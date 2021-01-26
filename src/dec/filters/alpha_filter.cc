// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------
//
// Alpha filter (bilateral filter).
//
// Author: Maryla (maryla@google.com)

#include "src/dec/filters/alpha_filter.h"

#include "src/dec/wp2_dec_i.h"

namespace WP2 {

static constexpr uint32_t kWeightBits = 10;
// Gaussian blur kernels (in 1 dimension).
// round(exp(-i*i/(2.*radius)) * (1<<kWeightBits))
constexpr uint32_t kGaussian1[] = {1024, 621};       // radius = 1
constexpr uint32_t kGaussian2[] = {1024, 797, 377};  // radius = 2
// Weights for pixel value differences.
// round(exp(-i*i/(2.*sigma*sigma)) * (1<<kWeightBits))
// sigma = 5.
constexpr uint32_t kValueWeights5[kAlphaMax + 1] = {
    1024, 1004, 945, 855, 744, 621, 498, 384, 285, 203,
    139,  91,   57,  35,  20,  11,  6,   3,   2,   1};  // Rest is 0.
// sigma = 10
constexpr uint32_t kValueWeights10[kAlphaMax + 1] = {
    1024, 1019, 1004, 979, 945, 904, 855, 801, 744, 683, 621, 559, 498, 440,
    384,  332,  285,  241, 203, 168, 139, 113, 91,  73,  57,  45,  35,  27,
    20,   15,   11,   8,   6,   4,   3,   2,   2,   1,   1,   1};  // Rest is 0.
// sigma = 20
constexpr uint32_t kValueWeights20[kAlphaMax + 1] = {
    1024, 1023, 1019, 1013, 1004, 992, 979, 963, 945, 925, 904, 880, 855, 829,
    801,  773,  744,  714,  683,  652, 621, 590, 559, 529, 498, 469, 440, 412,
    384,  358,  332,  308,  285,  262, 241, 221, 203, 185, 168, 153, 139, 125,
    113,  102,  91,   81,   73,   65,  57,  51,  45,  40,  35,  31,  27,  23,
    20,   18,   15,   13,   11,   10,  8,   7,   6,   5,   4,   4,   3,   3,
    2,    2,    2,    1,    1,    1,   1,   1,   1};  // Rest is 0.

// Quality factor threshold to enable the filter.
static constexpr uint16_t kEnableThreshold = 10;

bool IsAlphaFilterEnabled(const DecoderConfig& config,
                          const GlobalParams& gparams) {
  return gparams.maybe_use_lossy_alpha_ &&
         (gparams.enable_alpha_filter_ || VDMatch(config, "alpha-filter")) &&
         (gparams.segments_[0].a_quality_factor_ >= kEnableThreshold);
}

AlphaFilter::AlphaFilter(const DecoderConfig& config,
                         const GlobalParams& gparams,
                         const FilterBlockMap& blocks)
    : enabled_(IsAlphaFilterEnabled(config, gparams)),
      config_(config),
      blocks_(blocks) {
  if (!enabled_) return;
  assert(!gparams.segments_.empty());
  const uint16_t quality_factor = gparams.segments_[0].a_quality_factor_;
  // TODO(maryla): fine tune thresholds/sigma values.
  if (quality_factor < 50) {
    kernel_radius_ = 1;
    gaussian_ = kGaussian1;
  } else {
    kernel_radius_ = 2;
    gaussian_ = kGaussian2;
  }
  if (quality_factor < 20) {
    value_weights_ = kValueWeights5;
  } else if (quality_factor < 30) {
    value_weights_ = kValueWeights10;
  } else {
    value_weights_ = kValueWeights20;
  }
}

WP2Status AlphaFilter::Allocate() {
  if (!enabled_) return WP2_STATUS_OK;
  WP2_CHECK_STATUS(top_taps_.Resize(blocks_.tile_rect_.width, kTapHeight));
  return WP2_STATUS_OK;
}

uint32_t AlphaFilter::Smooth(uint32_t num_rows) {
  const uint32_t h = blocks_.tile_rect_.height;
  const uint32_t w = blocks_.tile_rect_.width;

  assert(num_rows <= h);
  assert(num_filtered_rows_ <= num_rows);

  Plane16* const pixels = &blocks_.pixels_->A;

  if (!enabled_ || blocks_.pixels_->A.IsEmpty() ||
      num_filtered_rows_ >= num_rows) {
    return num_rows;
  }

#if !defined(WP2_REDUCE_BINARY_SIZE)
  if (VDMatch(config_, "alpha-filter")) SavePixelsForVDebug(num_rows);
#endif  // WP2_REDUCE_BINARY_SIZE

  uint32_t num_filterable_rows;
  if (num_rows == h) {
    // All pixels are available: filter everything that is left.
    num_filterable_rows = num_rows;
  } else {
    // Wait for bottom taps to be available.
    num_filterable_rows = SafeSub(num_rows, (uint32_t)kMaxKernelRadius);
  }

  for (uint32_t y = num_filtered_rows_; y < num_filterable_rows; ++y) {
    const FilterBlockMap::BlockFeatures* const blocks =
        blocks_.GetRow(y / kMinBlockSizePix);

    // Copy the unfiltered version of the current row to 'top_taps_'.
    int16_t* const src = pixels->Row(y);
    std::copy(src, src + w, top_taps_.Row(y % kTapHeight));

    // Filter the w x h area.
    const uint32_t y_min = std::max(y, kernel_radius_) - kernel_radius_;
    const uint32_t y_max = std::min(y + kernel_radius_ + 1u, h);
    for (uint32_t x = 0; x < w; ++x) {
      if (!blocks[x / kMinBlockSizePix].has_lossy_alpha) {
        x += kMinBlockSizePix - 1;
        continue;
      }
      uint32_t sum = 0u;
      uint32_t weights = 0u;

      const uint32_t x_min = std::max(x, kernel_radius_) - kernel_radius_;
      const uint32_t x_max = std::min(x + kernel_radius_ + 1u, w);
      const int16_t center_pix = Clamp<int16_t>(src[x], 0, kAlphaMax);
      static constexpr uint32_t kMultNumBits = WP2Log2Ceil_k(kMaxKernelSize2);
      static constexpr uint32_t kSumNumBits =
          kAlphaBits + kWeightBits + kMultNumBits;
      static constexpr uint32_t kWeightsBits = kWeightBits + kMultNumBits;
      static_assert(kSumNumBits <= sizeof(sum) * 8 - 1, "Possible overflow.");
      static_assert(kWeightsBits <= sizeof(weights) * 8 - 1,
                    "Possible overflow.");

      for (uint32_t Y = y_min; Y < y_max; ++Y) {
        const uint32_t y_weight = gaussian_[(Y >= y) ? Y - y : y - Y];
        const int16_t* const tap =
            (Y <= y) ? top_taps_.Row(Y % kTapHeight) : pixels->Row(Y);
        for (uint32_t X = x_min; X < x_max; ++X) {
          const int16_t pix = Clamp<int16_t>(tap[X], 0, kAlphaMax);
          const int32_t diff = std::abs(center_pix - pix);
          const uint32_t value_weight = value_weights_[diff];
          const uint32_t x_weight = gaussian_[(X >= x) ? X - x : x - X];
          const uint32_t weight =
              ChangePrecision(value_weight * x_weight * y_weight,
                              kWeightBits * 3, kWeightBits);
          static_assert(kWeightBits * 3 <= sizeof(x_weight) * 8 - 1,
                        "Possible overflow.");
          weights += weight;
          sum += weight * pix;
        }
      }
      // TODO(skal): could be sped up if we had a constant 'weights' denominator
      src[x] = DivRound(sum, weights);
    }
  }
  num_filtered_rows_ = num_filterable_rows;

#if !defined(WP2_REDUCE_BINARY_SIZE)
  if (VDMatch(config_, "alpha-filter")) ApplyVDebug(num_rows);
#endif  // WP2_REDUCE_BINARY_SIZE
  return num_filtered_rows_;
}

}  // namespace WP2
