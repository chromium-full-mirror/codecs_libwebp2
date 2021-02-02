// Copyright 2019 Google LLC
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
// Block position/size scoring functions.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/enc/partitioning/partition_score_func_multi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "src/common/integral.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/wp2_enc_i.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

struct BaseSlope {
  float base, slope;
};

//------------------------------------------------------------------------------

static constexpr uint32_t kMaxCertainty = 3;
static constexpr int32_t kKernelSize = 5, kMaxDist = kKernelSize / 2;

static void GetMinMax(const Plane16& src, Plane16& min, Plane16& max) {
  assert(min.w_ == max.w_ && min.h_ == max.h_);
  const uint32_t step = src.Step();
  const uint32_t num_blocks = min.w_;
  const int16_t* row = (const int16_t*)src.Row(0);
  for (uint32_t y = 0; y < min.h_; ++y) {
    int16_t* const min_row = (int16_t*)min.Row(y);
    int16_t* const max_row = (int16_t*)max.Row(y);
    WP2::GetBlockMinMax(row, step, num_blocks, min_row, max_row);
    row += kMinBlockSizePix * step;
  }
}

// Range of original YUV values in a kKernelSize window around each pixel.
// The higher the value, the more heterogenous the area is.
static void GetSpread(const Plane16& src, Plane16* const dst) {
  const int32_t w = src.w_, h = src.h_;
  int16_t* p_dst = dst->Row(0);
  const uint32_t dst_step = dst->Step();
  const int16_t* p_src = src.Row(0);
  const uint32_t src_step = src.Step();
  for (int32_t y = 0; y < h; ++y) {
    int16_t min, max;
    if (y < kMaxDist || y >= h - kMaxDist) {
      const int32_t min_sub_y = std::max(-kMaxDist, -y);
      const int32_t max_sub_y = std::min(kMaxDist, h - 1 - y);
      const int16_t* const row = &src.At(0, y + min_sub_y);
      for (int32_t x = 0; x < w; ++x) {
        const int32_t min_sub_x = std::max(-kMaxDist, -x);
        const int32_t max_sub_x = std::min(kMaxDist, w - 1 - x);
        GetBlockMinMaxGeneric(row + x + min_sub_x, src_step,
                              max_sub_x + 1 - min_sub_x,
                              max_sub_y + 1 - min_sub_y, &min, &max);
        p_dst[x] = (int16_t)ClampToSigned(max - min, kMaxYuvBits + 1u);
      }
    } else {
      const int16_t* const row = p_src - src_step * kMaxDist;
      for (int32_t x = 0; x < w; ++x) {
        if (x < kMaxDist || x - kMaxDist + 8 > w) {
          const int32_t min_sub_x = std::max(-kMaxDist, -x);
          const int32_t max_sub_x = std::min(kMaxDist, w - 1 - x);
          GetBlockMinMaxGeneric(row + x + min_sub_x, src_step,
                                max_sub_x + 1 - min_sub_x, kKernelSize,
                                &min, &max);
        } else {
          GetBlockMinMax_5x5(row + x - kMaxDist, src_step, &min, &max);
        }
        p_dst[x] = (int16_t)ClampToSigned(max - min, kMaxYuvBits + 1u);
      }
    }
    p_dst += dst_step;
    p_src += src_step;
  }
}

constexpr float MultiScoreFunc::kMinScore;
constexpr int MultiScoreFunc::kMinEffortForGoodQuantDCT;

WP2Status MultiScoreFunc::Init(const EncoderConfig& config,
                               const Rectangle& tile_rect, const YUVPlane& yuv,
                               const GlobalParams& gparams,
                               const ProgressRange& progress) {
  DrctFilterInit();
  ScoreDspInit();

  WP2_CHECK_STATUS(PartitionScoreFunc::Init(config, tile_rect, yuv, gparams,
                                            ProgressRange(progress, 0.5)));
  yuv_range_ =
      (float)(gparams.transf_.GetYUVMax() - gparams.transf_.GetYUVMin());
  a_range_ratio_ = yuv_range_ / kAlphaMax;

  // Cache the 'min_' and 'max_' luma/chroma values per kMinBlockSize square.
  WP2_CHECK_STATUS(min_.Resize(SizeBlocks(src_->Y.w_), SizeBlocks(src_->Y.h_)));
  WP2_CHECK_STATUS(max_.Resize(min_.Y.w_, min_.Y.h_));
  for (Channel channel : {kYChannel, kUChannel, kVChannel}) {
    const Plane16& src_plane = src_->GetChannel(channel);
    assert(src_plane.w_ % kMinBlockSizePix == 0 &&
           src_plane.h_ % kMinBlockSizePix == 0);
    GetMinMax(src_plane, min_.GetChannel(channel), max_.GetChannel(channel));
  }

  if (config_->effort >= kMinEffortForGoodQuantDCT) {
    // Image processing.
    WP2_CHECK_STATUS(spread_.Resize(src_->GetWidth(), src_->GetHeight(),
                                    /*pad=*/1, with_alpha_));
    for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
      if (channel == kAChannel && !with_alpha_) continue;
      GetSpread(src_->GetChannel(channel), &spread_.GetChannel(channel));
    }
  }

  // Per-block standard deviation.
  WP2_CHECK_STATUS(
      stddev_.Allocate(num_block_cols_, num_block_rows_, kMinBlockSizePix));
  stddev_.AddValues(*src_);

  if (with_alpha_) {
    WP2_CHECK_STATUS(
        a_stddev_.Allocate(num_block_cols_, num_block_rows_, kMinBlockSizePix));
    a_stddev_.AddValues(src_->A);
  }

  // Per-block luma general direction.
  // In might give better results to compute that for each NxN block instead of
  // aggregating pre-computed 4x4 ones but it is probably too expensive.
  WP2_CHECK_ALLOC_OK(direction_.resize(num_block_cols_ * num_block_rows_));
  WP2_CHECK_ALLOC_OK(
      direction_certainty_.resize(num_block_cols_ * num_block_rows_));

  const uint32_t bitdepth = gparams.transf_.GetYUVPrecisionBits() + 1;
  for (uint32_t y = 0; y < num_block_rows_; ++y) {
    const int16_t* x_ptr = &src_->Y.At(/*x=*/0, y * kMinBlockSizePix);
    for (uint32_t i = y * num_block_cols_; i < (y + 1) * num_block_cols_;
         ++i, x_ptr += kMinBlockSizePix) {
      uint32_t variance;
      CdefDirection4x4(x_ptr, src_->Y.Step(), bitdepth, &direction_[i],
                       &variance);
      direction_certainty_[i] = Clamp(variance >> 4, 0u, kMaxCertainty);
      // TODO(yguyon): Also compute, store, use 8x8 direction for bigger blocks
    }
  }

  WP2_CHECK_REDUCED_STATUS(DrawVDebug());

  WP2_CHECK_STATUS(progress.AdvanceBy(0.5));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status MultiScoreFunc::ComputeScore(const Block& block,
                                       const ProgressRange& progress,
                                       float* const score) {
  float value = 0.f, threshold = 1.f;  // Passing if 'value <= threshold'.
  switch (pass_) {
    case Pass::LumaAlphaGradient:
      value = GetLumaAlphaGradient(block);
      threshold = GetLumaAlphaGradientThreshold(block);
      break;
    case Pass::NarrowStdDev:
      value = GetStdDevRange(block);
      threshold = GetStdDevRangeThreshold(block);
      break;
    case Pass::GoodQuantDCT:
      value = GetQuantDCT(block);
      threshold = GetQuantDCTThreshold(block);
      break;
    case Pass::Direction:
      value = GetDirection(block);
      threshold = GetDirectionThreshold(block);
      break;
    case Pass::Any:
      value = 0.f;
      threshold = 1.f;
      break;
    default:
      assert(false);
  }

  // Convert to "higher score is better" in [0:1], 0.5 being the threshold.
  if (value <= threshold) {
    *score = 1.001f - 0.5f * value / threshold;  // Will pass.
  } else {
    *score = 0.499f * threshold / value;  // Will not pass.
  }
  WP2_CHECK_REDUCED_STATUS(RegisterScoreForVDebug(block, *score));
  WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void MultiScoreFunc::GetCoeffs(Channel channel, const Block& block,
                               int16_t coeffs[kMaxBlockSizePix2]) const {
  const Plane16& plane = src_->GetChannel(channel);
  // Copy luma pixels.
  int16_t* dst_row = coeffs;
  const int16_t* src_row = &plane.At(block.x_pix(), block.y_pix());
  for (uint32_t y = 0; y < block.h_pix(); ++y) {
    for (uint32_t x = 0; x < block.w_pix(); ++x) {
      dst_row[x] = src_row[x];
    }
    dst_row += block.w_pix();
    src_row += plane.Step();
  }
}

void MultiScoreFunc::QuantizeCoeffs(Channel channel, const Block& block,
                                    int16_t coeffs[kMaxBlockSizePix2]) const {
  GetCoeffs(channel, block, coeffs);

  // Transform them.
  int32_t out[kMaxBlockSizePix2];
  WP2Transform2D(coeffs, kDct, kDct, block.w_pix(), block.h_pix(), out,
                 /*reduced=*/false);

  // Find the segment of the block.
  const Rectangle padded_tile_rect = {tile_rect_.x, tile_rect_.y,
                                      src_->GetWidth(), src_->GetHeight()};
  const uint8_t segment_id = (channel == kAChannel) ? 0 :
      AssignSegmentId(*config_, *gparams_, padded_tile_rect, block);
  const QuantMtx& quant_mtx =
      gparams_->segments_[segment_id].GetQuant(channel);

  // Quantize and dequantize coefficients.
  const TrfSize tdim = GetTransform(block.dim());
  uint32_t num_coeffs;
  quant_mtx.Quantize(out, tdim, /*first_is_dc=*/true, coeffs, &num_coeffs);
  quant_mtx.Dequantize(coeffs, num_coeffs, tdim, coeffs);

  // Inverse transform them back and compare the result.
  WP2InvTransform2D(coeffs, kDct, kDct, block.w_pix(), block.h_pix(),
                    /*reduced=*/false);
}

void MultiScoreFunc::QuantizeCoeffs(Channel channel, const Block& block,
                                    BlockSize sub_block_size,
                                    int16_t coeffs[kMaxBlockSizePix2],
                                    int32_t* const max_range) const {
  if (max_range != nullptr) *max_range = 0;
  Block sub_block(block.x(), block.y(), sub_block_size);
  while (sub_block.y() + sub_block.h() <= block.y() + block.h()) {
    int16_t sub_coeffs[kMaxBlockSizePix2];
    QuantizeCoeffs(channel, sub_block, sub_coeffs);
    int16_t min = sub_coeffs[0], max = sub_coeffs[0];
    const uint32_t y = sub_block.y_pix() - block.y_pix();
    const uint32_t x = sub_block.x_pix() - block.x_pix();
    for (uint32_t sub_y = 0; sub_y < sub_block.h_pix(); ++sub_y) {
      for (uint32_t sub_x = 0; sub_x < sub_block.w_pix(); ++sub_x) {
        const int16_t sub_coeff = sub_coeffs[sub_y * sub_block.w_pix() + sub_x];
        min = std::min(min, sub_coeff);
        max = std::max(max, sub_coeff);
        coeffs[(y + sub_y) * block.w_pix() + (x + sub_x)] = sub_coeff;
      }
    }
    if (max_range != nullptr) *max_range = std::max(*max_range, max - min);
    sub_block.SetXY(sub_block.x() + sub_block.w(), sub_block.y());
    if (sub_block.x() + sub_block.w() > block.x() + block.w()) {
      sub_block.SetXY(block.x(), sub_block.y() + sub_block.h());
    }
  }
}

//------------------------------------------------------------------------------

// Returns the maximum difference between 'src' and its predicted gradient.
// 'step' goes from a 'src' line to the next.
static int32_t GetGradientDiff(const int16_t* src, int32_t step,
                               int32_t w, int32_t h) {
  assert(w >= 4 && h >= 4);
  const int32_t max_x = w - 1, max_y = h - 1;

  // Average the corners (division by 3 is done at the very end).
  const int32_t top_left = src[0] + src[1] + src[step];
  const int32_t bottom_left =
      src[(max_y - 1) * step] + src[max_y * step] + src[max_y * step + 1];
  const int32_t top_right = src[max_x - 1] + src[max_x] + src[step + max_x];
  const int32_t bottom_right = src[(max_y - 1) * step + max_x] +
                               src[max_y * step + max_x - 1] +
                               src[max_y * step + max_x];

  // Create a gradient by bidimensional interpolation and compare with 'src'.
  int32_t max_diff = 0;
  for (int32_t y = 0; y <= max_y; ++y) {
    const int32_t left = top_left * (max_y - y) + bottom_left * y;
    const int32_t right = top_right * (max_y - y) + bottom_right * y;
    for (int32_t x = 0; x <= max_x; ++x) {
      const int32_t gradient_pixel =
          DivRound(left * (max_x - x) + right * x, 3 * max_x * max_y);
      max_diff = std::max(max_diff, std::abs(src[x] - gradient_pixel));
    }
    src += step;
  }
  return max_diff;
}

float MultiScoreFunc::GetLumaAlphaGradient(const Block& block) const {
  // Empirically chosen values.
  const float kDiffScale[] = {1.f, 1.f, 1.f, 0.25f / kAlphaMax * yuv_range_};
  float max_diff = 0.f;
  // kUChannel and kVChannel do not bring valuable partition decision-making
  // here so skip them for speed.
  for (Channel c : {kYChannel, kAChannel}) {
    if (c == kAChannel && !with_alpha_) continue;
    const int32_t diff = GetGradientDiff(
        &src_->GetChannel(c).At(block.x_pix(), block.y_pix()),
        src_->GetChannel(c).Step(), block.w_pix(), block.h_pix());
    max_diff = std::max(max_diff, diff * kDiffScale[c]);
  }
  return max_diff;
}

float MultiScoreFunc::GetLumaAlphaGradientThreshold(const Block& block) const {
  // The threshold is tighter for bigger blocks at higher qualities.
  // Medium blocks are ignored except at high qualities.
  constexpr BaseSlope kBaseSlope[] = {
      {0.f, 0.f},        //  4x4, unused
      {0.f, 0.f},        //  8x4, unused
      {-0.6f, 0.7f},     //  8x8  16x4
      {-0.06f, 0.08f},   // 16x8
      {-0.03f, 0.045f},  // 16x16 32x8
      {0.04f, -0.035f},  // 32x16
      {0.05f, -0.045f},  // 32x32
  };
  STATIC_ASSERT_ARRAY_SIZE(kBaseSlope, WP2Log2Ceil_k(kMaxBlockSize2) + 1);
  const uint32_t index = (uint32_t)WP2Log2Floor(block.rect().GetArea());
  assert((1u << index) == block.rect().GetArea());
  return yuv_range_ *
         MapQuality(*config_, kBaseSlope[index].base, kBaseSlope[index].slope);
}

//------------------------------------------------------------------------------

static float StdDevRange(const Block& block, const Integral& variance) {
  // Consider the standard deviation of the whole block as sub-blocks could
  // be coherent within themselves but not with other sub-blocks.
  const uint8_t overall_variance = variance.StdDevUint8(
      block.x(), block.y(), block.x() + block.w(), block.y() + block.h());
  uint8_t min = overall_variance, max = overall_variance;

  // Now check the sub-blocks.
  for (uint32_t sub_y = block.y(); sub_y < block.y() + block.h(); ++sub_y) {
    for (uint32_t sub_x = block.x(); sub_x < block.x() + block.w(); ++sub_x) {
      const uint8_t variance_tmp =
          variance.StdDevUint8(sub_x, sub_y, sub_x + 1, sub_y + 1);
      if (variance_tmp < min) {
        min = variance_tmp;
      } else if (variance_tmp > max) {
        max = variance_tmp;
      }
    }
  }
  return (max - min) / 255.f;
}

float MultiScoreFunc::GetStdDevRange(const Block& block) const {
  float range = StdDevRange(block, stddev_);
  if (!a_stddev_.empty()) {
    const float a_range =
        StdDevRange(block, a_stddev_) * a_range_ratio_ * a_range_ratio_;
    range = std::max(range, a_range);
  }

  return range;
}

float MultiScoreFunc::GetStdDevRangeThreshold(const Block& block) const {
  // The higher the quality, the narrower the standard deviation range needs to
  // be for a block to be accepted. Accept bigger blocks during partitioning at
  // low qualities, and seek smaller blocks at high qualities.
  return MapQuality(*config_, 0.50f, -0.38f);
}

//------------------------------------------------------------------------------

// Returns the average difference between the original luma coefficients and the
// quantized ones, weighted per pixel by the 'spread_' in order to give more
// importance to flat areas (penalize distant ripples more than noise on edges).
float MultiScoreFunc::GetQuantDCT(const Block& block, Channel channel) const {
  int16_t coeffs[kMaxBlockSizePix2];
  QuantizeCoeffs(channel, block, coeffs);

  float avg_diff = 0.f;
  int16_t* dst_row = coeffs;
  const int16_t* src_row =
      &src_->GetChannel(channel).At(block.x_pix(), block.y_pix());
  const int16_t* spread_row =
      &spread_.GetChannel(channel).At(block.x_pix(), block.y_pix());
  for (uint32_t y = 0; y < block.h_pix(); ++y) {
    for (uint32_t x = 0; x < block.w_pix(); ++x) {
      const int32_t diff = std::abs(dst_row[x] - src_row[x]);
      avg_diff += diff / Clamp(spread_row[x] / 20.f, 0.1f, 10.f);
    }
    dst_row += block.w_pix();
    src_row += src_->GetChannel(channel).Step();
    spread_row += spread_.GetChannel(channel).Step();
  }
  avg_diff /= (block.w_pix() * block.h_pix());
  return avg_diff;
}

float MultiScoreFunc::GetQuantDCT(const Block& block) const {
  // Take chroma and alpha into account just enough to discard bad layouts (for
  // example an entirely black image with alpha patterns).
  // TODO(yguyon): Check ADST too
  constexpr float kScale[] = {1.f, 0.1f, 0.1f, 0.1f};
  return std::max({GetQuantDCT(block, kYChannel) * kScale[kYChannel],
                   GetQuantDCT(block, kUChannel) * kScale[kUChannel],
                   GetQuantDCT(block, kVChannel) * kScale[kVChannel],
                   with_alpha_
                       ? GetQuantDCT(block, kAChannel) * kScale[kAChannel]
                       : 0.f});
}

float MultiScoreFunc::GetQuantDCTThreshold(const Block& block) const {
  // This metric is only useful at low qualities, for large blocks that would
  // still look fine with an aggressive quantization.
  return MapQuality(*config_, 4.00f, -4.75f);
}

//------------------------------------------------------------------------------

// Returns a low score for a 'block' that appears to have an obvious and uniform
// orientation.
float MultiScoreFunc::GetDirection(const Block& block) const {
  const uint32_t stride = num_block_cols_;
  const uint32_t* direction =
      direction_.data() + block.y() * stride + block.x();
  const uint32_t* certainty =
      direction_certainty_.data() + block.y() * stride + block.x();

  uint32_t weight[kDrctFltNumDirs] = {0};
  for (uint32_t sub_y = 0; sub_y < block.h(); ++sub_y) {
    for (uint32_t sub_x = 0; sub_x < block.w(); ++sub_x) {
      assert(direction[sub_x] < kDrctFltNumDirs);
      weight[direction[sub_x]] += std::min(certainty[sub_x], kMaxCertainty);
    }
    direction += stride;
    certainty += stride;
  }
  const uint32_t heaviest =
      std::max_element(weight, weight + kDrctFltNumDirs) - weight;
  const uint32_t max_weight = block.w() * block.h() * kMaxCertainty;
  assert(weight[heaviest] <= max_weight);
  const uint32_t previous_direction =
      (heaviest - 1 + kDrctFltNumDirs) % kDrctFltNumDirs;
  const uint32_t next_direction = (heaviest + 1) % kDrctFltNumDirs;
  const uint32_t weight_with_close_directions =
      weight[heaviest] +
      (weight[previous_direction] + weight[next_direction]) / 4;
  const float direction_score =
      1.f - Clamp(weight_with_close_directions / (float)max_weight, 0.f, 1.f);

  return direction_score;
}

float MultiScoreFunc::GetDirectionThreshold(const Block& block) const {
  return MapQuality(*config_, 0.15f, -0.0475f);
}

//------------------------------------------------------------------------------

}  // namespace WP2
