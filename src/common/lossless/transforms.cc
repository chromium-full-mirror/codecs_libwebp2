// Copyright 2025 Google LLC
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
// Transform definition for WebP Lossless.
//
// Authors: Vincent Rabaud (vrabaud@google.com)

#include "src/common/lossless/transforms.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#include "src/common/color_precision.h"
#include "src/dsp/math.h"
#include "src/utils/ans.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2L {
// A segment of integers, defining the min/max of a channel.
struct Segment {
  Segment operator+(const Segment& other) const {
    return Segment{min + other.min, max + other.max};
  }
  Segment operator-(const Segment& other) const {
    return Segment{min - other.max, max - other.min};
  }
  Segment operator*(int32_t v) const {
    return v >= 0 ? Segment{min * v, max * v} : Segment{max * v, min * v};
  }
  Segment operator/(int32_t v) const {
    return v > 0 ? Segment{min / v, max / v} : Segment{max / v, min / v};
  }
  Segment operator|(const Segment& other) const {
    return Segment{std::min(min, other.min), std::max(max, other.max)};
  }
  // This is the only operation that can fail, if the two intervals do not
  // intersect.
  WP2_NO_DISCARD
  bool IntersectAndUpdate(const Segment& other) {
    if (other.min > max || other.max < min) return false;
    min = std::max(min, other.min);
    max = std::min(max, other.max);
    return true;
  }
  int32_t min, max;
};

}  // namespace WP2L

namespace WP2 {

template <>
inline WP2L::Segment RightShiftRound<WP2L::Segment>(WP2L::Segment v,
                                                    uint32_t shift) {
  return {RightShiftRound(v.min, shift), RightShiftRound(v.max, shift)};
}

}  // namespace WP2

namespace WP2L {

WP2Status GetARGBRanges(
    const std::array<TransformHeader, kPossibleTransformCombinationSize>&
        headers,
    WP2SampleFormat format, std::array<int32_t, 4>& minima_range,
    std::array<int32_t, 4>& maxima_range, uint32_t num_transforms) {
  Segment segments[4];
  // Define the original range according to the pixel format.
  const int32_t num_bits_alpha = ::WP2::kAlphaBits;
  const int32_t num_bits_non_alpha = static_cast<int32_t>(WP2Formatbpc(format));
  for (uint32_t c = 0; c < 4; ++c) {
    segments[c] = {0, static_cast<int32_t>(WP2::FormatMax(format, c))};
  }
  assert(segments[0].max == WP2::kAlphaMax);  // WP2_A***_64 is unsupported.
  bool do_clamp = true;
  for (uint32_t i = 0; i < num_transforms; ++i) {
    const TransformHeader& header = headers[i];
    if (header.type == TransformType::kNum) break;
    switch (header.type) {
      case TransformType::kPredictor:
      case TransformType::kPredictorWSub:
        // The predictors are convex or clamped in the current bounds. Only the
        // black predictor can give results out of those bounds.
        // The ranges are updated to be the ranges of the residuals.
        for (uint32_t c = 0; c < 4; ++c) {
          segments[c] =
              (segments[c] + segments[c]) | (segments[c] - segments[c]);
          // For alpha and a value of 0, because of the predictor
          // (kAlphaMax,0,0,0), the residual can be -WP2::kAlphaMax.
          if (c == 0) {
            segments[0].min = std::min(segments[0].min,
                                       -static_cast<int32_t>(WP2::kAlphaMax));
          }
          if (do_clamp) {
            const uint32_t num_bits =
                (c == 0) ? num_bits_alpha : num_bits_non_alpha;
            const int32_t abs_value_max =
                (num_bits == 8) ? kMaxAbsResidual8 : kMaxAbsResidual10;
            static_assert(2 * kMaxAbsResidual8 + 1 <= ::WP2::kANSMaxRange,
                          "invalid kMaxAbsResidual8 value");
            static_assert(2 * kMaxAbsResidual10 + 1 <= ::WP2::kANSMaxRange,
                          "invalid kMaxAbsResidual10 value");
            if (!segments[c].IntersectAndUpdate(
                    Segment{-abs_value_max, abs_value_max})) {
              return WP2_STATUS_INVALID_PARAMETER;
            }
            if (segments[c].min > segments[c].max) {
              // Should not happen.
              assert(false);
              return WP2_STATUS_INVALID_PARAMETER;
            }
          }
        }
        break;
      case TransformType::kCrossColor: {
        auto update_min_max = [&segments](uint32_t c0, uint32_t c1,
                                          int32_t coeff) {
          // The color transforms adds rounded(channel * coeff >>
          // kColorTransformBits). Extend the interval with all possible deltas.
          const Segment delta_pos =
              WP2::RightShiftRound(segments[c1] * coeff, kColorTransformBits);
          const Segment delta_neg =
              WP2::RightShiftRound(segments[c1] * -coeff, kColorTransformBits);
          segments[c0] =
              (segments[c0] + delta_pos) | (segments[c0] - delta_pos) |
              (segments[c0] + delta_neg) | (segments[c0] - delta_neg);
        };
        // First, red is modified by green.
        update_min_max(/*c0=*/1, /*c1=*/2, kGreenToRedMax);
        // Then blue by green and red.
        update_min_max(/*c0=*/3, /*c1=*/2, kGreenToBlueMax);
        update_min_max(/*c0=*/3, /*c1=*/1, kRedToBlueMax);
        break;
      }
      case TransformType::kCrossColorGlobal: {
        const uint32_t c1 =
            static_cast<uint32_t>(header.cc_global_first_channel);
        const uint32_t c2 =
            static_cast<uint32_t>(header.cc_global_second_channel);
        const Segment& segment1 = segments[c1];
        Segment& segment2 = segments[c2];
        if (header.cc_global_third_transform !=
            TransformHeader::CCTransform::kNothing) {
          Segment& segment3 = segments[1 + 2 + 3 - c1 - c2];
          if (header.cc_global_third_transform ==
              TransformHeader::CCTransform::kSubtractFirst) {
            segment3 = segment3 - segment1;
          } else if (header.cc_global_third_transform ==
                     TransformHeader::CCTransform::kSubtractSecond) {
            segment3 = segment3 - segment2;
          } else {
            segment3 = segment3 - (segment1 + segment2) / 2;
          }
        }
        if (header.cc_global_second_transform !=
            TransformHeader::CCTransform::kNothing) {
          segment2 = segment2 - segment1;
        }
        break;
      }
      case TransformType::kYCoCgR: {
        // co: r-b.
        const Segment segment_g = segments[2];
        segments[2] = segments[1] - segments[3];
        // tmp = (b + (co / 2)); cg = g - tmp;
        const Segment segment_tmp = segments[3] + (segments[2] / 2);
        segments[3] = segment_g - segment_tmp;
        // y = tmp + (cg / 2);
        segments[1] = segment_tmp + (segments[3] / 2);
        break;
      }
      case TransformType::kSubtractGreen:
        // green is subtracted from red and blue
        segments[1] = segments[1] - segments[2];
        segments[3] = segments[3] - segments[2];
        break;
      case TransformType::kColorIndexing:
        assert(i == 0);
        // With color indexing, values can go up to the number of colors so do
        // not clamp anymore after that.
        do_clamp = false;
        // A is set to its maximum value.
        segments[0] = {::WP2::kAlphaMax, ::WP2::kAlphaMax};
        // R and B are set to 0.
        segments[1] = segments[3] = {0, 0};
        // Only the color index in [0, num_colors-1] is stored in G.
        segments[2] = {0, static_cast<int32_t>(header.indexing_num_colors) - 1};
        break;
      case TransformType::kNormalizeChannels: {
        if (header.normalize_channels_has_palette) {
          for (uint32_t c = 0; c < 4; ++c) {
            segments[c] = {
                0, static_cast<int32_t>(header.normalize_channels_max[c])};
          }
        } else {
          for (uint32_t c = 0; c < 4; ++c) {
            segments[c] =
                segments[c] + Segment{-header.normalize_channels_offset[c],
                                      -header.normalize_channels_offset[c]};
          }
        }
        break;
      }
      case TransformType::kNum:
        assert(false);
    }
  }

  // Make sure we fit in the ANS ranges [0,(1 << WP2::kANSMaxRangeBits)-1].
  constexpr int32_t kMaxRange = 1 << WP2::kANSMaxRangeBits;
  for (uint32_t c = 0; c < 4; ++c) {
    // Assign to the output.
    if (!segments[c].IntersectAndUpdate(
            Segment{-kMaxRange / 2, kMaxRange / 2 - 1})) {
      return WP2_STATUS_INVALID_PARAMETER;
    }
    minima_range[c] = segments[c].min;
    maxima_range[c] = segments[c].max;
    assert(minima_range[c] <= maxima_range[c]);
  }
  return WP2_STATUS_OK;
}

}  // namespace WP2L
