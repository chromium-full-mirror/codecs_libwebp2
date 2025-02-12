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

#ifndef WP2_COMMON_LOSSLESS_TRANSFORMS_H_
#define WP2_COMMON_LOSSLESS_TRANSFORMS_H_

#include <array>
#include <cstdint>

#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

// Information that defines a transform to apply.
struct TransformHeader {
  enum class Channel { kRed = 1, kGreen = 2, kBlue = 3 };
  // 0 (nothing), 1 (subtract the first color), 2 (subtract the second color),
  // 3 (subtract the average of the first and second colors).
  enum class CCTransform {
    kNothing = 0,
    kSubtractFirst = 1,
    kSubtractSecond = 2,
    kSubtractAverage = 3
  };
  bool operator==(const TransformHeader& other) const {
    if (type != other.type) return false;
    if (type == TransformType::kColorIndexing) {
      return indexing_num_colors == other.indexing_num_colors;
    }
    return true;
  };

  TransformType type = TransformType::kNum;
  uint32_t indexing_num_colors = 0;
  // First and second channels to be compressed. The second channel will have a
  // transform applied to it. Indices are in ARGB so the indices are in [1-3].
  Channel cc_global_first_channel;
  Channel cc_global_second_channel;
  // Transforms to apply to the second channel. Transforms can be: kNothing,
  // kSubtractFirst.
  CCTransform cc_global_second_transform;
  // Transforms to apply to the third channel. Transforms can be: kNothing,
  // kSubtractFirst, kSubtractSecond, kSubtractAverage.
  CCTransform cc_global_third_transform;

  // The following are parameters that also define the transform but after it
  // has been computed on the image. They are necessary for GetARGBRanges.
  std::array<uint32_t, 4> normalize_channels_max = {0u, 0u, 0u, 0u};
  std::array<int16_t, 4> normalize_channels_min = {0, 0, 0, 0};
  bool normalize_channels_has_palette;
};

// Fills ranges of ARGB symbols according to the transforms applied to the
// image. 'num_colors' is the number of colors used in the palette if any (it is
// unused if no color indexing transform is present).
// If 'transforms' is not one of the official sets of transforms you can specify
// its length with 'num_transforms'.
WP2Status GetARGBRanges(
    const std::array<TransformHeader, kPossibleTransformCombinationSize>&
        headers,
    WP2SampleFormat format, std::array<int32_t, 4>& minima_range,
    std::array<int32_t, 4>& maxima_range,
    uint32_t num_transforms = kPossibleTransformCombinationSize);

}  // namespace WP2L

#endif  // WP2_COMMON_LOSSLESS_TRANSFORMS_H_
