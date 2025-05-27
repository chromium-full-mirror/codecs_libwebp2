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
  bool operator==(const TransformHeader& other) const {
    if (type != other.type) return false;
    if (type == TransformType::kColorIndexing) {
      return indexing_num_colors == other.indexing_num_colors;
    }
    return true;
  };

  TransformType type = TransformType::kNum;
  uint32_t indexing_num_colors = 0;
  // y index is in [1-9] and uv index in [1-12] respectively indices in section
  // A and B of "multiplierless reversible colour transforms and their automatic
  // selection for image data compression" by Tilo Strutz.
  uint32_t cc_global_y_index;
  uint32_t cc_global_uv_index;

  // The following are parameters that also define the transform but after it
  // has been computed on the image. They are necessary for GetARGBRanges.
  // After normalization that offsets the values by normalize_channels_offset,
  // the values end up in [0, normalize_channels_max].
  std::array<uint32_t, 4> normalize_channels_max = {0u, 0u, 0u, 0u};
  std::array<int16_t, 4> normalize_channels_offset = {0, 0, 0, 0};
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
