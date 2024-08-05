// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <utility>

#include "include/helpers.h"
#include "src/dsp/dsp.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace {

TEST(RgbConversionTest, From32bTo32b) {
  WP2ArgbConverterInit();

  for (WP2SampleFormat from :
       {WP2_Argb_32, WP2_ARGB_32, WP2_XRGB_32, WP2_rgbA_32, WP2_RGBA_32,
        WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24,
        WP2_BGR_24}) {
    ArgbBuffer src(from);
    ASSERT_WP2_OK(src.Resize(1, 1));
    ArgbBuffer roundtrip(from);
    ASSERT_WP2_OK(roundtrip.Resize(1, 1));

    for (WP2SampleFormat to :
         {WP2_Argb_32, WP2_ARGB_32, WP2_XRGB_32, WP2_rgbA_32, WP2_RGBA_32,
          WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24,
          WP2_BGR_24}) {
      uint8_t values[] = {123, 61, 200, 211};  // Arbitrary.

      uint32_t alpha_channel_index;
      if (WP2FormatHasAlpha(from, &alpha_channel_index)) {
        if (!WP2FormatHasAlpha(to)) {
          // Make sure input is opaque.
          values[alpha_channel_index] = 255;
        } else if (WP2IsPremultiplied(from)) {
          // Make sure alpha is at least as big as the values of the other
          // channels.
          uint8_t* max_element = std::max_element(values, values + 4);
          std::swap(*max_element, values[alpha_channel_index]);
        }
      } else if (alpha_channel_index < WP2FormatBpp(from)) {
        values[alpha_channel_index] = 255;
      }

      std::copy(values, values + WP2FormatBpp(from), src.GetRow8(0));

      ArgbBuffer dst(to);
      ASSERT_WP2_OK(dst.ConvertFrom(src));
      ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));

      const int tolerance =
          WP2IsPremultiplied(from) == WP2IsPremultiplied(to) ? 0 : 1;
      for (uint32_t i = 0; i < WP2FormatBpp(from); ++i) {
        ASSERT_NEAR(roundtrip.GetRow8(0)[i], values[i], tolerance)
            << " (channel " << i << ") from " << from << " to " << to << " ("
            << static_cast<int>(dst.GetRow8(0)[i]) << ")";
      }
    }
  }
}

}  // namespace
}  // namespace WP2
