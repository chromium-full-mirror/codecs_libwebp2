// Copyright 2021 Google LLC
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

// Average color test. Ensures that average color is correctly calculated.

#include <cstdint>

#include "include/helpers.h"
#include "src/enc/preview/preview_enc.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------
// Testing buffers(WP2_Argb_32)

TEST(GetAverageColorTest_Argb_32, OneColor) {
  // Checks if a buffer filled with one color only is correctly averaged.

  for (uint16_t color = 0; color < 255; color++) {
    ArgbBuffer buffer(WP2_Argb_32);
    Argb32b org_color{255, (uint8_t)color, (uint8_t)color, (uint8_t)color};
    ASSERT_WP2_OK(buffer.Resize(100, 100));
    buffer.Fill(org_color);

    const Argb32b avg_color =
        GetAverageColor(buffer, /*radius=*/10, /*pixel_x=*/50, /*pixel_y=*/50);

    ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(org_color, avg_color));
  }
}

TEST(GetAverageColorTest_Argb_32, OneColorPerChannel) {
  // Checks if each channel having different color is correctly averaged.

  for (int i = 0; i < 3; i++) {
    for (uint8_t color = 0; color < 250; color++) {
      ArgbBuffer buffer(WP2_Argb_32);
      ASSERT_WP2_OK(buffer.Resize(100, 100));
      const Argb32b org_color{255, (i == 0) ? color : (uint8_t)0,
                              (i == 1) ? color : (uint8_t)0,
                              (i == 2) ? color : (uint8_t)0};
      buffer.Fill(org_color);

      const Argb32b avg_color = GetAverageColor(buffer, /*radius=*/10,
                                                /*pixel_x=*/50,
                                                /*pixel_y=*/50);

      ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(org_color, avg_color));
    }
  }
}

TEST(GetAverageColorTest_Argb_32, ZeroRadius) {
  // Checks if average color of one pixel is the color of the pixel.

  ArgbBuffer buffer(WP2_Argb_32);
  const uint32_t max_pixel_x = 16;
  const uint32_t max_pixel_y = 16;
  ASSERT_WP2_OK(buffer.Resize(max_pixel_x, max_pixel_y));
  uint8_t color_val = 0;
  for (uint16_t i = 0; i < max_pixel_x * max_pixel_y * 4; i++) {
    // Color always will be in the range [0, 255];
    buffer.GetRow8(0)[i] = (i % 4 == 0) ? kAlphaMax : color_val;
    if (i % 4 == 3) {
      ++color_val;
    }
  }
  for (uint32_t pixel_x = 0; pixel_x < max_pixel_x; pixel_x++) {
    for (uint32_t pixel_y = 0; pixel_y < max_pixel_y; pixel_y++) {
      // radius = 0 causes that only one pixel is taken into account while
      // computing the average color.
      const Argb32b org_color = testutil::GetPixel32b(buffer, pixel_x, pixel_y);
      const Argb32b avg_color =
          GetAverageColor(buffer, /*radius=*/0, pixel_x, pixel_y);
      ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(org_color, avg_color));
    }
  }
}

TEST(GetAverageColorTest_Argb_32, AlphaAveraging) {
  // Checks if averaging of alpha is correctly implemented.

  for (uint8_t alpha = 0; alpha < 250; alpha++) {
    ArgbBuffer buffer(WP2_Argb_32);
    Argb32b org_color = Argb32b({alpha, 0, 0, 0});
    ASSERT_WP2_OK(buffer.Resize(100, 100));
    buffer.Fill(org_color);

    const Argb32b avg_color =
        GetAverageColor(buffer, /*radius=*/10, /*pixel_x=*/50,
                        /*pixel_y=*/50);

    ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(org_color, avg_color));
  }
}

TEST(GetAverageColorTest_Argb_32, IndepentedFromPosition) {
  // Checks if color averaging works correctly for many different centers.

  for (uint8_t color = 0; color < 230; color += 20) {
    ArgbBuffer buffer(WP2_Argb_32);
    ASSERT_WP2_OK(buffer.Resize(50, 50));
    Argb32b org_color = Argb32b({255, color, color, color});
    buffer.Fill(org_color);

    uint32_t radius = 5;
    uint32_t min_pixel_x = 10;
    uint32_t min_pixel_y = 10;
    uint32_t max_pixel_x = 40;
    uint32_t max_pixel_y = 40;
    for (uint32_t pixel_x = min_pixel_x; pixel_x <= max_pixel_x; pixel_x += 5) {
      for (uint32_t pixel_y = min_pixel_y; pixel_y <= max_pixel_y;
           pixel_y += 5) {
        Argb32b avg_color = GetAverageColor(buffer, radius, pixel_x, pixel_y);
        ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(org_color, avg_color));
      }
    }
  }
}

TEST(GetAverageColorTest_Argb_32, DifferentColorsConstantAlpha) {
  // Checks if color averaging works correctly for a pixel surrounded with
  // pixels having other colors, and constant alpha. The tested buffer looks
  // as follows:
  //   *#*
  //   #*#
  //   *#*
  // # and * denote different but fixed colors. The test checks if the average
  // color for middle pixel is calculated correctly. The values of colors were
  // hand-designed, and the average color was calculated by hand.

  Argb32b asterisk_color = Argb32b({255, 0, 0, 0});
  Argb32b hash_color = Argb32b({255, 255, 255, 255});

  ArgbBuffer buffer(WP2_Argb_32);
  ASSERT_WP2_OK(buffer.Resize(3, 3));
  for (uint8_t i = 0; i < 3; i++) {
    for (uint8_t j = 0; j < 3; j++) {
      const Rectangle rectangle({i, j, 1, 1});
      // choose color based on (i + j) mod 2 to achieve chess grid
      if ((i + j) % 2 == 0) {
        buffer.Fill(rectangle, asterisk_color);
      } else {
        buffer.Fill(rectangle, hash_color);
      }
    }
  }

  Argb32b avg_color = GetAverageColor(buffer, /*radius=*/1, /*pixel_x*/ 1,
                                      /*pixel_y=*/1);

  // Calculated by hand
  Argb32b expected_color = Argb32b({255, 113, 113, 113});

  ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(expected_color, avg_color));
}

TEST(GetAverageColorTest_Argb_32, DifferentAlphaConstantColors) {
  // Checks if color averaging works correctly for a pixel surrounded with
  // pixels having other alpha, and constant color. The tested buffer looks
  // as follows:
  //   *#*
  //   #*#
  //   *#*
  // # and * denote different but fixed alpha. The test checks if the average
  // color for middle pixel is calculated correctly. The values of alpha were
  // hand-designed, and the average color was calculated by hand.

  Argb32b asterisk_color = Argb32b({0, 0, 0, 0});
  Argb32b hash_color = Argb32b({255, 0, 0, 0});

  ArgbBuffer buffer(WP2_Argb_32);
  ASSERT_WP2_OK(buffer.Resize(3, 3));
  for (uint8_t i = 0; i < 3; i++) {
    for (uint8_t j = 0; j < 3; j++) {
      const Rectangle rectangle({i, j, 1, 1});
      // choose color based on (i + j) mod 2 to achieve chess grid
      if ((i + j) % 2 == 0) {
        buffer.Fill(rectangle, asterisk_color);
      } else {
        buffer.Fill(rectangle, hash_color);
      }
    }
  }

  Argb32b avg_color = GetAverageColor(buffer, /*radius=*/1, /*pixel_x*/ 1,
                                      /*pixel_y=*/1);

  // Calculated by hand
  Argb32b expected_color = Argb32b({113, 0, 0, 0});

  ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(expected_color, avg_color));
}

TEST(GetAverageColorTest, Premultiplied) {
  // Checks if averaging works correctly for alpha premultiplication.
  ArgbBuffer unmul(WP2_ARGB_32), premul(WP2_Argb_32);
  for (ArgbBuffer* buffer : {&unmul, &premul}) {
    ASSERT_WP2_OK(buffer->Resize(2, 1));
    buffer->Fill({0, 0, 1, 1}, Argb32b{0, 0, 0, 0});
    buffer->Fill({1, 0, 1, 1}, Argb32b{255, 100, 100, 100});
  }
  // Only fully transparent and fully opaque pixels means the result should not
  // depend on alpha premultiplication mode.
  ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(
      GetAverageColor(unmul, kMaxBufferDimension, 0, 0),
      GetAverageColor(premul, kMaxBufferDimension, 0, 0)));
}

}  // namespace
}  // namespace WP2
