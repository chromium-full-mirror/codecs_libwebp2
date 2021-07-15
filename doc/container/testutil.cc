// Copyright 2020 Google LLC
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

#include "./testutil.h"

#include <random>

#include "./container.h"

namespace container {
namespace testutil {

//------------------------------------------------------------------------------

void VerifyEqual(const Image& a, const Image& b) {
  // Note: Returning an AssertionResult, a Matcher<T> or using a MATCHER_P were
  //       considered but this helper func seemed the best compromise of
  //       elegance, low boilerplate and clear stderr output.
  EXPECT_EQ(a.width, b.width);
  EXPECT_EQ(a.height, b.height);
  EXPECT_EQ(a.orientation, b.orientation);
  EXPECT_EQ(a.format, b.format);
  EXPECT_EQ(a.is_opaque, b.is_opaque);
  EXPECT_EQ(a.preview_color, b.preview_color);
  EXPECT_EQ(a.icc, b.icc);
  EXPECT_EQ(a.xmp, b.xmp);
  EXPECT_EQ(a.exif, b.exif);
  EXPECT_EQ(a.is_animation, b.is_animation);
  if (a.is_animation && b.is_animation) {
    EXPECT_EQ(a.loop_forever, b.loop_forever);
    EXPECT_EQ(a.background_color, b.background_color);
  }
  EXPECT_EQ(a.pixels, b.pixels);
}

//------------------------------------------------------------------------------

Image GenerateRandomImage() {
  std::mt19937 random;

  // Demo image settings.
  Image input;
  input.width = 1000;
  input.height = 1000;
  input.orientation = Orientation::kOriginal;
  input.format = Format::kARGB8;
  input.is_opaque = true;
  input.preview_color = 0x0;
  input.is_animation = false;
  input.loop_forever = true;
  input.background_color = 0xFF000000u;

  // Random metadata. The content does not matter.
  for (Data* data : {&input.icc, &input.xmp, &input.exif}) {
    data->resize(1 + random() % 2048);
    for (uint8_t& byte : *data) byte = random() % 255;
  }

  // Random pixel values. The content does not matter.
  input.pixels.resize(input.width * input.height * (input.is_opaque ? 3 : 4));
  for (uint8_t& byte : input.pixels) byte = random() % 255;

  return input;
}

// -----------------------------------------------------------------------------

}  // namespace testutil
}  // namespace container
