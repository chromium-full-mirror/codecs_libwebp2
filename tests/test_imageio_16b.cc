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

#include <string>

#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

TEST(ReadWriteRead16bImageTest, Opaque) {
  for (const std::string prefix : {"gray", "rgb"}) {
    ArgbBuffer buffer(WP2_ARGB_64);
    ASSERT_WP2_OK(ReadImage(
        testutil::GetTestDataPath(prefix + "16.png").c_str(), &buffer));
    EXPECT_FALSE(buffer.HasTransparency());

    const std::string dst =
        testutil::GetTempDataPath("test_imageio_16b_" + prefix + "16.png");
    ASSERT_WP2_OK(SaveImage(buffer, dst.c_str()));
    ArgbBuffer written(WP2_ARGB_64);
    ASSERT_WP2_OK(ReadImage(dst.c_str(), &written));
    EXPECT_TRUE(testutil::Compare(buffer, written, dst.c_str()));
  }
}

TEST(ReadWriteRead16bImageTest, Translucent) {
  for (const std::string prefix : {"grayalpha", "rgba"}) {
    ArgbBuffer buffer(WP2_ARGB_64);
    ASSERT_WP2_OK(ReadImage(
        testutil::GetTestDataPath(prefix + "16.png").c_str(), &buffer));
    EXPECT_TRUE(buffer.HasTransparency());

    const std::string dst =
        testutil::GetTempDataPath("test_imageio_16b_" + prefix + "16.png");
    ASSERT_WP2_OK(SaveImage(buffer, dst.c_str()));
    ArgbBuffer written(WP2_ARGB_64);
    ASSERT_WP2_OK(ReadImage(dst.c_str(), &written));
    EXPECT_TRUE(testutil::Compare(buffer, written, dst.c_str()));
  }
}

}  // namespace
}  // namespace WP2
