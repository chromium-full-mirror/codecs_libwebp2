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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

TEST(DecodeTest, InteropApiOpaque) {
  ArgbBuffer original(WP2_RGB_24);
  ASSERT_WP2_OK(ReadImage(
      testutil::GetTestDataPath("source3_222x167.jpg").c_str(), &original));

  uint8_t* encoded_bytes = nullptr;
  size_t num_encoded_bytes = 0;
  ASSERT_WP2_OK(WP2EncodeLosslessRgb8(222, 167, original.GetRow8(0),
                                      original.stride(), 3, encoded_bytes,
                                      num_encoded_bytes));
  std::unique_ptr<uint8_t, decltype(&WP2Release)> encoded_bytes_deleter(
      encoded_bytes, WP2Release);

  uint32_t width, height;
  ASSERT_WP2_OK(WP2Parse(encoded_bytes, num_encoded_bytes, &width, &height));

  ArgbBuffer decoded(original.format());
  ASSERT_WP2_OK(decoded.Resize(width, height));
  ASSERT_WP2_OK(WP2DecodeRgb8(encoded_bytes, num_encoded_bytes, width, height,
                              decoded.GetRow8(0), decoded.stride()));

  ASSERT_TRUE(testutil::Compare(original, decoded, "source3_222x167.jpg"));
}

TEST(DecodeTest, InteropApiAlpha) {
  ArgbBuffer original(WP2_RGBA_32);
  ASSERT_WP2_OK(ReadImage(
      testutil::GetTestDataPath("source1_64x48.png").c_str(), &original));

  uint8_t* encoded_bytes = nullptr;
  size_t num_encoded_bytes = 0;
  ASSERT_WP2_OK(WP2EncodeLosslessRgba8(64, 48, original.GetRow8(0),
                                       original.stride(), 3, encoded_bytes,
                                       num_encoded_bytes));
  std::unique_ptr<uint8_t, decltype(&WP2Release)> encoded_bytes_deleter(
      encoded_bytes, WP2Release);

  uint32_t width, height;
  ASSERT_WP2_OK(WP2Parse(encoded_bytes, num_encoded_bytes, &width, &height));

  ArgbBuffer decoded(original.format());
  ASSERT_WP2_OK(decoded.Resize(width, height));
  ASSERT_WP2_OK(WP2DecodeRgba8(encoded_bytes, num_encoded_bytes, width, height,
                               decoded.GetRow8(0), decoded.stride()));

  ASSERT_TRUE(testutil::Compare(original, decoded, "source1_64x48.png"));
}

}  // namespace
}  // namespace WP2
