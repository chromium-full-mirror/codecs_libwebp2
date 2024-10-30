// Copyright 2022 Google LLC
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
//
// Tests LZW-like algorithm implementation, both encoding and decoding.

#include <algorithm>
#include <cstdint>
#include <unordered_set>

#include "imageio/image_dec.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

constexpr uint32_t kSegmentCapacity = 4000;

// Build artificial 3x3 RGB square for testing.
WP2Status BuildArgbBuffer(WP2::ArgbBuffer& buf) {
  WP2_CHECK_STATUS(buf.SetFormat(WP2_ARGB_32));
  constexpr uint32_t kSize = 3;
  WP2_CHECK_STATUS(buf.Resize(kSize, kSize));
  uint8_t color[4] = {0xFF, 0xFF, 0x00, 0x00};
  buf.Fill({0, 0, 3, 1}, color);
  color[2] = color[1];
  color[1] = 0x00;
  buf.Fill({0, 1, 3, 1}, color);
  color[3] = color[2];
  color[2] = 0x00;
  buf.Fill({0, 2, 3, 1}, color);
  return WP2_STATUS_OK;
}

WP2Status LZWEncodeDecode(const WP2::ArgbBuffer& input_buf,
                          WP2::ArgbBuffer& decoded_buf,
                          uint32_t segment_capacity = kSegmentCapacity) {
  WP2MathInit();
  MemoryWriter memory_writer;
  EncoderConfig encoder_config;
  encoder_config.quality = 100;
  encoder_config.lossless_algorithm = WP2L::EncodingAlgorithm::kLZW;
  encoder_config.keep_unmultiplied = true;
  WP2_ASSERT_STATUS(Encode(input_buf, &memory_writer, encoder_config));

  DecoderConfig decoder_config;
  WP2_ASSERT_STATUS(Decode(memory_writer.mem_, memory_writer.size_,
                           &decoded_buf, decoder_config));

  return WP2_STATUS_OK;
}

// Tests if the LZW-like algorithm correctly encodes and decodes a simple image.
TEST(LZW, LZWEncodeAndDecodeSimpleImage) {
  ArgbBuffer input_buf(WP2_ARGB_32);

  ASSERT_WP2_OK(BuildArgbBuffer(input_buf));

  ArgbBuffer decoded_buf(WP2_ARGB_32);
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(
      testutil::Compare(input_buf, decoded_buf, /*file_name=*/"input_file"));
}

// Tests if the LZW-like algorithm correctly encodes and decodes another image.
TEST(LZW, LZWEncodeAndDecodeNormalImage) {
  ArgbBuffer input_buf(WP2_ARGB_32);

  constexpr const char kInputFileName[] = "source1_64x48.png";

  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kInputFileName).c_str(), &input_buf));

  ArgbBuffer decoded_buf(WP2_ARGB_32);
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));
}

// Tests if the LZW-like algorithm correctly encodes and decodes a large image.
TEST(LZW, LZWEncodeAndDecodeLargeImage) {
  ArgbBuffer input_buf(WP2_ARGB_32);

  constexpr const char kInputFileName[] = "large.png";

  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kInputFileName).c_str(), &input_buf));

  // Limit to kMaxPaletteSize colors per 256x256 tile.
  for (uint32_t Y = 0; Y < input_buf.height(); Y += 256) {
    for (uint32_t X = 0; X < input_buf.width(); X += 256) {
      std::unordered_set<uint32_t> colors;
      bool fill = false;
      for (uint32_t y = Y; y < std::min(Y + 256, input_buf.height()); ++y) {
        uint32_t* row = reinterpret_cast<uint32_t*>(input_buf.GetRow(y));
        for (uint32_t x = X; x < std::min(X + 256, input_buf.width()); ++x) {
          uint32_t* const pixel = row + x;
          if (fill) {
            *pixel = *colors.begin();
          } else {
            colors.insert(*pixel);
            fill = colors.size() >= WP2L::kMaxPaletteSize;
          }
        }
      }
      // TODO(vrabaud) fix this.
      // Make sure there are at least 2 colors for LZW as it needs a palette.
      if (colors.size() == 1) {
        reinterpret_cast<uint32_t*>(input_buf.GetRow(Y))[X] =
            ~reinterpret_cast<uint32_t*>(input_buf.GetRow(Y))[X];
      }
    }
  }

  ArgbBuffer decoded_buf(WP2_ARGB_32);
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));
}

}  // namespace
}  // namespace WP2
