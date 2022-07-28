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
#include <string>
#include <vector>

#include "imageio/image_dec.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/lossless/palette.h"
#include "src/utils/ans_enc.h"
#include "tests/include/helpers.h"

#if 0

namespace WP2 {
namespace {

constexpr uint32_t kSegmentCapacity = 4000;

// Build artificial 3x3 RGB square for testing.
WP2Status BuildArgbBuffer(WP2::ArgbBuffer& buf) {
  WP2_CHECK_STATUS(buf.SetFormat(WP2_Argb_32));
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

static WP2Status ConvertArgbBuffer(const WP2::Vector_s16& decoded_argb,
                                   uint32_t width, uint32_t height,
                                   WP2::ArgbBuffer& decoded_buf) {
  WP2_CHECK_STATUS(decoded_buf.Resize(width, height));
  for (uint32_t y = 0; y < height; ++y) {
    uint8_t* const row = decoded_buf.GetRow8(y);
    std::copy(&decoded_argb[y * 4 * width], &decoded_argb[(y + 1) * 4 * width],
              row);
  }
  return WP2_STATUS_OK;
}

static WP2Status BuildDataStream(const WP2::ArgbBuffer& buf,
                                 WP2::Vector_s16& data_stream) {
  WP2_CHECK_OK(data_stream.resize(4 * buf.width() * buf.height()),
               WP2_STATUS_OUT_OF_MEMORY);
  for (uint32_t y = 0; y < buf.height(); ++y) {
    const uint8_t* const row = buf.GetRow8(y);
    std::copy(row, row + 4 * buf.width(), &data_stream[y * 4 * buf.width()]);
  }
  return WP2_STATUS_OK;
}

WP2Status LZWEncodeDecode(const WP2::ArgbBuffer& input_buf,
                          WP2::ArgbBuffer& decoded_buf,
                          uint32_t segment_capacity = kSegmentCapacity) {
  WP2MathInit();
  WP2::Vector_s16 data_stream;
  WP2_CHECK_STATUS(BuildDataStream(input_buf, data_stream));
  const WP2L::Buffer_s16& buffer = {input_buf.width(), input_buf.height(),
                                    &data_stream[0],
                                    input_buf.HasTransparency(), 3};

  WP2L::Palette palette;
  palette.Init(input_buf.format(), input_buf.HasTransparency());
  WP2_CHECK_STATUS(palette.AnalyzeAndCreate(input_buf));

  const WP2::Vector_s16& colors = palette.GetColors();
  WP2::ANSEnc enc;
  WP2_CHECK_STATUS(WP2L::LZWEncode(buffer, colors, segment_capacity, &enc));

  WP2_CHECK_STATUS(enc.AssembleToBitstream(/*clear_tokens=*/false));
  Vector_u8 bytes;
  WP2_CHECK_STATUS(enc.WriteBitstreamTo(bytes));

  // Prepare the symbol reader to send to decoder.
  WP2::SymbolsInfo info;
  WP2L::InitLZW(segment_capacity, &info);
  ExternalDataSource data_source(bytes.data(), bytes.size());
  ANSDec dec(&data_source);
  SymbolReader sr;
  WP2_CHECK_STATUS(sr.Init(info, &dec));
  WP2_CHECK_STATUS(sr.Allocate());

  WP2::Vector_s16 decoded_argb;
  WP2_CHECK_STATUS(WP2L::LZWDecode(buffer.width, buffer.height,
                                   segment_capacity, colors, &sr,
                                   &decoded_argb));

  WP2_CHECK_STATUS(ConvertArgbBuffer(decoded_argb, buffer.width, buffer.height,
                                     decoded_buf));

  return WP2_STATUS_OK;
}

// Tests if the LZW-like algorithm correctly encodes and decodes a simple image.
TEST(LZW, LZWEncodeAndDecodeSimpleImage) {
  ArgbBuffer input_buf;

  ASSERT_WP2_OK(BuildArgbBuffer(input_buf));

  ArgbBuffer decoded_buf;
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(
      testutil::Compare(input_buf, decoded_buf, /*file_name=*/"input_file"));
}

// Tests if the LZW-like algorithm correctly encodes and decodes another image.
TEST(LZW, LZWEncodeAndDecodeNormalImage) {
  ArgbBuffer input_buf;

  constexpr const char kInputFileName[] = "source1_64x48.png";

  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kInputFileName).c_str(), &input_buf));

  ArgbBuffer decoded_buf;
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));
}

// Tests if the LZW-like algorithm correctly encodes and decodes a large image.
TEST(LZW, LZWEncodeAndDecodeLargeImage) {
  ArgbBuffer input_buf;

  constexpr const char kInputFileName[] = "large.png";

  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kInputFileName).c_str(), &input_buf));

  ArgbBuffer decoded_buf;
  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf));

  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));
}

// Tests if the LZW-like algorithm works when 'capacity' is exactly equal to
// number of colors in the image.
TEST(LZW, LZWEncodeAndDecodeWhenCapacityEqualToNumberOfColors) {
  ArgbBuffer input_buf;

  constexpr const char kInputFileName[] = "logo.png";

  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kInputFileName).c_str(), &input_buf));

  WP2L::Palette palette;
  palette.Init(input_buf.format(), input_buf.HasTransparency());
  ASSERT_WP2_OK(palette.AnalyzeAndCreate(input_buf));

  const uint32_t num_colors = palette.GetColors().size() / 4;

  ArgbBuffer decoded_buf;
  EXPECT_EQ(LZWEncodeDecode(input_buf, decoded_buf, num_colors - 1),
            WP2_STATUS_OUT_OF_MEMORY);

  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf, num_colors));
  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));

  ASSERT_WP2_OK(LZWEncodeDecode(input_buf, decoded_buf, num_colors + 1));
  EXPECT_TRUE(testutil::Compare(input_buf, decoded_buf, kInputFileName));
}

}  // namespace
}  // namespace WP2
#endif
