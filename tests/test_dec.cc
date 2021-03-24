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

// Simple decoding test.

#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------
// Test 10 bit lossless.

class Lossless10Test : public ::testing::TestWithParam<const char*> {};

TEST_P(Lossless10Test, Simple) {
  ArgbBuffer src(WP2_Argb_38);
  {
    // Load 10-bit source.
    ArgbBuffer src_32;
    ASSERT_WP2_OK(
        ReadImage(testutil::GetTestDataPath(GetParam()).c_str(), &src_32));
    ASSERT_WP2_OK(src.ConvertFrom(src_32));

    // Make some noise.
    UniformIntDistribution random(/*seed=*/9);
    uint16_t* const px = src.GetRow16(0);
    for (uint32_t i = 0; i < src.width() * src.height() * 4; ++i) {
      if (i % 4 == 0 || px[i] < 3) continue;  // Leave alpha as is.
      px[i] -= random.Get(0, 3);  // Make sure it stays valid alpha-premul.
    }
  }

  EncoderConfig config_enc;
  config_enc.quality = 100;
  MemoryWriter mem;
  ASSERT_WP2_OK(Encode(src, &mem, config_enc));

  ArgbBuffer output(WP2_Argb_38);
  ASSERT_WP2_OK(Decode(mem.mem_, mem.size_, &output));

  ASSERT_TRUE(testutil::Compare(src, output, GetParam(),
                                testutil::GetExpectedDistortion(config_enc)));
}

INSTANTIATE_TEST_SUITE_P(Lossless10TestInstantiation, Lossless10Test,
                         ::testing::Values("source1_1x1.png",
                                           "source1_64x48.png"));

//------------------------------------------------------------------------------

class DecodeTest
    : public ::testing::TestWithParam<std::tuple<std::string, bool, uint32_t>> {
};

TEST_P(DecodeTest, Simple) {
  const std::string src_file_name = std::get<0>(GetParam());
  const float quality = std::get<1>(GetParam()) ? 100.f : 75.f;
  const uint32_t thread_level = std::get<2>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage(src_file_name, &data, &src, quality,
                                        /*effort=*/3));

  ArgbBuffer output;
  DecoderConfig config;
  config.thread_level = thread_level;
  ASSERT_WP2_OK(Decode(data.mem_, data.size_, &output, config));

  ASSERT_TRUE(testutil::Compare(src, output, src_file_name,
                                testutil::GetExpectedDistortion(quality)));
}

INSTANTIATE_TEST_SUITE_P(
    DecodeTestInstantiationSingleTile, DecodeTest,
    ::testing::Combine(
        ::testing::Values("source1_64x48.png") /* one-tile image */,
        ::testing::Values(true) /* lossless */,
        ::testing::Values(0) /* no multithread; just one tile anyway */));

INSTANTIATE_TEST_SUITE_P(
    DecodeTestInstantiationMultiTile, DecodeTest,
    ::testing::Combine(::testing::Values("source0.pgm") /* several tiles */,
                       ::testing::Values(false) /* lossy */,
                       ::testing::Values(0, 64) /* extra threads */));

//------------------------------------------------------------------------------

TEST(DecodeTest, RawBuffer) {
  const std::string src_file_name = "source1.png";
  const float quality = 50.f;

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage(src_file_name, &data, &src, quality));

  Data output_buffer;
  const uint32_t min_stride = src.width() * WP2FormatBpp(src.format());
  const uint32_t useless_gap_between_lines = 237;
  const uint32_t arbitrary_stride = min_stride + useless_gap_between_lines;
  const uint32_t min_size = (src.height() - 1) * arbitrary_stride + min_stride;
  ASSERT_WP2_OK(output_buffer.Resize(min_size, /*keep_bytes=*/false));

  ASSERT_WP2_OK(DecodeArgb_32(data.mem_, data.size_, output_buffer.bytes,
                              arbitrary_stride, output_buffer.size));

  ArgbBuffer output;
  ASSERT_WP2_OK(output.SetExternal(src.width(), src.height(),
                                   output_buffer.bytes, arbitrary_stride,
                                   /*is_slow=*/true));

  ASSERT_TRUE(testutil::Compare(src, output, src_file_name,
                                testutil::GetExpectedDistortion(quality)));
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
