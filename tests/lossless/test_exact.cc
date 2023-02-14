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
// -----------------------------------------------------------------------------
//
// Test lossless exact compression (no pre-multiplication).
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "tests/include/helpers.h"

using WP2::ArgbBuffer;
using WP2::DecoderConfig;
using WP2::EncoderConfig;
using WP2::MemoryWriter;

namespace WP2L {
namespace {

void EncodeDecodeExact(const ArgbBuffer &original) {
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 100.f;
  encoder_config.alpha_quality = 100.f;
  encoder_config.exact = true;
  DecoderConfig decoder_config = DecoderConfig::kDefault;
  decoder_config.exact = true;

  MemoryWriter memory_writer;
  WP2Status status = Encode(original, &memory_writer, encoder_config);
  EXPECT_EQ(status, WP2_STATUS_OK);

  ArgbBuffer decompressed(WP2_ARGB_32);
  status = Decode(memory_writer.mem_, memory_writer.size_, &decompressed,
                  decoder_config);
  EXPECT_EQ(status, WP2_STATUS_OK);

  ASSERT_TRUE(WP2::testutil::Compare(original, decompressed, "internal"));
}

class ExactTest
    : public testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(ExactTest, Random) {
  const uint32_t width = std::get<0>(GetParam());
  const uint32_t height = std::get<1>(GetParam());

  ArgbBuffer original(WP2_ARGB_32);
  EXPECT_EQ(original.Resize(width, height), WP2_STATUS_OK);
  WP2::testutil::PrecalculatedRandom<65536u, uint8_t> rnd(0, 255);

  rnd.Fill(original.GetRow8(0),
           original.width() * WP2FormatBpp(original.format()),
           original.height(), original.stride());

  EncodeDecodeExact(original);
}

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestWide, ExactTest,
                         testing::Values(std::make_tuple(1, 1),
                                         std::make_tuple(11, 12),
                                         std::make_tuple(31, 47)));

TEST(ExactTest, ReadFile) {
  ArgbBuffer original(WP2_ARGB_32);
  ASSERT_WP2_OK(WP2::ReadImage(
      WP2::testutil::GetTestDataPath("alpha_ramp.png").c_str(), &original));
  EncodeDecodeExact(original);
}

}  // namespace
}  // namespace WP2L
