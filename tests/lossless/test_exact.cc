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

#include <cstdint>
#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

bool EncodeDecodeExact(const ArgbBuffer &original, bool exact) {
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 100.f;
  encoder_config.alpha_quality = 100.f;
  encoder_config.exact = exact;
  DecoderConfig decoder_config = DecoderConfig::kDefault;
  decoder_config.exact = exact;

  MemoryWriter memory_writer;
  EXPECT_EQ(Encode(original, &memory_writer, encoder_config), WP2_STATUS_OK);

  ArgbBuffer decompressed(original.format());
  EXPECT_EQ(Decode(memory_writer.mem_, memory_writer.size_, &decompressed,
                   decoder_config),
            WP2_STATUS_OK);

  return testutil::Compare(original, decompressed, "internal");
}

//------------------------------------------------------------------------------
// Exact lossless compression with non-premultiplied samples

class ExactTest
    : public testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(ExactTest, Random) {
  const uint32_t width = std::get<0>(GetParam());
  const uint32_t height = std::get<1>(GetParam());

  ArgbBuffer original(WP2_ARGB_32);
  EXPECT_EQ(original.Resize(width, height), WP2_STATUS_OK);
  testutil::PrecalculatedRandom<65536u, uint8_t> rnd(0, 255);

  rnd.Fill(original.GetRow8(0),
           original.width() * WP2FormatBpp(original.format()),
           original.height(), original.stride());

  ASSERT_TRUE(EncodeDecodeExact(original, /*exact=*/true));
}

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestWide, ExactTest,
                         testing::Values(std::make_tuple(1, 1),
                                         std::make_tuple(11, 12),
                                         std::make_tuple(31, 47)));

TEST(ExactTest, ReadFile) {
  ArgbBuffer original(WP2_ARGB_32);
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("alpha_ramp.png").c_str(),
                          &original));
  ASSERT_TRUE(EncodeDecodeExact(original, /*exact=*/true));
}

//------------------------------------------------------------------------------
// Exact settings and premultiplied modes combinations

ArgbBuffer CreateArgbBuffer(WP2SampleFormat format) {
  ArgbBuffer original(format);
  EXPECT_EQ(original.Resize(1, 1), WP2_STATUS_OK);
  const uint8_t kColor[] = {67, 66, 65, 64};
  original.Fill({0, 0, original.width(), original.height()}, kColor);
  return original;
}

class ExactFormatTest : public testing::TestWithParam<
                            std::tuple<WP2SampleFormat, /*exact=*/bool>> {};

TEST_P(ExactFormatTest, StillImageAPI) {
  const ArgbBuffer original = CreateArgbBuffer(std::get<0>(GetParam()));
  const bool exact = std::get<1>(GetParam());

  const bool is_truly_lossless = EncodeDecodeExact(original, exact);
  const bool uses_premultiplied_alpha_internally = !exact;
  const bool is_expected_to_be_truly_lossless =
      uses_premultiplied_alpha_internally ==
      WP2IsPremultiplied(original.format());
  ASSERT_EQ(is_truly_lossless, is_expected_to_be_truly_lossless);
}

TEST_P(ExactFormatTest, AnimationAPI) {
  const ArgbBuffer original = CreateArgbBuffer(std::get<0>(GetParam()));
  const bool exact = !std::get<1>(GetParam());

  AnimationEncoder animation_encoder;
  ASSERT_WP2_OK(animation_encoder.AddFrame(original, /*duration_ms=*/1000));
  ASSERT_WP2_OK(animation_encoder.AddFrame(original, /*duration_ms=*/1000));
  MemoryWriter memory_writer;
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 100.f;
  encoder_config.alpha_quality = 100.f;
  encoder_config.exact = exact;
  ASSERT_WP2_OK(animation_encoder.Encode(&memory_writer, encoder_config));

  DecoderConfig decoder_config = DecoderConfig::kDefault;
  decoder_config.exact = exact;
  ArrayDecoder decoder(memory_writer.mem_, memory_writer.size_, decoder_config);
  while (decoder.ReadFrame()) {
    const bool is_truly_lossless =
        testutil::Compare(original, decoder.GetPixels(), "internal");
    const bool uses_premultiplied_alpha_internally = !exact;
    const bool is_expected_to_be_truly_lossless =
        uses_premultiplied_alpha_internally ==
        WP2IsPremultiplied(original.format());
    ASSERT_EQ(is_truly_lossless, is_expected_to_be_truly_lossless);
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, ExactFormatTest,
    testing::Values(std::make_tuple(WP2_Argb_32, /*exact=*/false),
                    std::make_tuple(WP2_ARGB_32, /*exact=*/false),
                    std::make_tuple(WP2_Argb_32, /*exact=*/true),
                    std::make_tuple(WP2_ARGB_32, /*exact=*/true)));

}  // namespace
}  // namespace WP2
