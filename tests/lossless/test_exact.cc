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
// Test exact compression (no pre-multiplication).
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include <cstdint>
#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "gtest/gtest.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

bool EncodeDecodeExact(const ArgbBuffer &original, bool lossless,
                       bool keep_unmultiplied) {
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = lossless ? 100.f : 95.f;
  encoder_config.alpha_quality = 100.f;
  encoder_config.keep_unmultiplied = keep_unmultiplied;

  MemoryWriter memory_writer;
  EXPECT_EQ(Encode(original, &memory_writer, encoder_config), WP2_STATUS_OK);

  ArgbBuffer decompressed(original.format());
  EXPECT_EQ(Decode(memory_writer.mem_, memory_writer.size_, &decompressed),
            WP2_STATUS_OK);

  const float expected_distortion = lossless ? 99.f : 30.f;
  return testutil::Compare(original, decompressed, "internal",
                           expected_distortion);
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

  ASSERT_TRUE(EncodeDecodeExact(original, /*lossless=*/true,
                                /*keep_unmultiplied=*/true));
}

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestWide, ExactTest,
                         testing::Values(std::make_tuple(1, 1),
                                         std::make_tuple(11, 12),
                                         std::make_tuple(31, 47)));

TEST(ExactTest, ReadFile) {
  ArgbBuffer original(WP2_ARGB_32);
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("alpha_ramp.png").c_str(),
                          &original));
  ASSERT_TRUE(EncodeDecodeExact(original, /*lossless=*/true,
                                /*keep_unmultiplied=*/true));
}

//------------------------------------------------------------------------------
// keep_unmultiplied values and premultiplied modes combinations

ArgbBuffer CreateArgbBuffer(WP2SampleFormat format) {
  ArgbBuffer original(format);
  EXPECT_EQ(original.Resize(32, 32), WP2_STATUS_OK);
  const uint8_t alpha = WP2IsPremultiplied(format) ? 67 : 0;
  const uint8_t color[] = {alpha, 66, 65, 64};
  original.Fill({0, 0, original.width(), original.height()}, color);
  return original;
}

class ExactFormatTest
    : public testing::TestWithParam<std::tuple<
          WP2SampleFormat, /*lossless=*/bool, /*keep_unmultiplied=*/bool>> {};

TEST_P(ExactFormatTest, StillImageAPI) {
  const ArgbBuffer original = CreateArgbBuffer(std::get<0>(GetParam()));
  const bool lossless = std::get<1>(GetParam());
  const bool keep_unmultiplied = std::get<2>(GetParam());

  const bool loss_is_small =
      EncodeDecodeExact(original, lossless, keep_unmultiplied);
  const bool loss_is_expected_to_be_small =
      WP2IsPremultiplied(original.format()) || keep_unmultiplied;
  ASSERT_EQ(loss_is_small, loss_is_expected_to_be_small);
}

TEST_P(ExactFormatTest, AnimationAPI) {
  const ArgbBuffer original = CreateArgbBuffer(std::get<0>(GetParam()));
  const bool lossless = std::get<1>(GetParam());
  const bool keep_unmultiplied = std::get<2>(GetParam());

  AnimationEncoder animation_encoder;
  ASSERT_WP2_OK(animation_encoder.AddFrame(original, /*duration_ms=*/1000));
  ASSERT_WP2_OK(animation_encoder.AddFrame(original, /*duration_ms=*/1000));
  MemoryWriter memory_writer;
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = lossless ? 100.f : 95.f;
  encoder_config.alpha_quality = 100.f;
  encoder_config.keep_unmultiplied = keep_unmultiplied;
  ASSERT_WP2_OK(animation_encoder.Encode(&memory_writer, encoder_config));

  ArrayDecoder decoder(memory_writer.mem_, memory_writer.size_);
  while (decoder.ReadFrame()) {
    const float expected_distortion = lossless ? 99.f : 30.f;
    const bool loss_is_small = testutil::Compare(
        original, decoder.GetPixels(), "internal", expected_distortion);
    const bool loss_is_expected_to_be_small =
        WP2IsPremultiplied(original.format()) || keep_unmultiplied;
    ASSERT_EQ(loss_is_small, loss_is_expected_to_be_small);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Premultiplied, ExactFormatTest,
    testing::Combine(testing::Values(WP2_Argb_32),
                     /*lossless=*/testing::Bool(),
                     /*keep_unmultiplied=*/testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Unmultiplied, ExactFormatTest,
    testing::Values(std::make_tuple(WP2_ARGB_32, /*lossless=*/false,
                                    /*keep_unmultiplied=*/false),
                    std::make_tuple(WP2_ARGB_32, /*lossless=*/true,
                                    /*keep_unmultiplied=*/false),
                    std::make_tuple(WP2_ARGB_32, /*lossless=*/true,
                                    /*keep_unmultiplied=*/true)));

}  // namespace
}  // namespace WP2
