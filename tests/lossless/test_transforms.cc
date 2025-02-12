// Copyright 2025 Google LLC
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

// Test several transforms.

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "src/common/lossless/transforms.h"
#include "src/enc/lossless/backward_references_enc.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "tests/include/helpers.h"

namespace WP2 {

class CrossColorGlobalTest
    : public testing::TestWithParam<std::tuple<
          WP2L::TransformHeader::Channel, WP2L::TransformHeader::Channel,
          WP2L::TransformHeader::CCTransform,
          WP2L::TransformHeader::CCTransform>> {};

// Tests the kCrossColorGlobal transform.
TEST_P(CrossColorGlobalTest, Pattern) {
  constexpr char kSource1_64x48[] = "source1_64x48.png";
  // First and second channel have to be different.
  if (std::get<0>(GetParam()) == std::get<1>(GetParam())) return;

  ArgbBuffer original(WP2_ARGB_32);
  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(kSource1_64x48).c_str(), &original));

  // Define the encoder config for lossless, including kCrossColorGlobal.
  EncoderConfig encoder_config;
  encoder_config.alpha_quality = 100;
  encoder_config.quality = 100;
  encoder_config.keep_unmultiplied = true;
  encoder_config.lossless_config =
      std::make_shared<WP2::EncoderConfig::LosslessCrunchConfig>();
  WP2L::CrunchConfig &config = *encoder_config.lossless_config;
  config.algorithm = WP2L::EncodingAlgorithm::kWebP;
  config.transform_bits = config.histo_bits = 5;
  config.lz77s_types_to_try[0] = WP2L::kLZ77Standard;
  config.lz77s_types_to_try_size = 1;
  config.transforms[0].type = WP2L::TransformType::kSubtractGreen;
  config.transforms[1].type = WP2L::TransformType::kPredictor;
  config.transforms[2].type = WP2L::TransformType::kCrossColorGlobal;
  config.transforms[2].cc_global_first_channel = std::get<0>(GetParam());
  config.transforms[2].cc_global_second_channel = std::get<1>(GetParam());
  config.transforms[2].cc_global_second_transform = std::get<2>(GetParam());
  config.transforms[2].cc_global_third_transform = std::get<3>(GetParam());

  MemoryWriter memory_writer;
  WP2_ASSERT_STATUS(Encode(original, &memory_writer, encoder_config));

  DecoderConfig decoder_config;
  ArgbBuffer decompressed(WP2_ARGB_32);
  WP2_ASSERT_STATUS(Decode(memory_writer.mem_, memory_writer.size_,
                           &decompressed, decoder_config));

  EXPECT_TRUE(testutil::Compare(original, decompressed,
                                /*file_name=*/"cross_color_global"));
}

INSTANTIATE_TEST_SUITE_P(
    TransformsTestInstantiation, CrossColorGlobalTest,
    testing::Combine(
        testing::Values(WP2L::TransformHeader::Channel::kRed,
                        WP2L::TransformHeader::Channel::kGreen,
                        WP2L::TransformHeader::Channel::kBlue),
        testing::Values(WP2L::TransformHeader::Channel::kRed,
                        WP2L::TransformHeader::Channel::kGreen,
                        WP2L::TransformHeader::Channel::kBlue),
        testing::Values(WP2L::TransformHeader::CCTransform::kNothing,
                        WP2L::TransformHeader::CCTransform::kSubtractFirst),
        testing::Values(WP2L::TransformHeader::CCTransform::kNothing,
                        WP2L::TransformHeader::CCTransform::kSubtractFirst,
                        WP2L::TransformHeader::CCTransform::kSubtractSecond,
                        WP2L::TransformHeader::CCTransform::kSubtractAverage)));

TEST(GenericTest, InvalidTransformCombinations) {
  // Create a 2x2 image with a single color.
  ArgbBuffer original(WP2_ARGB_32);
  constexpr uint32_t dim = 2;
  WP2_ASSERT_STATUS(original.Resize(dim, dim));
  original.Fill(Argb32b{.a = 255, .r = 26, .g = 4, .b = 79});

  // Create a config with a non-existent recipe.
  EncoderConfig encoder_config;
  encoder_config.alpha_quality = 100;
  encoder_config.quality = 100;
  encoder_config.lossless_config =
      std::make_shared<WP2::EncoderConfig::LosslessCrunchConfig>();
  WP2L::CrunchConfig &config = *encoder_config.lossless_config;
  config.algorithm = WP2L::EncodingAlgorithm::kGroup4;
  config.transforms[0].type = WP2L::TransformType::kNum;
  config.transforms[1].type = WP2L::TransformType::kNum;
  config.transforms[2].type = WP2L::TransformType::kNum;

  MemoryWriter memory_writer;
  ASSERT_EQ(Encode(original, &memory_writer, encoder_config),
            WP2_STATUS_INVALID_CONFIGURATION);

  // Make the recipe valid but that does not work because there is only one
  // color.
  config.transforms[0].type = WP2L::TransformType::kColorIndexing;

  ASSERT_EQ(Encode(original, &memory_writer, encoder_config),
            WP2_STATUS_INVALID_PARAMETER);
}

// Tests the kNormalizeChannels transform.
TEST(NormalizeChannelsTest, Pattern) {
  constexpr char kSource1_64x48[] = "source1_64x48.png";
  ArgbBuffer full_original(WP2_ARGB_32);
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath(kSource1_64x48).c_str(),
                          &full_original));
  for (int test_case = 0; test_case < 3; ++test_case) {
    ArgbBuffer original(WP2_ARGB_32);
    if (test_case == 0) {
      WP2_ASSERT_STATUS(original.SetView(full_original));
    } else if (test_case == 1) {
      // Trigger the non-palette code.
      WP2_ASSERT_STATUS(
          original.SetView(full_original, WP2::Rectangle(0, 0, 10, 10)));
    } else {
      // Remove the alpha channel.
      WP2_ASSERT_STATUS(original.Import(
          WP2_XRGB_32, 20, 20, full_original.GetRow8(0),
          full_original.stride()));
      ASSERT_TRUE(!original.HasTransparency());
    }

    // Define the encoder config for lossless, including kNormalizeChannels.
    EncoderConfig encoder_config;
    encoder_config.alpha_quality = 100;
    encoder_config.quality = 100;
    encoder_config.keep_unmultiplied = true;
    encoder_config.lossless_config =
        std::make_shared<WP2::EncoderConfig::LosslessCrunchConfig>();
    WP2L::CrunchConfig &config = *encoder_config.lossless_config;
    config.algorithm = WP2L::EncodingAlgorithm::kWebP;
    config.transform_bits = config.histo_bits = 5;
    config.lz77s_types_to_try[0] = WP2L::kLZ77Standard;
    config.lz77s_types_to_try_size = 1;
    config.transforms[0].type = WP2L::TransformType::kNormalizeChannels;
    config.transforms[1].type = WP2L::TransformType::kPredictor;
    config.transforms[2].type = WP2L::TransformType::kNum;

    MemoryWriter memory_writer;
    WP2_ASSERT_STATUS(Encode(original, &memory_writer, encoder_config));

    DecoderConfig decoder_config;
    ArgbBuffer decompressed(WP2_ARGB_32);
    WP2_ASSERT_STATUS(Decode(memory_writer.mem_, memory_writer.size_,
                             &decompressed, decoder_config));

    EXPECT_TRUE(testutil::Compare(original, decompressed,
                                  /*file_name=*/"cross_color_global"));
  }
}
}  // namespace WP2
