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

// Decoding progress hook test.

#include <array>
#include <tuple>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/common/constants.h"
#include "src/common/progress_watcher.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class EncDecProgressTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, uint32_t>> {
};

constexpr double kFailurePoints[] = {0.,     kOnePercent, kProgressBeforeFrames,
                                     0.5,    0.75,        1. - kProgressEnd,
                                     0.9999, 1.};

// Encode() progression
TEST_P(EncDecProgressTest, Encode) {
  const char* const file_name = std::get<0>(GetParam());
  EncoderConfig encoder_config;
  encoder_config.quality = std::get<1>(GetParam()) ? 100.f : 75.f;
  encoder_config.thread_level = std::get<2>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(ReadImage(testing::GetTestDataPath(file_name).c_str(), &src));

  testing::ProgressTester progress_tester;
  encoder_config.progress_hook = &progress_tester;

  // Test that it fails as expected.
  for (float fail_at : kFailurePoints) {
    progress_tester.Reset();
    progress_tester.fail_at_ = fail_at;
    ASSERT_EQ(Encode(src, &data, encoder_config), WP2_STATUS_USER_ABORT);
  }

  // Test that it ends at 1.
  progress_tester.Reset();
  ASSERT_WP2_OK(Encode(src, &data, encoder_config));
  ASSERT_EQ(progress_tester.last_progress_, 1.);
}

// Decode() progression
TEST_P(EncDecProgressTest, Decode) {
  const char* const file_name = std::get<0>(GetParam());
  const float quality = std::get<1>(GetParam()) ? 100.f : 75.f;
  const uint32_t thread_level = std::get<2>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testing::CompressImage(file_name, &data, &src, quality));

  testing::ProgressTester progress_tester;

  ArgbBuffer output;
  DecoderConfig config;
  config.thread_level = thread_level;
  config.progress_hook = &progress_tester;
  ArrayDecoder decoder(config, &output);

  // Test that it starts at 0.
  for (size_t data_size = 0; data_size < kHeaderMinSize; ++data_size) {
    progress_tester.Reset();
    ASSERT_EQ(Decode(data.mem_, data_size, &output, config),
              WP2_STATUS_BITSTREAM_ERROR);
    ASSERT_EQ(progress_tester.last_progress_, 0.);
  }

  // Test that it fails as expected.
  for (double fail_at : kFailurePoints) {
    progress_tester.Reset();
    progress_tester.fail_at_ = fail_at;
    ASSERT_EQ(Decode(data.mem_, data.size_, &output, config),
              WP2_STATUS_USER_ABORT);
  }

  // Test that it ends at 1.
  progress_tester.Reset();
  ASSERT_WP2_OK(Decode(data.mem_, data.size_, &output, config));
  ASSERT_EQ(progress_tester.last_progress_, 1.);

  ASSERT_TRUE(testing::Compare(src, output, file_name,
                               testing::GetExpectedDistortion(quality)));
}

// Decoder progression (same as above but use the class API)
TEST_P(EncDecProgressTest, Decoder) {
  const char* const file_name = std::get<0>(GetParam());
  const float quality = std::get<1>(GetParam()) ? 100.f : 75.f;
  const uint32_t thread_level = std::get<2>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testing::CompressImage(file_name, &data, &src, quality));

  testing::ProgressTester progress_tester;

  ArgbBuffer output;
  DecoderConfig config;
  config.thread_level = thread_level;
  config.progress_hook = &progress_tester;
  ArrayDecoder decoder(config, &output);

  // Test that it starts at 0.
  for (size_t data_size = 0; data_size < kHeaderMinSize; ++data_size) {
    progress_tester.Reset();
    decoder.SetInput(data.mem_, data_size);
    ASSERT_FALSE(decoder.ReadFrame());
    ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_NOT_ENOUGH_DATA);
    ASSERT_EQ(progress_tester.last_progress_, 0.);
  }

  // Test that it fails as expected.
  for (double fail_at : kFailurePoints) {
    progress_tester.Reset();
    progress_tester.fail_at_ = fail_at;
    decoder.Rewind();
    decoder.SetInput(data.mem_, data.size_);
    ASSERT_FALSE(decoder.ReadFrame());
    ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_USER_ABORT);
  }

  // Test that it ends at 1.
  progress_tester.Reset();
  decoder.Rewind();
  decoder.SetInput(data.mem_, data.size_);
  ASSERT_TRUE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
  ASSERT_FALSE(decoder.ReadFrame());
  ASSERT_EQ(progress_tester.last_progress_, 1.);

  ASSERT_TRUE(testing::Compare(src, output, file_name,
                               testing::GetExpectedDistortion(quality)));
}

//------------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    ProgressTestInstantiationSingleTile, EncDecProgressTest,
    ::testing::Combine(
        ::testing::Values("source1_1x1.png",
                          "source1_64x48.png") /* one-tile image */,
        ::testing::Values(true) /* lossless */,
        ::testing::Values(0) /* no multithread; just one tile anyway */));

INSTANTIATE_TEST_SUITE_P(
    ProgressTestInstantiationMultiTile, EncDecProgressTest,
    ::testing::Combine(::testing::Values("source0.pgm") /* several tiles */,
                       ::testing::Values(false) /* lossy */,
                       ::testing::Values(4) /* extra threads */));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_ProgressTestInstantiation, EncDecProgressTest,
    ::testing::Combine(::testing::Values("source3.jpg", "alpha_ramp.png"),
                       ::testing::Values(true, false) /* lossless, lossy */,
                       ::testing::Values(0, 1, 64) /* extra threads */));

//------------------------------------------------------------------------------

// Test that WP2_STATUS_USER_ABORT is returned first, even with invalid or
// undefined pixels.
TEST(EncProgressTest, UserAbortBeforeAnyEncoding) {
  ArgbBuffer src(WP2_Argb_32);  // Premultiplied input.
  ASSERT_WP2_OK(src.Resize(16, 16));
  // First pixel is invalid (alpha < rgb), other pixels are undefined.
  for (int i : {0, 1, 2, 3}) src.GetRow8(0)[i] = 100 + 25 * i;

  testing::ProgressTester progress_tester;
  progress_tester.fail_at_ = 0.;
  EncoderConfig encoder_config;
  encoder_config.progress_hook = &progress_tester;
  MemoryWriter data;
  ASSERT_EQ(Encode(src, &data, encoder_config), WP2_STATUS_USER_ABORT);
}

// Test that WP2_STATUS_USER_ABORT is returned first, even with a failing
// bitstream.
TEST(DecProgressTest, UserAbortBeforeAnyDecoding) {
  std::array<uint8_t, 64> encoded_data = {(uint8_t)'u'};  // Junk

  DecoderConfig config = DecoderConfig::kDefault;
  testing::ProgressTester progress_stopper;
  config.progress_hook = &progress_stopper;

  std::array<uint8_t, encoded_data.size()> decoded_pixels{0};
  const uint32_t stride = encoded_data.size() / 2;
  ArgbBuffer decoded_buffer;
  ArrayDecoder decoder(encoded_data.data(), encoded_data.size(), config);

  progress_stopper.fail_at_ = 0.;
  ASSERT_EQ(DecodeArgb_32(encoded_data.data(), encoded_data.size(),
                          decoded_pixels.data(), stride, decoded_pixels.size(),
                          config),
            WP2_STATUS_USER_ABORT);
  ASSERT_EQ(
      Decode(encoded_data.data(), encoded_data.size(), &decoded_buffer, config),
      WP2_STATUS_USER_ABORT);
  ASSERT_FALSE(decoder.ReadFrame());
  ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_USER_ABORT);

  progress_stopper.fail_at_ = 0.001;
  ASSERT_EQ(DecodeArgb_32(encoded_data.data(), encoded_data.size(),
                          decoded_pixels.data(), stride, decoded_pixels.size(),
                          config),
            WP2_STATUS_BITSTREAM_ERROR);
  ASSERT_EQ(
      Decode(encoded_data.data(), encoded_data.size(), &decoded_buffer, config),
      WP2_STATUS_BITSTREAM_ERROR);
  decoder.Rewind();
  ASSERT_FALSE(decoder.ReadFrame());
  ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_BITSTREAM_ERROR);
}

//------------------------------------------------------------------------------
// Test progress precision with a lot of pixels.

class ProgressPrecisionTest : public ::testing::TestWithParam<
                                  std::tuple<uint32_t, uint32_t, bool, float>> {
};

TEST_P(ProgressPrecisionTest, Encode) {
  const uint32_t width = std::get<0>(GetParam());
  const uint32_t height = std::get<1>(GetParam());
  const uint8_t alpha = std::get<2>(GetParam()) ? 130 : 255;
  const float quality = std::get<3>(GetParam());

  ArgbBuffer big_source;
  ASSERT_WP2_OK(big_source.Resize(width, height));
  big_source.Fill({alpha, 129, 128, 127});

  MemoryWriter encoded_bitstream;
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  testing::ProgressTester progress_tester;
  encoder_config.progress_hook = &progress_tester;
  encoder_config.effort = 0;
  encoder_config.quality = quality;
  ASSERT_WP2_OK(Encode(big_source, &encoded_bitstream, encoder_config));

  DecoderConfig decoder_config = DecoderConfig::kDefault;
  decoder_config.progress_hook = &progress_tester;
  progress_tester.Reset();

  ArgbBuffer decoded_buffer;
  ASSERT_WP2_OK(Decode(encoded_bitstream.mem_, encoded_bitstream.size_,
                       &decoded_buffer, decoder_config));

  progress_tester.Reset();
  ArrayDecoder decoder(encoded_bitstream.mem_, encoded_bitstream.size_,
                       decoder_config);
  ASSERT_TRUE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());  // There is no trailing metadata.
  ASSERT_EQ(progress_tester.last_progress_, 1.);
}

INSTANTIATE_TEST_SUITE_P(
    ProgressPrecisionTestInstantiation, ProgressPrecisionTest,
    ::testing::Combine(::testing::Values(1, 2048),    // width
                       ::testing::Values(1, 2048),    // height
                       ::testing::Values(true),       // has_alpha
                       ::testing::Values(25.f, 96.f)  // quality
                       ));

//------------------------------------------------------------------------------

TEST(ProgressionTest, GetFrameProgression) {
  for (uint32_t num_frames : {1u, 2u, 5u, kMaxNumFrames}) {
    float progress = kProgressBeforeFrames;
    for (uint32_t i = 0; i < num_frames; ++i) {
      const bool is_last = (i + 1 == num_frames);
      progress += GetFrameProgression(i, is_last);
    }
    progress += kProgressEnd;
    EXPECT_NEAR(progress, 1.f, 0.00001f);
  }

  ASSERT_NEAR(kProgressBeforeFrames + kProgressFrames, 1.f - kProgressEnd,
              0.00001f);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
