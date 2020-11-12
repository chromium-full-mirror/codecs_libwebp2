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
#include <string>
#include <tuple>

#include "include/helpers.h"
#include "src/common/constants.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/common/progress_watcher.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class DecProgressTest : public ::testing::TestWithParam<
                            std::tuple<std::string, bool, uint32_t, bool>> {};

TEST_P(DecProgressTest, Simple) {
  const std::string src_file_name = std::get<0>(GetParam());
  const float quality = std::get<1>(GetParam()) ? 100.f : 75.f;
  const uint32_t thread_level = std::get<2>(GetParam());
  const bool use_decoder = std::get<3>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testing::CompressImage(src_file_name, &data, &src, quality));

  testing::ProgressTester progress_tester;

  ArgbBuffer output;
  DecoderConfig config;
  config.thread_level = thread_level;
  config.progress_hook = &progress_tester;
  ArrayDecoder decoder(config, &output);

  // Test that it starts at 0.
  for (size_t data_size = 0; data_size < kHeaderMinSize; ++data_size) {
    progress_tester.Reset();
    if (use_decoder) {
      decoder.SetInput(data.mem_, data_size);
      ASSERT_FALSE(decoder.ReadFrame());
      ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_NOT_ENOUGH_DATA);
    } else {
      ASSERT_EQ(Decode(data.mem_, data_size, &output, config),
                WP2_STATUS_BITSTREAM_ERROR);
    }
    ASSERT_EQ(progress_tester.last_progress_, 0.f);
  }

  // Test that it fails as expected.
  for (float fail_at :
       {0.f, kProgressDecHeader, 0.5f, 0.75f, 1.f - kProgressDecEnd, 0.9999f,
        1.f}) {
    progress_tester.Reset();
    progress_tester.fail_at_ = fail_at;
    if (use_decoder) {
      decoder.Rewind();
      decoder.SetInput(data.mem_, data.size_);
      ASSERT_FALSE(decoder.ReadFrame());
      ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_USER_ABORT);
    } else {
      ASSERT_EQ(Decode(data.mem_, data.size_, &output, config),
                WP2_STATUS_USER_ABORT);
    }
  }

  // Test that it ends at 1.
  progress_tester.Reset();
  if (use_decoder) {
    decoder.Rewind();
    decoder.SetInput(data.mem_, data.size_);
    ASSERT_TRUE(decoder.ReadFrame());
    ASSERT_WP2_OK(decoder.GetStatus());
    ASSERT_FALSE(decoder.ReadFrame());
  } else {
    ASSERT_WP2_OK(Decode(data.mem_, data.size_, &output, config));
  }
  ASSERT_EQ(progress_tester.last_progress_, 1.f);

  ASSERT_TRUE(testing::Compare(src, output, src_file_name,
                               testing::GetExpectedDistortion(quality)));
}

//------------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    ProgressTestInstantiationSingleTile, DecProgressTest,
    ::testing::Combine(
        ::testing::Values("source1_1x1.png",
                          "source1_64x48.png") /* one-tile image */,
        ::testing::Values(true) /* lossless */,
        ::testing::Values(0) /* no multithread; just one tile anyway */,
        ::testing::Values(false, true) /* use ArrayDecoder or Decode() */));

INSTANTIATE_TEST_SUITE_P(
    ProgressTestInstantiationMultiTile, DecProgressTest,
    ::testing::Combine(::testing::Values("source0.pgm") /* several tiles */,
                       ::testing::Values(false) /* lossy */,
                       ::testing::Values(4) /* extra threads */,
                       ::testing::Values(false, true) /* Decoder/Decode() */));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_ProgressTestInstantiation, DecProgressTest,
    ::testing::Combine(::testing::Values("source3.jpg", "alpha_ramp.png"),
                       ::testing::Values(true, false) /* lossless, lossy */,
                       ::testing::Values(0, 1, 64) /* extra threads */,
                       ::testing::Values(false, true) /* Decoder/Decode() */));

//------------------------------------------------------------------------------
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

  progress_stopper.fail_at_ = 0.f;
  ASSERT_EQ(DecodeArgb_32(encoded_data.data(), encoded_data.size(),
                          decoded_pixels.data(), stride, decoded_pixels.size(),
                          config),
            WP2_STATUS_USER_ABORT);
  ASSERT_EQ(
      Decode(encoded_data.data(), encoded_data.size(), &decoded_buffer, config),
      WP2_STATUS_USER_ABORT);
  ASSERT_FALSE(decoder.ReadFrame());
  ASSERT_EQ(decoder.GetStatus(), WP2_STATUS_USER_ABORT);

  progress_stopper.fail_at_ = 0.001f;
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

TEST(DecProgressTest, Precision) {
  ArgbBuffer big_source;
  ASSERT_WP2_OK(big_source.Resize(2048, 2048));
  big_source.Fill({130, 129, 128, 127});

  MemoryWriter encoded_bitstream;
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.effort = 0;
  ASSERT_WP2_OK(Encode(big_source, &encoded_bitstream, encoder_config));

  DecoderConfig decoder_config = DecoderConfig::kDefault;
  testing::ProgressTester progress_tester;
  decoder_config.progress_hook = &progress_tester;

  ArgbBuffer decoded_buffer;
  ASSERT_WP2_OK(Decode(encoded_bitstream.mem_, encoded_bitstream.size_,
                       &decoded_buffer, decoder_config));

  progress_tester.Reset();
  ArrayDecoder decoder(encoded_bitstream.mem_, encoded_bitstream.size_,
                       decoder_config);
  ASSERT_TRUE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());  // There is no trailing metadata.
  ASSERT_EQ(progress_tester.last_progress_, 1.f);
}

//------------------------------------------------------------------------------
// Test GetProgression() functions.

class ProgressionTest
    : public ::testing::TestWithParam<
          std::tuple<bool, bool, bool, uint32_t, uint32_t, uint32_t>> {};

TEST_P(ProgressionTest, Progression) {
  const bool has_preview = std::get<0>(GetParam());
  const bool has_icc = std::get<1>(GetParam());
  const bool has_trailing_data = std::get<2>(GetParam());
  const bool is_animation = (std::get<3>(GetParam()) == 0);
  const uint32_t num_frames = std::max(1u, std::get<3>(GetParam()));
  const uint32_t num_tiles_x = std::get<4>(GetParam());
  const uint32_t frame_height = std::get<5>(GetParam());

  float progress = 0.f;
  progress += GetProgressionBeforeFrames(has_preview, has_icc);
  for (uint32_t i = 0; i < num_frames; ++i) {
    const bool is_last = (i + 1 == num_frames);
    const float frame_progression = GetFrameProgression(
        has_preview, has_icc, has_trailing_data, i, is_last);

    const float anmf_progression =
        GetANMFProgression(has_preview, has_icc, has_trailing_data, i, is_last);
    const float row_progression =
        GetRowProgression(has_preview, has_icc, has_trailing_data, is_animation,
                          num_tiles_x, i, frame_height, is_last);
    EXPECT_NEAR(frame_progression,
                (is_animation ? anmf_progression : 0.f) +
                    frame_height * num_tiles_x * row_progression,
                0.00001f);

    progress += frame_progression;
  }
  progress += GetProgressionAfterFrames(has_trailing_data);
  EXPECT_NEAR(progress, 1.f, 0.001f);
}

INSTANTIATE_TEST_SUITE_P(
    ProgressionTestInstantiation, ProgressionTest,
    ::testing::Combine(
        ::testing::Values(false, true) /* has_preview */,
        ::testing::Values(false, true) /* has_icc */,
        ::testing::Values(false, true) /* has_trailing_data */,
        ::testing::Values(0, 1, 2, 5, kMaxNumFrames) /* num_frames */,
        ::testing::Values(1, 2, kImageDimMax / kMaxTileSize) /* num_tiles_x */,
        ::testing::Values(1, 2, 5, kImageDimMax) /* frame_height */));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
