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
// -----------------------------------------------------------------------------

// Incremental animation decoding test.

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "include/helpers.h"
#include "include/helpers_incr.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

typedef std::tuple<std::vector<const char*>, std::vector<uint32_t>, float, int,
                   uint32_t, testutil::IncrementalDecodingTestSetup>
    Param;

class AnimTestIncr : public testing::TestWithParam<Param> {};

TEST_P(AnimTestIncr, Simple) {
  const std::vector<const char*>& file_names = std::get<0>(GetParam());
  const std::vector<uint32_t>& durations_ms = std::get<1>(GetParam());
  const float quality = std::get<2>(GetParam());
  const int effort = std::get<3>(GetParam());
  const uint32_t thread_level = std::get<4>(GetParam());
  testutil::IncrementalDecodingTestSetup setup = std::get<5>(GetParam());

  std::vector<ArgbBuffer> frames;
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(testutil::CompressAnimation(file_names, durations_ms,
                                            &encoded_data, &frames, quality,
                                            effort, thread_level));

  DecoderConfig config = DecoderConfig::kDefault;
  config.thread_level = thread_level;

  // Try incremental decoding with different steps.
  const DataView input = {encoded_data.mem_, encoded_data.size_};
  for (const size_t step : {(size_t)1, input.size / 20, input.size}) {
    setup.incr_size_step = step;
    std::vector<ArgbBuffer> decoded_frames;
    std::vector<uint32_t> decoded_durations_ms;
    ASSERT_WP2_OK(testutil::DecodeIncremental(
        config, input, setup, &decoded_frames, &decoded_durations_ms));

    // Hard to know which frames to compare if some were skipped.
    if (setup.actions.empty() ||
        setup.actions.back().type != testutil::DecoderAction::Type::kSkip) {
      ASSERT_EQ(decoded_frames.size(), frames.size());
      for (size_t i = 0; i < frames.size(); ++i) {
        EXPECT_TRUE(
            testutil::Compare(decoded_frames[i], frames[i], file_names[i],
                              testutil::GetExpectedDistortion(quality)));
        EXPECT_EQ(decoded_durations_ms[i], durations_ms[i]);
      }
    }
  }
}

constexpr bool kDoNotCompare = false;
constexpr bool kCompareWithSingleChunkDec = true;

// Input images are expected to be of equal sizes inside a Param.
INSTANTIATE_TEST_SUITE_P(
    AnimTestIncrInstantiation, AnimTestIncr,
    testing::Values(
        Param(
            {"source1_64x48.png"},
            /*frame durations (ms):*/ {1000},
            /*quality=*/0.0f, /*effort=*/4, /*thread_level=*/2,
            {1, testutil::DecoderType::kArray, WP2_Argb_32, kDoNotCompare, {}}),
        Param({"alpha_ramp.png", "alpha_ramp.webp"},
              /*frame durations (ms):*/ {50, 1},
              /*quality=*/100.0f, /*effort=*/0, /*thread_level=*/4,
              {1,
               testutil::DecoderType::kUnstableCustom,
               WP2_ARGB_32,
               kDoNotCompare,
               {}}),
        Param({"source1_64x48.png"},
              /*frame durations (ms):*/ {1000},
              /*quality=*/40.0f, /*effort=*/0, /*thread_level=*/0,
              {1,
               testutil::DecoderType::kSwappingCustom,
               WP2_rgbA_32,
               kDoNotCompare,
               {}})));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_AnimTestIncrInstantiation, AnimTestIncr,
    testing::Values(
        Param({"alpha_ramp.png", "alpha_ramp.lossy.webp", "alpha_ramp.webp"},
              /*frame durations (ms):*/ {50, 70, 1},
              /*quality=*/100.0f, /*effort=*/2, /*thread_level=*/0,
              {1,
               testutil::DecoderType::kUnstableCustom,
               WP2_RGBA_32,
               kDoNotCompare,
               {}}),
        Param({"source0.pgm", "source1.png", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {kMaxFrameDurationMs, 1, 20, 3},
              /*quality=*/75.0f, /*effort=*/4, /*thread_level=*/8,
              {1,
               testutil::DecoderType::kStream,
               WP2_XRGB_32,
               kDoNotCompare,
               {}}),
        Param({"source3.jpg", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {123, 456, 789},
              /*quality=*/25.0f, /*effort=*/3, /*thread_level=*/5,
              {1,
               testutil::DecoderType::kSwappingCustom,
               WP2_RGBX_32,
               kDoNotCompare,
               {}}),
        Param({"source0.pgm", "source3.jpg"},
              /*frame durations (ms):*/ {kMaxFrameDurationMs, 3},
              /*quality=*/75.0f, /*effort=*/2, /*thread_level=*/8,
              {1,
               testutil::DecoderType::kStream,
               WP2_bgrA_32,
               kDoNotCompare,
               {}})));

INSTANTIATE_TEST_SUITE_P(
    AnimTestIncrRewind, AnimTestIncr,
    testing::Values(Param({"source1_64x48.png"},
                          /*frame durations (ms):*/ {1000},
                          /*quality=*/0.0f, /*effort=*/4, /*thread_level=*/2,
                          {1,
                           testutil::DecoderType::kArray,
                           WP2_BGRA_32,
                           kDoNotCompare,
                           {{testutil::DecoderAction::Type::kRewind, 0, 0},
                            {testutil::DecoderAction::Type::kRewind, 100,
                             0}}})));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_AnimTestIncrRewind, AnimTestIncr,
    testing::Values(
        Param({"source1_64x48.png"},
              /*frame durations (ms):*/ {1},
              /*quality=*/100.0f, /*effort=*/7, /*thread_level=*/0,
              {1,
               testutil::DecoderType::kArray,
               WP2_BGRX_32,
               kCompareWithSingleChunkDec,
               {{testutil::DecoderAction::Type::kRewind, 100000000, 0}}}),
        Param({"test_exif_xmp.webp"},
              /*frame durations (ms):*/ {1},
              /*quality=*/33.0f, /*effort=*/0, /*thread_level=*/1,
              {1,
               testutil::DecoderType::kArray,
               WP2_Argb_32,
               kDoNotCompare,
               {{testutil::DecoderAction::Type::kSkip, 35, 1}}}),
        Param({"alpha_ramp.png", "alpha_ramp.lossy.webp", "alpha_ramp.webp"},
              /*frame durations (ms):*/ {50, 70, 1},
              /*quality=*/100.0f, /*effort=*/2, /*thread_level=*/0,
              {1,
               testutil::DecoderType::kUnstableCustom,
               WP2_RGB_24,
               kDoNotCompare,
               {{testutil::DecoderAction::Type::kSkip, 355, 1},
                {testutil::DecoderAction::Type::kRewindKeepBytes, 356, 0},
                {testutil::DecoderAction::Type::kSkip, 357, 0}}}),
        Param({"source0.pgm", "source1.png", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {kMaxFrameDurationMs, 1, 20, 3},
              /*quality=*/75.0f, /*effort=*/4, /*thread_level=*/8,
              {1,
               testutil::DecoderType::kStream,
               WP2_BGR_24,
               kDoNotCompare,
               {{testutil::DecoderAction::Type::kSkip, 15, kMaxNumFrames + 1},
                {testutil::DecoderAction::Type::kRewindKeepBytes, 12000, 1}}}),
        Param({"source0.pgm", "source1.png", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {5689, 21121, 52, 3},
              /*quality=*/50.0f, /*effort=*/5, /*thread_level=*/4,
              {1,
               testutil::DecoderType::kArray,
               WP2_Argb_32,
               kCompareWithSingleChunkDec,
               {{testutil::DecoderAction::Type::kSkip, 0, kMaxNumFrames},
                {testutil::DecoderAction::Type::kRewind, 100000000, 0},
                {testutil::DecoderAction::Type::kSkip, 0, 1},
                {testutil::DecoderAction::Type::kRewind,
                 kHeaderMaxSize + kANMFMaxHeaderSize + 1, 0},
                {testutil::DecoderAction::Type::kSkip, 0, kMaxNumFrames},
                {testutil::DecoderAction::Type::kRewind, 100000000, 0}}}),
        Param({"source3.jpg", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {123, 456, 789},
              /*quality=*/25.0f, /*effort=*/3, /*thread_level=*/5,
              {1,
               testutil::DecoderType::kSwappingCustom,
               WP2_Argb_38,
               kDoNotCompare,
               {{testutil::DecoderAction::Type::kSkip, 0, 2},
                {testutil::DecoderAction::Type::kRewind, 100000000, 1}}})));

//------------------------------------------------------------------------------

// Tests that skipping frames returns the same frame features as decoding all
// pixels.
TEST(AnimTestIncr, SkipAndCompareFeatures) {
  MemoryWriter writer;
  ASSERT_WP2_OK(testutil::CompressAnimation(
      {"source0.pgm", "source1.png", "source3.jpg", "source4.webp"},
      {50, 50, 50, 50}, &writer, nullptr, EncoderConfig::kDefault.quality,
      EncoderConfig::kDefault.effort, EncoderConfig::kDefault.thread_level,
      /*num_downsamplings=*/2));

  ArrayDecoder feature_decoder, frame_decoder;
  feature_decoder.SkipNumNextFrames(kMaxNumFrames);

  for (size_t size = 1; size <= writer.size_; ++size) {
    feature_decoder.SetInput(writer.mem_, size);
    frame_decoder.SetInput(writer.mem_, size);
    ASSERT_FALSE(feature_decoder.ReadFrame());
    frame_decoder.ReadFrame();
    ASSERT_FALSE(feature_decoder.Failed());
    ASSERT_FALSE(frame_decoder.Failed());

    // Byte-by-byte incremental decoding should give the same values.
    // Otherwise the 'frame_decoder' might halt after finishing a frame.
    ASSERT_EQ(feature_decoder.GetNumFrameDecodedFeatures(),
              frame_decoder.GetNumFrameDecodedFeatures());
    for (uint32_t i = 0; i < frame_decoder.GetNumFrameDecodedFeatures(); ++i) {
      const FrameFeatures* const frame_a =
          feature_decoder.TryGetFrameDecodedFeatures(i);
      const FrameFeatures* const frame_b =
          frame_decoder.TryGetFrameDecodedFeatures(i);
      ASSERT_NE(frame_a, nullptr);
      ASSERT_NE(frame_b, nullptr);
      ASSERT_EQ(frame_a->duration_ms, frame_b->duration_ms);
      ASSERT_EQ(frame_a->window, frame_b->window);
      ASSERT_EQ(frame_a->is_last, frame_b->is_last);
      ASSERT_EQ(frame_a->last_dispose_frame_index,
                frame_b->last_dispose_frame_index);

      size_t offset_a = 0, offset_b = 0, length_a = 0, length_b = 0;
      ASSERT_EQ(feature_decoder.TryGetFrameLocation(i, &offset_a),
                frame_decoder.TryGetFrameLocation(i, &offset_b));
      ASSERT_EQ(offset_a, offset_b);
      ASSERT_EQ(feature_decoder.TryGetFrameLocation(i, nullptr, &length_a),
                frame_decoder.TryGetFrameLocation(i, nullptr, &length_b));
      ASSERT_EQ(length_a, length_b);
    }
  }
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
