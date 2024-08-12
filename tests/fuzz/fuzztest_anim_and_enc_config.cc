// Copyright 2019 Google LLC
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
//  Fuzzing of raw pixels and animation wp2 encoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "gtest/gtest.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {
namespace {

::testing::Environment* const kStackLimitEnv = SetStackLimitTo256KiB();

WP2SampleFormat GetPremulFormat(WP2SampleFormat format) {
  switch (format) {
    case WP2_ARGB_32:
    case WP2_XRGB_32:
      return WP2_Argb_32;
    case WP2_RGBA_32:
    case WP2_RGBX_32:
      return WP2_rgbA_32;
    case WP2_BGRA_32:
    case WP2_BGRX_32:
      return WP2_bgrA_32;
    default:
      return WP2_Argb_32;
  }
}

//------------------------------------------------------------------------------

struct FramesAndDurations {
  // Cannot use a std::vector<ArgbBuffer> directly because of b/300630831.
  uint32_t width;
  uint32_t height;
  WP2SampleFormat format;
  std::vector<uint8_t> samples;

  std::vector<uint32_t> durations_ms;
};

void EncodeDecode(const FramesAndDurations& frames_and_durations,
                  EncoderConfig config, double progress_fail_at,
                  bool loop_forever) {
  // Make sure the encoding stops before the fuzzer timeout.
  ProgressTimeout progress(kEncodingMaxNumSeconds);
  config.progress_hook = &progress;
  progress.fail_at_ = progress_fail_at;

  const size_t num_frames = frames_and_durations.durations_ms.size();
  std::vector<ArgbBuffer> original_frames;
  uint8_t* samples = const_cast<uint8_t*>(frames_and_durations.samples.data());
  const uint32_t stride =
      WP2FormatBpp(frames_and_durations.format) * frames_and_durations.width;
  // Make sure the fuzzer generates enough samples as described by the domains
  // below (b/358211151).
  ASSERT_GE(frames_and_durations.samples.size(),
            stride * frames_and_durations.height * num_frames);

  original_frames.reserve(num_frames);
  for (size_t i = 0; i < num_frames; ++i) {
    original_frames.emplace_back(frames_and_durations.format);
    ASSERT_EQ(original_frames.back().SetExternal(frames_and_durations.width,
                                                 frames_and_durations.height,
                                                 samples, stride),
              WP2_STATUS_OK);
    samples += stride * frames_and_durations.height;
  }

  // Input can be random noise; output can get heavily distorted but it's fine.
  const float expected_distortion = GetExpectedDistortion(config) - 15.f;
  EXPECT_EQ(
      TestAnimEncConfig(original_frames, frames_and_durations.durations_ms,
                        loop_forever, config, expected_distortion),
      0);
}

FUZZ_TEST(EncodeDecodeWebP2Anim, EncodeDecode)
    .WithDomains(
        fuzztest::FlatMap(
            [](size_t num_frames) {
              return fuzztest::FlatMap(
                  [](uint32_t width, uint32_t height, WP2SampleFormat format,
                     size_t num_frames) {
                    return fuzztest::ConstructorOf<FramesAndDurations>(
                        fuzztest::Just(width), fuzztest::Just(height),
                        fuzztest::Just(format),
                        fuzztest::Arbitrary<std::vector<uint8_t>>().WithSize(
                            width * height * num_frames * WP2FormatBpp(format)),
                        fuzztest::VectorOf(
                            fuzztest::InRange<uint32_t>(1, kMaxFrameDurationMs))
                            .WithSize(num_frames));
                  },
                  /*width=*/fuzztest::InRange<uint32_t>(1, 128 / num_frames),
                  /*height=*/fuzztest::InRange<uint32_t>(1, 128 / num_frames),
                  fuzztest::ElementOf<WP2SampleFormat>({
                      WP2_ARGB_32, WP2_XRGB_32, WP2_RGBA_32, WP2_RGBX_32,
                      WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24, WP2_BGR_24
                      // TODO: b/311360339 - Generate valid premult samples:
                      // WP2_Argb_32, WP2_rgbA_32, WP2_bgrA_32, WP2_Argb_38,
                  }),
                  fuzztest::Just(num_frames));
            },
            /*num_frames=*/fuzztest::InRange<size_t>(1, 16)),
        ArbitraryEncoderConfig(), ArbitraryProgressFailAt(),
        /*loop_forever=*/fuzztest::Arbitrary<bool>());

}  // namespace
}  // namespace testutil
}  // namespace WP2
