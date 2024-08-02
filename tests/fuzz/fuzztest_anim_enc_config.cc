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
//  Fuzzing of animation wp2 encoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "imageio/anim_image_dec.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {
namespace {

::testing::Environment* const kStackLimitEnv = SetStackLimitTo256KiB();

//------------------------------------------------------------------------------

void ReadEncodeDecode(const std::string& file_name, EncoderConfig config,
                      double progress_fail_at) {
  const Data data = ReadTestImage(file_name);
  // Read a randomly chosen valid image file (possibly an animation).
  std::vector<ArgbBuffer> original_frames;
  std::vector<uint32_t> durations_ms;
  uint32_t loop_count = 1;
  ASSERT_EQ(ReadAnimation(data.bytes, data.size, &original_frames,
                          &durations_ms, &loop_count),
            WP2_STATUS_OK);
  const bool loop_forever = (loop_count != 1);

  // Encode up to ~1M pixels to avoid timeouts.
  const size_t kMaxNumPixels = (1u << 20);
  const size_t num_pixels_per_frame =
      original_frames.front().width() * original_frames.front().height();
  const size_t max_num_frames = 1u + kMaxNumPixels / num_pixels_per_frame;
  if (original_frames.size() > max_num_frames) {
    original_frames.resize(max_num_frames);
    durations_ms.resize(max_num_frames);
  }

  // Make sure durations are valid for wp2 (to keep the same number of frames).
  for (size_t i = 0; i < original_frames.size(); ++i) {
    if (durations_ms[i] == ImageReader::kInfiniteDuration ||
        durations_ms[i] > kMaxFrameDurationMs) {
      durations_ms[i] = kMaxFrameDurationMs;
    }
  }

  // Make sure the encoding stops before the fuzzer timeout.
  ProgressTimeout progress(kEncodingMaxNumSeconds);
  config.progress_hook = &progress;
  progress.fail_at_ = progress_fail_at;

  const float expected_distortion = GetExpectedDistortion(config) - 10.f;
  EXPECT_EQ(TestAnimEncConfig(original_frames, durations_ms, loop_forever,
                              config, expected_distortion),
            0);
}

FUZZ_TEST(ReadEncodeDecodeAnim, ReadEncodeDecode)
    .WithDomains(fuzztest::ElementOf(GetTestImageNames()),
                 ArbitraryEncoderConfig(), ArbitraryProgressFailAt());

}  // namespace
}  // namespace testutil
}  // namespace WP2
