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

// Exercise frame skipping API.

#include <cstdint>

#include "include/helpers.h"
#include "include/helpers_filter.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

WP2Status CreateNoiseAnimation(uint32_t num_frames, Writer* const writer) {
  AnimationEncoder animation_encoder;
  ArgbBuffer frame;
  WP2_CHECK_STATUS(frame.Resize(32, 32));
  for (uint32_t i = 1; i <= num_frames; ++i) {
    frame.Fill(Argb32b{128, 128, 128, 128});
    testutil::Noise(/*seed=*/i, /*strength=*/128, &frame);
    WP2_CHECK_STATUS(animation_encoder.AddFrame(frame, /*duration_ms=*/i));
  }
  WP2_CHECK_STATUS(animation_encoder.Encode(writer));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

constexpr uint32_t kNumFrames = 32;

TEST(AnimTest, SkipSome) {
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(CreateNoiseAnimation(kNumFrames, &encoded_data));

  for (bool decode_frame_features_first : {false, true}) {
    ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_);

    if (decode_frame_features_first) {
      decoder.SkipNumNextFrames(kMaxNumFrames);
      ASSERT_FALSE(decoder.ReadFrame());
      ASSERT_FALSE(decoder.Failed());
      decoder.Rewind();
    }

    // Decode some frames.
    for (uint32_t i = 0; i <= 10; ++i) {
      ASSERT_TRUE(decoder.ReadFrame());
      ASSERT_EQ(decoder.GetCurrentFrameIndex(), i);
      ASSERT_EQ(decoder.GetDecodedArea(), Rectangle(0, 0, 32, 32));
    }

    // Skip some frames.
    decoder.SkipNumNextFrames(10);
    ASSERT_EQ(decoder.GetCurrentFrameIndex(), 20u);
    ASSERT_EQ(decoder.GetDecodedArea(), Rectangle(0, 0, 0, 0));
    // Read the next one.
    ASSERT_TRUE(decoder.ReadFrame());
    ASSERT_EQ(decoder.GetCurrentFrameIndex(), 20u);
    // Skip two.
    decoder.SkipNumNextFrames(1);
    ASSERT_EQ(decoder.GetCurrentFrameIndex(), 21u);
    decoder.SkipNumNextFrames(1);
    ASSERT_EQ(decoder.GetCurrentFrameIndex(), 22u);

    // Finish.
    for (uint32_t i = 22; i < kNumFrames; ++i) {
      ASSERT_TRUE(decoder.ReadFrame());
      ASSERT_EQ(decoder.GetCurrentFrameIndex(), i);
    }
    ASSERT_WP2_OK(decoder.GetStatus());
  }
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
