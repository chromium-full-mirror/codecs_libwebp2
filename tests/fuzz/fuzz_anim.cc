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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/utils/orientation.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {

int TestAnimEncConfig(const std::vector<ArgbBuffer>& original_frames,
                      const std::vector<uint32_t>& durations_ms,
                      bool loop_forever, const EncoderConfig& encoder_config,
                      float expected_distortion) {
  // std::cerr << encoder_config << std::endl;  // Uncomment to print config
  if (!encoder_config.IsValid()) abort();
  if (original_frames.empty()) return 0;

  // Add frames to the encoder.
  AnimationEncoder animation_encoder;
  for (size_t i = 0; i < original_frames.size(); ++i) {
    if (animation_encoder.AddFrame(original_frames[i], durations_ms[i]) !=
        WP2_STATUS_OK) {
      abort();
    }
  }

  // Encode with a reasonable config, without diving into detailed settings.
  MemoryWriter memory_writer;
  const WP2Status encoder_status =
      animation_encoder.Encode(&memory_writer, encoder_config, loop_forever);
  if (encoder_status != WP2_STATUS_OK) {
    if (encoder_config.progress_hook != nullptr &&
        encoder_status == WP2_STATUS_USER_ABORT) {
      return 0;
    }
    abort();
  }

  // Verify that it decodes fine and that the result is close enough.
  ArgbBuffer decoded(original_frames.front().format());
  ArgbBuffer decoded_unoriented(original_frames.front().format());
  DecoderConfig decoder_config = DecoderConfig::kDefault;
  // AnimationEncoder does not take EncoderConfig::exact into account.
  // It encodes the input frames in their original premul/unpremul state.
  // The decoder has no way to know that, so it must be told.
  decoder_config.exact = !WP2IsPremultiplied(original_frames.front().format());
  ArrayDecoder decoder(memory_writer.mem_, memory_writer.size_, decoder_config,
                       &decoded);
  for (size_t i = 0; i < original_frames.size(); ++i) {
    uint32_t duration_ms;
    if (!decoder.ReadFrame(&duration_ms)) abort();
    assert(!decoder.Failed());
    if (duration_ms != durations_ms[i]) abort();

    if (expected_distortion > 0.f) {
      // 'decoded' can't be modified as long as 'decoder' is used, so
      // 'decoded_unoriented' is compared instead.
      if (encoder_config.decoding_orientation == Orientation::kOriginal) {
        if (decoded_unoriented.SetView(decoded) != WP2_STATUS_OK) {
          abort();
        }
      } else if (OrientateBuffer(
                     GetInverseOrientation(encoder_config.decoding_orientation),
                     decoded, &decoded_unoriented) != WP2_STATUS_OK) {
        abort();
      }

      // Tiny frames are not interesting to compare.
      if (original_frames[i].width() >= 8 && original_frames[i].height() >= 8 &&
          !Compare(original_frames[i], decoded_unoriented, "anim",
                   expected_distortion)) {
        // Uncomment to save the original and decoded frames as PNG files.
        // SaveBeforeAfter(original_frames[i], decoded_unoriented,
        //                 "/tmp/comparison_" + std::to_string(i));
        abort();
      }
    }
  }
  if (decoder.GetStatus() != WP2_STATUS_OK) abort();
  return 0;
}

}  // namespace testutil
}  // namespace WP2
