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
//
//  Fuzzing of a wp2 animated bitstream and decoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <vector>

#include "src/dec/tile_dec.h"
#include "src/dsp/math.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers_incr.h"

//------------------------------------------------------------------------------

namespace WP2 {
namespace testutil {
namespace {

class FuzzedParametersSetup : public FuzzedParameters {
 public:
  using FuzzedParameters::FuzzedParameters;

  IncrementalDecodingTestSetup ExtractSetup(size_t max_bitstream_size) {
    const size_t incr_size_step =
        ExtractUInt32(1, std::max((size_t)kHeaderMinSize, max_bitstream_size));
    const DecoderType decoder_type =
        (DecoderType)ExtractUInt32((uint32_t)DecoderType::kSwappingCustom);
    const WP2SampleFormat output_format =
        (WP2SampleFormat)ExtractUInt32((uint32_t)WP2_Argb_38);
    const bool compare_with_single_chunk = ExtractBool();

    std::vector<DecoderAction> actions;
    actions.reserve(ExtractUInt32(0, 15));
    for (uint32_t i = 0; i < actions.capacity(); ++i) {
      const DecoderAction::Type type = (DecoderAction::Type)ExtractUInt32(
          (uint32_t)DecoderAction::Type::kSkip);
      const size_t bistream_position = ExtractUInt32(0, max_bitstream_size);
      switch (type) {
        case DecoderAction::Type::kRewind:
        case DecoderAction::Type::kRewindKeepBytes: {
          const bool keep_output_buffer = ExtractBool();
          actions.emplace_back(type, bistream_position,
                               (uint32_t)keep_output_buffer);
          break;
        }
        case DecoderAction::Type::kSkip: {
          const uint32_t num_frames_to_skip =
              ExtractBool() ? ExtractUInt32(0, 7) : kMaxNumFrames;
          actions.emplace_back(type, bistream_position, num_frames_to_skip);
          break;
        }
      }
    }
    return IncrementalDecodingTestSetup(
        incr_size_step, decoder_type, output_format, compare_with_single_chunk,
        std::move(actions));
  }
};

void LimitSetupForBigImages(const Data& bitstream,
                            IncrementalDecodingTestSetup* const setup) {
  WP2::BitstreamFeatures features;
  if (features.Read(bitstream.bytes, bitstream.size) != WP2_STATUS_OK) return;

  // Too many tiles and/or too many decodings generate timeouts because heavy
  // memset()/memcpy() usage slows sanitizers down considerably.
  const uint32_t num_tiles =
      WP2::GetNumTiles(features.raw_width, features.raw_height,
                       features.tile_width, features.tile_height);
  const uint32_t num_decodings =
      WP2::DivCeil(bitstream.size, setup->incr_size_step);
  // The threshold was found empirically.
  const uint32_t max_num_tiles = WP2::DivRound(1024u, num_decodings);
  if (num_tiles > max_num_tiles) {
    setup->compare_with_single_chunk = false;
    if (setup->decoder_type == DecoderType::kSwappingCustom) {
      setup->decoder_type = DecoderType::kArray;
    }
  }
}

}  // namespace
}  // namespace testutil
}  // namespace WP2

//------------------------------------------------------------------------------

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Input is considered random data and converted to parameters.
  WP2::testutil::FuzzedParametersSetup params(data, size);

  // Extract a random decoding pattern (when to rewind, which frames to skip).
  WP2::testutil::IncrementalDecodingTestSetup setup = params.ExtractSetup(size);

  // Extract a random config.
  WP2::DecoderConfig config;
  params.ExtractConfig(&config);

  // Create a (possibly invalid) wp2 bitstream, starting with the wp2 signature.
  // It's best to extract other params before because it will try to use all
  // remaining bytes.
  WP2::Data encoded_data;
  if (!params.TryExtractWP2Bitstream(&encoded_data)) return 0;  // malloc fail.
  WP2::testutil::LimitSetupForBigImages(encoded_data, &setup);
  WP2::testutil::LimitConfigForBigImages(encoded_data, &config);

  // Success is not expected but it should not crash or return a weird error.
  const WP2Status status = WP2::testutil::DecodeIncremental(
      config, {encoded_data.bytes, encoded_data.size}, setup);
  if (status != WP2_STATUS_OK &&
      status != WP2_STATUS_NOT_ENOUGH_DATA &&
      status != WP2_STATUS_BITSTREAM_ERROR &&
      status != WP2_STATUS_OUT_OF_MEMORY &&
      status != WP2_STATUS_UNSUPPORTED_FEATURE) {
    // WP2_STATUS_UNSUPPORTED_FEATURE is returned for neural decoding without
    // the correct compilation flags.
    abort();
  }
  return 0;
}
