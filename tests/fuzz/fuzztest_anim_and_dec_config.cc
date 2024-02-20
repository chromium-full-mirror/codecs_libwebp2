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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "src/dec/tile_dec.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"
#include "tests/include/helpers_incr.h"

//------------------------------------------------------------------------------

namespace WP2 {
namespace testutil {
namespace {

auto ArbitraryDecoderAction(size_t max_bitstream_size) {
  return fuzztest::FlatMap(
      [](size_t max_bitstream_size, DecoderAction::Type type) {
        return fuzztest::ConstructorOf<DecoderAction>(
            fuzztest::Just(type),
            /*bistream_position=*/
            fuzztest::InRange<size_t>(0, max_bitstream_size),
            type == DecoderAction::Type::kSkip
                ?
                // Number of frames to skip.
                fuzztest::ElementOf<uint32_t>(
                    {0, 1, 2, 3, 4, 7, kMaxNumFrames - 1, kMaxNumFrames})
                :
                // Whether to keep the output buffer.
                fuzztest::ElementOf<uint32_t>({0, 1}));
      },
      fuzztest::Just(max_bitstream_size),
      fuzztest::ElementOf({DecoderAction::Type::kRewind,
                           DecoderAction::Type::kRewindKeepBytes,
                           DecoderAction::Type::kSkip}));
}

auto ArbitraryIncrementalDecodingTestSetup(size_t max_bitstream_size) {
  return fuzztest::ConstructorOf<IncrementalDecodingTestSetup>(
      /*step=*/
      fuzztest::InRange<uint32_t>(
          1, std::max((size_t)kHeaderMinSize, max_bitstream_size)),
      fuzztest::ElementOf({DecoderType::kArray, DecoderType::kStream,
                           DecoderType::kUnstableCustom,
                           DecoderType::kSwappingCustom}),
      fuzztest::ElementOf<WP2SampleFormat>(
          {WP2_Argb_32, WP2_ARGB_32, WP2_XRGB_32, WP2_rgbA_32, WP2_RGBA_32,
           WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24,
           WP2_BGR_24, WP2_Argb_38}),
      /*compare_with_single_chunk=*/fuzztest::Arbitrary<bool>(),
      fuzztest::VectorOf(ArbitraryDecoderAction(max_bitstream_size))
          .WithMaxSize(15));
}

struct BytesAndIncrementalDecodingTestSetup {
  std::string bytes;                   // A (possibly invalid) WebP2 bitstream.
  IncrementalDecodingTestSetup setup;  // Random decoding pattern
                                       // (when to rewind, which frames to skip)
};

auto ArbitraryBytesAndIncrementalDecodingTestSetup() {
  return fuzztest::FlatMap(
      [](const std::string& bytes) {
        return fuzztest::StructOf<BytesAndIncrementalDecodingTestSetup>(
            fuzztest::Just(bytes),
            ArbitraryIncrementalDecodingTestSetup(bytes.size()));
      },
      ArbitraryImageWithSeeds());
}

void LimitSetupForBigImages(DataView bitstream,
                            IncrementalDecodingTestSetup* const setup) {
  BitstreamFeatures features;
  if (features.Read(bitstream.bytes, bitstream.size) != WP2_STATUS_OK) return;

  // Too many tiles and/or too many decodings generate timeouts because heavy
  // memset()/memcpy() usage slows sanitizers down considerably.
  const uint32_t num_tiles =
      GetNumTiles(features.raw_width, features.raw_height, features.tile_width,
                  features.tile_height);
  const uint32_t num_decodings = DivCeil(bitstream.size, setup->incr_size_step);
  // The threshold was found empirically.
  const uint32_t max_num_tiles = DivRound(1024u, num_decodings);
  if (num_tiles > max_num_tiles) {
    setup->compare_with_single_chunk = false;
    if (setup->decoder_type == DecoderType::kSwappingCustom) {
      setup->decoder_type = DecoderType::kArray;
    }
  }
}

//------------------------------------------------------------------------------

void Decode(BytesAndIncrementalDecodingTestSetup bytes_and_setup,
            DecoderConfig config) {
  const DataView encoded_data = {
      reinterpret_cast<const uint8_t*>(bytes_and_setup.bytes.data()),
      bytes_and_setup.bytes.size()};
  LimitSetupForBigImages(encoded_data, &bytes_and_setup.setup);

  Data encoded_data_wrapper;  // TODO: b/311360339 - Remove.
  encoded_data_wrapper.bytes = const_cast<uint8_t*>(encoded_data.bytes);
  encoded_data_wrapper.size = bytes_and_setup.bytes.size();
  LimitConfigForBigImages(encoded_data_wrapper, &config);
  encoded_data_wrapper.bytes = nullptr;
  encoded_data_wrapper.size = 0;

  // Success is not expected but it should not crash or return a weird error.
  EXPECT_THAT(
      (std::array{WP2_STATUS_OK, WP2_STATUS_NOT_ENOUGH_DATA,
                  WP2_STATUS_BITSTREAM_ERROR, WP2_STATUS_OUT_OF_MEMORY,
                  // WP2_STATUS_UNSUPPORTED_FEATURE is returned for neural
                  // decoding without the correct compilation flags.
                  WP2_STATUS_UNSUPPORTED_FEATURE}),
      testing::Contains(
          DecodeIncremental(config, encoded_data, bytes_and_setup.setup)));
}

FUZZ_TEST(DecodeWebP2Anim, Decode)
    .WithDomains(ArbitraryBytesAndIncrementalDecodingTestSetup(),
                 ArbitraryDecoderConfig());

}  // namespace
}  // namespace testutil
}  // namespace WP2
