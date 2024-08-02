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
//  Fuzzing of a wp2 bitstream and still image decoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <array>
#include <cstdint>

#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {
namespace {

::testing::Environment* const kStackLimitEnv = SetStackLimitTo256KiB();

void Decode(const std::string& bytes, DecoderConfig config) {
  const DataView encoded_data = {reinterpret_cast<const uint8_t*>(bytes.data()),
                                 bytes.size()};
  LimitConfigForBigImages(encoded_data, &config);

  ArgbBuffer decoded;
  // Success is not expected but it should not crash nor return a weird error.
  EXPECT_THAT(
      (std::array{WP2_STATUS_OK, WP2_STATUS_NOT_ENOUGH_DATA,
                  WP2_STATUS_BITSTREAM_ERROR, WP2_STATUS_OUT_OF_MEMORY,
                  // WP2_STATUS_UNSUPPORTED_FEATURE is returned for neural
                  // decoding without the correct compilation flags.
                  WP2_STATUS_UNSUPPORTED_FEATURE}),
      testing::Contains(
          Decode(encoded_data.bytes, encoded_data.size, &decoded, config)));
}

FUZZ_TEST(DecodeWebP2Anim, Decode)
    .WithDomains(ArbitraryImageWithSeeds(), ArbitraryDecoderConfig());

}  // namespace
}  // namespace testutil
}  // namespace WP2
