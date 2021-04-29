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

// Test a random image with alpha to make sure all lossless configs at maximum
// effort are dealt with properly.

#include <tuple>

#include "include/helpers.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class SlowLosslessTest
    : public testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(SlowLosslessTest, Lossless) {
  const uint32_t width = std::get<0>(GetParam());
  const uint32_t height = std::get<1>(GetParam());

  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 100.f;
  encoder_config.effort = 9;
  const DecoderConfig decoder_config = DecoderConfig::kDefault;

  ArgbBuffer original(WP2_ARGB_32);
  EXPECT_EQ(original.Resize(width, height), WP2_STATUS_OK);
  testutil::PrecalculatedRandom<65536u, uint8_t> rnd(0, 255);

  rnd.Fill(original.GetRow8(0), original.stride(), original.height(),
           original.stride());

  MemoryWriter memory_writer;
  WP2Status status = Encode(original, &memory_writer, encoder_config);
  EXPECT_EQ(status, WP2_STATUS_OK);

  ArgbBuffer decompressed(WP2_Argb_32);
  status = Decode(memory_writer.mem_, memory_writer.size_, &decompressed,
                  decoder_config);
  EXPECT_EQ(status, WP2_STATUS_OK);

  ArgbBuffer converted_original(decompressed.format());
  ASSERT_WP2_OK(converted_original.ConvertFrom(original));
  ASSERT_TRUE(testutil::Compare(converted_original, decompressed, "internal"));
}

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestWide, SlowLosslessTest,
                         testing::Values(std::make_tuple(256, 8)));

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestTall, SlowLosslessTest,
                         testing::Values(std::make_tuple(8, 256)));

INSTANTIATE_TEST_SUITE_P(SlowLosslessTestSquare, SlowLosslessTest,
                         testing::Values(std::make_tuple(32, 32)));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(DISABLED_SlowLosslessTestLarge, SlowLosslessTest,
                         testing::Values(std::make_tuple(256, 256)));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
