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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "include/helpers.h"
#include "src/common/constants.h"
#include "src/common/header_enc_dec.h"
#include "src/utils/data_source.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

TEST(VarIntTest, Zero) {
  // 0 bit written, 0 bit read.
  uint8_t dst[kMaxVarIntLength];
  ASSERT_EQ(WriteVarInt(/*value=*/0, /*min_value=*/0, /*max_value=*/0, dst),
            0u);
  ExternalDataSource data_source(dst, 0);
  uint32_t decoded_value;
  ASSERT_WP2_OK(TryReadVarInt(&data_source, /*min_value=*/0, /*max_value=*/0,
                              &decoded_value));
  ASSERT_EQ(decoded_value, 0u);
  ASSERT_EQ(data_source.GetNumReadBytes(), 0u);
}

TEST(VarIntTest, NumberOfEncodedBytes) {
  constexpr uint32_t kMin = 0;
  uint8_t dst[kMaxVarIntLength];
  ASSERT_EQ(WriteVarInt(/*value=*/0, kMin, /*max_value=*/1, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/1, kMin, /*max_value=*/1, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/127, kMin, /*max_value=*/255, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/128, kMin, /*max_value=*/255, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/127, kMin, /*max_value=*/256, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/128, kMin, /*max_value=*/256, dst), 2u);
  ASSERT_EQ(WriteVarInt(/*value=*/255, kMin, /*max_value=*/256, dst), 2u);
  ASSERT_EQ(WriteVarInt(/*value=*/256, kMin, /*max_value=*/256, dst), 2u);
}

TEST(VarIntTest, EncodedGreaterValueThanMaxDecoded) {
  uint8_t dst[kMaxVarIntLength];
  const uint32_t num_bytes =
      WriteVarInt(/*value=*/200, /*min_value=*/10, /*max_value=*/202, dst);
  ExternalDataSource data_source(dst, num_bytes);
  uint32_t decoded_value;
  ASSERT_EQ(TryReadVarInt(&data_source, /*min_value=*/10, /*max_value=*/198,
                          &decoded_value),
            WP2_STATUS_BITSTREAM_ERROR);
}

//------------------------------------------------------------------------------

TEST(VarIntTest, EncodeDecode) {
  // clang-format off
  const uint32_t kValues[] =
      {0, 1, 2, 3, 5, 8, 126, 127, 128, 129, 254, 255, 256, 257, 260768,
       (kMaxVarInt >> 7) - 1, (kMaxVarInt >> 7), (kMaxVarInt >> 7) + 1,
       kMaxVarInt - 2, kMaxVarInt - 1, kMaxVarInt};
  // clang-format on

  for (size_t min_value : kValues) {
    for (size_t max_value : kValues) {
      if (min_value > max_value) break;
      for (size_t value : kValues) {
        if (value < min_value || value > max_value) break;

        uint8_t dst[kMaxVarIntLength];
        const uint32_t num_bytes =
            WriteVarInt(value, min_value, max_value, dst);
        ASSERT_GE(num_bytes, (min_value == max_value) ? 0u : 1u);
        ASSERT_LE(num_bytes, GetMaxVarIntLength(min_value, max_value));

        ExternalDataSource data_source(dst, num_bytes);
        uint32_t decoded_value;
        ASSERT_WP2_OK(
            TryReadVarInt(&data_source, min_value, max_value, &decoded_value));
        ASSERT_EQ(decoded_value, value);

        for (uint32_t not_enough = 0; not_enough < num_bytes; ++not_enough) {
          ExternalDataSource shorter_data_source(dst, not_enough);
          ASSERT_EQ(TryReadVarInt(&shorter_data_source, min_value, max_value,
                                  &decoded_value),
                    WP2_STATUS_NOT_ENOUGH_DATA);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

// Makes sure variable integer encoding is bijective and monotonic.
// DISABLED because it is slow.
TEST(DISABLED_VarIntTest, Pessimization) {
  uint32_t num_values = 0;
  for (uint64_t buffer = 0; buffer <= std::numeric_limits<uint32_t>::max();
       ++buffer) {
    // Skip invalid values (more bytes defined than read).
    if (buffer & ~0xFFu) buffer |= 0x80u;
    if (buffer & ~0xFFFFu) buffer |= 0x8000u;
    if (buffer & ~0xFFFFFFu) buffer |= 0x800000u;

    // Map 'buffer' to a raw byte buffer.
    ExternalDataSource data_source(reinterpret_cast<uint8_t *>(&buffer),
                                   sizeof(buffer));
    uint32_t decoded_value;
    ASSERT_WP2_OK(TryReadVarInt(&data_source, /*min_value=*/0, kMaxVarInt,
                                &decoded_value));
    ASSERT_EQ(decoded_value, num_values);
    ++num_values;
  }
  ASSERT_EQ(num_values, kMaxVarInt + 1);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
