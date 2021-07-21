// Copyright 2021 Google LLC
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

#include "./bit_packing.h"

#include <limits>

#include "./testutil.h"

namespace container {
namespace {

constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();

//------------------------------------------------------------------------------

TEST(BitPackingTest, PackUnpack) {
  BitPacker packer;
  packer.EncodeBool(false);
  packer.EncodeBool(true);
  packer.EncodeUInt(5, /*num_bits=*/5);
  packer.EncodeUInt(1, /*min=*/0, /*max=*/1);
  packer.EncodeUInt(5, /*min=*/0, /*max=*/kMax);
  packer.EncodeUInt(kMax, /*min=*/kMax, /*max=*/kMax);
  packer.EncodeUInt(kMax, /*num_bits=*/32);
  ASSERT_EQ(packer.GetNumBits(), 1 + 1 + 5 + 1 + 32 + 0 + 32);

  BitUnpacker unpacker(packer.GetBytes(), packer.GetNumBytes());
  ASSERT_FALSE(unpacker.DecodeBool());
  ASSERT_TRUE(unpacker.DecodeBool());
  ASSERT_EQ(unpacker.DecodeUInt(/*num_bits=*/5), 5);
  ASSERT_EQ(unpacker.DecodeUInt(/*min=*/0, /*max=*/1), 1);
  ASSERT_EQ(unpacker.DecodeUInt(/*min=*/0, /*max=*/kMax), 5);
  ASSERT_EQ(unpacker.DecodeUInt(/*min=*/kMax, /*max=*/kMax), kMax);
  ASSERT_EQ(unpacker.DecodeUInt(/*num_bits=*/32), kMax);
  ASSERT_TRUE(unpacker.Ok());
  ASSERT_EQ(unpacker.GetNumReadBytes(), packer.GetNumBytes());
}

TEST(BitPackingTest, PackUnpackNotEnoughBytes) {
  BitPacker packer;
  packer.EncodeUInt(1023, /*min=*/0, /*max=*/1023);
  ASSERT_EQ(packer.GetNumBits(), 10);

  BitUnpacker unpacker(packer.GetBytes(), /*num_bytes=*/1);
  ASSERT_EQ(unpacker.DecodeUInt(/*num_bits=*/10), 0);
  ASSERT_FALSE(unpacker.Ok());
  ASSERT_EQ(unpacker.GetNumReadBytes(), 1);
}

//------------------------------------------------------------------------------

TEST(BitPackingTest, VariableLengthInteger) {
  Data data;
  EncodeVarInt(12, /*min=*/5, /*max=*/13, data);
  EncodeVarInt(0, /*min=*/0, /*max=*/0, data);
  EncodeVarInt(kMax - kMaxVarInt, /*min=*/kMax - kMaxVarInt, /*max=*/kMax,
               data);
  EncodeVarInt(kMaxVarInt, /*min=*/0, /*max=*/kMaxVarInt, data);
  EncodeVarInt(kMax, /*min=*/kMax, /*max=*/kMax, data);

  uint32_t data_pos = 0, value;
  ASSERT_TRUE(DecodeVarInt(/*min=*/5, /*max=*/13, data.data(), data.size(),
                           data_pos, value));
  ASSERT_EQ(value, 12);
  ASSERT_TRUE(DecodeVarInt(/*min=*/0, /*max=*/0, data.data(), data.size(),
                           data_pos, value));
  ASSERT_EQ(value, 0);
  ASSERT_TRUE(DecodeVarInt(/*min=*/kMax - kMaxVarInt, /*max=*/kMax, data.data(),
                           data.size(), data_pos, value));
  ASSERT_EQ(value, kMax - kMaxVarInt);
  ASSERT_TRUE(DecodeVarInt(/*min=*/0, /*max=*/kMaxVarInt, data.data(),
                           data.size(), data_pos, value));
  ASSERT_EQ(value, kMaxVarInt);
  ASSERT_TRUE(DecodeVarInt(/*min=*/kMax, /*max=*/kMax, data.data(), data.size(),
                           data_pos, value));
  ASSERT_EQ(value, kMax);
}

TEST(BitPackingTest, VariableLengthIntegerNotEnoughBytes) {
  Data data = {255};

  uint32_t data_pos = 0, value;
  ASSERT_FALSE(DecodeVarInt(/*min=*/0, /*max=*/2047, data.data(), data.size(),
                           data_pos, value));
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace container
