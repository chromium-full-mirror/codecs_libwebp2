// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./bit_packing.h"

#include <cassert>
#include <cmath>

namespace container {

namespace {

uint32_t GetNumBitsNeeded(uint32_t value) {
  uint32_t index = 0;
  while (value > 0) {
    value >>= 1;
    ++index;
  }
  return index;
}

}  // namespace

//------------------------------------------------------------------------------

void EncodeVarInt(uint32_t value, uint32_t min, uint32_t max, Data& data) {
  assert(min <= value && value <= max);
  value -= min;
  max -= min;
  assert(max <= kMaxVarInt);
  if (max == 0) return;
  while (value > 0x7F && max > 0xFF) {
    // Use one bit per byte to signal whether another byte is needed.
    data.push_back((value & 0x7F) | 0x80);
    value >>= 7;
    max >>= 7;
  }
  data.push_back(value);
}

bool DecodeVarInt(uint32_t min, uint32_t max, const uint8_t data[],
                  size_t data_size, uint32_t& data_pos, uint32_t& value) {
  assert(min <= max && max - min <= kMaxVarInt);
  value = 0;
  uint32_t max_remaining_value = max - min;
  bool read_next_byte = (max_remaining_value > 0);
  for (uint32_t i = 0; read_next_byte; ++i, ++data_pos) {
    if (data_pos >= data_size) {
      return false;
    }
    const uint8_t byte = data[data_pos];
    read_next_byte = (max_remaining_value > 0xFF && (byte & 0x80));
    // If there is a next byte, use only 7 bits in current 'byte'.
    value |= static_cast<uint32_t>(read_next_byte ? (byte & 0x7F) : byte)
             << (7 * i);
    max_remaining_value >>= 7;
  }
  value += min;
  return (value <= max);
}

//------------------------------------------------------------------------------

void BitPacker::EncodeBool(bool value) {
  if (next_bit_mask_ == 0) {    // The last byte was completed.
    bytes_.push_back(0);        // Start a new byte.
    next_bit_mask_ = (1 << 7);  // Highest order in the new byte.
  }
  if (value) bytes_.back() |= next_bit_mask_;
  next_bit_mask_ >>= 1;
}

void BitPacker::EncodeUInt(uint32_t value, uint32_t num_bits) {
  assert(GetNumBitsNeeded(value) <= num_bits);
  for (uint32_t i = num_bits; i > 0; --i) {
    EncodeBool((value >> (i - 1)) & 1);
  }
}

void BitPacker::EncodeUInt(uint32_t value, uint32_t min, uint32_t max) {
  assert(value >= min && value <= max);
  EncodeUInt(value - min, GetNumBitsNeeded(max - min));
}

size_t BitPacker::GetNumBits() const {
  return bytes_.size() * 8 -
         GetNumBitsNeeded(static_cast<uint32_t>(next_bit_mask_));
}

//------------------------------------------------------------------------------

bool BitUnpacker::DecodeBool() {
  if (next_bit_mask_ == 0) {
    if (num_read_bytes_ >= num_bytes_) {
      error_ = true;
      return false;
    }
    next_bit_mask_ = (1 << 7);
    ++num_read_bytes_;
  }
  const bool value = (bytes_[num_read_bytes_ - 1] & next_bit_mask_) != 0;
  next_bit_mask_ >>= 1;
  return value;
}

uint32_t BitUnpacker::DecodeUInt(uint32_t num_bits) {
  uint32_t v = 0;
  for (uint32_t i = num_bits; i > 0; --i) {
    v |= (DecodeBool() ? (1 << (i - 1)) : 0);
  }
  return (error_ ? 0 : v);
}

uint32_t BitUnpacker::DecodeUInt(uint32_t min, uint32_t max) {
  assert(min <= max);
  return min + DecodeUInt(GetNumBitsNeeded(max - min));
}

//------------------------------------------------------------------------------

}  // namespace container
