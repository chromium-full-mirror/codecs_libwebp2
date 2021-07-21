// Copyright 2021 Google LLC
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

#ifndef CONTAINER_BIT_PACKING_H_
#define CONTAINER_BIT_PACKING_H_

#include <cstddef>
#include <cstdint>

#include "./container.h"

namespace container {

//------------------------------------------------------------------------------
// Write bit-packed values, from high to low order (~ big endian).

class BitPacker {
 public:
  BitPacker() = default;

  // Appends the 'value'.
  void EncodeBool(bool value);
  void EncodeUInt(uint32_t value, uint32_t num_bits);
  void EncodeUInt(uint32_t value, uint32_t min, uint32_t max);

  // Access to the encoded bytes.
  const uint8_t* GetBytes() const { return bytes_.data(); }
  size_t GetNumBytes() const { return bytes_.size(); }
  size_t GetNumBits() const;

 private:
  Data bytes_;
  uint8_t next_bit_mask_ = 0;  // Mask applied to 'bytes_.back()' to get the
                               // next bit to be written.
};

//------------------------------------------------------------------------------
// Read bit-packed values, from high to low order (~ big endian).

class BitUnpacker {
 public:
  // 'bytes' contain the bit-packed values to be decoded.
  BitUnpacker(const uint8_t bytes[], size_t num_bytes)
      : bytes_(bytes), num_bytes_(num_bytes) {}

  // Returns the decoded value.
  bool DecodeBool();
  uint32_t DecodeUInt(uint32_t num_bits);
  uint32_t DecodeUInt(uint32_t min, uint32_t max);

  // Access to the decoder status.
  bool Ok() const { return !error_; }
  size_t GetNumReadBytes() const { return num_read_bytes_; }

 private:
  const uint8_t* const bytes_;
  const size_t num_bytes_;
  size_t num_read_bytes_ = 0;  // Number of read 'bytes_', partial one included.
  size_t next_bit_mask_ = 0;   // Mask applied to the trailing byte of 'bytes_'
                               // to get the next bit to be read.
  bool error_ = false;
};

//------------------------------------------------------------------------------
// Write and read variable-length integers.

// Maximum number of bytes used by a variable-length integer.
static constexpr uint32_t kMaxVarIntLength = 4;
// Maximum value expressible with a variable-length integer (included).
static constexpr uint32_t kMaxVarInt =
    (1u << (7 * (kMaxVarIntLength - 1) + 8)) - 1u;  // 29 bits

void EncodeVarInt(uint32_t value, uint32_t min, uint32_t max, Data& data);

bool DecodeVarInt(uint32_t min, uint32_t max, const uint8_t data[],
                  size_t data_size, uint32_t& data_pos, uint32_t& value);

//------------------------------------------------------------------------------

}  // namespace container

#endif  // CONTAINER_BIT_PACKING_H_
