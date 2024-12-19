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
#include <string>

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
// Variable-length integer
// It is based on https://en.wikipedia.org/wiki/LEB128#Unsigned_LEB128
// with the removed pessimization described at
// https://en.wikipedia.org/wiki/Variable-length_quantity#Removing_redundancy

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
    --value;  // 'value' is known to be > 0, so avoid literally encoding a 0.
    max >>= 7;
    --max;
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
    // One bit is reserved to know if the next byte should be read, unless the
    // remaining value fits in this byte.
    read_next_byte = (max_remaining_value > 0xFF && (data[data_pos] & 0x80));
    // If 'read_next_byte', at least 1 remains, meaning storing 0 is pointless,
    // so add 1 to the next byte. The following is equivalent to keeping 7 bits
    // of this byte and adding 1 to the next byte.
    value += (uint32_t)data[data_pos] << (7 * i);
    max_remaining_value >>= 7;
    if (read_next_byte) --max_remaining_value;
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

size_t BitUnpacker::GetNumReadBits() const {
  return num_read_bytes_ * 8 -
         GetNumBitsNeeded(static_cast<uint32_t>(next_bit_mask_));
}

//------------------------------------------------------------------------------

void BitCoder::Code(const std::string& name, bool& field) {
  uint32_t value = field ? 1 : 0;
  Code(name, /*num_bits=*/1, value);
  field = (value == 1);
}

void BitCoder::Force(const std::string& name, bool value) {
  Force(name, /*num_bits=*/1, value ? 1 : 0);
}

void BitCoder::Set(const std::string& name, bool& field, bool value) {
  uint32_t uint_field = field ? 1 : 0;
  Set(name, uint_field, value ? 1 : 0);
  field = (uint_field == 1);
}

void BitCoder::CodeDelta(const std::string& name, uint32_t num_bits,
                         uint32_t& field) {
  bool delta_coded = (field != 0);
  Code(name, delta_coded);
  if (delta_coded) {
    Code(name, num_bits, field);
  } else {
    field = 0;  // Set() would be cleaner but it logs.
  }
}

//------------------------------------------------------------------------------

BitWriter::~BitWriter() {
  if (!Ok()) return;
  output_.insert(output_.end(), bit_packer_.GetBytes(),
                 bit_packer_.GetBytes() + bit_packer_.GetNumBytes());
}

void BitWriter::Code(const std::string& name, uint32_t num_bits,
                     uint32_t& field) {
  if (!Ok()) return;

  log_ << name << " = " << field << " (uint" << num_bits << ")\n";
  if (field < (1u << num_bits)) {
    bit_packer_.EncodeUInt(field, num_bits);
  } else {
    log_ << "/!\\ Bad value, out of bounds\n";
    error_ = true;
  }
}

void BitWriter::Force(const std::string& name, uint32_t num_bits,
                      uint32_t value) {
  if (!Ok()) return;

  log_ << name << " = " << value << " (uint" << num_bits << ", forced)\n";
  if (value < (1u << num_bits)) {
    bit_packer_.EncodeUInt(value, num_bits);
  } else {
    log_ << "/!\\ Bad value, out of bounds\n";
    error_ = true;
  }
}

void BitWriter::Set(const std::string& name, uint32_t& field, uint32_t value) {
  if (!Ok()) return;

  log_ << name << " = " << field << " (set)\n";
  if (field != value) {
    log_ << "/!\\ Bad value, expected " << value << "\n";
    error_ = true;
  }
}

void BitWriter::CodeOnesTillEqual(const std::string& name, uint32_t& field,
                                  uint32_t min, uint32_t max) {
  if (!Ok()) return;

  log_ << name << " = " << field << " (min " << min << ", max " << max << ")\n";
  if (field >= min && field <= max) {
    for (uint32_t i = min; i < max; ++i) {
      bit_packer_.EncodeBool(i != field);
      if (i == field) break;
    }
  } else {
    log_ << "/!\\ Bad value, out of bounds\n";
    error_ = true;
  }
}

void BitWriter::NonSymmetric(const std::string& name, uint32_t& field,
                             uint32_t max) {
  if (!Ok()) return;

  const uint32_t w = GetNumBitsNeeded(max + 1);
  const uint32_t m = (1 << w) - (max + 1);
  log_ << name << " = " << field << " (max " << max << ", "
       << ((field < m) ? w - 1 : w) << " bits)\n";
  if (field <= max) {
    if (field < m) {
      bit_packer_.EncodeUInt(field, /*num_bits=*/w - 1);
    } else {
      bit_packer_.EncodeUInt(m + ((field - m) >> 1), /*num_bits=*/w - 1);
      bit_packer_.EncodeBool((field - m) & 0x1);
    }
  } else {
    log_ << "/!\\ Bad value, out of bounds\n";
    error_ = true;
  }
}

void BitWriter::AppendBytes(const std::string& name, const uint8_t bytes[],
                            size_t num_bytes) {
  if (!Ok()) return;

  log_ << name << " = [" << num_bytes << " bytes]\n";
  for (size_t i = 0; i < num_bytes; ++i) {
    bit_packer_.EncodeUInt(bytes[i], /*num_bits=*/8);
  }
}

//------------------------------------------------------------------------------

void BitReader::Code(const std::string& name, uint32_t num_bits,
                     uint32_t& field) {
  if (!Ok()) {
    field = 0;
  } else {
    field = bit_unpacker_.DecodeUInt(num_bits);
    log_ << name << " = " << field << " (uint" << num_bits << ")\n";
  }
}

void BitReader::Force(const std::string& name, uint32_t num_bits,
                      uint32_t value) {
  if (!Ok()) return;

  uint32_t decoded_value = bit_unpacker_.DecodeUInt(num_bits);
  log_ << name << " = " << decoded_value << " (uint" << num_bits
       << ", forced)\n";
  if (value != decoded_value) {
    log_ << "/!\\ Bad value, expected " << value << "\n";
    error_ = true;
  }
}

void BitReader::Set(const std::string& name, uint32_t& field, uint32_t value) {
  if (!Ok()) return;

  field = value;
  log_ << name << " = " << field << " (set)\n";
}

void BitReader::CodeOnesTillEqual(const std::string& name, uint32_t& field,
                                  uint32_t min, uint32_t max) {
  if (!Ok()) {
    field = 0;
  } else {
    for (field = min; field < max; ++field) {
      if (!bit_unpacker_.DecodeBool()) break;
    }
    log_ << name << " = " << field << " (min " << min << ", max " << max
         << ")\n";
  }
}

void BitReader::NonSymmetric(const std::string& name, uint32_t& field,
                             uint32_t max) {
  if (!Ok()) {
    field = 0;
  } else {
    // See "AV1 Bitstream & Decoding Process Specification - 4.10.7."
    const uint32_t w = GetNumBitsNeeded(max + 1);
    const uint32_t m = (1 << w) - (max + 1);
    field = bit_unpacker_.DecodeUInt(/*num_bits=*/w - 1);
    if (field >= m) {
      field = (field << 1) - m + (bit_unpacker_.DecodeBool() ? 1 : 0);
    }
    log_ << name << " = " << field << " (max " << max << ", "
         << ((field < m) ? w - 1 : w) << " bits)\n";
  }
}

//------------------------------------------------------------------------------

}  // namespace container
