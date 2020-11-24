// Copyright 2019 Google LLC
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
// -----------------------------------------------------------------------------
//
// Bitstream/chunk header reader and writer.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/header_enc_dec.h"

namespace WP2 {

//------------------------------------------------------------------------------

static_assert(GetMaxVarIntLength(0, kMaxVarInt) == kMaxVarIntLength, "Error");

WP2Status TryReadVarInt(DataSource* const data_source, uint32_t min_value,
                        uint32_t max_value, uint32_t* const value) {
  assert(min_value <= max_value);
  *value = 0;
  uint32_t num_bytes = 0;
  uint32_t max_remaining_value = max_value - min_value;
  assert(max_remaining_value <= kMaxVarInt);
  bool read_next_byte = (max_remaining_value > 0);
  while (read_next_byte) {
    const uint32_t i = num_bytes++;         // Current byte index.
    assert(num_bytes <= kMaxVarIntLength);  // Otherwise 'max_value>kMaxVarInt'.
    const uint8_t* bytes;
    WP2_CHECK_OK(data_source->TryGetNext(num_bytes, &bytes),
                 WP2_STATUS_NOT_ENOUGH_DATA);

    const uint8_t byte = bytes[i];  // Current byte. Next byte is 'bytes[i+1]'.
    read_next_byte = (max_remaining_value > 0xFF && (byte & 0x80));
    // If there is a next byte, use only 7 bits in current 'byte'.
    *value |= (uint32_t)(read_next_byte ? (byte & 0x7F) : byte) << (7 * i);
    max_remaining_value >>= 7;
  }
  *value += min_value;
  WP2_CHECK_OK(*value <= max_value, WP2_STATUS_BITSTREAM_ERROR);
  data_source->MarkNumBytesAsRead(num_bytes);
  return WP2_STATUS_OK;
}

uint32_t WriteVarInt(uint32_t value, uint32_t min_value, uint32_t max_value,
                     uint8_t dst[]) {
  assert(min_value <= value && value <= max_value);
  value -= min_value;      // Get rid of 'min_value'.
  max_value -= min_value;  // Consider 'value' in [0:max_value].
  assert(max_value <= kMaxVarInt);

  uint32_t num_used_bytes = 0;
  if (max_value == 0) return num_used_bytes;
  while (value > 0x7F && max_value > 0xFF) {
    dst[num_used_bytes++] = (value & 0x7F) | 0x80;
    value >>= 7;
    max_value >>= 7;
  }
  dst[num_used_bytes++] = value;
  return num_used_bytes;
}

//------------------------------------------------------------------------------

uint32_t HeaderDec::ReadBits(uint32_t num_bits, WP2_OPT_LABEL) {
  const uint32_t value = ReadBitsInternal(num_bits);
  WP2Trace("%s: value=%u num_bits=%d", debug_prefix_, label, value, num_bits);
  return value;
}

int32_t HeaderDec::ReadSBits(uint32_t num_bits, WP2_OPT_LABEL) {
  const int32_t value =
      (int32_t)ReadBitsInternal(num_bits) - (1 << (num_bits - 1));
  WP2Trace("%s: value=%d num_bits=%d (signed)", debug_prefix_, label, value,
           num_bits);
  return value;
}

uint32_t HeaderDec::ReadVarUInt(uint32_t min_value, uint32_t max_value,
                                WP2_OPT_LABEL) {
  assert(min_value <= max_value);
  if (status_ != WP2_STATUS_OK || min_value == max_value) return 0;
  assert(bit_pos_ == 8);  // Not-aligned variable-length ints are not supported.
  uint32_t value;
  status_ = TryReadVarInt(data_source_, min_value, max_value, &value);
  WP2Trace("%s: value=%u in [%u:%u]", debug_prefix_, label, value, min_value,
           max_value);
  // No need to update 'byte_' or 'bit_pos_' because whole bytes were read.
  return (status_ == WP2_STATUS_OK) ? value : 0;
}

Rectangle HeaderDec::ReadRect(uint32_t canvas_width, uint32_t canvas_height,
                              WP2_OPT_LABEL) {
  assert(canvas_width > 0 && canvas_height > 0);
  // Read reversed width and height first, as these tend to be near 'canvas_*'.
  Rectangle rect;
  rect.width = canvas_width - ReadVarUInt(0, canvas_width - 1, label);
  rect.height = canvas_height - ReadVarUInt(0, canvas_height - 1, label);
  rect.x = ReadVarUInt(0, canvas_width - rect.width, label);
  rect.y = ReadVarUInt(0, canvas_height - rect.height, label);
  return rect;
}

uint32_t HeaderDec::ReadBitsInternal(uint32_t num_bits) {
  assert(num_bits <= 24);
  if (status_ != WP2_STATUS_OK || num_bits == 0) return 0;

  assert(bit_pos_ > 0 && bit_pos_ <= 8);
  // (8 - bit_pos_) is the number of not-yet-used bits in 'byte_'.
  // Number of new bytes to read in order to have at least 'num_bits':
  //   num_bytes_to_read = ceil((num_bits - (8 - bit_pos_)) / 8)
  const uint32_t num_bytes_to_read = (bit_pos_ + num_bits - 1) >> 3;
  const uint8_t* bytes;
  if (!data_source_->TryReadNext(num_bytes_to_read, &bytes)) {
    status_ = WP2_STATUS_NOT_ENOUGH_DATA;
    return 0;
  }

  // Concatenate all bytes containing at least some of the 'num_bits'.
  uint32_t value = byte_ >> bit_pos_;  // Retrieve unused bits, if any.
  uint32_t shift = 8 - bit_pos_;  // Next bits are inserted as most significant.
  for (uint32_t i = 0; i < num_bytes_to_read; ++i, shift += 8) {
    value |= (uint32_t)bytes[i] << shift;
    byte_ = bytes[i];  // Remember last read byte.
  }
  value &= ((1u << num_bits) - 1u);      // Remove the unrelated bits.
  bit_pos_ = (bit_pos_ + num_bits) & 7;  // Remember the used bits of 'byte_'.
  if (bit_pos_ == 0) bit_pos_ = 8;  // If so, set the whole 'byte_' as used.
  return value;
}

//------------------------------------------------------------------------------

void HeaderEnc::PutBits(uint32_t value, uint32_t num_bits, WP2_OPT_LABEL) {
  WP2Trace("%s: value=%u num_bits=%u", debug_prefix_, label, value, num_bits);
  PutBitsInternal(value, num_bits);
}

void HeaderEnc::PutSBits(int32_t value, uint32_t num_bits, WP2_OPT_LABEL) {
  const int32_t max_abs_value = (1 << (num_bits - 1));
  assert(value >= -max_abs_value && value < max_abs_value);
  WP2Trace("%s: value=%d num_bits=%u (signed)", debug_prefix_, label, value,
           num_bits);
  PutBitsInternal((uint32_t)(value + max_abs_value), num_bits);
}

void HeaderEnc::PutVarUInt(uint32_t value, uint32_t min_value,
                           uint32_t max_value, WP2_OPT_LABEL) {
  assert(min_value <= value && value <= max_value);
  // Make sure any 'value' would fit in 'buf_'.
  if (((max_pos_ - bit_pos_ + 7) >> 3) <
      GetMaxVarIntLength(min_value, max_value)) {
    error_ = true;
  } else {
    assert((bit_pos_ & 7) == 0);
    WP2Trace("%s: value=%u in [%u:%u]", debug_prefix_, label, value, min_value,
             max_value);
    const size_t num_written_bytes =
        WriteVarInt(value, min_value, max_value, buf_ + (bit_pos_ >> 3));
    bit_pos_ += num_written_bytes << 3;
  }
}

void HeaderEnc::PutRect(const Rectangle& rect, uint32_t canvas_width,
                        uint32_t canvas_height, WP2_OPT_LABEL) {
  assert(rect.width > 0 && rect.height > 0);
  assert(rect.x + rect.width <= canvas_width);
  assert(rect.y + rect.height <= canvas_height);
  // Put reversed width and height first, as these tend to be near 'canvas_*'.
  PutVarUInt(canvas_width - rect.width, 0, canvas_width - 1, label);
  PutVarUInt(canvas_height - rect.height, 0, canvas_height - 1, label);
  PutVarUInt(rect.x, 0, canvas_width - rect.width, label);
  PutVarUInt(rect.y, 0, canvas_height - rect.height, label);
}

uint8_t* HeaderEnc::Align() {
  bit_pos_ = (bit_pos_ + 7) & ~7;
  return buf_ + (bit_pos_ >> 3);
}

void HeaderEnc::PutBitsInternal(uint32_t value, uint32_t num_bits) {
  assert(num_bits <= 24);
  assert(value < (1u << num_bits));

  const uint32_t shift = bit_pos_ & 7;
  size_t pos = bit_pos_ >> 3;
  bit_pos_ += num_bits;
  if (bit_pos_ > max_pos_) {  // Make sure 'value' fits in 'buf_'.
    error_ = true;
    return;
  }
  value <<= shift;
  for (uint32_t n = 0; n < num_bits + shift; n += 8) {
    buf_[pos++] |= value & 0xff;
    value >>= 8;
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
