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

#include "./container.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace container {
namespace {

static_assert(kMaxChunkSize - 1u <= kMaxVarInt,
              "The maximum chunk size cannot be represented with var ints.");

//------------------------------------------------------------------------------
// Helper classes to write or read bit-packed values.

class BitPacker {
 public:
  void EncodeBool(bool value) {
    if ((current_bit_ & 7) == 0) bytes_.push_back(0);
    if (value) bytes_[current_bit_ >> 3] |= (1 << (current_bit_ & 7));
    ++current_bit_;
  }
  void EncodeUInt(uint32_t value, uint32_t min, uint32_t max) {
    assert(value >= min && value <= max);
    const uint32_t num_bits = std::lround(std::ceil(std::log2(1. + max - min)));
    value -= min;
    for (uint32_t i = 0; i < num_bits; ++i) EncodeBool((value >> i) & 1);
  }
  void Write(Data* bitstream) {
    bitstream->insert(bitstream->end(), bytes_.begin(), bytes_.end());
  }

 private:
  Data bytes_;
  size_t current_bit_ = 0;
};

class BitUnpacker {
 public:
  BitUnpacker(const uint8_t bytes[], size_t num_bytes)
      : bytes_(bytes), num_bytes_(num_bytes) {}
  bool DecodeBool() {
    if (current_bit_ * 8 >= num_bytes_) return !(error_ = true);
    const bool value =
        (bytes_[current_bit_ >> 3] & (1 << (current_bit_ & 7))) != 0;
    ++current_bit_;
    return value;
  }
  uint32_t DecodeUInt(uint32_t min, uint32_t max) {
    assert(min <= max);
    const uint32_t num_bits = std::lround(std::ceil(std::log2(1. + max - min)));
    uint32_t v = 0;
    for (uint32_t i = 0; i < num_bits; ++i) v |= (DecodeBool() ? (1 << i) : 0);
    return min + v;
  }
  bool Ok() { return !error_; }
  size_t GetNumReadBytes() { return (current_bit_ + 7) >> 3; }

 private:
  const uint8_t* bytes_;
  const size_t num_bytes_;
  size_t current_bit_ = 0;
  bool error_ = false;
};

//------------------------------------------------------------------------------
// Helper functions to write or read variable-length integers.
// These use one bit per byte to signal whether another byte is needed, up to 4.

void EncodeVarInt(uint32_t value, uint32_t min_value, uint32_t max_value,
                  Data* bitstream) {
  assert(min_value <= value && value <= max_value);
  value -= min_value;
  max_value -= min_value;
  assert(max_value <= kMaxVarInt);
  if (max_value == 0) return;
  while (value > 0x7F && max_value > 0xFF) {
    bitstream->push_back((value & 0x7F) | 0x80);
    value >>= 7;
    max_value >>= 7;
  }
  bitstream->push_back(value);
}

uint32_t DecodeVarInt(uint32_t min_value, uint32_t max_value,
                      const uint8_t bitstream[],
                      uint32_t* const bitstream_pos) {
  assert(min_value <= max_value && max_value - min_value <= kMaxVarInt);
  uint32_t value = 0, max_remaining_value = max_value - min_value;
  bool read_next_byte = (max_remaining_value > 0);
  for (uint32_t i = 0; read_next_byte; ++i, ++*bitstream_pos) {
    const uint8_t byte = bitstream[*bitstream_pos];
    read_next_byte = (max_remaining_value > 0xFF && (byte & 0x80));
    // If there is a next byte, use only 7 bits in current 'byte'.
    value |= static_cast<uint32_t>(read_next_byte ? (byte & 0x7F) : byte)
             << (7 * i);
    max_remaining_value >>= 7;
  }
  value += min_value;
  return std::min(value, max_value);
}

// Writes the bytes of a 'chunk', after writing its size as a VarInt.
void WriteChunk(const Data& chunk, Data* bitstream) {
  EncodeVarInt(chunk.size(), 1, kMaxChunkSize, bitstream);
  bitstream->insert(bitstream->end(), chunk.begin(), chunk.end());
}

// Reads the bytes of a 'chunk', after reading its size as a VarInt.
void ReadChunk(const uint8_t bitstream[], uint32_t* bitstream_pos,
               Data* chunk) {
  const uint32_t chunk_size =
      DecodeVarInt(1, kMaxChunkSize, bitstream, bitstream_pos);
  chunk->resize(chunk_size);
  std::copy(bitstream + *bitstream_pos, bitstream + *bitstream_pos + chunk_size,
            chunk->data());
  *bitstream_pos += chunk_size;
}

}  // namespace

//------------------------------------------------------------------------------
// The content of these functions can be replaced by actual image compression
// algorithms. Currently it is just raw 8b pixel packing, with or without alpha.

void EncodeRawPixels(const Image& image, Data& data) {
  // TODO(yguyon): Implement if (image.is_animation)
  if (image.format == Format::kARGB8) {
    data.insert(data.end(), image.pixels.begin(), image.pixels.end());
  } else {
    abort();  // TODO(yguyon): Implement
  }
}

bool DecodeRawPixels(const uint8_t data[], size_t data_size, Image& image) {
  // TODO(yguyon): Implement if (image.is_animation)
  if (image.format == Format::kARGB8) {
    if (data_size != image.width * image.height * (image.is_opaque ? 3 : 4)) {
      return false;
    }
    image.pixels.resize(data_size);
    std::copy(data, data + data_size, image.pixels.data());
  } else {
    abort();  // TODO(yguyon): Implement
  }
  return true;
}

//------------------------------------------------------------------------------

void WrapContainer(const Image& image, const EncoderConfig& config,
                   uint32_t codec, const uint8_t codec_bytes[],
                   size_t num_codec_bytes, Data& data) {
  data.clear();

  BitPacker header;
  header.EncodeUInt(kContainerTag, 0, (1 << kContainerTagNumBits) - 1);
  header.EncodeUInt(codec, 0, (1 << kContainerCodecNumBits) - 1);
  header.EncodeUInt(image.width, 1, 1 << kImageDimNumBits);
  header.EncodeUInt(image.height, 1, 1 << kImageDimNumBits);
  header.EncodeUInt(static_cast<uint32_t>(image.orientation), 0, 7);
  header.EncodeUInt((image.format == Format::kARGB8)    ? 0
                    : (image.format == Format::kAYUV10) ? 1
                                                        : 3,
                    0, 3);
  header.EncodeBool(image.is_opaque);
  header.EncodeUInt(image.preview_color, 0, (1 << 12) - 1);
  header.EncodeBool(config.create_preview);
  header.EncodeBool(!image.icc.empty());
  header.EncodeBool(!image.xmp.empty());
  header.EncodeBool(!image.exif.empty());
  header.EncodeBool(image.is_animation);
  if (image.is_animation) {
    header.EncodeBool(image.loop_forever);
    header.EncodeUInt(image.background_color, 0u, 4294967295u);
  }
  header.Write(&data);

  if (config.create_preview) {
    abort();  // TODO(yguyon): Implement
  }
  if (!image.icc.empty()) {
    WriteChunk(image.icc, &data);
  }

  WriteChunk(Data(codec_bytes, codec_bytes + num_codec_bytes), &data);

  if (!image.xmp.empty()) {
    WriteChunk(image.xmp, &data);
  }
  if (!image.exif.empty()) {
    WriteChunk(image.exif, &data);
  }
}

//------------------------------------------------------------------------------

namespace {

// Extracts 'num_read_bytes' from 'data' into 'image' features.
bool UnwrapContainerHeader(const uint8_t data[], size_t data_size, Image& image,
                           uint32_t& codec, bool& has_preview, bool& has_icc,
                           bool& has_xmp, bool& has_exif,
                           uint32_t& num_read_bytes) {
  BitUnpacker header(data, data_size);
  const uint32_t tag = header.DecodeUInt(0, (1 << kContainerTagNumBits) - 1);
  codec = header.DecodeUInt(0, (1 << kContainerCodecNumBits) - 1);
  if (tag != kContainerTag) return false;

  image.width = header.DecodeUInt(1, 1 << kImageDimNumBits);
  image.height = header.DecodeUInt(1, 1 << kImageDimNumBits);
  image.orientation = static_cast<Orientation>(header.DecodeUInt(0, 7));
  const uint32_t format = header.DecodeUInt(0, 3);
  image.format = (format == 0)   ? Format::kARGB8
                 : (format == 1) ? Format::kAYUV10
                                 : Format::kUnknown;
  image.is_opaque = header.DecodeBool();
  image.preview_color = header.DecodeUInt(0, (1 << 12) - 1);
  has_preview = header.DecodeBool();
  has_icc = header.DecodeBool();
  has_xmp = header.DecodeBool();
  has_exif = header.DecodeBool();
  image.is_animation = header.DecodeBool();
  image.loop_forever = image.is_animation ? header.DecodeBool() : false;
  image.background_color =
      image.is_animation ? header.DecodeUInt(0u, 4294967295u) : 0;
  num_read_bytes = header.GetNumReadBytes();
  return header.Ok();
}

}  // namespace

bool UnwrapContainer(const uint8_t data[], size_t data_size, Image& image,
                     uint32_t& codec, size_t& offset_till_codec_bytes,
                     size_t& num_codec_bytes) {
  image = Image();

  bool has_preview, has_icc, has_xmp, has_exif;
  uint32_t data_pos;
  if (!UnwrapContainerHeader(data, data_size, image, codec, has_preview,
                             has_icc, has_xmp, has_exif, data_pos)) {
    return false;
  }

  Data preview;
  if (has_preview) ReadChunk(data, &data_pos, &preview);
  if (has_icc) ReadChunk(data, &data_pos, &image.icc);

  const uint32_t chunk_size = DecodeVarInt(1, kMaxChunkSize, data, &data_pos);
  if (data_pos + chunk_size > data_size) {
    return false;
  }
  offset_till_codec_bytes = data_pos;
  num_codec_bytes = chunk_size;
  data_pos += chunk_size;

  if (has_xmp) ReadChunk(data, &data_pos, &image.xmp);
  if (has_exif) ReadChunk(data, &data_pos, &image.exif);
  return true;
}

//------------------------------------------------------------------------------

}  // namespace container
