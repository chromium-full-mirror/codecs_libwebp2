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
#include <cassert>
#include <cmath>

namespace container {

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

//------------------------------------------------------------------------------
// The content of these functions can be replaced by actual image compression
// algorithms. Currently it is just raw 8b pixel packing, with or without alpha.

Data EncodeTile(const ImageFeatures& f, const uint8_t pixels[], uint32_t tile_x,
                uint32_t tile_y) {
  Data tile_bytes;
  const uint32_t tile_height = std::min(f.tile_size, f.height - tile_y);
  const uint32_t tile_width = std::min(f.tile_size, f.width - tile_x);
  const uint32_t mult = f.is_opaque ? 3 : 4;
  for (uint32_t y = 0; y < tile_height; ++y) {
    const uint8_t* src_row = &pixels[((tile_y + y) * f.width + tile_x) * mult];
    tile_bytes.insert(tile_bytes.end(), src_row, src_row + tile_width * mult);
  }
  return tile_bytes;
}

void DecodeTile(const ImageFeatures& f, uint32_t tile_x, uint32_t tile_y,
                const Data& tile_bytes, uint8_t pixels[]) {
  const uint32_t tile_height = std::min(f.tile_size, f.height - tile_y);
  const uint32_t tile_width = std::min(f.tile_size, f.width - tile_x);
  const uint32_t mult = f.is_opaque ? 3 : 4;
  for (uint32_t y = 0, bitstream_pos = 0; y < tile_height; ++y) {
    const uint8_t* src_row = tile_bytes.data() + bitstream_pos;
    uint8_t* dst_row = &pixels[((tile_y + y) * f.width + tile_x) * mult];
    std::copy(src_row, src_row + tile_width * mult, dst_row);
    bitstream_pos += tile_width * mult;
  }
}

//------------------------------------------------------------------------------

void EncodeStillImage(const ImageFeatures& features, const Data& preview,
                      const Data& icc, const Data& xmp, const Data& exif,
                      const Data& pixels, Data* bitstream) {
  bitstream->clear();

  BitPacker header;
  header.EncodeUInt(kTag, 0, (1 << (kTagNumBytes * 8)) - 1);
  header.EncodeUInt(features.width, 1, 1 << kImageDimNumBits);
  header.EncodeUInt(features.height, 1, 1 << kImageDimNumBits);
  header.EncodeUInt(features.orientation, 0, 3);
  header.EncodeUInt(features.tile_size == 128   ? 0
                    : features.tile_size == 256 ? 1
                    : features.tile_size == 512 ? 2
                                                : 3,
                    0, 3);
  header.EncodeBool(features.is_opaque);
  header.EncodeBool(features.is_animation);
  header.EncodeUInt(features.preview_color, 0, 1 << 12);
  header.EncodeBool(features.has_preview);
  header.EncodeBool(features.has_icc);
  header.EncodeBool(features.has_xmp);
  header.EncodeBool(features.has_exif);
  if (features.is_animation) {
    header.EncodeBool(features.loop_forever);
    header.EncodeUInt(features.background_color, 0, 4294967295u);
  }
  header.Write(bitstream);

  if (features.has_preview) WriteChunk(preview, bitstream);
  if (features.has_icc) WriteChunk(icc, bitstream);

  for (uint32_t tile_y = 0; tile_y < features.height;
       tile_y += features.tile_size) {
    for (uint32_t tile_x = 0; tile_x < features.width;
         tile_x += features.tile_size) {
      const Data tile_bytes =
          EncodeTile(features, pixels.data(), tile_x, tile_y);
      WriteChunk(tile_bytes, bitstream);
    }
  }

  if (features.has_xmp) WriteChunk(xmp, bitstream);
  if (features.has_exif) WriteChunk(exif, bitstream);
}

bool DecodeStillImage(const Data& bitstream, ImageFeatures* features,
                      Data* preview, Data* icc, Data* xmp, Data* exif,
                      Data* pixels) {
  BitUnpacker header(bitstream.data(), bitstream.size());
  const uint32_t tag = header.DecodeUInt(0, (1 << (kTagNumBytes * 8)) - 1);
  if (tag != kTag) return false;
  features->width = header.DecodeUInt(1, 1 << kImageDimNumBits);
  features->height = header.DecodeUInt(1, 1 << kImageDimNumBits);
  features->orientation = header.DecodeUInt(0, 3);
  const uint32_t tile_size = header.DecodeUInt(0, 3);
  features->tile_size = tile_size == 0   ? 128
                        : tile_size == 1 ? 256
                        : tile_size == 2 ? 512
                                         : 1024;
  features->is_opaque = header.DecodeBool();
  features->is_animation = header.DecodeBool();
  features->preview_color = header.DecodeUInt(0, 1 << 12);
  features->has_preview = header.DecodeBool();
  features->has_icc = header.DecodeBool();
  features->has_xmp = header.DecodeBool();
  features->has_exif = header.DecodeBool();
  features->loop_forever = features->is_animation ? header.DecodeBool() : false;
  features->background_color =
      features->is_animation ? header.DecodeUInt(0, 4294967295u) : 0;
  if (!header.Ok()) return false;
  uint32_t pos = header.GetNumReadBytes();

  if (features->has_preview) ReadChunk(bitstream.data(), &pos, preview);
  if (features->has_icc) ReadChunk(bitstream.data(), &pos, icc);

  pixels->resize(features->width * features->height *
                 (features->is_opaque ? 3 : 4));
  for (uint32_t tile_y = 0; tile_y < features->height;
       tile_y += features->tile_size) {
    for (uint32_t tile_x = 0; tile_x < features->width;
         tile_x += features->tile_size) {
      Data tile_bytes;
      ReadChunk(bitstream.data(), &pos, &tile_bytes);
      DecodeTile(*features, tile_x, tile_y, tile_bytes, pixels->data());
    }
  }

  if (features->has_xmp) ReadChunk(bitstream.data(), &pos, xmp);
  if (features->has_exif) ReadChunk(bitstream.data(), &pos, exif);

  return true;
}

//------------------------------------------------------------------------------

}  // namespace container
