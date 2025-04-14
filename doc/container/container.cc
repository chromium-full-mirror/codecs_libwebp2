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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

#include "./bit_packing.h"

namespace container {
namespace {

static_assert(kMaxChunkSize - 1u <= kMaxVarInt,
              "The maximum chunk size cannot be represented with var ints.");

//------------------------------------------------------------------------------

// Writes the bytes of a 'chunk', after writing its size as a variable int.
void WriteChunk(const uint8_t chunk[], size_t chunk_size, Data& data) {
  EncodeVarInt(chunk_size, 1, kMaxChunkSize, data);
  data.insert(data.end(), chunk, chunk + chunk_size);
}

// Reads the bytes of a 'chunk', after reading its size as a variable int.
// Advances 'data_pos' accordingly.
bool ReadChunk(const uint8_t data[], size_t data_size, uint32_t& data_pos,
               Data& chunk) {
  uint32_t chunk_size;
  if (!DecodeVarInt(1, kMaxChunkSize, data, data_size, data_pos, chunk_size) ||
      data_pos + chunk_size > data_size) {
    return false;
  }
  assert(chunk.empty());
  chunk.reserve(chunk_size);
  chunk.insert(chunk.end(), data + data_pos, data + data_pos + chunk_size);

  data_pos += chunk_size;
  return true;
}

// Same as ReadChunk() but without copying bytes.
bool SkipChunk(const uint8_t data[], size_t data_size, uint32_t& data_pos) {
  uint32_t chunk_size;
  if (!DecodeVarInt(1, kMaxChunkSize, data, data_size, data_pos, chunk_size) ||
      data_pos + chunk_size > data_size) {
    return false;
  }
  data_pos += chunk_size;
  return true;
}

}  // namespace

//------------------------------------------------------------------------------
// The content of these functions can be replaced by actual image compression
// algorithms. Currently it is just raw 8b pixel packing, with or without alpha.

void EncodeRawPixels(const Image& image, Data& data) {
  // TODO(yguyon): Implement if (image.is_animation)
  if (image.format == Format::kARGB8 || image.format == Format::kAYUV8) {
    data.insert(data.end(), image.pixels.begin(), image.pixels.end());
  } else {
    abort();  // TODO(yguyon): Implement
  }
}

bool DecodeRawPixels(const uint8_t data[], size_t data_size, Image& image) {
  // TODO(yguyon): Implement if (image.is_animation)
  if (image.format == Format::kARGB8 || image.format == Format::kAYUV8) {
    if (data_size != image.width * image.height * (image.has_alpha ? 4 : 3)) {
      return false;
    }
    assert(image.pixels.empty());
    image.pixels.reserve(data_size);
    image.pixels.insert(image.pixels.end(), data, data + data_size);
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
  header.EncodeBool(image.has_alpha);
  header.EncodeBool(image.is_animation);
  header.EncodeBool(image.is_animation ? image.loop_forever : false);
  header.EncodeUInt(image.preview_color, 0, (1 << 12) - 1);
  header.EncodeBool(config.create_preview);
  header.EncodeBool(!image.icc.empty());
  header.EncodeBool(!image.xmp.empty());
  header.EncodeBool(!image.exif.empty());
  header.EncodeUInt((image.format == Format::kARGB8)    ? 0
                    : (image.format == Format::kAYUV8) ? 1
                    : (image.format == Format::kAYUV10) ? 2
                                                        : 3,
                    0, 3);
  header.EncodeUInt(kContainerHeaderPaddingValue, 0, 15);
  assert(header.GetNumBits() == kContainerHeaderNumBits);
  data.reserve(header.GetNumBytes());
  data.insert(data.end(), header.GetBytes(),
              header.GetBytes() + header.GetNumBytes());

  if (!image.icc.empty()) {
    WriteChunk(image.icc.data(), image.icc.size(), data);
  }
  if (config.create_preview) {
    abort();  // TODO(yguyon): Implement
  }

  WriteChunk(codec_bytes, num_codec_bytes, data);

  if (!image.xmp.empty()) {
    WriteChunk(image.xmp.data(), image.xmp.size(), data);
  }
  if (!image.exif.empty()) {
    WriteChunk(image.exif.data(), image.exif.size(), data);
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
  const uint32_t tag = header.DecodeUInt(kContainerTagNumBits);
  codec = header.DecodeUInt(kContainerCodecNumBits);
  if (tag != kContainerTag) return false;

  image.width = header.DecodeUInt(1, 1 << kImageDimNumBits);
  image.height = header.DecodeUInt(1, 1 << kImageDimNumBits);
  image.orientation = static_cast<Orientation>(header.DecodeUInt(0, 7));
  image.has_alpha = header.DecodeBool();
  image.is_animation = header.DecodeBool();
  image.loop_forever = header.DecodeBool() && image.is_animation;
  image.preview_color = header.DecodeUInt(/*num_bits=*/12);
  has_preview = header.DecodeBool();
  has_icc = header.DecodeBool();
  has_xmp = header.DecodeBool();
  has_exif = header.DecodeBool();
  const uint32_t format = header.DecodeUInt(0, 3);
  image.format = (format == 0)   ? Format::kARGB8
                 : (format == 1) ? Format::kAYUV8
                 : (format == 2) ? Format::kAYUV10
                                 : Format::kAYUV12;
  if (header.DecodeUInt(0, 15) != kContainerHeaderPaddingValue) return false;
  assert(header.GetNumReadBits() == kContainerHeaderNumBits);
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

  if ((has_icc && !ReadChunk(data, data_size, data_pos, image.icc)) ||
      (has_preview && !SkipChunk(data, data_size, data_pos))) {
    return false;
  }

  uint32_t chunk_size;
  if (!DecodeVarInt(1, kMaxChunkSize, data, data_size, data_pos, chunk_size) ||
      data_pos + chunk_size > data_size) {
    return false;
  }
  offset_till_codec_bytes = data_pos;
  num_codec_bytes = chunk_size;
  data_pos += chunk_size;

  if ((has_xmp && !ReadChunk(data, data_size, data_pos, image.xmp)) ||
      (has_exif && !ReadChunk(data, data_size, data_pos, image.exif))) {
    return false;
  }
  return true;
}

//------------------------------------------------------------------------------

std::string ContainerHeaderToStr(const uint8_t data[], size_t data_size) {
  static const char* kOrientationStr[] = {"Original",
                                          "FlipLeftToRight",
                                          "180",
                                          "FlipTopToBottom",
                                          "FlipBotLeftToTopRgt",
                                          "90Clockwise",
                                          "FlipTopLeftToBotRgt",
                                          "270Clockwise"};
  static const char* kFormatStr[] = {"ARGB8", "AYUV10", "AYUV12", "Unknown"};
  static const char* kBoolStr[] = {"false", "true"};

  std::stringstream ss;

  Image image;
  uint32_t codec;
  bool has_preview, has_icc, has_xmp, has_exif;
  uint32_t num_read_bytes;
  if (!UnwrapContainerHeader(data, data_size, image, codec, has_preview,
                             has_icc, has_xmp, has_exif, num_read_bytes)) {
    ss << "Bad container header" << std::endl;
    return ss.str();
  }
  const int orientation = static_cast<int>(image.orientation);
  const int format = static_cast<int>(image.format);
  if (orientation < 0 || orientation >= array_size(kOrientationStr) ||
      format < 0 || format >= array_size(kFormatStr)) {
    ss << "Bad container header value" << std::endl;
    return ss.str();
  }

  ss << "Container header (" << num_read_bytes << " bytes, codec " << codec
     << ")" << std::endl
     << "  width " << image.width << std::endl
     << "  height " << image.height << std::endl
     << "  orientation " << kOrientationStr[orientation] << std::endl
     << "  has_alpha " << kBoolStr[image.has_alpha] << std::endl
     << "  is_animation " << kBoolStr[image.is_animation] << std::endl
     << "  loop_forever " << kBoolStr[image.loop_forever] << std::endl
     << "  preview_color " << image.preview_color << std::endl
     << "  has_preview " << kBoolStr[has_preview] << std::endl
     << "  has_icc " << kBoolStr[has_icc] << std::endl
     << "  has_xmp " << kBoolStr[has_xmp] << std::endl
     << "  has_exif " << kBoolStr[has_exif] << std::endl
     << "  format " << kFormatStr[format] << std::endl;
  return ss.str();
}

//------------------------------------------------------------------------------

}  // namespace container
