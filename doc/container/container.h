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

#ifndef CONTAINER_CONTAINER_H_
#define CONTAINER_CONTAINER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace container {

typedef std::vector<uint8_t> Data;  // Chunk of bytes.

// Returns the length of a C array. std::size() exists in C++17 but it is C++11.
template <class T, size_t N>
constexpr std::size_t array_size(const T (&array)[N]) {
  return N;
}

//------------------------------------------------------------------------------
// Format constants.

static constexpr uint32_t kContainerTagNumBits = 20;
static constexpr uint32_t kContainerTag = 0xEF4FF;
static constexpr uint32_t kContainerCodecNumBits = 4;
static constexpr uint32_t kContainerCodecObu = 0x8;  // Full av1-obu bitstream
static constexpr uint32_t kContainerCodecAv1 = 0x9;  // av1-obu tile group only
static constexpr uint32_t kContainerCodecWp2 = 0x6;  // WebP2 (unimplemented)
static constexpr uint32_t kImageDimNumBits = 14;  // Maximum log2 width, height.
static constexpr uint32_t kContainerHeaderNumBits = 80;       // 10 bytes
static constexpr uint32_t kContainerHeaderPaddingValue = 15;  // 4 bits

// Maximum (inclusive) size of a data chunk.
static constexpr uint32_t kMaxChunkSize = 1u << 29;  // ~500 MB

//------------------------------------------------------------------------------
// Format enums and structs.

// Follows orientation tags defined in JEITA CP-3451C (section 4.6.4.A)
enum class Orientation {
  kOriginal = 0,
  kFlipLeftToRight,  // Mirror horizontally = vertical symmetry axis
  k180,
  kFlipTopToBottom,      // Mirror vertically = horizontal symmetry axis
  kFlipBotLeftToTopRgt,  // 90 clockwise then mirror horizontally
  k90Clockwise,
  kFlipTopLeftToBotRgt,  // 270 clockwise then mirror horizontally
  k270Clockwise
};

// Describes the channel layout/content and the number of bits per channel.
enum class Format { kARGB8, kAYUV10, kAYUV12, kUnknown };

struct Image {
  uint32_t width;   // Width in pixels, before applying 'orientation'.
  uint32_t height;  // Height in pixels, before applying 'orientation'.
  Orientation orientation = Orientation::kOriginal;  // To be applied.

  Format format = Format::kARGB8;  // Describes 'pixels'.
  bool has_alpha = false;          // The image is opaque if false.
  uint32_t preview_color = 0;      // Average color. RGB444 format.

  Data icc;   // Color profile chunk. Empty if none.
  Data xmp;   // XMP metadata chunk. Empty if none.
  Data exif;  // EXIF metadata chunk. Empty if none.

  bool is_animation = false;  // True if the image is animated (unsupported).
  // For 'is_animation=true' only:
  bool loop_forever = false;      // Play once if false, forever if true.

  Data pixels;  // Raw samples in 'format'. 'orientation' is not applied.
};

struct EncoderConfig {
  bool create_preview = false;  // If true, prepend a small image at encoding.
};

//------------------------------------------------------------------------------

// Copies/converts the raw samples from 'image' to 'data'.
void EncodeRawPixels(const Image& image, Data& data);

// Copies/converts the raw samples from 'data' to 'image'.
// Returns false in case of error.
bool DecodeRawPixels(const uint8_t data[], size_t data_size, Image& image);

//------------------------------------------------------------------------------

// Wraps the input 'image' and 'codec_bytes' into contained 'data'.
// 'image.pixels' is ignored.
void WrapContainer(const Image& image, const EncoderConfig& config,
                   uint32_t codec, const uint8_t codec_bytes[],
                   size_t num_codec_bytes, Data& data);

// Parses the contained 'data' into 'image' characteristics and returns the
// 'offset_till_codec_bytes', 'num_codec_bytes' to be decoded by the 'codec'.
// Does not decode the pixels. Returns false in case of error.
bool UnwrapContainer(const uint8_t data[], size_t data_size, Image& image,
                     uint32_t& codec, size_t& offset_till_codec_bytes,
                     size_t& num_codec_bytes);

//------------------------------------------------------------------------------

// Returns a string containing the features of the image encoded in 'data'.
std::string ContainerHeaderToStr(const uint8_t data[], size_t data_size);

//------------------------------------------------------------------------------

}  // namespace container

#endif  // CONTAINER_CONTAINER_H_
