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

#ifndef CONTAINER_H_
#define CONTAINER_H_

#include <cstdint>
#include <vector>

namespace container {

typedef std::vector<uint8_t> Data;  // Chunk of bytes.

// Format constants.
static constexpr uint32_t kTagNumBytes = 3;
static constexpr uint32_t kTag = 123456;          // Value of the first 3 bytes.
static constexpr uint32_t kImageDimNumBits = 14;  // Maximum width, height.

// Maximum number of bytes used by a variable-length integer.
static constexpr uint32_t kMaxVarIntLength = 4;
// Maximum value expressible with a variable-length integer (included).
static constexpr uint32_t kMaxVarInt =
    (1u << (7 * (kMaxVarIntLength - 1) + 8)) - 1u;  // 29 bits
// Maximum (inclusive) size of a data chunk.
static constexpr uint32_t kMaxChunkSize = 1u << 29;  // ~500 MB

struct ImageFeatures {
  uint32_t width;            // Width in pixels.
  uint32_t height;           // Height in pixels.
  uint32_t orientation;      // Rotation applied during decoding.
  uint32_t tile_size;        // Width and height of tiles in pixels.

  bool is_opaque;            // True if no alpha. False has no guarantee.
  bool is_animation;         // True if the image is animated
                             // (contains frames and ANMF separators).
  uint32_t preview_color;    // Average color in RGB444 format.

  bool has_preview;          // True if the bitstream has a preview image
                             // (irrespective of the preview color).
  bool has_icc;              // Indicates the presence of ICC information.
  bool has_xmp;              // XMP chunk at the end of the bitstream.
  bool has_exif;             // EXIF chunk at the end of the bitstream.

  bool loop_forever;         // Play once if false, forever if true (anim only).
  uint32_t background_color;  // Color to dispose the canvas with (anim only).
};

void EncodeStillImage(const ImageFeatures& features, const Data& preview,
                      const Data& icc, const Data& xmp, const Data& exif,
                      const Data& pixels, Data* bitstream);

bool DecodeStillImage(const Data& bitstream, ImageFeatures* features,
                      Data* preview, Data* icc, Data* xmp, Data* exif,
                      Data* pixels);

}  // namespace container

#endif  // CONTAINER_H_
