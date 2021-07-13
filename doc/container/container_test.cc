// Copyright 2020 Google LLC
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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>

#include "./container.h"

//------------------------------------------------------------------------------

namespace {

constexpr uint32_t kContainerCodecRaw = 0x5;

bool operator==(const container::Data& a, const container::Data& b) {
  return (a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()));
}

bool operator==(const container::Image& a, const container::Image& b) {
  return (a.width == b.width && a.height == b.height &&
          a.orientation == b.orientation && a.format == b.format &&
          a.is_opaque == b.is_opaque && a.preview_color == b.preview_color &&
          a.icc == b.icc && a.xmp == b.xmp && a.exif == b.exif &&
          a.is_animation == b.is_animation &&
          (!a.is_animation || (a.loop_forever == b.loop_forever &&
                               a.background_color == b.background_color)) &&
          a.pixels == b.pixels);
}

}  // namespace

//------------------------------------------------------------------------------

int main(int, const char*[]) {
  std::mt19937 random;

  // Demo image settings.
  container::Image input;
  input.width = 1000;
  input.height = 1000;
  input.orientation = container::Orientation::kOriginal;
  input.format = container::Format::kARGB8;
  input.is_opaque = true;
  input.preview_color = 0x0;
  input.is_animation = false;
  input.loop_forever = true;
  input.background_color = 0xFF000000u;

  // Random preview and metadata. The content does not matter.
  for (container::Data* data : {&input.icc, &input.xmp, &input.exif}) {
    data->resize(1 + random() % 2048);
    for (uint8_t& byte : *data) byte = random() % 255;
  }

  // Random pixel values. The content does not matter.
  input.pixels.resize(input.width * input.height * (input.is_opaque ? 3 : 4));
  for (uint8_t& byte : input.pixels) byte = random() % 255;

  // Encode.
  container::Data codec_bytes, bitstream;
  container::EncodeRawPixels(input, codec_bytes);
  container::WrapContainer(input, container::EncoderConfig(),
                           kContainerCodecRaw, codec_bytes.data(),
                           codec_bytes.size(), bitstream);

  // Decode.
  container::Image output;
  uint32_t codec;
  size_t offset_till_codec_bytes, num_codec_bytes;
  if (!container::UnwrapContainer(bitstream.data(), bitstream.size(), output,
                                  codec, offset_till_codec_bytes,
                                  num_codec_bytes) ||
      codec != kContainerCodecRaw ||
      !container::DecodeRawPixels(bitstream.data() + offset_till_codec_bytes,
                                  num_codec_bytes, output)) {
    fprintf(stderr, "Failure\n");
    return 1;
  }

  // Compare.
  assert(input == output);

  printf("Success\n");
  return 0;
}
