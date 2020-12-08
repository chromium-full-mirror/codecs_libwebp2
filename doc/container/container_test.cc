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

bool operator==(const container::Data& a, const container::Data& b) {
  return (a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()));
}

bool operator==(const container::ImageFeatures& a,
                const container::ImageFeatures& b) {
  return (a.width == b.width && a.height == b.height &&
      a.orientation == b.orientation && a.tile_size == b.tile_size &&
      a.is_opaque == b.is_opaque && a.is_animation == b.is_animation &&
      a.has_preview == b.has_preview &&
      a.preview_color == b.preview_color && a.has_icc == b.has_icc &&
      a.has_xmp == b.has_xmp && a.has_exif == b.has_exif &&
      (!a.is_animation || (a.loop_forever == b.loop_forever &&
          a.background_color == b.background_color)));
}

}  // namespace

//------------------------------------------------------------------------------

int main(int, const char*[]) {
  std::mt19937 random;

  // Demo image settings.
  container::ImageFeatures enc_features;
  enc_features.width = 1000;
  enc_features.height = 1000;
  enc_features.orientation = 0;
  enc_features.tile_size = 512;
  enc_features.is_opaque = true;
  enc_features.is_animation = false;
  enc_features.has_preview = true;
  enc_features.preview_color = 0xFFF;
  enc_features.has_icc = true;
  enc_features.has_xmp = true;
  enc_features.has_exif = true;
  enc_features.loop_forever = true;
  enc_features.background_color = 0xFF000000u;

  // Random preview and metadata. The content does not matter.
  container::Data enc_preview, enc_icc, enc_xmp, enc_exif, enc_pixels;
  for (container::Data* data : {&enc_preview, &enc_icc, &enc_xmp, &enc_exif}) {
    data->resize(1 + random() % 2048);
    for (uint8_t& byte : *data) byte = random() % 255;
  }

  // Random pixel values. The content does not matter; it could be YUV or ARGB,
  // for example.
  enc_pixels.resize(enc_features.width * enc_features.height *
                    (enc_features.is_opaque ? 3 : 4));
  for (uint8_t& byte : enc_pixels) byte = random() % 255;

  // Encode.
  container::Data bitstream;
  container::EncodeStillImage(enc_features, enc_preview, enc_icc, enc_xmp,
                              enc_exif, enc_pixels, &bitstream);

  // Decode.
  container::ImageFeatures dec_features;
  container::Data dec_preview, dec_icc, dec_xmp, dec_exif, dec_pixels;
  if (!container::DecodeStillImage(bitstream, &dec_features, &dec_preview,
                                   &dec_icc, &dec_xmp, &dec_exif,
                                   &dec_pixels)) {
    fprintf(stderr, "Failure\n");
    return 1;
  }

  // Compare.
  assert(enc_features == dec_features);
  assert(enc_preview == dec_preview);
  assert(enc_icc == dec_icc);
  assert(enc_xmp == dec_xmp);
  assert(enc_exif == dec_exif);
  assert(enc_pixels == dec_pixels);

  printf("Success\n");
  return 0;
}
