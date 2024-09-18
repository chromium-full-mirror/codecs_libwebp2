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
// WP2 lossless decoding.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/common/global_params.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/ans.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"

namespace WP2 {

WP2Status LosslessDecode(const BitstreamFeatures& features,
                         const DecoderConfig& config,
                         const GlobalParams& gparams, ANSDec* const dec,
                         Tile* const tile) {
  assert(!tile->rgb_output.IsEmpty());
  tile->output_is_yuv = false;

  WP2L::Decoder dec_lossless;
  dec_lossless.Init(config, gparams, features.is_premultiplied, dec, tile);

  WP2_CHECK_STATUS(dec_lossless.DecodeHeader());
  WP2_CHECK_STATUS(dec_lossless.DecodeImage());

  WP2_CHECK_STATUS(dec->GetStatus());
  // Not using the exact whole chunk is an issue.
  WP2_CHECK_OK(dec->GetNumUsedBytes() == tile->chunk_size,
               WP2_STATUS_BITSTREAM_ERROR);

  return WP2_STATUS_OK;
}

}  // namespace WP2

//------------------------------------------------------------------------------

int WP2Parse(const uint8_t* encoded_bytes, size_t num_encoded_bytes,
             uint32_t* width, uint32_t* height) {
  WP2::BitstreamFeatures features;
  WP2Status status = features.Read(encoded_bytes, num_encoded_bytes);
  if (status != WP2_STATUS_OK) {
    return static_cast<int>(status);
  }
  *width = features.raw_width;
  *height = features.raw_height;
  return static_cast<int>(WP2_STATUS_OK);
}

namespace WP2 {
namespace {

int Decode8(const uint8_t* encoded_bytes, size_t num_encoded_bytes,
            uint32_t width, uint32_t height, WP2SampleFormat pixels_format,
            uint8_t* pixels, uint32_t stride) {
  ArgbBuffer buffer(pixels_format);
  WP2_CHECK_STATUS(buffer.SetExternal(width, height, pixels, stride));
  return Decode(encoded_bytes, num_encoded_bytes, &buffer);
}

}  // namespace
}  // namespace WP2

int WP2DecodeRgb8(const uint8_t* encoded_bytes, size_t num_encoded_bytes,
                  uint32_t width, uint32_t height, uint8_t* rgb,
                  uint32_t stride) {
  return static_cast<int>(WP2::Decode8(encoded_bytes, num_encoded_bytes, width,
                                       height, WP2_RGB_24, rgb, stride));
}

int WP2DecodeRgba8(const uint8_t* encoded_bytes, size_t num_encoded_bytes,
                   uint32_t width, uint32_t height, uint8_t* rgba,
                   uint32_t stride) {
  return static_cast<int>(WP2::Decode8(encoded_bytes, num_encoded_bytes, width,
                                       height, WP2_RGBA_32, rgba, stride));
}
