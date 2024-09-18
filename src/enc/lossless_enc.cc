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
// WP2 lossless encoding.

#include <cstddef>
#include <cstdint>

#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/tile_enc.h"
#include "src/utils/ans_enc.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"

namespace WP2 {

WP2Status TileEncoder::LosslessEncode(ANSEnc* const enc) {
  WP2_CHECK_STATUS(WP2L::EncodeImage(
      *config_, tile_->rgb_input, tiles_layout_->gparams->has_alpha_,
      tiles_layout_->image_is_premultiplied, tile_->progress, enc));
  return enc->GetStatus();
}

}  // namespace WP2

//------------------------------------------------------------------------------

namespace WP2 {
namespace {

WP2Status EncodeLossless8(uint32_t width, uint32_t height,
                          WP2SampleFormat pixel_format, const uint8_t* pixels,
                          uint32_t stride, int effort, uint8_t*& encoded_bytes,
                          size_t& num_encoded_bytes) {
  ArgbBuffer buffer(pixel_format);
  WP2_CHECK_STATUS(
      buffer.SetExternal(width, height, const_cast<uint8_t*>(pixels), stride));
  EncoderConfig config;
  config.quality = config.alpha_quality = 100;
  config.keep_unmultiplied = true;
  config.effort = effort;
  MemoryWriter writer;
  WP2_CHECK_STATUS(Encode(buffer, &writer, config));
  encoded_bytes = writer.mem_;
  num_encoded_bytes = writer.size_;
  writer.mem_ = nullptr;
  writer.size_ = 0;
  return WP2_STATUS_OK;
}

}  // namespace
}  // namespace WP2

int WP2EncodeLosslessRgb8(uint32_t width, uint32_t height, const uint8_t* rgb,
                          uint32_t stride, int effort, uint8_t*& encoded_bytes,
                          size_t& num_encoded_bytes) {
  return static_cast<int>(WP2::EncodeLossless8(width, height, WP2_RGB_24, rgb,
                                               stride, effort, encoded_bytes,
                                               num_encoded_bytes));
}

int WP2EncodeLosslessRgba8(uint32_t width, uint32_t height, const uint8_t* rgba,
                           uint32_t stride, int effort, uint8_t*& encoded_bytes,
                           size_t& num_encoded_bytes) {
  return static_cast<int>(WP2::EncodeLossless8(width, height, WP2_RGBA_32, rgba,
                                               stride, effort, encoded_bytes,
                                               num_encoded_bytes));
}

void WP2Release(uint8_t* encoded_bytes) { WP2Free(encoded_bytes); }
