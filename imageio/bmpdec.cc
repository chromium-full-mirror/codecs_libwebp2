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
// -----------------------------------------------------------------------------
//
// simple BMP Image reader. Only guaranted to read BMP files that *we* wrote!

#ifdef HAVE_CONFIG_H
#include "src/wp2/config.h"
#endif

#include "./anim_image_dec.h"
#include "src/dsp/dsp.h"

namespace WP2 {

namespace {

uint32_t ReadLE16(const uint8_t* const dst) {
  return (uint32_t)dst[0] | (dst[1] << 8);
}

uint32_t ReadLE32(const uint8_t* const dst) {
  return ReadLE16(dst + 0) | (ReadLE16(dst + 2) << 16);
}

}  // namespace

class ImageReaderBMP : public ImageReader::Impl {
 public:
  ImageReaderBMP(const uint8_t* data, size_t data_size,
                 ArgbBuffer* const buffer, LogLevel log_level,
                 size_t max_num_pixels)
      : ImageReader::Impl(buffer, data, data_size, log_level, max_num_pixels) {}

  WP2Status ReadFrame(bool* const is_last,
                      uint32_t* const duration_ms) override {
    WP2_CHECK_STATUS(CheckData());
    static constexpr uint32_t kBMPHeaderSize = 54u;
    static constexpr uint32_t kBMPAlphaHeaderSize = 70u;
    WP2_CHECK_OK(data_size_ >= kBMPHeaderSize,  WP2_STATUS_NOT_ENOUGH_DATA);
    WP2_CHECK_OK(ReadLE16(data_ + 0) == 0x4d42, WP2_STATUS_BITSTREAM_ERROR);
    const uint32_t total_size = ReadLE32(data_ + 2);
    WP2_CHECK_OK(data_size_ >= total_size, WP2_STATUS_BITSTREAM_ERROR);
    WP2_CHECK_OK(ReadLE16(data_ + 6) == 0, WP2_STATUS_BITSTREAM_ERROR);
    const uint32_t offset = ReadLE32(data_ + 10);
    const uint32_t width  = ReadLE32(data_ + 18);
    const uint32_t height = ReadLE32(data_ + 22);
    const bool top_down = ((int32_t)height < 0);
    const uint32_t real_height = top_down ? -(int32_t)height : height;
    WP2_CHECK_STATUS(CheckDimensions(width, real_height));
    const uint32_t bpp = ReadLE16(data_ + 28);
    WP2_CHECK_OK(bpp == 24 || bpp == 32, WP2_STATUS_UNSUPPORTED_FEATURE);
    WP2_CHECK_OK(ReadLE16(data_ + 26) == 1, WP2_STATUS_UNSUPPORTED_FEATURE);
    const uint32_t format = ReadLE32(data_ + 30);
    // only BI_RGB or BI_BITFIELDS are supported
    WP2_CHECK_OK(format == 0 || format == 3, WP2_STATUS_UNSUPPORTED_FEATURE);
    WP2_CHECK_OK(ReadLE32(data_ + 46) == 0, WP2_STATUS_UNSUPPORTED_FEATURE);
    WP2_CHECK_OK(ReadLE32(data_ + 50) == 0, WP2_STATUS_UNSUPPORTED_FEATURE);
    const bool has_alpha = (bpp == 32 && format == 3);
    const uint32_t header_size =
        (format == 3) ? kBMPAlphaHeaderSize : kBMPHeaderSize;
    const size_t stride = (width * (has_alpha ? 4 : 3) + 3) & ~3;   // pad to 4
    WP2_CHECK_OK(data_size_ >= header_size &&
                 offset >= header_size &&
                 data_size_ >= offset + real_height * stride,
                 WP2_STATUS_NOT_ENOUGH_DATA);

    if (format == 3) {  // check ARGB masks
      WP2_CHECK_OK(ReadLE32(data_ + 54) == 0x00ff0000 &&
                   ReadLE32(data_ + 58) == 0x0000ff00 &&
                   ReadLE32(data_ + 62) == 0x000000ff &&
                   ReadLE32(data_ + 66) == 0xff000000,
                   WP2_STATUS_UNSUPPORTED_FEATURE);
    }

    WP2_CHECK_STATUS(buffer_->Resize(width, real_height));
    WP2ArgbConverterInit();

    const uint8_t* src = data_ + offset;
    for (uint32_t y = 0; y < real_height; ++y) {
      uint8_t* const dst = buffer_->GetRow8(top_down ? y : real_height - 1 - y);
      WP2ArgbConvertFrom[has_alpha ? WP2_BGRA_32 : WP2_BGR_24](src, width, dst);
      src += stride;
      assert(src <= data_ + data_size_);
    }
    *is_last = true;
    *duration_ms = ImageReader::kInfiniteDuration;
    return WP2_STATUS_OK;
  }
};

void ImageReader::SetImplBMP(ArgbBuffer* const buffer,
                             LogLevel log_level, size_t max_num_pixels) {
  impl_.reset(new (WP2Allocable::nothrow) ImageReaderBMP(
      data_.bytes, data_.size, buffer, log_level, max_num_pixels));
  if (impl_ == nullptr) status_ = WP2_STATUS_OUT_OF_MEMORY;
}

}  // namespace WP2
