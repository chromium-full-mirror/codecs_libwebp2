// Copyright 2023 Google LLC
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
// JPEG XL decode.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include <cstdio>

#include "imageio/anim_image_dec.h"
#include "imageio/imageio_util.h"
#include "src/wp2/base.h"

#ifdef HAVE_CONFIG_H
#include "wp2/config.h"
#endif

#ifdef WP2_HAVE_JXL
#include <cstdint>

#include "src/utils/utils.h"
#include "third_party/jpegxl/lib/include/jxl/codestream_header.h"
#include "third_party/jpegxl/lib/include/jxl/decode.h"
#include "third_party/jpegxl/lib/include/jxl/decode_cxx.h"
#include "third_party/jpegxl/lib/include/jxl/types.h"

namespace {

//------------------------------------------------------------------------------
// JPEG XL decoding

}  // namespace

namespace WP2 {
namespace {
JxlPixelFormat ArgbBufferToJxlPixelFormat(const WP2::ArgbBuffer& image) {
  JxlPixelFormat pixel_format;
  const uint32_t bytes_per_channel = (WP2Formatbpc(image.format()) + 7) / 8;
  pixel_format.num_channels = WP2FormatBpp(image.format()) / bytes_per_channel;
  pixel_format.data_type =
      WP2Formatbpc(image.format()) == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
  // TODO(yguyon): Fix endianness TODO in libwebp2/public/src/wp2/decode.h
  pixel_format.endianness = JXL_NATIVE_ENDIAN;
  pixel_format.align = image.stride();
  return pixel_format;
}

size_t ArgbBufferSize(const WP2::ArgbBuffer& image) {
  return static_cast<size_t>(image.height() - 1) * image.stride() +
         static_cast<size_t>(image.width()) * WP2FormatBpp(image.format());
}
}  // namespace

class ImageReaderJXL : public ImageReader::Impl {
 public:
  ImageReaderJXL(const uint8_t* data, size_t data_size,
                 ArgbBuffer* const buffer, LogLevel log_level,
                 size_t max_num_pixels)
      : ImageReader::Impl(buffer, data, data_size, log_level, max_num_pixels) {}

  WP2Status ReadFrame(bool* const is_last,
                      uint32_t* const duration_ms) override {
    const JxlDecoderPtr decoder = JxlDecoderMake(nullptr);
    WP2_CHECK_ALLOC_OK(decoder != nullptr);

    JxlDecoderStatus status = JxlDecoderSubscribeEvents(
        decoder.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE);
    WP2_CHECK_OK(status == JXL_DEC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

    status = JxlDecoderSetInput(decoder.get(), data_, data_size_);
    WP2_CHECK_OK(status == JXL_DEC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
    JxlDecoderCloseInput(decoder.get());

    status = JxlDecoderProcessInput(decoder.get());
    WP2_CHECK_OK(status == JXL_DEC_BASIC_INFO,
                 WP2_STATUS_INVALID_CONFIGURATION);

    JxlBasicInfo info;
    status = JxlDecoderGetBasicInfo(decoder.get(), &info);
    WP2_CHECK_OK(status == JXL_DEC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

    status = JxlDecoderProcessInput(decoder.get());
    WP2_CHECK_OK(status == JXL_DEC_NEED_IMAGE_OUT_BUFFER,
                 WP2_STATUS_INVALID_CONFIGURATION);

    ArgbBuffer buffer_tmp;
    ArgbBuffer* buffer = buffer_;
    if (buffer->format() != (info.alpha_bits > 0 ? WP2_RGBA_32 : WP2_RGB_24)) {
      WP2_CHECK_STATUS(
          buffer_tmp.SetFormat(info.alpha_bits > 0 ? WP2_RGBA_32 : WP2_RGB_24));
      buffer = &buffer_tmp;
    }
    WP2_CHECK_STATUS(buffer->Resize(info.xsize, info.ysize));
    const JxlPixelFormat pixel_format = ArgbBufferToJxlPixelFormat(*buffer);
    status =
        JxlDecoderSetImageOutBuffer(decoder.get(), &pixel_format,
                                    buffer->GetRow(0), ArgbBufferSize(*buffer));
    WP2_CHECK_OK(status == JXL_DEC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

    status = JxlDecoderProcessInput(decoder.get());
    WP2_CHECK_OK(status == JXL_DEC_FULL_IMAGE,
                 WP2_STATUS_INVALID_CONFIGURATION);

    status = JxlDecoderProcessInput(decoder.get());
    WP2_CHECK_OK(status == JXL_DEC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
    if (buffer != buffer_) WP2_CHECK_STATUS(buffer_->ConvertFrom(*buffer));
    return WP2_STATUS_OK;
  }

 protected:
  JxlDecoderPtr decoder_;
  int last_timestamp_ = 0;
  size_t frame_index_ = 0;
};

void ImageReader::SetImplJXL(ArgbBuffer* const buffer, LogLevel log_level,
                             size_t max_num_pixels) {
  impl_.reset(new (WP2Allocable::nothrow) ImageReaderJXL(
      data_.bytes, data_.size, buffer, log_level, max_num_pixels));
  if (impl_ == nullptr) status_ = WP2_STATUS_OUT_OF_MEMORY;
}

}  // namespace WP2

#else  // !WP2_HAVE_JXL

void WP2::ImageReader::SetImplJXL(WP2::ArgbBuffer* const buffer,
                                  WP2::LogLevel log_level,
                                  size_t max_num_pixels) {
  (void)buffer;
  (void)max_num_pixels;
  if (log_level >= WP2::LogLevel::DEFAULT) {
    fprintf(stderr,
            "JPEG XL support not compiled. Please install the libjxl "
            "development package before building.\n");
  }
  status_ = WP2_STATUS_UNSUPPORTED_FEATURE;
}

#endif  // WP2_HAVE_JXL

// -----------------------------------------------------------------------------
