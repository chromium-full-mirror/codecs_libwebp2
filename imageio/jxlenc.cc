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
// JPEG-XL encode.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "./image_enc.h"
#include "src/wp2/base.h"

#ifdef HAVE_CONFIG_H
#include "wp2/config.h"
#endif

#ifdef WP2_HAVE_JXL
#include <cstddef>
#include <cstdint>

#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/format_constants.h"
#include "third_party/jpegxl/lib/include/jxl/codestream_header.h"
#include "third_party/jpegxl/lib/include/jxl/color_encoding.h"
#include "third_party/jpegxl/lib/include/jxl/encode.h"
#include "third_party/jpegxl/lib/include/jxl/encode_cxx.h"
#include "third_party/jpegxl/lib/include/jxl/types.h"
#endif

namespace WP2 {

// -----------------------------------------------------------------------------

#if defined(WP2_HAVE_JXL)

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

WP2Status CompressJXL(const ArgbBuffer& buffer_in, float quality, int effort,
                      Writer* const output) {
  ArgbBuffer buffer(buffer_in.HasTransparency() ? WP2_RGBA_32 : WP2_RGB_24);
  WP2_CHECK_STATUS(buffer.ConvertFrom(buffer_in));
  WP2_CHECK_OK(buffer.format() == WP2_RGBA_32 || buffer.format() == WP2_RGB_24,
               WP2_STATUS_INVALID_CONFIGURATION);

  const JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
  WP2_CHECK_ALLOC_OK(encoder != nullptr);

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = buffer.width();
  basic_info.ysize = buffer.height();
  basic_info.bits_per_sample = WP2Formatbpc(buffer.format());
  basic_info.uses_original_profile = JXL_TRUE;  // mandatory for lossless
  basic_info.num_color_channels = 3;
  if (WP2FormatHasAlpha(buffer.format())) {
    basic_info.num_extra_channels = 1;
    basic_info.alpha_bits = basic_info.bits_per_sample;
    basic_info.alpha_premultiplied = WP2IsPremultiplied(buffer.format());
    // JxlEncoderSetExtraChannelInfo() does not need to be called for alpha
    // apparently.
  }
  JxlEncoderStatus status = JxlEncoderSetBasicInfo(encoder.get(), &basic_info);
  WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

  JxlColorEncoding color_encoding = {};
  JxlColorEncodingSetToSRGB(&color_encoding, /*is_gray=*/JXL_FALSE);
  status = JxlEncoderSetColorEncoding(encoder.get(), &color_encoding);
  WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
  WP2_CHECK_ALLOC_OK(frame_settings != nullptr);
  if (quality == kMaxQuality) {
    status = JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
    WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
    // JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE should be ON by default if lossless.
  } else {
    const float distance = quality >= 100  ? 0.0
                           : quality >= 30 ? 0.1 + (100 - quality) * 0.09
                                           : 53.0 / 3000.0 * quality * quality -
                                                 23.0 / 20.0 * quality + 25.0;
    status = JxlEncoderSetFrameDistance(frame_settings, distance);
    WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
  }
  // Allow for the maximum effort.
  const int scaled_effort = std::round(1 + 10. * effort / kMaxEffort);
  JxlEncoderAllowExpertOptions(encoder.get());
  status = JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, scaled_effort);
  WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);

  const JxlPixelFormat pixel_format = ArgbBufferToJxlPixelFormat(buffer);
  status = JxlEncoderAddImageFrame(frame_settings, &pixel_format,
                                   buffer.GetRow(0), ArgbBufferSize(buffer));
  WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
  JxlEncoderCloseInput(encoder.get());

  WP2::Vector<uint8_t> vec;
  WP2_CHECK_ALLOC_OK(vec.resize(64));

  uint8_t* next_out = vec.data();
  size_t avail_out = vec.size() - (next_out - vec.data());
  do {
    status = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - vec.data();
      WP2_CHECK_ALLOC_OK(vec.resize(vec.size() * 2));
      next_out = vec.data() + offset;
      avail_out = vec.size() - offset;
    }
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  WP2_CHECK_ALLOC_OK(output->Append(vec.data(), next_out - vec.data()));
  WP2_CHECK_OK(status == JXL_ENC_SUCCESS, WP2_STATUS_INVALID_CONFIGURATION);
  return WP2_STATUS_OK;
}

#else

WP2Status CompressJXL(const ArgbBuffer&, float, int, Writer*) {
  return WP2_STATUS_UNSUPPORTED_FEATURE;
}

#endif  // defined(WP2_HAVE_JXL)

// -----------------------------------------------------------------------------

}  // namespace WP2
