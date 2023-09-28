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
// WP2ArgbBuffer
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cmath>
#include <cstring>  // for memcpy()

#include "src/common/color_precision.h"
#include "src/dsp/dsp.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

uint32_t WP2FormatBpp(WP2SampleFormat format) {
  if (format <= WP2_BGRX_32) return 4;
  if (format <= WP2_BGR_24) return 3;
  assert(format == WP2_Argb_38);
  return 8;  // actually 38 bits, stored as uint16
}

uint32_t WP2Formatbpc(WP2SampleFormat format) {
  if (format < WP2_Argb_38) return 8;
  assert(format == WP2_Argb_38);
  return 10;
}

bool WP2FormatHasAlpha(WP2SampleFormat format, uint32_t* alpha_channel_index) {
  switch (format) {
    case WP2_Argb_32:
    case WP2_ARGB_32:
    case WP2_Argb_38:
      if (alpha_channel_index != nullptr) *alpha_channel_index = 0;
      return true;
    case WP2_rgbA_32:
    case WP2_RGBA_32:
    case WP2_bgrA_32:
    case WP2_BGRA_32:
      if (alpha_channel_index != nullptr) *alpha_channel_index = 3;
      return true;
    case WP2_XRGB_32:
      if (alpha_channel_index != nullptr) *alpha_channel_index = 0;
      return false;
    case WP2_RGBX_32:
    case WP2_BGRX_32:
      if (alpha_channel_index != nullptr) *alpha_channel_index = 3;
      return false;
    default:
      if (alpha_channel_index != nullptr) {
        *alpha_channel_index = WP2FormatBpp(format);
      }
      return false;
  }
}

bool WP2IsPremultiplied(WP2SampleFormat format) {
  switch (format) {
    case WP2_Argb_32:
    case WP2_Argb_38:
    case WP2_rgbA_32:
    case WP2_bgrA_32:
      return true;
    case WP2_ARGB_32:
    case WP2_XRGB_32:
    case WP2_RGBA_32:
    case WP2_RGBX_32:
    case WP2_BGRA_32:
    case WP2_BGRX_32:
    case WP2_RGB_24:
    case WP2_BGR_24:
      return false;
    default:
      assert(false);
      return false;
  }
}

namespace WP2 {

static WP2Status CheckDimensions(uint32_t width, uint32_t height,
                                 uint32_t stride, WP2SampleFormat format) {
  WP2_CHECK_OK(width > 0 && height > 0, WP2_STATUS_BAD_DIMENSION);
  WP2_CHECK_OK(width <= kMaxBufferDimension, WP2_STATUS_BAD_DIMENSION);
  WP2_CHECK_OK(height <= kMaxBufferDimension, WP2_STATUS_BAD_DIMENSION);
  WP2_CHECK_OK((uint64_t)width * height <= kMaxBufferArea,
               WP2_STATUS_BAD_DIMENSION);
  WP2_CHECK_OK((uint64_t)stride >= (uint64_t)WP2FormatBpp(format) * width,
               WP2_STATUS_BAD_DIMENSION);
#if defined(WP2_MAX_ARGB_MEMORY)
  WP2_CHECK_OK((uint64_t)stride * height <= WP2_MAX_ARGB_MEMORY,
               WP2_STATUS_OUT_OF_MEMORY);
#endif
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

Rectangle Rectangle::ClipWith(const Rectangle& r) const {
  const uint32_t x_end = std::min(x + width, r.x + r.width);
  const uint32_t y_end = std::min(y + height, r.y + r.height);
  Rectangle clipped;
  clipped.x = std::max(x, r.x);
  clipped.y = std::max(y, r.y);
  clipped.width = (x_end > clipped.x) ? x_end - clipped.x : 0;
  clipped.height = (y_end > clipped.y) ? y_end - clipped.y : 0;
  return clipped;
}

Rectangle Rectangle::MergeWith(const Rectangle& r) const {
  Rectangle merged;
  merged.x = std::min(x, r.x);
  merged.y = std::min(y, r.y);
  merged.width = std::max(x + width, r.x + r.width) - merged.x;
  merged.height = std::max(y + height, r.y + r.height) - merged.y;
  return merged;
}

void Rectangle::Exclude(const Rectangle& r, Rectangle* const top,
                        Rectangle* const bot, Rectangle* const lft,
                        Rectangle* const rgt) const {
  const Rectangle excluded = r.ClipWith(*this);
  *top = {x, y, width, excluded.y - y};
  *bot = {x, excluded.y + excluded.height, width,
          (y + height) - (excluded.y + excluded.height)};
  *lft = {x, excluded.y, excluded.x - x, excluded.height};
  *rgt = {excluded.x + excluded.width, excluded.y,
          (x + width) - (excluded.x + excluded.width), excluded.height};
}

//------------------------------------------------------------------------------

ArgbBuffer::ArgbBuffer(WP2SampleFormat format_in) noexcept
    : format_(format_in) {}

ArgbBuffer::ArgbBuffer(ArgbBuffer&& other) noexcept : format_(other.format()) {
  TrivialMoveCtor(this, &other);
}

void ArgbBuffer::Deallocate() {
  if (!is_external_memory_) {
    WP2Free(private_memory_);
    private_memory_ = nullptr;
  } else {
    is_external_memory_ = false;
  }
  width_ = 0;
  height_ = 0;
  stride_ = 0;
  metadata_.Clear();
  pixels_ = nullptr;
  is_slow_memory_ = false;
}

const void* ArgbBuffer::GetRow(uint32_t y) const {
  return (WP2Formatbpc(format_) <= 8) ? (const void*)GetRow8(y)
                                      : (const void*)GetRow16(y);
}
void* ArgbBuffer::GetRow(uint32_t y) {
  return (WP2Formatbpc(format_) <= 8) ? (void*)GetRow8(y) : (void*)GetRow16(y);
}
const uint8_t* ArgbBuffer::GetRow8(uint32_t y) const {
  assert(y < height_ && WP2Formatbpc(format_) <= 8);
  return (const uint8_t*)pixels_ + y * stride_;
}
uint8_t* ArgbBuffer::GetRow8(uint32_t y) {
  assert(y < height_ && WP2Formatbpc(format_) <= 8);
  return (uint8_t*)pixels_ + y * stride_;
}
const uint16_t* ArgbBuffer::GetRow16(uint32_t y) const {
  assert(y < height_ && WP2Formatbpc(format_) > 8);
  return (uint16_t*)((const uint8_t*)pixels_ + y * stride_);
}
uint16_t* ArgbBuffer::GetRow16(uint32_t y) {
  assert(y < height_ && WP2Formatbpc(format_) > 8);
  return (uint16_t*)((uint8_t*)pixels_ + y * stride_);
}
const uint8_t* ArgbBuffer::GetPosition(uint32_t x, uint32_t y) const {
  assert(y < height_ && x < width_);
  return ((const uint8_t*)pixels_ + y * stride_ + x * WP2FormatBpp(format_));
}
uint8_t* ArgbBuffer::GetPosition(uint32_t x, uint32_t y) {
  assert(y < height_ && x < width_);
  return ((uint8_t*)pixels_ + y * stride_ + x * WP2FormatBpp(format_));
}

WP2Status ArgbBuffer::SetFormat(WP2SampleFormat format_in) {
  WP2_CHECK_OK(IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  format_ = format_in;
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::Resize(uint32_t new_width, uint32_t new_height) {
  const uint32_t new_stride = new_width * WP2FormatBpp(format_);
  WP2_CHECK_STATUS(CheckDimensions(new_width, new_height, new_stride, format_));

  if (width_ == new_width && height_ == new_height) {
    metadata_.Clear();
    return WP2_STATUS_OK;
  }
  // For external memory, we can't reallocate the buffer nor change the layout.
  WP2_CHECK_OK(!is_external_memory_, WP2_STATUS_BAD_DIMENSION);

  if (width_ == new_height && height_ == new_width) {
    // Changing layout is enough.
    width_  = new_width;
    height_ = new_height;
    stride_ = new_stride;
    metadata_.Clear();
    return WP2_STATUS_OK;
  }

  Deallocate();
  void* const new_pixels = WP2Malloc(new_height, new_stride);
  WP2_CHECK_ALLOC_OK(new_pixels != nullptr);
  pixels_ = new_pixels;
  width_  = new_width;
  height_ = new_height;
  stride_ = new_stride;
  is_external_memory_ = false;
  private_memory_ = new_pixels;
  is_slow_memory_ = false;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Import functions

WP2Status ArgbBuffer::CopyFrom(const ArgbBuffer& src) {
  WP2_CHECK_OK(src.format() == format_, WP2_STATUS_INVALID_COLORSPACE);
  if (src.IsEmpty()) {
    Deallocate();
    return WP2_STATUS_OK;
  }
  WP2_CHECK_STATUS(Resize(src.width_, src.height_));
  const size_t dst_stride = src.width_ * WP2FormatBpp(format_);
  for (uint32_t y = 0; y < height_; ++y) {
    std::memcpy(GetRow(y), src.GetRow(y), dst_stride);
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::ConvertFrom(const ArgbBuffer& src) {
  if (format_ == src.format()) return CopyFrom(src);
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(src.format(), format_);
  WP2_CHECK_OK(convert != nullptr, WP2_STATUS_INVALID_COLORSPACE);

  WP2_CHECK_STATUS(Resize(src.width_, src.height_));
  for (uint32_t y = 0; y < height_; ++y) {
    convert(src.GetRow(y), width_, GetRow(y));
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::Import(WP2SampleFormat input_format,
                             uint32_t new_width, uint32_t new_height,
                             const uint8_t* samples, uint32_t samples_stride) {
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(input_format, format_);
  WP2_CHECK_OK(convert != nullptr, WP2_STATUS_INVALID_COLORSPACE);

  WP2_CHECK_STATUS(Resize(new_width, new_height));
  for (uint32_t y = 0; y < new_height; ++y) {
    convert(samples, new_width, GetRow(y));
    samples += samples_stride;
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::ImportRow(WP2SampleFormat input_format,
                                uint32_t row_index, const uint8_t* samples) {
  WP2_CHECK_OK(samples != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(row_index <= height_, WP2_STATUS_INVALID_PARAMETER);
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(input_format, format_);
  WP2_CHECK_OK(convert != nullptr, WP2_STATUS_INVALID_COLORSPACE);

  convert(samples, width_, GetRow(row_index));
  return WP2_STATUS_OK;
}

void ArgbBuffer::Fill(const Rectangle& window, Argb32b color) {
  assert(format_ == WP2_Argb_32 || format_ == WP2_ARGB_32);
  assert(CheckPremultiplied(color));
  uint8_t bytes[4];
  ToUInt8(color, bytes);
  // Unmultiply.
  if (format_ == WP2_ARGB_32 && color.a != 255u && color.a != 0u) {
    for (uint32_t c = 1; c <= 3; ++c) {
      bytes[c] = DivRound<uint16_t>(bytes[c] * 255u, bytes[0]);
    }
  }
  Fill(window, bytes);
}

void ArgbBuffer::Fill(const Rectangle& window, Argb38b color) {
  assert(format_ == WP2_Argb_38);
  assert(CheckPremultiplied(color));
  const uint16_t color16[] = {color.a, color.r, color.g, color.b};
  Fill(window, color16);
}

void ArgbBuffer::Fill(const Rectangle& window, const uint8_t color[]) {
  assert(WP2Formatbpc(format_) <= 8);
  FillImpl(window, color);
}

void ArgbBuffer::Fill(const Rectangle& window, const uint16_t color[]) {
  assert(WP2Formatbpc(format_) > 8);
  FillImpl(window, color);
}

void ArgbBuffer::Fill(const Argb32b color) {
  Fill(AsRect(), color);
}

void ArgbBuffer::FillImpl(const Rectangle& window, const void* color) {
  const Rectangle r = window.ClipWith(AsRect());
  const uint32_t pixel_depth = WP2FormatBpp(format_);
  if (r.width == 0 || r.height == 0) return;
  // Even if channels are uint16_t, uint8_t is used as the stride and
  // pixel_depth take care of memory increments.
  uint8_t* dst_row = GetPosition(r.x, r.y);
  for (uint32_t y = 0; y < r.height; ++y) {
    uint8_t* dst_pixel = dst_row;
    for (uint32_t x = 0; x < r.width; ++x) {
      std::memcpy(dst_pixel, color, pixel_depth);
      dst_pixel += pixel_depth;
    }
    dst_row += stride_;
  }
}

void ArgbBuffer::DrawRect(const Rectangle& window, Argb32b color,
                          uint32_t edges_to_draw) {
  assert(CheckPremultiplied(color));
  // top
  if (edges_to_draw & 1) Fill({window.x, window.y, window.width, 1}, color);
  // left
  if (edges_to_draw & 2) Fill({window.x, window.y, 1, window.height}, color);
  // bottom
  if (edges_to_draw & 4) Fill({window.x, window.y + window.height - 1,
                               window.width, 1}, color);
  // right
  if (edges_to_draw & 8) Fill({window.x + window.width - 1, window.y,
                              1, window.height}, color);
  // mid horizontal
  if (edges_to_draw & 16) Fill({window.x, window.y + window.height / 2,
                                window.width, 1}, color);
  // mid vertical
  if (edges_to_draw & 32) Fill({window.x + window.width / 2, window.y,
                                1, window.height}, color);
}

WP2Status ArgbBuffer::CompositeOver(Argb32b background) {
  WP2_CHECK_OK(format_ == WP2_Argb_32, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(CheckPremultiplied(background), WP2_STATUS_INVALID_PARAMETER);
  for (uint32_t y = 0; y < height_; ++y) {
    uint8_t* const row = GetRow8(y);
    for (uint32_t x = 0; x < width_; ++x) {
      const uint8_t b = kAlphaMax - row[4 * x + 0];  // ~ 255 * (1-alpha)
      row[4 * x + 0] += DivBy255(b * background.a);
      row[4 * x + 1] += DivBy255(b * background.r);
      row[4 * x + 2] += DivBy255(b * background.g);
      row[4 * x + 3] += DivBy255(b * background.b);
    }
  }
  return WP2_STATUS_OK;
}

static WP2Status CompositeBuffers(const ArgbBuffer& background,
                                  const ArgbBuffer& foreground,
                                  const Rectangle& window,
                                  ArgbBuffer* const dest) {
  WP2_CHECK_OK(background.format() == WP2_Argb_32,
               WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(foreground.format() == WP2_Argb_32,
               WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(foreground.width() == background.width(),
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(foreground.height() == background.height(),
               WP2_STATUS_INVALID_PARAMETER);
  const Rectangle w =
      (window.GetArea() > 0) ? window : background.AsRect();
  WP2_CHECK_OK(window.x + window.width <= background.width(),
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(window.y + window.height <= background.height(),
               WP2_STATUS_INVALID_PARAMETER);

  for (uint32_t y = w.y; y < w.y + w.height; ++y) {
    const uint8_t* const fg_row = foreground.GetPosition(w.x, y);
    const uint8_t* const bg_row = background.GetPosition(w.x, y);
    uint8_t* const dest_row = dest->GetPosition(w.x, y);
    for (uint32_t x = 0; x < w.width; ++x) {
      const uint8_t b = kAlphaMax - fg_row[4 * x + 0];  // ~ 255 * (1-alpha)
      for (uint32_t c = 0; c < 4; ++c) {
        dest_row[4 * x + c] =
            fg_row[4 * x + c] + DivBy255(b * bg_row[4 * x + c]);
      }
    }
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::CompositeOver(const ArgbBuffer& background) {
  return CompositeBuffers(background, /*foreground=*/*this, /*window=*/{},
                          /*dest=*/this);
}

WP2Status ArgbBuffer::CompositeUnder(const ArgbBuffer& foreground,
                                     const Rectangle& window) {
  return CompositeBuffers(/*background=*/*this, foreground, window,
                          /*dest=*/this);
}

WP2Status ArgbBuffer::SetView(const ArgbBuffer& src) {
  return SetView(src, src.AsRect());
}

WP2Status ArgbBuffer::SetView(const ArgbBuffer& src, const Rectangle& window) {
  WP2_CHECK_OK(src.format() == format_, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(
      CheckDimensions(window.width, window.height, src.stride_, src.format()));
  WP2_CHECK_OK(window.x + window.width <= src.width_,
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(window.y + window.height <= src.height_,
               WP2_STATUS_INVALID_PARAMETER);
  if (!IsEmpty() && !IsView() && !src.IsEmpty() && src.IsView()) {
    // 'this' cannot be a view of 'src' if 'src' is already a view of 'this'.
    const void* end_pixels =
        (const uint8_t*)pixels_ +
        (SafeSub(height_, 1u) * stride_ + width_ * WP2FormatBpp(format_));
    WP2_CHECK_OK(src.pixels_ < pixels_ || src.pixels_ >= end_pixels,
                 WP2_STATUS_INVALID_PARAMETER);
  }
  if (&src != this) Deallocate();
  pixels_ = (void*)src.GetPosition(window.x, window.y);
  width_ = window.width;
  height_ = window.height;
  stride_ = src.stride_;
  is_external_memory_ = (&src != this) || IsView();
  is_slow_memory_ = src.is_slow_memory_;
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::SetExternalImpl(uint32_t new_width, uint32_t new_height,
                                      void* samples, uint32_t new_stride,
                                      bool is_slow) {
  WP2_CHECK_OK(samples != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_STATUS(CheckDimensions(new_width, new_height, new_stride, format_));
  Deallocate();
  width_ = new_width;
  height_ = new_height;
  pixels_ = samples;
  stride_ = new_stride;
  is_external_memory_ = true;
  is_slow_memory_ = is_slow;
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::SetExternal(uint32_t new_width, uint32_t new_height,
                                  uint8_t* samples, uint32_t new_stride,
                                  bool is_slow) {
  WP2_CHECK_OK(WP2Formatbpc(format_) <= 8, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(
      SetExternalImpl(new_width, new_height, samples, new_stride, is_slow));
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::SetExternal(uint32_t new_width, uint32_t new_height,
                                  uint16_t* samples, uint32_t new_stride,
                                  bool is_slow) {
  WP2_CHECK_OK(WP2Formatbpc(format_) > 8, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(
      SetExternalImpl(new_width, new_height, samples, new_stride, is_slow));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

bool ArgbBuffer::HasTransparency() const {
  if (IsEmpty()) return false;
  uint32_t alpha_channel_index;
  if (!WP2FormatHasAlpha(format_, &alpha_channel_index)) return false;
  if (WP2Formatbpc(format_) <= 8) {
    WP2AlphaInit();
    for (uint32_t y = 0; y < height_; ++y) {
      const uint8_t* const row = GetRow8(y) + alpha_channel_index;
      if (WP2HasOtherValue8b32b(row, width_, 0xff)) return true;
    }
  } else {
    for (uint32_t y = 0; y < height_; ++y) {
      const uint16_t* const row = GetRow16(y) + alpha_channel_index;
      for (uint32_t x = 0; x < width_; ++x) {
        if (row[4 * x + 0] != WP2::kAlphaMax) return true;
      }
    }
  }
  return false;
}

//------------------------------------------------------------------------------

void swap(ArgbBuffer& a, ArgbBuffer& b) {
  using std::swap;
  swap(a.format_, b.format_);
  swap(a.width_, b.width_);
  swap(a.height_, b.height_);
  swap(a.stride_, b.stride_);

  swap(a.metadata_, b.metadata_);

  swap(a.pixels_, b.pixels_);
  swap(a.is_external_memory_, b.is_external_memory_);
  swap(a.private_memory_, b.private_memory_);
  swap(a.is_slow_memory_, b.is_slow_memory_);
}

//------------------------------------------------------------------------------

void ArgbBuffer::SimpleHalfDownsample() {
  assert(WP2Formatbpc(format_) <= 8);  // not supported yet
  const uint32_t half_w = (width_ + 1) >> 1;
  const uint32_t half_w_low = width_ >> 1;
  const uint32_t half_h = (height_ + 1) >> 1;
  for (uint32_t y = 0; y < half_h; ++y) {
    const uint8_t* row1 = GetRow8(2 * y + 0);
    const uint32_t next_row = (2 * y + 1 == height_) ? 0 : 1;
    const uint8_t* row2 = GetRow8(2 * y + next_row);
    uint8_t* dst = GetRow8(y);
    for (uint32_t x = 0; x < half_w_low; ++x) {
      // TODO(skal): implement in dsp/
      for (uint32_t c = 0; c < 4; ++c) {
        *dst++ = (row1[c + 0] + row1[c + 4] +
                  row2[c + 0] + row2[c + 4] + 2) >> 2;
      }
      row1 += 2 * 4;
      row2 += 2 * 4;
    }
    if (half_w != half_w_low) {
      for (uint32_t c = 0; c < 4; ++c) dst[c] = row1[c];
    }
  }
  const WP2Status status = SetView(*this, {0, 0, half_w, half_h});
  (void)status;
  assert(status == WP2_STATUS_OK);  // should never fail
}

//------------------------------------------------------------------------------

}  // namespace WP2
