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
  if (format <= WP2_Argb_38) return 8;  // actually 38 bits, stored as uint16
  assert(false);
  return 2;
}

uint32_t WP2FormatBpc(WP2SampleFormat format) {
  if (format == WP2_Argb_38) return 2;
  return 1;
}

uint32_t WP2AlphaChannelIndex(WP2SampleFormat format) {
  switch (format) {
    case WP2_Argb_32:
    case WP2_ARGB_32:
    case WP2_Argb_38:
      return 0;
    case WP2_rgbA_32:
    case WP2_RGBA_32:
    case WP2_bgrA_32:
    case WP2_BGRA_32:
      return 3;
    default:
      return WP2FormatBpp(format);
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
                        Rectangle* const rgt) {
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
  return (WP2FormatBpc(format_) == 1) ? (const void*)GetRow8(y)
                                      : (const void*)GetRow16(y);
}
void* ArgbBuffer::GetRow(uint32_t y) {
  return (WP2FormatBpc(format_) == 1) ? (void*)GetRow8(y) : (void*)GetRow16(y);
}
const uint8_t* ArgbBuffer::GetRow8(uint32_t y) const {
  assert(y < height_ && WP2FormatBpc(format_) == 1);
  return (const uint8_t*)pixels_ + y * stride_;
}
uint8_t* ArgbBuffer::GetRow8(uint32_t y) {
  assert(y < height_ && WP2FormatBpc(format_) == 1);
  return (uint8_t*)pixels_ + y * stride_;
}
const uint16_t* ArgbBuffer::GetRow16(uint32_t y) const {
  assert(y < height_ && WP2FormatBpc(format_) == 2);
  return (uint16_t*)((const uint8_t*)pixels_ + y * stride_);
}
uint16_t* ArgbBuffer::GetRow16(uint32_t y) {
  assert(y < height_ && WP2FormatBpc(format_) == 2);
  return (uint16_t*)((uint8_t*)pixels_ + y * stride_);
}

WP2Status ArgbBuffer::SetFormat(WP2SampleFormat format_in) {
  WP2_CHECK_OK(IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  const_cast<WP2SampleFormat&>(format_) = format_in;
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
    width_ = new_width;
    height_ = new_height;
    stride_ = new_stride;
    metadata_.Clear();
    return WP2_STATUS_OK;
  }

  Deallocate();
  void* const new_pixels = WP2Malloc(new_height, new_stride);
  WP2_CHECK_ALLOC_OK(new_pixels != nullptr);
  pixels_ = new_pixels;
  width_ = new_width;
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
    void* const dst_ptr = GetRow(y);
    const void* const src_ptr = src.GetRow(y);
    std::memcpy(dst_ptr, src_ptr, dst_stride);
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::ConvertFrom(const ArgbBuffer& src) {
  if (format_ == src.format()) return CopyFrom(src);
  WP2_CHECK_OK(src.format() == WP2_Argb_32 || src.format() == WP2_ARGB_32 ||
                   format_ == WP2_Argb_32 || format_ == WP2_ARGB_32,
               WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(WP2FormatBpc(format_) == 1, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(WP2FormatBpc(src.format()) == 1, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(Resize(src.width_, src.height_));
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(src.format(), format_);
  for (uint32_t y = 0; y < height_; ++y) {
    uint8_t* dst_ptr = GetRow8(y);
    const uint8_t* src_ptr = src.GetRow8(y);
    convert(src_ptr, width_, dst_ptr);
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::Import(WP2SampleFormat input_format,
                             uint32_t new_width, uint32_t new_height,
                             const uint8_t* samples, uint32_t samples_stride) {
  WP2_CHECK_OK(input_format == WP2_Argb_32 || input_format == WP2_ARGB_32 ||
                   format_ == WP2_Argb_32 || format_ == WP2_ARGB_32,
               WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(Resize(new_width, new_height));
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(input_format, format_);
  for (uint32_t y = 0; y < new_height; ++y) {
    convert(samples, new_width, (uint8_t*)GetRow(y));
    samples += samples_stride;
  }
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::ImportRow(WP2SampleFormat input_format,
                                uint32_t row_index, const uint8_t* samples) {
  WP2_CHECK_OK(samples != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(input_format == WP2_Argb_32 || input_format == WP2_ARGB_32 ||
                   format_ == WP2_Argb_32 || format_ == WP2_ARGB_32,
               WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(row_index <= height_, WP2_STATUS_INVALID_PARAMETER);
  WP2ArgbConverterInit();
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(input_format, format_);
  convert(samples, width_, (uint8_t*)GetRow(row_index));
  return WP2_STATUS_OK;
}

void ArgbBuffer::Fill(const Rectangle& window, Argb32b color) {
  assert(format_ == WP2_Argb_32);
  assert(CheckPremultiplied(color));
  uint8_t bytes[4];
  ToUInt8(color, bytes);
  Fill(window, bytes);
}

void ArgbBuffer::Fill(const Rectangle& window, Argb38b color) {
  assert(format_ == WP2_Argb_38);
  assert(CheckPremultiplied(color));
  const uint16_t color16[] = {color.a, color.r, color.g, color.b};
  Fill(window, color16);
}

void ArgbBuffer::FillImpl(const Rectangle& window, const void* color) {
  const Rectangle r = window.ClipWith({0, 0, width_, height_});
  const uint32_t pixel_depth = WP2FormatBpp(format_);
  if (r.width == 0 || r.height == 0) return;
  // Even if channels are uint16_t, uint8_t is used as the stride and
  // pixel_depth take care of memory increments.
  uint8_t* dst_row = (uint8_t*)GetRow(r.y) + r.x * pixel_depth;
  for (uint32_t y = 0; y < r.height; ++y) {
    uint8_t* dst_pixel = dst_row;
    for (uint32_t x = 0; x < r.width; ++x) {
      std::memcpy(dst_pixel, color, pixel_depth);
      dst_pixel += pixel_depth;
    }
    dst_row += stride_;
  }
}

void ArgbBuffer::Fill(const Rectangle& window, const uint8_t color[]) {
  assert(WP2FormatBpc(format_) == 1);
  FillImpl(window, color);
}

void ArgbBuffer::Fill(const Rectangle& window, const uint16_t color[]) {
  assert(WP2FormatBpc(format_) == 2);
  FillImpl(window, color);
}

void ArgbBuffer::Fill(const Argb32b color) {
  Fill({0, 0, width_, height_}, color);
}

void ArgbBuffer::DrawRect(const Rectangle& window, Argb32b color) {
  assert(CheckPremultiplied(color));
  Fill({window.x, window.y, 1, window.height}, color);
  Fill({window.x, window.y, window.width, 1}, color);
  Fill({window.x + window.width - 1, window.y, 1, window.height}, color);
  Fill({window.x, window.y + window.height - 1, window.width, 1}, color);
}

WP2Status ArgbBuffer::CompositeOver(Argb32b background) {
  WP2_CHECK_OK(format_ == WP2_Argb_32, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_OK(CheckPremultiplied(background), WP2_STATUS_INVALID_PARAMETER);
  for (uint32_t y = 0; y < height_; ++y) {
    uint8_t* row = (uint8_t*)GetRow(y);
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
      (window.GetArea() > 0)
          ? window
          : Rectangle{0, 0, background.width(), background.height()};
  WP2_CHECK_OK(window.x + window.width <= background.width(),
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(window.y + window.height <= background.height(),
               WP2_STATUS_INVALID_PARAMETER);

  for (uint32_t y = w.y; y < w.y + w.height; ++y) {
    const uint8_t* const fg_row = foreground.GetRow8(y);
    const uint8_t* const bg_row = background.GetRow8(y);
    uint8_t* const dest_row = dest->GetRow8(y);
    for (uint32_t x = w.x; x < w.x + w.width; ++x) {
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
  return SetView(src, {0, 0, src.width_, src.height_});
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
  pixels_ =
      (uint8_t*)src.GetRow(window.y) + window.x * WP2FormatBpp(src.format());
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
  WP2_CHECK_OK(WP2FormatBpc(format_) == 1, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(
      SetExternalImpl(new_width, new_height, samples, new_stride, is_slow));
  return WP2_STATUS_OK;
}

WP2Status ArgbBuffer::SetExternal(uint32_t new_width, uint32_t new_height,
                                  uint16_t* samples, uint32_t new_stride,
                                  bool is_slow) {
  WP2_CHECK_OK(WP2FormatBpc(format_) == 2, WP2_STATUS_INVALID_COLORSPACE);
  WP2_CHECK_STATUS(
      SetExternalImpl(new_width, new_height, samples, new_stride, is_slow));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

bool ArgbBuffer::HasTransparency() const {
  if (IsEmpty()) return false;
  if (WP2FormatBpp(format_) == 3) return false;  // TODO(skal): refine
  if (WP2FormatBpc(format_) == 1) {
    WP2AlphaInit();
    for (uint32_t y = 0; y < height_; ++y) {
      const uint8_t* const row = (const uint8_t*)GetRow(y);
      if (WP2HasOtherValue8b32b(row, width_, 0xff)) return true;
    }
  } else {
    for (uint32_t y = 0; y < height_; ++y) {
      const int16_t* const row = (const int16_t*)GetRow(y);
      for (uint32_t x = 0; x < width_; ++x) {
        if (row[4 * x + 0] != WP2::kAlphaMax) return true;
      }
    }
  }
  return false;
}

//------------------------------------------------------------------------------

WP2Status ArgbBuffer::Swap(ArgbBuffer* const other) {
  WP2_CHECK_OK(format_ == other->format(), WP2_STATUS_INVALID_PARAMETER);

  using std::swap;
  swap(width_, other->width_);
  swap(height_, other->height_);
  swap(stride_, other->stride_);

  swap(metadata_, other->metadata_);

  swap(pixels_, other->pixels_);
  swap(private_memory_, other->private_memory_);
  swap(is_external_memory_, other->is_external_memory_);

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void ArgbBuffer::SimpleHalfDownsample() {
  assert(WP2FormatBpc(format_) == 1);  // not supported yet
  const uint32_t half_w = (width_ + 1) >> 1;
  const uint32_t half_w_low = width_ >> 1;
  const uint32_t half_h = (height_ + 1) >> 1;
  for (uint32_t y = 0; y < half_h; ++y) {
    const uint8_t* row1 = (const uint8_t*)GetRow(2 * y + 0);
    const uint32_t next_row = (2 * y + 1 == height_) ? 0 : 1;
    const uint8_t* row2 = (const uint8_t*)GetRow(2 * y + next_row);
    uint8_t* dst = (uint8_t*)GetRow(y);
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
