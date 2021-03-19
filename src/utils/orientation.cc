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
//  Orientation, rotation functions
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/utils/orientation.h"

#include <algorithm>
#include <cassert>

#include "src/dsp/dsp.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

namespace WP2 {

//------------------------------------------------------------------------------

uint32_t OrientateWidth(Orientation orientation, uint32_t width,
                        uint32_t height) {
  return (orientation == Orientation::k90Clockwise ||
          orientation == Orientation::k270Clockwise ||
          orientation == Orientation::kFlipBotLeftToTopRgt ||
          orientation == Orientation::kFlipTopLeftToBotRgt)
             ? height
             : width;
}

uint32_t OrientateHeight(Orientation orientation, uint32_t width,
                         uint32_t height) {
  return (orientation == Orientation::k90Clockwise ||
          orientation == Orientation::k270Clockwise ||
          orientation == Orientation::kFlipBotLeftToTopRgt ||
          orientation == Orientation::kFlipTopLeftToBotRgt)
             ? width
             : height;
}

//------------------------------------------------------------------------------

uint32_t OrientatePointX(Orientation orientation, uint32_t old_width,
                         uint32_t old_height, uint32_t x, uint32_t y) {
  assert(x < old_width && y < old_height);

  if (orientation == Orientation::k90Clockwise ||
      orientation == Orientation::kFlipTopLeftToBotRgt) {
    return old_height - y - 1u;
  } else if (orientation == Orientation::k180 ||
             orientation == Orientation::kFlipLeftToRight) {
    return old_width - x - 1u;
  } else if (orientation == Orientation::k270Clockwise ||
             orientation == Orientation::kFlipBotLeftToTopRgt) {
    return y;
  }
  return x;
}

uint32_t OrientatePointY(Orientation orientation, uint32_t old_width,
                         uint32_t old_height, uint32_t x, uint32_t y) {
  assert(x < old_width && y < old_height);

  if (orientation == Orientation::k90Clockwise ||
      orientation == Orientation::kFlipBotLeftToTopRgt) {
    return x;
  } else if (orientation == Orientation::k180 ||
             orientation == Orientation::kFlipTopToBottom) {
    return old_height - y - 1u;
  } else if (orientation == Orientation::k270Clockwise ||
             orientation == Orientation::kFlipTopLeftToBotRgt) {
    return old_width - x - 1u;
  }
  return y;
}

//------------------------------------------------------------------------------

Rectangle OrientateRectangle(Orientation orientation, uint32_t old_width,
                             uint32_t old_height, const Rectangle& rect) {
  assert(rect.x + rect.width <= old_width &&
         rect.y + rect.height <= old_height);
  Rectangle rotated_rect(
      OrientatePointX(orientation, old_width, old_height, rect.x, rect.y),
      OrientatePointY(orientation, old_width, old_height, rect.x, rect.y),
      OrientateWidth(orientation, rect.width, rect.height),
      OrientateHeight(orientation, rect.width, rect.height));
  // x, y represents top left, not any corner.
  if (orientation == Orientation::k90Clockwise ||
      orientation == Orientation::k180 ||
      orientation == Orientation::kFlipLeftToRight ||
      orientation == Orientation::kFlipTopLeftToBotRgt) {
    if (rotated_rect.width > 1) rotated_rect.x -= rotated_rect.width - 1;
  }
  if (orientation == Orientation::k180 ||
      orientation == Orientation::k270Clockwise ||
      orientation == Orientation::kFlipTopToBottom ||
      orientation == Orientation::kFlipTopLeftToBotRgt) {
    if (rotated_rect.height > 1) rotated_rect.y -= rotated_rect.height - 1;
  }
  return rotated_rect;
}

//------------------------------------------------------------------------------

WP2Status OrientateBuffer(Orientation orientation, ArgbBuffer* const buffer) {
  WP2_CHECK_OK(buffer != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!buffer->IsEmpty(), WP2_STATUS_INVALID_PARAMETER);

  if (orientation == Orientation::k90Clockwise ||
      orientation == Orientation::k270Clockwise ||
      orientation == Orientation::kFlipBotLeftToTopRgt ||
      orientation == Orientation::kFlipTopLeftToBotRgt) {
    // Note: Rotating or mirroring a non-square image in place is not trivial;
    //       using a temporary array instead.
    // TODO(yguyon): rotate directly during decoding
    ArgbBuffer tmp(buffer->format());
    WP2_CHECK_STATUS(tmp.Resize(buffer->height(), buffer->width()));
    WP2_CHECK_STATUS(OrientateSubBuffer(
        orientation, *buffer, buffer->AsRect(), &tmp));

    if (buffer->IsView()) {
      WP2_CHECK_STATUS(buffer->CopyFrom(tmp));
    } else {
      WP2_CHECK_STATUS(buffer->Swap(&tmp));
    }
  } else if (orientation == Orientation::k180) {
    // Swap first half of all pixels in scan-order with opposites.
    const uint32_t pixel_depth = WP2FormatBpp(buffer->format());
    uint8_t* row = buffer->GetRow8(0);
    uint8_t* opposite_row = buffer->GetRow8(buffer->height() - 1) +
                            (buffer->width() - 1) * pixel_depth;
    while (row < opposite_row) {
      uint8_t* pixel = row;
      uint8_t* opposite_pixel = opposite_row;
      for (uint32_t x = 0; x < buffer->width() && pixel < opposite_pixel; ++x) {
        for (uint32_t c = 0; c < pixel_depth; ++c) {
          std::swap(pixel[c], opposite_pixel[c]);
        }
        pixel += pixel_depth;
        opposite_pixel -= pixel_depth;
      }
      row += buffer->stride();
      opposite_row -= buffer->stride();
    }
  } else if (orientation == Orientation::kFlipLeftToRight) {
    // Reverse each line.
    const uint32_t pixel_depth = WP2FormatBpp(buffer->format());
    uint8_t* row = buffer->GetRow8(0);
    for (uint32_t y = 0; y < buffer->height(); ++y, row += buffer->stride()) {
      uint8_t* pixel = row;
      uint8_t* opposite_pixel = row + (buffer->width() - 1) * pixel_depth;
      while (pixel < opposite_pixel) {
        for (uint32_t c = 0; c < pixel_depth; ++c) {
          std::swap(pixel[c], opposite_pixel[c]);
        }
        pixel += pixel_depth;
        opposite_pixel -= pixel_depth;
      }
    }
  } else if (orientation == Orientation::kFlipTopToBottom) {
    // Swap each line with its opposite.
    const uint32_t pixel_depth = WP2FormatBpp(buffer->format());
    uint8_t* row = buffer->GetRow8(0);
    uint8_t* opposite_row = buffer->GetRow8(buffer->height() - 1);
    while (row < opposite_row) {
      std::swap_ranges(row, row + buffer->width() * pixel_depth, opposite_row);
      row += buffer->stride();
      opposite_row -= buffer->stride();
    }
  }
  return WP2_STATUS_OK;
}

WP2Status OrientateBuffer(Orientation orientation, const ArgbBuffer& src,
                          ArgbBuffer* const dst) {
  WP2_CHECK_OK(dst != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!src.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_STATUS(
      dst->Resize(OrientateWidth(orientation, src.width(), src.height()),
                  OrientateHeight(orientation, src.width(), src.height())));
  WP2_CHECK_STATUS(OrientateSubBuffer(orientation, src, src.AsRect(), dst));
  return WP2_STATUS_OK;
}

WP2Status OrientateSubBuffer(Orientation orientation, const ArgbBuffer& src,
                             const Rectangle& rect_in_src,
                             ArgbBuffer* const dst) {
  WP2_CHECK_OK(dst != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!src.IsEmpty() && !dst->IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(src.format() == WP2_Argb_32 || src.format() == WP2_ARGB_32 ||
                   dst->format() == WP2_Argb_32 || dst->format() == WP2_ARGB_32,
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK((rect_in_src.x + rect_in_src.width <= src.width()) &&
                   (rect_in_src.y + rect_in_src.height <= src.height()),
               WP2_STATUS_INVALID_PARAMETER);
  if (rect_in_src.width == 0 || rect_in_src.height == 0) return WP2_STATUS_OK;

  const uint32_t dst_x = OrientatePointX(orientation, src.width(), src.height(),
                                         rect_in_src.x, rect_in_src.y);
  const uint32_t dst_y = OrientatePointY(orientation, src.width(), src.height(),
                                         rect_in_src.x, rect_in_src.y);
  const uint32_t dst_width =
      OrientateWidth(orientation, rect_in_src.width, rect_in_src.height);
  const uint32_t dst_height =
      OrientateHeight(orientation, rect_in_src.width, rect_in_src.height);
  WP2_CHECK_OK(dst_width <= dst->width(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(dst_height <= dst->height(), WP2_STATUS_INVALID_PARAMETER);

  const uint32_t src_pixel_depth = WP2FormatBpp(src.format());
  const uint32_t dst_pixel_depth = WP2FormatBpp(dst->format());
  const uint8_t* src_row = (const uint8_t*)src.GetRow(rect_in_src.y) +
                           rect_in_src.x * src_pixel_depth;
  uint8_t* dst_line = (uint8_t*)dst->GetRow(dst_y) + dst_x * dst_pixel_depth;

  int32_t dst_next_line;
  int32_t dst_next_pixel;
  if (orientation == Orientation::kOriginal) {
    dst_next_line = (int32_t)dst->stride();
    dst_next_pixel = (int32_t)dst_pixel_depth;
  } else if (orientation == Orientation::k90Clockwise) {
    dst_next_line = -(int32_t)dst_pixel_depth;
    dst_next_pixel = (int32_t)dst->stride();
  } else if (orientation == Orientation::k180) {
    dst_next_line = -(int32_t)dst->stride();
    dst_next_pixel = -(int32_t)dst_pixel_depth;
  } else if (orientation == Orientation::k270Clockwise) {
    dst_next_line = (int32_t)dst_pixel_depth;
    dst_next_pixel = -(int32_t)dst->stride();
  } else if (orientation == Orientation::kFlipLeftToRight) {
    dst_next_line = (int32_t)dst->stride();
    dst_next_pixel = -(int32_t)dst_pixel_depth;
  } else if (orientation == Orientation::kFlipBotLeftToTopRgt) {
    dst_next_line = (int32_t)dst_pixel_depth;
    dst_next_pixel = (int32_t)dst->stride();
  } else if (orientation == Orientation::kFlipTopToBottom) {
    dst_next_line = -(int32_t)dst->stride();
    dst_next_pixel = (int32_t)dst_pixel_depth;
  } else {
    assert(orientation == Orientation::kFlipTopLeftToBotRgt);
    dst_next_line = -(int32_t)dst_pixel_depth;
    dst_next_pixel = -(int32_t)dst->stride();
  }

  if (src.format() == dst->format()) {
    for (uint32_t y = 0; y < rect_in_src.height; ++y) {
      const uint8_t* src_pixel = src_row;
      uint8_t* dst_pixel = dst_line;
      for (uint32_t x = 0; x < rect_in_src.width; ++x) {
        for (uint32_t c = 0; c < src_pixel_depth; ++c) {
          dst_pixel[c] = src_pixel[c];
        }
        src_pixel += src_pixel_depth;
        dst_pixel += dst_next_pixel;
      }
      src_row += src.stride();
      dst_line += dst_next_line;
    }
  } else {
    WP2ArgbConverterInit();
    const WP2ArgbConverterF convert =
        WP2ConversionFunction(src.format(), dst->format());
    for (uint32_t y = 0; y < rect_in_src.height; ++y) {
      const uint8_t* src_pixel = src_row;
      uint8_t* dst_pixel = dst_line;
      for (uint32_t x = 0; x < rect_in_src.width; ++x) {
        // Not optimized but the orientation stuff should be removed eventually.
        convert(src_pixel, /*width=*/1u, dst_pixel);
        src_pixel += src_pixel_depth;
        dst_pixel += dst_next_pixel;
      }
      src_row += src.stride();
      dst_line += dst_next_line;
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status OrientateBuffer(Orientation orientation, Plane16* const buffer) {
  WP2_CHECK_OK(buffer != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!buffer->IsEmpty(), WP2_STATUS_INVALID_PARAMETER);

  if (orientation == Orientation::k90Clockwise ||
      orientation == Orientation::k270Clockwise ||
      orientation == Orientation::kFlipBotLeftToTopRgt ||
      orientation == Orientation::kFlipTopLeftToBotRgt) {
    // Note: Rotating or mirroring a non-square image in place is not trivial;
    //       using a temporary array instead.
    Plane16 tmp;
    WP2_CHECK_STATUS(tmp.Resize(buffer->h_, buffer->w_));
    WP2_CHECK_STATUS(OrientateSubBuffer(orientation, *buffer,
                                        buffer->AsRect(), &tmp));
    if (buffer->IsView()) {
      WP2_CHECK_STATUS(buffer->Copy(tmp, /*resize_if_needed=*/false));
    } else {
      using std::swap;
      swap(*buffer, tmp);
    }
  } else if (orientation == Orientation::k180) {
    // Swap first half with opposites.
    int16_t* row = &buffer->At(0, 0);
    int16_t* opposite_row = &buffer->At(buffer->w_ - 1, buffer->h_ - 1);
    while (row < opposite_row) {
      int16_t* pixel = row;
      int16_t* opposite_pixel = opposite_row;
      for (uint32_t x = 0; x < buffer->w_ && pixel < opposite_pixel; ++x) {
        std::swap(*pixel, *opposite_pixel);
        pixel += 1;
        opposite_pixel -= 1;
      }
      row += buffer->Step();
      opposite_row -= buffer->Step();
    }
  } else if (orientation == Orientation::kFlipLeftToRight) {
    // Reverse each line.
    int16_t* row = buffer->Row(0);
    for (uint32_t y = 0; y < buffer->h_; ++y, row += buffer->step_) {
      std::reverse(row, row + buffer->w_);
    }
  } else if (orientation == Orientation::kFlipTopToBottom) {
    // Swap each line with its opposite.
    int16_t* row = buffer->Row(0);
    int16_t* opposite_row = buffer->Row(buffer->h_ - 1);
    while (row < opposite_row) {
      std::swap_ranges(row, row + buffer->w_, opposite_row);
      row += buffer->step_;
      opposite_row -= buffer->step_;
    }
  }
  return WP2_STATUS_OK;
}

WP2Status OrientateBuffer(Orientation orientation, const Plane16& src,
                          Plane16* const dst) {
  WP2_CHECK_OK(dst != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!src.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  const uint32_t dst_width = OrientateWidth(orientation, src.w_, src.h_);
  const uint32_t dst_height = OrientateHeight(orientation, src.w_, src.h_);
  if (!dst->IsView()) WP2_CHECK_STATUS(dst->Resize(dst_width, dst_height));
  WP2_CHECK_STATUS(OrientateSubBuffer(orientation, src, src.AsRect(), dst));
  return WP2_STATUS_OK;
}

WP2Status OrientateSubBuffer(Orientation orientation, const Plane16& src,
                             const Rectangle& rect_in_src, Plane16* const dst) {
  WP2_CHECK_OK(dst != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!src.IsEmpty() && !dst->IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK((rect_in_src.x + rect_in_src.width <= src.w_) &&
                   (rect_in_src.y + rect_in_src.height <= src.h_),
               WP2_STATUS_INVALID_PARAMETER);
  if (rect_in_src.width == 0 || rect_in_src.height == 0) return WP2_STATUS_OK;

  const uint32_t dst_x = OrientatePointX(orientation, src.w_, src.h_,
                                         rect_in_src.x, rect_in_src.y);
  const uint32_t dst_y = OrientatePointY(orientation, src.w_, src.h_,
                                         rect_in_src.x, rect_in_src.y);
  const uint32_t dst_width =
      OrientateWidth(orientation, rect_in_src.width, rect_in_src.height);
  const uint32_t dst_height =
      OrientateHeight(orientation, rect_in_src.width, rect_in_src.height);
  WP2_CHECK_OK(dst_width <= dst->w_, WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(dst_height <= dst->h_, WP2_STATUS_INVALID_PARAMETER);

  const int16_t* src_row = &src.At(rect_in_src.x, rect_in_src.y);
  int16_t* dst_line = &dst->At(dst_x, dst_y);

  int32_t dst_next_line;
  int32_t dst_next_pixel;
  if (orientation == Orientation::kOriginal) {
    dst_next_line = (int32_t)dst->Step();
    dst_next_pixel = 1;
  } else if (orientation == Orientation::k90Clockwise) {
    dst_next_line = -1;
    dst_next_pixel = (int32_t)dst->Step();
  } else if (orientation == Orientation::k180) {
    dst_next_line = -(int32_t)dst->Step();
    dst_next_pixel = -1;
  } else if (orientation == Orientation::k270Clockwise) {
    dst_next_line = 1;
    dst_next_pixel = -(int32_t)dst->Step();
  } else if (orientation == Orientation::kFlipLeftToRight) {
    dst_next_line = (int32_t)dst->Step();
    dst_next_pixel = -1;
  } else if (orientation == Orientation::kFlipBotLeftToTopRgt) {
    dst_next_line = 1;
    dst_next_pixel = (int32_t)dst->Step();
  } else if (orientation == Orientation::kFlipTopToBottom) {
    dst_next_line = -(int32_t)dst->Step();
    dst_next_pixel = 1;
  } else {
    assert(orientation == Orientation::kFlipTopLeftToBotRgt);
    dst_next_line = -1;
    dst_next_pixel = -(int32_t)dst->Step();
  }

  for (uint32_t y = 0; y < rect_in_src.height; ++y) {
    const int16_t* src_pixel = src_row;
    int16_t* dst_pixel = dst_line;
    for (uint32_t x = 0; x < rect_in_src.width; ++x) {
      *dst_pixel = *src_pixel;
      src_pixel += 1;
      dst_pixel += dst_next_pixel;
    }
    src_row += src.Step();
    dst_line += dst_next_line;
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status OrientateBuffer(Orientation orientation, YUVPlane* const buffer) {
  WP2_CHECK_OK(buffer != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!buffer->IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && buffer->GetChannel(channel).IsEmpty()) continue;
    WP2_CHECK_STATUS(
        OrientateBuffer(orientation, &buffer->GetChannel(channel)));
  }
  return WP2_STATUS_OK;
}

WP2Status OrientateBuffer(Orientation orientation, const YUVPlane& src,
                          YUVPlane* const dst) {
  WP2_CHECK_OK(!src.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(dst != nullptr, WP2_STATUS_NULL_PARAMETER);
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && src.GetChannel(channel).IsEmpty()) {
      dst->GetChannel(channel).Clear();
      continue;
    }
    WP2_CHECK_STATUS(OrientateBuffer(orientation, src.GetChannel(channel),
                                     &dst->GetChannel(channel)));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

Orientation GetInverseOrientation(Orientation orientation) {
  static constexpr Orientation kInverseOrientation[8] = {
      Orientation::kOriginal,
      Orientation::kFlipLeftToRight,
      Orientation::k180,
      Orientation::kFlipTopToBottom,
      Orientation::kFlipBotLeftToTopRgt,
      Orientation::k270Clockwise,
      Orientation::kFlipTopLeftToBotRgt,
      Orientation::k90Clockwise
  };

  return kInverseOrientation[(uint32_t)orientation];
}

//------------------------------------------------------------------------------

}  // namespace WP2
