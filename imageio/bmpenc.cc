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
// BMP Image writer

#include <cstdint>
#include <cstdio>

#include "imageio/image_enc.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

namespace WP2 {

namespace {

void PutLE16(uint8_t* const dst, uint32_t value) {
  dst[0] = (value >> 0) & 0xff;
  dst[1] = (value >> 8) & 0xff;
}

void PutLE32(uint8_t* const dst, uint32_t value) {
  PutLE16(dst + 0, (value >> 0) & 0xffff);
  PutLE16(dst + 2, (value >> 16) & 0xffff);
}

}  // namespace

static constexpr uint32_t kBMPHeaderSize = 54u;
static constexpr uint32_t kBMPAlphaHeaderSize = 70u;  // + 16 bytes of masks

WP2Status WriteBMP(const ArgbBuffer& buffer, FILE* fout) {
  WP2_CHECK_OK(fout != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!buffer.IsEmpty(), WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(WP2Formatbpc(buffer.format()) == 8,
               WP2_STATUS_INVALID_PARAMETER);
  const bool has_alpha = buffer.HasTransparency();

  ArgbBuffer converted_row(has_alpha ? WP2_BGRA_32 : WP2_BGR_24);
  const uint8_t bytes_per_px = WP2FormatBpp(converted_row.format());

  const uint32_t width = buffer.width();
  const uint32_t height = buffer.height();
  const uint32_t line_size = bytes_per_px * width;
  const uint32_t bmp_stride = (line_size + 3) & ~3;  // pad to 4
  const uint32_t header_size = has_alpha ? kBMPAlphaHeaderSize : kBMPHeaderSize;
  const uint32_t total_size = bmp_stride * height + header_size;
  const uint32_t format = has_alpha ? 3 : 0;  // BI_BITFIELDS or BI_RGB

  // bitmap file header
  uint8_t bmp_header[kBMPAlphaHeaderSize] = {0};
  PutLE16(bmp_header + 0, 0x4d42);        // signature 'BM'
  PutLE32(bmp_header + 2, total_size);    // size including header
  PutLE32(bmp_header + 6, 0);             // reserved
  PutLE32(bmp_header + 10, header_size);  // offset to pixel array
  // bitmap info header
  PutLE32(bmp_header + 14, header_size - 14);  // DIB header size
  PutLE32(bmp_header + 18, width);             // dimensions
  PutLE32(bmp_header + 22, height);            // no vertical flip
  PutLE16(bmp_header + 26, 1);                 // number of planes
  PutLE16(bmp_header + 28, bytes_per_px * 8);  // bits per pixel
  PutLE32(bmp_header + 30, format);            // compression format
  PutLE32(bmp_header + 34, 0);                 // image size (dummy)
  PutLE32(bmp_header + 38, 2400);              // x pixels/meter
  PutLE32(bmp_header + 42, 2400);              // y pixels/meter
  PutLE32(bmp_header + 46, 0);                 // number of palette colors
  PutLE32(bmp_header + 50, 0);                 // important color count
  if (format == 3) {                           // BITMAPV3INFOHEADER complement
    PutLE32(bmp_header + 54, 0x00ff0000);      // red mask
    PutLE32(bmp_header + 58, 0x0000ff00);      // green mask
    PutLE32(bmp_header + 62, 0x000000ff);      // blue mask
    PutLE32(bmp_header + 66, 0xff000000);      // alpha mask
  }

  // TODO(skal): color profile

  // write header
  WP2_CHECK_OK(fwrite(bmp_header, header_size, 1, fout) == 1,
               WP2_STATUS_BAD_WRITE);

  // write pixel array
  for (uint32_t y = 0; y < height; ++y) {
    ArgbBuffer buffer_row(buffer.format());
    WP2_CHECK_STATUS(buffer_row.SetView(buffer, {0, height - y - 1, width, 1}));
    if (converted_row.format() == buffer_row.format()) {
      WP2_CHECK_STATUS(converted_row.SetView(buffer_row));
    } else {
      WP2_CHECK_STATUS(converted_row.ConvertFrom(buffer_row));
    }
    WP2_CHECK_OK(fwrite(converted_row.GetRow8(0), bytes_per_px,
                        converted_row.width(), fout) == converted_row.width(),
                 WP2_STATUS_BAD_WRITE);
    // Pad till the line length is a multiple of 4 bytes.
    if (bmp_stride > line_size) {
      const uint8_t padding[3] = {0, 0, 0};
      WP2_CHECK_OK(fwrite(padding, bmp_stride - line_size, 1, fout) == 1,
                   WP2_STATUS_BAD_WRITE);
    }
  }
  return WP2_STATUS_OK;
}

}  // namespace WP2
