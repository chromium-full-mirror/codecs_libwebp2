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
// Speed-critical convert-to-Argb functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cstring>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

namespace {

//------------------------------------------------------------------------------
// C conversion from a format to another

void Unmult(uint32_t a, uint8_t* const v) {
  if (a > 0) {
    const uint32_t M = WP2::kAlphaDiv[a];
    for (int c : {0, 1, 2}) {
      v[c] = std::min(WP2::DivByAlphaDiv(v[c], M), 255u);
    }
  } else {
    v[0] = v[1] = v[2] = 0;
  }
}

void Premult(uint32_t a, uint8_t* const v) {
  if (a < 255) {
    if (a > 0) {
      for (int c : {0, 1, 2}) {
        v[c] = WP2::DivBy255(v[c] * a);
      }
    } else {
      v[0] = v[1] = v[2] = 0;
    }
  }
}

#define CONVERT_32_FUNC(NAME, SRC_A, SRC_R, SRC_G, SRC_B, SRC_STEP, DST_A,   \
                        DST_R, DST_G, DST_B, DST_STEP, PREMULT, FILL)        \
  void NAME(const void* src_data, uint32_t width, void* dst_data) {          \
    const uint8_t* src = (const uint8_t*)src_data;                           \
    uint8_t* dst = (uint8_t*)dst_data;                                       \
    const int min_pos = std::min({(DST_R), (DST_G), (DST_B)});               \
    for (uint32_t i = 0; i < width; ++i, src += SRC_STEP, dst += DST_STEP) { \
      if ((DST_STEP) == 4) dst[(DST_A)] = (FILL) ? 0xff : src[(SRC_A)];      \
      dst[(DST_R)] = src[(SRC_R)];                                           \
      dst[(DST_G)] = src[(SRC_G)];                                           \
      dst[(DST_B)] = src[(SRC_B)];                                           \
      if ((PREMULT) > 0) {                                                   \
        Premult(src[(SRC_A)], dst + min_pos);                                \
      } else if ((PREMULT) < 0) {                                            \
        Unmult(src[(SRC_A)], dst + min_pos);                                 \
      }                                                                      \
    }                                                                        \
  }

#define CONVERT_32_TO_ARGB32_FUNC(NAME, A, R, G, B, PREMULT, FILL, SRC_STEP) \
  CONVERT_32_FUNC(NAME, A, R, G, B, SRC_STEP, 0, 1, 2, 3, /*DST_STEP=*/4,    \
                  PREMULT, FILL)

#define CONVERT_ARGB32_TO_32_FUNC(NAME, A, R, G, B, PREMULT, FILL, DST_STEP) \
  CONVERT_32_FUNC(NAME, 0, 1, 2, 3, /*SRC_STEP=*/4, A, R, G, B, DST_STEP,    \
                  PREMULT, FILL)

CONVERT_32_TO_ARGB32_FUNC(Convert_Argb32_Argb32_C, 0, 1, 2, 3, 0, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_ARGB32_ARGB32_C, 0, 1, 2, 3, 0, 0, 4);

CONVERT_32_TO_ARGB32_FUNC(Convert_ARGB32_Argb32_C, 0, 1, 2, 3, 1, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_XRGB32_Argb32_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_rgbA32_Argb32_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGBA32_Argb32_C, 3, 0, 1, 2, 1, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGBX32_Argb32_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_bgrA32_Argb32_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGRA32_Argb32_C, 3, 2, 1, 0, 1, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGRX32_Argb32_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGB32_Argb32_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGR32_Argb32_C, 0, 2, 1, 0, 0, 1, 3);

CONVERT_32_TO_ARGB32_FUNC(Convert_XRGB32_ARGB32_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_rgbA32_ARGB32_C, 3, 0, 1, 2, -1, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGBA32_ARGB32_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGBX32_ARGB32_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_bgrA32_ARGB32_C, 3, 2, 1, 0, -1, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGRA32_ARGB32_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGRX32_ARGB32_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_32_TO_ARGB32_FUNC(Convert_RGB32_ARGB32_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_32_TO_ARGB32_FUNC(Convert_BGR32_ARGB32_C, 0, 2, 1, 0, 0, 1, 3);

CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_ARGB32_C, 0, 1, 2, 3, -1, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_XRGB32_C, 0, 1, 2, 3, -1, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_rgbA32_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_RGBA32_C, 3, 0, 1, 2, -1, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_RGBX32_C, 3, 0, 1, 2, -1, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_bgrA32_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_BGRA32_C, 3, 2, 1, 0, -1, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_BGRX32_C, 3, 2, 1, 0, -1, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_RGB32_C, 0, 0, 1, 2, -1, 1, 3);
CONVERT_ARGB32_TO_32_FUNC(Convert_Argb32_BGR32_C, 0, 2, 1, 0, -1, 1, 3);

CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_XRGB32_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_rgbA32_C, 3, 0, 1, 2, 1, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_RGBA32_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_RGBX32_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_bgrA32_C, 3, 2, 1, 0, 1, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_BGRA32_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_BGRX32_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_RGB32_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_ARGB32_TO_32_FUNC(Convert_ARGB32_BGR32_C, 0, 2, 1, 0, 0, 1, 3);

#undef CONVERT_FUNC
#undef CONVERT_ARGB32_TO_32_FUNC
#undef CONVERT_32_TO_ARGB32_FUNC

//------------------------------------------------------------------------------
// SSE4.1 implementation

#if defined(WP2_USE_SSE)

void Premultiply(uint8_t* dst, uint32_t width) {
  for (uint32_t i = 0; i < width; ++i, dst += 4) Premult(dst[0], dst + 1);
}

#define PSHUFB_CST32(CST) \
  _mm_set_epi32(CST | 0x0c0c0c0c, CST | 0x08080808, CST | 0x04040404, CST)

#define CONVERT_32_TO_32_FUNC(NAME, CST)                                      \
  void NAME##_SSE(const void* src_data, uint32_t width, void* dst_data) {     \
    const uint8_t* src = (const uint8_t*)src_data;                            \
    uint8_t* dst = (uint8_t*)dst_data;                                        \
    const __m128i kShuffle = PSHUFB_CST32(CST);                               \
    uint32_t x;                                                               \
    for (x = 0; x + 8 <= width; x += 8) {                                     \
      const __m128i A0 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 0));  \
      const __m128i A1 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 16)); \
      const __m128i B0 = _mm_shuffle_epi8(A0, kShuffle);                      \
      const __m128i B1 = _mm_shuffle_epi8(A1, kShuffle);                      \
      _mm_storeu_si128((__m128i*)(dst + 4 * x + 0), B0);                      \
      _mm_storeu_si128((__m128i*)(dst + 4 * x + 16), B1);                     \
    }                                                                         \
    if (x < width) NAME##_C(src + 4 * x, width - x, dst + 4 * x);             \
  }

#define CONVERT_32_TO_32_FILL_FUNC(NAME, CST, CST_A)                          \
  void NAME##_SSE(const void* src_data, uint32_t width, void* dst_data) {     \
    const uint8_t* src = (const uint8_t*)src_data;                            \
    uint8_t* dst = (uint8_t*)dst_data;                                        \
    const __m128i kShuffle = PSHUFB_CST32(CST);                               \
    const __m128i kFill = _mm_set1_epi32(CST_A);                              \
    uint32_t x;                                                               \
    for (x = 0; x + 8 <= width; x += 8) {                                     \
      const __m128i A0 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 0));  \
      const __m128i A1 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 16)); \
      const __m128i B0 = _mm_or_si128(_mm_shuffle_epi8(A0, kShuffle), kFill); \
      const __m128i B1 = _mm_or_si128(_mm_shuffle_epi8(A1, kShuffle), kFill); \
      _mm_storeu_si128((__m128i*)(dst + 4 * x + 0), B0);                      \
      _mm_storeu_si128((__m128i*)(dst + 4 * x + 16), B1);                     \
    }                                                                         \
    if (x < width) NAME##_C(src + 4 * x, width - x, dst + 4 * x);             \
  }

void Convert_Argb32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  memcpy(dst, src, 4 * width);
}

CONVERT_32_TO_32_FILL_FUNC(Convert_XRGB32_Argb32, 0x030201ff, 0x000000ffu)
void Convert_ARGB32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_Argb32_SSE(src, width, dst);
  Premultiply((uint8_t*)dst, width);
}

CONVERT_32_TO_32_FUNC(Convert_rgbA32_Argb32, 0x02010003)
CONVERT_32_TO_32_FILL_FUNC(Convert_RGBX32_Argb32, 0x020100ff, 0x000000ffu)
void Convert_RGBA32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_rgbA32_Argb32_SSE(src, width, dst);
  Premultiply((uint8_t*)dst, width);
}

CONVERT_32_TO_32_FUNC(Convert_bgrA32_Argb32, 0x00010203)
CONVERT_32_TO_32_FILL_FUNC(Convert_BGRX32_Argb32, 0x000102ff, 0x000000ffu)
void Convert_BGRA32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_bgrA32_Argb32_SSE(src, width, dst);
  Premultiply((uint8_t*)dst, width);
}

// ConvertTo

void Unmultiply(uint8_t* const dst, const uint8_t* dst_a, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x) Unmult(dst_a[4 * x], dst + 4 * x);
}
void FillAlpha(uint8_t* dst_a, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x) {
    dst_a[4 * x] = 255;
  }
}

void Convert_Argb32_ARGB32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_Argb32_SSE(src, width, dst);
  Unmultiply((uint8_t*)dst + 1, (uint8_t*)dst + 0, width);
}
void Convert_Argb32_XRGB32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_ARGB32_SSE(src, width, dst);
  FillAlpha((uint8_t*)dst + 0, width);
}

CONVERT_32_TO_32_FUNC(Convert_Argb32_rgbA32, 0x00030201)
void Convert_Argb32_RGBA32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_rgbA32_SSE(src, width, dst);
  Unmultiply((uint8_t*)dst + 0, (uint8_t*)dst + 3, width);
}
void Convert_Argb32_RGBX32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_RGBA32_SSE(src, width, dst);
  FillAlpha((uint8_t*)dst + 3, width);
}

CONVERT_32_TO_32_FUNC(Convert_Argb32_bgrA32, 0x00010203)
void Convert_Argb32_BGRA32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_bgrA32_SSE(src, width, dst);
  Unmultiply((uint8_t*)dst + 0, (uint8_t*)dst + 3, width);
}
void Convert_Argb32_BGRX32_SSE(const void* src, uint32_t width, void* dst) {
  Convert_Argb32_BGRA32_SSE(src, width, dst);
  FillAlpha((uint8_t*)dst + 3, width);
}

#undef CONVERT_32_TO_32_FUNC
#undef PSHUFB_CST32

WP2_TSAN_IGNORE_FUNCTION void ArgbConverterDspInitSSE() {
  WP2ArgbConvertFrom[WP2_Argb_32] = Convert_Argb32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Convert_ARGB32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Convert_XRGB32_Argb32_SSE;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Convert_rgbA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Convert_RGBA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Convert_RGBX32_Argb32_SSE;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Convert_bgrA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Convert_BGRA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Convert_BGRX32_Argb32_SSE;

  WP2ArgbConvertTo[WP2_Argb_32] = Convert_Argb32_Argb32_SSE;
  WP2ArgbConvertTo[WP2_ARGB_32] = Convert_Argb32_ARGB32_SSE;
  WP2ArgbConvertTo[WP2_XRGB_32] = Convert_Argb32_XRGB32_SSE;

  WP2ArgbConvertTo[WP2_rgbA_32] = Convert_Argb32_rgbA32_SSE;
  WP2ArgbConvertTo[WP2_RGBA_32] = Convert_Argb32_RGBA32_SSE;
  WP2ArgbConvertTo[WP2_RGBX_32] = Convert_Argb32_RGBX32_SSE;

  WP2ArgbConvertTo[WP2_bgrA_32] = Convert_Argb32_bgrA32_SSE;
  WP2ArgbConvertTo[WP2_BGRA_32] = Convert_Argb32_BGRA32_SSE;
  WP2ArgbConvertTo[WP2_BGRX_32] = Convert_Argb32_BGRX32_SSE;
}

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------
// NEON implementation (TODO)

//------------------------------------------------------------------------------

}  // namespace

WP2ArgbConverterF WP2ArgbConvertFrom[WP2_FORMAT_NUM];
WP2ArgbConverterF WP2ArgbConvertTo[WP2_FORMAT_NUM];
WP2ArgbConverterF WP2ARGBConvertFrom[WP2_FORMAT_NUM];
WP2ArgbConverterF WP2ARGBConvertTo[WP2_FORMAT_NUM];

static volatile WP2CPUInfo argb_converter_last_cpuinfo_used =
    (WP2CPUInfo)&argb_converter_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void WP2ArgbConverterInit() {
  if (argb_converter_last_cpuinfo_used == WP2GetCPUInfo) return;

  WP2ArgbConvertFrom[WP2_Argb_32] = Convert_Argb32_Argb32_C;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Convert_ARGB32_Argb32_C;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Convert_XRGB32_Argb32_C;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Convert_rgbA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Convert_RGBA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Convert_RGBX32_Argb32_C;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Convert_bgrA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Convert_BGRA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Convert_BGRX32_Argb32_C;

  WP2ArgbConvertFrom[WP2_RGB_24] = Convert_RGB32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGR_24] = Convert_BGR32_Argb32_C;

  WP2ArgbConvertFrom[WP2_Argb_38] = nullptr;

  WP2ArgbConvertTo[WP2_Argb_32] = Convert_Argb32_Argb32_C;
  WP2ArgbConvertTo[WP2_ARGB_32] = Convert_Argb32_ARGB32_C;
  WP2ArgbConvertTo[WP2_XRGB_32] = Convert_Argb32_XRGB32_C;

  WP2ArgbConvertTo[WP2_rgbA_32] = Convert_Argb32_rgbA32_C;
  WP2ArgbConvertTo[WP2_RGBA_32] = Convert_Argb32_RGBA32_C;
  WP2ArgbConvertTo[WP2_RGBX_32] = Convert_Argb32_RGBX32_C;

  WP2ArgbConvertTo[WP2_bgrA_32] = Convert_Argb32_bgrA32_C;
  WP2ArgbConvertTo[WP2_BGRA_32] = Convert_Argb32_BGRA32_C;
  WP2ArgbConvertTo[WP2_BGRX_32] = Convert_Argb32_BGRX32_C;

  WP2ArgbConvertTo[WP2_RGB_24] = Convert_Argb32_RGB32_C;
  WP2ArgbConvertTo[WP2_BGR_24] = Convert_Argb32_BGR32_C;

  WP2ArgbConvertTo[WP2_Argb_38] = nullptr;

  WP2ARGBConvertFrom[WP2_Argb_32] = Convert_Argb32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_ARGB_32] = Convert_ARGB32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_XRGB_32] = Convert_XRGB32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_rgbA_32] = Convert_rgbA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_RGBA_32] = Convert_RGBA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_RGBX_32] = Convert_RGBX32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_bgrA_32] = Convert_bgrA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGRA_32] = Convert_BGRA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGRX_32] = Convert_BGRX32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_RGB_24] = Convert_RGB32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGR_24] = Convert_BGR32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_Argb_38] = nullptr;

  WP2ARGBConvertTo[WP2_Argb_32] = Convert_ARGB32_Argb32_C;
  WP2ARGBConvertTo[WP2_ARGB_32] = Convert_ARGB32_ARGB32_C;
  WP2ARGBConvertTo[WP2_XRGB_32] = Convert_ARGB32_XRGB32_C;

  WP2ARGBConvertTo[WP2_rgbA_32] = Convert_ARGB32_rgbA32_C;
  WP2ARGBConvertTo[WP2_RGBA_32] = Convert_ARGB32_RGBA32_C;
  WP2ARGBConvertTo[WP2_RGBX_32] = Convert_ARGB32_RGBX32_C;

  WP2ARGBConvertTo[WP2_bgrA_32] = Convert_ARGB32_bgrA32_C;
  WP2ARGBConvertTo[WP2_BGRA_32] = Convert_ARGB32_BGRA32_C;
  WP2ARGBConvertTo[WP2_BGRX_32] = Convert_ARGB32_BGRX32_C;

  WP2ARGBConvertTo[WP2_RGB_24] = Convert_ARGB32_RGB32_C;
  WP2ARGBConvertTo[WP2_BGR_24] = Convert_ARGB32_BGR32_C;

  WP2ARGBConvertTo[WP2_Argb_38] = nullptr;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) ArgbConverterDspInitSSE();
#endif
  }
  argb_converter_last_cpuinfo_used = WP2GetCPUInfo;
}

WP2ArgbConverterF WP2ConversionFunction(WP2SampleFormat from,
                                        WP2SampleFormat to) {
  return (to == WP2_Argb_32)     ? WP2ArgbConvertFrom[from]
         : (to == WP2_ARGB_32)   ? WP2ARGBConvertFrom[from]
         : (from == WP2_Argb_32) ? WP2ArgbConvertTo[to]
         : (from == WP2_ARGB_32) ? WP2ARGBConvertTo[to]
                                 : nullptr;
}

//------------------------------------------------------------------------------
