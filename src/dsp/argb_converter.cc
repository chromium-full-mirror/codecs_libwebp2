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
// ConvertFrom...

void Premultiply(uint8_t* ARGB, uint32_t width) {
  uint32_t i;
  for (i = 0; i < width; ++i, ARGB += 4) {
    const uint32_t A = ARGB[0];
    if (A < 255) {
      ARGB[1] = WP2::DivBy255(ARGB[1] * A);
      ARGB[2] = WP2::DivBy255(ARGB[2] * A);
      ARGB[3] = WP2::DivBy255(ARGB[3] * A);
    }
  }
}


//------------------------------------------------------------------------------
// ConvertTo...

void Unmult(uint32_t a, uint8_t* const v) {
  if (a > 0) {
    const uint32_t M = WP2::kAlphaDiv[a];
    v[0] = std::min(WP2::DivByAlphaDiv(v[0], M), 255u);
    v[1] = std::min(WP2::DivByAlphaDiv(v[1], M), 255u);
    v[2] = std::min(WP2::DivByAlphaDiv(v[2], M), 255u);
  } else {
    v[0] = v[1] = v[2] = 0;
  }
}

void Premult(uint32_t a, uint8_t* const v) {
  if (a > 0) {
    v[0] = WP2::DivBy255(v[0] * a);
    v[1] = WP2::DivBy255(v[1] * a);
    v[2] = WP2::DivBy255(v[2] * a);
  } else {
    v[0] = v[1] = v[2] = 0;
  }
}

#define CONVERT_FUNC(NAME, SRC_A, SRC_R, SRC_G, SRC_B, DST_A, DST_R, DST_G,  \
                     DST_B, PREMULT, FILL, SRC_STEP, DST_STEP)               \
  void NAME(const uint8_t* src, uint32_t width, uint8_t* dst) {              \
    const int min_pos = std::min({(DST_R), (DST_G), (DST_B)});               \
    for (uint32_t i = 0; i < width; ++i, src += SRC_STEP, dst += DST_STEP) { \
      if ((DST_STEP) == 4) dst[(DST_A)] = (FILL) ? 0xff : src[(SRC_A)];      \
      dst[(DST_R)] = src[(SRC_R)];                                           \
      dst[(DST_G)] = src[(SRC_G)];                                           \
      dst[(DST_B)] = src[(SRC_B)];                                           \
      if ((PREMULT) != 0) {                                                  \
        const uint8_t a = (((DST_STEP) == 4) ? dst[(DST_A)] : src[(SRC_A)]); \
        if ((PREMULT) > 0) Premult(a, dst + min_pos);                        \
        if ((PREMULT) < 0) Unmult(a, dst + min_pos);                         \
      }                                                                      \
    }                                                                        \
  }

#define CONVERT_FROM_FUNC(NAME, A, R, G, B, PREMULT, FILL, STEP) \
  CONVERT_FUNC(NAME, A, R, G, B, 0, 1, 2, 3, PREMULT, FILL, STEP, 4)

#define CONVERT_TO_FUNC(NAME, A, R, G, B, PREMULT, FILL, STEP) \
  CONVERT_FUNC(NAME, 0, 1, 2, 3, A, R, G, B, PREMULT, FILL, 4, STEP)

CONVERT_FROM_FUNC(Convert_Argb_Argb_C, 0, 1, 2, 3, 0, 0, 4);
CONVERT_FROM_FUNC(Convert_ARGB_ARGB_C, 0, 1, 2, 3, 0, 0, 4);

CONVERT_FROM_FUNC(Convert_ARGB_Argb_C, 0, 1, 2, 3, 1, 0, 4);
CONVERT_FROM_FUNC(Convert_XRGB_Argb_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_rgbA_Argb_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_FROM_FUNC(Convert_RGBA_Argb_C, 3, 0, 1, 2, 1, 0, 4);
CONVERT_FROM_FUNC(Convert_RGBX_Argb_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_bgrA_Argb_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_FROM_FUNC(Convert_BGRA_Argb_C, 3, 2, 1, 0, 1, 0, 4);
CONVERT_FROM_FUNC(Convert_BGRX_Argb_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_RGB_Argb_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_FROM_FUNC(Convert_BGR_Argb_C, 0, 2, 1, 0, 0, 1, 3);

CONVERT_FROM_FUNC(Convert_XRGB_ARGB_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_rgbA_ARGB_C, 3, 0, 1, 2, -1, 0, 4);
CONVERT_FROM_FUNC(Convert_RGBA_ARGB_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_FROM_FUNC(Convert_RGBX_ARGB_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_bgrA_ARGB_C, 3, 2, 1, 0, -1, 0, 4);
CONVERT_FROM_FUNC(Convert_BGRA_ARGB_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_FROM_FUNC(Convert_BGRX_ARGB_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_FROM_FUNC(Convert_RGB_ARGB_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_FROM_FUNC(Convert_BGR_ARGB_C, 0, 2, 1, 0, 0, 1, 3);

CONVERT_TO_FUNC(Convert_Argb_ARGB_C, 0, 1, 2, 3, -1, 0, 4);
CONVERT_TO_FUNC(Convert_Argb_XRGB_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_TO_FUNC(Convert_Argb_rgbA_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_TO_FUNC(Convert_Argb_RGBA_C, 3, 0, 1, 2, -1, 0, 4);
CONVERT_TO_FUNC(Convert_Argb_RGBX_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_TO_FUNC(Convert_Argb_bgrA_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_TO_FUNC(Convert_Argb_BGRA_C, 3, 2, 1, 0, -1, 0, 4);
CONVERT_TO_FUNC(Convert_Argb_BGRX_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_TO_FUNC(Convert_Argb_RGB_C, 0, 0, 1, 2, -1, 1, 3);
CONVERT_TO_FUNC(Convert_Argb_BGR_C, 0, 2, 1, 0, -1, 1, 3);

CONVERT_TO_FUNC(Convert_ARGB_XRGB_C, 0, 1, 2, 3, 0, 1, 4);
CONVERT_TO_FUNC(Convert_ARGB_rgbA_C, 3, 0, 1, 2, 1, 0, 4);
CONVERT_TO_FUNC(Convert_ARGB_RGBA_C, 3, 0, 1, 2, 0, 0, 4);
CONVERT_TO_FUNC(Convert_ARGB_RGBX_C, 3, 0, 1, 2, 0, 1, 4);
CONVERT_TO_FUNC(Convert_ARGB_bgrA_C, 3, 2, 1, 0, 1, 0, 4);
CONVERT_TO_FUNC(Convert_ARGB_BGRA_C, 3, 2, 1, 0, 0, 0, 4);
CONVERT_TO_FUNC(Convert_ARGB_BGRX_C, 3, 2, 1, 0, 0, 1, 4);
CONVERT_TO_FUNC(Convert_ARGB_RGB_C, 0, 0, 1, 2, 0, 1, 3);
CONVERT_TO_FUNC(Convert_ARGB_BGR_C, 0, 2, 1, 0, 0, 1, 3);

#undef CONVERT_FUNC
#undef CONVERT_TO_FUNC
#undef CONVERT_FROM_FUNC

//------------------------------------------------------------------------------
// SSE4.1 implementation

#if defined(WP2_USE_SSE)

#define PSHUFB_CST32(CST)  \
  _mm_set_epi32(CST | 0x0c0c0c0c, CST | 0x08080808, CST | 0x04040404, CST)

#define CONVERT_FUNC(NAME, CST)                                              \
void NAME##_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {          \
  const __m128i kShuffle = PSHUFB_CST32(CST);                                \
  uint32_t x;                                                                \
  for (x = 0; x + 8 <= width; x += 8) {                                      \
    const __m128i A0 = _mm_loadu_si128((const __m128i*)(src + 4 * x +  0));  \
    const __m128i A1 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 16));  \
    const __m128i B0 = _mm_shuffle_epi8(A0, kShuffle);                       \
    const __m128i B1 = _mm_shuffle_epi8(A1, kShuffle);                       \
    _mm_storeu_si128((__m128i*)(dst + 4 * x +  0), B0);                      \
    _mm_storeu_si128((__m128i*)(dst + 4 * x + 16), B1);                      \
  }                                                                          \
  if (x < width) NAME##_C(src + 4 * x, width - x, dst + 4 * x);              \
}

#define CONVERT_FUNC_FILL(NAME, CST, CST_A)                                  \
void NAME##_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {          \
  const __m128i kShuffle = PSHUFB_CST32(CST);                                \
  const __m128i kFill = _mm_set1_epi32(CST_A);                               \
  uint32_t x;                                                                \
  for (x = 0; x + 8 <= width; x += 8) {                                      \
    const __m128i A0 = _mm_loadu_si128((const __m128i*)(src + 4 * x +  0));  \
    const __m128i A1 = _mm_loadu_si128((const __m128i*)(src + 4 * x + 16));  \
    const __m128i B0 = _mm_or_si128(_mm_shuffle_epi8(A0, kShuffle), kFill);  \
    const __m128i B1 = _mm_or_si128(_mm_shuffle_epi8(A1, kShuffle), kFill);  \
    _mm_storeu_si128((__m128i*)(dst + 4 * x +  0), B0);                      \
    _mm_storeu_si128((__m128i*)(dst + 4 * x + 16), B1);                      \
  }                                                                          \
  if (x < width) NAME##_C(src + 4 * x, width - x, dst + 4 * x);              \
}

void Convert_Argb_Argb_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  memcpy(dst, src, 4 * width);
}

CONVERT_FUNC_FILL(Convert_XRGB_Argb, 0x030201ff, 0x000000ffu)
void Convert_ARGB_Argb_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_Argb_Argb_SSE(src, width, dst);
  Premultiply(dst, width);
}

CONVERT_FUNC(Convert_rgbA_Argb, 0x02010003)
CONVERT_FUNC_FILL(Convert_RGBX_Argb, 0x020100ff, 0x000000ffu)
void Convert_RGBA_Argb_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_rgbA_Argb_SSE(src, width, dst);
  Premultiply(dst, width);
}

CONVERT_FUNC(Convert_bgrA_Argb, 0x00010203)
CONVERT_FUNC_FILL(Convert_BGRX_Argb, 0x000102ff, 0x000000ffu)
void Convert_BGRA_Argb_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_bgrA_Argb_SSE(src, width, dst);
  Premultiply(dst, width);
}

// ConvertTo

void Unmultiply(uint8_t* const dst, const uint8_t* dst_a, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x) Unmult(dst_a[4 * x], dst + 4 * x);
}

CONVERT_FUNC_FILL(Convert_Argb_XRGB, 0x030201ff, 0x000000ffu)
void Convert_Argb_ARGB_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_Argb_Argb_SSE(src, width, dst);
  Unmultiply(dst + 1, dst + 0, width);
}

CONVERT_FUNC(Convert_Argb_rgbA, 0x00030201)
CONVERT_FUNC_FILL(Convert_Argb_RGBX, 0xff030201, 0xff000000u)
void Convert_Argb_RGBA_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_Argb_rgbA_SSE(src, width, dst);
  Unmultiply(dst + 0, dst + 3, width);
}

CONVERT_FUNC(Convert_Argb_bgrA, 0x00010203)
CONVERT_FUNC_FILL(Convert_Argb_BGRX, 0xff010203, 0xff000000u)
void Convert_Argb_BGRA_SSE(const uint8_t* src, uint32_t width, uint8_t* dst) {
  Convert_Argb_bgrA_SSE(src, width, dst);
  Unmultiply(dst + 0, dst + 3, width);
}

#undef CONVERT_FUNC
#undef PSHUFB_CST32

WP2_TSAN_IGNORE_FUNCTION void ArgbConverterDspInitSSE() {
  WP2ArgbConvertFrom[WP2_Argb_32] = Convert_Argb_Argb_SSE;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Convert_ARGB_Argb_SSE;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Convert_XRGB_Argb_SSE;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Convert_rgbA_Argb_SSE;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Convert_RGBA_Argb_SSE;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Convert_RGBX_Argb_SSE;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Convert_bgrA_Argb_SSE;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Convert_BGRA_Argb_SSE;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Convert_BGRX_Argb_SSE;

  WP2ArgbConvertTo[WP2_Argb_32] = Convert_Argb_Argb_SSE;
  WP2ArgbConvertTo[WP2_ARGB_32] = Convert_Argb_ARGB_SSE;
  WP2ArgbConvertTo[WP2_XRGB_32] = Convert_Argb_XRGB_SSE;

  WP2ArgbConvertTo[WP2_rgbA_32] = Convert_Argb_rgbA_SSE;
  WP2ArgbConvertTo[WP2_RGBA_32] = Convert_Argb_RGBA_SSE;
  WP2ArgbConvertTo[WP2_RGBX_32] = Convert_Argb_RGBX_SSE;

  WP2ArgbConvertTo[WP2_bgrA_32] = Convert_Argb_bgrA_SSE;
  WP2ArgbConvertTo[WP2_BGRA_32] = Convert_Argb_BGRA_SSE;
  WP2ArgbConvertTo[WP2_BGRX_32] = Convert_Argb_BGRX_SSE;
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

  WP2ArgbConvertFrom[WP2_Argb_32] = Convert_Argb_Argb_C;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Convert_ARGB_Argb_C;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Convert_XRGB_Argb_C;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Convert_rgbA_Argb_C;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Convert_RGBA_Argb_C;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Convert_RGBX_Argb_C;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Convert_bgrA_Argb_C;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Convert_BGRA_Argb_C;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Convert_BGRX_Argb_C;

  WP2ArgbConvertFrom[WP2_RGB_24] = Convert_RGB_Argb_C;
  WP2ArgbConvertFrom[WP2_BGR_24] = Convert_BGR_Argb_C;

  WP2ArgbConvertTo[WP2_Argb_32] = Convert_Argb_Argb_C;
  WP2ArgbConvertTo[WP2_ARGB_32] = Convert_Argb_ARGB_C;
  WP2ArgbConvertTo[WP2_XRGB_32] = Convert_Argb_XRGB_C;

  WP2ArgbConvertTo[WP2_rgbA_32] = Convert_Argb_rgbA_C;
  WP2ArgbConvertTo[WP2_RGBA_32] = Convert_Argb_RGBA_C;
  WP2ArgbConvertTo[WP2_RGBX_32] = Convert_Argb_RGBX_C;

  WP2ArgbConvertTo[WP2_bgrA_32] = Convert_Argb_bgrA_C;
  WP2ArgbConvertTo[WP2_BGRA_32] = Convert_Argb_BGRA_C;
  WP2ArgbConvertTo[WP2_BGRX_32] = Convert_Argb_BGRX_C;

  WP2ArgbConvertTo[WP2_RGB_24] = Convert_Argb_RGB_C;
  WP2ArgbConvertTo[WP2_BGR_24] = Convert_Argb_BGR_C;

  WP2ARGBConvertFrom[WP2_Argb_32] = Convert_Argb_ARGB_C;
  WP2ARGBConvertFrom[WP2_ARGB_32] = Convert_ARGB_ARGB_C;
  WP2ARGBConvertFrom[WP2_XRGB_32] = Convert_XRGB_ARGB_C;

  WP2ARGBConvertFrom[WP2_rgbA_32] = Convert_rgbA_ARGB_C;
  WP2ARGBConvertFrom[WP2_RGBA_32] = Convert_RGBA_ARGB_C;
  WP2ARGBConvertFrom[WP2_RGBX_32] = Convert_RGBX_ARGB_C;

  WP2ARGBConvertFrom[WP2_bgrA_32] = Convert_bgrA_ARGB_C;
  WP2ARGBConvertFrom[WP2_BGRA_32] = Convert_BGRA_ARGB_C;
  WP2ARGBConvertFrom[WP2_BGRX_32] = Convert_BGRX_ARGB_C;

  WP2ARGBConvertFrom[WP2_RGB_24] = Convert_RGB_ARGB_C;
  WP2ARGBConvertFrom[WP2_BGR_24] = Convert_BGR_ARGB_C;

  WP2ARGBConvertTo[WP2_Argb_32] = Convert_ARGB_Argb_C;
  WP2ARGBConvertTo[WP2_ARGB_32] = Convert_ARGB_ARGB_C;
  WP2ARGBConvertTo[WP2_XRGB_32] = Convert_ARGB_XRGB_C;

  WP2ARGBConvertTo[WP2_rgbA_32] = Convert_ARGB_rgbA_C;
  WP2ARGBConvertTo[WP2_RGBA_32] = Convert_ARGB_RGBA_C;
  WP2ARGBConvertTo[WP2_RGBX_32] = Convert_ARGB_RGBX_C;

  WP2ARGBConvertTo[WP2_bgrA_32] = Convert_ARGB_bgrA_C;
  WP2ARGBConvertTo[WP2_BGRA_32] = Convert_ARGB_BGRA_C;
  WP2ARGBConvertTo[WP2_BGRX_32] = Convert_ARGB_BGRX_C;

  WP2ARGBConvertTo[WP2_RGB_24] = Convert_ARGB_RGB_C;
  WP2ARGBConvertTo[WP2_BGR_24] = Convert_ARGB_BGR_C;

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
