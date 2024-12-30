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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "src/dsp/dsp.h"
#include "src/wp2/base.h"

namespace {

//------------------------------------------------------------------------------
// Plain copy (same source and destination formats)

template <int bytes_per_pixel>
void WP2Copy_C(const void* src, uint32_t width, void* dst) {
  std::memcpy(dst, src, static_cast<size_t>(width) * bytes_per_pixel);
}

//------------------------------------------------------------------------------
// C conversion from a format to another

// Demultiplies color 'v' by alpha 'a'.
template <typename Type, int kAlphaMax, int kColorMax>
void Unmult(uint32_t a, Type& r, Type& g, Type& b) {
  if (a == 0) {
    r = g = b = 0;
    return;
  }
  if (a == kAlphaMax) return;

  const uint32_t round = a >> 1;
  r = static_cast<Type>(
      std::min<uint32_t>((r * kAlphaMax + round) / a, kColorMax));
  g = static_cast<Type>(
      std::min<uint32_t>((g * kAlphaMax + round) / a, kColorMax));
  b = static_cast<Type>(
      std::min<uint32_t>((b * kAlphaMax + round) / a, kColorMax));
}

// Premultiplies color 'v' by alpha 'a'.
template <typename Type, int kAlphaMax>
void Premult(uint32_t a, Type& r, Type& g, Type& b) {
  if (a == 0) {
    r = g = b = 0;
    return;
  }
  if (a == kAlphaMax) return;

  constexpr uint32_t kAlphaRound = kAlphaMax >> 1;
  r = static_cast<Type>((r * a + kAlphaRound) / kAlphaMax);
  g = static_cast<Type>((g * a + kAlphaRound) / kAlphaMax);
  b = static_cast<Type>((b * a + kAlphaRound) / kAlphaMax);
}

// Converts 'width' px from 'src_data' of 'SrcType' to 'dst_data' of 'DstType'.
// Channel ordering is specified by the indices in 'k[Src/Dst][A/R/G/B]'.
// Source samples are considered premultiplied by input alpha if 'kSrcMult'.
// 'kFillAlpha' will replace the output alpha by 255.
// Destination samples will be premultiplied by output alpha if 'kDstMult'.
template <typename SrcType, int kSrcDepth, int kSrcMult, int kSrcA, int kSrcR,
          int kSrcG, int kSrcB, int kSrcStep, typename DstType, int kDstDepth,
          int kDstMult, int kDstA, int kDstR, int kDstG, int kDstB,
          int kDstStep, bool kFillAlpha>
void Cvrt_C(const void* src_data, uint32_t width, void* dst_data) {
  static_assert((kSrcDepth == 8 || kSrcDepth == 10 || kSrcDepth == 16) &&
                    (kDstDepth == 8 || kDstDepth == 10 || kDstDepth == 16),
                "Unimplemented");

  constexpr uint32_t kSrcMax = (1u << kSrcDepth) - 1u;
  constexpr uint32_t kDstMax = (1u << kDstDepth) - 1u;

  // The alpha channel of WP2_Argb_38 (depth = 10) uses only 8 bits.
  constexpr uint32_t kSrcAlphaMax = kSrcDepth <= 10 ? 255u : kSrcMax;
  constexpr uint32_t kSrcAlphaRound = kSrcAlphaMax >> 1;
  constexpr uint32_t kDstAlphaMax = kDstDepth <= 10 ? 255u : kDstMax;

  assert(src_data != dst_data);
  const SrcType* src = reinterpret_cast<const SrcType*>(src_data);
  DstType* dst = reinterpret_cast<DstType*>(dst_data);
  for (uint32_t i = 0; i < width; ++i, src += kSrcStep, dst += kDstStep) {
    if (kDstStep == 4) {
      if (kFillAlpha) {
        dst[kDstA] = kDstAlphaMax;
      } else if (kSrcAlphaMax == kDstAlphaMax) {
        dst[kDstA] = src[kSrcA];
      } else {
        dst[kDstA] =
            (src[kSrcA] * kDstAlphaMax + kSrcAlphaRound) / kSrcAlphaMax;
      }
    }
    if (kSrcDepth == kDstDepth && kSrcMult == kDstMult) {
      // Just shuffling channels.
      dst[kDstR] = src[kSrcR];
      dst[kDstG] = src[kSrcG];
      dst[kDstB] = src[kSrcB];
    } else {
      const uint32_t src_max =
          kSrcMult
              ? (kSrcDepth == 10) ? (src[kSrcA] * 1023 + 127) / 255 : src[kSrcA]
              : kSrcMax;
      const uint32_t dst_max =
          kDstMult
              ? (kDstDepth == 10) ? (dst[kDstA] * 1023 + 127) / 255 : dst[kDstA]
              : kDstMax;
      if (src_max == 0 || dst_max == 0) {
        dst[kDstR] = 0;
        dst[kDstG] = 0;
        dst[kDstB] = 0;
      } else if (src_max == dst_max) {
        dst[kDstR] = src[kSrcR];
        dst[kDstG] = src[kSrcG];
        dst[kDstB] = src[kSrcB];
      } else {
        const uint32_t src_round = src_max >> 1;
        dst[kDstR] = static_cast<DstType>(std::min<uint32_t>(
            (src[kSrcR] * dst_max + src_round) / src_max, dst_max));
        dst[kDstG] = static_cast<DstType>(std::min<uint32_t>(
            (src[kSrcG] * dst_max + src_round) / src_max, dst_max));
        dst[kDstB] = static_cast<DstType>(std::min<uint32_t>(
            (src[kSrcB] * dst_max + src_round) / src_max, dst_max));
      }
    }
  }
}

// From the specified format to Argb32 or ARGB32.
#define CVRT_TO_Argb32(SRC_TYPE, SRC_DEPTH, SRC_MULT, A, R, G, B, FILL, \
                       SRC_STEP)                                        \
  Cvrt_C<SRC_TYPE, SRC_DEPTH, SRC_MULT, A, R, G, B, SRC_STEP, uint8_t,  \
         /*kDstDepth=*/8, /*kDstMult=*/1, /*kDstA=*/0, /*kDstR=*/1,     \
         /*kDstG=*/2, /*kDstB=*/3, /*kDstStep=*/4, FILL>
#define CVRT_TO_ARGB32(SRC_TYPE, SRC_DEPTH, SRC_MULT, A, R, G, B, FILL, \
                       SRC_STEP)                                        \
  Cvrt_C<SRC_TYPE, SRC_DEPTH, SRC_MULT, A, R, G, B, SRC_STEP, uint8_t,  \
         /*kDstDepth=*/8, /*kDstMult=*/0, /*kDstA=*/0, /*kDstR=*/1,     \
         /*kDstG=*/2, /*kDstB=*/3, /*kDstStep=*/4, FILL>

// From the Argb32 or ARGB32 to specified format.
#define CVRT_Argb32_TO(DST_TYPE, DST_DEPTH, DST_MULT, A, R, G, B, FILL,      \
                       DST_STEP)                                             \
  Cvrt_C<uint8_t, /*kSrcDepth=*/8, /*kSrcMult=*/1, /*kSrcA=*/0, /*kSrcR=*/1, \
         /*kSrcG=*/2, /*kSrcB=*/3, /*kSrcStep=*/4, DST_TYPE, DST_DEPTH,      \
         DST_MULT, A, R, G, B, DST_STEP, FILL>
#define CVRT_ARGB32_TO(DST_TYPE, DST_DEPTH, DST_MULT, A, R, G, B, FILL,      \
                       DST_STEP)                                             \
  Cvrt_C<uint8_t, /*kSrcDepth=*/8, /*kSrcMult=*/0, /*kSrcA=*/0, /*kSrcR=*/1, \
         /*kSrcG=*/2, /*kSrcB=*/3, /*kSrcStep=*/4, DST_TYPE, DST_DEPTH,      \
         DST_MULT, A, R, G, B, DST_STEP, FILL>

//                                                   d
//                                             t     e  m              f  s
//                                     src's:  y     p  u              i  t
//                                             p     t  l   indices    l  e
//          | src  | dst  |                    e     h  t  A  R  G  B  l  p

#define Cvrt_Argb32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 1, 0, 1, 2, 3, 0, 4)
#define Cvrt_ARGB32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 0, 1, 2, 3, 0, 4)

#define Cvrt_ARGB32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 0, 1, 2, 3, 0, 4)
#define Cvrt_XRGB32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 0, 1, 2, 3, 1, 4)
#define Cvrt_rgbA32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 1, 3, 0, 1, 2, 0, 4)
#define Cvrt_RGBA32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 3, 0, 1, 2, 0, 4)
#define Cvrt_RGBX32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 3, 0, 1, 2, 1, 4)
#define Cvrt_bgrA32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 1, 3, 2, 1, 0, 0, 4)
#define Cvrt_BGRA32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 3, 2, 1, 0, 0, 4)
#define Cvrt_BGRX32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 3, 2, 1, 0, 1, 4)
#define Cvrt_RGB32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 0, 0, 1, 2, 1, 3)
#define Cvrt_BGR32_Argb32_C CVRT_TO_Argb32(uint8_t, 8, 0, 0, 2, 1, 0, 1, 3)

#define Cvrt_XRGB32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 0, 1, 2, 3, 1, 4)
#define Cvrt_rgbA32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 1, 3, 0, 1, 2, 0, 4)
#define Cvrt_RGBA32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 3, 0, 1, 2, 0, 4)
#define Cvrt_RGBX32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 3, 0, 1, 2, 1, 4)
#define Cvrt_bgrA32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 1, 3, 2, 1, 0, 0, 4)
#define Cvrt_BGRA32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 3, 2, 1, 0, 0, 4)
#define Cvrt_BGRX32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 3, 2, 1, 0, 1, 4)
#define Cvrt_RGB32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 0, 0, 1, 2, 1, 3)
#define Cvrt_BGR32_ARGB32_C CVRT_TO_ARGB32(uint8_t, 8, 0, 0, 2, 1, 0, 1, 3)

#define Cvrt_Argb38_Argb32_C CVRT_TO_Argb32(uint16_t, 10, 1, 0, 1, 2, 3, 0, 4)
#define Cvrt_Argb38_ARGB32_C CVRT_TO_ARGB32(uint16_t, 10, 1, 0, 1, 2, 3, 0, 4)

#define Cvrt_ARGB64_Argb32_C CVRT_TO_Argb32(uint16_t, 16, 0, 0, 1, 2, 3, 0, 4)
#define Cvrt_ARGB64_ARGB32_C CVRT_TO_ARGB32(uint16_t, 16, 0, 0, 1, 2, 3, 0, 4)

//                                                   d
//                                             t     e  m              f  s
//                                     dst's:  y     p  u              i  t
//                                             p     t  l   indices    l  e
//          | src  | dst  |                    e     h  t  A  R  G  B  l  p

#define Cvrt_Argb32_ARGB32_C CVRT_Argb32_TO(uint8_t, 8, 0, 0, 1, 2, 3, 0, 4)
#define Cvrt_Argb32_XRGB32_C CVRT_Argb32_TO(uint8_t, 8, 0, 0, 1, 2, 3, 1, 4)
#define Cvrt_Argb32_rgbA32_C CVRT_Argb32_TO(uint8_t, 8, 1, 3, 0, 1, 2, 0, 4)
#define Cvrt_Argb32_RGBA32_C CVRT_Argb32_TO(uint8_t, 8, 0, 3, 0, 1, 2, 0, 4)
#define Cvrt_Argb32_RGBX32_C CVRT_Argb32_TO(uint8_t, 8, 0, 3, 0, 1, 2, 1, 4)
#define Cvrt_Argb32_bgrA32_C CVRT_Argb32_TO(uint8_t, 8, 1, 3, 2, 1, 0, 0, 4)
#define Cvrt_Argb32_BGRA32_C CVRT_Argb32_TO(uint8_t, 8, 0, 3, 2, 1, 0, 0, 4)
#define Cvrt_Argb32_BGRX32_C CVRT_Argb32_TO(uint8_t, 8, 0, 3, 2, 1, 0, 1, 4)
#define Cvrt_Argb32_RGB32_C CVRT_Argb32_TO(uint8_t, 8, 0, 0, 0, 1, 2, 0, 3)
#define Cvrt_Argb32_BGR32_C CVRT_Argb32_TO(uint8_t, 8, 0, 0, 2, 1, 0, 0, 3)

#define Cvrt_ARGB32_XRGB32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 0, 1, 2, 3, 1, 4)
#define Cvrt_ARGB32_rgbA32_C CVRT_ARGB32_TO(uint8_t, 8, 1, 3, 0, 1, 2, 0, 4)
#define Cvrt_ARGB32_RGBA32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 3, 0, 1, 2, 0, 4)
#define Cvrt_ARGB32_RGBX32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 3, 0, 1, 2, 1, 4)
#define Cvrt_ARGB32_bgrA32_C CVRT_ARGB32_TO(uint8_t, 8, 1, 3, 2, 1, 0, 0, 4)
#define Cvrt_ARGB32_BGRA32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 3, 2, 1, 0, 0, 4)
#define Cvrt_ARGB32_BGRX32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 3, 2, 1, 0, 1, 4)
#define Cvrt_ARGB32_RGB32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 0, 0, 1, 2, 0, 3)
#define Cvrt_ARGB32_BGR32_C CVRT_ARGB32_TO(uint8_t, 8, 0, 0, 2, 1, 0, 0, 3)

#define Cvrt_Argb32_Argb38_C CVRT_Argb32_TO(uint16_t, 10, 1, 0, 1, 2, 3, 0, 4)
#define Cvrt_ARGB32_Argb38_C CVRT_ARGB32_TO(uint16_t, 10, 1, 0, 1, 2, 3, 0, 4)

#define Cvrt_Argb32_ARGB64_C CVRT_Argb32_TO(uint16_t, 16, 0, 0, 1, 2, 3, 0, 4)
#define Cvrt_ARGB32_ARGB64_C CVRT_ARGB32_TO(uint16_t, 16, 0, 0, 1, 2, 3, 0, 4)

//------------------------------------------------------------------------------
// SSE4.1 implementation

#if defined(WP2_USE_SSE)

void PremultiplyARGB32(uint8_t* dst, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x, dst += 4) {
    Premult<uint8_t, 255>(*dst, dst[1], dst[2], dst[3]);
  }
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

void Cvrt_Argb32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  memcpy(dst, src, 4 * width);
}

CONVERT_32_TO_32_FILL_FUNC(Cvrt_XRGB32_Argb32, 0x030201ff, 0x000000ffu)
void Cvrt_ARGB32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_Argb32_SSE(src, width, dst);  // ARGB32 -> ARGB32
  PremultiplyARGB32((uint8_t*)dst, width);  // ARGB32 -> Argb32
}

CONVERT_32_TO_32_FUNC(Cvrt_rgbA32_Argb32, 0x02010003)
CONVERT_32_TO_32_FILL_FUNC(Cvrt_RGBX32_Argb32, 0x020100ff, 0x000000ffu)
void Cvrt_RGBA32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_rgbA32_Argb32_SSE(src, width, dst);  // RGBA32 -> ARGB32
  PremultiplyARGB32((uint8_t*)dst, width);  // ARGB32 -> Argb32
}

CONVERT_32_TO_32_FUNC(Cvrt_bgrA32_Argb32, 0x00010203)
CONVERT_32_TO_32_FILL_FUNC(Cvrt_BGRX32_Argb32, 0x000102ff, 0x000000ffu)
void Cvrt_BGRA32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_bgrA32_Argb32_SSE(src, width, dst);  // BGRA32 -> ARGB32
  PremultiplyARGB32((uint8_t*)dst, width);  // ARGB32 -> Argb32
}

// ConvertTo

void Unmultiply8b(const uint8_t* dst_a, uint8_t* dst_rgb, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x, dst_a += 4, dst_rgb += 4) {
    Unmult<uint8_t, 255, 255>(dst_a[0], dst_rgb[0], dst_rgb[1], dst_rgb[2]);
  }
}
void FillAlpha8b(uint8_t* dst_a, uint32_t width) {
  for (uint32_t x = 0; x < width; ++x) {
    dst_a[4 * x] = 255;
  }
}

void Cvrt_Argb32_ARGB32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_Argb32_SSE(src, width, dst);  // Argb32 -> Argb32
  Unmultiply8b(const_cast<const uint8_t*>(reinterpret_cast<uint8_t*>(dst) + 0),
               reinterpret_cast<uint8_t*>(dst) + 1,
               width);  // Argb32 -> ARGB32
}
void Cvrt_Argb32_XRGB32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_ARGB32_SSE(src, width, dst);                  // Argb32 -> ARGB32
  FillAlpha8b(reinterpret_cast<uint8_t*>(dst) + 0, width);  // ARGB32 -> XRGB32
}

CONVERT_32_TO_32_FUNC(Cvrt_Argb32_rgbA32, 0x00030201)
void Cvrt_Argb32_RGBA32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_rgbA32_SSE(src, width, dst);  // Argb32 -> rgbA32
  Unmultiply8b(const_cast<const uint8_t*>(reinterpret_cast<uint8_t*>(dst) + 3),
               reinterpret_cast<uint8_t*>(dst) + 0,
               width);  // rgbA32 -> RGBA32
}
void Cvrt_Argb32_RGBX32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_RGBA32_SSE(src, width, dst);                  // Argb32 -> RGBA32
  FillAlpha8b(reinterpret_cast<uint8_t*>(dst) + 3, width);  // RGBA32 -> RGBX32
}

CONVERT_32_TO_32_FUNC(Cvrt_Argb32_bgrA32, 0x00010203)
void Cvrt_Argb32_BGRA32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_bgrA32_SSE(src, width, dst);  // Argb32 -> bgrA32
  Unmultiply8b(const_cast<const uint8_t*>(reinterpret_cast<uint8_t*>(dst) + 3),
               reinterpret_cast<uint8_t*>(dst) + 0,
               width);  // bgrA32 -> BGRA32
}
void Cvrt_Argb32_BGRX32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_BGRA32_SSE(src, width, dst);                  // Argb32 -> BGRA32
  FillAlpha8b(reinterpret_cast<uint8_t*>(dst) + 3, width);  // BGRA32 -> BGRX32
}

#undef CONVERT_32_TO_32_FUNC
#undef PSHUFB_CST32

WP2_TSAN_IGNORE_FUNCTION void ArgbConverterDspInitSSE() {
  WP2ArgbConvertFrom[WP2_Argb_32] = Cvrt_Argb32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Cvrt_ARGB32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Cvrt_XRGB32_Argb32_SSE;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Cvrt_rgbA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Cvrt_RGBA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Cvrt_RGBX32_Argb32_SSE;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Cvrt_bgrA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Cvrt_BGRA32_Argb32_SSE;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Cvrt_BGRX32_Argb32_SSE;

  WP2ArgbConvertTo[WP2_Argb_32] = Cvrt_Argb32_Argb32_SSE;
  WP2ArgbConvertTo[WP2_ARGB_32] = Cvrt_Argb32_ARGB32_SSE;
  WP2ArgbConvertTo[WP2_XRGB_32] = Cvrt_Argb32_XRGB32_SSE;

  WP2ArgbConvertTo[WP2_rgbA_32] = Cvrt_Argb32_rgbA32_SSE;
  WP2ArgbConvertTo[WP2_RGBA_32] = Cvrt_Argb32_RGBA32_SSE;
  WP2ArgbConvertTo[WP2_RGBX_32] = Cvrt_Argb32_RGBX32_SSE;

  WP2ArgbConvertTo[WP2_bgrA_32] = Cvrt_Argb32_bgrA32_SSE;
  WP2ArgbConvertTo[WP2_BGRA_32] = Cvrt_Argb32_BGRA32_SSE;
  WP2ArgbConvertTo[WP2_BGRX_32] = Cvrt_Argb32_BGRX32_SSE;
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

  WP2ArgbConvertFrom[WP2_Argb_32] = Cvrt_Argb32_Argb32_C;
  WP2ArgbConvertFrom[WP2_ARGB_32] = Cvrt_ARGB32_Argb32_C;
  WP2ArgbConvertFrom[WP2_XRGB_32] = Cvrt_XRGB32_Argb32_C;

  WP2ArgbConvertFrom[WP2_rgbA_32] = Cvrt_rgbA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_RGBA_32] = Cvrt_RGBA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_RGBX_32] = Cvrt_RGBX32_Argb32_C;

  WP2ArgbConvertFrom[WP2_bgrA_32] = Cvrt_bgrA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGRA_32] = Cvrt_BGRA32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGRX_32] = Cvrt_BGRX32_Argb32_C;

  WP2ArgbConvertFrom[WP2_RGB_24] = Cvrt_RGB32_Argb32_C;
  WP2ArgbConvertFrom[WP2_BGR_24] = Cvrt_BGR32_Argb32_C;

  WP2ArgbConvertFrom[WP2_Argb_38] = Cvrt_Argb38_Argb32_C;

  WP2ArgbConvertFrom[WP2_ARGB_64] = Cvrt_ARGB64_Argb32_C;

  WP2ArgbConvertTo[WP2_Argb_32] = Cvrt_Argb32_Argb32_C;
  WP2ArgbConvertTo[WP2_ARGB_32] = Cvrt_Argb32_ARGB32_C;
  WP2ArgbConvertTo[WP2_XRGB_32] = Cvrt_Argb32_XRGB32_C;

  WP2ArgbConvertTo[WP2_rgbA_32] = Cvrt_Argb32_rgbA32_C;
  WP2ArgbConvertTo[WP2_RGBA_32] = Cvrt_Argb32_RGBA32_C;
  WP2ArgbConvertTo[WP2_RGBX_32] = Cvrt_Argb32_RGBX32_C;

  WP2ArgbConvertTo[WP2_bgrA_32] = Cvrt_Argb32_bgrA32_C;
  WP2ArgbConvertTo[WP2_BGRA_32] = Cvrt_Argb32_BGRA32_C;
  WP2ArgbConvertTo[WP2_BGRX_32] = Cvrt_Argb32_BGRX32_C;

  WP2ArgbConvertTo[WP2_RGB_24] = Cvrt_Argb32_RGB32_C;
  WP2ArgbConvertTo[WP2_BGR_24] = Cvrt_Argb32_BGR32_C;

  WP2ArgbConvertTo[WP2_Argb_38] = Cvrt_Argb32_Argb38_C;

  WP2ArgbConvertTo[WP2_ARGB_64] = Cvrt_Argb32_ARGB64_C;

  WP2ARGBConvertFrom[WP2_Argb_32] = Cvrt_Argb32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_ARGB_32] = Cvrt_ARGB32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_XRGB_32] = Cvrt_XRGB32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_rgbA_32] = Cvrt_rgbA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_RGBA_32] = Cvrt_RGBA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_RGBX_32] = Cvrt_RGBX32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_bgrA_32] = Cvrt_bgrA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGRA_32] = Cvrt_BGRA32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGRX_32] = Cvrt_BGRX32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_RGB_24] = Cvrt_RGB32_ARGB32_C;
  WP2ARGBConvertFrom[WP2_BGR_24] = Cvrt_BGR32_ARGB32_C;

  WP2ARGBConvertFrom[WP2_Argb_38] = Cvrt_Argb38_ARGB32_C;

  WP2ARGBConvertFrom[WP2_ARGB_64] = Cvrt_ARGB64_ARGB32_C;

  WP2ARGBConvertTo[WP2_Argb_32] = Cvrt_ARGB32_Argb32_C;
  WP2ARGBConvertTo[WP2_ARGB_32] = Cvrt_ARGB32_ARGB32_C;
  WP2ARGBConvertTo[WP2_XRGB_32] = Cvrt_ARGB32_XRGB32_C;

  WP2ARGBConvertTo[WP2_rgbA_32] = Cvrt_ARGB32_rgbA32_C;
  WP2ARGBConvertTo[WP2_RGBA_32] = Cvrt_ARGB32_RGBA32_C;
  WP2ARGBConvertTo[WP2_RGBX_32] = Cvrt_ARGB32_RGBX32_C;

  WP2ARGBConvertTo[WP2_bgrA_32] = Cvrt_ARGB32_bgrA32_C;
  WP2ARGBConvertTo[WP2_BGRA_32] = Cvrt_ARGB32_BGRA32_C;
  WP2ARGBConvertTo[WP2_BGRX_32] = Cvrt_ARGB32_BGRX32_C;

  WP2ARGBConvertTo[WP2_RGB_24] = Cvrt_ARGB32_RGB32_C;
  WP2ARGBConvertTo[WP2_BGR_24] = Cvrt_ARGB32_BGR32_C;

  WP2ARGBConvertTo[WP2_Argb_38] = Cvrt_ARGB32_Argb38_C;

  WP2ARGBConvertTo[WP2_ARGB_64] = Cvrt_ARGB32_ARGB64_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) ArgbConverterDspInitSSE();
#endif
  }
  argb_converter_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------
// Combinations

namespace {

template <WP2SampleFormat from, WP2SampleFormat to>
void WP2ConvertFromTo(const void* src, uint32_t width, void* dst) {
  const size_t src_bytes_per_pixel =
      WP2FormatBpp(static_cast<WP2SampleFormat>(from));
  const size_t dst_bytes_per_pixel =
      WP2FormatBpp(static_cast<WP2SampleFormat>(to));

  // Use a temporary layout Argb32 or ARGB32 to go from src to dst.
  // Bit depth changes can only go from 8-bit or to 8-bit.
  static_assert(WP2Formatbpc(from) == 8 || WP2Formatbpc(to) == 8,
                "Unimplemented");

  // Do not premultiply or unmultiply unless necessary.
  // Premultiply or unmultiply during the bit depth change if any, to benefit
  // from the increased precision. The other step of shuffling channels is
  // lossless.
  bool tmp_is_premul;
  if (WP2Formatbpc(static_cast<WP2SampleFormat>(from)) <
      WP2Formatbpc(static_cast<WP2SampleFormat>(to))) {
    // For example, for going from 8-bit to 16-bit, it is better to first
    // shuffle channels losslessly, then change the bit depth and unmul/premul.
    tmp_is_premul = WP2IsPremultiplied(static_cast<WP2SampleFormat>(from));
  } else {
    // For example, for going from 16-bit to 8-bit, it is better to first change
    // the bit depth and unmul/premul, then shuffle channels losslessly.
    tmp_is_premul = WP2IsPremultiplied(static_cast<WP2SampleFormat>(to));
  }

  // The functions below do not work inplace so use an 8-bit temporary buffer.
  uint8_t tmp[128];
  while (width > 0) {
    const uint32_t num_pixels = std::min<uint32_t>(width, sizeof(tmp) / 4);
    if (tmp_is_premul) {
      WP2ArgbConvertFrom[from](src, num_pixels, tmp);
      WP2ArgbConvertTo[to](tmp, num_pixels, dst);
    } else {
      WP2ARGBConvertFrom[from](src, num_pixels, tmp);
      WP2ARGBConvertTo[to](tmp, num_pixels, dst);
    }
    src = reinterpret_cast<const uint8_t*>(src) +
          num_pixels * src_bytes_per_pixel;
    dst = reinterpret_cast<uint8_t*>(dst) + num_pixels * dst_bytes_per_pixel;
    width -= num_pixels;
  }
}

template <WP2SampleFormat from>
WP2ArgbConverterF WP2ConvertFromTo8bFunc(WP2SampleFormat to) {
  if (to == WP2_XRGB_32) return WP2ConvertFromTo<from, WP2_XRGB_32>;
  if (to == WP2_rgbA_32) return WP2ConvertFromTo<from, WP2_rgbA_32>;
  if (to == WP2_RGBA_32) return WP2ConvertFromTo<from, WP2_RGBA_32>;
  if (to == WP2_RGBX_32) return WP2ConvertFromTo<from, WP2_RGBX_32>;
  if (to == WP2_bgrA_32) return WP2ConvertFromTo<from, WP2_bgrA_32>;
  if (to == WP2_BGRA_32) return WP2ConvertFromTo<from, WP2_BGRA_32>;
  if (to == WP2_BGRX_32) return WP2ConvertFromTo<from, WP2_BGRX_32>;
  if (to == WP2_RGB_24) return WP2ConvertFromTo<from, WP2_RGB_24>;
  if (to == WP2_BGR_24) return WP2ConvertFromTo<from, WP2_BGR_24>;
  return nullptr;
}

template <WP2SampleFormat from>
WP2ArgbConverterF WP2ConvertFromToFunc(WP2SampleFormat to) {
  if (to == WP2_Argb_38) return WP2ConvertFromTo<from, WP2_Argb_38>;
  if (to == WP2_ARGB_64) return WP2ConvertFromTo<from, WP2_ARGB_64>;
  return WP2ConvertFromTo8bFunc<from>(to);
}

}  // namespace

//------------------------------------------------------------------------------

WP2ArgbConverterF WP2ConversionFunction(WP2SampleFormat from,
                                        WP2SampleFormat to) {
  if (to == WP2_Argb_32) return WP2ArgbConvertFrom[from];
  if (to == WP2_ARGB_32) return WP2ARGBConvertFrom[from];
  if (from == WP2_Argb_32) return WP2ArgbConvertTo[to];
  if (from == WP2_ARGB_32) return WP2ARGBConvertTo[to];
  if (from == to) {
    if (WP2FormatBpp(from) == 3) return WP2Copy_C<3>;
    if (WP2FormatBpp(from) == 4) return WP2Copy_C<4>;
    if (WP2FormatBpp(from) == 8) return WP2Copy_C<8>;
  }

  if (from == WP2_XRGB_32) return WP2ConvertFromToFunc<WP2_XRGB_32>(to);
  if (from == WP2_rgbA_32) return WP2ConvertFromToFunc<WP2_rgbA_32>(to);
  if (from == WP2_RGBA_32) return WP2ConvertFromToFunc<WP2_RGBA_32>(to);
  if (from == WP2_RGBX_32) return WP2ConvertFromToFunc<WP2_RGBX_32>(to);
  if (from == WP2_bgrA_32) return WP2ConvertFromToFunc<WP2_bgrA_32>(to);
  if (from == WP2_BGRA_32) return WP2ConvertFromToFunc<WP2_BGRA_32>(to);
  if (from == WP2_BGRX_32) return WP2ConvertFromToFunc<WP2_BGRX_32>(to);
  if (from == WP2_RGB_24) return WP2ConvertFromToFunc<WP2_RGB_24>(to);
  if (from == WP2_BGR_24) return WP2ConvertFromToFunc<WP2_BGR_24>(to);
  if (to < WP2_Argb_38) {
    if (from == WP2_Argb_38) return WP2ConvertFromTo8bFunc<WP2_Argb_38>(to);
    if (from == WP2_ARGB_64) return WP2ConvertFromTo8bFunc<WP2_ARGB_64>(to);
  }
  return nullptr;
}

//------------------------------------------------------------------------------
