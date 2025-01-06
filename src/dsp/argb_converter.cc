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
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

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
template <typename SrcType, WP2SampleFormat from, typename DstType,
          WP2SampleFormat to>
void Cvrt_C(const void* src_data, uint32_t width, void* dst_data) {
  constexpr int kSrcDepth = WP2Formatbpc(from);
  constexpr bool kSrcMult = WP2IsPremultiplied(from);

  constexpr auto kSrcIndices = WP2FormatIndices(from);
  constexpr uint32_t kSrcA = kSrcIndices[0];
  constexpr uint32_t kSrcR = kSrcIndices[1];
  constexpr uint32_t kSrcG = kSrcIndices[2];
  constexpr uint32_t kSrcB = kSrcIndices[3];
  constexpr uint32_t kSrcStep = kSrcIndices[4];

  constexpr int kDstDepth = WP2Formatbpc(to);
  // Destination samples will be premultiplied by output alpha if 'kDstMult'.
  constexpr bool kDstMult = WP2IsPremultiplied(to);

  constexpr auto kDstIndices = WP2FormatIndices(to);
  constexpr uint32_t kDstA = kDstIndices[0];
  constexpr uint32_t kDstR = kDstIndices[1];
  constexpr uint32_t kDstG = kDstIndices[2];
  constexpr uint32_t kDstB = kDstIndices[3];
  constexpr uint32_t kDstStep = kDstIndices[4];

  constexpr uint32_t kSrcMax = (1u << kSrcDepth) - 1u;
  constexpr uint32_t kDstMax = (1u << kDstDepth) - 1u;

  constexpr uint32_t kSrcAlphaMax = (1u << WP2Formatbpalpha(from)) - 1u;
  constexpr uint32_t kSrcAlphaRound = kSrcAlphaMax >> 1;
  constexpr uint32_t kDstAlphaMax = (1u << WP2Formatbpalpha(to)) - 1u;
  constexpr uint32_t kDstAlphaRound = kDstAlphaMax >> 1;

  assert(src_data != dst_data);
  const SrcType* src = reinterpret_cast<const SrcType*>(src_data);
  DstType* dst = reinterpret_cast<DstType*>(dst_data);
  for (uint32_t i = 0; i < width; ++i, src += kSrcStep, dst += kDstStep) {
    // TODO(vrabaud): should be WP2FormatHasAlpha(to).
    if (kDstStep == 4) {
      if (!WP2FormatHasAlpha(from) || !WP2FormatHasAlpha(to)) {
        dst[kDstA] = kDstAlphaMax;
      } else if (kSrcAlphaMax == kDstAlphaMax) {
        dst[kDstA] = src[kSrcA];
      } else {
        dst[kDstA] = static_cast<DstType>(std::min<uint32_t>(
            (src[kSrcA] * kDstAlphaMax + kSrcAlphaRound) / kSrcAlphaMax,
            kDstAlphaMax));
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
              ? kSrcMax != kSrcAlphaMax
                    ? (src[kSrcA] * kSrcMax + kSrcAlphaRound) / kSrcAlphaMax
                    : src[kSrcA]
              : kSrcMax;
      const uint32_t dst_max =
          kDstMult
              ? kDstMax != kDstAlphaMax
                    ? (dst[kDstA] * kDstMax + kDstAlphaRound) / kDstAlphaMax
                    : dst[kDstA]
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

template <std::underlying_type_t<WP2SampleFormat> Start,
          std::underlying_type_t<WP2SampleFormat> End, class F>
constexpr void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + 1, End>(f);
  }
}

constexpr std::array<std::array<WP2ArgbConverterF, WP2_FORMAT_NUM>,
                     WP2_FORMAT_NUM>
InitConversion() {
  std::array<std::array<WP2ArgbConverterF, WP2_FORMAT_NUM>, WP2_FORMAT_NUM>
      res = {};
  constexpr_for<0, static_cast<int>(WP2_FORMAT_NUM)>([&res](auto i_from) {
    constexpr WP2SampleFormat from = static_cast<WP2SampleFormat>(i_from.value);
    constexpr_for<0, static_cast<int>(WP2_FORMAT_NUM)>([from, &res](auto i_to) {
      constexpr WP2SampleFormat to = static_cast<WP2SampleFormat>(i_to.value);
      if (from == to) {
        if (WP2FormatBpp(from) == 3) res[from][to] = WP2Copy_C<3>;
        if (WP2FormatBpp(from) == 4) res[from][to] = WP2Copy_C<4>;
        if (WP2FormatBpp(from) == 6) res[from][to] = WP2Copy_C<6>;
        if (WP2FormatBpp(from) == 8) res[from][to] = WP2Copy_C<8>;
        return;
      }
      constexpr int kSrcDepth = WP2Formatbpc(from);
      constexpr int kDstDepth = WP2Formatbpc(to);
      if constexpr (kSrcDepth == 8 && kDstDepth == 8) {
        res[from][to] =
            Cvrt_C</*SrcType=*/uint8_t, from, /*DstType=*/uint8_t, to>;
      } else if constexpr (kSrcDepth == 8 && kDstDepth != 8) {
        res[from][to] =
            Cvrt_C</*SrcType=*/uint8_t, from, /*DstType=*/uint16_t, to>;
      } else if constexpr (kSrcDepth != 8 && kDstDepth == 8) {
        res[from][to] =
            Cvrt_C</*SrcType=*/uint16_t, from, /*DstType=*/uint8_t, to>;
      } else {
        res[from][to] =
            Cvrt_C</*SrcType=*/uint16_t, from, /*DstType=*/uint16_t, to>;
      }
    });
  });
  return res;
}
constexpr std::array<std::array<WP2ArgbConverterF, WP2_FORMAT_NUM>,
                     WP2_FORMAT_NUM>
    WP2Convert = InitConversion();

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

#define CONVERT_32_TO_32_FUNC(NAME, CST, FUNC_C)                              \
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
    if (x < width) FUNC_C(src + 4 * x, width - x, dst + 4 * x);               \
  }

#define CONVERT_32_TO_32_FILL_FUNC(NAME, CST, CST_A, FUNC_C)                  \
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
    if (x < width) FUNC_C(src + 4 * x, width - x, dst + 4 * x);               \
  }

void Cvrt_Argb32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  memcpy(dst, src, 4 * width);
}

CONVERT_32_TO_32_FILL_FUNC(Cvrt_XRGB32_Argb32, 0x030201ff, 0x000000ffu,
                           WP2Convert[WP2_XRGB_32][WP2_Argb_32])
void Cvrt_ARGB32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_Argb32_Argb32_SSE(src, width, dst);  // ARGB32 -> ARGB32
  PremultiplyARGB32((uint8_t*)dst, width);  // ARGB32 -> Argb32
}

CONVERT_32_TO_32_FUNC(Cvrt_rgbA32_Argb32, 0x02010003,
                      WP2Convert[WP2_rgbA_32][WP2_Argb_32])
CONVERT_32_TO_32_FILL_FUNC(Cvrt_RGBX32_Argb32, 0x020100ff, 0x000000ffu,
                           WP2Convert[WP2_RGBX_32][WP2_Argb_32])
void Cvrt_RGBA32_Argb32_SSE(const void* src, uint32_t width, void* dst) {
  Cvrt_rgbA32_Argb32_SSE(src, width, dst);  // RGBA32 -> ARGB32
  PremultiplyARGB32((uint8_t*)dst, width);  // ARGB32 -> Argb32
}

CONVERT_32_TO_32_FUNC(Cvrt_bgrA32_Argb32, 0x00010203,
                      WP2Convert[WP2_bgrA_32][WP2_Argb_32])
CONVERT_32_TO_32_FILL_FUNC(Cvrt_BGRX32_Argb32, 0x000102ff, 0x000000ffu,
                           WP2Convert[WP2_BGRX_32][WP2_Argb_32])
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

CONVERT_32_TO_32_FUNC(Cvrt_Argb32_rgbA32, 0x00030201,
                      WP2Convert[WP2_Argb_32][WP2_rgbA_32])
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

CONVERT_32_TO_32_FUNC(Cvrt_Argb32_bgrA32, 0x00010203,
                      WP2Convert[WP2_Argb_32][WP2_bgrA_32])
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

  InitConversion();
  for (auto i = 0; i < WP2_FORMAT_NUM; ++i) {
    WP2ArgbConvertFrom[i] = WP2Convert[i][WP2_Argb_32];
    WP2ArgbConvertTo[i] = WP2Convert[WP2_Argb_32][i];
    WP2ARGBConvertFrom[i] = WP2Convert[i][WP2_ARGB_32];
    WP2ARGBConvertTo[i] = WP2Convert[WP2_ARGB_32][i];
  }

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) ArgbConverterDspInitSSE();
#endif
  }
  argb_converter_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------

WP2ArgbConverterF WP2ConversionFunction(WP2SampleFormat from,
                                        WP2SampleFormat to) {
  return WP2Convert[from][to];
}

//------------------------------------------------------------------------------
