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
// Image transform methods for lossless encoder.
//
// Authors: Vincent Rabaud (vrabaud@google.com)

#include "src/dsp/dsp.h"

#if defined(WP2_USE_SSE)

#include <nmmintrin.h>

#include <cassert>
#include <cstdint>

#include "src/dsp/dsp_x86.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/dsp/math.h"

namespace WP2L {

namespace {

//------------------------------------------------------------------------------

#define CSE_CHECK_ZERO(m, j)                      \
  do {                                            \
    if (my & m) {                                 \
      retval += WP2SLog2m1(X[i + j] + Y[i + j]);  \
      if (mx & m) retval += WP2SLog2m1(X[i + j]); \
    }                                             \
  } while (false)

float CombinedShannonEntropy_SSE(const uint32_t* X, const uint32_t* Y,
                                 uint32_t length) {
  float retval = 0.f;
  uint32_t i = 0;
  const __m128i zero = _mm_setzero_si128();
  for (; i + 16 <= length; i += 16) {
    const __m128i x0 = _mm_loadu_si128((const __m128i*)(X + i + 0));
    const __m128i y0 = _mm_loadu_si128((const __m128i*)(Y + i + 0));
    const __m128i x1 = _mm_loadu_si128((const __m128i*)(X + i + 4));
    const __m128i y1 = _mm_loadu_si128((const __m128i*)(Y + i + 4));
    const __m128i x2 = _mm_loadu_si128((const __m128i*)(X + i + 8));
    const __m128i y2 = _mm_loadu_si128((const __m128i*)(Y + i + 8));
    const __m128i x3 = _mm_loadu_si128((const __m128i*)(X + i + 12));
    const __m128i y3 = _mm_loadu_si128((const __m128i*)(Y + i + 12));
    const __m128i x4 =
        _mm_packs_epi16(_mm_packs_epi32(x0, x1), _mm_packs_epi32(x2, x3));
    const __m128i y4 =
        _mm_packs_epi16(_mm_packs_epi32(y0, y1), _mm_packs_epi32(y2, y3));
    const int32_t mx = _mm_movemask_epi8(_mm_cmpgt_epi8(x4, zero));
    int32_t my = _mm_movemask_epi8(_mm_cmpgt_epi8(y4, zero)) | mx;
    while (my) {
      const int32_t j = WP2Ctz(my);
      retval += WP2SLog2m1(X[i + j] + Y[i + j]);
      if ((mx >> j) & 1) retval += WP2SLog2m1(X[i + j]);
      my &= my - 1;
    }
  }
  for (; i + 4 <= length; i += 4) {
    const __m128i x0 = _mm_loadu_si128((const __m128i*)(X + i));
    const __m128i y0 = _mm_loadu_si128((const __m128i*)(Y + i));
    const int32_t mx = _mm_movemask_epi8(_mm_cmpgt_epi32(x0, zero));
    const int32_t my = _mm_movemask_epi8(_mm_cmpgt_epi32(y0, zero)) | mx;
    if (!my) continue;
    CSE_CHECK_ZERO(0x000f, 0);
    CSE_CHECK_ZERO(0x00f0, 1);
    CSE_CHECK_ZERO(0x0f00, 2);
    CSE_CHECK_ZERO(0xf000, 3);
  }
  for (; i < length; ++i) {
    const uint32_t x = X[i];
    const uint32_t y = Y[i];
    if (x != 0) {
      retval += WP2SLog2m1(x + y);
      retval += WP2SLog2m1(x);
    } else if (y != 0) {
      retval += WP2SLog2m1(y);
    }
  }
  return retval;
}
#undef CSE_CHECK_ZERO

inline bool DoCollectTransforms_SSE(const int16_t* src, const __m128i mult,
                                    const __m128i vmaxval,
                                    const __m128i vmaxidx,
                                    uint32_t* const histo) {
  const __m128i kRound = _mm_set1_epi32(15);
  const __m128i argb0 = _mm_loadu_si128((const __m128i*)(src + 0));
  const __m128i argb1 = _mm_loadu_si128((const __m128i*)(src + 8));
  const __m128i t0 = _mm_madd_epi16(argb0, mult);
  const __m128i t1 = _mm_madd_epi16(argb1, mult);
  const __m128i t2 = _mm_add_epi32(_mm_hadd_epi32(t0, t1), kRound);
  const __m128i value = _mm_srai_epi32(t2, 5);
  const __m128i absvalue = _mm_abs_epi32(value);

  if (_mm_movemask_epi8(_mm_cmpgt_epi32(absvalue, vmaxval))) return false;

  const __m128i sign = _mm_srai_epi32(value, 31);
  const __m128i index1 = _mm_xor_si128(_mm_add_epi32(value, value), sign);
  const __m128i index = _mm_min_epi32(index1, vmaxidx);
  ++histo[_mm_extract_epi32(index, 0)];
  ++histo[_mm_extract_epi32(index, 1)];
  ++histo[_mm_extract_epi32(index, 2)];
  ++histo[_mm_extract_epi32(index, 3)];
  return true;
}

bool CollectColorRedTransforms_SSE(const int16_t* const argb_ini,
                                   uint32_t width, uint32_t tile_width,
                                   uint32_t tile_height, int16_t green_to_red,
                                   uint16_t abs_value_max, uint32_t histo_size,
                                   uint32_t* const histo) {
  __m128i mult = _mm_cvtsi32_si128(32 << 16);
  mult = _mm_insert_epi32(mult, (uint16_t)(-green_to_red), 1);
  mult = _mm_unpacklo_epi64(mult, mult);
  const __m128i vmaxval = _mm_set1_epi32(abs_value_max);
  const __m128i vmaxidx = _mm_set1_epi32(histo_size - 1);
  const int16_t* argb = argb_ini;

  for (uint32_t i = 0; i < tile_height; ++i, argb += 4 * width) {
    uint32_t x = 0;
    for (; x + 4 <= tile_width; x += 4) {
      if (!DoCollectTransforms_SSE(argb + 4 * x, mult, vmaxval, vmaxidx, histo))
        return false;
    }
    const bool result = CollectColorRedTransforms_C(
        argb + 4 * (tile_width & ~3), width, tile_width - (tile_width & ~3),
        /*tile_height=*/1, green_to_red, abs_value_max, histo_size, histo);
    if (!result) return false;
  }
  return true;
}

bool CollectColorBlueTransforms_SSE(const int16_t* argb, uint32_t width,
                                    uint32_t tile_width, uint32_t tile_height,
                                    int16_t green_to_blue, int16_t red_to_blue,
                                    uint16_t abs_value_max, uint32_t histo_size,
                                    uint32_t* const histo) {
  __m128i mult = _mm_cvtsi32_si128((uint16_t)(-red_to_blue) << 16);
  mult = _mm_insert_epi32(mult, (uint16_t)(-green_to_blue) | (32 << 16), 1);
  mult = _mm_unpacklo_epi64(mult, mult);
  const __m128i vmaxval = _mm_set1_epi32(abs_value_max);
  const __m128i vmaxidx = _mm_set1_epi32(histo_size - 1);

  for (uint32_t i = 0; i < tile_height; ++i, argb += 4 * width) {
    uint32_t x = 0;
    for (; x + 4 <= tile_width; x += 4) {
      if (!DoCollectTransforms_SSE(argb + 4 * x, mult, vmaxval, vmaxidx, histo))
        return false;
    }
    const bool result = CollectColorBlueTransforms_C(
        argb + 4 * (tile_width & ~3), width, tile_width - (tile_width & ~3),
        /*tile_height=*/1, green_to_blue, red_to_blue, abs_value_max,
        histo_size, histo);
    if (!result) return false;
  }
  return true;
}

void BufferAdd_SSE(const uint32_t* const a, const uint32_t* const b,
                   uint32_t size, uint32_t* const out,
                   uint32_t* const nonzeros) {
  assert(out != nullptr);
  const __m128i zero = _mm_setzero_si128();
  VecI32_SSE zero_cnt(0);
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    const VecI32_SSE m = VecI32_SSE(a + i) + VecI32_SSE(b + i);
    m.Store(out + i);
    zero_cnt = zero_cnt - _mm_cmpeq_epi32(m, zero);
  }
  if (nonzeros != nullptr) {
    uint32_t tmp[4];
    zero_cnt.Store(tmp);
    *nonzeros = i - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
  }
  for (; i < size; ++i) {
    out[i] = a[i] + b[i];
    if (nonzeros != nullptr && out[i]) ++*nonzeros;
  }
}

}  // namespace

extern void EncLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void EncLDspInitSSE() {
  CombinedShannonEntropy = CombinedShannonEntropy_SSE;
  CollectColorRedTransforms = CollectColorRedTransforms_SSE;
  CollectColorBlueTransforms = CollectColorBlueTransforms_SSE;
  BufferAdd = BufferAdd_SSE;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(EncLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
