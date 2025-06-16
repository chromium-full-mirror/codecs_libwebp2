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

#include <algorithm>

#include "src/dsp/dsp.h"

#if defined(WP2_USE_SSE)

#include <nmmintrin.h>

#include <cassert>
#include <cstdint>

#include "src/dsp/dsp_x86.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/dsp/math.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

namespace {

//------------------------------------------------------------------------------
// Subtract-Green Transform

void SubtractGreenFromBlueAndRed_SSE(int16_t* WP2_RESTRICT argb,
                                     uint32_t num_pixels) {
  const __m128i kCstShuffle =
      _mm_set_epi8(13, 12, -1, -1, 13, 12, -1, -1, 5, 4, -1, -1, 5, 4, -1, -1);
  const int16_t* const argb_end = argb + 4 * num_pixels;
  for (; argb + 8 <= argb_end; argb += 8) {
    const __m128i in = _mm_loadu_si128((__m128i*)argb);
    const __m128i in_g0g0 = _mm_shuffle_epi8(in, kCstShuffle);
    const __m128i out = _mm_sub_epi16(in, in_g0g0);
    _mm_storeu_si128((__m128i*)argb, out);
  }
  // fallthrough and finish off with plain-C
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    SubtractGreenFromBlueAndRed_C(argb, remainder);
  }
}

//------------------------------------------------------------------------------
// Color Transform

// For sign-extended multiplying constants, pre-shifted by 5:
#define MK_CST_16(HI, LO) _mm_set_epi32(HI, LO, HI, LO)

void TransformColor_SSE(const Multipliers& m, uint32_t num_pixels,
                        int16_t* WP2_RESTRICT argb) {
  const __m128i mults_rb = MK_CST_16(m.green_to_blue, m.green_to_red);
  const __m128i mults_b2 = MK_CST_16(m.red_to_blue, 0);
  const __m128i kRound = _mm_set1_epi32(1 << kColorTransformBits >> 1);
  const __m128i kCstShuffleG =
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 13, 12, 5, 4, 5, 4);
  uint32_t i;
  for (i = 0; i + 2 <= num_pixels; i += 2) {
    const __m128i in = _mm_loadu_si128((__m128i*)&argb[4 * i]);  // bgra
    // Get: x x x x g2 g2 g1 g1
    const __m128i A = _mm_shuffle_epi8(in, kCstShuffleG);
    const __m128i B = _mm_cvtepi16_epi32(A);  // g2 g2 g1 g1
    // The comments now show the lowest 64 bits of the registers.
    const __m128i C = _mm_mullo_epi32(B, mults_rb);                // db1 dr
    const __m128i D = _mm_slli_epi64(_mm_srai_epi32(in, 16), 32);  // r 0
    const __m128i E = _mm_mullo_epi32(D, mults_b2);                // db2 0
    const __m128i F = _mm_add_epi32(E, C);                         // db  dr
    const __m128i G =
        _mm_srai_epi32(_mm_add_epi32(F, kRound), kColorTransformBits);

    // Shift to get 16 bits: db 0 dr 0
    const __m128i H = _mm_slli_epi32(G, 16);
    const __m128i out = _mm_sub_epi16(in, H);
    _mm_storeu_si128((__m128i*)&argb[4 * i], out);
  }
  // fallthrough and finish off with plain-C
  if (i != num_pixels) {
    TransformColor_C(m, num_pixels - i, argb + 4 * i);
  }
}

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// Entropy

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

//------------------------------------------------------------------------------
// Batch version of Predictor Transform subtraction

static inline void Clamp(const __m128i& MIN, const __m128i& MAX, __m128i& A) {
  const __m128i CLAMPED_MIN = _mm_max_epi16(MIN, A);
  A = _mm_min_epi16(MAX, CLAMPED_MIN);
}

static inline void Average2_m128i(const __m128i& A, const __m128i& B,
                                  __m128i& OUT) {
  const __m128i C = _mm_add_epi16(A, B);
  OUT = _mm_srai_epi16(C, 1);
}

const __m128i kAlphaMax =
    _mm_set_epi16(0, 0, 0, WP2::kAlphaMax, 0, 0, 0, WP2::kAlphaMax);

// Predictor0: ARGB_BLACK.
static void PredictorNonClampedSub0_SSE(const int16_t* in, bool has_alpha,
                                        [[maybe_unused]] const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  if (!has_alpha) {
    std::copy(in, in + 4 * num_pixels, out);
    return;
  }
  const __m128i kBlack = kAlphaMax;
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, out += 8) {
    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    const __m128i RES = _mm_sub_epi16(IN, kBlack);
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[0](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorNonClampedSub1_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TR = _mm_loadu_si128((const __m128i*)(upper + 4));
    __m128i PRED;
    Average2_m128i(L, TR, PRED);
    Average2_m128i(PRED, T, PRED);
    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[1](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorNonClampedSub2_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    __m128i PRED;
    Average2_m128i(L, T, PRED);
    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[2](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorNonClampedSub3_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TR = _mm_loadu_si128((const __m128i*)(upper + 4));
    __m128i A, B, PRED;
    Average2_m128i(L, TL, A);
    Average2_m128i(T, TR, B);
    Average2_m128i(A, B, PRED);
    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[3](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorNonClampedSub4_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));

    const __m128i TTL0 = _mm_sub_epi16(T, TL);
    const __m128i LTL0 = _mm_sub_epi16(L, TL);
    const __m128i TTL = _mm_abs_epi16(TTL0);
    const __m128i LTL = _mm_abs_epi16(LTL0);
    const __m128i A = _mm_hadd_epi16(TTL, LTL);
    const __m128i B = _mm_hadd_epi16(A, A);
    const __m128i C = _mm_shuffle_epi8(
        B, _mm_set_epi8(3, 2, 3, 2, 3, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1, 0));
    const __m128i D = _mm_shuffle_epi8(
        B, _mm_set_epi8(7, 6, 7, 6, 7, 6, 7, 6, 5, 4, 5, 4, 5, 4, 5, 4));
    const __m128i MASK = _mm_cmplt_epi16(C, D);
    const __m128i PRED = _mm_blendv_epi8(T, L, MASK);

    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[4](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorNonClampedSub5_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out) {
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));

    const __m128i A = _mm_add_epi16(L, T);
    __m128i PRED = _mm_sub_epi16(A, TL);
    const __m128i MIN = _mm_min_epi16(L, T);
    const __m128i MAX = _mm_max_epi16(L, T);
    Clamp(MIN, MAX, PRED);

    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedSub_C[5](in, has_alpha, upper, remainder, out);
  }
}

static void PredictorClampedSub0_SSE(const int16_t* in, bool has_alpha,
                                     const int16_t* upper, uint32_t num_pixels,
                                     const int32_t min_values[4],
                                     const int32_t max_values[4],
                                     int16_t* out) {
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);

  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));

    const __m128i A = _mm_add_epi16(L, T);
    __m128i PRED = _mm_sub_epi16(A, TL);
    Clamp(MIN, MAX, PRED);

    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsClampedSub_C[0](in, has_alpha, upper, remainder, min_values,
                              max_values, out);
  }
}

static void PredictorClampedSub1_SSE(const int16_t* in, bool has_alpha,
                                     const int16_t* upper, uint32_t num_pixels,
                                     const int32_t min_values[4],
                                     const int32_t max_values[4],
                                     int16_t* out) {
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);

  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    const __m128i L = _mm_loadu_si128((const __m128i*)(in - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));

    __m128i A;
    Average2_m128i(L, T, A);
    const __m128i B = _mm_sub_epi16(A, TL);
    const __m128i C = _mm_srai_epi16(B, 1);
    __m128i PRED = _mm_add_epi16(A, C);
    Clamp(MIN, MAX, PRED);

    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i RES = _mm_sub_epi16(IN, PRED);
    if (!has_alpha) {
      RES = _mm_blend_epi16(RES, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, RES);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsClampedSub_C[1](in, has_alpha, upper, remainder, min_values,
                              max_values, out);
  }
}

template <int MAIN_ANGLE>
static void AnglePredictorSub_SSE(int delta, const int16_t* in, bool has_alpha,
                                  const int16_t* upper, uint32_t num_pixels,
                                  int16_t* out) {
  if (delta == 0) {
    const int16_t* const in_end = in + 4 * num_pixels;
    for (; in + 8 <= in_end; in += 8, out += 8) {
      // Load the context.
      [[maybe_unused]] __m128i L;
      [[maybe_unused]] __m128i TL;
      [[maybe_unused]] __m128i T;
      [[maybe_unused]] __m128i TR;
      if constexpr (MAIN_ANGLE <= 90) {
        TR = _mm_loadu_si128((const __m128i*)(upper + 4));
      }
      if constexpr (MAIN_ANGLE >= 67 && MAIN_ANGLE <= 113) {
        T = _mm_loadu_si128((const __m128i*)upper);
      }
      if constexpr (MAIN_ANGLE >= 113 && MAIN_ANGLE <= 157) {
        TL = _mm_loadu_si128((const __m128i*)(upper - 4));
      }
      if constexpr (MAIN_ANGLE >= 157) {
        L = _mm_loadu_si128((const __m128i*)(in - 4));
      }
      // Compute the prediction.
      __m128i pred;
      if constexpr (MAIN_ANGLE == 45) {
        pred = TR;
      } else if constexpr (MAIN_ANGLE == 67) {
        const __m128i A = _mm_add_epi16(T, TR);
        pred = _mm_srai_epi16(A, 1);
      } else if constexpr (MAIN_ANGLE == 90) {
        pred = T;
      } else if constexpr (MAIN_ANGLE == 113) {
        const __m128i A = _mm_add_epi16(TL, T);
        pred = _mm_srai_epi16(A, 1);
      } else if constexpr (MAIN_ANGLE == 135) {
        pred = TL;
      } else if constexpr (MAIN_ANGLE == 157) {
        const __m128i A = _mm_add_epi16(L, TL);
        pred = _mm_srai_epi16(A, 1);
      } else if constexpr (MAIN_ANGLE == 180) {
        pred = L;
      }
      const __m128i src = _mm_loadu_si128((const __m128i*)in);
      __m128i res = _mm_sub_epi16(src, pred);
      if (!has_alpha) {
        res = _mm_blend_epi16(res, kAlphaMax, 0b00010001);
      }
      _mm_storeu_si128((__m128i*)out, res);
      if constexpr (MAIN_ANGLE != 180) upper += 8;
    }
    num_pixels = num_pixels % 2;
  }
  if (num_pixels != 0) {
    int mode;
    if constexpr (MAIN_ANGLE == 45) {
      mode = 0;
    } else if constexpr (MAIN_ANGLE == 67) {
      mode = 1;
    } else if constexpr (MAIN_ANGLE == 90) {
      mode = 2;
    } else if constexpr (MAIN_ANGLE == 113) {
      mode = 3;
    } else if constexpr (MAIN_ANGLE == 135) {
      mode = 4;
    } else if constexpr (MAIN_ANGLE == 157) {
      mode = 5;
    } else if constexpr (MAIN_ANGLE == 180) {
      mode = 6;
    }
    PredictorsAngleSub_C[mode](delta, in, has_alpha, upper, num_pixels, out);
  }
}

}  // namespace

extern void EncLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void EncLDspInitSSE() {
  SubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_SSE;
  TransformColor = TransformColor_SSE;
  CollectColorRedTransforms = CollectColorRedTransforms_SSE;
  CollectColorBlueTransforms = CollectColorBlueTransforms_SSE;
  BufferAdd = BufferAdd_SSE;
  CombinedShannonEntropy = CombinedShannonEntropy_SSE;

  PredictorsNonClampedSub[0] = PredictorNonClampedSub0_SSE;
  PredictorsNonClampedSub[1] = PredictorNonClampedSub1_SSE;
  PredictorsNonClampedSub[2] = PredictorNonClampedSub2_SSE;
  PredictorsNonClampedSub[3] = PredictorNonClampedSub3_SSE;
  PredictorsNonClampedSub[4] = PredictorNonClampedSub4_SSE;
  PredictorsNonClampedSub[5] = PredictorNonClampedSub5_SSE;

  PredictorsClampedSub[0] = PredictorClampedSub0_SSE;
  PredictorsClampedSub[1] = PredictorClampedSub1_SSE;

  PredictorsAngleSub[0] = AnglePredictorSub_SSE<45>;
  PredictorsAngleSub[1] = AnglePredictorSub_SSE<67>;
  PredictorsAngleSub[2] = AnglePredictorSub_SSE<90>;
  PredictorsAngleSub[3] = AnglePredictorSub_SSE<113>;
  PredictorsAngleSub[4] = AnglePredictorSub_SSE<135>;
  PredictorsAngleSub[5] = AnglePredictorSub_SSE<157>;
  PredictorsAngleSub[6] = AnglePredictorSub_SSE<180>;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(EncLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
