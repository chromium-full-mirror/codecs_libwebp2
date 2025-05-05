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
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include <algorithm>

#include "src/dsp/dsp.h"

#if defined(WP2_USE_SSE)

#include <nmmintrin.h>

#include <cassert>
#include <cstdint>

#include "src/dsp/lossless/decl_dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/wp2/format_constants.h"

namespace WP2L {
namespace {

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void AddGreenToBlueAndRed_SSE(const int16_t* argb, uint32_t num_pixels,
                                     int16_t* dst) {
  const __m128i kCstShuffle =
      _mm_set_epi8(13, 12, -1, -1, 13, 12, -1, -1, 5, 4, -1, -1, 5, 4, -1, -1);
  const int16_t* const argb_end = argb + 4 * num_pixels;
  for (; argb + 8 <= argb_end; argb += 8, dst += 8) {
    const __m128i in = _mm_loadu_si128((const __m128i*)argb);  // bgra
    const __m128i A = _mm_shuffle_epi8(in, kCstShuffle);       // g 0 g 0
    const __m128i out = _mm_add_epi16(in, A);
    _mm_storeu_si128((__m128i*)dst, out);
  }
  // fallthrough and finish off with plain-C
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    AddGreenToBlueAndRed_C(argb, remainder, dst);
  }
}

//------------------------------------------------------------------------------
// Color Transform

void TransformColorInverse_SSE(const WP2L::Multipliers& m, const int16_t* src,
                               uint32_t num_pixels, int16_t* dst,
                               int16_t* predicted) {
  const int32_t green_to_red = m.green_to_red;
  const int32_t green_to_blue = m.green_to_blue;
  const int32_t red_to_blue = m.red_to_blue;
  const int16_t* const src_end = src + 4 * num_pixels;
  const __m128i kMask_ag = _mm_set1_epi32(0xffff);  // A0G0 mask
  const __m128i kRound = _mm_set1_epi32(0x8000);
  const __m128i perm =
      _mm_setr_epi8(2, 3, 4, 5, 6, 7, 4, 5, 10, 11, 12, 13, 14, 15, 12, 13);

  __m128i mult1 = _mm_cvtsi32_si128((uint32_t)green_to_red << 16 | 32);
  __m128i mult2 = _mm_cvtsi32_si128((uint32_t)red_to_blue << 16);
  mult1 = _mm_insert_epi32(mult1, (uint32_t)green_to_blue << 16 | 32u, 1);
  mult1 = _mm_unpacklo_epi64(mult1, mult1);
  mult2 = _mm_unpacklo_epi64(mult2, mult2);

  for (; src + 8 <= src_end; src += 8, dst += 8) {
    if (predicted != nullptr) {
      for (int c : {0, 4}) {
        predicted[c + 0] = src[c + 0];
        const int16_t red = src[c + 1];
        const int16_t green = src[c + 2];
        predicted[c + 1] = ColorTransformDelta(m.green_to_red * green);
        predicted[c + 2] = green;
        const int16_t new_red =
            red + ColorTransformDelta(m.green_to_red * green);
        predicted[c + 3] =
            ColorTransformDelta(green_to_blue * green + red_to_blue * new_red);
      }
      predicted += 8;
    }
    const __m128i A = _mm_loadu_si128((const __m128i*)src);
    const __m128i B = _mm_shuffle_epi8(A, perm);  // ARGB -> RGBG
    const __m128i C = _mm_madd_epi16(B, mult1);
    const __m128i D =
        _mm_add_epi32(_mm_slli_epi32(C, 16 - kColorTransformBits), kRound);
    const __m128i E = _mm_madd_epi16(D, mult2);
    const __m128i F = _mm_slli_epi64(E, 48 - 5);
    const __m128i G = _mm_add_epi32(F, D);
    const __m128i out = _mm_blendv_epi8(G, A, kMask_ag);
    _mm_storeu_si128((__m128i*)dst, out);
  }
  // fallthrough and finish off with plain-C
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    TransformColorInverse_C(m, src, remainder, dst, predicted);
  }
}

//------------------------------------------------------------------------------
// Batch version of Predictor Transform addition.

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
static void PredictorNonClampedAdd0_SSE(const int16_t* in, bool has_alpha,
                                        [[maybe_unused]] const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out,
                                        int16_t* predicted) {
  if (!has_alpha) {
    std::copy(in, in + 4 * num_pixels, out);
    return;
  }
  const __m128i kBlack = kAlphaMax;
  const int16_t* const in_end = in + 4 * num_pixels;
  for (; in + 8 <= in_end; in += 8, out += 8) {
    const __m128i IN = _mm_loadu_si128((const __m128i*)in);
    if (predicted != nullptr) {
      _mm_storeu_si128((__m128i*)predicted, kBlack);
      predicted += 8;
    }
    const __m128i OUT = _mm_add_epi16(IN, kBlack);
    _mm_storeu_si128((__m128i*)out, OUT);
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedAdd_C[0](in, has_alpha, upper, remainder, out,
                                 predicted);
  }
}

// PredictorNonClampedAdd3: average of (average of (L,TL), average of (T, TR)).
template <int ITER>
static inline void PredictorNonClampedAdd3_Step0(const __m128i& IN,
                                                 const __m128i& TL,
                                                 const __m128i& avgTTR,
                                                 __m128i& L,
                                                 int16_t* predicted) {
  __m128i avgLTL, PRED;
  Average2_m128i(L, TL, avgLTL);
  Average2_m128i(avgTTR, avgLTL, PRED);
  if (predicted != nullptr) {
    _mm_storeu_si64((__m128i*)(predicted + 4 * ITER), PRED);
  }
  L = _mm_add_epi16(PRED, IN);
}

static inline void PredictorNonClampedAdd3_Step1(__m128i& IN, __m128i& TL,
                                                 __m128i& avgTTR) {
  // Rotate the pre-computed values for the next iteration.
  avgTTR = _mm_srli_si128(avgTTR, 8);
  TL = _mm_srli_si128(TL, 8);
  IN = _mm_srli_si128(IN, 8);
}

static void PredictorNonClampedAdd3_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out,
                                        int16_t* predicted) {
  const __m128i kMask = _mm_set_epi32(0, 0, 0xffffffff, 0xffffffff);
  const int16_t* const in_end = in + 4 * num_pixels;
  __m128i L = _mm_loadu_si128((const __m128i*)(out - 4));
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TR = _mm_loadu_si128((const __m128i*)(upper + 4));
    __m128i avgTTR;
    Average2_m128i(T, TR, avgTTR);

    PredictorNonClampedAdd3_Step0<0>(IN, TL, avgTTR, L, predicted);
    __m128i OUT = _mm_and_si128(L, kMask);

    PredictorNonClampedAdd3_Step1(IN, TL, avgTTR);
    PredictorNonClampedAdd3_Step0<1>(IN, TL, avgTTR, L, predicted);
    OUT = _mm_or_si128(_mm_slli_si128(L, 8), OUT);

    if (!has_alpha) {
      OUT = _mm_blend_epi16(OUT, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, OUT);
    if (predicted != nullptr) predicted += 8;
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedAdd_C[3](in, has_alpha, upper, remainder, out,
                                 predicted);
  }
}

template <int ITER>
static inline void PredictorNonClampedAdd4_Step0(
    const __m128i& IN, const __m128i& T, const __m128i& TL, const __m128i& C,
    bool has_alpha, __m128i& L, int16_t* predicted) {
  const __m128i LTL0 = _mm_sub_epi16(L, TL);
  const __m128i LTL = _mm_abs_epi16(LTL0);
  const __m128i A = _mm_hadd_epi16(LTL, LTL);
  const __m128i B = _mm_hadd_epi16(A, A);
  const __m128i D = _mm_shuffle_epi8(
      B, _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0));
  const __m128i MASK = _mm_cmplt_epi16(C, D);
  const __m128i PRED = _mm_blendv_epi8(T, L, MASK);
  if (predicted != nullptr) {
    _mm_storeu_si64((__m128i*)(predicted - 4 * ITER), PRED);
  }
  L = _mm_add_epi16(PRED, IN);
  if (!has_alpha) {
    // Alpha has to be fixed here as alpha impacts other channels.
    L = _mm_blend_epi16(L, kAlphaMax, 0b00010001);
  }
}

static inline void PredictorNonClampedAdd4_Step1(__m128i& IN, __m128i& T,
                                                 __m128i& TL, __m128i& C) {
  // Rotate the pre-computed values for the next iteration.
  IN = _mm_srli_si128(IN, 8);
  T = _mm_srli_si128(T, 8);
  TL = _mm_srli_si128(TL, 8);
  C = _mm_srli_si128(C, 8);
}

static void PredictorNonClampedAdd4_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out,
                                        int16_t* predicted) {
  const __m128i kMask = _mm_set_epi32(0, 0, 0xffffffff, 0xffffffff);
  const int16_t* const in_end = in + 4 * num_pixels;
  __m128i L = _mm_loadu_si128((const __m128i*)(out - 4));
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i T = _mm_loadu_si128((const __m128i*)upper);
    __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));
    __m128i C;
    {
      const __m128i TTL0 = _mm_sub_epi16(T, TL);
      const __m128i TTL = _mm_abs_epi16(TTL0);
      const __m128i A = _mm_hadd_epi16(TTL, TTL);
      const __m128i B = _mm_hadd_epi16(A, A);
      C = _mm_shuffle_epi8(
          B, _mm_set_epi8(3, 2, 3, 2, 3, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1, 0));
    }

    PredictorNonClampedAdd4_Step0<0>(IN, T, TL, C, has_alpha, L, predicted);
    __m128i OUT = _mm_and_si128(L, kMask);

    PredictorNonClampedAdd4_Step1(IN, T, TL, C);
    PredictorNonClampedAdd4_Step0<1>(IN, T, TL, C, has_alpha, L, predicted);
    OUT = _mm_or_si128(_mm_slli_si128(L, 8), OUT);

    if (!has_alpha) {
      OUT = _mm_blend_epi16(OUT, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, OUT);
    if (predicted != nullptr) predicted += 8;
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedAdd_C[4](in, has_alpha, upper, remainder, out,
                                 predicted);
  }
}

template <int ITER>
static inline void PredictorNonClampedAdd5_Step0(const __m128i& IN,
                                                 const __m128i& T,
                                                 const __m128i& SUB, __m128i& L,
                                                 int16_t* predicted) {
  const __m128i MIN = _mm_min_epi16(L, T);
  const __m128i MAX = _mm_max_epi16(L, T);
  __m128i PRED = _mm_add_epi16(L, SUB);
  Clamp(MIN, MAX, PRED);
  L = _mm_add_epi16(IN, PRED);
  if (predicted != nullptr) {
    _mm_storeu_si64((__m128i*)(predicted + 4 * ITER), PRED);
  }
  L = _mm_add_epi16(PRED, IN);
}

static inline void PredictorNonClampedAdd5_Step1(__m128i& IN, __m128i& T,
                                                 __m128i& SUB) {
  // Rotate the pre-computed values for the next iteration.
  IN = _mm_srli_si128(IN, 8);
  T = _mm_srli_si128(T, 8);
  SUB = _mm_srli_si128(SUB, 8);
}

static void PredictorNonClampedAdd5_SSE(const int16_t* in, bool has_alpha,
                                        const int16_t* upper,
                                        uint32_t num_pixels, int16_t* out,
                                        int16_t* predicted) {
  const __m128i kMask = _mm_set_epi32(0, 0, 0xffffffff, 0xffffffff);
  const int16_t* const in_end = in + 4 * num_pixels;
  __m128i L = _mm_loadu_si128((const __m128i*)(out - 4));
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    __m128i IN = _mm_loadu_si128((const __m128i*)in);
    __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));
    __m128i SUB = _mm_sub_epi16(T, TL);

    PredictorNonClampedAdd5_Step0<0>(IN, T, SUB, L, predicted);
    __m128i OUT = _mm_and_si128(L, kMask);

    PredictorNonClampedAdd5_Step1(IN, T, SUB);
    PredictorNonClampedAdd5_Step0<1>(IN, T, SUB, L, predicted);
    OUT = _mm_or_si128(_mm_slli_si128(L, 8), OUT);

    if (!has_alpha) {
      OUT = _mm_blend_epi16(OUT, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, OUT);
    if (predicted != nullptr) predicted += 8;
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsNonClampedAdd_C[5](in, has_alpha, upper, remainder, out,
                                 predicted);
  }
}

template <int ITER>
static inline void PredictorClampedAdd0_Step0(const __m128i& IN,
                                              const __m128i& SUB,
                                              const __m128i& MIN,
                                              const __m128i& MAX, __m128i& L,
                                              int16_t* predicted) {
  __m128i PRED = _mm_add_epi16(L, SUB);
  Clamp(MIN, MAX, PRED);
  L = _mm_add_epi16(IN, PRED);
  if (predicted != nullptr) {
    _mm_storeu_si64((__m128i*)(predicted + 4 * ITER), PRED);
  }
  L = _mm_add_epi16(PRED, IN);
}

static inline void PredictorClampedAdd0_Step1(__m128i& IN, __m128i& SUB) {
  // Rotate the pre-computed values for the next iteration.
  IN = _mm_srli_si128(IN, 8);
  SUB = _mm_srli_si128(SUB, 8);
}

static void PredictorClampedAdd0_SSE(const int16_t* in, bool has_alpha,
                                     const int16_t* upper, uint32_t num_pixels,
                                     const int32_t min_values[4],
                                     const int32_t max_values[4], int16_t* out,
                                     int16_t* predicted) {
  const __m128i kMask = _mm_set_epi32(0, 0, 0xffffffff, 0xffffffff);
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);

  const int16_t* const in_end = in + 4 * num_pixels;
  __m128i L = _mm_loadu_si128((const __m128i*)(out - 4));
  for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
    __m128i IN = _mm_loadu_si128((const __m128i*)in);
    const __m128i T = _mm_loadu_si128((const __m128i*)upper);
    const __m128i TL = _mm_loadu_si128((const __m128i*)(upper - 4));
    __m128i SUB = _mm_sub_epi16(T, TL);

    PredictorClampedAdd0_Step0<0>(IN, SUB, MIN, MAX, L, predicted);
    __m128i OUT = _mm_and_si128(L, kMask);

    PredictorClampedAdd0_Step1(IN, SUB);
    PredictorClampedAdd0_Step0<1>(IN, SUB, MIN, MAX, L, predicted);
    OUT = _mm_or_si128(_mm_slli_si128(L, 8), OUT);

    if (!has_alpha) {
      OUT = _mm_blend_epi16(OUT, kAlphaMax, 0b00010001);
    }
    _mm_storeu_si128((__m128i*)out, OUT);
    if (predicted != nullptr) predicted += 8;
  }
  const uint32_t remainder = num_pixels % 2;
  if (remainder > 0) {
    PredictorsClampedAdd_C[0](in, has_alpha, upper, remainder, min_values,
                              max_values, out, predicted);
  }
}

template <int MAIN_ANGLE>
static void AnglePredictorAdd_SSE(int delta, const int16_t* in, bool has_alpha,
                                  const int16_t* upper, uint32_t num_pixels,
                                  int16_t* out, int16_t* predicted) {
  if (delta == 0) {
    const int16_t* const in_end = in + 4 * num_pixels;
    for (; in + 8 <= in_end; in += 8, upper += 8, out += 8) {
      // Load the context.
      [[maybe_unused]] __m128i TL;
      [[maybe_unused]] __m128i T;
      [[maybe_unused]] __m128i TR;
      if constexpr (MAIN_ANGLE <= 90) {
        TR = _mm_loadu_si128((const __m128i*)(upper + 4));
      }
      if constexpr (MAIN_ANGLE >= 67 && MAIN_ANGLE <= 113) {
        T = _mm_loadu_si128((const __m128i*)upper);
      }
      if constexpr (MAIN_ANGLE >= 113 && MAIN_ANGLE <= 135) {
        TL = _mm_loadu_si128((const __m128i*)(upper - 4));
      }
      // Compute the prediction.
      __m128i PRED;
      if constexpr (MAIN_ANGLE == 45) {
        PRED = TR;
      } else if constexpr (MAIN_ANGLE == 67) {
        const __m128i A = _mm_add_epi16(T, TR);
        PRED = _mm_srai_epi16(A, 1);
      } else if constexpr (MAIN_ANGLE == 90) {
        PRED = T;
      } else if constexpr (MAIN_ANGLE == 113) {
        const __m128i A = _mm_add_epi16(TL, T);
        PRED = _mm_srai_epi16(A, 1);
      } else if constexpr (MAIN_ANGLE == 135) {
        PRED = TL;
      }
      if (predicted != nullptr) {
        _mm_storeu_si128((__m128i*)predicted, PRED);
        predicted += 8;
      }
      const __m128i IN = _mm_loadu_si128((const __m128i*)in);
      __m128i OUT = _mm_add_epi16(IN, PRED);
      if (!has_alpha) {
        OUT = _mm_blend_epi16(OUT, kAlphaMax, 0b00010001);
      }
      _mm_storeu_si128((__m128i*)out, OUT);
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
    }
    PredictorsAngleAdd_C[mode](delta, in, has_alpha, upper, num_pixels, out,
                               predicted);
  }
}

static void AnglePredictorAdd180_SSE(int delta, const int16_t* in,
                                     bool has_alpha, const int16_t* upper,
                                     uint32_t num_pixels, int16_t* out,
                                     int16_t* predicted) {
  if (delta == 0) {
    const int16_t* const in_end = in + 4 * num_pixels;
    __m128i PREV = _mm_loadu_si64((const __m128i*)(out - 4));
    PREV = _mm_shuffle_epi32(PREV, (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
    for (; in + 8 <= in_end; in += 8, out += 8) {
      // Compute the prediction.
      const __m128i IN = _mm_loadu_si128((const __m128i*)in);
      const __m128i SHIFT0 = _mm_slli_si128(IN, 8);
      const __m128i SUM = _mm_add_epi16(IN, SHIFT0);
      __m128i PRED = _mm_add_epi16(SUM, PREV);
      if (predicted != nullptr) {
        _mm_storeu_si128((__m128i*)predicted, PRED);
        predicted += 8;
      }
      // Replicate prev output on the four lanes.
      PREV = _mm_shuffle_epi32(PRED, (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
      if (!has_alpha) {
        PRED = _mm_blend_epi16(PRED, kAlphaMax, 0b00010001);
      }
      _mm_storeu_si128((__m128i*)out, PRED);
    }
    num_pixels = num_pixels % 2;
  }
  if (num_pixels != 0) {
    PredictorsAngleAdd_C[6](delta, in, has_alpha, /*upper=*/nullptr, num_pixels,
                            out, predicted);
  }
}

}  // namespace

extern void DecLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void DecLDspInitSSE() {
  AddGreenToBlueAndRed = AddGreenToBlueAndRed_SSE;
  TransformColorInverse = TransformColorInverse_SSE;

  PredictorsNonClampedAdd[0] = PredictorNonClampedAdd0_SSE;
  PredictorsNonClampedAdd[3] = PredictorNonClampedAdd3_SSE;
  PredictorsNonClampedAdd[4] = PredictorNonClampedAdd4_SSE;
  PredictorsNonClampedAdd[5] = PredictorNonClampedAdd5_SSE;

  PredictorsClampedAdd[0] = PredictorClampedAdd0_SSE;

  PredictorsAngleAdd[0] = AnglePredictorAdd_SSE<45>;
  PredictorsAngleAdd[1] = AnglePredictorAdd_SSE<67>;
  PredictorsAngleAdd[2] = AnglePredictorAdd_SSE<90>;
  PredictorsAngleAdd[3] = AnglePredictorAdd_SSE<113>;
  PredictorsAngleAdd[4] = AnglePredictorAdd_SSE<135>;
  PredictorsAngleAdd[6] = AnglePredictorAdd180_SSE;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(DecLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
