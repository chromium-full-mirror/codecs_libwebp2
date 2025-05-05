// Copyright 2025 Google LLC
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
// Authors: Vincent Rabaud (vrabaud@google.com)

#include "src/dsp/dsp.h"

#if defined(WP2_USE_SSE)

#include <nmmintrin.h>

#include <cassert>
#include <cstdint>

#include "src/dsp/lossless/dspl.h"

namespace WP2L {
namespace {

//------------------------------------------------------------------------------
// Predictor Transform

static inline void Clamp(const __m128i& MIN, const __m128i& MAX, __m128i& A) {
  const __m128i CLAMPED_MIN = _mm_max_epi16(MIN, A);
  A = _mm_min_epi16(MAX, CLAMPED_MIN);
}

static inline void Average2_m128i(const __m128i& A, const __m128i& B,
                                  __m128i& OUT) {
  const __m128i C = _mm_add_epi16(A, B);
  OUT = _mm_srai_epi16(C, 1);
}

static void PredictorNonClamped1_SSE(const int16_t* const left,
                                     const int16_t* const top,
                                     int16_t* const out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i TR = _mm_loadu_si64((const __m128i*)(top + 4));
  __m128i PRED;
  Average2_m128i(L, TR, PRED);
  Average2_m128i(PRED, T, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}

static void PredictorNonClamped2_SSE(const int16_t* const left,
                                     const int16_t* const top,
                                     int16_t* const out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  __m128i PRED;
  Average2_m128i(L, T, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}

static void PredictorNonClamped3_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i TL = _mm_loadu_si64((const __m128i*)(top - 4));
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i TR = _mm_loadu_si64((const __m128i*)(top + 4));
  __m128i A, B, PRED;
  Average2_m128i(L, TL, A);
  Average2_m128i(T, TR, B);
  Average2_m128i(A, B, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}

static void PredictorNonClamped4_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i TL = _mm_loadu_si64((const __m128i*)(top - 4));
  const __m128i TTL0 = _mm_sub_epi16(T, TL);
  const __m128i LTL0 = _mm_sub_epi16(L, TL);
  const __m128i TTL = _mm_abs_epi16(TTL0);
  const __m128i LTL = _mm_abs_epi16(LTL0);
  const __m128i A = _mm_hadd_epi16(TTL, LTL);
  int16_t diff[8];
  _mm_storeu_si128((__m128i*)diff, A);
  if (diff[4] + diff[5] <= diff[0] + diff[1]) {
    _mm_storeu_si64((__m128i*)out, T);
  } else {
    _mm_storeu_si64((__m128i*)out, L);
  }
}

static void PredictorNonClamped5_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i TL = _mm_loadu_si64((const __m128i*)(top - 4));
  const __m128i A = _mm_add_epi16(L, T);
  __m128i PRED = _mm_sub_epi16(A, TL);
  const __m128i MIN = _mm_min_epi16(L, T);
  const __m128i MAX = _mm_max_epi16(L, T);
  Clamp(MIN, MAX, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}

static void PredictorClamped0_SSE(const int16_t* left, const int16_t* top,
                                  const int32_t min_values[4],
                                  const int32_t max_values[4], int16_t* out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i TL = _mm_loadu_si64((const __m128i*)(top - 4));
  const __m128i A = _mm_add_epi16(L, T);
  __m128i PRED = _mm_sub_epi16(A, TL);
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);
  Clamp(MIN, MAX, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}

// Disabled for now as it triggers MSAN errors. It seems to be an error
// upstream.
#if 0
static void PredictorClamped1_SSE(const int16_t* left, const int16_t* top,
                                  const int32_t min_values[4],
                                  const int32_t max_values[4], int16_t* out) {
  const __m128i L = _mm_loadu_si64((const __m128i*)left);
  const __m128i T = _mm_loadu_si64((const __m128i*)top);
  const __m128i TL = _mm_loadu_si64((const __m128i*)top - 4);
  __m128i A;
  Average2_m128i(L, T, A);
  const __m128i B = _mm_sub_epi16(A, TL);
  const __m128i C = _mm_srai_epi16(B, 1);
  __m128i PRED = _mm_add_epi16(A, C);
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);
  Clamp(MIN, MAX, PRED);
  _mm_storeu_si64((__m128i*)out, PRED);
}
#endif

template <int MAIN_ANGLE>
static void AnglePredictor_SSE(int delta, const int16_t* const left,
                               const int16_t* const top, int16_t* const out) {
  if (delta == 0) {
    __m128i A, B;
    if constexpr (MAIN_ANGLE == 67) {
      A = _mm_loadu_si64((const __m128i*)top);
      B = _mm_loadu_si64((const __m128i*)(top + 4));
    } else if constexpr (MAIN_ANGLE == 113) {
      A = _mm_loadu_si64((const __m128i*)(top - 4));
      B = _mm_loadu_si64((const __m128i*)top);
    } else if constexpr (MAIN_ANGLE == 157) {
      A = _mm_loadu_si64((const __m128i*)left);
      B = _mm_loadu_si64((const __m128i*)(top - 4));
    }
    __m128i C;
    Average2_m128i(A, B, C);
    _mm_storeu_si64((__m128i*)out, C);
  } else {
    AnglePredict<MAIN_ANGLE>(delta, left, top, out);
  }
}

}  // namespace

extern void DspLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void DspLDspInitSSE() {
  PredictorsNonClamped[1] = PredictorNonClamped1_SSE;
  PredictorsNonClamped[2] = PredictorNonClamped2_SSE;
  PredictorsNonClamped[3] = PredictorNonClamped3_SSE;
  PredictorsNonClamped[4] = PredictorNonClamped4_SSE;
  PredictorsNonClamped[5] = PredictorNonClamped5_SSE;

  PredictorsClamped[0] = PredictorClamped0_SSE;
  // Disabled for now as it triggers MSAN errors. It seems to be an error
  // upstream.
  // PredictorsClamped[1] = PredictorClamped1_SSE;

  PredictorsAngle[1] = AnglePredictor_SSE<67>;
  PredictorsAngle[3] = AnglePredictor_SSE<113>;
  PredictorsAngle[5] = AnglePredictor_SSE<157>;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(DspLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
