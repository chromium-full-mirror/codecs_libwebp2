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

static inline void Clamp(const __m128i* const MIN, const __m128i* const MAX,
                         __m128i* const A) {
  const __m128i CLAMPED_MIN = _mm_max_epi16(*MIN, *A);
  *A = _mm_min_epi16(*MAX, CLAMPED_MIN);
}

static inline void AddSubtractFull_SSE(const int16_t* const a,
                                       const int16_t* const b,
                                       const int16_t* const c, __m128i* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)a);
  const __m128i B = _mm_loadu_si64((const __m128i*)b);
  const __m128i C = _mm_loadu_si64((const __m128i*)c);
  *out = _mm_add_epi16(A, B);
  *out = _mm_sub_epi16(*out, C);
}

static inline void Select_SSE(const int16_t* const a, const int16_t* const b,
                              const int16_t* const c, int16_t* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)a);
  const __m128i B = _mm_loadu_si64((const __m128i*)b);
  const __m128i C = _mm_loadu_si64((const __m128i*)c);
  const __m128i AC0 = _mm_sub_epi16(A, C);
  const __m128i BC0 = _mm_sub_epi16(B, C);
  const __m128i AC = _mm_abs_epi16(AC0);
  const __m128i BC = _mm_abs_epi16(BC0);
  const __m128i D = _mm_hadd_epi16(AC, BC);
  int16_t diff[8];
  _mm_storeu_si128((__m128i*)diff, D);
  if (diff[4] + diff[5] <= diff[0] + diff[1]) {
    _mm_storeu_si64((__m128i*)out, A);
  } else {
    _mm_storeu_si64((__m128i*)out, B);
  }
}

static inline void Average2_SSE(const int16_t* const a, const int16_t* const b,
                                int16_t* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)a);
  const __m128i B = _mm_loadu_si64((const __m128i*)b);
  const __m128i C = _mm_add_epi16(A, B);
  const __m128i D = _mm_srai_epi16(C, 1);
  _mm_storeu_si64((__m128i*)out, D);
}

static inline void Average3_SSE(const int16_t* const a, const int16_t* const b,
                                const int16_t* const c, int16_t* out) {
  Average2_SSE(a, c, out);
  Average2_SSE(out, b, out);
}

static inline void Average4_SSE(const int16_t* const a, const int16_t* const b,
                                const int16_t* const c, const int16_t* const d,
                                int16_t* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)a);
  const __m128i B = _mm_loadu_si64((const __m128i*)b);
  const __m128i C = _mm_loadu_si64((const __m128i*)c);
  const __m128i D = _mm_loadu_si64((const __m128i*)d);
  const __m128i E = _mm_add_epi16(A, B);
  const __m128i F = _mm_srai_epi16(E, 1);
  const __m128i G = _mm_add_epi16(C, D);
  const __m128i H = _mm_srai_epi16(G, 1);
  const __m128i I = _mm_add_epi16(F, H);
  const __m128i J = _mm_srai_epi16(I, 1);
  _mm_storeu_si64((__m128i*)out, J);
}

static void PredictorNonClamped1_SSE(const int16_t* const left,
                                     const int16_t* const top,
                                     int16_t* const out) {
  Average3_SSE(left, top, top + 4, out);
}

static void PredictorNonClamped2_SSE(const int16_t* const left,
                                     const int16_t* const top,
                                     int16_t* const out) {
  Average2_SSE(left, top, out);
}

static void PredictorNonClamped3_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  Average4_SSE(left, top - 4, top, top + 4, out);
}

static void PredictorNonClamped4_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  Select_SSE(top, left, top - 4, out);
}

static void PredictorNonClamped5_SSE(const int16_t* left, const int16_t* top,
                                     int16_t* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)left);
  const __m128i B = _mm_loadu_si64((const __m128i*)top);
  __m128i TMP;
  AddSubtractFull_SSE(left, top, top - 4, &TMP);
  const __m128i MIN = _mm_min_epi16(A, B);
  const __m128i MAX = _mm_max_epi16(A, B);
  Clamp(&MIN, &MAX, &TMP);
  _mm_storeu_si64((__m128i*)out, TMP);
}

static void PredictorClamped0_SSE(const int16_t* left, const int16_t* top,
                                  const int32_t min_values[4],
                                  const int32_t max_values[4], int16_t* out) {
  __m128i TMP;
  AddSubtractFull_SSE(left, top, top - 4, &TMP);
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);
  Clamp(&MIN, &MAX, &TMP);
  _mm_storeu_si64((__m128i*)out, TMP);
}

// Disabled for now as it triggers MSAN errors. It seems to be an error
// upstream.
#if 0
static void PredictorClamped1_SSE(const int16_t* left, const int16_t* top,
                                  const int32_t min_values[4],
                                  const int32_t max_values[4], int16_t* out) {
  const __m128i A = _mm_loadu_si64((const __m128i*)left);
  const __m128i B = _mm_loadu_si64((const __m128i*)top);
  const __m128i C = _mm_loadu_si64((const __m128i*)top - 4);
  const __m128i D = _mm_sub_epi16(A, B);
  const __m128i E = _mm_srai_epi16(D, 1);
  const __m128i F = _mm_sub_epi16(E, C);
  const __m128i G = _mm_srai_epi16(F, 1);
  __m128i H = _mm_add_epi16(E, G);
  __m128i MIN = _mm_loadu_si128((const __m128i*)min_values);
  __m128i MAX = _mm_loadu_si128((const __m128i*)max_values);
  MIN = _mm_packs_epi32(MIN, MIN);
  MAX = _mm_packs_epi32(MAX, MAX);
  Clamp(&MIN, &MAX, &H);
  _mm_storeu_si64((__m128i*)out, H);
}
#endif

static void AnglePredictor67_SSE(int delta, const int16_t* const left,
                                 const int16_t* const top, int16_t* const out) {
  if (delta == 0) {
    Average2_SSE(top, top + 4, out);
  } else {
    AnglePredict<67>(delta, left, top, out);
  }
}

static void AnglePredictor113_SSE(int delta, const int16_t* const left,
                                  const int16_t* const top,
                                  int16_t* const out) {
  if (delta == 0) {
    Average2_SSE(top - 4, top, out);
  } else {
    AnglePredict<67>(delta, left, top, out);
  }
}

static void AnglePredictor157_SSE(int delta, const int16_t* const left,
                                  const int16_t* const top,
                                  int16_t* const out) {
  if (delta == 0) {
    Average2_SSE(left, top - 4, out);
  } else {
    AnglePredict<67>(delta, left, top, out);
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

  PredictorsAngle[1] = AnglePredictor67_SSE;
  PredictorsAngle[3] = AnglePredictor113_SSE;
  PredictorsAngle[5] = AnglePredictor157_SSE;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(DspLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
