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
// Speed-critical deblocking functions
//
// Author: Yannis Guyon (yguyon@google.com)

#include <algorithm>
#include <cstring>
#include <numeric>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

constexpr uint32_t kWeightLog2 = 8;
constexpr uint32_t kStrengthShift = WP2Log2Ceil_k(kDblkMaxStrength);
constexpr uint32_t kBlurShift = (kWeightLog2 + kStrengthShift + 1);  // 16bit
constexpr uint32_t kBlurUnit = 1u << kBlurShift;

static_assert(kDblkMaxHalf <= 8u, "kDblkMaxHalf should be adjusted");

// Weights representing how much to blend a pixel with the average.
// Per line length, per distance to edge. Fixed point precision is kWeightLog2.
// Values are chosen experimentally.
constexpr uint16_t kBlurWeights[kDblkMaxHalf][8] = {
    {160},
    {180, 70},
    {190, 89, 22},
    {200, 113, 50, 13},
    {200, 128, 72, 32, 8},
    {200, 139, 89, 50, 22, 6},
    {200, 147, 102, 65, 37, 16, 4},
    {200, 153, 113, 78, 50, 28, 13, 3},
};
constexpr uint16_t kWeightThreshold(int i) {
  return kBlurUnit / (2 * kBlurWeights[i][0]);
}
constexpr uint16_t kBlurWeightsThreshold[kDblkMaxHalf] = {
  kWeightThreshold(0), kWeightThreshold(1),
  kWeightThreshold(2), kWeightThreshold(3),
  kWeightThreshold(4), kWeightThreshold(5),
  kWeightThreshold(6), kWeightThreshold(7),
};

// Per sharpness. Threshold to decide whether to filter a line or not-at-all.
// Values are chosen experimentally with a fixed-point precision of 10 bits.
constexpr int32_t kSharpnessThreshold[kDblkMaxSharpness + 1] = {
  220, 212, 204, 168, 108, 72, 40, 20
};

// Per sharpness, per distance to edge. Threshold between this pixel and the
// previous one (hence index 0 is irrelevant). Values are chosen experimentally.
// TODO(yguyon): Take 'num_precision_bits' into account.
// clang-format off
constexpr int8_t kFlatnessThreshold[kDblkMaxSharpness + 1][kDblkMaxHalf] = {
  // 0x7f is the max int8_t value for thresh, to guaranty that the comparison
  // "abs_all_8b > thresh" will always be false (and result in a leading 0 bit).
  {0x7f, 82, 15, 7, 4, 3, 2, 2},
  {0x7f, 80, 10, 6, 4, 3, 2, 2},
  {0x7f, 78,  7, 5, 4, 3, 2, 2},
  {0x7f, 66,  6, 4, 4, 3, 2, 1},
  {0x7f, 54,  6, 4, 3, 2, 1, 1},
  {0x7f, 40,  6, 4, 3, 2, 1, 1},
  {0x7f, 37,  4, 3, 2, 2, 1, 1},
  {0x7f, 25,  3, 2, 2, 1, 1, 1}
};
// clang-format on

//------------------------------------------------------------------------------
// plain-C implementation

// returns ...p0|q0... edge difference
static int32_t EdgeDelta(const int16_t* const q0, uint32_t half) {
  assert(half > 0 && half <= kDblkMaxHalf);

  // Extrapolate the edge color from the two closest pixels (consider it to be
  // between p0 and q0, at a distance of 0.5 pixels from each).
  const int16_t* const p0 = q0 - 1;
  int32_t diff = (q0[0] - p0[0]);
  if (half > 1) {
    diff += RightShift(diff - q0[1] + p0[-1], 1);
  }
  return diff;
}

uint8_t WouldDeblockLine_C(int32_t threshold, uint32_t half,
                           const int16_t q0[]) {
  return (std::abs(EdgeDelta(q0, half)) <= threshold) ? 1 : 0;
}

void DeblockLine_C(uint32_t filter_strength, int32_t threshold,
                   uint8_t half, int32_t min, int32_t max, int16_t* q0) {
  assert(filter_strength > 0 && half > 0);
  int32_t delta = EdgeDelta(q0, half);
  if (std::abs(delta) > threshold) return;
  // Do not filter invisible block edges, consider them already deblocked.
  delta *= (int32_t)filter_strength;
  if (std::abs(delta) <= kBlurWeightsThreshold[half - 1]) return;
  int16_t* p = q0 - 1;
  int16_t* q = q0;
  for (uint32_t i = 0; i < half; ++i, p -= 1, q += 1) {
    // warning! -(b>>1) is not equal to (-b>>1), that's why we need two deltas
    const int16_t p_weighted_delta = RightShiftRound(
        +delta * kBlurWeights[half - 1][i], kBlurShift);
    const int16_t m_weighted_delta = RightShiftRound(
        -delta * kBlurWeights[half - 1][i], kBlurShift);

    // since kBlurWeights[] is decreasing, check if we can early-out
    if ((m_weighted_delta | p_weighted_delta) == 0) break;
    *p = (int16_t)Clamp(*p + p_weighted_delta, min, max);
    *q = (int16_t)Clamp(*q + m_weighted_delta, min, max);
  }
}

uint32_t MeasureFlatLength_2(uint32_t filter_sharpness,
                             const int16_t* const q0) {
  const int16_t* const p = q0 - 1;
  const int16_t* const q = q0;
  const int32_t diff0 = q[0] - p[0];
  const int32_t diff_q = q[1] - q[0];
  const int32_t diff_p = p[0] - p[-1];
  const int32_t t = kFlatnessThreshold[filter_sharpness][1];
  if (std::abs(diff_q - diff0) > t || std::abs(diff_p - diff0) > t) return 1;
  return 2;
}

uint32_t MeasureFlatLength_C(uint32_t filter_sharpness, uint32_t half,
                             const int16_t* q0) {
  assert(half > 0 && half <= kDblkMaxHalf);
  if (half == 1) return 1;
  if (half == 2) return MeasureFlatLength_2(filter_sharpness, q0);
  const int16_t* p = q0 - 1;
  const int16_t* q = q0;
  int32_t previous_diff_p = *p - *q;
  int32_t previous_diff_q = -previous_diff_p;
  for (uint32_t i = 1; i < half; ++i) {
    const int32_t previous_p = *p, previous_q = *q;
    p -= 1;
    q += 1;
    const int32_t diff_p = *p - previous_p, diff_q = *q - previous_q;
    const int32_t t = kFlatnessThreshold[filter_sharpness][i];
    if (std::abs(diff_p - previous_diff_p) > t ||
        std::abs(diff_q - previous_diff_q) > t) {
      return i;
    }
    previous_diff_p = diff_p;
    previous_diff_q = diff_q;
  }
  return half;
}

void MeasureFlatLengths_C(uint32_t filter_sharpness, uint32_t half,
                          const int16_t* q0, uint32_t step,
                          uint8_t out[], uint32_t size) {
  for (uint32_t i = 0; i < size; ++i, q0 += step) {
    out[i] = MeasureFlatLength_C(filter_sharpness, half, q0);
  }
}

//------------------------------------------------------------------------------
// buffer I/O

void CopyIn_C(const int16_t* src, uint32_t src_step,
              int16_t* dst, int32_t width, int32_t height) {
  src -= height * src_step;
  dst -= height;
  for (int j = 0; j < 2 * height; ++j, src += src_step, dst += 1) {
    for (int i = 0; i < width; ++i) dst[i * kDblCacheStep] = src[i];
  }
}

void CopyOut_C(const int16_t* src, int16_t* dst, uint32_t dst_step,
               int32_t width, int32_t height) {
  src -= height;
  dst -= height * dst_step;
  for (int j = 0; j < 2 * height; ++j, src += 1, dst += dst_step) {
    for (int i = 0; i < width; ++i) dst[i] = src[i * kDblCacheStep];
  }
}

}  // namespace

//------------------------------------------------------------------------------
// SSE4.1 implementation

#if defined(WP2_USE_SSE)

namespace {

// loads 8 int16's of input
#define LOAD_ROW8(ROW) \
    _mm_loadu_si128((const __m128i*)(src + (ROW) * kDblCacheStep))
// stores 2x4 int16's of output
#define STORE_ROW2x4(REG, ROW, STEP)                                \
  _mm_storel_epi64((__m128i*)(dst + ((ROW) + 0) * (STEP)), (REG));  \
  _mm_storel_epi64((__m128i*)(dst + ((ROW) + 1) * (STEP)),          \
                   _mm_srli_si128((REG), 8))
void Transpose8x4_SSE(const int16_t* src, int16_t* dst, uint32_t dst_step) {
  const __m128i a0 = LOAD_ROW8(0);  // a00 a01 ... a07
  const __m128i a1 = LOAD_ROW8(1);  // a10 a11 ... a17
  const __m128i a2 = LOAD_ROW8(2);  // a20 a21 ... a27
  const __m128i a3 = LOAD_ROW8(3);  // a30 a31 ... a37
  const __m128i b0 = _mm_unpacklo_epi16(a0, a1);  // a00 a10 a01 a11 ... a03 a13
  const __m128i b1 = _mm_unpackhi_epi16(a0, a1);  // a04 a14 a05 a15 ... a07 a17
  const __m128i b2 = _mm_unpacklo_epi16(a2, a3);  // a20 a30 a21 a31 ... a23 a33
  const __m128i b3 = _mm_unpackhi_epi16(a2, a3);  // a24 a34 a25 a35 ... a27 a37
  const __m128i c0 = _mm_unpacklo_epi32(b0, b2);  // a00 a10 a20 a30 ... a21 a31
  const __m128i c1 = _mm_unpackhi_epi32(b0, b2);  // a02 a12 a22 a32 ... a23 a33
  const __m128i c2 = _mm_unpacklo_epi32(b1, b3);  // a04 a14 a24 a34 ... a25 a35
  const __m128i c3 = _mm_unpackhi_epi32(b1, b3);  // a06 a16 a17 a18 ... a27 a37
  STORE_ROW2x4(c0, 0, dst_step);
  STORE_ROW2x4(c1, 2, dst_step);
  STORE_ROW2x4(c2, 4, dst_step);
  STORE_ROW2x4(c3, 6, dst_step);
}
#undef STORE_ROW2x4
#undef LOAD_ROW8

// loads 2x4 int16's of input
#define LOAD_ROW2x4(ROW, STEP) _mm_unpacklo_epi16(                   \
  _mm_loadl_epi64((const __m128i*)(src + (ROW + 0) * (STEP))),  \
  _mm_loadl_epi64((const __m128i*)(src + (ROW + 1) * (STEP))))
// stores 8 int16's of output
#define STORE_ROW8(REG, ROW) \
  _mm_storeu_si128((__m128i*)(dst + (ROW) * kDblCacheStep), REG)
void Transpose4x8_SSE(const int16_t* src, uint32_t src_step, int16_t* dst) {
  const __m128i a0 = LOAD_ROW2x4(0, src_step);    // a00 a10 a01 a11 ... a03 a13
  const __m128i a1 = LOAD_ROW2x4(2, src_step);    // a20 a30 a21 a31 ... a23 a33
  const __m128i a2 = LOAD_ROW2x4(4, src_step);    // a40 a50 a41 a51 ... a43 a53
  const __m128i a3 = LOAD_ROW2x4(6, src_step);    // a60 a70 a61 a71 ... a63 a73
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);  // a00 a10 a20 a30 ... a21 a31
  const __m128i b1 = _mm_unpackhi_epi32(a0, a1);  // a02 a12 a22 a32 ... a23 a33
  const __m128i b2 = _mm_unpacklo_epi32(a2, a3);  // a40 a50 a60 a70 ... a61 a71
  const __m128i b3 = _mm_unpackhi_epi32(a2, a3);  // a42 a52 a62 a72 ... a63 a73
  const __m128i c0 = _mm_unpacklo_epi64(b0, b2);  // a00 a10 ... a70
  const __m128i c1 = _mm_unpackhi_epi64(b0, b2);  // a01 a11 ... a71
  const __m128i c2 = _mm_unpacklo_epi64(b1, b3);  // a02 a12 ... a72
  const __m128i c3 = _mm_unpackhi_epi64(b1, b3);  // a03 a13 ... a73
  STORE_ROW8(c0, 0);
  STORE_ROW8(c1, 1);
  STORE_ROW8(c2, 2);
  STORE_ROW8(c3, 3);
}
#undef STORE_ROW8
#undef LOAD_ROW2x4

void CopyIn_SSE(const int16_t* src, uint32_t src_step,
                int16_t* dst, int32_t width, int32_t height) {
  for (int y = 0; y < width; y += 4) {
    if (height <= 4) {
      Transpose4x8_SSE(src - 4 * src_step, src_step, dst - 4);
    } else {
      Transpose4x8_SSE(src - 8 * src_step, src_step, dst - 8);
      Transpose4x8_SSE(src - 0 * src_step, src_step, dst - 0);
    }
    src += 4;
    dst += 4 * kDblCacheStep;
  }
}

void CopyOut_SSE(const int16_t* src, int16_t* dst, uint32_t dst_step,
                 int32_t width, int32_t height) {
  for (int y = 0; y < width; y += 4) {
    if (height <= 4) {
      Transpose8x4_SSE(src - 4, dst - 4 * dst_step, dst_step);
    } else {
      Transpose8x4_SSE(src - 8, dst - 8 * dst_step, dst_step);
      Transpose8x4_SSE(src - 0, dst - 0 * dst_step, dst_step);
    }
    dst += 4;
    src += 4 * kDblCacheStep;
  }
}

//------------------------------------------------------------------------------
// Actual deblocking

// clang-format off
constexpr int16_t kBlurWeights_SSE[kDblkMaxHalf][16] = {
  { 0,  0,  0,  0,  0,   0,   0, 160, -160,    0,    0,   0,   0,   0,   0,  0},
  { 0,  0,  0,  0,  0,   0,  70, 180, -180,  -70,    0,   0,   0,   0,   0,  0},
  { 0,  0,  0,  0,  0,  22,  89, 190, -190,  -89,  -22,   0,   0,   0,   0,  0},
  { 0,  0,  0,  0, 13,  50, 113, 200, -200, -113,  -50, -13,   0,   0,   0,  0},
  { 0,  0,  0,  8, 32,  72, 128, 200, -200, -128,  -72, -32,  -8,   0,   0,  0},
  { 0,  0,  6, 22, 50,  89, 139, 200, -200, -139,  -89, -50, -22,  -6,   0,  0},
  { 0,  4, 16, 37, 65, 102, 147, 200, -200, -147, -102, -65, -37, -16,  -4,  0},
  { 3, 13, 28, 50, 78, 113, 153, 200, -200, -153, -113, -78, -50, -28, -13, -3}
};
// clang-format on
static_assert(kBlurShift == 15, "_mm_mulhrs_epi16 needs to be changed.");

void DeblockLine_SSE(uint32_t filter_strength, int32_t threshold,
                     uint8_t half, int32_t min, int32_t max, int16_t* q0) {
  assert(filter_strength > 0 && half > 0);
  const __m128i m_min = _mm_set1_epi16(min);
  const __m128i m_max = _mm_set1_epi16(max);
  int32_t delta = EdgeDelta(q0, half);
  if (std::abs(delta) > threshold) return;
  // Do not filter invisible block edges, consider them already deblocked.
  delta *= (int32_t)filter_strength;
  if (std::abs(delta) <= kBlurWeightsThreshold[half - 1]) return;
  assert(std::abs(delta) < 0x7fff);
  const int16_t* const weights = kBlurWeights_SSE[half - 1];
  const __m128i m_delta = _mm_set1_epi16(delta);
  if (half <= 4) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(weights + 4));
    const __m128i b0 = _mm_mulhrs_epi16(a0, m_delta);
    const __m128i c0 = _mm_loadu_si128((const __m128i*)(q0 - 4));
    const __m128i d0 = _mm_adds_epi16(c0, b0);
    const __m128i e0 = _mm_min_epi16(_mm_max_epi16(d0, m_min), m_max);
    _mm_storeu_si128((__m128i*)(q0 - 4), e0);
  } else {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(weights + 0));
    const __m128i a1 = _mm_loadu_si128((const __m128i*)(weights + 8));
    const __m128i b0 = _mm_mulhrs_epi16(a0, m_delta);
    const __m128i b1 = _mm_mulhrs_epi16(a1, m_delta);
    const __m128i c0 = _mm_loadu_si128((const __m128i*)(q0 - 8));
    const __m128i c1 = _mm_loadu_si128((const __m128i*)(q0 - 0));
    const __m128i d0 = _mm_adds_epi16(c0, b0);
    const __m128i d1 = _mm_adds_epi16(c1, b1);
    const __m128i e0 = _mm_min_epi16(_mm_max_epi16(d0, m_min), m_max);
    const __m128i e1 = _mm_min_epi16(_mm_max_epi16(d1, m_min), m_max);
    _mm_storeu_si128((__m128i*)(q0 - 8), e0);
    _mm_storeu_si128((__m128i*)(q0 - 0), e1);
  }
}

void MeasureFlatLengths_SSE(uint32_t filter_sharpness, uint32_t half,
                          const int16_t* q0, uint32_t step,
                          uint8_t out[], uint32_t size) {
  assert(half > 0 && half <= kDblkMaxHalf);
  if (size <= 0) return;
  if (half <= 2) {
    if (half < 2) {
      for (uint32_t j = 0; j < size; ++j) out[j] = 1;
    } else {
      const int32_t t = kFlatnessThreshold[filter_sharpness][1];
      for (uint32_t j = 0; j < size; ++j, q0 += step) {
        const int32_t diff_p = q0[-1] - q0[-2];
        const int32_t diff0  = q0[ 0] - q0[-1];
        const int32_t diff_q = q0[ 1] - q0[ 0];
        out[j] = (std::abs(diff_q - diff0) > t ||
                  std::abs(diff_p - diff0) > t) ? 1 : 2;
      }
    }
    return;
  }

  static const uint8_t kThresh_SSE[kDblkMaxSharpness + 1][2 * kDblkMaxHalf] = {
    {82, 15, 7, 4, 3, 2, 2, 0x7f, 0x7f, 2, 2, 3, 4, 7, 15, 82},
    {80, 10, 6, 4, 3, 2, 2, 0x7f, 0x7f, 2, 2, 3, 4, 6, 10, 80},
    {78,  7, 5, 4, 3, 2, 2, 0x7f, 0x7f, 2, 2, 3, 4, 5,  7, 78},
    {66,  6, 4, 4, 3, 2, 1, 0x7f, 0x7f, 1, 2, 3, 4, 4,  6, 66},
    {54,  6, 4, 3, 2, 1, 1, 0x7f, 0x7f, 1, 1, 2, 3, 4,  6, 54},
    {40,  6, 4, 3, 2, 1, 1, 0x7f, 0x7f, 1, 1, 2, 3, 4,  6, 40},
    {37,  4, 3, 2, 2, 1, 1, 0x7f, 0x7f, 1, 1, 2, 2, 3,  4, 37},
    {25,  3, 2, 2, 1, 1, 1, 0x7f, 0x7f, 1, 1, 1, 2, 2,  3, 25}
  };
  static const uint16_t limit[8] = {
    0xffff, 0x7ffe, 0x3ffc, 0x1ff8, 0x0ff0, 0x07e0, 0x03c0, 0x0180
  };
  const __m128i thresh =
      _mm_loadu_si128((const __m128i*)(kThresh_SSE[filter_sharpness]));
  const uint32_t or_mask = limit[half - 1];

  if (half <= 4) {
    for (uint32_t j = 0; j < size; ++j, q0 += step) {
      const __m128i Q = _mm_loadu_si128((const __m128i*)(q0 - 4));
      const __m128i diff1 = _mm_subs_epi16(Q, _mm_srli_si128(Q, 2));
      const __m128i diff2 = _mm_slli_si128(diff1, 2);
      const __m128i abs16 = _mm_abs_epi16(_mm_subs_epi16(diff1, diff2));
      const __m128i abs8 = _mm_packs_epi16(abs16, abs16);
      const __m128i abs_dPQ = _mm_shuffle_epi32(abs8, 0x05);
      const __m128i bits_PQ = _mm_cmpgt_epi8(abs_dPQ, thresh);
      const uint32_t bits = _mm_movemask_epi8(bits_PQ) | or_mask;
#if defined(__GNUC__)
      if (!bits) __builtin_unreachable();
#endif
      out[j] = std::min(15 - WP2Log2Floor(bits), WP2Ctz(bits)) + 1;
    }
  } else {
    for (uint32_t j = 0; j < size; ++j, q0 += step) {
      const __m128i P0 = _mm_loadu_si128((const __m128i*)(q0 - 8));
      const __m128i Q0 = _mm_loadu_si128((const __m128i*)(q0 + 0));
      const __m128i P1 = _mm_alignr_epi8(Q0, P0, 2);
      const __m128i Q1 = _mm_alignr_epi8(Q0, P0, 14);
      const __m128i diff_dP = _mm_subs_epi16(P1, P0);
      const __m128i diff_dQ = _mm_subs_epi16(Q1, Q0);
      const __m128i diff_dP1 = _mm_slli_si128(diff_dP, 2);
      const __m128i diff_dQ1 = _mm_srli_si128(diff_dQ, 2);
      const __m128i abs_dP = _mm_abs_epi16(_mm_subs_epi16(diff_dP, diff_dP1));
      const __m128i abs_dQ = _mm_abs_epi16(_mm_subs_epi16(diff_dQ, diff_dQ1));
      const __m128i abs_dPQ = _mm_packs_epi16(abs_dQ, abs_dP);
      const __m128i bits_PQ = _mm_cmpgt_epi8(abs_dPQ, thresh);
      const uint32_t bits = _mm_movemask_epi8(bits_PQ) | or_mask;
#if defined(__GNUC__)
      if (!bits) __builtin_unreachable();
#endif
      out[j] = std::min(15 - WP2Log2Floor(bits), WP2Ctz(bits)) + 1;
    }
  }
}

WP2_TSAN_IGNORE_FUNCTION void DblkFilterInitSSE() {
  DeblockLine = DeblockLine_SSE;

  FilterCopyIn = CopyIn_SSE;
  FilterCopyOut = CopyOut_SSE;

  MeasureFlatLengths = MeasureFlatLengths_SSE;
}

#if defined(WP2_USE_AVX2)
void DeblockLine_AVX2(uint32_t filter_strength, int32_t threshold,
                     uint8_t half, int32_t min, int32_t max, int16_t* q0) {
  if (half <= 4) {
    DeblockLine_SSE(filter_strength, threshold, half, min, max, q0);
    return;
  }
  assert(filter_strength > 0 && half > 0);
  int32_t delta = EdgeDelta(q0, half);
  if (std::abs(delta) > threshold) return;
  // Do not filter invisible block edges, consider them already deblocked.
  delta *= (int32_t)filter_strength;
  if (std::abs(delta) <= kBlurWeightsThreshold[half - 1]) return;
  assert(std::abs(delta) < 0x7fff);
  const __m256i m_min = _mm256_set1_epi16(min);
  const __m256i m_max = _mm256_set1_epi16(max);
  const int16_t* const weights = kBlurWeights_SSE[half - 1];
  const __m256i m_delta = _mm256_set1_epi16(delta);
  const __m256i a = _mm256_lddqu_si256((const __m256i*)weights);
  const __m256i b = _mm256_mulhrs_epi16(a, m_delta);
  const __m256i c = _mm256_lddqu_si256((const __m256i*)(q0 - 8));
  const __m256i d = _mm256_adds_epi16(b, c);
  const __m256i e = _mm256_min_epi16(d, m_max);
  const __m256i f = _mm256_max_epi16(e, m_min);
  _mm256_storeu_si256((__m256i*)(q0 - 8), f);
}

WP2_TSAN_IGNORE_FUNCTION void DblkFilterInitAVX2() {
  DeblockLine = DeblockLine_AVX2;
}
#endif

}  // namespace

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------

int32_t DeblockThresholdFromSharpness(uint32_t filter_sharpness,
                                      uint32_t num_precision_bits) {
  // This threshold is only used after calling EdgeDelta() which uses one extra
  // bit of precision to avoid a ">>1". Hence the "2 * " here.
  return ChangePrecision(kSharpnessThreshold[filter_sharpness], 10u,
                         num_precision_bits);
}

void (*DeblockLine)(uint32_t filter_strength, int32_t threshold, uint8_t half,
                    int32_t min, int32_t max, int16_t* q0) = nullptr;
uint8_t (*WouldDeblockLine)(int32_t threshold, uint32_t half,
                            const int16_t q0[]) = nullptr;


void (*MeasureFlatLengths)(uint32_t filter_sharpness, uint32_t half,
                           const int16_t* q0, uint32_t step,
                           uint8_t out[], uint32_t size) = nullptr;

void (*FilterCopyIn)(const int16_t* src, uint32_t src_step,
                     int16_t* dst, int32_t W, int32_t H) = nullptr;
void (*FilterCopyOut)(const int16_t* src, int16_t* dst, uint32_t dst_step,
                      int32_t W, int32_t H) = nullptr;

static volatile WP2CPUInfo dblk_filter_last_cpuinfo_used =
    (WP2CPUInfo)&dblk_filter_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DblkFilterInit() {
  if (dblk_filter_last_cpuinfo_used == WP2GetCPUInfo) return;

  DeblockLine = DeblockLine_C;
  WouldDeblockLine = WouldDeblockLine_C;
  MeasureFlatLengths = MeasureFlatLengths_C;
  FilterCopyIn = CopyIn_C;
  FilterCopyOut = CopyOut_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) DblkFilterInitSSE();
#if defined(WP2_USE_AVX2)
    if (WP2GetCPUInfo(kAVX2)) DblkFilterInitAVX2();
#endif  // AVX2
#endif  // SSE
  }

  dblk_filter_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------

}  // namespace WP2
