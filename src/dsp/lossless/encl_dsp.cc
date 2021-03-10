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

#include "src/dsp/lossless/encl_dsp.h"

#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/dsp/dsp_x86.h"
#include "src/dsp/math.h"

namespace {

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

// Compute the combined Shanon's entropy for distribution {X} and {X+Y}
float CombinedShannonEntropy_C(const uint32_t* X, const uint32_t* Y,
                               uint32_t length) {
  float retval = 0.f;
  // This critical function is extremely memory-bond! Any change in code should
  // be tested on various compilers.
  for (uint32_t i = 0; i < length; ++i) {
    const uint32_t x = X[i];
    if (x != 0) {
      retval += WP2SLog2m1(x + Y[i]);
      retval += WP2SLog2m1(x);
    } else if (Y[i] != 0) {
      retval += WP2SLog2m1(Y[i]);
    }
  }
  return (float)retval;
}

//------------------------------------------------------------------------------
// Color transforms.

void RGB2YCoCgR_C(int16_t* argb_data, uint32_t num_pixels) {
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const int16_t r = argb_data[4 * i + 1];
    const int16_t g = argb_data[4 * i + 2];
    const int16_t b = argb_data[4 * i + 3];
    const int16_t tmp_co = r - b;
    const int16_t tmp = (b + (tmp_co / 2));
    const int16_t tmp_cg = g - tmp;
    const int16_t tmp_y = tmp + (tmp_cg / 2);
    argb_data[4 * i + 1] = tmp_y;
    argb_data[4 * i + 2] = tmp_co;
    argb_data[4 * i + 3] = tmp_cg;
  }
}

void SubtractGreenFromBlueAndRed_C(int16_t* argb_data, uint32_t num_pixels) {
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const int16_t green = argb_data[4 * i + 2];
    argb_data[4 * i + 1] -= green;
    argb_data[4 * i + 3] -= green;
  }
}

void TransformColor_C(const WP2L::Multipliers& m, uint32_t num_pixels,
                      int16_t* const data) {
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const int16_t green = data[4 * i + 2];
    const int16_t red = data[4 * i + 1];
    int new_red = red;
    int new_blue = data[4 * i + 3];
    new_red -= WP2L::ColorTransformDelta(m.green_to_red * green);
    new_blue -= WP2L::ColorTransformDelta(m.green_to_blue * green +
                                          m.red_to_blue * red);
    data[4 * i + 1] = new_red;
    data[4 * i + 3] = new_blue;
  }
}

inline int16_t TransformColorRed(int16_t green_to_red,
                                 const int16_t* const argb) {
  const int16_t green = argb[2];
  int16_t new_red = argb[1];
  new_red -= WP2L::ColorTransformDelta(green_to_red * green);
  return new_red;
}

inline int16_t TransformColorBlue(int16_t green_to_blue,
                                  int16_t red_to_blue,
                                  const int16_t* const argb) {
  const int16_t green = argb[2];
  const int16_t red = argb[1];
  int16_t new_blue = argb[3];
  new_blue -=
      WP2L::ColorTransformDelta(green_to_blue * green + red_to_blue * red);
  return new_blue;
}

bool CollectColorRedTransforms_C(const int16_t* argb, uint32_t width,
                                 uint32_t tile_width, uint32_t tile_height,
                                 int16_t green_to_red, uint16_t abs_value_max,
                                 uint32_t histo_size, uint32_t* const histo) {
  while (tile_height-- > 0) {
    for (uint32_t x = 0; x < tile_width; ++x) {
      const int16_t value = TransformColorRed(green_to_red, &argb[4 * x]);
      if (std::abs(value) > abs_value_max) return false;
      ++histo[WP2L::MakeIndex(value, histo_size - 1)];
    }
    argb += 4 * width;
  }
  return true;
}

bool CollectColorBlueTransforms_C(const int16_t* argb, uint32_t width,
                                  uint32_t tile_width, uint32_t tile_height,
                                  int16_t green_to_blue, int16_t red_to_blue,
                                  uint16_t abs_value_max, uint32_t histo_size,
                                  uint32_t* const histo) {
  while (tile_height-- > 0) {
    for (uint32_t x = 0; x < tile_width; ++x) {
      const int16_t value =
          TransformColorBlue(green_to_blue, red_to_blue, &argb[4 * x]);
      if (std::abs(value) > abs_value_max) return false;
      ++histo[WP2L::MakeIndex(value, histo_size - 1)];
    }
    argb += 4 * width;
  }
  return true;
}

//------------------------------------------------------------------------------

float ExtraCost_C(const uint32_t* population, int length) {
  int i;
  float cost = 0.f;
  for (i = 2; i < length - 2; ++i) cost += (i >> 1) * population[i + 2];
  return cost;
}

float ExtraCostCombined_C(const uint32_t* X, const uint32_t* Y, int length) {
  int i;
  float cost = 0.f;
  for (i = 2; i < length - 2; ++i) {
    const int xy = X[i + 2] + Y[i + 2];
    cost += (i >> 1) * xy;
  }
  return cost;
}

//------------------------------------------------------------------------------

void BufferAdd_C(const uint32_t* const a, const uint32_t* const b,
                 uint32_t size, uint32_t* const out,
                 uint32_t* const nonzeros) {
  assert(out != nullptr);
  if (nonzeros != nullptr) *nonzeros = 0;
  if (b != out) {
    for (uint32_t i = 0; i < size; ++i) {
      out[i] = a[i] + b[i];
      if (nonzeros != nullptr && out[i]) ++*nonzeros;
    }
  } else {
    for (uint32_t i = 0; i < size; ++i) {
      out[i] += a[i];
      if (nonzeros != nullptr && out[i]) ++*nonzeros;
    }
  }
}

//------------------------------------------------------------------------------
// Predictors

void PredictorNonClampedSub0_C(const int16_t* in, bool has_alpha,
                               const int16_t*, uint32_t num_pixels,
                               int16_t* out) {
  for (const int16_t* const in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    out[0] = has_alpha ? in[0] - WP2::kAlphaMax : WP2::kAlphaMax;
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
  }
}

void PredictorNonClampedSub1_C(const int16_t* in, bool has_alpha,
                               const int16_t*, uint32_t num_pixels,
                               int16_t* out) {
  for (const int16_t* const in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    WP2L::SubPixels(in, has_alpha, &in[-4], out);
  }
}

// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// It subtracts the prediction from the input pixel and stores the residual
// in the output pixel.
#define GENERATE_PREDICTOR_NON_CLAMPED_SUB(I)                                \
  void PredictorNonClampedSub##I##_C(const int16_t* in, bool has_alpha,      \
                                     const int16_t* upper,                   \
                                     uint32_t num_pixels, int16_t* out) {    \
    int16_t pred[4];                                                         \
    assert(upper != nullptr);                                                \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         out += 4, in += 4, upper += 4) {                                    \
      WP2L::PredictorsNonClamped_C[I](in - 4, upper, pred);                  \
      WP2L::SubPixels(in, has_alpha, pred, out);                             \
    }                                                                        \
  }

GENERATE_PREDICTOR_NON_CLAMPED_SUB(2)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(3)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(4)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(5)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(6)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(7)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(8)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(9)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(10)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(11)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(12)

#define GENERATE_PREDICTOR_CLAMPED_SUB(I)                                      \
  void PredictorClampedSub##I##_C(const int16_t* in, bool has_alpha,           \
                                  const int16_t* upper, uint32_t num_pixels,   \
                                  const int16_t min_values[4],                 \
                                  const int16_t max_values[4], int16_t* out) { \
    int16_t pred[4];                                                           \
    assert(upper != nullptr);                                                  \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end;   \
         out += 4, in += 4, upper += 4) {                                      \
      WP2L::PredictorsClamped_C[I](in - 4, upper, min_values, max_values,      \
                                   pred);                                      \
      WP2L::SubPixels(in, has_alpha, pred, out);                               \
    }                                                                          \
  }

GENERATE_PREDICTOR_CLAMPED_SUB(0)
GENERATE_PREDICTOR_CLAMPED_SUB(1)

//------------------------------------------------------------------------------
// SSE version

#if defined(WP2_USE_SSE)

#define CSE_CHECK_ZERO(m, j) do {               \
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
    const __m128i x0 = _mm_loadu_si128((const __m128i*)(X + i +  0));
    const __m128i y0 = _mm_loadu_si128((const __m128i*)(Y + i +  0));
    const __m128i x1 = _mm_loadu_si128((const __m128i*)(X + i +  4));
    const __m128i y1 = _mm_loadu_si128((const __m128i*)(Y + i +  4));
    const __m128i x2 = _mm_loadu_si128((const __m128i*)(X + i +  8));
    const __m128i y2 = _mm_loadu_si128((const __m128i*)(Y + i +  8));
    const __m128i x3 = _mm_loadu_si128((const __m128i*)(X + i + 12));
    const __m128i y3 = _mm_loadu_si128((const __m128i*)(Y + i + 12));
    const __m128i x4 = _mm_packs_epi16(_mm_packs_epi32(x0, x1),
                                       _mm_packs_epi32(x2, x3));
    const __m128i y4 = _mm_packs_epi16(_mm_packs_epi32(y0, y1),
                                       _mm_packs_epi32(y2, y3));
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

inline bool DoCollectTransforms_SSE(
    const int16_t* src, const __m128i mult, const __m128i vmaxval,
    const __m128i vmaxidx, uint32_t* const histo) {
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

bool CollectColorRedTransforms_SSE(const int16_t* argb, uint32_t width,
                                   uint32_t tile_width, uint32_t tile_height,
                                   int16_t green_to_red, uint16_t abs_value_max,
                                   uint32_t histo_size, uint32_t* const histo) {
  __m128i mult = _mm_cvtsi32_si128(32 << 16);
  mult = _mm_insert_epi32(mult, (uint16_t)(-green_to_red), 1);
  mult = _mm_unpacklo_epi64(mult, mult);
  const __m128i vmaxval = _mm_set1_epi32(abs_value_max);
  const __m128i vmaxidx = _mm_set1_epi32(histo_size - 1);

  for (uint32_t i = 0; i < tile_height; ++i, argb += 4 * width) {
    uint32_t x = 0;
    for (; x + 4 <= tile_width; x += 4) {
      if (!DoCollectTransforms_SSE(argb + 4 * x, mult,
                                   vmaxval, vmaxidx, histo)) return false;
    }
    for (; x < tile_width; ++x) {
      const int16_t value = TransformColorRed(green_to_red, &argb[4 * x]);
      if (std::abs(value) > abs_value_max) return false;
      ++histo[WP2L::MakeIndex(value, histo_size - 1)];
    }
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
      if (!DoCollectTransforms_SSE(argb + 4 * x, mult,
                                   vmaxval, vmaxidx, histo)) return false;
    }
    for (; x < tile_width; ++x) {
      const int32_t value = TransformColorBlue(
          green_to_blue, red_to_blue, &argb[4 * x]);
      if (std::abs(value) > abs_value_max) return false;
      ++histo[WP2L::MakeIndex(value, histo_size - 1)];
    }
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

WP2_TSAN_IGNORE_FUNCTION void EncLDspInitSSE() {
  WP2L::CombinedShannonEntropy = CombinedShannonEntropy_SSE;
  WP2L::CollectColorRedTransforms = CollectColorRedTransforms_SSE;
  WP2L::CollectColorBlueTransforms = CollectColorBlueTransforms_SSE;
  WP2L::BufferAdd = BufferAdd_SSE;
}

#endif  // WP2_USE_SSE

}    // namespace

//------------------------------------------------------------------------------

namespace WP2L {

SimpleColorSpaceFunc SubtractGreenFromBlueAndRed;
SimpleColorSpaceFunc RGB2YCoCgR;

TransformColorFunc TransformColor;

CollectColorBlueTransformsFunc CollectColorBlueTransforms;
CollectColorRedTransformsFunc CollectColorRedTransforms;

CostFunc ExtraCost;
CostCombinedFunc ExtraCostCombined;
CombinedShannonEntropyFunc CombinedShannonEntropy;

BufferAddFunc BufferAdd;

PredictorNonClampedAddSubFunc PredictorsNonClampedSub[kNumPredictorsNonClamped];
PredictorNonClampedAddSubFunc
    PredictorsNonClampedSub_C[kNumPredictorsNonClamped];
PredictorClampedAddSubFunc PredictorsClampedSub[kNumPredictorsClamped];
PredictorClampedAddSubFunc PredictorsClampedSub_C[kNumPredictorsClamped];

static volatile WP2CPUInfo lossless_enc_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_enc_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void EncLDspInit() {
  if (lossless_enc_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  SubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_C;
  RGB2YCoCgR = RGB2YCoCgR_C;

  TransformColor = TransformColor_C;

  CollectColorBlueTransforms = CollectColorBlueTransforms_C;
  CollectColorRedTransforms = CollectColorRedTransforms_C;

  ExtraCost = ExtraCost_C;
  ExtraCostCombined = ExtraCostCombined_C;
  CombinedShannonEntropy = CombinedShannonEntropy_C;

  BufferAdd = BufferAdd_C;

  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Sub, Sub);
  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Sub, Sub_C);
  COPY_PREDICTOR_CLAMPED_ARRAY(Sub, Sub);
  COPY_PREDICTOR_CLAMPED_ARRAY(Sub, Sub_C);

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) EncLDspInitSSE();
#endif
  }
  lossless_enc_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2L
