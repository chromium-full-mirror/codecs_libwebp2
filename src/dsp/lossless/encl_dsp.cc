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
  uint32_t sumX = 0, sumXY = 0;
  for (uint32_t i = 0; i < length; ++i) {
    const uint32_t x = X[i];
    if (x != 0) {
      const uint32_t xy = x + Y[i];
      sumX += x;
      retval -= WP2SLog2(x);
      sumXY += xy;
      retval -= WP2SLog2(xy);
    } else if (Y[i] != 0) {
      sumXY += Y[i];
      retval -= WP2SLog2(Y[i]);
    }
  }
  retval += WP2SLog2(sumX) + WP2SLog2(sumXY);
  return (float)retval;
}

//------------------------------------------------------------------------------
// Color transforms.

void SubtractGreenFromBlueAndRed_C(uint16_t* argb_data, uint32_t channel_bits,
                                   uint32_t num_pixels) {
  uint16_t mask = (1u << channel_bits) - 1;
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const int green = argb_data[4 * i + 2];
    argb_data[4 * i + 1] = (argb_data[4 * i + 1] - green) & mask;
    argb_data[4 * i + 3] = (argb_data[4 * i + 3] - green) & mask;
  }
}

void TransformColor_C(const WP2L::Multipliers* const m, int num_pixels,
                      int16_t* const data) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const int16_t green = data[4 * i + 2];
    const int16_t red = data[4 * i + 1];
    int new_red = red;
    int new_blue = data[4 * i + 3];
    new_red -= WP2L::ColorTransformDelta(m->green_to_red, green);
    new_blue -= WP2L::ColorTransformDelta(m->green_to_blue, green);
    new_blue -= WP2L::ColorTransformDelta(m->red_to_blue, red);
    data[4 * i + 1] = new_red;
    data[4 * i + 3] = new_blue;
  }
}

inline int16_t TransformColorRed(int16_t green_to_red,
                                 const int16_t* const argb) {
  const int16_t green = argb[2];
  int16_t new_red = argb[1];
  new_red -= WP2L::ColorTransformDelta(green_to_red, green);
  return new_red;
}

inline int16_t TransformColorBlue(int16_t green_to_blue,
                                  int16_t red_to_blue,
                                  const int16_t* const argb) {
  const int16_t green = argb[2];
  const int16_t red = argb[1];
  int16_t new_blue = argb[3];
  new_blue -= WP2L::ColorTransformDelta(green_to_blue, green);
  new_blue -= WP2L::ColorTransformDelta(red_to_blue, red);
  return new_blue;
}

void CollectColorRedTransforms_C(const int16_t* argb, uint32_t width,
                                 uint32_t tile_width, uint32_t tile_height,
                                 int16_t green_to_red, uint32_t* const histo) {
  while (tile_height-- > 0) {
    for (uint32_t x = 0; x < tile_width; ++x) {
      ++histo[WP2L::MakeIndex(TransformColorRed(green_to_red, &argb[4 * x]))];
    }
    argb += 4 * width;
  }
}

void CollectColorBlueTransforms_C(const int16_t* argb, uint32_t width,
                                  uint32_t tile_width, uint32_t tile_height,
                                  int16_t green_to_blue, int16_t red_to_blue,
                                  uint32_t* const histo) {
  while (tile_height-- > 0) {
    for (uint32_t x = 0; x < tile_width; ++x) {
      ++histo[WP2L::MakeIndex(
          TransformColorBlue(green_to_blue, red_to_blue, &argb[4 * x]))];
    }
    argb += 4 * width;
  }
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

void PredictorSub0_C(const int16_t* in, bool has_alpha, const int16_t*,
                     uint32_t num_pixels, int16_t, int16_t, int16_t mask,
                     int16_t* out) {
  for (const int16_t* const in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    out[0] = has_alpha ? ((in[0] - WP2::kAlphaMax) & mask) : WP2::kAlphaMax;
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
  }
}

void PredictorSub1_C(const int16_t* in, bool has_alpha, const int16_t*,
                     uint32_t num_pixels, int16_t, int16_t, int16_t mask,
                     int16_t* out) {
  for (const int16_t* const in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    WP2L::SubPixels(in, has_alpha, &in[-4], mask, out);
  }
}

// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// It subtracts the prediction from the input pixel and stores the residual
// in the output pixel.
#define GENERATE_PREDICTOR_SUB(PREDICTOR, PREDICTOR_SUB)                     \
  void PREDICTOR_SUB(const int16_t* in, bool has_alpha,                      \
                     const int16_t* upper, uint32_t num_pixels,              \
                     int16_t min_value, int16_t max_value,                   \
                     int16_t mask, int16_t* out) {                           \
    int16_t pred[4];                                                         \
    assert(upper != nullptr);                                                \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         out += 4, in += 4, upper += 4) {                                    \
      (WP2L::PREDICTOR)(in - 4, upper, min_value, max_value, pred);          \
      WP2L::SubPixels(in, has_alpha, pred, mask, out);                       \
    }                                                                        \
  }

GENERATE_PREDICTOR_SUB(Predictors_C[2], PredictorSub2_C)
GENERATE_PREDICTOR_SUB(Predictors_C[3], PredictorSub3_C)
GENERATE_PREDICTOR_SUB(Predictors_C[4], PredictorSub4_C)
GENERATE_PREDICTOR_SUB(Predictors_C[5], PredictorSub5_C)
GENERATE_PREDICTOR_SUB(Predictors_C[6], PredictorSub6_C)
GENERATE_PREDICTOR_SUB(Predictors_C[7], PredictorSub7_C)
GENERATE_PREDICTOR_SUB(Predictors_C[8], PredictorSub8_C)
GENERATE_PREDICTOR_SUB(Predictors_C[9], PredictorSub9_C)
GENERATE_PREDICTOR_SUB(Predictors_C[10], PredictorSub10_C)
GENERATE_PREDICTOR_SUB(Predictors_C[11], PredictorSub11_C)
GENERATE_PREDICTOR_SUB(Predictors_C[12], PredictorSub12_C)
GENERATE_PREDICTOR_SUB(Predictors_C[13], PredictorSub13_C)
GENERATE_PREDICTOR_SUB(Predictors_C[14], PredictorSub14_C)

//------------------------------------------------------------------------------
// SSE version

#if defined(WP2_USE_SSE)

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
  WP2L::BufferAdd = BufferAdd_SSE;
}

#endif  // WP2_USE_SSE

}    // namespace

//------------------------------------------------------------------------------

namespace WP2L {

ProcessEncBlueAndRedFunc SubtractGreenFromBlueAndRed;

TransformColorFunc TransformColor;

CollectColorBlueTransformsFunc CollectColorBlueTransforms;
CollectColorRedTransformsFunc CollectColorRedTransforms;

CostFunc ExtraCost;
CostCombinedFunc ExtraCostCombined;
CombinedShannonEntropyFunc CombinedShannonEntropy;

BufferAddFunc BufferAdd;

PredictorAddSubFunc PredictorsSub[kNumPredictors];
PredictorAddSubFunc PredictorsSub_C[kNumPredictors];

static volatile WP2CPUInfo lossless_enc_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_enc_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void EncLDspInit() {
  if (lossless_enc_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  SubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_C;

  TransformColor = TransformColor_C;

  CollectColorBlueTransforms = CollectColorBlueTransforms_C;
  CollectColorRedTransforms = CollectColorRedTransforms_C;

  ExtraCost = ExtraCost_C;
  ExtraCostCombined = ExtraCostCombined_C;
  CombinedShannonEntropy = CombinedShannonEntropy_C;

  BufferAdd = BufferAdd_C;

  COPY_PREDICTOR_ARRAY(PredictorSub, PredictorsSub);
  COPY_PREDICTOR_ARRAY(PredictorSub, PredictorsSub_C);

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) EncLDspInitSSE();
#endif
  }
  lossless_enc_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2L
