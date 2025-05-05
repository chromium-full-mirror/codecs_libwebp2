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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/dsp/dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/math.h"

namespace {

//------------------------------------------------------------------------------
// Misc

void FindExtrema_C(const int16_t* in, uint32_t width, uint32_t height,
                   bool has_alpha, std::array<int16_t, 4>& min,
                   std::array<int16_t, 4>& max) {
  for (int c = has_alpha ? 0 : 1; c < 4; c++) {
    min[c] = max[c] = in[c];
  }
  // Figure out the min and max values.
  const size_t num_pixels = static_cast<size_t>(width) * height;
  for (size_t i = 1; i < num_pixels; ++i) {
    for (int c = has_alpha ? 0 : 1; c < 4; c++) {
      if (in[4 * i + c] < min[c]) {
        min[c] = in[4 * i + c];
      } else if (in[4 * i + c] > max[c]) {
        max[c] = in[4 * i + c];
      }
    }
  }
}

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
}  // namespace

namespace WP2L {
void SubtractGreenFromBlueAndRed_C(int16_t* argb, uint32_t num_pixels) {
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const int16_t green = argb[4 * i + 2];
    argb[4 * i + 1] -= green;
    argb[4 * i + 3] -= green;
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
}  // namespace WP2L

namespace {
inline int16_t TransformColorRed(int16_t green_to_red,
                                 const int16_t* const argb) {
  const int16_t green = argb[2];
  int16_t new_red = argb[1];
  new_red -= WP2L::ColorTransformDelta(green_to_red * green);
  return new_red;
}

inline int16_t TransformColorBlue(int16_t green_to_blue, int16_t red_to_blue,
                                  const int16_t* const argb) {
  const int16_t green = argb[2];
  const int16_t red = argb[1];
  int16_t new_blue = argb[3];
  new_blue -=
      WP2L::ColorTransformDelta(green_to_blue * green + red_to_blue * red);
  return new_blue;
}
}  // namespace

namespace WP2L {
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
}  // namespace WP2L

//------------------------------------------------------------------------------

namespace {

//------------------------------------------------------------------------------

void BufferAdd_C(const uint32_t* const a, const uint32_t* const b,
                 uint32_t size, uint32_t* const out, uint32_t* const nonzeros) {
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

// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// It subtracts the prediction from the input pixel and stores the residual
// in the output pixel.
#define GENERATE_PREDICTOR_NON_CLAMPED_SUB(I)                                \
  void PredictorNonClampedSub##I##_C(const int16_t* in, bool has_alpha,      \
                                     const int16_t* upper,                   \
                                     uint32_t num_pixels, int16_t* out) {    \
    int16_t pred[4];                                                         \
    assert(in != nullptr);                                                   \
    if (I != 0) assert(upper != nullptr);                                    \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         in += 4, out += 4) {                                                \
      WP2L::PredictorsNonClamped_C[I](in - 4, upper, pred);                  \
      WP2L::SubPixels(in, has_alpha, pred, out);                             \
      if (I != 0) upper += 4;                                                \
    }                                                                        \
  }

GENERATE_PREDICTOR_NON_CLAMPED_SUB(0)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(1)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(2)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(3)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(4)
GENERATE_PREDICTOR_NON_CLAMPED_SUB(5)

#define GENERATE_PREDICTOR_CLAMPED_SUB(I)                                      \
  void PredictorClampedSub##I##_C(const int16_t* in, bool has_alpha,           \
                                  const int16_t* upper, uint32_t num_pixels,   \
                                  const int32_t min_values[4],                 \
                                  const int32_t max_values[4], int16_t* out) { \
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

template <int MAIN_ANGLE>
void PredictorAngleSub_C(int delta, const int16_t* in, bool has_alpha,
                         const int16_t* upper, uint32_t num_pixels,
                         int16_t* out) {
  int16_t pred[4];
  if (MAIN_ANGLE != 180) assert(upper != nullptr);
  for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end;
       out += 4, in += 4, upper += 4) {
    WP2L::AnglePredict<MAIN_ANGLE>(delta, in - 4, upper, pred);
    WP2L::SubPixels(in, has_alpha, pred, out);
  }
}

}  // namespace

//------------------------------------------------------------------------------

namespace WP2L {

FindExtremaFunc FindExtrema;
SimpleColorSpaceFunc SubtractGreenFromBlueAndRed;
SimpleColorSpaceFunc RGB2YCoCgR;

TransformColorFunc TransformColor;

CollectColorBlueTransformsFunc CollectColorBlueTransforms;
CollectColorRedTransformsFunc CollectColorRedTransforms;

CombinedShannonEntropyFunc CombinedShannonEntropy;

BufferAddFunc BufferAdd;

PredictorNonClampedSubFunc PredictorsNonClampedSub[kNumPredictorsNonClamped];
PredictorNonClampedSubFunc PredictorsNonClampedSub_C[kNumPredictorsNonClamped];
PredictorClampedSubFunc PredictorsClampedSub[kNumPredictorsClamped];
PredictorClampedSubFunc PredictorsClampedSub_C[kNumPredictorsClamped];
std::array<PredictorAngleSubFunc, kNumPredictorsAngle> PredictorsAngleSub;
std::array<PredictorAngleSubFunc, kNumPredictorsAngle> PredictorsAngleSub_C;

PredictorNonClampedSubFunc PredictorSubBlack;
PredictorAngleSubFunc PredictorSubTop;
PredictorAngleSubFunc PredictorSubLeft;

static volatile WP2CPUInfo lossless_enc_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_enc_last_cpuinfo_used;

extern void EncLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void EncLDspInit() {
  if (lossless_enc_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  FindExtrema = FindExtrema_C;
  SubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_C;
  RGB2YCoCgR = RGB2YCoCgR_C;

  TransformColor = TransformColor_C;

  CollectColorBlueTransforms = CollectColorBlueTransforms_C;
  CollectColorRedTransforms = CollectColorRedTransforms_C;

  CombinedShannonEntropy = CombinedShannonEntropy_C;

  BufferAdd = BufferAdd_C;

  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Sub, Sub);
  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Sub, Sub_C);
  COPY_PREDICTOR_CLAMPED_ARRAY(Sub, Sub);
  COPY_PREDICTOR_CLAMPED_ARRAY(Sub, Sub_C);
  PredictorsAngleSub[0] = PredictorAngleSub_C<45>;
  PredictorsAngleSub[1] = PredictorAngleSub_C<67>;
  PredictorsAngleSub[2] = PredictorAngleSub_C<90>;
  PredictorsAngleSub[3] = PredictorAngleSub_C<113>;
  PredictorsAngleSub[4] = PredictorAngleSub_C<135>;
  PredictorsAngleSub[5] = PredictorAngleSub_C<157>;
  PredictorsAngleSub[6] = PredictorAngleSub_C<180>;
  std::copy(PredictorsAngleSub.begin(), PredictorsAngleSub.end(),
            PredictorsAngleSub_C.begin());

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) EncLDspInitSSE();
#endif

    PredictorSubBlack = PredictorsNonClampedSub[0];
    PredictorSubTop = PredictorsAngleSub[kPredictorAngleTopIndex];
    PredictorSubLeft = PredictorsAngleSub[kPredictorAngleLeftIndex];
  }
  lossless_enc_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2L
