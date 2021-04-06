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

#include "src/dsp/lossless/dspl.h"

#include "src/common/constants.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/dsp/dsp.h"

//------------------------------------------------------------------------------
// Predictors

namespace {

inline void Average2(const int16_t* const a0, const int16_t* const a1,
                     int16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) out[i] = (a0[i] + a1[i]) / 2;
}

inline void Average3(const int16_t* const a0, const int16_t* const a1,
                     const int16_t* const a2, int16_t* const out) {
  Average2(a0, a2, out);
  Average2(out, a1, out);
}

inline void Average4(const int16_t* const a0, const int16_t* const a1,
                     const int16_t* const a2, const int16_t* const a3,
                     int16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) {
    const int16_t tmp0 = (a0[i] + a1[i]) / 2;
    const int16_t tmp1 = (a2[i] + a3[i]) / 2;
    out[i] = (tmp0 + tmp1) / 2;
  }
}

inline int AddSubtractComponentFull(int a, int b, int c, int min_value,
                                    int max_value) {
  return WP2::Clamp(a + b - c, min_value, max_value);
}

// Same as above but clamped between a and b.
inline int ClampedAddSubtractComponentFull(int a, int b, int c) {
  const int min = std::min(a, b);
  const int max = std::max(a, b);
  return WP2::Clamp(a + b - c, min, max);
}

inline void AddSubtractFull(const int16_t* const c0, const int16_t* const c1,
                            const int16_t* const c2,
                            const int16_t min_values[4],
                            const int16_t max_values[4], int16_t* const out) {
  for (uint32_t c = 0; c < 4; ++c) {
    out[c] = AddSubtractComponentFull(c0[c], c1[c], c2[c], min_values[c],
                                      max_values[c]);
  }
}

inline void ClampedAddSubtractFull(const int16_t* const c0,
                                   const int16_t* const c1,
                                   const int16_t* const c2,
                                   int16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) {
    out[i] = ClampedAddSubtractComponentFull(c0[i], c1[i], c2[i]);
  }
}

inline int AddSubtractComponentHalf(int a, int b, int min_value,
                                    int max_value) {
  return WP2::Clamp(a + (a - b) / 2, min_value, max_value);
}

inline void AddSubtractHalf(const int16_t* const c0, const int16_t* const c1,
                            const int16_t* const c2,
                            const int16_t min_values[4],
                            const int16_t max_values[4], int16_t* const out) {
  Average2(c0, c1, out);
  for (uint32_t c = 0; c < 4; ++c) {
    out[c] =
        AddSubtractComponentHalf(out[c], c2[c], min_values[c], max_values[c]);
  }
}

inline void Select(const int16_t* const a, const int16_t* const b,
                   const int16_t* const c, int16_t* const out) {
  int pa_minus_pb = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    pa_minus_pb += std::abs(b[i] - c[i]) - std::abs(a[i] - c[i]);
  }
  if (pa_minus_pb <= 0) {
    for (uint32_t i = 0; i < 4; ++i) out[i] = a[i];
  } else {
    for (uint32_t i = 0; i < 4; ++i) out[i] = b[i];
  }
}

// Predictors of type NonClamped.
void PredictorNonClamped0_C(const int16_t*, const int16_t*, int16_t* out) {
  out[0] = WP2::kAlphaMax;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
}
void PredictorNonClamped1_C(const int16_t* left, const int16_t* top,
                            int16_t* out) {
  Average3(left, top, top + 4, out);
}
void PredictorNonClamped2_C(const int16_t* left, const int16_t* top,
                            int16_t* out) {
  Average2(left, top, out);
}
void PredictorNonClamped3_C(const int16_t* left, const int16_t* top,
                            int16_t* out) {
  Average4(left, top - 4, top, top + 4, out);
}
void PredictorNonClamped4_C(const int16_t* left, const int16_t* top,
                            int16_t* out) {
  Select(top, left, top - 4, out);
}
void PredictorNonClamped5_C(const int16_t* left, const int16_t* top,
                            int16_t* out) {
  ClampedAddSubtractFull(left, top, top - 4, out);
}
// Predictors of type Clamped.
void PredictorClamped0_C(const int16_t* left, const int16_t* top,
                         const int16_t min_values[4],
                         const int16_t max_values[4], int16_t* out) {
  AddSubtractFull(left, top, top - 4, min_values, max_values, out);
}
void PredictorClamped1_C(const int16_t* left, const int16_t* top,
                         const int16_t min_values[4],
                         const int16_t max_values[4], int16_t* out) {
  AddSubtractHalf(left, top, top - 4, min_values, max_values, out);
}

}  // namespace

// Angle predictors.
namespace WP2L {
template <>
void AnglePredict<45>(const int16_t* left, const int16_t* top, int16_t* out) {
  out[0] = top[4];
  out[1] = top[5];
  out[2] = top[6];
  out[3] = top[7];
}
template <>
void AnglePredict<67>(const int16_t* left, const int16_t* top, int16_t* out) {
  Average2(top, top + 4, out);
}
template <>
void AnglePredict<90>(const int16_t* left, const int16_t* top, int16_t* out) {
  out[0] = top[0];
  out[1] = top[1];
  out[2] = top[2];
  out[3] = top[3];
}
template <>
void AnglePredict<113>(const int16_t* left, const int16_t* top, int16_t* out) {
  Average2(top - 4, top, out);
}
template <>
void AnglePredict<135>(const int16_t* left, const int16_t* top, int16_t* out) {
  out[0] = top[-4];
  out[1] = top[-3];
  out[2] = top[-2];
  out[3] = top[-1];
}
template <>
void AnglePredict<157>(const int16_t* left, const int16_t* top, int16_t* out) {
  Average2(left, top - 4, out);
}
template <>
void AnglePredict<180>(const int16_t* left, const int16_t* top, int16_t* out) {
  out[0] = left[0];
  out[1] = left[1];
  out[2] = left[2];
  out[3] = left[3];
}

}  // namespace WP2L

//------------------------------------------------------------------------------

namespace WP2L {

PredictorNonClampedFunc PredictorsNonClamped[kNumPredictorsNonClamped];
PredictorClampedFunc PredictorsClamped[kNumPredictorsClamped];
PredictorAngleFunc PredictorsAngle[kNumPredictorsAngle];

// exposed plain-C implementations
PredictorNonClampedFunc PredictorsNonClamped_C[kNumPredictorsNonClamped];
PredictorClampedFunc PredictorsClamped_C[kNumPredictorsClamped];

static volatile WP2CPUInfo lossless_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DspInit() {
  if (lossless_last_cpuinfo_used == WP2GetCPUInfo) return;

  WP2MathInit();

  COPY_PREDICTOR_NON_CLAMPED_ARRAY(, )
  COPY_PREDICTOR_NON_CLAMPED_ARRAY(, _C)
  COPY_PREDICTOR_CLAMPED_ARRAY(, )
  COPY_PREDICTOR_CLAMPED_ARRAY(, _C)
  PredictorsAngle[0] = AnglePredict<45>;
  PredictorsAngle[1] = AnglePredict<67>;
  PredictorsAngle[2] = AnglePredict<90>;
  PredictorsAngle[3] = AnglePredict<113>;
  PredictorsAngle[4] = AnglePredict<135>;
  PredictorsAngle[5] = AnglePredict<157>;
  PredictorsAngle[6] = AnglePredict<180>;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
  // TODO(skal): SSE2, etc.
  }
  lossless_last_cpuinfo_used = WP2GetCPUInfo;
}
}  // namespace WP2L
