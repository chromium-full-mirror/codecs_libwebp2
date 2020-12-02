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

inline void Average2(const uint16_t* const a0, const uint16_t* const a1,
                     uint16_t mask, uint16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) out[i] = ((a0[i] + a1[i]) >> 1) & mask;
}

inline void Average3(const uint16_t* const a0, const uint16_t* const a1,
                     const uint16_t* const a2, uint16_t mask,
                     uint16_t* const out) {
  Average2(a0, a2, mask, out);
  Average2(out, a1, mask, out);
}

inline void Average4(const uint16_t* const a0, const uint16_t* const a1,
                     const uint16_t* const a2, const uint16_t* const a3,
                     uint16_t mask, uint16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) {
    const uint16_t tmp0 = ((a0[i] + a1[i]) >> 1) & mask;
    const uint16_t tmp1 = ((a2[i] + a3[i]) >> 1) & mask;
    out[i] = ((tmp0 + tmp1) >> 1) & mask;
  }
}

inline int AddSubtractComponentFull(int a, int b, int c, int max_value) {
  return WP2::Clamp(a + b - c, 0, max_value);
}

// Same as above but clamped between a and b.
inline int ClampedAddSubtractComponentFull(int a, int b, int c) {
  const int min = std::min(a, b);
  const int max = std::max(a, b);
  return WP2::Clamp(a + b - c, min, max);
}

inline void AddSubtractFull(const uint16_t* const c0, const uint16_t* const c1,
                            const uint16_t* const c2, uint16_t mask,
                            uint16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) {
    out[i] = AddSubtractComponentFull(c0[i], c1[i], c2[i], mask) & mask;
  }
}

inline void ClampedAddSubtractFull(const uint16_t* const c0,
                                   const uint16_t* const c1,
                                   const uint16_t* const c2, uint16_t mask,
                                   uint16_t* const out) {
  for (uint32_t i = 0; i < 4; ++i) {
    out[i] = ClampedAddSubtractComponentFull(c0[i], c1[i], c2[i]);
  }
}

inline int AddSubtractComponentHalf(int a, int b, int max_value) {
  return WP2::Clamp(a + (a - b) / 2, 0, max_value);
}

inline void AddSubtractHalf(const uint16_t* const c0, const uint16_t* const c1,
                            const uint16_t* const c2, uint16_t mask,
                            uint16_t* const out) {
  Average2(c0, c1, mask, out);
  for (uint32_t i = 0; i < 4; ++i) {
    out[i] = AddSubtractComponentHalf(out[i], c2[i], mask) & mask;
  }
}

// gcc-4.9 on ARM generates incorrect code in Select() when Sub3() is inlined.
#if defined(__arm__) && \
    (LOCAL_GCC_VERSION == 0x409 || LOCAL_GCC_VERSION == 0x408)
#define LOCAL_INLINE __attribute__((noinline))
#else
#define LOCAL_INLINE inline
#endif

LOCAL_INLINE int Sub3(int a, int b, int c) {
  const int pb = b - c;
  const int pa = a - c;
  return std::abs(pb) - std::abs(pa);
}

#undef LOCAL_INLINE

inline void Select(const uint16_t* const a, const uint16_t* const b,
                   const uint16_t* const c, uint16_t* const out) {
  int pa_minus_pb = 0;
  for (uint32_t i = 0; i < 4; ++i) pa_minus_pb += Sub3(a[i], b[i], c[i]);
  if (pa_minus_pb <= 0) {
    for (uint32_t i = 0; i < 4; ++i) out[i] = a[i];
  } else {
    for (uint32_t i = 0; i < 4; ++i) out[i] = b[i];
  }
}

void Predictor0_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)top;
  (void)left;
  out[0] = max_value;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
}
void Predictor1_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)top;
  (void)max_value;
  out[0] = left[0];
  out[1] = left[1];
  out[2] = left[2];
  out[3] = left[3];
}
void Predictor2_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)left;
  (void)max_value;
  out[0] = top[0];
  out[1] = top[1];
  out[2] = top[2];
  out[3] = top[3];
}
void Predictor3_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)left;
  (void)max_value;
  out[0] = top[4];
  out[1] = top[5];
  out[2] = top[6];
  out[3] = top[7];
}
void Predictor4_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)left;
  (void)max_value;
  out[0] = top[-4];
  out[1] = top[-3];
  out[2] = top[-2];
  out[3] = top[-1];
}
void Predictor5_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  Average3(left, top, top + 4, max_value, out);
}
void Predictor6_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  Average2(left, top - 4, max_value, out);
}
void Predictor7_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  Average2(left, top, max_value, out);
}
void Predictor8_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)left;
  Average2(top - 4, top, max_value, out);
}
void Predictor9_C(const uint16_t* left, const uint16_t* top, uint16_t max_value,
                  uint16_t* out) {
  (void)left;
  Average2(top, top + 4, max_value, out);
}
void Predictor10_C(const uint16_t* left, const uint16_t* top,
                   uint16_t max_value, uint16_t* out) {
  Average4(left, top - 4, top, top + 4, max_value, out);
}
void Predictor11_C(const uint16_t* left, const uint16_t* top,
                   uint16_t max_value, uint16_t* out) {
  (void)left;
  Select(top, left, top - 4, out);
}
void Predictor12_C(const uint16_t* left, const uint16_t* top,
                   uint16_t max_value, uint16_t* out) {
  AddSubtractFull(left, top, top - 4, max_value, out);
}
void Predictor13_C(const uint16_t* left, const uint16_t* top,
                   uint16_t max_value, uint16_t* out) {
  AddSubtractHalf(left, top, top - 4, max_value, out);
}
void Predictor14_C(const uint16_t* left, const uint16_t* top,
                   uint16_t max_value, uint16_t* out) {
  ClampedAddSubtractFull(left, top, top - 4, max_value, out);
}

}  // namespace

//------------------------------------------------------------------------------

namespace WP2L {

PredictorFunc Predictors[kNumPredictors];

// exposed plain-C implementations
PredictorFunc Predictors_C[kNumPredictors];

static volatile WP2CPUInfo lossless_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DspInit() {
  if (lossless_last_cpuinfo_used == WP2GetCPUInfo) return;

  WP2MathInit();

  COPY_PREDICTOR_ARRAY(Predictor, Predictors)
  COPY_PREDICTOR_ARRAY(Predictor, Predictors_C)

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
  // TODO(skal): SSE2, etc.
  }
  lossless_last_cpuinfo_used = WP2GetCPUInfo;
}
}  // namespace WP2L
