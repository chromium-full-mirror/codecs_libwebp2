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
// Image transforms and color space conversion methods common to
// decoder/encoder.
//
// Authors: Vincent Rabaud (vrabaud@google.com)

#ifndef WP2_DSP_LOSSLESS_DSPL_H_
#define WP2_DSP_LOSSLESS_DSPL_H_

#include "src/dsp/dsp.h"

namespace WP2L {

static constexpr uint32_t kNumPredictors = 15;

typedef struct {
  int16_t green_to_red;
  int16_t green_to_blue;
  int16_t red_to_blue;
} Multipliers;

//------------------------------------------------------------------------------
// Misc methods.

// Transforms a signed value into an unsigned index.
static inline uint16_t MakeIndex(int16_t val) {
  // TODO(vrabaud) remove this wrap around.
  return (val & 0xff);
}

// Computes sampled size of 'size' when sampling using 'sampling bits'.
// Equivalent to: ceil(size / 2^sampling_bits)
static inline uint32_t SubSampleSize(uint32_t size, uint32_t sampling_bits) {
  return (size + (1 << sampling_bits) - 1) >> sampling_bits;
}

//------------------------------------------------------------------------------
// Predictors

// These Add/Sub function expects upper[-1] and out[-1] to be readable.
typedef void (*PredictorAddSubFunc)(const int16_t* in, bool has_alpha,
                                    const int16_t* upper, uint32_t num_pixels,
                                    int16_t min_value, int16_t max_value,
                                    int16_t mask, int16_t* out);

typedef void (*PredictorFunc)(const int16_t* left, const int16_t* top,
                              int16_t min_value, int16_t max_value,
                              int16_t* out);
extern PredictorFunc Predictors[kNumPredictors];
extern PredictorFunc Predictors_C[kNumPredictors];

#define COPY_PREDICTOR_ARRAY(IN, OUT) \
  do {                                \
    (OUT)[0] = IN##0_C;               \
    (OUT)[1] = IN##1_C;               \
    (OUT)[2] = IN##2_C;               \
    (OUT)[3] = IN##3_C;               \
    (OUT)[4] = IN##4_C;               \
    (OUT)[5] = IN##5_C;               \
    (OUT)[6] = IN##6_C;               \
    (OUT)[7] = IN##7_C;               \
    (OUT)[8] = IN##8_C;               \
    (OUT)[9] = IN##9_C;               \
    (OUT)[10] = IN##10_C;             \
    (OUT)[11] = IN##11_C;             \
    (OUT)[12] = IN##12_C;             \
    (OUT)[13] = IN##13_C;             \
    (OUT)[14] = IN##14_C;             \
  } while (0);

// Must be called before calling any of the above methods.
void DspInit();

}  // namespace WP2L

#endif  // WP2_DSP_LOSSLESS_DSPL_H_
