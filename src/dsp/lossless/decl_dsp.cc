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

#include "src/dsp/lossless/decl_dsp.h"

#include "src/common/constants.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/dsp/dsp.h"

//------------------------------------------------------------------------------
// Predictors

typedef void (*PredictoAddFunc)(const uint16_t* in, const uint16_t* upper,
                                uint32_t num_pixels, uint32_t num_bits,
                                uint16_t* out);

// Sum of each component, mod 256.
static WP2_UBSAN_IGNORE_UNSIGNED_OVERFLOW inline void AddPixels(
    const uint16_t* const a, bool has_alpha, const uint16_t* const b,
    uint16_t mask, uint16_t* const out) {
  out[0] = has_alpha ? ((a[0] + b[0]) & mask) : WP2::kAlphaMax;
  for (uint32_t i = 1; i < 4; ++i) out[i] = (a[i] + b[i]) & mask;
}

static void PredictorAdd0_C(const uint16_t* in, bool has_alpha, const uint16_t*,
                            uint32_t num_pixels, uint32_t num_bits,
                            uint16_t* out) {
  uint16_t pred[4];
  const uint16_t max_value = (1u << num_bits) - 1;
  WP2L::Predictors_C[0](/*left=*/nullptr, /*top=*/nullptr, max_value, pred);
  for (const uint16_t* in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    AddPixels(in, has_alpha, pred, max_value, out);
  }
}
static void PredictorAdd1_C(const uint16_t* in, bool has_alpha, const uint16_t*,
                            uint32_t num_pixels, uint32_t num_bits,
                            uint16_t* out) {
  const uint16_t max_value = (1u << num_bits) - 1;
  for (const uint16_t* in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    AddPixels(in, has_alpha, &out[-4], max_value, out);
  }
}


// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// The predictor is added to the output pixel (which
// is therefore considered as a residual) to get the final prediction.
#define GENERATE_PREDICTOR_ADD(PREDICTOR, PREDICTOR_ADD)                      \
  static void PREDICTOR_ADD(const uint16_t* in, bool has_alpha,               \
                            const uint16_t* upper, uint32_t num_pixels,       \
                            uint32_t num_bits, uint16_t* out) {               \
    uint16_t pred[4];                                                         \
    const uint16_t mask = (1u << num_bits) - 1;                               \
    assert(upper != nullptr);                                                 \
    for (const uint16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         out += 4, in += 4, upper += 4) {                                     \
      (PREDICTOR)(&out[-4], upper, mask, pred);                               \
      AddPixels(in, has_alpha, pred, mask, out);                              \
    }                                                                         \
  }

namespace WP2L {

GENERATE_PREDICTOR_ADD(Predictors_C[2], PredictorAdd2_C)
GENERATE_PREDICTOR_ADD(Predictors_C[3], PredictorAdd3_C)
GENERATE_PREDICTOR_ADD(Predictors_C[4], PredictorAdd4_C)
GENERATE_PREDICTOR_ADD(Predictors_C[5], PredictorAdd5_C)
GENERATE_PREDICTOR_ADD(Predictors_C[6], PredictorAdd6_C)
GENERATE_PREDICTOR_ADD(Predictors_C[7], PredictorAdd7_C)
GENERATE_PREDICTOR_ADD(Predictors_C[8], PredictorAdd8_C)
GENERATE_PREDICTOR_ADD(Predictors_C[9], PredictorAdd9_C)
GENERATE_PREDICTOR_ADD(Predictors_C[10], PredictorAdd10_C)
GENERATE_PREDICTOR_ADD(Predictors_C[11], PredictorAdd11_C)
GENERATE_PREDICTOR_ADD(Predictors_C[12], PredictorAdd12_C)
GENERATE_PREDICTOR_ADD(Predictors_C[13], PredictorAdd13_C)
GENERATE_PREDICTOR_ADD(Predictors_C[14], PredictorAdd14_C)

//------------------------------------------------------------------------------
// Inverse transforms.

// Inverse prediction.
static void PredictorInverseTransform_C(const Transform* const transform,
                                        int y_start, int y_end,
                                        uint32_t num_bits, const uint16_t* in,
                                        bool has_alpha, uint16_t* out) {
  const int width = transform->width_;
  if (y_start == 0) {  // First Row follows the L (mode=1) mode.
    PredictorAdd0_C(in, has_alpha, nullptr, 1, num_bits, out);
    PredictorAdd1_C(in + 4, has_alpha, nullptr, width - 1, num_bits, out + 4);
    in += 4 * width;
    out += 4 * width;
    ++y_start;
  }

  {
    int y = y_start;
    const int tile_width = 1 << transform->bits_;
    const int mask = tile_width - 1;
    const int tiles_per_row = SubSampleSize(width, transform->bits_);
    const int16_t* pred_mode_base =
        transform->data_.data() + 4 * (y >> transform->bits_) * tiles_per_row;

    while (y < y_end) {
      const int16_t* pred_mode_src = pred_mode_base;
      int x = 1;
      // First pixel follows the T (mode=2) mode.
      PredictorAdd2_C(in, has_alpha, out - 4 * width, 1, num_bits, out);
      // .. the rest:
      while (x < width) {
        const PredictorAddSubFunc pred_func = PredictorsAdd[pred_mode_src[2]];
        int x_end = (x & ~mask) + tile_width;
        if (x_end > width) x_end = width;
        pred_func(in + 4 * x, has_alpha, out + 4 * x - 4 * width, x_end - x,
                  num_bits, out + 4 * x);
        x = x_end;
        pred_mode_src += 4;
      }
      in += 4 * width;
      out += 4 * width;
      ++y;
      if ((y & mask) == 0) {   // Use the same mask, since tiles are squares.
        pred_mode_base += 4 * tiles_per_row;
      }
    }
  }
}

// Add green to blue and red channels (i.e. perform the inverse transform of
// 'subtract green').
void AddGreenToBlueAndRed_C(const uint16_t* src, uint32_t num_pixels,
                            uint32_t channel_bits, uint16_t* dst) {
  const uint16_t mask = (uint16_t)((1u << channel_bits) - 1);
  for (const uint16_t* src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const uint16_t green = src[2];
    dst[0] = src[0];
    dst[1] = (src[1] + green) & mask;
    dst[2] = src[2];
    dst[3] = (src[3] + green) & mask;
  }
}

static inline int32_t ColorTransformDelta(int16_t color_pred,
                                          uint16_t max_value_half,
                                          uint16_t color) {
  const int color_int =
      (color >= max_value_half) ? (int)color - 2 * max_value_half : (int)color;
  return (((int32_t)color_pred * color_int) >> 5);
}

static inline void ColorCodeToMultipliers(const int16_t* const color_code,
                                          Multipliers* const m) {
  m->green_to_red = color_code[3];
  m->green_to_blue = color_code[2];
  m->red_to_blue = color_code[1];
}

static void TransformColorInverse_C(const Multipliers* const m,
                                    const uint16_t* src, uint32_t num_pixels,
                                    uint32_t num_bits, uint16_t* dst) {
  const uint16_t max_value = (1u << num_bits) - 1;
  const uint16_t max_value_half = (max_value + 1) / 2;
  for (const uint16_t* src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const uint16_t red = src[1];
    const uint16_t green = src[2];
    int new_red = red;
    int new_blue = src[3];
    new_red += ColorTransformDelta(m->green_to_red, max_value_half, green);
    new_red &= max_value;
    new_blue += ColorTransformDelta(m->green_to_blue, max_value_half, green);
    new_blue += ColorTransformDelta(m->red_to_blue, max_value_half, new_red);
    new_blue &= max_value;
    dst[0] = src[0];
    dst[1] = (uint16_t)new_red;
    dst[2] = src[2];
    dst[3] = (uint16_t)new_blue;
  }
}

// Color space inverse transform.
static void ColorSpaceInverseTransform_C(const Transform* const transform,
                                         int y_start, int y_end,
                                         uint32_t num_bits, const uint16_t* src,
                                         uint16_t* dst) {
  const uint32_t width = transform->width_;
  const int tile_width = 1 << transform->bits_;
  const int mask = tile_width - 1;
  const int safe_width = width & ~mask;
  const int remaining_width = width - safe_width;
  const int tiles_per_row = SubSampleSize(width, transform->bits_);
  int y = y_start;
  const int16_t* pred_row =
      transform->data_.data() + 4 * (y >> transform->bits_) * tiles_per_row;

  while (y < y_end) {
    const int16_t* pred = pred_row;
    Multipliers m = { 0, 0, 0 };
    const uint16_t* const src_safe_end = src + 4 * safe_width;
    const uint16_t* const src_end = src + 4 * width;
    while (src < src_safe_end) {
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(&m, src, tile_width, num_bits, dst);
      src += 4 * tile_width;
      dst += 4 * tile_width;
      pred += 4;
    }
    if (src < src_end) {  // Left-overs using C-version.
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(&m, src, remaining_width, num_bits, dst);
      src += 4 * remaining_width;
      dst += 4 * remaining_width;
      pred += 4;
    }
    ++y;
    if ((y & mask) == 0) pred_row += 4 * tiles_per_row;
  }
}

static void MapARGB_C(const uint16_t* src, const int16_t* const color_map,
                      uint32_t color_map_size, uint32_t y_start, uint32_t y_end,
                      uint32_t width, uint16_t* dst) {
  for (const uint16_t* const src_end = src + 4 * (y_end - y_start) * width;
       src < src_end; src += 4, dst += 4) {
    if (src[2] >= color_map_size) assert(false);
    // TODO(vrabaud) use the following once switched to int16_t;
    // ColorCopy(&color_map[4 * src[2]], dst);
    std::copy(&color_map[4 * src[2]], &color_map[4 * src[2]] + 4, dst);
  }
}
static void ColorIndexInverseTransform_C(const Transform* const transform,
                                         uint32_t y_start, uint32_t y_end,
                                         const uint16_t* const src,
                                         uint16_t* const dst) {
  const uint32_t width = transform->width_;
  MapColor(src, /*color_map=*/transform->data_.data(),
           /*color_map_size=*/transform->data_.size() / 4, y_start, y_end,
           width, dst);
}

void InverseTransform(const Transform* const transform, uint32_t row_start,
                      uint32_t row_end, uint32_t channel_bits,
                      const uint16_t* const in, bool has_alpha,
                      uint16_t* const out) {
  const uint32_t width = transform->width_;
  assert(row_start < row_end);
  assert(row_end <= transform->height_);

  switch (transform->type_) {
    case SUBTRACT_GREEN:
      AddGreenToBlueAndRed(in, (row_end - row_start) * width, channel_bits,
                           out);
      break;
    case PREDICTOR_TRANSFORM: {
      PredictorInverseTransform_C(transform, row_start, row_end, channel_bits,
                                  in, has_alpha, out);
      if (row_end != transform->height_) {
        // The last predicted row in this iteration will be the top-pred row
        // for the first row in next iteration.
        std::copy(out + 4 * (row_end - row_start - 1) * width,
                  out + 4 * (row_end - row_start) * width, out - 4 * width);
      }
      break;
    }
    case CROSS_COLOR_TRANSFORM:
      ColorSpaceInverseTransform_C(transform, row_start, row_end, channel_bits,
                                   in, out);
      break;
    case GROUP4:
    case COLOR_INDEXING_TRANSFORM:
      ColorIndexInverseTransform_C(transform, row_start, row_end, in, out);
      break;
    default:
      assert(false);
  }
}

//------------------------------------------------------------------------------

ProcessDecBlueAndRedFunc AddGreenToBlueAndRed;
PredictorAddSubFunc PredictorsAdd[kNumPredictors];

// exposed plain-C implementations
PredictorAddSubFunc PredictorsAdd_C[kNumPredictors];

TransformColorInverseFunc TransformColorInverse;

MapARGBFunc MapColor;

static volatile WP2CPUInfo lossless_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DecLDspInit() {
  if (lossless_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  COPY_PREDICTOR_ARRAY(PredictorAdd, PredictorsAdd)
  COPY_PREDICTOR_ARRAY(PredictorAdd, PredictorsAdd_C)

  AddGreenToBlueAndRed = AddGreenToBlueAndRed_C;

  TransformColorInverse = TransformColorInverse_C;

  MapColor = MapARGB_C;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (WP2GetCPUInfo != NULL) {
  // TODO(skal): SSE2, etc.
  }
  lossless_last_cpuinfo_used = WP2GetCPUInfo;
}
#undef COPY_PREDICTOR_ARRAY

}  // namespace WP2L
