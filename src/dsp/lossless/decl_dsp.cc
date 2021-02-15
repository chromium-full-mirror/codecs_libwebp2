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

namespace WP2L {
namespace {

//------------------------------------------------------------------------------
// Predictors

// Sum of each component, mod 256.
WP2_UBSAN_IGNORE_UNSIGNED_OVERFLOW inline void AddPixels(
    const int16_t* const a, bool has_alpha, const int16_t* const b,
    int16_t* const out) {
  out[0] = has_alpha ? a[0] + b[0] : WP2::kAlphaMax;
  for (uint32_t i = 1; i < 4; ++i) out[i] = a[i] + b[i];
}

void PredictorNonClampedAdd0_C(
    const int16_t* in, bool has_alpha, const int16_t*, uint32_t num_pixels,
    int16_t* out) {
  int16_t pred[4];
  WP2L::PredictorsNonClamped_C[0](/*left=*/nullptr, /*top=*/nullptr, pred);
  for (const int16_t* in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    AddPixels(in, has_alpha, pred, out);
  }
}
void PredictorNonClampedAdd1_C(
    const int16_t* in, bool has_alpha, const int16_t*, uint32_t num_pixels,
    int16_t* out) {
  for (const int16_t* in_end = in + 4 * num_pixels; in < in_end;
       in += 4, out += 4) {
    AddPixels(in, has_alpha, &out[-4], out);
  }
}

// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// The predictor is added to the output pixel (which
// is therefore considered as a residual) to get the final prediction.
#define GENERATE_PREDICTOR_NON_CLAMPED_ADD(I)                              \
void PredictorNonClampedAdd##I##_C(                                        \
    const int16_t* in, bool has_alpha, const int16_t* upper,               \
    uint32_t num_pixels, int16_t* out) {                                   \
  int16_t pred[4];                                                         \
  assert(upper != nullptr);                                                \
  for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
       out += 4, in += 4, upper += 4) {                                    \
    PredictorsNonClamped_C[I](out - 4, upper, pred);                       \
    AddPixels(in, has_alpha, pred, out);                                   \
  }                                                                        \
}

GENERATE_PREDICTOR_NON_CLAMPED_ADD(2)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(3)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(4)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(5)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(6)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(7)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(8)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(9)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(10)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(11)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(12)

#define GENERATE_PREDICTOR_CLAMPED_ADD(I)                                  \
void PredictorClampedAdd##I##_C(                                           \
    const int16_t* in, bool has_alpha, const int16_t* upper,               \
    uint32_t num_pixels, const int16_t min_values[4],                      \
    const int16_t max_values[4], int16_t* out) {                           \
  int16_t pred[4];                                                         \
  assert(upper != nullptr);                                                \
  for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
       out += 4, in += 4, upper += 4) {                                    \
    PredictorsClamped_C[I](out - 4, upper, min_values, max_values, pred);  \
    AddPixels(in, has_alpha, pred, out);                                   \
  }                                                                        \
}

GENERATE_PREDICTOR_CLAMPED_ADD(0)
GENERATE_PREDICTOR_CLAMPED_ADD(1)

//------------------------------------------------------------------------------
// Inverse transforms.

// Inverse prediction.
void PredictorInverseTransform_C(const Transform* const transform,
                                 int y_start, int y_end,
                                 const int16_t min_values[4],
                                 const int16_t max_values[4],
                                 const int16_t* in, bool has_alpha,
                                 int16_t* out) {
  const int width = transform->width_;
  if (y_start == 0) {  // First Row follows the L (mode=1) mode.
    PredictorNonClampedAdd0_C(in, has_alpha, nullptr, 1, out);
    PredictorNonClampedAdd1_C(in + 4, has_alpha, nullptr, width - 1, out + 4);
    in += 4 * width;
    out += 4 * width;
    ++y_start;
  }

  {
    int y = y_start;
    const int tile_width = 1 << transform->bits_;
    const int tile_mask = tile_width - 1;
    const int tiles_per_row = SubSampleSize(width, transform->bits_);
    const int16_t* pred_mode_base =
        transform->data_.data() + 4 * (y >> transform->bits_) * tiles_per_row;

    while (y < y_end) {
      const int16_t* pred_mode_src = pred_mode_base;
      int x = 1;
      // First pixel follows the T (mode=2) mode.
      PredictorNonClampedAdd2_C(in, has_alpha, out - 4 * width, 1, out);
      // .. the rest:
      while (x < width) {
        int x_end = (x & ~tile_mask) + tile_width;
        if (x_end > width) x_end = width;
        switch (kPredictorType[pred_mode_src[2]]) {
          case PredictorType::NonClamped:
            PredictorsNonClampedAdd[kPredictorIndex[pred_mode_src[2]]](
                in + 4 * x, has_alpha, out + 4 * x - 4 * width, x_end - x,
                out + 4 * x);
            break;
          case PredictorType::Clamped:
            PredictorsClampedAdd[kPredictorIndex[pred_mode_src[2]]](
                in + 4 * x, has_alpha, out + 4 * x - 4 * width, x_end - x,
                min_values, max_values, out + 4 * x);
            break;
          default:
            assert(false);
        }
        x = x_end;
        pred_mode_src += 4;
      }
      in += 4 * width;
      out += 4 * width;
      ++y;
      // Use the same mask, since tiles are squares.
      if ((y & tile_mask) == 0) {
        pred_mode_base += 4 * tiles_per_row;
      }
    }
  }
}

// Add green to blue and red channels (i.e. perform the inverse transform of
// 'subtract green').
void AddGreenToBlueAndRed_C(const int16_t* src, uint32_t num_pixels,
                            int16_t* dst) {
  for (const int16_t* src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const int16_t green = src[2];
    dst[0] = src[0];
    dst[1] = src[1] + green;
    dst[2] = src[2];
    dst[3] = src[3] + green;
  }
}

inline void ColorCodeToMultipliers(const int16_t* const color_code,
                                   Multipliers* const m) {
  m->green_to_red = color_code[3];
  m->green_to_blue = color_code[2];
  m->red_to_blue = color_code[1];
}

void TransformColorInverse_C(const Multipliers& m,
                             const int16_t* src, uint32_t num_pixels,
                             int16_t* dst) {
  for (const int16_t* src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const int16_t red = src[1];
    const int16_t green = src[2];
    int new_red = red;
    int new_blue = src[3];
    new_red += ColorTransformDelta(m.green_to_red, green);
    new_blue += (m.green_to_blue * green + m.red_to_blue * new_red + 16) >> 5;
    dst[0] = src[0];
    dst[1] = new_red;
    dst[2] = src[2];
    dst[3] = new_blue;
  }
}

// Color space inverse transform.
void ColorSpaceInverseTransform_C(const Transform* const transform,
                                  int y_start, int y_end,
                                  const int16_t* src, int16_t* dst) {
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
    const int16_t* const src_safe_end = src + 4 * safe_width;
    const int16_t* const src_end = src + 4 * width;
    while (src < src_safe_end) {
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(m, src, tile_width, dst);
      src += 4 * tile_width;
      dst += 4 * tile_width;
      pred += 4;
    }
    if (src < src_end) {  // Left-overs using C-version.
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(m, src, remaining_width, dst);
      src += 4 * remaining_width;
      dst += 4 * remaining_width;
      pred += 4;
    }
    ++y;
    if ((y & mask) == 0) pred_row += 4 * tiles_per_row;
  }
}

void MapARGB_C(const int16_t* src, const int16_t* const color_map,
               uint32_t color_map_size, uint32_t y_start, uint32_t y_end,
               uint32_t width, int16_t* dst) {
  for (const int16_t* const src_end = src + 4 * (y_end - y_start) * width;
       src < src_end; src += 4, dst += 4) {
    if (src[2] >= (int16_t)color_map_size) assert(false);
    ColorCopy(&color_map[4 * src[2]], dst);
  }
}
void ColorIndexInverseTransform_C(const Transform* const transform,
                                  uint32_t y_start, uint32_t y_end,
                                  const int16_t* const src,
                                  int16_t* const dst) {
  const uint32_t width = transform->width_;
  MapColor(src, /*color_map=*/transform->data_.data(),
           /*color_map_size=*/transform->data_.size() / 4, y_start, y_end,
           width, dst);
}

//------------------------------------------------------------------------------
// SSE version

#if defined(WP2_USE_SSE)

void TransformColorInverse_SSE(const WP2L::Multipliers& m,
                               const int16_t* src, uint32_t num_pixels,
                               int16_t* dst) {
  const int32_t green_to_red = m.green_to_red;
  const int32_t green_to_blue = m.green_to_blue;
  const int32_t red_to_blue = m.red_to_blue;
  const int16_t* const src_end = src + 4 * num_pixels;
  const __m128i kMask_ag = _mm_set1_epi32(0xffff);  // A0G0 mask
  const __m128i kRound = _mm_set1_epi32(0x8000);
  const __m128i perm =
      _mm_setr_epi8(2, 3, 4, 5, 6, 7, 4, 5, 10, 11, 12, 13, 14, 15, 12, 13);

  __m128i mult1 = _mm_cvtsi32_si128((uint32_t)green_to_red << 16 | 32);
  __m128i mult2 = _mm_cvtsi32_si128((uint32_t)red_to_blue << 16);
  mult1 = _mm_insert_epi32(mult1, (uint32_t)green_to_blue << 16 | 32u, 1);
  mult1 = _mm_unpacklo_epi64(mult1, mult1);
  mult2 = _mm_unpacklo_epi64(mult2, mult2);

  for (; src + 8 <= src_end; src += 8, dst += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)src);
    const __m128i B = _mm_shuffle_epi8(A, perm);  // ARGB -> RGBG
    const __m128i C = _mm_madd_epi16(B, mult1);
    const __m128i D = _mm_add_epi32(_mm_slli_epi32(C, 16 - 5), kRound);
    const __m128i E = _mm_madd_epi16(D, mult2);
    const __m128i F = _mm_slli_epi64(E, 48 - 5);
    const __m128i G = _mm_add_epi32(F, D);
    const __m128i out = _mm_blendv_epi8(G, A, kMask_ag);
    _mm_storeu_si128((__m128i*)dst, out);
  }
  for (; src < src_end; src += 4, dst += 4) {
    const int16_t red = src[1];
    const int16_t green = src[2];
    int32_t new_red = red;
    int32_t new_blue = src[3];
    new_red += (green_to_red * green + 16) >> 5;
    new_blue += (green_to_blue * green + red_to_blue * new_red + 16) >> 5;
    dst[0] = src[0];
    dst[1] = new_red;
    dst[2] = src[2];
    dst[3] = new_blue;
  }
}

WP2_TSAN_IGNORE_FUNCTION void DecLDspInitSSE() {
  WP2L::TransformColorInverse = TransformColorInverse_SSE;
}

#endif  // WP2_USE_SSE

}    // namespace

//------------------------------------------------------------------------------

void InverseTransform(const Transform* const transform, uint32_t row_start,
                      uint32_t row_end, const int16_t min_values[4],
                      const int16_t max_values[4], const int16_t* const in,
                      bool has_alpha, int16_t* const out) {
  const uint32_t width = transform->width_;
  assert(row_start < row_end);
  assert(row_end <= transform->height_);

  switch (transform->type_) {
    case TransformType::kSubtractGreen:
      AddGreenToBlueAndRed(in, (row_end - row_start) * width, out);
      break;
    case TransformType::kPredictor: {
      PredictorInverseTransform_C(transform, row_start, row_end, min_values,
                                  max_values, in, has_alpha, out);
      if (row_end != transform->height_) {
        // The last predicted row in this iteration will be the top-pred row
        // for the first row in next iteration.
        std::copy(out + 4 * (row_end - row_start - 1) * width,
                  out + 4 * (row_end - row_start) * width, out - 4 * width);
      }
      break;
    }
    case TransformType::kCrossColor:
      ColorSpaceInverseTransform_C(transform, row_start, row_end, in, out);
      break;
    case TransformType::kGroup4:
    case TransformType::kColorIndexing:
      ColorIndexInverseTransform_C(transform, row_start, row_end, in, out);
      break;
    default:
      assert(false);
  }
}

//------------------------------------------------------------------------------

ProcessDecBlueAndRedFunc AddGreenToBlueAndRed;
PredictorNonClampedAddSubFunc PredictorsNonClampedAdd[kNumPredictorsNonClamped];
PredictorClampedAddSubFunc PredictorsClampedAdd[kNumPredictorsClamped];

// exposed plain-C implementations
PredictorNonClampedAddSubFunc
    PredictorsNonClampedAdd_C[kNumPredictorsNonClamped];
PredictorClampedAddSubFunc PredictorsClampedAdd_C[kNumPredictorsClamped];

TransformColorInverseFunc TransformColorInverse;

MapARGBFunc MapColor;

static volatile WP2CPUInfo lossless_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DecLDspInit() {
  if (lossless_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Add, Add)
  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Add, Add_C)
  COPY_PREDICTOR_CLAMPED_ARRAY(Add, Add)
  COPY_PREDICTOR_CLAMPED_ARRAY(Add, Add_C)

  AddGreenToBlueAndRed = AddGreenToBlueAndRed_C;

  TransformColorInverse = TransformColorInverse_C;

  MapColor = MapARGB_C;

  if (WP2GetCPUInfo != NULL) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) DecLDspInitSSE();
#endif
  }
  lossless_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2L
