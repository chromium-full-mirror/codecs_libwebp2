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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/common/lossless/color_cache.h"
#include "src/common/lossless/transforms.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/wp2/format_constants.h"

namespace WP2L {
namespace {

//------------------------------------------------------------------------------
// Predictors

// Sum of each component, mod 256.
WP2_UBSAN_IGNORE_UNSIGNED_OVERFLOW inline void AddPixels(const int16_t* const a,
                                                         bool has_alpha,
                                                         const int16_t* const b,
                                                         int16_t* const out) {
  out[0] = has_alpha ? a[0] + b[0] : WP2::kAlphaMax;
  for (uint32_t i = 1; i < 4; ++i) out[i] = a[i] + b[i];
}

// Macros used to create a batch predictor that iteratively uses a
// one-pixel predictor.

// The predictor is added to the output pixel (which
// is therefore considered as a residual) to get the final prediction.
#define GENERATE_PREDICTOR_NON_CLAMPED_ADD(I)                                \
  void PredictorNonClampedAdd##I##_C(                                        \
      const int16_t* in, bool has_alpha, const int16_t* upper,               \
      uint32_t num_pixels, int16_t* out, int16_t* predicted) {               \
    int16_t pred[4];                                                         \
    if (I != 0) assert(upper != nullptr);                                    \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         out += 4, in += 4) {                                                \
      PredictorsNonClamped_C[I](out - 4, upper, pred);                       \
      AddPixels(in, has_alpha, pred, out);                                   \
      if (predicted != nullptr) {                                            \
        std::copy(pred, pred + 4, predicted);                                \
        predicted += 4;                                                      \
      }                                                                      \
      if (I != 0) upper += 4;                                                \
    }                                                                        \
  }

GENERATE_PREDICTOR_NON_CLAMPED_ADD(0)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(1)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(2)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(3)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(4)
GENERATE_PREDICTOR_NON_CLAMPED_ADD(5)

#define GENERATE_PREDICTOR_CLAMPED_ADD(I)                                    \
  void PredictorClampedAdd##I##_C(                                           \
      const int16_t* in, bool has_alpha, const int16_t* upper,               \
      uint32_t num_pixels, const int32_t min_values[4],                      \
      const int32_t max_values[4], int16_t* out, int16_t* predicted) {       \
    int16_t pred[4];                                                         \
    assert(upper != nullptr);                                                \
    for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end; \
         out += 4, in += 4, upper += 4) {                                    \
      PredictorsClamped_C[I](out - 4, upper, min_values, max_values, pred);  \
      AddPixels(in, has_alpha, pred, out);                                   \
      if (predicted != nullptr) {                                            \
        std::copy(pred, pred + 4, predicted);                                \
        predicted += 4;                                                      \
      }                                                                      \
    }                                                                        \
  }

GENERATE_PREDICTOR_CLAMPED_ADD(0)
GENERATE_PREDICTOR_CLAMPED_ADD(1)

template <int MAIN_ANGLE>
void PredictorAngleAdd_C(int delta, const int16_t* in, bool has_alpha,
                         const int16_t* upper, uint32_t num_pixels,
                         int16_t* out, int16_t* predicted) {
  int16_t pred[4];
  if (MAIN_ANGLE != 180) assert(upper != nullptr);
  for (const int16_t* const out_end = out + 4 * num_pixels; out < out_end;
       out += 4, in += 4) {
    AnglePredict<MAIN_ANGLE>(delta, out - 4, upper, pred);
    AddPixels(in, has_alpha, pred, out);
    if (predicted != nullptr) {
      std::copy(pred, pred + 4, predicted);
      predicted += 4;
    }
    if (upper != nullptr) upper += 4;
  }
}

//------------------------------------------------------------------------------
// Inverse transforms.

PredictorNonClampedAddFunc PredictorsAddBlack;
PredictorAngleAddFunc PredictorsAddTop;
PredictorAngleAddFunc PredictorsAddLeft;

// Inverse prediction.
void PredictorInverseTransform_C(const Transform* const transform, int y_start,
                                 int y_end, const int32_t min_values[4],
                                 const int32_t max_values[4], const int16_t* in,
                                 bool has_alpha, int16_t* out,
                                 int16_t* predicted) {
  const int width = transform->width_pic_;
  if (y_start == 0) {  // First Row follows the L (mode=1) mode.
    PredictorsAddBlack(in, has_alpha, nullptr, 1, out, predicted);
    PredictorsAddLeft(/*delta=*/0, in + 4, has_alpha, nullptr, width - 1,
                      out + 4, predicted != nullptr ? predicted + 4 : nullptr);
    in += 4 * width;
    out += 4 * width;
    if (predicted != nullptr) predicted += 4 * width;
    ++y_start;
  }

  int y = y_start;
  const int tile_width = 1 << transform->bits_;
  const int tile_mask = tile_width - 1;
  const int16_t* pred_mode_base =
      transform->data_.data() + 4 * (y >> transform->bits_) * transform->width_;

  while (y < y_end) {
    const int16_t* pred_mode_src = pred_mode_base;
    int x = 1;
    // First pixel follows the T (mode=2) mode.
    PredictorsAddTop(/*delta=*/0, in, has_alpha, out - 4 * width, 1, out,
                     predicted);
    // .. the rest:
    while (x < width) {
      int x_end = (x & ~tile_mask) + tile_width;
      if (x_end > width) x_end = width;
      const uint32_t mode = pred_mode_src[2] & ((1 << kNumPredictorsBits) - 1);
      switch (kPredictorType[mode]) {
        case PredictorType::NonClamped:
          PredictorsNonClampedAdd[kPredictorIndex[mode]](
              in + 4 * x, has_alpha, out + 4 * x - 4 * width, x_end - x,
              out + 4 * x, predicted != nullptr ? predicted + 4 * x : nullptr);
          break;
        case PredictorType::Clamped:
          PredictorsClampedAdd[kPredictorIndex[mode]](
              in + 4 * x, has_alpha, out + 4 * x - 4 * width, x_end - x,
              min_values, max_values, out + 4 * x,
              predicted != nullptr ? predicted + 4 * x : nullptr);
          break;
        case PredictorType::Angle: {
          const int sub_mode_positive =
              (pred_mode_src[2] >> kNumPredictorsBits);
          const int sub_mode = (sub_mode_positive % 2 == 0)
                                   ? sub_mode_positive / 2
                                   : (sub_mode_positive + 1) / -2;
          PredictorsAngleAdd[kPredictorIndex[mode]](
              /*delta=*/sub_mode, in + 4 * x, has_alpha,
              out + 4 * x - 4 * width, x_end - x, out + 4 * x,
              predicted != nullptr ? predicted + 4 * x : nullptr);
          break;
        }
        default:
          assert(false);
      }
      x = x_end;
      pred_mode_src += 4;
    }
    in += 4 * width;
    out += 4 * width;
    if (predicted != nullptr) predicted += 4 * width;
    ++y;
    // Use the same mask, since tiles are squares.
    if ((y & tile_mask) == 0) {
      pred_mode_base += 4 * transform->width_;
    }
  }
}

void YCoCgR2RGB_C(const int16_t* src, uint32_t num_pixels, int16_t* dst) {
  for (const int16_t* const src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const int16_t y = src[1];
    const int16_t co = src[2];
    const int16_t cg = src[3];
    const int16_t tmp = y - (cg / 2);
    const int16_t tmp_g = cg + tmp;
    const int16_t tmp_b = tmp - (co / 2);
    const int16_t tmp_r = tmp_b + co;
    dst[0] = src[0];
    dst[1] = tmp_r;
    dst[2] = tmp_g;
    dst[3] = tmp_b;
  }
}
}  // namespace

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

namespace {
static inline void ColorCodeToMultipliers(const int16_t* const color_code,
                                          Multipliers* const m) {
  m->green_to_red = color_code[3];
  m->green_to_blue = color_code[2];
  m->red_to_blue = color_code[1];
}
}  // namespace

void TransformColorInverse_C(const Multipliers& m, const int16_t* src,
                             uint32_t num_pixels, int16_t* dst,
                             int16_t* predicted) {
  for (const int16_t* src_end = src + 4 * num_pixels; src < src_end;
       src += 4, dst += 4) {
    const int16_t red = src[1];
    const int16_t green = src[2];
    int new_red = red;
    int new_blue = src[3];
    if (predicted != nullptr) {
      predicted[0] = src[0];
      predicted[1] = ColorTransformDelta(m.green_to_red * green);
      predicted[2] = src[2];
    }
    new_red += ColorTransformDelta(m.green_to_red * green);
    if (predicted != nullptr) {
      predicted[3] = ColorTransformDelta(m.green_to_blue * green +
                                         m.red_to_blue * new_red);
    }
    new_blue +=
        ColorTransformDelta(m.green_to_blue * green + m.red_to_blue * new_red);
    dst[0] = src[0];
    dst[1] = new_red;
    dst[2] = src[2];
    dst[3] = new_blue;
    if (predicted != nullptr) predicted += 4;
  }
}

namespace {
// Color space inverse transform.
void ColorSpaceInverseTransform_C(const Transform* const transform, int y_start,
                                  int y_end, const int16_t* src, int16_t* dst,
                                  int16_t* predicted) {
  const uint32_t width = transform->width_pic_;
  const int tile_width = 1 << transform->bits_;
  const int mask = tile_width - 1;
  const int safe_width = width & ~mask;
  const int remaining_width = width - safe_width;
  const int tiles_per_row = transform->width_;
  int y = y_start;
  const int16_t* pred_row =
      transform->data_.data() + 4 * (y >> transform->bits_) * tiles_per_row;

  while (y < y_end) {
    const int16_t* pred = pred_row;
    Multipliers m = {0, 0, 0};
    const int16_t* const src_safe_end = src + 4 * safe_width;
    const int16_t* const src_end = src + 4 * width;
    while (src < src_safe_end) {
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(m, src, tile_width, dst, predicted);
      src += 4 * tile_width;
      dst += 4 * tile_width;
      if (predicted != nullptr) predicted += 4 * tile_width;
      pred += 4;
    }
    if (src < src_end) {  // Left-overs using C-version.
      ColorCodeToMultipliers(pred, &m);
      TransformColorInverse(m, src, remaining_width, dst, predicted);
      src += 4 * remaining_width;
      dst += 4 * remaining_width;
      if (predicted != nullptr) predicted += 4 * remaining_width;
      pred += 4;
    }
    ++y;
    if ((y & mask) == 0) pred_row += 4 * tiles_per_row;
  }
}

// Color space inverse transform.
void CrossColorGlobalInverseTransform_C(const Transform* const transform,
                                        int y_start, int y_end,
                                        const int16_t* src, int16_t* dst,
                                        int16_t* predicted) {
  const uint32_t width = transform->width_pic_;
  const uint32_t c1 =
      static_cast<uint32_t>(transform->header_.cc_global_first_channel);
  const uint32_t c2 =
      static_cast<uint32_t>(transform->header_.cc_global_second_channel);
  const uint32_t c3 = 1 + 2 + 3 - c1 - c2;
  const TransformHeader::CCTransform cc_second_transform =
      transform->header_.cc_global_second_transform;
  assert(cc_second_transform != TransformHeader::CCTransform::kSubtractSecond &&
         cc_second_transform != TransformHeader::CCTransform::kSubtractAverage);
  const TransformHeader::CCTransform cc_third_transform =
      transform->header_.cc_global_third_transform;

  for (const int16_t* const src_end = src + 4 * (y_end - y_start) * width;
       src < src_end; src += 4, dst += 4) {
    dst[0] = src[0];
    dst[c1] = src[c1];
    switch (cc_second_transform) {
      case TransformHeader::CCTransform::kNothing:
        dst[c2] = src[c2];
        break;
      case TransformHeader::CCTransform::kSubtractFirst:
        dst[c2] = dst[c1] + src[c2];
        break;
      default:
        assert(false);
        break;
    }
    switch (cc_third_transform) {
      case TransformHeader::CCTransform::kNothing:
        dst[c3] = src[c3];
        break;
      case TransformHeader::CCTransform::kSubtractFirst:
        dst[c3] = dst[c1] + src[c3];
        break;
      case TransformHeader::CCTransform::kSubtractSecond:
        dst[c3] = dst[c2] + src[c3];
        break;
      case TransformHeader::CCTransform::kSubtractAverage:
        dst[c3] = (dst[c1] + dst[c2]) / 2 + src[c3];
    }
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
  const uint32_t width = transform->width_pic_;
  MapColor(src, /*color_map=*/transform->data_.data(),
           /*color_map_size=*/transform->data_.size() / 4, y_start, y_end,
           width, dst);
}

void NormalizeChannelsInverseTransform_C(const Transform& transform,
                                         uint32_t y_start, uint32_t y_end,
                                         bool has_alpha, const int16_t* src,
                                         int16_t* dst) {
  const uint32_t width = transform.width_pic_;
  if (transform.header_.normalize_channels_has_palette) {
    for (const int16_t* const src_end = src + 4 * (y_end - y_start) * width;
         src < src_end; src += 4, dst += 4) {
      if (!has_alpha) dst[0] = WP2::kAlphaMax;
      for (int c = has_alpha ? 0 : 1; c < 4; ++c) {
        dst[c] = transform.header_.normalize_channels_offset[c] +
                 transform.normalize_channels_mapping_[c][src[c]];
      }
    }
  } else {
    for (const int16_t* const src_end = src + 4 * (y_end - y_start) * width;
         src < src_end; src += 4, dst += 4) {
      if (!has_alpha) dst[0] = WP2::kAlphaMax;
      for (int c = has_alpha ? 0 : 1; c < 4; ++c) {
        dst[c] = transform.header_.normalize_channels_offset[c] + src[c];
      }
    }
  }
}

}  // namespace

//------------------------------------------------------------------------------

void InverseTransform(const Transform* const transform, uint32_t row_start,
                      uint32_t row_end, const int32_t min_values[4],
                      const int32_t max_values[4], const int16_t* const in,
                      bool has_alpha, int16_t* const out,
                      int16_t* const predicted) {
  const uint32_t width = transform->width_pic_;
  assert(row_start < row_end);
  assert(row_end <= transform->height_pic_);

  switch (transform->header_.type) {
    case TransformType::kYCoCgR:
      YCoCgRToRGB(in, (row_end - row_start) * width, out);
      break;
    case TransformType::kSubtractGreen:
      AddGreenToBlueAndRed(in, (row_end - row_start) * width, out);
      break;
    case TransformType::kPredictor:
    case TransformType::kPredictorWSub: {
      PredictorInverseTransform_C(transform, row_start, row_end, min_values,
                                  max_values, in, has_alpha, out, predicted);
      if (row_end != transform->height_pic_) {
        // The last predicted row in this iteration will be the top-pred row
        // for the first row in next iteration.
        std::copy(out + 4 * (row_end - row_start - 1) * width,
                  out + 4 * (row_end - row_start) * width, out - 4 * width);
      }
      break;
    }
    case TransformType::kCrossColor:
      ColorSpaceInverseTransform_C(transform, row_start, row_end, in, out,
                                   predicted);
      break;
    case TransformType::kCrossColorGlobal:
      CrossColorGlobalInverseTransform_C(transform, row_start, row_end, in, out,
                                         predicted);
      break;
    case TransformType::kColorIndexing:
      ColorIndexInverseTransform_C(transform, row_start, row_end, in, out);
      break;
    case TransformType::kNormalizeChannels:
      NormalizeChannelsInverseTransform_C(*transform, row_start, row_end,
                                          has_alpha, in, out);
      break;
    case TransformType::kNum:
      assert(false);
  }
}

//------------------------------------------------------------------------------

SimpleColorSpaceInverseFunc AddGreenToBlueAndRed;
SimpleColorSpaceInverseFunc YCoCgRToRGB;
std::array<PredictorNonClampedAddFunc, kNumPredictorsNonClamped>
    PredictorsNonClampedAdd;
std::array<PredictorNonClampedAddFunc, kNumPredictorsNonClamped>
    PredictorsNonClampedAdd_C;
std::array<PredictorClampedAddFunc, kNumPredictorsClamped> PredictorsClampedAdd;
std::array<PredictorClampedAddFunc, kNumPredictorsClamped>
    PredictorsClampedAdd_C;
std::array<PredictorAngleAddFunc, kNumPredictorsAngle> PredictorsAngleAdd;
std::array<PredictorAngleAddFunc, kNumPredictorsAngle> PredictorsAngleAdd_C;

TransformColorInverseFunc TransformColorInverse;

MapARGBFunc MapColor;

extern void DecLDspInitSSE();

static volatile WP2CPUInfo lossless_last_cpuinfo_used =
    (WP2CPUInfo)&lossless_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void DecLDspInit() {
  if (lossless_last_cpuinfo_used == WP2GetCPUInfo) return;

  DspInit();

  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Add, Add)
  COPY_PREDICTOR_NON_CLAMPED_ARRAY(Add, Add_C)
  COPY_PREDICTOR_CLAMPED_ARRAY(Add, Add)
  COPY_PREDICTOR_CLAMPED_ARRAY(Add, Add_C)
  PredictorsAngleAdd[0] = PredictorAngleAdd_C<45>;
  PredictorsAngleAdd[1] = PredictorAngleAdd_C<67>;
  PredictorsAngleAdd[2] = PredictorAngleAdd_C<90>;
  PredictorsAngleAdd[3] = PredictorAngleAdd_C<113>;
  PredictorsAngleAdd[4] = PredictorAngleAdd_C<135>;
  PredictorsAngleAdd[5] = PredictorAngleAdd_C<157>;
  PredictorsAngleAdd[6] = PredictorAngleAdd_C<180>;
  std::copy(PredictorsAngleAdd.begin(), PredictorsAngleAdd.end(),
            PredictorsAngleAdd_C.begin());

  AddGreenToBlueAndRed = AddGreenToBlueAndRed_C;
  YCoCgRToRGB = YCoCgR2RGB_C;

  TransformColorInverse = TransformColorInverse_C;

  MapColor = MapARGB_C;

  if (WP2GetCPUInfo != NULL) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) DecLDspInitSSE();
#endif
  }

  PredictorsAddBlack = PredictorsNonClampedAdd[0];
  PredictorsAddTop = PredictorsAngleAdd[kPredictorAngleTopIndex];
  PredictorsAddLeft = PredictorsAngleAdd[kPredictorAngleLeftIndex];

  lossless_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2L
