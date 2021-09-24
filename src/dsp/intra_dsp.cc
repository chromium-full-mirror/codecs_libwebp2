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
// Intra predictions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cmath>
#include <numeric>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace {

//------------------------------------------------------------------------------

inline void FillBlock(int16_t value, uint32_t bw, uint32_t bh,
                      int16_t* dst, size_t step) {
  for (uint32_t j = 0; j < bh; ++j) {
    for (uint32_t i = 0; i < bw; ++i) dst[i] = value;
    dst += step;
  }
}

void DC_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
          int16_t* dst, size_t step) {
  // Only smooth over the left, top-left and top contexts.
  const int32_t size = bh + 1 + bw;
  const int32_t sum = std::accumulate(ctx, ctx + size, 0);
  FillBlock(WP2::DivRound(sum, size), bw, bh, dst, step);
}

void DC_L_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
            int16_t* dst, size_t step) {
  const int32_t sum = std::accumulate(ctx, ctx + bh, 0);
  FillBlock(WP2::DivRound(sum, (int32_t)bh), bw, bh, dst, step);
}

void DC_T_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
            int16_t* dst, size_t step) {
  const int32_t sum = std::accumulate(ctx + bh + 1, ctx + bh + 1 + bw, 0);
  FillBlock(WP2::DivRound(sum, (int32_t)bw), bw, bh, dst, step);
}

int16_t CalcMedian(int16_t* ctx_to_find_median, const int32_t size) {
  int16_t* median_position = ctx_to_find_median + size / 2;
  std::nth_element(ctx_to_find_median, median_position,
                   ctx_to_find_median + size);
  if (size & 1) {
    return *median_position;
  }

  // ctx_to_find_median has an even size
  const int16_t left_median =
      *std::max_element(ctx_to_find_median, median_position - 1);
  const int16_t right_median = *median_position;

  return (left_median + right_median) / 2;
}

void MEDIAN_DC_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
                 int16_t* dst, size_t step) {
  const int32_t size = bh + 1 + bw;
  // TODO(maciekdragula): instead of using temporary ctx_to_find_median array,
  // try use dst instead as it will be fill in later.
  int16_t ctx_to_find_median[WP2::kMaxContextSize];
  std::copy(ctx, ctx + size, ctx_to_find_median);
  const int16_t median = CalcMedian(ctx_to_find_median, size);
  FillBlock(median, bw, bh, dst, step);
}

void MEDIAN_DC_L_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t,
                   int16_t, int16_t* dst, size_t step) {
  const int32_t size = bh;
  int16_t ctx_to_find_median[WP2::kMaxContextSize];
  std::copy(ctx, ctx + size, ctx_to_find_median);
  const int16_t median = CalcMedian(ctx_to_find_median, size);
  FillBlock(median, bw, bh, dst, step);
}

void MEDIAN_DC_T_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t,
                   int16_t, int16_t* dst, size_t step) {
  const int32_t size = bw;
  int16_t ctx_to_find_median[WP2::kMaxContextSize];
  std::copy(ctx + bh + 1, ctx + bh + 1 + size, ctx_to_find_median);
  const int16_t median = CalcMedian(ctx_to_find_median, size);
  FillBlock(median, bw, bh, dst, step);
}

//------------------------------------------------------------------------------

// The Sm_Weights_Tx_* in AV1.
static constexpr int16_t kSmoothWeights[] = {
  0, 0, 0, 0,   // <- dummy entries to make the offsets easy
  255, 149, 85, 64,  // 4
  255, 197, 146, 105, 73, 50, 37, 32,  // 8
  255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,  // 16
  255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
  66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,  // 32
};
STATIC_ASSERT_ARRAY_SIZE(kSmoothWeights, 4 + 4 + 8 + 16 + 32);
constexpr int32_t kSmoothShift = 8;

const int16_t* GetWeights(uint32_t d) { return &kSmoothWeights[d]; }

void Smooth_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
              int16_t* dst, size_t step) {
  const int16_t* const wLR = GetWeights(bw);  // left / right weights
  const int16_t* const wAB = GetWeights(bh);  // above / below weights
  const int16_t below = ctx[0];
  for (uint32_t j = 0; j < bh; ++j) {
    const int16_t left = ctx[bh - 1 - j];
    const int16_t right = ctx[bh + 1 + bw + 1 + j];
    for (uint32_t i = 0; i < bw; ++i) {
      const int16_t above = ctx[bh + 1 + i];
      const int32_t v = above * wAB[j] + below * (256 - wAB[j])
                      + left * wLR[i] + right * (256 - wLR[i]);
      dst[i] = WP2::RightShiftRound(v, kSmoothShift + 1);
    }
    dst += step;
  }
}

void Smooth_H_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
                int16_t* dst, size_t step) {
  const int16_t* const wLR = GetWeights(bw);  // left / right weights
  for (uint32_t j = 0; j < bh; ++j) {
    const int16_t left = ctx[bh - 1 - j];
    const int16_t right = ctx[bh + 1 + bw + 1 + j];
    for (uint32_t i = 0; i < bw; ++i) {
      const int32_t v = left * wLR[i] + right * (256 - wLR[i]);
      dst[i] = WP2::RightShiftRound(v, kSmoothShift);
    }
    dst += step;
  }
}

void Smooth_V_C(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t, int16_t,
                int16_t* dst, size_t step) {
  const int16_t* const wAB = GetWeights(bh);  // above / below weights
  const int16_t below = ctx[0];
  for (uint32_t j = 0; j < bh; ++j) {
    for (uint32_t i = 0; i < bw; ++i) {
      const int16_t above = ctx[bh + 1 + i];
      const int32_t v = above * wAB[j] + below * (256 - wAB[j]);
      dst[i] = WP2::RightShiftRound(v, kSmoothShift);
    }
    dst += step;
  }
}

//------------------------------------------------------------------------------

void TrueMotion_C(const int16_t* ctx, uint32_t bw, uint32_t bh,
                  int16_t min_value, int16_t max_value, int16_t* dst,
                  size_t step) {
  const int16_t* const top = ctx + bh + 1;
  const int16_t top_left = ctx[bh];
  for (uint32_t y = 0; y < bh; ++y) {
    const int16_t base = ctx[bh - 1 - y] - top_left;
    for (uint32_t x = 0; x < bw; ++x) {
      dst[x] = WP2::Clamp<int16_t>(top[x] + base, min_value, max_value);
    }
    dst += step;
  }
}

}  // namespace

//------------------------------------------------------------------------------
// Paeth

namespace WP2 {

static inline int16_t PaethPredictOnePixel(int16_t left, int16_t top,
                                           int16_t top_left) {
  // abs((top + left - top_left) - left)
  const uint16_t p_left = std::abs(top - top_left);
  // abs((top + left - top_left) - top)
  const uint16_t p_top = std::abs(left - top_left);
  // abs((top + left - top_left) - top_left)
  const uint16_t p_top_left = std::abs((top - top_left) + (left - top_left));
  // Return nearest to base of left, top and top_left.
  return (p_left <= p_top && p_left <= p_top_left)
             ? left
             : (p_top <= p_top_left) ? top : top_left;
}

void BasePaethPredictor(const int16_t* ctx, uint32_t bw, uint32_t bh, int16_t,
                        int16_t, int16_t* dst, size_t step) {
  const int16_t top_left = ctx[bh];
  for (uint32_t y = 0; y < bh; ++y) {
    const int16_t left = ctx[bh - 1 - y];
    for (uint32_t x = 0; x < bw; ++x) {
      const int16_t top = ctx[bh + 1 + x];
      dst[x] = PaethPredictOnePixel(left, top, top_left);
    }
    dst += step;
  }
}

}  // namespace WP2

//------------------------------------------------------------------------------

namespace {

// Fixed-point precision in bits, maximized to have all asserts pass.
constexpr uint32_t kAnglePredPrecision = 15;
static_assert(kAnglePredPrecision + 1 + 10 + 1 <= 32,
              "Too many precision bits");
constexpr int32_t kOne = (1u << kAnglePredPrecision);
constexpr uint32_t kMask = kOne - 1;
// (int32_t)(1. / std::tan(alpha) * (1u << kAnglePredPrecision))
constexpr int32_t kCoTanTable[] = {
    143565,      113740,  93645,   79108,  68043,  59289,  52149,   46182,
    41089,       36667,   32768,   29283,  26131,  23250,  20589,   18110,
    15780,       13572,   11466,   9440,   7479,   5567,   3692,    1840,
    0,  // 90 degrees
    -1840,       -3692,   -5567,   -7479,  -9440,  -11466, -13572,  -15780,
    -18110,      -20589,  -23250,  -26131, -29283, -32767, -36667,  -41089,
    -46182,      -52149,  -59289,  -68043, -79108, -93645, -113740, -143565,
    -192858,     -290824, -583488,
    -2147483648,  // 180 degrees
    583488,      290824,  192858,  143565, 113740, 93645,  79108,   68043,
    59289,       52149,   46182,   41089,  36667,  32767,  29283,   26131,
    23250};
STATIC_ASSERT_ARRAY_SIZE(kCoTanTable, WP2::kNumDirectionalAngles);

// (int32_t)(std::tan(alpha) * (1u << kAnglePredPrecision))
constexpr int32_t kTanTable[] = {
    // (these 'angle < 90' entries are not actually used)
    7479,        9440,    11466,   13572,   15780,   18110,  20589,  23250,
    26131,       29283,   32767,   36667,   41089,   46182,  52149,  59289,
    68043,       79108,   93645,   113740,  143565,  192858, 290824, 583488,
    -2147483648,  // 90 degrees
    -583488,     -290824, -192858, -143565, -113740, -93645, -79108, -68043,
    -59289,      -52149,  -46182,  -41089,  -36667,  -32768, -29283, -26131,
    -23250,      -20589,  -18110,  -15780,  -13572,  -11466, -9440,  -7479,
    -5567,       -3692,   -1840,
    0,  // 180 degrees
    1840,        3692,    5567,    7479,    9440,    11466,  13572,  15780,
    18110,       20589,   23250,   26131,   29283,   32768,  36667,  41089,
    46182};
STATIC_ASSERT_ARRAY_SIZE(kTanTable, WP2::kNumDirectionalAngles);

void AnglePredInterpolate_C(const int16_t* src,
                            int32_t frac, int16_t* dst, uint32_t len) {
  assert(frac >= 0 && frac <= kOne);
  uint32_t x = 0;
  for (; x < len; ++x) {
    // Computes src[x] * (kOne - frac) + src[x + 1] * frac
    dst[x] = src[x] + (((src[x + 1] - src[x]) * frac) >> kAnglePredPrecision);
  }
}

}  // namespace

namespace WP2 {

void SimpleAnglePredictor(uint8_t angle_idx, const int16_t ctx[],
                          uint32_t log2_bw, uint32_t log2_bh,
                          int16_t* dst, size_t step) {
  const uint32_t bw = 1u << log2_bw;
  const uint32_t bh = 1u << log2_bh;
  const ContextType context_type = GetContextType(angle_idx);
  const uint32_t context_size = ContextSize(context_type, bw, bh);

  if (angle_idx == kAngle_45) {
    for (uint32_t y = 0; y < bh; ++y, dst += step) {
      std::copy(ctx + (bh + 1 + y + 1), ctx + (bh + 1 + y + 1) + bw, dst);
    }
  } else if (angle_idx == kAngle_90) {
    for (uint32_t y = 0; y < bh; ++y, dst += step) {
      std::copy(&ctx[bh + 1], &ctx[bh + 1 + bw], dst);
    }
  } else if (angle_idx == kAngle_135) {
    for (uint32_t y = 0; y < bh; ++y, dst += step) {
      std::copy(&ctx[bh - y], &ctx[bh - y + bw], dst);
    }
  } else if (angle_idx == kAngle_180) {
    for (uint32_t y = 0; y < bh; ++y, dst += step) {
      std::fill(dst, dst + bw, ctx[bh - 1 - y]);
    }
  } else {
    const int16_t* dst_end = dst + step * bh;
    const int32_t cot = kCoTanTable[angle_idx];
    // Line equation in X and Y of slope 'angle' going through pixel (x, y):
    // (X-x)*sin(-angle) - (Y-y)*cos(-angle) = 0, with x axis going
    // right, y axis going down (hence the angle opposition).
    // Can be simplified to: (X-x)*sin(angle) + (Y-y)*cos(angle) = 0
    if (angle_idx < kAngle_90) {
      // Intersection index (in fixed-point precision) for the first context
      // sample at Y=-1 (hence the '+ cot').
      uint32_t ind_cur = ((bh + 1) << kAnglePredPrecision) + cot;
      const uint32_t ind_end =
          std::min((context_size - 1) << kAnglePredPrecision,
                   ind_cur + bh * cot);
      for (; ind_cur < ind_end; ind_cur += cot, dst += step) {
        assert(dst < dst_end);
        const uint32_t ind = ind_cur >> kAnglePredPrecision;
        const uint32_t frac = ind_cur & kMask;
        assert(ind < context_size);
        const uint32_t interpolength = std::min(bw, context_size - (ind + 1));
        // Copy pixels from 'ctx' to 'dst' then spatially-left-shift them by
        // 'frac' (which is in [0:1[).
        AnglePredInterpolate(ctx + ind, frac, dst, interpolength);
        // For the remaining pixels that have no source within 'context_size',
        // copy the last 'ctx' value.
        if (interpolength < bw) {
          std::fill(dst + interpolength, dst + bw, ctx[context_size - 1]);
        }
      }
      // last lines
      if (dst < dst_end) {
        if (ind_cur < ind_end + cot) {
          // If we reach outside the context, repeat the last element of the
          // line above.
          std::fill(dst, dst + bw, dst[-(int)step + (int)bw - 1]);
          dst += step;
        }
        for (; dst < dst_end; dst += step) {
          // If we cannot interpolate with the next element because it is out
          // of context, repeat the last context element.
          std::fill(dst, dst + bw, ctx[context_size - 1]);
        }
      }
    } else {        // angle >= 90.f
      const int32_t t = kTanTable[angle_idx];
      const int32_t ctx_height =
          ((context_type == kContextExtendLeft) ? 2 * bh : bh);
      const int32_t ctx_height_minus_1 =
          ChangePrecision((int32_t)ctx_height - 1, 0, kAnglePredPrecision);
      int32_t ind_base = ctx_height_minus_1;
      for (int32_t y = 0; y < (int32_t)bh; ++y, dst += step, ind_base -= kOne) {
        int32_t x_max;
        // Figure out where the span predicted with the left-context samples
        // ends. For angle < 180, we intersect the top-left sample with the
        // row 'y'. For angle > 180, we intersect the bottom-left one.
        if (angle_idx > kAngle_180) {
          x_max = (ctx_height - 1 - y) * cot;
        } else {
          x_max = (-1 - y) * cot;
        }
        // we now save the fractional part for later, to avoid re-doing the calc
        const int32_t frac0 = kOne - (x_max & kMask);
        assert(x_max >= 0);
        x_max = std::min(x_max >> kAnglePredPrecision, (int32_t)bw);
        // Hit the left context. X = -1.
        for (int32_t x = 0, ind = ind_base; x < x_max; ++x) {
          ind -= t;
          const int32_t frac = (ind & kMask);
          const int32_t pos = (ind >> kAnglePredPrecision);
          assert(pos <= ctx_height - 1 && pos >= 0);
          // Compute ctx[pos] * (kOne - frac) + ctx[pos + 1] * frac.
          dst[x] =
              ctx[pos] + ChangePrecision((ctx[pos + 1] - ctx[pos]) * frac,
                                         kAnglePredPrecision, 0u);
        }
        if (x_max == (int32_t)bw) continue;
        if (angle_idx > kAngle_180) {
          // Repeat the bottom of the left context.
          std::fill(dst + x_max, dst + bw, ctx[0]);
        } else {
          // Hit the top context. Y = -1.
          AnglePredInterpolate(ctx + bh, frac0, dst + x_max, bw - x_max);
        }
      }
    }
  }
}

}  // namespace WP2

namespace WP2L {

namespace {
// Computes src1 * (kOne - frac) + src2 * frac
void AnglePredInterpolate_C(const int16_t* const src1,
                            const int16_t* const src2, int32_t frac,
                            int16_t* const dst) {
  assert(frac >= 0 && frac <= kOne);
  for (uint32_t c = 0; c < 4; ++c) {
    dst[c] = src1[c] +
             WP2::RightShift((src2[c] - src1[c]) * frac, kAnglePredPrecision);
  }
}
}  // namespace

void SubAnglePredictor(uint8_t angle_idx, const int16_t* const left,
                       const int16_t* const top, int16_t* dst) {
  const int32_t cot = kCoTanTable[angle_idx];
  // Line equation in X and Y of slope 'angle' going through pixel (x, y):
  // (X-x)*sin(-angle) - (Y-y)*cos(-angle) = 0, with x axis going
  // right, y axis going down (hence the angle opposition).
  // Can be simplified to: (X-x)*sin(angle) + (Y-y)*cos(angle) = 0
  if (angle_idx < WP2::kAngle_90) {
    // (X-0)*sin(angle) + (-1-0)*cos(angle) = 0
    // Hence X = cotan(angle).
    AnglePredInterpolate_C(top, top + 4, cot, dst);
  } else if (angle_idx < WP2::kAngle_135) {
    // (-1 + X-0)*sin(angle) + (-1-0)*cos(angle) = 0
    // Hence X = cotan(angle) + 1.
    AnglePredInterpolate_C(top - 4, top, cot + kOne, dst);
  } else {
    // (-1-0)*sin(angle) + (0-Y-0)*cos(angle) = 0
    // Hence Y = -tan(angle).
    AnglePredInterpolate_C(left, top - 4, -kTanTable[angle_idx], dst);
  }
}
}  // namespace WP2L

//------------------------------------------------------------------------------

namespace {

// Weighted distance between predicted pixel (x0, y0) and context pixel (x, y).
// The larger 'strength' is, the more faster the weight decreases with distance.
constexpr float kMinWeight = 0.01f;     // "too-far" threshold
float WDistance(int32_t x0, int32_t y0, int32_t x, int32_t y, float strength) {
  // Threshold below which strength is considered flat.
  constexpr float kStrengthThreshold = 0.01f;
  if (strength <= kStrengthThreshold) return 1.f;
  const int32_t d2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
  if (d2 == 0) return 0.;
  const float weight = std::pow(d2, -6.f * strength);
  return (weight < kMinWeight) ? kMinWeight : weight;
}

inline float WDistance(int32_t x0, int32_t y0, int32_t x, int32_t y,
                       int32_t w, const WP2::LargeWeightTable table) {
  static constexpr int32_t limit = WP2::LargeWeightTableDim - 1u;
  const int32_t dx = std::min(std::abs(x - x0), limit);
  const int32_t dy = std::min(std::abs(y - y0), limit);
  const float weight = table[dx + WP2::LargeWeightTableDim * dy];
  if (weight == kMinWeight) {
    const int32_t min_distance = std::min({x0 + 1, w - x0, y0 + 1});
    // If the weight is low but this is the closest we'll ever be anyway,
    // set a weight of 1, so the pixel is the average of its closest neighbors.
    // Otherwise, ignore this context pixel completely.
    return (dx == min_distance || dy == min_distance) ? 1.f : kMinWeight;
  }
  return weight;
}

int16_t WeightedSum(float weights[], const int16_t ctx[], uint32_t size,
                    int16_t min_value, int16_t max_value) {
  float sum = 0.f, v = 0.f;
  for (uint32_t m = 0; m < size; ++m) {
    v += ctx[m] * weights[m];
    sum += weights[m];
  }
  assert(sum > 0.f);
  return WP2::Clamp((int16_t)(v / sum), min_value, max_value);
}

}  // namespace

namespace WP2 {

// Table-based version
void PrecomputeLargeWeightTable(float strength, LargeWeightTable table) {
  for (uint32_t y = 0; y < LargeWeightTableDim; ++y) {
    for (uint32_t x = 0; x < LargeWeightTableDim; ++x) {
      table[x + y * LargeWeightTableDim] = WDistance(0, 0, x, y, strength);
    }
  }
}

uint32_t ComputeFuseWeights(uint32_t w, uint32_t h, int32_t x, int32_t y,
                            const LargeWeightTable table,
                            float weights[kMaxContextSize]) {
  uint32_t m = 0;
  // left
  for (uint32_t j = 0; j < h; ++j) {
    weights[m++] = WDistance(x, y, -1, h - 1 - j, w, table);
  }
  // top-left
  weights[m++] = WDistance(x, y, -1, -1, w, table);
  // top
  for (uint32_t i = 0; i < w; ++i) {
    weights[m++] = WDistance(x, y, i, -1, w, table);
  }
  // top-right
  weights[m++] = WDistance(x, y, w, -1, w, table);
  // right
  for (uint32_t j = 0; j < h; ++j) {
    weights[m++] = WDistance(x, y, w, j, w, table);
  }
  assert(m <= kMaxContextSize);
  return m;
}

void BaseFusePredictor(const LargeWeightTable table, const int16_t ctx[],
                       uint32_t bw, uint32_t bh, int16_t* dst, size_t step,
                       int16_t min_value, int16_t max_value) {
  float weights[kMaxContextSize];
  for (uint32_t idx = 0, y = 0; y < bh; ++y) {
    for (uint32_t x = 0; x < bw; ++x, ++idx) {
      const uint32_t num_weights =
          ComputeFuseWeights(bw, bh, x, y, table, weights);
      dst[x] = WeightedSum(weights, ctx, num_weights, min_value, max_value);
    }
    dst += step;
  }
}

}  // namespace WP2

//------------------------------------------------------------------------------
// Add / Sub predictions

namespace {

void AddRow_C(const int16_t src[], const int16_t res[],
              int32_t min, int32_t max, int16_t dst[], uint32_t len) {
  assert((len & 3) == 0);
  for (uint32_t x = 0; x < len; ++x) {
    dst[x] = WP2::Clamp(res[x] + src[x], min, max);
  }
}
void AddRowEq_C(const int16_t res[], int16_t dst[],
                int32_t min, int32_t max, uint32_t len) {
  assert((len & 3) == 0);
  for (uint32_t x = 0; x < len; ++x) {
    dst[x] = WP2::Clamp(dst[x] + res[x], min, max);
  }
}

void SubtractRow_C(const int16_t src[], const int16_t pred[],
                   int16_t dst[], uint32_t len) {
  assert((len & 3) == 0);
  for (uint32_t x = 0; x < len; ++x) dst[x] = src[x] - pred[x];
}

#define ADD_SUB_BLOCK_FUNCS_DCL(WIDTH, ADD_FUNC, ADDEQ_FUNC, SUB_FUNC, EXT) \
void AddBlock_## WIDTH ## EXT(const int16_t src[], uint32_t src_step,       \
                              const int16_t res[], uint32_t res_step,       \
                              int32_t min, int32_t max,                     \
                              int16_t dst[], uint32_t dst_step,             \
                              uint32_t height) {                            \
  for (uint32_t y = 0; y < height; ++y) {                                   \
    ADD_FUNC(src, res, min, max, dst, WIDTH);                               \
    src += src_step;                                                        \
    res += res_step;                                                        \
    dst += dst_step;                                                        \
  }                                                                         \
}                                                                           \
void AddBlockEq_## WIDTH ## EXT(const int16_t res[], uint32_t res_step,     \
                                int16_t dst[], uint32_t dst_step,           \
                                int32_t min, int32_t max,                   \
                                uint32_t height) {                          \
  for (uint32_t y = 0; y < height; ++y) {                                   \
    ADDEQ_FUNC(res, dst, min, max, WIDTH);                                  \
    res += res_step;                                                        \
    dst += dst_step;                                                        \
  }                                                                         \
}                                                                           \
void SubBlock_## WIDTH ## EXT(const int16_t src[], uint32_t src_step,       \
                              const int16_t pred[], uint32_t pred_step,     \
                              int16_t dst[], uint32_t dst_step,             \
                              uint32_t height) {                            \
  for (uint32_t y = 0; y < height; ++y) {                                   \
    SUB_FUNC(src, pred, dst, WIDTH);                                        \
    src += src_step;                                                        \
    pred += pred_step;                                                      \
    dst += dst_step;                                                        \
  }                                                                         \
}

ADD_SUB_BLOCK_FUNCS_DCL( 4, AddRow_C, AddRowEq_C, SubtractRow_C, _C)
ADD_SUB_BLOCK_FUNCS_DCL( 8, AddRow_C, AddRowEq_C, SubtractRow_C, _C)
ADD_SUB_BLOCK_FUNCS_DCL(16, AddRow_C, AddRowEq_C, SubtractRow_C, _C)
ADD_SUB_BLOCK_FUNCS_DCL(32, AddRow_C, AddRowEq_C, SubtractRow_C, _C)

//------------------------------------------------------------------------------
// Chroma-From-Luma

void CflPredict_C(const int16_t* src, int16_t a, int32_t b,
                  int16_t* dst, int16_t min, int16_t max, uint32_t len) {
  for (uint32_t x = 0; x < len; ++x) {
    dst[x] = WP2::CflScale(src[x], a, b, min, max);
  }
}

//------------------------------------------------------------------------------
// SSE4.1 implementation

#if defined(WP2_USE_SSE)

void AnglePredInterpolate_SSE(const int16_t* src, int32_t frac,
                              int16_t* dst, uint32_t len) {
  assert(frac >= 0 && frac <= kOne);
  static_assert(kAnglePredPrecision == 15, "loop is tuned for 15bit!");
  uint32_t x = 0;
  const __m128i M = _mm_set1_epi16(frac);
  for (; x + 8 <= len; x += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(src + x + 0));
    const __m128i B = _mm_loadu_si128((const __m128i*)(src + x + 1));
    const __m128i C = _mm_sub_epi16(B, A);
    // _mm_mulhi_epi16 will perform the >>16 automatically. Since we want
    // diff >> 15, we pre-multiply diff by 2: (2*diff)>>16 = diff>>15
    const __m128i D = _mm_add_epi16(C, C);   // diff *= 2
    const __m128i E = _mm_mulhi_epi16(D, M);
    const __m128i F = _mm_add_epi16(A, E);
    _mm_storeu_si128((__m128i*)(dst + x), F);
  }
  if (x < len) AnglePredInterpolate_C(src + x, frac, dst + x, len - x);
}

void AddRow_SSE(const int16_t src[], const int16_t res[],
                int32_t min, int32_t max, int16_t dst[], uint32_t len) {
  assert((len & 3) == 0);
  const __m128i m_max = _mm_set1_epi16(max);
  const __m128i m_min = _mm_set1_epi16(min);
  uint32_t x = 0;
  for (; x + 8 <= len; x += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(res + x));
    const __m128i B = _mm_loadu_si128((const __m128i*)(src + x));
    const __m128i C = _mm_adds_epi16(A, B);
    const __m128i D = _mm_min_epi16(C, m_max);
    const __m128i E = _mm_max_epi16(D, m_min);
    _mm_storeu_si128((__m128i*)(dst + x), E);
  }
  if (x < len) {
    const __m128i A = _mm_loadl_epi64((const __m128i*)(res + x));
    const __m128i B = _mm_loadl_epi64((const __m128i*)(src + x));
    const __m128i C = _mm_adds_epi16(A, B);
    const __m128i D = _mm_min_epi16(C, m_max);
    const __m128i E = _mm_max_epi16(D, m_min);
    _mm_storel_epi64((__m128i*)(dst + x), E);
  }
}

void AddRowEq_SSE(const int16_t res[], int16_t dst[],
                  int32_t min, int32_t max, uint32_t len) {
  assert((len & 3) == 0);
  const __m128i m_max = _mm_set1_epi16(max);
  const __m128i m_min = _mm_set1_epi16(min);
  uint32_t x = 0;
  for (; x + 8 <= len; x += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(res + x));
    const __m128i B = _mm_loadu_si128((const __m128i*)(dst + x));
    const __m128i C = _mm_add_epi16(A, B);
    const __m128i D = _mm_min_epi16(C, m_max);
    const __m128i E = _mm_max_epi16(D, m_min);
    _mm_storeu_si128((__m128i*)(dst + x), E);
  }
  if (x < len) {
    const __m128i A = _mm_loadl_epi64((const __m128i*)(res + x));
    const __m128i B = _mm_loadl_epi64((const __m128i*)(dst + x));
    const __m128i C = _mm_add_epi16(A, B);
    const __m128i D = _mm_min_epi16(C, m_max);
    const __m128i E = _mm_max_epi16(D, m_min);
    _mm_storel_epi64((__m128i*)(dst + x), E);
  }
}

void SubtractRow_SSE(const int16_t src[], const int16_t pred[],
                     int16_t dst[], uint32_t len) {
  assert((len & 3) == 0);
  uint32_t x = 0;
  for (; x + 8 <= len; x += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(src + x));
    const __m128i B = _mm_loadu_si128((const __m128i*)(pred + x));
    const __m128i C = _mm_sub_epi16(A, B);
    _mm_storeu_si128((__m128i*)(dst + x), C);
  }
  if (x < len) {   // left over 4 pixels
    const __m128i A = _mm_loadl_epi64((const __m128i*)(src + x));
    const __m128i B = _mm_loadl_epi64((const __m128i*)(pred + x));
    const __m128i C = _mm_sub_epi16(A, B);
    _mm_storel_epi64((__m128i*)(dst + x), C);
  }
}

ADD_SUB_BLOCK_FUNCS_DCL( 4, AddRow_SSE, AddRowEq_SSE, SubtractRow_SSE, _SSE)
ADD_SUB_BLOCK_FUNCS_DCL( 8, AddRow_SSE, AddRowEq_SSE, SubtractRow_SSE, _SSE)
ADD_SUB_BLOCK_FUNCS_DCL(16, AddRow_SSE, AddRowEq_SSE, SubtractRow_SSE, _SSE)
ADD_SUB_BLOCK_FUNCS_DCL(32, AddRow_SSE, AddRowEq_SSE, SubtractRow_SSE, _SSE)

#undef ADD_SUB_BLOCK_FUNCS_DCL

//------------------------------------------------------------------------------
// Chroma-From-Luma

void CflPredict_SSE(const int16_t* src, int16_t a, int32_t b,
                    int16_t* dst, int16_t min, int16_t max, uint32_t len) {
  assert((len & 3) == 0);
  const __m128i A = _mm_set1_epi16(a);
  const __m128i B = _mm_set1_epi32(b);
  const __m128i m = _mm_set1_epi16(min);
  const __m128i M = _mm_set1_epi16(max);
  uint32_t x = 0;
  for (; x + 8 <= len; x += 8) {
    const __m128i v0 = _mm_loadu_si128((const __m128i*)(src + x));
    const __m128i v1l = _mm_mullo_epi16(v0, A);
    const __m128i v1h = _mm_mulhi_epi16(v0, A);
    const __m128i v2l = _mm_unpacklo_epi16(v1l, v1h);
    const __m128i v2h = _mm_unpackhi_epi16(v1l, v1h);
    const __m128i v3l = _mm_add_epi32(v2l, B);
    const __m128i v3h = _mm_add_epi32(v2h, B);
    const __m128i v4l = _mm_srai_epi32(v3l, WP2::kCflFracBits);
    const __m128i v4h = _mm_srai_epi32(v3h, WP2::kCflFracBits);
    const __m128i v5 = _mm_packs_epi32(v4l, v4h);
    const __m128i v6 = _mm_min_epi16(M, _mm_max_epi16(m, v5));
    _mm_storeu_si128((__m128i*)(dst + x), v6);
  }
  if (x < len) {   // 4px left-over?
    const __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + x));
    const __m128i v1l = _mm_mullo_epi16(v0, A);
    const __m128i v1h = _mm_mulhi_epi16(v0, A);
    const __m128i v2l = _mm_unpacklo_epi16(v1l, v1h);
    const __m128i v3l = _mm_add_epi32(v2l, B);
    const __m128i v4l = _mm_srai_epi32(v3l, WP2::kCflFracBits);
    const __m128i v5 = _mm_packs_epi32(v4l, v4l);
    const __m128i v6 = _mm_min_epi16(M, _mm_max_epi16(m, v5));
    _mm_storel_epi64((__m128i*)(dst + x), v6);
  }
}

//------------------------------------------------------------------------------

WP2_TSAN_IGNORE_FUNCTION void PredictionInitSSE() {
  WP2::AnglePredInterpolate = AnglePredInterpolate_SSE;
  WP2::SubtractRow = SubtractRow_SSE;
  WP2::AddRow = AddRow_SSE;

  WP2::SubtractBlock[0] = SubBlock_4_SSE;
  WP2::SubtractBlock[1] = SubBlock_8_SSE;
  WP2::SubtractBlock[2] = SubBlock_16_SSE;
  WP2::SubtractBlock[3] = SubBlock_32_SSE;
  WP2::AddBlock[0] = AddBlock_4_SSE;
  WP2::AddBlock[1] = AddBlock_8_SSE;
  WP2::AddBlock[2] = AddBlock_16_SSE;
  WP2::AddBlock[3] = AddBlock_32_SSE;
  WP2::AddBlockEq[0] = AddBlockEq_4_SSE;
  WP2::AddBlockEq[1] = AddBlockEq_8_SSE;
  WP2::AddBlockEq[2] = AddBlockEq_16_SSE;
  WP2::AddBlockEq[3] = AddBlockEq_32_SSE;

  WP2::CflPredict = CflPredict_SSE;
}

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------

}  // namespace

namespace WP2 {

LPredictF BasePredictors[BPRED_LAST];

void (*AnglePredInterpolate)(const int16_t* src, int32_t frac,
                             int16_t* dst, uint32_t len) = nullptr;

void (*SubtractRow)(const int16_t src[], const int16_t pred[],
                    int16_t dst[], uint32_t len) = nullptr;

void (*AddRow)(const int16_t src[], const int16_t res[],
               int32_t min, int32_t max,
               int16_t dst[], uint32_t len) = nullptr;

SubtractBlockFunc SubtractBlock[4] = { nullptr };
AddBlockFunc AddBlock[4] = { nullptr };
AddBlockEqFunc AddBlockEq[4] = { nullptr };

void (*CflPredict)(const int16_t* src, int16_t a, int32_t b, int16_t* dst,
                   int16_t min, int16_t max, uint32_t len) = nullptr;

static volatile WP2CPUInfo intra_last_cpuinfo_used =
    (WP2CPUInfo)&intra_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void PredictionInit() {
  if (intra_last_cpuinfo_used == WP2GetCPUInfo) return;

  BasePredictors[BPRED_DC] = DC_C;
  BasePredictors[BPRED_DC_L] = DC_L_C;
  BasePredictors[BPRED_DC_T] = DC_T_C;
  BasePredictors[BPRED_MEDIAN_DC] = MEDIAN_DC_C;
  BasePredictors[BPRED_MEDIAN_DC_L] = MEDIAN_DC_L_C;
  BasePredictors[BPRED_MEDIAN_DC_T] = MEDIAN_DC_T_C;
  BasePredictors[BPRED_SMOOTH] = Smooth_C;
  BasePredictors[BPRED_SMOOTH_H] = Smooth_H_C;
  BasePredictors[BPRED_SMOOTH_V] = Smooth_V_C;
  BasePredictors[BPRED_TM] = TrueMotion_C;

  AnglePredInterpolate = AnglePredInterpolate_C;

  SubtractRow = SubtractRow_C;
  AddRow = AddRow_C;

  SubtractBlock[0] = SubBlock_4_C;
  SubtractBlock[1] = SubBlock_8_C;
  SubtractBlock[2] = SubBlock_16_C;
  SubtractBlock[3] = SubBlock_32_C;
  AddBlock[0] = AddBlock_4_C;
  AddBlock[1] = AddBlock_8_C;
  AddBlock[2] = AddBlock_16_C;
  AddBlock[3] = AddBlock_32_C;
  AddBlockEq[0] = AddBlockEq_4_C;
  AddBlockEq[1] = AddBlockEq_8_C;
  AddBlockEq[2] = AddBlockEq_16_C;
  AddBlockEq[3] = AddBlockEq_32_C;

  CflPredict = CflPredict_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) PredictionInitSSE();
#endif
  }
  intra_last_cpuinfo_used = WP2GetCPUInfo;
}

}  // namespace WP2

//------------------------------------------------------------------------------
