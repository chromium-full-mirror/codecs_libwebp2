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

#include <limits>

#include "src/common/color_precision.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/dsp/math.h"
#include "src/enc/lossless/losslessi_enc.h"

using WP2::Vector_u32;

static const float kSpatialPredictorBias = 15.f;
static const int kPredLowEffort = 11;

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

namespace WP2L {
namespace {

// Computes the cost of a histogram for signed values where the index of a value
// v is defined by MakeIndex(v).
float PredictionCostSpatial(const Vector_u32& counts, int weight_0,
                            float exp_val) {
  // TODO(vrabaud) make this constant be dependent on counts.size().
  //               This will matter for 10-bit.
  constexpr int kSignificantSymbols = 16;
  const float kExpDecayFactor = 0.6f;
  float bits = weight_0 * counts[MakeIndex(0)];
  for (int i = 1; i < kSignificantSymbols; ++i) {
    const uint32_t ind1 = MakeIndex(i);
    const uint32_t ind2 = MakeIndex(-i);
    uint32_t count = (ind1 < counts.size()) ? counts[ind1] : 0u;
    count += (ind2 < counts.size()) ? counts[ind2] : 0u;
    bits += exp_val * count;
    exp_val *= kExpDecayFactor;
  }
  return -0.1f * bits;
}

float PredictionCostSpatialHistogram(const Vector_u32 accumulated[4],
                                     const Vector_u32 tile[4]) {
  int i;
  float retval = 0.f;
  for (i = 0; i < 4; ++i) {
    const float kExpValue = 0.94f;
    retval += PredictionCostSpatial(tile[i], 1, kExpValue);
    retval += CombinedShannonEntropy(tile[i].data(), accumulated[i].data(),
                                     tile[i].size());
  }
  return retval;
}

inline void UpdateHisto(Vector_u32 histo_argb[4], const int16_t* const argb,
                        uint32_t max_x) {
  for (uint32_t x = 0; x < max_x; ++x) {
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t index =
          MakeIndex(argb[4 * x + i], histo_argb[i].size() - 1);
      ++histo_argb[i][index];
    }
  }
}
}  // namespace

//------------------------------------------------------------------------------
// Spatial transform functions.

static inline void PredictBatch(int mode, int x_start, int y,
                                uint32_t num_pixels,
                                const int16_t min_values[4],
                                const int16_t max_values[4],
                                const int16_t* const current, bool has_alpha,
                                const int16_t* const upper, int16_t* out) {
  if (x_start == 0) {
    if (y == 0) {
      // ARGB_BLACK.
      PredictorsNonClampedSub[0](current, has_alpha, nullptr, 1, out);
    } else {
      // Top one.
      PredictorsNonClampedSub[2](current, has_alpha, upper, 1, out);
    }
    ++x_start;
    out += 4;
    --num_pixels;
  }
  if (y == 0) {
    // Left one.
    PredictorsNonClampedSub[1](&current[4 * x_start], has_alpha, nullptr,
                               num_pixels, out);
  } else {
    switch (kPredictorType[mode]) {
      case PredictorType::NonClamped:
        PredictorsNonClampedSub[kPredictorIndex[mode]](
            &current[4 * x_start], has_alpha, &upper[4 * x_start], num_pixels,
            out);
        break;
      case PredictorType::Clamped:
        PredictorsClampedSub[kPredictorIndex[mode]](
            &current[4 * x_start], has_alpha, &upper[4 * x_start], num_pixels,
            min_values, max_values, out);
        break;
      default:
        assert(false);
    }
  }
}

// Stores the difference between the pixel and its prediction in "out".
// In case of a lossy encoding, updates the source image to avoid propagating
// the deviation further to pixels which depend on the current pixel for their
// predictions.
static inline void GetResidual(int width, const int16_t min_values[4],
                               const int16_t max_values[4],
                               int16_t* const upper_row,
                               int16_t* const current_row, bool has_alpha,
                               int mode, int x_start, int x_end, int y,
                               int16_t* const out) {
  PredictorNonClampedFunc pred_non_clamped;
  PredictorClampedFunc pred_clamped;
  switch (kPredictorType[mode]) {
    case PredictorType::NonClamped:
      pred_non_clamped = PredictorsNonClamped[kPredictorIndex[mode]];
      break;
    case PredictorType::Clamped:
      pred_clamped = PredictorsClamped[kPredictorIndex[mode]];
      break;
    default:
      assert(false);
  }
  int x;
  for (x = x_start; x < x_end; ++x) {
    int16_t predict[4], residual[4];
    if (y == 0) {
      if (x == 0) {
        predict[0] = WP2::kAlphaMax;
        predict[1] = 0;
        predict[2] = 0;
        predict[3] = 0;
      } else {
        ColorCopy(&current_row[4 * (x - 1)], predict);  // Left.
      }
    } else if (x == 0) {
      ColorCopy(&upper_row[4 * x], predict);  // Top.
    } else {
      switch (kPredictorType[mode]) {
        case PredictorType::NonClamped:
          pred_non_clamped(&current_row[4 * (x - 1)], &upper_row[4 * x],
                           predict);
          break;
        case PredictorType::Clamped:
          pred_clamped(&current_row[4 * (x - 1)], &upper_row[4 * x], min_values,
                       max_values, predict);
          break;
        default:
          assert(false);
      }
    }
    SubPixels(&current_row[4 * x], has_alpha, predict, residual);
    if (current_row[4 * x] == 0) {
      // If alpha is 0, cleanup RGB. We can choose the RGB values of the
      // residual for best compression. The prediction of alpha itself can be
      // non-zero and must be kept though. We choose RGB of the residual to be
      // 0.
      residual[1] = 0;
      residual[2] = 0;
      residual[3] = 0;
      // Update the source image.
      current_row[4 * x + 0] = has_alpha ? 0 : WP2::kAlphaMax;
      current_row[4 * x + 1] = predict[1];
      current_row[4 * x + 2] = predict[2];
      current_row[4 * x + 3] = predict[3];
      // The prediction for the rightmost pixel in a row uses the leftmost
      // pixel in that row as its top-right context pixel. Hence if we change
      // the leftmost pixel of current_row, the corresponding change must be
      // applied to upper_row as well where top-right context is being read
      // from.
      if (x == 0 && y != 0) {
        ColorCopy(current_row, &upper_row[4 * width]);
      }
    }
    ColorCopy(residual, &out[4 * (x - x_start)]);
  }
}

// Returns best predictor and updates the accumulated histogram.
static int GetBestPredictorForTile(
    int tile_x, int tile_y, uint32_t bits, const int16_t min_values[4],
    const int16_t max_values[4], const Buffer_s16& buffer,
    const int16_t* const modes, int16_t* const argb_scratch,
    Vector_u32 accumulated[4], Vector_u32 tmp_histo[][4]) {
  const int start_x = tile_x << bits;
  const int start_y = tile_y << bits;
  const int tile_size = 1 << bits;
  const int width = buffer.width, height = buffer.height;
  const int max_y = std::min(tile_size, height - start_y);
  const int max_x = std::min(tile_size, width - start_x);
  // Whether there exist columns just outside the tile.
  const int have_left = (start_x > 0);
  // Position and size of the strip covering the tile and adjacent columns if
  // they exist.
  const int context_start_x = start_x - have_left;
  const int tiles_per_row = SubSampleSize(width, bits);
  // Prediction modes of the left and above neighbor tiles.
  const uint32_t left_mode =
      (tile_x > 0) ? modes[4 * (tile_y * tiles_per_row + tile_x - 1) + 2]
                   : 0xff;
  const uint32_t above_mode =
      (tile_y > 0) ? modes[4 * ((tile_y - 1) * tiles_per_row + tile_x) + 2]
                   : 0xff;
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  int16_t* upper_row = argb_scratch;
  int16_t* current_row = &upper_row[4 * (width + 1)];
  float best_diff = std::numeric_limits<float>::max();
  uint32_t best_mode = 0;
  uint32_t mode;
  // Need pointers to be able to swap arrays.
  Vector_u32* histo_argb = &tmp_histo[0][0];
  Vector_u32* best_histo = &tmp_histo[1][0];
  int16_t residuals[4 * (1 << kTransformBitsMax)];
  assert(bits >= kTransformBitsMin && bits <= kTransformBitsMax);
  assert(max_x <= (1 << kTransformBitsMax));

  for (mode = 0; mode < kNumPredictors; ++mode) {
    float cur_diff;
    int relative_y;
    for (uint32_t i = 0; i < 4; ++i) {
      std::fill(histo_argb[i].begin(), histo_argb[i].end(), 0);
    }
    if (start_y > 0) {
      // Read the row above the tile which will become the first upper_row.
      // Include a pixel to the left if it exists; include a pixel to the right
      // in all cases (wrapping to the leftmost pixel of the next row if it does
      // not exist).
      std::copy(buffer.GetPosition(context_start_x, start_y - 1),
                buffer.GetPosition(context_start_x, start_y - 1) +
                    4 * (max_x + have_left + 1),
                &current_row[4 * context_start_x]);
    }
    for (relative_y = 0; relative_y < max_y; ++relative_y) {
      const int y = start_y + relative_y;
      std::swap(upper_row, current_row);
      // Read current_row. Include a pixel to the left if it exists; include a
      // pixel to the right in all cases except at the bottom right corner of
      // the image (wrapping to the leftmost pixel of the next row if it does
      // not exist in the current row).
      std::copy(buffer.GetPosition(context_start_x, y),
                buffer.GetPosition(context_start_x, y) +
                    4 * (max_x + have_left + (y + 1 < height)),
                &current_row[4 * context_start_x]);

      GetResidual(width, min_values, max_values, upper_row, current_row,
                  buffer.has_alpha, mode, start_x, start_x + max_x, y,
                  residuals);
      UpdateHisto(histo_argb, residuals, max_x);
    }
    cur_diff = PredictionCostSpatialHistogram(accumulated, histo_argb);
    // Favor keeping the areas locally similar.
    if (mode == left_mode) cur_diff -= kSpatialPredictorBias;
    if (mode == above_mode) cur_diff -= kSpatialPredictorBias;

    if (cur_diff < best_diff) {
      std::swap(histo_argb, best_histo);
      best_diff = cur_diff;
      best_mode = mode;
    }
  }

  for (uint32_t i = 0; i < 4; i++) {
    for (uint32_t j = 0; j < best_histo[i].size(); j++) {
      accumulated[i][j] += best_histo[i][j];
    }
  }

  return best_mode;
}

// Converts pixels of the image to residuals with respect to predictions.
static void CopyImageWithPrediction(int bits, const int16_t min_values[4],
                                    const int16_t max_values[4],
                                    const int16_t* const modes, int effort,
                                    int16_t* const argb_scratch,
                                    Buffer_s16* const buffer) {
  const uint32_t width = buffer->width, height = buffer->height;
  const int tiles_per_row = SubSampleSize(width, bits);
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  int16_t* upper_row = argb_scratch;
  int16_t* current_row = &upper_row[4 * (width + 1)];
  for (uint32_t y = 0; y < height; ++y) {
    std::swap(upper_row, current_row);
    std::copy(buffer->GetRow(y), buffer->GetPosition(y + 1 < height, y + 1),
              current_row);

    if (effort == 0) {
      PredictBatch(kPredLowEffort, 0, y, width, min_values, max_values,
                   current_row, buffer->has_alpha, upper_row,
                   buffer->GetRow(y));
    } else {
      for (uint32_t x = 0; x < width;) {
        const int mode =
            modes[4 * ((y >> bits) * tiles_per_row + (x >> bits)) + 2];
        uint32_t x_end = x + (1 << bits);
        if (x_end > width) x_end = width;
        GetResidual(width, min_values, max_values, upper_row, current_row,
                    buffer->has_alpha, mode, x, x_end, y,
                    buffer->GetPosition(x, y));
        x = x_end;
      }
    }
  }
}

// Finds the best predictor for each tile, and converts the image to residuals
// with respect to predictions.
WP2Status CreateResidualImage(uint32_t bits, const int16_t min_values[4],
                              const int16_t max_values[4], int effort,
                              Buffer_s16* const buffer,
                              int16_t* const argb_scratch,
                              int16_t* const image) {
  const uint32_t tiles_per_row = SubSampleSize(buffer->width, bits);
  const uint32_t tiles_per_col = SubSampleSize(buffer->height, bits);
  if (effort == 0) {
    for (uint32_t i = 0; i < tiles_per_row * tiles_per_col; ++i) {
      image[4 * i + 0] = 0;
      image[4 * i + 1] = 0;
      image[4 * i + 2] = kPredLowEffort;
      image[4 * i + 3] = 0;
    }
  } else {
    Vector_u32 histo[4];
    Vector_u32 tmp_histo[2][4];
    for (uint32_t c = 0; c < 4; ++c) {
      uint32_t histo_size = 0;
      for (int16_t i : {(int16_t)-min_values[c], min_values[c],
                        (int16_t)-max_values[c], max_values[c]}) {
        histo_size = std::max(histo_size, 1u + MakeIndex(i));
      }
      WP2_CHECK_ALLOC_OK(histo[c].resize(histo_size));
      std::fill(histo[c].begin(), histo[c].end(), 0);
      for (uint32_t j = 0; j < 2; ++j) {
        WP2_CHECK_ALLOC_OK(tmp_histo[j][c].resize(histo_size));
      }
    }
    for (uint32_t tile_y = 0; tile_y < tiles_per_col; ++tile_y) {
      for (uint32_t tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
        const int pred = GetBestPredictorForTile(
            tile_x, tile_y, bits, min_values, max_values, *buffer, image,
            argb_scratch, histo, tmp_histo);
        image[4 * (tile_y * tiles_per_row + tile_x) + 0] = 0;
        image[4 * (tile_y * tiles_per_row + tile_x) + 1] = 0;
        image[4 * (tile_y * tiles_per_row + tile_x) + 2] = pred;
        image[4 * (tile_y * tiles_per_row + tile_x) + 3] = 0;
      }
    }
  }

  CopyImageWithPrediction(bits, min_values, max_values, image, effort,
                          argb_scratch, buffer);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Color transform functions.

static inline void MultipliersClear(Multipliers* const m) {
  m->green_to_red = 0;
  m->green_to_blue = 0;
  m->red_to_blue = 0;
}

static inline void ColorCodeToMultipliers(const int16_t* const color,
                                          Multipliers* const m) {
  m->green_to_red = color[3];
  m->green_to_blue = color[2];
  m->red_to_blue = color[1];
}

static inline void MultipliersToColorCode(const Multipliers* const m,
                                          int16_t* const res) {
  res[0] = 0;
  res[1] = m->red_to_blue;
  res[2] = m->green_to_blue;
  res[3] = m->green_to_red;
}

static float PredictionCostCrossColor(const Vector_u32& accumulated,
                                      const Vector_u32& counts) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const float kExpValue = 2.4f;
  assert(accumulated.size() == counts.size());
  return CombinedShannonEntropy(counts.data(), accumulated.data(),
                                counts.size()) +
         PredictionCostSpatial(counts, 3, kExpValue);
}

static float GetPredictionCostCrossColorRed(
    const int16_t* const argb, uint32_t width, uint32_t abs_value_max,
    int tile_width, int tile_height, Multipliers prev_x, Multipliers prev_y,
    int16_t green_to_red, const Vector_u32& accumulated_red_histo,
    Vector_u32* const tmp_histo) {
  std::fill(tmp_histo->begin(), tmp_histo->end(), 0);
  float cur_diff;

  if (!CollectColorRedTransforms(argb, width, tile_width, tile_height,
                                 green_to_red, abs_value_max, tmp_histo->size(),
                                 tmp_histo->data())) {
    return std::numeric_limits<float>::max();
  }

  cur_diff =
      PredictionCostCrossColor(accumulated_red_histo, *tmp_histo);
  // cur_diff -= 3 favors keeping the areas locally similar
  if (green_to_red == prev_x.green_to_red) {
    cur_diff -= 3;
  }
  if (green_to_red == prev_y.green_to_red) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_red == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

static void GetBestGreenToRed(const int16_t* argb, uint32_t width,
                              uint32_t abs_value_max, int tile_width,
                              int tile_height, Multipliers prev_x,
                              Multipliers prev_y, int effort,
                              const Vector_u32& accumulated_red_histo,
                              Multipliers* const best_tx,
                              Vector_u32* const tmp_histo) {
  const int kMaxIters = (effort < 5) ? 4 + effort * 2 / 5 : 6;  // in [4..6]
  int green_to_red_best = 0;
  int iter, offset;
  float best_diff = GetPredictionCostCrossColorRed(
      argb, width, abs_value_max, tile_width, tile_height, prev_x, prev_y,
      green_to_red_best, accumulated_red_histo, tmp_histo);
  for (iter = 0; iter < kMaxIters; ++iter) {
    // ColorTransformDelta is a 3.5 bit fixed point, so 32 is equal to
    // one in color computation. Having initial delta here as 1 is sufficient
    // to explore the range of (-2, 2).
    const int delta = 32 >> iter;
    // Try a negative and a positive delta from the best known value.
    for (offset = -delta; offset <= delta; offset += 2 * delta) {
      const int green_to_red_cur =
          WP2::Clamp(offset + green_to_red_best, -(int32_t)kGreenToRedMax,
                     (int32_t)kGreenToRedMax);
      const float cur_diff = GetPredictionCostCrossColorRed(
          argb, width, abs_value_max, tile_width, tile_height, prev_x, prev_y,
          green_to_red_cur, accumulated_red_histo, tmp_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_red_best = green_to_red_cur;
      }
    }
  }
  assert(best_diff < std::numeric_limits<float>::max());
  best_tx->green_to_red = green_to_red_best;
}

static float GetPredictionCostCrossColorBlue(
    const int16_t* const argb, int stride, uint32_t abs_value_max,
    int tile_width, int tile_height, Multipliers prev_x, Multipliers prev_y,
    int16_t green_to_blue, int16_t red_to_blue,
    const Vector_u32& accumulated_blue_histo, Vector_u32* tmp_histo) {
  std::fill(tmp_histo->begin(), tmp_histo->end(), 0);
  float cur_diff;

  if (!CollectColorBlueTransforms(argb, stride, tile_width, tile_height,
                                  green_to_blue, red_to_blue, abs_value_max,
                                  tmp_histo->size(), tmp_histo->data())) {
    return std::numeric_limits<float>::max();
  }

  cur_diff = PredictionCostCrossColor(accumulated_blue_histo, *tmp_histo);
  // cur_diff -= 3 favors keeping the areas locally similar
  if (green_to_blue == prev_x.green_to_blue) {
    cur_diff -= 3;
  }
  if (green_to_blue == prev_y.green_to_blue) {
    cur_diff -= 3;
  }
  if (red_to_blue == prev_x.red_to_blue) {
    cur_diff -= 3;
  }
  if (red_to_blue == prev_y.red_to_blue) {
    cur_diff -= 3;
  }
  if (green_to_blue == 0) {
    cur_diff -= 3;
  }
  if (red_to_blue == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

static constexpr uint32_t kGreenRedToBlueNumAxis = 8;
static constexpr uint32_t kGreenRedToBlueMaxIters = 7;
static void GetBestGreenRedToBlue(const int16_t* argb, uint32_t width,
                                  uint32_t abs_value_max, int tile_width,
                                  int tile_height, Multipliers prev_x,
                                  Multipliers prev_y, int effort,
                                  const Vector_u32& accumulated_blue_histo,
                                  Multipliers* const best_tx,
                                  Vector_u32* tmp_histo) {
  const int8_t offset[kGreenRedToBlueNumAxis][2] =
      {{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int8_t delta_lut[kGreenRedToBlueMaxIters] = { 16, 16, 8, 4, 2, 2, 2 };
  const int iters =
      (effort < 3) ? 1 : (effort >= 5) ? kGreenRedToBlueMaxIters : 4;
  int green_to_blue_best = 0;
  int red_to_blue_best = 0;
  int iter;
  // Initial value at origin:
  float best_diff = GetPredictionCostCrossColorBlue(
      argb, width, abs_value_max, tile_width, tile_height, prev_x, prev_y,
      green_to_blue_best, red_to_blue_best, accumulated_blue_histo, tmp_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    for (uint32_t axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          WP2::Clamp(offset[axis][0] * delta + green_to_blue_best,
                     -(int32_t)kGreenToBlueMax, (int32_t)kGreenToBlueMax);
      const int red_to_blue_cur =
          WP2::Clamp(offset[axis][1] * delta + red_to_blue_best,
                     -(int32_t)kRedToBlueMax, (int32_t)kRedToBlueMax);
      const float cur_diff = GetPredictionCostCrossColorBlue(
          argb, width, abs_value_max, tile_width, tile_height, prev_x, prev_y,
          green_to_blue_cur, red_to_blue_cur, accumulated_blue_histo,
          tmp_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_blue_best = green_to_blue_cur;
        red_to_blue_best = red_to_blue_cur;
      }
      if (effort < 3 && iter == 4) {
        // Only axis aligned diffs for lower effort.
        break;  // next iter.
      }
    }
    if (delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0) {
      // Further iterations would not help.
      break;  // out of iter-loop.
    }
  }
  assert(best_diff < std::numeric_limits<float>::max());
  best_tx->green_to_blue = green_to_blue_best;
  best_tx->red_to_blue = red_to_blue_best;
}

static Multipliers GetBestColorTransformForTile(
    int tile_x, int tile_y, int bits, Multipliers prev_x, Multipliers prev_y,
    int effort, int xsize, int ysize, const Vector_u32& accumulated_red_histo,
    const Vector_u32& accumulated_blue_histo, const int16_t* const argb,
    uint32_t abs_value_max, Vector_u32* const tmp_histo) {
  const int max_tile_size = 1 << bits;
  const int tile_y_offset = tile_y * max_tile_size;
  const int tile_x_offset = tile_x * max_tile_size;
  const int all_x_max = std::min(tile_x_offset + max_tile_size, xsize);
  const int all_y_max = std::min(tile_y_offset + max_tile_size, ysize);
  const int tile_width = all_x_max - tile_x_offset;
  const int tile_height = all_y_max - tile_y_offset;
  const int16_t* const tile_argb =
      &argb[4 * (tile_y_offset * xsize + tile_x_offset)];
  Multipliers best_tx;
  MultipliersClear(&best_tx);

  GetBestGreenToRed(tile_argb, xsize, abs_value_max, tile_width, tile_height,
                    prev_x, prev_y, effort, accumulated_red_histo, &best_tx,
                    tmp_histo);
  GetBestGreenRedToBlue(tile_argb, xsize, abs_value_max, tile_width,
                        tile_height, prev_x, prev_y, effort,
                        accumulated_blue_histo, &best_tx, tmp_histo);
  return best_tx;
}

static void CopyTileWithColorTransform(int tile_x, int tile_y,
                                       uint32_t max_tile_size,
                                       Multipliers color_transform,
                                       Buffer_s16* buffer) {
  const int xscan = std::min(max_tile_size, buffer->width - tile_x);
  int yscan = std::min(max_tile_size, buffer->height - tile_y);
  int16_t* argb = &buffer->GetRow(tile_y)[4 * tile_x];
  while (yscan-- > 0) {
    TransformColor(&color_transform, xscan, argb);
    argb += 4 * buffer->width;
  }
}

WP2Status ColorSpaceTransform(uint32_t bits, int effort,
                              Buffer_s16* const buffer, int16_t* image) {
  const int max_tile_size = 1 << bits;
  const uint32_t width = buffer->width, height = buffer->height;
  const int tile_xsize = SubSampleSize(width, bits);
  const int tile_ysize = SubSampleSize(height, bits);
  int tile_x, tile_y;
  int16_t* data = buffer->data;
  Multipliers prev_x, prev_y;
  MultipliersClear(&prev_y);
  MultipliersClear(&prev_x);
  Vector_u32 tmp_histo, accumulated_red_histo, accumulated_blue_histo;
  // We have deltas up to several times a color. In order to limit the size of
  // the histograms, we restrict to +1/-1 the color.
  const int32_t channel_max = (1 << buffer->channel_bits) - 1;
  const int32_t abs_value_max =
      (buffer->channel_bits == 8) ? kMaxAbsResidual8 : kMaxAbsResidual10;
  const uint32_t histo_size =
      1 + std::max(MakeIndex(channel_max), MakeIndex(-channel_max));
  WP2_CHECK_ALLOC_OK(tmp_histo.resize(histo_size));
  WP2_CHECK_ALLOC_OK(accumulated_red_histo.resize(histo_size));
  std::fill(accumulated_red_histo.begin(), accumulated_red_histo.end(), 0);
  WP2_CHECK_ALLOC_OK(accumulated_blue_histo.resize(histo_size));
  std::fill(accumulated_blue_histo.begin(), accumulated_blue_histo.end(), 0);
  for (tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      int y;
      const uint32_t tile_x_offset = tile_x * max_tile_size;
      const uint32_t tile_y_offset = tile_y * max_tile_size;
      const int all_x_max = std::min(tile_x_offset + max_tile_size, width);
      const int all_y_max = std::min(tile_y_offset + max_tile_size, height);
      const int offset = tile_y * tile_xsize + tile_x;
      if (tile_y != 0) {
        ColorCodeToMultipliers(&image[4 * (offset - tile_xsize)], &prev_y);
      }
      prev_x = GetBestColorTransformForTile(
          tile_x, tile_y, bits, prev_x, prev_y, effort, width, height,
          accumulated_red_histo, accumulated_blue_histo, data, abs_value_max,
          &tmp_histo);
      MultipliersToColorCode(&prev_x, &image[4 * offset]);
      CopyTileWithColorTransform(tile_x_offset, tile_y_offset, max_tile_size,
                                 prev_x, buffer);

      // Gather accumulated histogram data.
      for (y = tile_y_offset; y < all_y_max; ++y) {
        uint32_t ix = y * width + tile_x_offset;
        const uint32_t ix_end = ix + all_x_max - tile_x_offset;
        for (; ix < ix_end; ++ix) {
          if (ix >= 2 && ColorEq(&data[4 * ix], &data[4 * (ix - 2)]) &&
              ColorEq(&data[4 * ix], &data[4 * (ix - 1)])) {
            continue;  // repeated pixels are handled by backward references
          }
          if (ix >= width + 2 &&
              ColorEq(&data[4 * (ix - 2)], &data[4 * (ix - width - 2)]) &&
              ColorEq(&data[4 * (ix - 1)], &data[4 * (ix - width - 1)]) &&
              ColorEq(&data[4 * ix], &data[4 * (ix - width)])) {
            continue;  // repeated pixels are handled by backward references
          }
          ++accumulated_red_histo[MakeIndex(data[4 * ix + 1], histo_size - 1)];
          ++accumulated_blue_histo[MakeIndex(data[4 * ix + 3], histo_size - 1)];
        }
      }
    }
  }

  return WP2_STATUS_OK;
}
}  // namespace WP2L
