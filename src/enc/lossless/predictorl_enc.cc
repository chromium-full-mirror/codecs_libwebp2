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
#include "src/common/progress_watcher.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/dsp/math.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/symbols_enc.h"

using WP2::Vector_u32;

static const float kSpatialPredictorBias = 15.f;
static const int kPredLowEffort = 11;  // select predictor
static constexpr uint32_t kNumPredictorsBitsMask =
    (1u << WP2L::kNumPredictorsBits) - 1;

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

// Class containing one counter vector per channel.
class ChannelVector {
 public:
  const Vector_u32& operator[](int i) const { return data_[i]; }
  Vector_u32& operator[](int i) { return data_[i]; }
  void swap(ChannelVector& v) {
    for (uint32_t i = 0; i < 4; ++i) {
      data_[i].swap(v[i]);
    }
  }
  void Update(const ChannelVector& v) {
    for (uint32_t i = 0; i < 4; i++) {
      for (uint32_t j = 0; j < v[i].size(); j++) {
        data_[i][j] += v[i][j];
      }
    }
  }

 private:
  Vector_u32 data_[4];
};

float PredictionCostSpatialHistogram(const ChannelVector& accumulated,
                                     const ChannelVector& tile) {
  float retval = 0.f;
  for (uint32_t c = 0; c < 4u; ++c) {
    const float kExpValue = 0.94f;
    retval += PredictionCostSpatial(tile[c], 1, kExpValue);
    retval += CombinedShannonEntropy(tile[c].data(), accumulated[c].data(),
                                     tile[c].size());
  }
  return retval;
}

inline void UpdateHisto(const int16_t* const argb, uint32_t max_x,
                        ChannelVector* const histo_argb) {
  for (uint32_t x = 0; x < max_x; ++x) {
    for (uint32_t c = 0; c < 4u; ++c) {
      const uint32_t index =
          MakeIndex(argb[4 * x + c], (*histo_argb)[c].size() - 1);
      ++(*histo_argb)[c][index];
    }
  }
}
}  // namespace

//------------------------------------------------------------------------------
// Spatial transform functions.

// Struct containing the info about a predictor.
struct Predictor {
  Predictor(uint32_t mode_in, int8_t sub_mode_in,
            const int16_t min_values_in[4], const int16_t max_values_in[4])
      : mode(mode_in),
        sub_mode(sub_mode_in),
        type(kPredictorType[mode]),
        index(kPredictorIndex[mode]),
        min_values(min_values_in),
        max_values(max_values_in) {}
  uint32_t mode;
  int8_t sub_mode;
  PredictorType type;
  uint32_t index;
  const int16_t* min_values;
  const int16_t* max_values;
};

static void WriteSubModesImpl(const int16_t* const modes, bool needs_shift,
                              const WP2::Vector_s8& sub_modes,
                              WP2::SymbolManager* const sw,
                              WP2::ANSEncBase* const enc) {
  for (uint32_t i = 0; i < sub_modes.size(); i += 4) {
    const uint32_t mode =
        needs_shift ? (modes[i + 2] >> kNumPredictorsBits) : modes[i + 2];
    if (kPredictorType[mode] != PredictorType::Angle) continue;
    const uint32_t cluster =
        (kPredictorIndex[mode] == 0)                         ? 0
        : (kPredictorIndex[mode] == kNumPredictorsAngle - 1) ? 1
                                                             : 2;
    sw->Process(kSymbolSubMode, cluster, sub_modes[i + 2], "sub_mode", enc);
  }
}

WP2Status WriteSubModes(const int16_t* const modes, bool needs_shift,
                        const WP2::Vector_s8& sub_modes, int effort,
                        WP2::ANSEncBase* const enc,
                        WP2::ANSDictionaries* const dicts) {
  LosslessSymbolsInfo symbols_info;
  WP2_CHECK_STATUS(symbols_info.InitSubMode());
  WP2::SymbolRecorder recorder;
  const uint32_t num_pixels = sub_modes.size() / 4;
  WP2_CHECK_STATUS(recorder.Allocate(symbols_info, /*num_records=*/num_pixels));
  WP2::ANSEncNoop enc_noop;
  WriteSubModesImpl(modes, needs_shift, sub_modes, &recorder, &enc_noop);

  WP2::SymbolWriter sw;
  WP2_CHECK_STATUS(sw.Init(symbols_info, effort));
  WP2_CHECK_STATUS(sw.Allocate());
  WP2_CHECK_STATUS(sw.WriteHeader(kSymbolSubMode, num_pixels, recorder,
                                  "sub_mode", enc, dicts));
  WriteSubModesImpl(modes, needs_shift, sub_modes, &sw, enc);
  return WP2_STATUS_OK;
}

// Stores the difference between the pixel and its prediction in "out".
// In case of a lossy encoding, updates the source image to avoid propagating
// the deviation further to pixels which depend on the current pixel for their
// predictions.
static inline void GetResidual(int width, int16_t* const upper_row,
                               int16_t* const current_row, bool has_alpha,
                               bool exact, const Predictor& pred, int x_start,
                               int x_end, int y, int16_t* const out) {
  PredictorNonClampedFunc pred_non_clamped = nullptr;
  PredictorClampedFunc pred_clamped = nullptr;
  PredictorAngleFunc pred_angle = nullptr;
  switch (pred.type) {
    case PredictorType::NonClamped:
      pred_non_clamped = PredictorsNonClamped[pred.index];
      break;
    case PredictorType::Clamped:
      pred_clamped = PredictorsClamped[pred.index];
      break;
    case PredictorType::Angle:
      pred_angle = PredictorsAngle[pred.index];
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
      switch (pred.type) {
        case PredictorType::NonClamped:
          pred_non_clamped(&current_row[4 * (x - 1)], &upper_row[4 * x],
                           predict);
          break;
        case PredictorType::Clamped:
          pred_clamped(&current_row[4 * (x - 1)], &upper_row[4 * x],
                       pred.min_values, pred.max_values, predict);
          break;
        case PredictorType::Angle:
          pred_angle(/*delta=*/pred.sub_mode, &current_row[4 * (x - 1)],
                     &upper_row[4 * x], predict);
          break;
        default:
          assert(false);
      }
    }
    SubPixels(&current_row[4 * x], has_alpha, predict, residual);
    if (current_row[4 * x] == 0 && !exact) {
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
static void GetBestPredictorForTile(
    int tile_x, int tile_y, uint32_t bits, bool exact, bool use_sub_modes,
    const int16_t min_values[4], const int16_t max_values[4],
    const Buffer_s16& buffer, const int16_t* const modes,
    int16_t* const argb_scratch, ChannelVector accumulated[2],
    ChannelVector tmp_histo[2], Predictor* const pred_best,
    Predictor* const pred_wsub_best) {
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
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  int16_t* upper_row = argb_scratch;
  int16_t* current_row = &upper_row[4 * (width + 1)];
  float best_diff = std::numeric_limits<float>::max();
  *pred_best =
      Predictor(/*mode_in=*/0, /*sub_mode_in=*/0, min_values, max_values);
  *pred_wsub_best = *pred_best;
  // Need pointers to be able to swap arrays.
  ChannelVector& histo_argb = tmp_histo[0];
  ChannelVector& best_histo = tmp_histo[1];
  int16_t residuals[4 * (1 << kTransformBitsMax)];
  assert(bits >= kTransformBitsMin && bits <= kTransformBitsMax);
  assert(max_x <= (1 << kTransformBitsMax));

  // First pass: not trying sub-modes, second pass: trying them.
  // If sub-modes are used, 'modes' contains in ARGB:
  //   - 0, best mode without using sub-modes, best mode when using sub-modes,
  //     sub-mode of best mode when using sub-modes
  // If sub-modes are not used, 'modes' contains in ARGB:
  //   - 0, 0, best mode without using sub-modes, 0.
  // TODO(vrabaud) For the second pass, we should only test the top angle
  // predictors like done for lossy (for CPU optimization).
  for (uint32_t pass = 0; pass < (use_sub_modes ? 2u : 1u); ++pass) {
    if (pass == 1u) *pred_wsub_best = *pred_best;
    // Prediction modes of the left and above neighbor tiles.
    const uint32_t ind_left = 4 * (tile_y * tiles_per_row + tile_x - 1);
    const uint32_t left_mode =
        (tile_x > 0) ? (pass == 0u)
                           ? modes[ind_left + 2] & kNumPredictorsBitsMask
                           : modes[ind_left + 2] >> kNumPredictorsBits
                     : 0xff;
    const uint32_t ind_above = 4 * ((tile_y - 1) * tiles_per_row + tile_x);
    const uint32_t above_mode =
        (tile_y > 0) ? (pass == 0u)
                           ? modes[ind_above + 2] & kNumPredictorsBitsMask
                           : modes[ind_above + 2] >> kNumPredictorsBits
                     : 0xff;
    for (uint32_t mode = 0; mode < kNumPredictors; ++mode) {
      int8_t sub_mode_min = 0, sub_mode_max = 0;
      if (pass == 1) {
        if (kPredictorType[mode] != PredictorType::Angle) continue;
        if (kPredictorIndex[mode] == 0) {
          // 45 degrees.
          sub_mode_min = 0;
          sub_mode_max = 3;
        } else if (kPredictorIndex[mode] == kNumPredictorsAngle - 1) {
          // 180 degrees.
          sub_mode_min = -3;
          sub_mode_max = 0;
        } else {
          // Anything else.
          sub_mode_min = -3;
          sub_mode_max = 3;
        }
      }
      for (int8_t sub_mode = sub_mode_min; sub_mode <= sub_mode_max;
           ++sub_mode) {
        const Predictor pred(mode, sub_mode, min_values, max_values);
        int relative_y;
        for (uint32_t i = 0; i < 4; ++i) {
          std::fill(histo_argb[i].begin(), histo_argb[i].end(), 0);
        }
        if (start_y > 0) {
          // Read the row above the tile which will become the first upper_row.
          // Include a pixel to the left if it exists; include a pixel to the
          // right in all cases (wrapping to the leftmost pixel of the next row
          // if it does not exist).
          std::copy(buffer.GetPosition(context_start_x, start_y - 1),
                    buffer.GetPosition(context_start_x, start_y - 1) +
                        4 * (max_x + have_left + 1),
                    &current_row[4 * context_start_x]);
        }
        for (relative_y = 0; relative_y < max_y; ++relative_y) {
          const int y = start_y + relative_y;
          std::swap(upper_row, current_row);
          // Read current_row. Include a pixel to the left if it exists; include
          // a pixel to the right in all cases except at the bottom right corner
          // of the image (wrapping to the leftmost pixel of the next row if it
          // does not exist in the current row).
          std::copy(buffer.GetPosition(context_start_x, y),
                    buffer.GetPosition(context_start_x, y) +
                        4 * (max_x + have_left + (y + 1 < height)),
                    &current_row[4 * context_start_x]);

          GetResidual(width, upper_row, current_row, buffer.has_alpha, exact,
                      pred, start_x, start_x + max_x, y, residuals);
          UpdateHisto(residuals, max_x, &histo_argb);
        }
        float cur_diff =
            PredictionCostSpatialHistogram(accumulated[pass], histo_argb);
        // Favor keeping the areas locally similar.
        if (mode == left_mode) cur_diff -= kSpatialPredictorBias;
        if (mode == above_mode) cur_diff -= kSpatialPredictorBias;

        if (cur_diff < best_diff) {
          histo_argb.swap(best_histo);
          best_diff = cur_diff;
          if (pass == 0) {
            *pred_best = Predictor(mode, sub_mode, min_values, max_values);
          } else {
            *pred_wsub_best = Predictor(mode, sub_mode, min_values, max_values);
          }
        }
      }
    }

    accumulated[pass].Update(best_histo);
  }
}

// Converts pixels of the image to residuals with respect to predictions.
static void CopyImageWithPrediction(int bits, bool exact,
                                    const int16_t min_values[4],
                                    const int16_t max_values[4],
                                    const int16_t* const modes, int effort,
                                    const WP2::Vector_s8& sub_modes,
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

    for (uint32_t x = 0; x < width;) {
      const uint32_t x_end = std::min(x + (1 << bits), width);
      const uint32_t ind = 4 * ((y >> bits) * tiles_per_row + (x >> bits));
      const Predictor pred(
          modes[ind + 2],
          /*sub_mode_in=*/sub_modes.empty() ? 0 : sub_modes[ind + 2],
          min_values, max_values);

      GetResidual(width, upper_row, current_row, buffer->has_alpha, exact, pred,
                  x, x_end, y, buffer->GetPosition(x, y));
      x = x_end;
    }
  }
}

// Finds the best predictor for each tile, and converts the image to residuals
// with respect to predictions.
WP2Status CreateResidualImage(uint32_t bits, bool exact, bool can_use_sub_modes,
                              const int16_t min_values[4],
                              const int16_t max_values[4], int effort,
                              const WP2::ProgressRange& progress,
                              Buffer_s16* const buffer,
                              int16_t* const argb_scratch, int16_t* const modes,
                              WP2::Vector_s8* const sub_modes) {
  // Define the mode image dimensions.
  const uint32_t width = SubSampleSize(buffer->width, bits);
  const uint32_t height = SubSampleSize(buffer->height, bits);
  if (can_use_sub_modes) {
    WP2_CHECK_ALLOC_OK(sub_modes->resize(4 * width * height));
  } else {
    sub_modes->clear();
  }
  if (effort == 0) {
    for (uint32_t i = 0; i < 4 * width * height; i += 4) {
      modes[i + 0] = 0;
      modes[i + 1] = 0;
      modes[i + 2] = kPredLowEffort;
      modes[i + 3] = 0;
    }
    WP2_CHECK_STATUS(progress.AdvanceBy(1.0));
  } else {
    ChannelVector histo[2];
    ChannelVector tmp_histo[2];
    for (uint32_t c = 0; c < 4; ++c) {
      uint32_t histo_size = 0;
      for (int16_t i : {(int16_t)-min_values[c], min_values[c],
                        (int16_t)-max_values[c], max_values[c]}) {
        histo_size = std::max(histo_size, 1u + MakeIndex(i));
      }
      for (uint32_t j = 0; j < 2; ++j) {
        WP2_CHECK_ALLOC_OK(tmp_histo[j][c].resize(histo_size));
        WP2_CHECK_ALLOC_OK(histo[j][c].resize(histo_size));
        std::fill(histo[j][c].begin(), histo[j][c].end(), 0);
      }
    }
    Predictor pred(/*mode_in=*/0, /*sub_mode_in=*/0, min_values, max_values);
    Predictor pred_wsub(/*mode_in=*/0, /*sub_mode_in=*/0, min_values,
                        max_values);
    WP2::ProgressScale tile_progress(progress, 1.0 / (width * height));
    for (uint32_t tile_y = 0; tile_y < height; ++tile_y) {
      for (uint32_t tile_x = 0; tile_x < width; ++tile_x) {
        GetBestPredictorForTile(tile_x, tile_y, bits, exact, can_use_sub_modes,
                                min_values, max_values, *buffer, modes,
                                argb_scratch, histo, tmp_histo, &pred,
                                &pred_wsub);
        const uint32_t ind = 4 * (tile_y * width + tile_x);
        for (uint32_t c : {0, 1, 3}) modes[ind + c] = 0;
        // Store the modes in the green channel.
        if (can_use_sub_modes) {
          (*sub_modes)[ind + 2] = pred_wsub.sub_mode;
          // Store the two possible predictors at the same spot.
          modes[ind + 2] = (pred_wsub.mode << kNumPredictorsBits) | pred.mode;
        } else {
          modes[ind + 2] = pred.mode;
        }
        WP2_CHECK_STATUS(tile_progress.AdvanceBy(1.0));
      }
    }

    // Figure out if writing the sub modes is worth it.
    if (can_use_sub_modes) {
      // Compare the costs of storing residuals without and with sub-modes.
      float costs[2];
      for (uint32_t i = 0; i < 2; ++i) {
        costs[i] = 0.f;
        for (uint32_t c = 0; c < 4; ++c) {
          costs[i] +=
              WP2::ANSCountsCost(histo[i][c].data(), histo[i][c].size());
        }
      }
      // Compute the cost of storing the sub-modes.
      WP2::ANSEncCounter enc;
      WP2::ANSDictionaries dicts;
      WP2_CHECK_STATUS(WriteSubModes(modes, /*needs_shift=*/true, *sub_modes,
                                     effort, &enc, &dicts));

      // Compare the cost of sub-mode storage to what we gain using sub-modes.
      if (enc.GetCost(dicts) + costs[1] < costs[0]) {
        for (uint32_t i = 0; i < 4 * width * height; i += 4) {
          modes[i + 2] >>= kNumPredictorsBits;
        }
      } else {
        sub_modes->clear();
        for (uint32_t i = 0; i < 4 * width * height; i += 4) {
          modes[i + 2] &= kNumPredictorsBitsMask;
        }
      }
    }
  }

  CopyImageWithPrediction(bits, exact, min_values, max_values, modes, effort,
                          *sub_modes, argb_scratch, buffer);
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

static inline void MultipliersToColorCode(const Multipliers& m,
                                          int16_t* const res) {
  res[0] = 0;
  res[1] = m.red_to_blue;
  res[2] = m.green_to_blue;
  res[3] = m.green_to_red;
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
    int tile_width, int tile_height, const Multipliers& prev_x,
    const Multipliers& prev_y, int16_t green_to_red,
    const Vector_u32& accumulated_red_histo, Vector_u32* const tmp_histo) {
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
                              int tile_height, const Multipliers& prev_x,
                              const Multipliers& prev_y, int effort,
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
    int tile_width, int tile_height, const Multipliers& prev_x,
    const Multipliers& prev_y, int16_t green_to_blue, int16_t red_to_blue,
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
                                  int tile_height, const Multipliers& prev_x,
                                  const Multipliers& prev_y, int effort,
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
    int tile_x, int tile_y, int bits, const Multipliers& prev_x,
    const Multipliers& prev_y, int effort, int xsize, int ysize,
    const Vector_u32& accumulated_red_histo,
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
                                       const Multipliers& color_transform,
                                       Buffer_s16* buffer) {
  const int xscan = std::min(max_tile_size, buffer->width - tile_x);
  int yscan = std::min(max_tile_size, buffer->height - tile_y);
  int16_t* argb = &buffer->GetRow(tile_y)[4 * tile_x];
  while (yscan-- > 0) {
    TransformColor(color_transform, xscan, argb);
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
      MultipliersToColorCode(prev_x, &image[4 * offset]);
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
