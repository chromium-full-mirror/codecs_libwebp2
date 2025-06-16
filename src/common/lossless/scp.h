// Copyright (c) the JPEG XL Project
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
// Forked from https://github.com/google/pik/blob/master/pik/lossless8.cc
// at 16268ef512a65b541c7b5e485468a7ed33bc13d8

#ifndef WP2_COMMON_LOSSLESS_SCP_H_
#define WP2_COMMON_LOSSLESS_SCP_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

namespace WP2L {
namespace scp {

////////////////////////////////////////////////////////////////////////////////
// Data structure that stores pixel information by planes, in order ARGB.

class ImagePlanes {
 public:
  WP2Status Create(size_t xsize, size_t ysize, bool has_alpha) {
    width_ = xsize;
    height_ = ysize;
    for (uint32_t c = has_alpha ? 0 : 1; c < 4; ++c) {
      WP2_CHECK_ALLOC_OK(planes_[c].resize(xsize * ysize));
    }
    return WP2_STATUS_OK;
  }
  int16_t* PlaneRow(uint32_t c, uint32_t y) {
    return planes_[c].data() + y * width_;
  }
  const int16_t* PlaneRow(uint32_t c, uint32_t y) const {
    return planes_[c].data() + y * width_;
  }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  bool has_alpha() const { return !planes_[0].empty(); }

 private:
  size_t width_;
  size_t height_;
  // Planes are in order ARGB.
  std::array<WP2::Vector_s16, 4> planes_;
};

////////////////////////////////////////////////////////////////////////////////
// Constants for the classical Self Correcting Predictor.

// Multiple prediction modes, each has its own 4 predictors and heuristics.
// Auto is for encoding only.
enum class PredictionMode { Regular, West, North, Auto, Num = Auto };

inline constexpr int kMmulWeights0and1_R[] = {
    34, 36,  // when errors are small,
    31, 37,  // we assume they are random noise,
    33, 37,  // and penalize predictors 0 and 1
    36, 40,  //
    39, 44,  //
    42, 46,  //
    43, 47,  //
    43, 42,  //
};

// Weights for the different predictors "te" stands for the true error weights.
// The notation is R for regular, W for west, N for north, E for east, EE for
// east-east, etc.
inline constexpr int kMulWeights3teNE_R[] = {
    28, 0, 24, 15, 24, 19, 24, 16, 23, 12, 23, 12, 25, 11, 32, 11,
};

inline constexpr int kMulWeights0and1_W[] = {
    27, 31,  // when errors are small,
    33, 31,  // we assume they are random noise,
    40, 34,  // and penalize predictors 0 and 1
    43, 36, 52, 43, 59, 45, 63, 43, 65, 28,
};

inline constexpr int kMulWeights3teNE_W[] = {
    31, 0, 31, 21, 29, 19, 28, 13, 26, 14, 28, 24, 32, 26, 43, 35,
};

inline constexpr int kMulWeights0and1_N[] = {
    43, 23,  // when errors are small,
    38, 21,  // we assume they are random noise,
    35, 24,  // and penalize predictors 0 and 1
    34, 27, 35, 29, 33, 31, 28, 31, 23, 31,
};

inline constexpr int kMulWeights3teNE_N[] = {
    27, 0, 23, 29, 26, 34, 29, 29, 30, 13, 35, 13, 40, 11, 51, 9,
};

inline constexpr int kWithSign = 7;
inline constexpr int kNumContexts = 8 + kWithSign + 2;
inline constexpr int kMaxError = 101;
inline constexpr int kMaxSumErrors = kMaxError * 7 + 1;

////////////////////////////////////////////////////////////////////////////////
// Constants for the JXL Self Correcting Predictor.

struct JxlHeader {
  int16_t p1c = 0, p2c = 0, p3ca = 0, p3cb = 0, p3cc = 0, p3cd = 0, p3ce = 0;
  std::array<uint32_t, 4> w;
};

inline constexpr std::array<JxlHeader, 4> kJxlHeaders = {{
    //  default lossless8 predictor
    {.p1c = 8,
     .p2c = 8,
     .p3ca = 4,
     .p3cb = 0,
     .p3cc = 3,
     .p3cd = 23,
     .p3ce = 2,
     .w = {0xd, 0xc, 0xc, 0xb}},
    //  west lossless8 predictor
    {.p1c = 10,
     .p2c = 9,
     .p3ca = 7,
     .p3cb = 0,
     .p3cc = 0,
     .p3cd = 16,
     .p3ce = 9,
     .w = {0xd, 0xc, 0xd, 0xc}},
    //  north lossless8 predictor
    {.p1c = 16,
     .p2c = 8,
     .p3ca = 0,
     .p3cb = 16,
     .p3cc = 0,
     .p3cd = 23,
     .p3ce = 0,
     .w = {0xd, 0xd, 0xc, 0xc}},
    // ~ north lossless8 predictor
    {.p1c = 10,
     .p2c = 10,
     .p3ca = 5,
     .p3cb = 5,
     .p3cc = 5,
     .p3cd = 12,
     .p3ce = 4,
     .w = {0xd, 0xc, 0xc, 0xc}},
}};

////////////////////////////////////////////////////////////////////////////////
// Class that stores the state of the Self Correcting Predictor and is able to
// do predictions.
class State {
 public:
  WP2Status Init(size_t width, size_t height);

  WP2Status Reset();

 public:  // Only for testing.
  // p = prediction, v = true pixel value, m = inclusive maximum true pixel
  // value
  static int16_t CalcDistanceFromPredictionAndTPV(int16_t p, int16_t v,
                                                  int16_t min, int16_t max);
  // d =distance
  static int16_t CalcTPVFromPredictionAndDistance(int16_t p, int16_t d,
                                                  int16_t min, int16_t max);

 protected:
  inline constexpr static size_t kNumPredictors = 4;
  // Multiplier (by shift) of pixel, prediction and error values, to perform
  // prediction computations in higher precision.
  static inline constexpr int kPredExtraBits = 3;
  static inline constexpr int kPredictionRound =
      ((1 << kPredExtraBits) >> 1) - 1;

  // Common API. Kept here for now as that function could be the same for both
  // methods.
  template <bool USE_JXL>
  inline void UpdateErrors(size_t x, int16_t val_no_added_bits, int q) {
    const size_t xc = cur_row_ + x, xp = prev_row_ + x;
    const int16_t val = WP2::LeftShift(val_no_added_bits, kPredExtraBits);
    error_[xc] = pred_ - val;
    q = Quantized(q);
    quantized_error_[xc] = q;
    for (size_t i = 0; i < kNumPredictors; i++) {
      if (USE_JXL) {
        const uint16_t err =
            (std::abs(prediction_[i] - val) + kPredictionRound) >>
            kPredExtraBits;
        // For predicting in the next row.
        pred_errors_[i][xc] = err;
      } else {
        pred_errors_[i][xc] = DiffToError(val_no_added_bits, prediction_[i]);
      }
      // Add the error on this pixel to the error on the NE pixel. This has the
      // effect of adding the error on this pixel to the E and EE pixels.
      pred_errors_[i][1 + xp] += pred_errors_[i][xc];
    }
  }

  inline static constexpr int16_t WeightedAverage(
      const std::array<int16_t, kNumPredictors>& p,
      const std::array<int32_t, kNumPredictors>& w) {
    int64_t weight_sum = 0, sum = 0;
    for (size_t i = 0; i < kNumPredictors; i++) {
      weight_sum += w[i];
      sum += (int64_t)p[i] * w[i];
    }
    return WP2::DivRound(sum, weight_sum);
  }

  //////////////////////////////////////////////////////////////////////////////

  inline void StartProcessingLine(size_t y, const int16_t* row) {
    std::swap(cur_row_, prev_row_);
    row_pp_ = y == 1 ? row_ : row_p_;
    row_p_ = row_;
    row_ = row;
  }

  inline int Quantized(int x) const {
    return static_cast<size_t>(x) >= quantized_table_.size()
               ? 14
               : quantized_table_[x];
  }

  // Their range is -255...510 rather than 0...255!
  // And -510..510 after subtracting truePixelValue
  std::array<int16_t, kNumPredictors> prediction_;

  int16_t pred_;  // prediction with added bits

  // Errors of the 4 predictors for current and previous scanline.
  // Range 0..kMaxError
  WP2::Vector_u16 pred_errors_[kNumPredictors];
  WP2::Vector_s16 error_;           // true errors, with added bits
  WP2::Vector_u8 quantized_error_;  // The range is 0...14, all are
                                    // even due to quantizedInit()

  int16_t min_tpv_, max_tpv_;  // Tpv is truePixelValue, left shifted by PBits
  size_t last_x_;              // xsize - 1, for prediction borders

  // Beginning of the current and previous row in the temporary buffers.
  size_t cur_row_, prev_row_;
  // The current row of the image for prediction.
  const int16_t* WP2_RESTRICT row_;
  // Previous rows of the image for prediction.
  const int16_t *WP2_RESTRICT row_p_, *WP2_RESTRICT row_pp_;

  size_t width_, height_;

 private:
  static inline constexpr size_t kBufferPadding = 2;

  // Classical specific. Kept here for now.
  uint8_t DiffToError(int16_t tpv, int16_t prediction) const;

  uint8_t diff_to_error_[512 * 2];

  // Heuristics table for prediction.
  std::array<uint8_t, 29> quantized_table_;  // const
};

////////////////////////////////////////////////////////////////////////////////
// State for the classical Self Correcting Predictor.
class ClassicalState : public State {
 public:
  static int GetNumContexts(int maxerr_shift);
  WP2Status Init(size_t width, size_t height);

 protected:
  int16_t PredictY0(size_t x, uint8_t* ctxt);
  int16_t PredictX0(uint8_t* ctxt);
  int16_t PredictRegular(size_t x, uint8_t* ctxt);
  int16_t PredictWest(size_t x, uint8_t* ctxt);
  int16_t PredictNorth(size_t x, uint8_t* ctxt);

 private:
  inline uint16_t ErrorToWeight(uint16_t error) const {
    if (error < std::size(error_to_weight_)) return error_to_weight_[error];
    return 150 * 512 / (58 + error * std::sqrt(error + 50));
  }

  uint16_t error_to_weight_[kMaxSumErrors];
};

////////////////////////////////////////////////////////////////////////////////
// State for the JXL Self Correcting Predictor.
class JXLState : public State {
 public:
  void SetHeaderIndex(size_t index);
  void SetHeader(const JxlHeader& header);
  static int GetNumContexts();

 protected:
  int16_t Predict(size_t x, uint8_t* ctxt);

 private:
  // Allows to approximate division by a number from 1 to 64.
  //  for (int i = 0; i < 64; i++) divlookup[i] = (1 << 24) / (i + 1);
  static inline constexpr uint32_t kDivLookup[64] = {
      16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152,
      1864135,  1677721, 1525201, 1398101, 1290555, 1198372, 1118481, 1048576,
      986895,   932067,  883011,  838860,  798915,  762600,  729444,  699050,
      671088,   645277,  621378,  599186,  578524,  559240,  541200,  524288,
      508400,   493447,  479349,  466033,  453438,  441505,  430185,  419430,
      409200,   399457,  390167,  381300,  372827,  364722,  356962,  349525,
      342392,   335544,  328965,  322638,  316551,  310689,  305040,  299593,
      294337,   289262,  284359,  279620,  275036,  270600,  266305,  262144};

  inline constexpr static int32_t kCutoffs[] = {
      -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
      -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
      15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};

  // Approximates 4+(maxweight<<24)/(x+1), avoiding division
  inline static constexpr uint32_t ErrorWeight(uint64_t x, uint32_t maxweight) {
    int shift = static_cast<int>(std::floor(std::log2(x + 1))) - 5;
    if (shift < 0) shift = 0;
    return 4 + ((maxweight * kDivLookup[x >> shift]) >> shift);
  }

  JxlHeader header_;
};

}  // namespace scp
}  // namespace WP2L

#endif  // WP2_COMMON_LOSSLESS_SCP_H_
