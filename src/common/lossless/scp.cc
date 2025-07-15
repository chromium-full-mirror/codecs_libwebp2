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

#include "src/common/lossless/scp.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/common/lossless/plane.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

namespace {

constexpr inline int quantizedInit(int x) {
  assert(0 <= x && x <= 255);
  x = (x + 1) >> 1;
  int res = (x >= 4 ? 4 : x);
  if (x >= 6) res = 5;
  if (x >= 9) res = 6;
  if (x >= 15) res = 7;
  return res * 2;
}

}  // namespace

namespace WP2L {
namespace scp {

////////////////////////////////////////////////////////////////////////////////
// ScpState.

WP2Status State::Init(uint32_t width, uint32_t height) {
  WP2_CHECK_STATUS(PlaneCodec::Init(width, height));

  WP2_CHECK_STATUS(Reset());

  // Cache some constants.
  for (int j = -512; j <= 511; ++j) {
    diff_to_error_[512 + j] = std::min(j < 0 ? -j : j, kMaxError);
  }
  for (size_t j = 0; j < quantized_table_.size(); ++j) {
    quantized_table_[j] = quantizedInit(j);
  }

  return WP2_STATUS_OK;
}

WP2Status State::Reset() {
  WP2_CHECK_STATUS(WP2L::PlaneCodec::Reset());
  cur_row_ = 0;
  prev_row_ = width_ + kBufferPadding;
  for (WP2::Vector_u16& pred_error : pred_errors_) {
    WP2_CHECK_ALLOC_OK(pred_error.resize((width_ + kBufferPadding) * 2));
    std::fill(pred_error.begin(), pred_error.end(), 0);
  }
  WP2_CHECK_ALLOC_OK(error_.resize((width_ + kBufferPadding) * 2));
  std::fill(error_.begin(), error_.end(), 0);
  WP2_CHECK_ALLOC_OK(quantized_error_.resize((width_ + kBufferPadding) * 2));
  std::fill(quantized_error_.begin(), quantized_error_.end(), 0);
  return WP2_STATUS_OK;
}

uint8_t State::DiffToError(int16_t tpv, int16_t prediction) const {
  const int16_t j = -tpv + ((prediction + kPredictionRound) >> kPredExtraBits);
  return std::min(j < 0 ? -j : j, kMaxError);
}

////////////////////////////////////////////////////////////////////////////////

WP2Status ClassicalState::Init(uint32_t width, uint32_t height) {
  WP2_CHECK_STATUS(State::Init(width, height));

  // Cache some constants.
  for (int j = 0; j < kMaxSumErrors; ++j) {
    error_to_weight_[j] = 150 * 512 / (58 + j * std::sqrt(j + 50));
  }

  return WP2_STATUS_OK;
}

int ClassicalState::GetNumContextsFromErrShift(int maxerr_shift) {
  return ((scp::kNumContexts - 1) >> maxerr_shift) + 1;
}

// If cond is true, clamp to min_pred and max_pred. Otherwise, clamp to the
// range of values that would be accepted by the predictors.
static int16_t Clamp(bool cond, int16_t pred, int16_t min_pred,
                     int16_t max_pred, int16_t W, int16_t N, int16_t NE) {
  if (cond) {
    return std::clamp(pred, min_pred, max_pred);
  } else {
    const int16_t max = std::max({W, N, NE});
    const int16_t min = std::min({W, N, NE});
    return std::clamp(pred, min, max);
  }
}

#define SET_POS                                                           \
  const size_t pos = cur_row_ + x;                                        \
  const size_t pos_W = x > 0 ? pos - 1 : pos;                             \
  const size_t pos_N = prev_row_ + x;                                     \
  const size_t pos_NE = x < width_ - 1 ? pos_N + 1 : pos_N;               \
  const size_t pos_NW = x > 0 ? pos_N - 1 : pos_N;                        \
  int16_t W = (x ? row_[x - 1] : row_p_ != nullptr ? row_p_[x] : 0);      \
  int16_t N = row_p_ != nullptr ? row_p_[x] : W;                          \
  int16_t NE = (x + 1 < width_ && row_p_ != nullptr ? row_p_[x + 1] : N); \
  int16_t NW = (x && row_p_ != nullptr ? row_p_[x - 1] : W);              \
  int16_t NN = (row_pp_ != nullptr ? row_pp_[x] : N);                     \
  N = WP2::LeftShift(N, kPredExtraBits);                                  \
  W = WP2::LeftShift(W, kPredExtraBits);                                  \
  NE = WP2::LeftShift(NE, kPredExtraBits);                                \
  NW = WP2::LeftShift(NW, kPredExtraBits);                                \
  NN = WP2::LeftShift(NN, kPredExtraBits);                                \
  int16_t teW = x == 0 ? 0 : error_[pos_W];                               \
  int16_t teN = error_[pos_N];                                            \
  int16_t teNW = error_[pos_NW];                                          \
  int16_t sumteWN = teN + teW;                                            \
  int16_t teNE = error_[pos_NE];                                          \
  std::array<int32_t, kNumPredictors> weights;

int16_t ClassicalState::PredictY0(size_t x, uint8_t* ctxt) {
  const size_t pos_W = cur_row_ + x - 1;
  *ctxt = (x == 0   ? kNumContexts - 3
           : x == 1 ? quantized_error_[pos_W]
                    : std::max(quantized_error_[pos_W],
                               quantized_error_[pos_W - 1]));
  prediction_[1] = prediction_[2] = prediction_[3] = (x > 0 ? row_[x - 1] : 27)
                                                     << kPredExtraBits;
  prediction_[0] =
      (x <= 1 ? prediction_[1]
              : prediction_[1] +
                    WP2::LeftShift(row_[x - 1] - row_[x - 2], kPredExtraBits) *
                        5 / 16);
  pred_ = Clamp(/*cond=*/true, prediction_[0], min_tpv_, max_tpv_, /*W=*/0,
                /*N=*/0, /*NE=*/0);
  return (pred_ + kPredictionRound) >> kPredExtraBits;
}

int16_t ClassicalState::PredictX0(uint8_t* ctxt) {
  const size_t pos_N = prev_row_;
  *ctxt = std::max(quantized_error_[pos_N],
                   quantized_error_[pos_N + (0 < last_x_ ? 1 : 0)]);
  prediction_[1] = prediction_[2] = prediction_[3] = row_p_[0]
                                                     << kPredExtraBits;
  pred_ = prediction_[0] =
      (((row_p_[0] * 7 + row_p_[0 < last_x_ ? 1 : 0]) << kPredExtraBits) + 4) >>
      3;
  return (pred_ + kPredictionRound) >> kPredExtraBits;
}

int16_t ClassicalState::PredictRegular(size_t x, uint8_t* ctxt) {
  if (x == 0) return PredictX0(ctxt);  // tobe fixed in Production

  SET_POS
  for (size_t i = 0; i < kNumPredictors; i++) {
    weights[i] = pred_errors_[i][pos_N] + pred_errors_[i][pos_NW] +
                 pred_errors_[i][pos_NE];
  }

  uint8_t mxe = std::max({quantized_error_[pos_W], quantized_error_[pos_N],
                          quantized_error_[pos_NW], quantized_error_[pos_NE]});
  if (x > 1) mxe = std::max(mxe, quantized_error_[pos_W - 1]);
  int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

  weights[0] = ErrorToWeight(weights[0]) * kMmulWeights0and1_R[0 + mE];
  weights[1] = ErrorToWeight(weights[1]) * kMmulWeights0and1_R[1 + mE];
  weights[2] = ErrorToWeight(weights[2]) * 32;  // Baseline
  weights[3] = ErrorToWeight(weights[3]) * kMulWeights3teNE_R[0 + mE];

  if (mE) {
    if (sumteWN * 40 + teNW * 20 + teNE * kMulWeights3teNE_R[1 + mE] <= 0) ++mE;
  } else {
    if (N == W && N == NE)
      mE = ((sumteWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
  }
  *ctxt = mE;

  prediction_[0] = W - (sumteWN + teNW) / 4;  // 7/32 works better than 1/4 ?
  prediction_[1] = N - (sumteWN + teNE) / 4;
  prediction_[2] = W + NE - N;
  int t = (teNE * 3 + teNW * 4 + 7) >> 5;
  prediction_[3] = N + (N - NN) * 23 / 32 + (W - NW) / 16 - t;

  const int16_t prediction = WeightedAverage(prediction_, weights);

  // if all three have the same sign
  pred_ = Clamp(/*cond=*/((teN ^ teW) | (teN ^ teNW)) > 0, prediction, min_tpv_,
                max_tpv_, W, N, NE);
  return (pred_ + kPredictionRound) >> kPredExtraBits;
}

// rowPrev must exist
int16_t ClassicalState::PredictWest(size_t x, uint8_t* ctxt) {
  if (x == 0) return PredictX0(ctxt);  // tobe fixed in Production

  SET_POS
  for (size_t i = 0; i < kNumPredictors; i++) {
    weights[i] = (pred_errors_[i][pos_N] * 3 >> 1) + pred_errors_[i][pos_NW] +
                 pred_errors_[i][pos_NE];
  }

  uint8_t mxe = std::max({quantized_error_[pos_W], quantized_error_[pos_N],
                          quantized_error_[pos_NW], quantized_error_[pos_NE]});
  if (x > 1) mxe = std::max(mxe, quantized_error_[pos_W - 1]);
  int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

  weights[0] = ErrorToWeight(weights[0]) * kMulWeights0and1_W[0 + mE];
  weights[1] = ErrorToWeight(weights[1]) * kMulWeights0and1_W[1 + mE];
  weights[2] = ErrorToWeight(weights[2]) * 32;  // Baseline
  weights[3] = ErrorToWeight(weights[3]) * kMulWeights3teNE_W[0 + mE];

  if (mE) {
    if (sumteWN * 40 + (teNW + teNE) * kMulWeights3teNE_W[1 + mE] <= 0) ++mE;
  } else {
    if (N == W && N == NE)
      mE = ((sumteWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
  }
  *ctxt = mE;

  prediction_[0] =
      W - (sumteWN + teNW) * 9 / 32;  // pr's 0 & 1 rely on true errors
  prediction_[1] =
      N - (sumteWN + teNE) * 171 / 512;  // clamping not needed, is it?
  prediction_[2] = W + NE - N;
  prediction_[3] = N + ((N - NN) >> 1) + ((W - NW) * 19 - teNW * 13) / 64;

  const int16_t prediction = WeightedAverage(prediction_, weights);

  // if all three have the same sign
  pred_ = Clamp(/*cond=*/((teN ^ teW) | (teN ^ teNE)) > 0, prediction, min_tpv_,
                max_tpv_, W, N, NE);
  return (pred_ + kPredictionRound) >> kPredExtraBits;
}

// rowPrev must exist
int16_t ClassicalState::PredictNorth(size_t x, uint8_t* ctxt) {
  if (x == 0) return PredictX0(ctxt);  // tobe fixed in Production

  SET_POS
  for (size_t i = 0; i < kNumPredictors; i++) {
    weights[i] = pred_errors_[i][pos_N] + pred_errors_[i][pos_NW] +
                 pred_errors_[i][pos_NE];
  }

  uint8_t mxe = std::max({quantized_error_[pos_W], quantized_error_[pos_N],
                          quantized_error_[pos_NW], quantized_error_[pos_NE]});
  if (x > 1) mxe = std::max(mxe, quantized_error_[pos_W - 1]);
  int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

  weights[0] = ErrorToWeight(weights[0]) * kMulWeights0and1_N[0 + mE];
  weights[1] = ErrorToWeight(weights[1]) * kMulWeights0and1_N[1 + mE];
  weights[2] = ErrorToWeight(weights[2]) * 32;  // Baseline
  weights[3] = ErrorToWeight(weights[3]) * kMulWeights3teNE_N[0 + mE];

  if (mE) {
    if (sumteWN * 40 + teNW * 23 + teNE * kMulWeights3teNE_N[1 + mE] <= 0) ++mE;
  } else {
    if (N == W && N == NE)
      mE = ((sumteWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
  }
  *ctxt = mE;

  prediction_[0] =
      N - (sumteWN + teNW + teNE) / 4;  // if bigger than 1/4,
                                        // clamping would be needed!
  prediction_[1] =
      W - ((teW * 2 + teNW) >> 2);  // pr's 0 & 1 rely on true errors
  prediction_[2] = W + NE - N;
  prediction_[3] = N + ((N - NN) * 47) / 64 - (teN >> 2);

  const int16_t prediction = WeightedAverage(prediction_, weights);

  // if all three have the same sign
  pred_ = Clamp(/*cond=*/((teN ^ teW) | (teN ^ teNE)) > 0, prediction, min_tpv_,
                max_tpv_, W, N, NE);
  return (pred_ + kPredictionRound) >> kPredExtraBits;
}

////////////////////////////////////////////////////////////////////////////////
// JXL functions.

void JXLState::SetHeaderIndex(size_t index) {
  assert(index >= 0 && index < kJxlHeaders.size());
  SetHeader(kJxlHeaders[index]);
}

void JXLState::SetHeader(const JxlHeader& header) { header_ = header; }

int JXLState::GetNumContexts() {
  return sizeof(kCutoffs) / sizeof(*kCutoffs) + 1;
}

int16_t JXLState::Predict(size_t x, uint8_t* ctxt) {
  SET_POS
  for (size_t i = 0; i < kNumPredictors; i++) {
    // pred_errors[pos_N] also contains the error of pixel W.
    // pred_errors[pos_NW] also contains the error of pixel WW.
    weights[i] = pred_errors_[i][pos_N] + pred_errors_[i][pos_NE] +
                 pred_errors_[i][pos_NW];
    weights[i] = ErrorWeight(weights[i], header_.w[i]);
  }

#if 0
  // Old JXL way of getting the context.
  int16_t p = teW;
  if (std::abs(teN) > std::abs(p)) p = teN;
  if (std::abs(teNW) > std::abs(p)) p = teNW;
  if (std::abs(teNE) > std::abs(p)) p = teNE;
  *ctxt =
      std::lower_bound(kCutoffsJXL, kCutoffsJXL + std::size(kCutoffsJXL), p) -
      kCutoffsJXL;
#else
  uint8_t mxe = std::max({quantized_error_[pos_W], quantized_error_[pos_N],
                          quantized_error_[pos_NW], quantized_error_[pos_NE]});
  if (x > 1) mxe = std::max(mxe, quantized_error_[pos_W - 1]);
  int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

  if (mE) {
    if (sumteWN * 40 + teNW * 20 + teNE * kMulWeights3teNE_R[1 + mE] <= 0) ++mE;
  } else {
    if (N == W && N == NE)
      mE = ((sumteWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
  }
  *ctxt = mE;
#endif

  prediction_[0] = W + NE - N;
  prediction_[1] = N - (((sumteWN + teNE) * header_.p1c) >> 5);
  prediction_[2] = W - (((sumteWN + teNW) * header_.p2c) >> 5);
  prediction_[3] =
      N - ((teNW * header_.p3ca + teN * header_.p3cb + teNE * header_.p3cc +
            (NN - N) * header_.p3cd + (NW - W) * header_.p3ce) >>
           5);

  pred_ = WeightedAverage(prediction_, weights);

  // If all three have the same sign, skip clamping.
  const int16_t prediction = Clamp(((teN ^ teW) | (teN ^ teNW)) > 0, pred_,
                                   min_tpv_, max_tpv_, W, N, NE);
  pred_ = prediction;

  return (prediction + kPredictionRound) >> kPredExtraBits;
}

}  // namespace scp
}  // namespace WP2L
