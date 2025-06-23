// Copyright 2025 Google LLC
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
// Common CALIC code from "Context-Based, Adaptive, Lossless Image Coding"
// IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 45, NO. 4, APRIL 1997
// by Xiaolin Wu and Nasir Memon
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/common/lossless/calic.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "src/common/lossless/plane.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

namespace WP2L {
namespace calic {

WP2Status CalicState::Init(uint32_t width, uint32_t height,
                           Quantization quantization) {
  WP2_CHECK_STATUS(WP2L::PlaneCodec::Init(width, height));
  WP2_CHECK_ALLOC_OK(curr_diff_col_.resize(width));
  WP2_CHECK_ALLOC_OK(curr_diff_row_.resize(width));
  WP2_CHECK_ALLOC_OK(prev_diff_col_.resize(width));
  WP2_CHECK_ALLOC_OK(prev_diff_row_.resize(width));

  // Initialize the quantized delta table.
  switch (quantization) {
    case Quantization::kOriginal:
      WP2_CHECK_ALLOC_OK(quantized_delta_.resize(141));
      std::fill(&quantized_delta_[0], &quantized_delta_[5], 0);
      std::fill(&quantized_delta_[5], &quantized_delta_[15], 1);
      std::fill(&quantized_delta_[15], &quantized_delta_[25], 2);
      std::fill(&quantized_delta_[25], &quantized_delta_[42], 3);
      std::fill(&quantized_delta_[42], &quantized_delta_[60], 4);
      std::fill(&quantized_delta_[60], &quantized_delta_[85], 5);
      std::fill(&quantized_delta_[85], &quantized_delta_[140], 6);
      quantized_delta_[140] = 7;
      break;
    case Quantization::kNum:
      assert(false);
      return WP2_STATUS_INVALID_PARAMETER;
  }
  if (quantized_delta_.back() + 1 != GetNumContexts(quantization)) {
    return WP2_STATUS_INVALID_PARAMETER;
  }

  const int kSize = ((quantized_delta_.back() + 1) / 2) * (1 << 8);
  WP2_CHECK_ALLOC_OK(mean_errors_.resize(kSize));
  WP2_CHECK_ALLOC_OK(s_errors_.resize(kSize));
  WP2_CHECK_ALLOC_OK(n_errors_.resize(kSize));
  WP2_CHECK_STATUS(Reset());

  return WP2_STATUS_OK;
}

void CalicState::SetMinMax(int16_t min_tpv, int16_t max_tpv) {
  min_tpv_ = min_tpv;
  max_tpv_ = max_tpv;
  default_value_ = (min_tpv + max_tpv + 1) / 2;
}

uint8_t CalicState::GetNumContexts(Quantization quantization) {
  switch (quantization) {
    case Quantization::kOriginal:
      return 8;
      break;
    case Quantization::kNum:
    default:
      assert(false);
      return WP2_STATUS_INVALID_PARAMETER;
  }
}

WP2Status CalicState::Reset() {
  std::fill(mean_errors_.begin(), mean_errors_.end(), 0);
  std::fill(s_errors_.begin(), s_errors_.end(), 0);
  std::fill(n_errors_.begin(), n_errors_.end(), 0);
  prev_pred_error_ = 0;
  return WP2_STATUS_OK;
}

void CalicState::StartProcessingLine(uint32_t y, const int16_t* row) {
  PlaneCodec::StartProcessingLine(y, row);

  curr_diff_row_.swap(prev_diff_row_);
  curr_diff_col_.swap(prev_diff_col_);
}

int16_t CalicState::PredictGAP(uint16_t dh, uint16_t dv, int16_t w, int16_t n,
                               int16_t ne, int16_t nw) {
  int16_t pred;
  if (dv - dh > 80) {
    // Sharp horizontal edge.
    pred = w;
  } else if (dv - dh < -80) {
    // Sharp vertical edge.
    pred = n;
  } else {
    int16_t p = (w + n) / 2 + (ne - nw) / 4;
    if (dv - dh > 32) {
      // Horizontal edge.
      p = (p + w) / 2;
    } else if (dv - dh > 8) {
      // Weak horizontal edge.
      p = (3 * p + w) / 4;
    } else if (dv - dh < -32) {
      // Vertical edge.
      p = (p + n) / 2;
    } else if (dv - dh < -8) {
      // Weak vertical edge.
      p = (3 * p + n) / 4;
    }
    pred = p;
  }
  return pred;
}

void CalicState::Predict(uint32_t x, int16_t& prediction, uint8_t& ctxt,
                         bool& is_mean_error_negative) {
  // Section 3: GAP, Gradient-Adjusted Prediction.
  uint16_t dh = 0, dv = 0;
  // Edges are not properly defined in the paper.
  if (x > 0) {
    dh = curr_diff_col_[x - 1];
    dv = curr_diff_row_[x - 1];
  } else {
    dh = dv = 0;
  }
  if (y_ > 0) {
    dh += prev_diff_col_[x];
    dv += prev_diff_row_[x];
    if (x + 1 < width_) {
      dh += prev_diff_col_[x + 1];
      dv += prev_diff_row_[x + 1];
    }
  }

  // Edge conditions are not specified by the paper so we take the closest
  // physical value, or the default one if none.
  const int16_t w = x > 0 ? row_[x - 1] : y_ > 0 ? row_p_[0] : default_value_;
  const int16_t n = y_ > 0 ? row_p_[x] : w;
  const int16_t ne = x + 1 < width_ && y_ > 0 ? row_p_[x + 1] : n;
  const int16_t nw = x > 0 && y_ > 0 ? row_p_[x - 1] : n;
  pred_ = PredictGAP(dh, dv, w, n, ne, nw);
  pred_ = std::clamp(pred_, min_tpv_, max_tpv_);

  // Section 4: Coding Context Selection and Quantization.
  static constexpr int kA = 1, kB = 1, kC = 2;
  const size_t delta = kA * dh + kB * dv + kC * std::abs(prev_pred_error_);
  const uint8_t quantized_delta = delta >= quantized_delta_.size()
                                      ? quantized_delta_.back()
                                      : quantized_delta_[delta];

  // Section 5.A: Formation and Quantization of Contexts.
  const int16_t nn = y_ > 1 ? row_pp_[x] : n;
  const int16_t ww = x > 1 ? row_[x - 2] : w;
  uint8_t b = 0;
  b |= (n < pred_) << 0;
  b |= (w < pred_) << 1;
  b |= (nw < pred_) << 2;
  b |= (ne < pred_) << 3;
  b |= (nn < pred_) << 4;
  b |= (ww < pred_) << 5;
  b |= ((2 * n - nn) < pred_) << 6;
  b |= ((2 * w - ww) < pred_) << 7;
  compound_context_ = ((quantized_delta / 2) << 8) | b;

  // Section 5.D: Error Feedback.
  const auto mean_error = mean_errors_[compound_context_];
  is_mean_error_negative = mean_error < 0;
  const int16_t rounded_abs_residual =
      WP2::RightShiftRound(std::abs(mean_error), kPrecisionBits);
  prediction =
      pred_ + (mean_error >= 0 ? rounded_abs_residual : -rounded_abs_residual);
  prediction = std::clamp(prediction, min_tpv_, max_tpv_);

  ctxt = quantized_delta;
}

void CalicState::Update(uint32_t x, int16_t tpv) {
  if (x > 0) {
    curr_diff_col_[x] = std::abs(tpv - row_[x - 1]);
  } else {
    curr_diff_col_[0] = 0;
  }
  if (y_ > 0) {
    curr_diff_row_[x] = std::abs(row_[x] - row_p_[x]);
  } else {
    curr_diff_row_[x] = 0;
  }
  prev_pred_error_ = tpv - pred_;

  // Section 5.C: Estimation of Error Magnitude Within a Context.
  // Use the final prediction and not the original one (cf section 5.D).
  s_errors_[compound_context_] += tpv - pred_;
  ++n_errors_[compound_context_];
  if (n_errors_[compound_context_] >= 128) {
    s_errors_[compound_context_] /= 2;
    n_errors_[compound_context_] = 64;
  }

  mean_errors_[compound_context_] = WP2::DivRound(
      static_cast<int32_t>(s_errors_[compound_context_]) << kPrecisionBits,
      static_cast<int32_t>(n_errors_[compound_context_]));
}

}  // namespace calic
}  // namespace WP2L
