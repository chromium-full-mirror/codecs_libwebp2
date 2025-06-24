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
// Implementation of the CALIC algorithm.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WP2_COMMON_LOSSLESS_CALIC_H_
#define WP2_COMMON_LOSSLESS_CALIC_H_

#include <cstdint>

#include "src/common/lossless/plane.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

// Define ORIGINAL_CALIC to get the original version of CALIC, with no
// enhancement on the coding context.
#define ORIGINAL_CALIC

namespace WP2L {
namespace calic {

// Class implementing the CALIC algorithm from Xiaolin Wu and Nasir Memon.
class CalicState : public ::WP2L::PlaneCodec {
 public:
  // This is part of the format.
#ifdef ORIGINAL_CALIC
  enum class Quantization { kOriginal, kNum };
#else
  enum class Quantization { kOriginal, k18, k20, kNum };
#endif

  WP2Status Init(uint32_t width, uint32_t height) override;
  WP2Status Reset() override;
  void StartProcessingLine(uint32_t y, const int16_t* row) override;

  // Set the quantization type for the plane to compress.
  WP2Status SetQuantization(Quantization quantization);
  // Sets the minimum and maximum true pixel inclusive values in the image.
  void SetMinMax(int16_t min_tpv, int16_t max_tpv);
  // Get the number of contexts for the given quantization method.
  static uint8_t GetNumContextsFromQuantization(Quantization quantization);
  // Predicts the pixel value at position x, as well as the context `ctxt` and
  // whether the mean error is negative.
  void Predict(uint32_t x, int16_t& prediction, uint8_t& ctxt,
               bool& is_mean_error_negative);
  // Updates the internals after we know the true pixel value `tpv` at pixel x.
  void Update(uint32_t x, int16_t tpv);

 private:
  using ::WP2L::PlaneCodec::Init;
  // Number of bits used to perform operations with.
  static inline constexpr int kPrecisionBits = 8;

  // Predicts the pixel value using the GAP method. The horizontal and vertical
  // gradients are given as well as the context of the 4 neighbors.
  static int16_t PredictGAP(uint16_t dh, uint16_t dv, int16_t w, int16_t n,
                            int16_t ne, int16_t nw);

  // The minimum and maximum true pixel values in the image.
  int16_t min_tpv_, max_tpv_;
  // The default value to use when there is no information: the mean of the
  // minimum and maximum true pixel values.
  int16_t default_value_;

  // The current prediction.
  int16_t pred_;
  // The previous error between the prediction and the true pixel value.
  int16_t prev_pred_error_;
  // Vectors containing the differences between the current and previous cols or
  // rows, in the current or previous row.
  WP2::Vector_u16 curr_diff_col_, curr_diff_row_, prev_diff_col_,
      prev_diff_row_;
  // Accumulator of the errors with precision kPrecisionBits.
  WP2::Vector_s16 s_errors_;
  WP2::Vector_u8 n_errors_;
  // The mean error, with precision kPrecisionBits.
  WP2::Vector_s32 mean_errors_;
  WP2::Vector_u8 quantized_delta_;
  uint16_t compound_context_;
  Quantization quantization_;
};

}  // namespace calic
}  // namespace WP2L

#endif  // WP2_COMMON_LOSSLESS_CALIC_H_
