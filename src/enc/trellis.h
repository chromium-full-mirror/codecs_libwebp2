// Copyright 2021 Google LLC
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
//   WP2 encoder: trellis quantization.
//
// Author: Maryla (maryla@google.com)

#ifndef WP2_ENC_TRELLIS_H_
#define WP2_ENC_TRELLIS_H_

#include "src/common/lossy/block_size.h"

namespace WP2 {

// Removes small coeffs surrounded by many zeros. This is a faster/less accurate
// alternative to full trellis optimization. The number of zeroes required to
// drop a coefficient are deduced based on 'quality_factor' in [0; kQFactorMax].
void DropoutCoeffs(TrfSize tx_size, int quality_factor, int16_t* coeffs,
                   uint32_t* num_coeffs, int16_t* dequantized_coeffs);
// Same as above except that the number of leading/trailing zeros required to
// drop a coeff are explicitly passed in.
void DropoutCoeffs(TrfSize tx_size, const uint32_t dropout_num_before,
                   const uint32_t dropout_num_after, int16_t coeffs[],
                   uint32_t* num_coeffs, int16_t dequantized_coeffs[]);

}  // namespace WP2

#endif  // WP2_ENC_TRELLIS_H_
