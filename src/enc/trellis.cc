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
//   WP2 encoder: trellis quantization. Branched from aom file encodemb.c at
//   commit 27aa84463d241054885c900a50fabc2393ddedbb
//
// Author: Maryla (maryla@google.com)

#include "src/enc/trellis.h"

#include "src/common/lossy/residuals.h"

namespace WP2 {

// Hyper-parameters for dropout optimization, based on following logics.
// (1) Coefficients which are large enough will ALWAYS be kept.
constexpr uint16_t kDropoutCoeffMax = 2;  // Max dropout-able coefficient
// (2) Continuous coefficients will ALWAYS be kept. Here rigorous continuity is
//     NOT required. For example, `5 0 0 0 7` is treated as two continuous
//     coefficients if three zeros do not fulfill the dropout condition.
// Max dropout-able continuous coeff
constexpr uint32_t kDropoutContinuityMax = 2;
// (3) Recall that dropout optimization will forcibly set some quantized
//     coefficients to zero. The key logic on determining whether a coefficient
//     should be dropped is to check the number of continuous zeros before AND
//     after this coefficient. The exact number of zeros for judgement depends
//     on block size and quality factor. More concretely, block size
//     determines the base number of zeros, while quality factor determines
//     the multiplier. Intuitively, larger block requires more zeros and larger
//     quality factor also requires more zeros (more information is lost
//     when using larger quality factor).
constexpr uint32_t kDropoutMax = 256;  // Max number of leading/trailing zeros
constexpr uint32_t kDropoutMin = 32;   // Min number of leading/trailing zeros
constexpr uint32_t kDropoutBeforeBaseMin = 32;  // Min base number of zeros
constexpr float kDropoutMultiplierMin = 1;  // Max multiplier on number of zeros
constexpr float kDropoutMultiplierMax = 4;  // Max multiplier on number of zeros
constexpr uint32_t kDropoutMultiplierQBase = 64;  // Base Q for multiplier

void DropoutCoeffs(TrfSize tx_size, int quality_factor, int16_t coeffs[],
                   uint32_t* num_coeffs, int16_t dequantized_coeffs[]) {
  if (*num_coeffs == 0) return;
  const uint32_t tx_width = TrfWidth[tx_size];
  const uint32_t tx_height = TrfHeight[tx_size];
  // Compute number of zeros used for dropout judgement.
  const uint32_t base_number =
      std::max(std::max(tx_width, tx_height) * 2, kDropoutBeforeBaseMin);
  const float multiplier =
      Clamp((float)quality_factor / kDropoutMultiplierQBase,
            kDropoutMultiplierMin, kDropoutMultiplierMax);
  const uint32_t dropout_num = Clamp<uint32_t>(
      std::round(multiplier * base_number), kDropoutMin, kDropoutMax);

  DropoutCoeffs(tx_size, dropout_num, dropout_num, coeffs, num_coeffs,
                dequantized_coeffs);
}

void DropoutCoeffs(TrfSize tx_size, const uint32_t dropout_num_before,
                   const uint32_t dropout_num_after, int16_t coeffs[],
                   uint32_t* num_coeffs, int16_t dequantized_coeffs[]) {
  if (*num_coeffs == 0) return;
  int16_t* const qcoeff = coeffs;
  int16_t* const dqcoeff = dequantized_coeffs;
  const uint32_t max_eob = kNumCoeffs[tx_size];

  if (max_eob <= dropout_num_before) return;

  // End of block index, i.e. index of the last non-zero coefficient + 1, in
  // zig-zag order. It differs from num_coeffs which is in lexico order.
  uint32_t eob = max_eob;
  const uint16_t* const scan_order = ResidualIterator::GetZigzag(tx_size);
  while (eob > 0 && qcoeff[scan_order[eob - 1]] == 0) --eob;
  assert(eob > 0);

  // Early return if there are not enough non-zero coefficients.
  if (eob <= dropout_num_before) return;

  uint32_t count_zeros_before = 0;
  uint32_t count_zeros_after = 0;
  uint32_t count_nonzeros = 0;
  // Index of the first non-zero coefficient after sufficient number of
  // continuous zeros. If equals to `-1`, it means number of leading zeros
  // hasn't reach `dropout_num_before`.
  int idx = -1;

  for (uint32_t i = 0; i < eob; ++i) {
    const uint32_t scan_idx = scan_order[i];
    if (std::abs(qcoeff[scan_idx]) > kDropoutCoeffMax) {
      // Keep large coefficients.
      count_zeros_before = 0;
      count_zeros_after = 0;
      idx = -1;
    } else if (qcoeff[scan_idx] == 0) {  // Count zeros.
      if (idx == -1) {
        ++count_zeros_before;
      } else {
        ++count_zeros_after;
      }
    } else {  // Count non-zeros.
      if (count_zeros_before >= dropout_num_before) {
        idx = (idx == -1) ? i : idx;
        ++count_nonzeros;

        // Handle continuity.
        if (count_nonzeros > kDropoutContinuityMax) {
          count_zeros_before = 0;
          count_zeros_after = 0;
          count_nonzeros = 0;
          idx = -1;
        }
      } else {
        count_zeros_before = 0;
        assert(count_zeros_after == 0);
        assert(idx == -1);
      }
    }

    if (idx == -1) continue;

    if (i == eob - 1) {
      // We consider that there are many zeros after the last coeff (even if
      // it's close to the end of the block).
      count_zeros_after = dropout_num_after;
    }

    // Set redundant coefficients to zeros if needed.
    if (count_zeros_before >= dropout_num_before &&
        count_zeros_after >= dropout_num_after) {
      for (uint32_t j = idx; j <= i; ++j) {
        qcoeff[scan_order[j]] = 0;
        dqcoeff[scan_order[j]] = 0;
      }
      count_zeros_before += (i - idx + 1);
      count_zeros_after = 0;
      count_nonzeros = 0;
    }
  }

  // Update num_coeffs.
  while (*num_coeffs > 0 && qcoeff[*num_coeffs - 1] == 0) {
    --(*num_coeffs);
  }
}

}  // namespace WP2
