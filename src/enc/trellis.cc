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
//   WP2 encoder: trellis quantization. Branched from aom file encodemb.c and
//   txb_rdopt.c at commit 27aa84463d241054885c900a50fabc2393ddedbb
//
// Author: Maryla (maryla@google.com)

#include "src/enc/trellis.h"

#include "src/common/lossy/quant_mtx.h"
#include "src/common/lossy/residuals.h"
#include "src/common/lossy/transforms.h"
#include "src/enc/block_enc.h"
#include "src/enc/residuals_enc_aom.h"
#include "src/enc/symbols_enc.h"

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

void DropoutCoeffs(TrfSize tdim, int quality_factor, int16_t coeffs[],
                   uint32_t* num_coeffs, int16_t dequantized_coeffs[]) {
  if (*num_coeffs == 0) return;
  const uint32_t tx_width = TrfWidth[tdim];
  const uint32_t tx_height = TrfHeight[tdim];
  // Compute number of zeros used for dropout judgement.
  const uint32_t base_number =
      std::max(std::max(tx_width, tx_height) * 2, kDropoutBeforeBaseMin);
  const float multiplier =
      Clamp((float)quality_factor / kDropoutMultiplierQBase,
            kDropoutMultiplierMin, kDropoutMultiplierMax);
  const uint32_t dropout_num = Clamp<uint32_t>(
      std::round(multiplier * base_number), kDropoutMin, kDropoutMax);

  DropoutCoeffs(tdim, dropout_num, dropout_num, coeffs, num_coeffs,
                dequantized_coeffs);
}

void DropoutCoeffs(TrfSize tdim, const uint32_t dropout_num_before,
                   const uint32_t dropout_num_after, int16_t coeffs[],
                   uint32_t* num_coeffs, int16_t dequantized_coeffs[]) {
  if (*num_coeffs == 0) return;
  const uint32_t max_eob = kNumCoeffs[tdim];

  if (max_eob <= dropout_num_before) return;

  // End of block index, i.e. index of the last non-zero coefficient + 1, in
  // zig-zag order. It differs from num_coeffs which is in lexico order.
  uint32_t eob = max_eob;
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);
  while (eob > 0 && coeffs[scan[eob - 1]] == 0) --eob;
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
    const uint32_t scan_idx = scan[i];
    if (std::abs(coeffs[scan_idx]) > kDropoutCoeffMax) {
      // Keep large coefficients.
      count_zeros_before = 0;
      count_zeros_after = 0;
      idx = -1;
    } else if (coeffs[scan_idx] == 0) {  // Count zeros.
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
        coeffs[scan[j]] = 0;
        dequantized_coeffs[scan[j]] = 0;
      }
      count_zeros_before += (i - idx + 1);
      count_zeros_after = 0;
      count_nonzeros = 0;
    }
  }

  // Update num_coeffs.
  while (*num_coeffs > 0 && coeffs[*num_coeffs - 1] == 0) {
    --(*num_coeffs);
  }
}

// -----------------------------------------------------------------------------

namespace {

// Returns the square difference between 'res' (original, non quantized coeff)
// and 'dequantized_coeff'.
inline int64_t GetDisto(int16_t res, int16_t dequantized_coeff) {
  const int64_t diff = (res - dequantized_coeff);
  return diff * diff;
}

// Computes 'coeff_low' which is coeff-1 for a positive coeff, or coeff+1 for
// a negative coeff, and the corresponding dequantized value.
inline void GetLow(int16_t abs_coeff, bool is_negative, int16_t dequant,
                   int16_t* coeff_low, int16_t* dequantized_coeff_low) {
  const int16_t abs_coeff_low = abs_coeff - 1;
  *coeff_low = (is_negative ? -abs_coeff_low : abs_coeff_low);
  int16_t abs_dequantized_coeff_low = (abs_coeff_low * dequant);
  *dequantized_coeff_low =
      (is_negative ? -abs_dequantized_coeff_low : abs_dequantized_coeff_low);
}

inline float RdScore(float lambda, float rate, float disto) {
  return lambda * rate + disto;
}

// Returns the cost of storing the given end of block position.
float GetEobCost(uint32_t eob, libgav1::PlaneType plane_type, TrfSize tdim,
                 TransformClass tx_class, SymbolCounter* const counter) {
  ANSEncCounter enc;
  libgav1::AOMResidualWriter::WriteEOB(eob, tdim, plane_type, tx_class, counter,
                                       &enc);
  return enc.GetCost();
}

// Returns the cost of storing the given 'coeff'.
float GetCoeffCost(bool is_last, uint32_t eob, uint16_t pos, int16_t coeff,
                   libgav1::PlaneType plane_type, TrfSize tdim,
                   TransformClass tx_class, bool first_is_dc,
                   SymbolCounter* const counter, uint8_t* const levels) {
  ANSEncCounter enc;
  libgav1::AOMResidualWriter::WriteCoeff(coeff, pos, is_last, eob, first_is_dc,
                                         plane_type, tdim, tx_class, levels,
                                         counter, &enc);
  return enc.GetCost();
}

// Attempts lowering the i-th coeff, and possibly updates it if the score is
// improved. Tries two options:
// - coeff unchanged
// - coeff - 1 (or coeff + 1 for negative coeffs)
void UpdateCoeffGeneral(uint32_t i, uint32_t eob, libgav1::PlaneType plane_type,
                        TrfSize tdim, TransformClass tx_class,
                        const QuantMtx& quant, bool first_id_dc, float lambda,
                        const uint16_t* const scan,
                        const int32_t* const residuals, int16_t* const coeffs,
                        int16_t* const dequantized_coeffs,
                        uint8_t* const levels, float* accu_rate,
                        int64_t* accu_disto_minus_disto0,
                        SymbolCounter* const counter) {
  const int16_t dequant = quant.Dequant()[scan[i]];
  const uint16_t pos = scan[i];
  const int16_t coeff = coeffs[pos];
  assert(eob > 0);
  const bool is_last = (i == (eob - 1));
  if (coeff == 0) {
    assert(!is_last);
    *accu_rate += GetCoeffCost(is_last, eob, pos, coeff, plane_type, tdim,
                               tx_class, first_id_dc, counter, levels);
  } else {
    const bool is_negative = (coeff < 0);
    const int16_t abs_coeff = std::abs(coeff);
    const int16_t res = residuals[pos];
    const int16_t dequantized_coeff = dequantized_coeffs[pos];
    const int64_t disto = GetDisto(res, dequantized_coeff);
    const int64_t disto0 = GetDisto(res, 0);
    const float rate = GetCoeffCost(is_last, eob, pos, coeff, plane_type, tdim,
                                    tx_class, first_id_dc, counter, levels);
    const float score = RdScore(lambda, rate, disto);

    int16_t coeff_low, dequantized_coeff_low;
    int64_t disto_low;
    if (abs_coeff == 1) {
      coeff_low = 0;
      dequantized_coeff_low = 0;
      disto_low = disto0;
    } else {
      GetLow(abs_coeff, is_negative, dequant, &coeff_low,
             &dequantized_coeff_low);
      disto_low = GetDisto(res, dequantized_coeff_low);
    }
    const float rate_low =
        GetCoeffCost(is_last, eob, pos, coeff_low, plane_type, tdim, tx_class,
                     first_id_dc, counter, levels);

    const float score_low = RdScore(lambda, rate_low, disto_low);
    if (score_low < score) {
      coeffs[pos] = coeff_low;
      dequantized_coeffs[pos] = dequantized_coeff_low;
      libgav1::AOMResidualIO::UpdateLevels(coeff_low, pos, tdim, levels);
      *accu_rate += rate_low;
      *accu_disto_minus_disto0 += disto_low - disto0;
    } else {
      *accu_rate += rate;
      *accu_disto_minus_disto0 += disto - disto0;
    }
  }
}

// Simpler version of 'UpdateCoeffGeneral' for coeffs that are neither DC nor
// the last one.
void UpdateCoeffMiddle(int i, int eob, libgav1::PlaneType plane_type,
                       TrfSize tdim, TransformClass tx_class,
                       const QuantMtx& quant, bool first_is_dc, float lambda,
                       const uint16_t* const scan,
                       const int32_t* const residuals, int16_t* const coeffs,
                       int16_t* const dequantized_coeffs, uint8_t* const levels,
                       float* accu_rate, SymbolCounter* const counter) {
  const int16_t dequant = quant.Dequant()[scan[i]];
  assert(i != eob - 1);
  assert(i > 0);
  const uint16_t pos = scan[i];
  const int16_t coeff = coeffs[pos];
  if (coeff == 0) {
    *accu_rate += GetCoeffCost(/*is_last=*/false, eob, pos, coeff, plane_type,
                               tdim, tx_class, first_is_dc, counter, levels);
  } else {
    const int16_t abs_coeff = std::abs(coeff);
    const int16_t abs_res = std::abs(residuals[pos]);
    const int16_t abs_dequantized_coeff = std::abs(dequantized_coeffs[pos]);
    float rate_low = 0;
    const float rate =
        GetCoeffCost(/*is_last=*/false, eob, pos, coeff, plane_type, tdim,
                     tx_class, first_is_dc, counter, levels);
    // If the dequantized coeff is already smaller than the original one, we
    // won't try lowering.
    if (abs_dequantized_coeff < abs_res) {
      *accu_rate += rate;
      return;
    }

    const int64_t disto = GetDisto(abs_res, abs_dequantized_coeff);
    const float score = RdScore(lambda, rate, disto);

    const int16_t abs_coeff_low = abs_coeff - 1;
    const int16_t abs_dequantized_coeff_low = (abs_coeff_low * dequant);
    const int64_t disto_low = GetDisto(abs_res, abs_dequantized_coeff_low);
    const float score_low = RdScore(lambda, rate_low, disto_low);

    if (score_low < score) {
      const bool is_negative = (coeff < 0);
      coeffs[pos] = (is_negative? -abs_coeff_low : abs_coeff_low);
      dequantized_coeffs[pos] = (is_negative ? -abs_dequantized_coeff_low
                                             : abs_dequantized_coeff_low);
      libgav1::AOMResidualIO::UpdateLevels(abs_coeff_low, pos, tdim, levels);
      *accu_rate += rate_low;
    } else {
      *accu_rate += rate;
    }
  }
}

// Attempts lowering a coeff that is close to the end of block, and possibly
// updates the coeff if the score is improved. Tries four options:
// - coeff unchanged
// - coeff - 1 (or coeff + 1 for negative coeffs)
// - coeff unchanged + eob moved just after this coeff (zeroing out following
//   ones)
// - coeff - 1 (or coeff + 1 for negative coeffs) + eob moved just after
void UpdateCoeffEob(uint32_t i, libgav1::PlaneType plane_type, TrfSize tdim,
                    TransformClass tx_class, const QuantMtx& quant,
                    bool first_is_dc, float lambda, bool sharp_mode,
                    const uint16_t* const scan, const int32_t* const residuals,
                    int16_t* const coeffs, int16_t* const dequantized_coeffs,
                    uint8_t* const levels, uint32_t* nz_num,
                    uint32_t* const nz_pos, uint32_t* eob, float* accu_rate,
                    int64_t* accu_disto_minus_disto0,
                    SymbolCounter* const counter) {
  const int16_t dequant = quant.Dequant()[scan[i]];
  assert(i + 1 != *eob);
  const uint16_t pos = scan[i];
  const int16_t coeff = coeffs[pos];
  if (coeff == 0) {
    *accu_rate += GetCoeffCost(/*is_last=*/false, *eob, pos, coeff, plane_type,
                               tdim, tx_class, first_is_dc, counter, levels);
  } else {
    bool choose_lower = false;
    const int16_t abs_coeff = std::abs(coeff);
    const int16_t res = residuals[pos];
    const int16_t dequantized_coeff = dequantized_coeffs[pos];
    const bool is_negative = (coeff < 0);
    const int64_t disto0 = GetDisto(res, 0);
    int64_t disto = GetDisto(res, dequantized_coeff) - disto0;
    float rate = GetCoeffCost(/*is_last=*/false, *eob, pos, coeff, plane_type,
                              tdim, tx_class, first_is_dc, counter, levels);
    float score =
        RdScore(lambda, *accu_rate + rate, *accu_disto_minus_disto0 + disto);

    int16_t coeff_low, dequantized_coeff_low;
    int16_t abs_coeff_low;
    int64_t disto_low;
    float rate_low;
    float score_low;
    if (abs_coeff == 1) {
      abs_coeff_low = 0;
      dequantized_coeff_low = coeff_low = 0;
      disto_low = 0;  // disto0 - disto0
      rate_low = GetCoeffCost(/*is_last=*/false, *eob, pos, 0, plane_type, tdim,
                              tx_class, first_is_dc, counter, levels);
      score_low =
          RdScore(lambda, *accu_rate + rate_low, *accu_disto_minus_disto0);
    } else {
      GetLow(abs_coeff, is_negative, dequant, &coeff_low,
             &dequantized_coeff_low);
      abs_coeff_low = abs_coeff - 1;
      disto_low = GetDisto(res, dequantized_coeff_low) - disto0;
      rate_low =
          GetCoeffCost(/*is_last=*/false, *eob, pos, coeff_low, plane_type,
                       tdim, tx_class, first_is_dc, counter, levels);
      score_low = RdScore(lambda, *accu_rate + rate_low,
                          *accu_disto_minus_disto0 + disto_low);
    }

    bool choose_lower_new_eob = false;
    const int new_eob = i + 1;
    const float new_eob_cost =
        GetEobCost(new_eob, plane_type, tdim, tx_class, counter);
    float rate_coeff_eob =
        new_eob_cost + GetCoeffCost(/*is_last=*/true, new_eob, pos, coeff,
                                    plane_type, tdim, tx_class, first_is_dc,
                                    counter, levels);
    int64_t disto_new_eob = disto;
    float score_new_eob = RdScore(lambda, rate_coeff_eob, disto_new_eob);

    if (abs_coeff_low > 0) {
      const float rate_coeff_eob_low =
          new_eob_cost + GetCoeffCost(
                             /*is_last=*/true, new_eob, pos, coeff_low,
                             plane_type, tdim, tx_class, first_is_dc, counter,
                             levels);
      const int64_t disto_new_eob_low = disto_low;
      const float score_new_eob_low =
          RdScore(lambda, rate_coeff_eob_low, disto_new_eob_low);
      if (score_new_eob_low < score_new_eob) {
        choose_lower_new_eob = true;
        score_new_eob = score_new_eob_low;
        rate_coeff_eob = rate_coeff_eob_low;
        disto_new_eob = disto_new_eob_low;
      }
    }

    if (score_low < score) {
      choose_lower = true;
      score = score_low;
      rate = rate_low;
      disto = disto_low;
    }

    if (!sharp_mode && score_new_eob < score) {
      for (uint32_t j = 0; j < *nz_num; ++j) {
        int last_pos = nz_pos[j];
        libgav1::AOMResidualIO::UpdateLevels(0, pos, tdim, levels);
        coeffs[last_pos] = 0;
        dequantized_coeffs[last_pos] = 0;
      }
      *eob = new_eob;
      *nz_num = 0;
      *accu_rate = rate_coeff_eob;
      *accu_disto_minus_disto0 = disto_new_eob;
      choose_lower = choose_lower_new_eob;
    } else {
      *accu_rate += rate;
      *accu_disto_minus_disto0 += disto;
    }

    if (choose_lower) {
      coeffs[pos] = coeff_low;
      dequantized_coeffs[pos] = dequantized_coeff_low;
      libgav1::AOMResidualIO::UpdateLevels(abs_coeff_low, pos, tdim, levels);
    }
    if (coeffs[pos]) {
      nz_pos[*nz_num] = pos;
      ++*nz_num;
    }
  }
}

// Attempts skipping the whole block (setting all coeffs to 0) if it improves
// the rd score.
static void UpdateSkip(float lambda, float* accu_rate,
                       int64_t accu_disto_minus_disto0, uint32_t* const eob,
                       uint32_t nz_num, uint32_t* const nz_pos,
                       int16_t* const coeffs,
                       int16_t* const dequantized_coeffs) {
  const float score = RdScore(lambda, *accu_rate, accu_disto_minus_disto0);
  const float score_skip = RdScore(lambda, 0, 0);
  if (score_skip < score) {
    for (uint32_t i = 0; i < nz_num; ++i) {
      const uint16_t pos = nz_pos[i];
      coeffs[pos] = 0;
      dequantized_coeffs[pos] = 0;
      // no need to set up levels because this is the last step
    }
    *accu_rate = 0;
    *eob = 0;
  }
}

// Initializes the 'levels' array which is basically the absolute value
// of coeffs capped to kQuantizerCoefficientBaseRangeContextClamp.
static void InitLevels(const int16_t* const coeff, const uint32_t width,
                       const uint32_t height, uint8_t* const levels) {
  const uint32_t stride = width + libgav1::kLevelBufferPadding;

  uint8_t* ls = levels;
  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      *ls++ = (uint8_t)Clamp<uint32_t>(
          std::abs(coeff[i * width + j]), 0u,
          libgav1::kQuantizerCoefficientBaseRangeContextClamp);
    }
    for (uint32_t j = 0; j < libgav1::kLevelBufferPadding; j++) {
      *ls++ = 0;
    }
  }

  memset(levels + stride * height, 0,
         sizeof(*levels) * (libgav1::kLevelBufferPadding * stride));
}

}  // namespace

void Av1CoeffOptimization(Channel channel, TransformPair tx_type, TrfSize tdim,
                          const QuantMtx& quant, bool first_is_dc,
                          const int32_t* const residuals, int16_t* const coeffs,
                          uint32_t* num_coeffs,
                          int16_t* const dequantized_coeffs,
                          SymbolCounter* const counter, float* rate_cost) {
  if (*num_coeffs == 0) return;
  counter->Clear();

  // When using sharp mode, the algorithm is less aggressive in removing tail
  // coefficients.
  // TODO(maryla): enable sharp mode in some situations?
  constexpr bool kSharpMode = false;

  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);

  const uint32_t max_eob = kNumCoeffs[tdim];
  uint32_t eob = max_eob;
  while (eob > 0 && coeffs[scan[eob - 1]] == 0) --eob;
  assert(eob > 0);

  const libgav1::PlaneType plane_type = libgav1::GetPlaneType(channel);
  const TransformClass tx_class = GetTransformClass(tx_type);
  const uint32_t width = TrfWidth[tdim];
  const uint32_t height = TrfHeight[tdim];

  // 'levels' are used to compute the cluster when storing coefficients.
  uint8_t levels[libgav1::kMaxTxBufferDim * libgav1::kMaxTxBufferDim];
  if (eob > 1) InitLevels(coeffs, width, height, levels);

  const float lambda = quant.lambda * 0.5;  // Empirical value.

  const float eob_cost = GetEobCost(eob, plane_type, tdim, tx_class, counter);
  // Accumulated rate of storing coeffs.
  float accu_rate = eob_cost;
  // Accumulated distortion with the chosen coeffs, minus the distortion when
  // setting all coeffs to 0.
  int64_t accu_disto_minus_disto0 = 0;

  // Process coeffs starting from the end.

  // Step 1: deal with the last coeff.
  int i = eob - 1;
  const uint16_t last_pos = scan[i];
  const int16_t last_coeff = coeffs[last_pos];
  // Maximum number of non zero coeffs that can be set to zero in order to
  // reduce the position of the end-of-block.
  constexpr uint32_t kMaxNzNum = 2;
  uint32_t nz_num = 1;  // Number of non zero coeffs between i and eob.
  uint32_t nz_pos[3] = {last_pos, 0, 0};  // Position of those non zero coeffs.
  if (std::abs(last_coeff) >= 2) {
    UpdateCoeffGeneral(i, eob, plane_type, tdim, tx_class, quant, first_is_dc,
                       lambda, scan, residuals, coeffs, dequantized_coeffs,
                       levels, &accu_rate, &accu_disto_minus_disto0, counter);
    --i;
  } else {
    assert(std::abs(last_coeff) == 1);
    accu_rate +=
        GetCoeffCost(/*is_last=*/true, eob, last_pos, last_coeff, plane_type,
                     tdim, tx_class, first_is_dc, counter, levels);
    const int16_t res = residuals[last_pos];
    const int16_t dequantized_coeff = dequantized_coeffs[last_pos];
    const int64_t disto = GetDisto(res, dequantized_coeff);
    const int64_t disto0 = GetDisto(res, 0);
    accu_disto_minus_disto0 += disto - disto0;
    --i;
  }

  // Step 2: deal with the coeffs close to the end of block. The end of block
  // can be moved during this step.
  for (; i >= 0 && nz_num <= kMaxNzNum; --i) {
    UpdateCoeffEob(i, plane_type, tdim, tx_class, quant, first_is_dc, lambda,
                   kSharpMode, scan, residuals, coeffs, dequantized_coeffs,
                   levels, &nz_num, nz_pos, &eob, &accu_rate,
                   &accu_disto_minus_disto0, counter);
  }

  // If we've reached the start of the block already, check if we can skip
  // the whole block (set all coeffs to 0).
  if (i == -1 && nz_num <= kMaxNzNum && !kSharpMode) {
    UpdateSkip(lambda, &accu_rate, accu_disto_minus_disto0, &eob, nz_num,
               nz_pos, coeffs, dequantized_coeffs);
  }

  // Step 3: deal with the remaining coeffs except DC.
  for (; i >= 1; --i) {
    UpdateCoeffMiddle(i, eob, plane_type, tdim, tx_class, quant, first_is_dc,
                      lambda, scan, residuals, coeffs, dequantized_coeffs,
                      levels, &accu_rate, counter);
  }

  // Step 4: deal with DC.
  if (i == 0) {
    // no need to update accu_disto_minus_disto0 because it's not used after
    // this point
    int64_t dummy_disto = 0;
    UpdateCoeffGeneral(i, eob, plane_type, tdim, tx_class, quant, first_is_dc,
                       lambda, scan, residuals, coeffs, dequantized_coeffs,
                       levels, &accu_rate, &dummy_disto, counter);
  }

  // Update num_coeffs.
  while (*num_coeffs > 0 && coeffs[*num_coeffs - 1] == 0) {
    --(*num_coeffs);
  }

  if (rate_cost != nullptr) *rate_cost = accu_rate;
}

}  // namespace WP2
