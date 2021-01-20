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
//   AOM residual decoding, branched from libgav1, with as little changes as
//   possible.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/dec/residuals_dec_aom.h"

#include <algorithm>
#include <numeric>

#include "src/utils/utils.h"

namespace WP2 {
namespace libgav1 {

//------------------------------------------------------------------------------
// From libgav1/src/tile/tile.cc

void AOMResidualReader::ReadSignAndApplyDequantization(
    int adjusted_tx_width_log2, int16_t* const coeffs, uint32_t max_value,
    SymbolReader* const sr, ANSDec* const dec) {
  int32_t coeff = *coeffs;
  if (coeff == 0) return;
  if (coeff >= kQuantizerCoefficientBaseRangeContextClamp) {
    coeff = ReadGolomb(kQuantizerCoefficientBaseRangeContextClamp, max_value,
                        /*prefix_size=*/1, dec, "golomb");
  }
  *coeffs = dec->ReadBool("is_negative") ? -coeff : coeff;
}

int AOMResidualReader::ReadCoeffBaseRange(
    int tx_size_context, int cdf_context, int plane_type,
    SymbolReader* const sr) {
  int level = 0;
  for (int j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
    const int coeff_base_range =
        sr->Read(kAOMCoeffBaseRange,
                 GetClusterAOM3<kAOMCoeffBaseRange>(tx_size_context,
                                                    plane_type, cdf_context),
                  "coeff_base_range");
    level += coeff_base_range;
    if (coeff_base_range < (kCoeffBaseRangeSymbolCount - 1)) break;
  }
  return level;
}

int AOMResidualReader::ReadEOB(TrfSize tdim, PlaneType plane_type, int context,
                               SymbolReader* const sr, ANSDec* const dec) {
  const int max_num_coeffs = kNumCoeffs[tdim];
  const int tx_size_context = kTrfContext[tdim];
  const int eob_multi_size = WP2Log2Floor(max_num_coeffs >> 2);
  int eob_pt;
  switch (eob_multi_size) {
    case 0:
      eob_pt =
          sr->Read(kAOMEOBPT4, GetClusterAOM2<kAOMEOBPT4>(plane_type, context),
                   "eob_pt");
      break;
    case 1:
      eob_pt =
          sr->Read(kAOMEOBPT8, GetClusterAOM2<kAOMEOBPT8>(plane_type, context),
                   "eob_pt");
      break;
    case 2:
      eob_pt = sr->Read(kAOMEOBPT16,
                        GetClusterAOM2<kAOMEOBPT16>(plane_type, context),
                        "eob_pt");
      break;
    case 3:
      eob_pt = sr->Read(kAOMEOBPT32,
                        GetClusterAOM2<kAOMEOBPT32>(plane_type, context),
                        "eob_pt");
      break;
    case 4:
      eob_pt = sr->Read(kAOMEOBPT64,
                        GetClusterAOM2<kAOMEOBPT64>(plane_type, context),
                        "eob_pt");
      break;
    case 5:
      eob_pt = sr->Read(kAOMEOBPT128,
                        GetClusterAOM2<kAOMEOBPT128>(plane_type, context),
                        "eob_pt");
      break;
    case 6:
      eob_pt = sr->Read(kAOMEOBPT256,
                        GetClusterAOM2<kAOMEOBPT256>(plane_type, context),
                        "eob_pt");
      break;
    case 7:
      eob_pt = sr->Read(kAOMEOBPT512, GetClusterAOM1<kAOMEOBPT512>(plane_type),
                        "eob_pt");
      break;
    case 8:
    default:
      eob_pt =
          sr->Read(kAOMEOBPT1024, GetClusterAOM1<kAOMEOBPT1024>(plane_type),
                   "eob_pt");
      break;
  }

  int eob = 1;
  if (eob_pt > 0) {
    eob += 1 << (eob_pt - 1);  // leading bit
    if (eob_pt >= 2) {
      eob_pt -= 2;
      const uint32_t bit2 = (1 << eob_pt);  // 2nd-leading bit
      const bool eob_extra = sr->Read(
          kAOMEOBExtra,
          GetClusterAOM3<kAOMEOBExtra>(tx_size_context, plane_type, eob_pt),
          "eob_extra");
      if (eob_extra) eob += bit2;
      // remaining extra bits
      if (bit2 > 1) eob += dec->ReadRange(0, bit2 - 1, "eob_more");
    }
  }
  assert(eob <= max_num_coeffs);
  return eob;
}

uint32_t AOMResidualReader::ReadCoeffs(
    Channel channel, TrfSize tdim, TransformPair tx_type, bool first_is_dc,
    SymbolReader* const sr, ANSDec* const dec, int16_t* const coeffs) {
  const int tx_size_context = kTrfContext[tdim];
  const int tx_width = TrfWidth[tdim];
  const int tx_height = TrfHeight[tdim];
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const int padded_tx_height = tx_height + kQuantizedCoefficientBufferPadding;
  const PlaneType plane_type = GetPlaneType(channel);
  const uint32_t tx_class = (uint32_t)GetTransformClass(tx_type);
  int context = (tx_class != (uint32_t)TransformClass::kTwoD) ? 1 : 0;

  const int eob = ReadEOB(tdim, plane_type, context, sr, dec);

  // Only the first |padded_tx_width| * |padded_tx_height| values of |quantized|
  // will be used by this function and the functions to which it is passed into.
  // So we simply need to zero out those values before it is being used. If
  // |eob| == 1, then only the first index will be populated and used. So there
  // is no need to initialize this array in that case.
  assert(padded_tx_width <= (int)kMaxTxBufferDim);
  assert(padded_tx_height <= (int)kMaxTxBufferDim);
  uint8_t quantized1[kMaxTxBufferDim * kMaxTxBufferDim];
  uint8_t quantized2[kMaxTxBufferDim * kMaxTxBufferDim];
  if (eob > 1) {
    std::fill(quantized1, quantized1 + padded_tx_width * padded_tx_height, 0u);
    std::fill(quantized2, quantized2 + padded_tx_width * padded_tx_height, 0u);
  }
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);
  const int adjusted_tx_width_log2 = TrfLog2[tx_width];
  // Read the last coefficient.
  {
    context = GetCoeffBaseContextEob(tdim, eob - 1);
    const uint16_t pos = scan[eob - 1];
    const uint16_t adjusted_pos = PaddedIndex(pos, adjusted_tx_width_log2);
    const uint32_t cluster =
        GetClusterAOM3<kAOMCoeffBaseEOB>(tx_size_context, plane_type, context);
    int level = 1 + sr->Read(kAOMCoeffBaseEOB, cluster, "level");
    if (level > kNumQuantizerBaseLevels) {
      level += ReadCoeffBaseRange(
          tx_size_context,
          GetCoeffBaseRangeContextEob(adjusted_tx_width_log2, pos),
          plane_type, sr);
    }
    coeffs[pos] = level;
    quantized1[adjusted_pos] = std::min(level, 3);
    quantized2[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  auto get_coeff_base_context_func =
      AOMResidualIO::GetCoeffsBaseContextFunc(tx_class);
  auto get_coeff_base_range_context_func =
      AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class);

  // Read all the other coefficients.
  for (int i = eob - 2; i >= 0; --i) {
    const uint16_t pos = scan[i];
    const uint16_t adjusted_pos = PaddedIndex(pos, adjusted_tx_width_log2);
    context = get_coeff_base_context_func(quantized1, tdim,
                                          adjusted_tx_width_log2, pos);
    const uint32_t cluster =
        GetClusterAOM3<kAOMCoeffBase>(tx_size_context, plane_type, context);
    int level = sr->Read(kAOMCoeffBase, cluster, "level");
    if (level > kNumQuantizerBaseLevels) {
      const uint32_t cdf_context = get_coeff_base_range_context_func(
          quantized2, adjusted_tx_width_log2, pos);
      level += ReadCoeffBaseRange(tx_size_context, cdf_context, plane_type, sr);
    }
    coeffs[pos] = level;
    quantized1[adjusted_pos] = std::min(level, 3);
    quantized2[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }
  assert(scan[0] == 0);
  if (first_is_dc) {
      ReadSignAndApplyDequantization(
          adjusted_tx_width_log2, &coeffs[0], kMaxDcValue, sr, dec);
  }
  uint32_t last_coeff = (first_is_dc && coeffs[0]) ? 1 : 0;
  for (int i = (first_is_dc ? 1 : 0); i < eob; ++i) {
    if (coeffs[scan[i]]) {
      ReadSignAndApplyDequantization(
          adjusted_tx_width_log2, &coeffs[scan[i]], kMaxCoeffValue, sr, dec);
      last_coeff = std::max(last_coeff, scan[i] + 1u);
    }
  }
  return last_coeff;
}

}  // namespace libgav1
}  // namespace WP2
