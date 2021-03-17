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

int16_t AOMResidualReader::ReadSign(int16_t coeff, uint32_t max_value,
                                    SymbolReader* const sr) {
  assert(coeff > 0);
  if (coeff >= (int16_t)kQuantizerCoefficientBaseRangeContextClamp) {
    coeff = ReadGolomb(kQuantizerCoefficientBaseRangeContextClamp, max_value,
                       /*prefix_size=*/0, sr->dec(), "golomb");
  }
  return (sr->dec()->ReadBool("is_negative") ? -coeff : coeff);
}

int AOMResidualReader::ReadCoeffBaseRange(
    uint32_t cdf_context, uint32_t plane_type, SymbolReader* const sr) {
  int level = 0;
  for (uint32_t j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
    const uint32_t coeff_base_range =
        sr->Read(kAOMCoeffBaseRange,
                 GetClusterAOM2<kAOMCoeffBaseRange>(plane_type, cdf_context),
                  "coeff_base_range");
    level += coeff_base_range;
    if (coeff_base_range < kCoeffBaseRangeSymbolCount) break;
  }
  return level;
}

int AOMResidualReader::ReadEOB(TrfSize tdim, PlaneType plane_type,
                               uint32_t context, SymbolReader* const sr) {
  const int max_num_coeffs = kNumCoeffs[tdim];
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
          kAOMEOBExtra, GetClusterAOM2<kAOMEOBExtra>(plane_type, eob_pt),
          "eob_extra");
      if (eob_extra) eob += bit2;
      // remaining extra bits
      if (bit2 > 1) eob += sr->dec()->ReadRange(0, bit2 - 1, "eob_more");
    }
  }
  assert(eob <= max_num_coeffs);
  return eob;
}

uint32_t AOMResidualReader::ReadCoeffs(
    Channel channel, const int16_t dequant[],
    TrfSize tdim, TransformPair tx_type, bool first_is_dc,
    SymbolReader* const sr, int16_t* const coeffs) {
  const uint32_t tx_width = TrfWidth[tdim];
  const uint32_t tx_height = TrfHeight[tdim];
  const uint32_t padded_tx_width = tx_width + kLevelBufferPadding;
  const uint32_t padded_tx_height = tx_height + kLevelBufferPadding;
  const PlaneType plane_type = GetPlaneType(channel);
  const uint32_t tx_class = (uint32_t)GetTransformClass(tx_type);

  // Read EOB
  const uint32_t context =
      (tx_class != (uint32_t)TransformClass::kTwoD) ? 1 : 0;
  const int eob = ReadEOB(tdim, plane_type, context, sr);

  // If eob == 1, then only the first index will be populated and used. So there
  // is no need to initialize this array in that case.
  assert(padded_tx_width <= (int)kMaxTxBufferDim);
  assert(padded_tx_height <= (int)kMaxTxBufferDim);
  uint8_t levels[kMaxTxBufferDim * kMaxTxBufferDim];
  if (eob > 1) {
    std::fill(levels, levels + padded_tx_width * padded_tx_height, 0u);
  }
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);
  const uint32_t tx_width_log2 = TrfLog2[tx_width];
  // Read the last coefficient.
  {
    const uint16_t pos = scan[eob - 1];
    const uint16_t adjusted_pos = PaddedIndex(pos, tx_width_log2);
    const uint32_t base_context = GetCoeffBaseContextEob(tdim, eob - 1);
    const uint32_t cluster =
        GetClusterAOM2<kAOMCoeffBaseEOB>(plane_type, base_context);
    uint32_t level = 1 + sr->Read(kAOMCoeffBaseEOB, cluster, "level");
    if (level == kNumQuantizerBaseLevels) {
      level += ReadCoeffBaseRange(
          GetCoeffBaseRangeContextEob(tx_width_log2, pos), plane_type, sr);
    }
    coeffs[pos] = level;
    levels[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  auto get_coeff_base_context_func =
      AOMResidualIO::GetCoeffsBaseContextFunc(tx_class);
  auto get_coeff_base_range_context_func =
      AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class);

  // Read all the other coefficients.
  for (int i = eob - 2; i >= 0; --i) {
    const uint16_t pos = scan[i];
    const uint16_t adjusted_pos = PaddedIndex(pos, tx_width_log2);
    const uint32_t base_context =
        get_coeff_base_context_func(levels, tdim, tx_width_log2, pos);
    const uint32_t cluster =
        GetClusterAOM2<kAOMCoeffBase>(plane_type, base_context);
    uint32_t level = sr->Read(kAOMCoeffBase, cluster, "level");
    if (level == kNumQuantizerBaseLevels) {
      const uint32_t cdf_context = get_coeff_base_range_context_func(
          levels, tx_width_log2, pos);
      level += ReadCoeffBaseRange(cdf_context, plane_type, sr);
    }
    coeffs[pos] = level;
    levels[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }
  assert(scan[0] == 0);
  uint32_t num_coeffs = (first_is_dc && coeffs[0]) ? 1 : 0;
  if (num_coeffs == 1) {
    coeffs[0] = ReadSign(coeffs[0], kMaxDcValue, sr);
    coeffs[0] *= dequant[0];
  }

  for (int i = (first_is_dc ? 1 : 0); i < eob; ++i) {
    const uint32_t pos = scan[i];
    if (coeffs[pos]) {
      coeffs[pos] = ReadSign(coeffs[pos], kMaxCoeffValue, sr);
      const uint32_t qpos = (pos % tx_width) + WP2QStride * (pos / tx_width);
      coeffs[pos] *= dequant[qpos];
      ++num_coeffs;
    }
  }
  return num_coeffs;
}

}  // namespace libgav1
}  // namespace WP2
