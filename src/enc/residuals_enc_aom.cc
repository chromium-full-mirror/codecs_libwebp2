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
//   AOM residual encoding, reverse-engineered from libgav1 decoder.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/enc/residuals_enc_aom.h"

#include "src/enc/wp2_enc_i.h"

namespace WP2 {
namespace libgav1 {

void AOMResidualWriter::WriteSign(
    int16_t res, uint32_t max_value,
    SymbolManager* const sm, ANSEncBase* const enc) {
  const uint32_t val = std::abs(res);
  assert(res != 0);
  assert(val <= max_value);
  if (val >= kQuantizerCoefficientBaseRangeContextClamp) {
    WriteGolomb(val, kQuantizerCoefficientBaseRangeContextClamp, max_value,
                /*prefix_size=*/1, enc, "golomb");
  }
  enc->PutBool(res < 0, "is_negative");
}

void AOMResidualWriter::WriteCoeffBaseRange(
    uint32_t level, uint32_t cdf_context, uint32_t plane_type,
    SymbolManager* const sm, ANSEncBase* const enc) {
  for (uint32_t j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
    const uint32_t chunk =
        std::min(level, (uint32_t)kCoeffBaseRangeSymbolCount);
    sm->Process(kAOMCoeffBaseRange,
                GetClusterAOM2<kAOMCoeffBaseRange>(plane_type, cdf_context),
                chunk, "coeff_base_range", enc);
    if (chunk < kCoeffBaseRangeSymbolCount) break;
    level -= chunk;
  }
}

void AOMResidualWriter::WriteEOB(
    int eob, TrfSize tdim, PlaneType plane_type,
    uint32_t context, SymbolManager* const sm, ANSEncBase* const enc) {
  const uint32_t eob_multi_size = WP2Log2Floor(kNumCoeffs[tdim] >> 2);

  eob -= 1;  // in [0..max_num_coeffs)
  int16_t eob_pt = 0;
  if (eob > 0) eob_pt = 1 + WP2Log2Floor(eob);

  assert(eob_pt <= kEobPt1024SymbolCount);
  switch (eob_multi_size) {
    case 0:
      sm->Process(kAOMEOBPT4, GetClusterAOM2<kAOMEOBPT4>(plane_type, context),
                  eob_pt, "eob_pt", enc);
      break;
    case 1:
      sm->Process(kAOMEOBPT8, GetClusterAOM2<kAOMEOBPT8>(plane_type, context),
                  eob_pt, "eob_pt", enc);
      break;
    case 2:
      sm->Process(kAOMEOBPT16, GetClusterAOM2<kAOMEOBPT16>(plane_type, context),
                  eob_pt, "eob_pt", enc);
      break;
    case 3:
      sm->Process(kAOMEOBPT32, GetClusterAOM2<kAOMEOBPT32>(plane_type, context),
                  eob_pt, "eob_pt", enc);
      break;
    case 4:
      sm->Process(kAOMEOBPT64, GetClusterAOM2<kAOMEOBPT64>(plane_type, context),
                  eob_pt, "eob_pt", enc);
      break;
    case 5:
      sm->Process(kAOMEOBPT128,
                  GetClusterAOM2<kAOMEOBPT128>(plane_type, context), eob_pt,
                  "eob_pt", enc);
      break;
    case 6:
      sm->Process(kAOMEOBPT256,
                  GetClusterAOM2<kAOMEOBPT256>(plane_type, context), eob_pt,
                  "eob_pt", enc);
      break;
    case 7:
      sm->Process(kAOMEOBPT512, GetClusterAOM1<kAOMEOBPT512>(plane_type),
                  eob_pt, "eob_pt", enc);
      break;
    case 8:
    default:
      sm->Process(kAOMEOBPT1024, GetClusterAOM1<kAOMEOBPT1024>(plane_type),
                  eob_pt, "eob_pt", enc);
      break;
  }

  if (eob_pt >= 2) {  // eob > 1
    eob_pt -= 2;
    const bool eob_extra = (eob >> eob_pt) & 1;
    sm->Process(
        kAOMEOBExtra,
        GetClusterAOM2<kAOMEOBExtra>(plane_type, eob_pt),
        eob_extra, "eob_extra", enc);
    // remaining extra bits
    const uint32_t bit2 = 1 << eob_pt;   // second-leading bit
    if (bit2 > 1) enc->PutRange(eob & (bit2 - 1), 0, bit2 - 1, "eob_more");
  }
}

void AOMResidualWriter::WriteCoeffs(
    const int16_t res[], Channel channel, TrfSize tdim, TransformPair tx_type,
    bool first_is_dc, SymbolManager* const sm, ANSEncBase* const enc) {
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);
  int eob = kNumCoeffs[tdim];
  while ((eob > 0) && (res[scan[eob - 1]] == 0)) --eob;
  if (eob == 0) return;  // all_zero

  const uint32_t tx_width = TrfWidth[tdim];
  const uint32_t tx_height = TrfHeight[tdim];
  const uint32_t padded_tx_width = tx_width + kLevelBufferPadding;
  const uint32_t padded_tx_height = tx_height + kLevelBufferPadding;
  const PlaneType plane_type = GetPlaneType(channel);
  const uint32_t tx_class = (uint32_t)GetTransformClass(tx_type);

  // Write EOB
  const uint32_t context =
      (tx_class != (uint32_t)TransformClass::kTwoD) ? 1 : 0;
  WriteEOB(eob, tdim, plane_type, context, sm, enc);

  assert(padded_tx_width <= (int)kMaxTxBufferDim);
  assert(padded_tx_height <= (int)kMaxTxBufferDim);
  uint8_t levels[kMaxTxBufferDim * kMaxTxBufferDim];
  if (eob > 1) {
    std::fill(levels, levels + padded_tx_width * padded_tx_height, 0u);
  }
  const uint32_t tx_width_log2 = TrfLog2[tx_width];

  // Write the last coefficient.
  {
    const uint16_t pos = scan[eob - 1];
    const uint16_t adjusted_pos = PaddedIndex(pos, tx_width_log2);
    uint32_t level = std::abs(res[pos]);
    assert(level != 0);
    const uint32_t base_level = std::min(level, kNumQuantizerBaseLevels);
    const uint32_t base_context = GetCoeffBaseContextEob(tdim, eob - 1);
    const uint32_t cluster =
        GetClusterAOM2<kAOMCoeffBaseEOB>(plane_type, base_context);
    sm->Process(kAOMCoeffBaseEOB, cluster, base_level - 1, "level", enc);
    if (base_level == kNumQuantizerBaseLevels) {
      const uint32_t cdf_context =
          GetCoeffBaseRangeContextEob(tx_width_log2, pos);
      WriteCoeffBaseRange(level - kNumQuantizerBaseLevels, cdf_context,
                          plane_type, sm, enc);
    }
    levels[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  const auto get_coeff_base_context_func =
      AOMResidualIO::GetCoeffsBaseContextFunc(tx_class);
  const auto get_coeff_base_range_context_func =
      AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class);

  // Write all the other coefficients.
  for (int i = eob - 2; i >= 0; --i) {
    const uint16_t pos = scan[i];
    const uint16_t adjusted_pos = PaddedIndex(pos, tx_width_log2);
    const uint32_t level = std::abs(res[pos]);
    const uint32_t base_level = std::min(level, kNumQuantizerBaseLevels);
    const uint32_t base_context =
        get_coeff_base_context_func(levels, tdim, tx_width_log2, pos);
    const uint32_t cluster =
        GetClusterAOM2<kAOMCoeffBase>(plane_type, base_context);
    sm->Process(kAOMCoeffBase, cluster, base_level, "level", enc);
    if (base_level == kNumQuantizerBaseLevels) {
      const uint32_t cdf_context = get_coeff_base_range_context_func(
          levels, tx_width_log2, pos);
      WriteCoeffBaseRange(level - kNumQuantizerBaseLevels, cdf_context,
                          plane_type, sm, enc);
    }
    levels[adjusted_pos] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  assert(scan[0] == 0);
  if (first_is_dc && res[0]) WriteSign(res[0], kMaxDcValue, sm, enc);
  for (int i = (first_is_dc ? 1 : 0); i < eob; ++i) {
    if (res[scan[i]]) WriteSign(res[scan[i]], kMaxCoeffValue, sm, enc);
  }
}

}  // namespace libgav1
}  // namespace WP2
