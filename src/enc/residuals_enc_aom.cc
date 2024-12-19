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

#include <algorithm>
#include <cstdlib>

#include "src/enc/wp2_enc_i.h"

namespace WP2 {
namespace libgav1 {

void AOMResidualWriter::WriteSignAndRest(int16_t res, bool is_dc,
                                         SymbolManager* const sm,
                                         ANSEncBase* const enc) {
  const uint32_t val = std::abs(res);
  assert(res != 0);
  if (val >= kQuantizerCoefficientBaseRangeContextClamp) {
    const uint32_t max_value = is_dc ? kMaxDcValue : kMaxCoeffValue;
    assert(val <= max_value);
    WritePrefixCode(val, kQuantizerCoefficientBaseRangeContextClamp, max_value,
                    /*prefix_size=*/0, enc, "prefix_code");
  }
  enc->PutBool(res < 0, "is_negative");
}

void AOMResidualWriter::WriteCoeff(int16_t res, uint32_t pos, bool is_eob,
                                   uint32_t eob, bool first_is_dc,
                                   PlaneType plane_type, TrfSize tdim,
                                   TransformClass tx_class,
                                   const uint8_t* const levels,
                                   SymbolManager* const sm,
                                   ANSEncBase* const enc) {
  const uint32_t tx_width_log2 = TrfLog2[TrfWidth[tdim]];
  const uint32_t level = std::abs(res);
  const uint32_t base_level = std::min(level, kNumQuantizerBaseLevels);

  const uint32_t base_context =
      is_eob ? GetCoeffBaseContextEob(tdim, eob - 1)
             : AOMResidualIO::GetCoeffsBaseContextFunc(tx_class)(
                   levels, tdim, tx_width_log2, pos);
  const uint32_t cluster =
      is_eob ? GetClusterAOM2<kAOMCoeffBaseEOB>(plane_type, base_context)
             : GetClusterAOM2<kAOMCoeffBase>(plane_type, base_context);

  sm->Process(is_eob ? kAOMCoeffBaseEOB : kAOMCoeffBase, cluster,
              is_eob ? base_level - 1 : base_level, "level", enc);
  if (level >= kNumQuantizerBaseLevels) {
    const uint32_t cdf_context =
        is_eob ? GetCoeffBaseRangeContextEob(tx_width_log2, pos)
               : AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class)(
                     levels, tx_width_log2, pos);
    WriteCoeffBaseRange(level - kNumQuantizerBaseLevels, cdf_context,
                        plane_type, sm, enc);
  }
  if (base_level > 0) {
    const bool is_dc = (pos == 0) && first_is_dc;
    WriteSignAndRest(res, is_dc, sm, enc);
  }
}

void AOMResidualWriter::WriteCoeffBaseRange(uint32_t level,
                                            uint32_t cdf_context,
                                            PlaneType plane_type,
                                            SymbolManager* const sm,
                                            ANSEncBase* const enc) {
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

void AOMResidualWriter::WriteEOB(int eob, TrfSize tdim, PlaneType plane_type,
                                 TransformClass tx_class,
                                 SymbolManager* const sm,
                                 ANSEncBase* const enc) {
  const uint32_t context = (tx_class != TransformClass::kTwoD) ? 1 : 0;
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
  const TransformClass tx_class = GetTransformClass(tx_type);

  // Write EOB
  WriteEOB(eob, tdim, plane_type, tx_class, sm, enc);

  assert(padded_tx_width <= (int)kMaxTxBufferDim);
  assert(padded_tx_height <= (int)kMaxTxBufferDim);
  uint8_t levels[kMaxTxBufferDim * kMaxTxBufferDim];
  if (eob > 1) {
    std::fill(levels, levels + padded_tx_width * padded_tx_height, 0u);
  }

  for (int i = eob - 1; i >= 0; --i) {
    const uint16_t pos = scan[i];
    const bool is_eob = (i == eob - 1);
    WriteCoeff(res[pos], pos, is_eob, eob, first_is_dc, plane_type, tdim,
               tx_class, levels, sm, enc);
    UpdateLevels(res[pos], pos, tdim, levels);
  }
}

}  // namespace libgav1
}  // namespace WP2
