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

WP2Status AOMResidualWriter::WriteCoeffs(
    uint32_t x_pix, uint32_t y_pix, WP2::BlockSize dim, TransformPair tx_type,
    bool first_is_dc, int max_num_coeffs, bool is420,
    const int16_t* const coeffs, Channel channel,
    SymbolManager* const sm, ANSEncBase* const enc) {
  // Convert WP2 data to libgav1 data.
  const TransformSize tx_size = ConvertSize(dim);
  const bool is_reduced =
      ((channel == kUChannel || channel == kVChannel) && is420);

  // Write the coefficients.
  int coefficient_level;
  const uint32_t subsampling = is_reduced ? 1 : 0;
  WriteTransformCoefficients(coeffs, max_num_coeffs, is_reduced,
                             channel,
                             x_pix >> subsampling, y_pix >> subsampling,
                             tx_size, tx_type, first_is_dc, sm, enc,
                             &coefficient_level);
  return WP2_STATUS_OK;
}

// 'levels' contains the quantized coefficients in zigzag order.
template <bool is_dc_coefficient>
void AOMResidualWriter::WriteSignAndApplyDequantization(
    int16_t res, uint32_t dc_sign_cluster, int* coefficient_level,
    SymbolManager* const sm, ANSEncBase* const enc) {
  assert(res != 0);
  const bool sign = (res < 0);
  int32_t val = std::abs(res);
  val = std::min(is_dc_coefficient ? kMaxDcValue : kMaxCoeffValue, (int)val);
  if (is_dc_coefficient) {
    sm->Process(kAOMDCSign, dc_sign_cluster, sign, "is_negative_dc", enc);
  } else {
    enc->PutBool(sign, "is_negative");
  }
  *coefficient_level += val;
  if (val >= kQuantizerCoefficientBaseRangeContextClamp) {
    constexpr int16_t max_range = is_dc_coefficient ? kMaxDcValue
                                                    : kMaxCoeffValue;
    WriteGolomb(val, kQuantizerCoefficientBaseRangeContextClamp, max_range,
                /*prefix_size=*/1, enc, "golomb");
  }
}

uint32_t AOMResidualWriter::WriteCoeffBaseRange(uint32_t level,
                                                int clamped_tx_size_context,
                                                int cdf_context, int plane_type,
                                                SymbolManager* const sm,
                                                ANSEncBase* const enc) {
  uint32_t sum = 0;
  for (int j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
    uint32_t chunk = std::min(level, (uint32_t)kCoeffBaseRangeSymbolCount - 1);
    sum += chunk;
    sm->Process(kAOMCoeffBaseRange,
                GetClusterAOM3<kAOMCoeffBaseRange>(clamped_tx_size_context,
                                                   plane_type, cdf_context),
                chunk, "coeff_base_range", enc);
    level -= chunk;
    if (chunk < (kCoeffBaseRangeSymbolCount - 1)) break;
  }
  return sum;
}

void AOMResidualWriter::WriteEOB(
    int eob, int max_num_coeffs, TransformSize tx_size, PlaneType plane_type,
    int context, SymbolManager* const sm, ANSEncBase* const enc) {
  const int tx_size_context = kTransformSizeContext[tx_size];
  assert((max_num_coeffs % 4) == 0 && max_num_coeffs <= 1024);
  const int eob_multi_size = WP2Log2Floor(max_num_coeffs >> 2);

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
        GetClusterAOM3<kAOMEOBExtra>(tx_size_context, plane_type, eob_pt),
        eob_extra, "eob_extra", enc);
    // remaining extra bits
    const uint32_t bit2 = 1 << eob_pt;   // second-leading bit
    if (bit2 > 1) enc->PutRange(eob & (bit2 - 1), 0, bit2 - 1, "eob_more");
  }
}

void AOMResidualWriter::WriteTransformCoefficients(
    const int16_t* const res, int max_num_coeffs, bool is_reduced,
    Channel channel, int start_x, int start_y, TransformSize tx_size,
    TransformPair tx_type, bool first_is_dc, SymbolManager* const sm,
    ANSEncBase* const enc, int* const coefficient_level) {
  const int tx_size_context = kTransformSizeContext[tx_size];
  int eob = max_num_coeffs - 1;
  const uint16_t* scan = UberZigZag(tx_size, is_reduced);
  for (; eob >= 0 && res[scan[eob]] == 0; --eob) {
  }
  ++eob;  // is actually the size.
  *coefficient_level = 0;
  if (eob == 0) return;  // all_zero

  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  const int clamped_tx_width = std::min(tx_width, 32);
  const int clamped_tx_height = std::min(tx_height, 32);
  const int padded_tx_width =
      clamped_tx_width + kQuantizedCoefficientBufferPadding;
  const int padded_tx_height =
      clamped_tx_height + kQuantizedCoefficientBufferPadding;
  const PlaneType plane_type = GetPlaneType(channel);

  const uint32_t tx_class = (uint32_t)GetTransformClass(tx_type);
  int context = (tx_class != (uint32_t)TransformClass::kTwoD) ? 1 : 0;

  WriteEOB(eob, max_num_coeffs, tx_size, plane_type, context, sm, enc);

  uint8_t quantized[2048], quantized2[2048];
  int16_t levels[kMaxBlockSizePix2];   // non-padded
  std::fill(quantized, quantized + padded_tx_width * padded_tx_height, 0u);
  std::fill(quantized2, quantized2 + padded_tx_width * padded_tx_height, 0u);
  const TransformSize adjusted_tx_size = kAdjustedTransformSize[tx_size];
  const int adjusted_tx_width_log2 = kTransformWidthLog2[adjusted_tx_size];
  const int clamped_tx_size_context = std::min(tx_size_context, 3);

  // Write the last coefficient.
  {
    const uint16_t pos = scan[eob - 1];
    uint32_t level = std::abs(res[pos]);
    assert(level != 0);
    --level;
    uint32_t chunk = std::min(level, (uint32_t)kCoeffBaseEobSymbolCount - 1);
    context = GetCoeffBaseContextEob(tx_size, eob - 1);
    sm->Process(
        kAOMCoeffBaseEOB,
        GetClusterAOM3<kAOMCoeffBaseEOB>(tx_size_context, plane_type, context),
        chunk, "level", enc);
    int real_level = 1 + chunk;
    if (1 + chunk > kNumQuantizerBaseLevels) {
      level -= chunk;
      real_level += WriteCoeffBaseRange(
          level, clamped_tx_size_context,
          GetCoeffBaseRangeContextEob(adjusted_tx_width_log2, pos), plane_type,
          sm, enc);
    }
    levels[eob - 1] = real_level;
    quantized[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(real_level, 3);
    quantized2[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(real_level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  const auto get_coeff_base_context_func =
      AOMResidualIO::GetCoeffsBaseContextFunc(tx_class);
  const auto get_coeff_base_range_context_func =
      AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class);

  // Write all the other coefficients.
  for (int i = eob - 2; i >= 0; --i) {
    const uint16_t pos = scan[i];
    context = get_coeff_base_context_func(quantized, tx_size,
                                          adjusted_tx_width_log2, pos);
    const uint32_t level_ini = std::abs(res[pos]);
    uint32_t chunk = std::min(level_ini, (uint32_t)kCoeffBaseEobSymbolCount);
    sm->Process(
        kAOMCoeffBase,
        GetClusterAOM3<kAOMCoeffBase>(tx_size_context, plane_type, context),
        chunk, "level", enc);
    int real_level = chunk;
    if (chunk > kNumQuantizerBaseLevels) {
      const uint32_t cdf_context = get_coeff_base_range_context_func(
          quantized2, adjusted_tx_width_log2, pos);
      const uint32_t level = level_ini - chunk;
      real_level += WriteCoeffBaseRange(level, clamped_tx_size_context,
                                        cdf_context, plane_type, sm, enc);
    }
    quantized[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(real_level, 3);
    quantized2[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(real_level, kQuantizerCoefficientBaseRangeContextClamp);
    levels[i] = real_level;
  }

  assert(scan[0] == 0);
  if (first_is_dc && res[0]) {
    const uint32_t dc_sign_cluster = GetClusterAOM1<kAOMDCSign>(plane_type);
    WriteSignAndApplyDequantization<true>(res[0], dc_sign_cluster,
                                          coefficient_level, sm, enc);
  }
  for (int i = (first_is_dc ? 1 : 0); i < eob; ++i) {
    if (levels[i] == 0) continue;
    WriteSignAndApplyDequantization<false>(res[scan[i]], /*dc_sign_cluster=*/0,
                                           coefficient_level, sm, enc);
  }
}

}  // namespace libgav1
}  // namespace WP2
