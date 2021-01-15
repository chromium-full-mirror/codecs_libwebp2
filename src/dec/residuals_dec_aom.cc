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

template <bool is_dc_coefficient>
void AOMResidualReader::ReadSignAndApplyDequantization(
    int adjusted_tx_width_log2, uint16_t dc_sign_cluster, int16_t* const coeffs,
    int* const coefficient_level) {
  int32_t coeff = *coeffs;
  if (coeff == 0) return;
  const bool sign = is_dc_coefficient ? sr_->Read(kAOMDCSign, dc_sign_cluster,
                                                  "is_negative_dc")
                                      : dec_->ReadBool("is_negative");
  if (coeff >= kQuantizerCoefficientBaseRangeContextClamp) {
    constexpr int16_t max_range = is_dc_coefficient ? kMaxDcValue
                                                    : kMaxCoeffValue;
    coeff = ReadGolomb(kQuantizerCoefficientBaseRangeContextClamp, max_range,
                        /*prefix_size=*/1, dec_, "golomb");
  }
  *coefficient_level += coeff;
  if (sign) coeff = -coeff;
  *coeffs = coeff;
}

int AOMResidualReader::ReadCoeffBaseRange(int clamped_tx_size_context,
                                          int cdf_context, int plane_type) {
  int level = 0;
  for (int j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
    const int coeff_base_range =
        sr_->Read(kAOMCoeffBaseRange,
                  GetClusterAOM3<kAOMCoeffBaseRange>(clamped_tx_size_context,
                                                     plane_type, cdf_context),
                  "coeff_base_range");
    level += coeff_base_range;
    if (coeff_base_range < (kCoeffBaseRangeSymbolCount - 1)) break;
  }
  return level;
}

void AOMResidualReader::ReadCoeffs(
    Channel channel, uint32_t x_pix, uint32_t y_pix, WP2::BlockSize dim,
    bool is420, TransformPair tx_type, bool first_is_dc, int max_num_coeffs,
    ANSDec* const dec, SymbolReader* const sr,
    int16_t* const coeffs, uint32_t* const num_coeffs) {
  // Convert WP2 data to libgav1 data.
  const bool is_reduced =
      (channel == kUChannel || channel == kVChannel) && is420;
  const TransformSize tx_size = ConvertSize(dim);
  dec_ = dec;
  sr_ = sr;

  const uint32_t subsampling = is_reduced ? 1u : 0u;

  const int num_coeffs_in_transform = ReadTransformCoefficients(
      channel, is_reduced, x_pix >> subsampling, y_pix >> subsampling,
      tx_size, tx_type, first_is_dc, max_num_coeffs, coeffs);
  assert(num_coeffs_in_transform >= 0);
  *num_coeffs = (uint32_t)num_coeffs_in_transform;
}

int AOMResidualReader::ReadEOB(int max_num_coeffs,
    TransformSize tx_size, PlaneType plane_type, int context) {
  const int tx_size_context = kTransformSizeContext[tx_size];
  assert((max_num_coeffs % 4) == 0 && max_num_coeffs <= 1024);
  const int eob_multi_size = WP2Log2Floor(max_num_coeffs >> 2);
  int eob_pt;
  switch (eob_multi_size) {
    case 0:
      eob_pt =
          sr_->Read(kAOMEOBPT4, GetClusterAOM2<kAOMEOBPT4>(plane_type, context),
                    "eob_pt");
      break;
    case 1:
      eob_pt =
          sr_->Read(kAOMEOBPT8, GetClusterAOM2<kAOMEOBPT8>(plane_type, context),
                    "eob_pt");
      break;
    case 2:
      eob_pt = sr_->Read(kAOMEOBPT16,
                         GetClusterAOM2<kAOMEOBPT16>(plane_type, context),
                         "eob_pt");
      break;
    case 3:
      eob_pt = sr_->Read(kAOMEOBPT32,
                         GetClusterAOM2<kAOMEOBPT32>(plane_type, context),
                         "eob_pt");
      break;
    case 4:
      eob_pt = sr_->Read(kAOMEOBPT64,
                         GetClusterAOM2<kAOMEOBPT64>(plane_type, context),
                         "eob_pt");
      break;
    case 5:
      eob_pt = sr_->Read(kAOMEOBPT128,
                         GetClusterAOM2<kAOMEOBPT128>(plane_type, context),
                         "eob_pt");
      break;
    case 6:
      eob_pt = sr_->Read(kAOMEOBPT256,
                         GetClusterAOM2<kAOMEOBPT256>(plane_type, context),
                         "eob_pt");
      break;
    case 7:
      eob_pt = sr_->Read(kAOMEOBPT512, GetClusterAOM1<kAOMEOBPT512>(plane_type),
                         "eob_pt");
      break;
    case 8:
    default:
      eob_pt =
          sr_->Read(kAOMEOBPT1024, GetClusterAOM1<kAOMEOBPT1024>(plane_type),
                    "eob_pt");
      break;
  }

  int eob = 1;
  if (eob_pt > 0) {
    eob += 1 << (eob_pt - 1);  // leading bit
    if (eob_pt >= 2) {
      eob_pt -= 2;
      const uint32_t bit2 = (1 << eob_pt);  // 2nd-leading bit
      const bool eob_extra = sr_->Read(
          kAOMEOBExtra,
          GetClusterAOM3<kAOMEOBExtra>(tx_size_context, plane_type, eob_pt),
          "eob_extra");
      if (eob_extra) eob += bit2;
      // remaining extra bits
      if (bit2 > 1) eob += dec_->ReadRange(0, bit2 - 1, "eob_more");
    }
  }
  assert(eob <= max_num_coeffs);
  return eob;
}

uint32_t AOMResidualReader::ReadTransformCoefficients(
    Channel channel, bool is_reduced, int start_x, int start_y,
    TransformSize tx_size, TransformPair tx_type,
    bool first_is_dc, int max_num_coeffs, int16_t* const coeffs) {
  if (max_num_coeffs == 0) {
    // This is not used in this case, so it can be set to any value.
    return 0;
  }
  const int tx_size_context = kTransformSizeContext[tx_size];
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

  const int eob = ReadEOB(max_num_coeffs, tx_size, plane_type, context);

  // Only the first |padded_tx_width| * |padded_tx_height| values of |quantized|
  // will be used by this function and the functions to which it is passed into.
  // So we simply need to zero out those values before it is being used. If
  // |eob| == 1, then only the first index will be populated and used. So there
  // is no need to initialize this array in that case.
  uint8_t quantized[2048], quantized2[2048];
  if (eob > 1) {
    std::fill(quantized, quantized + padded_tx_width * padded_tx_height, 0u);
    std::fill(quantized2, quantized2 + padded_tx_width * padded_tx_height, 0u);
  }
  const uint16_t* scan = UberZigZag(tx_size, is_reduced);
  const TransformSize adjusted_tx_size = kAdjustedTransformSize[tx_size];
  const int adjusted_tx_width_log2 = kTransformWidthLog2[adjusted_tx_size];
  const int clamped_tx_size_context = std::min(tx_size_context, 3);
  // Read the last coefficient.
  {
    context = GetCoeffBaseContextEob(tx_size, eob - 1);
    const uint16_t pos = scan[eob - 1];
    const uint32_t cluster =
        GetClusterAOM3<kAOMCoeffBaseEOB>(tx_size_context, plane_type, context);
    int level = 1 + sr_->Read(kAOMCoeffBaseEOB, cluster, "level");
    if (level > kNumQuantizerBaseLevels) {
      level += ReadCoeffBaseRange(
          clamped_tx_size_context,
          GetCoeffBaseRangeContextEob(adjusted_tx_width_log2, pos), plane_type);
    }
    coeffs[pos] = level;
    quantized[PaddedIndex(pos, adjusted_tx_width_log2)] = std::min(level, 3);
    quantized2[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }

  auto get_coeff_base_context_func =
      AOMResidualIO::GetCoeffsBaseContextFunc(tx_class);
  auto get_coeff_base_range_context_func =
      AOMResidualIO::GetCoeffBaseRangeContextFunc(tx_class);

  // Read all the other coefficients.
  for (int i = eob - 2; i >= 0; --i) {
    const uint16_t pos = scan[i];

    context = get_coeff_base_context_func(quantized, tx_size,
                                          adjusted_tx_width_log2, pos);
    const uint32_t cluster =
        GetClusterAOM3<kAOMCoeffBase>(tx_size_context, plane_type, context);
    int level = sr_->Read(kAOMCoeffBase, cluster, "level");
    if (level > kNumQuantizerBaseLevels) {
      const uint32_t cdf_context = get_coeff_base_range_context_func(
          quantized2, adjusted_tx_width_log2, pos);
      level +=
          ReadCoeffBaseRange(clamped_tx_size_context, cdf_context, plane_type);
    }
    coeffs[pos] = level;
    quantized[PaddedIndex(pos, adjusted_tx_width_log2)] = std::min(level, 3);
    quantized2[PaddedIndex(pos, adjusted_tx_width_log2)] =
        std::min(level, kQuantizerCoefficientBaseRangeContextClamp);
  }
  int coefficient_level = 0;
  assert(scan[0] == 0);
  if (first_is_dc) {
     const uint32_t dc_sign_cluster =
        (quantized[0] != 0) ? GetClusterAOM1<kAOMDCSign>(plane_type) : 0;
      ReadSignAndApplyDequantization</*is_dc_coefficient=*/true>(
          adjusted_tx_width_log2, dc_sign_cluster, &coeffs[0],
          &coefficient_level);
  }
  uint32_t last_coeff = (first_is_dc && coeffs[0]) ? 1 : 0;
  for (int i = (first_is_dc ? 1 : 0); i < eob; ++i) {
    if (coeffs[scan[i]]) {
      ReadSignAndApplyDequantization</*is_dc_coefficient=*/false>(
          adjusted_tx_width_log2, /*dc_sign_cluster=*/0,
          &coeffs[scan[i]], &coefficient_level);
      last_coeff = std::max(last_coeff, scan[i] + 1u);
    }
  }

  return last_coeff;
}

}  // namespace libgav1
}  // namespace WP2
