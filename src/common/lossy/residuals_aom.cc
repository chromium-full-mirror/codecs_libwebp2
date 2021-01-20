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
//   Common code for AOM residuals, branched from libgav1, with as little
//   changes as possible.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/common/lossy/residuals_aom.h"

#include "src/utils/utils.h"

namespace WP2 {
namespace libgav1 {

//------------------------------------------------------------------------------
// From libgav1/src/utils/constants.cc

namespace {

/* clang-format off */
constexpr uint8_t kCoeffBaseContextOffset[TRF_LAST][5][5] = {
    {{0,  1,  0,  0, 0},
     {1,  6,  0,  0, 0},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0}},    // 2x2

    {{0,  1,  6,  6, 0},
     {1,  6,  6, 21, 0},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0}},    // 4x2

    {{0,  1,  6,  6, 21},
     {1,  6,  6, 21, 21},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0},
     {0,  0,  0,  0, 0}},    // 8x2

    {{0,  1,  0,  0, 0},
     {1,  6,  0,  0, 0},
     {6,  6,  0,  0, 0},
     {6, 21,  0,  0, 0},
     {0,  0,  0,  0, 0}},    // 2x4

    {{0,  1,  6,  6, 0},
     {1,  6,  6, 21, 0},
     {6,  6, 21, 21, 0},
     {6, 21, 21, 21, 0},
     {0,  0,  0,  0, 0}},    // 4x4

    {{ 0, 16,  6,  6, 21},
     {16, 16,  6, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21},
     { 0,  0,  0,  0,  0}},  // 8x4

    {{ 0, 16,  6,  6, 21},
     {16, 16,  6, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21},
     { 0,  0,  0,  0,  0}},  // 16x4

    {{ 0,  1,  0,  0, 0},
     { 1,  6,  0,  0, 0},
     { 6,  6,  0,  0, 0},
     { 6, 21,  0,  0, 0},
     {21, 21,  0,  0, 0}},    // 2x8

    {{ 0, 11, 11, 11, 0},
     {11, 11, 11, 11, 0},
     { 6,  6, 21, 21, 0},
     { 6, 21, 21, 21, 0},
     {21, 21, 21, 21, 0}},   // 4x8

    {{ 0,  1,  6,  6, 21},
     { 1,  6,  6, 21, 21},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 8x8

    {{ 0, 16,  6,  6, 21},
     {16, 16,  6, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}},  // 16x8

    {{ 0, 16,  6,  6, 21},
     {16, 16,  6, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}},  // 32x8

    {{ 0, 11, 11, 11, 0},
     {11, 11, 11, 11, 0},
     { 6,  6, 21, 21, 0},
     { 6, 21, 21, 21, 0},
     {21, 21, 21, 21, 0}},   // 4x16

    {{ 0, 11, 11, 11, 11},
     {11, 11, 11, 11, 11},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 8x16

    {{ 0,  1,  6,  6, 21},
     { 1,  6,  6, 21, 21},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 16x16

    {{ 0, 16,  6,  6, 21},
     {16, 16,  6, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}},  // 32x16

    {{ 0, 11, 11, 11, 11},
     {11, 11, 11, 11, 11},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 8x32

    {{ 0, 11, 11, 11, 11},
     {11, 11, 11, 11, 11},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 16x32

    {{ 0,  1,  6,  6, 21},
     { 1,  6,  6, 21, 21},
     { 6,  6, 21, 21, 21},
     { 6, 21, 21, 21, 21},
     {21, 21, 21, 21, 21}},  // 32x32
};
/* clang-format on */

constexpr uint8_t kCoeffBasePositionContextOffset[3] = {26, 31, 36};

}  // namespace

// Section 8.3.2 in the spec, under coeff_base_eob.
int AOMResidualIO::GetCoeffBaseContextEob(TrfSize tdim, int index) {
  if (index == 0) return 0;
  const int tx_width_log2 = TrfLog2[TrfWidth[tdim]];
  const int tx_height = TrfHeight[tdim];
  if (index <= DivideBy8(tx_height << tx_width_log2)) return 1;
  if (index <= DivideBy4(tx_height << tx_width_log2)) return 2;
  return 3;
}

// Section 8.3.2 in the spec, under coeff_base.
static int GetCoeffBaseContext2D(const uint8_t qlevels[],
                                 TrfSize tdim,
                                 int adjusted_tx_width_log2,
                                 uint16_t pos) {
  if (pos == 0) return 0;
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + levels[1] + levels[2] +
                       levels[padded_tx_width] +
                       levels[padded_tx_width + 1] +
                       levels[2 * padded_tx_width]));
  const int row = pos >> adjusted_tx_width_log2;
  const int column = pos & (tx_width - 1);
  return context +
         kCoeffBaseContextOffset[tdim][std::min(row, 4)]
                                      [std::min(column, 4)];
}

// Section 8.3.2 in the spec, under coeff_base.
static int GetCoeffBaseContextHorizontal(
    const uint8_t qlevels[], TrfSize /*tdim*/,
    int adjusted_tx_width_log2, uint16_t pos) {
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + levels[1] + levels[2] + levels[3] + levels[4] +
                       levels[padded_tx_width]));
  const int index = pos & (tx_width - 1);
  return context + kCoeffBasePositionContextOffset[std::min(index, 2)];
}

// Section 8.3.2 in the spec, under coeff_base.
static int GetCoeffBaseContextVertical(
    const uint8_t qlevels[], TrfSize /*tdim*/,
    int adjusted_tx_width_log2, uint16_t pos) {
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + levels[1] +
                       levels[1 * padded_tx_width] +
                       levels[2 * padded_tx_width] +
                       levels[3 * padded_tx_width] +
                       levels[4 * padded_tx_width]));
  const int index = pos >> adjusted_tx_width_log2;
  return context + kCoeffBasePositionContextOffset[std::min(index, 2)];
}

static constexpr CoeffBaseContextFunc kCoeffBaseContextFuncs[] = {
  GetCoeffBaseContext2D,
  GetCoeffBaseContextHorizontal,
  GetCoeffBaseContextVertical
};

CoeffBaseContextFunc
  AOMResidualIO::GetCoeffsBaseContextFunc(uint32_t tx_class) {
  return kCoeffBaseContextFuncs[tx_class];
}

// Section 8.3.2 in the spec, under coeff_br.
static int GetCoeffBaseRangeContext2D(
    const uint8_t qlevels[], int adjusted_tx_width_log2,
    int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(1 + levels[1] +
                       levels[padded_tx_width] +
                       levels[padded_tx_width + 1]));
  if (pos == 0) return context;
  const int row = pos >> adjusted_tx_width_log2;
  const int column = pos & (tx_width - 1);
  return context + (((row | column) < 2) ? 7 : 14);
}

// Section 8.3.2 in the spec, under coeff_br.
static int GetCoeffBaseRangeContextHorizontal(
    const uint8_t qlevels[], int adjusted_tx_width_log2, int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(1 + levels[1] + levels[2] + levels[padded_tx_width]));
  if (pos == 0) return context;
  const int column = pos & (tx_width - 1);
  return context + ((column == 0) ? 7 : 14);
}

// Section 8.3.2 in the spec, under coeff_br.
static int GetCoeffBaseRangeContextVertical(
    const uint8_t qlevels[], int adjusted_tx_width_log2, int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  const uint8_t* const levels =
      &qlevels[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(1 + levels[1] +
                       levels[padded_tx_width] + levels[2 * padded_tx_width]));
  if (pos == 0) return context;
  const int row = pos >> adjusted_tx_width_log2;
  return context + ((row == 0) ? 7 : 14);
}

static constexpr CoeffBaseRangeContextFunc kCoeffBaseRangeContextFuncs[] = {
  GetCoeffBaseRangeContext2D,
  GetCoeffBaseRangeContextHorizontal,
  GetCoeffBaseRangeContextVertical
};

CoeffBaseRangeContextFunc
  AOMResidualIO::GetCoeffBaseRangeContextFunc(uint32_t tx_class) {
  return kCoeffBaseRangeContextFuncs[tx_class];
}

// Section 8.3.2 in the spec, under coeff_br. Optimized for end of block based
// on the fact that {0, 1}, {1, 0}, {1, 1}, {0, 2} and {2, 0} will all be 0 in
// the end of block case.
int AOMResidualIO::GetCoeffBaseRangeContextEob(int adjusted_tx_width_log2,
                                               int pos) {
  if (pos == 0) return 0;
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int row = pos >> adjusted_tx_width_log2;
  const int column = pos & (tx_width - 1);
  return ((row | column) < 2) ? 7 : 14;
}

}  // namespace libgav1
}  // namespace WP2
