// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------
//
// Test for trellis
//
// Author: Maryla (maryla@google.com)

#include <tuple>

#include "include/helpers.h"
#include "src/common/lossy/residuals.h"
#include "src/enc/trellis.h"

namespace WP2 {
namespace {

using ::testing::ElementsAre;

//------------------------------------------------------------------------------

// Reorders 'coeffs_lexico', which is in lexicographic order, into zigzag order
// in 'coeffs_zig_zag'.
void ToZigzagOrder(TrfSize tx_size, int16_t* coeffs_lexico,
                   int16_t* coeffs_zig_zag) {
  const uint16_t* scan_order = ResidualIterator::GetZigzag(tx_size);
  for (uint32_t i = 0; i < kNumCoeffs[tx_size]; ++i) {
    coeffs_zig_zag[i] = coeffs_lexico[scan_order[i]];
  }
}

// Reorders 'coeffs_zig_zag', which is in zigzag order, into lexicographic
// order in 'coeffs_lexico'.
void ToLexicoOrder(TrfSize tx_size, int16_t* coeffs_zig_zag,
                   int16_t* coeffs_lexico) {
  const uint16_t* scan_order = ResidualIterator::GetZigzag(tx_size);
  for (uint32_t i = 0; i < kNumCoeffs[tx_size]; ++i) {
    coeffs_lexico[scan_order[i]] = coeffs_zig_zag[i];
  }
}

// Runs coefficient dropout on 'coeffs_zig_zag'.
void Dropout(TrfSize tx_size, uint32_t dropout_num_before,
             uint32_t dropout_num_after, int16_t coeffs_zig_zag[]) {
  uint32_t max_coeffs = kNumCoeffs[tx_size];

  int16_t coeffs[kMaxBlockSizePix2];
  // Coeffs are assumed to be in zigzag order, since tests are easier to
  // understand this way, but DropoutCoeffs takes coeffs in lexico order so we
  // convert to lexico then back to zigzag afterwards.
  ToLexicoOrder(tx_size, coeffs_zig_zag, coeffs);

  int16_t kDequantFactor = 10;
  int16_t dequantized_coeffs[kMaxBlockSizePix2];
  for (uint32_t i = 0; i < max_coeffs; ++i) {
    dequantized_coeffs[i] = coeffs[i] * kDequantFactor;
  }

  uint32_t num_coeffs = max_coeffs;
  while (num_coeffs > 0 && coeffs[num_coeffs - 1] == 0) --num_coeffs;

  DropoutCoeffs(tx_size, dropout_num_before, dropout_num_after, coeffs,
                &num_coeffs, dequantized_coeffs);

  ToZigzagOrder(tx_size, coeffs, coeffs_zig_zag);

  // Check updated num_coeffs value is valid.
  uint32_t new_num_coeffs = max_coeffs;
  while (new_num_coeffs > 0 && coeffs[new_num_coeffs - 1] == 0) {
    --new_num_coeffs;
  }
  EXPECT_EQ(new_num_coeffs, num_coeffs);

  // Check dequantized_coeffs is still valid.
  for (uint32_t i = 0; i < max_coeffs; ++i) {
    EXPECT_EQ(coeffs[i] * kDequantFactor, dequantized_coeffs[i]);
  }
}

TEST(TrellisTest, KeepsLargeCoeffs) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Large isolated coeffs should be preserved.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 0, 0, 42, 0,    // should be kept
                              0, 0, 0, 0, 0, 0, 0,  0,    //
                              0, 0, 0, 0, 0, 0, 0,  -30,  // should be kept
                              0, 0, 0, 0, 0, 0, 0,  0};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 42, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, -30,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, RemovesSmallIsolatedCoeffs) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small isolated coeffs should be removed.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 1,  0, 0, 0,  // should be removed
                              0, 0, 0, 0, 0,  0, 0, 0,  //
                              0, 0, 0, 0, -2, 0, 0, 0,  // should be removed
                              0, 0, 0, 0, 0,  0, 0, 0};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, KeepsSmallCoeffsAmongLargeOnes) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small coeffs that are not isolated (not enough zeros before/after should be
  // kept).
  int16_t coeffs_zig_zag[] = {1, 0,  0, 0,  -5, 0, 0, -1,  // should be kept
                              0, 0,  0, 10, 0,  0, 2, 0,   // should be kept
                              0, 0,  0, 0,  0,  0, 0, 0,   //
                              0, -2, 0, 0,  0,  0, 0, 0};  // should be removed
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(1, 0, 0, 0, -5, 0, 0, -1,  //
                                          0, 0, 0, 10, 0, 0, 2, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, KeepsSmallCoeffsCloseToStart) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small coeffs that are too close the the beginning of the block
  // should also be kept (counts as not enough zeroes before).
  // However, coeffs at the end of the block should be removed (we pretend that
  // there are more zeros after it if needed).
  int16_t coeffs_zig_zag[] = {0, 0, -1, 0,  0, 0, 0,  0,   // should be kept
                              0, 0, 0,  10, 0, 0, 0,  0,   // should be kept
                              0, 0, 0,  2,  0, 0, 0,  0,   // should be removed
                              0, 0, 0,  0,  0, 0, -1, 0};  // should be removed
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, -1, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 10, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, RemovesSmallClusterOfCoeffs) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small clusters (<= kDropoutContinuityMax) of small coeffs can be removed.
  int16_t coeffs_zig_zag_two[] = {0, 0, 0, 0, 1, 0, 0, -1,  // should be removed
                                  0, 0, 0, 0, 0, 0, 0, 0,   //
                                  0, 0, 0, 0, 0, 0, 1, 0,   // should be removed
                                  0, 0, 0, 0, 0, 0, 0, 0};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag_two);
  EXPECT_THAT(coeffs_zig_zag_two, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, KeepsLargeClusterOfCoeffs) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Large clusters (> kDropoutContinuityMax) of small coeffs should be kept.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 1, 0,  1, -1,  // should be kept
                              0, 0, 0, 0, 0, 0,  0, 0,   //
                              0, 0, 0, 0, 0, -2, 0, 0,   // should be removed
                              0, 0, 0, 0, 0, 0,  0, 0};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 1, 0, 1, -1,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TrellisTest, NumBeforeLargerThanNumAfter) {
  const TrfSize tx_size = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 2;
  // The second coeff (-2) doesn't seem to meet the dropout_num_before
  // criteria. But since the first coeff (1) will be dropped, it will meet
  // the criteria and should be dropped too.
  int16_t coeffs_zig_zag[] = {0,  0, 0, 0, 1, 0, 0, 0,  // should be removed
                              -2, 0, 0, 0, 0, 0, 0, 0,  // should be removed
                              0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

// More complex test combining other test cases.
TEST(TrellisTest, ComplexTest) {
  const TrfSize tx_size = TRF_8x8;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 2;
  int16_t coeffs_zig_zag[] = {1, 12, 0,  0,   0, 0, 1,  0,   //
                              0, 0,  0,  -12, 0, 0, 0,  1,   //
                              0, 0,  -2, 0,   1, 0, 0,  1,   //
                              0, 0,  0,  0,   5, 0, -1, 0,   //
                              0, 0,  0,  1,   0, 0, 0,  -1,  //
                              0, 0,  0,  0,   2, 0, 0,  0,   //
                              0, 1,  0,  0,   0, 5, 0,  0,   //
                              0, 0,  1,  1,   0, 0, 0,  -2};
  Dropout(tx_size, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(1, 12, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, -12, 0, 0, 0, 1,  //
                                          0, 0, -2, 0, 1, 0, 0, 1,   //
                                          0, 0, 0, 0, 5, 0, -1, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 5, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
