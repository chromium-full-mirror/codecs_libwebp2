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
#include "src/enc/block_enc.h"
#include "src/enc/trellis.h"

namespace WP2 {
namespace {

using ::testing::ElementsAre;

//------------------------------------------------------------------------------

// Reorders 'coeffs_lexico', which is in lexicographic order, into zigzag order
// in 'coeffs_zig_zag'.
void ToZigzagOrder(TrfSize tdim, int16_t* coeffs_lexico,
                   int16_t* coeffs_zig_zag) {
  const uint16_t* scan_order = ResidualIterator::GetZigzag(tdim);
  for (uint32_t i = 0; i < kNumCoeffs[tdim]; ++i) {
    coeffs_zig_zag[i] = coeffs_lexico[scan_order[i]];
  }
}

// Reorders 'coeffs_zig_zag', which is in zigzag order, into lexicographic
// order in 'coeffs_lexico'.
void ToLexicoOrder(TrfSize tdim, int16_t* coeffs_zig_zag,
                   int16_t* coeffs_lexico) {
  const uint16_t* scan_order = ResidualIterator::GetZigzag(tdim);
  for (uint32_t i = 0; i < kNumCoeffs[tdim]; ++i) {
    coeffs_lexico[scan_order[i]] = coeffs_zig_zag[i];
  }
}

// Runs coefficient dropout on 'coeffs_zig_zag'.
void Dropout(TrfSize tdim, uint32_t dropout_num_before,
             uint32_t dropout_num_after, int16_t coeffs_zig_zag[]) {
  uint32_t max_coeffs = kNumCoeffs[tdim];

  int16_t coeffs[kMaxBlockSizePix2];
  // Coeffs are assumed to be in zigzag order, since tests are easier to
  // understand this way, but DropoutCoeffs takes coeffs in lexico order so we
  // convert to lexico then back to zigzag afterwards.
  ToLexicoOrder(tdim, coeffs_zig_zag, coeffs);

  int16_t kDequantFactor = 10;
  int16_t dequantized_coeffs[kMaxBlockSizePix2];
  for (uint32_t i = 0; i < max_coeffs; ++i) {
    dequantized_coeffs[i] = coeffs[i] * kDequantFactor;
  }

  uint32_t num_coeffs = max_coeffs;
  while (num_coeffs > 0 && coeffs[num_coeffs - 1] == 0) --num_coeffs;

  DropoutCoeffs(tdim, dropout_num_before, dropout_num_after, coeffs,
                &num_coeffs, dequantized_coeffs);

  ToZigzagOrder(tdim, coeffs, coeffs_zig_zag);

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

TEST(DropoutTest, KeepsLargeCoeffs) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Large isolated coeffs should be preserved.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 0, 0, 42, 0,    // should be kept
                              0, 0, 0, 0, 0, 0, 0,  0,    //
                              0, 0, 0, 0, 0, 0, 0,  -30,  // should be kept
                              0, 0, 0, 0, 0, 0, 0,  0};
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 42, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, -30,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, RemovesSmallIsolatedCoeffs) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small isolated coeffs should be removed.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 1,  0, 0, 0,  // should be removed
                              0, 0, 0, 0, 0,  0, 0, 0,  //
                              0, 0, 0, 0, -2, 0, 0, 0,  // should be removed
                              0, 0, 0, 0, 0,  0, 0, 0};
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, KeepsSmallCoeffsAmongLargeOnes) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small coeffs that are not isolated (not enough zeros before/after should be
  // kept).
  int16_t coeffs_zig_zag[] = {1, 0,  0, 0,  -5, 0, 0, -1,  // should be kept
                              0, 0,  0, 10, 0,  0, 2, 0,   // should be kept
                              0, 0,  0, 0,  0,  0, 0, 0,   //
                              0, -2, 0, 0,  0,  0, 0, 0};  // should be removed
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(1, 0, 0, 0, -5, 0, 0, -1,  //
                                          0, 0, 0, 10, 0, 0, 2, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,    //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, KeepsSmallCoeffsCloseToStart) {
  const TrfSize tdim = TRF_8x4;
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
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, -1, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 10, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, RemovesSmallClusterOfCoeffs) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Small clusters (<= kDropoutContinuityMax) of small coeffs can be removed.
  int16_t coeffs_zig_zag_two[] = {0, 0, 0, 0, 1, 0, 0, -1,  // should be removed
                                  0, 0, 0, 0, 0, 0, 0, 0,   //
                                  0, 0, 0, 0, 0, 0, 1, 0,   // should be removed
                                  0, 0, 0, 0, 0, 0, 0, 0};
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag_two);
  EXPECT_THAT(coeffs_zig_zag_two, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0,  //
                                              0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, KeepsLargeClusterOfCoeffs) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 6;
  // Large clusters (> kDropoutContinuityMax) of small coeffs should be kept.
  int16_t coeffs_zig_zag[] = {0, 0, 0, 0, 1, 0,  1, -1,  // should be kept
                              0, 0, 0, 0, 0, 0,  0, 0,   //
                              0, 0, 0, 0, 0, -2, 0, 0,   // should be removed
                              0, 0, 0, 0, 0, 0,  0, 0};
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 1, 0, 1, -1,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0,   //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DropoutTest, NumBeforeLargerThanNumAfter) {
  const TrfSize tdim = TRF_8x4;
  const uint32_t dropout_num_before = 4;
  const uint32_t dropout_num_after = 2;
  // The second coeff (-2) doesn't seem to meet the dropout_num_before
  // criteria. But since the first coeff (1) will be dropped, it will meet
  // the criteria and should be dropped too.
  int16_t coeffs_zig_zag[] = {0,  0, 0, 0, 1, 0, 0, 0,  // should be removed
                              -2, 0, 0, 0, 0, 0, 0, 0,  // should be removed
                              0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
  EXPECT_THAT(coeffs_zig_zag, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0,  //
                                          0, 0, 0, 0, 0, 0, 0, 0));
}

// More complex test combining other test cases.
TEST(DropoutTest, ComplexTest) {
  const TrfSize tdim = TRF_8x8;
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
  Dropout(tdim, dropout_num_before, dropout_num_after, coeffs_zig_zag);
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

class Av1CoeffOptimizationTest : public testing::TestWithParam<int> {};

static const uint32_t kMaxResidual = 512;

void FillRandomResiduals(TrfSize tdim, uint32_t seed,
                         int32_t residuals[kMaxBlockSizePix2]) {
  std::fill(residuals, residuals + kMaxBlockSizePix2, 0);
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);

  UniformIntDistribution rand(seed);
  const uint32_t max_coeffs = kNumCoeffs[tdim];
  // Random coeffs.
  for (uint32_t i = 0; i < max_coeffs; ++i) {
    // Random eob.
    if (i > (max_coeffs * 0.75) && rand.FlipACoin()) break;
    // Make coeffs smaller and smaller.
    const int max =
        std::max<int>(kMaxResidual * (1.f - ((float)i / max_coeffs)), 3);
    residuals[scan[i]] = rand.Get(-max, max);
  }
}

TEST_P(Av1CoeffOptimizationTest, RandomTest) {
  WP2QuantizeInit();
  ANSInit();
  WP2MathInit();

  const int seed = GetParam();

  const Channel channel = kYChannel;
  const TransformPair tx_type = kDctDct;
  const TrfSize tdim = TRF_16x16;
  QuantMtx quant;
  quant.lambda_trellis = 1000.f;
  const uint16_t quant_steps[] = {20, 50, 50, 50, 50};
  ASSERT_WP2_OK(quant.AllocateForEncoder());
  quant.Init(kMaxResidual, /*q_scale=*/1.f, quant_steps);
  const bool first_is_dc = true;
  int32_t residuals[kMaxBlockSizePix2];

  FillRandomResiduals(tdim, seed, residuals);

  int16_t coeffs[kMaxBlockSizePix2];
  int16_t dequantized_coeffs[kMaxBlockSizePix2];
  uint32_t num_coeffs;
  ASSERT_WP2_OK(quant.AllocateForEncoder());
  quant.Quantize(residuals, tdim, first_is_dc, coeffs, &num_coeffs,
                 dequantized_coeffs);
  int16_t coeffs_copy[kMaxBlockSizePix2];
  std::copy(coeffs, coeffs + kMaxBlockSizePix2, coeffs_copy);

  constexpr bool kUseAomCoeffs = true;
  SymbolsInfo symbols_info;
  ASSERT_WP2_OK(symbols_info.InitLossy(/*num_segments=*/2,
                                       PartitionSet::SOME_RECTS,
                                       /*has_alpha=*/false, kUseAomCoeffs,
                                       /*use_splits=*/false));
  SymbolRecorder recorder;
  ASSERT_WP2_OK(recorder.Allocate(symbols_info, /*num_records=*/0));
  Counters counters;
  ASSERT_WP2_OK(counters.Init(/*effort=*/5, kUseAomCoeffs, recorder));

  float rate_cost;
  Av1CoeffOptimization(channel, tx_type, tdim, quant, first_is_dc, residuals,
                       coeffs, &num_coeffs, dequantized_coeffs,
                       counters.residuals(), &rate_cost);

  // Check updated num_coeffs value is valid.
  uint32_t new_num_coeffs = kNumCoeffs[tdim];
  while (new_num_coeffs > 0 && coeffs[new_num_coeffs - 1] == 0) {
    --new_num_coeffs;
  }
  EXPECT_EQ(new_num_coeffs, num_coeffs);

  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);

  // Check coeffs have only been lowered by one, or set to 0 if before the new
  // eob.
  uint32_t seen_eob = false;
  uint32_t num_changed = 0;
  for (int32_t i = kNumCoeffs[tdim] - 1; i >= 0; --i) {
    const uint32_t pos = scan[i];
    int16_t before = coeffs_copy[pos];
    int16_t after = coeffs[pos];
    if (after != 0) seen_eob = true;
    // If we haven't seen any coeffs yet there is nothing to check, any coeff
    // can be set to 0 before the new eob.
    if (after == before) continue;  // Coeff unchanged, nothing to check.
    ++num_changed;
    if (!seen_eob) continue;

    // Check coeffs have only been lowered by one.
    EXPECT_EQ(std::abs(before - after), 1);
    if (before > 0) {
      EXPECT_LT(after, before);
    } else {
      EXPECT_LT(before, after);
    }

    // Check 'dequantized_coeffs' has been correctly updated.
    const int16_t dequant = quant.Dequant(pos, tdim);
    EXPECT_EQ(dequantized_coeffs[pos], after * dequant);
  }
  // Check that the optimization actually did something. Since this is a random
  // test, it's not 100% guaranteed, but is very very likely.
  EXPECT_GT(num_changed, 0u);
}

INSTANTIATE_TEST_SUITE_P(Av1CoeffOptimizationTestInstantiation,
                         Av1CoeffOptimizationTest, testing::Range(0, 10));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
