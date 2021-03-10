// Copyright 2020 Google LLC
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

// Partition scoring function test.

#include <string>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/common/global_params.h"
#include "src/enc/partitioning/partition_score_func_block.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

TEST(PartitionScoreFuncTest, BlockScoreFunc) {
  ArgbBuffer rgb;
  ASSERT_WP2_OK(
      ReadImage(testing::GetTestDataPath("background.png").c_str(), &rgb));

  const EncoderConfig& config = EncoderConfig::kDefault;
  CSPTransform transf;
  YUVPlane yuv;
  ASSERT_WP2_OK(yuv.Import(rgb, /*import_alpha=*/rgb.HasTransparency(), transf,
                           /*resize_if_needed=*/true, /*pad=*/kPredWidth));
  GlobalParams gparams;
  FeatureMap features;
  gparams.features_ = &features;
  ASSERT_WP2_OK(GlobalAnalysis(rgb, yuv, transf, config, &gparams));
  ASSERT_WP2_OK(gparams.InitRndMtxSet());

  const Rectangle tile_rect = yuv.AsRect();
  const Block kBlocks[] = {{0, 0, BLK_4x4}, {1, 0, BLK_4x8}, {0, 1, BLK_4x4}};
  constexpr uint32_t kNumBlocks = WP2_ARRAY_SIZE(kBlocks);
  float score;

  BlockScoreFunc score_func(/*use_splits=*/false);
  ASSERT_WP2_OK(
      score_func.Init(config, tile_rect, yuv, gparams, ProgressRange()));

  // Test block by block.
  // Check that we can run ComputeScore multiple times and get the same result.
  float block_scores[kNumBlocks];
  for (uint32_t pass = 0; pass < 2; ++pass) {
    for (uint32_t i = 0; i < kNumBlocks; ++i) {
      const Block& block = kBlocks[i];
      if (block.y() != 0) continue;  // Blocks above must be used.
      ASSERT_WP2_OK(score_func.ComputeScore(block, ProgressRange(), &score));
      EXPECT_GT(score, 0);
      if (pass > 0) {
        EXPECT_EQ(score, block_scores[i]);
      }
      block_scores[i] = score;
    }
  }

  // Compute score again, this time marking each block as Use'd as we go.
  float inverted_scores[kNumBlocks];
  for (uint32_t i = 0; i < kNumBlocks; ++i) {
    const Block& block = kBlocks[i];
    ASSERT_WP2_OK(score_func.ComputeScore(block, ProgressRange(), &score));
    EXPECT_GT(score, 0);
    ASSERT_WP2_OK(score_func.Use(block));
    inverted_scores[i] = 1.f / score - 1.f;
  }

  // Fresh clean score func.
  BlockScoreFunc score_func2(/*use_splits=*/false);
  ASSERT_WP2_OK(
      score_func2.Init(config, tile_rect, yuv, gparams, ProgressRange()));

  // Test all block at once.
  ASSERT_WP2_OK(
      score_func2.ComputeScore(kBlocks, sizeof(kBlocks) / sizeof(kBlocks[0]),
                               /*extra_rate=*/0, /*best_score=*/0, &score));
  EXPECT_GT(score, 0);
  const float inverted_score = 1.f / score - 1.f;
  // Check the result is a weighted average of the scores of individual blocks.
  // Currently fails for images with alpha (e.g. source1_32x32.png).
  // TODO(maryla): debug and fix for images with alpha.
  const float expected_score =
      (inverted_scores[0] + 2.f * inverted_scores[1] + inverted_scores[2]) /
      4.f;
  EXPECT_NEAR(inverted_score, expected_score, 0.001f);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
