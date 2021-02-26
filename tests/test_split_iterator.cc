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

// SplitIterator test.

#include "include/helpers.h"
#include "src/common/lossy/block_size_io.h"
#include "src/utils/split_iterator.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

enum class IteratorType { kLexico, kRecurse };

// iterator type, partition set
class TestSplitIterator
    : public ::testing::TestWithParam<std::tuple<IteratorType, PartitionSet>> {
};

TEST_P(TestSplitIterator, Random) {
  const IteratorType type = std::get<0>(GetParam());
  const PartitionSet partition_set = std::get<1>(GetParam());

  constexpr uint32_t kWidth = 111;
  constexpr uint32_t kHeight = 52;
  std::unique_ptr<SplitIteratorBase> it(
      (type == IteratorType::kLexico)
          ? (SplitIteratorBase*)new SplitIteratorLexico()
          : (SplitIteratorBase*)new SplitIteratorRecurse());
  ASSERT_WP2_OK(it->Init(partition_set, kWidth, kHeight));

  const BlockSize* const sizes = GetBlockSizes(partition_set);
  const BlockSize smallest_block = sizes[0];
  BlockSize largest_block = sizes[0];
  for (const BlockSize* size = sizes; *size != BLK_LAST; ++size) {
    largest_block = *size;
  }

  const uint32_t min_blocks = DivCeil(kWidth, BlockWidthPix(largest_block)) *
                              DivCeil(kHeight, BlockHeightPix(largest_block));
  const uint32_t max_blocks = DivCeil(kWidth, BlockWidthPix(smallest_block)) *
                              DivCeil(kHeight, BlockHeightPix(smallest_block));

  UniformIntDistribution gen(/*seed=*/0);

  // Used blocks, to check we cover the whole image.
  FrontMgrBase front_mgr;
  ASSERT_WP2_OK(front_mgr.InitBase(kWidth, kHeight));

  uint32_t num_blocks = 0;
  while (!it->Done()) {
    const uint32_t num_split_patterns = it->NumSplitPatterns();
    ASSERT_GT(num_split_patterns, 0u);
    if (it->CurrentBlock().splittable) {
      ASSERT_GE(it->NumSplitPatterns(), 1u);
      Block sub_blocks[4];
      const uint32_t num_sub_blocks = it->GetSplitPatternBlocks(0, sub_blocks);
      ASSERT_GE(num_sub_blocks, 0u);
      ASSERT_LT(num_sub_blocks, 4u);
      const uint32_t split_idx =
          gen.Get<uint32_t>(0, it->NumSplitPatterns() - 1);
      it->SplitCurrentBlock(split_idx);
    } else {
      front_mgr.Use(it->CurrentBlock().block);
      it->NextBlock();
      ++num_blocks;
    }
  }

  EXPECT_TRUE(front_mgr.Done());
  EXPECT_GE(num_blocks, min_blocks);
  EXPECT_LE(num_blocks, max_blocks);
}

INSTANTIATE_TEST_SUITE_P(SplitRecurseIteratorInstanciation, TestSplitIterator,
                         ::testing::Combine(
                             /*type=*/::testing::Values(IteratorType::kLexico,
                                                        IteratorType::kRecurse),
                             /*partition_set=*/
                             ::testing::Values(SMALL_SQUARES, MEDIUM_SQUARES,
                                               SMALL_RECTS, SOME_RECTS,
                                               ALL_RECTS)));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
