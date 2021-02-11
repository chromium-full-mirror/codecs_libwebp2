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

// Partitioning test.

#include <string>

#include "include/helpers.h"
#include "src/enc/partitioning/partition_score_func.h"
#include "src/enc/partitioning/partitioner_split.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

enum class IteratorType { kLexico, kRecurse };

// iterator type, partition set, snapped
class TestSplitIterator : public ::testing::TestWithParam<
                              std::tuple<IteratorType, PartitionSet, bool>> {};

TEST_P(TestSplitIterator, Random) {
  const IteratorType type = std::get<0>(GetParam());
  const PartitionSet partition_set = std::get<1>(GetParam());
  const bool snapped = std::get<2>(GetParam());

  constexpr uint32_t kWidth = 111;
  constexpr uint32_t kHeight = 52;
  std::unique_ptr<SplitIteratorBase> it(
      (type == IteratorType::kLexico)
          ? (SplitIteratorBase*)new SplitIteratorLexico()
          : (SplitIteratorBase*)new SplitIteratorRecurse());
  ASSERT_WP2_OK(it->Init(partition_set, snapped, kWidth, kHeight));

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
                                               ALL_RECTS),
                             /*snapping=*/::testing::Values(true, false)));

//------------------------------------------------------------------------------

class PartitionsTest
    : public ::testing::TestWithParam<std::tuple<
          std::string, float, int, PartitionMethod, PartitionSet, bool>> {};

TEST_P(PartitionsTest, Simple) {
  const std::string& file_name = std::get<0>(GetParam());
  EncoderConfig config = EncoderConfig::kDefault;
  config.quality = std::get<1>(GetParam());
  config.effort = std::get<2>(GetParam());
  config.partition_method = std::get<3>(GetParam());
  config.partition_set = std::get<4>(GetParam());
  config.partition_snapping = std::get<5>(GetParam());

  ASSERT_TRUE(testing::EncodeDecodeCompare(file_name, config));
}

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiation, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("source1_64x48.png"),
        /*quality=*/::testing::Values(80),
        ::testing::Values(EncoderConfig::kDefault.effort),
        ::testing::Values(
            MULTIPASS_PARTITIONING, BLOCK_ENCODE_PARTITIONING,
            AREA_ENCODE_PARTITIONING
            // TILE_ENCODE and EXHAUSTIVE_PARTITIONING are too slow.
            ),
        ::testing::Values(ALL_RECTS),
        ::testing::Values(EncoderConfig::kDefault.partition_snapping)));

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiationMultipass, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("source1_64x48.png"),
        /*quality=*/::testing::Values(50),
        ::testing::Values(EncoderConfig::kDefault.effort),
        ::testing::Values(MULTIPASS_PARTITIONING),
        ::testing::Values(ALL_SQUARES),
        ::testing::Values(EncoderConfig::kDefault.partition_snapping)));

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiationSlow, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("source1_32x32.png"),
        /*quality=*/::testing::Values(50),
        ::testing::Values(EncoderConfig::kDefault.effort),
        ::testing::Values(TILE_ENCODE_PARTITIONING),
        ::testing::Values(ALL_RECTS),
        ::testing::Values(EncoderConfig::kDefault.partition_snapping)));

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiationMegaSlow, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("source1_4x4.png"),
        /*quality=*/::testing::Values(35),
        ::testing::Values(EncoderConfig::kDefault.effort),
        ::testing::Values(EXHAUSTIVE_PARTITIONING),
        ::testing::Values(ALL_RECTS),
        ::testing::Values(EncoderConfig::kDefault.partition_snapping)));

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiationFixed, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("source1_64x48.png"),
        /*quality=*/::testing::Values(30),
        ::testing::Values(EncoderConfig::kDefault.effort),
        ::testing::Range(ALL_4X4_PARTITIONING,
                         NUM_PARTITION_METHODS),
        ::testing::Values(ALL_SQUARES),
        ::testing::Values(EncoderConfig::kDefault.partition_snapping)));

INSTANTIATE_TEST_SUITE_P(
    PartitionsTestInstantiationRDOpt, PartitionsTest,
    ::testing::Combine(::testing::Values("source1_64x48.png"),
                       /*quality=*/::testing::Values(75),
                       ::testing::Values(EncoderConfig::kDefault.effort),
                       ::testing::Values(SPLIT_RECURSE_PARTITIONING,
                                         AREA_ENCODE_PARTITIONING,
                                         SUB_AREA_ENCODE_PARTITIONING),
                       ::testing::Values(ALL_RECTS),
                       /*partition_snapping=*/::testing::Values(true)));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_PartitionsTestInstantiationAll, PartitionsTest,
    ::testing::Combine(
        ::testing::Values("alpha_ramp.png", "source3.jpeg"),
        /*quality=*/::testing::Values(0.f, 25.f, 50.f, 85.f),
        /*effort=*/::testing::Values(0, 3, 6, 9),
        ::testing::Range(MULTIPASS_PARTITIONING, NUM_PARTITION_METHODS),
        ::testing::Values(SMALL_SQUARES, MEDIUM_SQUARES, ALL_SQUARES,
                          SMALL_RECTS, ALL_RECTS, THICK_RECTS),
        /*partition_snapping=*/::testing::Values(true, false)));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
