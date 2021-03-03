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
//   Block sizes and partitions
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/lossy/block_size_io.h"

// if defined, record statistics about block size distribution
// #define PRINT_STATS

#if defined(PRINT_STATS)
#include <cstdio>

#include "src/utils/stats.h"
#endif

namespace WP2 {

//------------------------------------------------------------------------------

// Block sizes to try for partitioning, in increasing surface order!
// Wider blocks are placed last to be tried first when reverse-iterating for
// partitioning, because horizontal edges are more common in photos.
// TODO(skal): settle on definitive ones.

// clang-format off
static const BlockSize kSmallSquares[] = {BLK_4x4, BLK_8x8,
                                          BLK_LAST};
static const BlockSize kSmallRects[] = {BLK_4x4,   BLK_4x8,  BLK_8x4,  BLK_8x8,
                                        BLK_4x16,  BLK_16x4, BLK_8x16, BLK_16x8,
                                        BLK_16x16,
                                        BLK_LAST};
static const BlockSize kAllRects[] = {BLK_4x4,   BLK_4x8,   BLK_8x4,  BLK_8x8,
                                      BLK_4x16,  BLK_16x4,  BLK_8x16, BLK_16x8,
                                      BLK_16x16, BLK_8x32,  BLK_32x8,
                                      BLK_16x32, BLK_32x16, BLK_32x32,
                                      BLK_LAST};
static const BlockSize kThickRects[] = {
    // This set contains square blocks and 'fill-in' blocks
    // no bigger than 2:1 ratio in width or height.
    BLK_4x4,   BLK_4x8,   BLK_8x4,   BLK_8x8,   BLK_16x8, BLK_8x16,
    BLK_16x16, BLK_16x32, BLK_32x16, BLK_32x32,
    BLK_LAST};
static const BlockSize kMediumSquares[] = {BLK_4x4, BLK_8x8, BLK_16x16,
                                           BLK_LAST};
static const BlockSize kAllSquares[] = {BLK_4x4, BLK_8x8, BLK_16x16, BLK_32x32,
                                        BLK_LAST};
static const BlockSize kSomeRects[] = {BLK_4x4,   BLK_4x8,  BLK_8x4,  BLK_8x8,
                                       BLK_4x16,  BLK_16x4, BLK_8x16, BLK_16x8,
                                       BLK_16x16, BLK_32x32,
                                       BLK_LAST};
// clang-format on

extern const PartitionSetStats kPartSetsStats[] = {
    {SMALL_SQUARES, kSmallSquares},   {SMALL_RECTS, kSmallRects},
    {ALL_RECTS, kAllRects},           {THICK_RECTS, kThickRects},
    {MEDIUM_SQUARES, kMediumSquares}, {ALL_SQUARES, kAllSquares},
    {SOME_RECTS, kSomeRects}};
STATIC_ASSERT_ARRAY_SIZE(kPartSetsStats, NUM_PARTITION_SETS);

constexpr uint8_t PartitionSetStats::kInvalidSlot;

static BlockSize GetExactBlockSize(PartitionSet set, uint32_t w, uint32_t h) {
  const BlockSize s = GetBlockSize(std::max(1u, w), std::max(1u, h));
  if (IsBlockSizeInSet(set, s) && BlockWidth[s] == w && BlockHeight[s] == h) {
    return s;
  }
  return BLK_LAST;
}

static void GetSplitPatterns(PartitionSet set, BlockSize size,
                             SplitPattern p[kMaxSplitPatterns],
                             uint32_t* num_patterns) {
  *num_patterns = 0;
  // This should always be first so the index matches kNoSplitIndex.
  if (IsBlockSizeInSet(set, size)) p[(*num_patterns)++] = {1, {size}};

  const uint32_t width = BlockWidth[size], height = BlockHeight[size];
  // The BlockSize variable names below are based on a 4x4 block reference.
  const BlockSize b2x2 = GetExactBlockSize(set, width / 2, height / 2);
  const BlockSize b4x1 = GetExactBlockSize(set, width, height / 4);
  const BlockSize b1x4 = GetExactBlockSize(set, width / 4, height);
  const BlockSize b4x2 = GetExactBlockSize(set, width, height / 2);
  const BlockSize b2x4 = GetExactBlockSize(set, width / 2, height);

  if (b2x2 != BLK_LAST) p[(*num_patterns)++] = {4, {b2x2, b2x2, b2x2, b2x2}};
  if (b4x1 != BLK_LAST) p[(*num_patterns)++] = {4, {b4x1, b4x1, b4x1, b4x1}};
  if (b1x4 != BLK_LAST) p[(*num_patterns)++] = {4, {b1x4, b1x4, b1x4, b1x4}};
  if (b4x2 != BLK_LAST) {
    p[(*num_patterns)++] = {2, {b4x2, b4x2}};
    if (b2x2 != BLK_LAST) p[(*num_patterns)++] = {3, {b4x2, b2x2, b2x2}};
    if (b2x2 != BLK_LAST) p[(*num_patterns)++] = {3, {b2x2, b2x2, b4x2}};
  }
  if (b2x4 != BLK_LAST) {
    p[(*num_patterns)++] = {2, {b2x4, b2x4}};
    if (b2x2 != BLK_LAST) p[(*num_patterns)++] = {3, {b2x4, b2x2, b2x2}};
    if (b2x2 != BLK_LAST) p[(*num_patterns)++] = {3, {b2x2, b2x4, b2x2}};
  }
}

PartitionSetStats::PartitionSetStats(PartitionSet partition_set,
                                     const BlockSize blocks[])
    : partition_set_(partition_set), num_unique_bounds_(0) {
  for (BlockSize bounds : kAllBlockSizes) {
    for (auto& b : blocks_[bounds]) b = BLK_LAST;
    for (auto& s : slots_[bounds]) s = kInvalidSlot;
    num_blocks_[bounds] = 0;
    uint32_t max_width = 0, max_height = 0;
    for (uint32_t i = 0; blocks[i] != BLK_LAST; ++i) {
      if (BlockWidth[blocks[i]] <= BlockWidth[bounds] &&
          BlockHeight[blocks[i]] <= BlockHeight[bounds]) {
        blocks_[bounds][num_blocks_[bounds]] = blocks[i];
        slots_[bounds][blocks[i]] = num_blocks_[bounds];
        ++num_blocks_[bounds];
        max_width = std::max(max_width, BlockWidth[blocks[i]]);
        max_height = std::max(max_height, BlockHeight[blocks[i]]);
      }
      GetSplitPatterns(partition_set, bounds, split_patterns_[bounds],
                       &num_split_patterns_[bounds]);
    }
    // For a given bounding BlockSize, 'smallest_bounds_' contains the smallest
    // BlockSize at least as wide and tall as each BlockSize belonging to the
    // PartitionSet fitting into these bounds.
    assert(max_width > 0 && max_height > 0);
    BlockSize& smallest_bounds = smallest_bounds_[bounds];
    smallest_bounds = GetBlockSize(max_width, max_height);
    assert(BlockWidth[smallest_bounds] <= max_width);
    assert(BlockHeight[smallest_bounds] <= max_height);
    if (smallest_bounds == bounds) {
      unique_bounds_index_[bounds] = num_unique_bounds_;
      unique_bounds_[num_unique_bounds_] = bounds;
      ++num_unique_bounds_;
    }
  }
  for (BlockSize bounds : kAllBlockSizes) {
    const BlockSize smallest_bounds = smallest_bounds_[bounds];
    if (smallest_bounds != bounds) {
      unique_bounds_index_[bounds] = unique_bounds_index_[smallest_bounds];
    }
  }
  std::fill(unique_bounds_ + num_unique_bounds_, unique_bounds_ + BLK_LAST,
            BLK_LAST);
  used_trfs_ = 0u;
  std::fill(block_index_, block_index_ + BLK_LAST, BLK_LAST);
  for (uint32_t i = 0; blocks[i] != BLK_LAST; ++i) {
    block_index_[blocks[i]] = i;

    const uint32_t trf = (uint32_t)blocks[i];
    const uint32_t trf1 = kFullDim[trf];
    const uint32_t trf2 = kHalfDim[trf];
    used_trfs_ |= 1u << trf1;
    used_trfs_ |= 1u << trf2;
  }
}

//------------------------------------------------------------------------------

const BlockSize* GetBlockSizes(PartitionSet set, BlockSize bounds) {
  assert(set < NUM_PARTITION_SETS);  // TODO(skal): allow custom set of blocks?
  return kPartSetsStats[set].blocks_[bounds];
}

uint32_t GetNumBlockSizes(PartitionSet set, BlockSize bounds) {
  assert(set < NUM_PARTITION_SETS);
  return kPartSetsStats[set].num_blocks_[bounds];
}

bool TrfIsUsed(PartitionSet set, TrfSize tdim) {
  assert(set < NUM_PARTITION_SETS);
  return !!(kPartSetsStats[set].used_trfs_ & (1u << (int)tdim));
}

BlockSize GetFittingBlockSize(PartitionSet set, BlockSize bounds) {
  return GetBlockSizes(set, bounds)[GetNumBlockSizes(set, bounds) - 1];
}

bool IsBlockSizeInSet(PartitionSet set, BlockSize size) {
  assert(set < NUM_PARTITION_SETS && size < BLK_LAST);
  return GetFittingBlockSize(set, size) == size;
}

BlockSize GetSmallestBounds(PartitionSet set, BlockSize bounds) {
  assert(set < NUM_PARTITION_SETS);
  return kPartSetsStats[set].smallest_bounds_[bounds];
}

uint32_t GetNumUniqueBounds(PartitionSet set) {
  assert(set < NUM_PARTITION_SETS);
  return kPartSetsStats[set].num_unique_bounds_;
}

BlockSize GetUniqueBounds(PartitionSet set, uint32_t index) {
  assert(set < NUM_PARTITION_SETS);
  assert(index < kPartSetsStats[set].num_unique_bounds_);
  assert(kPartSetsStats[set].unique_bounds_[index] != BLK_LAST);
  return kPartSetsStats[set].unique_bounds_[index];
}

uint32_t GetNumSplitPatterns(PartitionSet set, BlockSize size) {
  assert(set < NUM_PARTITION_SETS);
  return kPartSetsStats[set].num_split_patterns_[size];
}

const SplitPattern* GetSplitPatterns(PartitionSet set, BlockSize size) {
  return kPartSetsStats[set].split_patterns_[size];
}

//------------------------------------------------------------------------------

void WriteBlockSize(const FrontMgrDoubleOrderBase& mgr, BlockSize dim,
                    SymbolManager* const sm, ANSEncBase* const enc) {
  const PartitionSetStats& stats = kPartSetsStats[mgr.GetPartitionSet()];
  const BlockSize bounds =
      stats.smallest_bounds_[mgr.GetMaxPossibleBlock().dim()];
  const uint32_t i = stats.slots_[bounds][dim];
  assert(i != PartitionSetStats::kInvalidSlot);
  const uint32_t cluster = stats.unique_bounds_index_[bounds];
  sm->Process(kSymbolBlockSize, cluster, i, "block_size", enc);
}

BlockSize ReadBlockSize(const FrontMgrDoubleOrderBase& mgr,
                        SymbolReader* const sr) {
  const PartitionSetStats& stats = kPartSetsStats[mgr.GetPartitionSet()];
  const BlockSize bounds =
      stats.smallest_bounds_[mgr.GetMaxPossibleBlock().dim()];
  const uint32_t cluster = stats.unique_bounds_index_[bounds];
  const uint32_t i = sr->Read(kSymbolBlockSize, cluster, "block_size");
  const BlockSize dim = stats.blocks_[bounds][i];
  assert(dim != BLK_LAST);
  return dim;
}

//------------------------------------------------------------------------------

void WriteBlockSplit(const SplitIteratorBase& it, uint32_t split_idx,
                     SymbolManager* const sm, ANSEncBase* const enc) {
  assert(it.CurrentBlock().splittable);
  const PartitionSetStats& stats = kPartSetsStats[it.partition_set()];
  const uint32_t cluster = stats.block_index_[it.CurrentBlock().block.dim()];
  const uint32_t max = it.NumSplitPatterns() - 1;
  assert(split_idx <= max);
  sm->Process(kSymbolBlockSize, cluster, split_idx, max, "block_split", enc);
}

WP2Status ReadBlockSplit(const SplitIteratorBase& it, SymbolReader* const sr,
                         uint32_t* const split_idx) {
  assert(it.CurrentBlock().splittable);
  const PartitionSetStats& stats = kPartSetsStats[it.partition_set()];
  const uint32_t cluster = stats.block_index_[it.CurrentBlock().block.dim()];
  const uint32_t max = it.NumSplitPatterns() - 1;
  int32_t read_value;
  WP2_CHECK_STATUS(
      sr->TryRead(kSymbolBlockSize, cluster, max, "block_split", &read_value));
  *split_idx = read_value;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}   // namespace WP2
