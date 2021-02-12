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
// Tool for finding the best block layout.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/enc/partitioning/partitioner_split.h"

#include <algorithm>
#include <numeric>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/progress_watcher.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partition_score_func_block.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status SplitRecursePartitioner::Init(const EncoderConfig& config,
                                        const YUVPlane& yuv,
                                        const Rectangle& tile_rect,
                                        PartitionScoreFunc* const score_func) {
  WP2_CHECK_STATUS(Partitioner::Init(config, yuv, tile_rect, score_func));
  return WP2_STATUS_OK;
}

// Returns true if the given split pattern (given as the whole block + sub
// blocks) is compatible with the given forced blocks.
static bool IsCompatibleWithForcedBlocks(
    const Block& block, Block* sub_blocks, uint32_t num_sub_blocks,
    const VectorNoCtor<Block>& forced_blocks) {
  auto forced_block_it =
      std::lower_bound(forced_blocks.begin(), forced_blocks.end(), block);
  while (forced_block_it != forced_blocks.end() &&
         (forced_block_it->y() < block.y() + block.h())) {
    if (block.rect().Contains(forced_block_it->rect())) {
      if (num_sub_blocks == 1 && block.dim() != forced_block_it->dim()) {
        return false;
      }
      for (uint32_t i = 0; i < num_sub_blocks; ++i) {
        if (sub_blocks[i].rect().Intersects(forced_block_it->rect()) &&
            !sub_blocks[i].rect().Contains(forced_block_it->rect())) {
          return false;
        }
      }
    }
    ++forced_block_it;
  }
  return true;
}

WP2Status SplitRecursePartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  const uint32_t max_num_blocks = num_block_cols_ * num_block_rows_;
  uint32_t max_num_blocks_left = max_num_blocks;

  // Begin with the user-defined blocks, if any (debug).
  WP2_CHECK_STATUS(RegisterForcedBlocks(*blocks, &max_num_blocks_left));

  // Sort forced blocks by position for binary search below.
  VectorNoCtor<Block> forced_blocks;
  WP2_CHECK_ALLOC_OK(forced_blocks.resize(blocks->size()));
  std::copy(blocks->begin(), blocks->end(), forced_blocks.begin());
  std::sort(forced_blocks.begin(), forced_blocks.end());
  blocks->clear();

  SplitIteratorLexico it;
  WP2_CHECK_STATUS(it.Init(config_->partition_set, config_->partition_snapping,
                           src_->Y.w_, src_->Y.h_));
  BlockScoreFunc* const score_func = (BlockScoreFunc*)score_func_;

  while (!it.Done()) {
    assert(max_num_blocks_left > 0);
    const SplitIteratorRecurse::SplitInfo split_info = it.CurrentBlock();

    if (split_info.splittable) {
      // Get all split patterns and keep the one with the best score.
      int32_t best_pattern_idx = -1;
      float best_score = 0.f;

      const uint32_t num_patterns = it.NumSplitPatterns();
      for (uint32_t split_idx = 0; split_idx < num_patterns; ++split_idx) {
        Block sub_blocks[4];
        const uint32_t num_sub_blocks =
            it.GetSplitPatternBlocks(split_idx, sub_blocks);

        // Handle forced blocks. Skip configurations which don't match the
        // forced blocks.
        const Block& block = it.CurrentBlock().block;
        if (!forced_blocks.empty() &&
            !IsCompatibleWithForcedBlocks(block, sub_blocks, num_sub_blocks,
                                          forced_blocks)) {
          continue;
        }

        float score;
        WP2_CHECK_STATUS(
            score_func->ComputeScore(sub_blocks, num_sub_blocks, &score));
        if (best_pattern_idx == -1 || score > best_score) {
          best_pattern_idx = split_idx;
          best_score = score;
        }
      }
      assert(best_pattern_idx != -1);

      // Remember the decision.
      it.SplitCurrentBlock(best_pattern_idx);
      WP2_CHECK_STATUS(progress.AdvanceBy(0.));  // Nothing decided.
    } else {
      const Block& block = split_info.block;
      // This 'block' was decided as is with no possibility of recursion.
      assert(IsBlockValid(block, block, forced_blocks.size()));
      WP2_CHECK_REDUCED_STATUS(
          RegisterOrderForVDebug(0, 0, block, blocks->size(), max_num_blocks));
      WP2_CHECK_ALLOC_OK(blocks->push_back(block));
      WP2_CHECK_STATUS(score_func_->Use(block));

      // The following instructions are done only for assertions/debug
      // because the front manager should handle everything, except for
      // the forced blocks but these are checked above.
      assert((Occupy(block), true));
      assert(max_num_blocks_left >= block.w() * block.h());
      max_num_blocks_left -= block.w() * block.h();
      assert(block.dim() != BLK_LAST);

      WP2_CHECK_STATUS(progress.AdvanceBy(1. * block.area() / max_num_blocks));
      it.NextBlock();
    }
  }
  assert(max_num_blocks_left == 0);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

static BlockSize GetExactBlockSize(PartitionSet set, uint32_t w, uint32_t h) {
  const BlockSize s = GetBlockSize(std::max(1u, w), std::max(1u, h));
  if (IsBlockSizeInSet(set, s) && BlockWidth[s] == w && BlockHeight[s] == h) {
    return s;
  }
  return BLK_LAST;
}

void SplitIteratorBase::GetSplitPatterns(PartitionSet set, BlockSize size,
                                         Vector<SplitPattern>* const p) {
  assert(p->empty() && p->capacity() >= 10);
  if (IsBlockSizeInSet(set, size)) p->push_back_no_resize({1, {size}});

  const uint32_t width = BlockWidth[size], height = BlockHeight[size];
  // The BlockSize variable names below are based on a 4x4 block reference.
  const BlockSize b2x2 = GetExactBlockSize(set, width / 2, height / 2);
  const BlockSize b4x1 = GetExactBlockSize(set, width, height / 4);
  const BlockSize b1x4 = GetExactBlockSize(set, width / 4, height);
  const BlockSize b4x2 = GetExactBlockSize(set, width, height / 2);
  const BlockSize b2x4 = GetExactBlockSize(set, width / 2, height);

  if (b2x2 != BLK_LAST) p->push_back_no_resize({4, {b2x2, b2x2, b2x2, b2x2}});
  if (b4x1 != BLK_LAST) p->push_back_no_resize({4, {b4x1, b4x1, b4x1, b4x1}});
  if (b1x4 != BLK_LAST) p->push_back_no_resize({4, {b1x4, b1x4, b1x4, b1x4}});
  if (b4x2 != BLK_LAST) {
    p->push_back_no_resize({2, {b4x2, b4x2}});
    if (b2x2 != BLK_LAST) p->push_back_no_resize({3, {b4x2, b2x2, b2x2}});
    if (b2x2 != BLK_LAST) p->push_back_no_resize({3, {b2x2, b2x2, b4x2}});
  }
  if (b2x4 != BLK_LAST) {
    p->push_back_no_resize({2, {b2x4, b2x4}});
    if (b2x2 != BLK_LAST) p->push_back_no_resize({3, {b2x4, b2x2, b2x2}});
    if (b2x2 != BLK_LAST) p->push_back_no_resize({3, {b2x2, b2x4, b2x2}});
  }
  assert(!p->empty());
}

void SplitIteratorBase::GetSplitPatternBlocks(const Block& block,
                                              const SplitPattern& split_pattern,
                                              Block* sub_blocks) {
  assert(split_pattern.num_blocks >= 1 && split_pattern.num_blocks <= 4);
  // Treat this special case differently. Others can be done row by row.
  const bool is_2x4_2x2_2x2 =
      (split_pattern.num_blocks == 3 &&
       BlockHeight[split_pattern.block_sizes[0]] == block.h());
  uint32_t num_blocks = 0, x = block.x(), y = block.y();
  while (num_blocks < split_pattern.num_blocks) {
    if (x == block.x() + block.w()) {  // End of a row.
      // Start a new row aligned under the first block, unless 'is_2x4_2x2_2x2'.
      x = sub_blocks[is_2x4_2x2_2x2 ? 1 : 0].x();
      y += sub_blocks[is_2x4_2x2_2x2 ? 1 : 0].h();
    }
    assert(x < block.x() + block.w() && y < block.y() + block.h());
    sub_blocks[num_blocks] = Block(x, y, split_pattern.block_sizes[num_blocks]);
    x += sub_blocks[num_blocks].w();
    ++num_blocks;
  }
}

WP2Status SplitIteratorBase::Init(PartitionSet partition_set, bool snapped,
                                  uint32_t width, uint32_t height) {
  partition_set_ = partition_set;
  snapped_ = snapped;
  WP2_CHECK_STATUS(front_mgr_.InitBase(width, height));

  const BlockSize* const sizes = GetBlockSizes(partition_set_);
  for (const BlockSize* size = sizes; *size != BLK_LAST; ++size) {
    WP2_CHECK_ALLOC_OK(split_patterns_[(uint32_t)*size].reserve(10));
    GetSplitPatterns(partition_set_, *size, &split_patterns_[(uint32_t)*size]);
  }

  return WP2_STATUS_OK;
}

uint32_t SplitIteratorBase::NumSplitPatterns() const {
  const SplitInfo& current_block = CurrentBlock();
  if (!current_block.splittable) return 1;
  return split_patterns_[current_block.block.dim()].size();
}

uint32_t SplitIteratorBase::GetSplitPatternBlocks(uint32_t split_idx,
                                                  Block sub_blocks[4]) const {
  assert(CurrentBlock().splittable);
  const Block& current_block = CurrentBlock().block;
  assert(split_idx < SplitPatterns(current_block.dim()).size());
  const SplitPattern& split_pattern =
      SplitPatterns(current_block.dim())[split_idx];
  GetSplitPatternBlocks(current_block, split_pattern, sub_blocks);
  return split_pattern.num_blocks;
}

bool SplitIteratorBase::Done() const { return front_mgr_.Done(); }

//------------------------------------------------------------------------------

constexpr SplitIteratorLexico::LightSplitInfo SplitIteratorLexico::kNoSplitInfo;

WP2Status SplitIteratorLexico::Init(PartitionSet partition_set, bool snapped,
                                    uint32_t width, uint32_t height) {
  WP2_CHECK_STATUS(
      SplitIteratorBase::Init(partition_set, snapped, width, height));
  WP2_CHECK_STATUS(split_info_.Resize(SizeBlocks(width), SizeBlocks(height)));
  split_info_.Fill(kNoSplitInfo);
  SetCurrentBlock(/*bx=*/0, /*by=*/0);
  return WP2_STATUS_OK;
}

void SplitIteratorLexico::SplitCurrentBlock(uint32_t idx) {
  assert(CurrentBlock().splittable);
  const Block current_block = CurrentBlock().block;
  assert(idx < SplitPatterns(current_block.dim()).size());
  const SplitPattern& split_pattern = SplitPatterns(current_block.dim())[idx];

  Block blocks[4];
  GetSplitPatternBlocks(current_block, split_pattern, blocks);

  for (uint32_t i = 0; i < split_pattern.num_blocks; ++i) {
    // TODO(yguyon): Recurse in fewer situations to save CPU.
    const bool recursive =
        (split_pattern.num_blocks > 1 && blocks[i].dim() != BLK_4x4);
    SetSplitInfo(blocks[i], recursive);
    if (i == 0) current_block_ = {blocks[i], recursive};
  }
}

void SplitIteratorLexico::NextBlock() {
  assert(!CurrentBlock().splittable);
  front_mgr_.Use(CurrentBlock().block);
  if (!Done()) {
    uint32_t bx, by;
    front_mgr_.FindNextLexico(/*added_last=*/CurrentBlock().block, &bx, &by);
    SetCurrentBlock(bx, by);
  }
}

const SplitIteratorBase::SplitInfo& SplitIteratorLexico::CurrentBlock() const {
  return current_block_;
}

void SplitIteratorLexico::SetCurrentBlock(uint32_t bx, uint32_t by) {
  // Retrieve the previous decision to split this block, if any.
  const LightSplitInfo& split_info = GetSplitInfo(bx, by);
  if (split_info == kNoSplitInfo) {
    const Block bounds = front_mgr_.LargestPossibleBlockAt(bx, by, snapped_);
    const BlockSize size = GetFittingBlockSize(partition_set_, bounds.dim());
    const bool splittable = (SplitPatterns(size).size() > 1);
    current_block_ = {{bx, by, size}, splittable};
  } else {
    current_block_ = {{bx, by, split_info.size}, split_info.splittable};
  }
}

const SplitIteratorLexico::LightSplitInfo& SplitIteratorLexico::GetSplitInfo(
    uint32_t x, uint32_t y) const {
  return split_info_.At(x, y);
}

void SplitIteratorLexico::SetSplitInfo(const Block& block, bool recursive) {
  LightSplitInfo* row = &split_info_.At(block.x(), block.y());
  for (uint32_t y = 0; y < block.h(); ++y, row += split_info_.Step()) {
    for (uint32_t x = 0; x < block.w(); ++x) row[x] = {block.dim(), recursive};
  }
}

//------------------------------------------------------------------------------

WP2Status SplitIteratorRecurse::Init(PartitionSet partition_set, bool snapped,
                                     uint32_t width, uint32_t height) {
  WP2_CHECK_STATUS(
      SplitIteratorBase::Init(partition_set, snapped, width, height));
  WP2_CHECK_ALLOC_OK(block_queue_.reserve(kMaxBlockSize2));
  SetCurrentBlockLexico();
  return WP2_STATUS_OK;
}

void SplitIteratorRecurse::SplitCurrentBlock(uint32_t idx) {
  assert(block_queue_.back().splittable);
  const Block current_block = block_queue_.back().block;
  assert(idx < SplitPatterns(current_block.dim()).size());
  block_queue_.pop_back();

  const SplitPattern& split_pattern = SplitPatterns(current_block.dim())[idx];

  Block blocks[4];
  GetSplitPatternBlocks(current_block, split_pattern, blocks);

  for (int32_t i = split_pattern.num_blocks - 1; i >= 0; --i) {
    const Block& block = blocks[i];
    const bool splittable =
        (split_pattern.num_blocks > 1 && SplitPatterns(block.dim()).size() > 1);
    block_queue_.push_back_no_resize({block, splittable});
  }
}

void SplitIteratorRecurse::NextBlock() {
  assert(!block_queue_.back().splittable);
  front_mgr_.Use(block_queue_.back().block);
  block_queue_.pop_back();

  if (block_queue_.empty()) {
    SetCurrentBlockLexico();
  }
}

const SplitIteratorRecurse::SplitInfo& SplitIteratorRecurse::CurrentBlock()
    const {
  return block_queue_.back();
}

void SplitIteratorRecurse::SetCurrentBlockLexico() {
  uint32_t bx, by;
  front_mgr_.FindNextLexico(&bx, &by);
  const Block bounds = front_mgr_.LargestPossibleBlockAt(bx, by, snapped_);
  const BlockSize size = GetFittingBlockSize(partition_set_, bounds.dim());
  const bool splittable = (SplitPatterns(size).size() > 1);
  block_queue_.push_back_no_resize({{bx, by, size}, splittable});
}

//------------------------------------------------------------------------------

}  // namespace WP2
