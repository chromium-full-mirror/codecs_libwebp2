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
// Iterator for splitting blocks.
//
// Author: Maryla (maryla@google.com)

#include "src/utils/split_iterator.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

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
  // Change the order of the blocks for -| shape so that left context is
  // available.
  const bool is_2x2_2x4_2x2 =
      (split_pattern.num_blocks == 3 &&
       BlockHeight[split_pattern.block_sizes[1]] == block.h());
  if (is_2x2_2x4_2x2) std::swap(sub_blocks[1], sub_blocks[2]);
}

WP2Status SplitIteratorBase::Init(PartitionSet partition_set, uint32_t width,
                                  uint32_t height) {
  partition_set_ = partition_set;
  WP2_CHECK_STATUS(front_mgr_.InitBase(width, height));
  return WP2_STATUS_OK;
}

uint32_t SplitIteratorBase::NumSplitPatterns() const {
  const SplitInfo& current_block = CurrentBlock();
  if (!current_block.splittable) return 1;
  return GetNumSplitPatterns(partition_set_, current_block.block.dim());
}

uint32_t SplitIteratorBase::GetSplitPatternBlocks(uint32_t split_idx,
                                                  Block sub_blocks[4],
                                                  bool splittable[4]) const {
  assert(CurrentBlock().splittable);
  const Block& current_block = CurrentBlock().block;
  assert(split_idx < GetNumSplitPatterns(partition_set_, current_block.dim()));
  const SplitPattern& split_pattern =
      GetSplitPatterns(partition_set_, current_block.dim())[split_idx];
  GetSplitPatternBlocks(current_block, split_pattern, sub_blocks);
  GetSplittable(sub_blocks, split_pattern.num_blocks, splittable);
  return split_pattern.num_blocks;
}

bool SplitIteratorBase::Done() const { return front_mgr_.Done(); }

WP2Status SplitIteratorBase::CopyInternal(const SplitIteratorBase& other) {
  WP2_CHECK_STATUS(front_mgr_.CopyInternal(other.front_mgr_));
  partition_set_ = other.partition_set_;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

constexpr SplitIteratorLexico::LightSplitInfo SplitIteratorLexico::kNoSplitInfo;

WP2Status SplitIteratorLexico::Init(PartitionSet partition_set, uint32_t width,
                                    uint32_t height) {
  WP2_CHECK_STATUS(SplitIteratorBase::Init(partition_set, width, height));
  WP2_CHECK_STATUS(split_info_.Resize(SizeBlocks(width), SizeBlocks(height)));
  split_info_.Fill(kNoSplitInfo);
  SetCurrentBlock(/*bx=*/0, /*by=*/0);
  return WP2_STATUS_OK;
}

void SplitIteratorLexico::SplitCurrentBlock(uint32_t split_idx) {
  assert(CurrentBlock().splittable);
  const Block current_block = CurrentBlock().block;
  assert(split_idx < GetNumSplitPatterns(partition_set_, current_block.dim()));
  const SplitPattern& split_pattern =
      GetSplitPatterns(partition_set_, current_block.dim())[split_idx];

  Block blocks[4];
  GetSplitPatternBlocks(current_block, split_pattern, blocks);

  for (uint32_t i = 0; i < split_pattern.num_blocks; ++i) {
    // TODO(yguyon): Recurse in fewer situations to save CPU.
    const bool splittable =
        (split_pattern.num_blocks > 1 && IsSplittable(blocks[i].dim()));
    SetSplitInfo(blocks[i], splittable);
    if (i == 0) current_block_ = {blocks[i], splittable};
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
    const Block bounds =
        front_mgr_.LargestPossibleBlockAt(bx, by, /*snapped=*/true);
    const BlockSize size = GetFittingBlockSize(partition_set_, bounds.dim());
    current_block_ = {{bx, by, size}, IsSplittable(size)};
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

WP2Status SplitIteratorLexico::CopyFrom(const SplitIteratorLexico& other) {
  WP2_CHECK_STATUS(SplitIteratorBase::CopyInternal(other));
  current_block_ = other.current_block_;
  WP2_CHECK_STATUS(
      split_info_.Copy(other.split_info_, /*resize_if_needed=*/true));
  return WP2_STATUS_OK;
}

bool SplitIteratorLexico::IsSplittable(BlockSize dim) const {
  return (GetNumSplitPatterns(partition_set_, dim) > 1);
}

void SplitIteratorLexico::GetSplittable(const Block sub_blocks[4],
                                        uint32_t num_sub_blocks,
                                        bool splittable[4]) const {
  if (num_sub_blocks == 1) {
    splittable[0] = false;
  } else {
    for (uint32_t i = 0; i < num_sub_blocks; ++i) {
      splittable[i] = IsSplittable(sub_blocks[i].dim());
    }
  }
}

//------------------------------------------------------------------------------

WP2Status SplitIteratorRecurse::Init(PartitionSet partition_set, uint32_t width,
                                     uint32_t height) {
  WP2_CHECK_STATUS(SplitIteratorBase::Init(partition_set, width, height));
  WP2_CHECK_ALLOC_OK(block_queue_.reserve(kMaxBlockSize2));
  SetCurrentBlockLexico();
  return WP2_STATUS_OK;
}

void SplitIteratorRecurse::SplitCurrentBlock(uint32_t idx) {
  assert(block_queue_.back().splittable);
  const Block current_block = block_queue_.back().block;
  assert(idx < GetNumSplitPatterns(partition_set_, current_block.dim()));
  block_queue_.pop_back();

  const SplitPattern& split_pattern =
      GetSplitPatterns(partition_set_, current_block.dim())[idx];

  Block blocks[4];
  GetSplitPatternBlocks(current_block, split_pattern, blocks);
  bool splittable[4];
  GetSplittable(blocks, split_pattern.num_blocks, splittable);
  for (int32_t i = split_pattern.num_blocks - 1; i >= 0; --i) {
    block_queue_.push_back_no_resize({blocks[i], splittable[i]});
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
  const Block bounds =
      front_mgr_.LargestPossibleBlockAt(bx, by, /*snapped=*/true);
  const BlockSize size = GetFittingBlockSize(partition_set_, bounds.dim());
  const bool splittable = (GetNumSplitPatterns(partition_set_, size) > 1);
  block_queue_.push_back_no_resize({{bx, by, size}, splittable});
}

WP2Status SplitIteratorRecurse::CopyFrom(const SplitIteratorRecurse& other) {
  WP2_CHECK_STATUS(SplitIteratorBase::CopyInternal(other));
  WP2_CHECK_ALLOC_OK(block_queue_.copy_from(other.block_queue_));
  return WP2_STATUS_OK;
}

void SplitIteratorRecurse::GetSplittable(const Block sub_blocks[4],
                                         uint32_t num_sub_blocks,
                                         bool splittable[4]) const {
  // Only recurse when using the 2x2 split.
  const bool recurse =
      (num_sub_blocks == 4) && (sub_blocks[0].x() != sub_blocks[3].x()) &&
      (sub_blocks[0].y() != sub_blocks[3].y()) && (sub_blocks[0].area() > 1);
  for (uint32_t i = 0; i < num_sub_blocks; ++i) {
    splittable[i] = recurse;
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
