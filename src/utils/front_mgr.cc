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
// Front-Line manager. This struct keeps track of the covering process, when
// blocks are added one by one.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/utils/front_mgr.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <numeric>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status FrontMgrBase::InitBase(uint32_t width, uint32_t height) {
  assert(width > 0 && height > 0);
  bwidth_ = SizeBlocks(width);
  bheight_ = SizeBlocks(height);

  WP2_CHECK_ALLOC_OK(occupancy_.resize(bwidth_));
  Clear();
  return WP2_STATUS_OK;
}

void FrontMgrBase::Clear() {
  std::fill(occupancy_.begin(), occupancy_.end(), 0);
  left_ = bwidth_ * bheight_;
}

bool FrontMgrBase::IsUsed(uint32_t x, uint32_t y) const {
  assert(!occupancy_.empty());
  assert(x < bwidth_);
  return (occupancy_[x] > y);
}

uint32_t FrontMgrBase::GetOccupancy(int x) const {
  assert(!occupancy_.empty());
  if (x < 0 || x >= (int)bwidth_) return 0;
  return occupancy_[x];
}

uint32_t FrontMgrBase::GetRightContextExtent(const Block& blk) const {
  assert(!occupancy_.empty());
  if (blk.y() == 0u) return 0u;
  const uint32_t i_min = blk.x() + blk.w();
  const uint32_t i_max =
      std::min(bwidth_, i_min + kMaxContextTRExtent / kMinBlockSizePix);
  uint32_t i = i_min;
  while (i < i_max && occupancy_[i] >= blk.y()) ++i;
  return (i - i_min);
}

void FrontMgrBase::Use(const Block& block) {
  assert(!occupancy_.empty());
  assert(left_ >= block.w() * block.h());
  const uint32_t x = block.x(), y = block.y();
  const uint32_t width = block.w(), height = block.h();
  const uint32_t new_y = y + height;
  assert(new_y <= bheight_ && x + width <= bwidth_);
  for (uint32_t i = 0; i < width; ++i) {
    // Make sure all values are the same.
    assert(occupancy_[x + i] == y);
  }
  for (uint32_t i = 0; i < width; ++i) {
    occupancy_[x + i] = new_y;
  }
  left_ -= block.w() * block.h();
}

void FrontMgrBase::UndoUse(const Block& block) {
  assert(block.x() + block.w() <= bwidth_);
  for (uint32_t i = 0; i < block.w(); ++i) {
    assert(occupancy_[block.x() + i] == block.y() + block.h());
    occupancy_[block.x() + i] = block.y();
  }
  left_ += block.w() * block.h();
}

WP2Status FrontMgrBase::CopyInternal(const FrontMgrBase& other) {
  WP2_CHECK_ALLOC_OK(occupancy_.copy_from(other.occupancy_));
  bwidth_ = other.bwidth_;
  bheight_ = other.bheight_;
  left_ = other.left_;
  return WP2_STATUS_OK;
}

void FrontMgrBase::FindNextLexico(uint32_t* const bx,
                                  uint32_t* const by) const {
  const auto iter = std::min_element(occupancy_.begin(), occupancy_.end());
  *bx = iter - occupancy_.begin();
  *by = *iter;
}

void FrontMgrBase::FindNextLexico(const Block& added_last, uint32_t* const bx,
                                  uint32_t* const by) const {
  const uint32_t last_y = added_last.y();
  const uint32_t next_x = added_last.x() + added_last.w();
  // If the level on the right of that last added block is the same, this is our
  // next block in lexicographic order.
  if (next_x < occupancy_.size() && occupancy_[next_x] == last_y) {
    *bx = next_x;
    *by = last_y;
  } else {
    FindNextLexico(bx, by);
  }
}

Block FrontMgrBase::LargestPossibleBlockAt(uint32_t x, uint32_t y,
                                           bool snapped) const {
  assert(GetOccupancy(x) == y);
  // Search for the maximum possible width.
  uint32_t w;
  for (w = 1; w < kMaxBlockSize; ++w) {
    const uint32_t next_x = x + w;
    if (next_x >= bwidth_ || GetOccupancy(next_x) > y) break;
  }
  assert(y <= bheight_);
  const uint32_t h = bheight_ - y;
  return Block(x, y,
               snapped ? GetSnappedBlockSize(x, y, w, h) : GetBlockSize(w, h));
}

//------------------------------------------------------------------------------

WP2Status FrontMgr4x4::CopyFrom(const FrontMgr4x4& other) {
  WP2_CHECK_STATUS(FrontMgrBase::CopyInternal(other));
  bx_ = other.bx_;
  by_ = other.by_;
  return WP2_STATUS_OK;
}

void FrontMgr4x4::Clear() {
  FrontMgrBase::Clear();
  bx_ = by_ = 0;
}

void FrontMgr4x4::Use4x4() {
  Use(Block(bx_, by_, BLK_4x4));
  ++bx_;
  if (bx_ == bwidth_) {
    bx_ = 0;
    ++by_;
  }
}

//------------------------------------------------------------------------------

WP2Status FrontMgrDoubleOrderBase::Init(PartitionSet partition_set,
                                        bool snapped, uint32_t width,
                                        uint32_t height) {
  WP2_CHECK_STATUS(occupancy_size_.InitBase(width, height));
  assert(partition_set < NUM_PARTITION_SETS);
  partition_set_ = partition_set;
  snapped_ = snapped;
  WP2_CHECK_STATUS(FrontMgrBase::InitBase(width, height));
  WP2_CHECK_ALLOC_OK(size_stack_.reserve(SizeBlocks(width + kMaxBlockSizePix)));
  return WP2_STATUS_OK;
}

void FrontMgrDoubleOrderBase::Clear() {
  FrontMgrBase::Clear();
  occupancy_size_.Clear();
  size_stack_.clear();
  if (left_ > 0) next_ = occupancy_size_.LargestPossibleBlockAt(0, 0, snapped_);
}

bool FrontMgrDoubleOrderBase::UseReady(Block* const block_out) {
  if (size_stack_.empty()) return false;
  const Block& block = size_stack_.back();
  if (IsReady(block)) {
    size_stack_.pop_back();
    Use(block);
    if (block_out != nullptr) *block_out = block;
    return true;
  } else {
    return false;
  }
}

bool FrontMgrDoubleOrderBase::UseSize(BlockSize dim, Block* const block,
                                      bool use_block) {
  Block block_tmp;
  if (!TryGetNextBlock(dim, &block_tmp)) {
    assert(false);
    return false;
  }
  if (block != nullptr) *block = block_tmp;
  occupancy_size_.Use(block_tmp);
  if (use_block) {
    assert(IsReady(block_tmp));
    Use(block_tmp);
  } else {
    size_stack_.push_back_no_resize(block_tmp);
  }
  if (!occupancy_size_.Done()) {
    next_ = FindNextBlock(/*last_block=*/block_tmp);
    assert(next_.x() + next_.w() <= bwidth_ &&
           next_.y() + next_.h() <= bheight_);
  }
  return true;
}

// Default implementation that creates a block of size 'size' in the top left
// corner of 'next_'. Can be overridden by subclasses.
bool FrontMgrDoubleOrderBase::TryGetNextBlock(BlockSize size,
                                              Block* const block) const {
  if (BlockWidth[size] > next_.w() || BlockHeight[size] > next_.h()) {
    return false;
  }
  assert(left_ >= BlockWidth[size] * BlockHeight[size]);
  *block = Block(next_.x(), next_.y(), size);
  return true;
}

Block FrontMgrDoubleOrderBase::GetMaxFittingBlock() const {
  const Block max_block = GetMaxPossibleBlock();
  // If 'max_block' is 'snapped_', all inner rectangles with the same position
  // will be snapped too, so no need to enforce it here. Just clamp to a
  // rectangle belonging to the 'partition_set_'.
  return Block(max_block.x(), max_block.y(),
               GetFittingBlockSize(partition_set_, max_block.dim()));
}

WP2Status FrontMgrDoubleOrderBase::CopyInternal(
    const FrontMgrDoubleOrderBase& other) {
  WP2_CHECK_STATUS(FrontMgrBase::CopyInternal(other));
  partition_set_ = other.partition_set_;
  snapped_ = other.snapped_;
  WP2_CHECK_STATUS(occupancy_size_.CopyInternal(other.occupancy_size_));
  WP2_CHECK_ALLOC_OK(size_stack_.copy_from(other.size_stack_));
  next_ = other.next_;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status FrontMgrLexico::CopyFrom(const FrontMgrLexico& other) {
  return FrontMgrDoubleOrderBase::CopyInternal(other);
}

Block FrontMgrLexico::FindNextBlock(const Block& last_block) const {
  uint32_t bx, by;
  occupancy_size_.FindNextLexico(last_block, &bx, &by);
  return occupancy_size_.LargestPossibleBlockAt(bx, by, snapped_);
}

void FrontMgrLexico::UndoUseSize(const Block& block) {
  occupancy_size_.UndoUse(block);
  uint32_t bx, by;
  occupancy_size_.FindNextLexico(&bx, &by);
  next_ = occupancy_size_.LargestPossibleBlockAt(bx, by, snapped_);
}

WP2Status FrontMgrLexico::Sort(VectorNoCtor<Block>& blocks,
                               Vector_u16& size_order_indices) const {
  std::sort(blocks.begin(), blocks.end());
  // Block sizes are written to the stream in the same order as the blocks.
  WP2_CHECK_ALLOC_OK(size_order_indices.resize(blocks.size()));
  std::iota(size_order_indices.begin(), size_order_indices.end(), 0);

  // Only CheckBlockList() in debug but don't assert WP2_STATUS_OUT_OF_MEMORY.
  WP2Status status = WP2_STATUS_OK;
  assert((status = CheckBlockList(blocks), true));  // NOLINT side effect assert
  WP2_CHECK_STATUS(status);
  return WP2_STATUS_OK;
}

WP2Status FrontMgrLexico::CheckBlockList(
    const VectorNoCtor<Block>& blocks) const {
  FrontMgrLexico mgr;
  const uint32_t width = bwidth_ * kMinBlockSizePix;
  const uint32_t height = bheight_ * kMinBlockSizePix;
  WP2_CHECK_STATUS(mgr.Init(partition_set_, snapped_, width, height));
  for (uint32_t i = 0; i < blocks.size(); ++i) {
    const Block& v = blocks[i];
    const Block b(mgr.next_.x(), mgr.next_.y(), v.dim());

    // Compare to the input block.
    if (!(b.x() == v.x() && b.y() == v.y())) {
      fprintf(stderr,
              "Block validation error: wrong x,y position! "
              "@%d,%d: expected %d,%d\n",
              v.x(), v.y(), b.x(), b.y());
      assert(false);
      return WP2_STATUS_INVALID_PARAMETER;
    }
    Block blk_tmp;
    WP2_CHECK_ALLOC_OK(mgr.UseSize(v.dim(), &blk_tmp));
    assert(b == blk_tmp);
    while (mgr.UseReady()) {
    }
  }
  return WP2_STATUS_OK;
}

bool FrontMgrLexico::IsReady(const WP2::Block& block) const {
  return (block.x() == 0 || occupancy_[block.x() - 1] > block.y());
}

//------------------------------------------------------------------------------

WP2Status FrontMgrMax::CopyFrom(const FrontMgrMax& other) {
  return FrontMgrDoubleOrderBase::CopyInternal(other);
}

Block FrontMgrMax::FindNextBlock(const Block& last_block) const {
  // Find an element in the queue that is not final.
  for (int32_t i = (int32_t)size_stack_.size() - 1; i >= 0; --i) {
    const Block& block = size_stack_[i];
    if (IsReadySize(block)) continue;
    uint32_t w, x, y;
    assert(block.x() > 0);
    const uint32_t right_x = block.x() - 1;
    for (x = right_x, w = 1; x > 0 && w < kMaxBlockSize; --x, ++w) {
      if (occupancy_size_.GetOccupancy(x - 1) !=
          occupancy_size_.GetOccupancy(right_x)) {
        break;
      }
    }
    y = occupancy_size_.GetOccupancy(right_x);
    const uint32_t h = bheight_ - y;
    // Align to the right.
    BlockSize largest_fit;
    if (snapped_) {
      largest_fit = GetSnappedBlockSize(kMaxTileSize - right_x - 1, y, w, h);
    } else {
      largest_fit = GetBlockSize(w, h);
    }
    x += w - BlockWidth[largest_fit];

    return Block(x, y, largest_fit);
  }

  uint32_t x, y;
  occupancy_size_.FindNextLexico(&x, &y);
  return occupancy_size_.LargestPossibleBlockAt(x, y, snapped_);
}

bool FrontMgrMax::TryGetNextBlock(BlockSize size, Block* const block) const {
  if (BlockWidth[size] > next_.w() || BlockHeight[size] > next_.h()) {
    return false;
  }
  // Find an element in the queue that is not final.
  int32_t i;
  for (i = (int32_t)size_stack_.size() - 1; i >= 0; --i) {
    if (!IsReadySize(size_stack_[i])) break;
  }
  uint32_t bx, by;
  if (i < 0) {
    occupancy_size_.FindNextLexico(&bx, &by);
  } else {
    assert(size_stack_[i].x() >= BlockWidth[size]);
    bx = size_stack_[i].x() - BlockWidth[size];
    by = occupancy_size_.GetOccupancy(bx);
  }
  *block = Block(bx, by, size);
  assert(block->x() >= next_.x());
  assert(block->y() >= next_.y());
  assert(block->x() + block->w() <= next_.x() + next_.w());
  assert(block->y() + block->h() <= next_.y() + next_.h());
  return true;
}

// Though this algorithm is O(N^2) in the number of blocks, it is actually O(N)
// in practice as the initial blocks are sorted and when searching for
// neighboring blocks, they are not far in that list.
WP2Status FrontMgrMax::Sort(VectorNoCtor<Block>& blocks,
                            Vector_u16& size_order_indices) const {
  // Sort() is const, we don't want to modify this instance's state, so we
  // use a new instance for sorting purposes.
  FrontMgrMax mgr_tmp;
  WP2_CHECK_STATUS(mgr_tmp.Init(partition_set_, snapped_,
                                bwidth_ * kMinBlockSizePix,
                                bheight_ * kMinBlockSizePix));

  uint32_t ind = 0;
  // blocks_size will contain the blocks in size order while blocks contains
  // the blocks sorted in block order.
  VectorNoCtor<WP2::Block> blocks_size;
  WP2_CHECK_ALLOC_OK(blocks_size.reserve(blocks.size()));
  while (ind < blocks.size()) {
    uint32_t best_ind = ind;
    if (mgr_tmp.size_stack_.empty()) {
      // If we have no queue of elements to check, find the leftmost block at
      // lowest heights.
      Block best = blocks[ind];
      for (uint32_t i = ind + 1; i < blocks.size(); ++i) {
        const Block& tmp = blocks[i];
        if (tmp.y() < best.y() || (tmp.y() == best.y() && tmp.x() < best.x())) {
          best = tmp;
          best_ind = i;
        }
      }
    } else {
      // If the left context is not full, find the first block that can fill
      // it.
      const uint32_t x = mgr_tmp.size_stack_.back().x() - 1;
      const uint32_t y = mgr_tmp.occupancy_size_.GetOccupancy(x);
      best_ind = blocks.size();
      for (uint32_t i = ind; i < blocks.size(); ++i) {
        const Block& b = blocks[i];
        if (b.x() <= x && x < b.x() + b.w() && b.y() <= y &&
            y < b.y() + b.h()) {
          best_ind = i;
          break;
        }
      }
      assert(best_ind < blocks.size());
    }
    // Add that new block to the list of blocks.
    const Block& block = blocks[best_ind];
    WP2_CHECK_ALLOC_OK(blocks_size.push_back(block));
    Block tmp;
    WP2_CHECK_ALLOC_OK(mgr_tmp.UseSize(block.dim(), &tmp));
    assert(block == tmp);

    // Empty the queue if needed.
    while (true) {
      Block blk;
      if (!mgr_tmp.UseReady(&blk)) break;
      // Place the block at its right spot.
      for (uint32_t i = ind; i < blocks.size(); ++i) {
        if (blocks[i] == blk) {
          std::swap(blocks[i], blocks[ind]);
          ++ind;
          break;
        }
      }
    }
    if (ind == blocks.size()) break;
  }

  // Convert blocks_size to indices.
  WP2_CHECK_ALLOC_OK(size_order_indices.resize(blocks.size()));
  std::iota(size_order_indices.begin(), size_order_indices.end(), 0);
  ind = 0;
  for (const Block& b : blocks_size) {
    for (uint32_t i = ind; i < blocks.size(); ++i) {
      if (blocks[size_order_indices[i]] == b) {
        std::swap(size_order_indices[i], size_order_indices[ind]);
        ++ind;
        break;
      }
    }
  }

  // Only CheckBlockList() in debug but don't assert WP2_STATUS_OUT_OF_MEMORY.
  WP2Status status = WP2_STATUS_OK;
  assert((status = CheckBlockList(blocks), true));  // NOLINT side effect assert
  WP2_CHECK_STATUS(status);

  return WP2_STATUS_OK;
}

WP2Status FrontMgrMax::CheckBlockList(const VectorNoCtor<Block>& blocks) const {
  FrontMgrBase occupancy;
  WP2_CHECK_STATUS(occupancy.InitBase(bwidth_ * kMinBlockSizePix,
                                      bheight_ * kMinBlockSizePix));
  for (const Block& b : blocks) {
    // Check we have context above.
    if (b.y() > 0) {
      for (uint32_t x = b.x(); x < b.x() + b.w(); ++x) {
        if (occupancy.GetOccupancy(x) != b.y()) {
          fprintf(stderr, "Context not good at %d %d ", b.x(), b.y());
          assert(false);
          return WP2_STATUS_INVALID_PARAMETER;
        }
      }
    }
    // Check we have context on the left.
    if (!DoIsReady(occupancy, b)) {
      fprintf(stderr, "Context not good at %d %d ", b.x(), b.y());
      assert(false);
      return WP2_STATUS_INVALID_PARAMETER;
    }
    occupancy.Use(b);
  }
  return WP2_STATUS_OK;
}

bool FrontMgrMax::IsReady(const WP2::Block& block) const {
  return DoIsReady(*this, block);
}

bool FrontMgrMax::DoIsReady(const FrontMgrBase& occupancy,
                            const WP2::Block& block) const {
  return (block.x() == 0 ||
          occupancy.GetOccupancy(block.x() - 1) >= block.y() + block.h());
}

bool FrontMgrMax::IsReadySize(const WP2::Block& block) const {
  return (block.x() == 0 ||
          occupancy_size_.GetOccupancy(block.x() - 1) >= block.y() + block.h());
}

//------------------------------------------------------------------------------

WP2Status FrontMgrArea::Init(PartitionSet partition_set, bool snapped,
                             uint32_t width, uint32_t height) {
  // TODO(yguyon): Handle partially snapped partitions
  //               (snapped areas, non-snapped blocks inside an area)
  WP2_CHECK_OK(snapped, WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_STATUS(
      FrontMgrDoubleOrderBase::Init(partition_set, snapped, width, height));
  return WP2_STATUS_OK;
}

WP2Status FrontMgrArea::CopyFrom(const FrontMgrArea& other) {
  return FrontMgrDoubleOrderBase::CopyInternal(other);
}

Block FrontMgrArea::FindNextBlock(const Block& last_block) const {
  uint32_t bx, by;
  FindNextInArea(occupancy_size_, &bx, &by);
  return occupancy_size_.LargestPossibleBlockAt(bx, by, snapped_);
}

WP2Status FrontMgrArea::Sort(VectorNoCtor<Block>& blocks,
                             Vector_u16& size_order_indices) const {
  std::sort(blocks.begin(), blocks.end(),
            Comp(kMaxTileSize / kMinBlockSizePix, area_width_, area_height_));
  // Same block and size order.
  WP2_CHECK_ALLOC_OK(size_order_indices.resize(blocks.size()));
  std::iota(size_order_indices.begin(), size_order_indices.end(), 0);
  return WP2_STATUS_OK;
}

bool FrontMgrArea::IsReady(const WP2::Block& block) const {
  return (block.x() == 0 || occupancy_[block.x() - 1] > block.y());
}

void FrontMgrArea::FindNextInArea(const FrontMgrBase& occupancy,
                                  uint32_t* const bx,
                                  uint32_t* const by) const {
  *bx = 0;
  *by = bheight_;
  const uint32_t area_y_outside = DivCeil(bheight_, area_height_);
  uint32_t area_x = 0, area_y = area_y_outside;
  for (uint32_t x = 0; x < bwidth_; ++x) {
    const uint32_t y = occupancy.GetOccupancy(x);
    const uint32_t a_x = x / area_width_;
    const uint32_t a_y = (y >= bheight_) ? area_y_outside : y / area_height_;
    // Select the available top-left most block in the top-left most area.
    if (a_y < area_y || (a_x == area_x && a_y == area_y && y < *by)) {
      *bx = x;
      *by = y;
      area_x = a_x;
      area_y = a_y;
    }
  }
}

FrontMgrArea::Comp::Comp(uint32_t tile_width, uint32_t area_width,
                         uint32_t area_height)
    : area_step_(DivCeil(tile_width, area_width)),
      area_width_(area_width),
      area_height_(area_height) {}

bool FrontMgrArea::Comp::operator()(const Block& a, const Block& b) const {
  const uint32_t a_index = GetAreaIndex(a), b_index = GetAreaIndex(b);
  if (a_index != b_index) return a_index < b_index;
  return a < b;
}

uint32_t FrontMgrArea::Comp::GetAreaIndex(const Block& block) const {
  return (block.y() / area_height_) * area_step_ + block.x() / area_width_;
}

//------------------------------------------------------------------------------

}  // namespace WP2
