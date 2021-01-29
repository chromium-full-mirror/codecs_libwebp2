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

constexpr SplitRecursePartitioner::SplitInfo
    SplitRecursePartitioner::kNoSplitInfo;

WP2Status SplitRecursePartitioner::Init(const EncoderConfig& config,
                                        const YUVPlane& yuv,
                                        const Rectangle& tile_rect,
                                        PartitionScoreFunc* const score_func) {
  WP2_CHECK_STATUS(Partitioner::Init(config, yuv, tile_rect, score_func));
  WP2_CHECK_STATUS(split_info_.Resize(num_block_cols_, num_block_rows_));
  return WP2_STATUS_OK;
}

WP2Status SplitRecursePartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  split_info_.Fill(kNoSplitInfo);
  const uint32_t max_num_blocks = num_block_cols_ * num_block_rows_;
  uint32_t max_num_blocks_left = max_num_blocks;

  // Begin with the user-defined blocks, if any (debug).
  WP2_CHECK_STATUS(RegisterForcedBlocks(*blocks, &max_num_blocks_left));

  // Sort forced blocks by position for binary search below.
  VectorNoCtor<Block> forced_blocks;
  WP2_CHECK_ALLOC_OK(forced_blocks.resize(blocks->size()));
  std::copy(blocks->begin(), blocks->end(), forced_blocks.begin());
  std::sort(forced_blocks.begin(), forced_blocks.end());

  // Use a front manager to easily browse the blocks.
  FrontMgrLexico front_mgr;
  WP2_CHECK_STATUS(front_mgr.Init(config_->partition_set,
                                  config_->partition_snapping, src_->Y.w_,
                                  src_->Y.h_));
  Vector<SplitPattern> patterns;
  WP2_CHECK_ALLOC_OK(patterns.reserve(10));  // maximum 10 ways to split a block
  BlockScoreFunc* const score_func = (BlockScoreFunc*)score_func_;

  while (!front_mgr.Done()) {
    const Block max_block = front_mgr.GetMaxPossibleBlock();

    // Binary search to find if 'max_block' is already one of the forced blocks.
    const auto forced_block_it =
        std::lower_bound(forced_blocks.begin(), forced_blocks.end(), max_block);

    if (forced_block_it != forced_blocks.end() &&
        forced_block_it->x() == max_block.x() &&
        forced_block_it->y() == max_block.y()) {
      const Block block = *forced_block_it;
      assert(block.dim() != BLK_LAST);
      WP2_CHECK_ALLOC_OK(front_mgr.UseSize(block.dim(), /*ind=*/0u, nullptr,
                                           /*use_block=*/true));
      WP2_CHECK_STATUS(score_func_->Use(block));
      WP2_CHECK_STATUS(
          progress.AdvanceBy(1. * block.rect().GetArea() / max_num_blocks));
    } else {
      assert(max_num_blocks_left > 0);

      // Retrieve the previous decision to split this block, if any.
      const SplitInfo& split_info = GetSplitInfo(max_block.x(), max_block.y());
      if (split_info == kNoSplitInfo || split_info.recursive) {
        // Nothing decided or recursive allowed: try more split patterns.
        Block block;
        if (split_info == kNoSplitInfo) {
          block =
              Block(max_block.x(), max_block.y(),
                    GetMaxFittingBlockSize(max_block, forced_blocks.size()));
        } else {
          block = Block(max_block.x(), max_block.y(), split_info.size);
        }
        assert(IsBlockValid(block, max_block, forced_blocks.size()));
        assert(IsSplitInfo(block, split_info));

        // Get all split patterns and keep the one with the best score.
        patterns.clear();
        GetSplitPatterns(config_->partition_set, block.dim(), &patterns);
        SplitPattern best_pattern;
        if (patterns.size() == 1) {
          best_pattern = patterns.front();
        } else {
          best_pattern = {/*num_blocks=*/0, {BLK_LAST}};
          float best_score = 0.f;
          for (const SplitPattern& pattern : patterns) {
            Block pattern_blocks[4];
            GetSplitPatternBlocks(block, pattern, pattern_blocks);

            float score;
            WP2_CHECK_STATUS(score_func->ComputeScore(
                pattern_blocks, pattern.num_blocks, &score));
            if (best_pattern.num_blocks == 0 || score > best_score) {
              best_pattern = pattern;
              best_score = score;
            }
          }
          assert(best_pattern.num_blocks > 0);
        }

        // Remember the decision.
        Block best_pattern_blocks[4];
        GetSplitPatternBlocks(block, best_pattern, best_pattern_blocks);
        for (uint32_t i = 0; i < best_pattern.num_blocks; ++i) {
          const bool recursive = (best_pattern.num_blocks > 1 &&
                                  best_pattern_blocks[i].dim() != BLK_4x4);
          SetSplitInfo(best_pattern_blocks[i], recursive);
          // TODO(yguyon): Recurse in fewer situations to save CPU.
        }

        WP2_CHECK_STATUS(progress.AdvanceBy(0.));  // Nothing decided.
      } else {
        // This 'block' was decided as is with no possibility of recursion.
        const Block block =
            Block(max_block.x(), max_block.y(), split_info.size);
        assert(IsBlockValid(block, max_block, forced_blocks.size()));
        assert(IsSplitInfo(block, split_info));
        WP2_CHECK_REDUCED_STATUS(
            RegisterOrderForVDebug(0, 0, block, blocks->size(),
                                   max_num_blocks));
        WP2_CHECK_ALLOC_OK(blocks->push_back(block));
        WP2_CHECK_ALLOC_OK(front_mgr.UseSize(block.dim(), /*ind=*/0u, nullptr,
                                             /*use_block=*/true));
        WP2_CHECK_STATUS(score_func_->Use(block));

        // The following instructions are done only for assertions/debug
        // because the front manager should handle everything, except for
        // the forced blocks but these are checked above.
        assert((Occupy(block), true));
        assert(max_num_blocks_left >= block.w() * block.h());
        max_num_blocks_left -= block.w() * block.h();
        assert(block.dim() != BLK_LAST);

        WP2_CHECK_STATUS(
            progress.AdvanceBy(1. * block.rect().GetArea() / max_num_blocks));
      }
    }
  }
  assert(max_num_blocks_left == 0);
  return WP2_STATUS_OK;
}

BlockSize SplitRecursePartitioner::GetMaxFittingBlockSize(
    const Block& max_block, uint32_t num_forced_blocks) const {
  BlockSize best_size = BLK_LAST;
  const BlockSize* const sizes = GetBlockSizes(config_->partition_set);
  for (const BlockSize* size = sizes; *size != BLK_LAST; ++size) {
    const Block block = Block(max_block.x(), max_block.y(), *size);

    if ((best_size == BLK_LAST ||
         BlockWidth[*size] * BlockHeight[*size] >
             BlockWidth[best_size] * BlockHeight[best_size]) &&
        IsBlockValid(block, max_block, num_forced_blocks) &&
        IsSplitInfo(block, kNoSplitInfo)) {
      best_size = block.dim();
    }
  }
  assert(best_size != BLK_LAST);
  return best_size;
}

static BlockSize GetExactBlockSize(PartitionSet set, uint32_t w, uint32_t h) {
  const BlockSize s = GetBlockSize(std::max(1u, w), std::max(1u, h));
  if (IsBlockSizeInSet(set, s) && BlockWidth[s] == w && BlockHeight[s] == h) {
    return s;
  }
  return BLK_LAST;
}

void SplitRecursePartitioner::GetSplitPatterns(PartitionSet set, BlockSize size,
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

void SplitRecursePartitioner::GetSplitPatternBlocks(const Block& block,
                                                    const SplitPattern& pattern,
                                                    Block blocks[4]) {
  assert(pattern.num_blocks >= 1 && pattern.num_blocks <= 4);
  // Treat this special case differently. Others can be done row by row.
  const bool is_2x4_2x2_2x2 =
      (pattern.num_blocks == 3 &&
       BlockHeight[pattern.block_sizes[0]] == block.h());
  uint32_t num_blocks = 0, x = block.x(), y = block.y();
  while (num_blocks < pattern.num_blocks) {
    if (x == block.x() + block.w()) {  // End of a row.
      // Start a new row aligned under the first block, unless 'is_2x4_2x2_2x2'.
      x = blocks[is_2x4_2x2_2x2 ? 1 : 0].x();
      y += blocks[is_2x4_2x2_2x2 ? 1 : 0].h();
    }
    assert(x < block.x() + block.w() && y < block.y() + block.h());
    blocks[num_blocks] = Block(x, y, pattern.block_sizes[num_blocks]);
    x += blocks[num_blocks].w();
    ++num_blocks;
  }
}

const SplitRecursePartitioner::SplitInfo& SplitRecursePartitioner::GetSplitInfo(
    uint32_t x, uint32_t y) const {
  return split_info_.At(x, y);
}
bool SplitRecursePartitioner::IsSplitInfo(const Block& block,
                                          const SplitInfo& split_info) const {
  const SplitInfo* row = &split_info_.At(block.x(), block.y());
  for (uint32_t y = 0; y < block.h(); ++y, row += split_info_.Step()) {
    for (uint32_t x = 0; x < block.w(); ++x) {
      if (row[x] != split_info) return false;
    }
  }
  return true;
}
void SplitRecursePartitioner::SetSplitInfo(const Block& block, bool recursive) {
  SplitInfo* row = &split_info_.At(block.x(), block.y());
  for (uint32_t y = 0; y < block.h(); ++y, row += split_info_.Step()) {
    for (uint32_t x = 0; x < block.w(); ++x) row[x] = {block.dim(), recursive};
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
