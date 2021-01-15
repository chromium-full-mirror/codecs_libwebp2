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

#include "src/enc/partitioning/partitioner_top_left.h"

#include <algorithm>
#include <numeric>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/progress_watcher.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status TopLeftBestPartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  const uint32_t max_num_blocks = num_block_cols_ * num_block_rows_;
  uint32_t max_num_blocks_left = max_num_blocks;
  const ProgressScale block_progress(progress, 1. / max_num_blocks);

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
                                  config_->partition_snapping,
                                  src_->Y.w_, src_->Y.h_));

  while (!front_mgr.Done()) {
    // TODO(yguyon): This is using a top-most first order. Find an order that
    //               maximizes top and left context instead.
    const Block max_block = front_mgr.GetMaxFittingBlock();

    // Binary search to find if 'max_block' is already one of the forced blocks.
    const auto forced_block_it =
        std::lower_bound(forced_blocks.begin(), forced_blocks.end(), max_block);

    Block best_block;
    if (forced_block_it != forced_blocks.end() &&
        forced_block_it->x() == max_block.x() &&
        forced_block_it->y() == max_block.y()) {
      best_block = *forced_block_it;
      WP2_CHECK_STATUS(block_progress.AdvanceBy(1.));
    } else {
      assert(max_num_blocks_left > 0);
      WP2_CHECK_STATUS(GetBestSize(max_block, forced_blocks.size(),
                                   block_progress, &best_block));
      WP2_CHECK_REDUCED_STATUS(
          RegisterOrderForVDebug(0, 0, best_block, blocks->size(),
                                 max_num_blocks));
      WP2_CHECK_ALLOC_OK(blocks->push_back(best_block));

      // The following instructions are done only for assertions/debug
      // because the front manager should handle everything, except for
      // the forced blocks but these are checked above.
      assert((Occupy(best_block), true));
      assert(max_num_blocks_left >= best_block.w() * best_block.h());
      max_num_blocks_left -= best_block.w() * best_block.h();
    }
    assert(best_block.dim() != BLK_LAST);
    WP2_CHECK_ALLOC_OK(front_mgr.UseSize(best_block.dim(),
                                         /*ind=*/0u, nullptr));
    front_mgr.Use(best_block);
    WP2_CHECK_STATUS(score_func_->Use(best_block));
    // Cover the whole block.
    WP2_CHECK_STATUS(
        block_progress.AdvanceBy(best_block.rect().GetArea() - 1.));
  }
  assert(max_num_blocks_left == 0);
  return WP2_STATUS_OK;
}

WP2Status TopLeftBestPartitioner::GetBestSize(const Block& max_block,
                                              uint32_t num_forced_blocks,
                                              const ProgressRange& progress,
                                              Block* const best_block) {
  const ProgressScale size_progress(
      progress, 1. / GetNumBlockSizes(config_->partition_set));
  const BlockSize* const sizes = GetBlockSizes(config_->partition_set);
  float best_score = 0.f;
  // Test each size and keep the best score.
  for (const BlockSize* size = sizes; *size != BLK_LAST; ++size) {
    const Block block = Block(max_block.x(), max_block.y(), *size);

    if (IsBlockValid(block, max_block, num_forced_blocks)) {
      float score;
      WP2_CHECK_STATUS(score_func_->ComputeScore(block, size_progress, &score));

      const bool first_score = (size == sizes);
      // Increasing sizes: keep the largest one if same score (rare).
      if (first_score || score >= best_score) {
        best_score = score;
        *best_block = block;
      }
    } else {
      WP2_CHECK_STATUS(size_progress.AdvanceBy(1.));
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
