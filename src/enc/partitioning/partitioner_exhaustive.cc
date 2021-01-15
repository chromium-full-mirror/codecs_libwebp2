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

#include "src/enc/partitioning/partitioner_exhaustive.h"

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

static constexpr uint32_t kBlkStart = BLK_LAST;

WP2Status ExhaustivePartitioner::Init(const EncoderConfig& config,
                                      const YUVPlane& yuv,
                                      const Rectangle& tile_rect,
                                      PartitionScoreFunc* const score_func) {
  WP2_CHECK_STATUS(Partitioner::Init(config, yuv, tile_rect, score_func));

  const BlockSize* size = GetBlockSizes(config_->partition_set);
  next_block_sizes_[kBlkStart] = *size;  // First one.
  for (; *size != BLK_LAST; ++size) next_block_sizes_[*size] = *(size + 1);

  WP2_CHECK_STATUS(front_mgr_.Init(config_->partition_set,
                                   config_->partition_snapping,
                                   src_->Y.w_, src_->Y.h_));
  return WP2_STATUS_OK;
}

WP2Status ExhaustivePartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);

  WP2_CHECK_ALLOC_OK(forced_blocks_.resize(blocks->size()));
  std::copy(blocks->begin(), blocks->end(), forced_blocks_.begin());
  // Only occupy the forced blocks, the others are determined by 'front_mgr_'.
  for (const Block& block : forced_blocks_) Occupy(block);
  // Sort the 'forced_blocks_' by position for binary search later on.
  std::sort(forced_blocks_.begin(), forced_blocks_.end());

  float best_score = 0.f;
  VectorNoCtor<Block>* const best_partition = blocks;
  best_partition->clear();
  front_mgr_.Clear();
  partition_.clear();

  uint64_t num_iterations = 0;
  while (true) {
    bool found;
    float score;
    // Using half-by-half because the number of iterations can be very large and
    // is not convenient to compute.
    const ProgressRange iteration_progress(progress,
                                           std::pow(0.5, 1. + num_iterations));
    WP2_CHECK_STATUS(GetNextPartitionScore(iteration_progress, &found, &score));
    if (!found) break;

    // Make sure the partition fills exactly the tile.
    assert(std::accumulate(partition_.begin(), partition_.end(), 0u,
                           [](uint32_t sum, const Block& block) {
                             return sum + (BlockWidth[block.dim()] *
                                           BlockHeight[block.dim()]);
                           }) == num_block_cols_ * num_block_rows_);

    if (best_partition->empty() || score > best_score) {
      best_score = score;
      WP2_CHECK_ALLOC_OK(best_partition->resize(partition_.size()));
      std::copy(partition_.begin(), partition_.end(), best_partition->begin());
    }
    ++num_iterations;
  }
  WP2_CHECK_STATUS(progress.AdvanceBy(std::pow(0.5, 1. * num_iterations)));

  WP2_CHECK_REDUCED_STATUS(
      RegisterScoreForVDebug(best_score, best_partition->size(),
                             num_iterations));
  return WP2_STATUS_OK;
}

WP2Status ExhaustivePartitioner::GetNextPartitionScore(
    const ProgressRange& progress, bool* const found, float* const score) {
  // Begin the tree search.
  BlockSize next_block_size = next_block_sizes_[kBlkStart];
  bool is_first_branch = partition_.empty();
  // Find a new branch in the search tree if possible.
  while (is_first_branch || FindNewBranch(&next_block_size)) {
    is_first_branch = false;
    // Try to go down the new branch.
    do {
      Block block;
      VectorNoCtor<Block>::iterator forced_block_it;
      // Check if the 'next_block_size' can fit, then if the 'block' is snapped
      // or does not need to, then if the 'block' is forced or intersecting one.
      if (front_mgr_.TryGetNextBlock(next_block_size, &block) &&
          (!config_->partition_snapping || block.IsSnapped()) &&
          (forced_blocks_.empty() || !IsOccupied(block) ||
           ((forced_block_it = std::lower_bound(forced_blocks_.begin(),
                                                forced_blocks_.end(), block)) !=
                forced_blocks_.end() &&
            *forced_block_it == block))) {
        if (forced_blocks_.empty()) assert(!IsOccupied(block));

        // 'block' is valid.
        WP2_CHECK_ALLOC_OK(front_mgr_.UseSize(next_block_size,
                                              /*ind=*/0u, &block));
        front_mgr_.Use(block);
        WP2_CHECK_ALLOC_OK(partition_.push_back(block));

        if (front_mgr_.Done()) {
          // 'partition_' is complete. Get its score.
          WP2_CHECK_STATUS(
              tile_score_func_->TryEncode(partition_, progress, score));
          *found = true;
          return WP2_STATUS_OK;
        } else {
          // 'partition_' is incomplete. Go deeper.
          next_block_size = next_block_sizes_[kBlkStart];
        }
      } else {
        // 'block' is invalid. Try the next size or leave this branch.
        next_block_size = next_block_sizes_[next_block_size];
      }
    } while (next_block_size != BLK_LAST);
  }

  *found = false;
  return WP2_STATUS_OK;
}

bool ExhaustivePartitioner::FindNewBranch(BlockSize* const next_block_size) {
  while (!partition_.empty()) {
    *next_block_size = next_block_sizes_[partition_.back().dim()];
    front_mgr_.UndoUse(partition_.back());
    front_mgr_.UndoUseSize(partition_.back());
    partition_.pop_back();
    if (*next_block_size != BLK_LAST) return true;  // Found new branch.
  }
  return false;  // Browsed the whole tree.
}

//------------------------------------------------------------------------------

}  // namespace WP2
