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

#include "src/enc/partitioning/partitioner_multi.h"

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

using Pass = MultiScoreFunc::Pass;

WP2Status MultiPassPartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks,
    Vector_u32* const) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  uint32_t max_num_blocks_left = num_block_cols_ * num_block_rows_;
  WP2_CHECK_STATUS(RegisterForcedBlocks(*blocks, &max_num_blocks_left));
  const uint32_t max_num_passes =
      8 + GetNumBlockSizes(config_->partition_set) * 3;
  const ProgressScale pass_progress(progress, 1. / max_num_passes);

  // Run several passes to select matching blocks with a given size.
  // GetBlocks() will skip any pass with a BlockSize not belonging to the
  // current PartitionSet.

  // Start with easy-to-spot flat blocks (narrow range of values in the
  // original luma and alpha planes).
  for (BlockSize block_size : {BLK_32x32, BLK_32x16, BLK_16x32, BLK_16x16,
                               BLK_16x8, BLK_8x16, BLK_8x8}) {
    WP2_CHECK_STATUS(GetBlocks(Pass::LumaAlphaGradient, block_size,
                               Grid::Snapped, pass_progress,
                               &max_num_blocks_left, blocks));
  }
  if (config_->enable_alt_tuning &&
      config_->effort >= MultiScoreFunc::kMinEffortForGoodQuantDCT) {
    // Blocks having a DCT that still gives a good result when quantized.
    WP2_CHECK_STATUS(GetBlocks(Pass::GoodQuantDCT, BLK_32x32, Grid::Snapped,
                               pass_progress, &max_num_blocks_left, blocks));
  }

  // Pick a few not-too-small blocks that are following a luma edge.
  for (const BlockSize* block_size =
           GetBlockSizes(ALL_RECTS) + GetNumBlockSizes(ALL_RECTS) - 1u;
       *block_size != BLK_4x4; --block_size) {
    if (BlockWidth[*block_size] * BlockHeight[*block_size] < 4) continue;
    WP2_CHECK_STATUS(GetBlocks(Pass::Direction, *block_size, Grid::All,
                               pass_progress, &max_num_blocks_left, blocks));
  }

  // Choose smaller and smaller blocks based on variance.
  constexpr PartitionSet variance_partition_set = SMALL_RECTS;
  assert(*GetBlockSizes(variance_partition_set) == BLK_4x4);
  for (const BlockSize* allowed_block_size =
           GetBlockSizes(variance_partition_set) +
           GetNumBlockSizes(variance_partition_set) - 1u;
       *allowed_block_size != BLK_4x4; --allowed_block_size) {
    WP2_CHECK_STATUS(GetBlocks(Pass::NarrowStdDev, *allowed_block_size,
                               Grid::All, pass_progress, &max_num_blocks_left,
                               blocks));
  }

  // Merge all remaining small blocks that are following a luma edge.
  for (const BlockSize* block_size =
           GetBlockSizes(ALL_RECTS) + GetNumBlockSizes(ALL_RECTS) - 1u;
       *block_size != BLK_4x4; --block_size) {
    if (BlockWidth[*block_size] * BlockHeight[*block_size] >= 4) continue;
    WP2_CHECK_STATUS(GetBlocks(Pass::Direction, *block_size, Grid::All,
                               pass_progress, &max_num_blocks_left, blocks));
  }

  // Fill the remaining empty areas.
  WP2_CHECK_STATUS(GetBlocks(Pass::Any, BLK_4x4, Grid::All, pass_progress,
                             &max_num_blocks_left, blocks));
  assert(max_num_blocks_left == 0);
  assert(pass_index_ <= max_num_passes);
  WP2_CHECK_STATUS(pass_progress.AdvanceBy(max_num_passes - pass_index_));
  return WP2_STATUS_OK;
}

WP2Status MultiPassPartitioner::GetBlocks(Pass pass, BlockSize block_size,
                                          Grid restrict_to,
                                          const ProgressRange& progress,
                                          uint32_t* const max_num_blocks_left,
                                          VectorNoCtor<Block>* const out) {
  if (GetFittingBlockSize(config_->partition_set, block_size) != block_size) {
    return WP2_STATUS_OK;
  }

  // Only keep snapped blocks if the config forces it.
  if (config_->partition_snapping) {
    if (restrict_to == Grid::NonSnapped) return WP2_STATUS_OK;
    restrict_to = Grid::Snapped;
  }

  const uint32_t num_blocks_before = out->size();
  Block block(0, 0, block_size);

  // Make sure the 'block_size' is valid and there is at least one remaining.
  if (block.w() > num_block_cols_ || block.h() > num_block_rows_ ||
      block.area() > *max_num_blocks_left) {
    WP2_CHECK_REDUCED_STATUS(RegisterPassForVDebug(pass, block_size, 0));
    ++pass_index_;
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }

  struct BlockScore {
    Block blk;
    float score;
  };
  VectorNoCtor<BlockScore> block_scores;
  multi_score_func_->SetPass(pass);

  const uint32_t incr_x = (restrict_to == Grid::Snapped) ? block.w() : 1;
  const uint32_t incr_y = (restrict_to == Grid::Snapped) ? block.h() : 1;
  const uint32_t max_x = num_block_cols_ - block.w();
  const uint32_t max_y = num_block_rows_ - block.h();
  const uint32_t num_incr_x = DivCeil(max_x + 1, incr_x);
  const uint32_t num_incr_y = DivCeil(max_y + 1, incr_y);
  const ProgressScale incr_progress(progress, 1. / (num_incr_x * num_incr_y));
  const bool* occupancy = occupancy_.data();

  // Browse the grid of selectable blocks.
  for (uint32_t y = 0; y <= max_y;
       y += incr_y, occupancy += incr_y * num_block_cols_) {
    for (uint32_t x = 0; x <= max_x; x += incr_x) {
      if (!occupancy[x]) {  // Quickly skip top-left-occupied blocks.
        block.SetXY(x, y);
        if (restrict_to == Grid::Snapped) assert(block.IsSnapped());
        // Assume snapped blocks were already tried if 'restrict_to==NonSnapped'
        const bool tried_block_earlier =
            (restrict_to == Grid::NonSnapped && block.IsSnapped());

        if (!tried_block_earlier && !IsOccupied(block)) {
          float score;
          WP2_CHECK_STATUS(
              score_func_->ComputeScore(block, incr_progress, &score));
          if (score >= MultiScoreFunc::kMinScore) {
            if (restrict_to == Grid::Snapped) {
              // Immediately select any passing block.
              WP2_CHECK_STATUS(
                  SelectBlock(pass, block, max_num_blocks_left, out));
            } else {
              // Keep the scores of the passing blocks in memory.
              WP2_CHECK_ALLOC_OK(block_scores.push_back({block, score}));
            }
          }
        } else {
          WP2_CHECK_STATUS(incr_progress.AdvanceBy(1.));
        }
      } else {
        WP2_CHECK_STATUS(incr_progress.AdvanceBy(1.));
      }
    }
  }

  if (restrict_to != Grid::Snapped) {
    // Select blocks with best (=lowest) score first.
    // stable_sort() is used for its determinism.
    std::stable_sort(block_scores.begin(), block_scores.end(),
                     [](const BlockScore& l, const BlockScore& r) {
                       return l.score < r.score;
                     });

    for (const BlockScore& bs : block_scores) {
      if (IsOccupied(bs.blk)) continue;  // 'occupancy_' might have changed.
      WP2_CHECK_STATUS(SelectBlock(pass, bs.blk, max_num_blocks_left, out));
    }
  }

  (void)num_blocks_before;
  WP2_CHECK_REDUCED_STATUS(
      RegisterPassForVDebug(pass, block_size, out->size() - num_blocks_before));
  ++pass_index_;
  return WP2_STATUS_OK;
}

WP2Status MultiPassPartitioner::SelectBlock(MultiScoreFunc::Pass pass,
                                            const Block& block,
                                            uint32_t* const max_num_blocks_left,
                                            VectorNoCtor<Block>* const out) {
  WP2_CHECK_REDUCED_STATUS(
      RegisterOrderForVDebug((uint32_t)pass, pass_index_, block, out->size(),
                             num_block_cols_ * num_block_rows_));
  Occupy(block);
  const uint32_t block_area = block.w() * block.h();
  assert(*max_num_blocks_left >= block_area);
  *max_num_blocks_left -= block_area;
  WP2_CHECK_ALLOC_OK(out->push_back(block));
  WP2_CHECK_STATUS(score_func_->Use(block));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
