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

#include "src/enc/partitioning/partitioner_area.h"

#include <algorithm>
#include <numeric>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/progress_watcher.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partition_score_func_area.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status AreaRDOptPartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  uint32_t max_num_blocks_left = num_block_cols_ * num_block_rows_;

  // Forced blocks are not supported unless they cover the whole buffer.
  // TODO(yguyon): Handle forced blocks
  WP2_CHECK_STATUS(RegisterForcedBlocks(*blocks, &max_num_blocks_left));
  if (max_num_blocks_left == 0) {
    return WP2_STATUS_OK;
  }
  WP2_CHECK_OK(blocks->empty(), WP2_STATUS_UNSUPPORTED_FEATURE);

  const uint32_t width = src_->GetWidth(), height = src_->GetHeight();
  const uint32_t num_areas =
      DivCeil(width, area_width_) * DivCeil(height, area_height_);
  const ProgressScale per_area_progress(progress, 1. / num_areas);

  for (uint32_t area_y = 0; area_y < height; area_y += area_height_) {
    for (uint32_t area_x = 0; area_x < width; area_x += area_width_) {
      const Rectangle area =
          Rectangle(area_x, area_y, area_width_, area_height_)
              .ClipWith({0, 0, src_->Y.w_, src_->Y.h_});  // Padded.

      WP2_CHECK_STATUS(GetAreaBestPartition(area, per_area_progress, blocks,
                                            &max_num_blocks_left));
    }
  }
  assert(max_num_blocks_left == 0);
  return WP2_STATUS_OK;
}

WP2Status AreaRDOptPartitioner::GetAreaBestPartition(
    const Rectangle& area, const ProgressRange& progress,
    VectorNoCtor<Block>* const blocks, uint32_t* const max_num_blocks_left) {
  AreaScoreFunc* const score_func =
      reinterpret_cast<AreaScoreFunc*>(score_func_);
  assert(area.x == score_func->GetCurrentArea().x &&
         area.y == score_func->GetCurrentArea().y);
  VectorNoCtor<Block> area_blocks, area_best_blocks;
  const double per_layout_progress =
      1. / (1 + GetNumBlockSizes(config_->partition_set));

  // Start with the default partition as a reference score.
  float best_score;
  WP2_CHECK_STATUS(
      score_func->GetAreaDefaultScore(&area_best_blocks, &best_score));
  WP2_CHECK_STATUS(progress.AdvanceBy(per_layout_progress));

  // Test a uniform block grid for each block size and keep the best score.
  for (const BlockSize* size = GetBlockSizes(config_->partition_set);
       *size != BLK_LAST; ++size) {
    if (BlockWidthPix(*size) <= area.width &&
        BlockHeightPix(*size) <= area.height) {
      area_blocks.clear();
      float score;
      WP2_CHECK_STATUS(
          score_func->GetAreaGridScore(*size, &area_blocks, &score));
      if (score > best_score) {
        best_score = score;
        WP2_CHECK_ALLOC_OK(area_best_blocks.resize(area_blocks.size()));
        std::copy(area_blocks.begin(), area_blocks.end(),
                  area_best_blocks.begin());
      }
    }
    WP2_CHECK_STATUS(progress.AdvanceBy(per_layout_progress));
  }

  // Mark each block of the best partition as final.
  for (const Block& block : area_best_blocks) {
    assert(*max_num_blocks_left > 0);
    *max_num_blocks_left -= block.rect().GetArea();
    WP2_CHECK_ALLOC_OK(blocks->push_back(block));
    WP2_CHECK_STATUS(score_func_->Use(block));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status SubAreaRDOptPartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks) {
  WP2_CHECK_STATUS(front_mgr_.Init(config_->partition_set,
                                   config_->partition_snapping,
                                   tile_rect_.width, tile_rect_.height));
  return AreaRDOptPartitioner::GetBestPartition(progress, blocks);
}

WP2Status SubAreaRDOptPartitioner::GetAreaBestPartition(
    const Rectangle& area, const ProgressRange& progress,
    VectorNoCtor<Block>* const blocks, uint32_t* const max_num_blocks_left) {
  const uint32_t max_num_blocks = num_block_cols_ * num_block_rows_;
  assert(area.x == front_mgr_.GetMaxPossibleBlock().x_pix() &&
         area.y == front_mgr_.GetMaxPossibleBlock().y_pix());

  const uint32_t area_max_num_blocks =
      DivCeil(area.width, kPredWidth) * DivCeil(area.height, kPredHeight);
  const ProgressScale block_progress(progress, 1. / area_max_num_blocks);
  const BlockSize* const sizes = GetBlockSizes(config_->partition_set);
  const ProgressScale size_progress(
      block_progress, 1. / GetNumBlockSizes(config_->partition_set));

  while (!front_mgr_.Done()) {
    const Block max_block = front_mgr_.GetMaxPossibleBlock();
    // Exit the loop when the 'area' is complete.
    if (!area.Contains(max_block.x_pix(), max_block.y_pix())) break;

    assert(*max_num_blocks_left > 0);
    Block best_block;
    float best_score = std::numeric_limits<float>::lowest();
    for (const BlockSize* size = sizes; *size != BLK_LAST; ++size) {
      const Block block = Block(max_block.x(), max_block.y(), *size);
      if (block.w() <= max_block.w() && block.h() <= max_block.h()) {
        float score;
        WP2_CHECK_STATUS(
            score_func_->ComputeScore(block, size_progress, &score));

        if (score >= best_score) {
          best_score = score;
          best_block = block;
        }
      } else {
        WP2_CHECK_STATUS(size_progress.AdvanceBy(1.));
      }
    }
    const uint32_t block_index = max_num_blocks - *max_num_blocks_left;
    (void)block_index;
    WP2_CHECK_REDUCED_STATUS(
        RegisterOrderForVDebug(0, 0, best_block, block_index, max_num_blocks));
    *max_num_blocks_left -= best_block.rect().GetArea();
    WP2_CHECK_ALLOC_OK(blocks->push_back(best_block));
    WP2_CHECK_ALLOC_OK(
        front_mgr_.UseSize(best_block.dim(), 0u, nullptr, /*use_block=*/true));
    WP2_CHECK_STATUS(score_func_->Use(best_block));

    // Cover the whole block area.
    WP2_CHECK_STATUS(
        block_progress.AdvanceBy(best_block.rect().GetArea() - 1.));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
