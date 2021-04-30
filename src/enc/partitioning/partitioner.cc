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

#include "src/enc/partitioning/partitioner.h"

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

WP2Status Partitioner::Init(const EncoderConfig& config, const YUVPlane& yuv,
                            const Rectangle& tile_rect,
                            PartitionScoreFunc* const score_func) {
  config_ = &config;
  tile_rect_ = tile_rect;
  src_ = &yuv;
  num_block_cols_ = SizeBlocks(yuv.Y.w_);
  num_block_rows_ = SizeBlocks(yuv.Y.h_);
  score_func_ = score_func;

  WP2_CHECK_ALLOC_OK(occupancy_.resize(num_block_cols_ * num_block_rows_));
  return WP2_STATUS_OK;
}

bool Partitioner::IsOccupied(const Block& block) const {
  const uint32_t stride = num_block_cols_;
  const bool* occupancy = occupancy_.data() + block.y() * stride + block.x();
  for (uint32_t y = 0; y < block.h(); ++y) {
    for (uint32_t x = 0; x < block.w(); ++x) {
      if (occupancy[x]) return true;
    }
    occupancy += num_block_cols_;
  }
  return false;
}

void Partitioner::Occupy(const Block& block) {
  const uint32_t stride = num_block_cols_;
  bool* occupancy = occupancy_.data() + block.y() * stride + block.x();
  for (uint32_t y = 0; y < block.h(); ++y) {
    for (uint32_t x = 0; x < block.w(); ++x) occupancy[x] = true;
    occupancy += num_block_cols_;
  }
}

bool Partitioner::IsBlockValid(const Block& block, const Block& max_block,
                               uint32_t num_forced_blocks) const {
  if (block.w() > max_block.w()) return false;
  if (block.h() > max_block.h()) return false;
  // Thanks to the FrontMgrLexico, this 'block' can only be rejected
  // because of other forced blocks or forced snapping.
  if (config_->partition_snapping && !block.IsSnapped()) return false;
  if (num_forced_blocks > 0) return !IsOccupied(block);
  assert(!IsOccupied(block));
  return true;
}

//------------------------------------------------------------------------------

WP2Status AddForcedBlocks(const EncoderConfig& config,
                          const Rectangle& padded_tile_rect,
                          VectorNoCtor<Block>* const out) {
  if (config.info == nullptr) return WP2_STATUS_OK;

  for (const Rectangle& rect_px : config.info->force_partition) {
    if (rect_px.x < padded_tile_rect.x || rect_px.y < padded_tile_rect.y) {
      continue;  // Ignore this block, it's outside of this tile.
    }
    const Rectangle local_rect = {rect_px.x - padded_tile_rect.x,
                                  rect_px.y - padded_tile_rect.y,
                                  rect_px.width, rect_px.height};
    if (local_rect.x >= padded_tile_rect.width ||
        local_rect.y >= padded_tile_rect.height) {
      continue;  // Ignore this block, it's outside of this tile.
    }
    WP2_CHECK_OK((local_rect.x + local_rect.width <= padded_tile_rect.width &&
                  local_rect.y + local_rect.height <= padded_tile_rect.height),
                 WP2_STATUS_INVALID_CONFIGURATION);

    WP2::BlockSize matching_size = BLK_LAST;
    for (const BlockSize* size = GetBlockSizes(config.partition_set);
         *size != BLK_LAST; ++size) {
      if (local_rect.width == WP2::BlockWidthPix(*size) &&
          local_rect.height == WP2::BlockHeightPix(*size)) {
        matching_size = *size;
        break;
      }
    }
    WP2_CHECK_OK(matching_size != BLK_LAST, WP2_STATUS_INVALID_CONFIGURATION);

    // Create a Block that matches the rectangle.
    Block block(local_rect.x / kMinBlockSizePix,
                local_rect.y / kMinBlockSizePix, matching_size);
    assert(local_rect.width == block.w_pix() &&
           local_rect.height == block.h_pix());
    // Verify that position coordinates are aligned on the block grid (multiples
    // of kMinBlockSizePix). Width and height are checked and asserted above.
    WP2_CHECK_OK(local_rect.x == block.x_pix() && local_rect.y == block.y_pix(),
                 WP2_STATUS_INVALID_CONFIGURATION);
    // No IsSnapped() check, trust the user's intention on that.

    WP2_CHECK_ALLOC_OK(out->push_back(block));
  }
  return WP2_STATUS_OK;
}

WP2Status Partitioner::RegisterForcedBlocks(
    const VectorNoCtor<Block>& forced_blocks,
    uint32_t* const max_num_blocks_left) {
  for (const Block& block : forced_blocks) {
    WP2_CHECK_OK(!IsOccupied(block), WP2_STATUS_INVALID_CONFIGURATION);
    Occupy(block);
    assert(*max_num_blocks_left >= block.w() * block.h());
    *max_num_blocks_left -= block.w() * block.h();
  }
  return WP2_STATUS_OK;
}

bool IsCompatibleWithForcedBlocks(const Block& block, const Block sub_blocks[4],
                                  const bool splittable[4],
                                  uint32_t num_sub_blocks,
                                  const VectorNoCtor<Block>& forced_blocks) {
  if (num_sub_blocks == 1) {
    assert(!splittable[0]);
    assert(sub_blocks[0].dim() == block.dim());
  }
  auto forced_block_it =
      std::lower_bound(forced_blocks.begin(), forced_blocks.end(), block);
  for (; forced_block_it != forced_blocks.end(); ++forced_block_it) {
    if (forced_block_it->y() >= block.y() + block.h()) break;
    const Rectangle forced_block = forced_block_it->rect();
    if (!block.rect().Intersects(forced_block)) continue;
    // Intersects with but does not contain it => not compatible.
    if (!block.rect().Contains(forced_block)) return false;

    for (uint32_t i = 0; i < num_sub_blocks; ++i) {
      if (!sub_blocks[i].rect().Intersects(forced_block)) continue;
      if (!sub_blocks[i].rect().Contains(forced_block)) return false;
      if (!splittable[i] && (sub_blocks[i].rect() != forced_block)) {
        return false;
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------

}  // namespace WP2
