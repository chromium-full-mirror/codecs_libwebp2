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
#include "src/utils/split_iterator.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status SplitPartitioner::Init(const EncoderConfig& config,
                                 const YUVPlane& yuv,
                                 const Rectangle& tile_rect,
                                 PartitionScoreFunc* const score_func) {
  WP2_CHECK_STATUS(Partitioner::Init(config, yuv, tile_rect, score_func));
  WP2_CHECK_OK(config.partition_snapping, WP2_STATUS_INVALID_CONFIGURATION);
  return WP2_STATUS_OK;
}

static float GetSplitCost(const SplitIteratorDefault& it, uint32_t split_idx,
                          Block sub_blocks[4], uint32_t num_sub_blocks,
                          SymbolCounter* const symbol_counter,
                          ANSEncCounter* const counter) {
  counter->Reset();
  WriteBlockSplit(it, split_idx, symbol_counter, counter);
  if (num_sub_blocks > 1) {
    // Add the cost of encoding the splits of the subblocks.
    SplitIteratorDefault it_copy;
    WP2_CHECK_STATUS(it_copy.CopyFrom(it));
    it_copy.SplitCurrentBlock(split_idx);
    for (uint32_t i = 0; i < num_sub_blocks; ++i) {
      const SplitIteratorBase::SplitInfo& current_block =
          it_copy.CurrentBlock();
      assert(current_block.block == sub_blocks[i]);
      if (current_block.splittable) {
        WriteBlockSplit(it_copy, kNoSplitIndex, symbol_counter, counter);
        it_copy.SplitCurrentBlock(kNoSplitIndex);
        assert(!it_copy.CurrentBlock().splittable);
        it_copy.NextBlock();
      } else {
        it_copy.NextBlock();
      }
    }
  }
  return counter->GetCost();
}

WP2Status SplitPartitioner::GetBestPartition(const ProgressRange& progress,
                                             VectorNoCtor<Block>* const blocks,
                                             Vector_u32* const splits) {
  std::fill(occupancy_.begin(), occupancy_.end(), false);
  const uint32_t max_num_blocks = num_block_cols_ * num_block_rows_;
  uint32_t max_num_blocks_left = max_num_blocks;

  // Sort forced blocks by position for IsCompatibleWithForcedBlocks.
  VectorNoCtor<Block> forced_blocks;
  WP2_CHECK_ALLOC_OK(forced_blocks.resize(blocks->size()));
  std::copy(blocks->begin(), blocks->end(), forced_blocks.begin());
  std::sort(forced_blocks.begin(), forced_blocks.end());
  blocks->clear();

  SplitIteratorDefault it;
  WP2_CHECK_STATUS(it.Init(config_->partition_set, src_->Y.w_, src_->Y.h_));
  BlockScoreFunc* const score_func = (BlockScoreFunc*)score_func_;

  // Used for computing cost of encoding the splits.
  SymbolsInfo symbols_info;
  const bool use_aom_coeffs = DecideAOMCoeffs(*config_, tile_rect_);
  const GlobalParams& gparams = score_func->gparams();
  WP2_CHECK_STATUS(symbols_info.InitLossy(
      gparams.segments_.size(), gparams.partition_set_,
      gparams.maybe_use_lossy_alpha_, use_aom_coeffs, /*use_splits=*/true));
  SymbolRecorder symbol_recorder;
  WP2_CHECK_STATUS(symbol_recorder.Allocate(symbols_info, max_num_blocks));
  SymbolCounter symbol_counter(&symbol_recorder);
  WP2_CHECK_STATUS(symbol_counter.Allocate({kSymbolBlockSize}));
  ANSEncNoop enc_noop;
  ANSEncCounter counter;

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
        bool splittable[4];
        const uint32_t num_sub_blocks =
            it.GetSplitPatternBlocks(split_idx, sub_blocks, splittable);

        // Handle forced blocks. Skip configurations which don't match the
        // forced blocks.
        const Block& block = it.CurrentBlock().block;
        if (!forced_blocks.empty() &&
            !IsCompatibleWithForcedBlocks(block, sub_blocks, splittable,
                                          num_sub_blocks, forced_blocks)) {
          continue;
        }

        // Cost of writing this split.
        const float split_cost =
            GetSplitCost(it, split_idx, sub_blocks, num_sub_blocks,
                         &symbol_counter, &counter);

        float score;
        WP2_CHECK_STATUS(score_func->ComputeScore(
            sub_blocks, num_sub_blocks, split_cost, best_score,
            /*force_selected=*/false, &score));
        if (best_pattern_idx == -1 || score > best_score) {
          best_pattern_idx = split_idx;
          best_score = score;
        }
      }
      if (best_pattern_idx == -1) {
        // All patterns were skipped because of forced blocks. Just pick the
        // first one.
        assert(!forced_blocks.empty());
        best_pattern_idx = 0;
      }

      // Remember the decision.
      WriteBlockSplit(it, best_pattern_idx, &symbol_recorder, &enc_noop);
      it.SplitCurrentBlock(best_pattern_idx);
      if (splits != nullptr) {
        WP2_CHECK_ALLOC_OK(splits->push_back(best_pattern_idx));
      }
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

}  // namespace WP2
