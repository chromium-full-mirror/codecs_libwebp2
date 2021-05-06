// Copyright 2021 Google LLC
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
// Author: Maryla (maryla@google.com)

#include "src/enc/partitioning/partitioner_split_recurse.h"

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

WP2Status SplitRecursePartitioner::Init(const EncoderConfig& config,
                                        const YUVPlane& yuv,
                                        const Rectangle& tile_rect,
                                        PartitionScoreFunc* const score_func) {
  WP2_CHECK_STATUS(Partitioner::Init(config, yuv, tile_rect, score_func));
  WP2_CHECK_OK(config.partition_snapping, WP2_STATUS_INVALID_CONFIGURATION);
  return WP2_STATUS_OK;
}

static float GetSplitCost(const SplitIteratorDefault& it, uint32_t split_idx,
                          SymbolCounter* const symbol_counter,
                          ANSEncCounter* const counter) {
  counter->Reset();
  WriteBlockSplit(it, split_idx, symbol_counter, counter);
  return counter->GetCost();
}

WP2Status SplitRecursePartitioner::ComputeScore(
    const VectorNoCtor<Block>& forced_blocks, const SplitIteratorDefault& it,
    uint32_t split_idx, Block sub_blocks[4], uint32_t num_sub_blocks,
    BlockScoreFunc* const score_func, SymbolCounter* const symbol_counter,
    ANSEncCounter* const counter, float best_score, float extra_rate,
    uint32_t recursion_level, bool selected,
    Vector_u32* const best_pattern_idxs, float* score) const {
  assert(it.CurrentBlock().splittable);

  const Block outer_block = it.CurrentBlock().block;
  const float split_rate = GetSplitCost(it, split_idx, symbol_counter, counter);
  const float new_extra_rate = extra_rate + split_rate;

  // Check if one of the sub-blocks is splittable.
  bool more_recursion = false;
  if (num_sub_blocks > 1) {
    SplitIteratorDefault it_copy;
    WP2_CHECK_STATUS(it_copy.CopyFrom(it));
    it_copy.SplitCurrentBlock(split_idx);
    for (uint32_t i = 0; i < num_sub_blocks; ++i, it_copy.NextBlock()) {
      assert(it_copy.CurrentBlock().block == sub_blocks[i]);
      if (it_copy.CurrentBlock().splittable) {
        more_recursion = true;
        break;
      }
    }
  }

  if (!more_recursion) {  // Speed optimization.
    if (selected) {
      WP2_CHECK_STATUS(VDebugRecursionLevel(outer_block, recursion_level));
    }
    WP2_CHECK_STATUS(score_func->ComputeScore(sub_blocks, num_sub_blocks,
                                              new_extra_rate, best_score,
                                              selected, score));
    return WP2_STATUS_OK;
  }

  SplitIteratorDefault it_copy;
  WP2_CHECK_STATUS(it_copy.CopyFrom(it));
  it_copy.SplitCurrentBlock(split_idx);

  BlockScoreFunc score_func_copy(/*use_splits=*/true);
  WP2_CHECK_STATUS(score_func_copy.CopyFrom(*score_func));

  float total_score = 0;
  float scores[4];
  for (uint32_t i = 0; i < num_sub_blocks; ++i) {
    const Block& sub_block = sub_blocks[i];
    assert(it_copy.CurrentBlock().block == sub_block);
    const float block_extra_rate =
        new_extra_rate * sub_block.area() / outer_block.area();
    if (it_copy.CurrentBlock().splittable) {
      const uint32_t pattern_idxs_size = best_pattern_idxs->size();
      WP2_CHECK_STATUS(FindBestPattern(
          forced_blocks, it_copy, &score_func_copy, symbol_counter, counter,
          block_extra_rate, recursion_level + 1, selected, best_pattern_idxs,
          &scores[i]));
      assert(best_pattern_idxs->size() > pattern_idxs_size);

      // If this is not the last sub-block, update the score func with the
      // chosen layout.
      if (i < num_sub_blocks - 1) {
        uint32_t j = pattern_idxs_size;
        while (sub_block.rect().Contains(it_copy.CurrentBlock().block.rect())) {
          if (it_copy.CurrentBlock().splittable) {
            assert(j < best_pattern_idxs->size());
            it_copy.SplitCurrentBlock(best_pattern_idxs->at(j));
            ++j;
          } else {
            // Note this re-encodes the blocks which is quite inefficient
            // (looks for the best predictor/transform and so on).
            WP2_CHECK_STATUS(score_func_copy.Use(it_copy.CurrentBlock().block));
            it_copy.NextBlock();
          }
        }
        assert(j == best_pattern_idxs->size());
      }
    } else {
      if (selected) {
        WP2_CHECK_STATUS(VDebugRecursionLevel(outer_block, recursion_level));
      }
      WP2_CHECK_STATUS(score_func_copy.ComputeScore(
          &sub_block, /*num_blocks=*/1, block_extra_rate,
          /*best_score=*/0, selected, &scores[i]));
      if (i < num_sub_blocks - 1) {
        WP2_CHECK_STATUS(score_func_copy.Use(it_copy.CurrentBlock().block));
        it_copy.NextBlock();
      }
    }
    // The scoring function returns an inverted score. Un-inverted to compute
    // th average.
    total_score += (1. / scores[i] - 1.) * sub_block.area();
  }
  // Re-invert the average.
  *score = 1. / (1. + total_score / outer_block.area());
  if (selected) {
    WP2_CHECK_STATUS(RegisterScoreForVDebug(outer_block, recursion_level,
                                            scores, new_extra_rate, *score,
                                            /*is_best=*/(*score > best_score)));
  }
  return WP2_STATUS_OK;
}

WP2Status SplitRecursePartitioner::FindBestPattern(
    const VectorNoCtor<Block>& forced_blocks, const SplitIteratorDefault& it,
    BlockScoreFunc* const score_func, SymbolCounter* const symbol_counter,
    ANSEncCounter* const counter, float extra_rate, uint32_t recursion_level,
    bool selected, Vector_u32* const best_pattern_idxs,
    float* const best_score) const {
  *best_score = 0.f;

  const uint32_t num_patterns = it.NumSplitPatterns();
  uint32_t best_pattern_idx = 0;  // Best pattern out of 'num_patterns'
  // If the best pattern have sub-blocks that are themselves splittable, this
  // list contains the pattern indices for those splits that were found
  // recursively.
  Vector_u32 best_pattern_idxs_tmp;
  for (uint32_t split_idx = 0; split_idx < num_patterns; ++split_idx) {
    Block sub_blocks[4];
    bool splittable[4];
    const uint32_t num_sub_blocks =
        it.GetSplitPatternBlocks(split_idx, sub_blocks, splittable);

    // Skip configurations which don't match the forced blocks.
    const Block& block = it.CurrentBlock().block;
    if (!forced_blocks.empty() &&
        !IsCompatibleWithForcedBlocks(block, sub_blocks, splittable,
                                      num_sub_blocks, forced_blocks)) {
      continue;
    }

    float score;
    Vector_u32 pattern_idxs_tmp;
    WP2_CHECK_STATUS(ComputeScore(
        forced_blocks, it, split_idx, sub_blocks, num_sub_blocks, score_func,
        symbol_counter, counter, *best_score, extra_rate, recursion_level,
        selected, &pattern_idxs_tmp, &score));

    if (split_idx == 0 || score > *best_score) {
      best_pattern_idx = split_idx;
      swap(best_pattern_idxs_tmp, pattern_idxs_tmp);
      *best_score = score;
    }
  }
  // Add the best pattern found.
  WP2_CHECK_ALLOC_OK(best_pattern_idxs->push_back(best_pattern_idx));
  // And the corresponding recursive splits (if any of the sub-blocks for the
  // best pattern are themselves splittable).
  for (uint32_t pattern_idx : best_pattern_idxs_tmp) {
    WP2_CHECK_ALLOC_OK(best_pattern_idxs->push_back(pattern_idx));
  }
  return WP2_STATUS_OK;
}

WP2Status SplitRecursePartitioner::GetBestPartition(
    const ProgressRange& progress, VectorNoCtor<Block>* const blocks,
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
    const SplitIteratorBase::SplitInfo outer_block = it.CurrentBlock();

    // Get all split patterns and keep the one with the best score.
    Vector_u32 best_pattern_idxs;
    if (outer_block.splittable) {
      const bool selected = VDebugBlockSelected(outer_block.block);
      float best_score;
      WP2_CHECK_STATUS(FindBestPattern(
          forced_blocks, it, score_func, &symbol_counter, &counter,
          /*extra_rate=*/0, /*recursion_level=*/0, selected, &best_pattern_idxs,
          &best_score));
      assert(!best_pattern_idxs.empty());
    }

    uint32_t i_split = 0;
    do {
      const SplitIteratorBase::SplitInfo& current_block = it.CurrentBlock();
      if (current_block.splittable) {
        assert(i_split < best_pattern_idxs.size());
        const uint32_t best_pattern_idx = best_pattern_idxs[i_split];
        // Remember the decision.
        WriteBlockSplit(it, best_pattern_idx, &symbol_recorder, &enc_noop);
        it.SplitCurrentBlock(best_pattern_idx);
        if (splits != nullptr) {
          WP2_CHECK_ALLOC_OK(splits->push_back(best_pattern_idx));
        }
        ++i_split;
        WP2_CHECK_STATUS(progress.AdvanceBy(0.));  // Nothing decided.
      } else {
        const Block& block = current_block.block;
        // This 'block' was decided as is with no possibility of recursion.
        assert(IsBlockValid(block, block, forced_blocks.size()));
        WP2_CHECK_REDUCED_STATUS(RegisterOrderForVDebug(
            0, 0, block, blocks->size(), max_num_blocks));
        WP2_CHECK_ALLOC_OK(blocks->push_back(block));
        WP2_CHECK_STATUS(score_func_->Use(block));

        // The following instructions are done only for assertions/debug
        // because the front manager should handle everything, except for
        // the forced blocks but these are checked above.
        assert((Occupy(block), true));
        assert(max_num_blocks_left >= block.w() * block.h());
        max_num_blocks_left -= block.w() * block.h();
        assert(block.dim() != BLK_LAST);

        WP2_CHECK_STATUS(
            progress.AdvanceBy(1. * block.area() / max_num_blocks));
        it.NextBlock();
      }
    } while (outer_block.block.rect().Contains(it.CurrentBlock().block.rect()));
    assert(i_split == best_pattern_idxs.size());
  }
  assert(max_num_blocks_left == 0);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
