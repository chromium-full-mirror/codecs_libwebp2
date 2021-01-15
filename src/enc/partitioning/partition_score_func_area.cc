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
// Block position/size scoring functions.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/enc/partitioning/partition_score_func_area.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "src/common/integral.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partition_score_func_multi.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/partitioning/partitioner_multi.h"
#include "src/enc/wp2_enc_i.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

AreaScoreFunc::AreaScoreFunc(uint32_t area_width, uint32_t area_height)
    : area_width_(area_width),
      area_height_(area_height),
      area_front_mgr_(area_width, area_height),
      comp_(kMaxTileSize / kMinBlockSizePix, area_width / kMinBlockSizePix,
            area_height / kMinBlockSizePix) {}

WP2Status AreaScoreFunc::Init(const EncoderConfig& config,
                              const Rectangle& tile_rect, const YUVPlane& yuv,
                              const GlobalParams& gparams,
                              const ProgressRange& progress) {
  const ProgressRange init_progress(progress, 0.1);
  const ProgressRange score_func_init_progress(progress, 0.1);
  const ProgressRange default_partition_progress(progress, 0.8);
  WP2_CHECK_STATUS(
      PartitionScoreFunc::Init(config, tile_rect, yuv, gparams, init_progress));

  // This scoring function needs a strict block layout.
  const BlockSize max_block_size =
      GetSmallestBounds(config.partition_set, BLK_32x32);
  WP2_CHECK_OK(config.partition_snapping, WP2_STATUS_INVALID_CONFIGURATION);
  WP2_CHECK_OK(BlockWidthPix(max_block_size) <= area_width_ &&
                   BlockHeightPix(max_block_size) <= area_height_,
               WP2_STATUS_INVALID_CONFIGURATION);

  // Matches TileEncoder::LossyEncode() behavior.
  const ChromaSubsampling chroma_subsampling =
      DecideChromaSubsampling(*config_, /*more_than_one_block=*/true);
  const bool use_aom_coeffs = DecideAOMCoeffs(*config_, tile_rect_);
  WP2_CHECK_STATUS(DecideTransforms(config, &transforms_, &transforms_subset_));

  // Retrieve the default partition (multipass) to have a reference.
  {
    MultiScoreFunc score_func;
    MultiPassPartitioner partitioner(&score_func);

    EncoderConfig cfg_no_dbg = config;
    cfg_no_dbg.info = nullptr;  // Do not use or output any debugging data.
    WP2_CHECK_STATUS(score_func.Init(cfg_no_dbg, tile_rect_, *src_, *gparams_,
                                     score_func_init_progress));
    WP2_CHECK_STATUS(
        partitioner.Init(cfg_no_dbg, *src_, tile_rect_, &score_func));
    // TODO(yguyon): Add forced blocks from 'config' to 'default_partition_'
    WP2_CHECK_STATUS(partitioner.GetBestPartition(default_partition_progress,
                                                  &default_partition_));
    // Sort the blocks to find those in a given area faster.
    std::sort(default_partition_.begin(), default_partition_.end(), comp_);
  }

  // Initialize the instances recording only final blocks.
  WP2_CHECK_STATUS(syntax_writer_.Init(
      &dicts_, *config_, *gparams_, yuv, chroma_subsampling, tile_rect,
      num_block_cols_ * num_block_rows_, use_aom_coeffs, ProgressRange()));
  WP2_CHECK_STATUS(syntax_writer_.SetInitialSegmentIds());
  WP2_CHECK_STATUS(syntax_writer_.InitPass());

  WP2_CHECK_STATUS(context_.Init(use_aom_coeffs, yuv.Y.w_, yuv.Y.h_));

  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(dc_error_u_.Init(tile_rect_.width));
    WP2_CHECK_STATUS(dc_error_v_.Init(tile_rect_.width));
  }

  // Store the reconstructed pixels of the current area partitioning and of
  // the final selected blocks.
  WP2_CHECK_STATUS(
      buffer_.Resize(src_->Y.w_, src_->Y.h_, /*pad=*/1, with_alpha_));

  // Setup the first area as the top-left corner.
  WP2_CHECK_STATUS(area_front_mgr_.Init(config_->partition_set,
                                        config_->partition_snapping,
                                        tile_rect_.width, tile_rect_.height));
  WP2_CHECK_STATUS(BeginArea(/*area_x=*/0, /*area_y=*/0));  // x,y within tile
  return WP2_STATUS_OK;
}

// Returns a score in [0:1] representing how much a 'disto/rate' pair is good
// compared to a reference, where 0 is discarded, 1 is way better than 'ref' and
// 1/3 is equivalent to 'ref'.
static float GetRelativeScore(const EncoderConfig& config, float ref_disto,
                              float ref_rate, float disto, float rate) {
  if (disto > ref_disto || rate > ref_rate) return 0.f;  // No mercy.
  if (ref_disto > 0.f) {
    disto /= ref_disto;  // Normalize.
  } else  {
    if (disto > 0.f) return 0.f;  // Discard, no way of assessing 'disto'/'ref'.
    disto = 1.f;  // Both values are 0 so set to 1 to signal it is equivalent.
  }
  if (ref_rate > 0.f) {
    rate /= ref_rate;  // Normalize.
  } else  {
    if (rate > 0.f) return 0.f;  // Discard, no way of assessing 'rate'/'ref'.
    rate = 1.f;  // Both values are 0 so set to 1 to signal it is equivalent.
  }
  const float delta = Clamp(config.quality / kMaxLossyQuality, 0.01f, 0.99f);
  return 1.f / (1.f + delta * disto + (1.f - delta) * rate);
}

WP2Status AreaScoreFunc::ComputeScore(const Block&, const ProgressRange&,
                                      float* const) {
  return WP2_STATUS_UNSUPPORTED_FEATURE;  // Unused.
}

WP2Status AreaScoreFunc::GetAreaDefaultScore(VectorNoCtor<Block>* const blocks,
                                             float* const score) {
  Vector<CodedBlock> partition;
  WP2_CHECK_STATUS(GetAreaDefaultPartition(&partition));
  assert(!partition.empty());
  WP2_CHECK_STATUS(GetDistoRate(&partition, &default_disto_, &default_rate_));
  // Compute the score with itself as reference, to be easily comparable.
  *score = GetRelativeScore(*config_, default_disto_, default_rate_,
                            default_disto_, default_rate_);
  for (const CodedBlock& cb : partition) {
    WP2_CHECK_ALLOC_OK(blocks->push_back(cb.blk()));
  }

  WP2_CHECK_REDUCED_STATUS(
    RegisterScoreForVDebug(BLK_LAST, partition, *score, default_disto_,
                           default_rate_));
  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::GetAreaGridScore(BlockSize block_size,
                                          VectorNoCtor<Block>* const blocks,
                                          float* const score) {
  Vector<CodedBlock> partition;
  // Fill 'partition' with as many 'block_size' as possible.
  // Fill the remaining space with blocks as big as possible.
  FrontMgrArea front_mgr(area_width_, area_height_);
  WP2_CHECK_STATUS(front_mgr.CopyFrom(area_front_mgr_));
  uint32_t num_block_units = 0;
  while (!front_mgr.Done()) {
    Block block;
    if (!front_mgr.TryGetNextBlock(block_size, &block)) {
      block = front_mgr.GetMaxFittingBlock();
    }
    if (!area_.Contains(block.x_pix(), block.y_pix())) break;
    WP2_CHECK_ALLOC_OK(front_mgr.UseSize(block.dim(), 0, nullptr));
    front_mgr.Use(block);

    WP2_CHECK_ALLOC_OK(partition.resize(partition.size() + 1));
    partition.back().SetDimDefault(block);
    WP2_CHECK_ALLOC_OK(blocks->push_back(block));
    num_block_units += block.rect().GetArea();
  }
  assert(num_block_units == SizeBlocks(area_.width) * SizeBlocks(area_.height));

  assert(default_disto_ >= 0.f && default_rate_ >= 0.f);
  float disto = 0.f, rate = 0.f;
  WP2_CHECK_STATUS(GetDistoRate(&partition, &disto, &rate));
  *score =
      GetRelativeScore(*config_, default_disto_, default_rate_, disto, rate);

  WP2_CHECK_REDUCED_STATUS(
      RegisterScoreForVDebug(block_size, partition, *score, disto, rate));
  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::GetDistoRate(Vector<CodedBlock>* const area_blocks,
                                      float* const disto,
                                      float* const rate) const {
  *rate = *disto = 0.f;

  ANSDictionaries dicts;
  WP2_CHECK_STATUS(dicts.CopyFrom(dicts_));
  SyntaxWriter syntax_writer;
  WP2_CHECK_STATUS(syntax_writer.CopyFrom(syntax_writer_, &dicts));

  DCDiffusionMap dc_error_u, dc_error_v;
  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(dc_error_u.CopyFrom(dc_error_u_));
    WP2_CHECK_STATUS(dc_error_v.CopyFrom(dc_error_v_));
  }

  // Encode all 'area_blocks' using previously finished areas and blocks in this
  // 'area_' as prediction context (stored in 'buffer_').
  {
    FrontMgrArea front_mgr(area_width_, area_height_);
    WP2_CHECK_STATUS(front_mgr.CopyFrom(area_front_mgr_));
    for (CodedBlock& cb : *area_blocks) {
      WP2_CHECK_OK(!front_mgr.Done(), WP2_STATUS_INVALID_PARAMETER);
      assert(cb.x() == front_mgr.GetMaxPossibleBlock().x() &&
             cb.y() == front_mgr.GetMaxPossibleBlock().y() &&
             front_mgr.GetMaxPossibleBlock().rect().Contains(cb.blk().rect()));
      cb.SetDim(cb.blk(), front_mgr);
      WP2_CHECK_STATUS(EncodeBlock(front_mgr, &cb, &syntax_writer, &dc_error_u,
                                   &dc_error_v, &buffer_));
      WP2_CHECK_ALLOC_OK(front_mgr.UseSize(cb.dim(),
                                           /*ind=*/0, /*block=*/nullptr));
      front_mgr.Use(cb.blk());
    }
  }

  // Now estimate the bits necessary to encode all blocks in 'area_'.
  {
    // Also write the headers so that symbols are correctly set up.
    ANSEnc enc;
    WP2_CHECK_STATUS(syntax_writer.WriteHeader(&enc));
    const float header_rate = enc.GetCost(dicts);
    FrontMgrArea front_mgr(area_width_, area_height_);
    WP2_CHECK_STATUS(front_mgr.CopyFrom(area_front_mgr_));

    for (const CodedBlock& cb : *area_blocks) {
      WP2_CHECK_STATUS(WriteBlock(front_mgr, cb, &syntax_writer, &enc));
      WP2_CHECK_ALLOC_OK(front_mgr.UseSize(cb.dim(),
                                           /*ind=*/0, /*block=*/nullptr));
      front_mgr.Use(cb.blk());
    }
    // Exclude the header to prevent early decisions from impacting later areas.
    *rate = (enc.GetCost(dicts) - header_rate) / area_.GetArea();
    assert(*rate >= 0.f);
  }

  // Measure the distortion of this tested partition of 'area_'.
  {
    constexpr float disto_scale[] = {0.4f, 0.2f, 0.2f, 0.2f};
    for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
      if (c == kAChannel && !with_alpha_) continue;
      Plane16 src_area_view, buffer_area_view;
      WP2_CHECK_STATUS(src_area_view.SetView(src_->GetChannel(c), area_));
      WP2_CHECK_STATUS(buffer_area_view.SetView(buffer_.GetChannel(c), area_));
      *disto +=
          disto_scale[c] * WP2SumSquaredErrorBlock(
                               src_area_view.Row(0), src_area_view.Step(),
                               buffer_area_view.Row(0), buffer_area_view.Step(),
                               area_.width, area_.height);
    }
    *disto /= area_.GetArea();
  }
  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::BeginArea(uint32_t area_x, uint32_t area_y) {
  // Make sure areas are done in order.
  assert((area_x == 0 && area_y == 0) ||
         (area_x == area_.x + area_width_ && area_y == area_.y) ||
         (area_x == 0 && area_y == area_.y + area_height_));
  area_ = Rectangle(area_x, area_y, area_width_, area_height_)
              .ClipWith({0, 0, tile_rect_.width, tile_rect_.height});  // no pad
  // Assuming all areas are done in order row by row, top and left contexts
  // outside this 'area_' are available, if any.

  // Next available block should match the current 'area_'.
  assert(area_front_mgr_.GetMaxFittingBlock().x_pix() == area_.x &&
         area_front_mgr_.GetMaxFittingBlock().y_pix() == area_.y);

  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::Use(const Block& block) {
  CodedBlock cb;
  cb.SetDim(block, area_front_mgr_);
  assert(cb.blk() == block);
  // Write the final pixels for future context and rate computation.
  WP2_CHECK_STATUS(EncodeBlock(area_front_mgr_, &cb, &syntax_writer_,
                               &dc_error_u_, &dc_error_v_, &buffer_));
  WP2_CHECK_ALLOC_OK(area_front_mgr_.UseSize(cb.dim(), /*ind=*/0,
                                             /*block=*/nullptr));
  area_front_mgr_.Use(cb.blk());
  if (area_front_mgr_.Done()) {
    // All areas are complete.
    area_ = {};
    area_front_mgr_.Clear();
  } else {
    const Block max_block = area_front_mgr_.GetMaxPossibleBlock();
    if (!area_.Contains(max_block.x_pix(), max_block.y_pix())) {
      // This 'area_' is complete. Prepare the next one.
      uint32_t area_x = area_.x + area_width_, area_y = area_.y;  // Next col.
      if (area_x >= tile_rect_.width) {                           // Next row.
        area_x = 0;
        area_y += area_height_;
      }
      assert(area_x < tile_rect_.width && area_y < tile_rect_.height);
      WP2_CHECK_STATUS(BeginArea(area_x, area_y));
    }
  }
  default_disto_ = default_rate_ = -1;
  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::GetAreaDefaultPartition(
    Vector<CodedBlock>* const area_blocks) const {
  // Find the blocks in 'default_partition_' belonging to the current 'area_'.
  const Block area_pos(area_.x / kMinBlockSizePix, area_.y / kMinBlockSizePix,
                       BLK_32x32);
  VectorNoCtor<Block>::const_iterator block_it = std::lower_bound(
      default_partition_.begin(), default_partition_.end(), area_pos, comp_);

  uint32_t num_block_units = 0;
  for (; block_it != default_partition_.end(); ++block_it) {
    if (!area_.Contains(block_it->x_pix(), block_it->y_pix())) break;
    WP2_CHECK_ALLOC_OK(area_blocks->resize(area_blocks->size() + 1));
    area_blocks->back().SetDimDefault(*block_it);
    num_block_units += block_it->rect().GetArea();
  }
  // If snapping is not enabled or if the 'area_' dimensions do not match it,
  // the default partition cannot be used as is.
  WP2_CHECK_OK(
      num_block_units == SizeBlocks(area_.width) * SizeBlocks(area_.height),
      WP2_STATUS_INVALID_CONFIGURATION);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status SubAreaScoreFunc::ComputeScore(const Block& block,
                                         const ProgressRange& progress,
                                         float* const score) {
  const ProgressRange default_partitioning_progress(progress, 0.5);
  const ProgressRange remaining_blocks_progress(progress, 0.5);
  if (default_block_.dim() == BLK_LAST) {
    // Get the default partitioning and score of the remaining non-final blocks.
    // This is done once for each block position within each area.
    Vector<CodedBlock> default_partition;
    WP2_CHECK_STATUS(GetAreaRemainingDefaultPartition(
        default_partitioning_progress, &default_partition));
    assert(!default_partition.empty() &&
           default_partition.front().dim() != BLK_LAST);
    // TODO(yguyon): Also include the 'area_used_blocks_' into the disto/rate
    WP2_CHECK_STATUS(GetDistoRate(&default_partition, &default_block_disto_,
                                  &default_block_rate_));
#if !defined(WP2_REDUCE_BINARY_SIZE)
    const float default_score =
        GetRelativeScore(*config_, default_block_disto_, default_block_rate_,
                         default_block_disto_, default_block_rate_);
    WP2_CHECK_REDUCED_STATUS(
        RegisterScoreForVDebug(default_partition.front().blk(),
                               default_partition, default_score,
                               default_block_disto_, default_block_rate_));
#endif  // WP2_REDUCE_BINARY_SIZE
    default_block_ = default_partition.front().blk();
  } else {
    WP2_CHECK_STATUS(default_partitioning_progress.AdvanceBy(1.));
  }
  // 'default_block_' now contains the size of the first block among the
  // remaining non-final ones given by the default partitioning.
  assert(default_block_.x() == block.x() && default_block_.y() == block.y());

  float disto = 0.f, rate = 0.f;
  if (block == default_block_) {
    disto = default_block_disto_;
    rate = default_block_rate_;
    *score = GetRelativeScore(*config_, default_block_disto_,
                              default_block_rate_, disto, rate);
    WP2_CHECK_STATUS(remaining_blocks_progress.AdvanceBy(1.));
  } else {
    // 'block' here is the currently evaluated size for a given position.
    // 'area_used_blocks_' represent the final blocks previously encoded and
    // recorded. 'area_remaining_blocks' exist to fill the 'area_' partition and
    // compare the same surface by rate-distortion with the default partition.
    Vector<CodedBlock> area_remaining_blocks;
    WP2_CHECK_ALLOC_OK(area_remaining_blocks.resize(1));
    area_remaining_blocks.back().SetDimDefault(block);  // Force it.
    WP2_CHECK_STATUS(GetAreaRemainingDefaultPartition(remaining_blocks_progress,
                                                      &area_remaining_blocks));

    WP2_CHECK_STATUS(GetDistoRate(&area_remaining_blocks, &disto, &rate));
    *score = GetRelativeScore(*config_, default_block_disto_,
                              default_block_rate_, disto, rate);
    WP2_CHECK_REDUCED_STATUS(
        RegisterScoreForVDebug(block, area_remaining_blocks,
                               *score, disto, rate));
  }
  return WP2_STATUS_OK;
}

WP2Status SubAreaScoreFunc::BeginArea(uint32_t area_x, uint32_t area_y) {
  WP2_CHECK_STATUS(AreaScoreFunc::BeginArea(area_x, area_y));
  area_used_blocks_.clear();
  return WP2_STATUS_OK;
}

WP2Status SubAreaScoreFunc::Use(const Block& block) {
  WP2_CHECK_ALLOC_OK(area_used_blocks_.resize(area_used_blocks_.size() + 1));
  area_used_blocks_.back().SetDim(block, area_front_mgr_);
  WP2_CHECK_STATUS(AreaScoreFunc::Use(block));
  default_block_ = Block();  // Reset.
  default_block_disto_ = default_block_rate_ = 0.f;
  return WP2_STATUS_OK;
}

WP2Status SubAreaScoreFunc::GetAreaRemainingDefaultPartition(
    const ProgressRange& progress,
    Vector<CodedBlock>* const area_remaining_blocks) const {
  const ProgressRange score_func_init_progress(progress, 0.01);
  const ProgressRange partitioner_progress(progress, 0.99);

  // It would be faster to reuse the 'AreaScoreFunc::default_partition_' (or
  // part of it) but it is not always compatible with the already selected
  // 'area_used_blocks_' so for simplicity it is always recomputed for each
  // block size of each block position.
  MultiScoreFunc score_func;
  MultiPassPartitioner partitioner(&score_func);

  EncoderConfig config = *config_;
  config.info = nullptr;  // Remove any debugging input/output.
  WP2_CHECK_STATUS(score_func.Init(config, tile_rect_, *src_, *gparams_,
                                   score_func_init_progress));
  WP2_CHECK_STATUS(partitioner.Init(config, *src_, tile_rect_, &score_func));

  // 'blocks' will first contain all irrelevant blocks that will be forced into
  // the 'partitioner' to extract only the interesting ones.
  VectorNoCtor<Block> blocks;
  // Force all blocks external to the current 'area_' (their size and count do
  // not matter).
  {
    FrontMgrArea front_mgr(area_width_, area_height_);
    WP2_CHECK_STATUS(front_mgr.Init(config_->partition_set,
                                    config_->partition_snapping,
                                    tile_rect_.width, tile_rect_.height));
    uint32_t surface_kept = 0;
    while (!front_mgr.Done()) {
      const Block max_block = front_mgr.GetMaxFittingBlock();
      WP2_CHECK_ALLOC_OK(front_mgr.UseSize(max_block.dim(), 0, nullptr));
      front_mgr.Use(max_block);
      if (!area_.Contains(max_block.x_pix(), max_block.y_pix())) {
        WP2_CHECK_ALLOC_OK(blocks.push_back(max_block));
      } else {
        surface_kept += max_block.rect().GetArea();
      }
    }
    WP2_CHECK_OK(
        surface_kept == SizeBlocks(area_.width) * SizeBlocks(area_.height),
        WP2_STATUS_INVALID_PARAMETER);
  }
  // Force all already selected final blocks.
  for (const CodedBlock& cb : area_used_blocks_) {
    WP2_CHECK_ALLOC_OK(blocks.push_back(cb.blk()));
  }
  // Force the block under review, if any.
  for (const CodedBlock& cb : *area_remaining_blocks) {
    WP2_CHECK_ALLOC_OK(blocks.push_back(cb.blk()));
  }
  const uint32_t num_default_blocks_to_ignore = blocks.size();
  const uint32_t num_already_added_area_blocks = area_remaining_blocks->size();

  // Get the default partitioning of the remaining empty spaces into 'blocks'.
  WP2_CHECK_STATUS(partitioner.GetBestPartition(partitioner_progress, &blocks));
  // Sort in lexico order only the new blocks.
  std::sort(blocks.begin() + num_default_blocks_to_ignore, blocks.end());

  // Copy from Block struct to CodedBlock class into 'area_remaining_blocks'.
  WP2_CHECK_ALLOC_OK(area_remaining_blocks->resize(
      area_remaining_blocks->size() +
      (blocks.size() - num_default_blocks_to_ignore)));
  for (uint32_t i = num_default_blocks_to_ignore,
                j = num_already_added_area_blocks;
       i < blocks.size(); ++i, ++j) {
    area_remaining_blocks->at(j).SetDimDefault(blocks[i]);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
