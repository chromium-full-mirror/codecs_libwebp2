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

#include "src/enc/partitioning/partition_score_func_block.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "src/common/integral.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/wp2_enc_i.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

BlockScoreFunc::BlockScoreFunc(bool use_splits) : use_splits_(use_splits) {}

WP2Status BlockScoreFunc::Init(const EncoderConfig& config,
                               const Rectangle& tile_rect, const YUVPlane& yuv,
                               const GlobalParams& gparams,
                               const ProgressRange& progress) {
  WP2EncDspInit();
  WP2_CHECK_STATUS(PartitionScoreFunc::Init(config, tile_rect, yuv, gparams,
                                            ProgressRange(progress, 0.5)));

  const ChromaSubsampling chroma_subsampling =
      DecideChromaSubsampling(*config_, /*more_than_one_block=*/true);
  const bool use_aom_coeffs = DecideAOMCoeffs(*config_, tile_rect_);
  WP2_CHECK_STATUS(
      DecideModes(config, gparams, &y_modes_, &uv_modes_, &a_modes_));
  WP2_CHECK_STATUS(scorer_.Init(config, gparams, tile_rect));

  // Store the reconstructed pixels of the temporary and final blocks.
  WP2_CHECK_STATUS(
      buffer_.Resize(src_->Y.w_, src_->Y.h_, /*pad=*/1, with_alpha_));
  WP2_CHECK_STATUS(front_mgr_.InitBase(tile_rect_.width, tile_rect_.height));

  // Call the ProgressHook during the encoding of this block but do not advance
  // progress. The steps woulb be insignificant anyway.
  const ProgressRange sub_progress(progress, 0.);
  // Initialize the instances recording only final blocks.
  WP2_CHECK_STATUS(
      syntax_writer_.Init(&dicts_, *config_, *gparams_, yuv, chroma_subsampling,
                          tile_rect, num_block_cols_ * num_block_rows_,
                          use_aom_coeffs, use_splits_, sub_progress));
  WP2_CHECK_STATUS(syntax_writer_.SetInitialSegmentIds());
  WP2_CHECK_STATUS(syntax_writer_.InitPass());

  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(dc_error_u_.Init(tile_rect_.width));
    WP2_CHECK_STATUS(dc_error_v_.Init(tile_rect_.width));
  }

  // Initialize the cache.
  WP2_CHECK_STATUS(progress.AdvanceBy(0.5));
  return WP2_STATUS_OK;
}

static WP2Status CopyBlockModes(const BlockModes& from, BlockModes* const to) {
  WP2_CHECK_ALLOC_OK(
      to->splits_tried_during_preds.copy_from(from.splits_tried_during_preds));
  WP2_CHECK_ALLOC_OK(
      to->splits_tried_after_preds.copy_from(from.splits_tried_after_preds));
  WP2_CHECK_ALLOC_OK(to->main_preds.copy_from(from.main_preds));
  WP2_CHECK_ALLOC_OK(to->sub_preds.copy_from(from.sub_preds));
  WP2_CHECK_ALLOC_OK(
      to->tf_tried_during_preds.copy_from(from.tf_tried_during_preds));
  WP2_CHECK_ALLOC_OK(
      to->tf_tried_after_preds.copy_from(from.tf_tried_after_preds));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::CopyFrom(const BlockScoreFunc& other) {
  WP2_CHECK_STATUS(PartitionScoreFunc::CopyInternal(other));
  WP2_CHECK_STATUS(CopyBlockModes(other.y_modes_, &y_modes_));
  WP2_CHECK_STATUS(CopyBlockModes(other.uv_modes_, &uv_modes_));
  WP2_CHECK_STATUS(CopyBlockModes(other.a_modes_, &a_modes_));
  use_splits_ = other.use_splits_;
  WP2_CHECK_STATUS(buffer_.Copy(other.buffer_, /*resize_if_needed=*/true));
  WP2_CHECK_STATUS(front_mgr_.CopyInternal(other.front_mgr_));
  WP2_CHECK_STATUS(dicts_.CopyFrom(other.dicts_));
  WP2_CHECK_STATUS(syntax_writer_.CopyFrom(other.syntax_writer_, &dicts_));
  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(dc_error_u_.CopyFrom(other.dc_error_u_));
    WP2_CHECK_STATUS(dc_error_v_.CopyFrom(other.dc_error_v_));
  }
  WP2_CHECK_STATUS(scorer_.Init(*config_, *gparams_, tile_rect_));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::ComputeScore(const Block& block,
                                       const ProgressRange& progress,
                                       float* const score) {
  WP2_CHECK_STATUS(ComputeScore(&block, /*num_blocks=*/1, /*extra_rate=*/0,
                                /*best_score=*/0,
                                /*force_selected=*/false, score));
  WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::ComputeScore(const Block blocks[4],
                                       uint32_t num_blocks, float extra_rate,
                                       float best_score, bool force_selected,
                                       float* const score) {
  // Copy the final state into the temporary scratch one.
  WP2_CHECK_STATUS(tmp_dicts_.CopyFrom(dicts_));
  WP2_CHECK_STATUS(tmp_syntax_writer_.CopyFrom(syntax_writer_, &tmp_dicts_));
  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(tmp_dc_error_u_.CopyFrom(dc_error_u_));
    WP2_CHECK_STATUS(tmp_dc_error_v_.CopyFrom(dc_error_v_));
  }

  float total_rate = extra_rate, total_disto = 0.f, rate[4], disto[4];
  assert(num_blocks >= 1 && num_blocks <= 4);
  uint32_t total_pixels = 0;
  for (uint32_t i = 0; i < num_blocks; ++i) {
    const Block& block = blocks[i];

    // Encode the block.
    tmp_cb_.SetDim(block, front_mgr_);
    WP2_CHECK_STATUS(EncodeBlock(front_mgr_, &tmp_syntax_writer_,
                                 &tmp_dc_error_u_, &tmp_dc_error_v_, &tmp_cb_,
                                 &scorer_, &buffer_));

    // Write the bits.
    ANSEnc enc;
    WP2_CHECK_STATUS(tmp_syntax_writer_.WriteHeader(&enc));
    const float header_rate = enc.GetCost(tmp_dicts_);
    WP2_CHECK_STATUS(
        WriteBlock(front_mgr_, tmp_cb_, &tmp_syntax_writer_, &enc));
    const uint32_t num_pixels = block.AsRect().GetArea();
    total_pixels += num_pixels;
    // Exclude the header to prevent early decision from impacting later blocks.
    const float block_rate = (enc.GetCost(tmp_dicts_) - header_rate);
    assert(block_rate >= 0.f);
    total_rate += block_rate;

    // Compute the distortion per pixel.
    float block_disto = 0.f;
    constexpr float disto_scale[] = {0.4f, 0.2f, 0.2f, 0.2f};
    for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
      if (c == kAChannel && !with_alpha_) continue;
      block_disto += disto_scale[c] * tmp_cb_.GetDisto(c);
    }
    total_disto += block_disto;

    // Per pixel is not necessary, just nicer debug.
    rate[i] = block_rate / num_pixels;
    disto[i] = block_disto / num_pixels;

    // Register the blocks in the 'front_mgr_' except the last one.
    if (i + 1 < num_blocks) {
      front_mgr_.Use(block);
    }
  }

  // Unregister the blocks in the 'front_mgr_' except the last one.
  for (uint32_t i = num_blocks - 1; i-- > 0;) {
    front_mgr_.UndoUse(blocks[i]);
  }

  total_rate /= total_pixels;  // Average per pixel values.
  total_disto /= total_pixels;

  // Estimate a score from the written bits and the distortion.
  const float lambda = MapQuality(*config_, 200.f, 1.f);  // Empirical.
  constexpr float kNiceScale = 0.01f;  // Has no impact on the result.
  *score = 1.0f / (1.0f + kNiceScale * (lambda * total_rate + total_disto));
  WP2_CHECK_REDUCED_STATUS(RegisterScoreForVDebug(
      blocks, num_blocks, rate, disto, extra_rate / total_pixels, total_rate,
      total_disto, *score, /*is_best=*/(*score > best_score), force_selected));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::Use(const Block& block) {
  assert(front_mgr_.GetOccupancy(block.x()) == block.y());
  tmp_cb_.SetDim(block, front_mgr_);
  // Write the final pixels for future context and rate computation.
  WP2_CHECK_STATUS(EncodeBlock(front_mgr_, &syntax_writer_, &dc_error_u_,
                               &dc_error_v_, &tmp_cb_, &scorer_, &buffer_));
  front_mgr_.Use(tmp_cb_.blk());
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::FindBestBlockParams(const FrontMgrBase& front_mgr,
                                              SyntaxWriter* const syntax_writer,
                                              DCDiffusionMap* const dc_error_u,
                                              DCDiffusionMap* const dc_error_v,
                                              CodedBlock* const cb,
                                              BlockScorer* const scorer) const {
  const Rectangle padded_tile_rect = {tile_rect_.x, tile_rect_.y,
                                      src_->GetWidth(), src_->GetHeight()};
  cb->id_ = AssignSegmentId(*config_, *gparams_, padded_tile_rect, cb->blk());
  cb->mtx_set_ = gparams_->use_rnd_mtx_ ? &gparams_->mtx_set_ : nullptr;
  cb->is420_ = false;  // Set in OptimizeModesChroma() but might trigger the
                       // undefined-behavior-sanitizer in OptimizeModesLuma().

  cb->ResetContextCache();
  cb->y_context_is_constant_ = cb->ContextIsConstant(kYChannel);

  WP2_CHECK_STATUS(OptimizeModes(
      *config_, tile_rect_, kYChannel, gparams_->y_preds_, y_modes_,
      syntax_writer->context(), cb, syntax_writer->counters(), scorer));

  WP2_CHECK_STATUS(OptimizeModesChroma(
      *config_, tile_rect_, gparams_->maybe_use_lossy_alpha_, front_mgr,
      gparams_->uv_preds_, syntax_writer->chroma_subsampling(), uv_modes_,
      syntax_writer->context(), cb, syntax_writer->counters(), dc_error_u,
      dc_error_v, scorer));

  if (with_alpha_) {
    WP2_CHECK_STATUS(syntax_writer->DecideAlpha(cb, a_modes_, scorer));
    if (!cb->HasLossyAlpha()) {
      // Consider no loss by copying original samples to the buffer.
      WP2_CHECK_STATUS(cb->out_.A.Copy(cb->in_.A, /*resize_if_needed=*/false));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::EncodeBlock(const FrontMgrBase& front_mgr,
                                      SyntaxWriter* const syntax_writer,
                                      DCDiffusionMap* const dc_error_u,
                                      DCDiffusionMap* const dc_error_v,
                                      CodedBlock* const cb,
                                      BlockScorer* const scorer,
                                      YUVPlane* const buffer) const {
  assert(front_mgr.GetOccupancy(cb->x()) == cb->y());

  cb->SetRange(gparams_->transf_.GetYUVMin(), gparams_->transf_.GetYUVMax());
  cb->SetSrcInput(*src_);
  ContextCache pred_context;
  cb->SetContextInput(buffer_, &pred_context);
  cb->SetReconstructedOutput(buffer);

  // This is the slowest part: finding the best transform, predictor etc.
  WP2_CHECK_STATUS(FindBestBlockParams(front_mgr, syntax_writer, dc_error_u,
                                       dc_error_v, cb, scorer));
  // CodedBlock::Quantize() should be called already.
  WP2_CHECK_STATUS(syntax_writer->FindBestEncodingMethods(cb));
  if (!use_splits_) {
    const Block max_possible_block = front_mgr.LargestPossibleBlockAt(
        cb->x(), cb->y(), config_->partition_snapping);
    WP2_CHECK_STATUS(syntax_writer->RecordSize(
        cb->dim(), max_possible_block.dim(), config_->partition_set));
  }
  WP2_CHECK_STATUS(syntax_writer->Record(*cb));
  if (with_alpha_) {
    WP2_CHECK_STATUS(syntax_writer->RecordAlpha(*cb));
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::WriteBlock(const FrontMgrBase& front_mgr,
                                     const CodedBlock& cb,
                                     SyntaxWriter* const syntax_writer,
                                     ANSEnc* const enc) const {
  assert(front_mgr.GetOccupancy(cb.x()) == cb.y());
  if (!use_splits_) {
    const Block max_possible_block = front_mgr.LargestPossibleBlockAt(
        cb.x(), cb.y(), config_->partition_snapping);
    assert(max_possible_block.rect().Contains(cb.blk().rect()));
    ANSDebugPrefix prefix(enc, "BlockHeader");
    WriteBlockSize(cb.dim(), max_possible_block.dim(), config_->partition_set,
                   syntax_writer->symbol_writer(), enc);
  }
  WP2_CHECK_STATUS(syntax_writer->WriteBlock(cb, /*block_index=*/0, enc));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
