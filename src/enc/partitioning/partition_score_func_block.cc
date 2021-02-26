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
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

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
  WP2_CHECK_STATUS(DecideTransforms(config, &transforms_, &transforms_subset_));

  // Store the reconstructed pixels of the temporary and final blocks.
  WP2_CHECK_STATUS(
      buffer_.Resize(src_->Y.w_, src_->Y.h_, /*pad=*/1, with_alpha_));
  WP2_CHECK_STATUS(front_mgr_.Init(config_->partition_set,
                                   config_->partition_snapping,
                                   tile_rect_.width, tile_rect_.height));

  // Initialize the instances recording only final blocks.
  WP2_CHECK_STATUS(syntax_writer_.Init(
      &dicts_, *config_, *gparams_, yuv, chroma_subsampling, tile_rect,
      num_block_cols_ * num_block_rows_, use_aom_coeffs, /*use_splits=*/false,
      ProgressRange()));
  WP2_CHECK_STATUS(syntax_writer_.SetInitialSegmentIds());
  WP2_CHECK_STATUS(syntax_writer_.InitPass());

  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(dc_error_u_.Init(tile_rect_.width));
    WP2_CHECK_STATUS(dc_error_v_.Init(tile_rect_.width));
  }

  // Initialize the cache.
  WP2_CHECK_STATUS(context_.Init(use_aom_coeffs));
  WP2_CHECK_STATUS(progress.AdvanceBy(0.5));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::ComputeScore(const Block& block,
                                       const ProgressRange& progress,
                                       float* const score) {
  WP2_CHECK_STATUS(ComputeScore(&block, /*num_blocks=*/1, score));
  WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::ComputeScore(const Block blocks[4],
                                       uint32_t num_blocks,
                                       float* const score) {
  // Copy the final state into the temporary scratch one.
  WP2_CHECK_STATUS(tmp_dicts_.CopyFrom(dicts_));
  WP2_CHECK_STATUS(tmp_syntax_writer_.CopyFrom(syntax_writer_, &tmp_dicts_));
  if (DCDiffusionMap::GetDiffusion(config_->error_diffusion) > 0) {
    WP2_CHECK_STATUS(tmp_dc_error_u_.CopyFrom(dc_error_u_));
    WP2_CHECK_STATUS(tmp_dc_error_v_.CopyFrom(dc_error_v_));
  }

  float total_rate = 0.f, total_disto = 0.f, rate[4], disto[4];
  assert(num_blocks >= 1 && num_blocks <= 4);
  uint32_t total_pixels = 0;
  for (uint32_t i = 0; i < num_blocks; ++i) {
    const Block& block = blocks[i];

    // Encode the block.
    Block tmp_blk;
    if (!front_mgr_.SetNextBlockPosition(block.x(), block.y())) assert(false);
    assert(front_mgr_.TryGetNextBlock(block.dim(), &tmp_blk));
    assert(block == tmp_blk);
    tmp_cb_.SetDim(block, front_mgr_);
    WP2_CHECK_STATUS(EncodeBlock(front_mgr_, &tmp_cb_, &tmp_syntax_writer_,
                                 &tmp_dc_error_u_, &tmp_dc_error_v_, &buffer_));

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
      if (!front_mgr_.UseSize(block.dim(), &tmp_blk, /*use_block=*/true)) {
        assert(false);
      }
      assert(block == tmp_blk);
    }
  }

  // Unregister the blocks in the 'front_mgr_' except the last one.
  for (uint32_t i = num_blocks - 1; i-- > 0;) {
    front_mgr_.UndoUse(blocks[i]);
    front_mgr_.UndoUseSize(blocks[i]);
  }

  total_rate /= total_pixels;  // Average per pixel values.
  total_disto /= total_pixels;

  // Estimate a score from the written bits and the distortion.
  const float lambda = MapQuality(*config_, 200.f, -199.f);  // Empirical.
  constexpr float kNiceScale = 0.01f;  // Has no impact on the result.
  *score = 1.0f / (1.0f + kNiceScale * (lambda * total_rate + total_disto));
  WP2_CHECK_REDUCED_STATUS(
    RegisterScoreForVDebug(blocks, num_blocks, rate, disto, total_rate,
                           total_disto, *score));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::Use(const Block& block) {
  if (!front_mgr_.SetNextBlockPosition(block.x(), block.y())) assert(false);
  tmp_cb_.SetDim(block, front_mgr_);
  // Write the final pixels for future context and rate computation.
  WP2_CHECK_STATUS(EncodeBlock(front_mgr_, &tmp_cb_, &syntax_writer_,
                               &dc_error_u_, &dc_error_v_, &buffer_));
  WP2_CHECK_ALLOC_OK(
      front_mgr_.UseSize(tmp_cb_.dim(), /*block=*/nullptr, /*use_block=*/true));
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::FindBestBlockParams(
    const FrontMgrDoubleOrderBase& front_mgr, const BlockContext& block_context,
    SyntaxWriter* const writer, DCDiffusionMap* const dc_error_u,
    DCDiffusionMap* const dc_error_v, CodedBlock* const cb) const {
  const Rectangle padded_tile_rect = {tile_rect_.x, tile_rect_.y,
                                      src_->GetWidth(), src_->GetHeight()};
  cb->id_ = AssignSegmentId(*config_, *gparams_, padded_tile_rect, cb->blk());
  const Segment& segment = gparams_->segments_.at(cb->id_);
  cb->mtx_set_ = gparams_->use_rnd_mtx_ ? &gparams_->mtx_set_ : nullptr;
  cb->is420_ = false;  // Set in OptimizeModesChroma() but might trigger the
                       // undefined-behavior-sanitizer in OptimizeModesLuma().

  cb->ResetContextCache();
  cb->y_context_is_constant_ = cb->ContextIsConstant(kYChannel);

  WP2_CHECK_STATUS(cb->OptimizeModesLuma(
      *config_, tile_rect_, gparams_->maybe_use_lossy_alpha_,
      gparams_->y_preds_, segment, writer->context(), transforms_,
      transforms_subset_, writer->counters()));

  WP2_CHECK_STATUS(cb->OptimizeModesChroma(
      *config_, tile_rect_, gparams_->maybe_use_lossy_alpha_, front_mgr,
      gparams_->uv_preds_, segment, writer->context(),
      writer->chroma_subsampling(), dc_error_u, dc_error_v,
      writer->counters()));

  if (with_alpha_) {
    WP2_CHECK_STATUS(writer->DecideAlpha(cb));
    if (!cb->HasLossyAlpha()) {
      // Consider no loss by copying original samples to the buffer.
      WP2_CHECK_STATUS(cb->out_.A.Copy(cb->in_.A, /*resize_if_needed=*/false));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::EncodeBlock(const FrontMgrDoubleOrderBase& front_mgr,
                                      CodedBlock* const cb,
                                      SyntaxWriter* const syntax_writer,
                                      DCDiffusionMap* const dc_error_u,
                                      DCDiffusionMap* const dc_error_v,
                                      YUVPlane* const buffer) const {
  cb->SetRange(gparams_->transf_.GetYUVMin(), gparams_->transf_.GetYUVMax());
  cb->SetSrcInput(*src_);
  ContextCache pred_context;
  cb->SetContextInput(buffer_, &pred_context);
  cb->SetReconstructedOutput(buffer);

  // This is the slowest part: finding the best transform, predictor etc.
  WP2_CHECK_STATUS(FindBestBlockParams(front_mgr, context_, syntax_writer,
                                       dc_error_u, dc_error_v, cb));
  // CodedBlock::Quantize() should be called already.
  WP2_CHECK_STATUS(syntax_writer->FindBestEncodingMethods(cb));
  WP2_CHECK_STATUS(syntax_writer->RecordSize(front_mgr, cb->dim()));
  WP2_CHECK_STATUS(syntax_writer->Record(*cb));
  if (with_alpha_) {
    WP2_CHECK_STATUS(syntax_writer->RecordAlpha(*cb));
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::WriteBlock(const FrontMgrDoubleOrderBase& front_mgr,
                                     const CodedBlock& cb,
                                     SyntaxWriter* const syntax_writer,
                                     ANSEnc* const enc) const {
  assert(front_mgr.GetMaxFittingBlock().x() == cb.x() &&
         front_mgr.GetMaxFittingBlock().y() == cb.y());
  assert(front_mgr.GetMaxPossibleBlock().rect().Contains(cb.blk().rect()));
  {
    ANSDebugPrefix prefix(enc, "BlockHeader");
    WriteBlockSize(front_mgr, cb.dim(), syntax_writer->symbol_writer(), enc);
  }
  WP2_CHECK_STATUS(syntax_writer->WriteBlock(cb, /*block_index=*/0, enc));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
