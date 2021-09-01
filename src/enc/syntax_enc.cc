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
// WP2 lossy encoding.
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

#include "src/common/lossy/transforms.h"
#include "src/enc/analysis.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

WP2Status SyntaxWriter::Init(ANSDictionaries* const dicts,
                             const EncoderConfig& config,
                             const GlobalParams& gparams, const YUVPlane& yuv,
                             ChromaSubsampling chroma_subsampling,
                             const Rectangle& tile_rect, uint32_t num_blocks,
                             bool use_aom_coeffs, bool use_splits,
                             const ProgressRange& alpha_progress) {
  config_ = &config;
  dicts_ = dicts;
  recorded_blocks_ = 0;
  tile_rect_ = tile_rect;
  gparams_ = &gparams;
  assert(gparams_->IsOk());
  chroma_subsampling_ = chroma_subsampling;
  use_splits_ = use_splits;

  num_blocks_ = num_blocks;
  // Number of transforms is unknown yet, use a higher bound.
  num_transforms_ = std::min(
      num_blocks * 4, SizeBlocks(yuv.GetWidth()) * SizeBlocks(yuv.GetHeight()));
  // Deal with context.
  WP2_CHECK_STATUS(context_.Init(use_aom_coeffs, gparams_->segments_.size(),
                                 gparams_->explicit_segment_ids_,
                                 tile_rect.width));

  WP2_CHECK_STATUS(residual_writer_.Init(context_.use_aom(),
                                         gparams_->maybe_use_lossy_alpha_));


  if (gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_writer_.Init(*config_, gparams, context_, yuv,
                                        tile_rect, alpha_progress));
  } else {
    WP2_CHECK_STATUS(alpha_progress.AdvanceBy(1.));
  }

  if (kDebugPrintRate) {
    WP2_CHECK_ALLOC_OK(residual_rate_.resize(num_blocks));
  }

  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::CopyFrom(const SyntaxWriter& other,
                                 ANSDictionaries* const dicts) {
  config_ = other.config_;

  WP2_CHECK_OK(dicts != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(dicts != other.dicts_, WP2_STATUS_INVALID_PARAMETER);
  dicts_ = dicts;

  WP2_CHECK_STATUS(context_.CopyFrom(other.context_));
  if (other.gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_writer_.CopyFrom(other.alpha_writer_, context_));
  }

  num_blocks_ = other.num_blocks_;
  num_transforms_ = other.num_transforms_;
  recorded_blocks_ = other.recorded_blocks_;
  tile_rect_ = other.tile_rect_;
  chroma_subsampling_ = other.chroma_subsampling_;
  gparams_ = other.gparams_;
  use_splits_ = other.use_splits_;

  WP2_CHECK_STATUS(
      symbol_writer_.CopyFrom(other.symbol_writer_, *dicts_, *other.dicts_));
  WP2_CHECK_STATUS(symbol_recorder_.CopyFrom(other.symbol_recorder_));
  WP2_CHECK_STATUS(counters_.CopyFrom(other.counters_, symbol_recorder_));
  residual_writer_.CopyFrom(other.residual_writer_);

  WP2_CHECK_ALLOC_OK(residual_rate_.copy_from(other.residual_rate_));
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::InitPass() {
  SymbolsInfo symbols_info;
  WP2_CHECK_STATUS(symbols_info.InitLossy(
      gparams_->segments_.size(), gparams_->partition_set_,
      gparams_->maybe_use_lossy_alpha_, context_.use_aom(), use_splits_));
  symbols_info.SetMinMax(kSymbolModeY, /*min=*/0,
                         /*max=*/gparams_->y_preds_.GetMaxMode());
  if (gparams_->maybe_use_lossy_alpha_) {
    symbols_info.SetMinMax(kSymbolModeA, /*min=*/0,
                           /*max=*/gparams_->a_preds_.GetMaxMode());
  }
  symbols_info.SetMinMax(kSymbolModeUV, /*min=*/0,
                         /*max=*/gparams_->uv_preds_.GetMaxMode());
  const uint32_t num_channels = (gparams_->maybe_use_lossy_alpha_ ? 4 : 3);
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !gparams_->maybe_use_lossy_alpha_) continue;
    WP2_CHECK_STATUS(symbols_info.SetMinMax(
        kSymbolDC, ResidualIO::GetCluster(c, num_channels), /*min=*/0,
        /*max=*/gparams_->GetMaxDCRange(c)));
  }
  symbols_info.SetClusters(kSymbolDC, symbols_info.NumClusters(kSymbolDC));

  WP2_CHECK_STATUS(symbol_writer_.Init(symbols_info, config_->effort));
  WP2_CHECK_STATUS(symbol_writer_.Allocate());
  if (pass_number_ == 0) {
    WP2_CHECK_STATUS(symbol_recorder_.Allocate(
        symbols_info, /*num_records=*/num_blocks_ * kMaxBlockSize2));
  } else {
    WP2_CHECK_STATUS(symbol_recorder_.MakeBackup());
  }
  ++pass_number_;
  WP2_CHECK_STATUS(
      counters_.Init(config_->effort, context_.use_aom(), symbol_recorder_));
  context_.Reset();
  if (recorded_blocks_ > 0) WP2_CHECK_STATUS(ResetRecord());
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::WriteHeader(ANSEncBase* const enc) {
  const uint32_t max_num_blocks =
      SizeBlocks(tile_rect_.width) * SizeBlocks(tile_rect_.height);
  // The minimum number of blocks is not accurate but it is not worth it to have
  // a more complex solution (neglictible pessimization).
  const uint32_t min_num_blocks = std::max(1u, max_num_blocks / kMaxBlockSize2);
  // This is also used for partitioning with an upper bound of 'num_blocks_'.
  assert((num_blocks_ == max_num_blocks) || (recorded_blocks_ == num_blocks_));
  ANSDebugPrefix prefix(enc, "GlobalHeader");

  // TODO(skal): differential-write of gparams_

  enc->PutBool(context_.use_aom(), "use_aom_coeffs");
  if (gparams_->partition_snapping_) {
    enc->PutBool(use_splits_, "use_splits");
  } else {
    assert(!use_splits_);
  }

  // Write dictionaries.
  PutLargeRange(num_blocks_, min_num_blocks, max_num_blocks, enc, "num_blocks");

  // If there's only one block, we don't need this global value, we'll signal
  // the subsampling mode for that block later.
  if (chroma_subsampling_ != ChromaSubsampling::kSingleBlock) {
    enc->PutRValue((int)chroma_subsampling_,
                   (int)ChromaSubsampling::kSingleBlock, "chroma_subsampling");
  }

  if (gparams_->explicit_segment_ids_) {
    WP2_CHECK_STATUS(symbol_writer_.WriteHeader(kSymbolSegmentId, num_blocks_,
                                                symbol_recorder_, "segment_id",
                                                enc, dicts_));
  }

  if (gparams_->use_rnd_mtx_) {
    WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
        kSymbolRndMtx0, num_blocks_, symbol_recorder_, "rnd_mtx", enc, dicts_));
    WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
        kSymbolUseRandomMatrix, num_blocks_, symbol_recorder_, "use_rnd_mtx",
        enc, dicts_));
  }
  WP2_CHECK_STATUS(
      symbol_writer_.WriteHeader(kSymbolSplitTransform, num_blocks_,
                                 symbol_recorder_, "split_tf", enc, dicts_));
  if (chroma_subsampling_ == ChromaSubsampling::kSingleBlock ||
      chroma_subsampling_ == ChromaSubsampling::kAdaptive) {
    WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
        kSymbolYuv420, num_blocks_, symbol_recorder_, "yuv420", enc, dicts_));
  }
  WP2_CHECK_STATUS(symbol_writer_.WriteHeader(kSymbolTransform, num_blocks_,
                                              symbol_recorder_, "transform",
                                              enc, dicts_));

  WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
      kSymbolModeY, num_blocks_, symbol_recorder_, "mode", enc, dicts_));
  WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
      kSymbolModeUV, num_blocks_, symbol_recorder_, "mode", enc, dicts_));

  // TODO(maryla): not necessary if the SignalingCflPredictor isn't used.
  const uint32_t num_channels_using_cfl =
      gparams_->maybe_use_lossy_alpha_ ? 3 : 2;
  WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
      kSymbolCflSlope, num_channels_using_cfl * num_blocks_, symbol_recorder_,
      "cfl", enc, dicts_));

  WP2_CHECK_STATUS(symbol_writer_.WriteHeader(kSymbolBlockSize, num_blocks_,
                                              symbol_recorder_, "block_size",
                                              enc, dicts_));

  const uint32_t num_coeffs_max =
      max_num_blocks * kMinBlockSizePix * kMinBlockSizePix;
  const uint32_t num_coeffs_max_uv =
      (chroma_subsampling_ == ChromaSubsampling::k420) ? num_coeffs_max / 4
                                                       : num_coeffs_max;
  // We should always have a UV block.
  assert(num_coeffs_max_uv > 0);

  WP2_CHECK_STATUS(context_.segment_id_predictor().WriteHeader(enc));

  if (gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_writer_.WriteHeader(num_coeffs_max, enc));

    if (alpha_writer_.GetAlphaMode() == kAlphaModeLossy) {
      WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
          kSymbolBlockAlphaMode, num_blocks_, symbol_recorder_,
          "block_alpha_mode", enc, dicts_));
    }
    if (alpha_writer_.GetAlphaMode() != kAlphaModeLossless) {
      WP2_CHECK_STATUS(symbol_writer_.WriteHeader(
          kSymbolModeA, num_blocks_, symbol_recorder_, "mode", enc, dicts_));
    }
  }

  WP2_CHECK_STATUS(residual_writer_.WriteHeader(
      num_coeffs_max, num_coeffs_max_uv, num_transforms_,
      gparams_->has_alpha_ &&
          (alpha_writer_.GetAlphaMode() != kAlphaModeLossless),
      symbol_recorder_, dicts_, enc, &symbol_writer_));

  return enc->GetStatus();
}

void SyntaxWriter::WriteBlockBeforeCoeffs(const CodedBlock& cb,
                                          SymbolManager* const sm,
                                          ANSEncBase* const enc) {
  context_.segment_id_predictor().WriteId(cb, sm, enc);

  if (gparams_->has_alpha_) {
    alpha_writer_.WriteBlockBeforeCoeffs(cb, sm, enc);
  } else if (!gparams_->maybe_use_lossy_alpha_) {
    assert(!cb.HasLossyAlpha());
  }

  for (Channel channel : {kYChannel, kAChannel, kUChannel, kVChannel}) {
    if (channel == kAChannel && !cb.HasLossyAlpha()) continue;
    WritePredictors(cb, channel, sm, enc);
  }

  if (chroma_subsampling_ == ChromaSubsampling::kSingleBlock ||
      chroma_subsampling_ == ChromaSubsampling::kAdaptive) {
    sm->Process(kSymbolYuv420, cb.is420_, "yuv420", enc);
  }

  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && !cb.HasLossyAlpha()) continue;
    WriteSplitTransform(cb, channel, sm, enc);
    WriteHasCoeffs(cb, channel, sm, enc);
    WriteTransform(cb, channel, sm, enc);
  }

  if (!context_.use_aom()) {
    ANSDebugPrefix prefix(enc, "coeff_method");
    gparams_->segments_[cb.id_].StoreEncodingMethods(cb, sm, enc);
  }

  if (cb.mtx_set_ != nullptr) {
#if defined(ENABLE_RND_MTX)
    const bool can_use =
        gparams_->mtx_set_.IsOk(cb.mtx_[kYChannel], cb.tdim(kYChannel));
    if (can_use) {
      sm->Process(kSymbolUseRandomMatrix, cb.use_mtx_, "use_rnd_mtx", enc);
      if (cb.use_mtx_) {
        sm->Process(kSymbolRndMtx0, cb.mtx_[kYChannel], "rnd_mtx_y", enc);
      }
    } else {
      assert(!cb.use_mtx_);
    }
#endif
  }
}

void SyntaxWriter::WritePredictors(const CodedBlock& cb, Channel channel,
                                   SymbolManager* const sm,
                                   ANSEncBase* const enc) {
  ANSDebugPrefix prefix(enc, "pred_modes");
  const CodedBlock::CodingParams& params = cb.GetCodingParams(channel);
  if (channel == kYChannel) {
    if (cb.y_context_is_constant_) return;
    sm->Process(kSymbolModeY, params.pred->mode(), "y", enc);
  } else if (channel == kUChannel) {
    sm->Process(kSymbolModeUV, params.pred->mode(), "uv", enc);
  } else if (channel == kVChannel) {
    // kVChannel is not written as it is the same as kUChannel.
  } else {
    sm->Process(kSymbolModeA, params.pred->mode(), "a", enc);
  }
  /* sub_mode = */params.pred->WriteParams(cb, channel, sm, enc);
}

void SyntaxWriter::WriteSplitTransform(const CodedBlock& cb, Channel channel,
                                       SymbolManager* const sm,
                                       ANSEncBase* const enc) {
  if (channel == kYChannel &&
      GetSplitSize(cb.dim(), /*split=*/true) != cb.dim()) {
    sm->Process(kSymbolSplitTransform, cb.GetCodingParams(channel).split_tf,
                "split_tf", enc);
  } else {
    assert(!cb.GetCodingParams(channel).split_tf);
  }
}

void SyntaxWriter::WriteHasCoeffs(const CodedBlock& cb, Channel channel,
                                  SymbolManager* const sm,
                                  ANSEncBase* const enc) {
  for (uint32_t tf_i = 0; tf_i < cb.GetNumTransforms(channel); ++tf_i) {
    const bool tf_has_coeffs = (cb.num_coeffs_[channel][tf_i] != 0);
    const uint32_t cluster = (uint32_t)channel;
    sm->Process(kSymbolHasCoeffs, cluster, tf_has_coeffs ? 1 : 0, "has_coeffs",
                enc);
  }
}

void SyntaxWriter::WriteTransform(const CodedBlock& cb, Channel channel,
                                  SymbolManager* const sm,
                                  ANSEncBase* const enc) {
  const CodedBlock::CodingParams& params = cb.GetCodingParams(channel);
  if (channel == kUChannel || channel == kVChannel || channel == kAChannel) {
    assert(params.tf == kDctDct);
  } else if (cb.HasCoeffs(channel)) {
    // TODO(maryla): try more ways to signal the transform which might be
    //               correlated to block size.
    sm->Process(kSymbolTransform, params.tf, "transform", enc);
  } else {
    // No need to signal the transform as there is no coeff.
  }
}

void SyntaxWriter::RecordBlockHeader(const CodedBlock& cb) {
  ANSEncNoop enc;
  WriteBlockBeforeCoeffs(cb, &symbol_recorder_, &enc);
}

WP2Status SyntaxWriter::WriteBlocks(const Vector<CodedBlock>& cblocks,
                                    const Vector_u16& size_order_indices,
                                    FrontMgrDoubleOrderBase* const mgr,
                                    ANSEnc* const enc) {
  assert(cblocks.size() == num_blocks_);
  assert(cblocks.size() == size_order_indices.size());
  // Note: we could call segment_ids_.InitMap(bw, bh) again here.
  WP2_CHECK_STATUS(residual_writer_.Init(context_.use_aom(),
                                         gparams_->maybe_use_lossy_alpha_));
  context_.Reset();
  assert(size_order_indices.size() == num_blocks_);
  mgr->Clear();
  uint32_t block_idx = 0;
  for (uint16_t ind : size_order_indices) {
    {
      ANSDebugPrefix prefix(enc, "BlockHeader");
      const CodedBlock& cb = cblocks[ind];
      WriteBlockSize(*mgr, cb.dim(), &symbol_writer_, enc);
      Block block_tmp;
      WP2_CHECK_ALLOC_OK(mgr->UseSize(cb.dim(), &block_tmp));
      assert(block_tmp == cb.blk());
    }
    while (true) {
      Block block;
      if (!mgr->UseReady(&block)) break;
      const CodedBlock& cb = cblocks[block_idx];
      assert(cb.blk() == block);
      WP2_CHECK_STATUS(WriteBlock(cb, block_idx, enc));
      ++block_idx;
    }
  }

  if (kDebugPrintRate) {
    float total_estimated = 0.f;
    float total_actual = 0.f;
    float total_difference = 0.f;
    float max_difference = 0.f;
    uint32_t num_non_zero = 0;
    for (uint32_t i = 0; i < cblocks.size(); ++i) {
      const CodedBlock& cb = cblocks[i];
      for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
        if (c == kAChannel && !cb.HasLossyAlpha()) continue;

        const float estimated_rate = residual_rate_[i][c];
        const float actual_rate = residual_rate_[i][4 + c];
        total_estimated += estimated_rate;
        total_actual += actual_rate;

        if (actual_rate == 0 && estimated_rate == 0) continue;

        fprintf(stderr,
                "block (%3d %3d) channel %d rate: estimated %6.2f vs real "
                "%6.2f, %+4.2f",
                cb.x(), cb.y(), c, estimated_rate, actual_rate,
                actual_rate - estimated_rate);

        if (estimated_rate != 0) {
          ++num_non_zero;
          const float relative_difference = actual_rate / estimated_rate - 1.f;
          const float abs_difference = std::abs(relative_difference);
          total_difference += abs_difference;
          if (abs_difference > max_difference) max_difference = abs_difference;
          fprintf(stderr, "  %+.2f%%\n", relative_difference * 100.f);
        } else {
          fprintf(stderr, "\n");
        }
      }
    }
    fprintf(
        stderr,
        "Total residual rate estimated %.2f vs real %.2f, "
        "%+.2f%%, average difference (for non zero rates) %.2f%% max %.2f%%\n",
        total_estimated, total_actual,
        (total_estimated / total_actual - 1.f) * 100.f,
        total_difference / num_non_zero * 100, max_difference * 100);
  }

  return enc->GetStatus();
}

WP2Status SyntaxWriter::WriteBlocksUseSplits(const Vector<CodedBlock>& cblocks,
                                             const Vector_u32& splits,
                                             ANSEnc* const enc) {
  assert(cblocks.size() == num_blocks_);
  // Note: we could call segment_ids_.InitMap(bw, bh) again here.
  WP2_CHECK_STATUS(residual_writer_.Init(context_.use_aom(),
                                         gparams_->maybe_use_lossy_alpha_));
  context_.Reset();

  SplitIteratorDefault it;
  WP2_CHECK_STATUS(
      it.Init(gparams_->partition_set_, tile_rect_.width, tile_rect_.height));

  uint32_t block_idx = 0;
  uint32_t split_idx = 0;
  while (!it.Done()) {
    if (it.CurrentBlock().splittable) {
      ANSDebugPrefix prefix(enc, "BlockHeader");
      const uint32_t split = splits[split_idx];
      WriteBlockSplit(it, split, &symbol_writer_, enc);
      it.SplitCurrentBlock(split);
      ++split_idx;
    } else {
      const CodedBlock& cb = cblocks[block_idx];
      assert(cb.blk() == it.CurrentBlock().block);
      WP2_CHECK_STATUS(WriteBlock(cb, block_idx, enc));
      it.NextBlock();
      ++block_idx;
    }
  }
  return enc->GetStatus();
}

//------------------------------------------------------------------------------

WP2Status SyntaxWriter::FindBestEncodingMethods(CodedBlock* const cb) {
  const uint32_t num_channels = (gparams_->maybe_use_lossy_alpha_ ? 4 : 3);
  for (auto c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !cb->HasLossyAlpha()) continue;
    for (uint32_t tf_i = 0; tf_i < cb->GetNumTransforms(c); ++tf_i) {
      if (context_.use_aom()) {
        ResidualWriter::SetGeometry(cb->num_coeffs_[c][tf_i],
                                    &cb->method_[c][tf_i]);
      } else {
        WP2_CHECK_STATUS(residual_writer_.FindBestEncodingMethod(
            cb->tdim(c), cb->coeffs_[c][tf_i], cb->num_coeffs_[c][tf_i],
            cb->IsFirstCoeffDC(c), c, num_channels, counters_.residuals(),
            &cb->method_[c][tf_i]));
      }
    }
  }
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::DecideAlpha(CodedBlock* const cb,
                                    const BlockModes& modes,
                                    BlockScorer* const scorer) {
  return alpha_writer_.DecideAlpha(cb, modes, residual_writer_, &counters_,
                                   scorer);
}

WP2Status SyntaxWriter::RecordAlpha(const CodedBlock& cb) {
  return alpha_writer_.Record(cb);
}

WP2Status SyntaxWriter::SetInitialSegmentIds() {
  const Rectangle padded_tile_rect = {tile_rect_.x, tile_rect_.y,
                                      Pad(tile_rect_.width, kPredWidth),
                                      Pad(tile_rect_.height, kPredWidth)};
  CodedBlock cb;
  cb.SetDimDefault(Block(0, 0, BLK_4x4));
  FrontMgr4x4 mgr;
  WP2_CHECK_STATUS(
      mgr.InitBase(padded_tile_rect.width, padded_tile_rect.height));
  for (uint32_t y = 0; y < SizeBlocks(padded_tile_rect.height); ++y) {
    for (uint32_t x = 0; x < SizeBlocks(padded_tile_rect.width); ++x) {
      cb.SetXY(mgr, x, y);
      cb.id_ = AssignSegmentId(*config_, *gparams_, padded_tile_rect, cb.blk());
      context_.segment_id_predictor().InitInitialSegmentId(cb, cb.id_);
      mgr.Use4x4();
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status SyntaxWriter::Record(const CodedBlock& cb) {
  assert(recorded_blocks_ < num_blocks_);

  RecordBlockHeader(cb);
  const uint32_t num_channels = (gparams_->maybe_use_lossy_alpha_ ? 4 : 3);
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && !cb.HasLossyAlpha()) continue;

    if (kDebugPrintRate) {
      float rate;
      WP2_CHECK_STATUS(cb.ResidualRate(context_, channel, num_channels,
                                       counters_.residuals(), &rate));
      residual_rate_[recorded_blocks_][channel] = rate;
    }

    WP2_CHECK_STATUS(residual_writer_.RecordCoeffs(
        cb, channel, &symbol_recorder_));
  }

  ++recorded_blocks_;

  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::RecordSize(const FrontMgrDoubleOrderBase& mgr,
                                   BlockSize dim) {
  return RecordSize(dim, mgr.GetMaxPossibleBlock().dim(),
                    mgr.GetPartitionSet());
}

WP2Status SyntaxWriter::RecordSize(BlockSize dim, BlockSize max_possible_size,
                                   PartitionSet partition_set) {
  assert(!use_splits_);
  ANSEncNoop enc;
  WriteBlockSize(dim, max_possible_size, partition_set, &symbol_recorder_,
                 &enc);
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::RecordSplit(const SplitIteratorBase& mgr,
                                    uint32_t split_idx) {
  assert(use_splits_);
  ANSEncNoop enc;
  WriteBlockSplit(mgr, split_idx, &symbol_recorder_, &enc);
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::ResetRecord() {
  recorded_blocks_ = 0;
  WP2_CHECK_STATUS(symbol_recorder_.ResetRecord(/*reset_backup=*/false));
  if (gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_writer_.ResetRecord());
  }
  return WP2_STATUS_OK;
}

WP2Status SyntaxWriter::WriteBlock(const CodedBlock& cb, uint32_t block_index,
                                   ANSEnc* const enc) {
  {
    ANSDebugPrefix prefix(enc, "BlockHeader");
    WriteBlockBeforeCoeffs(cb, &symbol_writer_, enc);
  }

  float previous_cost;
  if (kDebugPrintRate) {
    previous_cost = enc->GetCost();
  }

  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !cb.HasLossyAlpha()) continue;

    WP2_CHECK_STATUS(residual_writer_.WriteCoeffs(cb, c, enc, &symbol_writer_));
    if (kDebugPrintRate) {
      const float cost = enc->GetCost();
      const float actual_rate = cost - previous_cost;
      residual_rate_[block_index][4 + c] = actual_rate;
      previous_cost = cost;
    }
  }

  if (gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_writer_.Write(cb, enc));
  }
  if (cb.HasLossyAlpha()) assert(gparams_->has_alpha_);

#if defined(WP2_ENC_DEC_DEEP_MATCH)
  PutRawPixels(cb, cb.out_, enc);  // Debug
#endif
  return WP2_STATUS_OK;
}

}  // namespace WP2
