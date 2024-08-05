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
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>

#include "src/common/global_params.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/context.h"
#include "src/common/lossy/predictor.h"
#include "src/common/lossy/segment.h"
#include "src/common/lossy/transforms.h"
#include "src/common/progress_watcher.h"
#include "src/common/symbols.h"
#include "src/common/vdebug.h"
#include "src/enc/analysis.h"
#include "src/enc/block_enc.h"
#include "src/enc/tile_enc.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/ans_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/csp.h"
#include "src/utils/front_mgr.h"
#include "src/utils/plane.h"
#include "src/utils/split_iterator.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

ChromaSubsampling DecideChromaSubsampling(const EncoderConfig& config,
                                          bool more_than_one_block) {
  if (!more_than_one_block) return ChromaSubsampling::kSingleBlock;
  if (!config.enable_alt_tuning) {
    return (config.uv_mode == EncoderConfig::UVMode444)
               ? ChromaSubsampling::k444
               : ChromaSubsampling::k420;
  }
  switch (config.uv_mode) {
    case EncoderConfig::UVMode444:
      return ChromaSubsampling::k444;
    case EncoderConfig::UVMode420:
      return ChromaSubsampling::k420;
    case EncoderConfig::UVModeAuto: {
      return (config.quality >= 90) ? ChromaSubsampling::k444
                                    : ChromaSubsampling::k420;
    }
    default:
      assert(config.uv_mode == EncoderConfig::UVModeAdapt);
      return ChromaSubsampling::kAdaptive;
  }
}

bool DecideAOMCoeffs(const EncoderConfig& config, const Rectangle& tile_rect) {
  return !config.enable_alt_tuning || (config.quality >= 35) ||
         (tile_rect.width >= 50 && tile_rect.height >= 50);
}

namespace {

WP2Status DecideLumaSplits(const EncoderConfig& config,
                           BlockModes* const modes) {
  WP2_CHECK_ALLOC_OK(modes->splits_tried_during_preds.push_back(false));
  if (config.effort >= 8) {
    WP2_CHECK_ALLOC_OK(modes->splits_tried_during_preds.push_back(true));
  } else if (config.effort >= 5) {
    WP2_CHECK_ALLOC_OK(modes->splits_tried_after_preds.push_back(true));
  }
  return WP2_STATUS_OK;
}

WP2Status DecidePreds(const EncoderConfig& config, const Predictors& preds,
                      int min_main_preds_effort, int min_sub_preds_effort,
                      BlockModes* const modes) {
  if (config.effort < min_main_preds_effort) {
    WP2_CHECK_ALLOC_OK(modes->main_preds.push_back(preds.GetPred(0)));
    return WP2_STATUS_OK;
  }

  WP2_CHECK_ALLOC_OK(modes->main_preds.reserve(kYBasePredNum + kAnglePredNum));
  WP2_CHECK_ALLOC_OK(modes->sub_preds.reserve(kAnglePredNum *
                                              kDirectionalMaxAngleDeltaYA * 2));

  for (const Predictor* pred : preds) {
    if (pred == preds.GetPred(pred->mode())) {
      modes->main_preds.push_back_no_resize(pred);
    } else if (config.effort >= min_sub_preds_effort) {
      modes->sub_preds.push_back_no_resize(pred);
    }
  }
  return WP2_STATUS_OK;
}

WP2Status DecideLumaTransforms(const EncoderConfig& config,
                               BlockModes* const modes) {
  // Not testing every transform with each predictor is faster.
  WP2_CHECK_ALLOC_OK(modes->tf_tried_during_preds.reserve(kNumTransformPairs));
  WP2_CHECK_ALLOC_OK(modes->tf_tried_after_preds.reserve(kNumTransformPairs));
  modes->tf_tried_during_preds.push_back_no_resize(kDctDct);
  if (config.effort >= 6) {
    modes->tf_tried_during_preds.push_back_no_resize(kAdstAdst);
  } else if (config.effort >= 3) {
    modes->tf_tried_after_preds.push_back_no_resize(kAdstAdst);
  }
  if (config.effort >= 7) {
    modes->tf_tried_during_preds.push_back_no_resize(kDctAdst);
    modes->tf_tried_during_preds.push_back_no_resize(kAdstDct);
  } else if (config.effort >= 4) {
    modes->tf_tried_after_preds.push_back_no_resize(kDctAdst);
    modes->tf_tried_after_preds.push_back_no_resize(kAdstDct);
  }
  if (config.effort >= 8) {
    modes->tf_tried_during_preds.push_back_no_resize(kIdentityIdentity);
  } else if (config.effort >= 5) {
    modes->tf_tried_after_preds.push_back_no_resize(kIdentityIdentity);
  }
  if (config.effort >= 9) {
    modes->tf_tried_during_preds.push_back_no_resize(kDctIdentity);
    modes->tf_tried_during_preds.push_back_no_resize(kIdentityDct);
  } else if (config.effort >= 6) {
    modes->tf_tried_after_preds.push_back_no_resize(kDctIdentity);
    modes->tf_tried_after_preds.push_back_no_resize(kIdentityDct);
  }
  return WP2_STATUS_OK;
}

WP2Status DecideLumaSegments(const EncoderConfig& config,
                             const Vector<Segment>& segments,
                             BlockModes* const modes) {
  if (config.effort >= 9) {
    WP2_CHECK_ALLOC_OK(
        modes->segment_ids_tried_during_preds.resize(segments.size()));
    std::iota(modes->segment_ids_tried_during_preds.begin(),
              modes->segment_ids_tried_during_preds.end(), 0);
  } else if (config.effort >= 5) {
    WP2_CHECK_ALLOC_OK(
        modes->segment_ids_tried_after_preds.resize(segments.size()));
    std::iota(modes->segment_ids_tried_after_preds.begin(),
              modes->segment_ids_tried_after_preds.end(), 0);
  }
  return WP2_STATUS_OK;
}

}  // namespace

WP2Status DecideModes(const EncoderConfig& config, const GlobalParams& gparams,
                      BlockModes* const y_modes, BlockModes* const uv_modes,
                      BlockModes* const a_modes) {
  // At which effort all main- or sub-predictors start to be tested.
  constexpr uint32_t kMinMainPredsEffort = 1, kMinSubPredsEffort = 2;

  // Luma
  WP2_CHECK_STATUS(DecideLumaSplits(config, y_modes));
  WP2_CHECK_STATUS(DecidePreds(config, gparams.y_preds_, kMinMainPredsEffort,
                               kMinSubPredsEffort, y_modes));
  WP2_CHECK_STATUS(DecideLumaTransforms(config, y_modes));
  WP2_CHECK_STATUS(DecideLumaSegments(config, gparams.segments_, y_modes));

  // Chroma
  WP2_CHECK_ALLOC_OK(uv_modes->splits_tried_during_preds.push_back(false));
  WP2_CHECK_STATUS(DecidePreds(config, gparams.uv_preds_, kMinMainPredsEffort,
                               kMinSubPredsEffort, uv_modes));
  WP2_CHECK_ALLOC_OK(uv_modes->tf_tried_during_preds.push_back(kDctDct));

  // Alpha
  if (gparams.maybe_use_lossy_alpha_) {
    WP2_CHECK_ALLOC_OK(a_modes->splits_tried_during_preds.push_back(false));
    WP2_CHECK_STATUS(DecidePreds(config, gparams.a_preds_, kMinMainPredsEffort,
                                 kMinSubPredsEffort, a_modes));
    WP2_CHECK_ALLOC_OK(a_modes->tf_tried_during_preds.push_back(kDctDct));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status TileEncoder::LossyEncode(const VectorNoCtor<Block>& forced_partition,
                                   ANSEnc* const enc) {
  assert(enc != nullptr && tiles_layout_ != nullptr && tile_ != nullptr);
  const GlobalParams& gparams = *tiles_layout_->gparams;
  const YUVPlane& yuv_buffer = tile_->yuv_input;
  const Rectangle& tile_rect = tile_->rect;

  if (gparams.type_ == GlobalParams::GP_BOTH) {
    ANSDebugPrefix prefix(enc, "GlobalHeader");
    enc->PutBool(config_->use_neural_compression, "is_neural");
  } else {
    assert(!config_->use_neural_compression);
  }
  if (config_->use_neural_compression) {
    assert(!tile_->rgb_input.IsEmpty());
    WP2_CHECK_STATUS(NeuralEncode(tile_->rgb_input, *config_, enc));
    WP2_CHECK_STATUS(tile_->progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }
  assert(!yuv_buffer.IsEmpty());

  const ProgressRange partitioning_progress(tile_->progress, 0.3);
  const ProgressRange alpha_progress(tile_->progress, 0.2);
  const ProgressRange passes_progress(tile_->progress, 0.4);
  const ProgressRange write_progress(tile_->progress, 0.1);

  // shortcuts
  const Rectangle padded_rect = {tile_rect.x, tile_rect.y,
                                 yuv_buffer.GetWidth(), yuv_buffer.GetHeight()};
  const CSPTransform& transf = gparams.transf_;

  // First, extract partitioning.
  VectorNoCtor<Block> blocks;
  WP2_CHECK_ALLOC_OK(blocks.resize(forced_partition.size()));
  std::copy(forced_partition.begin(), forced_partition.end(), blocks.begin());
  Vector_u32 splits;
  WP2_CHECK_STATUS(ExtractBlockPartition(*config_, gparams, yuv_buffer,
                                         tile_rect, partitioning_progress,
                                         &blocks, &splits));
  const bool use_splits = !splits.empty();
  if (use_splits) assert(gparams.partition_snapping_);

  FrontMgrDefault mgr;
  WP2_CHECK_STATUS(mgr.Init(config_->partition_set, config_->partition_snapping,
                            padded_rect.width, padded_rect.height));

  Vector_u16 size_order_indices;
  if (!use_splits) {
    WP2_CHECK_STATUS(mgr.Sort(blocks, size_order_indices));
  }
  // process all blocks

  // Setup the top-level coding info for each cblock
  Vector<CodedBlock> cblocks;
  WP2_CHECK_ALLOC_OK(cblocks.resize(blocks.size()));
  for (uint32_t i = 0; i < blocks.size(); ++i) {
    const auto& blk = blocks[i];
    CodedBlock& cb = cblocks[i];
    cb.SetRange(transf.GetYUVMin(), transf.GetYUVMax());
    cb.SetDim(blk, mgr);
    mgr.Use(blk);
  }

  // Assign Ids and simplify
  WP2_CHECK_STATUS(AssignSegmentIds(*config_, gparams, padded_rect, &cblocks));

  bool chroma_depends_on_luma = false;
  for (const Predictor* p : gparams.uv_preds_) {
    chroma_depends_on_luma |= p->DependsOnLuma();
  }
  assert(chroma_depends_on_luma);  // For now CflPredictor is always tried.

  // Use AOM coeffs at high quality for large images.
  const bool use_aom_coeffs = DecideAOMCoeffs(*config_, tile_rect);
  const ChromaSubsampling chroma_subsampling = DecideChromaSubsampling(
      *config_, /*more_than_one_block=*/(cblocks.size() > 1));

  // populate the dictionaries
  ANSDictionaries dicts;
  SyntaxWriter writer;
  WP2_CHECK_STATUS(writer.Init(
      &dicts, *config_, gparams, yuv_buffer, chroma_subsampling, tile_rect,
      (uint32_t)cblocks.size(), use_aom_coeffs, use_splits, alpha_progress));

  const uint32_t diffusion =
      DCDiffusionMap::GetDiffusion(config_->error_diffusion);
  DCDiffusionMap dc_error_u, dc_error_v;
  if (diffusion > 0) {
    WP2_CHECK_STATUS(dc_error_u.Init(tile_rect.width));
    WP2_CHECK_STATUS(dc_error_v.Init(tile_rect.width));
  }

  // Setup the remaining top-level coding info for each cblock
  YUVPlane out;  // reconstructed output
  WP2_CHECK_STATUS(out.Resize(padded_rect.width, padded_rect.height,
                              /*pad=*/1, gparams.maybe_use_lossy_alpha_));
  if (out.A.IsEmpty() && !yuv_buffer.A.IsEmpty()) {
    // No lossy alpha but there still is alpha output, so simulate it.
    WP2_CHECK_STATUS(out.A.SetView(yuv_buffer.A));
  }
  ContextCache context_cache;
  for (auto& cb : cblocks) {
    cb.SetSrcInput(yuv_buffer);
    // Predict from other reconstructed pixels.
    cb.SetContextInput(out, &context_cache);
    cb.SetReconstructedOutput(&out);
    cb.mtx_set_ = gparams.use_rnd_mtx_ ? &gparams.mtx_set_ : nullptr;
  }

  const uint32_t num_passes =
      std::max(config_->pass, (config_->effort >= 7) ? 2 : 1);

#if !defined(WP2_REDUCE_BINARY_SIZE)
  ArgbBuffer debug_output;  // Tile view of 'config.info->debug_output'
  Plane16 vd_plane;
  if (VDMatch(*config_, "")) {
    WP2_CHECK_STATUS(
        debug_output.SetView(config_->info->debug_output, tile_rect));
    if (VDMatch(*config_, "original-residuals") ||
        VDMatch(*config_, "chroma-from-luma") ||
        VDMatch(*config_, "error-diffusion")) {
      WP2_CHECK_STATUS(vd_plane.Resize(tile_rect.width, tile_rect.height));
      // Areas that are not overwritten will show as black in the debug UI.
      vd_plane.Fill(-512);
    }
  }
#endif  // WP2_REDUCE_BINARY_SIZE

  BlockModes y_modes, uv_modes, a_modes;
  WP2_CHECK_STATUS(
      DecideModes(*config_, gparams, &y_modes, &uv_modes, &a_modes));
  BlockScorer scorer;
  WP2_CHECK_STATUS(scorer.Init(*config_, gparams, tile_rect));

  for (uint32_t pass = 0; pass < num_passes; ++pass) {
    const ProgressRange pass_progress(passes_progress, 1. / num_passes);
    const ProgressScale row_progress(pass_progress, 1. / tile_rect.height);

    // main coding loop
    WP2_CHECK_STATUS(writer.InitPass());
    mgr.Clear();
    if (diffusion > 0) {
      dc_error_u.Clear();
      dc_error_v.Clear();
    }
    for (auto& cb : cblocks) {
      assert(!mgr.Done());
#if defined(WP2_BITTRACE)
      BlockInfo block_info;
      cb.original_res_ = &block_info.original_res;
#endif

      cb.ResetContextCache();
      cb.y_context_is_constant_ = cb.ContextIsConstant(kYChannel);

      WP2_CHECK_STATUS(OptimizeModes(
          *config_, tile_rect, kYChannel, gparams.y_preds_, y_modes,
          writer.context(), &cb, writer.counters(), &scorer));

      // If the U/V planes don't depend on luma, we only need to process them
      // during the first pass.
      if (pass == 0 || chroma_depends_on_luma) {
        WP2_CHECK_STATUS(OptimizeModesChroma(
            *config_, tile_rect, gparams.maybe_use_lossy_alpha_, mgr,
            gparams.uv_preds_, chroma_subsampling, uv_modes, writer.context(),
            &cb, writer.counters(), &dc_error_u, &dc_error_v, &scorer));
      }
      if (gparams.has_alpha_) {
        WP2_CHECK_STATUS(writer.DecideAlpha(&cb, a_modes, &scorer));
        WP2_CHECK_STATUS(writer.RecordAlpha(cb));
      }

#if !defined(WP2_REDUCE_BINARY_SIZE)
      // original_coeffs is only available in BITTRACE mode.
#if defined(WP2_BITTRACE)
      if (VDMatch(*config_, "original-residuals")) {
        const Channel channel = VDChannel(*config_);
        if (channel != kAChannel || cb.HasLossyAlpha()) {
          cb.StoreOriginalResiduals(*config_, tile_rect.x, tile_rect.y,
                                    block_info.original_res[channel],
                                    &vd_plane);
        }
      }
#endif
      if (VDMatch(*config_, "chroma-from-luma")) {
        const Channel channel = VDChannel(*config_);
        if (channel != kAChannel || cb.HasLossyAlpha()) {
          const bool selected =
              VDSelected(tile_rect.x, tile_rect.y, cb.AsRect(), *config_);
          std::string* const debug_str =
              selected ? &config_->info->selection_info : nullptr;

          if (VDMatch(*config_, "best-prediction")) {
            cb.StoreBestCflPrediction(channel, transf.GetYUVMin(),
                                      transf.GetYUVMax(), &vd_plane, debug_str);
          } else if (VDMatch(*config_, "best-residuals")) {
            cb.StoreBestCflResiduals(channel, transf.GetYUVMin(),
                                     transf.GetYUVMax(), &vd_plane, debug_str);
          } else if (VDMatch(*config_, "best-slope")) {
            cb.StoreBestCflSlope(channel, transf.GetYUVMin(),
                                 transf.GetYUVMax(), &vd_plane, debug_str);
          } else if (VDMatch(*config_, "best-intercept")) {
            cb.StoreBestCflIntercept(channel, transf.GetYUVMin(),
                                     transf.GetYUVMax(), &vd_plane, debug_str);
          }
        }
      }

      if (VDMatch(*config_, "encoder")) {
        WP2_CHECK_STATUS(
            cb.StoreLambdaMult(*config_, tile_rect.x, tile_rect.y));
        cb.StoreErrorDiffusion(*config_, tile_rect.x, tile_rect.y, &vd_plane);
      }

      if (config_->info != nullptr && config_->info->store_blocks) {
#if defined(WP2_BITTRACE)
        cb.ToBlockInfo(use_aom_coeffs, &block_info);
        block_info.rect.x += tile_rect.x;
        block_info.rect.y += tile_rect.y;
        block_info.bits = 0;

        WP2_CHECK_STATUS(tiles_layout_->assignment_lock.Acquire());
        config_->info->blocks.push_back(block_info);
        tiles_layout_->assignment_lock.Release();
#else
        static bool warning_printed = false;
        if (!warning_printed) {
          printf("Warning! 'store_blocks' needs WP2_BITTRACE compile flag!\n");
          warning_printed = true;
        }
#endif  // defined(WP2_BITTRACE)
      }
#endif  // WP2_REDUCE_BINARY_SIZE

      writer.FindBestEncodingMethods(&cb);
      writer.Record(cb);
      mgr.Use(cb.blk());

      if (cb.blk().x_pix() + cb.blk().w_pix() >= tile_rect.width) {
        // Progress is advanced by the height of each block on the right,
        // the total being the number of rows in the tile (not padded).
        const uint32_t block_height =
            std::min(cb.blk().h_pix(), tile_rect.height - cb.y_pix());
        WP2_CHECK_STATUS(row_progress.AdvanceBy(block_height));
      } else {
        // No progress advancement but check if the user cancelled the encoding.
        WP2_CHECK_STATUS(row_progress.AdvanceBy(0.0));
      }
    }

    if (use_splits) {
      SplitIteratorDefault it;
      WP2_CHECK_STATUS(
          it.Init(gparams.partition_set_, tile_rect.width, tile_rect.height));
      for (uint32_t split_idx : splits) {
        while (!it.CurrentBlock().splittable) it.NextBlock();
        WP2_CHECK_STATUS(writer.RecordSplit(it, split_idx));
        it.SplitCurrentBlock(split_idx);
      }
    } else {
      // Record info about the block sizes.
      // Sizes might be written in a different order than the blocks, hence a
      // separate recording: we need to follow the SizeIndices() order.
      mgr.Clear();
      for (uint16_t ind : size_order_indices) {
        const auto& block = blocks[ind];
        writer.RecordSize(mgr, block.dim());
        Block block_tmp;
        WP2_CHECK_ALLOC_OK(mgr.UseSize(block.dim(), &block_tmp));
        assert(block == block_tmp);
        // Empty the size stack if possible.
        while (mgr.UseReady()) {
        }
      }
    }

    // TODO(maryla): add an early-exit stopping criterion, so we don't do more
    //               passes if we don't think they're needed.
  }

  // write header (features, dictionaries...)
  WP2_CHECK_STATUS(writer.WriteHeader(enc));

  // Write coded blocks
  if (use_splits) {
    WP2_CHECK_STATUS(writer.WriteBlocksUseSplits(cblocks, splits, enc));
  } else {
    WP2_CHECK_STATUS(
        writer.WriteBlocks(cblocks, size_order_indices, &mgr, enc));
  }
  WP2_CHECK_STATUS(write_progress.AdvanceBy(1.));

#if !defined(WP2_REDUCE_BINARY_SIZE)
  if (!vd_plane.IsEmpty()) {
    if (VDMatch(*config_, "a")) {
      const bool is_res = VDMatch(*config_, "original-residuals");
      WP2_CHECK_STATUS(vd_plane.ToGray(
          &debug_output,
          /*bit_depth=*/{is_res ? 9u : 8u, /*is_signed=*/is_res}));
    } else {
      WP2_CHECK_STATUS(vd_plane.ToGray(&debug_output,
                                       /*bit_depth=*/{10, /*is_signed=*/true}));
    }
  }
#endif  // WP2_REDUCE_BINARY_SIZE

  return enc->GetStatus();
}

// -----------------------------------------------------------------------------
// Empty neural encoding

#if !defined(WP2_NEURAL)
WP2Status NeuralEncode(const ArgbBuffer& buffer, const EncoderConfig& config,
                       ANSEnc* const enc) {
  (void)buffer;
  (void)config;
  (void)enc;
  fprintf(stderr,
          "Attempted to use neural compression in a "
          "non-experimental build.\n");
  return WP2_STATUS_UNSUPPORTED_FEATURE;
}
#endif  // !WP2_NEURAL

}  // namespace WP2
