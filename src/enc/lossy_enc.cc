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

#include "src/dsp/math.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/ans_utils.h"
#include "src/utils/csp.h"
#include "src/utils/utils.h"

namespace WP2 {

//------------------------------------------------------------------------------

ChromaSubsampling DecideChromaSubsampling(const EncoderConfig& config,
                                          bool more_than_one_block) {
  if (!more_than_one_block) return ChromaSubsampling::kSingleBlock;
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
  assert(false);
}

bool DecideAOMCoeffs(const EncoderConfig& config, const Rectangle& tile_rect) {
  return (config.quality >= 35) &&
         (tile_rect.width >= 100 && tile_rect.height >= 100);
}

WP2Status DecideTransforms(const EncoderConfig& config,
                           Vector<TransformPair>* const transforms,
                           Vector<TransformPair>* const transforms_subset) {
  if (config.info != nullptr && config.info->disable_transforms) {
    WP2_CHECK_ALLOC_OK(transforms->reserve(1));
    transforms->push_back_no_resize(kDctDct);
    WP2_CHECK_ALLOC_OK(transforms_subset->copy_from(*transforms));
    return WP2_STATUS_OK;
  }

  WP2_CHECK_ALLOC_OK(transforms->reserve(kNumTransformPairs));
  transforms->push_back_no_resize(kDctDct);
  if (config.effort >= 3) {
    transforms->push_back_no_resize(kAdstAdst);
    if (config.effort >= 4) {
      transforms->push_back_no_resize(kDctAdst);
      transforms->push_back_no_resize(kAdstDct);
      if (config.effort >= 5) {
        transforms->push_back_no_resize(kIdentityIdentity);
        if (config.effort >= 6) {
          transforms->push_back_no_resize(kDctIdentity);
          transforms->push_back_no_resize(kIdentityDct);
        }
      }
    }
  }
  // This is used to find a predictor faster
  // (= not testing every transform with each predictor).
  WP2_CHECK_ALLOC_OK(transforms_subset->reserve(transforms->size()));
  transforms_subset->push_back_no_resize(kDctDct);
  if (config.effort >= 6) {
    transforms_subset->push_back_no_resize(kAdstAdst);
    if (config.effort >= 7) {
      transforms_subset->push_back_no_resize(kDctAdst);
      transforms_subset->push_back_no_resize(kAdstDct);
      if (config.effort >= 8) {
        transforms_subset->push_back_no_resize(kIdentityIdentity);
        if (config.effort >= 9) {
          transforms_subset->push_back_no_resize(kDctIdentity);
          transforms_subset->push_back_no_resize(kIdentityDct);
        }
      }
    }
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
  const Vector<Segment>& segments = gparams.segments_;
  const CSPTransform& transf = gparams.transf_;

  // First, extract partitioning.
  VectorNoCtor<Block> blocks;
  WP2_CHECK_ALLOC_OK(blocks.resize(forced_partition.size()));
  std::copy(forced_partition.begin(), forced_partition.end(), blocks.begin());
  WP2_CHECK_STATUS(ExtractBlockPartition(*config_, gparams, yuv_buffer,
                                         tile_rect, partitioning_progress,
                                         &blocks));
  FrontMgrDefault mgr;
  WP2_CHECK_STATUS(mgr.Init(config_->partition_set, config_->partition_snapping,
                            padded_rect.width, padded_rect.height));

  Vector_u16 size_order_indices;
  WP2_CHECK_STATUS(mgr.Sort(blocks, size_order_indices));
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
      (uint32_t)cblocks.size(), use_aom_coeffs, alpha_progress));

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

  Vector<TransformPair> transforms, transforms_subset;
  WP2_CHECK_STATUS(DecideTransforms(*config_, &transforms, &transforms_subset));

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

      const Segment& segment = segments[cb.id_];

      cb.ResetContextCache();
      cb.y_context_is_constant_ = cb.ContextIsConstant(kYChannel);

      WP2_CHECK_STATUS(cb.OptimizeModesLuma(
          *config_, tile_rect, gparams.maybe_use_lossy_alpha_, gparams.y_preds_,
          segment, writer.context(), transforms, transforms_subset,
          writer.counters()));

      // If the U/V planes don't depend on luma, we only need to process them
      // during the first pass.
      if (pass == 0 || chroma_depends_on_luma) {
        WP2_CHECK_STATUS(cb.OptimizeModesChroma(
            *config_, tile_rect, gparams.maybe_use_lossy_alpha_, mgr,
            gparams.uv_preds_, segment, writer.context(), chroma_subsampling,
            &dc_error_u, &dc_error_v, writer.counters()));
      }
      if (gparams.has_alpha_) {
        WP2_CHECK_STATUS(writer.DecideAlpha(&cb));
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
          const bool selected = VDSelected(tile_rect.x, tile_rect.y,
                                           cb.blk().rect_pix(), *config_);
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
        cb.ToBlockInfo(&block_info);
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

      WP2_CHECK_STATUS(writer.FindBestEncodingMethods(&cb));
      WP2_CHECK_STATUS(writer.Record(cb));
      mgr.Use(cb.blk());

      if (cb.blk().x_pix() + cb.blk().w_pix() >= tile_rect.width) {
        // Progress is advanced by the height of each block on the right,
        // the total being the number of rows in the tile (not padded).
        const Rectangle cb_rect = cb.blk().rect_pix().ClipWith(
            {0, 0, tile_rect.width, tile_rect.height});
        WP2_CHECK_STATUS(row_progress.AdvanceBy(cb_rect.height));
      }
    }

    // Record info about the block sizes.
    // Sizes might be written in a different order than the blocks, hence a
    // separate recording: we need to follow the SizeIndices() order.
    mgr.Clear();
    for (uint16_t ind : size_order_indices) {
      const auto& block = blocks[ind];
      WP2_CHECK_STATUS(writer.RecordSize(mgr, block.dim()));
      Block block_tmp;
      WP2_CHECK_ALLOC_OK(mgr.UseSize(block.dim(), ind, &block_tmp));
      assert(block == block_tmp);
      // Empty the size stack if possible.
      while (mgr.UseFinal()) {
      }
    }

    // TODO(maryla): add an early-exit stopping criterion, so we don't do more
    //               passes if we don't think they're needed.
  }

  // write header (features, dictionaries...)
  WP2_CHECK_STATUS(writer.WriteHeader(enc));

  // Write coded blocks
  WP2_CHECK_STATUS(writer.WriteBlocks(cblocks, size_order_indices, &mgr, enc));
  WP2_CHECK_STATUS(write_progress.AdvanceBy(1.));

#if !defined(WP2_REDUCE_BINARY_SIZE)
  if (!vd_plane.IsEmpty()) {
    if (VDMatch(*config_, "a")) {
      WP2_CHECK_STATUS(
          vd_plane.ToGray(&debug_output, /*num_bits=*/8,
                          /*shift=*/VDMatch(*config_, "original-residuals")));
    } else {
      WP2_CHECK_STATUS(vd_plane.ToGray(&debug_output, /*num_bits=*/10));
    }
  }
#endif  // WP2_REDUCE_BINARY_SIZE

  return enc->GetStatus();
}

// -----------------------------------------------------------------------------
// Empty neural encoding

#if !defined(WP2_EXPERIMENTAL)
WP2Status NeuralEncode(const ArgbBuffer& buffer, const EncoderConfig& config,
                       ANSEnc* const enc) {
  (void)buffer;
  (void)config;
  (void)enc;
  fprintf(stderr, "Attempted to use neural compression in a "
                  "non-experimental build.\n");
  return WP2_STATUS_UNSUPPORTED_FEATURE;
}
#endif  // !WP2_EXPERIMENTAL

}    // namespace WP2
