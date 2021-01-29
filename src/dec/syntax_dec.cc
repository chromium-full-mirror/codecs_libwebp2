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
// WP2 lossy decoding of syntax.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/constants.h"
#include "src/common/lossy/block_size.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/ans_utils.h"
#include "src/utils/front_mgr.h"
#include "src/utils/utils.h"
#include "src/wp2/decode.h"

namespace WP2 {

//------------------------------------------------------------------------------

SyntaxReader::SyntaxReader(ANSDec* const dec, const Rectangle& rect)
    : dec_(dec), width_(rect.width), height_(rect.height) {}

WP2Status SyntaxReader::ReadHeader(const BitstreamFeatures& features,
                                   const GlobalParams& gparams,
                                   const Tile& tile) {
  // TODO(skal): use differential coding to update gparam locally
  gparams_ = &gparams;

  if (gparams_->has_alpha_) {
    alpha_reader_.reset(new (WP2Allocable::nothrow) AlphaReader(dec_, tile));
    WP2_CHECK_ALLOC_OK(alpha_reader_ != nullptr);
    WP2_CHECK_STATUS(alpha_reader_->Allocate());
  }

  ANSDebugPrefix prefix(dec_, "GlobalHeader");

  const bool use_aom = dec_->ReadBool("use_aom_coeffs");
  residual_reader_.Init(use_aom);

  WP2_CHECK_STATUS(symbols_info_.InitLossy(
      (uint32_t)gparams_->segments_.size(), gparams.partition_set_,
      gparams_->maybe_use_lossy_alpha_, features.quality_hint, use_aom));
  const uint32_t num_channels = (gparams_->maybe_use_lossy_alpha_ ? 4 : 3);
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !gparams_->maybe_use_lossy_alpha_) continue;
    WP2_CHECK_STATUS(symbols_info_.SetMinMax(
        kSymbolDC, ResidualIO::GetCluster(c, num_channels), /*min=*/0,
        /*max=*/gparams_->GetMaxDCRange(c) - 1));
  }
  symbols_info_.SetClusters(kSymbolDC, symbols_info_.NumClusters(kSymbolDC));
  WP2_CHECK_STATUS(sr_.Init(symbols_info_, dec_));
  WP2_CHECK_STATUS(sr_.Allocate());

  const uint32_t max_num_blocks = SizeBlocks(width_) * SizeBlocks(height_);
  const uint32_t min_num_blocks = std::max(1u, max_num_blocks / kMaxBlockSize2);
  num_blocks_ =
      ReadLargeRange(min_num_blocks, max_num_blocks, dec_, "num_blocks");
  num_transforms_ = std::min(num_blocks_ * 4, max_num_blocks);

  if (num_blocks_ == 1) {
    chroma_subsampling_ = ChromaSubsampling::kSingleBlock;
  } else {
    chroma_subsampling_ = (ChromaSubsampling)dec_->ReadRValue(
        (int)ChromaSubsampling::kSingleBlock, "chroma_subsampling");
  }

  WP2_CHECK_STATUS(LoadDictionaries());

  WP2_CHECK_STATUS(dec_->GetStatus());
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status SyntaxReader::LoadDictionaries() {
  const uint32_t max_num_blocks = SizeBlocks(width_) * SizeBlocks(height_);
  if (gparams_->explicit_segment_ids_) {
    WP2_CHECK_STATUS(
        sr_.ReadHeader(kSymbolSegmentId, num_blocks_, "segment_id"));
  }
  if (gparams_->use_rnd_mtx_) {
    WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolRndMtx0, num_blocks_, "rnd_mtx"));
    WP2_CHECK_STATUS(
        sr_.ReadHeader(kSymbolUseRandomMatrix, num_blocks_, "use_rnd_mtx"));
  }
  WP2_CHECK_STATUS(
      sr_.ReadHeader(kSymbolSplitTransform, num_blocks_, "split_tf"));
  if (chroma_subsampling_ == ChromaSubsampling::kSingleBlock ||
      chroma_subsampling_ == ChromaSubsampling::kAdaptive) {
    WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolYuv420, num_blocks_, "yuv420"));
  }
  WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolTransform, num_blocks_, "transform"));
  sr_.SetRange(kSymbolModeY, /*min=*/0,
               /*max=*/gparams_->y_preds_.GetMaxMode());
  WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolModeY, num_blocks_, "mode"));
  sr_.SetRange(kSymbolModeUV, /*min=*/0,
               /*max=*/gparams_->uv_preds_.GetMaxMode());
  WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolModeUV, num_blocks_, "mode"));

  WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolCflSlope, 2 * num_blocks_, "cfl"));

  for (uint32_t i = 0; i < GetNumUniqueBounds(gparams_->partition_set_); ++i) {
    WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolBlockSize, /*cluster=*/i,
                                    num_blocks_, "block_size"));
  }

  const uint32_t num_coeffs_max =
      max_num_blocks * kMinBlockSizePix * kMinBlockSizePix;
  const uint32_t num_coeffs_max_uv =
      (chroma_subsampling_ == ChromaSubsampling::k420) ? num_coeffs_max / 4
                                                       : num_coeffs_max;

  WP2_CHECK_STATUS(
      segment_ids_.ReadHeader(dec_, (uint32_t)gparams_->segments_.size(),
                              gparams_->explicit_segment_ids_, width_));

  if (gparams_->has_alpha_) {
    WP2_CHECK_STATUS(alpha_reader_->ReadHeader(*gparams_));

    if (alpha_reader_->GetAlphaMode() == kAlphaModeLossy) {
      WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolBlockAlphaMode, num_blocks_,
                                      "block_alpha_mode"));
    }
    if (alpha_reader_->GetAlphaMode() != kAlphaModeLossless) {
      sr_.SetRange(kSymbolModeA, /*min=*/0, /*max=*/kAPredModeNum - 1);
      WP2_CHECK_STATUS(sr_.ReadHeader(kSymbolModeA, num_blocks_, "mode"));
    }
  }

  WP2_CHECK_STATUS(residual_reader_.ReadHeader(
      &sr_, num_coeffs_max, num_coeffs_max_uv, num_transforms_,
      gparams_->maybe_use_lossy_alpha_,
      gparams_->has_alpha_ &&
          (alpha_reader_->GetAlphaMode() != kAlphaModeLossless)));

  sr_.Clear();   // reclaim some tmp memory
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status SyntaxReader::ReadPredModes(CodedBlockBase* const cb) {
  ANSDebugPrefix prefix(dec_, "pred_modes");
  cb->y_context_is_constant_ = cb->ContextIsConstant(kYChannel);
  // no ReadParams() for kVChannel (supposed to be the same as kUChannel)
  for (Channel channel : {kYChannel, kAChannel, kUChannel}) {
    if (channel == kAChannel && !cb->HasLossyAlpha()) continue;
    const Predictors& preds = gparams_->GetPredictors(channel);
    if (channel == kYChannel && cb->y_context_is_constant_) {
      cb->SetLumaUniformPredictor(preds);
    } else {
      const uint32_t mode =
          (channel == kYChannel) ? sr_.Read(kSymbolModeY, "y") :
          (channel == kAChannel) ? sr_.Read(kSymbolModeA, "a") :
                                   sr_.Read(kSymbolModeUV, "uv");
      const uint32_t sub_mode =
          preds.GetPred(mode)->ReadParams(cb, channel, &sr_, dec_);
      if (channel == kUChannel) {
        // we need to read the 'params' of the v-channel, even if it
        // doesn't change the predictor. Useful mostly for SignalingCfl, where
        // U and V channel have a different param in cfl_[].
        preds.GetPred(mode)->ReadParams(cb, kVChannel, &sr_, dec_);
      }
      cb->SetPredictor(channel, preds.GetPred(mode, sub_mode));
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void SyntaxReader::ReadSplitTransform(Channel channel,
                                      CodedBlockBase* const cb) {
  cb->GetCodingParams(channel)->split_tf =
      (channel == kYChannel &&
       GetSplitSize(cb->dim(), /*split=*/true) != cb->dim() &&
       (bool)sr_.Read(kSymbolSplitTransform, "split_tf"));
}

void SyntaxReader::ReadHasCoeffs(Channel channel, CodedBlockBase* const cb) {
  bool block_has_coeffs = false;
  for (uint32_t tf_i = 0; tf_i < cb->GetNumTransforms(channel); ++tf_i) {
    // TODO(yguyon): If '!is420_' was signaled, deduce 'has_coeffs'.
    // TODO(yguyon): If 'has_lossy_alpha_' was signaled, deduce 'has_coeffs'.
    bool tf_has_coeffs;
    if (!block_has_coeffs && cb->GetCodingParams(channel)->split_tf &&
        tf_i + 1 == cb->GetNumTransforms(channel)) {
      // At least one transform has at least one non-zero coeff, otherwise
      // 'split_tf' would be pointless.
      tf_has_coeffs = true;
    } else {
      const uint32_t cluster = (uint32_t)channel;
      tf_has_coeffs =
          (bool)sr_.Read(kSymbolHasCoeffs, cluster, "has_coeffs", nullptr);
    }
    // 1 means "at least one coeff is not zero", 0 means "all coeffs are 0".
    cb->num_coeffs_[channel][tf_i] = tf_has_coeffs ? 1 : 0;
    if (tf_has_coeffs) block_has_coeffs = true;
  }
}

void SyntaxReader::ReadTransform(Channel channel, CodedBlockBase* const cb) {
  CodedBlockBase::CodingParams* params = cb->GetCodingParams(channel);
  const TransformPair implicit_tf = cb->GetImplicitTf(channel);
  if (implicit_tf != kUnknownTf) {
    // The transform is already known, no need to signal it.
    params->tf = implicit_tf;
  } else if (cb->HasCoeffs(channel)) {
    params->tf = (TransformPair)sr_.Read(kSymbolTransform, "transform");
  } else {
    // 0 coeff, transform does not matter.
    // TODO(maryla): we could have a noop transform to avoid doing any
    //               computation.
    params->tf = kIdentityIdentity;
  }
}

void SyntaxReader::ReadIs420(CodedBlockBase* const cb) {
  dec_->PushBitTracesCustomPrefix("is_420");
  if (chroma_subsampling_ == ChromaSubsampling::kSingleBlock ||
      chroma_subsampling_ == ChromaSubsampling::kAdaptive) {
    cb->is420_ = (bool)sr_.Read(kSymbolYuv420, "yuv420");
  } else if (chroma_subsampling_ == ChromaSubsampling::k420) {
    cb->is420_ = true;
  } else if (chroma_subsampling_ == ChromaSubsampling::k444) {
    cb->is420_ = false;
  } else {
    assert(false);
  }
  dec_->PopBitTracesCustomPrefix("is_420");
}

WP2Status SyntaxReader::GetBlock(CodedBlockBase* const cb, YUVPlane* const out,
                                 BlockInfo* const info) {
  // Read some block data.
  dec_->AddDebugPrefix("BlockHeader");

  // Initialize the views now that 'blk_.dim' is known.
  cb->SetContextInput(*out);  // Predict from other reconstructed pixels.
  cb->SetReconstructedOutput(out);

  dec_->PushBitTracesCustomPrefix("segment_id");
  segment_ids_.ReadId(&sr_, cb);  // Look at surrounding availability.
  dec_->PopBitTracesCustomPrefix("segment_id");

  if (gparams_->has_alpha_) {
    alpha_reader_->GetBlockHeader(&sr_, cb);
  } else {
    cb->alpha_mode_ = kBlockAlphaLossless;
  }

  // Decode the predictor used and thus get access to the implicit tf, if any.
  dec_->PushBitTracesCustomPrefix("mode");
  WP2_CHECK_STATUS(ReadPredModes(cb));
  dec_->PopBitTracesCustomPrefix("mode");

  // Bound the number of coeffs to decode for chroma.
  ReadIs420(cb);

  // If there is at least one coeff to read and if the transform is not
  // implicit, decode the transform. The coeff method may differ depending on
  // the transform used.
  dec_->PushBitTracesCustomPrefix("transforms");
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && !cb->HasLossyAlpha()) continue;
    ReadSplitTransform(channel, cb);
    ReadHasCoeffs(channel, cb);
    ReadTransform(channel, cb);
  }
  dec_->PopBitTracesCustomPrefix("transforms");

  // Signal the method with which coeffs were encoded.
  if (!residual_reader_.use_aom()) {
    ANSDebugPrefix prefix(dec_, "coeff_method");
    const Segment& segment = gparams_->segments_[cb->id_];
    WP2_CHECK_STATUS(segment.ReadEncodingMethods(&sr_, cb));
  }

  // Read random matrix.
  // TODO(skal): check rnd_mtx behavior when no coeff and/or identity transform
  dec_->PushBitTracesCustomPrefix("rnd_mtx");
  if (gparams_->use_rnd_mtx_) {
#if defined(ENABLE_RND_MTX)
    const bool can_use =
        gparams_->mtx_set_.IsOk(cb->mtx_[kYChannel], cb->tdim(kYChannel));
    cb->use_mtx_ =
        can_use && (bool)sr_.Read(kSymbolUseRandomMatrix, "use_rnd_mtx");
    if (cb->use_mtx_) {
      cb->mtx_[kYChannel] = (uint8_t)sr_.Read(kSymbolRndMtx0, "rnd_mtx_y");
    }
#else
    cb->use_mtx_ = false;
#endif
  }
  dec_->PopBitTracesCustomPrefix("rnd_mtx");

  dec_->PopDebugPrefix();

  // Read coefficients.
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !cb->HasLossyAlpha()) continue;
    dec_->PushBitTracesCustomPrefix(kCoeffsStr[c]);
    const Segment& segment =
        gparams_->segments_[(c == kAChannel) ? 0 : cb->id_];
    const int16_t* const dequant = segment.GetQuant(c).Dequant();
    WP2_CHECK_STATUS(
        residual_reader_.ReadCoeffs(c, dequant, dec_, &sr_, cb, info));
    dec_->PopBitTracesCustomPrefix(kCoeffsStr[c]);
  }

  // alpha
  if (gparams_->has_alpha_) {
    dec_->PushBitTracesCustomPrefix("alpha");
    WP2_CHECK_STATUS(alpha_reader_->GetBlock(cb));
    dec_->PopBitTracesCustomPrefix("alpha");
  }

  WP2_CHECK_STATUS(dec_->GetStatus());
  return WP2_STATUS_OK;
}

BlockSize SyntaxReader::GetBlockSize(const FrontMgrDoubleOrderBase& mgr) {
  ANSDebugPrefix prefix(dec_, "BlockHeader");
  return ReadBlockSize(mgr, &sr_);
}

//------------------------------------------------------------------------------

}  // namespace WP2
