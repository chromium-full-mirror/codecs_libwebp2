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
//   CodedBlock, for encoding
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/enc/block_enc.h"

#include <algorithm>
#include <cassert>

#include "src/common/lossy/block.h"
#include "src/common/lossy/predictor.h"
#include "src/common/lossy/transforms.h"
#include "src/common/symbols.h"
#include "src/enc/trellis.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/front_mgr.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------
// Counters

WP2Status Counters::Init(uint32_t effort, bool use_aom_residuals,
                         const SymbolRecorder& recorder) {
  effort_ = effort;
  use_aom_residuals_ = use_aom_residuals;
  recorder_ = &recorder;

  predictor_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
  WP2_CHECK_ALLOC_OK(predictor_ != nullptr);
  WP2_CHECK_STATUS(
      predictor_->Allocate({kSymbolModeY, kSymbolModeUV, kSymbolModeA}));

  transform_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
  WP2_CHECK_ALLOC_OK(transform_ != nullptr);
  WP2_CHECK_STATUS(transform_->Allocate(
      {kSymbolHasCoeffs, kSymbolTransform, kSymbolSplitTransform}));

  segment_id_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
  WP2_CHECK_ALLOC_OK(segment_id_ != nullptr);
  WP2_CHECK_STATUS(segment_id_->Allocate({kSymbolSegmentId}));

  if (effort >= kSlowCounterEffortThreshold) {
    // Using the UpdatingSymbolCounter slows down encoding by ~9%
    residuals_slow_.reset(new (WP2Allocable::nothrow)
                              UpdatingSymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_slow_ != nullptr);
    residuals_ = residuals_slow_.get();
  } else {
    residuals_fast_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_fast_ != nullptr);
    residuals_ = residuals_fast_.get();
  }
  if (use_aom_residuals) {
    WP2_CHECK_STATUS(residuals()->Allocate(
        {kSymbolTransform, kAOMEOBPT4, kAOMEOBPT8, kAOMEOBPT16, kAOMEOBPT32,
         kAOMEOBPT64, kAOMEOBPT128, kAOMEOBPT256, kAOMEOBPT512, kAOMEOBPT1024,
         kAOMEOBExtra, kAOMCoeffBaseEOB, kAOMCoeffBase, kAOMCoeffBaseRange}));
  } else {
    WP2_CHECK_STATUS(residuals()->Allocate(
        {kSymbolDC, kSymbolResidualUseBounds, kSymbolResidualBound1IsX,
         kSymbolResidualUseBound2, kSymbolResidualIsZero, kSymbolResidualIsOne,
         kSymbolResidualIsTwo, kSymbolResidualEndOfBlock,
         kSymbolResidualHasOnlyOnesLeft}));
  }
  return WP2_STATUS_OK;
}

WP2Status Counters::CopyFrom(const Counters& other,
                             const SymbolRecorder& recorder) {
  if (&recorder != recorder_) {
    WP2_CHECK_STATUS(Init(other.effort_, other.use_aom_residuals_, recorder));
  } else {
    // Init() was already called.
    assert(transform_ != nullptr && residuals_ != nullptr &&
           predictor_ != nullptr);
  }
  if (residuals_slow_ != nullptr) {
    WP2_CHECK_STATUS(residuals_slow_->CopyFrom(*other.residuals_slow_));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// CodedBlock  (encoder)

void CodedBlock::SetSrcInput(const YUVPlane& in) {
  WP2_ASSERT_STATUS(in_.SetView(in, AsRect()));
}

//------------------------------------------------------------------------------

void CodedBlock::ExtractFrom(const YUVPlane& yuv, Channel channel) const {
  Plane16 src, dst;
  WP2_ASSERT_STATUS(src.SetView(yuv.GetChannel(channel), AsRect()));
  WP2_ASSERT_STATUS(dst.SetView(out_.GetChannel(channel), AsRect()));
  WP2_ASSERT_STATUS(dst.Copy(src, /*resize_if_needed=*/false));
}

uint64_t CodedBlock::GetDisto(Channel channel) const {
  const Plane16& p_in = in_.GetChannel(channel);
  const Plane16& p_out = out_.GetChannel(channel);
  // TODO(skal): experiment calling WP2SumSquaredErrorHalfBlock for U/V channel
  return WP2SumSquaredErrorBlock(p_in.Row(0), p_in.Step(), p_out.Row(0),
                                 p_out.Step(), w_pix(), h_pix());
}

void CodedBlock::GetResiduals(Channel channel, uint32_t tf_i,
                              const Plane16& prediction,
                              BlockCoeffs16* const res) const {
  const Plane16& in = in_.GetChannel(channel);

  const Rectangle tf_rect =
      blk().LocalTfRect(GetCodingParams(channel).split_tf, tf_i);
  SubtractBlockFunc sub_block = WP2::SubtractBlock[TrfLog2[tf_rect.width] - 2];

  sub_block(&in.At(tf_rect.x, tf_rect.y), in.Step(),
            &prediction.At(tf_rect.x, tf_rect.y), prediction.Step(),
            (*res)[tf_i], tf_rect.width, tf_rect.height);

#if defined(WP2_BITTRACE)
  if (original_res_ != nullptr) {
    memcpy((*original_res_)[channel][tf_i], &(*res)[tf_i][0],
           tf_rect.width * tf_rect.height * sizeof((*res)[tf_i][0]));
  }
#endif
}

void CodedBlock::TransformAndReconstruct(
    const EncoderConfig& config, const BlockContext& context,
    const Segment& segment, Channel channel, uint32_t tf_i,
    bool reduced_transform, const Plane16& prediction,
    const BlockCoeffs16* const res, BlockCoeffs16* const tmp,
    Counters* const counters) {
  assert(tf_i < GetNumTransforms(channel));
  const CodingParams* const params = GetCodingParams(channel);
  const QuantMtx& quant = segment.GetQuant(channel);
  BlockCoeffs16& dequantized = *tmp;

  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(channel)->split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);

  int32_t tf_res[kMaxBlockSizePix2];
  WP2Transform2D((*res)[tf_i], params->tf_x(), params->tf_y(), split_w, split_h,
                 tf_res, reduced_transform);
  if (channel == kUChannel || channel == kVChannel) {
    assert(params->tf == kDctDct && !params->split_tf);
    // Note DC is scaled by NumPix(dim) by WP2Transform2D. We don't
    // normalize it when propagating the error, even though the same value
    // will have a bigger impact on a smaller block than on a large one.
    // This is intentional: since we have fewer pixels in a smaller block,
    // we shift them more. But it would also be possible to normalize by a
    // factor of NumPix(dim) (multiply dc_error_ by this and devide
    // dc_error_next by the same factor). This would mean we apply the same
    // color shift regardless of the block size.
    tf_res[0] += dc_error_[channel];
    dc_error_next_[channel] = quant.DCError(tf_res[0], tdim(channel));
  }
  // Use some kind of coeff optimization (rd-opt based on av1 coeffs, or simple
  // dropout).
  const bool use_coeff_optim =
      (params->tf != kIdentityIdentity && config.effort >= 7);
  const bool use_av1_coeff_optim = (use_coeff_optim && context.use_aom() &&
                                    channel == kYChannel && config.effort >= 8);
  // Av1CoeffOptimization works better if quantization uses simple rounding
  // instead of biasing towards 0.
  const bool use_bias = !use_av1_coeff_optim;
  quant.Quantize(tf_res, tdim(channel), IsFirstCoeffDC(channel),
                 coeffs_[channel][tf_i], &num_coeffs_[channel][tf_i],
                 dequantized[tf_i], use_bias);

  if (use_coeff_optim) {
    if (use_av1_coeff_optim) {
      Av1CoeffOptimization(
          channel, params->tf, tdim(channel), quant, IsFirstCoeffDC(channel),
          tf_res, coeffs_[channel][tf_i], &num_coeffs_[channel][tf_i],
          dequantized[tf_i], counters->residuals(), /*rate_cost=*/nullptr);
    } else {
      DropoutCoeffs(tdim(channel), segment.quality_factor_,
                    coeffs_[channel][tf_i], &num_coeffs_[channel][tf_i],
                    dequantized[tf_i]);
    }
  }

  if (channel == kYChannel && mtx_set_ != nullptr) {
    mtx_set_->DecideUseRndMtx(this);  // TODO(skal): later: U/V channel?
  }

  Reconstruct(channel, tf_i, reduced_transform, &dequantized, prediction);
}

void CodedBlock::Reconstruct(Channel channel, uint32_t tf_i,
                             bool reduced_transform, BlockCoeffs16* const res,
                             const Plane16& prediction) const {
  assert(tf_i < GetNumTransforms(channel));
  const CodingParams& params = GetCodingParams(channel);
  const Rectangle tf_rect = blk().LocalTfRect(params.split_tf, tf_i);

  WP2InvTransform2D((*res)[tf_i], params.tf_x(), params.tf_y(), tf_rect.width,
                    tf_rect.height, reduced_transform);

  Plane16* const out = &out_.GetChannel(channel);
  Plane16 pred_view;
  if (prediction.IsEmpty()) {
    WP2_ASSERT_STATUS(pred_view.SetView(*out));
    PredictBlock(channel, tf_i, pred_view.Row(0), pred_view.Step());
  } else {
    WP2_ASSERT_STATUS(pred_view.SetView(prediction));
  }

  AddBlockFunc add_block = WP2::AddBlock[TrfLog2[tf_rect.width] - 2];
  const int32_t min = (channel == kAChannel) ? 0 : yuv_min_;
  const int32_t max = (channel == kAChannel) ? kAlphaMax : yuv_max_;

  add_block(&pred_view.At(tf_rect.x, tf_rect.y), pred_view.Step(), (*res)[tf_i],
            tf_rect.width, min, max, &out->At(tf_rect.x, tf_rect.y),
            out->Step(), tf_rect.height);

  if (params.split_tf && tf_i < 3) {
    context_cache_->Reset(/*only_small_right_or_bot_contexts=*/true);
  }
}

void CodedBlock::QuantizeAll(const EncoderConfig& config,
                             const BlockContext& context,
                             const Segment& segment, Channel channel,
                             bool reduced_transform, Counters* const counters) {
  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    Quantize(config, context, segment, channel, tf_i, reduced_transform,
             counters);
  }
}

void CodedBlock::QuantizeAllButFirst(const EncoderConfig& config,
                                     const BlockContext& context,
                                     const Segment& segment, Channel channel,
                                     bool reduced_transform,
                                     Counters* const counters) {
  for (uint32_t tf_i = 1; tf_i < GetNumTransforms(channel); ++tf_i) {
    Quantize(config, context, segment, channel, tf_i, reduced_transform,
             counters);
  }
}

void CodedBlock::Quantize(const EncoderConfig& config,
                          const BlockContext& context, const Segment& segment,
                          Channel channel, uint32_t tf_i,
                          bool reduced_transform, Counters* const counters) {
  BlockCoeffs16 res, tmp;
  Plane16* const out = &out_.GetChannel(channel);
  PredictBlock(channel, tf_i, out->Row(0), out->Step());
  GetResiduals(channel, tf_i, /*prediction=*/*out, &res);
  TransformAndReconstruct(config, context, segment, channel, tf_i,
                          reduced_transform,
                          /*prediction=*/*out, &res, &tmp, counters);
}

//------------------------------------------------------------------------------

float CodedBlock::ResidualRate(const BlockContext& context, Channel channel,
                               uint32_t num_channels,
                               SymbolCounter* const counter) const {
  if (context.use_aom()) {
    return ResidualWriter::GetRateAOM(*this, channel, counter);
  }
  // Whether to use the more accurate but a lot more expensive rate.
  // Even this one is not perfect because the cost depends on global
  // distributions which are not known at this time.
  static constexpr bool kUseExpensiveRate = false;
  float rate = 0.f;
  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    if (kUseExpensiveRate) {
      rate += ResidualWriter::GetRate(
          channel, num_channels, tdim(channel), coeffs_[channel][tf_i],
          num_coeffs_[channel][tf_i], IsFirstCoeffDC(channel), counter);
    } else {
      rate += ResidualWriter::GetPseudoRate(
          channel, num_channels, tdim(channel), coeffs_[channel][tf_i],
          num_coeffs_[channel][tf_i], IsFirstCoeffDC(channel), counter);
    }
  }
  return rate;
}

// Quantizes 'coeffs', then dequantizes, detransforms them and computes
// distortion compared to 'res'.
static uint32_t RoundTripDisto(CodedBlock* const cb, const QuantMtx& quant,
                               Channel channel, TrfSize tdim,
                               const int16_t* const res,
                               const int32_t* const coeffs, bool reduced,
                               int16_t* const quantized_coeffs,
                               uint32_t* const num_coeffs) {
  int16_t dequantized_coeffs[kMaxBlockSizePix2];
  quant.Quantize(coeffs, tdim, cb->IsFirstCoeffDC(channel), quantized_coeffs,
                 num_coeffs, dequantized_coeffs);
  const CodedBlock::CodingParams& params = *cb->GetCodingParams(channel);
  WP2InvTransform2D(dequantized_coeffs, params.tf_x(), params.tf_y(),
                    cb->w_pix(), cb->h_pix(), reduced);
  return WP2SumSquaredError16s(res, dequantized_coeffs, *num_coeffs);
}

// Computes the rate-distortion score for the U and V channels.
// 'res_u' and 'res_v' should be in frequency space. 'orig_u' and 'orig_v' are
// the original spatial residuals to compare against for distortion.
static WP2Status UVScore(const BlockContext& context,
                         const EncoderConfig& config, uint32_t tile_pos_x,
                         uint32_t tile_pos_y, bool has_alpha,
                         const QuantMtx& quant_u, const QuantMtx& quant_v,
                         const int16_t orig_u[kMaxBlockSizePix2],
                         const int16_t orig_v[kMaxBlockSizePix2],
                         const int32_t res_u[kMaxBlockSizePix2],
                         const int32_t res_v[kMaxBlockSizePix2], TrfSize dim,
                         bool reduced, SymbolCounter* const counter,
                         CodedBlock* const cb, float* score) {
  const float lambda_u = quant_u.lambda * 2.f;  // Another magic constant.
  const float lambda_v = quant_v.lambda * 2.f;

  uint32_t tmp_num_coeffs_u, tmp_num_coeffs_v;
  int16_t tmp_coeffs_u[kMaxBlockSizePix2];
  int16_t tmp_coeffs_v[kMaxBlockSizePix2];
  const uint32_t disto =
      RoundTripDisto(cb, quant_u, kUChannel, dim, orig_u, res_u, reduced,
                     tmp_coeffs_u, &tmp_num_coeffs_u) +
      RoundTripDisto(cb, quant_v, kVChannel, dim, orig_v, res_v, reduced,
                     tmp_coeffs_v, &tmp_num_coeffs_v);
  // early out if there's ~only DC
  if (!reduced && tmp_num_coeffs_u <= 1 && tmp_num_coeffs_v <= 1) {
    WP2_CHECK_REDUCED_STATUS(
        cb->Store420Scores(config, tile_pos_x, tile_pos_y, lambda_u, lambda_v,
                           reduced, disto, 0, 0));
    *score = 0;
    return WP2_STATUS_OK;
  }
  const uint32_t num_channels = (has_alpha ? 4 : 3);
  const float rate_u =
      cb->ResidualRate(context, kUChannel, num_channels, counter);
  const float rate_v =
      cb->ResidualRate(context, kVChannel, num_channels, counter);
  *score = disto + lambda_u * rate_u + lambda_v * rate_v;
  WP2_CHECK_REDUCED_STATUS(cb->Store420Scores(config, tile_pos_x, tile_pos_y,
                                              lambda_u, lambda_v, reduced,
                                              disto, rate_u, rate_v));
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::DecideChromaSubsampling(
    const EncoderConfig& config, const BlockContext& context,
    uint32_t tile_pos_x, uint32_t tile_pos_y, bool has_alpha,
    const QuantMtx& quant_u, const QuantMtx& quant_v,
    Counters* const counters) {
  assert(!GetCodingParams(kUChannel)->split_tf);
  assert(!GetCodingParams(kVChannel)->split_tf);
  const BlockSize dim = blk_.dim();

  const TrfSize full_dim = kFullDim[dim];
  const TrfSize half_dim = kHalfDim[dim];
  const uint32_t w_pix = blk_.w_pix();
  const uint32_t h_pix = blk_.h_pix();
  SubtractBlockFunc sub_block = WP2::SubtractBlock[TrfLog2[w_pix] - 2];

  const CodingParams* const u_params = GetCodingParams(kUChannel);
  const CodingParams* const v_params = GetCodingParams(kVChannel);

  int16_t res_u[kMaxBlockSizePix2], res_v[kMaxBlockSizePix2];
  {
    int16_t prediction[kMaxBlockSizePix2];
    u_params->pred->Predict(*this, kUChannel, /*split_tf=*/false, /*tf_i=*/0,
                            prediction, w_pix);
    sub_block(in_.U.Row(0), in_.U.Step(), prediction, /*preds_step=*/w_pix,
              res_u, /*dst_step=*/w_pix, h_pix);
    v_params->pred->Predict(*this, kVChannel, /*split_tf=*/false, /*tf_i=*/0,
                            prediction, w_pix);
    sub_block(in_.V.Row(0), in_.V.Step(), prediction, /*preds_step=*/w_pix,
              res_v, /*dst_step=*/w_pix, h_pix);
  }

  int32_t tmp_u[kMaxBlockSizePix2], tmp_v[kMaxBlockSizePix2];
  // transform at full resolution
  WP2Transform2D(res_u, u_params->tf_x(), u_params->tf_y(), w_pix, h_pix,
                 tmp_u);
  WP2Transform2D(res_v, v_params->tf_x(), v_params->tf_y(), w_pix, h_pix,
                 tmp_v);

  float score444;
  WP2_CHECK_STATUS(UVScore(context, config, tile_pos_x, tile_pos_y, has_alpha,
                           quant_u, quant_v, res_u, res_v, tmp_u, tmp_v,
                           full_dim, /*reduced=*/false, counters->residuals(),
                           this, &score444));
  if (score444 == 0) {
    is420_ = false;
    WP2_CHECK_REDUCED_STATUS(Store420Decision(config, tile_pos_x, tile_pos_y,
                                              Debug420Decision::k444EarlyExit));
    return WP2_STATUS_OK;
  }

  // inspect half-resolution now
  WP2ReduceCoeffs(tmp_u, w_pix, h_pix, tmp_v);
  WP2ReduceCoeffs(tmp_u, w_pix, h_pix, tmp_v);
  float score420;
  WP2_CHECK_STATUS(UVScore(context, config, tile_pos_x, tile_pos_y, has_alpha,
                           quant_u, quant_v, res_u, res_v, tmp_u, tmp_v,
                           half_dim, /*reduced=*/true, counters->residuals(),
                           this, &score420));

  is420_ = (score420 < score444);
  WP2_CHECK_REDUCED_STATUS(Store420Decision(
      config, tile_pos_x, tile_pos_y,
      is420_ ? Debug420Decision::k420 : Debug420Decision::k444));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

float CodedBlock::PredictorRate(Channel channel,
                                SymbolCounter* const counter) const {
  counter->Clear();
  ANSEncCounter enc;
  SyntaxWriter::WritePredictors(*this, channel, counter, &enc);
  return enc.GetCost();
}

float CodedBlock::TransformRate(Channel channel,
                                SymbolCounter* const counter) const {
  counter->Clear();
  ANSEncCounter enc;
  SyntaxWriter::WriteSplitTransform(*this, channel, counter, &enc);
  SyntaxWriter::WriteHasCoeffs(*this, channel, counter, &enc);
  SyntaxWriter::WriteTransform(*this, channel, counter, &enc);
  return enc.GetCost();
}

float CodedBlock::SegmentIdRate(const BlockContext& context,
                                SymbolCounter* const counter) const {
  counter->Clear();
  ANSEncCounter enc;
  context.segment_id_predictor().WriteId(*this, counter, &enc);
  return enc.GetCost();
}

//------------------------------------------------------------------------------

namespace {

// Returns the index of 'channel' in one of the following sets:
//   {luma Y}, {chroma U, chroma V} or {alpha}
uint32_t GetChannelIndex(Channel channel) {
  return (channel == kVChannel) ? 1 : 0;
}

}  // namespace

float BlockRates::GetScore() const {
  return distortion +
         lambda * ((predictor_rate.is_defined ? predictor_rate.rate : 0.f) +
                   (transform_rate.is_defined ? transform_rate.rate : 0.f) +
                   (segment_id_rate.is_defined ? segment_id_rate.rate : 0.f) +
                   (residuals_rate.is_defined ? residuals_rate.rate : 0.f));
}

bool BlockRates::HasSameScoreFormulaAs(const BlockRates& other) const {
  return predictor_rate.is_defined == other.predictor_rate.is_defined &&
         transform_rate.is_defined == other.transform_rate.is_defined &&
         segment_id_rate.is_defined == other.segment_id_rate.is_defined &&
         residuals_rate.is_defined == other.residuals_rate.is_defined &&
         lambda == other.lambda;
}

bool BlockRates::HasScoreFormulaSubsetOf(const BlockRates& other) const {
  return (!predictor_rate.is_defined || other.predictor_rate.is_defined) &&
         (!transform_rate.is_defined || other.transform_rate.is_defined) &&
         (!segment_id_rate.is_defined || other.segment_id_rate.is_defined) &&
         (!residuals_rate.is_defined || other.residuals_rate.is_defined) &&
         (lambda == 0.f || lambda == other.lambda);
}

float BlockScore::GetScore() const {
  return results[0].GetScore() + results[1].GetScore();
}

float BlockScore::GetDistortion() const {
  return results[0].distortion + results[1].distortion;
}

bool BlockScore::IsBetterThan(const BlockScore& other) const {
  // Make sure the scores are comparable.
  assert(results[0].HasSameScoreFormulaAs(other.results[0]));
  assert(results[1].HasSameScoreFormulaAs(other.results[1]));
  return (GetScore() < other.GetScore());
}

bool BlockScore::MightBeBetterThan(const BlockScore& other) const {
  // The overall GetScore() can only increase when more rates are defined so a
  // partial comparison is valid, as long as 'other' does not change after.
  assert(results[0].HasScoreFormulaSubsetOf(other.results[0]));
  assert(results[1].HasScoreFormulaSubsetOf(other.results[1]));
  return (GetScore() < other.GetScore());
}

//------------------------------------------------------------------------------

WP2Status BlockScorer::Init(const EncoderConfig& config,
                            const GlobalParams& gparams, Rectangle tile_rect) {
  config_ = &config;
  gparams_ = &gparams;
  tile_rect_ = tile_rect;

  WP2_CHECK_ALLOC_OK(refined_preds_.resize(
      std::max({kYPredModeNum, kUVPredModeNum, kAPredModeNum})));
  return WP2_STATUS_OK;
}

void BlockScorer::Set(const BlockContext& context, CodedBlock* cb,
                      Counters* counters) {
  cb_ = cb;
  context_ = &context;
  counters_ = counters;
  // Use the segment given by AssignSegmentId() to compute the lambdas because
  // we may end up comparing several 'segment_ids_' which would lead to
  // different score computation formulas otherwise.
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    const Segment& segment =
        gparams_->segments_[(channel == kAChannel) ? 0 : cb_->id_];
    lambdas_[channel] = cb_->lambda_mult_ * segment.GetQuant(channel).lambda;
  }

  cached_predictor_ = nullptr;
  last_params_ = CodedBlock::CodingParams();

  best_combination_ = BlockScore();
}

WP2Status BlockScorer::ComputeScore(const VectorNoCtor<bool>** refined_preds) {
  refine_ = (refined_preds != nullptr);
  best_combination_no_angle_ = BlockScore();
  best_combinations_angle_.clear();

  CodedBlock::CodingParams& params = *cb_->GetCodingParams(channels_.front());
  const bool can_split =
      (channels_.front() == kYChannel &&
       GetSplitSize(cb_->dim(), /*split=*/true) != cb_->dim());
  const CodedBlock::SplitTf forced_split =
      can_split ? cb_->GetForcedSplitTf(*config_, tile_rect_)
                : CodedBlock::SplitTf::kUnknown;

  if (forced_split != CodedBlock::SplitTf::kUnknown) {
    // Set by EncoderConfig::info (debug).
    params.split_tf = (forced_split == CodedBlock::SplitTf::kForcedSplit);
    WP2_CHECK_STATUS(ComputeScoreForEachPredictor(/*forced=*/true));
  } else {
    for (const bool split_tf : splits_) {
      if (split_tf && !can_split) continue;
      params.split_tf = split_tf;
      WP2_CHECK_STATUS(ComputeScoreForEachPredictor(/*forced=*/false));
    }
  }

  assert(best_combination_.IsDefined());

  if (refine_) {
    RefineBestAnglePredictors();
    *refined_preds = &refined_preds_;
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScorer::ComputeScoreForEachPredictor(bool forced) {
  const Predictors& preds = gparams_->GetPredictors(channels_.front());
  CodedBlock::CodingParams& params = *cb_->GetCodingParams(channels_.front());
  const Predictor* const forced_pred =
      cb_->GetForcedPredictor(*config_, tile_rect_, preds, channels_.front());

  if (channels_.front() == kYChannel && cb_->y_context_is_constant_) {
    // If the context is constant, all luma predictors will predict the same
    // values (plus or minus 1, because of rounding errors), so we just force
    // the predictor to 0 and we do not write it to the bitstream. We cannot
    // do the same for UV and alpha because the chroma from luma predictor
    // does not have this property, but we could send a single bit saying
    // whether the predictor is cfl or "other".
    params.pred = preds.GetPred(0);  // Assumes it is the DC predictor.
    WP2_CHECK_STATUS(ComputeScoreForEachTransform(forced));
  } else if (forced_pred != nullptr) {
    // Predictor is set by EncoderConfig::info (debug).
    params.pred = forced_pred;
    WP2_CHECK_STATUS(ComputeScoreForEachTransform(/*forced=*/true));
  } else {
    for (const Predictor* pred : predictors_) {
      params.pred = pred;
      WP2_CHECK_STATUS(ComputeScoreForEachTransform(forced));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScorer::ComputeScoreForEachTransform(bool forced) {
  CodedBlock::CodingParams& params = *cb_->GetCodingParams(channels_.front());
  for (Channel c : channels_) {
    if (!params.pred->ComputeParams(cb_, c)) {
      // TODO(skal): deal with useless preds
    }
  }
  const TransformPair forced_tf = cb_->GetForcedTransform(*config_, tile_rect_);

  if (forced_tf != kUnknownTf) {
    // Transform is set by EncoderConfig::info (debug).
    params.tf = forced_tf;
    WP2_CHECK_STATUS(ComputeScoreForEachSegment(/*forced=*/true));
  } else {
    for (const TransformPair tf : transforms_) {
      params.tf = tf;
      WP2_CHECK_STATUS(ComputeScoreForEachSegment(forced));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScorer::ComputeScoreForEachSegment(bool forced) {
  if (segment_ids_.empty()) {
    // Just use the 'cb_->id_' value which was set by AssignSegmentId().
    WP2_CHECK_STATUS(ComputeScoreForEachChannel(forced));
  } else {
    for (uint8_t segment_id : segment_ids_) {
      cb_->id_ = segment_id;
      WP2_CHECK_STATUS(ComputeScoreForEachChannel(forced));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status BlockScorer::ComputeScoreForEachChannel(bool forced) {
  BlockScore combination;
  combination.params = *cb_->GetCodingParams(channels_.front());
  combination.segment_id = cb_->id_;
  if (!forced) {
    // Each unique combination should be tried only once.
    assert(combination.params != last_params_ ||
           combination.segment_id != last_segment_id_);
    assert(combination.params != best_combination_.params ||
           combination.segment_id != best_combination_.segment_id);
  }

  // Recompute the prediction of the top left sub-block if needed.
  // TODO(yguyon): See if the prediction of !split can be reused for split.
  const bool recompute_top_left_residuals =
      (cached_predictor_ != combination.params.pred ||
       cached_split_tf_ != combination.params.split_tf);
  const uint32_t num_channels = (gparams_->maybe_use_lossy_alpha_ ? 4 : 3);
  const bool reduced_tf = (channels_.front() == kUChannel && cb_->is420_);
  const Segment& segment =
      gparams_->segments_[(channels_.front() == kAChannel) ? 0 : cb_->id_];
  bool skipped = false;

  for (Channel c : channels_) {
    const uint32_t ci = GetChannelIndex(c);
    BlockRates& results = combination.results[ci];
    results.lambda = lambdas_[c];

    Plane16 prediction;
    WP2_ASSERT_STATUS(prediction.SetView(prediction_cache_[ci][0],
                                         kMaxBlockSizePix, kMaxBlockSizePix,
                                         kMaxBlockSizePix));
    BlockCoeffs16& residuals_cache = residuals_cache_[ci];
    if (recompute_top_left_residuals) {
      cb_->PredictBlock(c, /*tf_i=*/0, prediction.Row(0), prediction.Step());
      cb_->GetResiduals(c, /*tf_i=*/0, prediction, &residuals_cache);

      if (c == channels_.back()) {  // Remember what is cached for all channels.
        cached_predictor_ = combination.params.pred;
        cached_split_tf_ = combination.params.split_tf;
      } else {
        cached_predictor_ = nullptr;  // Modified but incomplete.
      }
    }
    BlockCoeffs16 residuals, reconstructed;  // Temporary buffers.
    // At least one of {pred, tf, segment} changed, so always reconstruct the
    // top left sub-block...
    cb_->TransformAndReconstruct(*config_, *context_, segment, c, /*tf_i=*/0,
                                 reduced_tf, prediction, &residuals_cache,
                                 &reconstructed, counters_);
    // TODO(yguyon): Transform output may be cached if only the segment changed.

    // ... and predict / reconstruct the other sub-blocks (that depend on
    // the results of top left one), if any.
    for (uint32_t tf_i = 1; tf_i < cb_->GetNumTransforms(c); ++tf_i) {
      cb_->PredictBlock(c, tf_i, prediction.Row(0), prediction.Step());
      cb_->GetResiduals(c, tf_i, prediction, &residuals);
      cb_->TransformAndReconstruct(*config_, *context_, segment, c, tf_i,
                                   reduced_tf, prediction, &residuals,
                                   &reconstructed, counters_);
    }
    if (c == channels_.back()) {  // Remember what is cached for all channels.
      last_params_ = combination.params;
      last_segment_id_ = combination.segment_id;
    } else {
      last_params_ = CodedBlock::CodingParams();  // Modified but incomplete.
    }

    // Compute distortion.
    results.distortion = cb_->GetDisto(c);

    // If the distortion is too high for this 'combination' to ever be the best,
    // and if there is no need to refine it later, discard it immediately to
    // avoid estimating unnecessary bit costs.
    if (CanBeSkipped(combination)) {
      skipped = true;
      break;
    }

    // Estimate the cost of this configuration.
    // TODO(yguyon): some rates may be skipped if all combinations lead to the
    //               same partial cost.
    results.predictor_rate = {/*is_defined=*/true,
                              cb_->PredictorRate(c, counters_->predictor())};
    results.transform_rate = {/*is_defined=*/true,
                              cb_->TransformRate(c, counters_->transform())};
    if (c == kYChannel) {
      // Segment ID is decided for luma only and signaled once per block.
      results.segment_id_rate = {
          /*is_defined=*/true,
          cb_->SegmentIdRate(*context_, counters_->segment_id())};
    } else {
      assert(segment_ids_.empty() && !results.segment_id_rate.is_defined);
    }

    // The 'residuals_rate' is the costliest to estimate.
    // Compute it last and only if needed.
    if (CanBeSkipped(combination)) {
      skipped = true;
      break;
    }
    results.residuals_rate = {
        /*is_defined=*/true,
        cb_->ResidualRate(*context_, c, num_channels, counters_->residuals())};
  }

  const bool is_best =
      !skipped && (!best_combination_.IsDefined() ||
                   combination.IsBetterThan(best_combination_));
  if (is_best) {
    best_combination_ = combination;
  }

  if (refine_) {
    if (combination.params.pred->IsAngle(nullptr)) {
      assert(!skipped);
      WP2_CHECK_ALLOC_OK(best_combinations_angle_.push_back(combination));
    } else if (!skipped &&
               (!best_combination_no_angle_.IsDefined() ||
                combination.IsBetterThan(best_combination_no_angle_))) {
      best_combination_no_angle_ = combination;
    }
  }

#if !defined(WP2_REDUCE_BINARY_SIZE)
  // Store scores for debug.
  for (Channel c : channels_) {
    cb_->StorePredictionScore(*config_, tile_rect_, c, *combination.params.pred,
                              combination.params.tf, combination.segment_id,
                              combination.results[GetChannelIndex(c)], is_best);
  }
#endif  // WP2_REDUCE_BINARY_SIZE

  return WP2_STATUS_OK;
}

void BlockScorer::RefineBestAnglePredictors() {
  assert(refine_);

  // Set all predictors to "should-not-be-evaluated".
  std::fill(refined_preds_.begin(), refined_preds_.end(), false);
  // Only angle predictors are refined and returned.
  if (best_combinations_angle_.empty()) return;

  // Choose to refine all the main angle predictors within a percentage of the
  // lowest score. This helps refine the predictors with similar scores.
  std::sort(best_combinations_angle_.begin(), best_combinations_angle_.end());
  const float min_score = best_combinations_angle_.front().GetScore();  // Best.
  const float max_score = best_combinations_angle_.back().GetScore();  // Worst.
  const float fraction = config_->effort / 9.f;  // Max effort includes all.
  const float min_allowed_score =
      min_score * (1.f - fraction) + max_score * fraction + 1.f;
  assert(config_->effort != 9 || max_score <= min_allowed_score);

  for (const BlockScore& main_angle : best_combinations_angle_) {
    if (!best_combination_no_angle_.IsBetterThan(main_angle) ||
        main_angle.GetDistortion() <=
            best_combination_no_angle_.GetDistortion() ||
        main_angle.GetScore() <= min_allowed_score) {
      assert(main_angle.params.pred->mode() < refined_preds_.size());
      refined_preds_[main_angle.params.pred->mode()] = true;
    }
  }
}

bool BlockScorer::CanBeSkipped(const BlockScore& combination) const {
  if (refine_) {
    return !combination.params.pred->IsAngle(nullptr) &&
           best_combination_no_angle_.IsDefined() &&
           !combination.MightBeBetterThan(best_combination_no_angle_);
  }
  return best_combination_.IsDefined() &&
         !combination.MightBeBetterThan(best_combination_);
}

const BlockScore& BlockScorer::GetBestCombination() const {
  return best_combination_;
}

void BlockScorer::ReconstructBestCombination() {
  assert(best_combination_.IsDefined());
  if (best_combination_.params != last_params_ ||
      best_combination_.segment_id != last_segment_id_) {
    *cb_->GetCodingParams(channels_.front()) = best_combination_.params;
    cb_->id_ = best_combination_.segment_id;
    const Segment& segment =
        gparams_->segments_[(channels_.front() == kAChannel) ? 0 : cb_->id_];
    for (Channel c : channels_) {
      const bool is420 = ((c == kUChannel || c == kVChannel) && cb_->is420_);
      cb_->QuantizeAll(*config_, *context_, segment, c, is420, counters_);
    }
  }
}

//------------------------------------------------------------------------------

WP2Status OptimizeModes(const EncoderConfig& config, const Rectangle& tile_rect,
                        Channel channel, const Predictors& preds,
                        const BlockModes& m, const BlockContext& context,
                        CodedBlock* const cb, Counters* const counters,
                        BlockScorer* const scorer) {
  scorer->Set(context, cb, counters);

  // Either luma, chroma or alpha.
  WP2_CHECK_ALLOC_OK(scorer->channels_.resize((channel == kUChannel) ? 2 : 1));
  scorer->channels_.front() = channel;
  if (channel == kUChannel) scorer->channels_.back() = kVChannel;
  const bool explicit_segment_id =
      scorer->GetGlobalParams().explicit_segment_ids_ && !cb->blk().IsSmall();

  // First round of combinations.
  WP2_CHECK_ALLOC_OK(scorer->splits_.copy_from(m.splits_tried_during_preds));
  WP2_CHECK_ALLOC_OK(scorer->predictors_.copy_from(m.main_preds));
  WP2_CHECK_ALLOC_OK(scorer->transforms_.copy_from(m.tf_tried_during_preds));
  scorer->segment_ids_.clear();
  if (explicit_segment_id) {
    WP2_CHECK_ALLOC_OK(
        scorer->segment_ids_.copy_from(m.segment_ids_tried_during_preds));
  }

  if (m.sub_preds.empty()) {
    WP2_CHECK_STATUS(scorer->ComputeScore());
  } else {
    const VectorNoCtor<bool>* refined_preds = nullptr;
    WP2_CHECK_STATUS(scorer->ComputeScore(&refined_preds));

    // Find all sub-angle predictors sharing the same mode (~angle) as the
    // best main-angle predictors.
    scorer->predictors_.clear();
    for (const Predictor* const sub_pred : m.sub_preds) {
      assert(sub_pred->IsAngle(nullptr));
      assert(std::find(m.main_preds.begin(), m.main_preds.end(), sub_pred) ==
             m.main_preds.end());
      if (refined_preds->at(sub_pred->mode())) {
        assert(std::find(scorer->predictors_.begin(), scorer->predictors_.end(),
                         sub_pred) == scorer->predictors_.end());
        WP2_CHECK_ALLOC_OK(scorer->predictors_.push_back(sub_pred));
      }
    }
    if (!scorer->predictors_.empty()) {
      WP2_CHECK_STATUS(scorer->ComputeScore());
    }
  }

  // Keep only the best of each from now on and try the others one by one.
  WP2_CHECK_ALLOC_OK(scorer->splits_.resize(1));
  scorer->splits_.front() = scorer->GetBestCombination().params.split_tf;
  WP2_CHECK_ALLOC_OK(scorer->predictors_.resize(1));
  scorer->predictors_.front() = scorer->GetBestCombination().params.pred;
  if (scorer->segment_ids_.empty()) {
    // Keep AssignSegmentId() value.
    assert(cb->id_ == scorer->GetBestCombination().segment_id);
  } else {
    WP2_CHECK_ALLOC_OK(scorer->segment_ids_.resize(1));
    scorer->segment_ids_.front() = scorer->GetBestCombination().segment_id;
  }

  // Try other transforms.
  if (!m.tf_tried_after_preds.empty()) {
    WP2_CHECK_ALLOC_OK(scorer->transforms_.copy_from(m.tf_tried_after_preds));
    WP2_CHECK_STATUS(scorer->ComputeScore());
    // Keep only the best transform from now on.
    WP2_CHECK_ALLOC_OK(scorer->transforms_.resize(1));
    scorer->transforms_.front() = scorer->GetBestCombination().params.tf;
  }

  // Try other segments.
  scorer->segment_ids_.clear();
  for (uint8_t segment_id : m.segment_ids_tried_after_preds) {
    // AssignSegmentId() could have returned a segment present in
    // 'm.segment_ids_tried_after_preds' so make sure there is no overlap.
    if (explicit_segment_id &&
        segment_id != scorer->GetBestCombination().segment_id) {
      WP2_CHECK_ALLOC_OK(scorer->segment_ids_.push_back(segment_id));
    }
  }
  if (!scorer->segment_ids_.empty()) {
    WP2_CHECK_STATUS(scorer->ComputeScore());
    // Keep only the best segment from now on.
    WP2_CHECK_ALLOC_OK(scorer->segment_ids_.resize(1));
    scorer->segment_ids_.front() = scorer->GetBestCombination().segment_id;
  }

  // Try other split_tf.
  if (!m.splits_tried_after_preds.empty()) {
    WP2_CHECK_ALLOC_OK(scorer->splits_.copy_from(m.splits_tried_after_preds));
    WP2_CHECK_STATUS(scorer->ComputeScore());
  }

  scorer->ReconstructBestCombination();
  return WP2_STATUS_OK;
}

WP2Status OptimizeModesChroma(
    const EncoderConfig& config, const Rectangle& tile_rect, bool has_alpha,
    const FrontMgrBase& mgr, const Predictors& preds,
    ChromaSubsampling chroma_subsampling, const BlockModes& modes,
    const BlockContext& context, CodedBlock* const cb, Counters* const counters,
    DCDiffusionMap* const dc_error_u, DCDiffusionMap* const dc_error_v,
    BlockScorer* const scorer) {
  const uint32_t diffusion =
      DCDiffusionMap::GetDiffusion(config.error_diffusion);
  const bool is420_for_sure =
      ((chroma_subsampling == ChromaSubsampling::kSingleBlock &&
        config.uv_mode == EncoderConfig::UVMode420) ||
       chroma_subsampling == ChromaSubsampling::k420);
  const bool is444_for_sure =
      ((chroma_subsampling == ChromaSubsampling::kSingleBlock &&
        config.uv_mode == EncoderConfig::UVMode444) ||
       chroma_subsampling == ChromaSubsampling::k444);
  const Segment& segment = scorer->GetGlobalParams().segments_[cb->id_];

  cb->is420_ = is420_for_sure;  // Must be set to something now even if unsure.
  WP2_CHECK_STATUS(OptimizeModes(config, tile_rect, kUChannel, preds, modes,
                                 context, cb, counters, scorer));
  if (diffusion > 0) {  // Add error diffusion
    cb->dc_error_[kUChannel] = dc_error_u->Get(cb->blk(), diffusion);
    cb->dc_error_[kVChannel] = dc_error_v->Get(cb->blk(), diffusion);
  }

  // Choose the final value of 'cb->is420_'.
  // TODO(skal): decide based on coded luma too?
  if (is420_for_sure) {
    cb->is420_ = true;
  } else if (is444_for_sure) {
    cb->is420_ = false;
  } else {
    assert(chroma_subsampling == ChromaSubsampling::kSingleBlock ||
           chroma_subsampling == ChromaSubsampling::kAdaptive);
    WP2_CHECK_STATUS(cb->DecideChromaSubsampling(
        config, context, tile_rect.x, tile_rect.y, has_alpha, segment.quant_u_,
        segment.quant_v_, counters));
  }
  if (diffusion > 0 || (!is420_for_sure && !is444_for_sure)) {
    cb->QuantizeAll(config, context, segment, kUChannel, cb->is420_, counters);
    cb->QuantizeAll(config, context, segment, kVChannel, cb->is420_, counters);
  }
  if (diffusion > 0) {
    dc_error_u->Store(mgr, cb->blk(), cb->dc_error_next_[kUChannel]);
    dc_error_v->Store(mgr, cb->blk(), cb->dc_error_next_[kVChannel]);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
