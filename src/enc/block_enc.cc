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

#include "src/common/symbols.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/front_mgr.h"
#include "src/utils/utils.h"

namespace WP2 {

//------------------------------------------------------------------------------
// Counters

WP2Status Counters::Init(uint32_t effort, const SymbolRecorder& recorder) {
  effort_ = effort;
  recorder_ = &recorder;

  transform_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
  WP2_CHECK_ALLOC_OK(transform_ != nullptr);
  WP2_CHECK_STATUS(transform_->Allocate(
      {kSymbolHasCoeffs, kSymbolTransform, kSymbolSplitTransform}));

  predictor_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
  WP2_CHECK_ALLOC_OK(predictor_ != nullptr);
  WP2_CHECK_STATUS(
      predictor_->Allocate({kSymbolModeY, kSymbolModeUV, kSymbolModeA}));

  if (effort >= kSlowCounterEffortThreshold) {
    // Using the UpdatingSymbolCounter slows down encoding by ~9%
    residuals_slow_.reset(new (WP2Allocable::nothrow)
                              UpdatingSymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_slow_ != nullptr);
    residuals_ = residuals_slow_.get();
    residuals_aom_slow_.reset(new (WP2Allocable::nothrow)
                                  UpdatingSymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_aom_slow_ != nullptr);
    residuals_aom_ = residuals_aom_slow_.get();
  } else {
    residuals_fast_.reset(new (WP2Allocable::nothrow) SymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_fast_ != nullptr);
    residuals_ = residuals_fast_.get();
    residuals_aom_fast_.reset(new (WP2Allocable::nothrow)
                                  SymbolCounter(recorder_));
    WP2_CHECK_ALLOC_OK(residuals_aom_fast_ != nullptr);
    residuals_aom_ = residuals_aom_fast_.get();
  }
  WP2_CHECK_STATUS(residuals()->Allocate(
      {kSymbolDC, kSymbolResidualUseBounds, kSymbolResidualBound1IsX,
       kSymbolResidualUseBound2, kSymbolResidualIsZero, kSymbolResidualIsOne,
       kSymbolResidualIsTwo, kSymbolResidualEndOfBlock,
       kSymbolResidualHasOnlyOnesLeft}));
  WP2_CHECK_STATUS(residuals_aom()->Allocate(
      {kSymbolTransform, kAOMEOBPT4, kAOMEOBPT8, kAOMEOBPT16, kAOMEOBPT32,
       kAOMEOBPT64, kAOMEOBPT128, kAOMEOBPT256, kAOMEOBPT512, kAOMEOBPT1024,
       kAOMEOBExtra, kAOMCoeffBaseEOB, kAOMCoeffBase, kAOMCoeffBaseRange}));
  return WP2_STATUS_OK;
}

WP2Status Counters::CopyFrom(const Counters& other,
                             const SymbolRecorder& recorder) {
  if (&recorder != recorder_) {
    WP2_CHECK_STATUS(Init(other.effort_, recorder));
  } else {
    // Init() was already called.
    assert(transform_ != nullptr && residuals_ != nullptr &&
           residuals_aom_ != nullptr && predictor_ != nullptr);
  }
  if (residuals_slow_ != nullptr) {
    WP2_CHECK_STATUS(residuals_slow_->CopyFrom(*other.residuals_slow_));
  }
  if (residuals_aom_slow_ != nullptr) {
    WP2_CHECK_STATUS(residuals_aom_slow_->CopyFrom(*other.residuals_aom_slow_));
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

uint32_t CodedBlock::GetDisto(Channel channel) const {
  const Plane16& p_in = in_.GetChannel(channel);
  const Plane16& p_out = out_.GetChannel(channel);
  // TODO(skal): experiment calling WP2SumSquaredErrorHalfBlock for U/V channel
  return WP2SumSquaredErrorBlock(p_in.Row(0), p_in.Step(), p_out.Row(0),
                                 p_out.Step(), w_pix(), h_pix());
}

void CodedBlock::GetResiduals(Channel channel, const Plane16& prediction,
                              BlockCoeffs16* const res) const {
  const Plane16& in = in_.GetChannel(channel);
  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(channel).split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  SubtractBlockFunc sub_block = WP2::SubtractBlock[TrfLog2[split_w] - 2];
  uint32_t tf_i = 0;
  for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
      sub_block(&in.At(split_x, split_y), in.Step(),
                &prediction.At(split_x, split_y), prediction.Step(),
                (*res)[tf_i], split_w, split_h);

#if defined(WP2_BITTRACE)
      if (original_res_ != nullptr) {
        memcpy((*original_res_)[channel][tf_i], &(*res)[tf_i][0],
               split_w * split_h * sizeof((*res)[tf_i][0]));
      }
#endif
      ++tf_i;
    }
  }
}

void CodedBlock::TransformAndReconstruct(const QuantMtx& quant, Channel channel,
                                         bool reduced_transform,
                                         const Plane16& prediction,
                                         const BlockCoeffs16* const res,
                                         BlockCoeffs16* const tmp) {
  const CodingParams* const params = GetCodingParams(channel);

  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(channel)->split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    int32_t tf_res[kMaxBlockSizePix2];
    WP2Transform2D((*res)[tf_i], params->tf_x(), params->tf_y(), split_w,
                   split_h, tf_res, reduced_transform);
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
    quant.Quantize(tf_res, tdim(channel), IsFirstCoeffDC(channel),
                   coeffs_[channel][tf_i], &num_coeffs_[channel][tf_i],
                   (*tmp)[tf_i]);

    if (channel == kYChannel && mtx_set_ != nullptr) {
      mtx_set_->DecideUseRndMtx(this);  // TODO(skal): later: U/V channel?
    }
  }
  Reconstruct(channel, reduced_transform, tmp, prediction);
}

void CodedBlock::Reconstruct(Channel channel, bool reduced_transform,
                             BlockCoeffs16* const res,
                             const Plane16& prediction) const {
  const CodingParams& params = GetCodingParams(channel);
  const BlockSize split_size = GetSplitSize(dim(), params.split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    WP2InvTransform2D((*res)[tf_i], params.tf_x(), params.tf_y(), split_w,
                      split_h, reduced_transform);
  }

  Plane16* const out = &out_.GetChannel(channel);
  Plane16 pred_view;
  if (prediction.IsEmpty()) {
    WP2_ASSERT_STATUS(pred_view.SetView(*out));
    PredictBlock(channel, pred_view.Row(0), pred_view.Step());
  } else {
    WP2_ASSERT_STATUS(pred_view.SetView(prediction));
  }

  AddBlockFunc add_block = WP2::AddBlock[TrfLog2[split_w] - 2];
  const int32_t min = (channel == kAChannel) ? 0 : yuv_min_;
  const int32_t max = (channel == kAChannel) ? kAlphaMax : yuv_max_;
  for (uint32_t tf_i = 0, split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w, ++tf_i) {
      add_block(&pred_view.At(split_x, split_y), pred_view.Step(), (*res)[tf_i],
                split_w, min, max, &out->At(split_x, split_y), out->Step(),
                split_h);
    }
  }
}

void CodedBlock::Quantize(const QuantMtx& quant, Channel channel,
                          bool reduced_transform) {
  BlockCoeffs16 res, tmp;
  Plane16* const out = &out_.GetChannel(channel);
  PredictBlock(channel, out->Row(0), out->Step());
  GetResiduals(channel, /*prediction=*/*out, &res);
  TransformAndReconstruct(quant, channel, reduced_transform,
                          /*prediction=*/*out, &res, &tmp);
}

//------------------------------------------------------------------------------

WP2Status CodedBlock::ResidualRate(const BlockContext& context, Channel channel,
                                   uint32_t num_channels,
                                   SymbolCounter* const counter,
                                   SymbolCounter* const aom_counter,
                                   float* rate) const {
  const bool is_uv = (channel == kUChannel || channel == kVChannel);
  // TODO(vrabaud) Using AOM context for U/V is worse for now...
  if (context.use_aom() && !is_uv) {
    WP2_CHECK_STATUS(ResidualWriter::GetRateAOM(*this, channel,
                                                aom_counter, rate));
  } else {
    // Whether to use the more accurate but a lot more expensive rate.
    // Even this one is not perfect because the cost depends on global
    // distributions which are not known at this time.
    static constexpr bool kUseExpensiveRate = false;
    *rate = 0.f;
    for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
      float tf_rate;
      if (kUseExpensiveRate) {
        WP2_CHECK_STATUS(ResidualWriter::GetRate(
            channel, num_channels, tdim(channel), coeffs_[channel][tf_i],
            num_coeffs_[channel][tf_i], IsFirstCoeffDC(channel), counter,
            &tf_rate));
      } else {
        WP2_CHECK_STATUS(ResidualWriter::GetPseudoRate(
            channel, num_channels, tdim(channel), coeffs_[channel][tf_i],
            num_coeffs_[channel][tf_i], IsFirstCoeffDC(channel), counter,
            &tf_rate));
      }
      *rate += tf_rate;
    }
  }
  return WP2_STATUS_OK;
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
static WP2Status UVScore(const EncoderConfig& config, uint32_t tile_pos_x,
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
  float rate_u, rate_v;
  WP2_CHECK_STATUS(ResidualWriter::GetPseudoRate(
      kUChannel, num_channels, dim, tmp_coeffs_u, tmp_num_coeffs_u,
      cb->IsFirstCoeffDC(kUChannel), counter, &rate_u));
  WP2_CHECK_STATUS(ResidualWriter::GetPseudoRate(
      kVChannel, num_channels, dim, tmp_coeffs_v, tmp_num_coeffs_v,
      cb->IsFirstCoeffDC(kVChannel), counter, &rate_v));
  *score = disto + lambda_u * rate_u + lambda_v * rate_v;
  WP2_CHECK_REDUCED_STATUS(
      cb->Store420Scores(config, tile_pos_x, tile_pos_y, lambda_u, lambda_v,
                         reduced, disto, rate_u, rate_v));
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::DecideChromaSubsampling(
    const EncoderConfig& config, uint32_t tile_pos_x, uint32_t tile_pos_y,
    bool has_alpha, const QuantMtx& quant_u, const QuantMtx& quant_v,
    Counters* const counters) {
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
    u_params->pred->Predict(*this, kUChannel, prediction, w_pix);
    sub_block(in_.U.Row(0), in_.U.Step(), prediction, /*preds_step=*/w_pix,
              res_u, /*dst_step=*/w_pix, h_pix);
    v_params->pred->Predict(*this, kVChannel, prediction, w_pix);
    sub_block(in_.V.Row(0), in_.V.Step(), prediction, /*preds_step=*/w_pix,
              res_v, /*dst_step=*/w_pix, h_pix);
  }
  int16_t orig_u[kMaxBlockSizePix2], orig_v[kMaxBlockSizePix2];
  std::copy(res_u, res_u + w_pix * h_pix, orig_u);
  std::copy(res_v, res_v + w_pix * h_pix, orig_v);

  int32_t tmp_u[kMaxBlockSizePix2], tmp_v[kMaxBlockSizePix2];
  // transform at full resolution
  WP2Transform2D(res_u, u_params->tf_x(), u_params->tf_y(), w_pix, h_pix,
                 tmp_u);
  WP2Transform2D(res_v, v_params->tf_x(), v_params->tf_y(), w_pix, h_pix,
                 tmp_v);

  float score444;
  WP2_CHECK_STATUS(UVScore(config, tile_pos_x, tile_pos_y, has_alpha,
                           quant_u, quant_v, orig_u, orig_v, tmp_u, tmp_v,
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
  WP2_CHECK_STATUS(UVScore(config, tile_pos_x, tile_pos_y, has_alpha,
                           quant_u, quant_v, orig_u, orig_v, tmp_u, tmp_v,
                           half_dim, /*reduced=*/true, counters->residuals(),
                           this, &score420));

  is420_ = (score420 < score444);
  WP2_CHECK_REDUCED_STATUS(Store420Decision(
      config, tile_pos_x, tile_pos_y,
      is420_ ? Debug420Decision::k420 : Debug420Decision::k444));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// TODO(skal): move to vdebug_enc.cc
TransformPair CodedBlock::GetForcedTransform(const EncoderConfig& config,
                                             uint32_t tile_pos_x,
                                             uint32_t tile_pos_y) {
  (void)config;
  (void)tile_pos_x;
  (void)tile_pos_y;
#if !defined(WP2_REDUCE_BINARY_SIZE)
  EncoderInfo* const info = config.info;
  if (info == nullptr) return kUnknownTf;

  for (const EncoderInfo::ForcedTransform& forced : info->force_transform) {
    if (forced.x != (x_pix() + tile_pos_x) ||
        forced.y != (y_pix() + tile_pos_y)) {
      continue;
    }

    if (forced.transform >= kNumTransformPairs) return kUnknownTf;
    return (TransformPair)forced.transform;
  }
#endif  // WP2_REDUCE_BINARY_SIZE
  return kUnknownTf;
}

//------------------------------------------------------------------------------

bool CodedBlock::SetForcedPredictor(const Predictors& preds,
                                    const EncoderConfig& config,
                                    uint32_t tile_pos_x, uint32_t tile_pos_y,
                                    Channel channel) {
  EncoderInfo* const info = config.info;
  if (info == nullptr || info->force_predictor.empty()) return false;

  for (const EncoderInfo::ForcedPredictor& forced : info->force_predictor) {
    if (forced.channel != channel ||
        forced.x != (x_pix() + tile_pos_x) ||
        forced.y != (y_pix() + tile_pos_y)) {
      continue;
    }

    assert(!preds.empty());
    const Predictor* const pred =
        preds[std::min(forced.predictor_id, (uint8_t)(preds.size() - 1))];
    SetPredictor(channel, pred);
    pred_scores_[(channel == kVChannel) ? kUChannel : channel] = 1.f;
    return true;
  }

  return false;
}

//------------------------------------------------------------------------------

WP2Status CodedBlock::FindBestTf(const EncoderConfig& config,
                                 const Rectangle& tile_rect,
                                 const QuantMtx& quant,
                                 const BlockContext& context, Channel channel,
                                 uint32_t num_channels, bool reduced,
                                 const Vector<TransformPair>& transforms,
                                 Counters* const counters) {
  assert(!transforms.empty());
  CodingParams* const params = GetCodingParams(channel);

  params->tf = GetImplicitTf(channel);
  if (params->tf != kUnknownTf) return WP2_STATUS_OK;
  params->tf = GetForcedTransform(config, tile_rect.x, tile_rect.y);
  if (params->tf != kUnknownTf) return WP2_STATUS_OK;
  params->tf = transforms.front();
  if (transforms.size() == 1) return WP2_STATUS_OK;

  TransformPair best_tf = transforms.front();
  float best_score = 0.f;
  const float lambda = lambda_mult_ * quant.lambda;
  bool is_first = true;

  // Predict from mode once and for all.
  int16_t prediction[kMaxBlockSizePix2];
  const Plane16 prediction_p = PredictInto(channel, prediction);

  for (const TransformPair tf : transforms) {
    params->tf = tf;

    BlockCoeffs16 tmp;
    GetResiduals(channel, prediction_p, &tmp);  // <- depends on 'params->tf'
    TransformAndReconstruct(quant, channel, reduced, prediction_p, &tmp, &tmp);

    const uint32_t distortion = GetDisto(channel);
    float current_score = distortion;
    if (!is_first && current_score >= best_score) continue;   // early-out

    float tf_rate, res_rate;
    WP2_CHECK_STATUS(ResidualRate(context, channel, num_channels,
                                  counters->residuals(),
                                  counters->residuals_aom(), &res_rate));
    current_score += lambda * res_rate;
    if (!is_first && current_score >= best_score) continue;   // early-out

    WP2_CHECK_STATUS(TransformRate(channel, transforms.size(),
                                   counters->transform(), &tf_rate));
    current_score += lambda * tf_rate;
    const bool is_best = (is_first || current_score < best_score);
    if (is_best) {
      best_tf = tf;
      best_score = current_score;
    }
    is_first = false;

#if !defined(WP2_REDUCE_BINARY_SIZE)
    StorePredictionScore(config, tile_rect, channel, *params->pred, tf,
                         distortion, lambda, res_rate, /*pred_rate=*/0, tf_rate,
                         current_score, is_best, SearchType::kTransform);
#endif  // WP2_REDUCE_BINARY_SIZE
  }

  params->tf = best_tf;
  return WP2_STATUS_OK;
}

void CodedBlock::FindBestPred(const Predictors& preds, Channel channel,
                              uint32_t improvement) {
  assert(preds.size() > 1);  // Otherwise this function should be skipped.
  // kVChannel is tested along kUChannel.
  assert(channel != kVChannel);
  Plane16& out = out_.GetChannel(channel);
  CodingParams* const params = GetCodingParams(channel);

  uint32_t best_score = 0;
  uint32_t best_score1 = 0;
  uint32_t best_score2 = 0;
  const Predictor* best_pred = nullptr;
  for (const Predictor* const p : preds) {
    params->pred = p;
    p->Predict(*this, channel, out.Row(0), out.Step());
    uint32_t score1 = GetDisto(channel);
    if (best_pred != nullptr && score1 > best_score) continue;
    uint32_t score2 = 0;
    if (channel == kUChannel) {
      p->Predict(*this, kVChannel, out_.V.Row(0), out_.V.Step());
      score2 = GetDisto(kVChannel);
    }
    // we want the global score to be overall better, but also each
    // individual U/V scores to be improved by at least 'improvement'.
    bool is_better = (score1 + score2 < best_score) && (score1 < best_score1);
    if (channel == kUChannel) is_better = is_better && (score2 < best_score2);
    if (best_pred == nullptr || is_better) {
      best_score1 = (score1 * improvement) >> 8;
      best_score2 = (score2 * improvement) >> 8;
      best_score = score1 + score2;
      best_pred = p;
    }
  }
  params->pred = best_pred;
  if (channel == kUChannel) {
    pred_scores_[kUChannel] = best_score1;
    pred_scores_[kVChannel] = best_score2;
  } else {
    pred_scores_[channel] = best_score;
  }
}

WP2Status CodedBlock::PredictorRate(Channel channel,
                                    SymbolCounter* const counter,
                                    float* const rate) const {
  counter->Clear();
  ANSEncCounter enc;
  SyntaxWriter::WritePredictors(*this, channel, counter, &enc);
  *rate = enc.GetCost();
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::TransformRate(Channel channel, uint32_t num_transforms,
                                    SymbolCounter* const counter,
                                    float* const rate) const {
  ANSEncCounter enc;
  counter->Clear();
  SyntaxWriter::WriteSplitTransform(*this, channel, counter, &enc);
  SyntaxWriter::WriteHasCoeffs(*this, channel, counter, &enc);
  SyntaxWriter::WriteTransform(*this, channel, counter, &enc);
  *rate = enc.GetCost();
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::FindBestPredTf(
    const EncoderConfig& config, const Rectangle& tile_rect,
    const Predictors& preds, const Segment& segment,
    const BlockContext& context, Channel channel, uint32_t num_channels,
    bool reduced, const Vector<TransformPair>& transforms,
    Counters* const counters) {
  assert(channel == kYChannel || channel == kUChannel || channel == kAChannel);
  if (channel == kUChannel) {
    is420_ = reduced;  // Final is420 will be decided later but is needed now.
    assert(transforms.empty());  // Empty because UV tfs are defined by preds.
  } else {
    assert(!transforms.empty());
  }

  float best_score = std::numeric_limits<float>::max();
  TransformPair best_tf = kUnknownTf;
  const Predictor* best_pred = nullptr;
  WP2_CHECK_STATUS(preds.FindBest(config, tile_rect, num_channels, context,
                                  segment, reduced, transforms, counters, this,
                                  &best_score, &best_pred, &best_tf));
  SetPredictor(channel, best_pred);
  pred_scores_[channel] = best_score;
  if (channel == kUChannel) {
    pred_scores_[kVChannel] = pred_scores_[kUChannel];
    assert(GetCodingParams(channel)->tf == best_tf);
  } else {
    GetCodingParams(channel)->tf = best_tf;
  }
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::FindBestSplitTf(const EncoderConfig& config,
                                      const Rectangle& tile_rect,
                                      const QuantMtx& quant,
                                      const BlockContext& context,
                                      Channel channel, uint32_t num_channels,
                                      bool reduced, Counters* const counters) {
  CodingParams* const params = GetCodingParams(channel);
  if (channel != kYChannel || GetSplitSize(dim(), /*split=*/true) == dim()) {
    assert(!params->split_tf);
    return WP2_STATUS_OK;
  }

  bool best_split_tf = false;  // avoids -Werror=maybe-uninitialized
  float best_score = 0.f;
  const float lambda = lambda_mult_ * quant.lambda;
  bool is_first = true;

  // Predict from mode once and for all.
  int16_t prediction[kMaxBlockSizePix2];
  const Plane16 prediction_p = PredictInto(channel, prediction);

  for (bool split_tf : {false, true}) {
    params->split_tf = split_tf;

    BlockCoeffs16 tmp;
    GetResiduals(channel, prediction_p, &tmp);  // <- depends on 'split_tf'
    TransformAndReconstruct(quant, channel, reduced, prediction_p, &tmp, &tmp);

    if (params->split_tf && !HasCoeffs(channel)) {
      continue;  // Split-tf with no coeff is forbidden.
    }

    const uint32_t distortion = GetDisto(channel);
    float score = distortion;
    if (!is_first && score >= best_score) continue;   // early-out

    float tf_rate, res_rate;
    WP2_CHECK_STATUS(ResidualRate(context, channel, num_channels,
                                  counters->residuals(),
                                  counters->residuals_aom(), &res_rate));
    score += lambda * res_rate;
    if (!is_first && score >= best_score) continue;   // early-out

    WP2_CHECK_STATUS(TransformRate(channel, /*num_transforms=*/1,
                                   counters->transform(), &tf_rate));
    score += lambda * tf_rate;

    const bool is_best = (is_first || score < best_score);
    if (is_best) {
      best_split_tf = split_tf;
      best_score = score;
    }
    is_first = false;

#if !defined(WP2_REDUCE_BINARY_SIZE)
    StorePredictionScore(config, tile_rect, channel, *params->pred, params->tf,
                         distortion, lambda, res_rate, /*pred_rate=*/0, tf_rate,
                         score, is_best, SearchType::kSplit);
#endif  // WP2_REDUCE_BINARY_SIZE
  }

  params->split_tf = best_split_tf;
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::OptimizeModesLuma(
    const EncoderConfig& config, const Rectangle& tile_rect, bool has_alpha,
    const Predictors& preds, const Segment& segment,
    const BlockContext& context, const Vector<TransformPair>& transforms,
    const Vector<TransformPair>& transforms_subset, Counters* const counters) {
  assert(preds.size() > 1);
  for (const Predictor* const pred : preds) {
    if (!pred->ComputeParams(this, kYChannel)) {
      // TODO(skal): deal with useless preds
    }
  }

  const uint32_t num_channels = (has_alpha ? 4 : 3);
  GetCodingParams(kYChannel)->split_tf = false;
  bool found_best_tf = false;
  if (y_context_is_constant_) {
    // If the context is constant, all luma predictors will predict the same
    // values (plus or minus 1, because of rounding errors), so we just force
    // the predictor to 0 and we don't write it to the bitstream. We can't do
    // the same for UV and alpha because the chroma from luma predictor doesn't
    // have this property, but we could send a single bit saying whether the
    // predictor is cfl or "other".
    SetLumaUniformPredictor(preds);
    pred_scores_[kYChannel] = 1.f;
  } else if (SetForcedPredictor(preds, config, tile_rect.x, tile_rect.y,
                                kYChannel)) {
    // Predictor is set by EncoderConfig::info (debug).
  } else if (config.effort == 0) {
    // Select the first predictor to save time.
    SetPredictor(kYChannel, preds[0]);
    pred_scores_[kYChannel] = 1.f;
  } else if (config.effort == 1) {
    // Quickly select the best predictor and then choose the best transform.
    const uint32_t improvement =
        200 + std::lround((256 - 200) * config.quality / kMaxLossyQuality);
    FindBestPred(preds, kYChannel, improvement);
  } else {
    const TransformPair forced_tf =
        GetForcedTransform(config, tile_rect.x, tile_rect.y);
    const bool tf_is_forced = (forced_tf != kUnknownTf);
    Vector<TransformPair> forced_tf_vector;
    if (tf_is_forced) {
      WP2_CHECK_ALLOC_OK(forced_tf_vector.resize(1));
      forced_tf_vector[0] = forced_tf;
    }
    const Vector<TransformPair>* const transforms_to_try =
        tf_is_forced ? &forced_tf_vector : &transforms_subset;
    // Select the best predictor+transform pair and then choose the best
    // transform if there are more to try.
    WP2_CHECK_STATUS(FindBestPredTf(config, tile_rect, preds, segment, context,
                                    kYChannel, num_channels, /*reduced=*/false,
                                    *transforms_to_try, counters));
    if (tf_is_forced || transforms.size() == transforms_subset.size()) {
      found_best_tf = true;
    }
  }
  if (!found_best_tf) {
    // TODO(yguyon): Do not recompute scores of 'transforms_subset'
    WP2_CHECK_STATUS(FindBestTf(config, tile_rect, segment.quant_y_, context,
                                kYChannel, num_channels, /*reduced=*/false,
                                transforms, counters));
  }
  WP2_CHECK_STATUS(FindBestSplitTf(config, tile_rect, segment.quant_y_, context,
                                   kYChannel, num_channels, /*reduced=*/false,
                                   counters));

  // TODO(yguyon): Remember the params during the last Quantize() call and omit
  //               this one if they are equivalent.
  Quantize(segment.quant_y_, kYChannel, /*reduced_transform=*/false);
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::OptimizeModesChroma(
    const EncoderConfig& config, const Rectangle& tile_rect, bool has_alpha,
    const FrontMgrBase& mgr, const UVPredictors& preds, const Segment& segment,
    const BlockContext& context, ChromaSubsampling chroma_subsampling,
    DCDiffusionMap* const dc_error_u, DCDiffusionMap* const dc_error_v,
    Counters* const counters) {
  const uint32_t diffusion =
      DCDiffusionMap::GetDiffusion(config.error_diffusion);
  for (const Predictor* const pred : preds) {
    // TODO(skal): deal with useless preds
    pred->ComputeParams(this, kUChannel);
    pred->ComputeParams(this, kVChannel);
  }
  // Decide if we reduce chroma for finding the best predictors. The
  // logic is based on what is done a few lines after.
  const bool reduced =
      ((chroma_subsampling == ChromaSubsampling::kSingleBlock &&
        config.uv_mode == EncoderConfig::UVMode420) ||
       chroma_subsampling == ChromaSubsampling::k420);

  if (SetForcedPredictor(preds, config, tile_rect.x, tile_rect.y, kUChannel)) {
    // Predictor is set by EncoderConfig::info (debug).
  } else if (config.effort == 0 || preds.size() == 1) {
    // Select the first predictor to save time.
    SetUVPredictor(preds[0]);
    pred_scores_[kUChannel] = pred_scores_[kVChannel] = 1.f;
  } else if (config.effort == 1) {
    // Quickly select the best predictor.
    const uint32_t improvement =
        100 + std::lround((256 - 100) * config.quality / kMaxLossyQuality);
    FindBestPred(preds, kUChannel, improvement);
  } else {
    const uint32_t num_channels = (has_alpha ? 4 : 3);
    // Empty because UV transforms are defined by predictors.
    const Vector<TransformPair> transforms;
    // Spend time to select the best predictor.
    WP2_CHECK_STATUS(FindBestPredTf(config, tile_rect, preds, segment, context,
                                    kUChannel, num_channels, reduced,
                                    transforms, counters));
  }

  if (diffusion > 0) {  // Add error diffusion
    dc_error_[kUChannel] = dc_error_u->Get(blk(), diffusion);
    dc_error_[kVChannel] = dc_error_v->Get(blk(), diffusion);
  }

  // TODO(skal): decide based on coded luma too?
  if (reduced) {
    is420_ = true;
  } else if ((chroma_subsampling == ChromaSubsampling::kSingleBlock &&
              config.uv_mode == EncoderConfig::UVMode444) ||
             chroma_subsampling == ChromaSubsampling::k444) {
    is420_ = false;
  } else {
    assert(chroma_subsampling == ChromaSubsampling::kSingleBlock ||
           chroma_subsampling == ChromaSubsampling::kAdaptive);
    WP2_CHECK_STATUS(DecideChromaSubsampling(config, tile_rect.x, tile_rect.y,
                                             has_alpha, segment.quant_u_,
                                             segment.quant_v_, counters));
  }
  Quantize(segment.quant_u_, kUChannel, is420_);
  Quantize(segment.quant_v_, kVChannel, is420_);
  if (diffusion > 0) {
    dc_error_u->Store(mgr, blk(), dc_error_next_[kUChannel]);
    dc_error_v->Store(mgr, blk(), dc_error_next_[kVChannel]);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
