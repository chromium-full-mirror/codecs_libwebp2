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
//  Predictor encoding code
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include <cstdio>
#include <limits>

#include "src/common/lossy/predictor.h"
#include "src/enc/block_enc.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------
// Predictors' FindBest

class PredictorScorer {    // Side-Struct computing some predictor scores.
 public:
  PredictorScorer(const EncoderConfig& config, const Rectangle& tile_rect,
                  const BlockContext& context, const Segment& segment,
                  Channel channel, uint32_t num_channels, bool reduced,
                  Counters* const counters, float best_score)
      : config_(config),
        tile_rect_(tile_rect),
        context_(context),
        segment_(segment),
        channel_(channel),
        num_channels_(num_channels),
        reduced_(reduced),
        counters_(counters),
        score_(0.f),
        dist_(0),
        best_score_(best_score),  // Needed for VDebug 'is_best'.
        best_dist_(std::numeric_limits<uint32_t>::max()),
        best_predictor_(nullptr),
        best_tf_(kUnknownTf) {
    (void)config_;  // only used by VDebug
  }

  // Computes the score and dist of the given predictor 'p' applied to 'cb'.
  WP2Status UpdateUVPredScore(const Predictor& p, CodedBlock* const cb) {
    constexpr uint32_t kU = 0, kV = 1;   // notation for easier reading
    constexpr Channel kChannels[] = {kUChannel, kVChannel};

    uint32_t dist[2];
    float scores[2], res_rate[2], pred_rate[2];
    for (uint32_t c : {kU, kV}) {
      cb->QuantizeAll(config_, context_, segment_, kChannels[c], reduced_,
                      counters_);

      dist[c] = cb->GetDisto(kChannels[c]);

      WP2_CHECK_STATUS(cb->ResidualRate(
          context_, kChannels[c], num_channels_, counters_->residuals(),
          counters_->residuals_aom(), &res_rate[c]));
      WP2_CHECK_STATUS(cb->PredictorRate(kChannels[c], counters_->predictor(),
                                         &pred_rate[c]));
      const float lambda =
          cb->lambda_mult_ * segment_.GetQuant(kChannels[c]).lambda;
      scores[c] = dist[c] + lambda * (res_rate[c] + pred_rate[c]);
    }

    score_ = scores[kU] + scores[kV];
    dist_ = dist[kU] + dist[kV];

#if !defined(WP2_REDUCE_BINARY_SIZE)
    // Store scores for debug.
    for (uint32_t c : {kU, kV}) {
      const QuantMtx& quant = segment_.GetQuant(kChannels[c]);
      cb->StorePredictionScore(config_, tile_rect_, kChannels[c], p,
                               cb->GetCodingParams(kChannels[c])->tf, dist[c],
                               cb->lambda_mult_ * quant.lambda, res_rate[c],
                               pred_rate[c], /*tf_rate=*/0.f, scores[c],
                               /*is_best=*/(score_ < best_score_),
                               CodedBlock::SearchType::kPredAndTransform);
    }
#endif  // WP2_REDUCE_BINARY_SIZE

    return WP2_STATUS_OK;
  }

  // Computes the score and dist of the given predictor 'p' and transform 'tf'
  // applied to 'cb'.
  WP2Status UpdateYAPredScore(const Predictor& p, const Plane16& prediction,
                              TransformPair tf, const BlockCoeffs16* const res,
                              BlockCoeffs16* const tmp, CodedBlock* const cb) {
    assert(channel_ == kYChannel || channel_ == kAChannel);
    CodedBlock::CodingParams* const params = cb->GetCodingParams(channel_);
    params->tf = tf;
    assert(cb->GetImplicitTf(channel_) == kUnknownTf ||
           cb->GetImplicitTf(channel_) == tf);

    cb->TransformAndReconstruct(config_, context_, segment_, channel_,
                                /*tf_i=*/0, reduced_, prediction, res, tmp,
                                counters_);
    cb->QuantizeAllButFirst(config_, context_, segment_, channel_,
                            /*reduced_transform=*/false, counters_);

    dist_ = cb->GetDisto(channel_);

    float res_rate, pred_rate, tf_rate;
    WP2_CHECK_STATUS(cb->ResidualRate(context_, channel_, num_channels_,
                                      counters_->residuals(),
                                      counters_->residuals_aom(), &res_rate));
    WP2_CHECK_STATUS(
        cb->PredictorRate(channel_, counters_->predictor(), &pred_rate));
    WP2_CHECK_STATUS(
        cb->TransformRate(channel_, counters_->transform(), &tf_rate));
    const float lambda = cb->lambda_mult_ * segment_.GetQuant(channel_).lambda;
    score_ = dist_ + lambda * (res_rate + pred_rate + tf_rate);
    // Note: some of the steps above could be skipped as soon as the distortion
    // and/or the rate is higher than the current 'best_score_' but angle
    // predictor scores are reused in Predictors::FindBest() to find sub-angles.

#if !defined(WP2_REDUCE_BINARY_SIZE)
    // Store score for debug.
    cb->StorePredictionScore(config_, tile_rect_, channel_, p, params->tf,
                             dist_, lambda, res_rate, pred_rate, tf_rate,
                             score_, /*is_best=*/(score_ < best_score_),
                             CodedBlock::SearchType::kPredAndTransform);
#endif  // WP2_REDUCE_BINARY_SIZE

    return WP2_STATUS_OK;
  }

  // Computes the score of the predictior 'p' and stores its score, dist etc. if
  // it is the best.
  WP2Status UpdatePredScore(const Predictor* const p, const Plane16& prediction,
                            TransformPair tf, const BlockCoeffs16* const res,
                            BlockCoeffs16* const tmp, CodedBlock* const cb) {
    if (channel_ == kUChannel) {
      WP2_CHECK_STATUS(UpdateUVPredScore(*p, cb));
    } else {
      WP2_CHECK_STATUS(UpdateYAPredScore(*p, prediction, tf, res, tmp, cb));
    }
    if (score_ < best_score_) {
      best_score_ = score_;
      best_dist_ = dist_;
      best_predictor_ = p;
      best_tf_ = tf;
    }
    return WP2_STATUS_OK;
  }

  float GetLastScore() const { return score_; }
  uint32_t GetLastDist() const { return dist_; }
  float GetBestScore() const { return best_score_; }
  uint32_t GetBestDist() const { return best_dist_; }
  const Predictor* GetBestPredictor() const { return best_predictor_; }
  TransformPair GetBestTf() const { return best_tf_; }

 private:
  const EncoderConfig& config_;
  const Rectangle tile_rect_;
  const BlockContext& context_;
  const Segment& segment_;
  const Channel channel_;
  const uint32_t num_channels_;
  const bool reduced_;
  Counters* counters_;

  float score_;
  uint32_t dist_;
  float best_score_;
  uint32_t best_dist_;
  const Predictor* best_predictor_;
  TransformPair best_tf_;
};

//------------------------------------------------------------------------------

WP2Status Predictors::FindBest(
    const EncoderConfig& config, const Rectangle& tile_rect,
    uint32_t num_channels, const BlockContext& context, const Segment& segment,
    bool reduced, const Vector<TransformPair>& transforms,
    Counters* const counters, CodedBlock* const cb, float* const best_score,
    const Predictor** const best_predictor, TransformPair* const best_tf,
    bool* const best_split) const {
  *best_predictor = nullptr;
  *best_score = std::numeric_limits<float>::max();
  // This function should only be called once for both U and V.
  assert(channel_ != kVChannel);
  const bool is_uv = (channel_ == kUChannel);
  PredictorScorer scorer(
      config, tile_rect, context, segment, channel_, num_channels, reduced,
      counters, *best_score);

  int16_t prediction[kMaxBlockSizePix2];
  Plane16 pred_view;
  WP2_CHECK_STATUS(pred_view.SetView(prediction, kMaxBlockSizePix,
                                     kMaxBlockSizePix, kMaxBlockSizePix));
  BlockCoeffs16 res, tmp;
  TransformPair overridden_tf;

  const bool force_dc = (config.info != nullptr) && config.info->disable_preds;

  // Test the non-angle predictors.
  for (const Predictor* const p : preds_no_angle_) {
    cb->SetPredictor(channel_, p);
    if (!is_uv) {
      cb->PredictBlock(channel_, /*tf_i=*/0, pred_view.Row(0),
                       pred_view.Step());
      cb->GetResiduals(channel_, /*tf_i=*/0, pred_view, &res);
    }
    if (p->OverridesTransform(channel_, &overridden_tf)) {
      WP2_CHECK_STATUS(
          scorer.UpdatePredScore(p, pred_view, overridden_tf, &res, &tmp, cb));
    } else {
      for (TransformPair tf : transforms) {
        WP2_CHECK_STATUS(
            scorer.UpdatePredScore(p, pred_view, tf, &res, &tmp, cb));
      }
    }

    if (force_dc) {
      // We assume DC is the first predictor and just stop here.
      assert(scorer.GetBestPredictor() != nullptr);
      *best_score = scorer.GetBestScore();
      *best_predictor = scorer.GetBestPredictor();
      *best_tf = scorer.GetBestTf();
      *best_split = cb->GetCodingParams(channel_)->split_tf;
      return WP2_STATUS_OK;
    }
  }
  const float best_score_no_angle = scorer.GetBestScore();
  const uint32_t best_dist_no_angle = scorer.GetBestDist();

  // Test the main angle predictors.
  struct AngleScore {
    float score;
    uint32_t dist;
    const Predictor* p;
    TransformPair tf;
    bool operator<(const AngleScore& s) const { return (score < s.score); }
  };
  // Store AngleScore in a static array if possible, dynamic otherwise.
  // Ideally, the static array would be of size
  // AnglePredictor::Type::Num * kNumTransformPairs but that is 1344 bytes.
  // So we limit to AnglePredictor::Type::Num.
  // Those could be moved to Predictors if it were not part of GlobalParams.
  VectorNoCtor<AngleScore> angle_score_dynamic;
  std::array<AngleScore, kAnglePredNum> angle_score_static;
  const uint32_t angle_score_size =
      preds_main_angle_.size() * std::max((size_t)1, transforms.size());
  AngleScore* beg;
  // Decide whether we use the dynamic array or the static one.
  if (angle_score_size > angle_score_static.size()) {
    WP2_CHECK_ALLOC_OK(angle_score_dynamic.resize(angle_score_size));
    beg = &angle_score_dynamic[0];
  } else {
    beg = &angle_score_static[0];
  }
  AngleScore* end = beg;

  for (const Predictor* const p : preds_main_angle_) {
    cb->SetPredictor(channel_, p);
    if (!is_uv) {
      cb->PredictBlock(channel_, /*tf_i=*/0, pred_view.Row(0),
                       pred_view.Step());
      cb->GetResiduals(channel_, /*tf_i=*/0, pred_view, &res);
    }
    if (p->OverridesTransform(channel_, &overridden_tf)) {
      WP2_CHECK_STATUS(
          scorer.UpdatePredScore(p, pred_view, overridden_tf, &res, &tmp, cb));

      *(end++) = {scorer.GetLastScore(), scorer.GetLastDist(), p,
                  overridden_tf};
    } else {
      for (TransformPair tf : transforms) {
        WP2_CHECK_STATUS(
            scorer.UpdatePredScore(p, pred_view, tf, &res, &tmp, cb));
        *(end++) = {scorer.GetLastScore(), scorer.GetLastDist(), p, tf};
      }
    }
  }
  assert(angle_score_size == end - beg && angle_score_size >= 1);

  std::sort(beg, end);

  // Choose to refine all the ones within a percentage of the lowest score. This
  // helps refine predictors that give similar scores.
  const int effort = config.effort;
  const float min_score = beg->score;
  const float max_score = end[-1].score;
  const float fraction = effort / 9.f;
  const float min_allowed_score =
      min_score * (1.f - fraction) + max_score * fraction + 1.f;
  assert(effort != 9 || max_score <= min_allowed_score);
  for (uint32_t i = 0; i < angle_score_size; ++i) {
    const AngleScore& s = beg[i];
    // We refine every predictor with a better score, or distortion and make
    // sure we refine all the ones within a percentage of the lowest score.
    if (s.score <= best_score_no_angle || s.dist <= best_dist_no_angle ||
        s.score <= min_allowed_score) {
      // Refine the angle predictor.
      for (const Predictor* const p : preds_) {
        if (p->mode() == s.p->mode() && p != s.p) {
          cb->SetPredictor(channel_, p);
          if (!is_uv) {
            cb->PredictBlock(channel_, /*tf_i=*/0, pred_view.Row(0),
                             pred_view.Step());
            cb->GetResiduals(channel_, /*tf_i=*/0, pred_view, &res);
          }
          WP2_CHECK_STATUS(
              scorer.UpdatePredScore(p, pred_view, s.tf, &res, &tmp, cb));
        }
      }
    }
  }

  if (scorer.GetBestScore() < *best_score) {
    assert(scorer.GetBestPredictor() != nullptr);
    *best_score = scorer.GetBestScore();
    *best_predictor = scorer.GetBestPredictor();
    *best_tf = scorer.GetBestTf();
    *best_split = cb->GetCodingParams(channel_)->split_tf;
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
