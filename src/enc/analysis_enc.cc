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
//  Source analysis
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include "src/dsp/dsp.h"
#include "src/enc/analysis.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/vector.h"

// #define PRINT_STATS   // define, perform stats on predictors

#if defined(PRINT_STATS)
#include "src/utils/stats.h"
#endif

namespace WP2 {

//------------------------------------------------------------------------------

static void AssignGrainLevel(const EncoderConfig& config,
                             Vector<Segment>* const segments) {
  bool store_grain = config.store_grain;
  if (store_grain) {
    bool have_grain = false;
    for (const Segment& s : *segments) {
      have_grain |= s.grain_.IsUsed();
      if (have_grain) break;
    }
    if (!have_grain) store_grain = false;
  }
  if (!store_grain) {
    for (Segment& s : *segments) s.grain_.Reset();
  }
}

#if 0   // unused for now
// Check if some segments are identical (same quantization) and reduce their
// number accordingly. This usually happens at very low or very high quality.
static WP2Status SimplifySegments(Vector<Segment>* const segments) {
  uint32_t size = 1;
  for (uint32_t i = 1; i < segments->size(); ++i) {
    bool found = false;
    for (uint32_t k = 0; k < size; ++k) {
      found = (*segments)[i].IsMergeableWith((*segments)[k]);
      if (found) break;
    }
    if (!found) {
      if (i > size) {
        // WP2_CHECK_STATUS((*segments)[size].CopyFrom((*segments)[i]));
        std::swap((*segments)[size], (*segments)[i]);
      }
      ++size;
    }
  }
  WP2_CHECK_ALLOC_OK(segments->resize(size));
  return WP2_STATUS_OK;
}
#endif

//------------------------------------------------------------------------------
// Perceptual: assignment of quantizer steps from the evaluated 'risk'

WP2Status GlobalParams::AssignQuantizations(const EncoderConfig& config) {
  const uint32_t num_segments = segments_.size();
  assert(num_segments <= (uint32_t)config.segments);
  assert(num_segments <= kMaxNumSegments);
  uint32_t qualities[kMaxNumSegments];
  const float quant_factor0 =
      kQFactorMax * (kMaxLossyQuality - config.quality) / kMaxLossyQuality;
  float min_risk = 1.0f;
  float max_risk = 0.0f;
  for (const auto& s : segments_) min_risk = std::min(min_risk, s.risk_);
  for (const auto& s : segments_) max_risk = std::max(max_risk, s.risk_);
  min_risk -= 0.01f;   // to avoid divide-by-zero
  const float amp = 1.4f * Clamp(config.sns / 100.f, 0.f, 1.f);
  const float adjust_amp =
      std::min(1.f, num_segments / 4.f) / (max_risk - min_risk);
  // Exponent for 'adjust' (1 = linear).
  const float exponent = 2.0f;
  for (uint32_t idx = 0; idx < num_segments; ++idx) {
    const Segment& s = segments_[idx];
    float quant = config.segment_factors[idx];
    if (quant == 0.f) {
      const float adjust = adjust_amp * (s.risk_ - min_risk);
      const float displaced = amp * pow(adjust, exponent);
      quant = quant_factor0 * (1.0f - displaced);
    } else {
      quant = kQFactorMax * (kMaxLossyQuality - quant) / kMaxLossyQuality;
    }
    qualities[idx] = Clamp<int32_t>(std::lround(quant), 0, kQFactorMax);
  }
  for (uint32_t idx = 0; idx < num_segments; ++idx) {
    WP2_CHECK_STATUS(segments_[idx].AllocateForEncoder());
    segments_[idx].SetQuantizationFactor(
        transf_, GetQualityHint(config.quality), u_quant_multiplier_,
        v_quant_multiplier_, qualities[idx]);
  }
  return WP2_STATUS_OK;
}

WP2Status GlobalParams::AssignAlphaQuantizations(const YUVPlane& yuv,
                                                 const EncoderConfig& config) {
  WP2AlphaInit();
  has_alpha_ = false;
  if (!yuv.A.IsEmpty()) {
    for (uint32_t y = 0; y < yuv.A.h_; ++y) {
      const int16_t* const row = yuv.A.Row(y);
      has_alpha_ = has_alpha_ || WP2HasOtherValue16b(row, yuv.A.w_, kAlphaMax);
      if (has_alpha_) break;
    }
  }
  maybe_use_lossy_alpha_ =
      has_alpha_ && (config.alpha_quality <= kMaxLossyQuality);

  if (maybe_use_lossy_alpha_) {
    // TODO(maryla): detect when to enable/disable alpha filtering.
    enable_alpha_filter_ = true;
    const uint32_t max_quality = kMaxLossyQuality;
    const uint32_t quant_factor0 =
        (uint32_t)std::lround(
            kQFactorMax * (max_quality - config.alpha_quality) / max_quality);
    assert(!segments_.empty());
    for (auto& s : segments_) {
      WP2_CHECK_STATUS(s.AllocateForEncoder(/*for_alpha=*/true));
      s.SetAlphaQuantizationFactor(quant_factor0);
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

GlobalParams::Type DecideGlobalParamsType(const EncoderConfig& config) {
  // Neural compression is considered as GP_BOTH for now because it is lossy
  // compression outputting RGB.
  if (config.use_neural_compression) return GlobalParams::GP_BOTH;
  // TODO(skal): mixed lossy / lossless case.
  if (config.quality <= kMaxLossyQuality) {
    return (config.use_av1 ? GlobalParams::GP_AV1 : GlobalParams::GP_LOSSY);
  }
  return GlobalParams::GP_LOSSLESS;
}

WP2Status GlobalAnalysis(const ArgbBuffer& rgb, const YUVPlane& yuv,
                         const CSPTransform& transf,
                         const EncoderConfig& config,
                         GlobalParams* const gparams) {
  WP2_CHECK_OK(gparams != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(!rgb.IsEmpty() || !yuv.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);

  gparams->type_ = DecideGlobalParamsType(config);

  if (gparams->type_ == GlobalParams::GP_LOSSLESS ||
      gparams->type_ == GlobalParams::GP_BOTH) {
    WP2_CHECK_OK(!rgb.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
    gparams->has_alpha_ = rgb.HasTransparency();
    // TODO(skal): extract global params from lossless code
  }
  if (gparams->type_ == GlobalParams::GP_LOSSY ||
      gparams->type_ == GlobalParams::GP_BOTH) {
    WP2_CHECK_OK(!yuv.IsEmpty(), WP2_STATUS_INVALID_PARAMETER);
    // If CSPTransform::Init() failed with kCustom, it felt back to another Csp.
    WP2_CHECK_OK(
        transf.GetType() == config.csp_type || config.csp_type == Csp::kCustom,
        WP2_STATUS_INVALID_PARAMETER);

    WP2MathInit();
    WP2TransformInit();
    PredictionInit();

    gparams->partition_set_ = config.partition_set;
    gparams->partition_snapping_ = config.partition_snapping;
    gparams->explicit_segment_ids_ =
        (config.segment_id_mode == WP2::EncoderConfig::SEGMENT_ID_AUTO) ?
            (config.quality > 15.f) :
        (config.segment_id_mode == WP2::EncoderConfig::SEGMENT_ID_EXPLICIT);
    gparams->use_rnd_mtx_ = config.use_random_matrix;
    gparams->transf_ = transf;
    gparams->u_quant_multiplier_ = config.u_quant_multiplier;
    gparams->v_quant_multiplier_ = config.v_quant_multiplier;

    // perform perceptual analysis to extract segments
    WP2_CHECK_STATUS(FindSegments(yuv, config, gparams));
    WP2_CHECK_STATUS(gparams->AssignQuantizations(config));
    WP2_CHECK_STATUS(gparams->AssignAlphaQuantizations(yuv, config));
    WP2_CHECK_STATUS(AssignGrainParams(yuv, config, gparams));
    AssignGrainLevel(config, &gparams->segments_);

    // TODO(skal): not working yet, we need to remap cluster_map[]
    // WP2_CHECK_STATUS(SimplifySegments(&gparams->segments_));

    if (config.info != nullptr) {
      for (const auto& forced : config.info->force_param) {
        if (forced.type != EncoderInfo::ForcedParam::Type::kSegment) continue;
        WP2_CHECK_OK(forced.value < gparams->segments_.size(),
                     WP2_STATUS_INVALID_CONFIGURATION);
      }
    }

    // No decision for large Y and U/V predictors.
    WP2_CHECK_STATUS(gparams->InitFixedPredictors());
  }
  return WP2_STATUS_OK;
}

}  // namespace WP2
