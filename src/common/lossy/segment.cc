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
//   Segment
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/lossy/segment.h"

#include <algorithm>

#include "src/common/constants.h"
#include "src/common/lossy/block.h"
#include "src/dec/symbols_dec.h"
#include "src/dsp/math.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------
// Grain read/write

static constexpr uint32_t kMaxNoiseStrength = 100;
static constexpr uint32_t kMaxNoiseStrengthIO = 15;

bool GrainParams::operator==(const GrainParams& other) const {
  return (y_ == other.y_) && (uv_ == other.uv_) && (freq_y_ == other.freq_y_) &&
         (freq_uv_ == other.freq_uv_);
}

void GrainParams::Write(ANSEnc* const enc) const {
  assert(y_ <= kMaxNoiseStrength);
  assert(uv_ <= kMaxNoiseStrength);
  // Scale down for i/o.
  const uint32_t scaled_y =
      DivRound(y_ * kMaxNoiseStrengthIO, kMaxNoiseStrength);
  enc->PutRange(scaled_y, 0, kMaxNoiseStrengthIO, "grain");
  if (scaled_y > 0) {
    enc->PutRange(freq_y_, 1, 6, "grain");
  }
  const uint32_t scaled_uv =
      DivRound(uv_ * kMaxNoiseStrengthIO, kMaxNoiseStrength);
  enc->PutRange(scaled_uv, 0, kMaxNoiseStrengthIO, "grain");
  if (scaled_uv > 0) {
    enc->PutRange(freq_uv_, 1, 6, "grain");
  }
}

void GrainParams::Read(ANSDec* const dec) {
  y_ = dec->ReadRange(0, kMaxNoiseStrengthIO, "grain") * kMaxNoiseStrength /
       kMaxNoiseStrengthIO;
  if (y_ > 0) {
    freq_y_ = dec->ReadRange(1, 6, "grain");
  }
  uv_ = dec->ReadRange(0, kMaxNoiseStrengthIO, "grain") * kMaxNoiseStrength /
        kMaxNoiseStrengthIO;
  if (uv_ > 0) {
    freq_uv_ = dec->ReadRange(1, 6, "grain");
  }
}

void GrainParams::Reset() {
  y_ = uv_ = 0;
  freq_y_ = freq_uv_ = 0;
}

void GrainParams::Print() const {
  printf("[y=%d uv=%d, cut y=%d uv=%d]\n", y_, uv_, freq_y_, freq_uv_);
}

//------------------------------------------------------------------------------
// Segment writing

WP2Status Segment::WriteHeader(ANSEnc* const enc,
                               bool write_alpha, bool write_grain) const {
  enc->PutBool(use_quality_factor_, "quant-factor");

  if (use_quality_factor_) {
    enc->PutRValue(quality_factor_, kQFactorMax + 1, "quant-factor");
    if (write_alpha) {
      enc->PutRValue(a_quality_factor_, kQFactorMax + 1, "quant-factor");
    }
  } else {
    // TODO(skal): use better delta-coding.
    for (uint32_t i = 0; i < kNumQuantZones; ++i) {
      for (auto c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
        if (c == kAChannel && !write_alpha) continue;
        enc->PutRange(quant_steps_[c][i], 0, kQFactorMax, "quant-factor");
      }
    }
  }

  if (write_grain && enc->PutUValue(use_grain_, 1, "grain")) grain_.Write(enc);
  return enc->GetStatus();
}

void Segment::StoreEncodingMethods(const CodedBlockBase& cb,
                                   SymbolManager* const sm,
                                   ANSEncBase* const enc) const {
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && !cb.HasLossyAlpha()) continue;
    for (uint32_t tf_i = 0; tf_i < cb.GetNumTransforms(channel); ++tf_i) {
      if (cb.num_coeffs_[channel][tf_i] == 0) {
        // TODO(yguyon): At least one method != kAllZero in block if split
        assert(cb.method_[channel][tf_i] == EncodingMethod::kAllZero);
      } else {
        const uint32_t cluster =
            (channel == kAChannel) ? 0 : (3 * cb.id_ + (uint32_t)channel);
        const Symbol symbol = (channel == kAChannel) ? kSymbolEncodingMethodA
                                                     : kSymbolEncodingMethod;
        if (cb.IsFirstCoeffDC(channel)) {
          sm->Process(symbol, cluster, (uint32_t) cb.method_[channel][tf_i],
                      kChannelStr[channel], enc);
        } else {
          // Use 'max_value' to remove the EncodingMethod::kDCOnly possibility.
          sm->Process(symbol, cluster, (uint32_t) cb.method_[channel][tf_i],
              /*max_value=*/(uint32_t) EncodingMethod::kMethod1,
                      kChannelStr[channel], enc);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Segment reading

WP2Status Segment::ReadHeader(uint32_t u_quant_offset, uint32_t v_quant_offset,
                              bool read_alpha, bool read_grain,
                              ANSDec* const dec) {
  const bool use_quality_factor = dec->ReadBool("quant-factor");
  if (use_quality_factor) {
    const uint32_t quality_factor =
        dec->ReadRValue(kQFactorMax + 1, "quant-factor");
    SetQuality(quality_factor, u_quant_offset, v_quant_offset);
    if (read_alpha) {
      const uint32_t a_quality_factor =
          dec->ReadRValue(kQFactorMax + 1, "quant-factor");
      SetAlphaQuality(a_quality_factor);
    }
  } else {
    for (uint32_t i = 0; i < kNumQuantZones; ++i) {
      for (auto c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
        if (c == kAChannel && !read_alpha) continue;
        quant_steps_[c][i] = dec->ReadRange(0, kQFactorMax, "quant-factor");
        SetQuantSteps(quant_steps_[c], c);
      }
    }
  }
  // Early check before calling heavier functions.
  WP2_CHECK_STATUS(dec->GetStatus());
  FinalizeQuant();
  if (read_alpha) FinalizeAlphaQuant();

  if (read_grain) {
    use_grain_ = dec->ReadUValue(1, "grain");
    if (use_grain_) grain_.Read(dec);
  } else {
    use_grain_ = false;
  }
  return dec->GetStatus();
}

WP2Status Segment::ReadEncodingMethods(SymbolReader* const sr,
                                       CodedBlockBase* const cb) const {
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (channel == kAChannel && !cb->HasLossyAlpha()) continue;
    for (uint32_t tf_i = 0; tf_i < cb->GetNumTransforms(channel); ++tf_i) {
      if (cb->num_coeffs_[channel][tf_i] == 0) {
        cb->method_[channel][tf_i] = EncodingMethod::kAllZero;
      } else {
        const uint32_t cluster =
            (channel == kAChannel) ? 0 : (3 * cb->id_ + (uint32_t)channel);
        const Symbol symbol = (channel == kAChannel) ? kSymbolEncodingMethodA
                                                     : kSymbolEncodingMethod;
        if (cb->IsFirstCoeffDC(channel)) {
          cb->method_[channel][tf_i] =
              (EncodingMethod)sr->Read(symbol, cluster, kChannelStr[channel]);
        } else {
          // Use 'max_value' to remove the EncodingMethod::kDCOnly possibility.
          int32_t method;
          WP2_CHECK_STATUS(
              sr->ReadWithMax(symbol, cluster,
                              /*max_value=*/(uint32_t)EncodingMethod::kMethod1,
                              kChannelStr[channel], &method));
          cb->method_[channel][tf_i] = (EncodingMethod)method;
        }
      }
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

uint32_t Segment::GetMaxAbsDC(Channel channel) const {
  const QuantMtx& mtx = GetQuant(channel);
  uint32_t max_abs_dc = 0;
  for (const TrfSize tdim : kAllTrfSizes) {
    max_abs_dc = std::max(max_abs_dc, mtx.GetMaxAbsResDC(tdim));
  }
  return max_abs_dc;
}

//------------------------------------------------------------------------------

bool Segment::IsMergeableWith(const Segment& other) const {
  const bool same =
      (q_scale_ == other.q_scale_) &&
      (use_quality_factor_ == other.use_quality_factor_) &&
      (use_grain_ == other.use_grain_) &&
      (!use_grain_ || grain_ == other.grain_);
  if (!same) return false;
  if (use_quality_factor_) {
    if (quality_factor_ != other.quality_factor_) return false;
    if (a_quality_factor_ != other.a_quality_factor_) return false;
  } else {
    for (uint32_t c = 0; c < 4; ++c) {
      for (uint32_t i = 0; i < kNumQuantZones; ++i) {
        if (quant_steps_[c][i] != other.quant_steps_[c][i]) return false;
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------

namespace {

uint32_t GetChromaACQuant(int32_t quality_factor, int32_t quant_offset) {
  return (uint32_t)Clamp<int32_t>(
      quality_factor + quant_offset - (int32_t)kNeutralQuantOffset, 0,
      kQFactorMax);
}

uint32_t GetChromaDCQuant(int32_t quality_factor, int32_t quant_offset) {
  const uint32_t q = GetChromaACQuant(quality_factor, quant_offset);
  // Quantize DC less than AC for UV, especially at lower qualities.
  // Values are found empirically.
  return RightShiftRound(34210 + 34210 * q - 76 * q * q, 16);
}

}  // namespace

void Segment::SetQuality(uint32_t quality_factor, uint32_t u_quant_offset,
                         uint32_t v_quant_offset) {
  // The following ad-hoc formulae maps the [0-kQFactorMax] range of quality
  // factor to approximately 0.05 bpp - 3 bpp usable range. Note that
  // quality_factor 0 is the highest quality (= smaller quant-step).
  quality_factor = std::min(quality_factor, (uint32_t)kQFactorMax);

  // DC
  quant_steps_[kYChannel][0] = quality_factor;
  quant_steps_[kUChannel][0] = GetChromaDCQuant(quality_factor, u_quant_offset);
  quant_steps_[kVChannel][0] = GetChromaDCQuant(quality_factor, v_quant_offset);
  // AC
  for (uint32_t zone = 1; zone < kNumQuantZones; ++zone) {
    quant_steps_[kYChannel][zone] = quality_factor;
    quant_steps_[kUChannel][zone] =
        GetChromaACQuant(quality_factor, u_quant_offset);
    quant_steps_[kVChannel][zone] =
        GetChromaACQuant(quality_factor, v_quant_offset);
  }

  quality_factor_ = quality_factor;
  use_quality_factor_ = true;
}

void Segment::SetAlphaQuality(uint32_t quality_factor) {
  assert(use_quality_factor_);
  a_quality_factor_ = std::min(quality_factor, (uint32_t)kQFactorMax);
  for (uint32_t i = 0; i < kNumQuantZones; ++i) {
    quant_steps_[kAChannel][i] = quality_factor / 2;
  }
}

void Segment::SetQuantSteps(uint16_t quants[kNumQuantZones], Channel channel) {
  for (uint32_t i = 0; i < kNumQuantZones; ++i) {
    if (quants[i]) {
      quant_steps_[channel][i] = std::min(quants[i], kQFactorMax);
    }
  }
  use_quality_factor_ = false;
}

void Segment::GetQuantSteps(uint16_t quants[kNumQuantZones],
                            Channel channel) const {
  for (uint32_t i = 0; i < kNumQuantZones; ++i) {
    quants[i] = quant_steps_[channel][i];
  }
}

WP2Status Segment::AllocateForEncoder(bool for_alpha) {
  if (for_alpha) {
    WP2_CHECK_STATUS(quant_a_.AllocateForEncoder());
  } else {
    WP2_CHECK_STATUS(quant_y_.AllocateForEncoder());
    WP2_CHECK_STATUS(quant_u_.AllocateForEncoder());
    WP2_CHECK_STATUS(quant_v_.AllocateForEncoder());
  }
  return WP2_STATUS_OK;
}

void Segment::SetYUVBounds(int16_t yuv_min, int16_t yuv_max) {
  max_residual_ = 1 + yuv_max - yuv_min;
  // Adjust quantization based on the actual yuv/alpha range, compared to
  // the max yuv range which is what the quantizer is tuned for by default.
  q_scale_ = (float)(1 << (kYuvMaxPrec + 1)) / max_residual_;
}

void Segment::FinalizeQuant() {
  assert(q_scale_ > 0.f);   // must have been set prior to calling this
  quant_y_.Init(max_residual_, q_scale_, quant_steps_[kYChannel]);
  quant_u_.Init(max_residual_, q_scale_, quant_steps_[kUChannel]);
  quant_v_.Init(max_residual_, q_scale_, quant_steps_[kVChannel]);
}

void Segment::FinalizeAlphaQuant() {
  // Max residual value is different for alpha since it doesn't go through the
  // CSP transform. Alpha is between 0 and kAlphaMax. After subtracting
  // predicted values from actual values, the residuals are between -kAlphaMax
  // and kAlphaMax. So the max absolute value is still kAlphaMax.
  const float q_scale = 1.0f;
  quant_a_.Init(/*max_residual=*/kAlphaMax, q_scale, quant_steps_[kAChannel]);
}

void Segment::SetQuantizationFactor(const CSPTransform& transform,
                                    uint32_t u_quant_offset,
                                    uint32_t v_quant_offset, uint32_t qfactor) {
  SetYUVBounds(transform.GetYUVMin(), transform.GetYUVMax());
  SetQuality(qfactor, u_quant_offset, v_quant_offset);
  FinalizeQuant();
  quant_y_.SetLambda();
  quant_u_.SetLambda();
  quant_v_.SetLambda();

  // Distortion is slightly more important for chroma than for luma or alpha
  // because the range of values is usually tighter.
  quant_u_.lambda *= 0.84f;
  quant_v_.lambda *= 0.84f;
}

void Segment::SetAlphaQuantizationFactor(uint32_t qfactor) {
  SetAlphaQuality(qfactor);
  FinalizeAlphaQuant();
  quant_a_.SetLambda();
}

//------------------------------------------------------------------------------

}   // namespace WP2
