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
//  Global parameters used by lossy / lossless tiles
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include "src/common/global_params.h"

#include "src/common/constants.h"
#include "src/common/lossy/segment.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"

namespace WP2 {

WP2Status GlobalParams::Write(bool image_has_alpha, ANSEnc* const enc) const {
  ANSDebugPrefix prefix(enc, "GlobalParams");

  enc->PutRValue(type_, image_has_alpha ? GP_AV1 : GP_LAST, "gp_type");

  if (type_ == GP_LOSSLESS) {
    assert(quality_hint_ == kLosslessQualityHint);
  } else if (type_ == GP_LOSSY || type_ == GP_BOTH) {
    enc->PutRValue(quality_hint_, kMaxLossyQualityHint + 1, "quality_hint");
  } else {
    assert(type_ == GP_AV1);
    assert(quality_hint_ == kLosslessQualityHint / 2);
  }

  if (image_has_alpha) {
    enc->PutBool(has_alpha_, "has_alpha");
  } else {
    assert(!has_alpha_);
  }

  if (type_ == GP_LOSSY || type_ == GP_BOTH) {
    if (has_alpha_) {
      enc->PutBool(maybe_use_lossy_alpha_, "maybe_use_lossy_alpha");
    } else {
      assert(!maybe_use_lossy_alpha_);
    }
    WP2_CHECK_STATUS(transf_.Write(enc));  // write inverse matrix
    enc->PutBool(explicit_segment_ids_, "explicit_segment_ids");
    enc->PutRValue(partition_set_, NUM_PARTITION_SETS, "block_size");
    enc->PutRange(
        segments_.size(), 1,
        GetMaxNumSegments(explicit_segment_ids_, quality_hint_, partition_set_),
        "num_segments_id");
    enc->PutBool(partition_snapping_, "block_size");
    bool use_grain = false;
    for (const Segment& s : segments_) {
      use_grain |= s.grain_.IsUsed();
      if (use_grain) break;
    }
    enc->PutBool(use_grain, "grain");

    const bool has_uv_quant_multiplier =
        (u_quant_multiplier_ != kDefaultQuantMultiplier) ||
        (v_quant_multiplier_ != kDefaultQuantMultiplier);
    enc->PutBool(has_uv_quant_multiplier, "uv_quant_multiplier");
    if (has_uv_quant_multiplier) {
      enc->PutRange(u_quant_multiplier_, 1, kNumQuantMultiplierValues,
                    "uv_quant_multiplier");
      if (u_quant_multiplier_ == kDefaultQuantMultiplier) {
        enc->PutRange((v_quant_multiplier_ > kDefaultQuantMultiplier)
                          ? (v_quant_multiplier_ - 1)
                          : v_quant_multiplier_,
                      1, kNumQuantMultiplierValues - 1, "uv_quant_multiplier");
      } else {
        enc->PutRange(v_quant_multiplier_, 1, kNumQuantMultiplierValues,
                      "uv_quant_multiplier");
      }
    }
    bool write_alpha = maybe_use_lossy_alpha_;
    for (const Segment& s : segments_) {
      WP2_CHECK_STATUS(s.WriteHeader(enc, write_alpha, use_grain));
      write_alpha = false;   // only for the first segment
    }
    // TODO(skal): use some fixed-value probas
    enc->PutBool(use_rnd_mtx_, "rnd_mtx");  // TODO(skal): write num_mtx?

    if (maybe_use_lossy_alpha_) {
      enc->PutBool(enable_alpha_filter_, "enable_alpha_filter");
    } else {
      assert(!enable_alpha_filter_);
    }
  }
  if (type_ == GP_LOSSLESS || type_ == GP_BOTH) {
    // write lossless
  }
  if (type_ == GP_AV1) {
    // write lossy AV1
  }

  return WP2_STATUS_OK;
}

// TODO(skal): use delta-coding instead, for saving more bits
WP2Status GlobalParams::Read(bool image_has_alpha, ANSDec* const dec) {
  ANSDebugPrefix prefix(dec, "GlobalParams");

  type_ = (Type)dec->ReadRValue(image_has_alpha ? GP_AV1 : GP_LAST, "gp_type");

  if (type_ == GP_LOSSLESS) {
    quality_hint_ = kLosslessQualityHint;
  } else if (type_ == GP_LOSSY || type_ == GP_BOTH) {
    quality_hint_ = dec->ReadRValue(kMaxLossyQualityHint + 1, "quality_hint");
  } else {
    assert(type_ == GP_AV1);
    quality_hint_ = kMaxLossyQualityHint / 2;  // Unused so whatever.
  }

  has_alpha_ = image_has_alpha && dec->ReadBool("has_alpha");

  if (type_ == GP_LOSSY || type_ == GP_BOTH) {
    maybe_use_lossy_alpha_ =
        has_alpha_ && dec->ReadBool("maybe_use_lossy_alpha");

    WP2_CHECK_STATUS(transf_.Read(dec));
    explicit_segment_ids_ = dec->ReadBool("explicit_segment_ids");
    // segments
    partition_set_ =
        (PartitionSet)dec->ReadRValue(NUM_PARTITION_SETS, "block_size");
    const uint32_t num_segments = dec->ReadRange(
        1,
        GetMaxNumSegments(explicit_segment_ids_, quality_hint_, partition_set_),
        "num_segments_id");
    WP2_CHECK_ALLOC_OK(segments_.resize(num_segments));
    partition_snapping_ = dec->ReadBool("block_size");

    const bool use_grain = dec->ReadBool("grain");

    const bool has_uv_quant_multiplier = dec->ReadBool("uv_quant_multiplier");
    if (has_uv_quant_multiplier) {
      u_quant_multiplier_ =
          dec->ReadRange(1, kNumQuantMultiplierValues, "uv_quant_multiplier");
      if (u_quant_multiplier_ == kDefaultQuantMultiplier) {
        v_quant_multiplier_ = dec->ReadRange(1, kNumQuantMultiplierValues - 1,
                                             "uv_quant_multiplier");
        if (v_quant_multiplier_ >= kDefaultQuantMultiplier) {
          ++v_quant_multiplier_;
        }
      } else {
        v_quant_multiplier_ =
            dec->ReadRange(1, kNumQuantMultiplierValues, "uv_quant_multiplier");
      }
    }

    bool read_alpha = maybe_use_lossy_alpha_;
    for (Segment& segment : segments_) {
      segment.SetYUVBounds(transf_.GetYUVMin(), transf_.GetYUVMax());
      WP2_CHECK_STATUS(segment.ReadHeader(quality_hint_, u_quant_multiplier_,
                                          v_quant_multiplier_, read_alpha,
                                          use_grain, dec));
      read_alpha = false;   // only for the first segment
    }
    use_rnd_mtx_ = dec->ReadBool("rnd_mtx");
    WP2_CHECK_STATUS(InitRndMtxSet());
    WP2_CHECK_STATUS(InitFixedPredictors());

    enable_alpha_filter_ =
        maybe_use_lossy_alpha_ && dec->ReadBool("enable_alpha_filter");
  }
  if (type_ == GP_LOSSLESS || type_ == GP_BOTH) {
    // read lossless
  }
  if (type_ == GP_AV1) {
    // read lossy AV1
  }
  return WP2_STATUS_OK;
}

WP2Status GlobalParams::InitFixedPredictors() {
  y_preds_.reset();
  WP2_CHECK_STATUS(y_preds_.Fill(transf_.GetYUVMin(), transf_.GetYUVMax()));
  uv_preds_.reset();
  WP2_CHECK_STATUS(uv_preds_.Fill(transf_.GetYUVMin(), transf_.GetYUVMax()));

  if (maybe_use_lossy_alpha_) {
    WP2_CHECK_STATUS(InitAlphaPredictors(transf_, &a_preds_));
  }

  return WP2_STATUS_OK;
}

WP2Status GlobalParams::InitRndMtxSet() {
  if (use_rnd_mtx_) {
    const uint32_t num_mtx = kMaxNumRndMtx;
    assert(!segments_.empty());
    WP2_CHECK_STATUS(mtx_set_.Init(num_mtx, segments_[0].quant_y_));
  } else {
    mtx_set_.Reset();
  }
  return WP2_STATUS_OK;
}

bool GlobalParams::IsOk() const {
  if (y_preds_.size() > kYPredNum) return false;
  if (uv_preds_.size() > kUVPredNum) return false;
  if (u_quant_multiplier_ < 1 ||
      u_quant_multiplier_ > kNumQuantMultiplierValues)
    return false;
  if (v_quant_multiplier_ < 1 ||
      v_quant_multiplier_ > kNumQuantMultiplierValues)
    return false;
  if (a_preds_.size() > kAPredNum) return false;
  return true;
}

uint32_t GlobalParams::GetMaxDCRange(Channel channel) const {
  uint32_t max_abs_dc = 0;
  for (const Segment& s : segments_) {
    // Figure out the maximum range.
    max_abs_dc = std::max(max_abs_dc, s.GetMaxAbsDC(channel));
  }
  assert(max_abs_dc <= kMaxDcValue);
  // 2 * max_abs_dc + 1 as DC is in [-max_abs_dc, max_abs_dc], zero included.
  return 2u * max_abs_dc + 1u;
}

void GlobalParams::Reset() {
  type_ = GP_LOSSY;
  has_alpha_ = false;
  transf_.InitYCoCg();
  segments_.reset();
  y_preds_.reset();
  uv_preds_.reset();
  maybe_use_lossy_alpha_ = false;
  enable_alpha_filter_ = false;
  a_preds_.reset();
  explicit_segment_ids_ = false;
  partition_set_ = PartitionSet::ALL_SQUARES;
  partition_snapping_ = false;
  use_rnd_mtx_ = false;
  mtx_set_.Reset();
  features_ = nullptr;
  u_quant_multiplier_ = kDefaultQuantMultiplier;
  v_quant_multiplier_ = kDefaultQuantMultiplier;
}

}    // namespace WP2
