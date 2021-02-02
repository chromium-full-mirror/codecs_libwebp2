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
//   CodedBlockBase, for decoding
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/lossy/block.h"

#include <algorithm>
#include <cassert>

#include "src/common/lossy/block_size.h"
#include "src/common/lossy/rnd_mtx.h"
#include "src/common/lossy/transforms.h"
#include "src/common/symbols.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/ans.h"
#include "src/utils/front_mgr.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------
// ContextCache

const int16_t* ContextCache::GetLarge(const CodedBlockBase& cb, Channel channel,
                                      ContextType context_type) {
  if (!is_computed_[channel][context_type]) {
    cb.GetContext(channel, context_type, large_[channel][context_type]);
    is_computed_[channel][context_type] = true;
  }
  return large_[channel][context_type];
}

void ContextCache::Reset() {
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    for (bool& b : is_computed_[channel]) b = false;
  }
}

//------------------------------------------------------------------------------
// CodedBlockBase

void CodedBlockBase::SetRange(int16_t yuv_min, int16_t yuv_max) {
  yuv_min_ = yuv_min;
  yuv_max_ = yuv_max;
}

void CodedBlockBase::SetDim(const Block& block, const FrontMgrBase& mgr) {
  blk_ = block;
  left_occupancy_ = mgr.GetOccupancy((int)blk_.x() - 1);
  right_occupancy_ = mgr.GetOccupancy(blk_.x() + blk_.w());
  top_context_extent_ = mgr.GetRightContextExtent(blk_);
}

void CodedBlockBase::SetDimDefault(const Block& block, bool full_left_ctx) {
  blk_ = block;
  left_occupancy_ =
      blk_.y() + (blk_.x() > 0 ? (full_left_ctx ? blk_.h() : 1) : 0);
  right_occupancy_ = blk_.y();
  top_context_extent_ = 0u;
}

void CodedBlockBase::SetXY(const FrontMgrBase& mgr, uint32_t x, uint32_t y) {
  blk_.SetXY(x, y);
  left_occupancy_ = mgr.GetOccupancy((int)blk_.x() - 1);
  right_occupancy_ = mgr.GetOccupancy(blk_.x() + blk_.w());
  top_context_extent_ = mgr.GetRightContextExtent(blk_);
}

void CodedBlockBase::SetContextInput(const YUVPlane& in,
                                     ContextCache* const context_cache) {
  WP2_ASSERT_STATUS(predict_from_.SetView(in, blk_.rect_pix()));
  if (context_cache != nullptr) context_cache_ = context_cache;
  ResetContextCache();
}

void CodedBlockBase::SetReconstructedOutput(YUVPlane* const out) {
  WP2_ASSERT_STATUS(out_.SetView(*out, blk_.rect_pix()));
}

//------------------------------------------------------------------------------

void CodedBlockBase::GetOccupancy(int8_t* const left, int8_t* const right,
                                  int8_t* const top_context_extent) const {
  *left = std::max(-1, (int)(left_occupancy_ - blk_.y()));
  *right = std::max(-1, (int)(right_occupancy_ - blk_.y()));
  if (top_context_extent != nullptr) {
    *top_context_extent = top_context_extent_;
  }
}

void CodedBlockBase::SetContextCache(ContextCache* const context_cache) {
  context_cache_ = context_cache;
}

void CodedBlockBase::ResetContextCache() const {
  assert(context_cache_ != nullptr);
  context_cache_->Reset();
}

const int16_t* CodedBlockBase::GetContext(Channel channel,
                                          ContextType context_type) const {
  assert(context_cache_ != nullptr);
  return context_cache_->GetLarge(*this, channel, context_type);
}

//------------------------------------------------------------------------------

static int CodingParamsIndex(Channel channel) {
  if (channel == kYChannel) return 0;
  if (channel == kUChannel || channel == kVChannel) return 1;
  if (channel == kAChannel) return 2;
  assert(false);
  return -1;
}

const CodedBlockBase::CodingParams& CodedBlockBase::GetCodingParams(
    Channel channel) const {
  return params_[CodingParamsIndex(channel)];
}

CodedBlockBase::CodingParams* CodedBlockBase::GetCodingParams(Channel channel) {
  return &params_[CodingParamsIndex(channel)];
}

TransformPair CodedBlockBase::GetImplicitTf(Channel channel) const {
  // TODO(maryla) check if it is still needed.
  if (channel == kAChannel) return kDctDct;
  TransformPair tf;
  if (GetCodingParams(channel).pred->OverridesTransform(channel, &tf)) {
    return tf;
  }
  return kUnknownTf;
}

//------------------------------------------------------------------------------

uint32_t CodedBlockBase::GetNumTransforms(Channel channel) const {
  return GetNumTransformsInBlock(dim(), GetCodingParams(channel).split_tf);
}

uint32_t CodedBlockBase::NumCoeffsPerTransform(Channel channel) const {
  return kNumBlocks[GetSplitSize(blk_.dim(),
                                 GetCodingParams(channel).split_tf)] *
         (((channel == kUChannel || channel == kVChannel) && is420_) ? 4 : 16);
}

bool CodedBlockBase::IsFirstCoeffDC(Channel channel) const {
  return (GetCodingParams(channel).tf_x() != kIdentity &&
          GetCodingParams(channel).tf_y() != kIdentity);
}

bool CodedBlockBase::HasCoeffs(Channel channel) const {
  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    if (num_coeffs_[channel][tf_i] != 0) return true;
  }
  return false;
}

//------------------------------------------------------------------------------

void CodedBlockBase::PredictBlock(Channel channel, int16_t* out,
                                  uint32_t step) const {
  const CodingParams& params = GetCodingParams(channel);
  params.pred->Predict(*this, channel, out, step);
}

//------------------------------------------------------------------------------
// Simplified all-in-one version for decoder

void CodedBlockBase::Reconstruct(Channel channel) {
  const bool reduced_transform =
      ((channel == kUChannel || channel == kVChannel) && is420_);
  Plane16& out = out_.GetChannel(channel);
  PredictBlock(channel, out.Row(0), out.Step());

  const int32_t min = (channel == kAChannel) ? 0 : yuv_min_;
  const int32_t max = (channel == kAChannel) ? kAlphaMax : yuv_max_;
  const CodingParams* const params = GetCodingParams(channel);
  const BlockSize split_size = GetSplitSize(dim(), params->split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  AddBlockEqFunc add_block_eq = WP2::AddBlockEq[TrfLog2[split_w] - 2];
  for (uint32_t tf_i = 0, split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w, ++tf_i) {
      // TODO(skal): if AllZero, we shouldn't be here (+special case for DCOnly)
      if (use_mtx_ && channel == kYChannel) {
        const TrfSize tsize = tdim(channel);
        assert(mtx_set_->IsOk(mtx_[channel], tsize));
        mtx_set_->AddMatrix(mtx_[channel], tsize, coeffs_[channel][tf_i]);
      }
      WP2InvTransform2D(coeffs_[channel][tf_i],  params->tf_x(), params->tf_y(),
                        split_w, split_h, reduced_transform);
      add_block_eq(coeffs_[channel][tf_i], split_w,
                   &out.At(split_x, split_y), out.Step(), min, max, split_h);
    }
  }
}

//------------------------------------------------------------------------------

bool CodedBlockBase::ContextIsConstant(Channel channel) const {
  assert(context_cache_ != nullptr);
  const int16_t* context =
      context_cache_->GetLarge(*this, channel, kContextSmall);
  const uint32_t context_size =
      ContextSize(kContextSmall, blk_.w_pix(), blk_.h_pix());
  for (uint32_t i = 1; i < context_size; ++i) {
    if (context[i] != context[0]) {
      return false;
    }
  }
  return true;
}

bool CodedBlockBase::HasLossyAlpha() const {
  return (alpha_mode_ == kBlockAlphaLossy);
}

//------------------------------------------------------------------------------

void CodedBlockBase::SetUVPredictor(const Predictor* const pred) {
  CodedBlockBase::CodingParams* const params = GetCodingParams(kUChannel);
  params->pred = pred;
  if (pred->OverridesTransform(kUChannel, &params->tf)) {
    assert(params->tf != kUnknownTf);
  } else {
    assert(false);  // The transform should always be overridden for U and V.
  }
}

void CodedBlockBase::SetPredictor(Channel channel,
                                  const Predictor* const pred) {
  if (channel == kUChannel || channel == kVChannel) {
    SetUVPredictor(pred);
  } else {
    CodingParams* const params = GetCodingParams(channel);
    params->pred = pred;
  }
}

void CodedBlockBase::SetLumaUniformPredictor(const Predictors& preds) {
  SetPredictor(kYChannel, preds[0]);
}

//------------------------------------------------------------------------------

// Macro replicating AtUnclamped for speed-up.
#define CONTEXT_SRC_AT_UNCLAMPED(X, Y) \
  *(context_src_data + (X) + (Y) * context_src_step)

void CodedBlockBase::GetContext(Channel channel, ContextType context_type,
                                int16_t context[]) const {
  const bool fill_in = (context_type != kContextSmallNoFillIn);
  const uint32_t w = blk_.w_pix(), h = blk_.h_pix();
  const Plane16& context_src = predict_from_.GetChannel(channel);
  if (context_src.IsEmpty()) {
    std::fill(context, context + kMaxContextSize, 0);
    return;
  }
  // absolute position within dst_ picture
  const uint32_t y_pix = blk_.y_pix();
  // implement all the neighbouring sample logic:
  const bool have_top = (y_pix > 0);
  int8_t y_left_occ, y_right_occ, right_extent;
  GetOccupancy(&y_left_occ, &y_right_occ, &right_extent);
  // Number of available context pixels on the left and right sides.
  const int left_context_size =
      (std::max(0, (int)y_left_occ) * (int)kMinBlockSizePix);
  const bool have_top_left = have_top && (-1 < y_left_occ);
  const bool have_top_right = have_top && (-1 < y_right_occ);

  // collect the context samples:
  uint32_t m = 0;
  // this is the value we reuse in case of non-availability
  int16_t last = fill_in ? 0 : kMissing;
  // left (bh samples)
  const int16_t* const context_src_data =
      context_src.IsEmpty() ? nullptr : &context_src.AtUnclamped(0, 0);
  const int context_src_step = context_src.Step();
  const int left_size = ((context_type == kContextExtendLeft) ? 2 * h : h);
  if (left_context_size > 0) {
    if (fill_in) last = CONTEXT_SRC_AT_UNCLAMPED(-1, left_context_size - 1);
    int j = left_size - 1;
    for (; j >= left_context_size; --j) context[m++] = last;
    for (; j >= 0; --j) context[m++] = CONTEXT_SRC_AT_UNCLAMPED(-1, j);
    if (fill_in) last = context[m - 1];
  } else {
    if (fill_in) {
      last =
          have_top ? CONTEXT_SRC_AT_UNCLAMPED(have_top_left ? -1 : 0, -1) : 0;
    }
    for (uint32_t j = 0; j < (uint32_t)left_size; ++j) context[m++] = last;
  }

  if (context_type != kContextExtendLeft) {
    // top-left
    if (have_top_left) {
      context[m++] = CONTEXT_SRC_AT_UNCLAMPED(-1, -1);
      if (fill_in) last = context[m - 1];
    } else {
      context[m++] = last;
    }
    // top (bw samples)
    if (have_top) {
      for (uint32_t i = 0; i < w; ++i) {
        context[m++] = CONTEXT_SRC_AT_UNCLAMPED(i, -1);
      }
      if (fill_in) last = context[m - 1];
    } else {
      if (fill_in) {
        last = (left_context_size > 0) ? CONTEXT_SRC_AT_UNCLAMPED(-1, 0) : 0;
      }
      for (uint32_t i = 0; i < w; ++i) context[m++] = last;
    }

    if (context_type == kContextExtendRight) {
      const uint32_t context_size = ContextSize(kContextExtendRight, w, h);
      right_extent *= kMinBlockSizePix;
      int j = 0;
      for (; j < right_extent && m < context_size; ++j) {
        context[m++] = CONTEXT_SRC_AT_UNCLAMPED(w + j, -1);
      }
      if (fill_in) last = context[m - 1];
      for (; m < context_size; ++j) context[m++] = last;
    } else {
      // top-right
      if (have_top_right) {
        context[m++] = CONTEXT_SRC_AT_UNCLAMPED(w, -1);
      } else {
        context[m++] = last;
      }
      // right (bh samples)
      uint32_t j = 0;
      const uint32_t right_context_size =
          std::min(h, std::max(0, (int)y_right_occ) * kMinBlockSizePix);
      for (; j < right_context_size; ++j) {
        context[m++] = CONTEXT_SRC_AT_UNCLAMPED(w, j);
      }
      if (fill_in) last = context[m - 1];
      for (; j < h; ++j) context[m++] = last;
      assert(m == ContextSize(kContextSmall, w, h));
    }
  }

  const uint32_t kSizeThreshold = 48;  // context size above which to filter
  const int32_t kDiffThreshold = 96;   // slope threshold below which to filter
  if (m > kSizeThreshold) {   // block is large enough?
    int32_t prev = context[0], cur = context[0];
    uint32_t prev_diff = 0;
    for (uint32_t i = 0; i < m - 1; ++i) {
      const int32_t next = context[i + 1];
      // Smaller or un-aligned blocks in the context can lead to edge smearing
      // during prediction (not fixed by the deblocking filter because it
      // happens inside the blocks). Hence low-pass smooth the context samples
      // that don't seem to belong to a real edge.
      const uint32_t next_diff = std::abs(cur - next);
      if (prev_diff + next_diff < kDiffThreshold) {
        context[i] = (6 * cur + prev + next + 4) >> 3;
      }
      prev_diff = next_diff;
      prev = cur;
      cur = next;
    }
    const int32_t next = context[m - 1];
    if (prev_diff < kDiffThreshold) {
      context[m - 1] = (6 * cur + prev + next + 4) >> 3;
    }
  }
}

#undef CONTEXT_SRC_AT_UNCLAMPED

//------------------------------------------------------------------------------

std::string CodedBlockBase::Print() const {
  std::string str;
#if defined(WP2_BITTRACE)
  str += WP2SPrint(" yuv: [%d - %d]\n", yuv_min_, yuv_max_);
  str += WP2SPrint(" cfl_: [%d, %d, %d]\n", cfl_[0], cfl_[1], cfl_[2]);
  str += WP2SPrint(" blk_: [%d, %d, %d x %d]\n",
                   x_pix(), y_pix(), w_pix(), h_pix());
  str += WP2SPrint(" occupancy:%d %d\n", left_occupancy_, right_occupancy_);
  str += WP2SPrint(" top-ctx:%d\n", top_context_extent_);
  str += WP2SPrint(" id:%d  420:%d ", id_, is420_);
  str += WP2SPrint(" use_mtx:%d ", use_mtx_);
  str += WP2SPrint(" alpha_mode_:%d ", alpha_mode_);
  str += WP2SPrint(" y_context_is_constant_:%d\n", y_context_is_constant_);
  for (const CodingParams& prm : params_) {
    str += WP2SPrint("  params_[%d]: trf:%d split:%d\n",
                      &prm - params_, prm.tf, prm.split_tf);
  }
#endif
  return str;
}

//------------------------------------------------------------------------------

}  // namespace WP2
