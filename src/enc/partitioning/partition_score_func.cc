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
// Block position/size scoring functions.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/enc/partitioning/partition_score_func.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "src/common/integral.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/wp2_enc_i.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status PartitionScoreFunc::Init(const EncoderConfig& config,
                                   const Rectangle& tile_rect,
                                   const YUVPlane& yuv,
                                   const GlobalParams& gparams,
                                   const ProgressRange& progress) {
  config_ = &config;
  gparams_ = &gparams;
  tile_rect_ = tile_rect;
  src_ = &yuv;
  with_alpha_ = src_->HasAlpha() && gparams_->maybe_use_lossy_alpha_;
  num_block_cols_ = SizeBlocks(yuv.Y.w_);
  num_block_rows_ = SizeBlocks(yuv.Y.h_);
  WP2_CHECK_REDUCED_STATUS(ClearVDebug());
  WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

WP2Status PartitionScoreFunc::Use(const Block& block) { return WP2_STATUS_OK; }

WP2Status PartitionScoreFunc::CopyInternal(const PartitionScoreFunc& other) {
  config_ = other.config_;
  gparams_ = other.gparams_;
  tile_rect_ = other.tile_rect_;
  src_ = other.src_;
  with_alpha_ = other.with_alpha_;
  num_block_cols_ = other.num_block_cols_;
  num_block_rows_ = other.num_block_rows_;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status FixedSizeScoreFunc::ComputeScore(const Block& block,
                                           const ProgressRange& progress,
                                           float* const score) {
  *score = (block.dim() == size_) ? 1.f : 0.f;
  WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

float MapQuality(const EncoderConfig& config, float min, float slope) {
  const float x = 1.f * config.quality / kMaxLossyQuality;
  return std::max(0.f, min + x * slope);
}

//------------------------------------------------------------------------------

}  // namespace WP2
