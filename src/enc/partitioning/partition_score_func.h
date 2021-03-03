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

#ifndef WP2_ENC_PARTITION_SCORE_FUNC_H_
#define WP2_ENC_PARTITION_SCORE_FUNC_H_

#include "src/common/integral.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/lossy/predictor.h"
#include "src/common/symbols.h"
#include "src/dec/wp2_dec_i.h"
#include "src/enc/symbols_enc.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/vector.h"
#include "src/wp2/encode.h"

namespace WP2 {

//------------------------------------------------------------------------------

// Used by Partitioner to evaluate a block layout in a tile.
class PartitionScoreFunc : public WP2Allocable {
 public:
  virtual ~PartitionScoreFunc() = default;

  // Configures the partition settings and allocates memory.
  // 'tile_rect' is in pixels, not padded.
  virtual WP2Status Init(const EncoderConfig& config,
                         const Rectangle& tile_rect, const YUVPlane& yuv,
                         const GlobalParams& gparams,
                         const ProgressRange& progress);

  // For a given block position and size, returns a score (higher means better).
  virtual WP2Status ComputeScore(const Block& block,
                                 const ProgressRange& progress,
                                 float* const score) = 0;

  // Called every time a block is definitively chosen.
  virtual WP2Status Use(const Block& block);

  const GlobalParams& gparams() const { return *gparams_; }

 protected:
  const EncoderConfig* config_ = nullptr;
  const GlobalParams* gparams_ = nullptr;
  Rectangle tile_rect_;            // In pixels, not padded.
  const YUVPlane* src_ = nullptr;  // Original image (tile).
  bool with_alpha_ = false;        // depends on gparams_
  uint32_t num_block_cols_ = 0;    // Maximum number of blocks on the horizontal
  uint32_t num_block_rows_ = 0;    // and vertical axis.

  // Visual debug.
  WP2Status ClearVDebug() const;
  WP2Status RegisterScoreForVDebug(const Block& block, float score,
                                   bool ending_new_line = true) const;
  bool VDebugBlockSelected(const Block& block) const;
};

//------------------------------------------------------------------------------

// Gives a high score to a predetermined block size and 0 to everything else.
class FixedSizeScoreFunc : public PartitionScoreFunc {
 public:
  explicit FixedSizeScoreFunc(BlockSize size) : size_(size) {}

  WP2Status ComputeScore(const Block& block, const ProgressRange& progress,
                         float* const score) override;

 protected:
  BlockSize size_;
};

//------------------------------------------------------------------------------

// Maps 'config.quality' from [0, 95] to [min, (min + slope)].
float MapQuality(const EncoderConfig& config, float min, float slope);

//------------------------------------------------------------------------------

}  // namespace WP2

#endif  // WP2_ENC_PARTITION_SCORE_FUNC_H_
