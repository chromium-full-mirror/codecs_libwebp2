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

#ifndef WP2_ENC_PARTITION_SCORE_FUNC_BLOCK_H_
#define WP2_ENC_PARTITION_SCORE_FUNC_BLOCK_H_

#include "src/common/integral.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/lossy/predictor.h"
#include "src/common/symbols.h"
#include "src/dec/wp2_dec_i.h"
#include "src/enc/partitioning/partition_score_func.h"
#include "src/enc/symbols_enc.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/vector.h"
#include "src/wp2/encode.h"

namespace WP2 {

//------------------------------------------------------------------------------

// Based on block encoding. Can only be used with a TopLeftBestPartitioner.
class BlockScoreFunc : public PartitionScoreFunc {
 public:
  WP2Status Init(const EncoderConfig& config, const Rectangle& tile_rect,
                 const YUVPlane& yuv, const GlobalParams& gparams,
                 const ProgressRange& progress) override;

  WP2Status ComputeScore(const Block& block, const ProgressRange& progress,
                         float* const score) override;

  // Returns the score for encoding several 'blocks' (at most 4).
  // Can be used for evaluating a block against its sub-blocks, for example.
  WP2Status ComputeScore(const Block blocks[4], uint32_t num_blocks,
                         float* const score);

  WP2Status Use(const Block& block) override;

 protected:
  // Tries some or all transform/predictor pairs and keeps the best one in 'cb'.
  WP2Status FindBestBlockParams(const FrontMgrNxNBase& front_mgr,
                                const BlockContext& block_context,
                                SyntaxWriter* const syntax_writer,
                                DCDiffusionMap* const dc_error_u,
                                DCDiffusionMap* const dc_error_v,
                                CodedBlock* const cb) const;
  // Finds the best encoding parameters then encodes and records 'cb'.
  // Advances the 'front_mgr'.
  WP2Status EncodeBlock(const FrontMgrNxNBase& front_mgr, CodedBlock* const cb,
                        SyntaxWriter* const syntax_writer,
                        DCDiffusionMap* const dc_error_u,
                        DCDiffusionMap* const dc_error_v,
                        YUVPlane* const buffer) const;
  // Writes 'cb' to 'enc'. Advances the 'front_mgr'.
  WP2Status WriteBlock(const FrontMgrNxNBase& front_mgr, const CodedBlock& cb,
                       SyntaxWriter* const syntax_writer,
                       ANSEnc* const enc) const;

  // Defined by 'config_' and not modified outside Init().
  Vector<TransformPair> transforms_, transforms_subset_;

  // Instances for recording and estimating the bit-rate/disto per block.
  mutable YUVPlane buffer_;  // Reconstructed blocks (final + under test).
  FrontMgrCustom front_mgr_;
  ANSDictionaries dicts_;
  SyntaxWriter syntax_writer_;
  DCDiffusionMap dc_error_u_, dc_error_v_;

  // Temporary instances.
  CodedBlock tmp_cb_;
  ANSDictionaries tmp_dicts_;
  SyntaxWriter tmp_syntax_writer_;
  DCDiffusionMap tmp_dc_error_u_, tmp_dc_error_v_;
  BlockContext context_;

 private:
  WP2Status RegisterScoreForVDebug(const Block blocks[], uint32_t num_blocks,
                                   const float rate[4], const float disto[4],
                                   float total_rate, float total_disto,
                                   float score);
};

//------------------------------------------------------------------------------

}  // namespace WP2

#endif  // WP2_ENC_PARTITION_SCORE_FUNC_BLOCK_H_
