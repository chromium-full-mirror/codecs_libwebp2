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
// Tool for finding the best block layout.
//
// Author: Yannis Guyon (yguyon@google.com)

#ifndef WP2_ENC_PARTITIONER_SPLIT_H_
#define WP2_ENC_PARTITIONER_SPLIT_H_

#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/predictor.h"
#include "src/common/symbols.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/vector.h"
#include "src/wp2/encode.h"

namespace WP2 {

//------------------------------------------------------------------------------

// For each block, tries some split patterns and keeps the best one then
// recurses on sub-blocks.
class SplitRecursePartitioner : public Partitioner {
 public:
  WP2Status Init(const EncoderConfig& config, const YUVPlane& yuv,
                 const Rectangle& tile_rect,
                 PartitionScoreFunc* const score_func) override;
  WP2Status GetBestPartition(const ProgressRange& progress,
                             VectorNoCtor<Block>* const blocks) override;

 private:
  // Returns the maximum BlockSize fitting in 'max_block' and respecting all the
  // constraints (not overlapping a forced block or a previously-decided split
  // pattern, belonging to the partition set etc.).
  BlockSize GetMaxFittingBlockSize(const Block& max_block,
                                   uint32_t num_forced_blocks) const;

  // Contains one way of splitting a block into smaller ones.
  struct SplitPattern {
    uint32_t num_blocks;       // Number of blocks in this split.
    BlockSize block_sizes[4];  // Top-left,top-right,bottom-left,bottom-right
                               // (or subset in same order if less than 4).
  };
  // Returns all possible SplitPatterns given a 'set' and a 'size'.
  static void GetSplitPatterns(PartitionSet set, BlockSize size,
                               Vector<SplitPattern>* const p);
  // Returns the 'blocks' given a 'max_block' (size and position) and a
  // 'split_pattern'.
  static void GetSplitPatternBlocks(const Block& max_block,
                                    const SplitPattern& split_pattern,
                                    Block blocks[4]);

  // Contains whether a SplitPattern was decided for a given block position.
  struct SplitInfo {
    BlockSize size;  // BLK_LAST if nothing decided yet.
    bool recursive;  // False if the decision is definitive
                     // or true if smaller blocks may be tried.
    bool operator==(const SplitInfo& other) const {
      return size == other.size && recursive == other.recursive;
    }
    bool operator!=(const SplitInfo& other) const { return !operator==(other); }
  };
  static constexpr SplitInfo kNoSplitInfo = {BLK_LAST, false};  // No decision.

  // Gets the split pattern decision for a given block position.
  const SplitInfo& GetSplitInfo(uint32_t x, uint32_t y) const;
  // Returns true if the split pattern decision is 'split_info' in all the given
  // 'block' area.
  bool IsSplitInfo(const Block& block, const SplitInfo& split_info) const;
  // Sets the split pattern decision for a given 'block' area.
  void SetSplitInfo(const Block& block, bool recursive);

  // Contains whether a SplitPattern was decided or not for each block position.
  Plane<SplitInfo> split_info_;
};

//------------------------------------------------------------------------------

}  // namespace WP2

#endif  // WP2_ENC_PARTITIONER_SPLIT_H_
