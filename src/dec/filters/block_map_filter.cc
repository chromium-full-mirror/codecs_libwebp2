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
// Map of block features to be used by filters.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/dec/filters/block_map_filter.h"

#include <cassert>

#include "src/common/vdebug.h"
#include "src/dec/filters/intertile_filter.h"
#include "src/dec/filters/intratile_filter.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/utils.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

bool IsFilterBlockMapNeeded(const DecoderConfig& config,
                            const BitstreamFeatures& features,
                            const GlobalParams& gparams,
                            const TilesLayout& tiles_layout) {
  return (IsIntratileFilterEnabled(config, features, gparams) ||
          IsIntertileFilterEnabled(config, features, gparams, tiles_layout) ||
          VDMatch(config, "filter-block-map"));
}

//------------------------------------------------------------------------------

void FilterBlockMap::Init(const Rectangle& tile_rect, BitDepth bit_depth,
                          int32_t yuv_min, int32_t yuv_max,
                          YUVPlane* const pixels) {
  if (pixels != nullptr) {
    assert(pixels->GetWidth() >= tile_rect.width &&
           pixels->GetHeight() >= tile_rect.height);
  }
  tile_rect_ = tile_rect;
  bit_depth_ = bit_depth;
  yuv_min_ = yuv_min;
  yuv_max_ = yuv_max;
  pixels_ = pixels;
  max_num_blocks_x_ = SizeBlocks(tile_rect_.width);
  max_num_blocks_y_ = SizeBlocks(tile_rect_.height);

  features_.clear();
  max_registered_height_ = kMinBlockSizePix;  // Minimum block size in pixels.
  current_block_index_ = 0;
}

WP2Status FilterBlockMap::Allocate() {
  const uint32_t max_num_blocks = max_num_blocks_x_ * max_num_blocks_y_;
  WP2_CHECK_ALLOC_OK(features_.resize(max_num_blocks));
  // No need to init 'features_': it will be set by RegisterBlock().
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Returns the mean number of non-zero residuals per coeff for a given channel.
static uint32_t ComputeResidualDensity(const CodedBlockBase& cb,
                                       Channel channel, uint32_t multiplier) {
  const uint32_t num_transforms = cb.GetNumTransforms(channel);
  const uint32_t num_coeffs = cb.NumCoeffsPerTransform(channel);
  uint32_t num_nz_coeffs = 0;
  for (uint32_t tf_i = 0; tf_i < num_transforms; ++tf_i) {
    num_nz_coeffs += cb.num_coeffs_[channel][tf_i];
  }
  return DivRound(num_nz_coeffs * multiplier, num_coeffs * num_transforms);
}

void FilterBlockMap::RegisterBlock(uint32_t x, uint32_t y, BlockSize dim,
                                   uint32_t segment_id, uint8_t min_bpp,
                                   bool has_lossy_alpha,
                                   const uint32_t res_den[4]) {
  const uint32_t stride = max_num_blocks_x_;
  for (uint32_t j = 0; j < BlockHeight[dim]; ++j) {
    BlockFeatures* const row = &features_[(y + j) * stride + x];
    for (uint32_t i = 0; i < BlockWidth[dim]; ++i) {
      row[i].index = current_block_index_;
      row[i].size = dim;
      row[i].segment_id = segment_id;
      row[i].min_bpp = min_bpp;
      row[i].has_lossy_alpha = has_lossy_alpha;
      for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
        row[i].res_den[c] = (uint8_t)res_den[c];
      }
    }
  }
  ++current_block_index_;
}

void FilterBlockMap::RegisterBlock(const CodedBlockBase& cb,
                                   uint32_t min_num_bytes) {
  if (features_.empty()) return;

  const uint32_t num_pixels = NumPix(cb.dim());
  const uint32_t min_bpp =
      Clamp((min_num_bytes * 8 * 128 + num_pixels / 2) / num_pixels, 0u, 255u);
  uint32_t res_den[4];
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !cb.HasLossyAlpha()) {
      res_den[c] = 0;
      continue;
    }
    res_den[c] = Clamp(ComputeResidualDensity(cb, c, 255), 0u, 255u);
  }

  // Fill the block 'features_' in a grid. This way they can be parsed and
  // easily jumped over in filters.
  // If the block has split_tf, we register it as several blocks. For simplicity
  // the split_tf of the Y channel counts for all channels.
  const bool split_tf = cb.GetCodingParams(kYChannel).split_tf;
  const BlockSize split_size = GetSplitSize(cb.dim(), split_tf);
  const uint32_t split_w = BlockWidth[split_size];
  const uint32_t split_h = BlockHeight[split_size];
  for (uint32_t split_y = 0; split_y < cb.h(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < cb.w(); split_x += split_w) {
      RegisterBlock(cb.x() + split_x, cb.y() + split_y, split_size, cb.id_,
                    min_bpp, cb.HasLossyAlpha(), res_den);
    }
  }

  max_registered_height_ = std::max(max_registered_height_, cb.h_pix());
}

void FilterBlockMap::Clear() {
  Init(/*tile_rect=*/{0, 0, 0, 0}, /*bit_depth=*/{0, false},
       /*yuv_min=*/0, /*yuv_max=*/0, /*pixels=*/nullptr);
}

//------------------------------------------------------------------------------

const FilterBlockMap::BlockFeatures* FilterBlockMap::GetRow(uint32_t y) const {
  assert(!features_.empty() && y < max_num_blocks_y_);
  return features_.data() + y * max_num_blocks_x_;
}

//------------------------------------------------------------------------------

}  // namespace WP2
