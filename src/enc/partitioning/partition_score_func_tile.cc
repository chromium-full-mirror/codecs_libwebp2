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

#include "src/enc/partitioning/partition_score_func_tile.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/common/constants.h"
#include "src/common/global_params.h"
#include "src/common/lossy/block_size.h"
#include "src/common/progress_watcher.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/dsp.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partition_score_func.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/ans.h"
#include "src/utils/ans_enc.h"
#include "src/utils/data_source.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status TileScoreFunc::Init(const EncoderConfig& config,
                              const Rectangle& tile_rect, const YUVPlane& yuv,
                              const GlobalParams& gparams,
                              const ProgressRange& progress) {
  WP2EncDspInit();
  const ProgressRange init_progress(progress, 0.2);
  const ProgressRange forced_partition_progress(progress, 0.8);
  WP2_CHECK_STATUS(
      PartitionScoreFunc::Init(config, tile_rect, yuv, gparams, init_progress));
  local_gparams_.features_ = &local_features_map_;
  WP2_CHECK_STATUS(GlobalAnalysis(ArgbBuffer(), yuv, gparams.transf_, config,
                                  &local_gparams_));
  WP2_CHECK_STATUS(InitForEncode());

  // Initialize the best score with the partition containing only the forced
  // blocks.
  const Rectangle padded_tile_rect = {tile_rect_.x, tile_rect_.y,
                                      yuv.GetWidth(), yuv.GetHeight()};
  WP2_CHECK_STATUS(AddForcedBlocks(config, padded_tile_rect, &blocks_));
  WP2_CHECK_STATUS(TryEncode(blocks_, forced_partition_progress, &best_score_));
  cached_best_score_ = 0.f;
  blocks_.clear();
  WP2_CHECK_REDUCED_STATUS(RegisterScoreForVDebug("starting", {}, best_score_));
  return WP2_STATUS_OK;
}

WP2Status TileScoreFunc::ComputeScore(const Block& block,
                                      const ProgressRange& progress,
                                      float* const score) {
  WP2_CHECK_ALLOC_OK(blocks_.push_back(block));
  WP2_CHECK_STATUS(TryEncode(blocks_, progress, score));
  if (*score > cached_best_score_) cached_best_score_ = *score;
  if (*score > best_score_) {
    WP2_CHECK_REDUCED_STATUS(RegisterScoreForVDebug("new best", block, *score));
  }
  blocks_.pop_back();
  return WP2_STATUS_OK;
}

WP2Status TileScoreFunc::InitForEncode() {
  WP2_CHECK_ALLOC_OK(blocks_.reserve((tile_rect_.width / kMaxBlockSizePix) *
                                     (tile_rect_.height / kMaxBlockSizePix)));

  enc_tiles_layout_.num_tiles_x = enc_tiles_layout_.num_tiles_y = 1;
  enc_tiles_layout_.tile_width = tile_rect_.width;
  enc_tiles_layout_.tile_height = tile_rect_.height;
  WP2_CHECK_ALLOC_OK(enc_tiles_layout_.tiles.resize(1));
  enc_tiles_layout_.first_unassigned_tile_index = 0;

  enc_tiles_layout_.tiles.front().rect = {0, 0, tile_rect_.width,
                                          tile_rect_.height};
  enc_tiles_layout_.tiles.front().rgb_input.Deallocate();  // This is lossy.
  assert(!src_->IsEmpty());
  WP2_CHECK_STATUS(enc_tiles_layout_.tiles.front().yuv_input.SetView(*src_));
  tmp_config_ = *config_;
  tmp_config_.partition_method = sub_partition_method_;
  tmp_config_.info = nullptr;
  tile_encoder_.config_ = &tmp_config_;
  tile_encoder_.use_lossless_ = (tmp_config_.quality > kMaxLossyQuality);
  tile_encoder_.tiles_layout_ = &enc_tiles_layout_;
  WP2_CHECK_STATUS(tile_encoder_.AssignNextTile());

  // Recursion is too dangerous here. It's potentially creating
  // (kMaxTileSize/kMinBlockSizePix)^2 = a lot of recursive encoding contexts.
  assert(sub_partition_method_ != AUTO_PARTITIONING &&
         sub_partition_method_ != TILE_ENCODE_PARTITIONING);

  dec_config_.thread_level = 0;

  WP2_CHECK_STATUS(decompressed_yuv_.Copy(*src_, /*resize_if_needed=*/true));
  // Needed for API compliance. The pixels will not be accessed.
  WP2_CHECK_STATUS(
      decompressed_argb_.Resize(tile_rect_.width, tile_rect_.height));

  // A BitstreamFeatures instance is needed by LossyDecode(). Make up one.
  const bool is_premultiplied =
      WP2IsPremultiplied(decompressed_argb_.format()) ||
      !tmp_config_.keep_unmultiplied;
  MemoryWriter writer;
  WP2_CHECK_STATUS(EncodeHeader(
      tmp_config_, tile_rect_.width, tile_rect_.height, /*rgb_bit_depth=*/8,
      with_alpha_, is_premultiplied, /*is_anim=*/false, /*loop_forever=*/true,
      kDefaultBackgroundColor, /*preview_color=*/{}, /*has_icc=*/false,
      /*has_trailing_data=*/false, &writer));
  WP2_CHECK_STATUS(features_.Read(writer.mem_, writer.size_));
  return WP2_STATUS_OK;
}

WP2Status TileScoreFunc::TryEncode(const VectorNoCtor<Block>& blocks,
                                   const ProgressRange& progress,
                                   float* const score) {
  ANSEnc& enc = enc_tiles_layout_.tiles.front().enc;
  enc.Reset();
  enc_tiles_layout_.gparams = &local_gparams_;
  enc_tiles_layout_.image_is_premultiplied = true;
  tile_encoder_.tile_->progress = progress;
  // Encode the whole tile with the forced 'blocks'.
  WP2_CHECK_STATUS(tile_encoder_.LossyEncode(blocks, &enc));
  WP2_CHECK_STATUS(enc.AssembleToBitstream(/*clear_tokens=*/true));
  Vector_u8 bits;
  WP2_CHECK_STATUS(enc.WriteBitstreamTo(bits));

  // Reset the unique tile to a fresh state.
  const uint32_t width = decompressed_argb_.width();
  const uint32_t height = decompressed_argb_.height();
  const uint32_t tile_width =
      TileWidth(FinalTileShape(*config_), width, height);
  const uint32_t tile_height =
      TileHeight(FinalTileShape(*config_), width, height);
  WP2_CHECK_STATUS(GetTilesLayout(width, height, tile_width, tile_height,
                                  ProgressRange(), &decompressed_argb_,
                                  &decompressed_yuv_, &tiles_layout_));
  assert(tiles_layout_.tiles.size() == 1 &&
         enc_tiles_layout_.tiles.size() == 1);

  // Plug ANSEnc output to ANSDec input.
  Tile* const tile = &tiles_layout_.tiles.front();
  tile->chunk_size_is_known = true;
  tile->chunk_size = bits.size();
  tiles_layout_.gparams = &local_gparams_;
  tile->private_input = ExternalDataSource(bits.data(), bits.size());
  tile->input = &tile->private_input;

  // Decode to 'decompressed_argb_'.
  ANSDec dec(tile->input);
  WP2_CHECK_STATUS(
      LossyDecode(features_, dec_config_, &tiles_layout_, &dec, tile));

  // Compare the pixels of the non-padded area only.
  YUVPlane original_view, decompressed_view;
  WP2_CHECK_STATUS(original_view.SetView(*src_, {0, 0, width, height}));
  WP2_CHECK_STATUS(
      decompressed_view.SetView(decompressed_yuv_, {0, 0, width, height}));
  WP2_CHECK_STATUS(decompressed_view.GetDistortion(
      original_view, {kYuvMaxPrec + 1, /*is_signed=*/true}, PSNR, distortion_));

  // Compute a score based on distortion and the number of bits per pixel.
  const float ssim = distortion_[4];
  const float bpp =
      std::max(1u, enc.GetBitstreamSize()) * 8.f / (width * height);
  const float lambda = MapQuality(*config_, 7.00f, 4.15f);
  *score = ssim - lambda * bpp;
  return WP2_STATUS_OK;
}

WP2Status TileScoreFunc::Use(const Block& block) {
  WP2_CHECK_ALLOC_OK(blocks_.push_back(block));
  if (cached_best_score_ > best_score_) best_score_ = cached_best_score_;
  cached_best_score_ = 0.f;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
