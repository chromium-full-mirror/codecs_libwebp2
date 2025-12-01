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
// Tile encoding
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/enc/tile_enc.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/common/constants.h"
#include "src/common/global_params.h"
#include "src/common/header_enc_dec.h"
#include "src/common/lossy/block_size.h"
#include "src/common/progress_watcher.h"
#include "src/dec/tile_dec.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/analysis.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/ans_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status TileEncoder::AssignNextTile() {
  tile_ = nullptr;

  // Get first unassigned tile, and increment the counter if valid.
  WP2_CHECK_STATUS(tiles_layout_->assignment_lock.Acquire());
  const uint32_t tile_index = tiles_layout_->first_unassigned_tile_index;
  if (tile_index < tiles_layout_->tiles.size()) {
    tile_ = &tiles_layout_->tiles[tile_index];
    ++tiles_layout_->first_unassigned_tile_index;
  }
  tiles_layout_->assignment_lock.Release();
  return WP2_STATUS_OK;
}

WP2Status TileEncoder::Execute() {
  while (tile_ != nullptr) {
    if (tiles_layout_->gparams->type_ == GlobalParams::GP_AV1) {
      WP2_CHECK_STATUS(Av1Encode(&tile_->data));
    } else {
      ANSEnc& enc = tile_->enc;
      assert(enc.GetBitstreamSize() == 0);
      if (tiles_layout_->gparams->type_ == GlobalParams::GP_BOTH) {
        ANSDebugPrefix prefix(&enc, "GlobalHeader");
        enc.PutBool(use_lossless_, "use_lossless");
      }
      if (use_lossless_) {
        WP2_CHECK_STATUS(LosslessEncode(&enc));
      } else {
        const Rectangle padded_tile_rect = {
            tile_->rect.x, tile_->rect.y, Pad(tile_->rect.width, kPredWidth),
            Pad(tile_->rect.height, kPredWidth)};
        VectorNoCtor<Block> partition;
        WP2_CHECK_STATUS(
            AddForcedBlocks(*config_, padded_tile_rect, &partition));
        WP2_CHECK_STATUS(LossyEncode(partition, &enc));
      }

      WP2_CHECK_STATUS(enc.AssembleToBitstream(/*clear_tokens=*/true));
    }

    WP2_CHECK_STATUS(AssignNextTile());
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

namespace {

// Fills the 'tiles_layout'. Dimensions are in pixels, not padded.
WP2Status GetEncTilesLayout(uint32_t frame_width, uint32_t frame_height,
                            uint32_t tile_width, uint32_t tile_height,
                            const ProgressRange& progress,
                            const ArgbBuffer& frame_rgb,
                            const YUVPlane& frame_yuv,
                            const GlobalParams& gparams,
                            bool image_is_premultiplied,
                            EncTilesLayout* const tiles_layout) {
  tiles_layout->tile_width = tile_width;
  tiles_layout->tile_height = tile_height;
  const uint32_t num_tiles =
      GetNumTiles(frame_width, frame_height, tile_width, tile_height,
                  &tiles_layout->num_tiles_x, &tiles_layout->num_tiles_y);
  WP2_CHECK_ALLOC_OK(tiles_layout->tiles.resize(num_tiles));
  const uint32_t num_pixels = frame_width * frame_height;

  uint32_t tile_y = 0, tile_x = 0;
  for (EncTile& tile : tiles_layout->tiles) {
    tile.rect = {tile_x * tile_width, tile_y * tile_height,
                 std::min(frame_width - tile_x * tile_width, tile_width),
                 std::min(frame_height - tile_y * tile_height, tile_height)};
    tile.progress =
        ProgressRange(progress, 1. * tile.rect.GetArea() / num_pixels);

    // Set views on RGB and YUV input buffers, one or both may be used.
    assert(!frame_rgb.IsEmpty() || !frame_yuv.IsEmpty());
    if (!frame_rgb.IsEmpty()) {
      assert(frame_rgb.width() == frame_width &&
             frame_rgb.height() == frame_height);
      WP2_CHECK_STATUS(tile.rgb_input.SetFormat(frame_rgb.format()));
      WP2_CHECK_STATUS(tile.rgb_input.SetView(frame_rgb, tile.rect));
    }
    if (!frame_yuv.IsEmpty()) {
      assert(frame_yuv.GetWidth() == Pad(frame_width, kPredWidth) &&
             frame_yuv.GetHeight() == Pad(frame_height, kPredWidth));
      WP2_CHECK_STATUS(tile.yuv_input.SetView(
          frame_yuv,
          {tile.rect.x, tile.rect.y,
           std::min(frame_yuv.GetWidth() - tile.rect.x, tile_width),
           std::min(frame_yuv.GetHeight() - tile.rect.y, tile_height)}));
    }

    if (++tile_x == tiles_layout->num_tiles_x) {
      ++tile_y;
      tile_x = 0;
    }
  }
  tiles_layout->first_unassigned_tile_index = 0;
  tiles_layout->gparams = &gparams;
  tiles_layout->image_is_premultiplied = image_is_premultiplied;
  return WP2_STATUS_OK;
}

WP2Status SetupWorkers(uint32_t width, uint32_t height,
                       const ArgbBuffer& rgb_buffer, const YUVPlane& yuv_buffer,
                       const EncoderConfig& config, const GlobalParams& gparams,
                       EncTilesLayout* const tiles_layout,
                       Vector<TileEncoder>* const workers) {
  // The number of workers is limited by the number of threads and tiles.
  const uint32_t num_workers = std::min((uint32_t)config.thread_level + 1u,
                                        (uint32_t)tiles_layout->tiles.size());
  WP2_CHECK_ALLOC_OK(workers->resize(num_workers));

  for (TileEncoder& worker : *workers) {
    worker.status_ = WP2_STATUS_OK;
    worker.config_ = &config;
    worker.use_lossless_ =
        (!config.use_neural_compression && config.quality > kMaxLossyQuality);
    worker.tiles_layout_ = tiles_layout;
    WP2_CHECK_STATUS(worker.AssignNextTile());
  }
  return WP2_STATUS_OK;
}

WP2Status CodeTiles(EncTilesLayout* const tiles_layout,
                    Vector<TileEncoder>* const workers, Writer* const output) {
  const bool do_mt = (workers->size() > 1);
  WP2Status status = WP2_STATUS_OK;
  for (TileEncoder& tile : *workers) {
    status = tile.Start(do_mt);
    if (status != WP2_STATUS_OK) break;
  }
  for (TileEncoder& tile : *workers) {
    const WP2Status thread_status = tile.End();
    if (thread_status != WP2_STATUS_OK) {
      status = thread_status;
      continue;
    }
    if (status == WP2_STATUS_OK) status = tile.status_;
  }
  WP2_CHECK_STATUS(status);
  for (EncTile& tile : tiles_layout->tiles) {
    const uint32_t rgb_bit_depth = WP2Formatbpc(tile.rgb_input.format());
    const uint32_t max_num_bytes = std::min(
        GetTileMaxNumBytes(rgb_bit_depth, *tiles_layout->gparams, tile.rect),
        kMaxVarInt);
    assert((tile.enc.GetBitstreamSize() == 0) ^ (tile.data.size == 0));
    const uint32_t num_bytes =
        (tile.data.size > 0) ? tile.data.size : tile.enc.GetBitstreamSize();
    {
      uint8_t buffer[kMaxVarIntLength];
      const uint32_t size = WriteVarInt(std::min(num_bytes, max_num_bytes), 1,
                                        max_num_bytes, buffer);
      WP2_CHECK_ALLOC_OK(output->Append(buffer, size));
    }

    if (num_bytes >= max_num_bytes) {
      // Discard ANS encoding because it takes at least as many bytes as raw
      // pixels.
      WP2_CHECK_STATUS(BypassTileEnc(*tiles_layout->gparams,
                                     tiles_layout->image_is_premultiplied,
                                     &tile, output));
    } else {
      if (tile.data.size > 0) {
        WP2_CHECK_ALLOC_OK(output->Append(tile.data.bytes, tile.data.size));
      } else {
        // TODO(yguyon): Check if 'tile.enc.GetBitstreamSize()' can even be 0.
        WP2_CHECK_STATUS(tile.enc.WriteBitstreamTo(*output));
      }
    }
    tile.enc.Reset();
    tile.data.Clear();
  }
  return status;
}

}  // namespace

//------------------------------------------------------------------------------

WP2Status EncodeTiles(uint32_t width, uint32_t height,
                      const ArgbBuffer& rgb_buffer, const YUVPlane& yuv_buffer,
                      const CSPTransform& transf, const EncoderConfig& config,
                      bool image_has_alpha, bool image_is_premultiplied,
                      const ProgressRange& progress, Writer* const output) {
  WP2EncDspInit();
  GlobalParams gparams;
  FeatureMap features;
  gparams.features_ = &features;
  const ArgbBuffer* buffer = &rgb_buffer;

  ArgbBuffer pre_processed_buffer(buffer->format());

  // near-lossless pre-processing
  constexpr uint32_t kMinNearLosslessDimension = 16;  // minimum dimension
  if (yuv_buffer.IsEmpty() && buffer->width() >= kMinNearLosslessDimension &&
      buffer->height() >= kMinNearLosslessDimension &&
      config.quality > kMaxLossyQuality && config.quality < kMaxQuality) {
    WP2_CHECK_STATUS(PreprocessNearLossless(*buffer, config, /*is_alpha=*/false,
                                            &pre_processed_buffer));
    buffer = &pre_processed_buffer;
  }

  WP2_CHECK_STATUS(
      GlobalAnalysis(*buffer, yuv_buffer, transf, config, &gparams));

  // TODO(skal): analysis and num_mtx_ decision...
  WP2_CHECK_STATUS(gparams.InitRndMtxSet());

  WP2_CHECK_STATUS(EncodeGLBL(config, gparams, image_has_alpha, output));
  WP2_CHECK_STATUS(progress.AdvanceBy(kOnePercent));  // GlobalParams
  // TILE_SHAPE_ALL is only valid for lossless for now.
  WP2_CHECK_OK(config.tile_shape != TILE_SHAPE_FULL ||
                   gparams.type_ == GlobalParams::GP_LOSSLESS,
               WP2_STATUS_INVALID_PARAMETER);
  const uint32_t tile_width = TileWidth(FinalTileShape(config), width, height);
  const uint32_t tile_height =
      TileHeight(FinalTileShape(config), width, height);

  const ProgressRange tiles_progress(progress, kProgressTiles);
  EncTilesLayout tiles_layout;
  WP2_CHECK_STATUS(GetEncTilesLayout(
      width, height, tile_width, tile_height, tiles_progress, *buffer,
      yuv_buffer, gparams, image_is_premultiplied, &tiles_layout));

  Vector<TileEncoder> workers;
  WP2_CHECK_STATUS(SetupWorkers(width, height, *buffer, yuv_buffer, config,
                                gparams, &tiles_layout, &workers));
  WP2_CHECK_STATUS(CodeTiles(&tiles_layout, &workers, output));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
