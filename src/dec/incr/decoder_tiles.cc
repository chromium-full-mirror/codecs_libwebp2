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
//  incremental decoding
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/common/global_params.h"
#include "src/dec/incr/decoder_info.h"
#include "src/dec/incr/decoder_state.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/math.h"
#include "src/utils/data_source.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"

namespace WP2 {

//------------------------------------------------------------------------------

namespace {

WP2Status TryReadTileChunkSize(uint32_t rgb_bit_depth,
                               const GlobalParams& gparams,
                               DataSource* const data_source,
                               const DecoderConfig& config, Tile* const tile) {
  assert(tile->rect.GetArea() > 0);
  if (!tile->chunk_size_is_known) {  // If not previously decoded.
    const size_t size_before = data_source->GetNumReadBytes();
    (void)size_before;
    WP2_CHECK_STATUS(DecodeTileChunkSize(rgb_bit_depth, gparams, tile->rect,
                                         data_source, &tile->chunk_size));
    // If not STATUS_OK, tile chunk size is unknown, do not decode this tile for
    // now. No byte is marked as read by DecodeTileChunkSize() if not STATUS_OK,
    // so UnmarkAllReadBytes() is not needed.
    tile->chunk_size_is_known = true;
    WP2_CHECK_REDUCED_STATUS(RegisterBitTrace(
        config, data_source->GetNumReadBytes() - size_before, "chunk_size"));
  }
  return WP2_STATUS_OK;
}

// Returns true if the next tile has all its chunk data available.
// 'bitstream_pos_after_tile' is set to 0 if it is too early to know it.
WP2Status TryReadTile(DataSource* const data_source,
                      const BitstreamFeatures& features,
                      const DecoderConfig& config,
                      TilesLayout* const tiles_layout, Tile* const tile,
                      PartialTileDecoder* const partial_tile_decoder,
                      size_t* const bitstream_position_at_end_of_tile) {
  *bitstream_position_at_end_of_tile = Decoder::State::kUnknown;
  WP2_CHECK_STATUS(TryReadTileChunkSize(features.rgb_bit_depth,
                                        *tiles_layout->gparams, data_source,
                                        config, tile));
  *bitstream_position_at_end_of_tile = data_source->GetNumDiscardedBytes() +
                                       data_source->GetNumReadBytes() +
                                       tile->chunk_size;

  if (data_source->TryReadNext(tile->chunk_size, &tile->data_handle)) {
    // All of the tile chunk data is available, decode it in one shot.
    ++tiles_layout->num_assignable_tiles;
    return WP2_STATUS_OK;
  } else {
    if (partial_tile_decoder != nullptr) {
      const size_t num_next_bytes = data_source->GetNumNextBytes();
      if (num_next_bytes > 0) {
        WP2_CHECK_OK(
            data_source->TryGetNext(num_next_bytes, &tile->data_handle),
            WP2_STATUS_BAD_READ);
        WP2_CHECK_STATUS(
            partial_tile_decoder->Init(features, config, tiles_layout, tile));
      }
    }
    return WP2_STATUS_NOT_ENOUGH_DATA;
  }
}

WP2Status TrySkipTile(DataSource* const data_source,
                      const BitstreamFeatures& features,
                      const DecoderConfig& config,
                      TilesLayout* const tiles_layout, Tile* const tile) {
  WP2_CHECK_STATUS(TryReadTileChunkSize(features.rgb_bit_depth,
                                        *tiles_layout->gparams, data_source,
                                        config, tile));
  data_source->MarkNumBytesAsRead(tile->chunk_size);
  ++tiles_layout->num_assignable_tiles;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Deblocks then converts available rows in each tile from YUV to RGB if needed.
// Updates 'state->frame_num_converted_rows' and 'frame_num_final_rows'.
WP2Status ConvertRows(Decoder::State* const state) {
  state->intertile_filter.Deblock(state->GetFrameNumDecodedRows());

  const uint32_t num_converted_rows = state->frame_num_converted_rows;
  const uint32_t num_convertible_rows =
      state->intertile_filter.GetNumFilteredRows();

  if (num_converted_rows < num_convertible_rows) {
    // Per-tile YUV to RGB conversion, if any.
    for (Tile& tile : state->tiles_layout.tiles) {
      // This Decoder does not support YUV output, final buffer is always RGB.
      assert(!tile.rgb_output.IsEmpty());
      if (!tile.output_is_yuv) continue;

      if (tile.rect.y + tile.rect.height <= num_converted_rows) continue;
      if (tile.rect.y >= num_convertible_rows) break;
      const uint32_t num_converted_rows_in_tile =
          SafeSub(num_converted_rows, tile.rect.y);
      const uint32_t num_convertible_rows_in_tile =
          std::min(tile.rect.height, num_convertible_rows - tile.rect.y);
      assert(num_converted_rows_in_tile < num_convertible_rows_in_tile);
      assert(num_convertible_rows_in_tile <= tile.num_decoded_rows);

      // Lines to convert within the current 'tile'.
      const Rectangle new_rows_rect = {
          0, num_converted_rows_in_tile, tile.rect.width,
          num_convertible_rows_in_tile - num_converted_rows_in_tile};

      ArgbBuffer rgb_output_rows(tile.rgb_output.format());
      WP2_CHECK_STATUS(rgb_output_rows.SetView(tile.rgb_output, new_rows_rect));
      YUVPlane yuv_output_rows;
      WP2_CHECK_STATUS(yuv_output_rows.SetView(tile.yuv_output, new_rows_rect));
      WP2_CHECK_STATUS(
          yuv_output_rows.Export(state->tiles_layout.gparams->transf_,
                                 /*resize_if_needed=*/false, &rgb_output_rows));
    }
  }

  state->frame_num_converted_rows = num_convertible_rows;
  state->frame_num_final_rows = state->frame_num_converted_rows;
  return WP2_STATUS_OK;
}

}  // namespace

//------------------------------------------------------------------------------

WP2Status Decoder::DecodeAvailableTiles() {
  TilesLayout* const tiles_layout = &state_->tiles_layout;
  PartialTileDecoder* const partial_tile_decoder =
      state_->partial_tile_decoder.get();
  if ((partial_tile_decoder != nullptr) &&
      (partial_tile_decoder->GetPartialTile() != nullptr)) {
    Tile* const partial_tile = partial_tile_decoder->GetPartialTile();
    assert(partial_tile->chunk_size_is_known);
    // DataSource::GetNumNextBytes() gets updated on TryGetNext() so try to get
    // the whole tile chunk first.
    if (!state_->data_source->TryGetNext(partial_tile->chunk_size,
                                         &partial_tile->data_handle)) {
      WP2_CHECK_OK(state_->data_source->TryGetNext(
                       state_->data_source->GetNumNextBytes(),
                       &partial_tile->data_handle),
                   WP2_STATUS_BITSTREAM_OUT_OF_MEMORY);
    }

    WP2_CHECK_STATUS(partial_tile_decoder->Continue());
    assert(partial_tile->input->GetNumDiscardedBytes() == 0);

    if (partial_tile_decoder->IsComplete()) {
      state_->data_source->MarkNumBytesAsRead(partial_tile->chunk_size);
      state_->data_source->Discard(state_->data_source->GetNumReadBytes());
      partial_tile_decoder->Clear();
      ++tiles_layout->num_decoded_tiles;
    } else if ((partial_tile->input->GetNumReadBytes() +
                partial_tile->input->GetNumNextBytes()) >=
               partial_tile->chunk_size) {
      // The whole tile chunk is available: missing data is an error.
      return WP2_STATUS_BITSTREAM_ERROR;
    } else {
      // If the partial tile is not fully decoded, do not look further.
      WP2_CHECK_STATUS(ConvertRows(state_));
      return WP2_STATUS_NOT_ENOUGH_DATA;
    }
  }

  while (tiles_layout->num_assignable_tiles < tiles_layout->tiles.size()) {
    const uint32_t tile_index = tiles_layout->num_assignable_tiles;
    Tile& tile = state_->tiles_layout.tiles[tile_index];
    size_t bitstream_position_at_end_of_tile;
    const WP2Status tile_status = TryReadTile(
        state_->data_source, features_, config_, &state_->tiles_layout, &tile,
        state_->partial_tile_decoder.get(), &bitstream_position_at_end_of_tile);
    if (tile_status != WP2_STATUS_NOT_ENOUGH_DATA) {  // Stop on true errors.
      WP2_CHECK_STATUS(tile_status);
    }

    // Save the frame byte size as soon as it is known: during last tile.
    if (tile_index + 1 == tiles_layout->tiles.size() &&
        bitstream_position_at_end_of_tile != State::kUnknown) {
      WP2_CHECK_STATUS(
          SaveFrameNumBytes(bitstream_position_at_end_of_tile,
                            &state_->frames[state_->current_frame_index]));
    }
    if (tile_status != WP2_STATUS_OK) {
      break;  // Missing bytes in this tile, no need to look at the next one.
    }
  }

  // Now that the primary 'state_->data_source' was read as much as possible and
  // won't be used till next DecodeAvailableTiles(), use ExternalDataSources as
  // input for each tile, to prevent slowness and concurrency issues.
  for (uint32_t tile_index = tiles_layout->num_decoded_tiles;
       tile_index < tiles_layout->num_assignable_tiles; ++tile_index) {
    Tile* const tile = &tiles_layout->tiles[tile_index];
    const uint8_t* const tile_data = tile->data_handle.GetBytes();
    // Maybe a DataSource::TryReadNext() invalidated previously read data.
    WP2_CHECK_OK(tile_data != nullptr, WP2_STATUS_BITSTREAM_OUT_OF_MEMORY);
    tile->private_input = ExternalDataSource(tile_data, tile->chunk_size);
    tile->input = &tile->private_input;
  }

  // Assign first tiles to available workers.
  const uint32_t num_full_tiles =
      tiles_layout->num_assignable_tiles - tiles_layout->num_decoded_tiles;
  const uint32_t num_workers =
      std::min((uint32_t)state_->workers.size(), num_full_tiles);
  for (uint32_t i = 0; i < num_workers; ++i) {
    TileDecoder* const worker = &state_->workers[i];
    WP2_CHECK_STATUS(worker->TryAssignNextTile());
    assert(worker->tile_ != nullptr);  // Has a job.
  }

  // Decode full tiles in parallel or sequentially.
  WP2Status status = WP2_STATUS_OK;
  for (uint32_t i = 0; i < num_workers; ++i) {
    TileDecoder* const worker = &state_->workers[i];

    // The last worker runs in the main thread, after starting others.
    const bool threaded = (i + 1 != num_workers);
    status = worker->Start(threaded);
    // Only the 'partial_tile_decoder' is allowed to return missing data.
    assert(status != WP2_STATUS_NOT_ENOUGH_DATA);
    if (status != WP2_STATUS_OK) break;
  }

  if ((status == WP2_STATUS_OK) && (partial_tile_decoder != nullptr) &&
      (partial_tile_decoder->GetPartialTile() != nullptr)) {
    // Start partial tile decoding, worker will suspend itself.
    status = partial_tile_decoder->Start();
  }

  // Loop over all threads and close them properly. No early return.
  for (uint32_t i = 0; i < num_workers; ++i) {
    TileDecoder* const worker = &state_->workers[i];
    WP2Status worker_status = worker->End();
    worker->tile_ = nullptr;
    // The whole tile chunk is received. Missing data implies a bitstream error.
    if (worker_status == WP2_STATUS_NOT_ENOUGH_DATA) {
      worker_status = WP2_STATUS_BITSTREAM_ERROR;
    }

    if ((worker_status != WP2_STATUS_OK) && (status == WP2_STATUS_OK)) {
      // The first encountered error is returned (by tile order, not time).
      status = worker_status;
    }
  }

  if ((partial_tile_decoder != nullptr) &&
      (partial_tile_decoder->GetPartialTile() != nullptr)) {
    if (status == WP2_STATUS_OK) {
      // Don't increase these earlier or regular workers might try to decode the
      // partial tile.
      ++tiles_layout->num_assignable_tiles;
      ++tiles_layout->first_unassigned_tile_index;
    } else {
      partial_tile_decoder->Clear();
    }
  }

  WP2_CHECK_STATUS(status);

  state_->data_source->Discard(state_->data_source->GetNumReadBytes());
  // Increase num_decoded_tiles only if there is no error. We don't know here
  // which tile failed.
  tiles_layout->num_decoded_tiles += num_full_tiles;

  WP2_CHECK_STATUS(ConvertRows(state_));

  if (tiles_layout->num_decoded_tiles < tiles_layout->tiles.size()) {
    return WP2_STATUS_NOT_ENOUGH_DATA;
  }
  return WP2_STATUS_OK;
}

WP2Status Decoder::DiscardAvailableTiles() {
  TilesLayout& tiles_layout = state_->tiles_layout;
  PartialTileDecoder* const partial_tile_decoder =
      state_->partial_tile_decoder.get();
  if (partial_tile_decoder != nullptr &&
      partial_tile_decoder->GetPartialTile() != nullptr) {
    const Tile& partial_tile = *partial_tile_decoder->GetPartialTile();
    state_->data_source->MarkNumBytesAsRead(partial_tile.chunk_size);
    state_->data_source->Discard(state_->data_source->GetNumReadBytes());
    partial_tile_decoder->Clear();
    WP2_CHECK_STATUS(partial_tile.row_progress.AdvanceBy(
        partial_tile.rect.height - partial_tile.num_decoded_rows));
  }

  while (tiles_layout.num_assignable_tiles < tiles_layout.tiles.size()) {
    Tile& tile = tiles_layout.tiles[tiles_layout.num_assignable_tiles];
    const WP2Status tile_status = TrySkipTile(
        state_->data_source, features_, config_, &state_->tiles_layout, &tile);
    if (tile_status != WP2_STATUS_NOT_ENOUGH_DATA) {
      WP2_CHECK_STATUS(tile_status);
    }
    state_->data_source->Discard(state_->data_source->GetNumReadBytes());
    if (tile_status != WP2_STATUS_OK) break;
    WP2_CHECK_STATUS(tile.row_progress.AdvanceBy(tile.rect.height));
  }

  WP2_CHECK_OK(tiles_layout.num_assignable_tiles >= tiles_layout.tiles.size(),
               WP2_STATUS_NOT_ENOUGH_DATA);

  // Save the frame byte size as soon as it is known.
  WP2_CHECK_STATUS(
      SaveFrameNumBytes(state_->data_source->GetNumDiscardedBytes(),
                        &state_->frames[state_->current_frame_index]));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
