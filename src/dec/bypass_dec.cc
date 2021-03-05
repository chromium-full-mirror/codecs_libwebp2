// Copyright 2020 Google LLC
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
//  Bypass with raw pixel coding
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/common/global_params.h"
#include "src/common/header_enc_dec.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/math.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status TileDecoder::BypassTileDec() {
  const uint32_t num_read_bytes_before =
      tile_->input->GetNumDiscardedBytes() + tile_->input->GetNumReadBytes();
  if (gparams_->type_ == GlobalParams::GP_LOSSLESS) {
    tile_->output_is_yuv = false;
    assert(!tile_->rgb_output.IsEmpty());

    if (features_->rgb_bit_depth == 8) {
      const uint32_t bytes_per_row =
          tile_->rect.width * (gparams_->has_alpha_ ? 4 : 3);
      assert(
          bytes_per_row * tile_->rect.height ==
          GetTileMaxNumBytes(features_->rgb_bit_depth, *gparams_, tile_->rect));

      for (tile_->num_decoded_rows = 0;
           tile_->num_decoded_rows < tile_->rect.height;
           ++tile_->num_decoded_rows) {
        DataSource::DataHandle handle;
        WP2_CHECK_OK(tile_->input->TryReadNext(bytes_per_row, &handle),
                     WP2_STATUS_NOT_ENOUGH_DATA);
        // TODO(yguyon): Also accept WP2_Argb_38 as output format
        uint8_t* const row = tile_->rgb_output.GetRow8(tile_->num_decoded_rows);
        if (gparams_->has_alpha_) {
          std::memcpy(row, handle.GetBytes(), handle.GetSize());
        } else {
          for (uint32_t x = 0; x < tile_->rect.width; ++x) {
            row[x * 4 + 0] = kAlphaMax;
            std::memcpy(row + x * 4 + 1, handle.GetBytes() + x * 3, 3);
          }
        }
        WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(1.));
      }
    } else {
      assert(features_->rgb_bit_depth == 10);
      BitUnpacker bit_unpacker(tile_->input, "raw_pixels");
      for (tile_->num_decoded_rows = 0;
           tile_->num_decoded_rows < tile_->rect.height;
           ++tile_->num_decoded_rows) {
        // TODO(yguyon): Also accept WP2_Argb_32 as output format
        uint16_t* const row =
            tile_->rgb_output.GetRow16(tile_->num_decoded_rows);
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x * 4 + 0] = gparams_->has_alpha_
                               ? bit_unpacker.ReadBits(kAlphaBits, "alpha")
                               : kAlphaMax;
          row[x * 4 + 1] = bit_unpacker.ReadBits(10, "red");
          row[x * 4 + 2] = bit_unpacker.ReadBits(10, "green");
          row[x * 4 + 3] = bit_unpacker.ReadBits(10, "blue");
        }
        WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(1.));
      }

      WP2_CHECK_STATUS(bit_unpacker.GetStatus());
    }
  } else {  // !GP_LOSSLESS
    tile_->output_is_yuv = true;
    assert(!tile_->yuv_output.IsEmpty());
    const uint32_t num_bits = gparams_->transf_.GetYUVPrecisionBits() + 1u;

    tile_->block_map.Init(tile_->rect, num_bits, gparams_->transf_.GetYUVMin(),
                          gparams_->transf_.GetYUVMax(), &tile_->yuv_output);
    if (IsFilterBlockMapNeeded(*config_, *features_, *gparams_,
                               *tiles_layout_)) {
      WP2_CHECK_STATUS(tile_->block_map.Allocate());
      // Simulate a grid of 4x4 blocks for the IntertileFilter settings.
      // Even though this is lossless it is likely to border a lossy tile which
      // might benefit from slight deblocking.
      // This must be done before 'tile_->num_decoded_rows' is increased because
      // the 'tile_->block_map' data may be accessed for each available line.
      constexpr BlockSize kSize = BLK_4x4;
      const uint32_t block_height = BlockHeightPix(kSize);
      const uint32_t block_width = BlockWidthPix(kSize);
      constexpr uint32_t kSegmentId = 0;  // Least quantized one.
      constexpr uint8_t kMinBpp = 255;  // Max accepted ~2bpp.
      constexpr uint32_t kResDen[4] = {255, 255, 255, 255};  // Full residuals.
      for (uint32_t y = 0; y < tile_->rect.height; y += block_height) {
        for (uint32_t x = 0; x < tile_->rect.width; x += block_width) {
          // Set segment id as the least quantized one.
          tile_->block_map.RegisterBlock(
              x / kMinBlockSizePix, y / kMinBlockSizePix, kSize, kSegmentId,
              kMinBpp, /*has_lossy_alpha=*/false, kResDen);
        }
      }
    }

    BitUnpacker bit_unpacker(tile_->input, "raw_pixels");
    if (!gparams_->has_alpha_ && !tile_->yuv_output.A.IsEmpty()) {
      tile_->yuv_output.A.Fill(kAlphaMax);
    }

    for (tile_->num_decoded_rows = 0;
         tile_->num_decoded_rows < tile_->rect.height;
         ++tile_->num_decoded_rows) {
      const uint32_t y = tile_->num_decoded_rows;
      // Read interleaved channels to output samples line by line.
      for (Channel channel : {kYChannel, kUChannel, kVChannel}) {
        int16_t* const row = tile_->yuv_output.GetChannel(channel).Row(y);
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x] = (int16_t)bit_unpacker.ReadSBits(num_bits, "yuv");
        }
      }
      if (gparams_->has_alpha_) {
        int16_t* const row = tile_->yuv_output.A.Row(y);
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x] = (int16_t)bit_unpacker.ReadBits(kAlphaBits, "alpha");
        }
      }
      WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(1.));
    }

    WP2_CHECK_STATUS(bit_unpacker.GetStatus());
  }
  const uint32_t num_read_bytes = tile_->input->GetNumDiscardedBytes() +
                                  tile_->input->GetNumReadBytes() -
                                  num_read_bytes_before;
  if (num_read_bytes !=
      GetTileMaxNumBytes(features_->rgb_bit_depth, *gparams_, tile_->rect)) {
    assert(false);
  }

#if defined(WP2_BITTRACE)
  if (config_->info != nullptr) {
    WP2_CHECK_STATUS(tiles_layout_->assignment_lock.Acquire());
    WP2_CHECK_STATUS(
        RegisterBitTrace(*config_, num_read_bytes, "BypassPixels"));
    tiles_layout_->assignment_lock.Release();
  }
#endif
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}   // namespace WP2
