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

#include <cassert>
#include <cstdint>
#include <cstring>

#include "src/common/global_params.h"
#include "src/common/header_enc_dec.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/dec/filters/block_map_filter.h"
#include "src/dec/tile_dec.h"
#include "src/dsp/math.h"
#include "src/utils/data_source.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

#if defined(WP2_BITTRACE)
#include "src/dec/wp2_dec_i.h"
#endif

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status TileDecoder::BypassTileDec() {
  const uint32_t num_read_bytes_before =
      tile_->input->GetNumDiscardedBytes() + tile_->input->GetNumReadBytes();
  if (gparams_->type_ == GlobalParams::GP_LOSSLESS ||
      gparams_->type_ == GlobalParams::GP_AV1) {
    tile_->output_is_yuv = false;
    assert(!tile_->rgb_output.IsEmpty());
    if (features_->rgb_bit_depth == 8) {
      WP2_CHECK_STATUS(BypassTileDecRgb8b());
    } else {
      assert(features_->rgb_bit_depth == 10);
      WP2_CHECK_STATUS(BypassTileDecRgb10b());
    }
  } else {
    tile_->output_is_yuv = true;
    assert(!tile_->yuv_output.IsEmpty());
    WP2_CHECK_STATUS(BypassTileDecYuv());
  }
  const uint32_t num_read_bytes = tile_->input->GetNumDiscardedBytes() +
                                  tile_->input->GetNumReadBytes() -
                                  num_read_bytes_before;
  (void)num_read_bytes;
#if !defined(WP2_ENC_DEC_MATCH)
  assert(num_read_bytes ==
         GetTileMaxNumBytes(features_->rgb_bit_depth, *gparams_, tile_->rect));
#endif

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

WP2Status TileDecoder::BypassTileDecRgb8b() {
  const uint32_t bytes_per_row =
      tile_->rect.width * (gparams_->has_alpha_ ? 4 : 3);
#if !defined(WP2_ENC_DEC_MATCH)
  assert(bytes_per_row * tile_->rect.height ==
         GetTileMaxNumBytes(features_->rgb_bit_depth, *gparams_, tile_->rect));
#endif

  for (tile_->num_decoded_rows = 0;
       tile_->num_decoded_rows < tile_->rect.height;
       ++tile_->num_decoded_rows) {
    const uint32_t y = tile_->num_decoded_rows;
    DataSource::DataHandle handle;
    WP2_CHECK_OK(tile_->input->TryReadNext(bytes_per_row, &handle),
                 WP2_STATUS_NOT_ENOUGH_DATA);
    if (tile_->rgb_output.format() == WP2_Argb_32 ||
        tile_->rgb_output.format() == WP2_ARGB_32) {
      uint8_t* const row = tile_->rgb_output.GetRow8(y);
      if (gparams_->has_alpha_) {
        std::memcpy(row, handle.GetBytes(), handle.GetSize());
      } else {
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x * 4 + 0] = kAlphaMax;
          std::memcpy(row + x * 4 + 1, handle.GetBytes() + x * 3, 3);
        }
      }
    } else {
      assert(tile_->rgb_output.format() == WP2_Argb_38);
      uint16_t* const row = tile_->rgb_output.GetRow16(y);
      if (gparams_->has_alpha_) {
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x * 4 + 0] = handle.GetBytes()[x * 4];
          const uint8_t* const rgb = handle.GetBytes() + x * 4 + 1;
          for (uint32_t i : {0, 1, 2}) {
            row[x * 4 + 1 + i] =
                ChangePrecision<uint16_t>(rgb[i], /*from=*/8, /*to=*/10);
          }
        }
      } else {
        for (uint32_t x = 0; x < tile_->rect.width; ++x) {
          row[x * 4 + 0] = kAlphaMax;
          const uint8_t* const rgb = handle.GetBytes() + x * 3;
          for (uint32_t i : {0, 1, 2}) {
            row[x * 4 + 1 + i] =
                ChangePrecision<uint16_t>(rgb[i], /*from=*/8, /*to=*/10);
          }
        }
      }
    }
    WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(1.));
  }
  return WP2_STATUS_OK;
}

WP2Status TileDecoder::BypassTileDecRgb10b() {
  BitUnpacker bit_unpacker(tile_->input, "raw_pixels");
  const uint32_t num_bits_per_row =
      ((gparams_->has_alpha_ ? kAlphaBits : 0) + 3 * 10) * tile_->rect.width;

  for (tile_->num_decoded_rows = 0;
       tile_->num_decoded_rows < tile_->rect.height;
       ++tile_->num_decoded_rows) {
    // Make sure enough data is fetched to read an entire row.
    // Otherwise the decoding might be suspended in the middle of a row and the
    // output buffer invalidated just before resuming in the middle of a pixel.
    WP2_CHECK_OK(bit_unpacker.Prefetch(num_bits_per_row),
                 WP2_STATUS_NOT_ENOUGH_DATA);

    const uint32_t y = tile_->num_decoded_rows;
    if (tile_->rgb_output.format() == WP2_Argb_32) {
      uint8_t* const row = tile_->rgb_output.GetRow8(y);
      for (uint32_t x = 0; x < tile_->rect.width; ++x) {
        row[x * 4 + 0] = gparams_->has_alpha_
                             ? bit_unpacker.ReadBits(kAlphaBits, "alpha")
                             : kAlphaMax;
        row[x * 4 + 1] = ChangePrecision(bit_unpacker.ReadBits(10, "red"),
                                         /*from=*/10, /*to=*/8);
        row[x * 4 + 2] = ChangePrecision(bit_unpacker.ReadBits(10, "green"),
                                         /*from=*/10, /*to=*/8);
        row[x * 4 + 3] = ChangePrecision(bit_unpacker.ReadBits(10, "blue"),
                                         /*from=*/10, /*to=*/8);
      }
    } else {
      assert(tile_->rgb_output.format() == WP2_Argb_38);
      uint16_t* const row = tile_->rgb_output.GetRow16(y);
      for (uint32_t x = 0; x < tile_->rect.width; ++x) {
        row[x * 4 + 0] = gparams_->has_alpha_
                             ? bit_unpacker.ReadBits(kAlphaBits, "alpha")
                             : kAlphaMax;
        row[x * 4 + 1] = bit_unpacker.ReadBits(10, "red");
        row[x * 4 + 2] = bit_unpacker.ReadBits(10, "green");
        row[x * 4 + 3] = bit_unpacker.ReadBits(10, "blue");
      }
    }
    WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(1.));
  }

  WP2_CHECK_OK(bit_unpacker.Pad(), WP2_STATUS_BITSTREAM_ERROR);
  WP2_CHECK_STATUS(bit_unpacker.GetStatus());
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status TileDecoder::BypassTileDecYuv() {
  tile_->block_map.Init(tile_->rect, gparams_->transf_.GetYuvDepth(),
                        gparams_->transf_.GetYUVMin(),
                        gparams_->transf_.GetYUVMax(), &tile_->yuv_output);
  if (IsFilterBlockMapNeeded(*config_, *gparams_, *tiles_layout_)) {
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
    constexpr uint8_t kMinBpp = 255;    // Max accepted ~2bpp.
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

  const uint32_t num_bits = gparams_->transf_.GetYuvDepth().num_bits;
  const uint32_t num_bits_per_row =
      (gparams_->has_alpha_ ? kAlphaBits : 0) + 3 * num_bits;
  BitUnpacker bit_unpacker(tile_->input, "raw_pixels");
  if (!gparams_->has_alpha_ && !tile_->yuv_output.A.IsEmpty()) {
    tile_->yuv_output.A.Fill(kAlphaMax);
  }

  for (tile_->num_decoded_rows = 0;
       tile_->num_decoded_rows < tile_->rect.height;
       ++tile_->num_decoded_rows) {
    // Make sure enough data is fetched to read an entire row.
    // Otherwise the decoding might be suspended in the middle of a row and the
    // output buffer invalidated just before resuming in the middle of a pixel.
    WP2_CHECK_OK(bit_unpacker.Prefetch(num_bits_per_row),
                 WP2_STATUS_NOT_ENOUGH_DATA);

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
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
