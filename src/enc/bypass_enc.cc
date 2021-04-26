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

#include "src/common/header_enc_dec.h"
#include "src/dec/tile_dec.h"
#include "src/dsp/math.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status BypassTileEnc(const GlobalParams& gparams, EncTile* const tile,
                        Writer* const output) {
  const uint32_t rgb_bit_depth =
      (tile->rgb_input.format() == WP2_Argb_38) ? 10 : 8;
  const uint32_t max_num_bytes =
      GetTileMaxNumBytes(rgb_bit_depth, gparams, tile->rect);

  if (gparams.type_ == GlobalParams::GP_LOSSLESS ||
      gparams.type_ == GlobalParams::GP_AV1) {
    // RGB input is expected.
    assert(!tile->rgb_input.IsEmpty());
    if (WP2Formatbpc(tile->rgb_input.format()) <= 8) {
      for (uint32_t y = 0; y < tile->rect.height; ++y) {
        const uint8_t* const row = tile->rgb_input.GetRow8(y);
        if (gparams.has_alpha_) {
          // Write rows of Argb samples.
          WP2_CHECK_ALLOC_OK(output->Append((void*)row, tile->rect.width * 4));
        } else {
          // Write only rgb.
          for (uint32_t x = 0; x < tile->rect.width; ++x) {
            WP2_CHECK_ALLOC_OK(output->Append((void*)&row[x * 4 + 1], 3));
          }
        }
      }
      // TODO(yguyon): Simulate ANSDebugPrefix to count these raw bytes?
    } else {
      assert(tile->rgb_input.format() == WP2_Argb_38);
      // Allocate, fill and write the 'data' to 'output'.
      // TODO(yguyon): Use less memory by outputting 8 lines by 8 lines?
      Data data;
      WP2_CHECK_STATUS(data.Resize(max_num_bytes, /*keep_bytes=*/false));
      // Use a BitPacker instance to easily pack sample bits (e.g. 10 by 10).
      BitPacker bit_packer(data.bytes, data.size, "raw_pixels");

      for (uint32_t y = 0; y < tile->rect.height; ++y) {
        const uint16_t* const row = tile->rgb_input.GetRow16(y);
        for (uint32_t x = 0; x < tile->rect.width; ++x) {
          if (gparams.has_alpha_) {
            bit_packer.PutBits(row[x * 4 + 0], kAlphaBits, "alpha");
          }
          bit_packer.PutBits(row[x * 4 + 1], 10, "red");
          bit_packer.PutBits(row[x * 4 + 2], 10, "green");
          bit_packer.PutBits(row[x * 4 + 3], 10, "blue");
        }
      }

      WP2_CHECK_OK(bit_packer.Ok(), WP2_STATUS_BAD_WRITE);
#if !defined(WP2_ENC_DEC_MATCH)
      assert(bit_packer.Used() == max_num_bytes);
#endif
      WP2_CHECK_ALLOC_OK(output->Append(data.bytes, data.size));
    }
  } else {
    // YUV input is expected even though RGB may also be available.
    assert(!tile->yuv_input.IsEmpty());
    const uint32_t channel_bits = gparams.transf_.GetYuvDepth().num_bits;

    // Allocate, fill and write the 'data' to 'output'.
    // TODO(yguyon): Use less memory by outputting 8 lines by 8 lines?
    Data data;
    WP2_CHECK_STATUS(data.Resize(max_num_bytes, /*keep_bytes=*/false));
    // Use a BitPacker instance to easily pack sample bits (e.g. 10 by 10).
    BitPacker bit_packer(data.bytes, data.size, "raw_pixels");

    for (uint32_t y = 0; y < tile->rect.height; ++y) {
      // Interleave channels to output samples line by line.
      for (Channel channel : {kYChannel, kUChannel, kVChannel}) {
        const int16_t* const row = tile->yuv_input.GetChannel(channel).Row(y);
        for (uint32_t x = 0; x < tile->rect.width; ++x) {
          bit_packer.PutSBits((int32_t)row[x], channel_bits, "yuv");
        }
      }
      if (gparams.has_alpha_) {
        const int16_t* const row = tile->yuv_input.A.Row(y);
        for (uint32_t x = 0; x < tile->rect.width; ++x) {
          assert(row[x] >= 0);
          bit_packer.PutBits((uint32_t)row[x], kAlphaBits, "alpha");
        }
      }
    }

    WP2_CHECK_OK(bit_packer.Ok(), WP2_STATUS_BAD_WRITE);
#if !defined(WP2_ENC_DEC_MATCH)
    assert(bit_packer.Used() == max_num_bytes);
#endif
    WP2_CHECK_ALLOC_OK(output->Append(data.bytes, data.size));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
