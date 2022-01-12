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
// Functions related to animation decoding.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <cassert>

#include "src/common/color_precision.h"
#include "src/common/header_enc_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/ans_utils.h"
#include "src/utils/data_source.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status DecodeANMF(const DecoderConfig& config, DataSource* const data_source,
                     const BitstreamFeatures& features, uint32_t frame_index,
                     AnimationFrame* const frame) {
  if (features.is_animation) {
    const size_t num_read_bytes_before = data_source->GetNumReadBytes();
    BitUnpacker dec(data_source, "anmf_chunk");
    const uint32_t tag = dec.ReadBits(kANMFTagNumBits, "anmf_tag");
    frame->dispose = (tag == kANMFTagDispose);
    frame->blend = (dec.ReadBits(1, "blend") == 1);
    frame->is_last = (dec.ReadBits(1, "is_last") == 1);
    // TODO(yguyon): Force read 'duration_ms>0' if kMaxNumPreframes || is_last
    frame->duration_ms = dec.ReadVarUInt(0, kMaxFrameDurationMs, "duration");
    frame->window =
        dec.ReadRect(features.raw_width, features.raw_height, "window");
    const WP2Status status =
        dec.Pad() ? dec.GetStatus() : WP2_STATUS_BITSTREAM_ERROR;
    if (status != WP2_STATUS_OK) {
      // The data consumption must be atomic for incr dec glimpses.
      data_source->UnmarkAllReadBytes();
      data_source->MarkNumBytesAsRead(num_read_bytes_before);
    }
    WP2_CHECK_STATUS(status);

    WP2_CHECK_OK((tag == kANMFTagDispose) || (tag == kANMFTagFrame),
                 WP2_STATUS_BITSTREAM_ERROR);
    assert(frame->window.x + frame->window.width <= features.raw_width);
    assert(frame->window.y + frame->window.height <= features.raw_height);
    if (frame_index == 0) {  // The first frame MUST dispose.
      WP2_CHECK_OK(frame->dispose, WP2_STATUS_BITSTREAM_ERROR);
    } else {
      WP2_CHECK_OK(frame_index < kMaxNumFrames, WP2_STATUS_BITSTREAM_ERROR);
      if (frame->duration_ms == 0) {
        WP2_CHECK_OK(!frame->is_last, WP2_STATUS_BITSTREAM_ERROR);
      }
    }
    WP2_CHECK_REDUCED_STATUS(
      RegisterBitTrace(config,
          data_source->GetNumReadBytes() - num_read_bytes_before, "ANMF"));
  } else {
    frame->dispose = true;
    frame->blend = false;
    frame->duration_ms = kMaxFrameDurationMs;
    frame->window = {0, 0, features.raw_width, features.raw_height};
    frame->is_last = true;
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Fills the area outside of 'frame.window' with 'features.background_color' if
// 'frame.dispose' is true. The destination is the non-null buffer among
// 'rgb_output' and 'yuv_output'.
static WP2Status FillBorders(const BitstreamFeatures& features,
                             const AnimationFrame& frame,
                             ArgbBuffer* const rgb_output,
                             const CSPTransform* const csp_transform,
                             YUVPlane* const yuv_output) {
  if (!frame.dispose) return WP2_STATUS_OK;

  WP2_CHECK_OK((rgb_output != nullptr) != (yuv_output != nullptr),
               WP2_STATUS_INVALID_PARAMETER);
  uint32_t width, height;
  if (rgb_output != nullptr) {
    width = rgb_output->width();
    height = rgb_output->height();
  } else {
    width = yuv_output->GetWidth();
    height = yuv_output->GetHeight();
  }
  const Rectangle& window = frame.window;
  assert(window.x + window.width <= width);
  assert(window.y + window.height <= height);
  Rectangle top, bot, lft, rgt;
  Rectangle(0, 0, width, height).Exclude(window, &top, &bot, &lft, &rgt);

  if (rgb_output != nullptr) {
    if (WP2Formatbpc(rgb_output->format()) <= 8) {
      const Argb32b color = ToArgb32b(features.background_color);

      rgb_output->Fill(top, color);
      rgb_output->Fill(bot, color);
      rgb_output->Fill(lft, color);
      rgb_output->Fill(rgt, color);
    } else {
      rgb_output->Fill(top, features.background_color);
      rgb_output->Fill(bot, features.background_color);
      rgb_output->Fill(lft, features.background_color);
      rgb_output->Fill(rgt, features.background_color);
    }
  } else {
    assert(csp_transform != nullptr);
    const Ayuv38b color =
        csp_transform->ToYuv(ToArgb32b(features.background_color));

    yuv_output->Fill(top, color);
    yuv_output->Fill(bot, color);
    yuv_output->Fill(lft, color);
    yuv_output->Fill(rgt, color);
  }
  return WP2_STATUS_OK;
}

WP2Status FillBorders(const BitstreamFeatures& features,
                      const AnimationFrame& frame, ArgbBuffer* const output) {
  return FillBorders(features, frame, output,
                     /*csp_transform=*/nullptr, /*yuv_output=*/nullptr);
}

WP2Status FillBorders(const BitstreamFeatures& features,
                      const AnimationFrame& frame,
                      const CSPTransform& csp_transform,
                      YUVPlane* const output) {
  return FillBorders(features, frame, /*rgb_output=*/nullptr,
                     /*csp_transform=*/&csp_transform, /*yuv_output=*/output);
}

//------------------------------------------------------------------------------

}  // namespace WP2
