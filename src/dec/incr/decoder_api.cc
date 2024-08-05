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

#include "src/dec/incr/decoder_context.h"
#include "src/dec/incr/decoder_state.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/context_switch.h"
#include "src/utils/data_source.h"
#include "src/utils/orientation.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

Decoder::Decoder(State* const state, const DecoderConfig& config,
                 ArgbBuffer* const output_buffer)
    : state_(state),
      config_(config),
      output_((output_buffer != nullptr) ? output_buffer : &internal_output_),
      internal_output_(config.exact ? WP2_ARGB_32 : WP2_Argb_32) {
  if (state_ != nullptr) {
#if defined(WP2_USE_CONTEXT_SWITCH) && (WP2_USE_CONTEXT_SWITCH > 0)
    if (config.incremental_mode ==
        DecoderConfig::IncrementalMode::PARTIAL_TILE_CONTEXT) {
      state_->partial_tile_decoder.reset(new (WP2Allocable::nothrow)
                                             ContextTileDecoder());
      // If the allocation fails, the partial_tile_decoder will not be used.
    }
#endif  // WP2_USE_CONTEXT_SWITCH

    output_->metadata_.Clear();
  }
}

Decoder::~Decoder() { delete state_; }

WP2Status Decoder::GetStatus() const {
  if (state_ == nullptr) return WP2_STATUS_OUT_OF_MEMORY;
  if (state_->status == WP2_STATUS_NOT_ENOUGH_DATA) {
    // The decoding halted, hence the WP2_STATUS_NOT_ENOUGH_DATA, but maybe
    // WP2_STATUS_OK can be returned because there is nothing left to decode.
    if (state_->stage >= State::Stage::WAITING_FOR_METADATA_CHUNKS &&
        !features_.has_trailing_data) {
      // Nothing left to process if there is only XMP/EXIF metadata left to
      // decode but there is no trailing metadata.
      return WP2_STATUS_OK;
    }
    if (state_->stage >= State::Stage::WAITING_FOR_TILE_DATA &&
        state_->current_frame_is_ready) {
      const FrameFeatures* const current_frame_features =
          TryGetFrameDecodedFeatures(
              state_->GetFinalFrameIndex(state_->current_frame_index));
      assert(current_frame_features != nullptr);  // As current_frame_is_ready.
      if (current_frame_features->is_last && !features_.has_trailing_data) {
        // Nothing left to process if all frames are decoded and if there is no
        // metadata afterwards.
        return WP2_STATUS_OK;
      }
    }
  }
  return state_->status;
}

bool Decoder::Failed() const {
  return (state_ == nullptr || (state_->status != WP2_STATUS_OK &&
                                state_->status != WP2_STATUS_NOT_ENOUGH_DATA));
}

//------------------------------------------------------------------------------

// These functions return the data if the chunk is entirely decoded, even if an
// error happened after.
const BitstreamFeatures* Decoder::TryGetDecodedFeatures() const {
  if ((state_ != nullptr) &&
      (state_->stage > State::Stage::WAITING_FOR_HEADER_CHUNK)) {
    // TODO(yguyon): Return &features_ even just after Rewind()
    return &features_;
  }
  return nullptr;
}

const FrameFeatures* Decoder::TryGetFrameDecodedFeatures(
    uint32_t frame_index) const {
  // 'frame_index' is actually a final frame index.
  if (TryGetDecodedFeatures() != nullptr && state_ != nullptr &&
      frame_index < state_->final_to_regular_index.size()) {
    const uint32_t regular_frame_index =
        state_->final_to_regular_index[frame_index];
    assert(regular_frame_index < state_->frames.size());
    return &state_->frames[regular_frame_index];
  }
  return nullptr;
}

bool Decoder::TryGetFrameLocation(uint32_t frame_index, size_t* offset,
                                  size_t* length) const {
  // 'frame_index' is actually a final frame index.
  // 'length' will be the sum of all pre- and regular frames belonging to it.
  if (TryGetDecodedFeatures() == nullptr) return false;  // Early exit.

  // Find the first pre- or regular frame belonging to the final 'frame_index'.
  uint32_t first_pre_or_regular_frame_index;
  if (frame_index == 0) {
    first_pre_or_regular_frame_index = 0;
  } else {
    const uint32_t previous_final_frame_index = frame_index - 1;
    if (previous_final_frame_index >= state_->final_to_regular_index.size()) {
      return false;
    }
    first_pre_or_regular_frame_index =
        (state_->final_to_regular_index[previous_final_frame_index] + 1);
  }
  if (first_pre_or_regular_frame_index >= state_->frames.size()) return false;
  const size_t first_pre_or_regular_frame_offset =
      state_->frames[first_pre_or_regular_frame_index].bitstream_position;
  if (offset != nullptr) *offset = first_pre_or_regular_frame_offset;

  // Find the ending regular frame belonging to the final 'frame_index'.
  if (length != nullptr) {
    const uint32_t regular_frame_index =
        state_->final_to_regular_index[frame_index];
    if (regular_frame_index >= state_->frames.size()) return false;
    const size_t pos = state_->frames[regular_frame_index].bitstream_position;
    const size_t size = state_->frames[regular_frame_index].num_bytes;
    if (size == Decoder::State::kUnknown) return false;  // Not yet decoded.
    assert(pos + size > first_pre_or_regular_frame_offset);
    *length = pos + size - first_pre_or_regular_frame_offset;
  }
  return true;
}

//------------------------------------------------------------------------------

uint32_t Decoder::GetCurrentFrameIndex() const {
  if (state_ != nullptr) {
    return std::max(state_->GetFinalFrameIndex(state_->current_frame_index),
                    state_->skip_final_frames_before_index);
  }
  return 0;
}

uint32_t Decoder::GetNumFrameDecodedFeatures() const {
  if (state_ == nullptr) return 0;
  // Return the number of known final frames.
  return (uint32_t)state_->final_to_regular_index.size();
}

//------------------------------------------------------------------------------

Rectangle Decoder::GetDecodedArea() const {
  if (TryGetDecodedFeatures() == nullptr) return Rectangle(0, 0, 0, 0);

  uint32_t num_decoded_rows = 0;
  uint32_t num_decoded_columns = 0;
  if (state_ != nullptr &&
      state_->stage >= State::Stage::WAITING_FOR_TILE_DATA &&
      state_->current_frame_index < state_->frames.size()) {
    const State::InternalFrameFeatures& frame =
        state_->frames[state_->current_frame_index];

    if (frame.final_frame_index < state_->skip_final_frames_before_index) {
      // Skipped frames are not available to the user.
    } else if (frame.info.duration_ms == 0) {
      // No guarantee on the position of the frame following this preframe,
      // so the decoded area is not given until we hit a regular frame.
    } else {
      assert(state_->tiles_layout.num_tiles_x > 0);
      if (state_->frame_num_final_rows < frame.info.window.height) {
        num_decoded_rows = frame.info.window.y + state_->frame_num_final_rows;
      } else {
        num_decoded_rows = state_->raw_output_buffer.height();
      }

      // Return the width only if we have at least one row.
      if (num_decoded_rows > 0) {
        num_decoded_columns = state_->raw_output_buffer.width();
      }
    }
  }

  // For consistency even an empty rectangle is rotated (for x, y).
  return OrientateRectangle(
      features_.orientation, features_.raw_width, features_.raw_height,
      Rectangle(0, 0, num_decoded_columns, num_decoded_rows));
}

bool Decoder::ReadFrame(uint32_t* const duration_ms) {
  DecodeAvailableData();
  if (!Failed() && state_->current_frame_is_ready) {
    const State::InternalFrameFeatures& frame =
        state_->frames[state_->current_frame_index];
    if (frame.final_frame_index >= state_->skip_final_frames_before_index) {
      if (duration_ms != nullptr) {
        *duration_ms = features_.is_animation ? frame.duration_ms : 0;
      }
      return true;
    }
  }
  return false;
}

//------------------------------------------------------------------------------

void Decoder::SkipNumNextFrames(uint32_t num_frames_to_skip) {
  if (state_ != nullptr) {
    // Unlikely but make sure there is no overflow.
    num_frames_to_skip = std::min(kMaxNumFrames, num_frames_to_skip);
    state_->skip_final_frames_before_index =
        std::min(kMaxNumFrames, GetCurrentFrameIndex() + num_frames_to_skip);
  }
}

void Decoder::Rewind() {
  // Only rewind if no failure (USER_ABORT not being considered as one here).
  if (!Failed() ||
      (state_ != nullptr && state_->status == WP2_STATUS_USER_ABORT)) {
    state_->Rewind();
  }
}

//------------------------------------------------------------------------------

ArrayDecoder::ArrayDecoder(const DecoderConfig& config,
                           ArgbBuffer* output_buffer)
    // May fail. If so WP2_STATUS_OUT_OF_MEMORY will be returned by GetStatus().
    : Decoder(new(WP2Allocable::nothrow) DecoderStateArray(config), config,
              output_buffer) {}

ArrayDecoder::ArrayDecoder(const uint8_t* data, size_t size,
                           const DecoderConfig& config,
                           ArgbBuffer* output_buffer)
    : ArrayDecoder(config, output_buffer) {
  SetInput(data, size);
}

void ArrayDecoder::SetInput(const uint8_t* data, size_t size) {
  if (Failed()) return;

  DecoderStateArray* const state = (DecoderStateArray*)state_;
  state->array_data_source.Update(data, size);
  state->last_input_set = {data, size};
}

//------------------------------------------------------------------------------

StreamDecoder::StreamDecoder(const DecoderConfig& config,
                             ArgbBuffer* output_buffer)
    // May fail. If so WP2_STATUS_OUT_OF_MEMORY will be returned by GetStatus().
    : Decoder(new(WP2Allocable::nothrow) DecoderStateStream(config), config,
              output_buffer) {}

void StreamDecoder::AppendInput(const uint8_t* data, size_t size,
                                bool data_is_persistent) {
  if (Failed()) return;

  DecoderStateStream* const state = (DecoderStateStream*)state_;
  // Store previous bytes in case they become unavailable after this call
  // (only relevant if 'data_is_persistent' was true during the previous call).
  WP2Status data_source_status =
      state->stream_data_source.AppendExternalToInternal();
  if (data_source_status == WP2_STATUS_OK) {
    // Append new bytes, copying them if previous unread bytes remain because
    // they need to be merged with the new ones into a single buffer.
    data_source_status = state->stream_data_source.AppendAsExternal(data, size);
  }
  if (data_source_status == WP2_STATUS_OK && !data_is_persistent) {
    // Store new bytes (if not already done).
    data_source_status = state->stream_data_source.AppendExternalToInternal();
  }
  if (data_source_status != WP2_STATUS_OK) {
    state_->status = data_source_status;
  }
}

//------------------------------------------------------------------------------

CustomDecoder::CustomDecoder(const DecoderConfig& config,
                             ArgbBuffer* output_buffer)
    // May fail. If so WP2_STATUS_OUT_OF_MEMORY will be returned by GetStatus().
    : Decoder(new(WP2Allocable::nothrow) DecoderStateCustom(config, this),
              config, output_buffer) {}

void CustomDecoder::DecodeAvailableData() {
  if (Failed()) return;

  DecoderStateCustom* const state = (DecoderStateCustom*)state_;
  // Buffer might have been invalidated since last CustomDecoder::Read(), so
  // force refresh it.
  const WP2Status data_source_status = state->custom_data_source.ForceFetch();
  if (data_source_status != WP2_STATUS_OK) {
    state_->status = data_source_status;
  } else {
    Decoder::DecodeAvailableData();
  }
}

WP2Status CustomDataSource::ForceFetch() {
  if (num_available_bytes_ > 0) {
    // Make sure data is up-to-date by fetching what is already available.
    assert(num_available_bytes_ >= num_read_bytes_);
    const size_t num_requested_bytes = num_available_bytes_ - num_read_bytes_;
    num_available_bytes_ = 0;
    const uint8_t* data;
    WP2_CHECK_OK(TryGetNext(num_requested_bytes, &data),
                 WP2_STATUS_BITSTREAM_OUT_OF_MEMORY);
  }
  return WP2_STATUS_OK;
}

void CustomDataSource::Reset() {
  DataSource::Reset();
  fetcher_->Reset();
}

bool CustomDataSource::Fetch(size_t num_requested_bytes) {
  // CustomDataSource just redirects the call to CustomDecoder with the number
  // of read bytes taken into account.
  fetcher_->Fetch(num_read_bytes_ + num_requested_bytes, &available_bytes_,
                  &num_available_bytes_);
  return (num_available_bytes_ >= num_read_bytes_ + num_requested_bytes);
}

void CustomDataSource::OnDiscard(size_t num_bytes) {
  fetcher_->Discard(num_bytes);
  if (num_available_bytes_ > 0) {
    fetcher_->Fetch(num_available_bytes_, &available_bytes_,
                    &num_available_bytes_);
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
