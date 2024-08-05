// Copyright 2022 Google LLC
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
// Implementation of a variant of the LZW encoder.
// But using a hash map instead of a look up table.
//
// Author: Felipe Mourad Pereira (mouradfelipe@google.com)

#include <algorithm>
#include <cmath>

#include "src/enc/lossless/losslessi_enc.h"
#include "src/utils/hash_map.h"
#include "src/utils/utils.h"

namespace WP2L {

namespace {
void ProcessCode(uint32_t start_pixel, uint32_t end_pixel, uint32_t code,
                 uint32_t max_range, bool* const has_just_cleared_cache,
                 WP2::ANSEncBase* const enc, EncodeInfo* const encode_info) {
  uint32_t min_range = 0;
  if (*has_just_cleared_cache) {
    // Exclude the possibility of sending clear code right after clearing cache.
    static_assert(kClearCode == 0, "kClearCode must be 0.");
    *has_just_cleared_cache = false;
    min_range = 1;
  }
  enc->PutRange(code, min_range, max_range, "code");
  if (encode_info != nullptr && !encode_info->bits_per_pixel.empty()) {
    std::fill(encode_info->bits_per_pixel.begin() + start_pixel,
              encode_info->bits_per_pixel.begin() + end_pixel,
              (log2(max_range - min_range + 1)) / (end_pixel - start_pixel));
  }
}

// Update line_tokens from last_written_row excluded to row_end excluded.
void UpdateLineTokens(uint32_t row_end, const WP2::ANSEncBase& enc,
                      int32_t* const last_written_row,
                      EncodeInfo* const encode_info) {
  while (*last_written_row + 1 < static_cast<int32_t>(row_end)) {
    ++*last_written_row;
    encode_info->line_tokens[*last_written_row] = enc.NumTokens();
  }
}

WP2Status Impl(const int16_t* const argb, uint32_t width, uint32_t height,
               const Palette& palette, uint32_t capacity,
               WP2::ANSEncBase* const enc, EncodeInfo* const encode_info) {
  WP2::SegmentMap segment_map(capacity);
  WP2_CHECK_STATUS(segment_map.Allocate());

  bool has_just_cleared_cache = true;  // start with a brand new cache

  // LZW-like compression algorithm.
  WP2_CHECK_STATUS(segment_map.InsertColors(palette));

  // Initialize lzw_buf with first pixel.
  ColorSegment lzw_buf = {argb, 1};

  uint32_t seg_index;
  WP2_CHECK_OK(segment_map.HasKey(lzw_buf, seg_index),
               WP2_STATUS_INVALID_PARAMETER);
  uint32_t prev_buff_index = seg_index;
  uint32_t prev_buff_num_pixels = 1;
  const uint32_t total_pixels = width * height;
  int32_t last_written_row = -1;
  bool new_line = false;
  for (uint32_t pixel_index = 1; pixel_index < total_pixels; ++pixel_index) {
    ++lzw_buf.num_pixels;
    // Check if lzw_buf is inside code table already.
    if (!segment_map.HasKey(lzw_buf, seg_index)) {
      ProcessCode(pixel_index - prev_buff_num_pixels, pixel_index,
                  prev_buff_index, segment_map.MaxCode(),
                  &has_just_cleared_cache, enc, encode_info);
      if (last_written_row + 1 < static_cast<int32_t>(pixel_index / width)) {
        new_line = true;
      }
      WP2_CHECK_STATUS(segment_map.FindOrAdd(lzw_buf, seg_index));
      if (seg_index == kClearCode) {
        // Cache is full and should be cleared.
        segment_map.Clear();
        has_just_cleared_cache = true;
      }
      lzw_buf.num_pixels = 1;
      lzw_buf.data = argb + 4 * pixel_index;
      WP2_CHECK_OK(segment_map.HasKey(lzw_buf, seg_index),
                   WP2_STATUS_INVALID_PARAMETER);
    }
    prev_buff_index = seg_index;
    prev_buff_num_pixels = lzw_buf.num_pixels;

    if (encode_info != nullptr && !encode_info->line_tokens.empty() &&
        new_line) {
      UpdateLineTokens(pixel_index / width, *enc, &last_written_row,
                       encode_info);
      new_line = false;
    }
  }

  // Put remaining from lzw_buf inside code stream.
  ProcessCode(total_pixels - prev_buff_num_pixels, total_pixels,
              prev_buff_index, segment_map.MaxCode(), &has_just_cleared_cache,
              enc, encode_info);
  if (encode_info != nullptr && !encode_info->line_tokens.empty()) {
    UpdateLineTokens(height, *enc, &last_written_row, encode_info);
  }
  return WP2_STATUS_OK;
}
}  // namespace

WP2Status LZWEncode(const Buffer_s16& input_buffer, const Palette& palette,
                    int effort, WP2::ANSEncBase* const enc,
                    EncodeInfo* const encode_info) {
  const uint32_t capacity = palette.Size() + kLZWCapacity;
  const uint32_t width = input_buffer.width, height = input_buffer.height;
  const int16_t* const argb = input_buffer.data;

  // Write the data.
  WP2_CHECK_STATUS(
      Impl(argb, width, height, palette, capacity, enc, encode_info));

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2L
