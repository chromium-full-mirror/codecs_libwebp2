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
// Implementation of a variant of the LZW decoder.
//
// Author: Felipe Mourad Pereira (mouradfelipe@google.com)

#include <algorithm>

#include "src/common/symbols.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/utils/hash_map.h"
#include "src/utils/utils.h"

namespace WP2L {

namespace {
void PushSegmentToVector(const ColorSegment& seg, const int16_t* const src_end,
                         int16_t** src) {
  const uint32_t num_copy_bytes =
      std::min(4 * seg.num_pixels, static_cast<uint32_t>(src_end - *src));
  if (seg.data + num_copy_bytes <= *src) {
    std::copy(seg.data, seg.data + num_copy_bytes, *src);
  } else {
    for (uint32_t i = 0; i < num_copy_bytes; ++i) {
      (*src)[i] = seg.data[i];
    }
  }
  *src += num_copy_bytes;
}

// Initializes code table for the first pixel or when clear cache signal.
WP2Status InitializeLZW(uint32_t num_colors, const int16_t* const src_end,
                        WP2::Vector<ColorSegment>* const segments,
                        WP2::ANSDec* const dec, ColorSegment* const seg,
                        int16_t** src) {
  segments->resize_no_check(num_colors + 1);
  // Exclude the possibility of receiving clear code.
  static_assert(kClearCode == 0, "kClearCode must be 0.");
  const int32_t cur_code = dec->ReadRange(1, num_colors, "code");
  WP2_CHECK_STATUS(dec->GetStatus());
  *seg = {*src, 1};
  PushSegmentToVector((*segments)[cur_code], src_end, src);
  return WP2_STATUS_OK;
}
}  // namespace

WP2Status Decoder::DecodeLZW(uint32_t width, uint32_t last_row,
                             int16_t** src_out) {
  uint32_t row = last_pixel_ / width;

  int16_t* const data = pixels_.data();
  int16_t* src = data + 4 * last_pixel_;
  const int16_t* const src_last =
      data + 4 * width * last_row;  // last pixel to decode (exclusive)
  const int16_t* const src_end =
      pixels_.data() + pixels_.size();  // last possible pixel (exclusive)
  // LZW-like decompression algorithm.
  ColorSegment prev_seg;

  if (last_pixel_ == 0) {
    // Initialize output as the first code segment.
    WP2_CHECK_STATUS(InitializeLZW(num_colors_, src_end, &lzw_segments_, dec_,
                                   &prev_seg, &src));
  } else {
    prev_seg = lzw_last_seg_;
  }

  while (src < src_last) {
    ColorSegment cur_seg;
    if (lzw_segments_.size() == lzw_segments_.capacity()) {
      // Segments must be restarted.
      // Next code is necessarily of a simple single color segment.
      // Behaviour should be similar to starting code.
      WP2_CHECK_STATUS(InitializeLZW(num_colors_, src_end, &lzw_segments_, dec_,
                                     &cur_seg, &src));
    } else {
      // The range can go up to segments size (inclusive),
      // since code read might be of a new code.
      const int32_t cur_code = dec_->ReadRange(0, lzw_segments_.size(), "code");
      WP2_CHECK_STATUS(dec_->GetStatus());
      if (cur_code == kClearCode) {
        WP2_CHECK_STATUS(InitializeLZW(num_colors_, src_end, &lzw_segments_,
                                       dec_, &cur_seg, &src));
      } else if (cur_code == static_cast<int32_t>(lzw_segments_.size())) {
        // New code.
        lzw_segments_.push_back_no_resize({src, prev_seg.num_pixels + 1});
        PushSegmentToVector(prev_seg, src_end, &src);
        std::copy(prev_seg.data, prev_seg.data + 4, src);
        src += 4;
        cur_seg = lzw_segments_[cur_code];
      } else {
        cur_seg = {src, lzw_segments_[cur_code].num_pixels};
        PushSegmentToVector(lzw_segments_[cur_code], src_end, &src);
        lzw_segments_.push_back_no_resize(
            {prev_seg.data, prev_seg.num_pixels + 1});
      }
    }

    const uint32_t latest_unfinished_row = (src - data) / 4 / width;
    while (row < latest_unfinished_row) {
      ++row;
      if (row <= last_row) {
        WP2_CHECK_STATUS(ProcessRows(row));
      }
    }
    prev_seg = cur_seg;
  }
  lzw_last_seg_ = prev_seg;
  if (src_out != nullptr) *src_out = src;
  return WP2_STATUS_OK;
}

}  // namespace WP2L
