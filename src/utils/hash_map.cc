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
// Defines a hash map for LZW-like algorithm.
//
// Author: Felipe Mourad Pereira (mouradfelipe@google.com)

#include "src/utils/hash_map.h"

#include <algorithm>
#include <cassert>

#include "src/dsp/math.h"
#include "src/utils/hash.h"
#include "src/utils/utils.h"

namespace WP2 {
SegmentMap::SegmentMap(uint32_t num_segments)
    : segments_capacity_(
          num_segments +
          1),  // it has a +1 due to 0-index reserved for clear code.
      num_bits_(WP2Log2Ceil_k(2 * segments_capacity_)),
      num_colors_(0) {}

void SegmentMap::Clear() {
  assert(num_colors_ < segments_.size());
  // Segments in [1;num_colors_] are pure colors, they shouldn't be cleared.
  segments_.resize_no_check(num_colors_ + 1);
  for (uint32_t& segment_index : hash_map_) {
    if (segment_index > num_colors_) {
      segment_index = 0;
    }
  }
}

WP2Status SegmentMap::Allocate() {
  WP2_CHECK_ALLOC_OK(segments_.reserve(segments_capacity_));
  WP2_CHECK_ALLOC_OK(hash_map_.resize(1 << num_bits_));
  segments_.push_back_no_resize({nullptr, 0});  // index 0 is for clear code
  return WP2_STATUS_OK;
}

WP2Status SegmentMap::InsertColors(const WP2L::Palette& palette) {
  ColorSegment seg;
  seg.num_pixels = 1;
  seg.data = &palette.GetColors()[0];
  for (uint32_t i = 0; i < palette.Size(); ++i, seg.data += 4) {
    assert(segments_.size() < segments_capacity_);
    uint32_t index;  // where to insert the segment in hash_map_
    if (HasKey(seg, index)) {
      // Color is duplicate.
      assert(false);
    }
    hash_map_[index] = segments_.size();
    segments_.push_back_no_resize(seg);
  }
  num_colors_ = segments_.size() - 1;
  return WP2_STATUS_OK;
}

bool SegmentMap::HasKey(const ColorSegment& segment, uint32_t& index) const {
  if (segment.num_pixels == 0) return false;
  const uint32_t hash =
      decoding::HashPixN(segment.data, segment.num_pixels, num_bits_);
  uint32_t i = hash;
  do {
    if (hash_map_[i] == 0) break;  // make sure it is not an empty position
    if (segments_[hash_map_[i]] == segment) {
      index = hash_map_[i];
      return true;
    }
    if (++i == hash_map_.size()) i = 0;
  } while (i != hash);
  index = i;  // this value is the position to add a new segment to hash_map_.
  return false;
}

WP2Status SegmentMap::FindOrAdd(const ColorSegment& segment, uint32_t& index) {
  WP2_CHECK_OK(segment.num_pixels != 0, WP2_STATUS_BAD_READ);
  if (HasKey(segment, index)) {
    return WP2_STATUS_OK;
  }

  if (segments_.size() == segments_capacity_) {
    // 'segments_' is full.
    // Request a clear cache.
    index = kClearCode;
    return WP2_STATUS_OK;
  }

  // If code reaches here, it means that 'index' is actually the index
  // in hash_map_ to insert the element.
  hash_map_[index] = segments_.size();
  index = segments_.size();
  segments_.push_back_no_resize(segment);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// SegmentCache.

WP2Status SegmentCache::Allocate(uint32_t num_bits) {
  num_bits_ = num_bits;
  const uint64_t hash_size = (1 << num_bits_);
  WP2_CHECK_ALLOC_OK(segments_.resize(hash_size));
  return WP2_STATUS_OK;
}

bool SegmentCache::HasKey(const ColorSegment& segment,
                          uint32_t* const index) const {
  assert(segment.num_pixels != 0);
  const uint32_t hash =
      decoding::HashPixN(segment.data, segment.num_pixels, num_bits_);
  if (segments_[hash] == segment) {
    if (index != nullptr) *index = hash;
    return true;
  } else {
    return false;
  }
}

bool SegmentCache::Insert(const ColorSegment& segment, uint32_t* const index) {
  const uint32_t hash =
      decoding::HashPixN(segment.data, segment.num_pixels, num_bits_);
  if (index != nullptr) *index = hash;
  if (segments_[hash] == segment) {
    return false;
  } else {
    segments_[hash] = segment;
    return true;
  }
}

void SegmentCache::Lookup(uint32_t index, ColorSegment* const segment) const {
  assert((index >> num_bits_) == 0u);
  *segment = segments_[index];
}
}  // namespace WP2
