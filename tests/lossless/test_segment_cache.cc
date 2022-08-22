// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Tests segment cache implementation, used during lossless compression.

#include "src/utils/hash_map.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

TEST(SegmentCache, Simple) {
  const int16_t kGreen[] = {255, 0, 255, 0};
  const int16_t kRed[] = {255, 255, 0, 0};
  const int16_t kRedGreen[] = {255, 255, 0, 0, 255, 0, 255, 0};

  const ColorSegment kGreenSegment = {&kGreen[0], 1};
  const ColorSegment kRedSegment = {&kRed[0], 1};
  const ColorSegment kRedGreenSegment = {&kRedGreen[0], 2};

  constexpr uint32_t num_bits = 1;  // make sure a collision happens
  SegmentCache segment_cache;
  ASSERT_WP2_OK(segment_cache.Allocate(num_bits));
  uint32_t cache_index;

  EXPECT_FALSE(segment_cache.HasKey(kGreenSegment, &cache_index));
  EXPECT_FALSE(segment_cache.HasKey(kRedSegment, &cache_index));
  EXPECT_FALSE(segment_cache.HasKey(kRedGreenSegment, &cache_index));

  uint32_t red_index;
  ASSERT_TRUE(segment_cache.Insert(kRedSegment, &red_index));
  EXPECT_TRUE(segment_cache.HasKey(kRedSegment, &red_index));
  EXPECT_FALSE(segment_cache.Insert(kRedSegment, &red_index));

  uint32_t green_index;
  ASSERT_FALSE(segment_cache.HasKey(kGreenSegment, &green_index));
  EXPECT_TRUE(segment_cache.Insert(kGreenSegment, &green_index));
  EXPECT_TRUE(segment_cache.HasKey(kGreenSegment, &green_index));
  ASSERT_FALSE(segment_cache.Insert(kGreenSegment, &green_index));

  bool red_removed = red_index == green_index;
  ASSERT_EQ(red_removed, !segment_cache.HasKey(kRedSegment, &red_index));

  uint32_t red_green_index;
  ASSERT_FALSE(segment_cache.HasKey(kRedGreenSegment, &red_green_index));
  EXPECT_TRUE(segment_cache.Insert(kRedGreenSegment, &red_green_index));
  EXPECT_TRUE(segment_cache.HasKey(kRedGreenSegment, &red_green_index));
  ASSERT_FALSE(segment_cache.Insert(kRedGreenSegment, &red_green_index));

  bool green_removed = green_index == red_green_index;
  red_removed = red_removed || (red_index == red_green_index);
  ASSERT_TRUE(green_removed || red_removed);
  EXPECT_EQ(red_removed, !segment_cache.HasKey(kRedSegment, &red_index));
  EXPECT_EQ(green_removed, !segment_cache.HasKey(kGreenSegment, &green_index));

  uint32_t new_red_index;
  ASSERT_EQ(red_removed, segment_cache.Insert(kRedSegment, &new_red_index));
  ASSERT_EQ(red_index, new_red_index);

  ColorSegment return_segment;
  segment_cache.Lookup(new_red_index, &return_segment);
  ASSERT_EQ(return_segment, kRedSegment);

  ASSERT_EQ(segment_cache.IndexRange(), 1u << num_bits);
}

}  // namespace
}  // namespace WP2
