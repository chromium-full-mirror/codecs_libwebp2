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
// Tests hash map implementation, that will be used for LZW algorithm.

#include <algorithm>
#include <array>
#include <vector>

#include "src/utils/hash_map.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

Vector_s16 BuildTestData() {
  constexpr std::array<int16_t, 20> data_array = {
      0xFF, 0xF0, 0xF1, 0xF2, 0xFF, 0xF3, 0xF4, 0xF5, 0xFF, 0xF6,
      0xF7, 0xF8, 0xFF, 0xF9, 0xFA, 0xFB, 0xFF, 0xFC, 0xFD, 0xFE,
  };
  Vector_s16 data;
  if (data.resize(20)) {
    std::copy(data_array.begin(), data_array.end(), data.begin());
  }
  return data;
}

// Build artificial 3x3 RGB square for testing.
WP2Status BuildArgbBuffer(WP2::ArgbBuffer& buf) {
  WP2_CHECK_STATUS(buf.SetFormat(WP2_Argb_32));
  constexpr uint32_t kSize = 3;
  WP2_CHECK_STATUS(buf.Resize(kSize, kSize));
  uint8_t color[4] = {0xFF, 0xFF, 0x00, 0x00};
  buf.Fill({0, 0, 3, 1}, color);
  color[2] = color[1];
  color[1] = 0x00;
  buf.Fill({0, 1, 3, 1}, color);
  color[3] = color[2];
  color[2] = 0x00;
  buf.Fill({0, 2, 3, 1}, color);
  return WP2_STATUS_OK;
}

// Tests if the hash map correctly adds a new segment.
TEST(LZWHashMap, InsertOneValue) {
  Vector_s16 data;

  ASSERT_TRUE(data.reserve(4));
  data.push_back_no_resize(0xFF);
  data.push_back_no_resize(0xF0);
  data.push_back_no_resize(0xF1);
  data.push_back_no_resize(0xF2);

  constexpr uint32_t size = 3;
  SegmentMap segment_map = SegmentMap(size);

  ASSERT_WP2_OK(segment_map.Allocate());

  uint32_t segment_index;
  const ColorSegment segment = {&data[0], 1};

  ASSERT_FALSE(segment_map.HasKey(segment, segment_index));

  ASSERT_WP2_OK(segment_map.FindOrAdd(segment, segment_index));

  ASSERT_TRUE(segment_map.HasKey(segment, segment_index));
}

// Tests if SegmentMap correctly adds until segments_ full.
TEST(LZWHashMap, InsertValuesUntilFull) {
  constexpr uint32_t size = 3;
  Vector_s16 data = BuildTestData();

  SegmentMap segment_map = SegmentMap(size);

  ASSERT_WP2_OK(segment_map.Allocate());

  const std::vector<ColorSegment> segments = {
      {&data[0], 1}, {&data[4], 1}, {&data[8], 2}};

  uint32_t segment_index;
  for (const ColorSegment& segment : segments) {
    ASSERT_FALSE(segment_map.HasKey(segment, segment_index));

    ASSERT_WP2_OK(segment_map.FindOrAdd(segment, segment_index));

    ASSERT_TRUE(segment_map.HasKey(segment, segment_index));
  }

  for (const ColorSegment& segment : segments) {
    ASSERT_TRUE(segment_map.HasKey(segment, segment_index));
  }

  // Clear cache and make sure segments are no longer found.
  // Since "StartLZW" was not called, "num_colors_" wasn't defined, and
  // every segment should be cleared.

  segment_map.Clear();
  for (const ColorSegment& segment : segments) {
    ASSERT_FALSE(segment_map.HasKey(segment, segment_index));
  }
}

// Tests if SegmentMap correctly fails to add when segments_ is full.
TEST(LZWHashMap, FailWhenFull) {
  constexpr uint32_t size = 3;
  Vector_s16 data = BuildTestData();

  SegmentMap segment_map = SegmentMap(size);

  ASSERT_WP2_OK(segment_map.Allocate());

  const std::vector<ColorSegment> segments = {
      {&data[0], 1}, {&data[4], 1}, {&data[8], 2}, {&data[16], 1}};

  ColorSegment segment;
  uint32_t segment_index;
  for (uint32_t i = 0; i < 3; ++i) {
    segment = segments[i];
    ASSERT_FALSE(segment_map.HasKey(segment, segment_index));

    ASSERT_WP2_OK(segment_map.FindOrAdd(segment, segment_index));

    ASSERT_TRUE(segment_map.HasKey(segment, segment_index));
  }

  // Now try to insert another segment, it should fail.
  segment = segments[3];
  ASSERT_FALSE(segment_map.HasKey(segment, segment_index));

  ASSERT_WP2_OK(segment_map.FindOrAdd(segment, segment_index));
  ASSERT_EQ(segment_index, kClearCode);

  ASSERT_FALSE(segment_map.HasKey(segment, segment_index));
}

// Tests if SegmentMap correctly inserts colors (1-pixel segments) from input.
TEST(LZWHashMap, InsertColors) {
  ArgbBuffer simple_rgb_buf;
  ASSERT_WP2_OK(BuildArgbBuffer(simple_rgb_buf));
  WP2L::Palette palette;
  palette.Init(WP2SampleFormat::WP2_Argb_32, true);
  ASSERT_WP2_OK(palette.AnalyzeAndCreate(simple_rgb_buf));

  constexpr uint32_t size = 10;
  SegmentMap segment_map = SegmentMap(size);
  ASSERT_WP2_OK(segment_map.Allocate());
  ASSERT_WP2_OK(segment_map.InsertColors(palette));

  constexpr int16_t kDataArray[] = {0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00,
                                    0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF};
  const ColorSegment segments[] = {
      {&kDataArray[0], 1}, {&kDataArray[4], 1}, {&kDataArray[8], 1}};

  uint32_t segment_index;
  for (const ColorSegment& segment : segments) {
    ASSERT_TRUE(segment_map.HasKey(segment, segment_index));
  }
}

}  // namespace
}  // namespace WP2
