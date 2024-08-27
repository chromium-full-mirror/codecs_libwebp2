// Copyright 2020 Google LLC
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

// Integration tests for tile shapes.

#include <cstdint>
#include <string>
#include <tuple>

#include "imageio/anim_image_dec.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class TileShapeTest
    : public testing::TestWithParam<std::tuple<std::string, TileShape>> {};

TEST_P(TileShapeTest, Simple) {
  const std::string& file_name = std::get<0>(GetParam());
  EncoderConfig config = EncoderConfig::kDefault;
  config.tile_shape = (TileShape)std::get<1>(GetParam());

  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name, config));
}

INSTANTIATE_TEST_SUITE_P(
    PredictorTestInstantiationAll, TileShapeTest,
    testing::Combine(testing::Values("source1_64x48.png", "background.png",
                                     "large.png", "verylarge.png"),
                     testing::Values(TILE_SHAPE_SQUARE_128,
                                     TILE_SHAPE_SQUARE_256,
                                     TILE_SHAPE_SQUARE_512, TILE_SHAPE_WIDE)));

//------------------------------------------------------------------------------

// Regression test for tile dimension mismatch due to being based on frame width
// and height at encoding and image width and height at decoding.
TEST(LosslessAnimationTileShapeWideTest, Regression) {
  ArgbBuffer frame(WP2_ARGB_32);
  const std::string path =
      testutil::GetTestDataPath("okfestival2014_sullivan_vs_sullivan.gif");
  ImageReader reader(path.c_str(), &frame);
  AnimationEncoder encoder;
  bool is_last;
  int frame_count = 0;
  do {
    uint32_t duration_ms;
    ASSERT_WP2_OK(reader.ReadFrame(&is_last, &duration_ms));
    ++frame_count;
    // Speed up the test while keeping the relevant code covered.
    if (frame_count != 1 && frame_count != 30) continue;
    ASSERT_WP2_OK(encoder.AddFrame(frame, duration_ms));
  } while (!is_last);

  EncoderConfig config = EncoderConfig::kDefault;
  config.quality = 100.0f;
  config.alpha_quality = 100.0f;
  config.effort = 1;
  config.keep_unmultiplied = true;
  config.tile_shape = WP2::TILE_SHAPE_WIDE;
  MemoryWriter writer;
  ASSERT_WP2_OK(encoder.Encode(&writer, config));

  ArrayDecoder decoder(writer.mem_, writer.size_);
  uint32_t duration_ms;
  while (decoder.ReadFrame(&duration_ms)) {
  }
  ASSERT_WP2_OK(decoder.GetStatus());
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
