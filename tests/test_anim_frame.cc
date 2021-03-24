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

// Tests for FrameEncoder.

#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "examples/example_utils.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "include/helpers.h"
#include "src/common/color_precision.h"
#include "src/common/progress_watcher.h"
#include "src/enc/anim/anim_enc.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/orientation.h"
#include "src/utils/utils.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

using WP2::testutil::GetTestDataPath;

constexpr ProgressRange kProgress;

TEST(AnimFrameTest, ForceLossless) {
  std::vector<ArgbBuffer> buffers;
  ASSERT_WP2_OK(testutil::ReadImages(
      {GetTestDataPath("logo.png"), GetTestDataPath("background.png")},
      &buffers));

  Vector<Frame> frames;
  ASSERT_TRUE(frames.resize(2));
  Frame& first_frame = frames[0];
  first_frame.duration_ms = 10;
  first_frame.force_lossless = true;
  ASSERT_WP2_OK(first_frame.rgb_pixels.SetView(buffers[0]));

  Frame& second_frame = frames[1];
  second_frame.duration_ms = 20;
  ASSERT_WP2_OK(second_frame.rgb_pixels.SetView(buffers[1]));

  const uint32_t rgb_bit_depth =
      (first_frame.rgb_pixels.format() == WP2_Argb_38) ? 10 : 8;
  const bool has_alpha = !AllFramesAreOpaque(frames);

  EncoderConfig config;
  config.quality = 80;
  config.tile_shape = TILE_SHAPE_SQUARE_256;
  MemoryWriter encoded_data;

  ASSERT_WP2_OK(EncodeHeader(
      config, first_frame.GetWidth(), first_frame.GetHeight(), rgb_bit_depth,
      has_alpha, /*is_anim=*/(frames.size() > 1), /*loop_forever=*/false,
      kDefaultBackgroundColor, RGB12b{10, 0, 30}, /*has_icc=*/false,
      /*has_trailing_data=*/false, &encoded_data));

  FrameEncoder frame_encode(config, kDefaultBackgroundColor, frames,
                            kProgress);
  ASSERT_WP2_OK(frame_encode.Encode(&encoded_data));

  // Decode.
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_);

  // First frame should be lossless (PSNR 99).
  uint32_t duration_ms = 0;
  ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
  EXPECT_EQ(duration_ms, first_frame.duration_ms);
  EXPECT_TRUE(testutil::Compare(first_frame.rgb_pixels, decoder.GetPixels(),
                                "logo", 99.f));

  // We don't force lossless on the second frame so it will probably be lossy.
  ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
  EXPECT_EQ(duration_ms, second_frame.duration_ms);
  EXPECT_TRUE(testutil::Compare(second_frame.rgb_pixels, decoder.GetPixels(),
                                "background", 35.f));

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

// orientation, is_slow.
typedef std::tuple<Orientation, bool> Param;

class ParameterizedAnimFrameTest : public ::testing::TestWithParam<Param> {
 public:
  // Inits decoded_frame_ buffer.
  WP2Status InitDecodedFrame(uint32_t width, uint32_t height) {
    const Orientation orientation = std::get<0>(GetParam());
    const bool is_slow = std::get<1>(GetParam());

    WP2_CHECK_STATUS(decoded_frame_internal_.Resize(
        OrientateWidth(orientation, width, height),
        OrientateHeight(orientation, width, height)));
    // Allows setting "is_slow".
    WP2_CHECK_STATUS(decoded_frame_.SetExternal(
        decoded_frame_internal_.width(), decoded_frame_internal_.height(),
        decoded_frame_internal_.GetRow8(0), decoded_frame_internal_.stride(),
        is_slow));
    return WP2_STATUS_OK;
  }

  ArgbBuffer decoded_frame_;

 protected:
  ArgbBuffer decoded_frame_internal_;
};

TEST_P(ParameterizedAnimFrameTest, Blend) {
  const Orientation orientation = std::get<0>(GetParam());

  std::vector<ArgbBuffer> buffers;
  ASSERT_WP2_OK(testutil::ReadImages(
      {GetTestDataPath("background.png"), GetTestDataPath("logo.png"),
       GetTestDataPath("composited.png")},
      &buffers));
  // Smaller view to speed up the test.
  for (ArgbBuffer& b : buffers) {
    ASSERT_WP2_OK(b.SetView(b, {0, 0, b.width() / 2, b.height() / 2}));
  }

  const Rectangle window = {0, 0, buffers[0].width(), buffers[0].height()};

  Vector<Subframe> subframes;
  ASSERT_TRUE(subframes.resize(2));
  Subframe& first_frame = subframes[0];
  first_frame.duration_ms = 10;
  first_frame.window = window;
  first_frame.dispose = true;
  ASSERT_WP2_OK(first_frame.rgb_pixels.SetView(buffers[0]));

  Subframe& second_frame = subframes[1];
  second_frame.duration_ms = 5;
  second_frame.blend = true;
  second_frame.window = window;
  ASSERT_WP2_OK(second_frame.rgb_pixels.SetView(buffers[1]));

  const uint32_t rgb_bit_depth =
      (first_frame.rgb_pixels.format() == WP2_Argb_38) ? 10 : 8;

  bool has_alpha = false;
  for (const Subframe& subframe : subframes) {
    has_alpha |= subframe.rgb_pixels.HasTransparency();
    if (has_alpha) break;
  }

  EncoderConfig config;
  config.quality = 80;
  config.tile_shape = TILE_SHAPE_SQUARE_256;
  config.decoding_orientation = orientation;
  MemoryWriter encoded_data;

  ASSERT_WP2_OK(EncodeHeader(
      config, first_frame.rgb_pixels.width(), first_frame.rgb_pixels.height(),
      rgb_bit_depth, has_alpha, /*is_anim=*/(subframes.size() > 1),
      /*loop_forever=*/false, kDefaultBackgroundColor, RGB12b{10, 0, 30},
      /*has_icc=*/false, /*has_trailing_data=*/false, &encoded_data));

  SubframeEncoder subframe_encoder(GetQualityHint(config.quality), has_alpha,
                                   config.tile_shape);
  ASSERT_WP2_OK(subframe_encoder.EncodeSubframe(
      config, first_frame, /*is_last=*/false, kProgress, &encoded_data));
  ASSERT_WP2_OK(subframe_encoder.EncodeSubframe(
      config, second_frame, /*is_last=*/true, kProgress, &encoded_data));
  // Decode.
  ASSERT_WP2_OK(InitDecodedFrame(window.width, window.height));
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_,
                       DecoderConfig::kDefault, &decoded_frame_);

  // First frame (shows "background.png").
  uint32_t duration_ms;
  ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
  EXPECT_EQ(duration_ms, first_frame.duration_ms);
  EXPECT_TRUE(testutil::Compare(buffers[0], decoded_frame_, "background", 35.,
                                PSNR, config.decoding_orientation));

  // Second frame (shows "logo.png" composited on "background.png" which is
  // "composited.png").
  ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
  EXPECT_EQ(duration_ms, second_frame.duration_ms);
  EXPECT_TRUE(testutil::Compare(buffers[2], decoded_frame_, "composited", 35.,
                                PSNR, config.decoding_orientation));

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

TEST_P(ParameterizedAnimFrameTest, BlendWithPreframe) {
  const Orientation orientation = std::get<0>(GetParam());

  std::vector<ArgbBuffer> buffers;
  ASSERT_WP2_OK(testutil::ReadImages(
      {GetTestDataPath("background.png"), GetTestDataPath("logo.png"),
       GetTestDataPath("composited.png")},
      &buffers));
  // Smaller view to speed up the test.
  for (ArgbBuffer& b : buffers) {
    ASSERT_WP2_OK(b.SetView(
        b, {b.width() / 3, b.height() / 3, b.width() / 3, b.height() / 3}));
  }

  const Rectangle window = {0, 0, buffers[0].width(), buffers[0].height()};

  Vector<Subframe> subframes;
  ASSERT_TRUE(subframes.resize(2));
  Subframe& first_frame = subframes[0];
  first_frame.duration_ms = 0;  // Preframe.
  first_frame.window = window;
  first_frame.dispose = true;
  ASSERT_WP2_OK(first_frame.rgb_pixels.SetView(buffers[0]));

  Subframe& second_frame = subframes[1];
  second_frame.duration_ms = 5;
  second_frame.blend = true;
  second_frame.window = window;
  ASSERT_WP2_OK(second_frame.rgb_pixels.SetView(buffers[1]));

  const uint32_t rgb_bit_depth =
      (first_frame.rgb_pixels.format() == WP2_Argb_38) ? 10 : 8;

  bool has_alpha = false;
  for (const Subframe& subframe : subframes) {
    has_alpha |= subframe.rgb_pixels.HasTransparency();
    if (has_alpha) break;
  }

  EncoderConfig config;
  config.quality = 80;
  config.tile_shape = TILE_SHAPE_SQUARE_256;
  config.decoding_orientation = orientation;
  MemoryWriter encoded_data;

  ASSERT_WP2_OK(EncodeHeader(
      config, first_frame.rgb_pixels.width(), first_frame.rgb_pixels.height(),
      rgb_bit_depth, has_alpha, /*is_anim=*/(subframes.size() > 1),
      /*loop_forever=*/false, kDefaultBackgroundColor, RGB12b{10, 0, 30},
      /*has_icc=*/false, /*has_trailing_data=*/false, &encoded_data));

  SubframeEncoder subframe_encoder(GetQualityHint(config.quality), has_alpha,
                                   config.tile_shape);
  ASSERT_WP2_OK(subframe_encoder.EncodeSubframe(
      config, first_frame, /*is_last=*/false, kProgress, &encoded_data));
  ASSERT_WP2_OK(subframe_encoder.EncodeSubframe(
      config, second_frame, /*is_last=*/true, kProgress, &encoded_data));
  // Decode.
  ASSERT_WP2_OK(InitDecodedFrame(window.width, window.height));
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_,
                       DecoderConfig::kDefault, &decoded_frame_);

  // Final frame (shows "logo.png" composited on "background.png" which is
  // "composited.png").
  uint32_t duration_ms;
  ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
  EXPECT_EQ(duration_ms, second_frame.duration_ms);
  EXPECT_TRUE(decoder.TryGetFrameDecodedFeatures(0)->is_last);
  EXPECT_TRUE(testutil::Compare(buffers[2], decoded_frame_, "composited", 35.,
                                PSNR, config.decoding_orientation));

  // Normal (non incremental) decode.
  ArgbBuffer decoded_first_frame;
  ASSERT_WP2_OK(Decode(encoded_data.mem_, encoded_data.size_,
                       &decoded_first_frame, DecoderConfig::kDefault));
  // We should get the same result as incremental decoding.
  EXPECT_TRUE(testutil::Compare(decoded_frame_, decoded_first_frame,
                                "composited", 99.f));

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

Argb32b RandomColor(bool has_alpha, UniformIntDistribution* const random) {
  const uint8_t alpha = (has_alpha ? random->Get(0u, kAlphaMax) : kAlphaMax);
  return {
      alpha,                           // premultiplied
      random->Get<uint8_t>(0, alpha),  // red
      random->Get<uint8_t>(0, alpha),  // green
      random->Get<uint8_t>(0, alpha)   // blue
  };
}

Rectangle RandomRect(uint32_t frame_width, uint32_t frame_height,
                     UniformIntDistribution* const random) {
  const uint32_t x = random->Get<uint32_t>(0, frame_width - 2);
  const uint32_t y = random->Get<uint32_t>(0, frame_height - 2);
  return {x, y, random->Get<uint32_t>(1, frame_width - x - 1),
          random->Get<uint32_t>(1, frame_height - y - 1)};
}

// Set (e.g. to /tmp) to debug RandomFrames test below.
const char* const kDebugOutputPath = "";

WP2Status FillRandom(const std::string& debug_str,
                     const ArgbBuffer* const previous,
                     UniformIntDistribution* const random,
                     ArgbBuffer* const buffer) {
  if (previous != nullptr && random->FlipACoin()) {
    WP2_CHECK_STATUS(buffer->CopyFrom(*previous));
  } else if (random->FlipACoin()) {
    buffer->Fill(RandomColor(/*has_alpha=*/true, random));
  } else {
    buffer->Fill({0, 0, 0, 0});
  }

  const uint32_t width = buffer->width();
  const uint32_t height = buffer->height();
  const uint32_t num_rects = random->Get(1, 3);
  for (uint32_t j = 0; j < num_rects; ++j) {
    Rectangle window = RandomRect(width, height, random);
    buffer->Fill(window, random->FlipACoin()
                             ? Argb32b{0, 0, 0, 0}
                             : RandomColor(/*has_alpha=*/true, random));
  }

  if (std::strlen(kDebugOutputPath) > 0) {
    WP2_CHECK_STATUS(WP2::SaveImage(
        *buffer,
        SPrintf("%s/%s.png", kDebugOutputPath, debug_str.c_str()).c_str(),
        true));
  }
  return WP2_STATUS_OK;
}

WP2Status SaveDebug(const ArgbBuffer& expected_buffer,
                    const ArgbBuffer& decoded_frame, Orientation orientation,
                    uint32_t test_id, uint32_t final_frame_index) {
  if (std::strlen(kDebugOutputPath) > 0) {
    ArgbBuffer expected_rotated;
    WP2_CHECK_STATUS(expected_rotated.CopyFrom(expected_buffer));
    WP2_CHECK_STATUS(OrientateBuffer(orientation, &expected_rotated));
    WP2_CHECK_STATUS(
        WP2::SaveImage(decoded_frame,
                       SPrintf("%s/test%d-%d-decoded.png", kDebugOutputPath,
                               test_id, final_frame_index)
                           .c_str(),
                       true));
    WP2_CHECK_STATUS(
        WP2::SaveImage(expected_rotated,
                       SPrintf("%s/test%d-%d-expected.png", kDebugOutputPath,
                               test_id, final_frame_index)
                           .c_str(),
                       true));
  }
  return WP2_STATUS_OK;
}

TEST_P(ParameterizedAnimFrameTest, RandomFrames) {
  const Orientation orientation = std::get<0>(GetParam());
  const bool is_slow = std::get<1>(GetParam());
  const int test_id = (int)orientation * 2 + is_slow;
  UniformIntDistribution random(test_id);

  Vector<Frame> frames;
  const int num_frames = random.Get(5, 10);
  ASSERT_TRUE(frames.resize(num_frames));

  const uint32_t width = random.Get(100, 600);
  const uint32_t height = random.Get(25, 50);

  for (uint32_t i = 0; i < frames.size(); ++i) {
    Frame& frame = frames[i];
    frame.duration_ms = random.Get(1, 100);
    frame.force_dispose = random.FlipACoin();
    frame.force_lossless = random.FlipACoin();
    ASSERT_WP2_OK(frame.rgb_pixels.Resize(width, height));
    const std::string debug_str =
        SPrintf("test%d-frame%d-duration%d-force_dispose%d", test_id, i,
                frame.duration_ms, frame.force_dispose);
    ASSERT_WP2_OK(
        FillRandom(debug_str,
                   /*previous=*/i > 0 ? &(frames[i - 1].rgb_pixels) : nullptr,
                   &random, &frame.rgb_pixels));

    printf("frame %d duration %d force_dispose %d\n", i, frame.duration_ms,
           frame.force_dispose);
  }

  const uint32_t rgb_bit_depth =
      (frames.front().rgb_pixels.format() == WP2_Argb_38) ? 10 : 8;
  const bool has_alpha = !AllFramesAreOpaque(frames);

  EncoderConfig config;
  config.quality = random.Get(0, 95);
  config.tile_shape = TILE_SHAPE_SQUARE_128;
  config.decoding_orientation = orientation;
  MemoryWriter encoded_data;

  const Argb32b background_color = RandomColor(has_alpha, &random);
  printf("Background_color: Argb32(%d, %d, %d, %d)\n", background_color.a,
         background_color.r, background_color.g, background_color.b);
  ASSERT_WP2_OK(EncodeHeader(
      config, width, height, rgb_bit_depth, has_alpha,
      /*is_anim=*/(frames.size() > 1), /*loop_forever=*/false,
      ToArgb38b(background_color), /*preview_color=*/RGB12b{10, 0, 30},
      /*has_icc=*/false, /*has_trailing_data=*/false, &encoded_data));

  FrameEncoder frame_encode(config, ToArgb38b(background_color), frames,
                            kProgress);
  ASSERT_WP2_OK(frame_encode.Encode(&encoded_data));

  // Decode.
  ASSERT_WP2_OK(InitDecodedFrame(width, height));
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_,
                       DecoderConfig::kDefault, &decoded_frame_);

  for (uint32_t i = 0; i < frames.size(); ++i) {
    const Frame& frame = frames[i];

    uint32_t duration_ms = 0;
    ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
    EXPECT_TRUE(testutil::Compare(
        frame.rgb_pixels, decoded_frame_, "fake data",
        testutil::GetExpectedDistortion(config.quality), PSNR, orientation))
        << "Error on final frame " << i;

    ASSERT_WP2_OK(
        SaveDebug(frame.rgb_pixels, decoded_frame_, orientation, test_id, i));
  }

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

TEST_P(ParameterizedAnimFrameTest, RandomSubframes) {
  const Orientation orientation = std::get<0>(GetParam());
  const bool is_slow = std::get<1>(GetParam());
  const int test_id = (int)orientation * 2 + is_slow;
  UniformIntDistribution random(test_id);

  Vector<Subframe> subframes;
  const uint32_t num_subframes = random.Get(20, 40);
  ASSERT_TRUE(subframes.resize(num_subframes));

  const uint32_t width = random.Get(1, 50);
  const uint32_t height = random.Get(1, 50);
  bool has_alpha = false;

  EncoderConfig config;
  config.quality = random.Get(50, 95);
  config.tile_shape = TILE_SHAPE_SQUARE_128;
  config.decoding_orientation = orientation;
  MemoryWriter encoded_data;

  uint32_t num_consecutive_preframes = 0;
  for (uint32_t i = 0; i < subframes.size(); ++i) {
    Subframe& subframe = subframes[i];
    const bool is_preframe =
        (i + 1 < subframes.size() &&
         num_consecutive_preframes < kMaxNumPreframes && random.FlipACoin());
    subframe.duration_ms = is_preframe ? 0 : random.Get(1, 100);
    subframe.dispose =
        (i == 0 || (num_consecutive_preframes == 0 && random.FlipACoin()));
    subframe.blend = random.FlipACoin();
    ASSERT_WP2_OK(subframe.rgb_pixels.Resize(width, height));
    const std::string debug_str =
        SPrintf("test%d-frame%d-duration%d-force_dispose%d-blend%d", test_id, i,
                subframe.duration_ms, subframe.dispose, subframe.blend);
    ASSERT_WP2_OK(FillRandom(
        debug_str,
        /*previous=*/(i > 0 ? &(subframes[i - 1].rgb_pixels) : nullptr),
        &random, &subframe.rgb_pixels));
    has_alpha |= subframe.rgb_pixels.HasTransparency();

    if (random.FlipACoin()) {
      subframe.window = {0, 0, width, height};
    } else {
      subframe.window = RandomRect(width, height, &random);
    }

    printf("frame %u duration %u dispose %d blend %d window %u,%u,%u,%u\n", i,
           subframe.duration_ms, subframe.dispose, subframe.blend,
           subframe.window.x, subframe.window.y, subframe.window.width,
           subframe.window.height);

    num_consecutive_preframes = is_preframe ? num_consecutive_preframes + 1 : 0;
  }

  const uint32_t rgb_bit_depth =
      (subframes.front().rgb_pixels.format() == WP2_Argb_38) ? 10 : 8;
  const Argb32b background_color = RandomColor(has_alpha, &random);
  printf("Background_color: Argb32(%d, %d, %d, %d)\n", background_color.a,
         background_color.r, background_color.g, background_color.b);

  ASSERT_WP2_OK(EncodeHeader(
      config, width, height, rgb_bit_depth, has_alpha, /*is_anim=*/true,
      /*loop_forever=*/false, ToArgb38b(background_color),
      /*preview_color=*/RGB12b{10, 0, 30}, /*has_icc=*/false,
      /*has_trailing_data=*/false, &encoded_data));

  SubframeEncoder subframe_encoder(GetQualityHint(config.quality), has_alpha,
                                   config.tile_shape);

  for (uint32_t i = 0; i < subframes.size(); ++i) {
    const bool is_last = (i == num_subframes - 1);
    ASSERT_WP2_OK(subframe_encoder.EncodeSubframe(config, subframes[i], is_last,
                                                  kProgress, &encoded_data));
  }

  // Decode.
  ASSERT_WP2_OK(InitDecodedFrame(width, height));
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_,
                       DecoderConfig::kDefault, &decoded_frame_);

  // Use the classic (non incremental) decoder to decode the first frame.
  ArgbBuffer decoded_first_frame;
  ASSERT_WP2_OK(Decode(encoded_data.mem_, encoded_data.size_,
                       &decoded_first_frame));

  ArgbBuffer expected_buffer;
  ASSERT_WP2_OK(expected_buffer.Resize(width, height));
  int final_frame_index = 0;
  for (uint32_t i = 0; i < subframes.size(); ++i) {
    const Subframe& subframe = subframes[i];
    if (subframe.dispose) expected_buffer.Fill(background_color);
    if (subframe.blend) {
      ASSERT_WP2_OK(
          expected_buffer.CompositeUnder(subframe.rgb_pixels, subframe.window));
    } else {
      ArgbBuffer view1, view2;
      ASSERT_WP2_OK(view1.SetView(expected_buffer, subframe.window));
      ASSERT_WP2_OK(view2.SetView(subframe.rgb_pixels, subframe.window));
      ASSERT_WP2_OK(view1.CopyFrom(view2));
    }

    if (subframe.duration_ms == 0) continue;

    uint32_t duration_ms = 0;
    ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
    ASSERT_TRUE(decoded_frame_.IsView() &&
                decoded_frame_.GetRow(0) == decoded_frame_internal_.GetRow(0));
    EXPECT_TRUE(testutil::Compare(
        expected_buffer, decoded_frame_, "fake data",
        testutil::GetExpectedDistortion(config.quality), PSNR, orientation))
        << "Error on final frame " << final_frame_index;

    if (final_frame_index == 0) {
      // Compare output of classic decoder (which only decodes the first frame).
      EXPECT_TRUE(testutil::Compare(decoded_frame_, decoded_first_frame,
                                    "fake data", 99.))
          << "Error on first frame";
    }

    ASSERT_WP2_OK(SaveDebug(expected_buffer, decoded_frame_, orientation,
                            test_id, final_frame_index));
    ++final_frame_index;
  }

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

INSTANTIATE_TEST_SUITE_P(
    ParameterizedAnimFrameTestInstantiationIsSlow, ParameterizedAnimFrameTest,
    ::testing::Combine(::testing::Values(Orientation::kOriginal,
                                         Orientation::k270Clockwise),
                       ::testing::Values(true, false)  // is_slow
                       ));

INSTANTIATE_TEST_SUITE_P(
    ParameterizedAnimFrameTestInstantiationRotate, ParameterizedAnimFrameTest,
    ::testing::Combine(::testing::Values(Orientation::k90Clockwise,
                                         Orientation::k180),
                       ::testing::Values(false)  // is_slow
                       ));

INSTANTIATE_TEST_SUITE_P(
    ParameterizedAnimFrameTestInstantiationFlip, ParameterizedAnimFrameTest,
    ::testing::Combine(::testing::Values(Orientation::kFlipLeftToRight,
                                         Orientation::kFlipBotLeftToTopRgt,
                                         Orientation::kFlipTopToBottom,
                                         Orientation::kFlipTopLeftToBotRgt),
                       ::testing::Values(false)  // is_slow
                       ));

}  // namespace
}  // namespace WP2
