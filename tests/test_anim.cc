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
// -----------------------------------------------------------------------------

// Animation encoding and decoding test.

#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "include/helpers.h"
#include "include/helpers_filter.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

typedef std::tuple<std::vector<const char*>, std::vector<uint32_t>, float, int,
                   uint32_t>
    Param;

class AnimTest : public testing::TestWithParam<Param> {};

// Test AnimationEncoder with Decoder::Update().
TEST_P(AnimTest, Simple) {
  const std::vector<const char*>& file_names = std::get<0>(GetParam());
  const std::vector<uint32_t>& durations_ms = std::get<1>(GetParam());
  const float quality = std::get<2>(GetParam());
  const int effort = std::get<3>(GetParam());
  const uint32_t thread_level = std::get<4>(GetParam());

  std::vector<ArgbBuffer> frames;
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(testutil::CompressAnimation(file_names, durations_ms,
                                            &encoded_data, &frames, quality,
                                            effort, thread_level));

  // Decode.
  DecoderConfig decoder_config;
  decoder_config.thread_level = thread_level;
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_, decoder_config);

  for (size_t i = 0; i < frames.size(); ++i) {
    // ReadFrame() halts once for each frame.
    uint32_t duration_ms;
    ASSERT_TRUE(decoder.ReadFrame(&duration_ms));
    EXPECT_EQ(duration_ms, durations_ms[i]);
    EXPECT_TRUE(testutil::Compare(decoder.GetPixels(), frames[i], file_names[i],
                                  testutil::GetExpectedDistortion(quality)));
    if (i + 1 < frames.size() ||
        decoder.TryGetDecodedFeatures()->has_trailing_data) {
      // Not the last frame, or last frame but waiting for metadata.
      EXPECT_EQ(decoder.GetStatus(), WP2_STATUS_NOT_ENOUGH_DATA);
    } else {
      EXPECT_EQ(decoder.GetStatus(), WP2_STATUS_OK);
    }
  }

  // Then ReadFrame() returns false when all data is decoded.
  EXPECT_FALSE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());
}

// Test AnimationEncoder with Decode() function: only first frame is decoded.
TEST_P(AnimTest, FirstFrame) {
  const std::vector<const char*>& file_names = std::get<0>(GetParam());
  const std::vector<uint32_t>& durations_ms = std::get<1>(GetParam());
  const float quality = std::get<2>(GetParam());
  const int effort = std::get<3>(GetParam());
  const uint32_t thread_level = std::get<4>(GetParam());

  std::vector<ArgbBuffer> frames;
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(testutil::CompressAnimation(file_names, durations_ms,
                                            &encoded_data, &frames, quality,
                                            effort, thread_level));

  // Decode.
  ArgbBuffer decoded_frame;
  DecoderConfig decoder_config;
  decoder_config.thread_level = thread_level;
  ASSERT_WP2_OK(Decode(encoded_data.mem_, encoded_data.size_, &decoded_frame,
                       decoder_config));
  EXPECT_TRUE(testutil::Compare(decoded_frame, frames[0], file_names[0],
                                testutil::GetExpectedDistortion(quality)));
}

// Input images are expected to be of equal sizes inside a Param.
INSTANTIATE_TEST_SUITE_P(
    AnimTestInstantiation, AnimTest,
    testing::Values(
        Param({"source1_64x48.png"},
              /*frame durations (ms):*/ {1000},
              /*quality=*/0.0f, /*effort=*/1, /*thread_level=*/0),
        Param({"alpha_ramp.png", "alpha_ramp.lossy.webp", "alpha_ramp.webp"},
              /*frame durations (ms):*/ {50, 70, 1},
              /*quality=*/100.0f, /*effort=*/5, /*thread_level=*/0)));

// Disabled because it takes too much time.
INSTANTIATE_TEST_SUITE_P(
    DISABLED_AnimTestInstantiation, AnimTest,
    testing::Values(
        Param({"source0.pgm", "source1.png", "source2.tiff", "source3.jpg"},
              /*frame durations (ms):*/ {kMaxFrameDurationMs, 1, 20, 3},
              /*quality=*/75.0f, /*effort=*/3, /*thread_level=*/1)));

//------------------------------------------------------------------------------

WP2Status Fill(ArgbBuffer* const buffer, uint32_t width, uint32_t height) {
  WP2_CHECK_STATUS(buffer->Resize(width, height));
  for (uint32_t y = 0; y < height; ++y) {
    memset(buffer->GetRow(y), 127, buffer->stride());
  }
  return WP2_STATUS_OK;
}

TEST(AnimTest, Fail) {
  AnimationEncoder animation_encoder;
  ArgbBuffer src;
  MemoryWriter encoded_data;
  static constexpr uint32_t ms = 1;  // Frame duration.

  // Empty encoder or frame.
  ASSERT_EQ(animation_encoder.Encode(&encoded_data),
            WP2_STATUS_INVALID_PARAMETER);
  ASSERT_EQ(animation_encoder.AddFrame(src, ms), WP2_STATUS_NULL_PARAMETER);

  // Badly sized frames.
  ASSERT_WP2_OK(Fill(&src, 1, kImageDimMax + 1));
  ASSERT_EQ(animation_encoder.AddFrame(src, ms), WP2_STATUS_BAD_DIMENSION);
  ASSERT_WP2_OK(Fill(&src, kImageDimMax + 1, 1));
  ASSERT_EQ(animation_encoder.AddFrame(src, ms), WP2_STATUS_BAD_DIMENSION);

  // OK.
  ASSERT_WP2_OK(Fill(&src, 128, 512));
  ASSERT_WP2_OK(animation_encoder.AddFrame(src, ms));
  ASSERT_WP2_OK(animation_encoder.Encode(&encoded_data));

  // Different dimension than first frame.
  ASSERT_WP2_OK(Fill(&src, 127, 512));
  ASSERT_EQ(animation_encoder.AddFrame(src, ms), WP2_STATUS_BAD_DIMENSION);

  ASSERT_WP2_OK(Fill(&src, 128, 512));
  // User-defined preframes are not allowed.
  ASSERT_EQ(animation_encoder.AddFrame(src, 0), WP2_STATUS_INVALID_PARAMETER);
  // Frame too long.
  ASSERT_EQ(animation_encoder.AddFrame(src, kMaxFrameDurationMs + 1),
            WP2_STATUS_INVALID_PARAMETER);

  ASSERT_EQ(animation_encoder.Encode(nullptr), WP2_STATUS_NULL_PARAMETER);
  ASSERT_WP2_OK(animation_encoder.Encode(&encoded_data));
}

//------------------------------------------------------------------------------

Metadata CreateMetadata(const char* const xmp, const char* const exif) {
  Metadata metadata;
  if (xmp != nullptr) {
    WP2_ASSERT_STATUS(
        metadata.xmp.CopyFrom((const uint8_t*)xmp, std::strlen(xmp)));
  }
  if (exif != nullptr) {
    WP2_ASSERT_STATUS(
        metadata.exif.CopyFrom((const uint8_t*)exif, std::strlen(exif)));
  }
  return metadata;
}

TEST(AnimTest, Metadata) {
  const Metadata metadata_good(CreateMetadata(nullptr, "exif"));
  const Metadata metadata_bad1(CreateMetadata("bAd", "BaD"));
  const Metadata metadata_bad2(CreateMetadata("spamely", "spam spam"));
  const Metadata metadata_bad3(CreateMetadata("xmp", nullptr));
  ArgbBuffer decoded_image;
  ASSERT_WP2_OK(decoded_image.metadata_.CopyFrom(metadata_bad3));

  AnimationEncoder animation_encoder;
  {
    ArgbBuffer blank_image;
    ASSERT_WP2_OK(Fill(&blank_image, 128, 128));

    // Add frames with various metadata.
    ASSERT_WP2_OK(blank_image.metadata_.CopyFrom(metadata_bad1));
    ASSERT_WP2_OK(animation_encoder.AddFrame(blank_image, 1));
    blank_image.metadata_.Clear();
    ASSERT_WP2_OK(animation_encoder.AddFrame(blank_image, 1));
    ASSERT_WP2_OK(blank_image.metadata_.CopyFrom(metadata_bad2));
    ASSERT_WP2_OK(animation_encoder.AddFrame(blank_image, 1));
  }

  {
    // Encode them with 'metadata_good'.
    MemoryWriter encoded_data;
    ASSERT_WP2_OK(
        animation_encoder.Encode(&encoded_data, EncoderConfig::kDefault,
                                 /*loop_forever=*/true, metadata_good));

    // Decode and expect to find 'metadata_good' and no 'metadata_bad*'.
    ASSERT_WP2_OK(
        Decode(encoded_data.mem_, encoded_data.size_, &decoded_image));
  }

  EXPECT_TRUE(
      testutil::HasSameData(decoded_image.metadata_.exif, metadata_good.exif));

  EXPECT_EQ(decoded_image.metadata_.xmp.size, metadata_good.xmp.size);
  EXPECT_EQ(decoded_image.metadata_.xmp.bytes, nullptr);
}

//------------------------------------------------------------------------------

// Test preframes.
TEST(AnimTest, Preframes) {
  std::vector<ArgbBuffer> frames(2);
  ArgbBuffer& first_frame = frames[0];
  ArgbBuffer& second_frame = frames[1];

  // First frame with random noise, hard to compress especially in lossless.
  // This way preframes should be way smaller than entire frames and should be
  // selected as the best option.
  ASSERT_WP2_OK(first_frame.Resize(512, 512));
  first_frame.Fill(Argb32b{128, 128, 128, 128});
  testutil::Noise(/*seed=*/42, /*strength=*/128, &first_frame);

  // Copy of first frame except for two areas that are far from each other
  // (top-left and bottom-right), to force the generation of one preframe.
  ASSERT_WP2_OK(second_frame.CopyFrom(first_frame));
  const uint32_t area_size = 8;
  ArgbBuffer top_left, bottom_right;
  ASSERT_WP2_OK(top_left.SetView(first_frame, {0, 0, area_size, area_size}));
  ASSERT_WP2_OK(bottom_right.SetView(
      first_frame, {first_frame.width() - area_size,
                    first_frame.height() - area_size, area_size, area_size}));
  testutil::Noise(/*seed=*/12, /*strength=*/255, &top_left);
  testutil::Noise(/*seed=*/0, /*strength=*/255, &bottom_right);

  // Encode losslessly.
  EncoderConfig config;
  config.quality = 100.f;
  config.effort = 2;
  config.thread_level = 8;
  MemoryWriter encoded_data;
  AnimationEncoder animation_encoder;
  ASSERT_WP2_OK(animation_encoder.AddFrame(first_frame, /*duration_ms=*/50u));
  ASSERT_WP2_OK(animation_encoder.AddFrame(second_frame, /*duration_ms=*/50u));
  ASSERT_WP2_OK(animation_encoder.Encode(&encoded_data, config));

  // Decode features, verify that no preframe can be seen from user side.
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_);
  decoder.SkipNumNextFrames(kMaxNumFrames);
  ASSERT_FALSE(decoder.ReadFrame());
  ASSERT_FALSE(decoder.Failed());
  ASSERT_EQ(decoder.GetNumFrameDecodedFeatures(), 2u);
  const FrameFeatures* first_features = decoder.TryGetFrameDecodedFeatures(0);
  const FrameFeatures* second_features = decoder.TryGetFrameDecodedFeatures(1);
  ASSERT_TRUE(first_features != nullptr && second_features != nullptr);
  ASSERT_EQ(first_features->window.x, second_features->window.x);
  ASSERT_EQ(first_features->window.y, second_features->window.y);
  ASSERT_EQ(first_features->window.width, second_features->window.width);
  ASSERT_EQ(first_features->window.height, second_features->window.height);

  // Check pixels just in case.
  decoder.Rewind();
  for (const ArgbBuffer& frame : frames) {
    ASSERT_TRUE(decoder.ReadFrame());
    ASSERT_FALSE(decoder.Failed());
    EXPECT_TRUE(testutil::Compare(decoder.GetPixels(), frame, "preframe",
                                  testutil::GetExpectedDistortion(config)));
  }
  ASSERT_WP2_OK(decoder.GetStatus());
}

//------------------------------------------------------------------------------

TEST(AnimTest, NoAnimButFrameData) {
  ArgbBuffer src;
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(testutil::CompressImage("source1.png", &encoded_data, &src));

  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_);
  ASSERT_TRUE(decoder.ReadFrame());
  ASSERT_WP2_OK(decoder.GetStatus());

  ASSERT_EQ(decoder.GetNumFrameDecodedFeatures(), 1u);
  const FrameFeatures* const features = decoder.TryGetFrameDecodedFeatures(0);
  ASSERT_NE(features, nullptr);
  ASSERT_EQ(features->duration_ms, kMaxFrameDurationMs);
  ASSERT_EQ(features->window, Rectangle(0, 0, src.width(), src.height()));
  ASSERT_TRUE(features->is_last);
  ASSERT_EQ(features->last_dispose_frame_index, 0u);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
