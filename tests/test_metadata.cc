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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Encode decode metadata of minimum size per ICC, EXIF and XMP chunk.
TEST(MetadataTest, MinChunkSize) {
  ArgbBuffer original, decoded;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  Metadata& a = original.metadata_;
  ASSERT_WP2_OK(a.iccp.Resize(2, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.xmp.Resize(1, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.exif.Resize(1, /*keep_bytes=*/false));
  std::fill(a.iccp.bytes, a.iccp.bytes + a.iccp.size, 42);
  std::fill(a.xmp.bytes, a.xmp.bytes + a.xmp.size, 44);
  std::fill(a.exif.bytes, a.exif.bytes + a.exif.size, 43);

  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(original, &memory_writer));
  ASSERT_WP2_OK(Decode(memory_writer.mem_, memory_writer.size_, &decoded));
  const Metadata& b = decoded.metadata_;

  EXPECT_TRUE(testutil::HasSameData(a.iccp, b.iccp));
  EXPECT_TRUE(testutil::HasSameData(a.xmp, b.xmp));
  EXPECT_TRUE(testutil::HasSameData(a.exif, b.exif));
}

TEST(MetadataTest, IccIsTooSmall) {
  ArgbBuffer original;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  Metadata& a = original.metadata_;
  ASSERT_WP2_OK(a.iccp.Resize(1, /*keep_bytes=*/false));
  std::fill(a.iccp.bytes, a.iccp.bytes + a.iccp.size, 42);

  MemoryWriter memory_writer;
  EXPECT_EQ(Encode(original, &memory_writer), WP2_STATUS_INVALID_PARAMETER);
}

//------------------------------------------------------------------------------

// Encode decode metadata of maximum size per ICC, EXIF and XMP chunk.
TEST(MetadataTest, MaxChunkSize) {
  ArgbBuffer original, decoded;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  Metadata& a = original.metadata_;
  ASSERT_WP2_OK(a.iccp.Resize(kMaxChunkSize, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.xmp.Resize(kMaxChunkSize, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.exif.Resize(kMaxChunkSize, /*keep_bytes=*/false));
  std::fill(a.iccp.bytes, a.iccp.bytes + a.iccp.size, 42);
  std::fill(a.xmp.bytes, a.xmp.bytes + a.xmp.size, 44);
  std::fill(a.exif.bytes, a.exif.bytes + a.exif.size, 43);

  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(original, &memory_writer));
  ASSERT_WP2_OK(Decode(memory_writer.mem_, memory_writer.size_, &decoded));
  const Metadata& b = decoded.metadata_;

  EXPECT_TRUE(testutil::HasSameData(a.iccp, b.iccp));
  EXPECT_TRUE(testutil::HasSameData(a.xmp, b.xmp));
  EXPECT_TRUE(testutil::HasSameData(a.exif, b.exif));
}

//------------------------------------------------------------------------------

// Make sure metadata of forbidden size fails the encoding.
TEST(MetadataTest, MaxChunkSizePlusOne) {
  ArgbBuffer original;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  ASSERT_WP2_OK(
      original.metadata_.iccp.Resize(kMaxChunkSize + 1, /*keep_bytes=*/false));

  MemoryWriter memory_writer;
  ASSERT_EQ(Encode(original, &memory_writer), WP2_STATUS_INVALID_PARAMETER);

  swap(original.metadata_.iccp, original.metadata_.xmp);
  ASSERT_EQ(Encode(original, &memory_writer), WP2_STATUS_INVALID_PARAMETER);

  swap(original.metadata_.xmp, original.metadata_.exif);
  ASSERT_EQ(Encode(original, &memory_writer), WP2_STATUS_INVALID_PARAMETER);
}

//------------------------------------------------------------------------------
// Exercise GetChunk() and GetNumFrames().

TEST(MetadataTest, GetChunk) {
  // Encode some image into 'data'. Chunks will be retrieved from that.
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage("test_exif_xmp.webp", &data));

  Counter header;
  ASSERT_WP2_OK(GetChunk(data.mem_, data.size_, ChunkType::kHeader, &header));
  EXPECT_GE(header.size_, kHeaderMinSize);  // Found.
  EXPECT_LE(header.size_, kHeaderMaxSize);

  for (ChunkType type : {ChunkType::kPreview, ChunkType::kIcc}) {
    Counter chunk;
    ASSERT_WP2_OK(GetChunk(data.mem_, data.size_, type, &chunk));
    EXPECT_EQ(chunk.size_, 0u);  // Not found.
  }

  Counter first_frame;
  ASSERT_WP2_OK(
      GetChunk(data.mem_, data.size_, ChunkType::kFrame, &first_frame));
  EXPECT_GT(first_frame.size_, 0u);  // Found.
  Counter second_frame;
  ASSERT_WP2_OK(
      GetChunk(data.mem_, data.size_, ChunkType::kFrame, &second_frame));
  EXPECT_GE(second_frame.size_, 0u);  // Not found, only one frame.

  for (ChunkType type : {ChunkType::kXmp, ChunkType::kExif}) {
    Counter chunk;
    ASSERT_WP2_OK(GetChunk(data.mem_, data.size_, type, &chunk));
    EXPECT_GT(chunk.size_, 0u);  // Found.
    EXPECT_LE(chunk.size_, kMaxChunkSize);
  }
}

TEST(MetadataTest, GetChunkTruncated) {
  // Encode some image into 'data'.
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage("test_exif_xmp.webp", &data));

  // A few bytes are missing.
  Counter chunk;
  ASSERT_EQ(GetChunk(data.mem_, data.size_ - 5, ChunkType::kExif, &chunk),
            WP2_STATUS_NOT_ENOUGH_DATA);
  // XMP is before EXIF in the bitstream and thus entirely available.
  ASSERT_WP2_OK(GetChunk(data.mem_, data.size_ - 5, ChunkType::kXmp, &chunk));
}

TEST(MetadataTest, GetChunkFailure) {
  MemoryWriter writer;
  ASSERT_EQ(GetChunk(/*bitstream_data=*/nullptr, /*bitstream_size=*/123,
                     ChunkType::kHeader, &writer),
            WP2_STATUS_NULL_PARAMETER);

  constexpr uint8_t bitstream[123] = {0};
  ASSERT_EQ(GetChunk(bitstream, 123, ChunkType::kHeader, /*writer=*/nullptr),
            WP2_STATUS_NULL_PARAMETER);

  ASSERT_EQ(GetChunk(bitstream, 123, ChunkType::kHeader, &writer),
            WP2_STATUS_BITSTREAM_ERROR);
}

TEST(MetadataTest, GetNumFrames) {
  // Encode some still image into 'data'.
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage("test_exif_xmp.webp", &data));

  // This is not an animation, there should be only one frame.
  size_t num_frames;
  ASSERT_WP2_OK(GetNumFrames(data.mem_, data.size_, &num_frames));
  ASSERT_EQ(num_frames, 1u);
}

TEST(MetadataTest, GetNumFramesFailure) {
  size_t num_frames;
  ASSERT_EQ(GetNumFrames(/*bitstream_data=*/nullptr, /*bitstream_size=*/123,
                         &num_frames),
            WP2_STATUS_NULL_PARAMETER);

  constexpr uint8_t bitstream[123] = {0};
  ASSERT_EQ(GetNumFrames(bitstream, 123, /*num_frames=*/nullptr),
            WP2_STATUS_NULL_PARAMETER);

  ASSERT_EQ(GetNumFrames(bitstream, 123, &num_frames),
            WP2_STATUS_BITSTREAM_ERROR);
}

TEST(MetadataTest, GetNumFramesAndGetChunkForAnimation) {
  // Encode some animated image into 'data'.
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressAnimation(
      {"source0.pgm", "source1.png", "source3.jpg"}, {1, 1, 1}, &data, nullptr,
      EncoderConfig::kDefault.quality, EncoderConfig::kDefault.effort,
      EncoderConfig::kDefault.thread_level, /*num_downsamplings=*/2));

  // This is an animation, there should be more than one frame.
  size_t num_frames;
  ASSERT_WP2_OK(GetNumFrames(data.mem_, data.size_, &num_frames));
  ASSERT_EQ(num_frames, 3u);

  // Make sure each frame chunk can be retrieved.
  for (size_t i = 0; i < num_frames; ++i) {
    Counter counter;
    ASSERT_WP2_OK(GetChunk(data.mem_, data.size_, ChunkType::kFrame, &counter,
                           /*frame_index=*/i));
    EXPECT_GT(counter.size_, 0u);
  }

  // Out-of-range frames are not written, but there is no error.
  MemoryWriter writer;
  ASSERT_WP2_OK(GetChunk(data.mem_, data.size_, ChunkType::kFrame, &writer,
                         /*frame_index=*/num_frames));
  EXPECT_EQ(writer.max_size_, 0u);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
