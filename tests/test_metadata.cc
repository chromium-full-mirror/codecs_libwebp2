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

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Encode decode metadata of 1 byte per ICC, EXIF and XMP chunk.
TEST(MetadataTest, MinChunkSize) {
  ArgbBuffer original, decoded;
  ASSERT_WP2_OK(ReadImage(testing::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  Metadata& a = original.metadata_;
  ASSERT_WP2_OK(a.iccp.Resize(1, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.xmp.Resize(1, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.exif.Resize(1, /*keep_bytes=*/false));
  std::fill(a.iccp.bytes, a.iccp.bytes + a.iccp.size, 42);
  std::fill(a.xmp.bytes, a.xmp.bytes + a.xmp.size, 44);
  std::fill(a.exif.bytes, a.exif.bytes + a.exif.size, 43);

  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(original, &memory_writer));
  ASSERT_WP2_OK(Decode(memory_writer.mem_, memory_writer.size_, &decoded));
  const Metadata& b = decoded.metadata_;

  EXPECT_TRUE(testing::HasSameData(a.iccp, b.iccp));
  EXPECT_TRUE(testing::HasSameData(a.xmp, b.xmp));
  EXPECT_TRUE(testing::HasSameData(a.exif, b.exif));
}

//------------------------------------------------------------------------------

// Encode decode metadata of maximum size per ICC, EXIF and XMP chunk.
TEST(MetadataTest, MaxChunkSize) {
  ArgbBuffer original, decoded;
  ASSERT_WP2_OK(ReadImage(testing::GetTestDataPath("source1_1x1.png").c_str(),
                          &original));

  Metadata& a = original.metadata_;
  // ICC needs to reserve 1 byte for the ICC type. Hence the '-1':
  ASSERT_WP2_OK(a.iccp.Resize(kMaxChunkSize - 1, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.xmp.Resize(kMaxChunkSize, /*keep_bytes=*/false));
  ASSERT_WP2_OK(a.exif.Resize(kMaxChunkSize, /*keep_bytes=*/false));
  std::fill(a.iccp.bytes, a.iccp.bytes + a.iccp.size, 42);
  std::fill(a.xmp.bytes, a.xmp.bytes + a.xmp.size, 44);
  std::fill(a.exif.bytes, a.exif.bytes + a.exif.size, 43);

  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(original, &memory_writer));
  ASSERT_WP2_OK(Decode(memory_writer.mem_, memory_writer.size_, &decoded));
  const Metadata& b = decoded.metadata_;

  EXPECT_TRUE(testing::HasSameData(a.iccp, b.iccp));
  EXPECT_TRUE(testing::HasSameData(a.xmp, b.xmp));
  EXPECT_TRUE(testing::HasSameData(a.exif, b.exif));
}

//------------------------------------------------------------------------------

// Make sure metadata of forbidden size fails the encoding.
TEST(MetadataTest, MaxChunkSizePlusOne) {
  ArgbBuffer original;
  ASSERT_WP2_OK(ReadImage(testing::GetTestDataPath("source1_1x1.png").c_str(),
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

}  // namespace
}  // namespace WP2
