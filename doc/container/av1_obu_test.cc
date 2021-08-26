// Copyright 2021 Google LLC
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

#include "./av1_obu.h"

#include <cstring>
#include <string>

#include "./avif.h"
#include "./container.h"
#include "./testutil.h"

namespace container {
namespace {

//------------------------------------------------------------------------------

TEST(ObuTest, UnwrapObuSequenceHeader) {
  const Data avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(avif.empty());

  Image unwrapped_avif;
  size_t offset_till_obu, obu_size;
  ASSERT_TRUE(UnwrapAvif(avif.data(), avif.size(), unwrapped_avif,
                         offset_till_obu, obu_size));

  SequenceHeaderObu unwrapped_obu;
  ASSERT_TRUE(UnwrapObuSequenceHeader(avif.data() + offset_till_obu, obu_size,
                                      unwrapped_obu));
  ASSERT_EQ(unwrapped_obu.width, unwrapped_avif.width);
  ASSERT_EQ(unwrapped_obu.height, unwrapped_avif.height);
}

//------------------------------------------------------------------------------

TEST(ObuTest, UnwrapWrapUnwrap) {
  const Data avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(avif.empty());
  Image unwrapped_avif;
  size_t offset_till_obu, obu_size;
  ASSERT_TRUE(UnwrapAvif(avif.data(), avif.size(), unwrapped_avif,
                         offset_till_obu, obu_size));

  // Unwrap the OBU part that was just extracted from the AVIF bitstream.
  ObuFeatures features;
  std::memset(&features, 0, sizeof(features));  // For memcmp correctness.
  size_t offset_till_tiles_bytes, num_tiles_bytes;
  std::string log;
  ASSERT_TRUE(UnwrapObu(avif.data() + offset_till_obu, obu_size,
                        unwrapped_avif.width, unwrapped_avif.height, features,
                        offset_till_tiles_bytes, num_tiles_bytes, log))
      << log;

  Data final_obu;
  ASSERT_TRUE(WrapObu(unwrapped_avif.width, unwrapped_avif.height, features,
                      avif.data() + offset_till_tiles_bytes, num_tiles_bytes,
                      final_obu, log))
      << log;

  // Unwrap again to see if all 'features' match.
  ObuFeatures comparison_features;
  std::memset(&comparison_features, 0, sizeof(comparison_features));
  size_t final_offset_till_tiles_bytes, final_num_tiles_bytes;
  ASSERT_TRUE(UnwrapObu(final_obu.data(), final_obu.size(),
                        unwrapped_avif.width, unwrapped_avif.height,
                        comparison_features, final_offset_till_tiles_bytes,
                        final_num_tiles_bytes, log))
      << log;
  EXPECT_EQ(std::memcmp(&features, &comparison_features, sizeof(features)), 0);
  EXPECT_EQ(num_tiles_bytes, final_num_tiles_bytes);

  // Wrap again to check the byte-wise equality of the OBU parts.
  Data comparison_obu;
  ASSERT_TRUE(WrapObu(unwrapped_avif.width, unwrapped_avif.height,
                      comparison_features,
                      final_obu.data() + final_offset_till_tiles_bytes,
                      final_num_tiles_bytes, comparison_obu, log))
      << log;
  EXPECT_TRUE(
      std::equal(final_obu.begin(), final_obu.end(), comparison_obu.begin()));

  EXPECT_FALSE(log.empty());
  EXPECT_EQ(log.find("Bad"), std::string::npos);
}

//------------------------------------------------------------------------------

TEST(StrippedObuTest, UnwrapWrapUnwrap) {
  const Data avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(avif.empty());
  Image unwrapped_avif;
  size_t offset_till_obu, obu_size;
  ASSERT_TRUE(UnwrapAvif(avif.data(), avif.size(), unwrapped_avif,
                         offset_till_obu, obu_size));

  // Unwrap the OBU part that was just extracted from the AVIF bitstream.
  ObuFeatures features;
  std::memset(&features, 0, sizeof(features));  // For memcmp correctness.
  size_t offset_till_tiles_bytes, num_tiles_bytes;
  std::string log;
  ASSERT_TRUE(UnwrapObu(avif.data() + offset_till_obu, obu_size,
                        unwrapped_avif.width, unwrapped_avif.height, features,
                        offset_till_tiles_bytes, num_tiles_bytes, log))
      << log;

  Data stripped_obu;
  ASSERT_TRUE(WrapStrippedObu(unwrapped_avif.width, unwrapped_avif.height,
                              features, avif.data() + offset_till_tiles_bytes,
                              num_tiles_bytes, stripped_obu, log))
      << log;

  // Unwrap again to see if all 'features' match.
  ObuFeatures comparison_features;
  std::memset(&comparison_features, 0, sizeof(comparison_features));
  size_t final_offset_till_tiles_bytes, final_num_tiles_bytes;
  ASSERT_TRUE(UnwrapStrippedObu(
      stripped_obu.data(), stripped_obu.size(), unwrapped_avif.width,
      unwrapped_avif.height, comparison_features, final_offset_till_tiles_bytes,
      final_num_tiles_bytes, log))
      << log;
  EXPECT_EQ(std::memcmp(&features, &comparison_features, sizeof(features)), 0);
  EXPECT_LE(num_tiles_bytes, final_num_tiles_bytes);

  // Wrap again to check the byte-wise equality of the OBU parts.
  Data comparison_obu;
  ASSERT_TRUE(WrapObu(unwrapped_avif.width, unwrapped_avif.height,
                      comparison_features,
                      stripped_obu.data() + final_offset_till_tiles_bytes,
                      final_num_tiles_bytes, comparison_obu, log))
      << log;
  // Official OBU and stripped OBU bitstreams should differ.
  EXPECT_FALSE(std::equal(stripped_obu.begin(), stripped_obu.end(),
                         comparison_obu.begin()));

  EXPECT_FALSE(log.empty());
  EXPECT_EQ(log.find("Bad"), std::string::npos);
}

//------------------------------------------------------------------------------

TEST(ObuTest, ObuUnitsToStr) {
  const Data avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(avif.empty());

  Image unwrapped_avif;
  size_t offset_till_obu, obu_size;
  ASSERT_TRUE(UnwrapAvif(avif.data(), avif.size(), unwrapped_avif,
                         offset_till_obu, obu_size));

  const std::string output_str =
      ObuUnitsToStr(avif.data() + offset_till_obu, obu_size);

  EXPECT_FALSE(output_str.empty());
  EXPECT_EQ(output_str.find("Bad"), std::string::npos);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace container
