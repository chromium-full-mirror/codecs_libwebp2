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

#include <string>

#include "./avif.h"
#include "./container.h"
#include "./testutil.h"

namespace container {
namespace {

Data GetAv1Obu(const char avif_file_name[]) {
  const Data input_avif = testutil::LoadFile(avif_file_name);
  Image unwrapped_input_avif;
  size_t offset_till_input_obu, input_obu_size;
  if (UnwrapAvif(input_avif.data(), input_avif.size(), unwrapped_input_avif,
                 offset_till_input_obu, input_obu_size)) {
    return Data(input_avif.data() + offset_till_input_obu,
                input_avif.data() + offset_till_input_obu + input_obu_size);
  }
  return Data();
}

//------------------------------------------------------------------------------

TEST(ObuTest, UnwrapOBUSequenceHeader) {
  const Data av1_obu = GetAv1Obu("1x1.avif");
  ASSERT_FALSE(av1_obu.empty());

  SequenceHeader header;
  ASSERT_TRUE(UnwrapOBUSequenceHeader(av1_obu.data(), av1_obu.size(), header));
}

//------------------------------------------------------------------------------

TEST(ObuTest, ObuUnitsToStr) {
  const Data av1_obu = GetAv1Obu("1x1.avif");
  ASSERT_FALSE(av1_obu.empty());

  const std::string output_str = ObuUnitsToStr(av1_obu.data(), av1_obu.size());

  EXPECT_FALSE(output_str.empty());
  EXPECT_EQ(output_str.find("Bad"), std::string::npos);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace container
