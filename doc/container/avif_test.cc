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

#include "./avif.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "./container.h"
#include "./testutil.h"

using testing::AnyOf;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Not;

namespace container {
namespace {

//------------------------------------------------------------------------------

TEST(ContainerTest, WrapUnwrapRawPixels) {
  const Data input_avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(input_avif.empty());

  // Decode.
  Image unwrapped_input_avif;
  size_t offset_till_input_obu, input_obu_size;
  ASSERT_TRUE(UnwrapAvif(input_avif.data(), input_avif.size(),
                         unwrapped_input_avif, offset_till_input_obu,
                         input_obu_size));

  // Encode.
  Data output_avif;
  ASSERT_TRUE(WrapAvif(unwrapped_input_avif,
                       input_avif.data() + offset_till_input_obu,
                       input_obu_size, output_avif));

  // Decode again.
  Image unwrapped_output_avif;
  size_t offset_till_output_obu, output_obu_size;
  ASSERT_TRUE(UnwrapAvif(output_avif.data(), output_avif.size(),
                         unwrapped_output_avif, offset_till_output_obu,
                         output_obu_size));

  // Compare image features.
  ASSERT_NO_FATAL_FAILURE(
      testutil::VerifyEqual(unwrapped_input_avif, unwrapped_output_avif));

  // Compare the untouched AV1-OBU chunks.
  ASSERT_EQ(input_obu_size, output_obu_size);
  EXPECT_TRUE(
      std::equal(input_avif.data() + offset_till_input_obu,
                 input_avif.data() + offset_till_input_obu + input_obu_size,
                 output_avif.data() + offset_till_output_obu));
}

//------------------------------------------------------------------------------

TEST(ContainerTest, BmffBoxesToStr) {
  const Data input_avif = testutil::LoadFile("1x1.avif");
  ASSERT_FALSE(input_avif.empty());

  CaptureTestStdout();
  BmffBoxesToStr(input_avif.data(), input_avif.size());

  EXPECT_THAT(GetCapturedTestStdout(), Not(AnyOf(IsEmpty(), HasSubstr("Bad"))));
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace container
