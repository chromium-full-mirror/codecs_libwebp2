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

#include "./container.h"

#include "./testutil.h"

//------------------------------------------------------------------------------

namespace container {
namespace {

constexpr uint32_t kContainerCodecRaw = 0x5;

TEST(ContainerTest, WrapUnwrapRawPixels) {
  const Image input = testutil::GenerateRandomImage();

  // Encode.
  Data codec_bytes, bitstream;
  EncodeRawPixels(input, codec_bytes);
  WrapContainer(input, container::EncoderConfig(), kContainerCodecRaw,
                codec_bytes.data(), codec_bytes.size(), bitstream);

  // Decode.
  Image output;
  uint32_t codec;
  size_t offset_till_codec_bytes, num_codec_bytes;
  ASSERT_TRUE(UnwrapContainer(bitstream.data(), bitstream.size(), output, codec,
                              offset_till_codec_bytes, num_codec_bytes));
  ASSERT_EQ(codec, kContainerCodecRaw);

  ASSERT_TRUE(DecodeRawPixels(bitstream.data() + offset_till_codec_bytes,
                              num_codec_bytes, output));

  ASSERT_NO_FATAL_FAILURE(testutil::VerifyEqual(input, output));
}

}  // namespace
}  // namespace container
