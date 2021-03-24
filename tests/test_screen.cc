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

// Test screen content encoding.

#include <cstdio>
#include <string>
#include <tuple>

#include "examples/example_utils.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "include/helpers.h"
#include "src/dec/preview/preview_dec.h"
#include "src/enc/preview/preview_enc.h"
#include "src/enc/screen_content/screen_enc.h"
#include "src/utils/data_source.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

TEST(ScreenEnc, Basic) {
  ArgbBuffer input;
  ASSERT_WP2_OK(ReadImage(
      testutil::GetTestDataPath("source1_64x48.png").c_str(), &input));

  const uint32_t block_size = 16;
  const uint32_t block_height = DivCeil(input.height(), block_size);
  const uint32_t block_width = DivCeil(input.width(), block_size);

  Plane<bool> is_lossy;
  ASSERT_WP2_OK(is_lossy.Resize(block_width, block_height));
  is_lossy.Fill(false);
  is_lossy.Row(1)[1] = is_lossy.Row(1)[2] = is_lossy.Row(2)[3] = true;

  MemoryWriter writer;
  EncoderConfig config;
  EXPECT_WP2_OK(
      EncodeScreenContent(input, is_lossy, block_size, &writer, config));

  ArgbBuffer output;
  EXPECT_WP2_OK(Decode(writer.mem_, writer.size_, &output));

  constexpr float kMinExpectedSimilarity = 35.f;
  ASSERT_TRUE(testutil::Compare(input, output, "source1_64x48.png",
                                kMinExpectedSimilarity));
}

}  // namespace
}  // namespace WP2
