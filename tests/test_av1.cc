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
//
// AV1-in-WP2 test

#include <string>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

TEST(Av1Test, Simple) {
  ArgbBuffer original;
  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath("source3.jpg").c_str(), &original));

  EncoderConfig encoder_config = EncoderConfig::kDefault;
#if defined(WP2_HAVE_AOM)
  encoder_config.use_av1 = true;
#endif
  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(original, &memory_writer, encoder_config));

  ArgbBuffer decompressed;
  ASSERT_WP2_OK(Decode(memory_writer.mem_, memory_writer.size_, &decompressed));
  ASSERT_TRUE(testutil::Compare(original, decompressed, "internal",
                                /*expected_distortion=*/40, PSNR));
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
