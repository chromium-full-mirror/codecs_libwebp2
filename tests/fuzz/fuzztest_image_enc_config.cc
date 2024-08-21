// Copyright 2019 Google LLC
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
//
//  Fuzzing of still image wp2 encoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include <string>

#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/imageio_util.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {
namespace {

::testing::Environment* const kStackLimitEnv = SetStackLimitTo256KiB();

//------------------------------------------------------------------------------

void ReadEncodeDecode(const std::string& file_name, EncoderConfig config,
                      double progress_fail_at) {
  const Data data = ReadTestImage(file_name);
  // Read a randomly chosen valid image file.
  ArgbBuffer original(config.keep_unmultiplied ? WP2_ARGB_32 : WP2_Argb_32);
  ASSERT_EQ(ReadImage(data.bytes, data.size, &original, FileFormat::AUTO,
                      LogLevel::QUIET),
            WP2_STATUS_OK);

  // Make sure the encoding stops before the fuzzer timeout.
  ProgressTimeout progress(kEncodingMaxNumSeconds);
  config.progress_hook = &progress;
  progress.fail_at_ = progress_fail_at;

  const float expected_distortion = GetExpectedDistortion(config) - 11.f;
  EXPECT_EQ(TestImageEncConfig(original, config, expected_distortion), 0);
}

FUZZ_TEST(ReadEncodeDecodeImage, ReadEncodeDecode)
    .WithDomains(fuzztest::ElementOf(GetTestImageNames()),
                 ArbitraryEncoderConfig(), ArbitraryProgressFailAt());

}  // namespace
}  // namespace testutil
}  // namespace WP2
