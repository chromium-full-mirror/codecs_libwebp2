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
//  Fuzzing of imageio (still image encoding).
//
// Author: Yannis Guyon (yguyon@google.com)

#include <cstddef>

#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/wp2/base.h"
#include "gtest/gtest.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/fuzz/fuzz_utils_temp_file.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {
namespace {

::testing::Environment* const kStackLimitEnv = SetStackLimitTo256KiB();

void Save(const ArgbBuffer& image, FileFormat file_format) {
  FuzzerTemporaryFile dst;

  // Write the random buffer in a randomly chosen valid format.
  size_t file_size;
  ASSERT_EQ(
      SaveImage(image, dst.path(), /*overwrite=*/true, file_format, &file_size),
      WP2_STATUS_OK);

  // PGM is written as 16 bits but read at 8 so it's disabled. BMP is unhandled.
  if (file_format != FileFormat::PGM && file_format != FileFormat::BMP) {
    // Read the freshly written file to ensure it's valid.
    ArgbBuffer buffer(image.format());
    EXPECT_EQ(ReadImage(dst.path(), &buffer, /*file_size=*/nullptr, file_format,
                        LogLevel::QUIET),
              WP2_STATUS_OK);
  }
}

FUZZ_TEST(SaveImage, Save)
    .WithDomains(ArbitraryArgbBuffer8b(),
                 fuzztest::ElementOf({FileFormat::PNG, FileFormat::TIFF,
                                      FileFormat::BMP, FileFormat::PAM,
                                      FileFormat::PGM, FileFormat::PPM}));

}  // namespace
}  // namespace testutil
}  // namespace WP2
