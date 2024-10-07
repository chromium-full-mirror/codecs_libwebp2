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
//  Fuzzing of imageio (still image decoding and encoding).
//
// Author: Yannis Guyon (yguyon@google.com)

#include <string>

#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/wp2/base.h"
#include "gtest/gtest.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/fuzz/fuzz_utils_temp_file.h"

namespace WP2 {
namespace testutil {
namespace {

void ReadSave(const std::string& file_name, FileFormat file_format) {
  const Data data = ReadTestImage(file_name);
  FuzzerTemporaryFile dst;

  // Read a randomly chosen valid image file.
  ArgbBuffer buffer;
  ASSERT_EQ(ReadImage(data.bytes, data.size, &buffer, FileFormat::AUTO,
                      LogLevel::QUIET),
            WP2_STATUS_OK);

  // Write it in a randomly chosen valid format.
  ASSERT_EQ(SaveImage(buffer, dst.path(), /*overwrite=*/true, file_format),
            WP2_STATUS_OK);

  // PGM is written as 16 bits but read at 8 so is disabled. BMP is unhandled.
  if (file_format != FileFormat::PGM && file_format != FileFormat::BMP) {
    // Read the freshly written file to ensure it is valid.
    EXPECT_EQ(ReadImage(dst.path(), &buffer, /*file_size=*/nullptr, file_format,
                        LogLevel::QUIET),
              WP2_STATUS_OK);
  }
}

FUZZ_TEST(ReadSaveImage, ReadSave)
    .WithDomains(fuzztest::ElementOf(GetTestImageNames()),
                 fuzztest::ElementOf({FileFormat::PNG, FileFormat::TIFF,
                                      FileFormat::BMP, FileFormat::PAM,
                                      FileFormat::PGM, FileFormat::PPM}));

}  // namespace
}  // namespace testutil
}  // namespace WP2
