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
//  Fuzzing of imageio (still and animated image decoding).
//
// Author: Yannis Guyon (yguyon@google.com)

#include <cstdint>
#include <string>
#include <string_view>

#include "imageio/anim_image_dec.h"
#include "imageio/file_format.h"
#include "imageio/imageio_util.h"
#include "src/wp2/base.h"
#include "fuzzing/fuzztest.h"
#include "tests/fuzz/fuzz_utils.h"

namespace WP2 {
namespace testutil {
namespace {

void Read(std::string_view bytes) {
  // Input data is an encoded animation (webp, gif etc.) that may be invalid.
  // It can also be a still image (jpeg, png etc.).
  ArgbBuffer buffer;
  ImageReader image_reader(reinterpret_cast<const uint8_t*>(bytes.data()),
                           bytes.size(), &buffer, FileFormat::AUTO,
                           LogLevel::QUIET, kMaxNumPixels);
  bool is_last_frame = true;
  while (image_reader.ReadFrame(&is_last_frame) == WP2_STATUS_OK &&
         !is_last_frame) {
  }
}

// Truncate lengthy inputs.
constexpr uint32_t kMaxSize = 1024 * 1024;  // 1 MB

FUZZ_TEST(ReadAnim, Read)
    .WithDomains(fuzztest::String().WithMaxSize(kMaxSize).WithDictionary(
        ReadDictionaryFromFiles({GetTestAssetPath("fuzz/common_tags.dict")})))
    .WithSeeds(ReadFilesFromDirectory(GetTestAssetPath("testdata")));

}  // namespace
}  // namespace testutil
}  // namespace WP2
