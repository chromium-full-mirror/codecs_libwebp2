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

#include <string>

#include "include/helpers.h"
#include "src/wp2/encode.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class AltTuningTest
    : public testing::TestWithParam<std::tuple<std::string, float>> {};

TEST_P(AltTuningTest, Disable) {
  const std::string& file_name = std::get<0>(GetParam());
  EncoderConfig config = EncoderConfig::kDefault;
  config.quality = std::get<1>(GetParam());
  config.enable_alt_tuning = false;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name, config));
}

INSTANTIATE_TEST_SUITE_P(AltTuningTestInstantiation, AltTuningTest,
                         testing::Combine(testing::Values("source1_64x48.png",
                                                          "source3.jpg"),
                                          testing::Values(0.f, 75.f)  // quality
                                          ));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
