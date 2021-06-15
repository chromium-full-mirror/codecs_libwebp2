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

class DebugOptionsTest
    : public testing::TestWithParam<std::tuple<std::string, float>> {
 public:
  void SetUp() override {
    file_name_ = std::get<0>(GetParam());
    config_.quality = std::get<1>(GetParam());
    config_.info = &encoder_info_;
  }

  std::string file_name_;
  EncoderConfig config_ = EncoderConfig::kDefault;
  EncoderInfo encoder_info_;
};

TEST_P(DebugOptionsTest, DisableAltTuning) {
  config_.enable_alt_tuning = false;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name_, config_));
}

TEST_P(DebugOptionsTest, DisablePreds) {
  encoder_info_.disable_preds = true;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name_, config_));
}

TEST_P(DebugOptionsTest, DisableTransforms) {
  encoder_info_.disable_transforms = true;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name_, config_));
}

TEST_P(DebugOptionsTest, DisableSplitTf) {
  encoder_info_.disable_split_tf = true;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name_, config_));
}

INSTANTIATE_TEST_SUITE_P(DebugOptionsTestInstantiation, DebugOptionsTest,
                         testing::Combine(testing::Values("source1_64x48.png",
                                                          "source3.jpg"),
                                          testing::Values(0.f, 75.f)  // quality
                                          ));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
