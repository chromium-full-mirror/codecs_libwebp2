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
//  Fuzzing of raw pixels and still image wp2 encoding settings.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/wp2/encode.h"
#include "tests/fuzz/fuzz_utils.h"
#include "tests/include/helpers.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Input is considered random data and converted to parameters.
  WP2::testutil::FuzzedParameters params(data, size);

  // Extract a random config.
  WP2::EncoderConfig config;
  params.ExtractConfig(&config);

  // Make sure the encoding stops before the fuzzer timeout.
  WP2::testutil::ProgressTimeout progress(
      WP2::testutil::kEncodingMaxNumSeconds);
  config.progress_hook = &progress;
  progress.fail_at_ = params.ExtractBool() ? params.ExtractDouble(0., 1.) : 2.;

  // Create a randomly sized valid buffer with random pixels. It's best to
  // extract other params before because it will try to use all remaining bytes.
  WP2::ArgbBuffer original =
      params.ExtractArgbBuffer(/*max_num_pixels=*/256u * 256u);
  if (original.IsEmpty()) return 0;  // No byte left to extract as raw pixels.

  // Input can be random noise; output can get heavily distorted but it's fine.
  const float expected_distortion =
      WP2::testutil::GetExpectedDistortion(config) - 15.f;
  return WP2::testutil::TestImageEncConfig(original, config,
                                           expected_distortion);
}
