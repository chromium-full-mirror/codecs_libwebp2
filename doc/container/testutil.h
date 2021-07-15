// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONTAINER_TESTUTIL_H_
#define CONTAINER_TESTUTIL_H_

/* copybara:strip_begin(Googletest include paths) */
#include "testing/base/public/gunit.h"
/* copybara:strip_end_and_replace
#include "gtest/gtest.h"
*/

#include "./container.h"

namespace container {
namespace testutil {

// -----------------------------------------------------------------------------

// Checks that the images are identical. Call it with ASSERT_NO_FATAL_FAILURE().
void VerifyEqual(const Image& a, const Image& b);

// Returns an image for testing.
Image GenerateRandomImage();

// -----------------------------------------------------------------------------

}  // namespace testutil
}  // namespace container

#endif  // CONTAINER_TESTUTIL_H_
