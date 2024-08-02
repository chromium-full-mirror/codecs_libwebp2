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
// -----------------------------------------------------------------------------

#ifndef WP2_TESTS_FUZZ_FUZZ_UTILS_TEMP_FILE_H_
#define WP2_TESTS_FUZZ_FUZZ_UTILS_TEMP_FILE_H_

// Adapter utility from fuzzer input to a temporary file, for fuzzing APIs that
// require a file instead of an input buffer.

#include <cstdio>
#include <cstring>

class FuzzerTemporaryFile {
 public:
  FuzzerTemporaryFile() {
    std::memset(path_, 0, sizeof(path_));
    std::tmpnam(path_);
  }

  ~FuzzerTemporaryFile() {
    if (std::strlen(path_) != 0) {
      std::remove(path_);
    }
  }

  const char* path() const { return path_; }

 private:
  char path_[L_tmpnam];
};

#endif  // WP2_TESTS_FUZZ_FUZZ_UTILS_TEMP_FILE_H_
