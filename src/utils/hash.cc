// Copyright 2022 Google LLC
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
// -----------------------------------------------------------------------------
//
// Author: Felipe Mourad Pereira (mouradfelipe@google.com)

#include "src/utils/hash.h"

#ifdef WP2_WORDS_BIGENDIAN
#define SPLIT(X) (uint8_t)((uint16_t)argb[X] >> 8), (uint8_t)(argb[X] & 0xff)
#else
#define SPLIT(X) (uint8_t)(argb[X] & 0xff), (uint8_t)((uint16_t)argb[X] >> 8)
#endif

namespace WP2 {
namespace decoding {

uint32_t HashPixN(const int16_t* const argb, uint32_t num_pixels,
                  uint32_t hash_bits) {
  uint64_t h1, h2, h3, h4 = 0;
  for (uint32_t i = 0; i < num_pixels; ++i) {
    const uint8_t aux[8] = {SPLIT(4 * i + 0), SPLIT(4 * i + 1),
                            SPLIT(4 * i + 2), SPLIT(4 * i + 3)};
    h1 = internal::HashPixImpl(aux);
    h2 = internal::HashPixImpl(aux + 4);
    h3 = internal::HashCombine(h1, h2);
    h4 = internal::HashCombine(h3, h4);
  }
  return internal::FinalizeHash(h4, hash_bits);
}

}  // namespace decoding
}  // namespace WP2
