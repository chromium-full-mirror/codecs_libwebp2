// Copyright 2020 Google LLC
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
//   SSE helper function
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WP2_DSP_DSP_X86_H_
#define WP2_DSP_DSP_X86_H_

#if defined(WP2_USE_SSE)
#include <stdio.h>

//------------------------------------------------------------------------------
// debugging

template<typename T>
void Print(const __m128i A, const char name[], int how_many = 32) {
  constexpr int size = 16 / sizeof(T);
  how_many = std::min(how_many, size);
  T tmp[size];
  _mm_storeu_si128((__m128i*)tmp, A);
  printf("%s :", name);
  for (int i = 0; i < how_many; ++i) {
    printf((size == 16) ? " %.2x" : (size == 8) ? " %.4x" : " %.8x", tmp[i]);
  }
  printf("\n");
}
static inline
void Print32b(const __m128i A, const char name[], int how_many = 4) {
  Print<uint32_t>(A, name, how_many);
}
static inline
void Print16b(const __m128i A, const char name[], int how_many = 8) {
  Print<uint16_t>(A, name, how_many);
}
static inline
void Print8b(const __m128i A, const char name[], int how_many = 16) {
  Print<uint8_t>(A, name, how_many);
}

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------

#endif  /* WP2_DSP_DSP_X86_H_ */
