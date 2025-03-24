// Copyright 2019 Google LLC
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
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include "src/dsp/dsp.h"

#if defined(WP2_USE_SSE)

#include <nmmintrin.h>

#include <cassert>
#include <cstdint>

#include "src/dsp/lossless/decl_dsp.h"
#include "src/dsp/lossless/dspl.h"

namespace WP2L {
namespace {

void TransformColorInverse_SSE(const WP2L::Multipliers& m, const int16_t* src,
                               uint32_t num_pixels, int16_t* dst,
                               int16_t* predicted) {
  const int32_t green_to_red = m.green_to_red;
  const int32_t green_to_blue = m.green_to_blue;
  const int32_t red_to_blue = m.red_to_blue;
  const int16_t* const src_end = src + 4 * num_pixels;
  const __m128i kMask_ag = _mm_set1_epi32(0xffff);  // A0G0 mask
  const __m128i kRound = _mm_set1_epi32(0x8000);
  const __m128i perm =
      _mm_setr_epi8(2, 3, 4, 5, 6, 7, 4, 5, 10, 11, 12, 13, 14, 15, 12, 13);

  __m128i mult1 = _mm_cvtsi32_si128((uint32_t)green_to_red << 16 | 32);
  __m128i mult2 = _mm_cvtsi32_si128((uint32_t)red_to_blue << 16);
  mult1 = _mm_insert_epi32(mult1, (uint32_t)green_to_blue << 16 | 32u, 1);
  mult1 = _mm_unpacklo_epi64(mult1, mult1);
  mult2 = _mm_unpacklo_epi64(mult2, mult2);

  for (; src + 8 <= src_end; src += 8, dst += 8) {
    if (predicted != nullptr) {
      for (int c : {0, 4}) {
        predicted[c + 0] = src[c + 0];
        const int16_t red = src[c + 1];
        const int16_t green = src[c + 2];
        predicted[c + 1] = ColorTransformDelta(m.green_to_red * green);
        predicted[c + 2] = green;
        const int16_t new_red =
            red + ColorTransformDelta(m.green_to_red * green);
        predicted[c + 3] =
            ColorTransformDelta(green_to_blue * green + red_to_blue * new_red);
      }
      predicted += 8;
    }
    const __m128i A = _mm_loadu_si128((const __m128i*)src);
    const __m128i B = _mm_shuffle_epi8(A, perm);  // ARGB -> RGBG
    const __m128i C = _mm_madd_epi16(B, mult1);
    const __m128i D = _mm_add_epi32(_mm_slli_epi32(C, 16 - 5), kRound);
    const __m128i E = _mm_madd_epi16(D, mult2);
    const __m128i F = _mm_slli_epi64(E, 48 - 5);
    const __m128i G = _mm_add_epi32(F, D);
    const __m128i out = _mm_blendv_epi8(G, A, kMask_ag);
    _mm_storeu_si128((__m128i*)dst, out);
  }
  for (; src < src_end; src += 4, dst += 4) {
    const int16_t red = src[1];
    const int16_t green = src[2];
    int32_t new_red = red;
    int32_t new_blue = src[3];
    if (predicted != nullptr) {
      predicted[1] = ColorTransformDelta(m.green_to_red * green);
    }
    new_red += ColorTransformDelta(green_to_red * green);
    if (predicted != nullptr) {
      predicted[3] =
          ColorTransformDelta(green_to_blue * green + red_to_blue * new_red);
    }
    new_blue +=
        ColorTransformDelta(green_to_blue * green + red_to_blue * new_red);
    dst[0] = src[0];
    dst[1] = new_red;
    dst[2] = src[2];
    dst[3] = new_blue;
  }
}

}  // namespace

extern void DecLDspInitSSE();

WP2_TSAN_IGNORE_FUNCTION void DecLDspInitSSE() {
  TransformColorInverse = TransformColorInverse_SSE;
}
}  // namespace WP2L

#else  // !WP2_USE_SSE

namespace WP2L {
WP2_DSP_INIT_STUB(DecLDspInitSSE)
}  // namespace WP2L

#endif  // WP2_USE_SSE
