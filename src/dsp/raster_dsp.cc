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
// Rasterization for preview
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cassert>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------
// C version

static void RasterAdvance_C(grad_t grads[3 * 4], grad_t dX, grad_t cur[4]) {
  cur[0] = grads[ 8] + grads[0] * dX;
  cur[1] = grads[ 9] + grads[1] * dX;
  cur[2] = grads[10] + grads[2] * dX;
  cur[3] = grads[11] + grads[3] * dX;
  grads[ 8] += grads[4];
  grads[ 9] += grads[5];
  grads[10] += grads[6];
  grads[11] += grads[7];
}

static uint32_t Square(int32_t v, uint8_t b) {
  v = (v >> kRasterPrec) - b;
  return v * v;
}

static void RasterLoss_C(const uint8_t* src, uint32_t size,
                         const grad_t value[4], const grad_t gradient[4],
                         uint32_t loss[4]) {
  grad_t cur[4] = { value[0], value[1], value[2], value[3] };
  for (uint32_t x = 0; x < size; ++x, src += 4) {
    loss[0] += Square(cur[0], src[0]);
    loss[1] += Square(cur[1], src[1]);
    loss[2] += Square(cur[2], src[2]);
    loss[3] += Square(cur[3], src[3]);
    cur[0] += gradient[0];
    cur[1] += gradient[1];
    cur[2] += gradient[2];
    cur[3] += gradient[3];
  }
}

static uint8_t FloorToUInt8(int32_t v) {
  v >>= kRasterPrec;
  return (v < 0) ? 0u : (v > 255) ? 255u : v;
}

static void RasterDraw_C(const grad_t value[4], const grad_t gradient[4],
                         uint8_t* dst, uint32_t size) {
  grad_t cur[4] = { value[0], value[1], value[2], value[3] };
  for (uint32_t x = 0; x < size; ++x, dst += 4) {
    dst[0] = FloorToUInt8(cur[0]);
    dst[1] = FloorToUInt8(cur[1]);
    dst[2] = FloorToUInt8(cur[2]);
    dst[3] = FloorToUInt8(cur[3]);
    cur[0] += gradient[0];
    cur[1] += gradient[1];
    cur[2] += gradient[2];
    cur[3] += gradient[3];
  }
}

//------------------------------------------------------------------------------
// SSE version

#if defined(WP2_USE_SSE)

static void RasterAdvance_SSE(grad_t grads[3 * 4], grad_t dX, grad_t cur[4]) {
  const __m128i A = _mm_loadu_si128((const __m128i*)&grads[0]);
  const __m128i B = _mm_loadu_si128((const __m128i*)&grads[4]);
  const __m128i C = _mm_loadu_si128((const __m128i*)&grads[8]);
  const __m128i D = _mm_mullo_epi32(A, _mm_set1_epi32(dX));
  const __m128i E = _mm_add_epi32(C, D);
  _mm_storeu_si128((__m128i*)cur, E);
  const __m128i F = _mm_add_epi32(C, B);
  _mm_storeu_si128((__m128i*)&grads[8], F);
}

static void RasterLoss_SSE(const uint8_t* src, uint32_t size,
                           const grad_t value[4], const grad_t gradient[4],
                           uint32_t loss[4]) {
  const __m128i grad = _mm_loadu_si128((const __m128i*)gradient);
  __m128i C = _mm_loadu_si128((const __m128i*)value);
  __m128i L = _mm_loadu_si128((const __m128i*)loss);
  const __m128i zero = _mm_setzero_si128();
  for (uint32_t x = 0; x < size; ++x, src += 4) {
    const __m128i C1 = _mm_srai_epi32(C, kRasterPrec);
    C = _mm_add_epi32(C, grad);
    const __m128i D = _mm_cvtsi32_si128(*(int32_t*)src);
    const __m128i E = _mm_unpacklo_epi8(D, zero);
    const __m128i F = _mm_unpacklo_epi16(E, zero);
    const __m128i G = _mm_sub_epi32(C1, F);
    const __m128i H = _mm_mullo_epi32(G, G);
    L = _mm_add_epi32(L, H);
  }
  _mm_storeu_si128((__m128i*)loss, L);
}

static void RasterDraw_SSE(const grad_t value[4], const grad_t gradient[4],
                           uint8_t* dst, uint32_t size) {
  const __m128i grad = _mm_loadu_si128((const __m128i*)gradient);
  __m128i A = _mm_loadu_si128((const __m128i*)value);
  for (uint32_t x = 0; x < size; ++x, dst += 4) {
    const __m128i B = _mm_srai_epi32(A, kRasterPrec);
    A = _mm_add_epi32(A, grad);
    const __m128i C = _mm_packs_epi32(B, B);
    const __m128i D = _mm_packus_epi16(C, C);
    *(int*)dst = _mm_cvtsi128_si32(D);
  }
}

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------

}  // namespace

void (*RasterAdvance)(grad_t grads[3 * 4], grad_t dX, grad_t cur[4]) = nullptr;
void (*RasterLoss)(const uint8_t* src, uint32_t size,
                   const grad_t value[4],
                   const grad_t gradient[4], uint32_t loss[4]) = nullptr;
void (*RasterDraw)(const grad_t value[4], const grad_t gradient[4],
                   uint8_t* dst, uint32_t size) = nullptr;

static volatile WP2CPUInfo raster_last_cpuinfo_used =
    (WP2CPUInfo)&raster_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void RasterInit() {
  if (raster_last_cpuinfo_used == WP2GetCPUInfo) return;

  RasterAdvance = RasterAdvance_C;
  RasterLoss = RasterLoss_C;
  RasterDraw = RasterDraw_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) {
      RasterAdvance = RasterAdvance_SSE;
      RasterLoss = RasterLoss_SSE;
      RasterDraw = RasterDraw_SSE;
    }
#endif
  }
  raster_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------

}  // namespace WP2
