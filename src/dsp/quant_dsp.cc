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
// Quantization
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cassert>

#include "src/dsp/dsp.h"

//------------------------------------------------------------------------------

namespace {

void Quantize_C(const uint32_t iq[], const uint32_t bias[],
                const int32_t res[], int16_t coeffs[], uint32_t W, uint32_t H) {
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      coeffs[x] = (res[x] < 0) ? -WP2Quantize16b(-res[x], iq[x], bias[x])
                               :  WP2Quantize16b( res[x], iq[x], bias[x]);
    }
    coeffs += W;
    res += W;
    iq += WP2QStride;
    bias += WP2QStride;
  }
}

void Dequantize_C(const int16_t in[], const int16_t dequants[], int32_t out[],
                  uint32_t W, uint32_t H, uint32_t len) {
  assert(len <= W * H);
  int32_t* const end = out + W * H;
  for (uint32_t y = 0; y < len; y += W) {
    for (uint32_t x = 0; x < W; ++x) out[x] = in[x] * dequants[x];
    dequants += WP2QStride;
    out += W;
    in += W;
  }
  std::fill(out, end, 0);
}

#if defined(WP2_USE_SSE)

//------------------------------------------------------------------------------
// SSE version

void Quantize_SSE(const uint32_t iq[], const uint32_t bias[],
                  const int32_t res[], int16_t coeffs[],
                  uint32_t W, uint32_t H) {
  for (uint32_t y = 0; y < H; ++y) {
    if (W > 4) {
      for (uint32_t x = 0; x < W; x += 8) {
        const __m128i IQ0 = _mm_loadu_si128((const __m128i*)&iq[x + 0]);
        const __m128i IQ1 = _mm_loadu_si128((const __m128i*)&iq[x + 4]);
        const __m128i BIAS0 = _mm_loadu_si128((const __m128i*)&bias[x + 0]);
        const __m128i BIAS1 = _mm_loadu_si128((const __m128i*)&bias[x + 4]);
        const __m128i r0 = _mm_loadu_si128((const __m128i*)&res[x + 0]);
        const __m128i r1 = _mm_loadu_si128((const __m128i*)&res[x + 4]);
        const __m128i abs_r0 = _mm_abs_epi32(r0);
        const __m128i abs_r1 = _mm_abs_epi32(r1);
        // |res| * iq + bias
        const __m128i A0 = _mm_mullo_epi32(abs_r0, IQ0);
        const __m128i A1 = _mm_mullo_epi32(abs_r1, IQ1);
        const __m128i B0 = _mm_add_epi32(A0, BIAS0);
        const __m128i B1 = _mm_add_epi32(A1, BIAS1);
        // >> WP2QBits
        const __m128i C0 = _mm_srli_epi32(B0, WP2QBits);
        const __m128i C1 = _mm_srli_epi32(B1, WP2QBits);
        // coeffs[] = sign ? -B : B;
        const __m128i D0 = _mm_sign_epi32(C0, r0);
        const __m128i D1 = _mm_sign_epi32(C1, r1);
        const __m128i E = _mm_packs_epi32(D0, D1);
        _mm_storeu_si128((__m128i*)&coeffs[x], E);
      }
    } else {
      const __m128i IQ0 = _mm_loadu_si128((const __m128i*)iq);
      const __m128i BIAS0 = _mm_loadu_si128((const __m128i*)bias);
      const __m128i r0 = _mm_loadu_si128((const __m128i*)res);
      const __m128i abs_r0 = _mm_abs_epi32(r0);
      // |res| * iq + bias
      const __m128i A0 = _mm_mullo_epi32(abs_r0, IQ0);
      const __m128i B0 = _mm_add_epi32(A0, BIAS0);
      // >> WP2QBits
      const __m128i C0 = _mm_srli_epi32(B0, WP2QBits);
      // coeffs[] = sign ? -B : B;
      const __m128i D0 = _mm_sign_epi32(C0, r0);
      const __m128i E = _mm_packs_epi32(D0, D0);
      _mm_storel_epi64((__m128i*)coeffs, E);
    }
    coeffs += W;
    res += W;
    iq += WP2QStride;
    bias += WP2QStride;
  }
}

void Dequantize_SSE(const int16_t in[], const int16_t dequants[], int32_t out[],
                    uint32_t W, uint32_t H, uint32_t len) {
  assert(len <= W * H);
  int32_t* const end = out + W * H;
  for (uint32_t y = 0; y < len; y += W) {
    if (W > 4) {
      for (uint32_t x = 0; x < W; x += 8) {
        const __m128i A0 = _mm_loadu_si128((const __m128i*)&in[x]);
        const __m128i B0 = _mm_loadu_si128((const __m128i*)&dequants[x]);
        const __m128i C0 = _mm_mullo_epi16(A0, B0);
        const __m128i C1 = _mm_mulhi_epi16(A0, B0);
        const __m128i D0 = _mm_unpacklo_epi16(C0, C1);
        const __m128i D1 = _mm_unpackhi_epi16(C0, C1);
        _mm_storeu_si128((__m128i*)&out[x + 0], D0);
        _mm_storeu_si128((__m128i*)&out[x + 4], D1);
      }
    } else {
      const __m128i A0 = _mm_loadl_epi64((const __m128i*)in);
      const __m128i B0 = _mm_loadl_epi64((const __m128i*)dequants);
      const __m128i C0 = _mm_mullo_epi16(A0, B0);
      const __m128i C1 = _mm_mulhi_epi16(A0, B0);
      const __m128i D0 = _mm_unpacklo_epi16(C0, C1);
      _mm_storeu_si128((__m128i*)out, D0);
    }
    dequants += WP2QStride;
    out += W;
    in += W;
  }
  std::fill(out, end, 0);
}

WP2_TSAN_IGNORE_FUNCTION void QuantizeInitSSE() {
  WP2Quantize = Quantize_SSE;
  WP2Dequantize = Dequantize_SSE;
}

#endif  // WP2_USE_SSE

//------------------------------------------------------------------------------

}  // namespace


void (*WP2Quantize)(const uint32_t iq[], const uint32_t bias[],
                    const int32_t res[], int16_t coeffs[],
                    uint32_t W, uint32_t H) = nullptr;
void (*WP2Dequantize)(const int16_t in[], const int16_t dequants[],
                      int32_t out[],
                      uint32_t W, uint32_t H, uint32_t len) = nullptr;

static volatile WP2CPUInfo quantize_last_cpuinfo_used =
    (WP2CPUInfo)&quantize_last_cpuinfo_used;

extern void WP2QuantizeInitSSE();

WP2_TSAN_IGNORE_FUNCTION void WP2QuantizeInit() {
  if (quantize_last_cpuinfo_used == WP2GetCPUInfo) return;

  WP2Quantize = Quantize_C;
  WP2Dequantize = Dequantize_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) QuantizeInitSSE();
#endif
  }

  quantize_last_cpuinfo_used = WP2GetCPUInfo;
}
