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
// ANS DSP
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cassert>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------
// C version

void ANSUpdateCDF_C(uint32_t n, const uint16_t cdf_base[],
                    const uint16_t cdf_var[], uint32_t mult, uint16_t cumul[]) {
  if (mult == 65535) {  // special case 'copy'
    for (uint32_t i = 1; i < n; ++i) {
      const int32_t v = (int32_t)cdf_var[i] + cdf_base[i] - cumul[i];
      cumul[i] += v;
    }
  } else {
    for (uint32_t i = 1; i < n; ++i) {
      const int32_t v = (int32_t)cdf_var[i] + cdf_base[i] - cumul[i];
      cumul[i] += RightShift(v * mult, 16);
    }
  }
  for (uint32_t i = 1; i < n; ++i) assert(cumul[i] >= cumul[i - 1]);
}

uint32_t ANSGetSymbol_C(uint32_t max_symbol, const uint16_t cumul[],
                        uint32_t proba) {
  (void)max_symbol;
  uint32_t s = 0;
#if 0
  // loop version
  for (; s < max_symbol - 1; ++s) {
    if (proba < cumul_[s + 1]) break;
  }
#else
  // binary search
#if (APROBA_MAX_SYMBOL > 16)
  if (proba >= cumul[16]) s += 16;
#endif
#if (APROBA_MAX_SYMBOL > 8)
  if (proba >= cumul[s + 8]) s += 8;
#endif
#if (APROBA_MAX_SYMBOL > 4)
  if (proba >= cumul[s + 4]) s += 4;
#endif
  if (proba >= cumul[s + 2]) s += 2;
  if (proba >= cumul[s + 1]) s += 1;
#endif
  return s;
}

//------------------------------------------------------------------------------
// SSE version

#if defined(WP2_USE_SSE)

void ANSUpdateCDF_SSE(uint32_t n, const uint16_t cdf_base[],
                      const uint16_t cdf_var[], uint32_t mult,
                      uint16_t cumul[]) {
  if (mult == 65535) {  // special case 'copy'
    for (uint32_t i = 0; i < n; i += 8) {
      const __m128i A = _mm_loadu_si128((const __m128i*)&cdf_var[i]);
      const __m128i B = _mm_loadu_si128((const __m128i*)&cdf_base[i]);
      const __m128i D = _mm_add_epi16(A, B);
      _mm_storeu_si128((__m128i*)&cumul[i], D);
    }
  } else {
    assert(mult < 32768);
    const __m128i M = _mm_set1_epi16(mult);
    for (uint32_t i = 0; i < n; i += 8) {
      const __m128i A = _mm_loadu_si128((const __m128i*)&cdf_var[i]);
      const __m128i B = _mm_loadu_si128((const __m128i*)&cdf_base[i]);
      const __m128i C = _mm_loadu_si128((const __m128i*)&cumul[i]);
      const __m128i D = _mm_add_epi16(A, B);
      const __m128i E = _mm_sub_epi16(D, C);
      const __m128i F = _mm_mulhi_epi16(E, M);
      const __m128i I = _mm_add_epi16(C, F);
      _mm_storeu_si128((__m128i*)&cumul[i], I);
    }
  }
}

uint32_t ANSGetSymbol_SSE(uint32_t max_symbol, const uint16_t cumul[],
                          uint32_t proba) {
  (void)max_symbol;
  // TODO(yguyon): SSE other than APROBA_MAX_SYMBOL=16
#if (APROBA_MAX_SYMBOL == 16)
  const __m128i A0 = _mm_loadu_si128((const __m128i*)&cumul[0]);
  const __m128i A1 = _mm_loadu_si128((const __m128i*)&cumul[8]);
  const __m128i B = _mm_set1_epi16(proba + 1);
  const __m128i C0 = _mm_cmplt_epi16(A0, B);  // we'd need mm_cmple_epi16 !!
  const __m128i C1 = _mm_cmplt_epi16(A1, B);
  const __m128i D = _mm_packs_epi16(C0, C1);
  const uint32_t bits = _mm_movemask_epi8(D);
  return WP2Log2Floor(bits);
#elif (APROBA_MAX_SYMBOL == 32)
  const __m128i A0 = _mm_loadu_si128((const __m128i*)&cumul[0]);
  const __m128i A1 = _mm_loadu_si128((const __m128i*)&cumul[8]);
  const __m128i A2 = _mm_loadu_si128((const __m128i*)&cumul[16]);
  const __m128i A3 = _mm_loadu_si128((const __m128i*)&cumul[24]);
  const __m128i B = _mm_set1_epi16(proba + 1);
  const __m128i C0 = _mm_cmplt_epi16(A0, B);  // we'd need mm_cmple_epi16 !!
  const __m128i C1 = _mm_cmplt_epi16(A1, B);
  const __m128i C2 = _mm_cmplt_epi16(A2, B);
  const __m128i C3 = _mm_cmplt_epi16(A3, B);
  const __m128i D0 = _mm_packs_epi16(C0, C1);
  const __m128i D1 = _mm_packs_epi16(C2, C3);
  const uint32_t E0 = _mm_movemask_epi8(D0);
  const uint32_t E1 = _mm_movemask_epi8(D1);
  const uint32_t bits = E0 | (E1 << 16);
  return WP2Log2Floor(bits);
#else
  return ANSGetSymbol_C(max_symbol, cumul, proba);
#endif
}

#endif  // WP2_USE_SSE

}  // namespace

//------------------------------------------------------------------------------

ANSUpdateCDFFunc ANSUpdateCDF = nullptr;
ANSGetSymbolFunc ANSGetSymbol = nullptr;

static volatile WP2CPUInfo ans_last_cpuinfo_used =
    (WP2CPUInfo)&ans_last_cpuinfo_used;

WP2_TSAN_IGNORE_FUNCTION void ANSInit() {
  if (ans_last_cpuinfo_used == WP2GetCPUInfo) return;

  ANSUpdateCDF = ANSUpdateCDF_C;
  ANSGetSymbol = ANSGetSymbol_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) {
      ANSUpdateCDF = ANSUpdateCDF_SSE;
      ANSGetSymbol = ANSGetSymbol_SSE;
    }
#endif
  }
  ans_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------

}  // namespace WP2
