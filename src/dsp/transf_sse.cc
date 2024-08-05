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
// SSE implementation of Transforms
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

#if defined(WP2_USE_SSE)

#include "src/dsp/dsp_x86.h"

namespace {

//------------------------------------------------------------------------------
// Constants for 10 bit precision

// kCosPi[j] = (int)round(cos(M_PI*j/128) * (1<<kPrecision));
constexpr uint32_t kPrecision = 10u;
constexpr int32_t kCosPi[64] = {
    1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
    972,  964,  955,  946,  936,  926,  915,  903,  891,  878, 865, 851, 837,
    822,  807,  792,  775,  759,  742,  724,  706,  688,  669, 650, 630, 610,
    590,  569,  548,  526,  505,  483,  460,  438,  415,  392, 369, 345, 321,
    297,  273,  249,  224,  200,  175,  150,  125,  100,  75,  50,  25};
// Pre-calculated constants kCosPi[32] * kCosPi[16/48]
constexpr int32_t kCos32Cos48 = 277;  // cos(pi/4) * cos(3 * pi / 8)
constexpr int32_t kCos32Cos16 = 669;  // cos(pi/4) * cos(pi / 8)
// These are the same constants but with one extra precision bit (p=pre-scaled).
constexpr int32_t kCos32Cos48p = 554;   // 2 * cos(pi/4) * cos(3 * pi / 8)
constexpr int32_t kCos32Cos16p = 1337;  // 2 * cos(pi/4) * cos(pi / 8)

//------------------------------------------------------------------------------

template <class V>
class ColTransformGen {
 public:
  static void idct2(int16_t* coeffs, uint32_t w);
  static void idct4(int16_t* coeffs, uint32_t w);
  static void idct8(int16_t* coeffs, uint32_t w);

  static void fdct2(int32_t* coeffs, uint32_t w);
  static void fdct4(int32_t* coeffs, uint32_t w);
  static void fdct8(int32_t* coeffs, uint32_t w);
  static void fdct16(int32_t* coeffs, uint32_t w);

 private:
  static constexpr uint32_t kStep = V::Step();
  static V Load(const int32_t in[]) { return V(in); }
  static constexpr V Rounding(uint32_t n) { return V::Rounding(n); }
};

//------------------------------------------------------------------------------
// column-wise idct

template <class V>
void ColTransformGen<V>::idct2(int16_t* coeffs, uint32_t w) {
  const V c_32 = kCosPi[32];
  const V kRnd0 = Rounding(kPrecision);
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    const V a0 = Load_s16_s32(coeffs + w * 0);
    const V a1 = Load_s16_s32(coeffs + w * 1);
    const V b0 = ((a0 + a1) * c_32 + kRnd0) >> kPrecision;
    const V b1 = ((a0 - a1) * c_32 + kRnd0) >> kPrecision;
    b0.Store16(coeffs + w * 0);
    b1.Store16(coeffs + w * 1);
  }
}

template <class V>
void ColTransformGen<V>::idct4(int16_t* coeffs, uint32_t w) {
  const V kRnd0 = Rounding(kPrecision);
  const V c_32_48 = kCos32Cos48p;
  const V c_32_16 = kCos32Cos16p;
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    V a0 = Load_s16_s32(coeffs + w * 0);
    V a1 = Load_s16_s32(coeffs + w * 2);
    V a2 = Load_s16_s32(coeffs + w * 1);
    V a3 = Load_s16_s32(coeffs + w * 3);

    const V b0 = a0 + a1;
    const V b1 = a0 - a1;
    const V b2 = (c_32_48 * a2 - c_32_16 * a3 + kRnd0) >> kPrecision;
    const V b3 = (c_32_16 * a2 + c_32_48 * a3 + kRnd0) >> kPrecision;

    a0 = (b0 + b3) >> 1;
    a1 = (b1 + b2) >> 1;
    a2 = (b1 - b2) >> 1;
    a3 = (b0 - b3) >> 1;

    a0.Store16(coeffs + w * 0);
    a1.Store16(coeffs + w * 1);
    a2.Store16(coeffs + w * 2);
    a3.Store16(coeffs + w * 3);
  }
}

template <class V>
void ColTransformGen<V>::idct8(int16_t* coeffs, uint32_t w) {
  const V c_08 = kCosPi[8];
  const V c_16 = kCosPi[16];
  const V c_24 = kCosPi[24];
  const V c_32 = kCosPi[32];
  const V c_40 = kCosPi[40];
  const V c_48 = kCosPi[48];
  const V c_56 = kCosPi[56];
  constexpr uint32_t shift = 1;
  const V kRnd0 = Rounding(kPrecision);
  const V kRnd1 = 1;  // rounder
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    V b0 = Load_s16_s32(coeffs + w * 0);
    V b1 = Load_s16_s32(coeffs + w * 4);
    V b2 = Load_s16_s32(coeffs + w * 2);
    V b3 = Load_s16_s32(coeffs + w * 6);

    V a0 = (((b0 + b1) * c_32 + kRnd0) >> kPrecision) + kRnd1;
    V a1 = (((b0 - b1) * c_32 + kRnd0) >> kPrecision) + kRnd1;
    V a2 = (c_48 * b2 - c_16 * b3 + kRnd0) >> kPrecision;
    V a3 = (c_16 * b2 + c_48 * b3 + kRnd0) >> kPrecision;

    b0 = a0 + a3;
    b1 = a1 + a2;
    b2 = a1 - a2;
    b3 = a0 - a3;

    V a4 = Load_s16_s32(coeffs + w * 1);
    V a5 = Load_s16_s32(coeffs + w * 5);
    V a6 = Load_s16_s32(coeffs + w * 3);
    V a7 = Load_s16_s32(coeffs + w * 7);

    V b4 = (c_56 * a4 - c_08 * a7 + kRnd0) >> kPrecision;
    V b7 = (c_08 * a4 + c_56 * a7 + kRnd0) >> kPrecision;
    V b5 = (c_24 * a5 - c_40 * a6 + kRnd0) >> kPrecision;
    V b6 = (c_40 * a5 + c_24 * a6 + kRnd0) >> kPrecision;

    a4 = b5 + b4;
    a5 = b4 - b5;
    a6 = b7 - b6;
    a7 = b6 + b7;

    b4 = a4;
    b5 = ((a6 - a5) * c_32 + kRnd0) >> kPrecision;
    b6 = ((a6 + a5) * c_32 + kRnd0) >> kPrecision;
    b7 = a7;

    a0 = (b0 + b7) >> shift;
    a1 = (b1 + b6) >> shift;
    a6 = (b1 - b6) >> shift;
    a7 = (b0 - b7) >> shift;

    a0.Store16(coeffs + w * 0);
    a1.Store16(coeffs + w * 1);
    a6.Store16(coeffs + w * 6);
    a7.Store16(coeffs + w * 7);

    a2 = (b2 + b5) >> shift;
    a3 = (b3 + b4) >> shift;
    a4 = (b3 - b4) >> shift;
    a5 = (b2 - b5) >> shift;

    a2.Store16(coeffs + w * 2);
    a3.Store16(coeffs + w * 3);
    a4.Store16(coeffs + w * 4);
    a5.Store16(coeffs + w * 5);
  }
}

//------------------------------------------------------------------------------
// column-wise fdct

template <class V>
void ColTransformGen<V>::fdct2(int32_t* coeffs, uint32_t w) {
  const V c_32 = kCosPi[32];
  const V kRnd0 = Rounding(kPrecision);
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    const V a0(coeffs + w * 0);
    const V a1(coeffs + w * 1);
    const V b0 = ((a0 + a1) * c_32 + kRnd0) >> kPrecision;
    const V b1 = ((a0 - a1) * c_32 + kRnd0) >> kPrecision;
    b0.Store(coeffs + w * 0);
    b1.Store(coeffs + w * 1);
  }
}

template <class V>
void ColTransformGen<V>::fdct4(int32_t* coeffs, uint32_t w) {
  const V kRnd1 = 1;  // rounder for shift=1
  const V kRnd0 = Rounding(kPrecision);
  const V c_32_48 = kCos32Cos48;
  const V c_32_16 = kCos32Cos16;
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    V b0(coeffs + w * 0);
    V b1(coeffs + w * 1);
    V b2(coeffs + w * 2);
    V b3(coeffs + w * 3);

    const V a0 = b3 + b0;
    const V a1 = b2 + b1;
    const V a2 = b1 - b2;
    const V a3 = b0 - b3;

    b0 = (a0 + a1 + kRnd1) >> 1;
    b1 = (a0 - a1 + kRnd1) >> 1;
    b2 = (c_32_48 * a2 + c_32_16 * a3 + kRnd0) >> kPrecision;
    b3 = (c_32_48 * a3 - c_32_16 * a2 + kRnd0) >> kPrecision;

    b0.Store(coeffs + w * 0);
    b2.Store(coeffs + w * 1);
    b1.Store(coeffs + w * 2);
    b3.Store(coeffs + w * 3);
  }
}

template <class V>
void ColTransformGen<V>::fdct8(int32_t* coeffs, uint32_t w) {
  constexpr uint32_t shift = kPrecision + 1;
  const V kRnd0 = Rounding(kPrecision);
  const V kRnd1 = Rounding(shift);
  const V c_08 = kCosPi[8];
  const V c_16 = kCosPi[16];
  const V c_24 = kCosPi[24];
  const V c_32 = kCosPi[32];
  const V c_40 = kCosPi[40];
  const V c_48 = kCosPi[48];
  const V c_56 = kCosPi[56];
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    V a0(coeffs + w * 0);
    V a1(coeffs + w * 1);
    V a2(coeffs + w * 2);
    V a3(coeffs + w * 3);

    V b3(coeffs + w * 4);
    V b2(coeffs + w * 5);
    V b1(coeffs + w * 6);
    V b0(coeffs + w * 7);

    // Save for later
    V a7 = a0;
    V a6 = a1;
    V a5 = a2;
    V a4 = a3;
    V b4 = b3;
    V b5 = b2;
    V b6 = b1;
    V b7 = b0;

    a2 = a2 + b2;
    a3 = a3 + b3;
    a0 = a0 + b0;
    a1 = a1 + b1;

    b0 = a3 + a0;
    b1 = a2 + a1;
    b2 = a1 - a2;
    b3 = a0 - a3;

    a0 = ((b0 + b1) * c_32 + kRnd1) >> shift;
    a1 = ((b0 - b1) * c_32 + kRnd1) >> shift;
    a2 = (c_48 * b2 + c_16 * b3 + kRnd1) >> shift;
    a3 = (c_48 * b3 - c_16 * b2 + kRnd1) >> shift;

    a0.Store(coeffs + w * 0);
    a2.Store(coeffs + w * 2);
    a1.Store(coeffs + w * 4);
    a3.Store(coeffs + w * 6);

    a4 = a4 - b4;
    a5 = a5 - b5;
    a6 = a6 - b6;
    a7 = a7 - b7;

    b5 = ((a6 - a5) * c_32 + kRnd0) >> kPrecision;
    b6 = ((a6 + a5) * c_32 + kRnd0) >> kPrecision;

    a5 = a4 - b5;
    a6 = a7 - b6;
    a4 = a4 + b5;
    a7 = a7 + b6;

    b4 = (c_56 * a4 + c_08 * a7 + kRnd1) >> shift;
    b7 = (c_56 * a7 - c_08 * a4 + kRnd1) >> shift;
    b5 = (c_24 * a5 + c_40 * a6 + kRnd1) >> shift;
    b6 = (c_24 * a6 - c_40 * a5 + kRnd1) >> shift;

    b4.Store(coeffs + w * 1);
    b6.Store(coeffs + w * 3);
    b5.Store(coeffs + w * 5);
    b7.Store(coeffs + w * 7);
  }
}

template <class V>
void ColTransformGen<V>::fdct16(int32_t* coeffs, uint32_t w) {
  constexpr uint32_t shift = kPrecision + 1;
  const V kRnd0 = Rounding(kPrecision);
  const V kRnd1 = Rounding(shift);
  for (uint32_t n = V::Size(w); n--; coeffs += kStep) {
    int32_t temp[8 * kStep];
    V c0, c1;

    // low
    V a0 = Load(coeffs + w * 0) + Load(coeffs + w * 15);
    V a1 = Load(coeffs + w * 1) + Load(coeffs + w * 14);
    V a2 = Load(coeffs + w * 2) + Load(coeffs + w * 13);
    V a3 = Load(coeffs + w * 3) + Load(coeffs + w * 12);
    V a4 = Load(coeffs + w * 4) + Load(coeffs + w * 11);
    V a5 = Load(coeffs + w * 5) + Load(coeffs + w * 10);
    V a6 = Load(coeffs + w * 6) + Load(coeffs + w * 9);
    V a7 = Load(coeffs + w * 7) + Load(coeffs + w * 8);

    V b0 = a7 + a0;
    V b1 = a6 + a1;
    V b2 = a5 + a2;
    V b3 = a4 + a3;
    V b4 = a3 - a4;
    V b5 = a2 - a5;
    V b6 = a1 - a6;
    V b7 = a0 - a7;

    // part 1
    a0 = b3 + b0;
    a1 = b2 + b1;
    a2 = b1 - b2;
    a3 = b0 - b3;
    c0 = kCosPi[32];
    b0 = ((a0 + a1) * c0 + kRnd0) >> kPrecision;
    b1 = ((a0 - a1) * c0 + kRnd0) >> kPrecision;
    c0 = kCosPi[48];
    c1 = kCosPi[16];
    b2 = (c0 * a2 + c1 * a3 + kRnd0) >> kPrecision;
    b3 = (c0 * a3 - c1 * a2 + kRnd0) >> kPrecision;
    c0 = kCosPi[32];
    b0 = (b0 * c0 + kRnd1) >> shift;
    b1 = (b1 * c0 + kRnd1) >> shift;
    b2 = (b2 * c0 + kRnd1) >> shift;
    b3 = (b3 * c0 + kRnd1) >> shift;

    b0.Store(temp + 0);   // 0
    b2.Store(temp + 4);   // 4
    b1.Store(temp + 8);   // 8
    b3.Store(temp + 12);  // 12

    // part 2
    c0 = kCosPi[32];
    a5 = ((b6 - b5) * c0 + kRnd0) >> kPrecision;
    a6 = ((b6 + b5) * c0 + kRnd0) >> kPrecision;
    b5 = b4 - a5;
    b6 = b7 - a6;
    b4 = b4 + a5;
    b7 = b7 + a6;
    c0 = kCosPi[24];
    c1 = kCosPi[40];
    a5 = (c0 * b5 + c1 * b6 + kRnd0) >> kPrecision;
    a6 = (c0 * b6 - c1 * b5 + kRnd0) >> kPrecision;
    c0 = kCosPi[56];
    c1 = kCosPi[8];
    a4 = (c0 * b4 + c1 * b7 + kRnd0) >> kPrecision;
    a7 = (c0 * b7 - c1 * b4 + kRnd0) >> kPrecision;
    c0 = kCosPi[32];
    a4 = (a4 * c0 + kRnd1) >> shift;
    a5 = (a5 * c0 + kRnd1) >> shift;
    a6 = (a6 * c0 + kRnd1) >> shift;
    a7 = (a7 * c0 + kRnd1) >> shift;

    a4.Store(temp + 16);  // 2
    a6.Store(temp + 20);  // 6
    a5.Store(temp + 24);  // 10
    a7.Store(temp + 28);  // 14

    // high
    a3 = Load(coeffs + w * 4) - Load(coeffs + w * 11);
    a4 = Load(coeffs + w * 3) - Load(coeffs + w * 12);
    c0 = kCosPi[32];
    b3 = ((a4 - a3) * c0 + kRnd0) >> kPrecision;
    b4 = ((a4 + a3) * c0 + kRnd0) >> kPrecision;
    a0 = Load(coeffs + w * 7) - Load(coeffs + w * 8);
    a7 = Load(coeffs + w * 0) - Load(coeffs + w * 15);
    b0 = a0 + b3;
    b7 = a7 + b4;
    b3 = a0 - b3;
    b4 = a7 - b4;

    a2 = Load(coeffs + w * 5) - Load(coeffs + w * 10);
    a5 = Load(coeffs + w * 2) - Load(coeffs + w * 13);
    b2 = ((a5 - a2) * c0 + kRnd0) >> kPrecision;
    b5 = ((a5 + a2) * c0 + kRnd0) >> kPrecision;
    b1 = Load(coeffs + w * 6) - Load(coeffs + w * 9);
    b6 = Load(coeffs + w * 1) - Load(coeffs + w * 14);
    a1 = b2 + b1;
    a2 = b2 - b1;
    a5 = b6 - b5;
    a6 = b6 + b5;
    c0 = kCosPi[48];
    c1 = kCosPi[16];
    b1 = (c0 * a6 - c1 * a1 + kRnd0) >> kPrecision;
    b2 = (c0 * a2 - c1 * a5 + kRnd0) >> kPrecision;
    b5 = (c0 * a5 + c1 * a2 + kRnd0) >> kPrecision;
    b6 = (c0 * a1 + c1 * a6 + kRnd0) >> kPrecision;

    a0 = b0 + b1;
    a1 = b0 - b1;
    a6 = b7 - b6;
    a7 = b7 + b6;
    c0 = kCosPi[60];
    c1 = kCosPi[4];
    b0 = (c0 * a0 + c1 * a7 + kRnd0) >> kPrecision;
    b7 = (c0 * a7 - c1 * a0 + kRnd0) >> kPrecision;
    c0 = kCosPi[28];
    c1 = kCosPi[36];
    b1 = (c0 * a1 + c1 * a6 + kRnd0) >> kPrecision;
    b6 = (c0 * a6 - c1 * a1 + kRnd0) >> kPrecision;
    c0 = kCosPi[32];
    b0 = (b0 * c0 + kRnd1) >> shift;
    b1 = (b1 * c0 + kRnd1) >> shift;
    b6 = (b6 * c0 + kRnd1) >> shift;
    b7 = (b7 * c0 + kRnd1) >> shift;

    b0.Store(coeffs + w * 1);
    b6.Store(coeffs + w * 7);
    b1.Store(coeffs + w * 9);
    b7.Store(coeffs + w * 15);

    a2 = b3 - b2;
    a3 = b3 + b2;
    a4 = b4 + b5;
    a5 = b4 - b5;
    c0 = kCosPi[12];
    c1 = kCosPi[52];
    b3 = (c0 * a3 + c1 * a4 + kRnd0) >> kPrecision;
    b4 = (c0 * a4 - c1 * a3 + kRnd0) >> kPrecision;
    c0 = kCosPi[44];
    c1 = kCosPi[20];
    b2 = (c0 * a2 + c1 * a5 + kRnd0) >> kPrecision;
    b5 = (c0 * a5 - c1 * a2 + kRnd0) >> kPrecision;
    c0 = kCosPi[32];
    b2 = (b2 * c0 + kRnd1) >> shift;
    b3 = (b3 * c0 + kRnd1) >> shift;
    b4 = (b4 * c0 + kRnd1) >> shift;
    b5 = (b5 * c0 + kRnd1) >> shift;

    b4.Store(coeffs + w * 3);
    b2.Store(coeffs + w * 5);
    b5.Store(coeffs + w * 11);
    b3.Store(coeffs + w * 13);

    a0 = Load(temp + 0);
    a1 = Load(temp + 4);
    a2 = Load(temp + 8);
    a3 = Load(temp + 12);
    a4 = Load(temp + 16);
    a5 = Load(temp + 20);
    a6 = Load(temp + 24);
    a7 = Load(temp + 28);

    a0.Store(coeffs + w * 0);
    a1.Store(coeffs + w * 4);
    a2.Store(coeffs + w * 8);
    a3.Store(coeffs + w * 12);
    a4.Store(coeffs + w * 2);
    a5.Store(coeffs + w * 6);
    a6.Store(coeffs + w * 10);
    a7.Store(coeffs + w * 14);
  }
}

//------------------------------------------------------------------------------
// row-wise idct

static void idct4_SSE(int16_t* coeffs) {
  constexpr uint32_t one = 1 << kPrecision;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  __m128i v0, v1, c0, c1;

  // stage 1
  v0 = Load_s16_s32(coeffs);
  v0 = _mm_shuffle_epi32(v0, 0xd8);

  // stage 2
  c0 = _mm_setr_epi32(one, -one, kCos32Cos48p, kCos32Cos48p);
  c1 = _mm_setr_epi32(one, one, -kCos32Cos16p, kCos32Cos16p);
  v1 = _mm_shuffle_epi32(v0, 0xb1);
  v0 = _mm_mullo_epi32(v0, c0);
  v1 = _mm_mullo_epi32(v1, c1);
  v0 = _mm_add_epi32(v0, v1);
  v0 = _mm_add_epi32(v0, kRnd0);
  v0 = _mm_srai_epi32(v0, kPrecision);

  // stage 3
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  v1 = _mm_sign_epi32(v0, c0);
  v0 = _mm_shuffle_epi32(v0, 0x1b);
  v0 = _mm_sub_epi32(v0, v1);
  v0 = _mm_srai_epi32(v0, 1);

  Store_s32_s16(v0, coeffs);
}

static void idct8_SSE(int16_t* coeffs) {
  constexpr uint32_t shift = 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  __m128i v4, v5, c0, c1;

  // stage 1
  __m128i v0 = Load_s16_s32(coeffs + 0);
  __m128i v1 = Load_s16_s32(coeffs + 4);
  __m128i v3 = _mm_shuffle_epi32(v0, 0xb1);
  __m128i v2 = _mm_shuffle_epi32(v1, 0xb1);
  v0 = _mm_blend_epi16(v0, v2, 0xcc);
  v1 = _mm_blend_epi16(v1, v3, 0x33);

  // stage 2
  c0 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[24], kCosPi[56]);
  c1 = _mm_setr_epi32(-kCosPi[8], -kCosPi[40], kCosPi[40], kCosPi[8]);
  v3 = _mm_shuffle_epi32(v1, 0x1b);
  v1 = _mm_mullo_epi32(v1, c0);
  v3 = _mm_mullo_epi32(v3, c1);
  v1 = _mm_add_epi32(v1, v3);
  v1 = _mm_add_epi32(v1, kRnd0);
  v1 = _mm_srai_epi32(v1, kPrecision);

  // stage 3
  c0 = _mm_setr_epi32(kCosPi[32], -kCosPi[32], kCosPi[48], kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[32], kCosPi[32], -kCosPi[16], kCosPi[16]);
  v2 = _mm_shuffle_epi32(v0, 0xb1);
  v0 = _mm_mullo_epi32(v0, c0);
  v2 = _mm_mullo_epi32(v2, c1);
  // +1=rounder
  c0 = _mm_setr_epi32((3 << kPrecision) >> 1, (3 << kPrecision) >> 1,
                      (1 << kPrecision) >> 1, (1 << kPrecision) >> 1);
  v0 = _mm_add_epi32(v0, v2);
  v0 = _mm_add_epi32(v0, c0);
  v0 = _mm_srai_epi32(v0, kPrecision);

  c0 = _mm_setr_epi32(1, -1, -1, 1);
  v3 = _mm_shuffle_epi32(v1, 0xb1);
  v1 = _mm_sign_epi32(v1, c0);
  v1 = _mm_add_epi32(v1, v3);

  // stage 4
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  c1 = _mm_set1_epi32(kCosPi[32]);
  v2 = _mm_sign_epi32(v0, c0);
  v0 = _mm_shuffle_epi32(v0, 0x1b);
  v0 = _mm_sub_epi32(v0, v2);

  v5 = _mm_shuffle_epi32(v1, 0x1b);
  v4 = _mm_sign_epi32(v1, c0);
  v5 = _mm_add_epi32(v5, v4);
  v5 = _mm_mullo_epi32(v5, c1);
  v5 = _mm_add_epi32(v5, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v1 = _mm_blend_epi16(v1, v5, 0x3c);

  // stage 5
  v3 = _mm_shuffle_epi32(v1, 0x1b);
  v2 = _mm_shuffle_epi32(v0, 0x1b);
  v3 = _mm_add_epi32(v3, v0);
  v2 = _mm_sub_epi32(v2, v1);
  v3 = _mm_srai_epi32(v3, shift);
  v2 = _mm_srai_epi32(v2, shift);

  Store_s32_s16(v3, coeffs + 0);
  Store_s32_s16(v2, coeffs + 4);
}

static void idct16_SSE(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, v6, v7, c0, c1;

  // stage 1
  v0 = Load_s16_s32(coeffs + 0);
  v1 = Load_s16_s32(coeffs + 4);
  v2 = Load_s16_s32(coeffs + 8);
  v3 = Load_s16_s32(coeffs + 12);
  v4 = _mm_unpacklo_epi64(v0, v2);  // 0189
  v5 = _mm_unpackhi_epi64(v0, v2);  // 23ab
  v6 = _mm_unpacklo_epi64(v1, v3);  // 45cd
  v7 = _mm_unpackhi_epi64(v1, v3);  // 67ef
  v0 = _mm_unpacklo_epi32(v4, v6);  // 0415
  v1 = _mm_unpackhi_epi32(v4, v6);  // 8c9d
  v2 = _mm_unpacklo_epi32(v5, v7);  // 2637
  v3 = _mm_unpackhi_epi32(v5, v7);  // aebf
  v4 = _mm_unpacklo_epi32(v0, v1);  // 084c
  v6 = _mm_unpackhi_epi32(v0, v1);  // 195d
  v5 = _mm_unpacklo_epi32(v2, v3);  // 2a6e
  v7 = _mm_unpackhi_epi32(v2, v3);  // 3b7f

  // stage 2
  c0 = _mm_setr_epi32(kCosPi[60], kCosPi[28], kCosPi[44], kCosPi[12]);
  c1 = _mm_setr_epi32(kCosPi[4], kCosPi[36], kCosPi[20], kCosPi[52]);
  v7 = _mm_shuffle_epi32(v7, 0x1b);  // fedc
  v0 = _mm_mullo_epi32(v6, c1);
  v6 = _mm_mullo_epi32(v6, c0);
  v1 = _mm_mullo_epi32(v7, c1);
  v7 = _mm_mullo_epi32(v7, c0);
  v6 = _mm_sub_epi32(v6, v1);
  v7 = _mm_add_epi32(v7, v0);
  v6 = _mm_add_epi32(v6, kRnd0);
  v7 = _mm_add_epi32(v7, kRnd0);
  v6 = _mm_srai_epi32(v6, kPrecision);
  v7 = _mm_srai_epi32(v7, kPrecision);

  // stage 3
  c0 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[24], kCosPi[56]);
  c1 = _mm_setr_epi32(kCosPi[8], kCosPi[40], -kCosPi[40], -kCosPi[8]);
  v0 = _mm_shuffle_epi32(v5, 0x1b);
  v5 = _mm_mullo_epi32(v5, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v5 = _mm_sub_epi32(v5, v0);
  v5 = _mm_add_epi32(v5, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);

  v7 = _mm_shuffle_epi32(v7, 0x4e);
  v0 = _mm_blend_epi16(v6, v7, 0x3c);  // 8cfb
  v1 = _mm_blend_epi16(v6, v7, 0xc3);
  v1 = _mm_shuffle_epi32(v1, 0xb1);  // 9dea
  v6 = _mm_add_epi32(v0, v1);
  v7 = _mm_sub_epi32(v0, v1);

  // stage 4
  c0 = _mm_setr_epi32(kCosPi[32], -kCosPi[32], kCosPi[48], kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[32], kCosPi[32], -kCosPi[16], kCosPi[16]);
  v0 = _mm_shuffle_epi32(v4, 0xb1);
  v4 = _mm_mullo_epi32(v4, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v4 = _mm_add_epi32(v4, v0);
  v4 = _mm_add_epi32(v4, kRnd0);
  v4 = _mm_srai_epi32(v4, kPrecision);

  c0 = _mm_setr_epi32(1, -1, -1, 1);
  v0 = _mm_shuffle_epi32(v5, 0xb1);
  v5 = _mm_sign_epi32(v5, c0);
  v5 = _mm_add_epi32(v5, v0);

  c0 = _mm_setr_epi32(-kCosPi[16], kCosPi[48], kCosPi[16], -kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[48], -kCosPi[16], kCosPi[48], -kCosPi[16]);
  v0 = _mm_shuffle_epi32(v7, 0x4e);
  v7 = _mm_mullo_epi32(v7, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v7 = _mm_add_epi32(v7, v0);
  v7 = _mm_add_epi32(v7, kRnd0);
  v7 = _mm_srai_epi32(v7, kPrecision);

  // stage 5
  v0 = _mm_unpacklo_epi32(v6, v7);  // 89cd
  v1 = _mm_unpackhi_epi32(v7, v6);  // efab
  v6 = _mm_unpacklo_epi64(v1, v0);  // ef89
  v7 = _mm_unpackhi_epi64(v1, v0);  // abcd

  v0 = _mm_shuffle_epi32(v6, 0x1b);
  v1 = _mm_shuffle_epi32(v7, 0x1b);
  v6 = _mm_add_epi32(v6, v1);
  v7 = _mm_sub_epi32(v0, v7);

  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  c1 = _mm_set1_epi32(kCosPi[32]);
  v0 = _mm_sign_epi32(v4, c0);
  v4 = _mm_shuffle_epi32(v4, 0x1b);
  v4 = _mm_sub_epi32(v4, v0);

  v3 = _mm_shuffle_epi32(v5, 0x1b);
  v1 = _mm_sign_epi32(v5, c0);
  v3 = _mm_add_epi32(v3, v1);
  v3 = _mm_mullo_epi32(v3, c1);
  v3 = _mm_add_epi32(v3, kRnd0);
  v3 = _mm_srai_epi32(v3, kPrecision);
  v5 = _mm_blend_epi16(v5, v3, 0x3c);

  // stage 6
  v0 = _mm_shuffle_epi32(v5, 0x1b);
  v1 = _mm_shuffle_epi32(v4, 0x1b);
  v0 = _mm_add_epi32(v0, v4);
  v1 = _mm_sub_epi32(v1, v5);

  v3 = _mm_sign_epi32(v7, c0);
  v7 = _mm_shuffle_epi32(v7, 0x1b);
  v7 = _mm_add_epi32(v7, v3);
  v7 = _mm_mullo_epi32(v7, c1);
  v7 = _mm_add_epi32(v7, kRnd0);
  v7 = _mm_srai_epi32(v7, kPrecision);
  v2 = _mm_alignr_epi8(v7, v6, 8);
  v3 = _mm_alignr_epi8(v6, v7, 8);

  // stage 7
  v4 = _mm_shuffle_epi32(v3, 0x1b);
  v5 = _mm_shuffle_epi32(v2, 0x1b);
  v6 = _mm_shuffle_epi32(v1, 0x1b);
  v7 = _mm_shuffle_epi32(v0, 0x1b);
  v4 = _mm_add_epi32(v4, v0);
  v5 = _mm_add_epi32(v5, v1);
  v6 = _mm_sub_epi32(v6, v2);
  v7 = _mm_sub_epi32(v7, v3);

  v4 = _mm_mullo_epi32(v4, c1);
  v5 = _mm_mullo_epi32(v5, c1);
  v6 = _mm_mullo_epi32(v6, c1);
  v7 = _mm_mullo_epi32(v7, c1);
  v4 = _mm_add_epi32(v4, kRnd1);
  v5 = _mm_add_epi32(v5, kRnd1);
  v6 = _mm_add_epi32(v6, kRnd1);
  v7 = _mm_add_epi32(v7, kRnd1);
  v4 = _mm_srai_epi32(v4, shift);
  v5 = _mm_srai_epi32(v5, shift);
  v6 = _mm_srai_epi32(v6, shift);
  v7 = _mm_srai_epi32(v7, shift);

  Store_s32_s16(v4, coeffs + 0);
  Store_s32_s16(v5, coeffs + 4);
  Store_s32_s16(v6, coeffs + 8);
  Store_s32_s16(v7, coeffs + 12);
}

//------------------------------------------------------------------------------
// row-wise fdct

static void fdct4_SSE(int32_t* coeffs) {
  constexpr uint32_t rounding = (1 << kPrecision) >> 1;
  const __m128i kRnd0 = _mm_set1_epi32(rounding);
  __m128i v0, v1, c0, c1;

  // stage 1
  v0 = Load128b(coeffs);
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  v1 = _mm_sign_epi32(v0, c0);
  v0 = _mm_shuffle_epi32(v0, 0x1b);
  v0 = _mm_sub_epi32(v0, v1);

  // stage 2
  c0 = _mm_setr_epi32(rounding, -rounding, kCos32Cos48, kCos32Cos48);
  c1 = _mm_setr_epi32(rounding, rounding, kCos32Cos16, -kCos32Cos16);
  v1 = _mm_shuffle_epi32(v0, 0xb1);
  v0 = _mm_mullo_epi32(v0, c0);
  v1 = _mm_mullo_epi32(v1, c1);
  v0 = _mm_add_epi32(v0, v1);
  v0 = _mm_add_epi32(v0, kRnd0);
  v0 = _mm_srai_epi32(v0, kPrecision);

  // stage 3
  v0 = _mm_shuffle_epi32(v0, 0xd8);
  Store128b(v0, coeffs);
}

static void fdct8_SSE(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, c0, c1;

  // stage 1
  v0 = Load128b(coeffs + 0);
  v1 = Load128b(coeffs + 4);
  v3 = _mm_shuffle_epi32(v0, 0x1b);
  v2 = _mm_shuffle_epi32(v1, 0x1b);
  v3 = _mm_sub_epi32(v3, v1);
  v2 = _mm_add_epi32(v2, v0);

  // stage 2
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  c1 = _mm_set1_epi32(kCosPi[32]);
  v0 = _mm_sign_epi32(v2, c0);
  v2 = _mm_shuffle_epi32(v2, 0x1b);
  v2 = _mm_sub_epi32(v2, v0);

  v5 = _mm_shuffle_epi32(v3, 0x1b);
  v4 = _mm_sign_epi32(v3, c0);
  v5 = _mm_add_epi32(v5, v4);
  v5 = _mm_mullo_epi32(v5, c1);
  v5 = _mm_add_epi32(v5, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v5 = _mm_shuffle_epi32(v5, 0xa5);

  // stage 3
  c0 = _mm_setr_epi32(1, -1, -1, 1);
  v3 = _mm_shuffle_epi32(v3, 0xf0);
  v5 = _mm_sign_epi32(v5, c0);
  v3 = _mm_add_epi32(v3, v5);

  c0 = _mm_setr_epi32(kCosPi[32], -kCosPi[32], kCosPi[48], kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[32], kCosPi[32], kCosPi[16], -kCosPi[16]);
  v4 = _mm_shuffle_epi32(v2, 0xb1);
  v2 = _mm_mullo_epi32(v2, c0);
  v4 = _mm_mullo_epi32(v4, c1);
  v2 = _mm_add_epi32(v2, v4);
  v2 = _mm_add_epi32(v2, kRnd1);
  v2 = _mm_srai_epi32(v2, shift);

  // stage 4
  c0 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[24], kCosPi[56]);
  c1 = _mm_setr_epi32(kCosPi[8], kCosPi[40], -kCosPi[40], -kCosPi[8]);
  v0 = _mm_shuffle_epi32(v3, 0x1b);
  v3 = _mm_mullo_epi32(v3, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v3 = _mm_add_epi32(v3, v0);
  v3 = _mm_add_epi32(v3, kRnd1);
  v3 = _mm_srai_epi32(v3, shift);

  // stage 5
  v1 = _mm_shuffle_epi32(v2, 0xb1);
  v0 = _mm_shuffle_epi32(v3, 0xb1);
  v1 = _mm_blend_epi16(v1, v3, 0xcc);
  v0 = _mm_blend_epi16(v0, v2, 0x33);

  Store128b(v0, coeffs + 0);
  Store128b(v1, coeffs + 4);
}

static void fdct16_SSE(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, v6, v7, c0, c1;

  // stage 1
  v0 = Load128b(coeffs + 0);
  v1 = Load128b(coeffs + 4);
  v2 = Load128b(coeffs + 8);
  v3 = Load128b(coeffs + 12);
  v4 = _mm_shuffle_epi32(v0, 0x1b);
  v5 = _mm_shuffle_epi32(v1, 0x1b);
  v6 = _mm_shuffle_epi32(v2, 0x1b);
  v7 = _mm_shuffle_epi32(v3, 0x1b);
  v1 = _mm_add_epi32(v1, v6);
  v2 = _mm_sub_epi32(v5, v2);
  v0 = _mm_add_epi32(v0, v7);
  v3 = _mm_sub_epi32(v4, v3);

  // stage 2
  v7 = _mm_shuffle_epi32(v1, 0x1b);
  v6 = _mm_shuffle_epi32(v0, 0x1b);
  v4 = _mm_add_epi32(v0, v7);
  v5 = _mm_sub_epi32(v6, v1);
  v6 = _mm_alignr_epi8(v3, v2, 8);  // abcd
  v3 = _mm_alignr_epi8(v2, v3, 8);  // ef89
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  v7 = _mm_shuffle_epi32(v6, 0x1b);
  v6 = _mm_sign_epi32(v6, c0);
  c1 = _mm_set1_epi32(kCosPi[32]);
  v6 = _mm_add_epi32(v6, v7);
  v6 = _mm_mullo_epi32(v6, c1);
  v6 = _mm_add_epi32(v6, kRnd0);
  v6 = _mm_srai_epi32(v6, kPrecision);

  // stage 3
  v0 = _mm_shuffle_epi32(v6, 0x1b);
  v1 = _mm_shuffle_epi32(v3, 0x1b);
  v0 = _mm_add_epi32(v0, v3);
  v1 = _mm_sub_epi32(v1, v6);
  v2 = _mm_shuffle_epi32(v4, 0x1b);
  v3 = _mm_sign_epi32(v4, c0);
  v2 = _mm_sub_epi32(v2, v3);

  v3 = _mm_shuffle_epi32(v5, 0x1b);
  v4 = _mm_sign_epi32(v5, c0);
  v3 = _mm_add_epi32(v3, v4);
  v3 = _mm_mullo_epi32(v3, c1);
  v3 = _mm_add_epi32(v3, kRnd0);
  v3 = _mm_srai_epi32(v3, kPrecision);
  v3 = _mm_shuffle_epi32(v3, 0xa5);

  // stage 4
  c0 = _mm_setr_epi32(1, -1, -1, 1);
  v5 = _mm_shuffle_epi32(v5, 0xf0);
  v3 = _mm_sign_epi32(v3, c0);
  v3 = _mm_add_epi32(v3, v5);

  c0 = _mm_setr_epi32(kCosPi[32], -kCosPi[32], kCosPi[48], kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[32], kCosPi[32], kCosPi[16], -kCosPi[16]);
  v4 = _mm_shuffle_epi32(v2, 0xb1);
  v2 = _mm_mullo_epi32(v2, c0);
  v4 = _mm_mullo_epi32(v4, c1);
  v2 = _mm_add_epi32(v2, v4);
  v2 = _mm_add_epi32(v2, kRnd0);
  v2 = _mm_srai_epi32(v2, kPrecision);

  v0 = _mm_shuffle_epi32(v0, 0xb1);
  v4 = _mm_blend_epi16(v0, v1, 0x3c);  // fbc8
  v5 = _mm_blend_epi16(v0, v1, 0xc3);  // ae9d

  c0 = _mm_setr_epi32(-kCosPi[48], kCosPi[16], -kCosPi[16], kCosPi[48]);
  c1 = _mm_setr_epi32(-kCosPi[16], kCosPi[48], kCosPi[48], -kCosPi[16]);
  v0 = _mm_shuffle_epi32(v5, 0x1b);
  v5 = _mm_mullo_epi32(v5, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v5 = _mm_add_epi32(v5, v0);
  v5 = _mm_add_epi32(v5, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);

  // stage 5
  c0 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[24], kCosPi[56]);
  c1 = _mm_setr_epi32(kCosPi[8], kCosPi[40], -kCosPi[40], -kCosPi[8]);
  v0 = _mm_shuffle_epi32(v3, 0x1b);
  v3 = _mm_mullo_epi32(v3, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v3 = _mm_add_epi32(v3, v0);
  v3 = _mm_add_epi32(v3, kRnd0);
  v3 = _mm_srai_epi32(v3, kPrecision);

  v0 = _mm_shuffle_epi32(v5, 0xb1);
  v5 = _mm_sub_epi32(v4, v0);  // ead9
  v4 = _mm_add_epi32(v4, v0);  // fbc8

  // stage 6
  v0 = _mm_shuffle_epi32(v5, 0x4e);  // d9ea
  v5 = _mm_unpacklo_epi64(v4, v0);   // fbd9
  v4 = _mm_unpackhi_epi64(v0, v4);   // eac8
  v4 = _mm_shuffle_epi32(v4, 0x1b);  // 8cae

  c0 = _mm_setr_epi32(kCosPi[4], -kCosPi[52], kCosPi[20], -kCosPi[36]);
  c1 = _mm_setr_epi32(kCosPi[60], kCosPi[12], kCosPi[44], kCosPi[28]);
  v0 = _mm_mullo_epi32(v4, c0);
  v1 = _mm_mullo_epi32(v5, c0);
  v4 = _mm_mullo_epi32(v4, c1);
  v5 = _mm_mullo_epi32(v5, c1);
  v4 = _mm_add_epi32(v4, v1);
  v5 = _mm_sub_epi32(v5, v0);
  v4 = _mm_add_epi32(v4, kRnd0);
  v5 = _mm_add_epi32(v5, kRnd0);
  v0 = _mm_srai_epi32(v4, kPrecision);
  v1 = _mm_srai_epi32(v5, kPrecision);
  v1 = _mm_shuffle_epi32(v1, 0x1b);

  // stage 7
  c0 = _mm_set1_epi32(kCosPi[32]);
  v2 = _mm_mullo_epi32(v2, c0);
  v3 = _mm_mullo_epi32(v3, c0);
  v0 = _mm_mullo_epi32(v0, c0);
  v1 = _mm_mullo_epi32(v1, c0);
  v2 = _mm_add_epi32(v2, kRnd1);
  v3 = _mm_add_epi32(v3, kRnd1);
  v0 = _mm_add_epi32(v0, kRnd1);
  v1 = _mm_add_epi32(v1, kRnd1);
  v2 = _mm_srai_epi32(v2, shift);
  v3 = _mm_srai_epi32(v3, shift);
  v0 = _mm_srai_epi32(v0, shift);
  v1 = _mm_srai_epi32(v1, shift);

  v4 = _mm_unpacklo_epi32(v2, v3);  // 0415
  v5 = _mm_unpackhi_epi32(v2, v3);  // 2637
  v6 = _mm_unpacklo_epi32(v4, v0);  // 084c
  v7 = _mm_unpackhi_epi32(v5, v1);  // 3b7f
  v0 = _mm_shuffle_epi32(v0, 0x4e);
  v1 = _mm_shuffle_epi32(v1, 0x4e);
  v5 = _mm_unpacklo_epi32(v5, v0);  // 2a6e
  v4 = _mm_unpackhi_epi32(v4, v1);  // 195d

  Store128b(v6, coeffs + 0);
  Store128b(v5, coeffs + 4);
  Store128b(v4, coeffs + 8);
  Store128b(v7, coeffs + 12);
}

void fdct32_SSE(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 2;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, v6, v7, c0, c1;
  int32_t temp[16];

  // stage 1
  v0 = Load128b(coeffs + 0);
  v1 = Load128b(coeffs + 4);
  v2 = Load128b(coeffs + 8);
  v3 = Load128b(coeffs + 12);
  v4 = Load128b(coeffs + 16);
  v5 = Load128b(coeffs + 20);
  v6 = Load128b(coeffs + 24);
  v7 = Load128b(coeffs + 28);
  v6 = _mm_shuffle_epi32(v6, 0x1b);
  v7 = _mm_shuffle_epi32(v7, 0x1b);
  v4 = _mm_shuffle_epi32(v4, 0x1b);
  v5 = _mm_shuffle_epi32(v5, 0x1b);
  v0 = _mm_add_epi32(v0, v7);
  v1 = _mm_add_epi32(v1, v6);
  v2 = _mm_add_epi32(v2, v5);
  v3 = _mm_add_epi32(v3, v4);

  // stage 2
  v4 = _mm_shuffle_epi32(v3, 0x1b);
  v5 = _mm_shuffle_epi32(v2, 0x1b);
  v6 = _mm_shuffle_epi32(v1, 0x1b);
  v7 = _mm_shuffle_epi32(v0, 0x1b);
  v4 = _mm_add_epi32(v4, v0);
  v5 = _mm_add_epi32(v5, v1);
  v6 = _mm_sub_epi32(v6, v2);
  v7 = _mm_sub_epi32(v7, v3);

  // stage 3
  v0 = _mm_shuffle_epi32(v5, 0x1b);
  v1 = _mm_shuffle_epi32(v4, 0x1b);
  v0 = _mm_add_epi32(v0, v4);
  v1 = _mm_sub_epi32(v1, v5);

  v2 = _mm_alignr_epi8(v6, v7, 8);  // ef89
  v3 = _mm_alignr_epi8(v7, v6, 8);  // abcd
  c0 = _mm_setr_epi32(-1, -1, 1, 1);
  c1 = _mm_set1_epi32(kCosPi[32]);
  v4 = _mm_sign_epi32(v3, c0);
  v3 = _mm_shuffle_epi32(v3, 0x1b);
  v3 = _mm_add_epi32(v3, v4);
  v3 = _mm_mullo_epi32(v3, c1);
  v3 = _mm_add_epi32(v3, kRnd0);
  v3 = _mm_srai_epi32(v3, kPrecision);

  // stage 4
  v4 = _mm_shuffle_epi32(v0, 0x1b);
  v0 = _mm_sign_epi32(v0, c0);
  v4 = _mm_sub_epi32(v4, v0);

  v6 = _mm_shuffle_epi32(v3, 0x1b);
  v7 = _mm_shuffle_epi32(v2, 0x1b);
  v6 = _mm_add_epi32(v6, v2);
  v7 = _mm_sub_epi32(v7, v3);

  v5 = _mm_shuffle_epi32(v1, 0x1b);
  v2 = _mm_sign_epi32(v1, c0);
  v5 = _mm_add_epi32(v5, v2);
  v5 = _mm_mullo_epi32(v5, c1);
  v5 = _mm_add_epi32(v5, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v5 = _mm_shuffle_epi32(v5, 0xa5);

  // stage 5
  c0 = _mm_setr_epi32(1, -1, -1, 1);
  v1 = _mm_shuffle_epi32(v1, 0xf0);
  v5 = _mm_sign_epi32(v5, c0);
  v1 = _mm_add_epi32(v1, v5);

  v6 = _mm_shuffle_epi32(v6, 0xb1);    // fe98
  v2 = _mm_blend_epi16(v6, v7, 0x3c);  // fbc8
  v3 = _mm_blend_epi16(v6, v7, 0xc3);  // ae9d

  c0 = _mm_setr_epi32(-kCosPi[48], kCosPi[16], -kCosPi[16], kCosPi[48]);
  c1 = _mm_setr_epi32(-kCosPi[16], kCosPi[48], kCosPi[48], -kCosPi[16]);
  v0 = _mm_shuffle_epi32(v3, 0x1b);
  v3 = _mm_mullo_epi32(v3, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v3 = _mm_add_epi32(v3, v0);
  v3 = _mm_add_epi32(v3, kRnd0);
  v3 = _mm_srai_epi32(v3, kPrecision);

  c0 = _mm_setr_epi32(kCosPi[32], -kCosPi[32], kCosPi[48], kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[32], kCosPi[32], kCosPi[16], -kCosPi[16]);
  v0 = _mm_shuffle_epi32(v4, 0xb1);
  v4 = _mm_mullo_epi32(v4, c0);
  v0 = _mm_mullo_epi32(v0, c1);
  v0 = _mm_add_epi32(v0, v4);
  v0 = _mm_add_epi32(v0, kRnd1);
  v0 = _mm_srai_epi32(v0, shift);

  // stage 6
  c0 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[24], kCosPi[56]);
  c1 = _mm_setr_epi32(kCosPi[8], kCosPi[40], -kCosPi[40], -kCosPi[8]);
  v4 = _mm_shuffle_epi32(v1, 0x1b);
  v1 = _mm_mullo_epi32(v1, c0);
  v4 = _mm_mullo_epi32(v4, c1);
  v1 = _mm_add_epi32(v1, v4);
  v1 = _mm_add_epi32(v1, kRnd1);
  v1 = _mm_srai_epi32(v1, shift);

  v7 = _mm_shuffle_epi32(v2, 0xb1);
  v6 = _mm_add_epi32(v3, v7);  // bf8c
  v7 = _mm_sub_epi32(v7, v3);  // ae9d

  // stage 7
  v6 = _mm_shuffle_epi32(v6, 0x4e);  // 8cbf
  v3 = _mm_unpackhi_epi64(v7, v6);   // 9dbf
  v2 = _mm_unpacklo_epi64(v6, v7);   // 8cae
  v3 = _mm_shuffle_epi32(v3, 0x1b);  // fbd9

  c0 = _mm_setr_epi32(kCosPi[4], -kCosPi[52], kCosPi[20], -kCosPi[36]);
  c1 = _mm_setr_epi32(kCosPi[60], kCosPi[12], kCosPi[44], kCosPi[28]);
  v4 = _mm_mullo_epi32(v2, c0);
  v5 = _mm_mullo_epi32(v3, c0);
  v2 = _mm_mullo_epi32(v2, c1);
  v3 = _mm_mullo_epi32(v3, c1);
  v2 = _mm_add_epi32(v2, v5);
  v3 = _mm_sub_epi32(v3, v4);
  v2 = _mm_add_epi32(v2, kRnd1);
  v3 = _mm_add_epi32(v3, kRnd1);
  v2 = _mm_srai_epi32(v2, shift);
  v3 = _mm_srai_epi32(v3, shift);
  v3 = _mm_shuffle_epi32(v3, 0x1b);

  // stage 8

  // stage 9
  v4 = _mm_unpacklo_epi32(v0, v1);  // 0415
  v5 = _mm_unpackhi_epi32(v0, v1);  // 2637
  v6 = _mm_unpacklo_epi32(v4, v2);  // 084c
  v7 = _mm_unpackhi_epi32(v5, v3);  // 3b7f
  v2 = _mm_shuffle_epi32(v2, 0x4e);
  v3 = _mm_shuffle_epi32(v3, 0x4e);
  v5 = _mm_unpacklo_epi32(v5, v2);  // 2a6e
  v4 = _mm_unpackhi_epi32(v4, v3);  // 195d

  Store128b(v6, temp + 0);
  Store128b(v5, temp + 4);
  Store128b(v4, temp + 8);
  Store128b(v7, temp + 12);

  // stage 1
  v3 = Load128b(coeffs + 0);
  v2 = Load128b(coeffs + 4);
  v1 = Load128b(coeffs + 8);
  v0 = Load128b(coeffs + 12);
  v4 = Load128b(coeffs + 16);
  v5 = Load128b(coeffs + 20);
  v6 = Load128b(coeffs + 24);
  v7 = Load128b(coeffs + 28);
  v0 = _mm_shuffle_epi32(v0, 0x1b);
  v1 = _mm_shuffle_epi32(v1, 0x1b);
  v2 = _mm_shuffle_epi32(v2, 0x1b);
  v3 = _mm_shuffle_epi32(v3, 0x1b);
  v0 = _mm_sub_epi32(v0, v4);
  v1 = _mm_sub_epi32(v1, v5);
  v2 = _mm_sub_epi32(v2, v6);
  v3 = _mm_sub_epi32(v3, v7);

  // stage 2
  c1 = _mm_set1_epi32(kCosPi[32]);
  v5 = _mm_shuffle_epi32(v2, 0x1b);
  v6 = _mm_shuffle_epi32(v1, 0x1b);
  v5 = _mm_sub_epi32(v5, v1);
  v6 = _mm_add_epi32(v6, v2);
  v5 = _mm_mullo_epi32(v5, c1);
  v6 = _mm_mullo_epi32(v6, c1);
  v5 = _mm_add_epi32(v5, kRnd0);
  v6 = _mm_add_epi32(v6, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v6 = _mm_srai_epi32(v6, kPrecision);

  // stage 3
  v4 = _mm_shuffle_epi32(v5, 0x1b);
  v1 = _mm_shuffle_epi32(v0, 0x1b);
  v2 = _mm_shuffle_epi32(v3, 0x1b);
  v7 = _mm_shuffle_epi32(v6, 0x1b);
  v0 = _mm_add_epi32(v0, v4);
  v1 = _mm_sub_epi32(v1, v5);
  v2 = _mm_sub_epi32(v2, v6);
  v3 = _mm_add_epi32(v3, v7);

  // stage 4
  v4 = _mm_unpacklo_epi32(v0, v2);  // 0819
  v5 = _mm_unpackhi_epi32(v0, v2);  // 2a3b
  v6 = _mm_unpacklo_epi32(v1, v3);  // 4c5d
  v7 = _mm_unpackhi_epi32(v1, v3);  // 6e7f
  c0 = _mm_setr_epi32(-kCosPi[16], kCosPi[48], -kCosPi[16], kCosPi[48]);
  c1 = _mm_setr_epi32(-kCosPi[48], kCosPi[16], -kCosPi[48], kCosPi[16]);
  v1 = _mm_shuffle_epi32(v6, 0x1b);
  v2 = _mm_shuffle_epi32(v5, 0x1b);
  v5 = _mm_mullo_epi32(v5, c0);
  v6 = _mm_mullo_epi32(v6, c1);
  v1 = _mm_mullo_epi32(v1, c1);
  v2 = _mm_mullo_epi32(v2, c0);
  v5 = _mm_sub_epi32(v5, v1);
  v6 = _mm_add_epi32(v6, v2);
  v5 = _mm_add_epi32(v5, kRnd0);
  v6 = _mm_add_epi32(v6, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v6 = _mm_srai_epi32(v6, kPrecision);

  // stage 5
  v0 = _mm_shuffle_epi32(v5, 0x4e);
  v1 = _mm_shuffle_epi32(v4, 0x4e);
  v2 = _mm_shuffle_epi32(v7, 0x4e);
  v3 = _mm_shuffle_epi32(v6, 0x4e);
  v0 = _mm_add_epi32(v0, v4);
  v1 = _mm_sub_epi32(v1, v5);
  v2 = _mm_sub_epi32(v2, v6);
  v3 = _mm_add_epi32(v3, v7);

  // stage 6
  v4 = _mm_unpacklo_epi64(v0, v2);  // 084c
  v5 = _mm_unpackhi_epi64(v0, v2);  // 195d
  v6 = _mm_unpacklo_epi64(v1, v3);  // 2a6e
  v7 = _mm_unpackhi_epi64(v1, v3);  // 3b7f
  c0 = _mm_setr_epi32(-kCosPi[8], kCosPi[24], -kCosPi[40], kCosPi[56]);
  c1 = _mm_setr_epi32(-kCosPi[56], kCosPi[40], -kCosPi[24], kCosPi[8]);
  v1 = _mm_shuffle_epi32(v6, 0x1b);
  v2 = _mm_shuffle_epi32(v5, 0x1b);
  v5 = _mm_mullo_epi32(v5, c0);
  v6 = _mm_mullo_epi32(v6, c1);
  v1 = _mm_mullo_epi32(v1, c1);
  v2 = _mm_mullo_epi32(v2, c0);
  v5 = _mm_sub_epi32(v5, v1);
  v6 = _mm_add_epi32(v6, v2);
  v5 = _mm_add_epi32(v5, kRnd0);
  v6 = _mm_add_epi32(v6, kRnd0);
  v5 = _mm_srai_epi32(v5, kPrecision);
  v6 = _mm_srai_epi32(v6, kPrecision);

  // stage 7
  v0 = _mm_add_epi32(v4, v5);
  v1 = _mm_sub_epi32(v4, v5);
  v2 = _mm_sub_epi32(v7, v6);
  v3 = _mm_add_epi32(v7, v6);

  // stage 8
  c0 = _mm_setr_epi32(kCosPi[62], kCosPi[6], kCosPi[54], kCosPi[14]);
  c1 = _mm_setr_epi32(kCosPi[2], -kCosPi[58], kCosPi[10], -kCosPi[50]);
  v3 = _mm_shuffle_epi32(v3, 0x1b);
  v4 = _mm_mullo_epi32(v0, c1);
  v7 = _mm_mullo_epi32(v3, c1);
  v0 = _mm_mullo_epi32(v0, c0);
  v3 = _mm_mullo_epi32(v3, c0);
  v0 = _mm_add_epi32(v0, v7);
  v3 = _mm_sub_epi32(v3, v4);
  v0 = _mm_add_epi32(v0, kRnd1);
  v3 = _mm_add_epi32(v3, kRnd1);
  v0 = _mm_srai_epi32(v0, shift);
  v3 = _mm_srai_epi32(v3, shift);
  v3 = _mm_shuffle_epi32(v3, 0x1b);

  c0 = _mm_setr_epi32(kCosPi[30], kCosPi[38], kCosPi[22], kCosPi[46]);
  c1 = _mm_setr_epi32(kCosPi[34], -kCosPi[26], kCosPi[42], -kCosPi[18]);
  v2 = _mm_shuffle_epi32(v2, 0x1b);
  v5 = _mm_mullo_epi32(v1, c1);
  v6 = _mm_mullo_epi32(v2, c1);
  v1 = _mm_mullo_epi32(v1, c0);
  v2 = _mm_mullo_epi32(v2, c0);
  v1 = _mm_add_epi32(v1, v6);
  v2 = _mm_sub_epi32(v2, v5);
  v1 = _mm_add_epi32(v1, kRnd1);
  v2 = _mm_add_epi32(v2, kRnd1);
  v1 = _mm_srai_epi32(v1, shift);
  v2 = _mm_srai_epi32(v2, shift);
  v2 = _mm_shuffle_epi32(v2, 0x1b);

  // stage 9
  v6 = Load128b(temp + 0);
  v7 = Load128b(temp + 4);
  v4 = _mm_unpacklo_epi32(v6, v0);
  v5 = _mm_unpackhi_epi32(v6, v0);
  v6 = _mm_unpacklo_epi32(v7, v2);
  v7 = _mm_unpackhi_epi32(v7, v2);
  v0 = Load128b(temp + 8);
  v2 = Load128b(temp + 12);

  Store128b(v4, coeffs + 0);
  Store128b(v5, coeffs + 4);
  Store128b(v6, coeffs + 8);
  Store128b(v7, coeffs + 12);

  v4 = _mm_unpacklo_epi32(v0, v1);
  v5 = _mm_unpackhi_epi32(v0, v1);
  v6 = _mm_unpacklo_epi32(v2, v3);
  v7 = _mm_unpackhi_epi32(v2, v3);

  Store128b(v4, coeffs + 16);
  Store128b(v5, coeffs + 20);
  Store128b(v6, coeffs + 24);
  Store128b(v7, coeffs + 28);
}

static void fadst8_SSE(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, c0, c1;

  // stage 1
  c0 = _mm_setr_epi32(1, -1, 1, -1);
  v0 = Load128b(coeffs + 0);
  v1 = Load128b(coeffs + 4);
  v0 = _mm_sign_epi32(v0, c0);
  v1 = _mm_sign_epi32(v1, c0);
  v0 = _mm_shuffle_epi32(v0, 0xb4);
  v1 = _mm_shuffle_epi32(v1, 0x4b);

  // stage 2
  v3 = _mm_unpackhi_epi64(v0, v1);
  v2 = _mm_unpacklo_epi32(v0, v1);
  v5 = _mm_unpackhi_epi32(v3, v3);
  v4 = _mm_unpacklo_epi32(v3, v3);
  v5 = _mm_sign_epi32(v5, c0);
  c0 = _mm_set1_epi32(kCosPi[32]);

  v4 = _mm_add_epi32(v4, v5);
  v4 = _mm_mullo_epi32(v4, c0);
  v4 = _mm_add_epi32(v4, kRnd0);
  v4 = _mm_srai_epi32(v4, kPrecision);

  // stage 3
  v3 = _mm_add_epi32(v2, v4);
  v5 = _mm_sub_epi32(v2, v4);
  v0 = _mm_unpacklo_epi64(v3, v5);
  v1 = _mm_unpackhi_epi64(v3, v5);

  // stage 4
  v2 = _mm_shuffle_epi32(v1, 0xa0);
  v3 = _mm_shuffle_epi32(v1, 0xf5);
  c0 = _mm_setr_epi32(kCosPi[16], kCosPi[48], -kCosPi[48], kCosPi[16]);
  c1 = _mm_setr_epi32(kCosPi[48], -kCosPi[16], kCosPi[16], kCosPi[48]);
  v2 = _mm_mullo_epi32(v2, c0);
  v3 = _mm_mullo_epi32(v3, c1);
  v1 = _mm_add_epi32(v2, v3);
  v1 = _mm_add_epi32(v1, kRnd0);
  v1 = _mm_srai_epi32(v1, kPrecision);

  // stage 5
  v2 = _mm_add_epi32(v0, v1);
  v3 = _mm_sub_epi32(v0, v1);

  // stage 6
  v0 = _mm_unpacklo_epi32(v2, v3);
  v1 = _mm_unpackhi_epi32(v2, v3);
  v2 = _mm_unpacklo_epi32(v0, v1);
  v3 = _mm_unpackhi_epi32(v0, v1);

  c0 = _mm_setr_epi32(kCosPi[4], kCosPi[20], kCosPi[36], kCosPi[52]);
  c1 = _mm_setr_epi32(kCosPi[60], kCosPi[44], kCosPi[28], kCosPi[12]);
  v0 = _mm_mullo_epi32(v2, c0);
  v1 = _mm_mullo_epi32(v2, c1);
  v4 = _mm_mullo_epi32(v3, c1);
  v5 = _mm_mullo_epi32(v3, c0);

  v2 = _mm_add_epi32(v0, v4);
  v3 = _mm_sub_epi32(v1, v5);
  v2 = _mm_add_epi32(v2, kRnd1);
  v3 = _mm_add_epi32(v3, kRnd1);
  v2 = _mm_srai_epi32(v2, shift);
  v3 = _mm_srai_epi32(v3, shift);

  // stage 7
  v2 = _mm_shuffle_epi32(v2, 0x1b);
  v0 = _mm_unpacklo_epi32(v3, v2);
  v1 = _mm_unpackhi_epi32(v3, v2);

  Store128b(v0, coeffs + 0);
  Store128b(v1, coeffs + 4);
}

//------------------------------------------------------------------------------
// row-wise fadst

#define HALF_BTF_VEC(v0, v1, t0, t1, t2, t3) \
  t0 = _mm_mullo_epi32(v0, c0);              \
  t1 = _mm_mullo_epi32(v0, c1);              \
  t2 = _mm_mullo_epi32(v1, c1);              \
  t3 = _mm_mullo_epi32(v1, c0);              \
  v0 = _mm_add_epi32(t0, t2);                \
  v1 = _mm_sub_epi32(t1, t3);                \
  v0 = _mm_add_epi32(v0, kRnd0);             \
  v1 = _mm_add_epi32(v1, kRnd0);             \
  v0 = _mm_srai_epi32(v0, kPrecision);       \
  v1 = _mm_srai_epi32(v1, kPrecision)

static void fadst16_SSE(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  const __m128i kRnd0 = _mm_set1_epi32((1 << kPrecision) >> 1);
  const __m128i kRnd1 = _mm_set1_epi32((1 << shift) >> 1);
  __m128i v0, v1, v2, v3, v4, v5, v6, v7, c0, c1;

  // stage 1
  c0 = _mm_setr_epi32(1, -1, 1, -1);
  v0 = Load128b(coeffs + 0);
  v2 = Load128b(coeffs + 4);
  v3 = Load128b(coeffs + 8);
  v1 = Load128b(coeffs + 12);
  v0 = _mm_sign_epi32(v0, c0);
  v1 = _mm_sign_epi32(v1, c0);
  v2 = _mm_sign_epi32(v2, c0);
  v3 = _mm_sign_epi32(v3, c0);
  v0 = _mm_shuffle_epi32(v0, 0x9c);  // 048c
  v1 = _mm_shuffle_epi32(v1, 0x63);  // 159d
  v2 = _mm_shuffle_epi32(v2, 0x63);  // 26ae
  v3 = _mm_shuffle_epi32(v3, 0x9c);  // 37bf

  // stage 2
  c0 = _mm_set1_epi32(kCosPi[32]);
  v4 = _mm_add_epi32(v2, v3);
  v5 = _mm_sub_epi32(v2, v3);
  v4 = _mm_mullo_epi32(v4, c0);
  v5 = _mm_mullo_epi32(v5, c0);
  v4 = _mm_add_epi32(v4, kRnd0);
  v5 = _mm_add_epi32(v5, kRnd0);
  v4 = _mm_srai_epi32(v4, kPrecision);
  v5 = _mm_srai_epi32(v5, kPrecision);

  // stage 3
  v2 = _mm_add_epi32(v0, v4);
  v3 = _mm_add_epi32(v1, v5);
  v4 = _mm_sub_epi32(v0, v4);
  v5 = _mm_sub_epi32(v1, v5);

  // stage 4
  v6 = _mm_unpacklo_epi32(v2, v4);
  v7 = _mm_unpacklo_epi32(v3, v5);
  v4 = _mm_unpackhi_epi32(v2, v4);
  v5 = _mm_unpackhi_epi32(v3, v5);

  v0 = _mm_unpacklo_epi64(v6, v4);  // 028a
  v1 = _mm_unpackhi_epi64(v6, v4);  // 46ce
  v2 = _mm_unpacklo_epi64(v7, v5);  // 139b
  v3 = _mm_unpackhi_epi64(v7, v5);  // 57df

  c0 = _mm_setr_epi32(kCosPi[16], -kCosPi[48], kCosPi[16], -kCosPi[48]);
  c1 = _mm_setr_epi32(kCosPi[48], kCosPi[16], kCosPi[48], kCosPi[16]);
  HALF_BTF_VEC(v1, v3, v4, v5, v6, v7);

  // stage 5
  v4 = _mm_add_epi32(v0, v1);
  v5 = _mm_sub_epi32(v0, v1);
  v6 = _mm_add_epi32(v2, v3);
  v7 = _mm_sub_epi32(v2, v3);

  // stage 6
  v0 = _mm_unpacklo_epi64(v4, v5);  // 0246
  v1 = _mm_unpackhi_epi64(v4, v5);  // 8ace
  v2 = _mm_unpacklo_epi64(v6, v7);  // 1357
  v3 = _mm_unpackhi_epi64(v6, v7);  // 9bdf

  c0 = _mm_setr_epi32(kCosPi[8], kCosPi[40], -kCosPi[56], -kCosPi[24]);
  c1 = _mm_setr_epi32(kCosPi[56], kCosPi[24], kCosPi[8], kCosPi[40]);
  HALF_BTF_VEC(v1, v3, v4, v5, v6, v7);

  // stage 7
  v4 = _mm_add_epi32(v0, v1);
  v5 = _mm_sub_epi32(v0, v1);
  v6 = _mm_add_epi32(v2, v3);
  v7 = _mm_sub_epi32(v2, v3);

  // stage 8
  c0 = _mm_setr_epi32(kCosPi[2], kCosPi[10], kCosPi[18], kCosPi[26]);
  c1 = _mm_setr_epi32(kCosPi[62], kCosPi[54], kCosPi[46], kCosPi[38]);
  HALF_BTF_VEC(v4, v6, v0, v1, v2, v3);
  c0 = _mm_setr_epi32(kCosPi[34], kCosPi[42], kCosPi[50], kCosPi[58]);
  c1 = _mm_setr_epi32(kCosPi[30], kCosPi[22], kCosPi[14], kCosPi[6]);
  HALF_BTF_VEC(v5, v7, v0, v1, v2, v3);

  // stage 9
  c0 = _mm_set1_epi32(kCosPi[32]);
  v4 = _mm_mullo_epi32(v4, c0);
  v5 = _mm_mullo_epi32(v5, c0);
  v6 = _mm_mullo_epi32(v6, c0);
  v7 = _mm_mullo_epi32(v7, c0);
  v4 = _mm_add_epi32(v4, kRnd1);
  v5 = _mm_add_epi32(v5, kRnd1);
  v6 = _mm_add_epi32(v6, kRnd1);
  v7 = _mm_add_epi32(v7, kRnd1);
  v4 = _mm_srai_epi32(v4, shift);
  v5 = _mm_srai_epi32(v5, shift);
  v6 = _mm_srai_epi32(v6, shift);
  v7 = _mm_srai_epi32(v7, shift);

  v4 = _mm_shuffle_epi32(v4, 0x1b);  // 6420
  v5 = _mm_shuffle_epi32(v5, 0x1b);  // eca8
  v0 = _mm_unpacklo_epi32(v6, v5);
  v1 = _mm_unpackhi_epi32(v6, v5);
  v2 = _mm_unpacklo_epi32(v7, v4);
  v3 = _mm_unpackhi_epi32(v7, v4);

  Store128b(v0, coeffs + 0);
  Store128b(v1, coeffs + 4);
  Store128b(v2, coeffs + 8);
  Store128b(v3, coeffs + 12);
}
#undef HALF_BTF_VEC

//------------------------------------------------------------------------------

inline void Transpose_4x4_32(const __m128i* const in0, const __m128i* const in1,
                             const __m128i* const in2, const __m128i* const in3,
                             __m128i* const out0, __m128i* const out1,
                             __m128i* const out2, __m128i* const out3) {
  const __m128i t_0 = _mm_unpacklo_epi32(*in0, *in1);
  const __m128i t_1 = _mm_unpacklo_epi32(*in2, *in3);
  const __m128i t_2 = _mm_unpackhi_epi32(*in0, *in1);
  const __m128i t_3 = _mm_unpackhi_epi32(*in2, *in3);
  *out0 = _mm_unpacklo_epi64(t_0, t_1);
  *out1 = _mm_unpackhi_epi64(t_0, t_1);
  *out2 = _mm_unpacklo_epi64(t_2, t_3);
  *out3 = _mm_unpackhi_epi64(t_2, t_3);
}

void Transpose32_SSE(const int32_t* in, uint32_t w, uint32_t h, int32_t* out) {
  assert(in != out);
  if (w == 2 || h == 2) {
    // fallback to C-impl for these cases
    // TODO(skal): remove need for 2x2 transforms
    for (size_t j = 0; j < h; ++j) {
      for (size_t i = 0; i < w; ++i) {
        out[j + h * i] = in[i + w * j];
      }
    }
    return;
  }
  for (size_t j = 0; j < h; j += 4) {
    const int32_t* const ptr0 = &in[w * (j + 0)];
    const int32_t* const ptr1 = &in[w * (j + 1)];
    const int32_t* const ptr2 = &in[w * (j + 2)];
    const int32_t* const ptr3 = &in[w * (j + 3)];
    for (size_t i = 0; i < w; i += 4) {
      __m128i out0, out1, out2, out3;
      const __m128i in0 = Load128b(ptr0 + i);
      const __m128i in1 = Load128b(ptr1 + i);
      const __m128i in2 = Load128b(ptr2 + i);
      const __m128i in3 = Load128b(ptr3 + i);
      Transpose_4x4_32(&in0, &in1, &in2, &in3, &out0, &out1, &out2, &out3);
      Store128b(out0, &out[j + h * (i + 0)]);
      Store128b(out1, &out[j + h * (i + 1)]);
      Store128b(out2, &out[j + h * (i + 2)]);
      Store128b(out3, &out[j + h * (i + 3)]);
    }
  }
}

inline void Transpose_4x4_16(const __m128i* const in0, const __m128i* const in1,
                             const __m128i* const in2, const __m128i* const in3,
                             __m128i* const out0, __m128i* const out1) {
  const __m128i t_0 = _mm_unpacklo_epi16(*in0, *in1);
  const __m128i t_1 = _mm_unpacklo_epi16(*in2, *in3);
  *out0 = _mm_unpacklo_epi32(t_0, t_1);
  *out1 = _mm_unpackhi_epi32(t_0, t_1);
}

void Transpose16_SSE(const int16_t* in, uint32_t w, uint32_t h, int16_t* out) {
  assert(in != out);
  if (w == 2 || h == 2) {
    // fallback to C-impl for these cases
    // TODO(skal): remove need for 2x2 transforms
    for (size_t j = 0; j < h; ++j) {
      for (size_t i = 0; i < w; ++i) {
        out[j + h * i] = in[i + w * j];
      }
    }
    return;
  }
  for (size_t j = 0; j < h; j += 4) {
    const int16_t* const ptr0 = &in[w * (j + 0)];
    const int16_t* const ptr1 = &in[w * (j + 1)];
    const int16_t* const ptr2 = &in[w * (j + 2)];
    const int16_t* const ptr3 = &in[w * (j + 3)];
    for (size_t i = 0; i < w; i += 4) {
      __m128i out0, out1;
      const __m128i in0 = Load64b(ptr0 + i);
      const __m128i in1 = Load64b(ptr1 + i);
      const __m128i in2 = Load64b(ptr2 + i);
      const __m128i in3 = Load64b(ptr3 + i);
      Transpose_4x4_16(&in0, &in1, &in2, &in3, &out0, &out1);
      Store64b(out0, &out[j + h * (i + 0)]);
      Store64b(_mm_srli_si128(out0, 8), &out[j + h * (i + 1)]);
      Store64b(out1, &out[j + h * (i + 2)]);
      Store64b(_mm_srli_si128(out1, 8), &out[j + h * (i + 3)]);
    }
  }
}

//------------------------------------------------------------------------------

// double horizontal-add
int32_t hadd(const __m128 v0, const __m128 v1) {
  float tmp[4];
  _mm_storeu_ps(tmp, _mm_add_ps(v0, v1));
  return (int32_t)lrintf((tmp[0] + tmp[1] + tmp[2] + tmp[3]) * 256.f);
}

int32_t SlowDct8x8_SSE(const int32_t in[64], const float cos_x[8],
                       const float cos_y[8]) {
  __m128 sum0 = _mm_set1_ps(0.f);
  __m128 sum4 = _mm_set1_ps(0.f);
  const __m128 cos_x0 = _mm_loadu_ps(cos_x + 0);
  const __m128 cos_x4 = _mm_loadu_ps(cos_x + 4);
  const __m128i* src = (const __m128i*)in;
  for (uint32_t j = 0; j < 8; ++j, src += 2) {
    const __m128 C = _mm_set1_ps(cos_y[j]);
    const __m128 in0 = _mm_cvtepi32_ps(_mm_loadu_si128(src + 0));
    const __m128 in4 = _mm_cvtepi32_ps(_mm_loadu_si128(src + 1));
    const __m128 a0 = _mm_mul_ps(in0, cos_x0);
    const __m128 a4 = _mm_mul_ps(in4, cos_x4);
    sum0 = _mm_add_ps(sum0, _mm_mul_ps(a0, C));
    sum4 = _mm_add_ps(sum4, _mm_mul_ps(a4, C));
  }
  return hadd(sum0, sum4);
}

//------------------------------------------------------------------------------
}  // namespace

extern void WP2TransformInitSSE();

WP2_TSAN_IGNORE_FUNCTION void WP2TransformInitSSE() {
  WP2Transpose16b = Transpose16_SSE;
  WP2Transpose32b = Transpose32_SSE;

  WP2InvDct[1] = idct4_SSE;
  WP2InvDct[2] = idct8_SSE;
  WP2InvDct[3] = idct16_SSE;
  WP2FwdDct[1] = fdct4_SSE;
  WP2FwdDct[2] = fdct8_SSE;
  WP2FwdDct[3] = fdct16_SSE;
  WP2FwdDct[4] = fdct32_SSE;
  WP2FwdAdst[2] = fadst8_SSE;
  WP2FwdAdst[3] = fadst16_SSE;

  for (int i = 1; i < 5; i++) {
    WP2InvDctCol[0][i] = ColTransformGen<VecI32_SSE>::idct2;
    WP2InvDctCol[1][i] = ColTransformGen<VecI32_SSE>::idct4;
    WP2InvDctCol[2][i] = ColTransformGen<VecI32_SSE>::idct8;

    WP2FwdDctCol[0][i] = ColTransformGen<VecI32_SSE>::fdct2;
    WP2FwdDctCol[1][i] = ColTransformGen<VecI32_SSE>::fdct4;
    WP2FwdDctCol[2][i] = ColTransformGen<VecI32_SSE>::fdct8;
    WP2FwdDctCol[3][i] = ColTransformGen<VecI32_SSE>::fdct16;
  }

  WP2SlowDct8x8 = SlowDct8x8_SSE;
}

#else  // !WP2_USE_SSE

WP2_DSP_INIT_STUB(WP2TransformInitSSE);

#endif  // WP2_USE_SSE
