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
// AV1 Transforms

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"

//------------------------------------------------------------------------------
// Constants for 10bit precision

// kCosPi[j] = (int)round(cos(M_PI*j/128) * (1<<kPrecision));
static constexpr uint32_t kPrecision = 10u;
static constexpr int32_t kCosPi[64] = {
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

//  kSinPi[j] =
//    (int)round((sqrt(2) * sin(j*Pi/9) * 2 / 3) * (1 << kPrecision))
//  modified so that elements j=1,2 sum to element j=4.
static constexpr int32_t kSinPi[5] = {0, 330, 621, 836, 951};
static_assert(kSinPi[1] + kSinPi[2] == kSinPi[4],
              "invalid pre-condition on kSinPi[]");

static inline uint32_t SizeToIdx(uint32_t size) {
  assert(size == 2 || size == 4 || size == 8 || size == 16 || size == 32);
  return (size == 2)    ? 0
         : (size == 4)  ? 1
         : (size == 8)  ? 2
         : (size == 16) ? 3
                        : 4;
}

//------------------------------------------------------------------------------

static inline int32_t round_shift(int64_t value, uint32_t bit = kPrecision) {
  return (int32_t)((value + ((int64_t)1 << bit >> 1)) >> bit);
}
static inline int16_t noround_shift(int64_t value, uint32_t bit = kPrecision) {
  return (int16_t)(value >> bit);
}

static inline int32_t half_btf(int32_t w0, int32_t in0, int32_t w1, int32_t in1,
                               uint32_t precision = kPrecision) {
  const int64_t tmp = (int64_t)w0 * in0 + (int64_t)w1 * in1;
  return round_shift(tmp, precision);
}

static inline int32_t div_sqrt2(int32_t v, uint32_t shift = 0u) {
  return round_shift((int64_t)v * kCosPi[32], kPrecision + shift);
}

//------------------------------------------------------------------------------
// Inverse transforms

static void idct2_C(int16_t* coeffs) {
  const int32_t A = coeffs[0], B = coeffs[1];
  coeffs[0] = div_sqrt2(A + B);
  coeffs[1] = div_sqrt2(A - B);
}

static void idct4_C(int16_t* coeffs) {
  int32_t tmp[4];

  tmp[0] = coeffs[0] + coeffs[2];
  tmp[1] = coeffs[0] - coeffs[2];
  tmp[2] = half_btf(kCos32Cos48p, coeffs[1], -kCos32Cos16p, coeffs[3]);
  tmp[3] = half_btf(kCos32Cos16p, coeffs[1], kCos32Cos48p, coeffs[3]);

  // stage 3
  coeffs[0] = noround_shift(tmp[0] + tmp[3], 1);
  coeffs[1] = noround_shift(tmp[1] + tmp[2], 1);
  coeffs[2] = noround_shift(tmp[1] - tmp[2], 1);
  coeffs[3] = noround_shift(tmp[0] - tmp[3], 1);
}

static void idct8_C(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  int32_t tmp0[8], tmp1[8];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = coeffs[4];
  bf1[2] = coeffs[2];
  bf1[3] = coeffs[6];
  bf1[4] = coeffs[1];
  bf1[5] = coeffs[5];
  bf1[6] = coeffs[3];
  bf1[7] = coeffs[7];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], -kCosPi[8], bf0[7]);
  bf1[5] = half_btf(kCosPi[24], bf0[5], -kCosPi[40], bf0[6]);
  bf1[6] = half_btf(kCosPi[40], bf0[5], kCosPi[24], bf0[6]);
  bf1[7] = half_btf(kCosPi[8], bf0[4], kCosPi[56], bf0[7]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = div_sqrt2(bf0[0] + bf0[1]) + 1;  // +1=rounder
  bf1[1] = div_sqrt2(bf0[0] - bf0[1]) + 1;
  bf1[2] = half_btf(kCosPi[48], bf0[2], -kCosPi[16], bf0[3]);
  bf1[3] = half_btf(kCosPi[16], bf0[2], kCosPi[48], bf0[3]);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = bf0[4] - bf0[5];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[6] + bf0[7];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = bf0[1] - bf0[2];
  bf1[3] = bf0[0] - bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = div_sqrt2(bf0[6] - bf0[5]);
  bf1[6] = div_sqrt2(bf0[6] + bf0[5]);
  bf1[7] = bf0[7];

  // stage 5
  std::swap(bf0, bf1);
  coeffs[0] = noround_shift(bf0[0] + bf0[7], shift - kPrecision);
  coeffs[1] = noround_shift(bf0[1] + bf0[6], shift - kPrecision);
  coeffs[2] = noround_shift(bf0[2] + bf0[5], shift - kPrecision);
  coeffs[3] = noround_shift(bf0[3] + bf0[4], shift - kPrecision);
  coeffs[4] = noround_shift(bf0[3] - bf0[4], shift - kPrecision);
  coeffs[5] = noround_shift(bf0[2] - bf0[5], shift - kPrecision);
  coeffs[6] = noround_shift(bf0[1] - bf0[6], shift - kPrecision);
  coeffs[7] = noround_shift(bf0[0] - bf0[7], shift - kPrecision);
}

static void idct16_C(int16_t* coeffs) {
  int32_t tmp0[16], tmp1[16];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = coeffs[8];
  bf1[2] = coeffs[4];
  bf1[3] = coeffs[12];
  bf1[4] = coeffs[2];
  bf1[5] = coeffs[10];
  bf1[6] = coeffs[6];
  bf1[7] = coeffs[14];
  bf1[8] = coeffs[1];
  bf1[9] = coeffs[9];
  bf1[10] = coeffs[5];
  bf1[11] = coeffs[13];
  bf1[12] = coeffs[3];
  bf1[13] = coeffs[11];
  bf1[14] = coeffs[7];
  bf1[15] = coeffs[15];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[60], bf0[8], -kCosPi[4], bf0[15]);
  bf1[9] = half_btf(kCosPi[28], bf0[9], -kCosPi[36], bf0[14]);
  bf1[10] = half_btf(kCosPi[44], bf0[10], -kCosPi[20], bf0[13]);
  bf1[11] = half_btf(kCosPi[12], bf0[11], -kCosPi[52], bf0[12]);
  bf1[12] = half_btf(kCosPi[52], bf0[11], kCosPi[12], bf0[12]);
  bf1[13] = half_btf(kCosPi[20], bf0[10], kCosPi[44], bf0[13]);
  bf1[14] = half_btf(kCosPi[36], bf0[9], kCosPi[28], bf0[14]);
  bf1[15] = half_btf(kCosPi[4], bf0[8], kCosPi[60], bf0[15]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], -kCosPi[8], bf0[7]);
  bf1[5] = half_btf(kCosPi[24], bf0[5], -kCosPi[40], bf0[6]);
  bf1[6] = half_btf(kCosPi[40], bf0[5], kCosPi[24], bf0[6]);
  bf1[7] = half_btf(kCosPi[8], bf0[4], kCosPi[56], bf0[7]);
  bf1[8] = bf0[8] + bf0[9];
  bf1[9] = bf0[8] - bf0[9];
  bf1[10] = -bf0[10] + bf0[11];
  bf1[11] = bf0[10] + bf0[11];
  bf1[12] = bf0[12] + bf0[13];
  bf1[13] = bf0[12] - bf0[13];
  bf1[14] = -bf0[14] + bf0[15];
  bf1[15] = bf0[14] + bf0[15];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[32], bf0[0], kCosPi[32], bf0[1]);
  bf1[1] = half_btf(kCosPi[32], bf0[0], -kCosPi[32], bf0[1]);
  bf1[2] = half_btf(kCosPi[48], bf0[2], -kCosPi[16], bf0[3]);
  bf1[3] = half_btf(kCosPi[16], bf0[2], kCosPi[48], bf0[3]);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = bf0[4] - bf0[5];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[6] + bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = half_btf(-kCosPi[16], bf0[9], kCosPi[48], bf0[14]);
  bf1[10] = half_btf(-kCosPi[48], bf0[10], -kCosPi[16], bf0[13]);
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = half_btf(-kCosPi[16], bf0[10], kCosPi[48], bf0[13]);
  bf1[14] = half_btf(kCosPi[48], bf0[9], kCosPi[16], bf0[14]);
  bf1[15] = bf0[15];

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = bf0[1] - bf0[2];
  bf1[3] = bf0[0] - bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = half_btf(-kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[6] = half_btf(kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[7] = bf0[7];
  bf1[8] = bf0[8] + bf0[11];
  bf1[9] = bf0[9] + bf0[10];
  bf1[10] = bf0[9] - bf0[10];
  bf1[11] = bf0[8] - bf0[11];
  bf1[12] = -bf0[12] + bf0[15];
  bf1[13] = -bf0[13] + bf0[14];
  bf1[14] = bf0[13] + bf0[14];
  bf1[15] = bf0[12] + bf0[15];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[7];
  bf1[1] = bf0[1] + bf0[6];
  bf1[2] = bf0[2] + bf0[5];
  bf1[3] = bf0[3] + bf0[4];
  bf1[4] = bf0[3] - bf0[4];
  bf1[5] = bf0[2] - bf0[5];
  bf1[6] = bf0[1] - bf0[6];
  bf1[7] = bf0[0] - bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(-kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[11] = half_btf(-kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[12] = half_btf(kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[13] = half_btf(kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];

  // stage 7
  std::swap(bf0, bf1);
  constexpr uint32_t shift = 1;
  coeffs[0] = div_sqrt2(bf0[0] + bf0[15], shift);
  coeffs[1] = div_sqrt2(bf0[1] + bf0[14], shift);
  coeffs[2] = div_sqrt2(bf0[2] + bf0[13], shift);
  coeffs[3] = div_sqrt2(bf0[3] + bf0[12], shift);
  coeffs[4] = div_sqrt2(bf0[4] + bf0[11], shift);
  coeffs[5] = div_sqrt2(bf0[5] + bf0[10], shift);
  coeffs[6] = div_sqrt2(bf0[6] + bf0[9], shift);
  coeffs[7] = div_sqrt2(bf0[7] + bf0[8], shift);
  coeffs[8] = div_sqrt2(bf0[7] - bf0[8], shift);
  coeffs[9] = div_sqrt2(bf0[6] - bf0[9], shift);
  coeffs[10] = div_sqrt2(bf0[5] - bf0[10], shift);
  coeffs[11] = div_sqrt2(bf0[4] - bf0[11], shift);
  coeffs[12] = div_sqrt2(bf0[3] - bf0[12], shift);
  coeffs[13] = div_sqrt2(bf0[2] - bf0[13], shift);
  coeffs[14] = div_sqrt2(bf0[1] - bf0[14], shift);
  coeffs[15] = div_sqrt2(bf0[0] - bf0[15], shift);
}

static void idct32_C(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 2;
  int32_t tmp0[32], tmp1[32];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = coeffs[16];
  bf1[2] = coeffs[8];
  bf1[3] = coeffs[24];
  bf1[4] = coeffs[4];
  bf1[5] = coeffs[20];
  bf1[6] = coeffs[12];
  bf1[7] = coeffs[28];
  bf1[8] = coeffs[2];
  bf1[9] = coeffs[18];
  bf1[10] = coeffs[10];
  bf1[11] = coeffs[26];
  bf1[12] = coeffs[6];
  bf1[13] = coeffs[22];
  bf1[14] = coeffs[14];
  bf1[15] = coeffs[30];
  bf1[16] = coeffs[1];
  bf1[17] = coeffs[17];
  bf1[18] = coeffs[9];
  bf1[19] = coeffs[25];
  bf1[20] = coeffs[5];
  bf1[21] = coeffs[21];
  bf1[22] = coeffs[13];
  bf1[23] = coeffs[29];
  bf1[24] = coeffs[3];
  bf1[25] = coeffs[19];
  bf1[26] = coeffs[11];
  bf1[27] = coeffs[27];
  bf1[28] = coeffs[7];
  bf1[29] = coeffs[23];
  bf1[30] = coeffs[15];
  bf1[31] = coeffs[31];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = half_btf(kCosPi[62], bf0[16], -kCosPi[2], bf0[31]);
  bf1[17] = half_btf(kCosPi[30], bf0[17], -kCosPi[34], bf0[30]);
  bf1[18] = half_btf(kCosPi[46], bf0[18], -kCosPi[18], bf0[29]);
  bf1[19] = half_btf(kCosPi[14], bf0[19], -kCosPi[50], bf0[28]);
  bf1[20] = half_btf(kCosPi[54], bf0[20], -kCosPi[10], bf0[27]);
  bf1[21] = half_btf(kCosPi[22], bf0[21], -kCosPi[42], bf0[26]);
  bf1[22] = half_btf(kCosPi[38], bf0[22], -kCosPi[26], bf0[25]);
  bf1[23] = half_btf(kCosPi[6], bf0[23], -kCosPi[58], bf0[24]);
  bf1[24] = half_btf(kCosPi[58], bf0[23], kCosPi[6], bf0[24]);
  bf1[25] = half_btf(kCosPi[26], bf0[22], kCosPi[38], bf0[25]);
  bf1[26] = half_btf(kCosPi[42], bf0[21], kCosPi[22], bf0[26]);
  bf1[27] = half_btf(kCosPi[10], bf0[20], kCosPi[54], bf0[27]);
  bf1[28] = half_btf(kCosPi[50], bf0[19], kCosPi[14], bf0[28]);
  bf1[29] = half_btf(kCosPi[18], bf0[18], kCosPi[46], bf0[29]);
  bf1[30] = half_btf(kCosPi[34], bf0[17], kCosPi[30], bf0[30]);
  bf1[31] = half_btf(kCosPi[2], bf0[16], kCosPi[62], bf0[31]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[60], bf0[8], -kCosPi[4], bf0[15]);
  bf1[9] = half_btf(kCosPi[28], bf0[9], -kCosPi[36], bf0[14]);
  bf1[10] = half_btf(kCosPi[44], bf0[10], -kCosPi[20], bf0[13]);
  bf1[11] = half_btf(kCosPi[12], bf0[11], -kCosPi[52], bf0[12]);
  bf1[12] = half_btf(kCosPi[52], bf0[11], kCosPi[12], bf0[12]);
  bf1[13] = half_btf(kCosPi[20], bf0[10], kCosPi[44], bf0[13]);
  bf1[14] = half_btf(kCosPi[36], bf0[9], kCosPi[28], bf0[14]);
  bf1[15] = half_btf(kCosPi[4], bf0[8], kCosPi[60], bf0[15]);
  bf1[16] = bf0[16] + bf0[17];
  bf1[17] = bf0[16] - bf0[17];
  bf1[18] = -bf0[18] + bf0[19];
  bf1[19] = bf0[18] + bf0[19];
  bf1[20] = bf0[20] + bf0[21];
  bf1[21] = bf0[20] - bf0[21];
  bf1[22] = -bf0[22] + bf0[23];
  bf1[23] = bf0[22] + bf0[23];
  bf1[24] = bf0[24] + bf0[25];
  bf1[25] = bf0[24] - bf0[25];
  bf1[26] = -bf0[26] + bf0[27];
  bf1[27] = bf0[26] + bf0[27];
  bf1[28] = bf0[28] + bf0[29];
  bf1[29] = bf0[28] - bf0[29];
  bf1[30] = -bf0[30] + bf0[31];
  bf1[31] = bf0[30] + bf0[31];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], -kCosPi[8], bf0[7]);
  bf1[5] = half_btf(kCosPi[24], bf0[5], -kCosPi[40], bf0[6]);
  bf1[6] = half_btf(kCosPi[40], bf0[5], kCosPi[24], bf0[6]);
  bf1[7] = half_btf(kCosPi[8], bf0[4], kCosPi[56], bf0[7]);
  bf1[8] = bf0[8] + bf0[9];
  bf1[9] = bf0[8] - bf0[9];
  bf1[10] = -bf0[10] + bf0[11];
  bf1[11] = bf0[10] + bf0[11];
  bf1[12] = bf0[12] + bf0[13];
  bf1[13] = bf0[12] - bf0[13];
  bf1[14] = -bf0[14] + bf0[15];
  bf1[15] = bf0[14] + bf0[15];
  bf1[16] = bf0[16];
  bf1[17] = half_btf(-kCosPi[8], bf0[17], kCosPi[56], bf0[30]);
  bf1[18] = half_btf(-kCosPi[56], bf0[18], -kCosPi[8], bf0[29]);
  bf1[19] = bf0[19];
  bf1[20] = bf0[20];
  bf1[21] = half_btf(-kCosPi[40], bf0[21], kCosPi[24], bf0[26]);
  bf1[22] = half_btf(-kCosPi[24], bf0[22], -kCosPi[40], bf0[25]);
  bf1[23] = bf0[23];
  bf1[24] = bf0[24];
  bf1[25] = half_btf(-kCosPi[40], bf0[22], kCosPi[24], bf0[25]);
  bf1[26] = half_btf(kCosPi[24], bf0[21], kCosPi[40], bf0[26]);
  bf1[27] = bf0[27];
  bf1[28] = bf0[28];
  bf1[29] = half_btf(-kCosPi[8], bf0[18], kCosPi[56], bf0[29]);
  bf1[30] = half_btf(kCosPi[56], bf0[17], kCosPi[8], bf0[30]);
  bf1[31] = bf0[31];

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[32], bf0[0], kCosPi[32], bf0[1]);
  bf1[1] = half_btf(kCosPi[32], bf0[0], -kCosPi[32], bf0[1]);
  bf1[2] = half_btf(kCosPi[48], bf0[2], -kCosPi[16], bf0[3]);
  bf1[3] = half_btf(kCosPi[16], bf0[2], kCosPi[48], bf0[3]);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = bf0[4] - bf0[5];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[6] + bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = half_btf(-kCosPi[16], bf0[9], kCosPi[48], bf0[14]);
  bf1[10] = half_btf(-kCosPi[48], bf0[10], -kCosPi[16], bf0[13]);
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = half_btf(-kCosPi[16], bf0[10], kCosPi[48], bf0[13]);
  bf1[14] = half_btf(kCosPi[48], bf0[9], kCosPi[16], bf0[14]);
  bf1[15] = bf0[15];
  bf1[16] = bf0[16] + bf0[19];
  bf1[17] = bf0[17] + bf0[18];
  bf1[18] = bf0[17] - bf0[18];
  bf1[19] = bf0[16] - bf0[19];
  bf1[20] = -bf0[20] + bf0[23];
  bf1[21] = -bf0[21] + bf0[22];
  bf1[22] = bf0[21] + bf0[22];
  bf1[23] = bf0[20] + bf0[23];
  bf1[24] = bf0[24] + bf0[27];
  bf1[25] = bf0[25] + bf0[26];
  bf1[26] = bf0[25] - bf0[26];
  bf1[27] = bf0[24] - bf0[27];
  bf1[28] = -bf0[28] + bf0[31];
  bf1[29] = -bf0[29] + bf0[30];
  bf1[30] = bf0[29] + bf0[30];
  bf1[31] = bf0[28] + bf0[31];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = bf0[1] - bf0[2];
  bf1[3] = bf0[0] - bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = half_btf(-kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[6] = half_btf(kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[7] = bf0[7];
  bf1[8] = bf0[8] + bf0[11];
  bf1[9] = bf0[9] + bf0[10];
  bf1[10] = bf0[9] - bf0[10];
  bf1[11] = bf0[8] - bf0[11];
  bf1[12] = -bf0[12] + bf0[15];
  bf1[13] = -bf0[13] + bf0[14];
  bf1[14] = bf0[13] + bf0[14];
  bf1[15] = bf0[12] + bf0[15];
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = half_btf(-kCosPi[16], bf0[18], kCosPi[48], bf0[29]);
  bf1[19] = half_btf(-kCosPi[16], bf0[19], kCosPi[48], bf0[28]);
  bf1[20] = half_btf(-kCosPi[48], bf0[20], -kCosPi[16], bf0[27]);
  bf1[21] = half_btf(-kCosPi[48], bf0[21], -kCosPi[16], bf0[26]);
  bf1[22] = bf0[22];
  bf1[23] = bf0[23];
  bf1[24] = bf0[24];
  bf1[25] = bf0[25];
  bf1[26] = half_btf(-kCosPi[16], bf0[21], kCosPi[48], bf0[26]);
  bf1[27] = half_btf(-kCosPi[16], bf0[20], kCosPi[48], bf0[27]);
  bf1[28] = half_btf(kCosPi[48], bf0[19], kCosPi[16], bf0[28]);
  bf1[29] = half_btf(kCosPi[48], bf0[18], kCosPi[16], bf0[29]);
  bf1[30] = bf0[30];
  bf1[31] = bf0[31];

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[7];
  bf1[1] = bf0[1] + bf0[6];
  bf1[2] = bf0[2] + bf0[5];
  bf1[3] = bf0[3] + bf0[4];
  bf1[4] = bf0[3] - bf0[4];
  bf1[5] = bf0[2] - bf0[5];
  bf1[6] = bf0[1] - bf0[6];
  bf1[7] = bf0[0] - bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(-kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[11] = half_btf(-kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[12] = half_btf(kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[13] = half_btf(kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = bf0[16] + bf0[23];
  bf1[17] = bf0[17] + bf0[22];
  bf1[18] = bf0[18] + bf0[21];
  bf1[19] = bf0[19] + bf0[20];
  bf1[20] = bf0[19] - bf0[20];
  bf1[21] = bf0[18] - bf0[21];
  bf1[22] = bf0[17] - bf0[22];
  bf1[23] = bf0[16] - bf0[23];
  bf1[24] = -bf0[24] + bf0[31];
  bf1[25] = -bf0[25] + bf0[30];
  bf1[26] = -bf0[26] + bf0[29];
  bf1[27] = -bf0[27] + bf0[28];
  bf1[28] = bf0[27] + bf0[28];
  bf1[29] = bf0[26] + bf0[29];
  bf1[30] = bf0[25] + bf0[30];
  bf1[31] = bf0[24] + bf0[31];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[15];
  bf1[1] = bf0[1] + bf0[14];
  bf1[2] = bf0[2] + bf0[13];
  bf1[3] = bf0[3] + bf0[12];
  bf1[4] = bf0[4] + bf0[11];
  bf1[5] = bf0[5] + bf0[10];
  bf1[6] = bf0[6] + bf0[9];
  bf1[7] = bf0[7] + bf0[8];
  bf1[8] = bf0[7] - bf0[8];
  bf1[9] = bf0[6] - bf0[9];
  bf1[10] = bf0[5] - bf0[10];
  bf1[11] = bf0[4] - bf0[11];
  bf1[12] = bf0[3] - bf0[12];
  bf1[13] = bf0[2] - bf0[13];
  bf1[14] = bf0[1] - bf0[14];
  bf1[15] = bf0[0] - bf0[15];
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = half_btf(-kCosPi[32], bf0[20], kCosPi[32], bf0[27]);
  bf1[21] = half_btf(-kCosPi[32], bf0[21], kCosPi[32], bf0[26]);
  bf1[22] = half_btf(-kCosPi[32], bf0[22], kCosPi[32], bf0[25]);
  bf1[23] = half_btf(-kCosPi[32], bf0[23], kCosPi[32], bf0[24]);
  bf1[24] = half_btf(kCosPi[32], bf0[23], kCosPi[32], bf0[24]);
  bf1[25] = half_btf(kCosPi[32], bf0[22], kCosPi[32], bf0[25]);
  bf1[26] = half_btf(kCosPi[32], bf0[21], kCosPi[32], bf0[26]);
  bf1[27] = half_btf(kCosPi[32], bf0[20], kCosPi[32], bf0[27]);
  bf1[28] = bf0[28];
  bf1[29] = bf0[29];
  bf1[30] = bf0[30];
  bf1[31] = bf0[31];

  // stage 9
  std::swap(bf0, bf1);
  coeffs[0] = round_shift(bf0[0] + bf0[31], shift - kPrecision);
  coeffs[1] = round_shift(bf0[1] + bf0[30], shift - kPrecision);
  coeffs[2] = round_shift(bf0[2] + bf0[29], shift - kPrecision);
  coeffs[3] = round_shift(bf0[3] + bf0[28], shift - kPrecision);
  coeffs[4] = round_shift(bf0[4] + bf0[27], shift - kPrecision);
  coeffs[5] = round_shift(bf0[5] + bf0[26], shift - kPrecision);
  coeffs[6] = round_shift(bf0[6] + bf0[25], shift - kPrecision);
  coeffs[7] = round_shift(bf0[7] + bf0[24], shift - kPrecision);
  coeffs[8] = round_shift(bf0[8] + bf0[23], shift - kPrecision);
  coeffs[9] = round_shift(bf0[9] + bf0[22], shift - kPrecision);
  coeffs[10] = round_shift(bf0[10] + bf0[21], shift - kPrecision);
  coeffs[11] = round_shift(bf0[11] + bf0[20], shift - kPrecision);
  coeffs[12] = round_shift(bf0[12] + bf0[19], shift - kPrecision);
  coeffs[13] = round_shift(bf0[13] + bf0[18], shift - kPrecision);
  coeffs[14] = round_shift(bf0[14] + bf0[17], shift - kPrecision);
  coeffs[15] = round_shift(bf0[15] + bf0[16], shift - kPrecision);
  coeffs[16] = round_shift(bf0[15] - bf0[16], shift - kPrecision);
  coeffs[17] = round_shift(bf0[14] - bf0[17], shift - kPrecision);
  coeffs[18] = round_shift(bf0[13] - bf0[18], shift - kPrecision);
  coeffs[19] = round_shift(bf0[12] - bf0[19], shift - kPrecision);
  coeffs[20] = round_shift(bf0[11] - bf0[20], shift - kPrecision);
  coeffs[21] = round_shift(bf0[10] - bf0[21], shift - kPrecision);
  coeffs[22] = round_shift(bf0[9] - bf0[22], shift - kPrecision);
  coeffs[23] = round_shift(bf0[8] - bf0[23], shift - kPrecision);
  coeffs[24] = round_shift(bf0[7] - bf0[24], shift - kPrecision);
  coeffs[25] = round_shift(bf0[6] - bf0[25], shift - kPrecision);
  coeffs[26] = round_shift(bf0[5] - bf0[26], shift - kPrecision);
  coeffs[27] = round_shift(bf0[4] - bf0[27], shift - kPrecision);
  coeffs[28] = round_shift(bf0[3] - bf0[28], shift - kPrecision);
  coeffs[29] = round_shift(bf0[2] - bf0[29], shift - kPrecision);
  coeffs[30] = round_shift(bf0[1] - bf0[30], shift - kPrecision);
  coeffs[31] = round_shift(bf0[0] - bf0[31], shift - kPrecision);
}

//------------------------------------------------------------------------------

static void iadst4_C(int16_t* coeffs) {
  // stage 0;
  int32_t x0 = coeffs[0];
  int32_t x1 = coeffs[1];
  int32_t x2 = coeffs[2];
  int32_t x3 = coeffs[3];

  if (!(x0 | x1 | x2 | x3)) {
    coeffs[0] = coeffs[1] = coeffs[2] = coeffs[3] = 0;
    return;
  }

  // stage 1
  int32_t s0 = round_shift((int64_t)kSinPi[1] * x0);
  int32_t s1 = round_shift((int64_t)kSinPi[2] * x0);
  int32_t s2 = round_shift((int64_t)kSinPi[3] * x1);
  int32_t s3 = round_shift((int64_t)kSinPi[4] * x2);
  int32_t s4 = round_shift((int64_t)kSinPi[1] * x2);
  int32_t s5 = round_shift((int64_t)kSinPi[2] * x3);
  int32_t s6 = round_shift((int64_t)kSinPi[4] * x3);
  int32_t s7 = x0 - x2;

  // stage 2
  s7 = s7 + x3;

  // stage 3
  s0 = s0 + s3;
  s1 = s1 - s4;
  s3 = s2;
  s2 = round_shift((int64_t)kSinPi[3] * s7);

  // stage 4
  s0 = s0 + s5;
  s1 = s1 - s6;

  // stage 5
  x0 = s0 + s3;
  x1 = s1 + s3;
  x2 = s2;
  x3 = s0 + s1;

  // stage 6
  x3 = x3 - s3;

  coeffs[0] = div_sqrt2(x0);
  coeffs[1] = div_sqrt2(x1);
  coeffs[2] = div_sqrt2(x2);
  coeffs[3] = div_sqrt2(x3);
}

static void iadst8_C(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  int32_t tmp0[8], tmp1[8];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[7];
  bf1[1] = coeffs[0];
  bf1[2] = coeffs[5];
  bf1[3] = coeffs[2];
  bf1[4] = coeffs[3];
  bf1[5] = coeffs[4];
  bf1[6] = coeffs[1];
  bf1[7] = coeffs[6];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[4], bf0[0], kCosPi[60], bf0[1]);
  bf1[1] = half_btf(-kCosPi[4], bf0[1], kCosPi[60], bf0[0]);
  bf1[2] = half_btf(kCosPi[20], bf0[2], kCosPi[44], bf0[3]);
  bf1[3] = half_btf(-kCosPi[20], bf0[3], kCosPi[44], bf0[2]);
  bf1[4] = half_btf(kCosPi[36], bf0[4], kCosPi[28], bf0[5]);
  bf1[5] = half_btf(-kCosPi[36], bf0[5], kCosPi[28], bf0[4]);
  bf1[6] = half_btf(kCosPi[52], bf0[6], kCosPi[12], bf0[7]);
  bf1[7] = half_btf(-kCosPi[52], bf0[7], kCosPi[12], bf0[6]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = -bf0[4] + bf0[0];
  bf1[5] = -bf0[5] + bf0[1];
  bf1[6] = -bf0[6] + bf0[2];
  bf1[7] = -bf0[7] + bf0[3];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(-kCosPi[16], bf0[5], kCosPi[48], bf0[4]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[48], bf0[7], kCosPi[16], bf0[6]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = -bf0[2] + bf0[0];
  bf1[3] = -bf0[3] + bf0[1];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = -bf0[6] + bf0[4];
  bf1[7] = -bf0[7] + bf0[5];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = round_shift(bf0[0], shift - kPrecision);
  bf1[1] = round_shift(bf0[1], shift - kPrecision);
  bf1[2] = half_btf(kCosPi[32], bf0[2], kCosPi[32], bf0[3], shift);
  bf1[3] = half_btf(-kCosPi[32], bf0[3], kCosPi[32], bf0[2], shift);
  bf1[4] = round_shift(bf0[4], shift - kPrecision);
  bf1[5] = round_shift(bf0[5], shift - kPrecision);
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[7], shift);
  bf1[7] = half_btf(-kCosPi[32], bf0[7], kCosPi[32], bf0[6], shift);

  // stage 7
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = -bf0[4];
  coeffs[2] = bf0[6];
  coeffs[3] = -bf0[2];
  coeffs[4] = bf0[3];
  coeffs[5] = -bf0[7];
  coeffs[6] = bf0[5];
  coeffs[7] = -bf0[1];
}

static void iadst16_C(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  int32_t tmp0[16], tmp1[16];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[15];
  bf1[1] = coeffs[0];
  bf1[2] = coeffs[13];
  bf1[3] = coeffs[2];
  bf1[4] = coeffs[11];
  bf1[5] = coeffs[4];
  bf1[6] = coeffs[9];
  bf1[7] = coeffs[6];
  bf1[8] = coeffs[7];
  bf1[9] = coeffs[8];
  bf1[10] = coeffs[5];
  bf1[11] = coeffs[10];
  bf1[12] = coeffs[3];
  bf1[13] = coeffs[12];
  bf1[14] = coeffs[1];
  bf1[15] = coeffs[14];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[2], bf0[0], kCosPi[62], bf0[1]);
  bf1[1] = half_btf(-kCosPi[2], bf0[1], kCosPi[62], bf0[0]);
  bf1[2] = half_btf(kCosPi[10], bf0[2], kCosPi[54], bf0[3]);
  bf1[3] = half_btf(-kCosPi[10], bf0[3], kCosPi[54], bf0[2]);
  bf1[4] = half_btf(kCosPi[18], bf0[4], kCosPi[46], bf0[5]);
  bf1[5] = half_btf(-kCosPi[18], bf0[5], kCosPi[46], bf0[4]);
  bf1[6] = half_btf(kCosPi[26], bf0[6], kCosPi[38], bf0[7]);
  bf1[7] = half_btf(-kCosPi[26], bf0[7], kCosPi[38], bf0[6]);
  bf1[8] = half_btf(kCosPi[34], bf0[8], kCosPi[30], bf0[9]);
  bf1[9] = half_btf(-kCosPi[34], bf0[9], kCosPi[30], bf0[8]);
  bf1[10] = half_btf(kCosPi[42], bf0[10], kCosPi[22], bf0[11]);
  bf1[11] = half_btf(-kCosPi[42], bf0[11], kCosPi[22], bf0[10]);
  bf1[12] = half_btf(kCosPi[50], bf0[12], kCosPi[14], bf0[13]);
  bf1[13] = half_btf(-kCosPi[50], bf0[13], kCosPi[14], bf0[12]);
  bf1[14] = half_btf(kCosPi[58], bf0[14], kCosPi[6], bf0[15]);
  bf1[15] = half_btf(-kCosPi[58], bf0[15], kCosPi[6], bf0[14]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[8];
  bf1[1] = bf0[1] + bf0[9];
  bf1[2] = bf0[2] + bf0[10];
  bf1[3] = bf0[3] + bf0[11];
  bf1[4] = bf0[4] + bf0[12];
  bf1[5] = bf0[5] + bf0[13];
  bf1[6] = bf0[6] + bf0[14];
  bf1[7] = bf0[7] + bf0[15];
  bf1[8] = -bf0[8] + bf0[0];
  bf1[9] = -bf0[9] + bf0[1];
  bf1[10] = -bf0[10] + bf0[2];
  bf1[11] = -bf0[11] + bf0[3];
  bf1[12] = -bf0[12] + bf0[4];
  bf1[13] = -bf0[13] + bf0[5];
  bf1[14] = -bf0[14] + bf0[6];
  bf1[15] = -bf0[15] + bf0[7];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[8], bf0[8], kCosPi[56], bf0[9]);
  bf1[9] = half_btf(-kCosPi[8], bf0[9], kCosPi[56], bf0[8]);
  bf1[10] = half_btf(kCosPi[40], bf0[10], kCosPi[24], bf0[11]);
  bf1[11] = half_btf(-kCosPi[40], bf0[11], kCosPi[24], bf0[10]);
  bf1[12] = half_btf(-kCosPi[56], bf0[12], kCosPi[8], bf0[13]);
  bf1[13] = half_btf(kCosPi[56], bf0[13], kCosPi[8], bf0[12]);
  bf1[14] = half_btf(-kCosPi[24], bf0[14], kCosPi[40], bf0[15]);
  bf1[15] = half_btf(kCosPi[24], bf0[15], kCosPi[40], bf0[14]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = -bf0[4] + bf0[0];
  bf1[5] = -bf0[5] + bf0[1];
  bf1[6] = -bf0[6] + bf0[2];
  bf1[7] = -bf0[7] + bf0[3];
  bf1[8] = bf0[8] + bf0[12];
  bf1[9] = bf0[9] + bf0[13];
  bf1[10] = bf0[10] + bf0[14];
  bf1[11] = bf0[11] + bf0[15];
  bf1[12] = -bf0[12] + bf0[8];
  bf1[13] = -bf0[13] + bf0[9];
  bf1[14] = -bf0[14] + bf0[10];
  bf1[15] = -bf0[15] + bf0[11];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(-kCosPi[16], bf0[5], kCosPi[48], bf0[4]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[48], bf0[7], kCosPi[16], bf0[6]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = half_btf(kCosPi[16], bf0[12], kCosPi[48], bf0[13]);
  bf1[13] = half_btf(-kCosPi[16], bf0[13], kCosPi[48], bf0[12]);
  bf1[14] = half_btf(-kCosPi[48], bf0[14], kCosPi[16], bf0[15]);
  bf1[15] = half_btf(kCosPi[48], bf0[15], kCosPi[16], bf0[14]);

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = -bf0[2] + bf0[0];
  bf1[3] = -bf0[3] + bf0[1];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = -bf0[6] + bf0[4];
  bf1[7] = -bf0[7] + bf0[5];
  bf1[8] = bf0[8] + bf0[10];
  bf1[9] = bf0[9] + bf0[11];
  bf1[10] = -bf0[10] + bf0[8];
  bf1[11] = -bf0[11] + bf0[9];
  bf1[12] = bf0[12] + bf0[14];
  bf1[13] = bf0[13] + bf0[15];
  bf1[14] = -bf0[14] + bf0[12];
  bf1[15] = -bf0[15] + bf0[13];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = div_sqrt2(bf0[0], shift - kPrecision);
  bf1[1] = div_sqrt2(bf0[1], shift - kPrecision);
  bf1[2] = round_shift(bf0[2] + bf0[3], shift - kPrecision + 1);
  bf1[3] = round_shift(bf0[2] - bf0[3], shift - kPrecision + 1);
  bf1[4] = div_sqrt2(bf0[4], shift - kPrecision);
  bf1[5] = div_sqrt2(bf0[5], shift - kPrecision);
  bf1[6] = round_shift(bf0[6] + bf0[7], shift - kPrecision + 1);
  bf1[7] = round_shift(bf0[6] - bf0[7], shift - kPrecision + 1);
  bf1[8] = div_sqrt2(bf0[8], shift - kPrecision);
  bf1[9] = div_sqrt2(bf0[9], shift - kPrecision);
  bf1[10] = round_shift(bf0[10] + bf0[11], shift - kPrecision + 1);
  bf1[11] = round_shift(bf0[10] - bf0[11], shift - kPrecision + 1);
  bf1[12] = div_sqrt2(bf0[12], shift - kPrecision);
  bf1[13] = div_sqrt2(bf0[13], shift - kPrecision);
  bf1[14] = round_shift(bf0[14] + bf0[15], shift - kPrecision + 1);
  bf1[15] = round_shift(bf0[14] - bf0[15], shift - kPrecision + 1);

  // stage 9
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = -bf0[8];
  coeffs[2] = bf0[12];
  coeffs[3] = -bf0[4];
  coeffs[4] = bf0[6];
  coeffs[5] = -bf0[14];
  coeffs[6] = bf0[10];
  coeffs[7] = -bf0[2];
  coeffs[8] = bf0[3];
  coeffs[9] = -bf0[11];
  coeffs[10] = bf0[15];
  coeffs[11] = -bf0[7];
  coeffs[12] = bf0[5];
  coeffs[13] = -bf0[13];
  coeffs[14] = bf0[9];
  coeffs[15] = -bf0[1];
}

static void iadst32_C(int16_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 2;
  int32_t tmp0[32], tmp1[32];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = -coeffs[31];
  bf1[2] = -coeffs[15];
  bf1[3] = coeffs[16];
  bf1[4] = -coeffs[7];
  bf1[5] = coeffs[24];
  bf1[6] = coeffs[8];
  bf1[7] = -coeffs[23];
  bf1[8] = -coeffs[3];
  bf1[9] = coeffs[28];
  bf1[10] = coeffs[12];
  bf1[11] = -coeffs[19];
  bf1[12] = coeffs[4];
  bf1[13] = -coeffs[27];
  bf1[14] = -coeffs[11];
  bf1[15] = coeffs[20];
  bf1[16] = -coeffs[1];
  bf1[17] = coeffs[30];
  bf1[18] = coeffs[14];
  bf1[19] = -coeffs[17];
  bf1[20] = coeffs[6];
  bf1[21] = -coeffs[25];
  bf1[22] = -coeffs[9];
  bf1[23] = coeffs[22];
  bf1[24] = coeffs[2];
  bf1[25] = -coeffs[29];
  bf1[26] = -coeffs[13];
  bf1[27] = coeffs[18];
  bf1[28] = -coeffs[5];
  bf1[29] = coeffs[26];
  bf1[30] = coeffs[10];
  bf1[31] = -coeffs[21];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = half_btf(kCosPi[32], bf0[2], kCosPi[32], bf0[3]);
  bf1[3] = half_btf(kCosPi[32], bf0[2], -kCosPi[32], bf0[3]);
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[7]);
  bf1[7] = half_btf(kCosPi[32], bf0[6], -kCosPi[32], bf0[7]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(kCosPi[32], bf0[10], kCosPi[32], bf0[11]);
  bf1[11] = half_btf(kCosPi[32], bf0[10], -kCosPi[32], bf0[11]);
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = half_btf(kCosPi[32], bf0[14], kCosPi[32], bf0[15]);
  bf1[15] = half_btf(kCosPi[32], bf0[14], -kCosPi[32], bf0[15]);
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = half_btf(kCosPi[32], bf0[18], kCosPi[32], bf0[19]);
  bf1[19] = half_btf(kCosPi[32], bf0[18], -kCosPi[32], bf0[19]);
  bf1[20] = bf0[20];
  bf1[21] = bf0[21];
  bf1[22] = half_btf(kCosPi[32], bf0[22], kCosPi[32], bf0[23]);
  bf1[23] = half_btf(kCosPi[32], bf0[22], -kCosPi[32], bf0[23]);
  bf1[24] = bf0[24];
  bf1[25] = bf0[25];
  bf1[26] = half_btf(kCosPi[32], bf0[26], kCosPi[32], bf0[27]);
  bf1[27] = half_btf(kCosPi[32], bf0[26], -kCosPi[32], bf0[27]);
  bf1[28] = bf0[28];
  bf1[29] = bf0[29];
  bf1[30] = half_btf(kCosPi[32], bf0[30], kCosPi[32], bf0[31]);
  bf1[31] = half_btf(kCosPi[32], bf0[30], -kCosPi[32], bf0[31]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = bf0[0] - bf0[2];
  bf1[3] = bf0[1] - bf0[3];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = bf0[4] - bf0[6];
  bf1[7] = bf0[5] - bf0[7];
  bf1[8] = bf0[8] + bf0[10];
  bf1[9] = bf0[9] + bf0[11];
  bf1[10] = bf0[8] - bf0[10];
  bf1[11] = bf0[9] - bf0[11];
  bf1[12] = bf0[12] + bf0[14];
  bf1[13] = bf0[13] + bf0[15];
  bf1[14] = bf0[12] - bf0[14];
  bf1[15] = bf0[13] - bf0[15];
  bf1[16] = bf0[16] + bf0[18];
  bf1[17] = bf0[17] + bf0[19];
  bf1[18] = bf0[16] - bf0[18];
  bf1[19] = bf0[17] - bf0[19];
  bf1[20] = bf0[20] + bf0[22];
  bf1[21] = bf0[21] + bf0[23];
  bf1[22] = bf0[20] - bf0[22];
  bf1[23] = bf0[21] - bf0[23];
  bf1[24] = bf0[24] + bf0[26];
  bf1[25] = bf0[25] + bf0[27];
  bf1[26] = bf0[24] - bf0[26];
  bf1[27] = bf0[25] - bf0[27];
  bf1[28] = bf0[28] + bf0[30];
  bf1[29] = bf0[29] + bf0[31];
  bf1[30] = bf0[28] - bf0[30];
  bf1[31] = bf0[29] - bf0[31];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(kCosPi[48], bf0[4], -kCosPi[16], bf0[5]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[16], bf0[6], kCosPi[48], bf0[7]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = half_btf(kCosPi[16], bf0[12], kCosPi[48], bf0[13]);
  bf1[13] = half_btf(kCosPi[48], bf0[12], -kCosPi[16], bf0[13]);
  bf1[14] = half_btf(-kCosPi[48], bf0[14], kCosPi[16], bf0[15]);
  bf1[15] = half_btf(kCosPi[16], bf0[14], kCosPi[48], bf0[15]);
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = half_btf(kCosPi[16], bf0[20], kCosPi[48], bf0[21]);
  bf1[21] = half_btf(kCosPi[48], bf0[20], -kCosPi[16], bf0[21]);
  bf1[22] = half_btf(-kCosPi[48], bf0[22], kCosPi[16], bf0[23]);
  bf1[23] = half_btf(kCosPi[16], bf0[22], kCosPi[48], bf0[23]);
  bf1[24] = bf0[24];
  bf1[25] = bf0[25];
  bf1[26] = bf0[26];
  bf1[27] = bf0[27];
  bf1[28] = half_btf(kCosPi[16], bf0[28], kCosPi[48], bf0[29]);
  bf1[29] = half_btf(kCosPi[48], bf0[28], -kCosPi[16], bf0[29]);
  bf1[30] = half_btf(-kCosPi[48], bf0[30], kCosPi[16], bf0[31]);
  bf1[31] = half_btf(kCosPi[16], bf0[30], kCosPi[48], bf0[31]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = bf0[0] - bf0[4];
  bf1[5] = bf0[1] - bf0[5];
  bf1[6] = bf0[2] - bf0[6];
  bf1[7] = bf0[3] - bf0[7];
  bf1[8] = bf0[8] + bf0[12];
  bf1[9] = bf0[9] + bf0[13];
  bf1[10] = bf0[10] + bf0[14];
  bf1[11] = bf0[11] + bf0[15];
  bf1[12] = bf0[8] - bf0[12];
  bf1[13] = bf0[9] - bf0[13];
  bf1[14] = bf0[10] - bf0[14];
  bf1[15] = bf0[11] - bf0[15];
  bf1[16] = bf0[16] + bf0[20];
  bf1[17] = bf0[17] + bf0[21];
  bf1[18] = bf0[18] + bf0[22];
  bf1[19] = bf0[19] + bf0[23];
  bf1[20] = bf0[16] - bf0[20];
  bf1[21] = bf0[17] - bf0[21];
  bf1[22] = bf0[18] - bf0[22];
  bf1[23] = bf0[19] - bf0[23];
  bf1[24] = bf0[24] + bf0[28];
  bf1[25] = bf0[25] + bf0[29];
  bf1[26] = bf0[26] + bf0[30];
  bf1[27] = bf0[27] + bf0[31];
  bf1[28] = bf0[24] - bf0[28];
  bf1[29] = bf0[25] - bf0[29];
  bf1[30] = bf0[26] - bf0[30];
  bf1[31] = bf0[27] - bf0[31];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[8], bf0[8], kCosPi[56], bf0[9]);
  bf1[9] = half_btf(kCosPi[56], bf0[8], -kCosPi[8], bf0[9]);
  bf1[10] = half_btf(kCosPi[40], bf0[10], kCosPi[24], bf0[11]);
  bf1[11] = half_btf(kCosPi[24], bf0[10], -kCosPi[40], bf0[11]);
  bf1[12] = half_btf(-kCosPi[56], bf0[12], kCosPi[8], bf0[13]);
  bf1[13] = half_btf(kCosPi[8], bf0[12], kCosPi[56], bf0[13]);
  bf1[14] = half_btf(-kCosPi[24], bf0[14], kCosPi[40], bf0[15]);
  bf1[15] = half_btf(kCosPi[40], bf0[14], kCosPi[24], bf0[15]);
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = bf0[20];
  bf1[21] = bf0[21];
  bf1[22] = bf0[22];
  bf1[23] = bf0[23];
  bf1[24] = half_btf(kCosPi[8], bf0[24], kCosPi[56], bf0[25]);
  bf1[25] = half_btf(kCosPi[56], bf0[24], -kCosPi[8], bf0[25]);
  bf1[26] = half_btf(kCosPi[40], bf0[26], kCosPi[24], bf0[27]);
  bf1[27] = half_btf(kCosPi[24], bf0[26], -kCosPi[40], bf0[27]);
  bf1[28] = half_btf(-kCosPi[56], bf0[28], kCosPi[8], bf0[29]);
  bf1[29] = half_btf(kCosPi[8], bf0[28], kCosPi[56], bf0[29]);
  bf1[30] = half_btf(-kCosPi[24], bf0[30], kCosPi[40], bf0[31]);
  bf1[31] = half_btf(kCosPi[40], bf0[30], kCosPi[24], bf0[31]);

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[8];
  bf1[1] = bf0[1] + bf0[9];
  bf1[2] = bf0[2] + bf0[10];
  bf1[3] = bf0[3] + bf0[11];
  bf1[4] = bf0[4] + bf0[12];
  bf1[5] = bf0[5] + bf0[13];
  bf1[6] = bf0[6] + bf0[14];
  bf1[7] = bf0[7] + bf0[15];
  bf1[8] = bf0[0] - bf0[8];
  bf1[9] = bf0[1] - bf0[9];
  bf1[10] = bf0[2] - bf0[10];
  bf1[11] = bf0[3] - bf0[11];
  bf1[12] = bf0[4] - bf0[12];
  bf1[13] = bf0[5] - bf0[13];
  bf1[14] = bf0[6] - bf0[14];
  bf1[15] = bf0[7] - bf0[15];
  bf1[16] = bf0[16] + bf0[24];
  bf1[17] = bf0[17] + bf0[25];
  bf1[18] = bf0[18] + bf0[26];
  bf1[19] = bf0[19] + bf0[27];
  bf1[20] = bf0[20] + bf0[28];
  bf1[21] = bf0[21] + bf0[29];
  bf1[22] = bf0[22] + bf0[30];
  bf1[23] = bf0[23] + bf0[31];
  bf1[24] = bf0[16] - bf0[24];
  bf1[25] = bf0[17] - bf0[25];
  bf1[26] = bf0[18] - bf0[26];
  bf1[27] = bf0[19] - bf0[27];
  bf1[28] = bf0[20] - bf0[28];
  bf1[29] = bf0[21] - bf0[29];
  bf1[30] = bf0[22] - bf0[30];
  bf1[31] = bf0[23] - bf0[31];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = half_btf(kCosPi[4], bf0[16], kCosPi[60], bf0[17]);
  bf1[17] = half_btf(kCosPi[60], bf0[16], -kCosPi[4], bf0[17]);
  bf1[18] = half_btf(kCosPi[20], bf0[18], kCosPi[44], bf0[19]);
  bf1[19] = half_btf(kCosPi[44], bf0[18], -kCosPi[20], bf0[19]);
  bf1[20] = half_btf(kCosPi[36], bf0[20], kCosPi[28], bf0[21]);
  bf1[21] = half_btf(kCosPi[28], bf0[20], -kCosPi[36], bf0[21]);
  bf1[22] = half_btf(kCosPi[52], bf0[22], kCosPi[12], bf0[23]);
  bf1[23] = half_btf(kCosPi[12], bf0[22], -kCosPi[52], bf0[23]);
  bf1[24] = half_btf(-kCosPi[60], bf0[24], kCosPi[4], bf0[25]);
  bf1[25] = half_btf(kCosPi[4], bf0[24], kCosPi[60], bf0[25]);
  bf1[26] = half_btf(-kCosPi[44], bf0[26], kCosPi[20], bf0[27]);
  bf1[27] = half_btf(kCosPi[20], bf0[26], kCosPi[44], bf0[27]);
  bf1[28] = half_btf(-kCosPi[28], bf0[28], kCosPi[36], bf0[29]);
  bf1[29] = half_btf(kCosPi[36], bf0[28], kCosPi[28], bf0[29]);
  bf1[30] = half_btf(-kCosPi[12], bf0[30], kCosPi[52], bf0[31]);
  bf1[31] = half_btf(kCosPi[52], bf0[30], kCosPi[12], bf0[31]);

  // stage 9
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[16];
  bf1[1] = bf0[1] + bf0[17];
  bf1[2] = bf0[2] + bf0[18];
  bf1[3] = bf0[3] + bf0[19];
  bf1[4] = bf0[4] + bf0[20];
  bf1[5] = bf0[5] + bf0[21];
  bf1[6] = bf0[6] + bf0[22];
  bf1[7] = bf0[7] + bf0[23];
  bf1[8] = bf0[8] + bf0[24];
  bf1[9] = bf0[9] + bf0[25];
  bf1[10] = bf0[10] + bf0[26];
  bf1[11] = bf0[11] + bf0[27];
  bf1[12] = bf0[12] + bf0[28];
  bf1[13] = bf0[13] + bf0[29];
  bf1[14] = bf0[14] + bf0[30];
  bf1[15] = bf0[15] + bf0[31];
  bf1[16] = bf0[0] - bf0[16];
  bf1[17] = bf0[1] - bf0[17];
  bf1[18] = bf0[2] - bf0[18];
  bf1[19] = bf0[3] - bf0[19];
  bf1[20] = bf0[4] - bf0[20];
  bf1[21] = bf0[5] - bf0[21];
  bf1[22] = bf0[6] - bf0[22];
  bf1[23] = bf0[7] - bf0[23];
  bf1[24] = bf0[8] - bf0[24];
  bf1[25] = bf0[9] - bf0[25];
  bf1[26] = bf0[10] - bf0[26];
  bf1[27] = bf0[11] - bf0[27];
  bf1[28] = bf0[12] - bf0[28];
  bf1[29] = bf0[13] - bf0[29];
  bf1[30] = bf0[14] - bf0[30];
  bf1[31] = bf0[15] - bf0[31];

  // stage 10
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[1], bf0[0], kCosPi[63], bf0[1], shift);
  bf1[1] = half_btf(kCosPi[63], bf0[0], -kCosPi[1], bf0[1], shift);
  bf1[2] = half_btf(kCosPi[5], bf0[2], kCosPi[59], bf0[3], shift);
  bf1[3] = half_btf(kCosPi[59], bf0[2], -kCosPi[5], bf0[3], shift);
  bf1[4] = half_btf(kCosPi[9], bf0[4], kCosPi[55], bf0[5], shift);
  bf1[5] = half_btf(kCosPi[55], bf0[4], -kCosPi[9], bf0[5], shift);
  bf1[6] = half_btf(kCosPi[13], bf0[6], kCosPi[51], bf0[7], shift);
  bf1[7] = half_btf(kCosPi[51], bf0[6], -kCosPi[13], bf0[7], shift);
  bf1[8] = half_btf(kCosPi[17], bf0[8], kCosPi[47], bf0[9], shift);
  bf1[9] = half_btf(kCosPi[47], bf0[8], -kCosPi[17], bf0[9], shift);
  bf1[10] = half_btf(kCosPi[21], bf0[10], kCosPi[43], bf0[11], shift);
  bf1[11] = half_btf(kCosPi[43], bf0[10], -kCosPi[21], bf0[11], shift);
  bf1[12] = half_btf(kCosPi[25], bf0[12], kCosPi[39], bf0[13], shift);
  bf1[13] = half_btf(kCosPi[39], bf0[12], -kCosPi[25], bf0[13], shift);
  bf1[14] = half_btf(kCosPi[29], bf0[14], kCosPi[35], bf0[15], shift);
  bf1[15] = half_btf(kCosPi[35], bf0[14], -kCosPi[29], bf0[15], shift);
  bf1[16] = half_btf(kCosPi[33], bf0[16], kCosPi[31], bf0[17], shift);
  bf1[17] = half_btf(kCosPi[31], bf0[16], -kCosPi[33], bf0[17], shift);
  bf1[18] = half_btf(kCosPi[37], bf0[18], kCosPi[27], bf0[19], shift);
  bf1[19] = half_btf(kCosPi[27], bf0[18], -kCosPi[37], bf0[19], shift);
  bf1[20] = half_btf(kCosPi[41], bf0[20], kCosPi[23], bf0[21], shift);
  bf1[21] = half_btf(kCosPi[23], bf0[20], -kCosPi[41], bf0[21], shift);
  bf1[22] = half_btf(kCosPi[45], bf0[22], kCosPi[19], bf0[23], shift);
  bf1[23] = half_btf(kCosPi[19], bf0[22], -kCosPi[45], bf0[23], shift);
  bf1[24] = half_btf(kCosPi[49], bf0[24], kCosPi[15], bf0[25], shift);
  bf1[25] = half_btf(kCosPi[15], bf0[24], -kCosPi[49], bf0[25], shift);
  bf1[26] = half_btf(kCosPi[53], bf0[26], kCosPi[11], bf0[27], shift);
  bf1[27] = half_btf(kCosPi[11], bf0[26], -kCosPi[53], bf0[27], shift);
  bf1[28] = half_btf(kCosPi[57], bf0[28], kCosPi[7], bf0[29], shift);
  bf1[29] = half_btf(kCosPi[7], bf0[28], -kCosPi[57], bf0[29], shift);
  bf1[30] = half_btf(kCosPi[61], bf0[30], kCosPi[3], bf0[31], shift);
  bf1[31] = half_btf(kCosPi[3], bf0[30], -kCosPi[61], bf0[31], shift);

  // stage 11
  std::swap(bf0, bf1);
  coeffs[0] = bf0[1];
  coeffs[1] = bf0[30];
  coeffs[2] = bf0[3];
  coeffs[3] = bf0[28];
  coeffs[4] = bf0[5];
  coeffs[5] = bf0[26];
  coeffs[6] = bf0[7];
  coeffs[7] = bf0[24];
  coeffs[8] = bf0[9];
  coeffs[9] = bf0[22];
  coeffs[10] = bf0[11];
  coeffs[11] = bf0[20];
  coeffs[12] = bf0[13];
  coeffs[13] = bf0[18];
  coeffs[14] = bf0[15];
  coeffs[15] = bf0[16];
  coeffs[16] = bf0[17];
  coeffs[17] = bf0[14];
  coeffs[18] = bf0[19];
  coeffs[19] = bf0[12];
  coeffs[20] = bf0[21];
  coeffs[21] = bf0[10];
  coeffs[22] = bf0[23];
  coeffs[23] = bf0[8];
  coeffs[24] = bf0[25];
  coeffs[25] = bf0[6];
  coeffs[26] = bf0[27];
  coeffs[27] = bf0[4];
  coeffs[28] = bf0[29];
  coeffs[29] = bf0[2];
  coeffs[30] = bf0[31];
  coeffs[31] = bf0[0];
}

//------------------------------------------------------------------------------
// Hadamard transform

// sequence reordering (=bit-reversal + Gray code)
static const uint32_t kHadamardR2[2] = {0, 1};
static const uint32_t kHadamardR4[4] = {0, 3, 1, 2};
static const uint32_t kHadamardR8[8] = {0, 7, 3, 4, 1, 6, 2, 5};
static const uint32_t kHadamardR16[16] = {  // https://oeis.org/A240909
    0, 15, 7, 8, 3, 12, 4, 11, 1, 14, 6, 9, 2, 13, 5, 10};
static const uint32_t kHadamardR32[32] = {  // https://oeis.org/A240910
    0, 31, 15, 16, 7, 24, 8, 23, 3, 28, 12, 19, 4, 27, 11, 20,
    1, 30, 14, 17, 6, 25, 9, 22, 2, 29, 13, 18, 5, 26, 10, 21};

static const uint32_t* const kHadamardReorder[5] = {
    kHadamardR2, kHadamardR4, kHadamardR8, kHadamardR16, kHadamardR32};

template <typename T>
static void CoreHadamard_C(T* coeffs, uint32_t size) {
  assert(size == 2 || size == 4 || size == 8 || size == 16 || size == 32);
  int32_t tmp[32];
  for (uint32_t i = 0; i < size; ++i) tmp[i] = coeffs[i];
  for (uint32_t half_size = 1; half_size < size; half_size *= 2) {
    for (uint32_t i = 0; i < size; i += 2 * half_size) {
      for (uint32_t j = 0; j < half_size; ++j) {
        const int32_t A = tmp[i + j + 0];
        const int32_t B = tmp[i + j + half_size];
        tmp[i + j + 0] = A + B;
        tmp[i + j + half_size] = A - B;
      }
    }
  }
  // Sequence-reordering
  const uint32_t* const reorder = kHadamardReorder[SizeToIdx(size)];
  const float norm = std::sqrt(1. / size);  // same norm for bwd/fwd
  for (uint32_t i = 0; i < size; ++i) {
    coeffs[i] = std::lround(tmp[reorder[i]] * norm);
  }
}

#define GENERATE_HADAMARD_FUNC(SIZE)                 \
  static void iHadamard##SIZE##_C(int16_t* coeffs) { \
    CoreHadamard_C<int16_t>(coeffs, SIZE);           \
  }                                                  \
  static void fHadamard##SIZE##_C(int32_t* coeffs) { \
    CoreHadamard_C<int32_t>(coeffs, SIZE);           \
  }

GENERATE_HADAMARD_FUNC(2)
GENERATE_HADAMARD_FUNC(4)
GENERATE_HADAMARD_FUNC(8)
GENERATE_HADAMARD_FUNC(16)
GENERATE_HADAMARD_FUNC(32)

#undef GENERATE_HADAMARD_FUNC

//------------------------------------------------------------------------------
// Forward transforms

static void fdct2_C(int32_t* coeffs) {
  const int32_t A = coeffs[0], B = coeffs[1];
  coeffs[0] = div_sqrt2(A + B);
  coeffs[1] = div_sqrt2(A - B);
}

static void fdct4_C(int32_t* coeffs) {
  int32_t tmp0[4], tmp1[4];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0] + coeffs[3];
  bf1[1] = coeffs[1] + coeffs[2];
  bf1[2] = coeffs[1] - coeffs[2];
  bf1[3] = coeffs[0] - coeffs[3];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = round_shift(bf0[0] + bf0[1], 1);
  bf1[1] = round_shift(bf0[0] - bf0[1], 1);
  bf1[2] = half_btf(kCos32Cos48, bf0[2], kCos32Cos16, bf0[3]);
  bf1[3] = half_btf(-kCos32Cos16, bf0[2], kCos32Cos48, bf0[3]);

  // stage 3
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = bf0[2];
  coeffs[2] = bf0[1];
  coeffs[3] = bf0[3];
}

static void fdct8_C(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 1;
  int32_t tmp0[8], tmp1[8];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0] + coeffs[7];
  bf1[1] = coeffs[1] + coeffs[6];
  bf1[2] = coeffs[2] + coeffs[5];
  bf1[3] = coeffs[3] + coeffs[4];
  bf1[4] = -coeffs[4] + coeffs[3];
  bf1[5] = -coeffs[5] + coeffs[2];
  bf1[6] = -coeffs[6] + coeffs[1];
  bf1[7] = -coeffs[7] + coeffs[0];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = -bf0[2] + bf0[1];
  bf1[3] = -bf0[3] + bf0[0];
  bf1[4] = bf0[4];
  bf1[5] = half_btf(-kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[5]);
  bf1[7] = bf0[7];

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[32], bf0[0], kCosPi[32], bf0[1], shift);
  bf1[1] = half_btf(-kCosPi[32], bf0[1], kCosPi[32], bf0[0], shift);
  bf1[2] = half_btf(kCosPi[48], bf0[2], kCosPi[16], bf0[3], shift);
  bf1[3] = half_btf(kCosPi[48], bf0[3], -kCosPi[16], bf0[2], shift);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = -bf0[5] + bf0[4];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[7] + bf0[6];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], kCosPi[8], bf0[7], shift);
  bf1[5] = half_btf(kCosPi[24], bf0[5], kCosPi[40], bf0[6], shift);
  bf1[6] = half_btf(kCosPi[24], bf0[6], -kCosPi[40], bf0[5], shift);
  bf1[7] = half_btf(kCosPi[56], bf0[7], -kCosPi[8], bf0[4], shift);

  // stage 5
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = bf0[4];
  coeffs[2] = bf0[2];
  coeffs[3] = bf0[6];
  coeffs[4] = bf0[1];
  coeffs[5] = bf0[5];
  coeffs[6] = bf0[3];
  coeffs[7] = bf0[7];
}

static void fdct16_C(int32_t* coeffs) {
  int32_t tmp0[16], tmp1[16];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0] + coeffs[15];
  bf1[1] = coeffs[1] + coeffs[14];
  bf1[2] = coeffs[2] + coeffs[13];
  bf1[3] = coeffs[3] + coeffs[12];
  bf1[4] = coeffs[4] + coeffs[11];
  bf1[5] = coeffs[5] + coeffs[10];
  bf1[6] = coeffs[6] + coeffs[9];
  bf1[7] = coeffs[7] + coeffs[8];
  bf1[8] = -coeffs[8] + coeffs[7];
  bf1[9] = -coeffs[9] + coeffs[6];
  bf1[10] = -coeffs[10] + coeffs[5];
  bf1[11] = -coeffs[11] + coeffs[4];
  bf1[12] = -coeffs[12] + coeffs[3];
  bf1[13] = -coeffs[13] + coeffs[2];
  bf1[14] = -coeffs[14] + coeffs[1];
  bf1[15] = -coeffs[15] + coeffs[0];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[7];
  bf1[1] = bf0[1] + bf0[6];
  bf1[2] = bf0[2] + bf0[5];
  bf1[3] = bf0[3] + bf0[4];
  bf1[4] = -bf0[4] + bf0[3];
  bf1[5] = -bf0[5] + bf0[2];
  bf1[6] = -bf0[6] + bf0[1];
  bf1[7] = -bf0[7] + bf0[0];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(-kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[11] = half_btf(-kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[12] = half_btf(kCosPi[32], bf0[12], kCosPi[32], bf0[11]);
  bf1[13] = half_btf(kCosPi[32], bf0[13], kCosPi[32], bf0[10]);
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = -bf0[2] + bf0[1];
  bf1[3] = -bf0[3] + bf0[0];
  bf1[4] = bf0[4];
  bf1[5] = half_btf(-kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[5]);
  bf1[7] = bf0[7];
  bf1[8] = bf0[8] + bf0[11];
  bf1[9] = bf0[9] + bf0[10];
  bf1[10] = -bf0[10] + bf0[9];
  bf1[11] = -bf0[11] + bf0[8];
  bf1[12] = -bf0[12] + bf0[15];
  bf1[13] = -bf0[13] + bf0[14];
  bf1[14] = bf0[14] + bf0[13];
  bf1[15] = bf0[15] + bf0[12];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[32], bf0[0], kCosPi[32], bf0[1]);
  bf1[1] = half_btf(-kCosPi[32], bf0[1], kCosPi[32], bf0[0]);
  bf1[2] = half_btf(kCosPi[48], bf0[2], kCosPi[16], bf0[3]);
  bf1[3] = half_btf(kCosPi[48], bf0[3], -kCosPi[16], bf0[2]);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = -bf0[5] + bf0[4];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[7] + bf0[6];
  bf1[8] = bf0[8];
  bf1[9] = half_btf(-kCosPi[16], bf0[9], kCosPi[48], bf0[14]);
  bf1[10] = half_btf(-kCosPi[48], bf0[10], -kCosPi[16], bf0[13]);
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = half_btf(kCosPi[48], bf0[13], -kCosPi[16], bf0[10]);
  bf1[14] = half_btf(kCosPi[16], bf0[14], kCosPi[48], bf0[9]);
  bf1[15] = bf0[15];

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], kCosPi[8], bf0[7]);
  bf1[5] = half_btf(kCosPi[24], bf0[5], kCosPi[40], bf0[6]);
  bf1[6] = half_btf(kCosPi[24], bf0[6], -kCosPi[40], bf0[5]);
  bf1[7] = half_btf(kCosPi[56], bf0[7], -kCosPi[8], bf0[4]);
  bf1[8] = bf0[8] + bf0[9];
  bf1[9] = -bf0[9] + bf0[8];
  bf1[10] = -bf0[10] + bf0[11];
  bf1[11] = bf0[11] + bf0[10];
  bf1[12] = bf0[12] + bf0[13];
  bf1[13] = -bf0[13] + bf0[12];
  bf1[14] = -bf0[14] + bf0[15];
  bf1[15] = bf0[15] + bf0[14];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[60], bf0[8], kCosPi[4], bf0[15]);
  bf1[9] = half_btf(kCosPi[28], bf0[9], kCosPi[36], bf0[14]);
  bf1[10] = half_btf(kCosPi[44], bf0[10], kCosPi[20], bf0[13]);
  bf1[11] = half_btf(kCosPi[12], bf0[11], kCosPi[52], bf0[12]);
  bf1[12] = half_btf(kCosPi[12], bf0[12], -kCosPi[52], bf0[11]);
  bf1[13] = half_btf(kCosPi[44], bf0[13], -kCosPi[20], bf0[10]);
  bf1[14] = half_btf(kCosPi[28], bf0[14], -kCosPi[36], bf0[9]);
  bf1[15] = half_btf(kCosPi[60], bf0[15], -kCosPi[4], bf0[8]);

  // stage 7
  std::swap(bf0, bf1);
  constexpr uint32_t shift = 1;
  coeffs[0] = div_sqrt2(bf0[0], shift);
  coeffs[1] = div_sqrt2(bf0[8], shift);
  coeffs[2] = div_sqrt2(bf0[4], shift);
  coeffs[3] = div_sqrt2(bf0[12], shift);
  coeffs[4] = div_sqrt2(bf0[2], shift);
  coeffs[5] = div_sqrt2(bf0[10], shift);
  coeffs[6] = div_sqrt2(bf0[6], shift);
  coeffs[7] = div_sqrt2(bf0[14], shift);
  coeffs[8] = div_sqrt2(bf0[1], shift);
  coeffs[9] = div_sqrt2(bf0[9], shift);
  coeffs[10] = div_sqrt2(bf0[5], shift);
  coeffs[11] = div_sqrt2(bf0[13], shift);
  coeffs[12] = div_sqrt2(bf0[3], shift);
  coeffs[13] = div_sqrt2(bf0[11], shift);
  coeffs[14] = div_sqrt2(bf0[7], shift);
  coeffs[15] = div_sqrt2(bf0[15], shift);
}

static void fdct32_C(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 2;
  int32_t tmp0[32], tmp1[32];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0] + coeffs[31];
  bf1[1] = coeffs[1] + coeffs[30];
  bf1[2] = coeffs[2] + coeffs[29];
  bf1[3] = coeffs[3] + coeffs[28];
  bf1[4] = coeffs[4] + coeffs[27];
  bf1[5] = coeffs[5] + coeffs[26];
  bf1[6] = coeffs[6] + coeffs[25];
  bf1[7] = coeffs[7] + coeffs[24];
  bf1[8] = coeffs[8] + coeffs[23];
  bf1[9] = coeffs[9] + coeffs[22];
  bf1[10] = coeffs[10] + coeffs[21];
  bf1[11] = coeffs[11] + coeffs[20];
  bf1[12] = coeffs[12] + coeffs[19];
  bf1[13] = coeffs[13] + coeffs[18];
  bf1[14] = coeffs[14] + coeffs[17];
  bf1[15] = coeffs[15] + coeffs[16];
  bf1[16] = -coeffs[16] + coeffs[15];
  bf1[17] = -coeffs[17] + coeffs[14];
  bf1[18] = -coeffs[18] + coeffs[13];
  bf1[19] = -coeffs[19] + coeffs[12];
  bf1[20] = -coeffs[20] + coeffs[11];
  bf1[21] = -coeffs[21] + coeffs[10];
  bf1[22] = -coeffs[22] + coeffs[9];
  bf1[23] = -coeffs[23] + coeffs[8];
  bf1[24] = -coeffs[24] + coeffs[7];
  bf1[25] = -coeffs[25] + coeffs[6];
  bf1[26] = -coeffs[26] + coeffs[5];
  bf1[27] = -coeffs[27] + coeffs[4];
  bf1[28] = -coeffs[28] + coeffs[3];
  bf1[29] = -coeffs[29] + coeffs[2];
  bf1[30] = -coeffs[30] + coeffs[1];
  bf1[31] = -coeffs[31] + coeffs[0];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[15];
  bf1[1] = bf0[1] + bf0[14];
  bf1[2] = bf0[2] + bf0[13];
  bf1[3] = bf0[3] + bf0[12];
  bf1[4] = bf0[4] + bf0[11];
  bf1[5] = bf0[5] + bf0[10];
  bf1[6] = bf0[6] + bf0[9];
  bf1[7] = bf0[7] + bf0[8];
  bf1[8] = -bf0[8] + bf0[7];
  bf1[9] = -bf0[9] + bf0[6];
  bf1[10] = -bf0[10] + bf0[5];
  bf1[11] = -bf0[11] + bf0[4];
  bf1[12] = -bf0[12] + bf0[3];
  bf1[13] = -bf0[13] + bf0[2];
  bf1[14] = -bf0[14] + bf0[1];
  bf1[15] = -bf0[15] + bf0[0];
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = half_btf(-kCosPi[32], bf0[20], kCosPi[32], bf0[27]);
  bf1[21] = half_btf(-kCosPi[32], bf0[21], kCosPi[32], bf0[26]);
  bf1[22] = half_btf(-kCosPi[32], bf0[22], kCosPi[32], bf0[25]);
  bf1[23] = half_btf(-kCosPi[32], bf0[23], kCosPi[32], bf0[24]);
  bf1[24] = half_btf(kCosPi[32], bf0[24], kCosPi[32], bf0[23]);
  bf1[25] = half_btf(kCosPi[32], bf0[25], kCosPi[32], bf0[22]);
  bf1[26] = half_btf(kCosPi[32], bf0[26], kCosPi[32], bf0[21]);
  bf1[27] = half_btf(kCosPi[32], bf0[27], kCosPi[32], bf0[20]);
  bf1[28] = bf0[28];
  bf1[29] = bf0[29];
  bf1[30] = bf0[30];
  bf1[31] = bf0[31];

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[7];
  bf1[1] = bf0[1] + bf0[6];
  bf1[2] = bf0[2] + bf0[5];
  bf1[3] = bf0[3] + bf0[4];
  bf1[4] = -bf0[4] + bf0[3];
  bf1[5] = -bf0[5] + bf0[2];
  bf1[6] = -bf0[6] + bf0[1];
  bf1[7] = -bf0[7] + bf0[0];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(-kCosPi[32], bf0[10], kCosPi[32], bf0[13]);
  bf1[11] = half_btf(-kCosPi[32], bf0[11], kCosPi[32], bf0[12]);
  bf1[12] = half_btf(kCosPi[32], bf0[12], kCosPi[32], bf0[11]);
  bf1[13] = half_btf(kCosPi[32], bf0[13], kCosPi[32], bf0[10]);
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = bf0[16] + bf0[23];
  bf1[17] = bf0[17] + bf0[22];
  bf1[18] = bf0[18] + bf0[21];
  bf1[19] = bf0[19] + bf0[20];
  bf1[20] = -bf0[20] + bf0[19];
  bf1[21] = -bf0[21] + bf0[18];
  bf1[22] = -bf0[22] + bf0[17];
  bf1[23] = -bf0[23] + bf0[16];
  bf1[24] = -bf0[24] + bf0[31];
  bf1[25] = -bf0[25] + bf0[30];
  bf1[26] = -bf0[26] + bf0[29];
  bf1[27] = -bf0[27] + bf0[28];
  bf1[28] = bf0[28] + bf0[27];
  bf1[29] = bf0[29] + bf0[26];
  bf1[30] = bf0[30] + bf0[25];
  bf1[31] = bf0[31] + bf0[24];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[3];
  bf1[1] = bf0[1] + bf0[2];
  bf1[2] = -bf0[2] + bf0[1];
  bf1[3] = -bf0[3] + bf0[0];
  bf1[4] = bf0[4];
  bf1[5] = half_btf(-kCosPi[32], bf0[5], kCosPi[32], bf0[6]);
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[5]);
  bf1[7] = bf0[7];
  bf1[8] = bf0[8] + bf0[11];
  bf1[9] = bf0[9] + bf0[10];
  bf1[10] = -bf0[10] + bf0[9];
  bf1[11] = -bf0[11] + bf0[8];
  bf1[12] = -bf0[12] + bf0[15];
  bf1[13] = -bf0[13] + bf0[14];
  bf1[14] = bf0[14] + bf0[13];
  bf1[15] = bf0[15] + bf0[12];
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = half_btf(-kCosPi[16], bf0[18], kCosPi[48], bf0[29]);
  bf1[19] = half_btf(-kCosPi[16], bf0[19], kCosPi[48], bf0[28]);
  bf1[20] = half_btf(-kCosPi[48], bf0[20], -kCosPi[16], bf0[27]);
  bf1[21] = half_btf(-kCosPi[48], bf0[21], -kCosPi[16], bf0[26]);
  bf1[22] = bf0[22];
  bf1[23] = bf0[23];
  bf1[24] = bf0[24];
  bf1[25] = bf0[25];
  bf1[26] = half_btf(kCosPi[48], bf0[26], -kCosPi[16], bf0[21]);
  bf1[27] = half_btf(kCosPi[48], bf0[27], -kCosPi[16], bf0[20]);
  bf1[28] = half_btf(kCosPi[16], bf0[28], kCosPi[48], bf0[19]);
  bf1[29] = half_btf(kCosPi[16], bf0[29], kCosPi[48], bf0[18]);
  bf1[30] = bf0[30];
  bf1[31] = bf0[31];

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[32], bf0[0], kCosPi[32], bf0[1], shift);
  bf1[1] = half_btf(-kCosPi[32], bf0[1], kCosPi[32], bf0[0], shift);
  bf1[2] = half_btf(kCosPi[48], bf0[2], kCosPi[16], bf0[3], shift);
  bf1[3] = half_btf(kCosPi[48], bf0[3], -kCosPi[16], bf0[2], shift);
  bf1[4] = bf0[4] + bf0[5];
  bf1[5] = -bf0[5] + bf0[4];
  bf1[6] = -bf0[6] + bf0[7];
  bf1[7] = bf0[7] + bf0[6];
  bf1[8] = bf0[8];
  bf1[9] = half_btf(-kCosPi[16], bf0[9], kCosPi[48], bf0[14]);
  bf1[10] = half_btf(-kCosPi[48], bf0[10], -kCosPi[16], bf0[13]);
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = half_btf(kCosPi[48], bf0[13], -kCosPi[16], bf0[10]);
  bf1[14] = half_btf(kCosPi[16], bf0[14], kCosPi[48], bf0[9]);
  bf1[15] = bf0[15];
  bf1[16] = bf0[16] + bf0[19];
  bf1[17] = bf0[17] + bf0[18];
  bf1[18] = -bf0[18] + bf0[17];
  bf1[19] = -bf0[19] + bf0[16];
  bf1[20] = -bf0[20] + bf0[23];
  bf1[21] = -bf0[21] + bf0[22];
  bf1[22] = bf0[22] + bf0[21];
  bf1[23] = bf0[23] + bf0[20];
  bf1[24] = bf0[24] + bf0[27];
  bf1[25] = bf0[25] + bf0[26];
  bf1[26] = -bf0[26] + bf0[25];
  bf1[27] = -bf0[27] + bf0[24];
  bf1[28] = -bf0[28] + bf0[31];
  bf1[29] = -bf0[29] + bf0[30];
  bf1[30] = bf0[30] + bf0[29];
  bf1[31] = bf0[31] + bf0[28];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[56], bf0[4], kCosPi[8], bf0[7], shift);
  bf1[5] = half_btf(kCosPi[24], bf0[5], kCosPi[40], bf0[6], shift);
  bf1[6] = half_btf(kCosPi[24], bf0[6], -kCosPi[40], bf0[5], shift);
  bf1[7] = half_btf(kCosPi[56], bf0[7], -kCosPi[8], bf0[4], shift);
  bf1[8] = bf0[8] + bf0[9];
  bf1[9] = -bf0[9] + bf0[8];
  bf1[10] = -bf0[10] + bf0[11];
  bf1[11] = bf0[11] + bf0[10];
  bf1[12] = bf0[12] + bf0[13];
  bf1[13] = -bf0[13] + bf0[12];
  bf1[14] = -bf0[14] + bf0[15];
  bf1[15] = bf0[15] + bf0[14];
  bf1[16] = bf0[16];
  bf1[17] = half_btf(-kCosPi[8], bf0[17], kCosPi[56], bf0[30]);
  bf1[18] = half_btf(-kCosPi[56], bf0[18], -kCosPi[8], bf0[29]);
  bf1[19] = bf0[19];
  bf1[20] = bf0[20];
  bf1[21] = half_btf(-kCosPi[40], bf0[21], kCosPi[24], bf0[26]);
  bf1[22] = half_btf(-kCosPi[24], bf0[22], -kCosPi[40], bf0[25]);
  bf1[23] = bf0[23];
  bf1[24] = bf0[24];
  bf1[25] = half_btf(kCosPi[24], bf0[25], -kCosPi[40], bf0[22]);
  bf1[26] = half_btf(kCosPi[40], bf0[26], kCosPi[24], bf0[21]);
  bf1[27] = bf0[27];
  bf1[28] = bf0[28];
  bf1[29] = half_btf(kCosPi[56], bf0[29], -kCosPi[8], bf0[18]);
  bf1[30] = half_btf(kCosPi[8], bf0[30], kCosPi[56], bf0[17]);
  bf1[31] = bf0[31];

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[60], bf0[8], kCosPi[4], bf0[15], shift);
  bf1[9] = half_btf(kCosPi[28], bf0[9], kCosPi[36], bf0[14], shift);
  bf1[10] = half_btf(kCosPi[44], bf0[10], kCosPi[20], bf0[13], shift);
  bf1[11] = half_btf(kCosPi[12], bf0[11], kCosPi[52], bf0[12], shift);
  bf1[12] = half_btf(kCosPi[12], bf0[12], -kCosPi[52], bf0[11], shift);
  bf1[13] = half_btf(kCosPi[44], bf0[13], -kCosPi[20], bf0[10], shift);
  bf1[14] = half_btf(kCosPi[28], bf0[14], -kCosPi[36], bf0[9], shift);
  bf1[15] = half_btf(kCosPi[60], bf0[15], -kCosPi[4], bf0[8], shift);
  bf1[16] = bf0[16] + bf0[17];
  bf1[17] = -bf0[17] + bf0[16];
  bf1[18] = -bf0[18] + bf0[19];
  bf1[19] = bf0[19] + bf0[18];
  bf1[20] = bf0[20] + bf0[21];
  bf1[21] = -bf0[21] + bf0[20];
  bf1[22] = -bf0[22] + bf0[23];
  bf1[23] = bf0[23] + bf0[22];
  bf1[24] = bf0[24] + bf0[25];
  bf1[25] = -bf0[25] + bf0[24];
  bf1[26] = -bf0[26] + bf0[27];
  bf1[27] = bf0[27] + bf0[26];
  bf1[28] = bf0[28] + bf0[29];
  bf1[29] = -bf0[29] + bf0[28];
  bf1[30] = -bf0[30] + bf0[31];
  bf1[31] = bf0[31] + bf0[30];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = half_btf(kCosPi[62], bf0[16], kCosPi[2], bf0[31], shift);
  bf1[17] = half_btf(kCosPi[30], bf0[17], kCosPi[34], bf0[30], shift);
  bf1[18] = half_btf(kCosPi[46], bf0[18], kCosPi[18], bf0[29], shift);
  bf1[19] = half_btf(kCosPi[14], bf0[19], kCosPi[50], bf0[28], shift);
  bf1[20] = half_btf(kCosPi[54], bf0[20], kCosPi[10], bf0[27], shift);
  bf1[21] = half_btf(kCosPi[22], bf0[21], kCosPi[42], bf0[26], shift);
  bf1[22] = half_btf(kCosPi[38], bf0[22], kCosPi[26], bf0[25], shift);
  bf1[23] = half_btf(kCosPi[6], bf0[23], kCosPi[58], bf0[24], shift);
  bf1[24] = half_btf(kCosPi[6], bf0[24], -kCosPi[58], bf0[23], shift);
  bf1[25] = half_btf(kCosPi[38], bf0[25], -kCosPi[26], bf0[22], shift);
  bf1[26] = half_btf(kCosPi[22], bf0[26], -kCosPi[42], bf0[21], shift);
  bf1[27] = half_btf(kCosPi[54], bf0[27], -kCosPi[10], bf0[20], shift);
  bf1[28] = half_btf(kCosPi[14], bf0[28], -kCosPi[50], bf0[19], shift);
  bf1[29] = half_btf(kCosPi[46], bf0[29], -kCosPi[18], bf0[18], shift);
  bf1[30] = half_btf(kCosPi[30], bf0[30], -kCosPi[34], bf0[17], shift);
  bf1[31] = half_btf(kCosPi[62], bf0[31], -kCosPi[2], bf0[16], shift);

  // stage 9
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = bf0[16];
  coeffs[2] = bf0[8];
  coeffs[3] = bf0[24];
  coeffs[4] = bf0[4];
  coeffs[5] = bf0[20];
  coeffs[6] = bf0[12];
  coeffs[7] = bf0[28];
  coeffs[8] = bf0[2];
  coeffs[9] = bf0[18];
  coeffs[10] = bf0[10];
  coeffs[11] = bf0[26];
  coeffs[12] = bf0[6];
  coeffs[13] = bf0[22];
  coeffs[14] = bf0[14];
  coeffs[15] = bf0[30];
  coeffs[16] = bf0[1];
  coeffs[17] = bf0[17];
  coeffs[18] = bf0[9];
  coeffs[19] = bf0[25];
  coeffs[20] = bf0[5];
  coeffs[21] = bf0[21];
  coeffs[22] = bf0[13];
  coeffs[23] = bf0[29];
  coeffs[24] = bf0[3];
  coeffs[25] = bf0[19];
  coeffs[26] = bf0[11];
  coeffs[27] = bf0[27];
  coeffs[28] = bf0[7];
  coeffs[29] = bf0[23];
  coeffs[30] = bf0[15];
  coeffs[31] = bf0[31];
}

static void fadst4_C(int32_t* coeffs) {
  // stage 0
  int64_t x0 = coeffs[0];
  int64_t x1 = coeffs[1];
  int64_t x2 = coeffs[2];
  int64_t x3 = coeffs[3];

  if (!(x0 | x1 | x2 | x3)) {
    coeffs[0] = coeffs[1] = coeffs[2] = coeffs[3] = 0;
    return;
  }

  // stage 1
  int64_t s0 = (int64_t)kSinPi[1] * x0;
  int64_t s1 = (int64_t)kSinPi[4] * x0;
  int64_t s2 = (int64_t)kSinPi[2] * x1;
  int64_t s3 = (int64_t)kSinPi[1] * x1;
  int64_t s4 = (int64_t)kSinPi[3] * x2;
  int64_t s5 = (int64_t)kSinPi[4] * x3;
  int64_t s6 = (int64_t)kSinPi[2] * x3;
  int64_t s7 = x0 + x1;

  // stage 2
  s7 = s7 - x3;

  // stage 3
  x0 = s0 + s2;
  x1 = (int64_t)kSinPi[3] * s7;
  x2 = s1 - s3;
  x3 = s4;

  // stage 4
  x0 = x0 + s5;
  x2 = x2 + s6;

  // stage 5
  s0 = x0 + x3;
  s1 = x1;
  s2 = x2 - x3;
  s3 = x2 - x0;

  // stage 6
  s3 = s3 + x3;

  coeffs[0] = div_sqrt2(round_shift(s0));
  coeffs[1] = div_sqrt2(round_shift(s1));
  coeffs[2] = div_sqrt2(round_shift(s2));
  coeffs[3] = div_sqrt2(round_shift(s3));
}

static void fadst8_C(int32_t* coeffs) {
  const uint32_t shift = kPrecision + 1;
  int32_t tmp0[8], tmp1[8];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = -coeffs[7];
  bf1[2] = -coeffs[3];
  bf1[3] = coeffs[4];
  bf1[4] = -coeffs[1];
  bf1[5] = coeffs[6];
  bf1[6] = coeffs[2];
  bf1[7] = -coeffs[5];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = half_btf(kCosPi[32], bf0[2], kCosPi[32], bf0[3]);
  bf1[3] = half_btf(kCosPi[32], bf0[2], -kCosPi[32], bf0[3]);
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[7]);
  bf1[7] = half_btf(kCosPi[32], bf0[6], -kCosPi[32], bf0[7]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = bf0[0] - bf0[2];
  bf1[3] = bf0[1] - bf0[3];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = bf0[4] - bf0[6];
  bf1[7] = bf0[5] - bf0[7];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(kCosPi[48], bf0[4], -kCosPi[16], bf0[5]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[16], bf0[6], kCosPi[48], bf0[7]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = bf0[0] - bf0[4];
  bf1[5] = bf0[1] - bf0[5];
  bf1[6] = bf0[2] - bf0[6];
  bf1[7] = bf0[3] - bf0[7];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[4], bf0[0], kCosPi[60], bf0[1], shift);
  bf1[1] = half_btf(kCosPi[60], bf0[0], -kCosPi[4], bf0[1], shift);
  bf1[2] = half_btf(kCosPi[20], bf0[2], kCosPi[44], bf0[3], shift);
  bf1[3] = half_btf(kCosPi[44], bf0[2], -kCosPi[20], bf0[3], shift);
  bf1[4] = half_btf(kCosPi[36], bf0[4], kCosPi[28], bf0[5], shift);
  bf1[5] = half_btf(kCosPi[28], bf0[4], -kCosPi[36], bf0[5], shift);
  bf1[6] = half_btf(kCosPi[52], bf0[6], kCosPi[12], bf0[7], shift);
  bf1[7] = half_btf(kCosPi[12], bf0[6], -kCosPi[52], bf0[7], shift);

  // stage 7
  std::swap(bf0, bf1);
  coeffs[0] = bf0[1];
  coeffs[1] = bf0[6];
  coeffs[2] = bf0[3];
  coeffs[3] = bf0[4];
  coeffs[4] = bf0[5];
  coeffs[5] = bf0[2];
  coeffs[6] = bf0[7];
  coeffs[7] = bf0[0];
}

static void fadst16_C(int32_t* coeffs) {
  const uint32_t shift = kPrecision + 1;
  int32_t tmp0[16], tmp1[16];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[0];
  bf1[1] = -coeffs[15];
  bf1[2] = -coeffs[7];
  bf1[3] = coeffs[8];
  bf1[4] = -coeffs[3];
  bf1[5] = coeffs[12];
  bf1[6] = coeffs[4];
  bf1[7] = -coeffs[11];
  bf1[8] = -coeffs[1];
  bf1[9] = coeffs[14];
  bf1[10] = coeffs[6];
  bf1[11] = -coeffs[9];
  bf1[12] = coeffs[2];
  bf1[13] = -coeffs[13];
  bf1[14] = -coeffs[5];
  bf1[15] = coeffs[10];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = half_btf(kCosPi[32], bf0[2], kCosPi[32], bf0[3]);
  bf1[3] = half_btf(kCosPi[32], bf0[2], -kCosPi[32], bf0[3]);
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[7]);
  bf1[7] = half_btf(kCosPi[32], bf0[6], -kCosPi[32], bf0[7]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = half_btf(kCosPi[32], bf0[10], kCosPi[32], bf0[11]);
  bf1[11] = half_btf(kCosPi[32], bf0[10], -kCosPi[32], bf0[11]);
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = half_btf(kCosPi[32], bf0[14], kCosPi[32], bf0[15]);
  bf1[15] = half_btf(kCosPi[32], bf0[14], -kCosPi[32], bf0[15]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = bf0[0] - bf0[2];
  bf1[3] = bf0[1] - bf0[3];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = bf0[4] - bf0[6];
  bf1[7] = bf0[5] - bf0[7];
  bf1[8] = bf0[8] + bf0[10];
  bf1[9] = bf0[9] + bf0[11];
  bf1[10] = bf0[8] - bf0[10];
  bf1[11] = bf0[9] - bf0[11];
  bf1[12] = bf0[12] + bf0[14];
  bf1[13] = bf0[13] + bf0[15];
  bf1[14] = bf0[12] - bf0[14];
  bf1[15] = bf0[13] - bf0[15];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(kCosPi[48], bf0[4], -kCosPi[16], bf0[5]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[16], bf0[6], kCosPi[48], bf0[7]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = half_btf(kCosPi[16], bf0[12], kCosPi[48], bf0[13]);
  bf1[13] = half_btf(kCosPi[48], bf0[12], -kCosPi[16], bf0[13]);
  bf1[14] = half_btf(-kCosPi[48], bf0[14], kCosPi[16], bf0[15]);
  bf1[15] = half_btf(kCosPi[16], bf0[14], kCosPi[48], bf0[15]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = bf0[0] - bf0[4];
  bf1[5] = bf0[1] - bf0[5];
  bf1[6] = bf0[2] - bf0[6];
  bf1[7] = bf0[3] - bf0[7];
  bf1[8] = bf0[8] + bf0[12];
  bf1[9] = bf0[9] + bf0[13];
  bf1[10] = bf0[10] + bf0[14];
  bf1[11] = bf0[11] + bf0[15];
  bf1[12] = bf0[8] - bf0[12];
  bf1[13] = bf0[9] - bf0[13];
  bf1[14] = bf0[10] - bf0[14];
  bf1[15] = bf0[11] - bf0[15];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[8], bf0[8], kCosPi[56], bf0[9]);
  bf1[9] = half_btf(kCosPi[56], bf0[8], -kCosPi[8], bf0[9]);
  bf1[10] = half_btf(kCosPi[40], bf0[10], kCosPi[24], bf0[11]);
  bf1[11] = half_btf(kCosPi[24], bf0[10], -kCosPi[40], bf0[11]);
  bf1[12] = half_btf(-kCosPi[56], bf0[12], kCosPi[8], bf0[13]);
  bf1[13] = half_btf(kCosPi[8], bf0[12], kCosPi[56], bf0[13]);
  bf1[14] = half_btf(-kCosPi[24], bf0[14], kCosPi[40], bf0[15]);
  bf1[15] = half_btf(kCosPi[40], bf0[14], kCosPi[24], bf0[15]);

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[8];
  bf1[1] = bf0[1] + bf0[9];
  bf1[2] = bf0[2] + bf0[10];
  bf1[3] = bf0[3] + bf0[11];
  bf1[4] = bf0[4] + bf0[12];
  bf1[5] = bf0[5] + bf0[13];
  bf1[6] = bf0[6] + bf0[14];
  bf1[7] = bf0[7] + bf0[15];
  bf1[8] = bf0[0] - bf0[8];
  bf1[9] = bf0[1] - bf0[9];
  bf1[10] = bf0[2] - bf0[10];
  bf1[11] = bf0[3] - bf0[11];
  bf1[12] = bf0[4] - bf0[12];
  bf1[13] = bf0[5] - bf0[13];
  bf1[14] = bf0[6] - bf0[14];
  bf1[15] = bf0[7] - bf0[15];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[2], bf0[0], kCosPi[62], bf0[1]);
  bf1[1] = half_btf(kCosPi[62], bf0[0], -kCosPi[2], bf0[1]);
  bf1[2] = half_btf(kCosPi[10], bf0[2], kCosPi[54], bf0[3]);
  bf1[3] = half_btf(kCosPi[54], bf0[2], -kCosPi[10], bf0[3]);
  bf1[4] = half_btf(kCosPi[18], bf0[4], kCosPi[46], bf0[5]);
  bf1[5] = half_btf(kCosPi[46], bf0[4], -kCosPi[18], bf0[5]);
  bf1[6] = half_btf(kCosPi[26], bf0[6], kCosPi[38], bf0[7]);
  bf1[7] = half_btf(kCosPi[38], bf0[6], -kCosPi[26], bf0[7]);
  bf1[8] = half_btf(kCosPi[34], bf0[8], kCosPi[30], bf0[9]);
  bf1[9] = half_btf(kCosPi[30], bf0[8], -kCosPi[34], bf0[9]);
  bf1[10] = half_btf(kCosPi[42], bf0[10], kCosPi[22], bf0[11]);
  bf1[11] = half_btf(kCosPi[22], bf0[10], -kCosPi[42], bf0[11]);
  bf1[12] = half_btf(kCosPi[50], bf0[12], kCosPi[14], bf0[13]);
  bf1[13] = half_btf(kCosPi[14], bf0[12], -kCosPi[50], bf0[13]);
  bf1[14] = half_btf(kCosPi[58], bf0[14], kCosPi[6], bf0[15]);
  bf1[15] = half_btf(kCosPi[6], bf0[14], -kCosPi[58], bf0[15]);

  // stage 9
  std::swap(bf0, bf1);
  coeffs[0] = div_sqrt2(bf0[1], shift - kPrecision);
  coeffs[1] = div_sqrt2(bf0[14], shift - kPrecision);
  coeffs[2] = div_sqrt2(bf0[3], shift - kPrecision);
  coeffs[3] = div_sqrt2(bf0[12], shift - kPrecision);
  coeffs[4] = div_sqrt2(bf0[5], shift - kPrecision);
  coeffs[5] = div_sqrt2(bf0[10], shift - kPrecision);
  coeffs[6] = div_sqrt2(bf0[7], shift - kPrecision);
  coeffs[7] = div_sqrt2(bf0[8], shift - kPrecision);
  coeffs[8] = div_sqrt2(bf0[9], shift - kPrecision);
  coeffs[9] = div_sqrt2(bf0[6], shift - kPrecision);
  coeffs[10] = div_sqrt2(bf0[11], shift - kPrecision);
  coeffs[11] = div_sqrt2(bf0[4], shift - kPrecision);
  coeffs[12] = div_sqrt2(bf0[13], shift - kPrecision);
  coeffs[13] = div_sqrt2(bf0[2], shift - kPrecision);
  coeffs[14] = div_sqrt2(bf0[15], shift - kPrecision);
  coeffs[15] = div_sqrt2(bf0[0], shift - kPrecision);
}

static void fadst32_C(int32_t* coeffs) {
  constexpr uint32_t shift = kPrecision + 2;
  int32_t tmp0[32], tmp1[32];
  int32_t *bf0 = tmp0, *bf1 = tmp1;

  // stage 1;
  bf1[0] = coeffs[31];
  bf1[1] = coeffs[0];
  bf1[2] = coeffs[29];
  bf1[3] = coeffs[2];
  bf1[4] = coeffs[27];
  bf1[5] = coeffs[4];
  bf1[6] = coeffs[25];
  bf1[7] = coeffs[6];
  bf1[8] = coeffs[23];
  bf1[9] = coeffs[8];
  bf1[10] = coeffs[21];
  bf1[11] = coeffs[10];
  bf1[12] = coeffs[19];
  bf1[13] = coeffs[12];
  bf1[14] = coeffs[17];
  bf1[15] = coeffs[14];
  bf1[16] = coeffs[15];
  bf1[17] = coeffs[16];
  bf1[18] = coeffs[13];
  bf1[19] = coeffs[18];
  bf1[20] = coeffs[11];
  bf1[21] = coeffs[20];
  bf1[22] = coeffs[9];
  bf1[23] = coeffs[22];
  bf1[24] = coeffs[7];
  bf1[25] = coeffs[24];
  bf1[26] = coeffs[5];
  bf1[27] = coeffs[26];
  bf1[28] = coeffs[3];
  bf1[29] = coeffs[28];
  bf1[30] = coeffs[1];
  bf1[31] = coeffs[30];

  // stage 2
  std::swap(bf0, bf1);
  bf1[0] = half_btf(kCosPi[1], bf0[0], kCosPi[63], bf0[1]);
  bf1[1] = half_btf(-kCosPi[1], bf0[1], kCosPi[63], bf0[0]);
  bf1[2] = half_btf(kCosPi[5], bf0[2], kCosPi[59], bf0[3]);
  bf1[3] = half_btf(-kCosPi[5], bf0[3], kCosPi[59], bf0[2]);
  bf1[4] = half_btf(kCosPi[9], bf0[4], kCosPi[55], bf0[5]);
  bf1[5] = half_btf(-kCosPi[9], bf0[5], kCosPi[55], bf0[4]);
  bf1[6] = half_btf(kCosPi[13], bf0[6], kCosPi[51], bf0[7]);
  bf1[7] = half_btf(-kCosPi[13], bf0[7], kCosPi[51], bf0[6]);
  bf1[8] = half_btf(kCosPi[17], bf0[8], kCosPi[47], bf0[9]);
  bf1[9] = half_btf(-kCosPi[17], bf0[9], kCosPi[47], bf0[8]);
  bf1[10] = half_btf(kCosPi[21], bf0[10], kCosPi[43], bf0[11]);
  bf1[11] = half_btf(-kCosPi[21], bf0[11], kCosPi[43], bf0[10]);
  bf1[12] = half_btf(kCosPi[25], bf0[12], kCosPi[39], bf0[13]);
  bf1[13] = half_btf(-kCosPi[25], bf0[13], kCosPi[39], bf0[12]);
  bf1[14] = half_btf(kCosPi[29], bf0[14], kCosPi[35], bf0[15]);
  bf1[15] = half_btf(-kCosPi[29], bf0[15], kCosPi[35], bf0[14]);
  bf1[16] = half_btf(kCosPi[33], bf0[16], kCosPi[31], bf0[17]);
  bf1[17] = half_btf(-kCosPi[33], bf0[17], kCosPi[31], bf0[16]);
  bf1[18] = half_btf(kCosPi[37], bf0[18], kCosPi[27], bf0[19]);
  bf1[19] = half_btf(-kCosPi[37], bf0[19], kCosPi[27], bf0[18]);
  bf1[20] = half_btf(kCosPi[41], bf0[20], kCosPi[23], bf0[21]);
  bf1[21] = half_btf(-kCosPi[41], bf0[21], kCosPi[23], bf0[20]);
  bf1[22] = half_btf(kCosPi[45], bf0[22], kCosPi[19], bf0[23]);
  bf1[23] = half_btf(-kCosPi[45], bf0[23], kCosPi[19], bf0[22]);
  bf1[24] = half_btf(kCosPi[49], bf0[24], kCosPi[15], bf0[25]);
  bf1[25] = half_btf(-kCosPi[49], bf0[25], kCosPi[15], bf0[24]);
  bf1[26] = half_btf(kCosPi[53], bf0[26], kCosPi[11], bf0[27]);
  bf1[27] = half_btf(-kCosPi[53], bf0[27], kCosPi[11], bf0[26]);
  bf1[28] = half_btf(kCosPi[57], bf0[28], kCosPi[7], bf0[29]);
  bf1[29] = half_btf(-kCosPi[57], bf0[29], kCosPi[7], bf0[28]);
  bf1[30] = half_btf(kCosPi[61], bf0[30], kCosPi[3], bf0[31]);
  bf1[31] = half_btf(-kCosPi[61], bf0[31], kCosPi[3], bf0[30]);

  // stage 3
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[16];
  bf1[1] = bf0[1] + bf0[17];
  bf1[2] = bf0[2] + bf0[18];
  bf1[3] = bf0[3] + bf0[19];
  bf1[4] = bf0[4] + bf0[20];
  bf1[5] = bf0[5] + bf0[21];
  bf1[6] = bf0[6] + bf0[22];
  bf1[7] = bf0[7] + bf0[23];
  bf1[8] = bf0[8] + bf0[24];
  bf1[9] = bf0[9] + bf0[25];
  bf1[10] = bf0[10] + bf0[26];
  bf1[11] = bf0[11] + bf0[27];
  bf1[12] = bf0[12] + bf0[28];
  bf1[13] = bf0[13] + bf0[29];
  bf1[14] = bf0[14] + bf0[30];
  bf1[15] = bf0[15] + bf0[31];
  bf1[16] = -bf0[16] + bf0[0];
  bf1[17] = -bf0[17] + bf0[1];
  bf1[18] = -bf0[18] + bf0[2];
  bf1[19] = -bf0[19] + bf0[3];
  bf1[20] = -bf0[20] + bf0[4];
  bf1[21] = -bf0[21] + bf0[5];
  bf1[22] = -bf0[22] + bf0[6];
  bf1[23] = -bf0[23] + bf0[7];
  bf1[24] = -bf0[24] + bf0[8];
  bf1[25] = -bf0[25] + bf0[9];
  bf1[26] = -bf0[26] + bf0[10];
  bf1[27] = -bf0[27] + bf0[11];
  bf1[28] = -bf0[28] + bf0[12];
  bf1[29] = -bf0[29] + bf0[13];
  bf1[30] = -bf0[30] + bf0[14];
  bf1[31] = -bf0[31] + bf0[15];

  // stage 4
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = bf0[12];
  bf1[13] = bf0[13];
  bf1[14] = bf0[14];
  bf1[15] = bf0[15];
  bf1[16] = half_btf(kCosPi[4], bf0[16], kCosPi[60], bf0[17]);
  bf1[17] = half_btf(-kCosPi[4], bf0[17], kCosPi[60], bf0[16]);
  bf1[18] = half_btf(kCosPi[20], bf0[18], kCosPi[44], bf0[19]);
  bf1[19] = half_btf(-kCosPi[20], bf0[19], kCosPi[44], bf0[18]);
  bf1[20] = half_btf(kCosPi[36], bf0[20], kCosPi[28], bf0[21]);
  bf1[21] = half_btf(-kCosPi[36], bf0[21], kCosPi[28], bf0[20]);
  bf1[22] = half_btf(kCosPi[52], bf0[22], kCosPi[12], bf0[23]);
  bf1[23] = half_btf(-kCosPi[52], bf0[23], kCosPi[12], bf0[22]);
  bf1[24] = half_btf(-kCosPi[60], bf0[24], kCosPi[4], bf0[25]);
  bf1[25] = half_btf(kCosPi[60], bf0[25], kCosPi[4], bf0[24]);
  bf1[26] = half_btf(-kCosPi[44], bf0[26], kCosPi[20], bf0[27]);
  bf1[27] = half_btf(kCosPi[44], bf0[27], kCosPi[20], bf0[26]);
  bf1[28] = half_btf(-kCosPi[28], bf0[28], kCosPi[36], bf0[29]);
  bf1[29] = half_btf(kCosPi[28], bf0[29], kCosPi[36], bf0[28]);
  bf1[30] = half_btf(-kCosPi[12], bf0[30], kCosPi[52], bf0[31]);
  bf1[31] = half_btf(kCosPi[12], bf0[31], kCosPi[52], bf0[30]);

  // stage 5
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[8];
  bf1[1] = bf0[1] + bf0[9];
  bf1[2] = bf0[2] + bf0[10];
  bf1[3] = bf0[3] + bf0[11];
  bf1[4] = bf0[4] + bf0[12];
  bf1[5] = bf0[5] + bf0[13];
  bf1[6] = bf0[6] + bf0[14];
  bf1[7] = bf0[7] + bf0[15];
  bf1[8] = -bf0[8] + bf0[0];
  bf1[9] = -bf0[9] + bf0[1];
  bf1[10] = -bf0[10] + bf0[2];
  bf1[11] = -bf0[11] + bf0[3];
  bf1[12] = -bf0[12] + bf0[4];
  bf1[13] = -bf0[13] + bf0[5];
  bf1[14] = -bf0[14] + bf0[6];
  bf1[15] = -bf0[15] + bf0[7];
  bf1[16] = bf0[16] + bf0[24];
  bf1[17] = bf0[17] + bf0[25];
  bf1[18] = bf0[18] + bf0[26];
  bf1[19] = bf0[19] + bf0[27];
  bf1[20] = bf0[20] + bf0[28];
  bf1[21] = bf0[21] + bf0[29];
  bf1[22] = bf0[22] + bf0[30];
  bf1[23] = bf0[23] + bf0[31];
  bf1[24] = -bf0[24] + bf0[16];
  bf1[25] = -bf0[25] + bf0[17];
  bf1[26] = -bf0[26] + bf0[18];
  bf1[27] = -bf0[27] + bf0[19];
  bf1[28] = -bf0[28] + bf0[20];
  bf1[29] = -bf0[29] + bf0[21];
  bf1[30] = -bf0[30] + bf0[22];
  bf1[31] = -bf0[31] + bf0[23];

  // stage 6
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = bf0[4];
  bf1[5] = bf0[5];
  bf1[6] = bf0[6];
  bf1[7] = bf0[7];
  bf1[8] = half_btf(kCosPi[8], bf0[8], kCosPi[56], bf0[9]);
  bf1[9] = half_btf(-kCosPi[8], bf0[9], kCosPi[56], bf0[8]);
  bf1[10] = half_btf(kCosPi[40], bf0[10], kCosPi[24], bf0[11]);
  bf1[11] = half_btf(-kCosPi[40], bf0[11], kCosPi[24], bf0[10]);
  bf1[12] = half_btf(-kCosPi[56], bf0[12], kCosPi[8], bf0[13]);
  bf1[13] = half_btf(kCosPi[56], bf0[13], kCosPi[8], bf0[12]);
  bf1[14] = half_btf(-kCosPi[24], bf0[14], kCosPi[40], bf0[15]);
  bf1[15] = half_btf(kCosPi[24], bf0[15], kCosPi[40], bf0[14]);
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = bf0[20];
  bf1[21] = bf0[21];
  bf1[22] = bf0[22];
  bf1[23] = bf0[23];
  bf1[24] = half_btf(kCosPi[8], bf0[24], kCosPi[56], bf0[25]);
  bf1[25] = half_btf(-kCosPi[8], bf0[25], kCosPi[56], bf0[24]);
  bf1[26] = half_btf(kCosPi[40], bf0[26], kCosPi[24], bf0[27]);
  bf1[27] = half_btf(-kCosPi[40], bf0[27], kCosPi[24], bf0[26]);
  bf1[28] = half_btf(-kCosPi[56], bf0[28], kCosPi[8], bf0[29]);
  bf1[29] = half_btf(kCosPi[56], bf0[29], kCosPi[8], bf0[28]);
  bf1[30] = half_btf(-kCosPi[24], bf0[30], kCosPi[40], bf0[31]);
  bf1[31] = half_btf(kCosPi[24], bf0[31], kCosPi[40], bf0[30]);

  // stage 7
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[4];
  bf1[1] = bf0[1] + bf0[5];
  bf1[2] = bf0[2] + bf0[6];
  bf1[3] = bf0[3] + bf0[7];
  bf1[4] = -bf0[4] + bf0[0];
  bf1[5] = -bf0[5] + bf0[1];
  bf1[6] = -bf0[6] + bf0[2];
  bf1[7] = -bf0[7] + bf0[3];
  bf1[8] = bf0[8] + bf0[12];
  bf1[9] = bf0[9] + bf0[13];
  bf1[10] = bf0[10] + bf0[14];
  bf1[11] = bf0[11] + bf0[15];
  bf1[12] = -bf0[12] + bf0[8];
  bf1[13] = -bf0[13] + bf0[9];
  bf1[14] = -bf0[14] + bf0[10];
  bf1[15] = -bf0[15] + bf0[11];
  bf1[16] = bf0[16] + bf0[20];
  bf1[17] = bf0[17] + bf0[21];
  bf1[18] = bf0[18] + bf0[22];
  bf1[19] = bf0[19] + bf0[23];
  bf1[20] = -bf0[20] + bf0[16];
  bf1[21] = -bf0[21] + bf0[17];
  bf1[22] = -bf0[22] + bf0[18];
  bf1[23] = -bf0[23] + bf0[19];
  bf1[24] = bf0[24] + bf0[28];
  bf1[25] = bf0[25] + bf0[29];
  bf1[26] = bf0[26] + bf0[30];
  bf1[27] = bf0[27] + bf0[31];
  bf1[28] = -bf0[28] + bf0[24];
  bf1[29] = -bf0[29] + bf0[25];
  bf1[30] = -bf0[30] + bf0[26];
  bf1[31] = -bf0[31] + bf0[27];

  // stage 8
  std::swap(bf0, bf1);
  bf1[0] = bf0[0];
  bf1[1] = bf0[1];
  bf1[2] = bf0[2];
  bf1[3] = bf0[3];
  bf1[4] = half_btf(kCosPi[16], bf0[4], kCosPi[48], bf0[5]);
  bf1[5] = half_btf(-kCosPi[16], bf0[5], kCosPi[48], bf0[4]);
  bf1[6] = half_btf(-kCosPi[48], bf0[6], kCosPi[16], bf0[7]);
  bf1[7] = half_btf(kCosPi[48], bf0[7], kCosPi[16], bf0[6]);
  bf1[8] = bf0[8];
  bf1[9] = bf0[9];
  bf1[10] = bf0[10];
  bf1[11] = bf0[11];
  bf1[12] = half_btf(kCosPi[16], bf0[12], kCosPi[48], bf0[13]);
  bf1[13] = half_btf(-kCosPi[16], bf0[13], kCosPi[48], bf0[12]);
  bf1[14] = half_btf(-kCosPi[48], bf0[14], kCosPi[16], bf0[15]);
  bf1[15] = half_btf(kCosPi[48], bf0[15], kCosPi[16], bf0[14]);
  bf1[16] = bf0[16];
  bf1[17] = bf0[17];
  bf1[18] = bf0[18];
  bf1[19] = bf0[19];
  bf1[20] = half_btf(kCosPi[16], bf0[20], kCosPi[48], bf0[21]);
  bf1[21] = half_btf(-kCosPi[16], bf0[21], kCosPi[48], bf0[20]);
  bf1[22] = half_btf(-kCosPi[48], bf0[22], kCosPi[16], bf0[23]);
  bf1[23] = half_btf(kCosPi[48], bf0[23], kCosPi[16], bf0[22]);
  bf1[24] = bf0[24];
  bf1[25] = bf0[25];
  bf1[26] = bf0[26];
  bf1[27] = bf0[27];
  bf1[28] = half_btf(kCosPi[16], bf0[28], kCosPi[48], bf0[29]);
  bf1[29] = half_btf(-kCosPi[16], bf0[29], kCosPi[48], bf0[28]);
  bf1[30] = half_btf(-kCosPi[48], bf0[30], kCosPi[16], bf0[31]);
  bf1[31] = half_btf(kCosPi[48], bf0[31], kCosPi[16], bf0[30]);

  // stage 9
  std::swap(bf0, bf1);
  bf1[0] = bf0[0] + bf0[2];
  bf1[1] = bf0[1] + bf0[3];
  bf1[2] = -bf0[2] + bf0[0];
  bf1[3] = -bf0[3] + bf0[1];
  bf1[4] = bf0[4] + bf0[6];
  bf1[5] = bf0[5] + bf0[7];
  bf1[6] = -bf0[6] + bf0[4];
  bf1[7] = -bf0[7] + bf0[5];
  bf1[8] = bf0[8] + bf0[10];
  bf1[9] = bf0[9] + bf0[11];
  bf1[10] = -bf0[10] + bf0[8];
  bf1[11] = -bf0[11] + bf0[9];
  bf1[12] = bf0[12] + bf0[14];
  bf1[13] = bf0[13] + bf0[15];
  bf1[14] = -bf0[14] + bf0[12];
  bf1[15] = -bf0[15] + bf0[13];
  bf1[16] = bf0[16] + bf0[18];
  bf1[17] = bf0[17] + bf0[19];
  bf1[18] = -bf0[18] + bf0[16];
  bf1[19] = -bf0[19] + bf0[17];
  bf1[20] = bf0[20] + bf0[22];
  bf1[21] = bf0[21] + bf0[23];
  bf1[22] = -bf0[22] + bf0[20];
  bf1[23] = -bf0[23] + bf0[21];
  bf1[24] = bf0[24] + bf0[26];
  bf1[25] = bf0[25] + bf0[27];
  bf1[26] = -bf0[26] + bf0[24];
  bf1[27] = -bf0[27] + bf0[25];
  bf1[28] = bf0[28] + bf0[30];
  bf1[29] = bf0[29] + bf0[31];
  bf1[30] = -bf0[30] + bf0[28];
  bf1[31] = -bf0[31] + bf0[29];

  // stage 10
  std::swap(bf0, bf1);
  bf1[0] = round_shift(bf0[0], shift - kPrecision);
  bf1[1] = round_shift(bf0[1], shift - kPrecision);
  bf1[2] = half_btf(kCosPi[32], bf0[2], kCosPi[32], bf0[3], shift);
  bf1[3] = half_btf(-kCosPi[32], bf0[3], kCosPi[32], bf0[2], shift);
  bf1[4] = round_shift(bf0[4], shift - kPrecision);
  bf1[5] = round_shift(bf0[5], shift - kPrecision);
  bf1[6] = half_btf(kCosPi[32], bf0[6], kCosPi[32], bf0[7], shift);
  bf1[7] = half_btf(-kCosPi[32], bf0[7], kCosPi[32], bf0[6], shift);
  bf1[8] = round_shift(bf0[8], shift - kPrecision);
  bf1[9] = round_shift(bf0[9], shift - kPrecision);
  bf1[10] = half_btf(kCosPi[32], bf0[10], kCosPi[32], bf0[11], shift);
  bf1[11] = half_btf(-kCosPi[32], bf0[11], kCosPi[32], bf0[10], shift);
  bf1[12] = round_shift(bf0[12], shift - kPrecision);
  bf1[13] = round_shift(bf0[13], shift - kPrecision);
  bf1[14] = half_btf(kCosPi[32], bf0[14], kCosPi[32], bf0[15], shift);
  bf1[15] = half_btf(-kCosPi[32], bf0[15], kCosPi[32], bf0[14], shift);
  bf1[16] = round_shift(bf0[16], shift - kPrecision);
  bf1[17] = round_shift(bf0[17], shift - kPrecision);
  bf1[18] = half_btf(kCosPi[32], bf0[18], kCosPi[32], bf0[19], shift);
  bf1[19] = half_btf(-kCosPi[32], bf0[19], kCosPi[32], bf0[18], shift);
  bf1[20] = round_shift(bf0[20], shift - kPrecision);
  bf1[21] = round_shift(bf0[21], shift - kPrecision);
  bf1[22] = half_btf(kCosPi[32], bf0[22], kCosPi[32], bf0[23], shift);
  bf1[23] = half_btf(-kCosPi[32], bf0[23], kCosPi[32], bf0[22], shift);
  bf1[24] = round_shift(bf0[24], shift - kPrecision);
  bf1[25] = round_shift(bf0[25], shift - kPrecision);
  bf1[26] = half_btf(kCosPi[32], bf0[26], kCosPi[32], bf0[27], shift);
  bf1[27] = half_btf(-kCosPi[32], bf0[27], kCosPi[32], bf0[26], shift);
  bf1[28] = round_shift(bf0[28], shift - kPrecision);
  bf1[29] = round_shift(bf0[29], shift - kPrecision);
  bf1[30] = half_btf(kCosPi[32], bf0[30], kCosPi[32], bf0[31], shift);
  bf1[31] = half_btf(-kCosPi[32], bf0[31], kCosPi[32], bf0[30], shift);

  // stage 11
  std::swap(bf0, bf1);
  coeffs[0] = bf0[0];
  coeffs[1] = -bf0[16];
  coeffs[2] = bf0[24];
  coeffs[3] = -bf0[8];
  coeffs[4] = bf0[12];
  coeffs[5] = -bf0[28];
  coeffs[6] = bf0[20];
  coeffs[7] = -bf0[4];
  coeffs[8] = bf0[6];
  coeffs[9] = -bf0[22];
  coeffs[10] = bf0[30];
  coeffs[11] = -bf0[14];
  coeffs[12] = bf0[10];
  coeffs[13] = -bf0[26];
  coeffs[14] = bf0[18];
  coeffs[15] = -bf0[2];
  coeffs[16] = bf0[3];
  coeffs[17] = -bf0[19];
  coeffs[18] = bf0[27];
  coeffs[19] = -bf0[11];
  coeffs[20] = bf0[15];
  coeffs[21] = -bf0[31];
  coeffs[22] = bf0[23];
  coeffs[23] = -bf0[7];
  coeffs[24] = bf0[5];
  coeffs[25] = -bf0[21];
  coeffs[26] = bf0[29];
  coeffs[27] = -bf0[13];
  coeffs[28] = bf0[9];
  coeffs[29] = -bf0[25];
  coeffs[30] = bf0[17];
  coeffs[31] = -bf0[1];
}

//------------------------------------------------------------------------------

static void Transpose16_C(const int16_t* in, uint32_t w, uint32_t h,
                          int16_t* out) {
  assert(in != out);
  for (size_t j = 0; j < h; ++j) {
    for (size_t i = 0; i < w; ++i) {
      out[j + h * i] = in[i + w * j];
    }
  }
}

static void Transpose32_C(const int32_t* in, uint32_t w, uint32_t h,
                          int32_t* out) {
  assert(in != out);
  for (size_t j = 0; j < h; ++j) {
    for (size_t i = 0; i < w; ++i) {
      out[j + h * i] = in[i + w * j];
    }
  }
}

//------------------------------------------------------------------------------

static int32_t SlowDct8x8_C(const int32_t in[64], const float cos_x[8],
                            const float cos_y[8]) {
  float sum = 0.;
  for (uint32_t j = 0; j < 8; ++j) {
    for (uint32_t i = 0; i < 8; ++i) {
      sum += in[i + j * 8] * cos_x[i] * cos_y[j];
    }
  }
  return (int32_t)lrintf(256.f * sum);
}

//------------------------------------------------------------------------------

WP2FwdTransposeF WP2Transpose32b = nullptr;
WP2InvTransposeF WP2Transpose16b = nullptr;
WP2InvTransformF WP2InvDct[5] = {nullptr}, WP2InvAdst[5] = {nullptr};
WP2FwdTransformF WP2FwdDct[5] = {nullptr}, WP2FwdAdst[5] = {nullptr};
WP2InvTransformF WP2InvHadamard[5] = {nullptr};
WP2FwdTransformF WP2FwdHadamard[5] = {nullptr};
WP2InvTransformColF WP2InvDctCol[5][5] = {{nullptr}};
WP2FwdTransformColF WP2FwdDctCol[5][5] = {{nullptr}};

int32_t (*WP2SlowDct8x8)(const int32_t in[64], const float cos_x[8],
                         const float cos_y[8]) = nullptr;

static volatile WP2CPUInfo transform_last_cpuinfo_used =
    (WP2CPUInfo)&transform_last_cpuinfo_used;

extern void WP2TransformInitSSE();

WP2_TSAN_IGNORE_FUNCTION void WP2TransformInit() {
  if (transform_last_cpuinfo_used == WP2GetCPUInfo) return;

  WP2Transpose16b = Transpose16_C;
  WP2Transpose32b = Transpose32_C;

  WP2InvDct[0] = idct2_C;
  WP2InvDct[1] = idct4_C;
  WP2InvDct[2] = idct8_C;
  WP2InvDct[3] = idct16_C;
  WP2InvDct[4] = idct32_C;
  WP2FwdDct[0] = fdct2_C;
  WP2FwdDct[1] = fdct4_C;
  WP2FwdDct[2] = fdct8_C;
  WP2FwdDct[3] = fdct16_C;
  WP2FwdDct[4] = fdct32_C;
  WP2InvAdst[0] = idct2_C;
  WP2InvAdst[1] = iadst4_C;
  WP2InvAdst[2] = iadst8_C;
  WP2InvAdst[3] = iadst16_C;
  WP2InvAdst[4] = iadst32_C;
  WP2FwdAdst[0] = fdct2_C;
  WP2FwdAdst[1] = fadst4_C;
  WP2FwdAdst[2] = fadst8_C;
  WP2FwdAdst[3] = fadst16_C;
  WP2FwdAdst[4] = fadst32_C;
  WP2InvHadamard[0] = iHadamard2_C;
  WP2InvHadamard[1] = iHadamard4_C;
  WP2InvHadamard[2] = iHadamard8_C;
  WP2InvHadamard[3] = iHadamard16_C;
  WP2InvHadamard[4] = iHadamard32_C;
  WP2FwdHadamard[0] = fHadamard2_C;
  WP2FwdHadamard[1] = fHadamard4_C;
  WP2FwdHadamard[2] = fHadamard8_C;
  WP2FwdHadamard[3] = fHadamard16_C;
  WP2FwdHadamard[4] = fHadamard32_C;

  for (uint32_t th : {0, 1, 2, 3, 4}) {
    for (uint32_t tw : {0, 1, 2, 3, 4}) {
      WP2InvDctCol[th][tw] = nullptr;
      WP2FwdDctCol[th][tw] = nullptr;
    }
  }

  WP2SlowDct8x8 = SlowDct8x8_C;

  if (WP2GetCPUInfo != nullptr) {
#if defined(WP2_USE_SSE)
    if (WP2GetCPUInfo(kSSE)) WP2TransformInitSSE();
#endif
  }

  transform_last_cpuinfo_used = WP2GetCPUInfo;
}

//------------------------------------------------------------------------------
// FwdTransform entry points

static WP2FwdTransformF GetFwdFunc(WP2TransformType type, uint32_t size) {
  const uint32_t idx = SizeToIdx(size);
  switch (type) {
    case WP2TransformType::kDct:
      return WP2FwdDct[idx];
    case WP2TransformType::kAdst:
      return WP2FwdAdst[idx];
    case WP2TransformType::kHadamard:
      return WP2FwdHadamard[idx];
    case WP2TransformType::kIdentity:
      return nullptr;
    default:
      assert(false);
  }
  return nullptr;
}

void WP2Transform(int32_t coeffs[], WP2TransformType type, uint32_t size,
                  uint32_t num_rows) {
  WP2FwdTransformF transform = GetFwdFunc(type, size);
  if (transform != nullptr) {
    for (size_t j = 0; j < num_rows; ++j) transform(coeffs + j * size);
  }
}

void WP2Transform(int16_t coeffs[], WP2TransformType type, uint32_t size) {
  WP2FwdTransformF transform = GetFwdFunc(type, size);
  if (transform != nullptr) {
    int32_t tmp[32];
    for (uint32_t i = 0; i < size; ++i) tmp[i] = coeffs[i];
    transform(tmp);
    for (uint32_t i = 0; i < size; ++i) {
      coeffs[i] = WP2::ClampToSigned(tmp[i], 16);
    }
  }
}

static void Transform2D(const int16_t src[], WP2TransformType tf_x,
                        WP2TransformType tf_y, uint32_t size_x, uint32_t size_y,
                        int32_t dst[]) {
  int32_t tmp[32 * 32];
  const uint32_t tw = size_x;
  const uint32_t th = size_y;
  for (uint32_t i = 0; i < tw * th; ++i) dst[i] = src[i];
  WP2Transform(dst, tf_x, size_x, th);
  if (tf_y == WP2TransformType::kDct) {
    const WP2FwdTransformColF transform =
        WP2FwdDctCol[SizeToIdx(size_y)][SizeToIdx(size_x)];
    if (transform != nullptr) {
      transform(dst, tw);
      return;
    }
  }
  WP2Transpose32b(dst, tw, th, tmp);  // tw x th -> th x tw
  WP2Transform(tmp, tf_y, size_y, tw);
  WP2Transpose32b(tmp, th, tw, dst);  // th x tw -> tw x th
}

//------------------------------------------------------------------------------
// InvTransform entry points

void WP2InvTransform(int16_t coeffs[], WP2TransformType type, uint32_t size,
                     uint32_t num_rows) {
  const uint32_t idx = SizeToIdx(size);
  WP2InvTransformF transform = nullptr;
  // clang-format off
  switch (type) {
    case WP2TransformType::kDct: transform = WP2InvDct[idx]; break;
    case WP2TransformType::kAdst: transform = WP2InvAdst[idx]; break;
    case WP2TransformType::kHadamard: transform = WP2InvHadamard[idx]; break;
    case WP2TransformType::kIdentity:
      return;
    default:
      assert(false);
  }
  for (size_t j = 0; j < num_rows; ++j) transform(coeffs + j * size);
  // clang-format on
}

// 'x_end' is the first x for which all coeffs are 0.
// TODO(vrabaud) the DC only case can be sped up.
static void InvTransform2D(int16_t coeffs[], WP2TransformType tf_x,
                           WP2TransformType tf_y, uint32_t size_x,
                           uint32_t x_end, uint32_t size_y) {
  const uint32_t tw = size_x;
  const uint32_t th = size_y;
  const WP2InvTransformColF transform =
      (tf_y == WP2TransformType::kDct)
          ? WP2InvDctCol[SizeToIdx(size_y)][SizeToIdx(size_x)]
          : nullptr;
  if (transform != nullptr) {
    transform(coeffs, tw);
  } else {
    int16_t tmp[32 * 32];
    WP2Transpose16b(coeffs, tw, th, tmp);  // tw x th -> th x tw
    WP2InvTransform(tmp, tf_y, size_y, x_end);
    std::fill(tmp + x_end * th, tmp + tw * th, 0);
    WP2Transpose16b(tmp, th, tw, coeffs);  // th x tw -> tw x th
  }
  WP2InvTransform(coeffs, tf_x, size_x, th);
}

//------------------------------------------------------------------------------

void WP2Transform2D(const int16_t src[], WP2TransformType tf_x,
                    WP2TransformType tf_y, uint32_t size_x, uint32_t size_y,
                    int32_t dst[], bool reduced) {
  if (!reduced) return Transform2D(src, tf_x, tf_y, size_x, size_y, dst);
  const uint32_t tw = size_x / 2;
  const uint32_t th = size_y / 2;

  int16_t tmp[32 * 32];
  for (uint32_t y = 0; y < th; ++y) {
    for (uint32_t x = 0; x < tw; ++x) {
      const size_t offset = 2 * y * 2 * tw + 2 * x;
      tmp[x + tw * y] =
          (src[offset + 0] + src[offset + 1] + src[offset + 2 * tw + 0] +
           src[offset + 2 * tw + 1] + 2) >>
          2;
    }
  }
  Transform2D(tmp, tf_x, tf_y, tw, th, dst);
}

//------------------------------------------------------------------------------
// Upsampling

// Upsampling taps
static constexpr int32_t kTaps[4][9] = {
    {64, 9, -9, 9, -9, 2, -1, -1, 0},
    {64, -9, 9, 9, -9, -1, 2, 0, -1},
    {64, 9, -9, -9, 9, -1, 0, 2, -1},
    {64, -9, 9, -9, 9, 0, -1, -1, 2},
};

// in-place 16 bit upsample
static void Upsample(uint32_t tw, uint32_t th, int16_t coeffs[]) {
  int16_t tmp[16 * 16];
  assert(tw <= 16 && th <= 16);
  std::copy(coeffs, coeffs + tw * th, tmp);
  for (uint32_t y = 0; y < th; ++y) {
    for (uint32_t x = 0; x < tw; ++x) {
      const uint32_t xp = (x + 1 < tw) ? x + 1 : x;
      const uint32_t yp = (y + 1 < th) ? y + 1 : y;
      const uint32_t xm = (x > 0) ? x - 1 : 0;
      const uint32_t ym = (y > 0) ? y - 1 : 0;
      // Source sample numbering (samples[]):
      // 5...3...6
      // .   .   .
      // .   .   .
      // .  x.x  .
      // 1...0...2
      // .  x.x  .
      // .   .   .
      // .   .   .
      // 7...4...8
      // (interpolated samples v[] marked with 'x')
      const int32_t samples[9] = {
          tmp[x + tw * y],                       // #0
          tmp[xm + tw * y],  tmp[xp + tw * y],   // #1 #2
          tmp[x + tw * ym],  tmp[x + tw * yp],   // #3 #4
          tmp[xm + tw * ym], tmp[xp + tw * ym],  // #5 #6
          tmp[xm + tw * yp], tmp[xp + tw * yp]   // #7 #8
      };
      int32_t v[4] = {32, 32, 32, 32};
      for (uint32_t i = 0; i < 9; ++i) {
        v[0] += kTaps[0][i] * samples[i];
        v[1] += kTaps[1][i] * samples[i];
        v[2] += kTaps[2][i] * samples[i];
        v[3] += kTaps[3][i] * samples[i];
      }
      const size_t offset = 2 * y * 2 * tw + 2 * x;
      coeffs[offset + 0] = v[0] >> 6;
      coeffs[offset + 1] = v[1] >> 6;
      coeffs[offset + 2 * tw + 0] = v[2] >> 6;
      coeffs[offset + 2 * tw + 1] = v[3] >> 6;
    }
  }
}

void WP2InvTransform2D(int16_t coeffs[], WP2TransformType tf_x,
                       WP2TransformType tf_y, uint32_t size_x, uint32_t size_y,
                       bool reduced) {
  const uint32_t tw = reduced ? size_x / 2 : size_x;
  const uint32_t th = reduced ? size_y / 2 : size_y;
  // Figure out the bounding column.
  uint32_t x_end = 0;
  for (uint32_t j = 0; j < th; ++j) {
    for (uint32_t i = x_end; i < tw; ++i) {
      if (coeffs[j * tw + i] != 0) x_end = i + 1;
    }
  }
  // If the residuals are null.
  if (x_end == 0) {
    std::fill(coeffs, coeffs + size_x * size_y, 0);
    return;
  }
  bool dc_only = false;
  // Check if we are DC only.
  if (x_end == 1 && tf_x == kDct && tf_y == kDct) {
    uint32_t y_end = th;
    while (y_end > 0 && coeffs[(y_end - 1) * tw] == 0) --y_end;
    assert(y_end > 0);
    dc_only = (y_end == 1);
  }

  InvTransform2D(coeffs, tf_x, tf_y, tw, x_end, th);
  if (!reduced) return;
  if (dc_only) {
    for (uint32_t i = tw * th; i < size_x * size_y; ++i) coeffs[i] = coeffs[0];
    return;
  }
  Upsample(tw, th, coeffs);
}

//------------------------------------------------------------------------------

void WP2ReduceCoeffs(const int32_t src[], uint32_t size_x, uint32_t size_y,
                     int32_t dst[]) {
  const uint32_t half_size_x = size_x / 2;
  const uint32_t half_size_y = size_y / 2;
  for (uint32_t y = 0; y < half_size_y; ++y) {
    for (uint32_t x = 0; x < half_size_x; ++x) {
      const int32_t v0 = src[x + size_x * y];
      // rounds toward 0 with a +1 bias, using >>'s instead of /2
      dst[x + half_size_x * y] = (v0 + (v0 < 0)) >> 1;
    }
  }
}

//------------------------------------------------------------------------------
