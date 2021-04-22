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
//   Coefficient Quantization
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cstdio>

#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size_io.h"
#include "src/utils/utils.h"

namespace WP2 {

//------------------------------------------------------------------------------

// idx at which we use the next quant parameter (for both x and y directions)
static constexpr uint32_t kZones[kNumQuantZones] = { 1, 4, 16, 32 };

// Linearly interpolate values[] at point 'n' in interval [s, e]
// TODO(skal): use fixed-point precision
static uint32_t Interpolate(const uint16_t values[],
                            uint32_t s, uint32_t e, uint32_t n) {
  const float v = 1.f * (values[0] * (e - n) + values[1] * (n - s)) / (e - s);
  return (uint32_t)std::lrintf(v);
}

WP2Status QuantMtx::AllocateForEncoder() {
  WP2_CHECK_ALLOC_OK(iquants.resize(kDataSize));
  WP2_CHECK_ALLOC_OK(bias.resize(kDataSize));
  return WP2_STATUS_OK;
}

// clang-format off
static constexpr uint16_t Qs[kQFactorMax + 1] = {
     12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,
     25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,
     38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,
     51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,
     64,   65,   66,   67,   68,   70,   71,   72,   73,   74,   75,   76,   77,
     78,   80,   81,   82,   83,   84,   86,   87,   88,   90,   91,   92,   94,
     95,   97,   98,   99,  101,  102,  104,  106,  107,  109,  110,  112,  114,
    115,  117,  119,  121,  122,  124,  126,  128,  130,  132,  134,  136,  138,
    140,  142,  144,  146,  149,  152,  153,  155,  158,  160,  162,  166,  167,
    170,  173,  175,  178,  180,  183,  186,  189,  191,  194,  197,  200,  203,
    206,  209,  212,  215,  219,  222,  225,  229,  232,  236,  239,  243,  246,
    250,  254,  258,  261,  265,  269,  273,  277,  282,  286,  291,  295,  299,
    303,  309,  312,  318,  322,  327,  332,  336,  341,  348,  352,  358,  362,
    368,  374,  381,  385,  390,  397,  402,  409,  414,  420,  428,  434,  439,
    448,  455,  461,  468,  474,  481,  489,  496,  504,  512,  520,  528,  537,
    541,  550,  560,  569,  574,  585,  595,  601,  612,  618,  630,  642,  648,
    661,  668,  682,  689,  697,  712,  720,  728,  744,  753,  762,  780,  789,
    799,  809,  819,  840,  851,  862,  873,  885,  897,  910,  923,  936,  963,
    978,  992, 1008, 1024,
};
// clang-format on

struct FindIdxLUT {
#if 0
  // Code to generate the LUT at compile time.
  constexpr FindIdxLUT() : kArr() {
    for (uint32_t i = 0, q = 0; i < kQFactorMax + 1; ++i) {
      for (; q <= Qs[i]; ++q) kArr[q] = i;
    }
  }
  uint8_t kArr[Qs[kQFactorMax] + 1];
#else
  static constexpr uint8_t kArr[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,
      3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
      18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
      33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  57,  58,  59,  60,  61,
      62,  63,  64,  65,  66,  66,  67,  68,  69,  70,  71,  71,  72,  73,  74,
      74,  75,  76,  77,  77,  78,  79,  79,  80,  81,  82,  82,  83,  84,  84,
      85,  85,  86,  87,  87,  88,  89,  89,  90,  90,  91,  92,  92,  93,  93,
      94,  94,  95,  96,  96,  97,  97,  98,  98,  99,  99,  100, 100, 101, 101,
      102, 102, 103, 103, 104, 104, 105, 105, 106, 106, 107, 107, 108, 108, 108,
      109, 109, 109, 110, 111, 111, 112, 112, 112, 113, 113, 114, 114, 115, 115,
      115, 115, 116, 117, 117, 117, 118, 118, 118, 119, 119, 120, 120, 120, 121,
      121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 126, 126, 126,
      127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
      132, 132, 132, 133, 133, 133, 134, 134, 134, 134, 135, 135, 135, 136, 136,
      136, 137, 137, 137, 137, 138, 138, 138, 139, 139, 139, 139, 140, 140, 140,
      141, 141, 141, 141, 142, 142, 142, 143, 143, 143, 143, 144, 144, 144, 144,
      145, 145, 145, 145, 146, 146, 146, 147, 147, 147, 147, 148, 148, 148, 148,
      149, 149, 149, 149, 150, 150, 150, 150, 151, 151, 151, 151, 151, 152, 152,
      152, 152, 153, 153, 153, 153, 153, 154, 154, 154, 154, 155, 155, 155, 155,
      156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 158, 158, 158, 159, 159,
      159, 159, 159, 159, 160, 160, 160, 160, 161, 161, 161, 161, 161, 162, 162,
      162, 162, 162, 163, 163, 163, 163, 164, 164, 164, 164, 164, 165, 165, 165,
      165, 165, 165, 165, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 168,
      168, 168, 168, 169, 169, 169, 169, 169, 169, 170, 170, 170, 170, 170, 170,
      171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 173, 173, 173, 173,
      173, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 176, 176,
      176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178,
      178, 179, 179, 179, 179, 179, 179, 179, 179, 180, 180, 180, 180, 180, 180,
      181, 181, 181, 181, 181, 182, 182, 182, 182, 182, 182, 182, 182, 182, 183,
      183, 183, 183, 183, 183, 183, 184, 184, 184, 184, 184, 184, 185, 185, 185,
      185, 185, 185, 185, 186, 186, 186, 186, 186, 186, 187, 187, 187, 187, 187,
      187, 187, 188, 188, 188, 188, 188, 188, 188, 188, 189, 189, 189, 189, 189,
      189, 189, 190, 190, 190, 190, 190, 190, 190, 190, 191, 191, 191, 191, 191,
      191, 191, 191, 192, 192, 192, 192, 192, 192, 192, 192, 193, 193, 193, 193,
      193, 193, 193, 193, 194, 194, 194, 194, 194, 194, 194, 194, 194, 195, 195,
      195, 195, 196, 196, 196, 196, 196, 196, 196, 196, 196, 197, 197, 197, 197,
      197, 197, 197, 197, 197, 197, 198, 198, 198, 198, 198, 198, 198, 198, 198,
      199, 199, 199, 199, 199, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
      200, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 202, 202, 202, 202,
      202, 202, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 204, 204,
      204, 204, 204, 204, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
      205, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 207, 207,
      207, 207, 207, 207, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208,
      208, 208, 209, 209, 209, 209, 209, 209, 209, 210, 210, 210, 210, 210, 210,
      210, 210, 210, 210, 210, 210, 210, 210, 211, 211, 211, 211, 211, 211, 211,
      212, 212, 212, 212, 212, 212, 212, 212, 213, 213, 213, 213, 213, 213, 213,
      213, 213, 213, 213, 213, 213, 213, 213, 214, 214, 214, 214, 214, 214, 214,
      214, 215, 215, 215, 215, 215, 215, 215, 215, 216, 216, 216, 216, 216, 216,
      216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 217, 217, 217, 217, 217,
      217, 217, 217, 217, 218, 218, 218, 218, 218, 218, 218, 218, 218, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 220, 220, 220, 220, 220, 220, 220, 220, 220, 221, 221, 221, 221, 221,
      221, 221, 221, 221, 221, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222,
      223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 224, 224, 224, 224, 224,
      224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224,
      224, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 226, 226, 226,
      226, 226, 226, 226, 226, 226, 226, 226, 227, 227, 227, 227, 227, 227, 227,
      227, 227, 227, 227, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228,
      228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 230, 230,
      230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 232, 232, 232, 232, 232, 232,
      232, 232, 232, 232, 232, 232, 232, 233, 233, 233, 233, 233, 233, 233, 233,
      233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233,
      233, 233, 233, 233, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234,
      234, 234, 234, 234, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235,
      235, 235, 235, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237,
      237, 237, 237, 237, 237};
#endif
};

constexpr uint8_t FindIdxLUT::kArr[];

//  clang-format off
// iQs[i] = (1<<16) / Qs[i], adjusted to maximize precision
static const uint16_t iQs[kQFactorMax + 1] = {
  5462, 5042, 4682, 4370, 4096, 3856, 3641, 3450, 3277, 3121, 2979, 2850, 2731,
  2622, 2521, 2428, 2341, 2260, 2185, 2115, 2048, 1986, 1928, 1873, 1821, 1772,
  1725, 1681, 1639, 1599, 1561, 1525, 1490, 1457, 1425, 1395, 1366, 1338, 1311,
  1286, 1261, 1237, 1214, 1192, 1171, 1150, 1130, 1111, 1093, 1075, 1058, 1041,
  1024, 1009,  993,  979,  964,  937,  924,  911,  898,  886,  874,  863,  852,
   841,  820,  810,  800,  790,  781,  763,  754,  745,  729,  721,  713,  698,
   690,  676,  669,  662,  649,  643,  631,  619,  613,  602,  596,  586,  575,
   570,  561,  551,  542,  538,  529,  521,  512,  505,  497,  490,  482,  475,
   469,  462,  456,  449,  440,  432,  429,  423,  415,  410,  405,  395,  393,
   386,  379,  375,  369,  365,  359,  353,  347,  344,  338,  333,  328,  323,
   319,  314,  310,  305,  300,  296,  292,  287,  283,  278,  275,  270,  267,
   263,  259,  255,  252,  248,  244,  241,  237,  233,  230,  226,  223,  220,
   217,  213,  211,  207,  204,  201,  198,  196,  193,  189,  187,  184,  182,
   179,  176,  173,  171,  169,  166,  164,  161,  159,  157,  154,  152,  150,
   147,  145,  143,  141,  139,  137,  135,  133,  131,  128,  127,  125,  123,
   122,  120,  118,  116,  115,  113,  111,  110,  108,  107,  105,  103,  102,
   100,   99,   97,   96,   95,   93,   92,   91,   89,   88,   87,   85,   84,
    83,   82,   81,   79,   78,   77,   76,   75,   74,   73,   72,   71,   69,
    68,   67,   66,   64,
};
// clang-format on

// Returns the closest quant-idx to "Qs[idx] / scale".
static uint32_t FindIdx(uint32_t idx, float scale) {
  assert(scale > 0.f);
  const uint16_t real_q = (uint16_t)std::round(Qs[idx] / scale);
  assert(real_q <= Qs[kQFactorMax]);
  static constexpr FindIdxLUT kFindIdxLUT;
  STATIC_ASSERT_ARRAY_SIZE(kFindIdxLUT.kArr, Qs[kQFactorMax] + 1);
  return kFindIdxLUT.kArr[real_q];
}

void QuantMtx::Init(uint32_t max_residual, float q_scale,
                    const uint16_t steps[kNumQuantZones]) {
  for (uint32_t i = 0; i < kNumQuantZones; ++i) {
    assert(steps[i] >= 0 && steps[i] <= kQFactorMax);
    quant_steps[i] = steps[i];
  }

  // build an interpolated ramp for the quant-steps
  uint32_t qramp[kMaxBlockSizePix];
  {
    uint32_t s = 0, last = 1u;
    for (uint32_t z = 0; z + 1 < kNumQuantZones; ++z) {
      const uint32_t e = kZones[z];
      for (uint16_t n = s; n < e; ++n) {
        last = qramp[n] = Interpolate(steps + z, s, e, n);
      }
      s = e;
    }
    // last segment is flat
    for (; s < kMaxBlockSizePix; ++s) qramp[s] = last;
  }

  // Compute quantization params for the big 32x32 unique matrix
  const bool for_encoding = !iquants.empty();
  uint32_t IQ0 = 0;
  for (uint32_t j = 0; j < 32; ++j) {
    const uint32_t qy = qramp[j];
    for (uint32_t i = j; i < 32; ++i) {
      const uint32_t qx = qramp[i];
      const uint32_t q_idx = FindIdx((qx + qy + 1) >> 1, q_scale);
      // Matrix is symmetric.
      const uint32_t idx1 = i + j * WP2QStride;
      const uint32_t idx2 = j + i * WP2QStride;
      if (for_encoding) {
        iquants[idx1] = iquants[idx2] = iQs[q_idx];
        bias[idx1] = bias[idx2] = (1u << WP2QBits) / 3;
      }
      dequants[idx1] = dequants[idx2] = (int16_t)Qs[q_idx];
      if (idx1 == 0) IQ0 = iQs[q_idx];  // save dc inv-quant
    }
  }
  for (const TrfSize tdim : kAllTrfSizes) {
    const uint32_t W = TrfWidth[tdim], H = TrfHeight[tdim];
    const uint32_t max_dc_res =
        WP2Quantize16b((uint32_t)(max_residual * std::sqrt(W * H) + .5), IQ0,
                                  (1u << WP2QBits) - 1u);
    // Use 'floor' rounding, since (1<<16)-1 must strictly not be overflowed.
    const uint32_t max_dc_16b = WP2Quantize16b((1 << 16) - 1, IQ0, /*bias=*/0);
    max_abs_res_dc[tdim] = std::min({kMaxDcValue, max_dc_res, max_dc_16b});
  }
}

void QuantMtx::SetLambda() {
  // this expression was fit over the sampling of several images encoded at
  // fixed quality (=quant_steps[1]) with varying lambda.
  lambda = 300. * std::pow(Qs[quant_steps[1]] / 50., 1.5);
  assert(lambda > 0.);
}

void QuantMtx::Quantize(const int32_t residuals[], TrfSize dim,
                        bool first_is_dc, int16_t coeffs[],
                        uint32_t* const num_coeffs,
                        int16_t dequantized_coeffs[]) const {
  WP2Quantize(iquants.data(), bias.data(),
              residuals, coeffs, TrfWidth[dim], TrfHeight[dim]);
  uint32_t last_nz = kNumCoeffs[dim];
  while (last_nz > 0 && coeffs[last_nz - 1] == 0) --last_nz;
  *num_coeffs = last_nz;

#if defined(WP2_ERROR_TRACE) || defined(WP2_TRACE)
  for (uint32_t i = 0; i < kNumCoeffs[dim]; ++i) {
    const uint32_t v = std::abs(coeffs[i]);
    const uint32_t max_v = (i == 0) ? max_abs_res_dc[dim] : kMaxCoeffValue;
    if (v > max_v) {
      // TODO(maryla): make sure clipping never occurs? Clipping is bad, it's
      // a form of unexpected quantization that happens at high quality (but
      // skal disagrees).
      fprintf(stderr,
              "WARNING!! Clipping occured. Clipped #%d value %d to %d\n",
              i, v, max_v);
    }
  }
#endif

  if (first_is_dc) {
    const uint32_t max_dc = max_abs_res_dc[dim];
    coeffs[0] = Clamp<int16_t>(coeffs[0], -max_dc, max_dc);
  }
  for (uint32_t i = (first_is_dc ? 1 : 0); i < last_nz; ++i) {
    coeffs[i] =
        Clamp<int16_t>(coeffs[i], -(int16_t)kMaxCoeffValue, kMaxCoeffValue);
  }
  // TODO(skal): merge the clamping to kMaxCoeffValue into WP2Dequantize()
  WP2Dequantize(coeffs, dequants, dequantized_coeffs,
                TrfWidth[dim], TrfHeight[dim], *num_coeffs);
}

int16_t QuantMtx::DCError(int32_t DC, TrfSize dim) const {
  const uint32_t iq = iquants[0];
  const uint32_t b = bias[0];
  int16_t v = WP2Quantize16b(std::abs(DC), iq, b);
  v = std::min(v, (int16_t)max_abs_res_dc[dim]) * dequants[0];
  if (DC < 0) v = -v;
  return DC - v;
}

void QuantMtx::Print() const {
  for (uint32_t j = 0; j < 32; ++j) {
    for (uint32_t i = 0; i < 32; ++i) {
      const uint32_t idx = i + WP2QStride * j;
      printf("%4u ", (uint32_t)((1ull << WP2QBits) / iquants[idx]));
    }
    printf("\n");
  }
  printf("quant_steps[]: ");
  for (const auto& q : quant_steps) printf("[%d]", q);
  printf("\n");
}

//------------------------------------------------------------------------------

}    // namespace WP2
