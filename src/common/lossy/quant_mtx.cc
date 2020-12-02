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

WP2Status QuantMtx::Allocate() {
  WP2_CHECK_ALLOC_OK(iquant_data.resize(kDataSize));
  WP2_CHECK_ALLOC_OK(bias_data.resize(kDataSize));
  WP2_CHECK_ALLOC_OK(range_data.resize(kDataSize));
  return WP2_STATUS_OK;
}

// clang-format off
static const uint16_t Qs[kQFactorMax + 1] = {
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
  const uint32_t
      new_idx = std::lower_bound(&Qs[0], &Qs[kQFactorMax] + 1, real_q) - Qs;
  return new_idx;
}

void QuantMtx::Init(uint32_t max_residual, float q_scale,
                    PartitionSet partition_set,
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
  // prepare the quant matrices
  const bool for_decoding = iquant_data.empty();
  int16_t* dq_ptr = dequant_data;
  uint32_t* iq_ptr = iquant_data.data();
  uint32_t* b_ptr = bias_data.data();
  uint16_t* range_ptr = range_data.data();
  for (const TrfSize tdim : kAllTrfSizes) {
    // Check if tdim is used in the PartitionSet.
    const bool is_used = TrfIsUsed(partition_set, tdim);
    const uint32_t W = TrfWidth[tdim];
    const uint32_t H = TrfHeight[tdim];
    const uint32_t max_residual_value =
        (uint32_t)(max_residual * std::sqrt(W * H));
    for (uint32_t j = is_used ? 0 : H; j < H; ++j) {
      const uint32_t qy = qramp[j * 32 / H];
      for (uint32_t i = 0; i < W; ++i) {
        const bool is_dc = (i == 0 && j == 0);
        const uint32_t qx = qramp[i * 32 / W];
        const uint32_t q_idx = FindIdx((qx + qy + 1) >> 1, q_scale);
        const uint32_t q = Qs[q_idx];
        const uint32_t IQ = iQs[q_idx];
        const uint32_t idx = i + j * W;
        const uint32_t B = (1u << WP2QBits) / 3;
        if (!for_decoding) {
          iq_ptr[idx] = IQ;
          b_ptr[idx] = B;
        }
        if (!for_decoding || is_dc) {
          const uint32_t max_range = is_dc ? kMaxDcValue : kMaxCoeffValue;
          // for max_range calculation, one must use the actual quantization
          // formula, instead of the exact divide by q.
          // This quantization formulae is the same as the one used in dsp/,
          // only rounded up for safety.
          const uint16_t range = std::min(max_range,
              (max_residual_value * IQ + (1 << WP2QBits) - 1) >> WP2QBits);
          if (!for_decoding) range_ptr[idx] = range;
          if (is_dc) max_abs_res_dc[tdim] = range;
        }
        dq_ptr[idx] = (int16_t)q;
      }
    }
    // Because of the assumption of WP2Quantize(), we need to round up the size
    // to next multiple of 8.
    const uint32_t size8 = (W * H + 7) & ~7;
    dequants[tdim] = dq_ptr;
    iquants[tdim] = iq_ptr;
    bias[tdim] = b_ptr;
    max_abs_res[tdim] = range_ptr;
    dq_ptr += size8;
    if (!for_decoding) {
      iq_ptr += size8;
      b_ptr += size8;
      range_ptr += size8;
    }
  }
  assert(dq_ptr == &dequant_data[kDataSize]);
}

void QuantMtx::SetLambda(float lambda) {
  // this expression was fit over the sampling of several images encoded at
  // fixed quality (=quant_steps[1]) with varying lambda.
  const double lambda_f = 450. * pow(Qs[quant_steps[1]] / 50., 1.2);
  assert(lambda_f > 0.);
  lambda1 = lambda_f;
  lambda2 = 0.84 * lambda_f;
  lambda3 = lambda_f;
}

void QuantMtx::Quantize(const int32_t residuals[], TrfSize dim,
                        bool first_is_dc, int16_t coeffs[],
                        uint32_t* const num_coeffs) const {
  const uint32_t num_coeffs_full = kNumCoeffs[dim];
  WP2Quantize(iquants[dim], bias[dim], residuals, coeffs, num_coeffs_full);
  uint32_t last_nz = num_coeffs_full;
  while (last_nz > 0 && coeffs[last_nz - 1] == 0) --last_nz;
  *num_coeffs = last_nz;

#if defined(WP2_ERROR_TRACE) || defined(WP2_TRACE)
  for (uint32_t i = 0; i < num_coeffs_full; ++i) {
    const uint32_t v = std::abs(coeffs[i]);
    assert(v <= max_abs_res[dim][i]);
    if (i == 0 && v > kMaxDcValue) {
      // TODO(maryla): make sure clipping never occurs? Clipping is bad, it's
      // a form of unexpected quantization that happens at high quality (but
      // skal disagrees).
      fprintf(stderr, "WARNING!! Clipping occured. Clipped DC %d to %d\n", v,
              kMaxDcValue);
    } else if (i > 0 && v > kMaxCoeffValue) {
      fprintf(stderr, "WARNING!! Clipping occured. Clipped value %d to %d\n", v,
              kMaxCoeffValue);
    }
  }
#endif
  if (first_is_dc) {
    coeffs[0] = Clamp<int16_t>(coeffs[0], -kMaxDcValue, kMaxDcValue);
  }
  for (uint32_t i = (first_is_dc ? 1 : 0); i < last_nz; ++i) {
    coeffs[i] = Clamp<int16_t>(coeffs[i], -kMaxCoeffValue, kMaxCoeffValue);
  }
}

int16_t QuantMtx::DCError(int32_t DC, TrfSize dim) const {
  const uint32_t iq = iquants[dim][0];
  const uint32_t b = bias[dim][0];
  int32_t v = ((uint64_t)std::abs(DC) * iq + b) >> WP2QBits;
  v = std::min(v, kMaxDcValue) * dequants[dim][0] * (DC < 0 ? -1 : 1);
  return DC - v;
}

void QuantMtx::Dequantize(const int16_t coeffs[], uint32_t num_coeffs,
                          TrfSize dim, int32_t residuals[]) const {
  WP2Dequantize(coeffs, dequants[dim], residuals, num_coeffs, kNumCoeffs[dim]);
}

uint32_t QuantMtx::RoundTrip(const int32_t residuals[], TrfSize dim,
                             bool first_is_dc, int16_t coeffs[],
                             uint32_t* const num_coeffs) const {
  int32_t tmp[kMaxBlockSizePix2];
  Quantize(residuals, dim, first_is_dc, coeffs, num_coeffs);
  // TODO(skal): rnd_mtx ?
  WP2Dequantize(coeffs, dequants[dim], tmp, *num_coeffs, kNumCoeffs[dim]);

  uint32_t error = 0;
  for (uint32_t i = 0; i < kNumCoeffs[dim]; ++i) {
    const int32_t diff = (int32_t)residuals[i] - tmp[i];
    error += diff * diff;
  }
  return error / kNumCoeffs[dim];
}

void QuantMtx::Print() const {
  for (const TrfSize tdim : kAllTrfSizes) {
    printf(" --- Q-%s ---\n", kTDimNames[tdim]);
    const uint32_t W = TrfWidth[tdim];
    const uint32_t H = TrfHeight[tdim];
    for (uint32_t j = 0; j < H; ++j) {
      for (uint32_t i = 0; i < W; ++i) {
        const uint32_t idx = i + W * j;
        printf("%4u ", (uint32_t)((1ull << WP2QBits) / iquants[tdim][idx]));
      }
      printf("\n");
    }
  }
  printf("quant_steps[]: ");
  for (const auto& q : quant_steps) printf("[%d]", q);
  printf("\n");
}

//------------------------------------------------------------------------------

void QuantStat::Reset() {
  for (const TrfSize tdim : kAllTrfSizes) {
    const uint32_t num_coeffs = kNumCoeffs[tdim];
    last_idx_[tdim] = 0;
    for (uint32_t i = 0; i < num_coeffs; ++i) {
      is_large_[tdim][i] = false;
    }
  }
  for (uint32_t i = 0; i < kNumQuantZones; ++i) {
    count_[i] = 0;
    value_[i] = 0;
  }
}

static uint16_t IdxToZone(TrfSize tdim, uint32_t idx) {
  const uint32_t W = TrfWidth[tdim];
  const uint32_t x = idx % W;
  const uint32_t y = idx / W;
  uint32_t zi = 0, zj = 0;
  while (x > kZones[zi]) ++zi;
  while (y > kZones[zj]) ++zj;
  return std::min(zi, zj);
}

// These global constants are inverse of kDynNorm[] * q_scale.
// TODO(maryla): this makes no sense since q_scale is not a constant.
static constexpr float kZoneNorm[TRF_LAST] = {
  0,      0,      0,
  0, 2.1609, 3.1458, 4.5276,
  0, 3.1458, 4.5796,  6.5912, 9.4160,
     4.5276, 6.5912, 9.4864, 13.5520,
            9.4160, 13.5520, 19.36000
};
STATIC_ASSERT_ARRAY_SIZE(kZoneNorm, TRF_LAST);

void QuantStat::Record(TrfSize tdim,
                       uint32_t idx, int32_t qvalue, int32_t value) {
  assert(idx < kNumCoeffs[tdim]);
  qvalue = std::abs(qvalue);
  const bool large = (qvalue >= 2);
  is_large_[tdim][idx] |= large;
  if (qvalue > 0) {
    last_idx_[tdim] = std::max(last_idx_[tdim], idx + 1);
  }
  if (qvalue == 1) {
    const uint32_t z = IdxToZone(tdim, idx);
    ++count_[z];
    value_[z] += std::abs(value) / kZoneNorm[tdim] / 2.;
  } else {
    // TODO(skal): value /= qvalue ?
  }
}

void QuantStat::Record(TrfSize tdim,
                       int32_t* const coeffs, int16_t* const qcoeffs) {
  const uint32_t num_coeffs = kNumCoeffs[tdim];
  for (uint32_t i = 0; i < num_coeffs; ++i) {
    if (qcoeffs[i]) Record(tdim, i, qcoeffs[i], coeffs[i]);
  }
}

void QuantStat::Print() const {
  for (uint32_t z = 0; z < kNumQuantZones; ++z) {
    if (count_[z] > 0) {
      const uint32_t q = (uint32_t)(value_[z] / count_[z] + .5);
      printf("%3d ", q);
    } else {
      printf("  .  ");
    }
  }
  printf("\n");
}

bool QuantStat::AdjustQuantSteps(uint16_t quant_steps[kNumQuantZones]) const {
  bool changed = false;
  for (uint32_t z = 0; z < kNumQuantZones; ++z) {
    if (count_[z] > 0) {
      const uint16_t q = (uint32_t)(value_[z] / count_[z] + .5);
      changed |= (q != quant_steps[z]);
      quant_steps[z] = q;
    }
  }
  return changed;
}

//------------------------------------------------------------------------------

}    // namespace WP2
