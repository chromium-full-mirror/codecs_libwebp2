// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------
//
// Measures performance of trellis related functions.

#include "src/common/lossy/quant_mtx.h"
#include "src/common/lossy/residuals.h"
#include "src/common/lossy/transforms.h"
#include "src/enc/block_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/enc/trellis.h"
#include "tests/bench/helpers_bm.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

static const uint32_t kMaxResidual = 512;

void FillRandomResiduals(TrfSize tdim, uint32_t seed,
                         int32_t residuals[kMaxBlockSizePix2]) {
  std::fill(residuals, residuals + kMaxBlockSizePix2, 0);
  const uint16_t* const scan = ResidualIterator::GetZigzag(tdim);

  UniformIntDistribution rand(seed);
  const uint32_t max_coeffs = kNumCoeffs[tdim];
  // Random coeffs.
  for (uint32_t i = 0; i < max_coeffs; ++i) {
    // Random eob.
    if (i > (max_coeffs * 0.75) && rand.FlipACoin()) break;
    // Make coeffs smaller and smaller.
    const int max =
        std::max<int>(kMaxResidual * (1.f - ((float)i / max_coeffs)), 3);
    residuals[scan[i]] = rand.Get(-max, max);
  }
}

void BM_Av1CoeffOptimization(benchmark::State& state) {
  WP2QuantizeInit();
  ANSInit();
  WP2MathInit();

  const int seed = state.range(0);

  const Channel channel = kYChannel;
  const TransformPair tx_type = kDctDct;
  const TrfSize tdim = TRF_16x16;
  QuantMtx quant;
  quant.lambda = 1000;
  const uint16_t quant_steps[] = {20, 50, 50, 50, 50};
  ASSERT_WP2_OK(quant.AllocateForEncoder());
  quant.Init(kMaxResidual, /*q_scale=*/1.f, quant_steps);
  const bool first_is_dc = true;
  int32_t residuals[kMaxBlockSizePix2];

  FillRandomResiduals(tdim, seed, residuals);

  int16_t coeffs[kMaxBlockSizePix2];
  int16_t dequantized_coeffs[kMaxBlockSizePix2];
  uint32_t num_coeffs;
  ASSERT_WP2_OK(quant.AllocateForEncoder());
  quant.Quantize(residuals, tdim, first_is_dc, coeffs, &num_coeffs,
                 dequantized_coeffs);
  int16_t coeffs_copy[kMaxBlockSizePix2];
  std::copy(coeffs, coeffs + kMaxBlockSizePix2, coeffs_copy);

  SymbolsInfo symbols_info;
  ASSERT_WP2_OK(
      symbols_info.InitLossy(/*num_segments=*/2, PartitionSet::SOME_RECTS,
                             /*has_alpha=*/false, /*use_aom_coeffs=*/true,
                             /*use_splits=*/false));
  SymbolRecorder recorder;
  ASSERT_WP2_OK(recorder.Allocate(symbols_info, /*num_records=*/0));
  Counters counters;
  ASSERT_WP2_OK(counters.Init(/*effort=*/5, recorder));

  for (const auto& _ : state) {
    (void)_;  // The content of this loop is benchmarked.
    float rate_cost;
    Av1CoeffOptimization(channel, tx_type, tdim, quant, first_is_dc, residuals,
                         coeffs, &num_coeffs, dequantized_coeffs,
                         counters.residuals_aom(), &rate_cost);
  }
}
BENCHMARK(BM_Av1CoeffOptimization)->DenseRange(0, 10);

}  // namespace
}  // namespace WP2
