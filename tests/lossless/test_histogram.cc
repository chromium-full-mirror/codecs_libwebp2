// Copyright 2020 Google LLC
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
// Test lossless histograms.
//
// Author: Vincent Rabaud (vincent.rabaud@gmail.com)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/common/symbols.h"
#include "src/enc/lossless/histogram_enc.h"
#include "src/utils/random.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "tests/include/helpers.h"

namespace WP2L {
namespace {

TEST(LosslessHistogram, HistoQueue) {
  constexpr uint32_t kNumHistos = 10;
  constexpr uint32_t seed = 0;
  constexpr uint32_t max_count = 0;
  WP2::UniformIntDistribution generator(seed);
  LosslessSymbolsInfo symbols_info;
  symbols_info.Init(/*num_pixels=*/1000, /*has_alpha=*/true, WP2_Argb_32);
  symbols_info.SetCacheRange(1 << 10, /*has_segment_cache=*/false);

  PopulationAnalyzer analyzer(kNumHistos);
  ASSERT_WP2_OK(analyzer.Allocate(symbols_info));

  // Fill the histograms with random values.
  HistogramSet histogram_set;
  ASSERT_EQ(histogram_set.Allocate(kNumHistos, symbols_info), true);
  auto& histograms = histogram_set.histograms_;
  for (Histogram* const h : histograms) {
    h->Init();
    for (uint32_t i = 0; i < 100; ++i) {
      const uint32_t s = generator.Get(0u, (uint32_t)kSymbolNum - 1);
      if (symbols_info.IsUnused(s)) continue;
      h->counts_[s][generator.Get(0u, symbols_info.Range(s, 0))] =
          generator.Get(0u, max_count);
    }
    analyzer.UpdateHistogramCost(h);
  }

  // Create a queue en fill it.
  HistoQueue queue;
  ASSERT_WP2_OK(queue.Init(&histograms, &analyzer));
  ASSERT_WP2_OK(queue.Resize((kNumHistos - 1) * kNumHistos / 2));
  auto& valid_indices = queue.ValidIndices();

  while (valid_indices.size() > 1) {
    // Get the best cost diff;
    float best_cost_diff = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < kNumHistos; ++i) {
      for (uint32_t j = i + 1; j < kNumHistos; ++j) {
        const auto info = analyzer.GetCombinedHistogramEntropy(
            *histograms[i], *histograms[j], std::numeric_limits<float>::max());
        best_cost_diff =
            std::min(best_cost_diff, info.value() - (histograms[i]->bit_cost_ +
                                                     histograms[j]->bit_cost_));
      }
    }

    // Add pairs at best_cost_diff but add a little margin.
    const size_t size_before = valid_indices.size();
    for (uint32_t i = 0; i < size_before; ++i) {
      for (uint32_t j = i + 1; j < size_before; ++j) {
        ASSERT_WP2_OK(
            queue.Push(valid_indices[i], valid_indices[j], best_cost_diff + 1));
      }
    }

    EXPECT_FALSE(queue.empty());
    WP2::Vector_s16 segments_vec;
    EXPECT_TRUE(segments_vec.resize(4 * kNumHistos));
    std::fill(segments_vec.begin(), segments_vec.end(), 0);
    SegmentImgWrapper segments(segments_vec.data());
    queue.MergeFront(&segments);
    EXPECT_EQ(valid_indices.size(), size_before - 1);
  }
}

}  // namespace
}  // namespace WP2L
