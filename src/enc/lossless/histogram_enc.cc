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
// Author: Jyrki Alakuijala (jyrki@google.com)
//

#include "src/enc/lossless/histogram_enc.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

#ifdef HAVE_CONFIG_H
#include "src/webp/config.h"
#endif

#include "src/common/color_precision.h"
#include "src/common/progress_watcher.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/random.h"
#include "src/utils/vector.h"

namespace WP2L {

// Maximum number of histograms allowed in greedy combining algorithm.
#define MAX_HISTO_GREEDY 100

//------------------------------------------------------------------------------

static inline void CopySubHistogram(const Histogram& a, const uint32_t s,
                                    Histogram* const out) {
  const uint32_t size =
      sizeof(out->counts_[s][0]) * a.symbols_info_->GetMaxRange(s);
  memcpy(out->counts_[s], a.counts_[s], size);
  out->trivial_symbols_[s] = a.trivial_symbols_[s];
  out->nonzeros_[s] = a.nonzeros_[s];
}

// Add two histograms, taking into account the number of non-zero elements for
// speed-up.
static void HistogramAdd(const Histogram& a, const Histogram& b,
                         Histogram* const out) {
  assert(a.symbols_info_ == b.symbols_info_);
  if (&b != out) {
    for (uint32_t s = 0; s < kSymbolNum; ++s) {
      if (a.nonzeros_[s] == 0) {
        CopySubHistogram(b, s, out);
      } else if (b.nonzeros_[s] == 0) {
        CopySubHistogram(a, s, out);
      } else if (a.nonzeros_[s] == 1 && b.nonzeros_[s] == 1 &&
                 a.trivial_symbols_[s] == b.trivial_symbols_[s]) {
        const auto& trivial_symbol = a.trivial_symbols_[s];
        const uint32_t size =
            sizeof(out->counts_[s][0]) * a.symbols_info_->GetMaxRange(s);
        memset(out->counts_[s], 0, size);
        out->counts_[s][trivial_symbol] =
            a.counts_[s][trivial_symbol] + b.counts_[s][trivial_symbol];
        out->trivial_symbols_[s] = trivial_symbol;
        out->nonzeros_[s] = 1;
      } else {
        out->trivial_symbols_[s] = 0;
        BufferAdd(a.counts_[s], b.counts_[s], a.symbols_info_->GetMaxRange(s),
                  out->counts_[s], &out->nonzeros_[s]);
        assert(out->nonzeros_[s] > 1);
      }
    }
  } else {
    for (uint32_t s = 0; s < kSymbolNum; ++s) {
      if (a.nonzeros_[s] == 0) {
        // out is b and no modification is added.
      } else if (b.nonzeros_[s] == 0) {
        CopySubHistogram(a, s, out);
      } else if (a.nonzeros_[s] == 1 && b.nonzeros_[s] == 1 &&
                 a.trivial_symbols_[s] == b.trivial_symbols_[s]) {
        // a and b are basically the same.
        out->counts_[s][a.trivial_symbols_[s]] +=
            a.counts_[s][a.trivial_symbols_[s]];
      } else {
        out->trivial_symbols_[s] = 0;
        BufferAdd(a.counts_[s], b.counts_[s], a.symbols_info_->GetMaxRange(s),
                  out->counts_[s], &out->nonzeros_[s]);
        assert(out->nonzeros_[s] > 1);
      }
    }
  }
}

bool HistogramSet::Allocate(uint32_t size,
                            const WP2::SymbolsInfo& symbols_info) {
  symbols_info_ = &symbols_info;
  const uint32_t cache_num = symbols_info.MaxRangeSum();
  if (!histograms_.resize(size) || !histo_buffer_.resize(size + 1) ||
      !buffer_.resize((size + 1) * cache_num)) {
    return false;
  }
  std::fill(buffer_.begin(), buffer_.end(), 0);
  uint32_t* memory = buffer_.data();
  for (uint32_t i = 0; i < size; ++i) {
    histograms_[i] = &histo_buffer_[i];
    histo_buffer_[i].Init();
    histo_buffer_[i].SetBuffer(memory, symbols_info);
    histo_buffer_[i].id_ = i;
    memory += cache_num;
  }
  tmp_histo_ = &histo_buffer_[size];
  tmp_histo_->Init();
  tmp_histo_->SetBuffer(memory, symbols_info);
  return true;
}

// -----------------------------------------------------------------------------
// Entropy-related functions.

PopulationAnalyzer::PopulationAnalyzer(uint32_t image_size)
    : image_size_(image_size) {
  EncLDspInit();
}

void PopulationAnalyzer::PopulationCost(const uint32_t* const population,
                                        uint32_t range,
                                        uint32_t* const nonzeros,
                                        uint32_t* const trivial_sym,
                                        float* cost) {
  WP2::Quantizer::Config* config;

  uint32_t size = 0;
  for (uint32_t k = 0; k < range; ++k) {
    if (population[k] == 0) continue;
    histogram_[size] = population[k];
    mapping_[size] = k;
    ++size;
  }
  if (nonzeros != nullptr) *nonzeros = size;
  // TODO(vrabaud) replace the cost logic below with a proper call to
  //               SymbolWriter::WriteHeader.
  if (size == 0) {
    if (cost != nullptr) *cost = WP2Log2(3);
    return;
  }
  if (size == 1) {
    if (trivial_sym != nullptr) *trivial_sym = mapping_[0];
    // TODO(vrabaud) do in the Quantizer.
    if (cost != nullptr) *cost = WP2Log2(3) + WP2Log2(range);
    return;
  }

  // TODO(vrabaud) use the 'effort' parameter instead of imposing 0.
  quantizer_.Quantize(histogram_.data(), mapping_.data(), size, range,
                      image_size_, /*effort=*/0, &config);
  if (cost != nullptr) {
    *cost =
        WP2Log2(3) + WP2Log2(std::min(image_size_, range) - 1) + config->cost;
  }
}

void PopulationAnalyzer::GetCombinedEntropy(
    const uint32_t* const X, const uint32_t* const Y, uint32_t X_nonzeros,
    uint32_t Y_nonzeros, uint32_t X_trivial, uint32_t Y_trivial,
    uint32_t length, float* cost) {
  // If one of the histograms is basically unused:
  if (X_nonzeros == 0) {
    PopulationCost(Y, length, nullptr, nullptr, cost);
    return;
  }
  if (Y_nonzeros == 0) {
    PopulationCost(X, length, nullptr, nullptr, cost);
    return;
  }
  // If they both contain the same element.
  if (X_nonzeros == 1 && Y_nonzeros == 1 && X_trivial == Y_trivial) {
    *cost = WP2Log2(3) + WP2Log2(length);
    return;
  }

  BufferAdd(X, Y, length, sum_tmp_.data(), nullptr);
  PopulationCost(sum_tmp_.data(), length, nullptr, nullptr, cost);
}

// -----------------------------------------------------------------------------
// Various histogram combine/cost-eval functions

int PopulationAnalyzer::GetCombinedHistogramEntropy(const Histogram& a,
                                                    const Histogram& b,
                                                    float cost_threshold,
                                                    float* cost) {
  assert(a.symbols_info_ == b.symbols_info_);

  // Deal with the raw bits of length and distance.
  for (const auto& sym : {kSymbolLen, kSymbolDist}) {
    *cost += ExtraCostCombined(a.counts_[sym], b.counts_[sym],
                               a.symbols_info_->GetMaxRange(sym));
    if (*cost > cost_threshold) return 0;
  }

  bool is_used[kSymbolTypeNum];
  for (uint32_t i = 0; i < kSymbolTypeNum; ++i) {
    is_used[i] = (a.counts_[kSymbolType][i] + b.counts_[kSymbolType][i] > 0);
  }

  // Deal with all the symbols.
  for (uint32_t s = 0; s < kSymbolNum; ++s) {
    if (a.symbols_info_->IsUnused(s) || b.symbols_info_->IsUnused(s)) {
      continue;
    }
    if (!WP2L::IsSymbolUsed((Symbol)s, is_used)) continue;
    float tmp;
    GetCombinedEntropy(a.counts_[s], b.counts_[s], a.nonzeros_[s],
                       b.nonzeros_[s], a.trivial_symbols_[s],
                       b.trivial_symbols_[s], a.symbols_info_->GetMaxRange(s),
                       &tmp);
    *cost += tmp;
    if (*cost > cost_threshold) return 0;
  }
  return 1;
}

// Performs out = a + b, computing the cost C(a+b) - C(a) - C(b) while comparing
// to the threshold value 'cost_threshold'. The score returned is
//  Score = C(a+b) - C(a) - C(b), where C(a) + C(b) is known and fixed.
// Since the previous score passed is 'cost_threshold', we only need to compare
// the partial cost against 'cost_threshold + C(a) + C(b)' to possibly bail-out
// early.
float PopulationAnalyzer::HistogramAddEval(const Histogram& a,
                                           const Histogram& b,
                                           Histogram* const out,
                                           float cost_threshold) {
  float cost = 0.f;
  const float sum_cost = a.bit_cost_ + b.bit_cost_;
  cost_threshold += sum_cost;

  if (GetCombinedHistogramEntropy(a, b, cost_threshold, &cost)) {
    HistogramAdd(a, b, out);
    out->bit_cost_ = cost;
    assert(a.symbols_info_ == b.symbols_info_);
  }

  return cost - sum_cost;
}

// Same as HistogramAddEval(), except that the resulting histogram
// is not stored. Only the cost C(a+b) - C(a) is evaluated. We omit
// the term C(b) which is constant over all the evaluations.
float PopulationAnalyzer::HistogramAddThresh(const Histogram& a,
                                             const Histogram& b,
                                             float cost_threshold) {
  float cost = -a.bit_cost_;
  GetCombinedHistogramEntropy(a, b, cost_threshold, &cost);
  return cost;
}

// -----------------------------------------------------------------------------

void PopulationAnalyzer::UpdateHistogramCost(Histogram* const h) {
  h->bit_cost_ = 0.;
  bool is_used[kSymbolTypeNum];
  bool has_at_least_one_symbol = false;
  for (uint32_t i = 0; i < kSymbolTypeNum; ++i) {
    is_used[i] = (h->counts_[kSymbolType][i] > 0);
    has_at_least_one_symbol |= is_used[i];
  }
  if (!has_at_least_one_symbol) {
    // If the histogram is completely empty, it contains no information so no
    // cost.
    std::fill(h->costs_, h->costs_ + kSymbolNum, 0.);
    return;
  }
  for (uint32_t s = 0; s < kSymbolNum; ++s) {
    const Symbol sym = (Symbol)s;
    if (h->symbols_info_->IsUnused(s) || !IsSymbolUsed(sym, is_used)) {
      h->costs_[s] = 0.;
    } else {
      PopulationCost(h->counts_[s], h->symbols_info_->GetMaxRange(sym),
                     &h->nonzeros_[s], &h->trivial_symbols_[s],
                     &(h->costs_[s]));
    }
  }
  for (const auto& sym : {kSymbolLen, kSymbolDist}) {
    h->costs_[sym] +=
        ExtraCost(h->counts_[sym], h->symbols_info_->GetMaxRange(sym));
  }
  // Update the total bit_cost.
  for (const auto& c : h->costs_) h->bit_cost_ += c;
}

WP2Status BuildHistograms(uint32_t width, uint32_t height, uint32_t histo_bits,
                          const LosslessSymbolsInfo& symbols_info,
                          const BackwardRefs& refs,
                          HistogramSet* const histogram_set) {
  // Count the symbols using a SymbolRecorder.
  WP2::SymbolRecorder recorder;
  WP2::ANSEncNoop enc_noop;
  LosslessSymbolsInfo symbols_info_counter;
  WP2_CHECK_STATUS(symbols_info_counter.CopyFrom(symbols_info));
  const uint32_t num_histos = histogram_set->histograms_.size();
  symbols_info_counter.SetNumClusters(num_histos);
  WP2_CHECK_STATUS(recorder.Allocate(symbols_info_counter, /*num_records=*/0));
  WP2_CHECK_STATUS(StorePixels(width, height, histo_bits, refs,
                               WP2::ProgressRange(nullptr, 0.),
                               /*segments=*/nullptr, &enc_noop, &recorder));

  // Copy the dictionaries to the HistogramSet.
  Histogram** const histograms = histogram_set->histograms_.data();
  for (uint32_t i = 0; i < num_histos; ++i) {
    for (uint32_t s = 0; s < kSymbolNum; ++s) {
      if (symbols_info_counter.Range(s, i) == 0) continue;
      const WP2::Vector_u32& counts = recorder.GetRecordedDict(s, i).Counts();
      std::copy(counts.begin(), counts.end(), histograms[i]->counts_[s]);
    }
  }
  return WP2_STATUS_OK;
}

// Copies the histograms and computes its bit_cost.
static WP2Status HistogramSetCopy(const HistogramSet& in,
                                  const WP2::SymbolsInfo& symbols_info,
                                  HistogramSet* const out) {
  uint32_t size = 0;
  for (Histogram* const histo : in.histograms_) {
    // Skip this histogram if it is not even used once.
    if (histo->bit_cost_ != 0) ++size;
  }
  WP2_CHECK_ALLOC_OK(out->Allocate(size, symbols_info));

  size = 0;
  for (Histogram* const histo : in.histograms_) {
    // Skip this histogram if it is not even used once.
    if (histo->bit_cost_ == 0) continue;
    // Copy histograms from 'in' to 'out'.
    histo->CopyTo(out->histograms_[size++]);
  }
  return WP2_STATUS_OK;
}

// Class that merges histograms together if they have common entropies for some
// of their symbols.
class BinMerger {
 public:
  BinMerger(uint32_t histo_size, uint32_t effort) : effort_(effort) {
    combine_cost_factor_ = 0.16;
    if (effort < 8) {
      if (histo_size > 256) combine_cost_factor_ /= 2.;
      if (histo_size > 512) combine_cost_factor_ /= 2.;
      if (histo_size > 1024) combine_cost_factor_ /= 2.;
      if (effort < 5) combine_cost_factor_ /= 2.;
    }
  }
  // Merges empty histograms together.
  static WP2Status RemoveEmptyHistograms(HistogramSet* const image_histo,
                                         SegmentImgWrapper* const segments) {
    uint32_t new_size = 0;
    auto& histograms = image_histo->histograms_;
    for (uint32_t i = 0; i < histograms.size(); ++i) {
      if (histograms[i]->bit_cost_ == 0.) {
        (*segments)[histograms[i]->id_] = -1;
      } else {
        std::swap(histograms[i], histograms[new_size]);
        ++new_size;
      }
    }
    WP2_CHECK_ALLOC_OK(histograms.resize(new_size));
    return WP2_STATUS_OK;
  }
  // Partition histograms to different entropy bins for three dominant (literal,
  // red and blue) symbol costs and compute the histogram aggregate bit_cost.
  bool AnalyzeEntropyBin(const HistogramSet& image_histo,
                         uint16_t* const bin_map) {
    // Don't attempt linear bin-partition heuristic for
    // histograms of small sizes (as bin_map will be very sparse) and
    // maximum effort 9 (to preserve the compression gains at that level).
    const uint32_t entropy_combine_num_bins =
        effort_ == 0 ? kNumPartitions : kBinSizeTot;
    const bool entropy_combine =
        (image_histo.histograms_.size() > entropy_combine_num_bins * 2) &&
        (effort_ < 9);
    if (!entropy_combine) return false;

    std::fill(max_costs_, max_costs_ + kSymbolNum, 0);
    std::fill(min_costs_, min_costs_ + kSymbolNum,
              std::numeric_limits<float>::max());

    // Analyze the entropy cost bounds.
    for (const Histogram* h : image_histo.histograms_) {
      for (uint32_t s = 0; s < h->symbols_info_->Size(); ++s) {
        if (h->costs_[s] < min_costs_[s]) {
          min_costs_[s] = h->costs_[s];
        } else if (h->costs_[s] > max_costs_[s]) {
          max_costs_[s] = h->costs_[s];
        }
      }
    }

    // Find the widest cost disparities. Variance and coefficient of variation
    // are actually worse. PCA should be tried.
    for (uint32_t i = 0; i < image_histo.symbols_info_->Size(); ++i) {
      disparities_[i].first = max_costs_[i] - min_costs_[i];
      disparities_[i].second = i;
    }
    std::sort(disparities_, disparities_ + kSymbolNum,
              std::greater<std::pair<float, uint32_t>>());

    // bin-hash histograms on three of the dominant (literal, red and blue)
    // symbol costs and store the resulting bin_id for each histogram.
    uint32_t i = 0;
    for (const Histogram* h : image_histo.histograms_) {
      bin_map[i++] = GetHistoBinIndex(*h);
    }
    return true;
  }

  // Compact image_histo[] by merging some histograms with same bin_id together
  // if it's advantageous.
  WP2Status CombineEntropyBin(const WP2::Vector_u16& bin_map,
                              PopulationAnalyzer* const analyzer,
                              HistogramSet* const image_histo,
                              SegmentImgWrapper* const segments) const {
    Histogram** const histograms = image_histo->histograms_.data();
    // Work in-place: processed histograms are put at the beginning of
    // image_histo[]. At the end, we just have to truncate the array.
    struct {
      int16_t first;  // position of the histogram that accumulates all
                      // histograms with the same bin_id
      uint16_t num_combine_failures;  // number of combine failures per bin_id
    } bin_info[kBinSizeTot];

    const uint32_t num_bins = (effort_ == 0) ? kNumPartitions : kBinSizeTot;
    assert(num_bins <= kBinSizeTot);
    for (uint32_t idx = 0; idx < num_bins; ++idx) {
      bin_info[idx].first = -1;
      bin_info[idx].num_combine_failures = 0;
    }

    uint32_t size = 0;
    for (uint32_t idx = 0; idx < image_histo->histograms_.size(); ++idx) {
      const int bin_id = bin_map[idx];
      const int first = bin_info[bin_id].first;
      assert(size <= idx);
      if (first == -1) {
        // just move histogram #idx to its final position
        histograms[size] = histograms[idx];
        bin_info[bin_id].first = size++;
      } else if (effort_ == 0) {
        HistogramAdd(*histograms[idx], *histograms[first], histograms[first]);
        (*segments)[histograms[first]->id_] =
            (*segments)[histograms[idx]->id_] =
                std::min(histograms[first]->id_, histograms[idx]->id_);
      } else {
        // try to merge #idx into #first (both share the same bin_id)
        const float bit_cost = histograms[idx]->bit_cost_;
        const float bit_cost_thresh = -bit_cost * combine_cost_factor_;
        const float curr_cost_diff = analyzer->HistogramAddEval(
            *histograms[first], *histograms[idx], image_histo->tmp_histo_,
            bit_cost_thresh);
        if (curr_cost_diff < bit_cost_thresh) {
          (*segments)[histograms[first]->id_] =
              (*segments)[histograms[idx]->id_] = image_histo->tmp_histo_->id_ =
                  std::min(histograms[first]->id_, histograms[idx]->id_);
          std::swap(image_histo->tmp_histo_, histograms[first]);
        } else {
          histograms[size++] = histograms[idx];
        }
      }
    }
    WP2_CHECK_ALLOC_OK(image_histo->histograms_.resize(size));
    if (effort_ == 0) {
      // for low_effort case, update the final cost when everything is merged
      for (uint32_t idx = 0; idx < size; ++idx) {
        analyzer->UpdateHistogramCost(histograms[idx]);
      }
    }
    return WP2_STATUS_OK;
  }

 private:
  uint32_t GetHistoBinIndex(const Histogram& h) const {
    uint32_t bin_id = 0;
    // Carve the space depending on the three widest disparities.
    for (uint32_t i = 0; i < (effort_ > 0 ? 3u : 1u); ++i) {
      const uint32_t s = disparities_[i].second;
      const float range = disparities_[i].first;
      if (range == 0.) break;
      // Fix potential rounding issues.
      const uint32_t new_bin_id = std::min(
          (uint32_t)(kNumPartitions * (h.costs_[s] - min_costs_[s]) / range),
          kNumPartitions - 1);
      bin_id = bin_id * kNumPartitions + new_bin_id;
      assert(bin_id < kBinSizeTot);
    }
    return bin_id;
  }

  // Number of partitions for the dominant symbols.
  static constexpr uint32_t kNumPartitions = 4;
  // The size of the bin-hash corresponding to the three dominant costs.
  static constexpr uint32_t kBinSizeTot =
      (kNumPartitions * kNumPartitions * kNumPartitions);

  float min_costs_[kSymbolNum];
  float max_costs_[kSymbolNum];
  std::pair<float, uint32_t> disparities_[kSymbolNum];

  float combine_cost_factor_;
  const uint32_t effort_;
};

// -----------------------------------------------------------------------------
// Histogram pairs priority queue

WP2Status HistoQueue::Init(WP2::VectorNoCtor<Histogram*>* const histograms,
                           PopulationAnalyzer* const analyzer) {
  histograms_ = histograms;
  analyzer_ = analyzer;
  return WP2_STATUS_OK;
}

WP2Status HistoQueue::Resize(uint32_t size) {
  WP2_CHECK_ALLOC_OK(queue_.reserve(size));
  WP2_CHECK_ALLOC_OK(indices_.resize(histograms_->size()));
  std::iota(indices_.begin(), indices_.end(), 0u);
  return WP2_STATUS_OK;
}

void HistoQueue::MergeFront(SegmentImgWrapper* const segments) {
  const uint16_t idx1 = queue_.front().idx1;
  const uint16_t idx2 = queue_.front().idx2;
  assert(idx1 < idx2);
  // Merge idx2 into idx1.
  Histogram* const histo1 = (*histograms_)[idx1];
  const Histogram& histo2 = *(*histograms_)[idx2];
  HistogramAdd(histo2, *histo1, histo1);
  histo1->bit_cost_ = queue_.front().cost_combo;
  const int16_t min_id = std::min(histo1->id_, histo2.id_);
  (*segments)[histo1->id_] = (*segments)[histo2.id_] = min_id;
  histo1->id_ = min_id;

  for (uint32_t i = 0; i < queue_.size();) {
    HistogramPair* const p = &queue_[i];
    if (p->idx1 == idx1 || p->idx2 == idx1 || p->idx1 == idx2 ||
        p->idx2 == idx2) {
      // Pop the pair from the list.
      *p = queue_.back();
      queue_.pop_back();
      continue;
    }
    MaybeUpdateHead(p);
    ++i;
  }
  // Remove idx2 from valid indices.
  const auto iter = std::lower_bound(indices_.begin(), indices_.end(), idx2);
  assert(iter != indices_.end() && *iter == idx2);
  std::copy(iter + 1, indices_.end(), iter);
  indices_.pop_back();
}

WP2Status HistoQueue::Push(uint16_t idx1, uint16_t idx2, float threshold,
                           float* const cost_diff) {
  if (idx1 > idx2) std::swap(idx1, idx2);
  HistogramPair pair;
  pair.idx1 = idx1;
  pair.idx2 = idx2;
  const Histogram* const h1 = (*histograms_)[idx1];
  const Histogram* const h2 = (*histograms_)[idx2];
  const float sum_cost = h1->bit_cost_ + h2->bit_cost_;
  pair.cost_combo = 0.;
  analyzer_->GetCombinedHistogramEntropy(*h1, *h2, sum_cost + threshold,
                                         &pair.cost_combo);
  pair.cost_diff = pair.cost_combo - sum_cost;

  // Do not even consider the pair if it does not improve the entropy.
  if (cost_diff != nullptr) *cost_diff = 0;
  if (pair.cost_diff >= threshold) return WP2_STATUS_OK;

  // We cannot add more elements than the capacity.
  assert(queue_.size() < queue_.capacity());
  WP2_CHECK_ALLOC_OK(queue_.push_back(pair, /*resize_if_needed=*/false));
  MaybeUpdateHead(&queue_.back());

  if (cost_diff != nullptr) *cost_diff = pair.cost_diff;
  return WP2_STATUS_OK;
}

WP2Status HistoQueue::RemoveInvalidHistograms() {
  uint32_t size = 0;
  for (uint32_t i : indices_) {
    std::swap((*histograms_)[i], (*histograms_)[size]);
    ++size;
  }
  WP2_CHECK_ALLOC_OK(histograms_->resize(size));
  queue_.clear();
  return WP2_STATUS_OK;
}

void HistoQueue::MaybeUpdateHead(HistogramPair* const pair) {
  assert(pair >= queue_.data() && pair <= &queue_.back());
  assert(!queue_.empty());
  if (pair->cost_diff < queue_.front().cost_diff) {
    // Replace the best pair.
    const HistogramPair tmp = queue_.front();
    queue_.front() = *pair;
    *pair = tmp;
  }
}

// -----------------------------------------------------------------------------

// Combines histograms by continuously choosing the one with the highest cost
// reduction.
// Returns true on success, false otherwise.
static WP2Status HistogramCombineGreedy(HistogramSet* const image_histo,
                                        SegmentImgWrapper* const segments,
                                        PopulationAnalyzer* const analyzer) {
  const uint32_t image_histo_size = image_histo->histograms_.size();

  // Priority queue of histogram pairs.
  HistoQueue queue;
  WP2_CHECK_STATUS(queue.Init(&image_histo->histograms_, analyzer));
  WP2_CHECK_STATUS(queue.Resize((image_histo_size - 1) * image_histo_size / 2));

  for (uint32_t i = 0; i < image_histo_size; ++i) {
    for (uint32_t j = i + 1; j < image_histo_size; ++j) {
      WP2_CHECK_STATUS(queue.Push(i, j, 0.));
    }
  }

  while (queue.ValidIndices().size() > 1 && !queue.empty()) {
    const uint32_t idx1 = queue.front().idx1;
    queue.MergeFront(segments);

     // Push new pairs formed with combined histogram to the queue.
    for (uint32_t i : queue.ValidIndices()) {
      if (i != idx1) {
        WP2_CHECK_STATUS(queue.Push(idx1, i, /*threshold=*/0.));
      }
    }
  }
  // Remove histograms that have been merged.
  WP2_CHECK_STATUS(queue.RemoveInvalidHistograms());

  return WP2_STATUS_OK;
}

// Perform histogram aggregation using a stochastic approach.
// 'do_greedy' is set to 1 if a greedy approach needs to be performed
// afterwards, 0 otherwise.
// Returns true on success, false otherwise.
static WP2Status HistogramCombineStochastic(
    HistogramSet* const image_histo, uint32_t min_cluster_size, float min_diff,
    SegmentImgWrapper* const segments, PopulationAnalyzer* const analyzer,
    const WP2::ProgressRange& progress, bool* const do_greedy) {
  WP2::UniformIntDistribution random;
  const uint32_t outer_iters = image_histo->histograms_.size();
  const uint32_t num_tries_no_success = outer_iters / 2;
  // Priority queue of histogram pairs. Its size impacts the quality of the
  // compression and the speed: the smaller the faster but the worse for the
  // compression.
  HistoQueue queue;
  constexpr uint32_t kHistoQueueSize = 9;
  const WP2::ProgressScale iter_progress(progress,
                                         1. / std::max(1u, outer_iters));

  WP2_CHECK_STATUS(queue.Init(&image_histo->histograms_, analyzer));
  WP2_CHECK_STATUS(queue.Resize(kHistoQueueSize));

  // Collapse similar histograms in 'image_histo'.
  ++min_cluster_size;
  uint32_t valid_size = queue.ValidIndices().size();
  uint32_t iter = 0, tries_with_no_success = 0;
  for (; iter < outer_iters && valid_size >= min_cluster_size &&
         ++tries_with_no_success < num_tries_no_success;
       ++iter) {
    float best_cost = queue.empty() ? 0.f : queue.front().cost_diff;
    // valid_size / 2 was chosen empirically. Less means faster but worse
    // compression.
    const uint32_t num_tries = valid_size / 2;
    for (uint32_t j = 0; j < num_tries; ++j) {
      float curr_cost;
      // Choose two different histograms at random and try to combine them.
      uint32_t idx1 = random.Get(0u, valid_size - 1);
      uint32_t idx2 = random.Get(0u, valid_size - 2);
      if (idx2 >= idx1) ++idx2;
      idx1 = queue.ValidIndices()[idx1];
      idx2 = queue.ValidIndices()[idx2];
      if (idx1 > idx2) std::swap(idx1, idx2);

      // Calculate cost reduction on combination.
      WP2_CHECK_STATUS(queue.Push(idx1, idx2, best_cost, &curr_cost));
      if (curr_cost < min_diff) {  // found a better pair?
        best_cost = curr_cost;
        // Empty the queue if we reached full capacity.
        if (queue.full()) break;
      }
    }
    WP2_CHECK_STATUS(iter_progress.AdvanceBy(1.));
    if (queue.empty()) continue;

    // Merge the two best histograms.
    queue.MergeFront(segments);
    valid_size = queue.ValidIndices().size();

    tries_with_no_success = 0;
  }
  WP2_CHECK_STATUS(queue.RemoveInvalidHistograms());
  *do_greedy = (valid_size <= min_cluster_size);
  WP2_CHECK_STATUS(iter_progress.AdvanceBy(outer_iters - iter));
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Histogram refinement

// Make sure the ARGB image is a label image.
// ARB are 0 and only G contains indices.
// Each G value is either:
//  - <= to the previous ones
//  - equal to the max so far + 1
bool SegmentImgWrapper::IsLabelImage(uint32_t size) const {
  if (size % 4 != 0) return false;
  int16_t max_value = 0;
  const int16_t* v = data_;
  for (const int16_t* const v_end = v + size; v < v_end; v += 4) {
    if (v[0] != 0 || v[1] != 0 || v[3] != 0) return false;
    if (v[2] <= max_value) continue;
    if (v[2] - max_value > 1) return false;
    max_value = v[2];
  }
  return true;
}

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
static WP2Status HistogramRemap(const HistogramSet& in, HistogramSet* const out,
                                PopulationAnalyzer* const analyzer,
                                int16_t* const segments) {
  Histogram* const* const in_histo = in.histograms_.data();
  Histogram* const* const out_histo = out->histograms_.data();
  const uint32_t in_size = in.histograms_.size();
  const uint32_t out_size = out->histograms_.size();
  // If all histograms have been merged into 1, they all belong to the same
  // segment.
  SegmentImgWrapper segments_wrap(segments);
  if (out->histograms_.size() == 1) {
    for (uint32_t i = 0; i < in_size; ++i) segments_wrap[i] = 0;
    analyzer->UpdateHistogramCost(out->histograms_[0]);
    return WP2_STATUS_OK;
  }
  WP2::Vector_u32 mapping_count;
  WP2_CHECK_ALLOC_OK(mapping_count.resize(in_size));
  std::fill(mapping_count.begin(), mapping_count.end(), 0);

  // Make sure the mapping is as minimal as possible.
  for (int16_t i = 0; i < (int16_t)in_size; ++i) {
    int16_t* const segment = &segments_wrap[i];
    // If we have an empty histogram.
    if (*segment == -1) continue;
    if (*segment != i) {
      // If we have a histogram that is not its own segment.
      assert(*segment < i);
      *segment = segments_wrap[*segment];
    }
    ++mapping_count[*segment];
    // Make sure we reached the minimal mapping.
    assert(segments_wrap[*segment] == *segment);
  }
  // Assign empty histograms to the most common used segment to improve
  // compression.
  // TODO(vrabaud) Test with assigning to the same as the previous block to
  //               favor LZ77.
  const uint32_t best_segment =
      std::max_element(mapping_count.begin(), mapping_count.end()) -
      mapping_count.begin();
  for (uint32_t i = 0; i < in_size; ++i) {
    if (segments_wrap[i] < 0) segments_wrap[i] = best_segment;
  }

  // Re-write 'segments' so that it is a proper label image with values
  // increasing.
  std::fill(mapping_count.begin(), mapping_count.end(), in_size);
  uint32_t mapping_max = 0;
  for (uint32_t i = 0; i < in_size; ++i) {
    int16_t* const segment = &segments_wrap[i];
    if (mapping_count[*segment] == in_size) {
      mapping_count[*segment] = mapping_max++;
    }
    *segment = mapping_count[*segment];
  }
  assert(segments_wrap.IsLabelImage(4 * in_size));

  // Recompute each out based on raw and symbols.
  for (uint32_t i = 0; i < out_size; ++i) {
    out_histo[i]->Clear();
  }

  for (uint32_t i = 0; i < in_size; ++i) {
    const int idx = segments_wrap[i];
    HistogramAdd(*in_histo[i], *out_histo[idx], out_histo[idx]);
  }
  for (uint32_t i = 0; i < out_size; ++i) {
    analyzer->UpdateHistogramCost(out_histo[i]);
  }
  return WP2_STATUS_OK;
}

// Number of histograms for which we try to merge them all into one to see if
// it is more efficient.
// TODO(vrabaud) Make this parameter effort dependent.
const int kAllHistogramMergingNum = 10;

WP2Status GetHistoImageSymbols(uint32_t width, uint32_t height,
                               const BackwardRefs& refs, int effort,
                               const WP2::ProgressRange& progress,
                               int histogram_bits,
                               const LosslessSymbolsInfo& symbols_info,
                               HistogramSet* const merged_histo,
                               int16_t* const segments) {
  const WP2::ProgressRange combine_progress(progress, 0.2);
  const WP2::ProgressRange encode_progress(progress, 0.6);
  const WP2::ProgressRange reduce_progress(progress, 0.2);

  const uint32_t histo_xsize =
      histogram_bits ? SubSampleSize(width, histogram_bits) : 1;
  const uint32_t histo_ysize =
      histogram_bits ? SubSampleSize(height, histogram_bits) : 1;
  const uint32_t num_segments = histo_xsize * histo_ysize;
  HistogramSet orig_histo;
  WP2_CHECK_ALLOC_OK(orig_histo.Allocate(num_segments, symbols_info));

  // Construct the histograms from backward references.
  WP2_CHECK_STATUS(BuildHistograms(width, height, histogram_bits, symbols_info,
                                   refs, &orig_histo));
  // Copies the histograms and computes their bit_cost.
  PopulationAnalyzer analyzer(num_segments);
  WP2_CHECK_STATUS(analyzer.Allocate(symbols_info));
  for (Histogram* const h : orig_histo.histograms_) {
    analyzer.UpdateHistogramCost(h);
  }
  WP2_CHECK_STATUS(HistogramSetCopy(orig_histo, symbols_info, merged_histo));
  SegmentImgWrapper segments_wrap(segments);
  // Invalidate all histograms.
  for (uint32_t i = 0; i < num_segments; ++i) segments_wrap[i] = -1;
  // Validate non-empty histograms.
  for (const Histogram* histo : merged_histo->histograms_) {
    segments_wrap[histo->id_] = histo->id_;
  }

  WP2::Vector_u16 bin_map;
  BinMerger merger(orig_histo.histograms_.size(), effort);
  WP2_CHECK_STATUS(merger.RemoveEmptyHistograms(merged_histo, &segments_wrap));
  WP2_CHECK_ALLOC_OK(bin_map.resize(merged_histo->histograms_.size()));
  const bool entropy_combine =
      merger.AnalyzeEntropyBin(*merged_histo, bin_map.data());
  if (entropy_combine) {
    // Merge histograms with similar entropy.
    WP2_CHECK_STATUS(merger.CombineEntropyBin(bin_map, &analyzer, merged_histo,
                                              &segments_wrap));
  }

  // Don't combine the histograms using stochastic and greedy heuristics for
  // low-effort compression mode.
  if (effort > 0 || !entropy_combine) {
    const float x = (effort == 5) ? 75.f / 100.f : effort / 9.f;
    // cubic ramp between 1 and MAX_HISTO_GREEDY:
    const auto threshold_size = (int)(1 + (x * x * x) * (MAX_HISTO_GREEDY - 1));
    bool do_greedy;
    WP2_CHECK_STATUS(HistogramCombineStochastic(merged_histo, threshold_size,
                                                0.f, &segments_wrap, &analyzer,
                                                combine_progress, &do_greedy));
    if (do_greedy) {
      WP2_CHECK_STATUS(
          HistogramCombineGreedy(merged_histo, &segments_wrap, &analyzer));
    }
  } else {
    WP2_CHECK_STATUS(combine_progress.AdvanceBy(1.));
  }

  // In case we have too many histograms, merge the least bad pairs.
  while (merged_histo->histograms_.size() > kMaxHistogramImageSize) {
    // Unknown loop count but ProgressWatcher should still be called.
    const WP2::ProgressScale merge_progress(reduce_progress, 0.);
    bool do_greedy;
    WP2_CHECK_STATUS(HistogramCombineStochastic(
        merged_histo, kMaxHistogramImageSize, std::numeric_limits<float>::max(),
        &segments_wrap, &analyzer, merge_progress, &do_greedy));
  }
  WP2_CHECK_STATUS(reduce_progress.AdvanceBy(1.));

  WP2_CHECK_STATUS(
      HistogramRemap(orig_histo, merged_histo, &analyzer, segments));

  // Sometimes, combining histograms provides worse entropy. But if we
  // combine all the histograms, we could have the overall extra entropy be
  // less than the cost of adding the entropy image.
  if (merged_histo->histograms_.size() > 1 &&
      merged_histo->histograms_.size() < kAllHistogramMergingNum &&
      merged_histo->histograms_.size() < (orig_histo.histograms_.size() - 1)) {
    // Cost of all the histograms and the entropy image.
    float cost_all;
    // Figure out the entropy of the symbols image.
    {
      WP2::ANSEnc enc;
      HashChain hash_chain;
      WP2_CHECK_ALLOC_OK(hash_chain.Allocate(num_segments));
      BackwardRefsPool ref_pool;
      ref_pool.Init(num_segments);

      WP2::ANSDictionaries dicts;
      LosslessSymbolsInfo symbols_info_tmp;
      // 'has_alpha' is false as it is a label image.
      symbols_info_tmp.Init(/*has_alpha=*/false, symbols_info.SampleFormat());
      symbols_info_tmp.SetAsLabelImage();
      WP2_CHECK_STATUS(EncodeImageNoClusters(
          &enc, &dicts, segments, &hash_chain, &ref_pool, histo_ysize,
          histo_xsize, symbols_info_tmp, effort, encode_progress));
      cost_all = enc.GetCost(dicts);
    }

    // Compute the cost if we merge all histograms and update cost_all with the
    // cost of each histogram.
    Histogram* const tmp_histo = merged_histo->tmp_histo_;
    merged_histo->histograms_[0]->CopyTo(tmp_histo);
    cost_all += merged_histo->histograms_[0]->bit_cost_;
    for (uint32_t i = 1; i < merged_histo->histograms_.size(); ++i) {
      cost_all += merged_histo->histograms_[i]->bit_cost_;
      HistogramAdd(*merged_histo->histograms_[i], *tmp_histo, tmp_histo);
    }
    analyzer.UpdateHistogramCost(tmp_histo);

    if (tmp_histo->bit_cost_ < cost_all) {
      // Use that merged histogram.
      WP2_CHECK_ALLOC_OK(merged_histo->histograms_.resize(1));
      tmp_histo->CopyTo(merged_histo->histograms_[0]);
      for (uint32_t i = 0; i < num_segments; ++i) segments_wrap[i] = 0;
    }
  } else {
    WP2_CHECK_STATUS(encode_progress.AdvanceBy(1.));
  }

  return WP2_STATUS_OK;
}

}  // namespace WP2L
