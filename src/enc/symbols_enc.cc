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
//  Encoding of ANS symbols.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/enc/symbols_enc.h"

#include <algorithm>
#include <limits>

#include "src/dsp/math.h"
#include "src/utils/ans.h"

namespace WP2 {

WP2Status SymbolRecorder::MakeBackup() {
  // Save the stats to the previous pass dicts.
  for (uint32_t i = 0; i < dicts_.size(); ++i) {
    WP2_CHECK_STATUS(dicts_previous_pass_[i]->CopyFrom(*dicts_[i]));
  }
  return WP2_STATUS_OK;
}

void SymbolRecorder::ResetRecord(bool reset_backup) {
  // Save the stats to the previous pass dicts.
  if (reset_backup) {
    for (auto& d : dicts_previous_pass_) d->ResetCounts();
  }
  for (ANSDictionary* const d : dicts_) d->ResetCounts();

  uint32_t i = 0;
  for (uint32_t sym = 0; sym < symbols_info_.Size(); ++sym) {
    if (symbols_info_.Method(sym) != SymbolsInfo::StorageMethod::kAdaptiveBit) {
      continue;
    }
    for (uint32_t cluster = 0; cluster < symbols_info_.NumClusters(sym);
         ++cluster) {
      a_bits_[i] = ANSBinSymbol(symbols_info_.StartingProbaP0(sym, cluster),
                                symbols_info_.StartingProbaP1(sym, cluster));
      ++i;
    }
  }
  assert(i == a_bits_.size());

  for (ANSAdaptiveSymbol& sym : a_symbols_) {
    sym.InitFromUniform(sym.NumSymbols());
  }
  for (Vector_u8& v : values_) v.clear();
}

WP2Status SymbolRecorder::CopyFrom(const SymbolRecorder& other) {
  WP2_CHECK_STATUS(symbols_info_.CopyFrom(other.symbols_info_));
  index_ = other.index_;
  values_index_ = other.values_index_;
  WP2_CHECK_STATUS(dicts_.CopyFrom(other.dicts_));
  WP2_CHECK_STATUS(dicts_previous_pass_.CopyFrom(other.dicts_previous_pass_));
  WP2_CHECK_STATUS(a_bits_.CopyFrom(other.a_bits_));
  WP2_CHECK_STATUS(a_symbols_.CopyFrom(other.a_symbols_));
  WP2_CHECK_ALLOC_OK(values_.resize(other.values_.size()));
  for (uint32_t i = 0; i < values_.size(); ++i) {
    values_[i].clear();
    // Each sub vector is allocated only once with a fixed capacity.
    WP2_CHECK_ALLOC_OK(values_[i].reserve(other.values_[i].capacity()));
    WP2_CHECK_ALLOC_OK(values_[i].copy_from(other.values_[i]));
  }
  return WP2_STATUS_OK;
}

void SymbolRecorder::DeepClear() {
  index_ = {};
  dicts_.DeepClear();
  dicts_previous_pass_.DeepClear();
  a_bits_.clear();
  a_symbols_.clear();
  values_.clear();
}

WP2Status SymbolRecorder::Allocate(const SymbolsInfo& symbols_info,
                                   uint32_t num_records) {
  WP2_CHECK_STATUS(symbols_info_.CopyFrom(symbols_info));
  DeepClear();

  // Allocate dictionaries.
  for (uint32_t sym = 0; sym < symbols_info_.Size(); ++sym) {
    const uint32_t num_clusters = symbols_info_.NumClusters(sym);
    switch (symbols_info_.Method(sym)) {
      case SymbolsInfo::StorageMethod::kAuto: {
        index_[sym] = dicts_.size();
        for (uint32_t cluster = 0; cluster < num_clusters; ++cluster) {
          const uint32_t range = symbols_info_.Range(sym, cluster);
          if (range > 0) {
            WP2_CHECK_STATUS(dicts_.Add(range));
            WP2_CHECK_STATUS(dicts_previous_pass_.Add(range));
          }
        }
        break;
      }
      case SymbolsInfo::StorageMethod::kAdaptiveBit: {
        index_[sym] = a_bits_.size();
        for (uint32_t cluster = 0; cluster < num_clusters; ++cluster) {
          const ANSBinSymbol s(symbols_info_.StartingProbaP0(sym, cluster),
                               symbols_info_.StartingProbaP1(sym, cluster));
          WP2_CHECK_ALLOC_OK(a_bits_.push_back(s));
        }
        break;
      }
      case SymbolsInfo::StorageMethod::kAdaptiveSym: {
        index_[sym] = a_symbols_.size();
        for (uint32_t cluster = 0; cluster < num_clusters; ++cluster) {
          const uint32_t range = symbols_info_.Range(sym, cluster);
          if (range > 0) WP2_CHECK_ALLOC_OK(a_symbols_.Add(range));
        }
        break;
      }
      case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed: {
        // Must fit in a uint8_t
        assert(symbols_info_.GetMaxRange(sym) < (1 << 8));
        index_[sym] = a_symbols_.size();
        values_index_[sym] = values_.size();
        WP2_CHECK_ALLOC_OK(values_.resize(values_.size() + num_clusters));
        for (uint32_t cluster = 0; cluster < num_clusters; ++cluster) {
          const uint32_t range = symbols_info_.Range(sym, cluster);
          if (range > 0) {
            WP2_CHECK_ALLOC_OK(a_symbols_.Add(range));
            WP2_CHECK_ALLOC_OK(
                values_[values_index_[sym] + cluster].reserve(num_records));
          }
        }
        break;
      }
      case SymbolsInfo::StorageMethod::kUnused: {
        break;
      }
    }
  }
  return WP2_STATUS_OK;
}

const ANSDictionary& SymbolRecorder::GetRecordedDict(uint32_t sym,
                                                     uint32_t cluster) const {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAuto);
  return *dicts_[index_[sym] + cluster];
}

const ANSDictionary& SymbolRecorder::GetDictPreviousPass(
    uint32_t sym, uint32_t cluster) const {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAuto);
  return *dicts_previous_pass_[index_[sym] + cluster];
}

ANSDictionary* SymbolRecorder::GetDict(uint32_t sym, uint32_t cluster) {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAuto);
  return dicts_[index_[sym] + cluster];
}

const ANSBinSymbol& SymbolRecorder::GetABit(uint32_t sym,
                                            uint32_t cluster) const {
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAdaptiveBit);
  return a_bits_[index_[sym] + cluster];
}

ANSBinSymbol* SymbolRecorder::GetABit(uint32_t sym, uint32_t cluster) {
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAdaptiveBit);
  return &a_bits_[index_[sym] + cluster];
}

const ANSAdaptiveSymbol& SymbolRecorder::GetASymbol(uint32_t sym,
                                                    uint32_t cluster) const {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAdaptiveSym);
  return a_symbols_[index_[sym] + cluster];
}

ANSAdaptiveSymbol* SymbolRecorder::GetASymbol(uint32_t sym, uint32_t cluster) {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAdaptiveSym);
  return &a_symbols_[index_[sym] + cluster];
}

const ANSAdaptiveSymbol& SymbolRecorder::GetASymbolWithSpeed(
    uint32_t sym, uint32_t cluster) const {
  assert(symbols_info_.Range(sym, cluster) != 0);
  assert(symbols_info_.Method(sym) ==
         SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed);
  return a_symbols_[index_[sym] + cluster];
}

const Vector_u8& SymbolRecorder::GetRecordedValues(uint32_t sym,
                                                   uint32_t cluster) const {
  assert(symbols_info_.Method(sym) ==
         SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed);
  return values_[values_index_[sym] + cluster];
}

int32_t SymbolRecorder::ProcessInternal(uint32_t sym, uint32_t cluster,
                                        int32_t value, bool use_max_value,
                                        uint32_t max_value, WP2_OPT_LABEL,
                                        ANSEncBase* const enc, float* const) {
  assert(value <= symbols_info_.Max(sym, cluster));
  assert(value >= symbols_info_.Min(sym, cluster));
  assert(!use_max_value || (uint32_t)std::abs(value) <= max_value);
  switch (symbols_info_.Method(sym)) {
    case SymbolsInfo::StorageMethod::kAuto:
      GetDict(sym, cluster)
          ->RecordSymbol(value - symbols_info().Min(sym, cluster));
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveBit:
      GetABit(sym, cluster)->Update(value);
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveSym:
      GetASymbol(sym, cluster)
          ->Update(value - symbols_info().Min(sym, cluster));
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed:
      a_symbols_[index_[sym] + cluster].Update(
          value - symbols_info().Min(sym, cluster));
      if (!values_[values_index_[sym] + cluster].push_back(
              value - symbols_info().Min(sym, cluster),
              /*resize_if_needed=*/false)) {
        // Fail if we get more values than expected.
        assert(false);
      }
      break;
    case SymbolsInfo::StorageMethod::kUnused:
      // Even though the object can be created with unused symbols (e.g. alpha
      // symbols), those should never be used.
      assert(false);
      break;
  }
  return value;
}

//------------------------------------------------------------------------------

constexpr uint32_t SymbolCounter::kInvalidIndex;

WP2Status SymbolCounter::Allocate(std::initializer_list<uint32_t> syms) {
  num_a_bits_ = 0u;
  num_a_symbols_ = 0u;
  indices_.fill(kInvalidIndex);
  for (uint32_t sym : syms) {
    assert(sym < kSymbolNumMax);
    // In some partitioning (e.g. BlockScoreFunc), symbols_info is
    // actually empty, so choose the maximum possible.
    const uint32_t num_clusters = (symbols_info_.Size() == 0)
                                      ? kMaxLossyClusters
                                      : symbols_info_.NumClusters(sym);
    switch (symbols_info_.Method(sym)) {
      case SymbolsInfo::StorageMethod::kAdaptiveBit:
        indices_[sym] = num_a_bits_;
        num_a_bits_ += num_clusters;
        break;
      case SymbolsInfo::StorageMethod::kAdaptiveSym:
      case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed:
        indices_[sym] = num_a_symbols_;
        num_a_symbols_ += num_clusters;
        break;
      case SymbolsInfo::StorageMethod::kUnused:
        break;
      default:
        // In practice, it should be assert(false) but some symbols might have
        // a different method during tests.
        break;
    }
  }
  WP2_CHECK_ALLOC_OK(a_bits_.resize(num_a_bits_));
  WP2_CHECK_ALLOC_OK(a_symbols_.resize(num_a_symbols_));
  // If the following asserts ever break, just increase the size of the arrays.
  // For now, the sizes are tuned to our current usage.
  assert(num_a_bits_ <= a_bit_initialized_.size());
  assert(num_a_symbols_ <= a_symbol_initialized_.size());
  Clear();
  return WP2_STATUS_OK;
}

WP2Status SymbolCounter::CopyFrom(const SymbolCounter& other) {
  // Nothing to copy in base classes SymbolManager and WP2Allocable.
  indices_ = other.indices_;
  num_a_bits_ = other.num_a_bits_;
  a_bit_initialized_ = other.a_bit_initialized_;
  WP2_CHECK_STATUS(a_bits_.CopyFrom(other.a_bits_));
  num_a_symbols_ = other.num_a_symbols_;
  a_symbol_initialized_ = other.a_symbol_initialized_;
  WP2_CHECK_STATUS(a_symbols_.CopyFrom(other.a_symbols_));
  return WP2_STATUS_OK;
}

void SymbolCounter::Clear() {
  // Somehow, memset is faster than std::fill.
  memset(&a_bit_initialized_[0], 0, num_a_bits_ * sizeof(false));
  memset(&a_symbol_initialized_[0], 0, num_a_symbols_ * sizeof(false));
}

ANSBinSymbol* SymbolCounter::GetABit(uint32_t sym, uint32_t cluster) {
  assert(symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAdaptiveBit);
  assert(indices_[sym] != kInvalidIndex);
  const uint32_t ind = indices_[sym] + cluster;
  if (!a_bit_initialized_[ind]) {
    a_bits_[ind] = recorder_->GetABit(sym, cluster);
    a_bit_initialized_[ind] = true;
  }
  return &a_bits_[ind];
}

ANSAdaptiveSymbol* SymbolCounter::GetASymbol(uint32_t sym, uint32_t cluster) {
  assert(symbols_info_.Range(sym, cluster) != 0);
  const SymbolsInfo::StorageMethod method = symbols_info_.Method(sym);
  assert(method == SymbolsInfo::StorageMethod::kAdaptiveSym ||
         method == SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed);
  assert(indices_[sym] != kInvalidIndex);
  const uint32_t ind = indices_[sym] + cluster;
  if (!a_symbol_initialized_[ind]) {
    if (method == SymbolsInfo::StorageMethod::kAdaptiveSym) {
      a_symbols_[ind] = recorder_->GetASymbol(sym, cluster);
    } else {
      // Adaptive symbol with auto speed. We approximate it as regular adaptive
      // symbol with the default speed.
      // TODO(maryla): not sure this is the best. Maybe just assume a range?
      a_symbols_[ind] = recorder_->GetASymbolWithSpeed(sym, cluster);
    }
    a_symbol_initialized_[ind] = true;
  }
  return &a_symbols_[ind];
}

int32_t SymbolCounter::ProcessInternal(uint32_t sym, uint32_t cluster,
                                       int32_t value_in, bool use_max_value,
                                       uint32_t max_value, WP2_OPT_LABEL,
                                       ANSEncBase* const enc, float* const) {
  assert(value_in <= symbols_info_.Max(sym, cluster));
  assert(value_in >= symbols_info_.Min(sym, cluster));
  assert(!use_max_value || (uint32_t)std::abs(value_in) <= max_value);
  switch (symbols_info_.Method(sym)) {
    case SymbolsInfo::StorageMethod::kAuto: {
      const ANSDictionary& current_dict =
          recorder_->GetRecordedDict(sym, cluster);
      const ANSDictionary& previous_dict =
          recorder_->GetDictPreviousPass(sym, cluster);

      // Whether the recorder has accumulated enough stats (by some arbitrary
      // criteria).
      const bool enough_stats =
          (current_dict.Total() > 50 ||
           current_dict.Total() > 10 * symbols_info_.Range(sym, cluster));
      const ANSDictionary* const dict =
          enough_stats ? &current_dict : &previous_dict;
      const uint32_t value = value_in - symbols_info().Min(sym, cluster);
      const bool symbol_present = (dict->Counts()[value] > 0);
      if (symbol_present) {
        // TODO(maryla): average the two dicts instead of sharply transitioning
        // between the two?
        if (use_max_value) {
          enc->PutSymbol(value, max_value, *dict, label);
        } else {
          enc->PutSymbol(value, *dict, label);
        }
      } else if (enough_stats || previous_dict.Total() > 0) {
        // If the symbol is not present but we have enough data, assume a proba
        // of 1/Total.
        // Emulate the cost of log2(1/Total) by using a range.
        enc->PutRValue(0, dict->Total(), label);
      } else {
        // Otherwise we assume all values have the same probability, i.e. it's
        // a range.
        uint32_t range = symbols_info_.Range(sym, cluster);
        if (use_max_value) range = std::min(range, max_value + 1);
        if (symbols_info_.ProbablyGeometric(sym)) {
          // Very approximate. Favors lower values. TODO(maryla): find a better
          // approximation of golomb cost.
          range = std::min(range, (value == 0) ? 2 : ((uint32_t)value << 2));
        }
        enc->PutRValue(value, range, label);
      }
      break;
    }
    case SymbolsInfo::StorageMethod::kAdaptiveBit:
      enc->PutABit(value_in, GetABit(sym, cluster), label);
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveSym:
    case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed:
      enc->PutASymbol(value_in, GetASymbol(sym, cluster), label);
      break;
    case SymbolsInfo::StorageMethod::kUnused:
      // Even though the object can be created with unused symbols (e.g. alpha
      // symbols), those should never be used.
      assert(false);
      break;
  }
  return value_in;
}

//------------------------------------------------------------------------------

WP2Status SymbolWriter::Allocate() {
  WP2_CHECK_STATUS(SymbolIO<SymbolWriterStatExtra>::Allocate());
  const uint32_t symbol_range_max = symbols_info_.GetMaxRange();
  WP2_CHECK_ALLOC_OK(histogram_.resize(symbol_range_max));
  WP2_CHECK_ALLOC_OK(mapping_.resize(symbol_range_max));
  WP2_CHECK_ALLOC_OK(infos_tmp_.resize(symbol_range_max));
  // The following buffers are also used when storing recursive histograms
  // (storage type HuffmanANS). Assuming a symbol appears at most once per
  // pixel, the range of a huffmanized histogram is log2(max pixels).
  constexpr uint32_t kMaxHistoValueBits = WP2Log2Ceil_k(kMaxTilePixels);
  const uint32_t histo_max_range =
      std::max(symbol_range_max, kMaxHistoValueBits);
  WP2_CHECK_ALLOC_OK(stats_buffer_.resize(histo_max_range));
  WP2_CHECK_ALLOC_OK(counts_tmp_.reserve(histo_max_range));

  const Golomb golomb1(symbol_range_max - 1, 1);
  const Golomb golomb2(symbol_range_max - 1, 2);
  const uint32_t golomb_range_max =
      1 + std::max(golomb1.prefix, golomb2.prefix);
  WP2_CHECK_ALLOC_OK(histogram_golomb_.resize(golomb_range_max));
  WP2_CHECK_ALLOC_OK(mapping_golomb_.resize(golomb_range_max));
  WP2_CHECK_STATUS(
      quantizer_.Allocate(std::max(symbol_range_max, golomb_range_max)));
  return WP2_STATUS_OK;
}

WP2Status SymbolWriter::CopyFrom(const SymbolWriter& other,
                                 const ANSDictionaries& original_dicts,
                                 const ANSDictionaries& copied_dicts) {
  // Nothing to copy in base classes SymbolManager and WP2Allocable.

  // Deep copy base class SymbolIO<SymbolWriterStatExtra>. This is easier to do
  // it here rather than in SymbolIO because the template is known.
  WP2_CHECK_STATUS(symbols_info_.CopyFrom(other.symbols_info_));
  WP2_CHECK_STATUS(Allocate());  // Needed for 'stats_start_', Stat::mappings

  WP2_CHECK_STATUS(a_bits_.CopyFrom(other.a_bits_));
  WP2_CHECK_STATUS(a_symbols_.CopyFrom(other.a_symbols_));
  // 'all_stats_' cannot be copied as is, it contains references.
  assert(all_stats_.size() == other.all_stats_.size());
  for (uint32_t i = 0; i < all_stats_.size(); ++i) {
    Stat& stat = all_stats_[i];
    const Stat& other_stat = other.all_stats_[i];

    stat.type = other_stat.type;
    stat.use_mapping = other_stat.use_mapping;
    // Keep the Stat::mappings that was assigned during Allocate().
    stat.range = other_stat.range;
    stat.param = other_stat.param;

    if (stat.type != SymbolWriter::Stat::Type::kUnknown) {
      SymbolWriterStatExtra& extra = stat.extra;
      const SymbolWriterStatExtra& other_extra = other_stat.extra;

      extra.dict = copied_dicts.GetEquivalent(original_dicts, other_extra.dict);
      assert((extra.dict != nullptr) == (other_extra.dict != nullptr));
      extra.mapping_size = other_extra.mapping_size;
    }
  }
  // 'mappings_buffer_' cannot be reallocated, it is referenced in Allocate().
  assert(mappings_buffer_.size() == other.mappings_buffer_.size());
  std::copy(other.mappings_buffer_.begin(), other.mappings_buffer_.end(),
            mappings_buffer_.begin());

  // Deep copy remaining direct members of SymbolWriter.
  WP2_CHECK_ALLOC_OK(infos_tmp_.copy_from(other.infos_tmp_));
  WP2_CHECK_ALLOC_OK(counts_tmp_.copy_from(other.counts_tmp_));
  WP2_CHECK_ALLOC_OK(histogram_.copy_from(other.histogram_));
  WP2_CHECK_ALLOC_OK(mapping_.copy_from(other.mapping_));
  WP2_CHECK_ALLOC_OK(histogram_golomb_.copy_from(other.histogram_golomb_));
  WP2_CHECK_ALLOC_OK(mapping_golomb_.copy_from(other.mapping_golomb_));
  // No need to copy Quantizer instances. They only contain data members to
  // avoid reallocations. TODO(yguyon): Check if more could be skipped
  WP2_CHECK_ALLOC_OK(stats_buffer_.copy_from(other.stats_buffer_));
  effort_ = other.effort_;
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void SymbolWriter::WriteHistogram(const Quantizer::Config& config,
                                  uint32_t symbol_range, uint32_t max_count,
                                  ANSEncBase* const enc) {
  config.WriteHistogramHeader(symbol_range, stats_buffer_.data(),
                              stats_buffer_.size(), enc);
  config.WriteHistogramCounts(symbol_range, max_count, stats_buffer_.data(),
                              stats_buffer_.size(), enc);
}

//------------------------------------------------------------------------------

static bool ANSSymbolInfoCmp(const ANSSymbolInfo& i1, const ANSSymbolInfo& i2) {
  return (i1.symbol < i2.symbol);
}

void SymbolWriter::AddTrivial(uint32_t sym, uint32_t cluster, bool is_symmetric,
                              int32_t value) {
  Stat* const stat = GetStats(sym, cluster);

  // No dictionary for trivial symbols.
  stat->type = Stat::Type::kTrivial;
  stat->is_symmetric = is_symmetric;
  stat->param.trivial_value = value;
  stat->extra.dict = nullptr;
  stat->extra.mapping_size = 0;
}

void SymbolWriter::AddRange(uint32_t sym, uint32_t cluster, bool is_symmetric,
                            const uint16_t* const mapping, uint32_t size,
                            uint16_t max_range) {
  Stat* const stat = GetStats(sym, cluster);

  stat->type = Stat::Type::kRange;
  stat->is_symmetric = is_symmetric;
  stat->use_mapping = (mapping != nullptr);
  stat->extra.dict = nullptr;
  stat->extra.mapping_size = 0;
  if (stat->use_mapping) {
    stat->range = size;
    stat->extra.mapping_size =
        1 + *std::max_element(&mapping[0], &mapping[0] + size);
    std::fill(stat->mappings, stat->mappings + stat->extra.mapping_size,
              Stat::kInvalidMapping);
    for (uint32_t i = 0; i < size; ++i) {
      stat->mappings[mapping[i]] = i;
    }
  } else {
    assert(size == 0);
    stat->range = max_range;
  }
}

WP2Status SymbolWriter::AddDict(uint32_t sym, uint32_t cluster,
                                bool is_symmetric, const uint32_t* const counts,
                                const uint32_t* const quantized_counts,
                                const uint16_t* const mapping, uint32_t size,
                                ANSDictionaries* const dicts) {
  assert(mapping != nullptr);
  assert(quantized_counts != nullptr);

  Stat* const stat = GetStats(sym, cluster);

  stat->type = Stat::Type::kDict;
  stat->is_symmetric = is_symmetric;
  stat->use_mapping = true;

  // Sort with the most frequent symbols first.
  for (uint32_t i = 0; i < size; ++i) {
    infos_tmp_[i].freq = quantized_counts[i];
    assert(infos_tmp_[i].freq == quantized_counts[i]); /* no overflow */
    infos_tmp_[i].symbol = mapping[i];
  }
  std::sort(&infos_tmp_[0], &infos_tmp_[0] + size, ANSSymbolInfoCmp);
  // Figure out the mappings.
  stat->extra.mapping_size = 0;
  for (uint32_t k = 0; k < size; ++k) {
    if (infos_tmp_[k].symbol >= stat->extra.mapping_size) {
      stat->extra.mapping_size = 1 + infos_tmp_[k].symbol;
    }
  }
  std::fill(stat->mappings, stat->mappings + stat->extra.mapping_size,
            Stat::kInvalidMapping);
  WP2_CHECK_ALLOC_OK(counts_tmp_.resize(size));
  for (uint32_t k = 0; k < size; ++k) {
    stat->mappings[infos_tmp_[k].symbol] = k;
    counts_tmp_[k] = infos_tmp_[k].freq;
  }
  WP2_CHECK_STATUS(dicts->Add(size));
  auto* const d = dicts->back();
  // In case the counts were already quantized.
  for (uint32_t k = 0; k < size; ++k) {
    d->RecordSymbol(stat->mappings[mapping[k]], counts[k]);
  }
  WP2_CHECK_STATUS(d->SetQuantizedCounts(counts_tmp_));
  WP2_CHECK_STATUS(d->ToCodingTable());
  stat->extra.dict = d;
  return WP2_STATUS_OK;
}

WP2Status SymbolWriter::AddGolomb(uint32_t sym, uint32_t cluster,
                                  bool is_symmetric,
                                  const uint32_t* const counts,
                                  const uint32_t* const quantized_counts,
                                  const uint16_t* const mapping, uint32_t size,
                                  uint32_t prefix_size,
                                  ANSDictionaries* const dicts) {
  WP2_CHECK_STATUS(AddDict(sym, cluster, is_symmetric, counts, quantized_counts,
                           mapping, size, dicts));
  SetGolombStat(sym, cluster, prefix_size);
  return WP2_STATUS_OK;
}

uint32_t SymbolWriter::FindBiggestMappingIndex(const Stat& stat,
                                               uint32_t max_index) {
  const uint32_t index_ini =
      std::min(max_index, (uint32_t)(stat.extra.mapping_size - 1));
  uint32_t index = index_ini;
  while (index > 0 && stat.mappings[index] == Stat::kInvalidMapping) {
    --index;
  }
  if (index == 0 && stat.mappings[index] == Stat::kInvalidMapping) {
    index = index_ini;
    while (index < stat.extra.mapping_size &&
           stat.mappings[index] == Stat::kInvalidMapping) {
      ++index;
    }
    assert(index < stat.extra.mapping_size);
  }
  assert(stat.mappings[index] != Stat::kInvalidMapping);
  return index;
}

int32_t SymbolWriter::ProcessInternal(uint32_t sym, uint32_t cluster,
                                      int32_t value_in, bool use_max_value,
                                      uint32_t max_value, WP2_OPT_LABEL,
                                      ANSEncBase* const enc,
                                      float* const cost) {
  if (use_max_value) assert(value_in <= (int32_t)max_value);
  const Stat& stat = *GetStats(sym, cluster);
  const int16_t min_symbol = symbols_info().Min(sym, cluster);
  ANSDebugPrefix debug_prefix(enc, label);
  const uint32_t value =
      (stat.is_symmetric) ? std::abs(value_in) : value_in - min_symbol;
  switch (stat.type) {
    case Stat::Type::kTrivial:
      assert(value == stat.param.trivial_value);
      break;
    case Stat::Type::kRange: {
      uint32_t range;
      if (!use_max_value) {
        range = stat.range;
      } else if (stat.use_mapping) {
        // Find the biggest range <= max_value that is valid.
        const uint32_t max_index = FindBiggestMappingIndex(stat, max_value);
        range = stat.mappings[max_index] + 1;
        assert(range <= stat.range);
      } else {
        range = std::min(stat.range, (uint16_t)(max_value + 1));
      }
      enc->PutRValue(stat.use_mapping ? stat.mappings[value] : value, range,
                     "range");
      if (cost != nullptr) *cost += std::log2(range);
      break;
    }
    case Stat::Type::kDict: {
      if (use_max_value) {
        // Get the info of the maximally usable symbol.
        max_value =
            std::min(max_value, (uint32_t)(stat.extra.mapping_size - 1));
        while (max_value > 0 &&
               stat.mappings[max_value] == Stat::kInvalidMapping) {
          --max_value;
        }
        assert(stat.mappings[max_value] != Stat::kInvalidMapping);
        enc->PutSymbol(stat.mappings[value], stat.mappings[max_value],
                       *stat.extra.dict, "dict");
        if (cost != nullptr) {
          *cost += stat.extra.dict->SymbolCost(stat.mappings[value],
                                               stat.mappings[max_value]);
        }
      } else {
        enc->PutSymbol(stat.mappings[value], *stat.extra.dict, "dict");
        if (cost != nullptr) {
          *cost += stat.extra.dict->SymbolCost(stat.mappings[value]);
        }
      }
      break;
    }
    case Stat::Type::kGolomb: {
      // TODO(vrabaud) restrict when use_max_value is set.
      const uint32_t prefix_size = stat.param.golomb.prefix_size;
      const Golomb golomb(value, prefix_size);
      bool use_range;
      uint32_t range = 0;
      if (use_max_value) {
        const Golomb golomb_max(max_value, prefix_size);
        const uint32_t max_prefix =
            FindBiggestMappingIndex(stat, golomb_max.prefix);
        enc->PutSymbol(stat.mappings[golomb.prefix], stat.mappings[max_prefix],
                       *stat.extra.dict, "golomb");
        use_range = (golomb.prefix == max_prefix);
        if (use_range) {
          // Get (the biggest value with a prefix of 'max_prefix') + 1.
          range = std::min(
              max_value + 1,
              Golomb::Merge(max_prefix, prefix_size, 0) +
                  (1 << Golomb::NumExtraBits(max_prefix, prefix_size)));
        }
        if (cost != nullptr) {
          *cost += stat.extra.dict->SymbolCost(stat.mappings[golomb.prefix],
                                               stat.mappings[max_prefix]);
        }
      } else {
        enc->PutSymbol(stat.mappings[golomb.prefix], *stat.extra.dict,
                       "golomb");
        // If the last bit interval is truncated, go with ranges.
        use_range = (golomb.prefix + 1 == stat.param.golomb.range);
        if (use_range) range = stat.range;
        if (cost != nullptr) {
          *cost += stat.extra.dict->SymbolCost(stat.mappings[golomb.prefix]);
        }
      }
      if (golomb.extra_bits_num > 0) {
        if (use_range) {
          assert(Golomb::Merge(golomb.prefix, prefix_size, 0) < range);
          range -= Golomb::Merge(golomb.prefix, prefix_size, 0);
          enc->PutRValue(golomb.extra_bits_value, range, "extra_bits_value");
          if (cost != nullptr) {
            *cost += std::log2(range);
          }
        } else {
          enc->PutUValue(golomb.extra_bits_value, golomb.extra_bits_num,
                         "extra_bits_value");
          if (cost != nullptr) {
            *cost += golomb.extra_bits_num;
          }
        }
      }
      break;
    }
    case Stat::Type::kAdaptiveBit: {
      if (!use_max_value || max_value > 0) {
        enc->PutBit(value, a_bits_[stat.param.a_bit_index].NumZeros(),
                    a_bits_[stat.param.a_bit_index].NumTotal(), "a_bit");
        if (cost != nullptr) {
          *cost += a_bits_[stat.param.a_bit_index].GetCost(value);
        }
      }
      a_bits_[stat.param.a_bit_index].Update(value);
      break;
    }
    case Stat::Type::kAdaptiveSymbol: {
      if (use_max_value) {
        enc->PutSymbol(value, max_value, a_symbols_[stat.param.a_symbol_index],
                       "a_symbol");
        if (cost != nullptr) {
          *cost +=
              a_symbols_[stat.param.a_symbol_index].GetCost(value, max_value);
        }
      } else {
        enc->PutSymbol(value, a_symbols_[stat.param.a_symbol_index],
                       "a_symbol");
        if (cost != nullptr) {
          *cost += a_symbols_[stat.param.a_symbol_index].GetCost(value);
        }
      }
      // TODO(vrabaud) take the max_value into account when updating.
      a_symbols_[stat.param.a_symbol_index].Update(value);
      break;
    }
    default:
      // Did you forget to call WriteHeader for this symbol?
      assert(false);
      break;
  }
  // Store the sign.
  if (value_in != 0 && stat.is_symmetric) {
    enc->PutBool(value_in < 0, "is_negative");
    if (cost != nullptr) *cost += 1.;
  }
  return value_in;
}

uint32_t SymbolWriter::FillCachedGolombHistogram(uint32_t range, uint32_t size,
                                                 uint32_t prefix_size) {
  // Analyze whether a Golomb encoding would work.
  const Golomb golomb(range - 1, prefix_size);
  const uint32_t range_golomb = golomb.prefix + 1;
  assert(range_golomb <= histogram_golomb_.size());
  std::fill(&histogram_golomb_[0], &histogram_golomb_[0] + range_golomb, 0);
  uint32_t size_golomb = 0;
  for (uint32_t i = 0; i < size; ++i) {
    const Golomb golomb_i(mapping_[i], prefix_size);
    if (size_golomb == 0 ||
        golomb_i.prefix != mapping_golomb_[size_golomb - 1]) {
      mapping_golomb_[size_golomb++] = golomb_i.prefix;
      assert(size_golomb - 1 < mapping_golomb_.size());
    }
    histogram_golomb_[size_golomb - 1] += histogram_[i];
  }
  return size_golomb;
}

bool SymbolWriter::ComputeCachedGolombHistogramCost(uint32_t range,
                                                    uint32_t max_nnz,
                                                    uint32_t size,
                                                    uint32_t prefix_size,
                                                    float cost_max) {
  const uint32_t size_golomb =
      FillCachedGolombHistogram(range, size, prefix_size);
  // Do not use Golomb if there is only one value for now.
  if (size_golomb == 1) {
    // TODO(vrabaud) Allow for a unique value (probably requires recursion).
    return false;
  }
  const Golomb golomb(range - 1, prefix_size);
  const uint32_t range_golomb = golomb.prefix + 1;
  const uint32_t nnz_range_golomb = std::min(max_nnz, range_golomb);
  // Add the prefix_size and size cost.
  float cost_extra = 1.f + WP2Log2(nnz_range_golomb - 2);
  for (uint32_t i = 0; i < size; ++i) {
    const Golomb golomb_i(mapping_[i], prefix_size);
    cost_extra += golomb_i.extra_bits_num * histogram_[i];
  }
  // Quantize the histogram to get the cost of using a dictionary.
  if (quantizer_.Quantize(histogram_golomb_.data(), mapping_golomb_.data(),
                          size_golomb, range_golomb, max_nnz,
                          cost_max, cost_extra, effort_)) {
    Quantizer::Config* config_best = quantizer_.GetBest();
    config_best->param.golomb_size = size_golomb;
    config_best->param.golomb_prefix_size = prefix_size;
    return true;
  } else {
    // We could not beat the previous configuration.
    return false;
  }
}

WP2Status SymbolWriter::WriteHeader(uint32_t sym, uint32_t cluster,
                                    uint32_t max_nnz,
                                    const SymbolRecorder& recorder,
                                    WP2_OPT_LABEL, ANSEncBase* const enc,
                                    ANSDictionaries* const dicts) {
  if (symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kUnused) {
    return WP2_STATUS_OK;
  }
  const uint32_t range = symbols_info_.Range(sym, cluster);
  if (range < 2) {
    assert(range > 0);
    // Set the value to 0 as we offset by the symbol minimum when is_symmetric
    // is false.
    AddTrivial(sym, cluster, /*is_symmetric=*/false, /*value=*/0);
    return WP2_STATUS_OK;
  }
  // We only allow negative values if we have kAuto.
  assert(symbols_info_.Min(sym, cluster) >= 0 ||
         symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kAuto);

  switch (symbols_info_.Method(sym)) {
    case SymbolsInfo::StorageMethod::kAdaptiveBit:
      WP2_CHECK_STATUS(AddAdaptiveBit(
          sym, cluster, symbols_info_.StartingProbaP0(sym, cluster),
          symbols_info_.StartingProbaP1(sym, cluster)));
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveSym: {
      WP2_CHECK_STATUS(AddAdaptiveSymbol(sym, cluster,
                                         ANSAdaptiveSymbol::Method::kAOM,
                                         kANSAProbaInvalidSpeed));
      break;
    }
    case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed: {
      if (max_nnz <= 1) {
        // Adaptation method/speed don't matter if we only use this symbol once.
        return AddAdaptiveSymbol(sym, cluster,
                                 ANSAdaptiveSymbol::Method::kConstant, 0);
      }
      ANSDebugPrefix prefix(enc, label);
      ANSAdaptiveSymbol s;
      s.SetAdaptationSpeed(ANSAdaptiveSymbol::Method::kAOM);
      s.InitFromUniform(symbols_info_.Range(sym, cluster));
      ANSAdaptiveSymbol::Method method;
      uint32_t speed;
      s.FindBestAdaptationSpeed(
          recorder.GetRecordedValues(sym, cluster), &method, &speed);
      enc->PutRValue((uint32_t)method,
                     (uint32_t)ANSAdaptiveSymbol::Method::kNum,
                     "adaptation_method");
      if (method == ANSAdaptiveSymbol::Method::kConstant) {
        enc->PutRValue(speed, kNumAdaptationSpeeds, "adaptation_speed");
        speed = kAdaptationSpeeds[speed];
      } else {
        speed = kANSAProbaInvalidSpeed;
      }
      WP2_CHECK_STATUS(AddAdaptiveSymbol(sym, cluster, method, speed));
      return enc->GetStatus();
    }
    case SymbolsInfo::StorageMethod::kAuto: {
      return WriteHeader(
          sym, cluster, max_nnz,
          recorder.GetRecordedDict(sym, cluster).Counts().data(),
          label, enc, dicts);
    }
    case SymbolsInfo::StorageMethod::kUnused:
      // Dealt with above.
      assert(false);
      break;
  }
  return WP2_STATUS_OK;
}

WP2Status SymbolWriter::WriteHeader(uint32_t sym, uint32_t max_nnz,
                                    const SymbolRecorder& syntax_recorder,
                                    WP2_OPT_LABEL, ANSEncBase* const enc,
                                    ANSDictionaries* const dicts) {
  for (uint32_t cluster = 0; cluster < symbols_info_.NumClusters(sym);
       ++cluster) {
    WP2_CHECK_STATUS(
        WriteHeader(sym, cluster, max_nnz, syntax_recorder, label, enc, dicts));
  }
  return WP2_STATUS_OK;
}

// If is_symmetric is set to true, the counts for v and -v are added
// as counts for abs(v).
void SymbolWriter::ConvertCountsToCachedHistogram(const uint32_t* const counts,
                                                  int min, int max,
                                                  bool is_symmetric,
                                                  uint32_t* size,
                                                  uint32_t* counts_total) {
  *size = 0;
  if (is_symmetric) {
    assert(min < 0);
    *counts_total = 0;
    for (int i = 0; i <= std::max(-min, max); ++i) {
      // If the values can be negative, we clump together the statistics for x
      // and -x.
      uint32_t count;
      if (i == 0) {
        count = counts[-min];
      } else {
        count = 0;
        if (-i >= min) count += counts[-min - i];
        if (i <= max) count += counts[-min + i];
      }
      if (count == 0) continue;
      histogram_[*size] = count;
      *counts_total += count;
      mapping_[*size] = i;
      ++*size;
    }
  } else {
    const int range = max - min + 1;
    for (int i = 0; i < range; ++i) {
      const uint32_t count = counts[i];
      if (count == 0) continue;
      histogram_[*size] = count;
      *counts_total += count;
      mapping_[*size] = i;
      ++*size;
    }
  }
}

WP2Status SymbolWriter::WriteHeader(uint32_t sym, uint32_t cluster,
                                    uint32_t max_nnz,
                                    const uint32_t* const counts, WP2_OPT_LABEL,
                                    ANSEncBase* const enc,
                                    ANSDictionaries* const dicts) {
  // Figure out how many symbols are used and the sizes in bits of the
  // probabilities and difference of consecutive symbols.
  uint32_t size;
  const uint32_t range = symbols_info_.Range(sym, cluster);
  uint32_t counts_total = 0;
  const int32_t min_symbol = symbols_info_.Min(sym, cluster);
  const int32_t max_symbol = symbols_info_.Max(sym, cluster);
  const bool changes_sign = (min_symbol < 0) && (max_symbol > 0);
  ConvertCountsToCachedHistogram(counts, min_symbol, max_symbol, changes_sign,
                                 &size, &counts_total);

  // Deal with the trivial cases.
  ANSDebugPrefix prefix(enc, label);
  const uint32_t nnz_range = std::min(max_nnz, range);
  assert(size <= nnz_range);
  if (size <= 1) {
    if (size == 0) {
      enc->PutRValue(kSymbolCountZero,
                     nnz_range == 1 ? kSymbolCountLast - 1 : kSymbolCountLast,
                     "scount");
      AddTrivial(sym, cluster, /*is_symmetric=*/false, 0);
    } else {
      enc->PutRValue(kSymbolCountOne,
                     nnz_range == 1 ? kSymbolCountLast - 1 : kSymbolCountLast,
                     "scount");
      if (changes_sign) {
        enc->PutRange(mapping_[0], 0, std::max(-min_symbol, max_symbol),
                      "symbol");
      } else {
        enc->PutRange(mapping_[0], 0, max_symbol - min_symbol, "symbol");
      }
      AddTrivial(sym, cluster, changes_sign, mapping_[0]);
    }
    return WP2_STATUS_OK;
  }
  enc->PutRValue(kSymbolCountMoreThanOne, kSymbolCountLast, "scount");

  float cost_best = std::numeric_limits<float>::max();
  Stat::Type type_best =  Stat::Type::kUnknown;
  Quantizer::Config* config_best = nullptr;
  bool is_symmetric_best = false;
  uint32_t prefix_size_best = 0;
  quantizer_.ResetBest();
  // Iterate over whether negative symbols are treated with a symmetric
  // distribution or not. i==0 is for changes_sign as we have already computed
  // counts through ConvertCountsToCachedHistogram. i==1 (if any) is for
  // non-symmetric mapping.
  for (uint32_t i = 0; i < (changes_sign ? 2u : 1u); ++i) {
    if (changes_sign && i == 1) {
      // Re-compute the counts for non symmetric negatives.
      ConvertCountsToCachedHistogram(counts, min_symbol, max_symbol,
                                     /*is_symmetric=*/false, &size,
                                     &counts_total);
    }
    // Figure out the cost of using ranges.
    // Cost of mapping + cost of storing each symbol as a range + cost of size.
    const float cost_range = StoreMapping(mapping_.data(), size, range,
                                          stats_buffer_.data(), nullptr) +
                             counts_total * WP2Log2(size) +
                             WP2Log2(nnz_range - 1);
    if (cost_range < cost_best) {
      config_best = nullptr;
      cost_best = cost_range;
      type_best = Stat::Type::kRange;
      is_symmetric_best = (changes_sign && i == 0);
    }

    // Compute the best cost of Golomb encoding.
    for (uint32_t prefix_size : {1, 2}) {
      if (ComputeCachedGolombHistogramCost(range, max_nnz, size, prefix_size,
                                           cost_best)) {
        prefix_size_best = prefix_size;
        config_best = quantizer_.GetBest();
        cost_best = config_best->cost;
        type_best = Stat::Type::kGolomb;
        is_symmetric_best = (changes_sign && i == 0);
      }
    }

    // If we have too many symbols for dictionaries, do not use a dictionary.
    if (size < ANS_MAX_SYMBOLS) {
      const float cost_extra = WP2Log2(nnz_range - 2);  // size cost
      // Quantize the histogram to get the cost of using a dictionary.
      if (quantizer_.Quantize(histogram_.data(), mapping_.data(), size, range,
                              max_nnz, cost_best, cost_extra, effort_)) {
        config_best = quantizer_.GetBest();
        cost_best = config_best->cost;
        type_best = Stat::Type::kDict;
        is_symmetric_best = (changes_sign && i == 0);
      }
    }
  }
  if (type_best == Stat::Type::kGolomb) {
    // Now that we know if negatives are treates symmetrically and the best
    // Golomb prefix, recompute the cached Golomg histograms.
    // TODO(vrabaud) we could cache the 4 combinations and just re-use the best
    //               one here: more RAM, less CPU.
    ConvertCountsToCachedHistogram(counts, min_symbol, max_symbol,
                                   is_symmetric_best, &size, &counts_total);
    // Re-initialize the cached Golomb parameters.
    FillCachedGolombHistogram(range, size, prefix_size_best);
  }

  // TODO(vrabaud) For very small images, always choose range to save bits on
  // this symbol.
  enc->PutRValue(type_best == Stat::Type::kRange ? 0 :
                 type_best == Stat::Type::kDict ? 1 : 2,
                 3, "type");
  if (changes_sign && enc->PutBool(is_symmetric_best, "is_symmetric")) {
    ConvertCountsToCachedHistogram(counts, min_symbol, max_symbol,
                                   /*is_symmetric=*/true, &size, &counts_total);
  }
  if (type_best == Stat::Type::kGolomb) {
    const uint32_t golomb_prefix_size = config_best->param.golomb_prefix_size;
    const uint32_t golomb_size = config_best->param.golomb_size;
    const Golomb golomb(range - 1, golomb_prefix_size);
    const uint32_t range_golomb = golomb.prefix + 1;
    const uint32_t nnz_range_golomb = std::min(max_nnz, range_golomb);
    enc->PutRange(golomb_prefix_size, 1, 2, "prefix_size");
    enc->PutRange(golomb_size, 2, nnz_range_golomb, "size");
    // Store the dictionary.
    WriteHistogram(*config_best, range_golomb, max_nnz, enc);
    // Create needed dictionaries.
    WP2_CHECK_STATUS(
        AddGolomb(sym, cluster, is_symmetric_best, histogram_golomb_.data(),
                  config_best->histo.counts, mapping_golomb_.data(),
                  golomb_size, golomb_prefix_size, dicts));
  } else if (type_best == Stat::Type::kDict) {
    // The number of symbols is bounded by the number of possible symbols
    // and the size of the image.
    const uint32_t dict_range = std::min(nnz_range, (uint32_t)ANS_MAX_SYMBOLS);
    assert(size <= dict_range);
    enc->PutRange(size, 2, dict_range, "size");
    // Store the dictionary.
    WriteHistogram(*config_best, range, max_nnz, enc);
    // Create needed dictionaries.
    WP2_CHECK_STATUS(AddDict(sym, cluster, is_symmetric_best, histogram_.data(),
                             config_best->histo.counts, mapping_.data(), size,
                             dicts));
  } else {  // kRange
    enc->PutRange(size, 1, nnz_range, "size");
    // Store the mapping.
    StoreMapping(mapping_.data(), size, range, stats_buffer_.data(), enc);
    // Initialize the SymbolWriter.
    AddRange(sym, cluster, is_symmetric_best, mapping_.data(), size, range);
  }
  return WP2_STATUS_OK;
}

void SymbolWriter::GetPotentialUsage(uint32_t sym, uint32_t cluster,
                                     bool is_maybe_used[],
                                     uint32_t size) const {
  const Stat& stat = *GetStats(sym, cluster);
  switch (stat.type) {
    case (Stat::Type::kTrivial):
      // Nothing is used but the one value.
      std::fill(is_maybe_used, is_maybe_used + size, false);
      is_maybe_used[stat.param.trivial_value] = true;
      break;
    case (Stat::Type::kRange):
      // We have no idea of what is used or not.
      std::fill(is_maybe_used, is_maybe_used + size, true);
      break;
    case (Stat::Type::kDict): {
      // Go over the stats to figure out what is used or not.
      if (stat.use_mapping) {
        std::fill(is_maybe_used, is_maybe_used + size, false);
      }
      const Vector_u32& v = stat.extra.dict->Counts();
      if (stat.use_mapping) {
        for (uint32_t i = 0; i < stat.extra.mapping_size; ++i) {
          if (stat.mappings[i] != Stat::kInvalidMapping) {
            is_maybe_used[i] = (v[stat.mappings[i]] > 0);
          }
        }
      } else {
        for (uint32_t i = 0; i < stat.extra.dict->MaxSymbol(); ++i) {
          is_maybe_used[i] = (v[i] > 0);
        }
      }
      break;
    }
    case Stat::Type::kGolomb: {
      // Go over the stats to figure out what is used or not.
      if (stat.use_mapping) {
        std::fill(is_maybe_used, is_maybe_used + size, false);
      }
      const Vector_u32& v = stat.extra.dict->Counts();
      const uint32_t prefix_size = stat.param.golomb.prefix_size;
      if (stat.use_mapping) {
        for (uint32_t i = 0; i < stat.extra.mapping_size; ++i) {
          const auto m = stat.mappings[i];
          if (m != Stat::kInvalidMapping && v[m] > 0) {
            const uint32_t extra_bits_num =
                Golomb::NumExtraBits(i, prefix_size);
            const uint32_t i1 = Golomb::Merge(i, prefix_size, 0);
            assert(i1 <= size);
            const uint32_t i2 =
                Golomb::Merge(i, prefix_size, (1 << extra_bits_num) - 1);
            std::fill(is_maybe_used + i1,
                      is_maybe_used + std::min(i2 + 1, size), true);
          }
        }
      } else {
        for (uint32_t i = 0; i < stat.extra.dict->MaxSymbol(); ++i) {
          if (v[i] == 0) continue;
          const uint32_t extra_bits_num = Golomb::NumExtraBits(i, prefix_size);
          const uint32_t i1 = Golomb::Merge(i, prefix_size, 0);
          assert(i1 <= size);
          const uint32_t i2 =
              Golomb::Merge(i, prefix_size, (1 << extra_bits_num) - 1);
          std::fill(is_maybe_used + i1, is_maybe_used + std::min(i2 + 1, size),
                    true);
        }
      }
      break;
    }
    default:
      assert(false);
  }
}

}  // namespace WP2
