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
//  Decoding of ANS symbols.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/dec/symbols_dec.h"

#include <algorithm>
#include <numeric>

#include "src/dsp/math.h"
#include "src/utils/ans_utils.h"

namespace WP2 {

// 'nnz' is the number of non-zero values in the histogram.
// 'max_count' is the maximum value of a count in the histogram.
static WP2Status ReadHistogram(uint32_t nnz, uint32_t symbol_range,
                               uint32_t max_count, ANSDec* const dec,
                               ANSCodes& infos) {
  // Read the mappings first.
  Vector_u16 mapping;
  uint32_t histo_size = nnz;
  bool is_sparse;
  if (nnz < symbol_range) {
    is_sparse = dec->ReadBool("is_sparse");
    if (is_sparse) {
      WP2_CHECK_ALLOC_OK(mapping.resize(nnz));
      WP2_CHECK_STATUS(LoadMapping(dec, nnz, symbol_range, mapping.data()));
    } else {
      histo_size = dec->ReadRange(nnz, symbol_range, "histogram_size");
    }
  } else {
    // If we use the whole range, actually force sparse usage. This seems
    // counter-intuitive but it's actually due to the optimization done for
    // sparse data: 1 is subtracted to all counts[] values.
    is_sparse = true;
  }

  Vector_u16 counts;
  WP2_CHECK_ALLOC_OK(counts.resize(histo_size));

  // do_huffman: false = raw probabilities, true = power-of-two probabilities
  const bool do_huffman = dec->ReadBool("use_huffman");
  const uint32_t max_count_bits =
      std::min(kMaxFreqBits, 1u + (uint32_t)WP2Log2Floor(max_count));
  // Read the probabilities.
  uint32_t max_freq_bits;
  if (do_huffman) {  // Huffman
    max_freq_bits =
        dec->ReadRange(1, 1 + WP2Log2Floor(max_count_bits), "max_freq_bits");
  } else {  // Raw
    max_freq_bits = dec->ReadRange(1, max_count_bits, "max_freq_bits");
  }
  if (is_sparse) {
    ReadVector(dec, (1 << max_freq_bits) - 1, counts);
    for (auto& c : counts) c += 1;
  } else {
    ReadVectorNnz(dec, nnz, (1 << max_freq_bits) - 1, counts);
  }

  // Store everything in the info.
  WP2_CHECK_ALLOC_OK(infos.resize(nnz));
  for (uint32_t ind = 0, k = 0; k < counts.size(); ++k) {
    if (counts[k] > 0) {
      assert(ind < nnz);  // We cannot have more non-zeros than needed.
      infos[ind].symbol = mapping.empty() ? k : mapping[k];
      infos[ind].freq = do_huffman ? (1 << (counts[k] - 1)) : counts[k];
      ++ind;
    }
  }
  return dec->GetStatus();
}

//------------------------------------------------------------------------------

WP2Status SymbolReader::Init(const SymbolsInfo& symbols_info,
                             ANSDec* const dec) {
  dec_ = dec;
  // Get an upper bound on the number of symbols that might use a dictionary.
  uint32_t num_auto = 0u;
  for (uint32_t s = 0; s < symbols_info.Size(); ++s) {
    if (symbols_info.Method(s) == SymbolsInfo::StorageMethod::kAuto) {
      num_auto += symbols_info.NumClusters(s);
    }
  }
  WP2_CHECK_ALLOC_OK(counts_.reserve(num_auto));
  WP2_CHECK_ALLOC_OK(codes_.reserve(num_auto));
  WP2_CHECK_STATUS(SymbolIO<StatExtra>::Init(symbols_info));
  return WP2_STATUS_OK;
}

void SymbolReader::AddTrivial(uint32_t sym, uint32_t cluster, bool is_symmetric,
                              int32_t value) {
  Stat* const stat = GetStats(sym, cluster);
  stat->type = Stat::Type::kTrivial;
  stat->is_symmetric = is_symmetric;
  stat->param.trivial_value = value;
}

SymbolReader::Stat* SymbolReader::AddRange(
    uint32_t sym, uint32_t cluster, bool is_symmetric, uint16_t range) {
  Stat* const stat = GetStats(sym, cluster);
  stat->type = Stat::Type::kRange;
  stat->is_symmetric = is_symmetric;
  stat->use_mapping = false;
  stat->range = range;
  assert(range > 0);
  return stat;
}

WP2Status SymbolReader::AddDict(uint32_t sym, uint32_t cluster,
                                bool is_symmetric, ANSCodes* const infos) {
  Stat* const stat = GetStats(sym, cluster);
  const uint32_t num_symbols = infos->size();
  assert(num_symbols > 1 && num_symbols <= ANS_MAX_SYMBOLS);
  stat->type = Stat::Type::kDict;
  stat->is_symmetric = is_symmetric;
  stat->use_mapping = true;

  // Create the mappings.
  WP2_CHECK_ALLOC_OK(counts_.resize(counts_.size() + 1));
  Vector_u32& counts = counts_.back();
  WP2_CHECK_ALLOC_OK(counts.resize(num_symbols + 1));  // note: one extra entry
  counts[0] = 0;
  for (uint32_t k = 0; k < num_symbols; ++k) {
    stat->mappings[k] = (*infos)[k].symbol;
    // verify that symbols are sorted, by syntax design
    assert(k == 0 || stat->mappings[k] > stat->mappings[k - 1]);
    counts[k + 1] = (*infos)[k].freq;
  }
  stat->extra.log2_tab_size = ANS_LOG_TAB_SIZE;
  const uint32_t tab_size = 1u << stat->extra.log2_tab_size;
  WP2_CHECK_STATUS(ANSNormalizeCounts(&counts[1], num_symbols, tab_size));

  // Convert the counts to a spread table.
  WP2_CHECK_ALLOC_OK(codes_.resize(codes_.size() + 1));
  // will allocate codes_.back()
  WP2_CHECK_STATUS(
      ANSCountsToSpreadTable(&counts[1], num_symbols, tab_size, codes_.back()));
  stat->extra.codes = codes_.back().data();
  stat->extra.cumul = counts_.back().data();
  stat->extra.num_symbols = num_symbols;
  // convert counts[] to cumulative
  for (uint32_t k = 1; k <= num_symbols; ++k) counts[k] += counts[k - 1];
  assert(counts[num_symbols] == tab_size);
  return WP2_STATUS_OK;
}

WP2Status SymbolReader::AddPrefixCode(uint32_t sym, uint32_t cluster,
                                      bool is_symmetric, ANSCodes* const infos,
                                      uint32_t prefix_size) {
  WP2_CHECK_STATUS(AddDict(sym, cluster, is_symmetric, infos));
  SetPrefixCodeStat(sym, cluster, prefix_size);

  return WP2_STATUS_OK;
}

int32_t SymbolReader::Read(uint32_t sym, uint32_t cluster, WP2_OPT_LABEL,
                           float* const cost) {
  int32_t value;
  const WP2Status status = ReadInternal(sym, cluster, /*use_max_value=*/false,
      /*max_value=*/0, label, &value, cost);
  (void)status;
  assert(status == WP2_STATUS_OK);
  // TODO(yguyon): Enforce dec_->GetStatus() checking more widely.
  //               Currently some failures are silent.
  return value;
}

WP2Status SymbolReader::ReadWithMax(uint32_t sym, uint32_t cluster,
                                    uint32_t max_value, WP2_OPT_LABEL,
                                    int32_t* const value, float* const cost) {
  const WP2Status status = ReadInternal(sym, cluster, /*use_max_value=*/true,
                                        max_value, label, value, cost);
  WP2_CHECK_STATUS(dec_->GetStatus());
  return status;
}

uint32_t SymbolReader::FindSymbol(const Stat& stat, uint32_t max_value) {
  const uint32_t num_symbols = stat.extra.num_symbols;
  // Stop if the first symbol is already larger than the max_value.
  if (stat.mappings[0] > max_value) return 0;
  // Stop if the last interval is actually the right one.
  // TODO(skal): this case should be made impossible by the syntax.
  if (max_value >= stat.mappings[num_symbols - 1]) return num_symbols - 1;
  const uint32_t idx =
      std::upper_bound(stat.mappings, stat.mappings + num_symbols - 1,
                       max_value) - stat.mappings;
  assert(idx >= 1);
  return idx - 1;
}

//------------------------------------------------------------------------------
// Main call.

WP2Status SymbolReader::ReadInternal(uint32_t sym, uint32_t cluster,
                                     bool use_max_value, uint32_t max_value,
                                     WP2_OPT_LABEL, int32_t* const value,
                                     float* const cost) {
  // Consider reading a symbol as a single read occurrence.
  dec_->PushBitTracesCustomPrefix(label, /*merge_until_pop=*/true);

#if !defined(WP2_BITTRACE)
  (void)cost;
#endif

  const Stat& stat = *GetStats(sym, cluster);
  if (use_max_value && stat.use_mapping) {
    // Make sure at least one value in the mapping is inferior to the max_value.
    // We check the minimal/first one.
    // TODO(vrabaud) could be optimized if at encoding time, every time we hit
    //               the minimal stat.mapping values, they are max_value. We
    //               would then remove them from stat.mapping values and let
    //               max_value be used instead.
    WP2_CHECK_OK(stat.mappings[0] <= max_value, WP2_STATUS_BITSTREAM_ERROR);
  }
  ANSDebugPrefix debug_prefix(dec_, label);
  switch (stat.type) {
    case Stat::Type::kTrivial:
      *value = stat.param.trivial_value;
      if (use_max_value) {
        // TODO(vrabaud) could be optimized if at encoding time we only have one
        //               value and hit max_value. pessimization: 0.50%.
        WP2_CHECK_OK(*value <= (int32_t)max_value, WP2_STATUS_BITSTREAM_ERROR);
      }
      break;
    case Stat::Type::kRange: {
      uint32_t range = stat.range;
      if (use_max_value) {
        if (stat.use_mapping) {
          if (stat.mappings[range - 1] > max_value) {
            // Find the biggest range such that mapping[range-1] <= max_value.
            range = std::upper_bound(stat.mappings, stat.mappings + range,
                                     max_value) - stat.mappings;
            assert(stat.mappings[range] > max_value &&
                   stat.mappings[range - 1] <= max_value);
          }
        } else {
          range = std::min(range, max_value + 1);
        }
      }
      *value = dec_->ReadRValue(range, "range");
#if defined(WP2_BITTRACE)
      if (cost != nullptr) *cost += std::log2(range);
#endif
      if (stat.use_mapping) *value = stat.mappings[*value];
      break;
    }
    case Stat::Type::kDict: {
      assert(stat.use_mapping);
      uint32_t raw_value;
      if (use_max_value &&
          max_value < stat.mappings[stat.extra.num_symbols - 1]) {
        const uint32_t max_idx = FindSymbol(stat, max_value);
        raw_value = dec_->ReadSymbol(stat.extra.codes, stat.extra.log2_tab_size,
                                     stat.extra.cumul[max_idx], "dict");
#if defined(WP2_BITTRACE)
        if (cost != nullptr) {
          const uint32_t freq = stat.extra.freq(raw_value);
          const uint32_t total_freq = stat.extra.cumul[max_idx + 1];
          *cost -= std::log2(1.* freq / total_freq);
        }
#endif
      } else {
        raw_value = dec_->ReadSymbol(&stat.extra.codes[0],
                                     stat.extra.log2_tab_size, "dict");
#if defined(WP2_BITTRACE)
        if (cost != nullptr) {
          const uint32_t freq = stat.extra.freq(raw_value);
          *cost -= std::log2(freq) - stat.extra.log2_tab_size;
        }
#endif
      }
      *value = stat.mappings[raw_value];
      break;
    }
    case Stat::Type::kPrefixCode: {
      const uint32_t prefix_size = stat.param.prefix_code.prefix_size;
      uint32_t prefix;
      bool use_range;
      uint32_t range = 0;
      if (use_max_value) {
        const PrefixCode prefix_code_max(max_value, prefix_size);
        const uint32_t i_inf = FindSymbol(stat, prefix_code_max.prefix);
        const uint32_t raw_prefix =
            dec_->ReadSymbol(&stat.extra.codes[0], stat.extra.log2_tab_size,
                             stat.extra.cumul[i_inf], "prefix_code");
        prefix = std::min(prefix_code_max.prefix,
                          (uint32_t)stat.mappings[raw_prefix]);
        const uint32_t max_prefix =
            std::min(prefix_code_max.prefix, (uint32_t)stat.mappings[i_inf]);
        // Use ranges if we are at the last interval.
        use_range = (prefix == max_prefix);
        if (use_range) {
          // Get (the biggest value with a prefix of 'max_prefix') + 1.
          range = std::min(
              max_value + 1,
              PrefixCode::Merge(max_prefix, prefix_size, 0) +
                  (1 << PrefixCode::NumExtraBits(max_prefix, prefix_size)));
        }
#if defined(WP2_BITTRACE)
        if (cost != nullptr) {
          const uint32_t freq = stat.extra.freq(raw_prefix);
          const uint32_t total_freq = stat.extra.cumul[i_inf + 1];
          *cost -= std::log2(1. * freq / total_freq);
        }
#endif
      } else {
        const uint32_t raw_prefix = dec_->ReadSymbol(
            &stat.extra.codes[0], stat.extra.log2_tab_size, "prefix_code");
        prefix = stat.mappings[raw_prefix];
        // Use ranges if we are at the last interval.
        use_range = (prefix + 1 == stat.param.prefix_code.range);
        if (use_range) range = stat.range;
#if defined(WP2_BITTRACE)
        if (cost != nullptr) {
          const uint32_t freq = stat.extra.freq(raw_prefix);
          *cost -= std::log2(freq) - stat.extra.log2_tab_size;
        }
#endif
      }
      const uint32_t extra_bits = PrefixCode::NumExtraBits(prefix, prefix_size);
      uint32_t extra_bits_value;
      if (extra_bits == 0) {
        extra_bits_value = 0;
      } else {
        if (use_range) {
          assert(PrefixCode::Merge(prefix, prefix_size, 0) < range);
          range -= PrefixCode::Merge(prefix, prefix_size, 0);
          extra_bits_value = dec_->ReadRValue(range, "extra_bits_value");
#if defined(WP2_BITTRACE)
          if (cost != nullptr) *cost += std::log2(range);
#endif
        } else {
          extra_bits_value = dec_->ReadUValue(extra_bits, "extra_bits_value");
#if defined(WP2_BITTRACE)
          if (cost != nullptr) *cost += extra_bits;
#endif
        }
      }
      *value = PrefixCode::Merge(prefix, prefix_size, extra_bits_value);
      break;
    }
    case Stat::Type::kAdaptiveBit: {
      if (use_max_value && max_value == 0) {
        *value = 0;
      } else {
        *value =
            dec_->ReadBit(a_bits_[stat.param.a_bit_index].NumZeros(),
                          a_bits_[stat.param.a_bit_index].NumTotal(), "a_bit");
#if defined(WP2_BITTRACE)
        if (cost != nullptr) {
          *cost += a_bits_[stat.param.a_bit_index].GetCost(*value);
        }
#endif
      }
      a_bits_[stat.param.a_bit_index].Update(*value);
      break;
    }
    case Stat::Type::kAdaptiveSymbol: {
      ANSAdaptiveSymbol& asym = a_symbols_[stat.param.a_symbol_index];
      if (use_max_value) {
        *value = dec_->ReadSymbol(asym, max_value, "a_symbol");
#if defined(WP2_BITTRACE)
        if (cost != nullptr) *cost += asym.GetCost(*value, max_value);
#endif
      } else {
        *value = dec_->ReadSymbol(asym, "a_symbol");
#if defined(WP2_BITTRACE)
        if (cost != nullptr) *cost += asym.GetCost(*value);
#endif
      }
      asym.Update(*value);
      break;
    }
    default:
      // Did you forget to call ReadHeader for this symbol?
      assert(false);
      *value = 0;
      dec_->PopBitTracesCustomPrefix(label);
      return WP2_STATUS_BITSTREAM_ERROR;
  }
  if (stat.is_symmetric) {
    if (*value != 0) {
      if (dec_->ReadBool("is_negative")) *value = -(*value);
#if defined(WP2_BITTRACE)
      if (cost != nullptr) *cost += 1.;
#endif
    }
  } else {
    *value += symbols_info_.Min(sym, cluster);
  }
  if (use_max_value) assert((uint32_t)std::abs(*value) <= max_value);
  dec_->PopBitTracesCustomPrefix(label);
  return WP2_STATUS_OK;
}

WP2Status SymbolReader::ReadHeader(uint32_t sym, uint32_t max_nnz,
                                   WP2_OPT_LABEL) {
  for (uint32_t cluster = 0; cluster < symbols_info_.NumClusters(sym);
       ++cluster) {
    WP2_CHECK_STATUS(ReadHeader(sym, cluster, max_nnz, label));
  }
  return dec_->GetStatus();
}

WP2Status SymbolReader::ReadHeader(uint32_t sym, uint32_t cluster,
                                   uint32_t max_nnz, WP2_OPT_LABEL) {
  if (symbols_info_.Method(sym) == SymbolsInfo::StorageMethod::kUnused) {
    return WP2_STATUS_OK;
  }
  const uint32_t range = symbols_info_.Range(sym, cluster);
  const int16_t min_symbol = symbols_info_.Min(sym, cluster);
  const int16_t max_symbol = symbols_info_.Max(sym, cluster);
  if (range < 2) {
    assert(range == 1);
    AddTrivial(sym, cluster, /*is_symmetric=*/false, 0);
    return WP2_STATUS_OK;
  }

  ANSDebugPrefix prefix(dec_, label);

  switch (symbols_info_.Method(sym)) {
    case SymbolsInfo::StorageMethod::kAuto:
      // This main case is handled after.
      break;
    case SymbolsInfo::StorageMethod::kAdaptiveBit:
      WP2_CHECK_STATUS(AddAdaptiveBit(
          sym, cluster, symbols_info_.StartingProbaP0(sym, cluster),
          symbols_info_.StartingProbaP1(sym, cluster)));
      return dec_->GetStatus();
    case SymbolsInfo::StorageMethod::kAdaptiveSym: {
      WP2_CHECK_STATUS(AddAdaptiveSymbol(sym, cluster,
                                         ANSAdaptiveSymbol::Method::kAOM,
                                         kANSAProbaInvalidSpeed));
      return dec_->GetStatus();
    }
    case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed: {
      if (max_nnz <= 1) {
        // Adaptation method/speed don't matter if we only use this symbol once.
        return AddAdaptiveSymbol(sym, cluster,
                                 ANSAdaptiveSymbol::Method::kConstant, 0);
      }
      const auto method = (ANSAdaptiveSymbol::Method)dec_->ReadRValue(
          (uint32_t)ANSAdaptiveSymbol::Method::kNum, "adaptation_method");
      uint32_t speed;
      if (method == ANSAdaptiveSymbol::Method::kConstant) {
        speed = kAdaptationSpeeds[dec_->ReadRValue(kNumAdaptationSpeeds,
                                                   "adaptation_speed")];
      } else {
        speed = kANSAProbaInvalidSpeed;
      }
      WP2_CHECK_STATUS(AddAdaptiveSymbol(sym, cluster, method, speed));
      return dec_->GetStatus();
    }
    case SymbolsInfo::StorageMethod::kUnused:
      break;
  }

  const uint32_t nnz_range = std::min(max_nnz, range);
  const SymbolCount c = (SymbolCount)dec_->ReadRValue(
      (nnz_range == 1) ? kSymbolCountLast - 1 : kSymbolCountLast, "scount");

  // Deal with trivial cases.
  if (c == kSymbolCountZero) {
    AddTrivial(sym, cluster, /*is_symmetric=*/false, 0);
    return dec_->GetStatus();
  }
  if (c == kSymbolCountOne) {
    const bool changes_sign = (min_symbol < 0) && (max_symbol > 0);
    int32_t value;
    if (changes_sign) {
      value = dec_->ReadRange(0, std::max((int16_t)-min_symbol, max_symbol),
                              "symbol");
    } else {
      value = dec_->ReadRange(0, max_symbol - min_symbol, "symbol");
    }
    AddTrivial(sym, cluster, changes_sign, value);
    return dec_->GetStatus();
  }

  const uint32_t type = dec_->ReadRValue(3, "type");
  const bool is_symmetric = (min_symbol < 0 && max_symbol > 0)
                                ? dec_->ReadBool("is_symmetric")
                                : false;
  if (type == 2) {  // kPrefixCode
    const uint32_t prefix_code_prefix_size =
        dec_->ReadRange(0, 1, "prefix_size");
    const PrefixCode prefix_code(range - 1, prefix_code_prefix_size);
    const uint32_t range_prefix_code = prefix_code.prefix + 1;
    const uint32_t nnz =
        dec_->ReadRange(2, std::min(max_nnz, range_prefix_code), "size");
    WP2_CHECK_STATUS(
        ReadHistogram(nnz, range_prefix_code, max_nnz, dec_, infos_));
    WP2_CHECK_STATUS(AddPrefixCode(sym, cluster, is_symmetric, &infos_,
                                   prefix_code_prefix_size));
  } else if (type == 1) {  // kDict
    // a kDict can't use more than ANS_MAX_SYMBOLS
    const uint32_t dict_range = std::min(nnz_range, (uint32_t)ANS_MAX_SYMBOLS);
    const uint32_t nnz = dec_->ReadRange(2, dict_range, "size");
    WP2_CHECK_STATUS(ReadHistogram(nnz, range, max_nnz, dec_, infos_));
    WP2_CHECK_STATUS(AddDict(sym, cluster, is_symmetric, &infos_));
  } else {  // kRange
    const uint32_t nnz = dec_->ReadRange(1, nnz_range, "size");
    Stat* const stat = AddRange(sym, cluster, is_symmetric, nnz);
    stat->use_mapping = true;
    WP2_CHECK_STATUS(LoadMapping(dec_, nnz, range, stat->mappings));
  }
  return dec_->GetStatus();
}

void SymbolReader::GetPotentialUsage(uint32_t sym, uint32_t cluster,
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
    case (Stat::Type::kDict):
      // Go over the stats to figure out what is used or not.
      if (stat.use_mapping) {
        std::fill(is_maybe_used, is_maybe_used + size, false);
      }
      for (uint32_t k = 0; k < stat.extra.num_symbols; ++k) {
        is_maybe_used[stat.use_mapping ? stat.mappings[k] : k] = true;
      }
      break;
    case Stat::Type::kPrefixCode: {
      // Go over the stats to figure out what is used or not.
      if (stat.use_mapping) {
        std::fill(is_maybe_used, is_maybe_used + size, false);
      }
      const uint32_t prefix_size = stat.param.prefix_code.prefix_size;
      for (uint32_t k = 0; k < stat.extra.num_symbols; ++k) {
        const uint32_t m = stat.use_mapping ? stat.mappings[k] : k;
        const uint32_t extra_bits_num =
            PrefixCode::NumExtraBits(m, prefix_size);
        const uint32_t m1 = PrefixCode::Merge(m, prefix_size, 0);
        const uint32_t m2 =
            PrefixCode::Merge(m, prefix_size, (1 << extra_bits_num) - 1);
        std::fill(is_maybe_used + m1, is_maybe_used + std::min(m2 + 1, size),
                  true);
      }
      break;
    }
    default:
      assert(false);
      break;
  }
}

}  // namespace WP2
