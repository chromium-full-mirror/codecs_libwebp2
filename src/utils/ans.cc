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
// Assymetric Numeral System coder
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/utils/ans.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <utility>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/utils/data_source.h"
#include "src/utils/utils.h"

// some derived constants:
#define IO_BYTES (IO_BITS / 8)
#define IO_LIMIT (1ull << IO_LIMIT_BITS)

namespace WP2 {

ANSSymbolInfo ANSSymbolInfo::Rescale(const ANSSymbolInfo& max,
                                     uint32_t tab_size) const {
  ANSSymbolInfo info;
  const uint32_t max_offset = max.offset + max.freq;
  const uint32_t offset_next = (offset + freq) * tab_size / max_offset;
  info.offset                =  offset         * tab_size / max_offset;
  // Compute the frequency with the next offset to not leave any gap.
  info.freq = offset_next - info.offset;
  info.symbol = symbol;
  assert(offset < max.offset || offset_next == tab_size);
  return info;
}

//------------------------------------------------------------------------------
// ANSBinSymbol

ANSBinSymbol::ANSBinSymbol(uint32_t p0, uint32_t p1) {
  num_zeros_ = p0;
  num_total_ = p0 + p1;
}

void ANSBinSymbol::UpdateCount(uint32_t bit) {
  assert(num_total_ < kMaxSum);
  num_zeros_ += !bit;
  ++num_total_;
}

//------------------------------------------------------------------------------
// update-table (hardcoded for APROBA_MAX_SYMBOL == 16)

// For mixing distributions while preserve norms, see:
//    https://fgiesen.wordpress.com/2015/02/20/mixing-discrete-probability-distributions/    NOLINT

void ANSAdaptiveSymbol::Update(uint32_t sym) {
  const uint32_t mult = adapt_factor_;
  if (method_ == Method::kAOM && adapt_factor_ > kMinAOMAdaptSpeed) {
    adapt_factor_ -= adapt_factor_ >> kAOMAdaptSpeedShift;
  }
  // TODO(skal): SSE or table-free for plain-C
  assert(sym < max_symbol_);
  const uint16_t* const cdf_var = &cdf_var_[APROBA_MAX_SYMBOL - sym - 1];
  ANSUpdateCDF(max_symbol_, cdf_base_.data(), cdf_var, mult, cumul_.data());
  assert(cumul_[0] == 0);
  assert(cumul_[max_symbol_ - 1] < APROBA_MAX);
  assert(max_symbol_ == APROBA_MAX_SYMBOL || cumul_[max_symbol_] == APROBA_MAX);
}

//------------------------------------------------------------------------------

static void GenerateVariableCDF(uint32_t nnz, uint16_t* const table) {
  for (uint32_t i = 0; i < APROBA_MAX_SYMBOL; ++i) table[i] = 0;
  for (uint32_t i = APROBA_MAX_SYMBOL; i < 2 * APROBA_MAX_SYMBOL - 1; ++i) {
    table[i] = APROBA_MAX - nnz;
  }
}

void ANSAdaptiveSymbol::InitCDF() {
  uint32_t nnz = 0;
  cumul_[max_symbol_] = APROBA_MAX;
  for (uint32_t s = 0; s < max_symbol_; ++s) {
    cdf_base_[s] = nnz;
    const bool is_used = (cumul_[s + 1] > cumul_[s]);
    if (is_used) nnz += 1;
  }
  for (uint32_t s = max_symbol_; s < APROBA_MAX_SYMBOL; ++s) cdf_base_[s] = nnz;

  // TODO(skal): could be static tables, indexed by 'nnz'.
  GenerateVariableCDF(nnz, &cdf_var_[0]);

  assert(max_symbol_ >= 1);
  if (max_symbol_ + 1 <= APROBA_MAX_SYMBOL) {
    // It is useful to fill those values for the later search within cumul_.
    std::fill(&cumul_[max_symbol_ + 1], &cumul_[APROBA_MAX_SYMBOL] + 1,
              APROBA_MAX + 1);
  }
}

void ANSAdaptiveSymbol::InitFromUniform(uint32_t max_symbol) {
  assert(max_symbol > 0 && max_symbol <= APROBA_MAX_SYMBOL);
  max_symbol_ = max_symbol;
  for (uint32_t i = 0; i < max_symbol_; ++i) {
    cumul_[i] = (uint16_t)(i * APROBA_MAX / max_symbol_);
  }
  InitCDF();
}

WP2Status ANSAdaptiveSymbol::InitFromCounts(const uint32_t counts[],
                                            uint32_t max_symbol) {
  assert(max_symbol <= APROBA_MAX_SYMBOL);
  max_symbol_ = 0;
  uint32_t pdf[APROBA_MAX_SYMBOL];

  std::copy(counts, counts + max_symbol, pdf);
  WP2_CHECK_STATUS(ANSNormalizeCounts(pdf, max_symbol, APROBA_MAX));
  counts = pdf;

  cumul_[0] = 0;
  for (uint32_t i = 0; i < max_symbol; ++i) {
    if (counts[i] > 0) max_symbol_ = i + 1;
    cumul_[i + 1] = cumul_[i] + (uint16_t)counts[i];
  }
  assert(cumul_[max_symbol_] == APROBA_MAX);
  InitCDF();
  return WP2_STATUS_OK;
}

WP2Status ANSAdaptiveSymbol::InitFromCDF(const uint16_t* const cdf,
                                         uint32_t max_symbol,
                                         uint32_t max_proba) {
  assert(max_symbol > 0 && max_symbol <= APROBA_MAX_SYMBOL);
  assert(max_proba > 0);
  assert(cdf[0] == 0);
  max_symbol_ = max_symbol;
  uint32_t counts[APROBA_MAX_SYMBOL];
  for (uint32_t i = 0; i < max_symbol - 1; ++i) {
    assert(cdf[i + 1] >= cdf[i] && cdf[i + 1] <= max_proba);
    counts[i] = cdf[i + 1] - cdf[i];
  }
  counts[max_symbol - 1] = max_proba - cdf[max_symbol - 1];
  WP2_CHECK_STATUS(InitFromCounts(counts, max_symbol));
  return WP2_STATUS_OK;
}

void ANSAdaptiveSymbol::CopyFrom(const ANSAdaptiveSymbol& other) {
  max_symbol_ = other.max_symbol_;
  adapt_factor_ = other.adapt_factor_;
  std::copy(std::begin(other.cumul_), std::end(other.cumul_),
            std::begin(cumul_));
  std::copy(std::begin(other.cdf_base_), std::end(other.cdf_base_),
            std::begin(cdf_base_));
  std::copy(std::begin(other.cdf_var_), std::end(other.cdf_var_),
            std::begin(cdf_var_));
}

void ANSAdaptiveSymbol::SetAdaptationSpeed(Method method, uint32_t speed) {
  method_ = method;
  if (method_ == Method::kAOM) {
    assert(speed == kANSAProbaInvalidSpeed);
    adapt_factor_ = kMaxAOMAdaptSpeed;
  } else {
    assert(speed != kANSAProbaInvalidSpeed);
    adapt_factor_ = std::min(speed, 0xffffu);
  }
}

void ANSAdaptiveSymbol::Print(bool use_percents) const {
  printf(" -- adaptive [max_symbol:%d] (speed factor=%u): ",
         max_symbol_, adapt_factor_);
  if (use_percents) {
    for (uint32_t i = 0; i < APROBA_MAX_SYMBOL; ++i) {
      printf("[%5.2f%%]", 100.f * GetProba(i));
    }
  } else {
    for (uint32_t i = 0; i <= APROBA_MAX_SYMBOL; ++i) {
      printf("[0x%.4x]", cumul_[i]);
    }
  }
  printf("\n");
}

void ANSAdaptiveSymbol::PrintProbas(uint32_t norm) const {
  if (norm == 0) norm = APROBA_MAX;
  uint32_t pdf[APROBA_MAX_SYMBOL];
  for (uint32_t i = 0; i < APROBA_MAX_SYMBOL; ++i) {
    pdf[i] = cumul_[i + 1] - cumul_[i];
  }
  if (norm != APROBA_MAX) {
    (void)ANSNormalizeCounts(pdf, APROBA_MAX_SYMBOL, APROBA_MAX);
  }
  printf(" { ");
  for (uint32_t i = 0; i < APROBA_MAX_SYMBOL; ++i) {
    if (i > 0) printf(", ");
    printf("%d", pdf[i]);
  }
  printf(" },\n");
}

//------------------------------------------------------------------------------
// Decoder

#if defined(WP2_BITTRACE)
constexpr double ANSDec::kBitCountAccuracy;
#endif

uint32_t ANSDec::ReadNextWord(uint32_t s) {
  s <<= IO_BITS;
  const uint8_t* data;
  if (data_source_->TryReadNext(IO_BYTES, &data)) {
    s |= (data[1] << 8) | data[0];   // written as endian-neutral code
  } else {
    status_ = WP2_STATUS_NOT_ENOUGH_DATA;
  }
  return s;
}

uint32_t ANSDec::ReadBitInternal(uint32_t num_zeros, uint32_t num_total) {
  if (num_zeros == 0) {
    return 1;
  } else if (num_zeros == num_total) {
    return 0;
  } else {
    const uint32_t p0 = (num_zeros << PROBA_BITS) / num_total;
    const int q0 = PROBA_MAX - p0;
    const uint32_t xfrac = state_ & (PROBA_MAX - 1);
    const uint32_t bit = (xfrac >= p0);
    if (!bit) {
      state_ = p0 * (state_ >> PROBA_BITS) + xfrac;
    } else {
      state_ = q0 * (state_ >> PROBA_BITS) + xfrac - p0;
    }
    if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
    cur_pos_ += PROBA_BITS - WP2Log2(bit ? q0 : p0);
#endif
    return bit;
  }
}

uint32_t ANSDec::ReadABitInternal(ANSBinSymbol* const stats) {
  const uint32_t bit = ReadBitInternal(stats->NumZeros(), stats->NumTotal());
  return stats->Update(bit);
}

uint32_t ANSDec::ReadSymbolInternal(const ANSSymbolInfo codes[],
                                    uint32_t log2_tab_size,
                                    uint32_t max_index) {
  const uint32_t tab_size = (1u << log2_tab_size);
  const uint32_t res = state_ & (tab_size - 1);
  // Deduce the scaled freq/offset.
  ANSSymbolInfo s;
  // A good start is the scaled index. It is almost always the right value,
  // except for very small frequencies, because of rounding errors.
  const uint32_t max_pos = max_index + codes[max_index].freq;
  uint32_t i = res * max_pos / tab_size;
  i -= codes[i].offset;
  uint32_t offset_next;
  while (true) {
    offset_next = (i + codes[i].freq) * tab_size / max_pos;
    if (res < offset_next) break;
    i += codes[i].freq;
  }
  s.offset = i * tab_size / max_pos;
  s.freq = offset_next - s.offset;
  s.symbol = codes[i].symbol;
  assert(res >= s.offset && res < offset_next);
  state_ = s.freq * (state_ >> log2_tab_size) + (res - s.offset);
  if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
  cur_pos_ += log2_tab_size - WP2Log2(s.freq);
#endif
  return s.symbol;
}

uint32_t ANSDec::ReadSymbolInternal(const ANSAdaptiveSymbol& asym,
                                    uint32_t max_index) {
  const uint32_t res = state_ & ((1u << ANS_LOG_TAB_SIZE) - 1);
  // Deduce the scaled freq/offset.
  // TODO(maryla): optimize so we don't do a linear scan from he first symbol.
  ANSSymbolInfo s;
  s.offset = s.freq = s.symbol = 0;
  const ANSSymbolInfo info_max = asym.GetInfo(max_index);
  for (uint32_t i = 0u; i <= max_index; ++i) {
    s = asym.GetInfo(i).Rescale(info_max);
    if (res < (uint32_t)s.offset + s.freq) break;
  }
  assert(res >= s.offset);
  state_ = s.freq * (state_ >> ANS_LOG_TAB_SIZE) + (res - s.offset);
  if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
  cur_pos_ += ANS_LOG_TAB_SIZE - WP2Log2(s.freq);
#endif
  return s.symbol;
}

uint32_t ANSDec::ReadSymbolInternal(const ANSSymbolInfo codes[],
                                    uint32_t log2_tab_size) {
  const uint32_t res = state_ & ((1u << log2_tab_size) - 1);
  const ANSSymbolInfo s = codes[res];
  state_ = s.freq * (state_ >> log2_tab_size) + s.offset;
  if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
  cur_pos_ += log2_tab_size - WP2Log2(s.freq);
#endif
  return s.symbol;
}

uint32_t ANSDec::ReadSymbolInternal(const ANSAdaptiveSymbol& asym) {
  const uint32_t proba = state_ & (APROBA_MAX - 1);
  const ANSSymbolInfo s = asym.GetSymbol(proba);
  state_ = s.freq * (state_ >> APROBA_BITS) + (proba - s.offset);
  if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
  cur_pos_ += ANSAdaptiveSymbol::GetFreqCost(s.freq);
#endif
  return s.symbol;
}

uint32_t ANSDec::ReadUValueInternal(uint32_t bits) {
  assert(bits <= kANSMaxUniformBits);
  const uint32_t value = state_ & ((1u << bits) - 1);
  state_ >>= bits;
  if (state_ < IO_LIMIT) state_ = ReadNextWord(state_);
#if defined(WP2_BITTRACE)
  cur_pos_ += bits;
#endif
  return value;
}

uint32_t ANSDec::ReadRValueInternal(uint32_t range) {
  assert(range <= (1u << kANSMaxRangeBits));
  uint32_t s = state_ % range;
  state_     = state_ / range;
  if (state_ < IO_LIMIT) {
    s = ReadNextWord(s);
    state_ = (state_ << IO_LIMIT_BITS) + s / range;
    s %= range;
  }
#if defined(WP2_BITTRACE)
  cur_pos_ += WP2Log2(range);
#endif
  return s;
}

void ANSDec::Init(DataSource* const data_source) {
  status_ = WP2_STATUS_OK;
  data_source_ = data_source;
  state_ = 0;
  if (data_source == nullptr) {
    status_ = WP2_STATUS_NULL_PARAMETER;
  } else {
    state_ = ReadNextWord(state_);
    state_ = ReadNextWord(state_);
  }
  if (status_ == WP2_STATUS_OK && state_ < IO_LIMIT) {
    status_ = WP2_STATUS_BITSTREAM_ERROR;
  }
}

//------------------------------------------------------------------------------
// Bit-counters

#if defined(WP2_ENC_DEC_MATCH)
uint32_t ANSDec::ReadDebugUValue(uint32_t bits) {
  return ReadUValueInternal(bits);
}
#endif

#if defined(WP2_BITTRACE)

void ANSDec::BitTrace(uint32_t value, LabelStats::Type type,
                      const char label[]) {
  if (!is_padding_counted_) {
    const double incr = kANSPaddingCost;
    counters_["ANS_padding"].bits += incr;
    cur_pos_ += incr;
    last_pos_ += incr;
    counters_["ANS_padding"].num_occurrences += 1;
    is_padding_counted_ = true;
  }
  const std::string key = debug_prefix_ + std::string(label);
  counters_[key].bits += cur_pos_ - last_pos_;
  ++counters_[key].num_occurrences;
  ++counters_[key].histo[value];
  counters_[key].type = type;

  // Add stats for the custom counter (even if no custom key, not to miss any).
  if (merge_custom_until_pop_) {
    assert(!counters_custom_key_.empty());
    counters_custom_[counters_custom_key_].bits += cur_pos_ - last_pos_;
  } else {
    const std::string counters_custom_key =
        counters_custom_key_.empty() ? key
                                     : (counters_custom_key_ + "/" + label);
    counters_custom_[counters_custom_key].bits += cur_pos_ - last_pos_;
    ++counters_custom_[counters_custom_key].num_occurrences;
  }
  last_pos_ = cur_pos_;
}

void ANSDec::PushBitTracesCustomPrefix(const char* const suffix_of_prefix,
                                       bool merge_until_pop) {
  assert(std::strlen(suffix_of_prefix) > 0);
  if (!counters_custom_key_.empty()) counters_custom_key_ += "/";
  counters_custom_key_ += suffix_of_prefix;

  assert(!merge_custom_until_pop_);
  if (merge_until_pop) {
    merge_custom_until_pop_ = true;
    // Increment by one now, merge next ones.
    ++counters_custom_[counters_custom_key_].num_occurrences;
  }
}

void ANSDec::PopBitTracesCustomPrefix(const char* const suffix_of_prefix) {
  if (counters_custom_key_ == suffix_of_prefix) {
    counters_custom_key_.clear();
  } else {
    const uint32_t suffix_length = std::strlen(suffix_of_prefix);
    assert(counters_custom_key_.size() > suffix_length);
    const uint32_t new_length = counters_custom_key_.size() - suffix_length - 1;
    assert(counters_custom_key_[new_length] == '/' &&
           counters_custom_key_.substr(new_length + 1) == suffix_of_prefix);
    counters_custom_key_.resize(new_length);
  }
  merge_custom_until_pop_ = false;
}
#endif

uint32_t ANSDec::GetNumUsedBytes() const {
  return data_source_->GetNumDiscardedBytes() + data_source_->GetNumReadBytes();
}

uint32_t ANSDec::GetMinNumUsedBytesDiff(uint32_t num_used_bytes_before,
                                        uint32_t num_used_bytes_after) {
  // Precision is +-IO_BYTES.
  return SafeSub(num_used_bytes_after, num_used_bytes_before + IO_BYTES);
}

//------------------------------------------------------------------------------

}   // namespace WP2
