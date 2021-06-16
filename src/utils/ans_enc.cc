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
// Assymetric Numeral System - encoding part
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/utils/ans_enc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <utility>

#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/utils/utils.h"
#include "src/wp2/encode.h"   // for Writer

namespace WP2 {

//------------------------------------------------------------------------------
// Encoding part of ANSAdaptiveSymbol

float ANSAdaptiveSymbol::ComputeDistance(const ANSAdaptiveSymbol& from) const {
  assert(from.max_symbol_ <= max_symbol_);
  float score = 0.f;
  if (max_symbol_ > 0) {
    for (uint32_t i = 1; i <= from.max_symbol_; ++i) {
      const int delta0 = cumul_[i] - cumul_[i - 1];
      const int delta1 = from.cumul_[i] - from.cumul_[i - 1];
      score += abs(delta0 - delta1);
    }
    score /= (APROBA_MAX * from.max_symbol_);
  }
  return score;
}

float ANSAdaptiveSymbol::ComputeNonUniformDistance() const {
  uint32_t score = 0;
  if (max_symbol_ > 0) {
    const float average = 1.f * APROBA_MAX / max_symbol_;
    for (uint32_t i = 1; i <= max_symbol_; ++i) {
      const int delta = cumul_[i] - cumul_[i - 1];
      score += (delta - average) * (delta - average);
    }
  }
  return score / (float)(APROBA_MAX * APROBA_MAX);
}

void ANSAdaptiveSymbol::FindBestAdaptationSpeed(
    const Vector_u8& syms, Method* const method, uint32_t* const speed_idx,
    uint32_t* const header_bits) const {
  if (header_bits != nullptr) *header_bits = 0;
  *speed_idx = 0;
  if (syms.empty()) {
    // Choose AOM by default if there are no elements as it does not require
    // storing a speed.
    *method = Method::kAOM;
    return;
  }

  *method = Method::kConstant;
  float best_cost;
  for (uint32_t idx = 0; idx < kNumAdaptationSpeeds; ++idx) {
    const uint32_t speed = kAdaptationSpeeds[idx];
    const float cost = ScoreSequence(Method::kConstant, speed, syms);
    if (idx == 0 || cost < best_cost) {
      best_cost = cost;
      *speed_idx = idx;
    }
    // Add the cost of including the speed.
    best_cost += WP2Log2Fast(kNumAdaptationSpeeds);
  }
  // Try out AOM's method.
  const float cost_aom =
      ScoreSequence(Method::kAOM, kANSAProbaInvalidSpeed, syms);
  if (cost_aom < best_cost) {
    best_cost = cost_aom;
    *method = Method::kAOM;
    *speed_idx = 0;
  }

  if (header_bits != nullptr) {
    uint32_t counts[APROBA_MAX_SYMBOL] = { 0 };
    for (uint32_t s : syms) ++counts[s];
    float cost = 0.;
    for (uint32_t c : counts) cost += WP2SLog2m1(c);
    // fixed probabilities seems better
    if (cost + 2.f < best_cost) {    // +2.f = headroom for inaccuracy
      *header_bits = (uint32_t)(best_cost - cost);
    }
  }
}

float ANSAdaptiveSymbol::ScoreSequence(Method method, uint32_t speed,
                                       const Vector_u8& syms) const {
  ANSAdaptiveSymbol repro;
  repro.InitFromUniform(max_symbol_);
  repro.SetAdaptationSpeed(method, speed);
  float cost = 0.f;
  for (uint32_t s : syms) {
    cost += repro.GetCost(s);
    repro.Update(s);
  }
  return cost;
}

//------------------------------------------------------------------------------
// Tokens.

// Internal token marking. There are 4 variants of token, for binary symbols,
// uniformly drawn values (range or bits) and arbitrary symbols.
typedef uint32_t ANSToken;

enum ANSTokenType {
  ANSTokenTypeBit    = (1u << 29),
  ANSTokenTypeU      = (2u << 29),
  ANSTokenTypeR      = (3u << 29),
  ANSTokenTypeSym    = (4u << 29),
  ANSTokenTypeASym   = (5u << 29),
#if defined(WP2_ENC_DEC_MATCH)
  ANSTokenTypeDebugU = (6u << 29)
#endif
};
inline ANSTokenType ANSTokenGetType(const ANSToken& t) {
  return (ANSTokenType)(t & (7u << 29));
}

// Class generating ANS tokens and extracting info from them.
// A token contains a type stored in the first 3 bits, then a first and a
// second value.
template <ANSTokenType T, uint32_t SecondBits, bool DoCheckSecondBits = true>
class ANSTokenInfo {
 public:
  static constexpr uint32_t kBitsFirst =
      (32 - 3 - SecondBits < IO_BITS) ? (32 - 3 - SecondBits) : IO_BITS;
  static constexpr uint32_t kBitsSecond = SecondBits;
  // We limit values to be at most IO_BITS bits.
  // TODO(skal): check if we can actually use more.
  static_assert(3 + kBitsFirst + kBitsSecond <= 32 && kBitsFirst > 0 &&
                    kBitsFirst <= IO_BITS,
                "wrong token bits");
  static_assert(!DoCheckSecondBits ||
                    (kBitsSecond > 0 && kBitsSecond <= IO_BITS),
                "wrong token bits");
  static uint32_t First(const ANSToken& t) {
    return (t >> kBitsSecond) & ((1u << kBitsFirst) - 1);
  }
  static uint32_t Second(const ANSToken& t) {
    return t & ((1u << kBitsSecond) - 1);
  }
  static ANSToken Init(const uint32_t& f, const uint32_t& s) {
    assert(f < (1u << kBitsFirst));
    assert(s < (1u << kBitsSecond));
    return (ANSToken)(T | (f << kBitsSecond) | s);
  }
};

// We need PROBA_BITS + 1, to be able to store PROBA_MAX.
typedef ANSTokenInfo<ANSTokenTypeBit, PROBA_BITS + 1, false> ANSTokenInfoBit;
// Dictionary and adaptive symbols have the same implementation, just a
// different name.
typedef ANSTokenInfo<ANSTokenTypeSym, APROBA_BITS> ANSTokenInfoSym;
typedef ANSTokenInfo<ANSTokenTypeASym, APROBA_BITS> ANSTokenInfoASym;

// We need enough bits to represent the number of bits with which the symbol
// will be represented. 5 is enough as (1 << 5) bits is enough to represent
// the symbol.
#if defined(WP2_ENC_DEC_MATCH)
typedef ANSTokenInfo<ANSTokenTypeDebugU, 5> ANSTokenInfoDebugU;
#endif
typedef ANSTokenInfo<ANSTokenTypeU, 5> ANSTokenInfoU;
static_assert(kANSMaxUniformBits < (1u << ANSTokenInfoU::kBitsSecond),
              "Too many bits for maximum uniform value.");
static_assert(3 + kANSMaxUniformBits + ANSTokenInfoU::kBitsSecond <= 32,
              "Too many bits for maximum uniform value.");
// kANSMaxRangeBits bits for range.
typedef ANSTokenInfo<ANSTokenTypeR, kANSMaxRangeBits> ANSTokenInfoR;
static_assert(3 + 2 * ANSTokenInfoR::kBitsSecond <= 32u,
              "Too many bits for range.");

// Specializations.
template <>
ANSToken ANSTokenInfoBit::Init(const uint32_t& bit, const uint32_t& proba) {
  assert(proba >= 0 && proba <= PROBA_MAX);
  return (ANSToken)(ANSTokenTypeBit | (!!bit << kBitsSecond) | proba);
}
template <>
ANSToken ANSTokenInfoBit::First(const uint32_t& t) {
  // Optimized computation.
  return !!(t & (1u << kBitsSecond));
}
template <>
ANSToken ANSTokenInfoR::Init(const uint32_t& v, const uint32_t& R) {
  assert(v < (1u << kBitsFirst));
  assert(R > 0 && R <= (1u << kBitsSecond));
  // The range is always > 0 so we can remove 1.
  return (ANSToken)(ANSTokenTypeR | (v << kBitsSecond) | (R - 1));
}
template <>
ANSToken ANSTokenInfoR::Second(const uint32_t& t) {
  return 1 + (t & ((1u << kBitsSecond) - 1));
}

// Adaptive symbol token. We store the offset+freq ANSSymbolInfo.
template <>
ANSToken ANSTokenInfoASym::Init(const uint32_t& s, const uint32_t& f) {
  assert(s < (1u << kBitsFirst));
  assert(f > 0 && f <= (1u << kBitsSecond));
  return (ANSToken)(ANSTokenTypeASym | (s << kBitsSecond) | (f - 1));
}
template <>
ANSToken ANSTokenInfoASym::Second(const uint32_t& f) {
  return 1 + (f & ((1u << kBitsSecond) - 1));
}

// Dictionary symbol: same as adaptive symbol.
template <>
ANSToken ANSTokenInfoSym::Init(const uint32_t& s, const uint32_t& f) {
  assert(s < (1u << kBitsFirst));
  assert(f > 0 && f <= (1u << kBitsSecond));
  return (ANSToken)(ANSTokenTypeSym | (s << kBitsSecond) | (f - 1));
}
template <>
ANSToken ANSTokenInfoSym::Second(const uint32_t& f) {
  return 1 + (f & ((1u << kBitsSecond) - 1));
}

//------------------------------------------------------------------------------
// Dictionary management

WP2Status ANSDictionary::CopyFrom(const ANSDictionary& dict) {
  max_symbol_ = dict.max_symbol_;
  total_ = dict.total_;
  total_quantized_ = dict.total_quantized_;
  WP2_CHECK_ALLOC_OK(counts_.copy_from(dict.counts_));
  WP2_CHECK_ALLOC_OK(quantized_counts_.copy_from(dict.quantized_counts_));
  WP2_CHECK_ALLOC_OK(infos_.copy_from(dict.infos_));
  log2_tab_size_ = dict.log2_tab_size_;
  return WP2_STATUS_OK;
}

WP2Status ANSDictionary::Init(uint32_t max_symbol) {
  // Note: max_symbol can be > ANS_MAX_SYMBOLS at this point
  // (then, ToCodingTable() will not be called on this dictionary)
  WP2_CHECK_OK(max_symbol > 0, WP2_STATUS_INVALID_PARAMETER);

  log2_tab_size_ = ANS_LOG_TAB_SIZE;

  Clear();
  if (max_symbol > 0) {
    WP2_CHECK_ALLOC_OK(infos_.resize(max_symbol));
    WP2_CHECK_ALLOC_OK(counts_.resize(max_symbol));
    std::fill(counts_.begin(), counts_.end(), 0);
    max_symbol_ = max_symbol;
  }
  return WP2_STATUS_OK;
}

void ANSDictionary::ResetCounts() {
  total_ = 0u;
  total_quantized_ = 0u;
  assert(counts_.size() == max_symbol_);
  std::fill(counts_.begin(), counts_.end(), 0);
  quantized_counts_.clear();
}

void ANSDictionary::Clear() {
  counts_.clear();
  quantized_counts_.clear();
  infos_.clear();
  max_symbol_ = 0;
  ResetCounts();
  // don't reset log2_tab_size_.
}

void ANSDictionary::RecordSymbol(uint32_t symbol, uint32_t count) {
  assert(symbol < max_symbol_);
  if (count == 0) return;
  counts_[symbol] += count;
  total_ += count;
}

float ANSDictionary::SymbolCost(uint32_t symbol) const {
  const uint32_t total = IsQuantized() ? total_quantized_ : total_;
  const float c = IsQuantized() ? quantized_counts_[symbol] : counts_[symbol];
  if (total == 0 || c == 0) return 0.f;
  return WP2Log2(total) - WP2Log2(c);
}

float ANSDictionary::SymbolCost(uint32_t symbol, uint32_t max_symbol) const {
  assert(symbol <= max_symbol);
  const uint32_t total_full = IsQuantized() ? total_quantized_ : total_;
  const Vector_u32& counts = IsQuantized() ? quantized_counts_ : counts_;
  const float c = counts[symbol];
  if (total_full == 0 || c == 0) return 0.f;
  const uint32_t total =
      std::accumulate(&counts[0], &counts[max_symbol + 1], 0u);
  assert(total != 0 && total <= total_full);
  return WP2Log2(total) - WP2Log2(c);
}

float ANSDictionary::Cost() const {
  // TODO(vrabaud) cache the following results if the dictionary has not
  // changed.
  float cost = 0.;
  for (uint32_t i = 0; i < max_symbol_; ++i) {
    // TODO(skal): WP2Log2(total_) is computed each time. Optimize.
    cost += counts_[i] * SymbolCost(i);
  }
  return cost;
}

WP2Status ANSCountsToSpreadTable(uint32_t counts[], uint32_t max_symbol,
                                 uint32_t table_size, ANSCodes& codes) {
  if (max_symbol == 0) return WP2_STATUS_OK;
  assert(table_size >= max_symbol);

  WP2_CHECK_STATUS(ANSNormalizeCounts(counts, max_symbol, table_size));
  WP2_CHECK_ALLOC_OK(codes.resize(table_size));
  uint32_t pos = 0;
  for (uint16_t s = 0; s < max_symbol; ++s) {
    const uint16_t freq = counts[s];
    for (uint16_t k = 0; k < freq; ++k, ++pos) {
      codes[pos].freq = freq;
      codes[pos].offset = k;
      codes[pos].symbol = s;
    }
  }
  return WP2_STATUS_OK;
}

static void CountsToInfos(const Vector_u32& counts, ANSCodes& infos) {
  assert(counts.size() <= infos.size());
  for (uint32_t total = 0, s = 0; s < counts.size(); ++s) {
    const uint32_t freq = counts[s];
    infos[s].freq = freq;
    infos[s].offset = total;
    infos[s].symbol = s;  // not really needed
    total += freq;
  }
}

WP2Status ANSDictionary::ToCodingTable() {
  WP2_CHECK_OK(max_symbol_ <= ANS_MAX_SYMBOLS, WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(total_ > 0, WP2_STATUS_INVALID_PARAMETER);

  for (uint32_t i = 0; IsQuantized() && i < max_symbol_; ++i) {
    // If a symbol is used, make sure its quantized count is non-zero.
    assert((counts_[i] == 0) == (quantized_counts_[i] == 0u));
  }
  // total_ has been updated all along with RecordSymbol
  assert(total_ == std::accumulate(counts_.begin(), counts_.end(), 0u));
  if (IsQuantized()) {  // total_quantized_ is updated by SetQuantizedCounts()
    assert(total_quantized_ ==
           std::accumulate(quantized_counts_.begin(),
                           quantized_counts_.end(), 0u));
  }
  const uint32_t tab_size = 1u << log2_tab_size_;
  if (total_quantized_ != tab_size) {
    Vector_u32 counts;
    WP2_CHECK_ALLOC_OK(
        counts.copy_from(IsQuantized() ? quantized_counts_ : counts_));
    WP2_CHECK_STATUS(ANSNormalizeCounts(&counts[0], max_symbol_, tab_size));
    CountsToInfos(counts, infos_);
  } else {
    CountsToInfos(quantized_counts_, infos_);
  }
  return WP2_STATUS_OK;
}

WP2Status ANSDictionary::SetQuantizedCounts(const uint32_t quantized_counts[]) {
  WP2_CHECK_ALLOC_OK(quantized_counts_.resize(max_symbol_));
  total_quantized_ = 0;
  for (uint32_t i = 0; i < max_symbol_; ++i) {
    quantized_counts_[i] = quantized_counts[i];
    total_quantized_ += quantized_counts[i];
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// ANSEncDictionaries

WP2Status ANSDictionaries::CopyFrom(const ANSDictionaries& dicts) {
  DeepClear();
  // Copy the dictionaries.
  WP2_CHECK_ALLOC_OK(reserve(dicts.size()));
  for (const ANSDictionary* dict : dicts) {
    std::unique_ptr<ANSDictionary>
        s(new (WP2Allocable::nothrow) ANSDictionary());
    WP2_CHECK_ALLOC_OK(s != nullptr);
    WP2_CHECK_STATUS(s->CopyFrom(*dict));
    WP2_CHECK_ALLOC_OK(push_back(s.get(), /*resize_if_needed=*/false));
    s.release();
  }
  return WP2_STATUS_OK;
}

ANSDictionary* ANSDictionaries::GetEquivalent(
    const ANSDictionaries& original,
    const ANSDictionary* const dict_to_find) const {
  if (size() != original.size()) return nullptr;
  const auto it = std::find(original.begin(), original.end(), dict_to_find);
  if (it == end()) return nullptr;
  ANSDictionary* const equivalent = at(it - original.begin());
  assert(equivalent->SeemsEquivalentTo(*dict_to_find));
  return equivalent;
}

WP2Status ANSDictionaries::Add(uint32_t max_symbol) {
  std::unique_ptr<ANSDictionary>
        s(new (WP2Allocable::nothrow) ANSDictionary());
  WP2_CHECK_ALLOC_OK(s != nullptr);
  WP2_CHECK_STATUS(s->Init(max_symbol));
  WP2_CHECK_ALLOC_OK(push_back(s.get()));
  s.release();
  return WP2_STATUS_OK;
}

WP2Status ANSDictionaries::AppendAndClear(ANSDictionaries* const in) {
  if (in->empty()) return WP2_STATUS_OK;

  WP2_CHECK_ALLOC_OK(reserve(size() + in->size()));
  for (const auto& d : *in) {
    WP2_CHECK_ALLOC_OK(push_back(d, /*resize_if_needed=*/false));
  }
  // Leave ownership to the current ANSDictionaries.
  in->clear();
  return WP2_STATUS_OK;
}

WP2Status ANSDictionaries::ToCodingTable() {
  for (auto& d : *this) WP2_CHECK_STATUS(d->ToCodingTable());
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// ANSAdaptiveBits

WP2Status ANSAdaptiveBits::CopyFrom(const ANSAdaptiveBits& dicts) {
  clear();
  WP2_CHECK_ALLOC_OK(reserve(dicts.size()));
  for (const ANSBinSymbol& other : dicts) {
    WP2_CHECK_ALLOC_OK(push_back(other, /*resize_if_needed=*/false));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// ANSAdaptiveSymbols

WP2Status ANSAdaptiveSymbols::Add(uint32_t max_symbol,
                                  const uint16_t* const cdf, uint16_t max_proba,
                                  bool resize_if_needed) {
  assert(max_symbol <= APROBA_MAX_SYMBOL);
  ANSAdaptiveSymbol s;
  if (cdf == nullptr) {
    // If no CDF has been set, assume a uniform distribution.
    s.InitFromUniform(max_symbol);
  } else {
    WP2_CHECK_STATUS(s.InitFromCDF(cdf, max_symbol, max_proba));
  }
  s.SetAdaptationSpeed(ANSAdaptiveSymbol::Method::kAOM);
  WP2_CHECK_ALLOC_OK(push_back(s, resize_if_needed));
  return WP2_STATUS_OK;
}

WP2Status ANSAdaptiveSymbols::CopyFrom(const ANSAdaptiveSymbols& dicts) {
  clear();
  WP2_CHECK_ALLOC_OK(reserve(dicts.size()));
  for (const ANSAdaptiveSymbol& other : dicts) {
    WP2_CHECK_ALLOC_OK(push_back(other, /*resize_if_needed=*/false));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// ANSEnc: Encoder

// internal structure for storing tokens in linked pages.
struct ANSTokenPage : public WP2Allocable {
  static constexpr uint32_t kAnsMaxTokenPerPage = 4096u;
  ANSToken tokens[kAnsMaxTokenPerPage];
  const ANSTokenPage* prev;  // previous complete pages
  int nb_tokens;
};

struct BufferPage : public WP2Allocable {
  // adjust the page size so that the total is exactly 4k.
  static const uint32_t kPageSize = 4096u - sizeof(BufferPage*);
  uint8_t data[kPageSize];
  BufferPage* prev;   // list of pages, in reverse order
};
static_assert(sizeof(BufferPage) <= 4096u, "non-adjusted BufferPage size");
static_assert(BufferPage::kPageSize % IO_BYTES == 0,
              "BufferPage::data must be able to hold uint16_t's");

class VectorWriter : public Writer {
 public:
  explicit VectorWriter(Vector_u8& out) : out_(out) {}
  WP2_NO_DISCARD bool Reserve(size_t size) override {
    return out_.reserve(out_.size() + size);
  }
  WP2_NO_DISCARD bool Append(const void* data, size_t data_size) override {
    return out_.append((const uint8_t*)data, data_size);
  }

 protected:
  Vector_u8& out_;
};

//------------------------------------------------------------------------------

ANSEnc::ANSEnc() {
  tokens_ = nullptr;
  pages_ = nullptr;
  status_ = WP2_STATUS_OK;
  Reset();
}

void ANSEnc::Reset() {
  FreeTokenPages();
  // there's no point in preserving the byte pages now that the token are gone
  FreeBufferPages();
}

ANSEnc::~ANSEnc() { Reset(); }

ANSEnc::ANSEnc(ANSEnc&& other) noexcept { TrivialMoveCtor(this, &other); }

WP2Status ANSEnc::Clone(const ANSEnc& e) {
  Reset();
#if defined(WP2_BITTRACE) || defined(WP2_TRACE) || defined(WP2_ENC_DEC_MATCH)
  debug_prefix_ = e.debug_prefix_;
#endif
  status_ = e.status_;
  WP2_CHECK_STATUS(status_);
  state_ = e.state_;
  // Copy the tokens.
  const ANSTokenPage* page_e = e.tokens_;
  while (page_e != nullptr) {
    if (!NewTokenPage()) {
      Reset();
      return WP2_STATUS_OUT_OF_MEMORY;
    }
    memcpy(tokens_->tokens, page_e->tokens, sizeof(page_e->tokens));
    tokens_->nb_tokens = page_e->nb_tokens;
    page_e = page_e->prev;
  }
  nb_tokens_ = e.nb_tokens_;

  // We don't copy the buffer pages. They need a new call to
  // AssembleToBitstream() to be regenerated anyway.
  assert(pages_ == nullptr && num_pages_ == 0);
  assert(buffer_ == nullptr && buffer_pos_ == 0);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// buffer / pages management

bool ANSEnc::NewBufferPage() {
  // too much memory allocated? potential overflow?
  if (num_pages_ >= 0xffffffffu / BufferPage::kPageSize) return false;
  BufferPage* const new_page = new (WP2Allocable::nothrow) BufferPage;
  if (new_page == nullptr) return false;
  new_page->prev = pages_;
  pages_ = new_page;
  buffer_ = new_page->data;
  buffer_pos_ = BufferPage::kPageSize;
  ++num_pages_;
  return true;
}

void ANSEnc::FreeBufferPages() {
  while (pages_ != nullptr) {
    BufferPage* const prev = pages_->prev;
    delete pages_;
    pages_ = prev;
  }
  num_pages_ = 0;
  buffer_ = nullptr;
  buffer_pos_ = 0;
  state_ = ANS_SIGNATURE;
  assert(state_ >= IO_LIMIT);
}

uint32_t ANSEnc::GetBitstreamSize() const {
  // should not overflow, since num_pages_ is capped
  return (num_pages_ * BufferPage::kPageSize - buffer_pos_);
}

WP2Status ANSEnc::WriteBitstreamTo(Writer& writer) const {
  WP2_CHECK_ALLOC_OK(writer.Reserve(GetBitstreamSize()));
  size_t size = BufferPage::kPageSize - buffer_pos_;  // fractional last size
  size_t pos = buffer_pos_;
  for (const BufferPage* p = pages_; p != nullptr; p = p->prev) {
    WP2_CHECK_ALLOC_OK(writer.Append((const void*)(p->data + pos), size));
    size = BufferPage::kPageSize;  // all other subsequent pages are complete
    pos = 0;
  }
  return WP2_STATUS_OK;
}

WP2Status ANSEnc::WriteBitstreamTo(Vector_u8& bits) const {
  bits.clear();
  VectorWriter w(bits);
  return WriteBitstreamTo(w);
}

//------------------------------------------------------------------------------
// emit symbol/bit/uniform-value/etc

uint32_t ANSEnc::EmitWord(uint32_t s) {
  if (buffer_pos_ < IO_BYTES) {
    if (!NewBufferPage()) return (s >> IO_BITS);   // error
  }
  buffer_pos_ -= IO_BYTES;
  // written in endian-neutral way. Compiler will optimize with a memcpy()
  // on little-endian platform.
  buffer_[buffer_pos_ + 0] = (s >> 0) & 0xff;
  buffer_[buffer_pos_ + 1] = (s >> 8) & 0xff;
  return (s >> IO_BITS);
}

void ANSEnc::EmitBit(uint32_t bit, uint32_t p0) {
  const uint32_t q0 = PROBA_MAX - p0;
  if (p0 == 0) {
    assert(bit != 0);
  } else if (q0 == 0) {
    assert(bit == 0);
  } else {
    uint32_t s = state_;
    if ((s >> (IO_LIMIT_BITS + IO_BITS - PROBA_BITS)) >= (bit ? q0 : p0)) {
      s = EmitWord(s);
      assert(s < IO_LIMIT);
    }
    if (bit) {
      s = ((s / q0) << PROBA_BITS) + (s % q0) + p0;
    } else {
      s = ((s / p0) << PROBA_BITS) + (s % p0);
    }
    state_ = s;
  }
  assert(state_ >= IO_LIMIT);
}

void ANSEnc::EmitASymbol(uint32_t offset, uint32_t freq) {
  uint32_t s = state_;
  assert(freq > 0);   // coding a symbol that was never seen ?!
  assert(freq <= APROBA_MAX);
  if ((s >> (IO_LIMIT_BITS + IO_BITS - APROBA_BITS)) >= freq) {
    s = EmitWord(s);
  }
  state_ = ((s / freq) << APROBA_BITS) + (s % freq) + offset;
  assert(state_ >= IO_LIMIT);
}

void ANSEnc::EmitUValue(uint32_t value, int bits) {
  // No need to do anything for 0 bits.
  // We cannot go through the code normally as the standard says: if the value
  // of the right operand is negative or is greater or equal to the number of
  // bits in the promoted left operand, the behavior is undefined.
  if (bits == 0) return;
  uint32_t s = state_;
  if (s >> (IO_LIMIT_BITS + IO_BITS - bits) > 0) s = EmitWord(s);
  state_ = (s << bits) + value;
  assert(state_ >= IO_LIMIT);
}

void ANSEnc::EmitRValue(uint32_t value, uint32_t range) {
  assert(state_ >= IO_LIMIT);
  // Warning! Unlike other symbols, for RValue we call EmitWord() first,
  // and use 64b for temporary state_.
  uint64_t state = state_ * (uint64_t)range + value;
  constexpr uint64_t kLimitHi = 1ull << (IO_LIMIT_BITS + IO_BITS);
  if (state >= kLimitHi) {
    EmitWord((uint32_t)state);   // flush the lower bits of 'state'
    state >>= IO_LIMIT_BITS;     // rotate-in the higher bits
  }
  assert(state >= IO_LIMIT && state < kLimitHi);
  state_ = (uint32_t)state;
}

//------------------------------------------------------------------------------

WP2Status ANSEnc::AssembleToBitstream(bool clear_tokens) {
  FreeBufferPages();
  const ANSTokenPage* page = tokens_;
  while (page != nullptr) {
    int n = page->nb_tokens;
    while (--n >= 0) {
      const ANSToken tok = page->tokens[n];
      switch (ANSTokenGetType(tok)) {
        case (ANSTokenTypeBit): {
          const uint32_t p0 = ANSTokenInfoBit::Second(tok);
          const int bit = ANSTokenInfoBit::First(tok);
          EmitBit(bit, p0);
          break;
        }
        case (ANSTokenTypeU):
          EmitUValue(ANSTokenInfoU::First(tok), ANSTokenInfoU::Second(tok));
          break;
#if defined(WP2_ENC_DEC_MATCH)
        case (ANSTokenTypeDebugU):
          EmitUValue(ANSTokenInfoDebugU::First(tok),
                     ANSTokenInfoDebugU::Second(tok));
          break;
#endif
        case (ANSTokenTypeR):
          EmitRValue(ANSTokenInfoR::First(tok), ANSTokenInfoR::Second(tok));
          break;
        case (ANSTokenTypeASym): {
          const uint32_t offset = ANSTokenInfoASym::First(tok);
          const uint32_t freq = ANSTokenInfoASym::Second(tok);
          EmitASymbol(offset, freq);
          break;
        }
        case (ANSTokenTypeSym): {
          const uint32_t offset = ANSTokenInfoSym::First(tok);
          const uint32_t freq = ANSTokenInfoSym::Second(tok);
          EmitASymbol(offset, freq);
          break;
        }
      }
    }
    page = page->prev;
    if (clear_tokens) {
      delete tokens_;
      tokens_ = (ANSTokenPage*)page;
    }
  }
  state_ = EmitWord(state_);
  state_ = EmitWord(state_);
  assert(!clear_tokens || tokens_ == nullptr);

  // finalize the last (incomplete) buffer
  assert(pages_ != nullptr);
  assert(num_pages_ > 0);
  return status_;
}

//------------------------------------------------------------------------------

// Analyze counts[] and renormalize with Squeaky Wheel fix, so that
// the total is rescaled to be equal to 'sum' exactly.
WP2Status ANSNormalizeCounts(uint32_t counts[], uint32_t size, uint32_t sum) {
  if (size == 0) return WP2_STATUS_OK;   // corner case, but ok.

  WP2_CHECK_OK(size <= 0xffffu, WP2_STATUS_INVALID_PARAMETER);

  uint64_t total = 0;
  uint32_t num_symbols = 0;
  for (uint32_t s = 0; s < size; ++s) {
    total += counts[s];
    num_symbols += (counts[s] > 0);
  }
  WP2_CHECK_OK(total <= 0xffffffffu, WP2_STATUS_INVALID_PARAMETER);  // overflow
  WP2_CHECK_OK(sum >= num_symbols, WP2_STATUS_INVALID_PARAMETER);
  if (total == sum) return WP2_STATUS_OK;   // already ok.
  if (total == 0) {   // force a sane state
    counts[0] = sum;
    return WP2_STATUS_OK;
  }

  constexpr uint32_t kFix = 16u;  // fixed-point precision
  static_assert(kFix <= 16u, "invalid fixed-point precision");
  // we combine error+symbol into a single uint32_t as
  //  key = (error << 16) | symbol;
  // => We can sort by decreasing error, and also resolve tie-breaks.
  VectorNoCtor<uint32_t> keys;
  WP2_CHECK_ALLOC_OK(keys.resize(num_symbols));

  const uint32_t norm = sum << kFix;
  assert((sum >> (32 - kFix)) == 0);
  uint32_t sub_sum = 0;
  for (uint32_t n = 0, s = 0; s < size; ++s) {
    if (counts[s] > 0) {
      const uint64_t v = (uint64_t)norm * counts[s] / total;
      counts[s] = std::max(1u, (uint32_t)(v >> kFix));  // round down
      sub_sum += counts[s];
      const uint32_t error = v & ((1u << kFix) - 1u);  // fractional error, 16b
      keys[n++] = (error << 16) | s;
    }
  }
  if (sub_sum == sum) return WP2_STATUS_OK;   // done!
  if (sub_sum < sum) {
    // Under-shoot -> increase the count of symbols with most errors.
    const uint32_t missing = sum - sub_sum;
    std::nth_element(keys.begin(), &keys[missing], keys.end(),
                     std::greater<uint32_t>());
    for (uint32_t i = 0; i < missing; ++i) ++counts[keys[i] & 0xffffu];
  } else {
    // Over-shoot -> decrease the count of symbols with largest counts.
    const uint32_t missing = sub_sum - sum;
    constexpr uint32_t kCountMax = 0xffffu;
    uint32_t n = 0;
    for (uint32_t s = 0; s < size; ++s) {
      if (counts[s] > 1u) {
        const uint32_t key = std::min(counts[s], kCountMax);
        keys[n++] = (key << 16) | s;
      }
    }
    if (missing < n) {
      std::nth_element(&keys[0], &keys[missing], &keys[n],
                       std::greater<uint32_t>());
    } else {
      std::sort(&keys[0], &keys[n], std::greater<uint32_t>());
    }

    // we loop until we've decreased enough entries. This *will* terminate.
    while (sub_sum > sum) {
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t s = keys[i] & 0xffffu;
        if (counts[s] > 1u) {
          --counts[s];
          if (--sub_sum == sum) break;
        }
      }
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Main call, that will gather stats and code the buffer out.

// Returns the cost of a token that is not a ANSTokenTypeSym. Debug tokens are
// ignored and have no cost.
static inline float GetTokenCost(const ANSToken& tok) {
  switch (ANSTokenGetType(tok)) {
    case ANSTokenTypeBit: {
      const uint32_t p0 = ANSTokenInfoBit::Second(tok);
      if (p0 > 0 && p0 < PROBA_MAX) {
        const bool bit = ANSTokenInfoBit::First(tok);
        return PROBA_BITS - WP2Log2(bit ? PROBA_MAX - p0 : p0);
      }
      break;
    }
    case ANSTokenTypeASym:
      return ANSAdaptiveSymbol::GetFreqCost(ANSTokenInfoASym::Second(tok));
    case ANSTokenTypeSym:
      // This has to be handled by the calling function.
      assert(false);
      break;
    case ANSTokenTypeU:
      return ANSTokenInfoU::Second(tok);
#if defined(WP2_ENC_DEC_MATCH)
    case ANSTokenTypeDebugU:
      return 0.f;
#endif
    case ANSTokenTypeR:
      return WP2Log2(ANSTokenInfoR::Second(tok));
    default:
      assert(false);
  }
  return 0.f;
}

float ANSEnc::GetCost() const {
  float cost = 0.f;
  for (const ANSTokenPage* page = tokens_; page != nullptr; page = page->prev) {
    int n = page->nb_tokens;
    while (--n >= 0) {
      const ANSToken& tok = page->tokens[n];
      switch (ANSTokenGetType(tok)) {
        case ANSTokenTypeSym:
          cost += ANSAdaptiveSymbol::GetFreqCost(ANSTokenInfoSym::Second(tok));
          break;
        default:
          cost += GetTokenCost(tok);
          break;
      }
    }
  }
  return cost;
}

float ANSEnc::GetCost(const ANSDictionaries& dicts) const {
  float cost = 0.f;
  for (const ANSTokenPage* page = tokens_; page != nullptr; page = page->prev) {
    int n = page->nb_tokens;
    while (--n >= 0) {
      const ANSToken& tok = page->tokens[n];
      switch (ANSTokenGetType(tok)) {
        case ANSTokenTypeSym:
          // This cost is handled by the dictionaries below. Skip them for now.
          break;
        default:
          cost += GetTokenCost(tok);
          break;
      }
    }
  }
  // Add the symbol costs.
  for (const auto* d : dicts) cost += d->Cost();
  return cost;
}

float ANSEncBase::GetCostFull() const { return GetCost() + kANSPaddingCost; }

float ANSEncBase::GetCostFull(const ANSDictionaries& dicts) const {
  return GetCost(dicts) + kANSPaddingCost;
}

WP2Status ANSEnc::AppendTokens(const ANSEnc& enc, uint32_t start_token,
                               uint32_t n_tokens) {
  // Figure out the pages in order
  uint32_t n_pages = 0;
  for (const ANSTokenPage* page = enc.tokens_; page != nullptr;
       page = page->prev) {
    ++n_pages;
  }
  VectorNoCtor<const ANSTokenPage*> pages;
  if (!pages.resize(n_pages)) return WP2_STATUS_OUT_OF_MEMORY;
  uint32_t idx = n_pages;
  for (const ANSTokenPage* page = enc.tokens_; page != nullptr;
       page = page->prev) {
    pages[--idx] = page;
  }

  uint32_t i_token = 0;
  for (const auto page : pages) {
    if (i_token + page->nb_tokens <= start_token) {
      // Skip this page completely.
      i_token += page->nb_tokens;
      continue;
    } else if (i_token >= start_token + n_tokens) {
      break;
    }
    const uint32_t skip = (start_token > i_token) ? start_token - i_token : 0;
    const uint32_t end_token =
        std::min((uint32_t)page->nb_tokens, start_token + n_tokens - i_token);
    i_token += skip;
    for (uint32_t n = skip; n < end_token; ++n, ++i_token) {
      const ANSToken tok = page->tokens[n];
      switch (ANSTokenGetType(tok)) {
        case ANSTokenTypeBit:
          PutBitInternal(ANSTokenInfoBit::First(tok),
                         ANSTokenInfoBit::Second(tok));
          break;
        case ANSTokenTypeU:
          PutUValueInternal(ANSTokenInfoU::First(tok),
                            ANSTokenInfoU::Second(tok));
          break;
#if defined(WP2_ENC_DEC_MATCH)
        case ANSTokenTypeDebugU:
          PutDebugUValue(ANSTokenInfoDebugU::First(tok),
                         ANSTokenInfoDebugU::Second(tok));
          break;
#endif
        case ANSTokenTypeR:
          PutRValueInternal(ANSTokenInfoR::First(tok),
                            ANSTokenInfoR::Second(tok));
          break;
        case ANSTokenTypeASym: {
          ANSSymbolInfo info;
          info.offset = ANSTokenInfoASym::First(tok);
          info.freq = ANSTokenInfoASym::Second(tok);
          PutSymbolInternal(info, /*is_adaptive=*/true);
          break;
        }
        case ANSTokenTypeSym: {
          ANSSymbolInfo info;
          info.offset = ANSTokenInfoSym::First(tok);
          info.freq = ANSTokenInfoSym::Second(tok);
          PutSymbolInternal(info, /*is_adaptive=*/false);
          break;
        }
        default:
          assert(0);
          return WP2_STATUS_BAD_WRITE;
      }
    }
  }

  return WP2_STATUS_OK;
}

uint32_t ANSEnc::PutBitInternal(uint32_t bit, uint32_t proba) {
  assert((proba == 0) ? (bit == 1) : (proba != PROBA_MAX || bit == 0));
  EnqueueToken(ANSTokenInfoBit::Init(bit, proba));
  return bit;
}

// Append an adaptive binary symbol 'bit', updating 'stats' afterward.
uint32_t ANSEnc::PutABitInternal(uint32_t bit, ANSBinSymbol* const stats) {
  const uint32_t proba = (stats->NumZeros() << PROBA_BITS) / stats->NumTotal();
  PutBitInternal(bit, proba);
  return stats->Update(bit);
}

// Append a symbol 'symbol' to the message. Probabilities are optimally
// evaluated at the end of the message, when calling ANSEncAssemble().
// 'symbol' should be in range [0, ANS_MAX_SYMBOLS).
// Returns the symbol.
void ANSEnc::PutSymbolInternal(const ANSSymbolInfo& info, bool is_adaptive) {
  if (is_adaptive) {
    EnqueueToken(ANSTokenInfoASym::Init(info.offset, info.freq));
  } else {
    EnqueueToken(ANSTokenInfoSym::Init(info.offset, info.freq));
  }
}

// Append a 'value' uniformly distributed in the range [0..1 << bits).
// Current constants impose 'bits' to fit in 8 bits.
// Returns the value, for convenience.
uint32_t ANSEnc::PutUValueInternal(uint32_t value, uint32_t bits) {
  EnqueueToken(ANSTokenInfoU::Init(value, bits));
  assert(bits <= kANSMaxUniformBits);
  assert(value < (1u << bits));
  return value;
}

// Append a 'value' uniformly distributed in [0..range).
// Current constants impose 'range' to fit in kANSMaxRangeBits bits.
// Returns the value, for convenience.
uint32_t ANSEnc::PutRValueInternal(uint32_t value, uint32_t range) {
  EnqueueToken(ANSTokenInfoR::Init(value, range));
  assert(value < range);
  return value;
}

// Append a debug token whose cost is not counted.
#if defined(WP2_ENC_DEC_MATCH)
uint32_t ANSEnc::PutDebugUValue(uint32_t value, uint32_t bits) {
  EnqueueToken(ANSTokenInfoDebugU::Init(value, bits));
  assert(bits <= kANSMaxUniformBits);
  assert(value < (1u << bits));
  return value;
}
#endif

//------------------------------------------------------------------------------
// Token pages management

bool ANSEnc::NewTokenPage() {
  ANSTokenPage* page = new (WP2Allocable::nothrow) ANSTokenPage;
  if (page == nullptr) {
    status_ = WP2_STATUS_OUT_OF_MEMORY;
    return false;
  }
  page->prev = tokens_;
  page->nb_tokens = 0;
  tokens_ = page;
  return true;
}

void ANSEnc::FreeTokenPages() {
  ANSTokenPage* page = tokens_;
  while (page != nullptr) {
    ANSTokenPage* const prev = (ANSTokenPage*)page->prev;
    delete page;
    page = prev;
  }
  tokens_ = nullptr;
  nb_tokens_ = 0;
}

void ANSEnc::EnqueueToken(ANSToken tok) {
  ANSTokenPage* page = tokens_;
  if (page == nullptr || page->nb_tokens == ANSTokenPage::kAnsMaxTokenPerPage) {
    if (!NewTokenPage()) return;
    page = tokens_;
  }
  page->tokens[page->nb_tokens++] = tok;
  ++nb_tokens_;
}

uint32_t ANSEnc::NumTokens() const { return nb_tokens_; }

//------------------------------------------------------------------------------

void swap(ANSEnc& e1, ANSEnc& e2) {
  using std::swap;
#if defined(WP2_BITTRACE) || defined(WP2_TRACE) || defined(WP2_ENC_DEC_MATCH)
  swap(e1.debug_prefix_, e2.debug_prefix_);
#endif
  swap(e1.status_, e2.status_);
  swap(e1.state_, e2.state_);
  swap(e1.pages_, e2.pages_);
  swap(e1.num_pages_, e2.num_pages_);
  swap(e1.buffer_, e2.buffer_);
  swap(e1.buffer_pos_, e2.buffer_pos_);
  swap(e1.tokens_, e2.tokens_);
  swap(e1.nb_tokens_, e2.nb_tokens_);
}

}   // namespace WP2
