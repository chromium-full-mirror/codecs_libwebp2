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
// Asymmetric Numeral System helper functions.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WP2_UTILS_ANS_UTILS_H_
#define WP2_UTILS_ANS_UTILS_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/dsp/math.h"
#include "src/utils/ans.h"
#include "src/utils/ans_enc.h"
#include "src/wp2/base.h"

namespace WP2 {

//------------------------------------------------------------------------------
// ANS quantizations.

// Scale the counts to fit in max_bits and replace them with the nearest powers
// of 2. 'cost' is set to the cost of encoding the original counts using the
// quantized counts.
void ANSCountsQuantizeHuffman(uint32_t size, const uint32_t* counts,
                              uint32_t* out, uint32_t max_bits, float* cost);

// Quantize the counts to be at most max_freq. Non-null values stay non-null.
// Returns true if the array is modified
// If do_expand is set to true, the counts can increase if the current
// max is inferior to max_freq.
// 'cost' is the cost of storing the data with quantized probabilities. A
// nullptr can be passed.
// The function exists as inplace or with a pre-allocated 'out' buffer.
bool ANSCountsQuantize(bool do_expand, uint32_t max_freq, uint32_t size,
                       uint32_t* counts, float* cost);
bool ANSCountsQuantize(bool do_expand, uint32_t max_freq, uint32_t size,
                       const uint32_t* counts, uint32_t* out, float* cost);

// Outputs the bit cost of a histogram.
float ANSCountsCost(const uint32_t* counts, uint32_t size);

//------------------------------------------------------------------------------
// ANS vector storage.

// Helper function for the function below.
struct OptimizeArrayStorageStat {
  uint32_t count;
  uint8_t n_bits;
};
// define to have a fast, but slightly less efficient OptimizeArrayStorage.
#define OPTIMIZE_ARRAY_STORAGE 1
void OptimizeArrayStorage(OptimizeArrayStorageStat* stats, size_t* size,
                          float overhead);

// Return the index of the highest set bit.
inline uint8_t FindLastSet(uint32_t val) {
  return (val == 0) ? 0 : (1 + WP2Log2Floor(val));
}

// Forward declaration.
template <typename T, bool USE_NNZ>
float StoreVectorImpl(const T* vals, uint32_t size, uint32_t nnz,
                      uint32_t val_upper, OptimizeArrayStorageStat* stats,
                      ANSEncBase* enc);

// This function efficiently stores a vector of values by splitting it into
// groups of numbers of similar bit size.
// val_upper is an upper bound (potentially the max) of the input values.
// The size of the array and this 'val_upper' must be stored independently.
// If no ANS encoder is provided (enc == nullptr), the function can still be
// used to estimate the resulting size in bits (and it is faster than calling
// the function with an ANSEncCounter). It stores the array with tuples (number
// of bits, number of numbers with that bit-size, a list of numbers).
// 'stats' is a pre-allocated buffer of the same size that will be used for
// temporary computation.
template <typename T>
float StoreVector(const T* const vals, uint32_t size, uint32_t val_upper,
                  OptimizeArrayStorageStat* stats, ANSEncBase* enc) {
  return StoreVectorImpl<T, false>(vals, size,
                                   /*nnz=*/std::numeric_limits<uint32_t>::max(),
                                   val_upper, stats, enc);
}

// Same as StoreVector except that we know the number 'nnz' of
// non-zero values as well as the fact that the last element (if size>0) is
// non-zero.
template <typename T>
float StoreVectorNnz(const T* const vals, uint32_t size, uint32_t nnz,
                     uint32_t val_upper, OptimizeArrayStorageStat* stats,
                     ANSEncBase* enc) {
  // Check the number of non-zero elements is correct.
  assert(std::count(vals, vals + size, 0) + nnz == size);
  assert(size == 0 || vals[size - 1] != 0);
  return StoreVectorImpl<T, true>(vals, size, nnz, val_upper, stats, enc);
}

// Implementation of StoreVector and StoreVectorNnz.
// If USE_NNZ is false the number of non-zero elements is not used.
template <typename T, bool USE_NNZ>
float StoreVectorImpl(const T* const vals, uint32_t size, uint32_t nnz,
                      uint32_t val_upper, OptimizeArrayStorageStat* stats,
                      ANSEncBase* enc) {
  if (size == 0 || val_upper == 0 || (USE_NNZ && nnz == 0)) return 0.f;
  WP2MathInit();  // Initialization for WP2Log2.
  const uint8_t n_bits_max = FindLastSet(val_upper);
  const float val_upper_log2 = WP2Log2(val_upper);
  const float val_upper_plus_one_log2 = WP2Log2(val_upper + 1);
  auto add_value = [enc, val_upper, n_bits_max](T val, uint32_t min) {
    if (val_upper <= kANSMaxRange) {
      enc->PutRange(val, min, val_upper, "value");
    } else {
      enc->PutUValue(val, n_bits_max, "value");
    }
  };
  auto add_value_cost = [val_upper, val_upper_log2, val_upper_plus_one_log2,
                         n_bits_max](uint32_t min) -> float {
    if (val_upper <= kANSMaxRange) {
      if (min == 0) return val_upper_plus_one_log2;
      assert(min == 1);
      return val_upper_log2;
    } else {
      return n_bits_max;
    }
  };
  if (USE_NNZ && nnz == 1) {
    // The last value is the non-zero one.
    add_value(vals[size - 1], 1);
    return add_value_cost(/*min=*/1);
  }
  if (size <= 2) {
    // For 2 values, storing as ranges is usually more efficient because of the
    // overhead that describes the data.
    if (USE_NNZ) {
      uint32_t nnz_left = nnz;
      float cost = 0.f;
      for (uint32_t i = 0; i < size && nnz_left > 0; ++i) {
        const uint32_t min = i + nnz_left == size ? 1 : 0;
        if (enc != nullptr) add_value(vals[i], min);
        cost += add_value_cost(min);
        if (vals[i] != 0) --nnz_left;
      }
      return cost;
    } else {
      if (enc != nullptr) {
        for (uint32_t i = 0; i < size; ++i) add_value(vals[i], 0);
      }
      return size * add_value_cost(/*min=*/0);
    }
  }

  // When USE_NNZ, there are two optimizations that can be done.
  // - all the last values that are > 0 can be encoded -1. index_subtract_one is
  // the index when such values start.
  // - if the last value is preceded by a 0, the previous 0s do not need to be
  // encoded and can be skipped starting at index index_skip_zeros.
  uint32_t index_subtract_one, index_skip_zeros;
  if (USE_NNZ) {
    for (index_subtract_one = size - 1; index_subtract_one > 0;
         --index_subtract_one) {
      if (vals[index_subtract_one - 1] == 0) break;
    }
    if (index_subtract_one + 1 == size) {
      // The only way for index_subtract_one to be 0 here is to have
      // a single value, because otherwise either
      // - the last value is not preceded by a 0, so index_subtract_one+1!=size,
      // or
      // - the last value is preceded by a 0, so index_subtract_one>0.
      // And the single value case is handled above as an early exit.
      assert(index_subtract_one > 0);
      for (index_skip_zeros = index_subtract_one - 1; index_skip_zeros > 0;
           --index_skip_zeros) {
        if (vals[index_skip_zeros - 1] != 0) break;
      }
    } else {
      index_skip_zeros = size;
    }
  } else {
    index_subtract_one = index_skip_zeros = size;
  }
  // Stats contains a list of pairs: number of bits, numbers of consecutive
  // values with that bit depth.
  size_t stats_size = 0;
  {
    T val_max = 0, val_min = 0;
    size_t i;
    for (i = 0; i < index_subtract_one && i < index_skip_zeros; ++i) {
      const T val = vals[i];
      assert(val <= val_upper);
      // Check if the highest set bit is the same (faster than calling
      // FindLastSet).
      if (val >= val_min && val < val_max) {
        ++stats[stats_size - 1].count;
      } else {
        stats[stats_size].count = 1;
        stats[stats_size].n_bits = FindLastSet(val);
        val_max = (1u << stats[stats_size].n_bits);
        val_min = (val_max >> 1);
        ++stats_size;
      }
    }
    if (USE_NNZ && i == index_skip_zeros) {
      // In the case where the last non-zero value is by itself, merge the
      // previous 0s in the previous stat.
      stats[stats_size - 1].count += (size - 1 - index_skip_zeros);
      i = size - 1;
    }
    // If only non-zero elements remain, store them with -1.
    for (; i < size; ++i) {
      const T val = vals[i] - 1;
      assert(val <= val_upper);
      // Check if the highest set bit is the same (faster than calling
      // FindLastSet).
      if (val >= val_min && val < val_max) {
        ++stats[stats_size - 1].count;
      } else {
        stats[stats_size].count = 1;
        stats[stats_size].n_bits = FindLastSet(val);
        val_max = (1u << stats[stats_size].n_bits);
        val_min = (val_max >> 1);
        ++stats_size;
      }
    }
  }
  // Merge the pairs to give an optimal cost.
  const float cost0 = WP2Log2Fast(n_bits_max);
  const float cost1 = WP2Log2Fast(n_bits_max + 1);
  OptimizeArrayStorage(stats, &stats_size, cost0 + WP2Log2(size + 1));

  // Figure out the cost of this storage. This is a simplified version of the
  // code below where 'enc' is replaced by 'cost'. Comments have been removed
  // for simplification.
  const T* v = vals;
  float cost = cost1;
  size_t size_left = size;
  uint8_t n_bits_max_in_val = 0;
  for (size_t i = 0; i < stats_size; ++i) {
    if (USE_NNZ && v >= vals + index_skip_zeros) {
      i = stats_size - 1;
      v = vals + index_subtract_one;
      size_left = 1;
    }
    const OptimizeArrayStorageStat& s = stats[i];
    assert(s.count > 0);
    if (i > 0) cost += cost0;
    cost += WP2Log2(size_left);
    const bool use_r_value =
        (s.n_bits == n_bits_max) && (n_bits_max <= kANSMaxRangeBits);
    if (s.n_bits == 0) {
      v += s.count;
    } else if (s.count == 1) {
      if (s.n_bits > 1) {
        if (use_r_value) {
          const uint32_t high_order_bit = 1u << (s.n_bits - 1);
          const bool subtract_one = (USE_NNZ && v >= vals + index_subtract_one);
          cost += WP2Log2(subtract_one ? val_upper - high_order_bit
                                       : val_upper + 1 - high_order_bit);
        } else {
          cost += s.n_bits - 1;
        }
      }
      ++v;
    } else {
      const T* const v_end_count = v + s.count;
      if (USE_NNZ) {
        const T* v_end = std::min(
            {v_end_count, vals + index_subtract_one, vals + index_skip_zeros});
        cost +=
            (v_end - v) * (use_r_value ? val_upper_plus_one_log2 : s.n_bits);
        v = v_end;
        if (v == vals + index_skip_zeros) v = vals + index_subtract_one;
        if (v < v_end_count) {
          cost += (v_end_count - v) * (use_r_value ? val_upper_log2 : s.n_bits);
        }
      } else {
        cost += s.count * (use_r_value ? val_upper_plus_one_log2 : s.n_bits);
      }
      v = v_end_count;
    }
    size_left -= s.count;
    n_bits_max_in_val = std::max(n_bits_max_in_val, s.n_bits);
  }

  // Compute the cost if all values are set to the same bit depth and choose it
  // accordingly. This is a simplified version of the code below where 'enc' is
  // replaced by 'cost', with only one statistic with count == size and n_bits
  // == n_bits_max_in_val.
  float cost_same_depth = cost1;
  {
    // Whether to write the values as RValue or UValue.
    const bool use_r_value =
        (n_bits_max_in_val == n_bits_max) && (n_bits_max <= kANSMaxRangeBits);

    if (USE_NNZ) {
      cost_same_depth +=
          std::min({size, index_subtract_one, index_skip_zeros}) *
          (use_r_value ? val_upper_plus_one_log2 : n_bits_max_in_val);
      cost_same_depth += (size - index_subtract_one) *
                         (use_r_value ? val_upper_log2 : n_bits_max_in_val);
    } else {
      cost_same_depth +=
          size * (use_r_value ? val_upper_plus_one_log2 : n_bits_max_in_val);
    }
  }

  const bool have_same_depth = (cost_same_depth < cost);
  if (have_same_depth) {
    cost = cost_same_depth;
    stats_size = 1;
    stats->count = size;
    stats->n_bits = n_bits_max_in_val;
  }

  // Add the cost for  the same depth bit.
  cost += 1;

  if (enc == nullptr) return cost;

  enc->PutBool(have_same_depth, "has_same_depth");
  // Store the optimized version of the array.
  v = vals;
  uint8_t n_bits_prev = 0;
  size_left = size;
  for (size_t i = 0; i < stats_size; ++i) {
    if (USE_NNZ && v >= vals + index_skip_zeros) {
      // Jump to encoding the last non-zero value.
      i = stats_size - 1;
      v = vals + index_subtract_one;
      size_left = 1;
    }
    const OptimizeArrayStorageStat& s = stats[i];
    // The number of bits is in [0:n_bits_max) for the first set.
    if (v == vals) {
      enc->PutRValue(s.n_bits, n_bits_max + 1, "n_bits");
    } else {
      // If we store n_bits0 for a set, the following n_bits1 will be stored as:
      //   n_bits1       if n_bits1 < n_bits0,
      //   n_bits1 - 1   otherwise.
      if (s.n_bits < n_bits_prev) {
        enc->PutRValue(s.n_bits, n_bits_max, "n_bits");
      } else {
        enc->PutRValue(s.n_bits - 1, n_bits_max, "n_bits");
      }
    }
    n_bits_prev = s.n_bits;
    // Store the number of values.
    if (!have_same_depth) enc->PutRange(s.count, 1, size_left, "count");

    // Whether to write the values as RValue or UValue.
    const bool use_r_value =
        (s.n_bits == n_bits_max) && (n_bits_max <= kANSMaxRangeBits);

    size_left -= s.count;
    if (s.n_bits == 0) {
      v += s.count;
    } else if (s.count == 1) {
      // If we only have one number, we don't need to store the high order
      // bit as it's always 1, otherwise it would fit in (n_bits_ - 1).
      // And if that number is only on one bit to begin with, we don't need
      // to store anything.
      if (s.n_bits > 1) {
        const uint32_t high_order_bit = 1u << (s.n_bits - 1);
        const bool subtract_one = (USE_NNZ && v >= vals + index_subtract_one);
        const uint32_t value_to_code =
            (subtract_one ? *v - 1 : *v) ^ high_order_bit;
        if (use_r_value) {
          enc->PutRValue(value_to_code,
                         subtract_one ? val_upper - high_order_bit
                                      : val_upper + 1 - high_order_bit,
                         "value");
        } else {
          enc->PutUValue(value_to_code, s.n_bits - 1, "value");
        }
      }
      ++v;
    } else {
      // Figure out the end of the segment.
      const T* const v_end_count = v + s.count;
      const T* v_end = v_end_count;
      if (USE_NNZ) {
        if (vals + index_subtract_one < v_end) {
          v_end = vals + index_subtract_one;
        }
        if (vals + index_skip_zeros < v_end) {
          v_end = vals + index_skip_zeros;
        }
      }
      for (; v < v_end; ++v) {
        if (use_r_value) {
          enc->PutRValue(*v, val_upper + 1, "value");
        } else {
          enc->PutUValue(*v, s.n_bits, "value");
        }
      }
      if (USE_NNZ) {
        if (v == vals + index_skip_zeros) {
          // Skip the last zeros.
          v = vals + index_subtract_one;
          assert(v <= v_end_count);
        }
        // The last elements can be encoded minus 1.
        for (; v < v_end_count; ++v) {
          if (use_r_value) {
            enc->PutRValue(*v - 1, val_upper, "value");
          } else {
            enc->PutUValue(*v - 1, s.n_bits, "value");
          }
        }
      }
    }
  }
  assert(size_left == 0);
  return cost;
}

// Forward declaration.
template <typename T, bool USE_NNZ>
void LoadVectorImpl(ANSDec* dec, size_t nnz, uint32_t val_upper, T& v);

// Reciprocal of the function above to read a stored vector.
// 'val_upper' is the *inclusive* upper bound expected for the vector's values.
// The vector must be pre-allocated (with length 'size').
template <typename T>
void LoadVector(ANSDec* dec, uint32_t val_upper, T& v) {
  LoadVectorImpl<T, false>(dec, /*nnz=*/std::numeric_limits<uint32_t>::max(),
                           val_upper, v);
}

// Same as above except we are also given the maximum number of non-zero values
// and we know the last element is non-zero.
template <typename T>
void LoadVectorNnz(ANSDec* dec, uint32_t nnz, uint32_t val_upper, T& v) {
  LoadVectorImpl<T, true>(dec, nnz, val_upper, v);
}

template <typename T, bool USE_NNZ>
void LoadVectorImpl(ANSDec* dec, size_t nnz, uint32_t val_upper, T& v) {
  uint32_t size = v.size();
  if (size == 0) return;
  if (val_upper == 0 || (USE_NNZ && nnz == 0)) {
    std::fill(&v[0], &v[size], 0);
    return;
  }
  const uint8_t n_bits_max = FindLastSet(val_upper);
  auto read_value = [dec, val_upper, n_bits_max](uint32_t min) -> uint32_t {
    if (val_upper <= kANSMaxRange) {
      return static_cast<uint32_t>(dec->ReadRange(min, val_upper, "value"));
    } else {
      return dec->ReadUValue(n_bits_max, "value");
    }
  };

  if (USE_NNZ && nnz == 1) {
    // The last value is the non-zero one.
    std::fill(&v[0], &v[size - 1], 0);
    v[size - 1] = read_value(1);
    return;
  }
  if (size <= 2) {
    uint32_t i = 0;
    if (USE_NNZ) {
      for (; i < size; ++i) {
        v[i] = read_value(i + nnz == size ? 1 : 0);
        if (v[i] > 0) --nnz;
      }
    } else {
      for (; i < size; ++i) v[i] = read_value(0);
    }
    std::fill(&v[i], &v[size], 0);
    return;
  }

  size_t k = 0;
  int n_bits_prev = -1;
  const bool have_same_depth = dec->ReadBool("has_same_depth");

  while (size > 0) {
    // Figure out the number of bits.
    uint32_t n_bits;
    if (n_bits_prev < 0) {
      // The first time, we have to use the full range of bits.
      n_bits = dec->ReadRValue(n_bits_max + 1, "n_bits");
    } else {
      // The following time, we save one unit as we know the value has to be
      // different from before.
      n_bits = dec->ReadRValue(n_bits_max, "n_bits");
      if ((int)n_bits >= n_bits_prev) ++n_bits;
    }
    n_bits_prev = (int)n_bits;

    // Figure out the number of elements with that bit depth.
    if (USE_NNZ && nnz == 1) {
      // We know we only have zeros until the last value.
      for (; size > 1; ++k, --size) v[k] = 0;
    }
    const uint32_t n =
        (have_same_depth) ? size : dec->ReadRange(1, size, "count");
    // Whether to read the values as RValue or UValue.
    const bool use_r_value =
        (n_bits == n_bits_max) && (n_bits_max <= kANSMaxRangeBits);

    if (n_bits == 0) {
      // For a depth of 0 bits, we know it is a 0.
      for (size_t i = 0; i < n; ++i) v[k++] = 0;
      assert(size >= n);
      size -= n;
      // If we have more nnz_left than elements after this segment, the last 0s
      // were actually elements reduced by 1.
      if (USE_NNZ && nnz >= size) {
        for (uint32_t i = k - (nnz - size); i < k; ++i) ++v[i];
        nnz = size;
      }
    } else if (n == 1) {
      // If we only have one number, we didn't store the high order
      // bit as it's always 1, otherwise it would fit in (n_bits_ - 1).
      uint32_t value;
      const bool add_one = (USE_NNZ && size == nnz);
      if (n_bits == 1) {
        // And if that number is only on one bit to begin with, we didn't
        // store anything.
        value = 1u;
      } else {
        value = 1u << (n_bits - 1);
        if (use_r_value) {
          value |= dec->ReadRValue(
              add_one ? val_upper - value : val_upper + 1 - value, "value");
        } else {
          value |= dec->ReadUValue(n_bits - 1, "value");
        }
      }
      v[k] = add_one ? value + 1 : value;
      if (USE_NNZ && v[k] != 0) --nnz;
      ++k;
      assert(size >= 1);
      size -= 1;
    } else {
      for (size_t i = 0; i < n; ++i, ++k, assert(size >= 1), --size) {
        // Use the fact that the last element is non-zero.
        if (USE_NNZ && nnz == 1 && size > 1) {
          v[k] = 0;
          continue;
        }
        const bool add_one = (USE_NNZ && size == nnz);
        if (use_r_value) {
          v[k] = dec->ReadRValue(add_one ? val_upper : val_upper + 1, "value");
        } else {
          v[k] = dec->ReadUValue(n_bits, "value");
        }
        if (add_one) ++v[k];
        if (USE_NNZ && v[k] > 0) --nnz;
      }
    }
  }
  if (USE_NNZ) assert(nnz == 0);
  if (k < v.size()) std::fill(&v[k], v.end(), 0);
}

//------------------------------------------------------------------------------
// ANSEncCounter

// ANSEncCounter behaves like an ANSEnc but does not store anything.
// Its only goal is to keep track of the cost of what its ANSEnc counterpart
// will store. It is useful to test the cost of an ANSEnc without dealing with
// all the token logic.
class ANSEncCounter : public ANSEncBase {
 public:
  ANSEncCounter() : cost_(0.), symbol_cost_(0.) {}
  uint32_t PutBit(uint32_t bit, uint32_t num_zeros, uint32_t num_total,
                  WP2_OPT_LABEL) override;
  uint32_t PutABit(uint32_t bit, ANSBinSymbol* stats, WP2_OPT_LABEL) override;
  uint32_t PutSymbol(uint32_t symbol, const ANSDictionary& dict,
                     WP2_OPT_LABEL) override;
  uint32_t PutSymbol(uint32_t symbol, const ANSAdaptiveSymbol& asym,
                     WP2_OPT_LABEL) override;
  uint32_t PutSymbol(uint32_t symbol, uint32_t max_symbol,
                     const ANSDictionary& dict, WP2_OPT_LABEL) override;
  uint32_t PutSymbol(uint32_t symbol, uint32_t max_symbol,
                     const ANSAdaptiveSymbol& asym, WP2_OPT_LABEL) override;
  uint32_t PutUValue(uint32_t value, uint32_t bits, WP2_OPT_LABEL) override;
  uint32_t PutRValue(uint32_t value, uint32_t range, WP2_OPT_LABEL) override;
  // Adds a fixed cost to the total cost.
  void AddCost(float cost) { cost_ += cost; }
  // Returns the total cost in bits. Cost of symbols is the cost based on
  // dictionary stats at time of writing.
  float GetCost() const override;
  // Returns the total cost. Cost of symbols is taken from the provided
  // dictionaries, and thus the result might be different from GetCost().
  float GetCost(const ANSDictionaries& dicts) const override;
  // The number of tokens is useless for a counter.
  uint32_t NumTokens() const override { return 0; };

  WP2Status GetStatus() const override { return WP2_STATUS_OK; }
  WP2Status Append(const ANSEncCounter& enc);

  // Resets costs to 0.
  void Reset();

 protected:
  float cost_;
  // Cost of symbols at time of writing.
  float symbol_cost_;
};

// ANS encoder that does nothing! Useful when a function expects an ANS encoder
// but we don't actually want to write to the bitstream.
class ANSEncNoop : public ANSEncBase {
 public:
  ANSEncNoop() = default;
  uint32_t PutBit(uint32_t bit, uint32_t num_zeros, uint32_t num_total,
                  WP2_OPT_LABEL) override {
    return bit;
  }
  uint32_t PutABit(uint32_t bit, ANSBinSymbol* const stats,
                   WP2_OPT_LABEL) override {
    return bit;
  }
  uint32_t PutSymbol(uint32_t symbol, const ANSDictionary& dict,
                     WP2_OPT_LABEL) override {
    return symbol;
  }
  uint32_t PutSymbol(uint32_t symbol, const ANSAdaptiveSymbol& asym,
                     WP2_OPT_LABEL) override {
    return symbol;
  }
  uint32_t PutSymbol(uint32_t symbol, uint32_t max_symbol,
                     const ANSDictionary& dict, WP2_OPT_LABEL) override {
    return symbol;
  }
  uint32_t PutSymbol(uint32_t symbol, uint32_t max_symbol,
                     const ANSAdaptiveSymbol& asym, WP2_OPT_LABEL) override {
    return symbol;
  }
  uint32_t PutUValue(uint32_t value, uint32_t bits, WP2_OPT_LABEL) override {
    return value;
  }
  uint32_t PutRValue(uint32_t value, uint32_t range, WP2_OPT_LABEL) override {
    return value;
  }
  float GetCost() const override {
    assert(false);
    return 0;
  }
  float GetCost(const ANSDictionaries& dicts) const override {
    assert(false);
    return 0;
  }
  uint32_t NumTokens() const override {
    assert(false);
    return 0;
  }
  WP2Status GetStatus() const override { return WP2_STATUS_OK; }
};

//------------------------------------------------------------------------------
// Very simple utilities to store/load elements.

// Loads a mapping where all values are in the range [ 0, 'range' ).
// A mapping is such that value i is mapped to 'mapping'[i]. All values are
// strictly increasing and are in the range [ 0, 'range' ).
// 'mapping' must point to a memory buffer of size at least 'size'.
WP2Status LoadMapping(ANSDec* dec, uint32_t size, uint32_t range,
                      uint16_t* mapping);

// Stores a mapping.
// 'stats' is the same as in StoreVector.
// Returns the cost of storing the mapping. It is faster to get this value than
// to call the function with an ANSEncCounter.
float StoreMapping(const uint16_t* mapping, size_t size, uint32_t range,
                   int effort, OptimizeArrayStorageStat* stats,
                   ANSEncBase* enc);

//------------------------------------------------------------------------------

// Writes a value in [min, max] where the size of the interval can be larger
// than kANSMaxRange.
uint32_t PutLargeRange(uint32_t v, uint32_t min, uint32_t max, ANSEncBase* enc,
                       WP2_OPT_LABEL);

// Reads a value written with PutLargeRange.
uint32_t ReadLargeRange(uint32_t min, uint32_t max, ANSDec* dec, WP2_OPT_LABEL);

//------------------------------------------------------------------------------

// Class that adds a debug prefix when it's instantiated and pop it in the
// destructor.
class ANSDebugPrefix {
 public:
#if defined(WP2_BITTRACE) || defined(WP2_TRACE) || defined(WP2_ENC_DEC_MATCH)
  ANSDebugPrefix(ANSEncBase* const enc, const char prefix[])
      : enc_(enc), dec_(nullptr) {
    if (enc_ != nullptr) enc_->AddDebugPrefix(prefix);
  }

  ANSDebugPrefix(ANSDec* const dec, const char prefix[])
      : enc_(nullptr), dec_(dec) {
    if (dec_ != nullptr) dec_->AddDebugPrefix(prefix);
  }

  ~ANSDebugPrefix() {
    if (enc_ != nullptr) {
      enc_->PopDebugPrefix();
    } else if (dec_ != nullptr) {
      dec_->PopDebugPrefix();
    }
  }

 private:
  ANSEncBase* enc_;
  ANSDec* dec_;
#else
  ANSDebugPrefix(ANSEncBase* const enc, const char prefix[]) {
    (void)enc;
    (void)prefix;
  }
  ANSDebugPrefix(ANSDec* const dec, const char prefix[]) {
    (void)dec;
    (void)prefix;
  }
#endif
};

}  // namespace WP2

#endif  // WP2_UTILS_ANS_UTILS_H_
