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
// Defines utilities to quantize a distribution to better store it and the
// population it represents.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/utils/quantizer.h"

#include <algorithm>
#include <limits>

#include "src/common/constants.h"
#include "src/dsp/math.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"

namespace WP2 {

WP2Status Quantizer::Allocate(uint32_t range_max) {
  // We need at least kMaxFreqBits elements to store Huffman probabilities.
  histogram_size_max_ = std::max(kMaxFreqBits + 1u, range_max);
  WP2_CHECK_ALLOC_OK(buffer_.resize(kConfigNbr * histogram_size_max_ * 2));
  uint32_t* buffer = buffer_.data();
  for (Config& c : configs_) {
    c.histogram_to_write = buffer;
    buffer += histogram_size_max_;
    c.histo.counts = buffer;
    buffer += histogram_size_max_;
  }

  WP2_CHECK_ALLOC_OK(stats_buffer_.resize(histogram_size_max_));
  return WP2_STATUS_OK;
}

bool Quantizer::Quantize(const uint32_t* const histogram,
                         const uint16_t* const mapping, size_t size_sparse,
                         uint32_t symbol_range, uint32_t max_count,
                         float cost_max, float cost_offset, int effort) {
  assert(effort >= 0 && effort <= 9);
  float cost_best_prev;
  if (config_best_ == nullptr) {
    config_best_ = &configs_[0];
    cost_best_prev = std::numeric_limits<float>::max();
  } else {
    cost_best_prev = config_best_->cost;
  }
  const float cost_to_beat = std::min(cost_best_prev, cost_max) - cost_offset;
  // Temporarily switch the best configuration cost as the new cost to beat.
  config_best_->cost = cost_to_beat;
  QuantizeImpl(histogram, mapping, size_sparse, symbol_range, max_count,
               effort);
  // Return whether we beat the best cost.
  if (config_best_->cost < cost_to_beat) {
    config_best_->cost += cost_offset;
    return true;
  } else {
    config_best_->cost = cost_best_prev;
    return false;
  }
}

void Quantizer::Config::WriteHistogramHeader(uint32_t symbol_range,
                                             OptimizeArrayStorageStat* stats,
                                             uint32_t stats_size,
                                             ANSEncBase* const enc) const {
  // Store the mapping first.
  if (histo.nnz < symbol_range) {
    enc->PutBool(param.is_sparse, "is_sparse");
    if (param.is_sparse) {
      // The diff between two consecutive elements is at most symbol_range-nbr
      // of symbols.
      assert(histo.nnz <= stats_size);
      StoreMapping(histo.mapping, histo.nnz, symbol_range, stats, enc);
    } else {
      enc->PutRange(size_to_write, histo.nnz, symbol_range, "histogram_size");
    }
  } else {
    // We use the full range so no need to do anything.
    assert(param.is_sparse);
  }
  if (param.is_sparse) assert(size_to_write == histo.nnz);
  assert(size_to_write >= histo.nnz);

  enc->PutBool(param.type == 1, "coding_type");
}

void Quantizer::Config::WriteHistogramCounts(uint32_t symbol_range,
                                             uint32_t max_count,
                                             OptimizeArrayStorageStat* stats,
                                             uint32_t stats_size,
                                             ANSEncBase* const enc) const {
  // Save the probabilities.
  const uint32_t max_count_bits =
      std::min(kMaxFreqBits, 1u + (uint32_t)WP2Log2Floor(max_count));
  for (uint32_t i = 0; i < histo.nnz; ++i) {
    assert(histo.counts[i] < (1u << (max_count_bits + 1)));
  }
  assert(param.max_freq_bits >= 1);
  if (param.type == Quantizer::Huffman) {
    enc->PutRange(param.max_freq_bits, 1, 1 + WP2Log2Floor(max_count_bits),
                  "max_freq_bits");
  } else {
    enc->PutRange(param.max_freq_bits, 1, max_count_bits, "max_freq_bits");
  }
  assert(size_to_write <= stats_size);
  if (param.is_sparse) {
    StoreVector(histogram_to_write, size_to_write,
                (1 << param.max_freq_bits) - 1, stats, enc);
  } else {
    StoreVectorNnz(histogram_to_write, size_to_write, histo.nnz,
                   (1 << param.max_freq_bits) - 1, stats, enc);
  }
}

// The quantization implementation uses the Config cache of the class during
// recursion.
void Quantizer::QuantizeImpl(const uint32_t* const histogram,
                             const uint16_t* const mapping, size_t size_sparse,
                             uint32_t symbol_range, uint32_t max_count,
                             int effort) {
  if (size_sparse == 0) {
    // Do not modify pointers if config_best_ is re-used.
    config_best_->param.type = Raw;
    config_best_->param.is_sparse = false;
    config_best_->param.max_freq_bits = 0;
    config_best_->size_to_write = 0;
    config_best_->histo.nnz = 0;
    config_best_->cost = 0.f;
    return;
  }
  // Find an unused config.
  Config* config = &configs_[config_best_ == &configs_[0] ? 1 : 0];

  // Pre-compute some quantization configurations for clarity.
  size_t n_max_freq_bits = 0;
  uint8_t bits[kMaxFreqBits];
  {
    uint32_t max_freq_bits_max =
        1 + WP2Log2Floor(*std::max_element(histogram, histogram + size_sparse));
    max_freq_bits_max = std::min(max_freq_bits_max, kMaxFreqBits);
    const uint32_t max_freq_bits_min =
        (1 * effort + max_freq_bits_max * (9 - effort)) / 9;
    for (uint32_t max_freq_bits = max_freq_bits_min;
         max_freq_bits <= max_freq_bits_max; ++max_freq_bits) {
      bits[n_max_freq_bits++] = max_freq_bits;
    }
  }

  // If we use the whole range, actually force sparse usage. This seems
  // counter-intuitive but it's actually due to the optimization done for
  // sparse data: 1 is subtracted to all values.
  constexpr bool kSparsity[2] = {false, true};
  const bool skip_non_sparse = (symbol_range == size_sparse);
  // Compute the histogram header costs for non_sparse and sparse.
  float cost_header[2];
  uint32_t histogram_to_write_sizes[2];
  config->histo.nnz = size_sparse;
  config->histo.mapping = mapping;
  for (uint32_t i = 0; i < 2; ++i) {
    if (!kSparsity[i] && skip_non_sparse) continue;
    // In the non-sparse case, we store the full probabilities: the vector is
    // not sparse and can contain 0 values.
    config->size_to_write = histogram_to_write_sizes[i] =
        kSparsity[i] ? size_sparse : mapping[size_sparse - 1] + 1;
    config->param.is_sparse = kSparsity[i];
    WP2::ANSEncCounter counter;
    config->WriteHistogramHeader(symbol_range, stats_buffer_.data(),
                                 stats_buffer_.size(), &counter);
    cost_header[i] = counter.GetCost();
  }

  // Go over the quantification configurations and pick the best one.
  for (size_t i = 0; i < n_max_freq_bits; ++i) {
    // No need to continue if any pre-computed cost we have is already bigger.
    if ((skip_non_sparse || cost_header[0] > config_best_->cost) &&
        cost_header[1] > config_best_->cost) {
      break;
    }
    for (const ConfigType type : {Raw, Huffman}) {
      uint8_t max_freq_bits = bits[i];
      // Compute the cost of storing the data with the quantized probabilities.
      float cost_ini;
      uint32_t* const histogram_quantized = config->histo.counts;

      if (type == Huffman) {
        ANSCountsQuantizeHuffman(size_sparse, histogram, histogram_quantized,
                                 max_freq_bits, &cost_ini);
      } else {
        const uint32_t freq_max = (1u << max_freq_bits) - 1;
        ANSCountsQuantize(false, freq_max, size_sparse, histogram,
                          histogram_quantized, &cost_ini);
      }
      // Iterate over sparsity.
      for (uint32_t j = 0; j < 2; ++j) {
        const bool is_sparse = kSparsity[j];
        if (!is_sparse && skip_non_sparse) continue;
        // Reset variables.
        max_freq_bits = bits[i];
        config->cost = cost_ini + cost_header[j];

        // No need to continue if the bit cost without the histogram cost is
        // already bigger.
        if (config->cost >= config_best_->cost) continue;

        // Prepare the vector in which we will write the histogram.
        uint32_t* const histogram_to_write = config->histogram_to_write;
        const size_t histogram_to_write_size = histogram_to_write_sizes[j];
        if (!is_sparse) {
          std::fill(histogram_to_write,
                    histogram_to_write + histogram_to_write_size, 0);
        }

        // Fill the histogram.
        if (type == Huffman) {
          uint32_t max_n_bits = 0;
          for (uint32_t k = 0; k < size_sparse; ++k) {
            // +1 as we encode 0 as a frequency of 0.
            uint32_t n_bits =
                WP2Log2Floor(histogram_quantized[k]) + (is_sparse ? 0 : 1);
            if (is_sparse) {
              histogram_to_write[k] = n_bits;
            } else {
              histogram_to_write[mapping[k]] = n_bits;
            }
            max_n_bits = std::max(max_n_bits, n_bits);
          }
          max_freq_bits = 1 + WP2Log2Floor(max_n_bits);
        } else {
          assert(stats_buffer_.size() >= histogram_to_write_size);
          if (is_sparse) {
            for (uint32_t k = 0; k < size_sparse; ++k) {
              histogram_to_write[k] = histogram_quantized[k] - 1;
            }
          } else {
            for (uint32_t k = 0; k < size_sparse; ++k) {
              histogram_to_write[mapping[k]] = histogram_quantized[k];
            }
          }
        }

        // Fill config with all the relevant local information.
        config->param.type = type;
        config->param.is_sparse = is_sparse;
        config->param.max_freq_bits = max_freq_bits;
        config->size_to_write = histogram_to_write_size;
        config->histo.mapping = mapping;
        config->histo.nnz = size_sparse;
        // Temporarily modify config to point to the right histogram.
        uint32_t* const counts_bak = config->histo.counts;
        config->histo.counts = histogram_quantized;

        // Add the cost of the probabilities.
        WP2::ANSEncCounter counter;
        config->WriteHistogramCounts(symbol_range, max_count,
                                     stats_buffer_.data(), stats_buffer_.size(),
                                     &counter);
        config->histo.counts = counts_bak;
        config->cost += counter.GetCost();

        // Keep the best configuration.
        if (config->cost >= config_best_->cost) {
          continue;
        }
        std::swap(config, config_best_);
        // Make sure the counts remain in the best config.
        if (histogram_quantized != config_best_->histo.counts) {
          std::swap(config->histo.counts, config_best_->histo.counts);
        }
        config_best_->histo.mapping = mapping;
      }
    }
  }
}

}  // namespace WP2
