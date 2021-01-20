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

#include "src/common/constants.h"
#include "src/dsp/math.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"

namespace WP2 {

WP2Status Quantizer::Allocate(uint32_t range_max) {
  // We need at least kMaxFreqBits elements to store Huffman probabilities.
  histogram_size_max_ = std::max(kMaxFreqBits + 1u, range_max);
  const size_t histogram_size = 4 * sizeof(uint32_t) + sizeof(uint16_t);
  WP2_CHECK_STATUS(buffer_.Resize(
      kConfigNbr * histogram_size_max_ * histogram_size, /*keep_bytes=*/false));

  for (size_t i = 0; i < kConfigNbr; ++i) {
    // Make sure the recursion buffers are properly allocated.
    WP2_CHECK_ALLOC_OK(histogram_sub_[i].resize(kMaxFreqBits + 1));
    WP2_CHECK_ALLOC_OK(mapping_sub_[i].resize(kMaxFreqBits + 1));
  }

  WP2_CHECK_ALLOC_OK(stats_buffer_.resize(histogram_size_max_));
  return WP2_STATUS_OK;
}

void Quantizer::Quantize(const uint32_t* const histogram,
                         const uint16_t* const mapping, size_t size_sparse,
                         uint32_t symbol_range, uint32_t max_count, int effort,
                         Config** const config_best) {
  assert(effort >= 0 && effort <= 9);
  *config_best = &configs_[0];
  configs_used_[0] = true;
  for (size_t i = 1; i < kConfigNbr; ++i) configs_used_[i] = false;

  uint32_t* buffer = (uint32_t*)buffer_.bytes;
  for (size_t i = 0; i < kConfigNbr; ++i) {
    // Make sure the recursion buffers are properly allocated.
    histogram_quantized_[i] = buffer;
    buffer += histogram_size_max_;
    histogram_to_write_[i] = buffer;
    buffer += histogram_size_max_;
    configs_[i].histogram_to_write = buffer;
    buffer += histogram_size_max_;
    configs_[i].histo.counts = buffer;
    buffer += histogram_size_max_;
    configs_[i].histo.mapping = (uint16_t*)buffer;
    buffer += histogram_size_max_ * sizeof(uint16_t) / sizeof(uint32_t);
  }
  QuantizeImpl(histogram, mapping, size_sparse, symbol_range,
               1 + WP2Log2Floor(max_count), effort, 0);
}

// The quantization implementation uses the Config cache of the class during
// recursion.
void Quantizer::QuantizeImpl(const uint32_t* const histogram,
                             const uint16_t* const mapping, size_t size_sparse,
                             uint32_t symbol_range, uint32_t n_pixels_bits,
                             int effort, size_t config_index, float cost_max) {
  assert(config_index < kConfigNbr);
  Config* const config_best = &configs_[config_index];
  if (size_sparse == 0) {
    config_best->param.type = Raw;
    config_best->param.is_sparse = false;
    config_best->param.max_freq_bits = 0;
    config_best->histogram_to_write = nullptr;
    config_best->size_to_write = 0;
    config_best->histo.counts = nullptr;
    config_best->histo.mapping = nullptr;
    config_best->histo.nnz = 0;
    config_best->cost = 0;
    return;
  }

  // We simplify the mapping to be stored as the difference between
  // consecutive terms.
  config_best->histo.nnz = size_sparse;
  std::copy(mapping, mapping + size_sparse, config_best->histo.mapping);

  // Pre-compute some quantization configurations for clarity.
  uint8_t* const bits = bits_[config_index];
  size_t n_max_freq_bits = 0;
  {
    uint32_t max_freq_bits_max =
        1 + WP2Log2Floor(*std::max_element(histogram, histogram + size_sparse));
    max_freq_bits_max = std::min(max_freq_bits_max, kMaxFreqBits);
    const uint32_t max_freq_bits_min =
        (1 * effort + max_freq_bits_max * (9 - effort)) / 9;
    for (uint32_t max_freq_bits = max_freq_bits_max;
         max_freq_bits >= max_freq_bits_min; --max_freq_bits) {
      bits[n_max_freq_bits++] = max_freq_bits;
    }
    // Start by doing some strong quantization to prune out the ones that would
    // barely change the bits cost.
    std::swap(bits[0], bits[(n_max_freq_bits - 1) / 2]);
    if (n_max_freq_bits > 1) {
      std::swap(bits[1], bits[(n_max_freq_bits - 1) / 4]);
    }
    if (n_max_freq_bits > 2) {
      std::swap(bits[2], bits[(n_max_freq_bits - 1) * 3 / 4]);
    }
  }

  uint32_t* histogram_quantized = histogram_quantized_[config_index];
  uint32_t* histogram_to_write = histogram_to_write_[config_index];

  // Compute some constant costs.
  float cost_probability_size, cost_mapping;
  if (symbol_range == size_sparse) {
    // If we use the whole range, actually force sparse usage. This seems
    // counter-intuitive but it's actually due to the optimization done for
    // sparse data: 1 is subtracted to all values.
    cost_probability_size = std::numeric_limits<float>::max();
    cost_mapping = 0.f;
  } else {
    cost_probability_size = WP2Log2(symbol_range + 1 - size_sparse);
    // Small speed improvements: if we use the whole range, no need to even
    // consider a potential mapping.
    cost_mapping = StoreMapping(config_best->histo.mapping, size_sparse,
                                symbol_range, stats_buffer_.data(), nullptr);
  }
  // Cost for the sparse bit and the histogram type.
  const float cost_sparse_bit_and_type = 1 + std::log2(3);
  n_pixels_bits = std::min(n_pixels_bits, kMaxFreqBits);

  // Go over the quantification configurations and pick the best one.
  config_best->cost = cost_max;
  for (size_t i = 0; i < n_max_freq_bits; ++i) {
    // No need to continue if any pre-computed cost we have is already bigger.
    if (cost_sparse_bit_and_type + cost_mapping > config_best->cost &&
        cost_sparse_bit_and_type + cost_probability_size > config_best->cost) {
      break;
    }
    for (const ConfigType type : {Raw, Huffman}) {
      uint8_t max_freq_bits = bits[i];
      // Compute the cost of storing the data with the quantized probabilities.
      float cost_ini;

      if (type == Huffman) {
        ANSCountsQuantizeHuffman(size_sparse,
                                 histogram,
                                 histogram_quantized,
                                 max_freq_bits,
                                 &cost_ini);
      } else {
        const uint32_t freq_max = (1u << max_freq_bits) - 1;
        ANSCountsQuantize(false, freq_max, size_sparse, histogram,
                          histogram_quantized, &cost_ini);
      }

      // Add the cost for the sparse bit and the histogram type.
      cost_ini += cost_sparse_bit_and_type;

      for (const bool is_sparse : {false, true}) {
        // Reset variables.
        max_freq_bits = bits[i];
        float cost = cost_ini;
        // Add the cost for the mapping or the probability size.
        cost += is_sparse ? cost_mapping : cost_probability_size;

        // No need to continue if the bit cost without the histogram cost is
        // already bigger.
        if (cost >= config_best->cost) continue;

        // Prepare the vector in which we will write the histogram.
        size_t histogram_to_write_size;
        if (is_sparse) {
          // Store the mapping and the probabilities: it's basically a sparse
          // vector.
          histogram_to_write_size = size_sparse;
        } else {
          // Store the full probabilities: the vector is not sparse and can
          // contain 0 values.
          histogram_to_write_size = mapping[size_sparse - 1] + 1;
          std::fill(histogram_to_write,
                    histogram_to_write + histogram_to_write_size, 0);
        }

        // Fill the histogram.
        float cost_histo;
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
          if (is_sparse) {
            cost_histo = StoreVector(
                histogram_to_write, histogram_to_write_size,
                (1 << max_freq_bits) - 1, stats_buffer_.data(), nullptr);
          } else {
            cost_histo = StoreVectorNnz(
                histogram_to_write, histogram_to_write_size, size_sparse,
                (1 << max_freq_bits) - 1, stats_buffer_.data(), nullptr);
          }
          cost_histo += WP2Log2Fast(1 + WP2Log2Floor(n_pixels_bits));
        } else {
          assert(stats_buffer_.size() >= histogram_to_write_size);
          if (is_sparse) {
            for (uint32_t k = 0; k < size_sparse; ++k) {
              histogram_to_write[k] = histogram_quantized[k] - 1;
            }
            cost_histo = StoreVector(
                histogram_to_write, histogram_to_write_size,
                (1 << max_freq_bits) - 1, stats_buffer_.data(), nullptr);
          } else {
            for (uint32_t k = 0; k < size_sparse; ++k) {
              histogram_to_write[mapping[k]] = histogram_quantized[k];
            }
            cost_histo = StoreVectorNnz(
                histogram_to_write, histogram_to_write_size, size_sparse,
                (1 << max_freq_bits) - 1, stats_buffer_.data(), nullptr);
          }
          cost_histo += WP2Log2Fast(n_pixels_bits);
        }

        // Add the cost of the probabilities.
        cost += cost_histo;

        // Keep the best configuration.
        if (cost >= config_best->cost) {
          continue;
        }
        config_best->param.type = type;
        config_best->param.is_sparse = is_sparse;
        config_best->param.max_freq_bits = max_freq_bits;
        config_best->size_to_write = histogram_to_write_size;
        std::swap(config_best->histogram_to_write, histogram_to_write);
        if (is_sparse) {
          std::swap(config_best->histo.counts, histogram_quantized);
        } else {
          std::copy(histogram_quantized,
                    histogram_quantized + histogram_to_write_size,
                    config_best->histo.counts);
        }
        config_best->cost = cost;
      }
    }
  }

  // Reset the common temporary variables.
  histogram_quantized_[config_index] = histogram_quantized;
  histogram_to_write_[config_index] = histogram_to_write;
}

}  // namespace WP2
