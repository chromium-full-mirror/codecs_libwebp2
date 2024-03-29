// Copyright 2019 Google LLC
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
//  Fuzzing tools.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "./fuzz_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "imageio/file_format.h"
#include "imageio/image_enc.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "fuzzing/fuzztest.h"

#include "examples/example_utils.h"  // For WP2::GetFilesIn()
#include "imageio/imageio_util.h"    // For WP2::IoUtilReadFile()
#include "tests/include/helpers.h"   // For WP2::GetTestDataPath()

namespace WP2 {
namespace testutil {

//------------------------------------------------------------------------------

uint32_t FuzzedParameters::ExtractUInt32(uint32_t max_value) {
  uint32_t extracted_value = 0;
  int max_extracted_value = 0;

  while (bit_pos_ < 8 * data_size_ && max_extracted_value < max_value) {
    extracted_value = (extracted_value << 1);
    max_extracted_value = (max_extracted_value << 1) | 1u;

    const uint8_t mask = (uint8_t)1 << (bit_pos_ & 7);
    if (data_[bit_pos_ >> 3] & mask) extracted_value |= 1;

    ++bit_pos_;
  }

  // Also handles the case where (max_value + 1) would overflow.
  if (max_value < max_extracted_value) {
    return extracted_value % (max_value + 1);
  }
  return extracted_value;
}

uint32_t FuzzedParameters::ExtractUInt32(uint32_t min_value,
                                         uint32_t max_value) {
  assert(max_value >= min_value);
  return min_value + ExtractUInt32(max_value - min_value);
}

double FuzzedParameters::ExtractDouble(double min_value, double max_value) {
  assert(max_value >= min_value);
  return min_value +
         (ExtractUInt32(4294967295u) / 4294967295.) * (max_value - min_value);
}

bool FuzzedParameters::ExtractBool() { return (ExtractUInt32(1) == 1); }

//------------------------------------------------------------------------------

bool FuzzedParameters::ExtractSource(const char** const file_name,
                                     const uint8_t** const file_data,
                                     size_t* const file_size) {
  std::vector<std::string> files;
  constexpr bool kRecursive = false;
  if (!GetFilesIn(GetTestDataPath(""), &files, kRecursive)) return false;
  if (files.size() == 0) return false;
  *file_name = files[ExtractUInt32((uint32_t)files.size() - 1)].c_str();
  Data data;
  if (IoUtilReadFile(*file_name, &data) != WP2_STATUS_OK) return false;
  *file_data = data.bytes;
  *file_size = data.size;
  return true;
}

FileFormat FuzzedParameters::ExtractOutputFormat() {
  const FileFormat kOutputFormats[] = {
      FileFormat::PNG, FileFormat::TIFF, FileFormat::BMP,
      FileFormat::PAM, FileFormat::PGM,  FileFormat::PPM,
  };
  const uint32_t kNumOutputFormats =
      sizeof(kOutputFormats) / sizeof(kOutputFormats[0]);

  return kOutputFormats[ExtractUInt32(kNumOutputFormats - 1)];
}

//------------------------------------------------------------------------------

namespace {

// Returns all unique divisors of 'number', unsorted.
std::vector<uint32_t> GetDivisors(uint32_t number, uint32_t max_divisor) {
  std::vector<uint32_t> divisors;
  uint32_t divisor = (uint32_t)std::lround(
      std::ceil(number / (double)std::min(max_divisor, number)));
  for (; divisor * divisor < number; ++divisor) {
    if (number % divisor == 0) {
      if (divisor > max_divisor) return divisors;
      divisors.push_back(divisor);
      divisors.push_back(number / divisor);
    }
  }
  if (divisor * divisor == number && divisor <= max_divisor) {
    divisors.push_back(divisor);
  }
  return divisors;
}

}  // namespace

ArgbBuffer FuzzedParameters::ExtractArgbBuffer(uint32_t max_num_pixels,
                                               uint32_t max_width_or_height,
                                               WP2SampleFormat format) {
  assert(max_num_pixels <= kMaxBufferDimension);
  const size_t previous_bit_pos = bit_pos_;

  // Some bytes will be used for extracting the size and for the alignment.
  const size_t max_num_bits_for_size = (sizeof(uint32_t) + sizeof(uint8_t)) * 8;
  assert(bit_pos_ <= data_size_ * 8);

  // Check if there is enough data left.
  size_t num_available_bits = data_size_ * 8 - bit_pos_;
  if (num_available_bits < max_num_bits_for_size) return ArgbBuffer(format);
  num_available_bits -= max_num_bits_for_size;

  uint32_t num_pixels =
      std::min((uint32_t)num_available_bits / (8 * WP2FormatBpp(format)),
               max_num_pixels);
  if (num_pixels < 1) return ArgbBuffer(format);

  // Compute random valid width and height based on the number of pixels.
  std::vector<uint32_t> divisors =
      GetDivisors(num_pixels, /*max_divisor=*/max_width_or_height);
  if (divisors.empty()) {  // No valid divisor respecting 'max_width_or_height'.
    num_pixels = max_width_or_height;
    divisors = GetDivisors(num_pixels, /*max_divisor=*/max_width_or_height);
  }
  const uint32_t width = divisors[ExtractUInt32((uint32_t)divisors.size() - 1)];
  const uint32_t height = num_pixels / width;

  bit_pos_ = (bit_pos_ + 7u) & ~7u;  // Align.
  const uint32_t stride = width * WP2FormatBpp(format);
  assert(data_size_ - (bit_pos_ >> 3) >= height * stride);

  // Copy bytes and consider them as ARGB. Allocation may fail.
  ArgbBuffer buffer(format);
  if (buffer.Import(WP2_ARGB_32, width, height, data_ + (bit_pos_ >> 3),
                    stride) != WP2_STATUS_OK) {
    bit_pos_ = previous_bit_pos;  // Unread everything in case of failure.
    return ArgbBuffer(format);
  }
  bit_pos_ += (height * stride) * 8;
  return buffer;
}

//------------------------------------------------------------------------------

bool FuzzedParameters::TryExtractWP2Bitstream(Data* const encoded_data) {
  bit_pos_ = (bit_pos_ + 7u) & ~7u;  // Align.
  const size_t byte_pos = bit_pos_ / 8u;
  const size_t num_available_bytes = data_size_ - byte_pos;
  if (encoded_data->Resize(3u + num_available_bytes, /*keep_bytes=*/false) !=
      WP2_STATUS_OK) {
    return false;
  }
  encoded_data->bytes[0] = (uint8_t)((kSignature >> 0) & 0xffu);
  encoded_data->bytes[1] = (uint8_t)((kSignature >> 8) & 0xffu);
  encoded_data->bytes[2] = (uint8_t)((kSignature >> 16) & 0xffu);
  std::memcpy(encoded_data->bytes + 3, data_ + byte_pos, num_available_bytes);
  return true;
}

//------------------------------------------------------------------------------

void FuzzedParameters::ExtractConfig(EncoderConfig* const config) {
  config->quality = (float)ExtractDouble(0., kMaxQuality);
  config->target_size = ExtractBool() ? 0 : (size_t)ExtractUInt32(1 << 20);
  config->target_psnr = ExtractBool() ? 0.f : (float)ExtractDouble(.01, 99.);
  config->alpha_quality = (float)ExtractDouble(0., kMaxQuality);
  config->effort = (int)ExtractUInt32(kMinEffort, kMaxEffort);
  config->use_av1 = false;  // TODO(yguyon): Test AV1

  config->decoding_orientation = (Orientation)ExtractUInt32(7);
  config->create_preview = ExtractBool();
  // 'preview' and 'preview_size' are not fuzzed.

  config->transfer_function =
      (TransferFunction)ExtractUInt32((uint32_t)WP2_TF_ARIB_STD_B67_HLG);

  config->pass = (int)ExtractUInt32(1, 10);
  config->sns = (float)ExtractDouble(0., 100.);
  config->error_diffusion = (int)ExtractUInt32(100);

  config->segments = (int)ExtractUInt32(1, kMaxNumSegments);
  config->segment_id_mode = (EncoderConfig::SegmentIdMode)ExtractUInt32(2);
  for (uint32_t segment = 0; segment < kMaxNumSegments; ++segment) {
    config->segment_factors[segment] = (float)ExtractUInt32(kMaxLossyQuality);
  }

  config->tile_shape = (TileShape)ExtractUInt32((uint32_t)TILE_SHAPE_AUTO);

  config->partition_method =
      (PartitionMethod)ExtractUInt32(NUM_PARTITION_METHODS - 1);
  config->partition_set = (PartitionSet)ExtractUInt32(NUM_PARTITION_SETS - 1);
  config->partition_snapping =
      (config->partition_method == AREA_ENCODE_PARTITIONING ||
       config->partition_method == SUB_AREA_ENCODE_PARTITIONING ||
       config->partition_method == SPLIT_PARTITIONING ||
       config->partition_method == SPLIT_RECURSE_PARTITIONING)
          ? true
          : ExtractBool();

  config->csp_type = (Csp)ExtractUInt32(kNumCspTypes - 1);
  config->uv_mode = (EncoderConfig::UVMode)ExtractUInt32(
      (uint32_t)EncoderConfig::NumUVMode - 1);
  config->preprocessing = (int)ExtractUInt32(2);
  config->preprocessing_strength = (int)ExtractUInt32(100);
  config->use_random_matrix = ExtractBool();
  config->store_grain = ExtractBool();
  config->tune_perceptual = ExtractBool();
  config->use_delta_palette = ExtractBool();

  config->thread_level = (int)ExtractUInt32((1u << 31) - 1u);
  config->low_memory = ExtractBool();
  // 'use_neural_compression' is not fuzzed.
}

//------------------------------------------------------------------------------

void FuzzedParameters::ExtractConfig(DecoderConfig* const config) {
  config->thread_level = ExtractUInt32((1u << 31) - 1u);
  config->enable_deblocking_filter = ExtractBool();
  config->enable_directional_filter = ExtractBool();
  config->enable_restoration_filter = ExtractBool();
  config->enable_alpha_filter = ExtractBool();
  config->grain_amplitude = (uint8_t)ExtractUInt32(100);
  config->incremental_mode = (DecoderConfig::IncrementalMode)ExtractUInt32(1);
}

void LimitConfigForBigImages(const Data& bitstream, DecoderConfig* config) {
  LimitConfigForBigImages(DataView{bitstream.bytes, bitstream.size}, config);
}

void LimitConfigForBigImages(DataView bitstream, DecoderConfig* config) {
  // Try to keep the peak allocated memory below 2 GB by reducing the number of
  // concurrently decoded tiles for big images.
  WP2::BitstreamFeatures features;
  if (features.Read(bitstream.bytes, bitstream.size) == WP2_STATUS_OK) {
    if (features.width * features.height <= (1u << 20)) {
      // Small image, leave as is.
    } else {
      constexpr uint32_t kMaxPeakMem = (1u << 28);  // 256 MB
      // Estimation of the peak memory allocation within each tile
      // (roughly 16 times the 16-bit 4-channel tile canvas).
      const uint32_t bytes_used_per_tile = features.tile_width *
                                           features.tile_height *
                                           (sizeof(int16_t) * 4) * 16;
      config->thread_level =
          std::min(config->thread_level,
                   std::max(kMaxPeakMem / bytes_used_per_tile, 1u) - 1u);
    }
  }
}

//------------------------------------------------------------------------------

std::vector<std::string> GetTestImagesContents(size_t max_file_size) {
  const char* test_data_dirs[] = {"testdata"};

  std::vector<std::string> seeds;
  for (const char* test_data_dir : test_data_dirs) {
    std::cout << "Reading seeds from " << test_data_dir << " (non recursively)"
              << std::endl;
    std::vector<std::tuple<std::string>> tuple_vector =
        fuzztest::ReadFilesFromDirectory(test_data_dir);
    seeds.reserve(seeds.size() + tuple_vector.size());
    for (auto& [file_content] : tuple_vector) {
      if (file_content.size() > max_file_size) continue;
      seeds.push_back(std::move(file_content));
    }
  }
  if (seeds.empty()) {
    std::cerr << "ERROR: no files found that match the given file size"
              << std::endl;
    std::abort();
  }
  std::cout << "Returning " << seeds.size() << " seed images" << std::endl;
  return seeds;
}

//------------------------------------------------------------------------------

EncoderConfig CreateEncoderConfig(
    float quality, size_t target_size, float target_psnr, float alpha_quality,
    int effort, bool exact, Orientation decoding_orientation,
    bool create_preview, TransferFunction transfer_function, int pass,
    float sns, int error_diffusion, const std::vector<float>& segments,
    EncoderConfig::SegmentIdMode segment_id_mode, bool enable_alt_tuning,
    TileShape tile_shape, PartitionMethod partition_method,
    PartitionSet partition_set, bool partition_snapping, Csp csp_type,
    EncoderConfig::UVMode uv_mode, int preprocessing,
    int preprocessing_strength, bool use_random_matrix, bool store_grain,
    bool tune_perceptual, bool use_delta_palette, int thread_level,
    bool low_memory) {
  EncoderConfig config;
  config.quality = quality;
  config.target_size = target_size;
  config.target_psnr = target_psnr;
  config.alpha_quality = alpha_quality;
  config.effort = effort;
  config.use_av1 = false;

  config.decoding_orientation = decoding_orientation;
  config.create_preview = create_preview;
  // 'preview' and 'preview_size' are not fuzzed.

  config.transfer_function = transfer_function;

  config.pass = pass;
  config.sns = sns;
  config.error_diffusion = error_diffusion;

  config.segments = static_cast<int>(segments.size());
  config.segment_id_mode = segment_id_mode;
  std::copy(segments.data(), segments.data() + segments.size(),
            config.segment_factors);

  config.tile_shape = tile_shape;

  config.partition_method = partition_method;
  config.partition_set = partition_set;
  config.partition_snapping =
      (partition_method == AREA_ENCODE_PARTITIONING ||
       partition_method == SUB_AREA_ENCODE_PARTITIONING ||
       partition_method == SPLIT_PARTITIONING ||
       partition_method == SPLIT_RECURSE_PARTITIONING)
          ? true
          : partition_snapping;

  config.csp_type = csp_type;
  config.uv_mode = uv_mode;
  config.preprocessing = preprocessing;
  config.preprocessing_strength = preprocessing_strength;
  config.use_random_matrix = use_random_matrix;
  config.store_grain = store_grain;
  config.tune_perceptual = tune_perceptual;
  config.use_delta_palette = use_delta_palette;

  config.thread_level = thread_level;
  config.low_memory = low_memory;
  // 'use_neural_compression' is not fuzzed.
  return config;
}

DecoderConfig CreateDecoderConfig(
    uint32_t thread_level, bool enable_deblocking_filter,
    bool enable_directional_filter, bool enable_restoration_filter,
    bool enable_alpha_filter, uint8_t grain_amplitude,
    DecoderConfig::IncrementalMode incremental_mode) {
  DecoderConfig config;
  config.thread_level = thread_level;
  config.enable_deblocking_filter = enable_deblocking_filter;
  config.enable_directional_filter = enable_directional_filter;
  config.enable_restoration_filter = enable_restoration_filter;
  config.enable_alpha_filter = enable_alpha_filter;
  config.grain_amplitude = grain_amplitude;
  config.incremental_mode = incremental_mode;
  return config;
}

//------------------------------------------------------------------------------

void SaveBeforeAfter(const ArgbBuffer& original, const ArgbBuffer& decompressed,
                     const std::string& path) {
  ArgbBuffer original_buffer(
      WP2IsPremultiplied(original.format()) ? WP2_Argb_32 : WP2_ARGB_32);
  WP2_ASSERT_STATUS(original.format() == original_buffer.format()
                        ? original_buffer.SetView(original)
                        : original_buffer.ConvertFrom(original));
  ArgbBuffer decompressed_buffer(
      WP2IsPremultiplied(decompressed.format()) ? WP2_Argb_32 : WP2_ARGB_32);
  WP2_ASSERT_STATUS(decompressed.format() == decompressed_buffer.format()
                        ? decompressed_buffer.SetView(decompressed)
                        : decompressed_buffer.ConvertFrom(decompressed));
  WP2_ASSERT_STATUS(SaveImage(original_buffer, (path + "_original.png").c_str(),
                              /*overwrite=*/true));
  WP2_ASSERT_STATUS(SaveImage(decompressed_buffer,
                              (path + "_decompressed.png").c_str(),
                              /*overwrite=*/true));
}

//------------------------------------------------------------------------------

}  // namespace testutil
}  // namespace WP2
