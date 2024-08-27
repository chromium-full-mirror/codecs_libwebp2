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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/common/constants.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "fuzzing/fuzztest.h"

namespace WP2 {
namespace testutil {

//------------------------------------------------------------------------------

void LimitConfigForBigImages(DataView bitstream, DecoderConfig* config) {
  // Try to keep the peak allocated memory below 2 GB by reducing the number of
  // concurrently decoded tiles for big images.
  BitstreamFeatures features;
  if (features.Read(bitstream.bytes, bitstream.size) == WP2_STATUS_OK) {
    if (features.width * features.height <= (1u << 20)) {
      // Small image, leave as is.
    } else {
      constexpr uint32_t kMaxPeakMem = (1u << 28);  // 256 MB
      // Estimation of the peak memory allocation within each tile
      // (roughly 16 times the 16-bit 4-channel tile canvas).
      const uint32_t tile_width = TileWidth(
          features.tile_shape, features.raw_width, features.raw_height);
      const uint32_t tile_height = TileHeight(
          features.tile_shape, features.raw_width, features.raw_height);
      const uint32_t bytes_used_per_tile =
          tile_width * tile_height * (sizeof(int16_t) * 4) * 16;
      config->thread_level =
          std::min(config->thread_level,
                   std::max(kMaxPeakMem / bytes_used_per_tile, 1u) - 1u);
    }
  }
}

//------------------------------------------------------------------------------

std::string GetTestAssetPath(std::string_view name) {
  return "tests/" + std::string(name);
}

// Returns a vector of strings instead of const char* to avoid
// fuzztest::ElementOf()-related errors.
std::vector<std::string> GetTestImageNames() {
  // Valid files only.
  return {"alpha_ramp.bmp",
          "alpha_ramp.lossy.webp",
          "alpha_ramp.pam",
          "alpha_ramp.png",
          "alpha_ramp.tiff",
          "alpha_ramp.webp",
          "animation0.gif",
          "animation0.webp",
          "animation1.webp",
          "background.png",
          "composited.png",
          "large.png",
          "logo.png",
          "source0.pgm",
          "source0.ppm",
          "source1_1x1.png",
          "source1_1x48.png",
          "source1_32x32.png",
          "source1_4x4.png",
          "source1_64x1.png",
          "source1_64x48.png",
          "source1.itl.png",
          "source1.png",
          "source2.tiff",
          "source3_222x167.jpg",
          "source3.jpg",
          "source4.ll.webp",
          "source4.webp",
          "source5.bmp",
          "taiwan.png",
          "test_exif_xmp.webp",
          "verylarge.png"};
}

Data ReadTestImage(const std::string& name) {
  Data data;
  if (IoUtilReadFile(GetTestAssetPath("testdata/" + name).c_str(), &data) !=
      WP2_STATUS_OK) {
    std::abort();
  }
  return data;
}

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

std::vector<std::tuple<std::string>> ReadFilesFromDirectory(
    std::string_view dir_path) {
  const std::vector<std::tuple<std::string>> files =
      fuzztest::ReadFilesFromDirectory(dir_path);
  if (files.empty()) {
    std::cerr << "fuzztest::ReadFilesFromDirectory() failed on " << dir_path
              << std::endl;
    abort();
  }
  return files;
}

std::vector<std::string> ReadDictionaryFromFiles(
    const std::vector<std::string_view>& file_paths) {
  std::vector<std::string> merged_dictionary;
  for (std::string_view file_path : file_paths) {
    std::vector<std::string> dictionary =
        fuzztest::ReadDictionaryFromFile(file_path);
    if (dictionary.empty()) {
      std::cerr << "fuzztest::ReadDictionaryFromFile() failed on " << file_path
                << std::endl;
      abort();
    }
    merged_dictionary.insert(merged_dictionary.end(),
                             std::make_move_iterator(dictionary.begin()),
                             std::make_move_iterator(dictionary.end()));
  }
  return merged_dictionary;
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

ArgbBuffer CreateArgbBuffer8b(size_t width, size_t height,
                              WP2SampleFormat format,
                              const std::vector<uint8_t>& samples) {
  assert(samples.size() == GetNumSamples(width, height, format));
  ArgbBuffer image(format);
  // Converting samples is simpler than generating alpha-premultiplied ones.
  const WP2SampleFormat fuzzed_format = format == WP2_Argb_32   ? WP2_ARGB_32
                                        : format == WP2_rgbA_32 ? WP2_RGBA_32
                                        : format == WP2_bgrA_32 ? WP2_BGRA_32
                                                                : format;
  assert(!WP2IsPremultiplied(fuzzed_format));
  if (image.Import(fuzzed_format, width, height, samples.data(),
                   WP2FormatBpp(format) * width) != WP2_STATUS_OK) {
    std::abort();
  }
  return image;
}

size_t GetNumSamples(size_t width, size_t height, WP2SampleFormat format) {
  return width * height * WP2FormatBpp(format) /
         (WP2Formatbpc(format) == 8 ? 1 : 2);
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
