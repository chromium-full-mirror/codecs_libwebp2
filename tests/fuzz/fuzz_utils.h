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

#ifndef WP2_TESTS_FUZZ_FUZZ_UTILS_H_
#define WP2_TESTS_FUZZ_FUZZ_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "fuzzing/fuzztest.h"

namespace WP2 {
namespace testutil {

// Limit read image size to avoid out-of-memory and timeout issues.
// Hint:  kMaxNumPixels * kMaxBytesPerPixel (4) * kMaxNumFullCanvasAlloc (3)
//        should be below kMaxFuzzMemory (2 GB).
// Hint:  Fuzzed inputs of a few megapixels tend to take longer than
//        kEncodingMaxNumSeconds to decode when sanitizers are enabled.
constexpr size_t kMaxNumPixels = 1u << 22;

// Limit encoding duration to avoid timeout issues (< 90 seconds).
constexpr double kEncodingMaxNumSeconds = 60.;

// Modifies the 'config' so that decoding the 'bitstream' is not too CPU- or
// memory-intensive.
void LimitConfigForBigImages(DataView bitstream, DecoderConfig* config);

// Verifies that 'original' can be encoded with 'config' or aborts. Also checks
// that the decoded image is not worse than the 'expected_distortion' if not 0.
int TestImageEncConfig(const ArgbBuffer& original, const EncoderConfig& config,
                       float expected_distortion);

int TestAnimEncConfig(const std::vector<ArgbBuffer>& original_frames,
                      const std::vector<uint32_t>& durations_ms,
                      bool loop_forever, const EncoderConfig& config,
                      float expected_distortion);

// Verifies that 'encoded_data' does not crash when decoded with 'config'.
// Aborts if it's not decoded without error in case of 'expected_success'.
int TestImageDecConfig(DataView encoded_data, const DecoderConfig& config,
                       bool expected_success);

//------------------------------------------------------------------------------

std::string GetTestAssetPath(std::string_view name);

// Returns the list of test image paths from the "testdata" directory.
std::vector<std::string> GetTestImageNames();

// Reads a file from the tests/testdata directory and returns its contents.
Data ReadTestImage(const std::string& name);

// Returns a list of test image contents (not paths) that are smaller than
// 'max_file_size'.
std::vector<std::string> GetTestImagesContents(size_t max_file_size);

// Same as fuzztest::ReadFilesFromDirectory() but checks for file existence.
std::vector<std::tuple<std::string>> ReadFilesFromDirectory(
    std::string_view dir);

std::vector<std::string> ReadDictionaryFromFiles(
    const std::vector<std::string_view>& file_paths);

// Generator for an arbitrary encoded byte sequence taken from
// GetTestImagesContents() or a mutation of it.
inline auto ArbitraryImageWithSeeds() {
  constexpr uint32_t kMaxSeedFileSize = 1024 * 1024;  // 1MB.
  return fuzztest::String()
      .WithMaxSize(kMaxSeedFileSize)
      .WithSeeds(GetTestImagesContents(kMaxSeedFileSize))
      .WithDictionary(ReadDictionaryFromFiles(
          {GetTestAssetPath("fuzz/common_wp2_tags.dict")}));
}

//------------------------------------------------------------------------------

// Exposed for convenient fuzztest reproducer output.
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
    bool low_memory);
DecoderConfig CreateDecoderConfig(
    uint32_t thread_level, bool enable_deblocking_filter,
    bool enable_directional_filter, bool enable_restoration_filter,
    bool enable_alpha_filter, uint8_t grain_amplitude,
    DecoderConfig::IncrementalMode incremental_mode);

ArgbBuffer CreateArgbBuffer8b(size_t width, size_t height,
                              WP2SampleFormat format,
                              const std::vector<uint8_t>& samples);

size_t GetNumSamples(size_t width, size_t height, WP2SampleFormat format);

//------------------------------------------------------------------------------

// Do not generate images wider or taller than this.
inline constexpr size_t kMaxDimension = 256;  // In pixels.

// ArgbBuffer generator type: Width, height, format and 8-bit samples.
inline auto ArbitraryArgbBuffer8b() {
  return fuzztest::FlatMap(
      [](size_t width, size_t height, WP2SampleFormat format) {
        return fuzztest::Map(
            CreateArgbBuffer8b, fuzztest::Just(width), fuzztest::Just(height),
            fuzztest::Just(format),
            fuzztest::Arbitrary<std::vector<uint8_t>>().WithSize(
                GetNumSamples(width, height, format)));
      },
      fuzztest::InRange<size_t>(1, kMaxDimension),
      fuzztest::InRange<size_t>(1, kMaxDimension), fuzztest::Just(WP2_Argb_32));
}

//------------------------------------------------------------------------------

// Generator for an arbitrary EncoderConfig.
inline auto ArbitraryEncoderConfig() {
  return fuzztest::Map(
      CreateEncoderConfig,

      /*quality=*/fuzztest::InRange<float>(0, kMaxQuality),
      /*target_size=*/
      fuzztest::OneOf(fuzztest::Just<size_t>(0),
                      fuzztest::InRange<size_t>(1, 1 << 20)),
      /*target_psnr=*/
      fuzztest::OneOf(fuzztest::Just<float>(0), fuzztest::Just<float>(99.),
                      fuzztest::InRange<float>(.01, 99.)),
      /*alpha_quality=*/fuzztest::InRange<float>(0, kMaxQuality),
      /*effort=*/fuzztest::InRange<int>(kMinEffort, kMaxEffort),
      /*exact=*/fuzztest::Arbitrary<bool>(),

      fuzztest::ElementOf(
          {Orientation::kOriginal, Orientation::kFlipLeftToRight,
           Orientation::k180, Orientation::kFlipTopToBottom,
           Orientation::kFlipBotLeftToTopRgt, Orientation::k90Clockwise,
           Orientation::kFlipTopLeftToBotRgt, Orientation::k270Clockwise}),
      /*create_preview=*/fuzztest::Arbitrary<bool>(),

      fuzztest::ElementOf<TransferFunction>(
          {WP2_TF_ITU_R_BT2020_10BIT, WP2_TF_ITU_R_BT709, WP2_TF_UNSPECIFIED,
           WP2_TF_GAMMA_22, WP2_TF_GAMMA_28, WP2_TF_SMPTE_170M,
           WP2_TF_SMPTE_240M, WP2_TF_LINEAR, WP2_TF_LOG, WP2_TF_SQRT,
           WP2_TF_IEC_61966_2_4, WP2_TF_ITU_R_BT1361_EXTENDED,
           WP2_TF_IEC_61966_2_1, WP2_TF_ITU_R_BT2020_12BIT,
           WP2_TF_SMPTE_ST_2084, WP2_TF_SMPTE_ST_428_1,
           WP2_TF_ARIB_STD_B67_HLG}),

      /*pass=*/fuzztest::InRange<int>(1, 10),
      /*sns=*/fuzztest::InRange<float>(0, 100),
      /*error_diffusion=*/fuzztest::InRange<int>(0, 100),

      /*segments=*/
      fuzztest::VectorOf(fuzztest::InRange<float>(0, kMaxLossyQuality))
          .WithMinSize(1)
          .WithMaxSize(kMaxNumSegments),
      fuzztest::ElementOf<EncoderConfig::SegmentIdMode>(
          {EncoderConfig::SEGMENT_ID_AUTO, EncoderConfig::SEGMENT_ID_EXPLICIT,
           EncoderConfig::SEGMENT_ID_IMPLICIT}),

      /*enable_alt_tuning=*/fuzztest::Arbitrary<bool>(),
      fuzztest::ElementOf<TileShape>(
          {TILE_SHAPE_SQUARE_128, TILE_SHAPE_SQUARE_256, TILE_SHAPE_SQUARE_512,
           TILE_SHAPE_WIDE, TILE_SHAPE_AUTO}),

      fuzztest::ElementOf<PartitionMethod>(
          {MULTIPASS_PARTITIONING, BLOCK_ENCODE_PARTITIONING,
           SPLIT_PARTITIONING, TILE_ENCODE_PARTITIONING,
           SPLIT_RECURSE_PARTITIONING, AREA_ENCODE_PARTITIONING,
           SUB_AREA_ENCODE_PARTITIONING, EXHAUSTIVE_PARTITIONING,
           ALL_4X4_PARTITIONING, ALL_8X8_PARTITIONING, ALL_16X16_PARTITIONING,
           ALL_32X32_PARTITIONING, AUTO_PARTITIONING}),
      fuzztest::ElementOf<PartitionSet>({SMALL_SQUARES, SMALL_RECTS, ALL_RECTS,
                                         THICK_RECTS, MEDIUM_SQUARES,
                                         ALL_SQUARES, SOME_RECTS}),
      /*partition_snapping=*/fuzztest::Arbitrary<bool>(),

      fuzztest::ElementOf<Csp>(
          {Csp::kYCoCg, Csp::kYCbCr, Csp::kCustom, Csp::kYIQ}),
      fuzztest::ElementOf<EncoderConfig::UVMode>(
          {EncoderConfig::UVModeAdapt, EncoderConfig::UVMode420,
           EncoderConfig::UVMode444, EncoderConfig::UVModeAuto}),
      /*preprocessing=*/fuzztest::InRange<int>(0, 2),
      /*preprocessing_strength=*/fuzztest::InRange<int>(0, 100),
      /*use_random_matrix=*/fuzztest::Arbitrary<bool>(),
      /*store_grain=*/fuzztest::Arbitrary<bool>(),
      /*tune_perceptual=*/fuzztest::Arbitrary<bool>(),
      /*use_delta_palette=*/fuzztest::Arbitrary<bool>(),

      /*thread_level=*/fuzztest::InRange<int>(0, (1u << 31) - 1u),
      /*low_memory=*/fuzztest::Arbitrary<bool>());
}

inline auto ArbitraryProgressFailAt() {
  return fuzztest::OneOf(fuzztest::InRange<double>(0, 1),
                         fuzztest::Just<double>(2));
}

// Generator for an arbitrary DecoderConfig.
inline auto ArbitraryDecoderConfig() {
  return fuzztest::Map(
      CreateDecoderConfig,
      /*thread_level=*/fuzztest::InRange<uint32_t>(0, (1u << 31) - 1u),
      /*enable_deblocking_filter=*/fuzztest::Arbitrary<bool>(),
      /*enable_directional_filter=*/fuzztest::Arbitrary<bool>(),
      /*enable_restoration_filter=*/fuzztest::Arbitrary<bool>(),
      /*enable_alpha_filter=*/fuzztest::Arbitrary<bool>(),
      /*grain_amplitude=*/fuzztest::InRange<uint8_t>(0, 100),
      fuzztest::ElementOf(
          {DecoderConfig::IncrementalMode::FULL_TILE,
           DecoderConfig::IncrementalMode::PARTIAL_TILE_CONTEXT}));
}

//------------------------------------------------------------------------------

// Exports 'original' and 'decompressed' to PNG files at 'path' for debugging.
void SaveBeforeAfter(const ArgbBuffer& original, const ArgbBuffer& decompressed,
                     const std::string& path);

}  // namespace testutil
}  // namespace WP2

#endif  // WP2_TESTS_FUZZ_FUZZ_UTILS_H_
