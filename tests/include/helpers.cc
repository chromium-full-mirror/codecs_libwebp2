// Copyright 2020 Google LLC
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

#include "./helpers.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "examples/example_utils.h"
#include "imageio/anim_image_dec.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "src/dsp/lossless/decl_dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/utils/front_mgr.h"
#include "src/utils/orientation.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace testutil {

//------------------------------------------------------------------------------

std::string GetTestDataPath(const std::string& file_name) {
  return "testdata/" + file_name;
}

void GetTestDataPaths(const std::vector<const char*>& file_names,
                      std::vector<std::string>* const file_paths) {
  file_paths->resize(file_names.size());
  for (size_t i = 0; i < file_names.size(); ++i) {
    file_paths->at(i) = GetTestDataPath(file_names[i]);
  }
}

std::string GetTempDataPath(const std::string& file_path) {
  return "/tmp/" + GetFileName(file_path);
}

//------------------------------------------------------------------------------

bool HasSameData(const Data& a, const Data& b) {
  if (a.size != b.size) return false;
  if (a.size == 0) return true;
  if (a.bytes == nullptr || b.bytes == nullptr) return false;
  return (std::memcmp((void*)a.bytes, (void*)b.bytes, a.size) == 0);
}

//------------------------------------------------------------------------------

WP2Status ReadImages(const std::vector<std::string>& file_paths,
                     std::vector<ArgbBuffer>* const frames) {
  frames->resize(file_paths.size());
  for (size_t i = 0; i < frames->size(); ++i) {
    WP2_CHECK_STATUS(ReadImage(file_paths[i].c_str(), &frames->at(i)));
  }
  return WP2_STATUS_OK;
}

WP2Status ReadAnimation(const uint8_t* const data, size_t size,
                        std::vector<ArgbBuffer>* const frames,
                        std::vector<uint32_t>* const frame_durations,
                        uint32_t* const loop_count) {
  ArgbBuffer buffer;
  ImageReader image_reader(data, size, &buffer);
  frames->clear();
  if (frame_durations != nullptr) frame_durations->clear();

  bool is_last_frame = false;
  do {
    uint32_t duration_ms = 0;
    WP2_CHECK_STATUS(image_reader.ReadFrame(&is_last_frame, &duration_ms));
    assert(!buffer.IsEmpty() && !buffer.IsView());
    frames->emplace_back(buffer.format());
    WP2_CHECK_STATUS(frames->back().CopyFrom(buffer));
    if (frame_durations != nullptr) frame_durations->push_back(duration_ms);
  } while (!is_last_frame);

  if (loop_count != nullptr) *loop_count = image_reader.GetLoopCount();
  return WP2_STATUS_OK;
}

WP2Status ReadAnimation(const std::string& file_path,
                        std::vector<ArgbBuffer>* const frames,
                        std::vector<uint32_t>* const frame_durations,
                        uint32_t* const loop_count) {
  Data data;
  WP2_CHECK_STATUS(IoUtilReadFile(file_path.c_str(), &data));
  WP2_CHECK_STATUS(ReadAnimation(data.bytes, data.size, frames, frame_durations,
                                 loop_count));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status CompressImage(const std::string& file_name,
                        Writer* const encoded_data, ArgbBuffer* original_image,
                        float quality, int effort, int thread_level,
                        uint32_t num_downsamplings) {
  ArgbBuffer internal_original_image;
  if (original_image == nullptr) original_image = &internal_original_image;
  WP2_CHECK_STATUS(
      ReadImage(GetTestDataPath(file_name).c_str(), original_image));
  for (uint32_t i = 0; i < num_downsamplings; ++i) {
    original_image->SimpleHalfDownsample();
  }

  EncoderConfig config;
  config.quality = quality;
  config.effort = effort;
  config.thread_level = thread_level;
  return Encode(*original_image, encoded_data, config);
}

WP2Status CompressAnimation(const std::vector<const char*>& frames_file_names,
                            const std::vector<uint32_t>& durations_ms,
                            Writer* const encoded_data,
                            std::vector<ArgbBuffer>* frames, float quality,
                            int effort, int thread_level,
                            uint32_t num_downsamplings) {
  assert(frames_file_names.size() == durations_ms.size());
  std::vector<std::string> file_paths;
  GetTestDataPaths(frames_file_names, &file_paths);
  std::vector<ArgbBuffer> internal_frames;
  if (frames == nullptr) frames = &internal_frames;
  WP2_CHECK_STATUS(ReadImages(file_paths, frames));
  for (uint32_t i = 0; i < num_downsamplings; ++i) {
    for (ArgbBuffer& frame : *frames) frame.SimpleHalfDownsample();
  }

  AnimationEncoder animation_encoder;
  for (size_t i = 0; i < frames->size(); ++i) {
    WP2_CHECK_STATUS(
        animation_encoder.AddFrame(frames->at(i), durations_ms[i]));
  }

  EncoderConfig config;
  config.quality = quality;
  config.effort = effort;
  config.thread_level = thread_level;
  return animation_encoder.Encode(encoded_data, config);
}

//------------------------------------------------------------------------------

float GetExpectedDistortion(float quality, float alpha_quality, int effort) {
  if (quality <= kMaxLossyQuality || alpha_quality <= kMaxLossyQuality) {
    const float malus = -0.5f * (5.f - std::min(5, effort));
    return (14.f + quality * 0.06f + alpha_quality * 0.03f + malus);  // Lossy
  } else if (quality < kMaxQuality || alpha_quality < kMaxQuality) {
    quality -= kMaxLossyQuality;
    alpha_quality -= kMaxLossyQuality;
    return (30.f + 1.f * quality + 1.f * alpha_quality);  // Near-lossless
  } else {
    return 99.f;  // Lossless
  }
}

float GetExpectedDistortion(const EncoderConfig& encoder_config) {
  return GetExpectedDistortion(encoder_config.quality,
                               encoder_config.alpha_quality,
                               encoder_config.effort);
}

namespace {

// Returns true and the coordinates of the first pixel that differ between 'a'
// and 'b'. Returns false if 'a' and 'b' are identical.
bool TryFindFirstDifference(const ArgbBuffer& a, const ArgbBuffer& b,
                            uint32_t& pixel_x, uint32_t& pixel_y) {
  assert(a.width() == b.width() && a.height() == b.height());
  assert(a.format() == b.format());
  for (uint32_t y = 0; y < a.height(); ++y) {
    for (uint32_t x = 0; x < a.width(); ++x) {
      const uint8_t* const pixel_a = a.GetPosition(x, y);
      const uint8_t* const pixel_b = b.GetPosition(x, y);
      if (!std::equal(pixel_a, pixel_a + WP2FormatBpp(a.format()), pixel_b)) {
        pixel_x = x;
        pixel_y = y;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

testing::AssertionResult Compare(const ArgbBuffer& src, const ArgbBuffer& dec,
                                 const std::string& file_name,
                                 float expected_distortion, MetricType metric,
                                 Orientation decoding_orientation) {
  if (src.width() !=
          OrientateWidth(decoding_orientation, dec.width(), dec.height()) ||
      src.height() !=
          OrientateHeight(decoding_orientation, dec.width(), dec.height())) {
    return testing::AssertionFailure()
           << "Original image dimensions " << src.width() << "x" << src.height()
           << " are different than the decompressed ones " << dec.width() << "x"
           << dec.height() << " with file " << file_name;
  }
  if (src.format() != dec.format()) {
    return testing::AssertionFailure() << "Format mismatch (" << src.format()
                                       << " vs " << dec.format() << ")";
  }

  ArgbBuffer dec_oriented(dec.format());
  if (decoding_orientation != Orientation::kOriginal) {
    if (OrientateBuffer(GetInverseOrientation(decoding_orientation), dec,
                        &dec_oriented) != WP2_STATUS_OK) {
      return testing::AssertionFailure() << "Failed to rotate buffer";
    }
  } else {
    if (dec_oriented.SetView(dec) != WP2_STATUS_OK) {
      return testing::AssertionFailure() << "Failed to set view";
    }
  }

  float disto[5];
  if (dec_oriented.GetDistortion(src, metric, disto) != WP2_STATUS_OK) {
    return testing::AssertionFailure()
           << "Can not get distortion with file " << file_name;
  }
  if (disto[4] < expected_distortion) {
    auto assertion_failure =
        testing::AssertionFailure()
        << "Distortion is too high: " << disto[0] << ", " << disto[1] << ", "
        << disto[2] << ", " << disto[3] << ", " << disto[4] << " with file "
        << file_name << " (expected at least " << expected_distortion << ")";

    uint32_t x, y;
    if (!TryFindFirstDifference(src, dec_oriented, x, y)) {
      assert(false);  // There should be a difference.
      return assertion_failure;
    }
    const uint8_t* const a = src.GetPosition(x, y);
    const uint8_t* const b = dec_oriented.GetPosition(x, y);
    int32_t aa, ar, ag, ab, ba, br, bg, bb;  // Print numbers, not chars
    if (WP2Formatbpc(src.format()) == 8) {
      aa = a[0], ar = a[1], ag = a[2], ab = a[3];
      ba = b[0], br = b[1], bg = b[2], bb = b[3];
    } else {
      const uint16_t* const a16 = (const uint16_t*)a;
      const uint16_t* const b16 = (const uint16_t*)b;
      aa = a16[0], ar = a16[1], ag = a16[2], ab = a16[3];
      ba = b16[0], br = b16[1], bg = b16[2], bb = b16[3];
    }
    assertion_failure << std::endl
                      << "First different pixel at " << x << ", " << y << ":   "
                      << aa << ", " << ar << ", " << ag << ", " << ab << " vs "
                      << ba << ", " << br << ", " << bg << ", " << bb;
    return assertion_failure;
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult Compare(const YUVPlane& src, const YUVPlane& dec,
                                 BitDepth bit_depth,
                                 const std::string& file_name,
                                 float expected_distortion, MetricType metric) {
  // some easy checks
  if ((src.Y.w_ != dec.Y.w_ || src.Y.h_ != dec.Y.h_) ||
      (src.U.w_ != dec.U.w_ || src.U.h_ != dec.U.h_) ||
      (src.V.w_ != dec.V.w_ || src.V.h_ != dec.V.h_) ||
      src.A.IsEmpty() != dec.A.IsEmpty()) {
    return testing::AssertionFailure() << "Different image dimensions";
  }

  float disto[5];
  if (dec.GetDistortion(src, bit_depth, metric, disto) != WP2_STATUS_OK) {
    return testing::AssertionFailure()
           << "Can not get distortion with file " << file_name;
  }
  if (disto[4] < expected_distortion) {
    auto assertion_failure =
        testing::AssertionFailure()
        << "Distortion is too high: " << disto[0] << ", " << disto[1] << ", "
        << disto[2] << ", " << disto[3] << ", " << disto[4] << " with file "
        << file_name << " (expected at least " << expected_distortion << ")";
    for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
      const Plane16& src_plane = src.GetChannel(channel);
      const Plane16& dec_plane = dec.GetChannel(channel);
      for (uint32_t y = 0; y < src_plane.h_; ++y) {
        for (uint32_t x = 0; x < src_plane.w_; ++x) {
          if (src_plane.At(x, y) != dec_plane.At(x, y)) {
            assertion_failure << std::endl
                              << "First different pixel in "
                              << kChannelStr[channel] << " at " << x << ", "
                              << y << ":   " << src_plane.At(x, y) << " vs "
                              << dec_plane.At(x, y);
            y = src_plane.h_;
            break;  // Don't spam.
          }
        }
      }
    }
    return assertion_failure;
  }
  return testing::AssertionSuccess();
}

bool ProgressTester::OnUpdate(double progress) {
  assert(progress >= 0. && progress <= 1.);
  assert(progress >= last_progress_);
  last_progress_ = progress;
  return (progress < fail_at_);
}

bool ProgressTimeout::OnUpdate(double progress) {
  const std::chrono::time_point<Clock> now = Clock::now();
  const std::chrono::duration<double> diff = now - start_;
#if WP2_CHECK_PROGRESS_GRANULARITY  // Enable to detect missing progress calls.
  if (diff.count() > 1.) abort();   // Checks that the duration between two
  start_ = now;                     // OnUpdate() calls is not too long.
#endif
  return (ProgressTester::OnUpdate(progress) && diff.count() < max_seconds_);
}

void ProgressTimeout::Reset() {
  ProgressTester::Reset();
  start_ = Clock::now();
}

//------------------------------------------------------------------------------

testing::AssertionResult EncodeDecodeCompare(
    const std::string& file_name, const EncoderConfig& encoder_config,
    const DecoderConfig& decoder_config) {
  ArgbBuffer original;
  WP2Status status = ReadImage(GetTestDataPath(file_name).c_str(), &original);
  if (status != WP2_STATUS_OK) {
    return testing::AssertionFailure() << "Failed to read file " << file_name
                                       << ": " << WP2GetStatusMessage(status);
  }

  MemoryWriter memory_writer;
  status = Encode(original, &memory_writer, encoder_config);
  if (status != WP2_STATUS_OK) {
    return testing::AssertionFailure()
           << "Failed to encode: " << WP2GetStatusMessage(status);
  }

  ArgbBuffer decompressed;
  status = Decode(memory_writer.mem_, memory_writer.size_, &decompressed,
                  decoder_config);
  if (status != WP2_STATUS_OK) {
    return testing::AssertionFailure()
           << "Failed to decode: " << WP2GetStatusMessage(status);
  }

  return Compare(original, decompressed, file_name,
                 GetExpectedDistortion(encoder_config));
}

WP2Status GetDistortionDiff(const std::string& file_name,
                            const Rectangle& window,
                            const EncoderConfig& encoder_config,
                            const DecoderConfig& decoder_config_a,
                            const DecoderConfig& decoder_config_b,
                            float disto_diff[5], MetricType metric) {
  ArgbBuffer original, cropped;
  WP2_CHECK_STATUS(ReadImage(GetTestDataPath(file_name).c_str(), &original));
  WP2_CHECK_STATUS(cropped.SetView(original, window));
  MemoryWriter writer;
  WP2_CHECK_STATUS(Encode(cropped, &writer, encoder_config));
  ArgbBuffer a, b;
  WP2_CHECK_STATUS(Decode(writer.mem_, writer.size_, &a, decoder_config_a));
  WP2_CHECK_STATUS(Decode(writer.mem_, writer.size_, &b, decoder_config_b));
  float disto_a[5], disto_b[5];
  WP2_CHECK_STATUS(a.GetDistortion(cropped, metric, disto_a));
  WP2_CHECK_STATUS(b.GetDistortion(cropped, metric, disto_b));
  for (int i = 0; i < 5; ++i) disto_diff[i] = disto_b[i] - disto_a[i];
  return WP2_STATUS_OK;
}

WP2Status GetDistortionDiff(const std::string& file_name,
                            const Rectangle& window,
                            const EncoderConfig& encoder_config_a,
                            const EncoderConfig& encoder_config_b,
                            const DecoderConfig& decoder_config,
                            float disto_diff[5], MetricType metric) {
  ArgbBuffer original, cropped;
  WP2_CHECK_STATUS(ReadImage(GetTestDataPath(file_name).c_str(), &original));
  WP2_CHECK_STATUS(cropped.SetView(original, window));
  MemoryWriter writer;
  ArgbBuffer a, b;
  WP2_CHECK_STATUS(Encode(cropped, &writer, encoder_config_a));
  WP2_CHECK_STATUS(Decode(writer.mem_, writer.size_, &a, decoder_config));
  writer.Reset();
  WP2_CHECK_STATUS(Encode(cropped, &writer, encoder_config_b));
  WP2_CHECK_STATUS(Decode(writer.mem_, writer.size_, &b, decoder_config));
  float disto_a[5], disto_b[5];
  WP2_CHECK_STATUS(a.GetDistortion(cropped, metric, disto_a));
  WP2_CHECK_STATUS(b.GetDistortion(cropped, metric, disto_b));
  for (int i = 0; i < 5; ++i) disto_diff[i] = disto_b[i] - disto_a[i];
  return WP2_STATUS_OK;
}

void DumpImage(const ArgbBuffer& image, const std::string& file_path,
               Rectangle rect) {
  ArgbBuffer buffer(WP2_Argb_32);
  if (image.format() == buffer.format()) {
    WP2_ASSERT_STATUS(buffer.SetView(image));
  } else {
    WP2_ASSERT_STATUS(buffer.ConvertFrom(image));
  }
  if (rect.GetArea() != 0) WP2_ASSERT_STATUS(buffer.SetView(buffer, rect));
  WP2_ASSERT_STATUS(SaveImage(buffer, file_path.c_str(), /*overwrite=*/true));
}

void DumpData(const void* data, size_t data_size,
              const std::string& file_path) {
  std::ofstream file(file_path.c_str(), std::ios::binary);
  file.write(reinterpret_cast<const char*>(data), data_size);
}

//------------------------------------------------------------------------------

namespace {

// Uniformly draws a random BlockSize whose width/height fit in bw/bh (in
// kMinBlockSize units).
BlockSize UniformBlockSizeDrawInBounds(uint32_t bw, uint32_t bh,
                                       UniformIntDistribution* const gen) {
  BlockSize dims[BLK_LAST + 1];
  uint32_t size = 0;
  for (uint32_t dim = 0; (BlockSize)dim != BLK_LAST; ++dim) {
    if (BlockWidth[dim] <= bw && BlockHeight[dim] <= bh) {
      dims[size++] = (BlockSize)dim;
    }
  }
  return dims[gen->Get(0u, size - 1)];
}

}  // namespace

//------------------------------------------------------------------------------

void CreatePartition(uint32_t w, uint32_t h, bool snapped,
                     UniformIntDistribution* const gen,
                     VectorNoCtor<Block>* const blocks) {
  const uint32_t bwidth = SizeBlocks(w), bheight = SizeBlocks(h);
  FrontMgrLexico manager;
  WP2_ASSERT_STATUS(manager.Init(ALL_RECTS, snapped, w, h));
  while (!manager.Done()) {
    // Find the beginning of the next block in lexicographic order.
    uint32_t bx = 0;
    uint32_t by = bheight;
    for (uint32_t x = 0; x < bwidth; ++x) {
      const uint32_t y = manager.GetOccupancy(x);
      if (y < by) {
        bx = x;
        by = y;
      }
    }
    // Find max width/height.
    uint32_t max_bw = 1, max_bh = std::min(kMaxBlockSize, bheight - by);
    while (bx + max_bw < bwidth && max_bw < kMaxBlockSize &&
           manager.GetOccupancy(bx + max_bw) == manager.GetOccupancy(bx)) {
      ++max_bw;
    }

    if (snapped) {
      const BlockSize snapped_block_size =
          GetSnappedBlockSize(bx, by, max_bw, max_bh);
      max_bw = BlockWidth[snapped_block_size];
      max_bh = BlockHeight[snapped_block_size];
    }
    const BlockSize dim = UniformBlockSizeDrawInBounds(max_bw, max_bh, gen);
    const uint32_t bh = BlockHeight[dim];

    const Block block(bx, by, (BlockSize)dim);

    if (!blocks->push_back(block)) assert(false);
    manager.Use(block);
    if (!manager.IsUsed(bx, by)) assert(false);
    if (manager.IsUsed(bx, by + bh)) assert(false);
  }
}

}  // namespace testutil
}  // namespace WP2

//------------------------------------------------------------------------------

namespace {

template <typename Type>
void PrintSetting(std::ostream& out, Type setting, Type default_setting,
                  const char setting_name[],
                  const char setting_type_name[] = "int") {
  out << std::endl
      << "config." << setting_name << " = (" << setting_type_name << ")"
      << static_cast<int>(setting) << ";";
  if (setting != default_setting) {
    out << "  // Default is " << static_cast<int>(default_setting);
  }
}

}  // namespace

std::ostream& operator<<(std::ostream& out, const WP2::EncoderConfig& c) {
  const WP2::EncoderConfig kD = WP2::EncoderConfig::kDefault;
  out << std::endl << "EncoderConfig config = EncoderConfig::kDefault;";
  PrintSetting(out, c.quality, kD.quality, "quality");
  PrintSetting(out, c.target_size, kD.target_size, "target_size");
  PrintSetting(out, c.target_psnr, kD.target_psnr, "target_psnr");
  PrintSetting(out, c.alpha_quality, kD.alpha_quality, "alpha_quality");
  PrintSetting(out, c.effort, kD.effort, "effort");
  PrintSetting(out, c.decoding_orientation, kD.decoding_orientation,
               "decoding_orientation", "Orientation");
  PrintSetting(out, c.create_preview, kD.create_preview, "create_preview",
               "bool");
  PrintSetting(out, c.transfer_function, kD.transfer_function,
               "transfer_function", "TransferFunction");
  PrintSetting(out, c.pass, kD.pass, "pass");
  PrintSetting(out, c.sns, kD.sns, "sns", "float");
  PrintSetting(out, c.error_diffusion, kD.error_diffusion, "error_diffusion");
  PrintSetting(out, c.segments, kD.segments, "segments");
  PrintSetting(out, c.partition_method, kD.partition_method, "partition_method",
               "PartitionMethod");
  PrintSetting(out, c.partition_set, kD.partition_set, "partition_set",
               "PartitionSet");
  PrintSetting(out, c.partition_snapping, kD.partition_snapping,
               "partition_snapping", "bool");
  PrintSetting(out, c.csp_type, kD.csp_type, "csp_type", "Csp");
  PrintSetting(out, c.uv_mode, kD.uv_mode, "uv_mode", "EncoderConfig::UVMode");
  PrintSetting(out, c.preprocessing, kD.preprocessing, "preprocessing");
  PrintSetting(out, c.preprocessing_strength, kD.preprocessing_strength,
               "preprocessing_strength");
  PrintSetting(out, c.use_random_matrix, kD.use_random_matrix,
               "use_random_matrix", "bool");
  PrintSetting(out, c.store_grain, kD.store_grain, "store_grain", "bool");
  PrintSetting(out, c.use_delta_palette, kD.use_delta_palette,
               "use_delta_palette", "bool");
  PrintSetting(out, c.thread_level, kD.thread_level, "thread_level");
  PrintSetting(out, c.low_memory, kD.low_memory, "low_memory", "bool");
  PrintSetting(out, c.use_neural_compression, kD.use_neural_compression,
               "use_neural_compression");
  return out << std::endl;
}

std::ostream& operator<<(std::ostream& out, const WP2::DecoderConfig& c) {
  const WP2::DecoderConfig kD = WP2::DecoderConfig::kDefault;
  out << std::endl << "DecoderConfig config = DecoderConfig::kDefault;";
  PrintSetting(out, c.thread_level, kD.thread_level, "thread_level");
  PrintSetting(out, c.enable_deblocking_filter, kD.enable_deblocking_filter,
               "enable_deblocking_filter", "bool");
  PrintSetting(out, c.enable_directional_filter, kD.enable_directional_filter,
               "enable_directional_filter", "bool");
  PrintSetting(out, c.enable_restoration_filter, kD.enable_restoration_filter,
               "enable_restoration_filter", "bool");
  PrintSetting(out, c.enable_alpha_filter, kD.enable_alpha_filter,
               "enable_alpha_filter", "bool");
  PrintSetting(out, c.grain_amplitude, kD.grain_amplitude, "grain_amplitude");
  PrintSetting(out, c.incremental_mode, kD.incremental_mode, "incremental_mode",
               "DecoderConfig::IncrementalMode");
  return out << std::endl;
}

//------------------------------------------------------------------------------
// this function is just here to offer a different address than GetCPUInfo,
// so that calling WP2xxxInit() really resets everything.
static bool FakeGetInfo(WP2CPUFeature feature) { return false; }

void WP2DspReset() {
  // first time, we save the original WP2GetCPUInfo, for later reset
  static WP2CPUInfo saved_cpu_info = nullptr;
  if (saved_cpu_info == nullptr) saved_cpu_info = WP2GetCPUInfo;

  // first we clear the local xxx_last_cpuinfo_used internal pointers
  WP2GetCPUInfo = FakeGetInfo;
  WP2EncDspInit();
  WP2DecDspInit();
  WP2ArgbConverterInit();
  WP2TransformInit();
  WP2::DblkFilterInit();
  WP2::DrctFilterInit();
  WP2::WienerFilterInit();
  WP2::GrainFilterInit();
  WP2::PredictionInit();
  WP2::ScoreDspInit();
  WP2SSIMInit();
  WP2PSNRInit();
  WP2AlphaInit();
  WP2QuantizeInit();
  WP2::ANSInit();
  WP2MathInit();

  WP2L::DecLDspInit();
  WP2L::DspInit();
  WP2L::EncLDspInit();

  // then, we reset the pointer-to-functions for good
  WP2GetCPUInfo = saved_cpu_info;

  for (auto& f : WP2ArgbConvertFrom) f = nullptr;
  for (auto& f : WP2ArgbConvertTo) f = nullptr;

  WP2YuvToCustom = nullptr;
  WP2AyuvToArgb32 = nullptr;
  WP2AyuvToARGB32 = nullptr;
  WP2AyuvToXRGB32 = nullptr;
  WP2YuvToArgb32 = nullptr;
  WP2AyuvToArgb38 = nullptr;
  WP2YuvToArgb38 = nullptr;

  WP2Transpose16b = nullptr;
  WP2Transpose32b = nullptr;
  for (uint32_t i = 0; i < 5; ++i) {
    WP2InvDct[i] = WP2InvAdst[i] = WP2InvHadamard[i] = nullptr;
    WP2FwdDct[i] = WP2FwdAdst[i] = WP2FwdHadamard[i] = nullptr;
  }
  for (uint32_t th : {0, 1, 2, 3, 4}) {
    for (uint32_t tw : {0, 1, 2, 3, 4}) {
      WP2InvDctCol[th][tw] = nullptr;
      WP2FwdDctCol[th][tw] = nullptr;
    }
  }
  WP2SlowDct8x8 = nullptr;

  WP2::GetBlockMinMax = nullptr;
  WP2::GetBlockMinMax_5x5 = nullptr;

  WP2::DeblockLine = nullptr;
  WP2::WouldDeblockLine = nullptr;
  WP2::FilterCopyIn = nullptr;
  WP2::FilterCopyOut = nullptr;
  WP2::MeasureFlatLengths = nullptr;

  WP2::CdefDirection4x4 = nullptr;
  WP2::CdefDirection8x8 = nullptr;
  WP2::CdefPad = nullptr;
  WP2::CdefFiltering = nullptr;

  WP2::WienerFilter = nullptr;

  WP2::AddGrain4x4 = nullptr;
  WP2::GenerateGrain4x4 = nullptr;

  for (auto& f : WP2::BasePredictors) f = nullptr;
  WP2::AnglePredInterpolate = nullptr;
  WP2::SubtractRow = nullptr;
  WP2::AddRow = nullptr;
  for (auto& f : WP2::SubtractBlock) f = nullptr;
  for (auto& f : WP2::AddBlock) f = nullptr;
  for (auto& f : WP2::AddBlockEq) f = nullptr;

  WP2::CflPredict = nullptr;

  WP2SSIMGet4x8u = nullptr;
  WP2SSIMGet8u = nullptr;
  WP2SSIMGet10s = nullptr;
  WP2SSIMGet12s = nullptr;

  WP2SSIMGetClipped4x8u = nullptr;
  WP2SSIMGetClipped8u = nullptr;
  WP2SSIMGetClipped10s = nullptr;
  WP2SSIMGetClipped12s = nullptr;

  WP2SumSquaredError8u = nullptr;
  WP2SumSquaredError16s = nullptr;
  WP2SumSquaredError4x8u = nullptr;
  WP2SumSquaredError3x8u = nullptr;
  WP2SumSquaredError4x16u = nullptr;

  WP2SumSquaredErrorBlock = nullptr;
  WP2SumSquaredErrorHalfBlock = nullptr;

  WP2HasValue8b = nullptr;
  WP2HasValue16b = nullptr;
  WP2HasOtherValue8b = nullptr;
  WP2HasOtherValue16b = nullptr;
  WP2HasOtherValue8b32b = nullptr;

  WP2::ANSUpdateCDF = nullptr;

  WP2::RasterAdvance = nullptr;
  WP2::RasterLoss = nullptr;
  WP2::RasterDraw = nullptr;

  WP2Quantize = nullptr;
  WP2Dequantize = nullptr;

  WP2Log2Slow = nullptr;
  WP2SLog2m1Slow = nullptr;
  WP2InnerProduct = nullptr;

  // decl_dsp
  for (auto& p : WP2L::PredictorsNonClampedAdd) p = nullptr;
  for (auto& p : WP2L::PredictorsClampedAdd) p = nullptr;
  for (auto& p : WP2L::PredictorsAngleAdd) p = nullptr;
  WP2L::AddGreenToBlueAndRed = nullptr;
  WP2L::YCoCgRToRGB = nullptr;
  WP2L::TransformColorInverse = nullptr;
  WP2L::MapColor = nullptr;

  // dspl
  for (auto& p : WP2L::PredictorsNonClamped) p = nullptr;
  for (auto& p : WP2L::PredictorsClamped) p = nullptr;
  for (auto& p : WP2L::PredictorsAngle) p = nullptr;

  // encl_dsp
  WP2L::SubtractGreenFromBlueAndRed = nullptr;
  WP2L::RGB2YCoCgR = nullptr;
  WP2L::TransformColor = nullptr;
  WP2L::CollectColorBlueTransforms = nullptr;
  WP2L::CollectColorRedTransforms = nullptr;
  WP2L::ExtraCost = nullptr;
  WP2L::ExtraCostCombined = nullptr;
  WP2L::CombinedShannonEntropy = nullptr;
  WP2L::BufferAdd = nullptr;
  for (auto& p : WP2L::PredictorsNonClampedSub) p = nullptr;
  for (auto& p : WP2L::PredictorsClampedSub) p = nullptr;
  for (auto& p : WP2L::PredictorsAngleSub) p = nullptr;
  WP2L::PredictorSubBlack = nullptr;
  WP2L::PredictorSubTop = nullptr;
  WP2L::PredictorSubLeft = nullptr;
}
