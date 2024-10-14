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
//  simple command line calling the WP2Encode function.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef HAVE_CONFIG_H
#include "src/wp2/config.h"
#endif

#include "examples/example_utils.h"
#include "examples/stopwatch.h"
#include "imageio/anim_image_dec.h"
#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/utils/csp.h"
#include "src/utils/orientation.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/debug.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace {

using WP2::ExUtilGetFloat;
using WP2::ExUtilGetFloats;
using WP2::ExUtilGetInt;
using WP2::ExUtilGetUInt;
using WP2::ExUtilTryGetInt;
using WP2::ProgramOptions;
using WP2::SPrintf;

//------------------------------------------------------------------------------

constexpr uint32_t kDefaultThreadLevel = 20;  // When -mt is set.

struct Params {
  bool allow_overwrite = false;
  WP2::LogLevel log_level = WP2::LogLevel::DEFAULT;
  bool short_output = false;
  int print_distortion = -1;  // -1=off, >=0 -> (int)MetricType
  bool print_info = false;
  bool no_metadata = false;
  WP2::Metadata forced_metadata;
  int bit_trace = 0;  // 0=none, 1=trace in bits, 2=trace in bytes
  uint32_t bit_trace_level = 0;
  bool show_histograms = false;  // show symbol histograms
  WP2::EncoderConfig config;
  bool loop_forever_was_specified = false;
  bool loop_forever = true;
  WP2::PictureHint pic_hint = WP2::HINT_NONE;
  bool crop = false;
  WP2::Rectangle crop_area;
};

//------------------------------------------------------------------------------

void HelpShort() {
  printf("Usage:\n\n");
  printf("   cwp2 [options] -q quality input.png -o output.wp2\n\n");
  printf("where quality is between 0 (poor) to 100 (very good).\n");
  printf("Typical value is around 80.\n\n");
  printf("Try -h for an exhaustive list of options.\n");
}

void HelpLong() {
  ProgramOptions opt;
  opt.Add("Usage:");
  opt.Add("  cwp2 [options] in_file [-o out_file]");
  opt.Add("");
  opt.Add("Options:");
  opt.Add("-q <float>",
          SPrintf("quality factor (0:small..100:big), default=%1.1f",
                  WP2::EncoderConfig::kDefault.quality));
  opt.Add("-alpha_q <float>",
          SPrintf("alpha quality factor (0:small..100:big), default=%1.1f",
                  WP2::EncoderConfig::kDefault.alpha_quality));
  opt.Add("-keep_unmultiplied",
          "No premultiplication is done (default when -q 100 -alpha_q 100).");
  opt.Add("-target_size <int>", "target size (in bytes)");
  opt.Add("-target_psnr <float>", "target PSNR (in dB. typically: 42)");
  opt.Add("-effort <int>",
          SPrintf("compression effort (%d:fast..%d:slower/better), default=%d",
                  WP2::kMinEffort, WP2::kMaxEffort,
                  WP2::EncoderConfig::kDefault.effort));
  opt.Add("-av1", "use lossy AV1 internally instead of lossy WP2");
  opt.Add("-tile_shape <int>",
          SPrintf("tile shape (0=128, 1=256, 2=512, 3=wide, default = %d)",
                  WP2::EncoderConfig::kDefault.tile_shape));
  opt.Add("-f [<str> <int>]",
          "create an animation from frames "
          "(alternate images and durations in ms)");
  opt.Add("-no_loop", "the animation will play once then stop");
  opt.Add("-loop_forever",
          "the animation will play in a loop "
          "(if it cannot be deduced from input, default is loop_forever)");
  opt.Add("-csp <int>", "color space (0=YCoCg, 1=YCbCr 2=Custom 3=YIQ)");
  opt.Add("-uv_mode <int>", "UV subsampling mode (0=Adapt,1=420,2=444,3=Auto)");
  opt.Add("-pass <int>", "entropy-analysis passes (1..10)");
  opt.Add("-nofilter", "disable all filters during decoding");
  opt.Add("-notransforms", "disable transforms other than DCT");
  opt.Add("-nopreds", "disable predictors other than DC");
  opt.Add("-nosplit_tf", "disable split transforms");
  opt.Add("-preview <file>", "add binary file as preview data directly");
  opt.Add("-create_preview", "add a tiny preview to the bitstream");
  opt.Add("-orientation <int>",
          "number of clockwise rotations by 90 degrees during decoding");
  opt.Add("-tf <int>", "transfer function");
  opt.Add("-nometadata", "ignore source metadata");
  opt.Add("-metadata [dst] file",
          "force metadata with content of 'file'. "
          "'dst' is one of iccp, exif or xmp\n");
  opt.Add("-crop <x> <y> <w> <h>", "crop picture with the given rectangle");
  opt.Add("-segments <int>", "number of segments");
  opt.Add("-segment_mode <mode>",
          "segment id mode (one of 'auto', 'explicit', 'implicit')");
  opt.Add("-quants <floats,...>", "force per-segment quantization factors");
  opt.Add("-alt_tuning on/off", "adapt config to the image and quality range");
  opt.Add("-pm <int>",
          SPrintf("partition method (0..%d)", WP2::NUM_PARTITION_METHODS - 1));
  opt.Add("-ps <int>",
          SPrintf("partition set (0..%d)", WP2::NUM_PARTITION_SETS - 1));
  opt.Add("-partition_snapping", "force quadtree-like partitioning");
  opt.Add("-[no_]perceptual", "turn perceptual tuning on/off");
  opt.Add("-sns <float>", "segmentation strength (0=off..100)");
  opt.Add("-diffusion <int>", "error diffusion strength (0=off..100)");
  opt.Add("-grain", "analyze and store grain information");
  opt.Add("-rnd_mtx", "enable use of random matrices");
  opt.AddMetricOptions();
  opt.Add("-mt [int]", SPrintf("enable multi-threading, with %u extra threads "
                               "if unspecified (0 to disable, default is %u)",
                               kDefaultThreadLevel,
                               WP2::EncoderConfig::kDefault.thread_level));
  opt.Add("-d <file>", "save the decoded result as 'file' (single image case)");
  opt.Add("");
  opt.Add("-r", "recurse into input directories");
  opt.Add("-frames",
          "create an animation (parse input file names for frame order and "
          "duration: \"frame[index]_[duration]ms.png\")");
  opt.Add("-inplace",
          "output files in the same directories as input files (replaces -o)");
  opt.Add("-force",
          "overwrite destination if it exists "
          "(default=off unless out_file is explicit)");
  opt.Add("-short", "condense printed message");
  opt.Add("-info", "print summary and statistics");
  opt.Add("-progress", "display encoding progression");
  opt.Add("-quiet", "don't print anything");
#ifdef WP2_BITTRACE
  opt.Add("-bt / -BT", "set bit trace level to 1 / 2");
  opt.Add("-histos", "display histograms per symbol when -bt is set");
#endif
  opt.Add("-neural",
          "enable neural compression and specify "
          "directory with enc/decoder graphdefs");
  opt.Add("");
  opt.AddMemoryOptionSection();
  opt.AddSystemOptionSection();
  opt.Print();
}

//------------------------------------------------------------------------------

struct InputFrame {
  const std::string* file_path;
  uint32_t index;
  uint32_t duration;
  bool operator<(const InputFrame& o) const { return (index < o.index); }
};

// Gets frames to encode as an animation (-frames).
WP2_NO_DISCARD WP2Status GetFramesToEncode(
    WP2::LogLevel log_level, std::vector<uint32_t>* const durations,
    WP2::FileList* const frames_paths) {
  // Keep only files with a matching frame name.
  std::vector<InputFrame> input_frames;
  for (const std::string& file_path : *frames_paths) {
    const std::string file_name = WP2::GetFileName(file_path);
    uint32_t frame_index;
    uint32_t duration_ms;
    const int num_assigned_variables =
        sscanf(file_name.c_str(), "frame%u_%ums", &frame_index, &duration_ms);
    if (num_assigned_variables == 2) {
      CHECK_TRUE(input_frames.size() < WP2::kMaxNumFrames,
                 "Error! Frame count can not exceed %u (file '%s')",
                 WP2::kMaxNumFrames, file_path.c_str());
      input_frames.emplace_back(
          InputFrame({&file_path, frame_index, duration_ms}));
    } else if (log_level >= WP2::LogLevel::VERBOSE) {
      fprintf(stderr, "Ignoring unformatted file %s.\n", file_path.c_str());
    }
  }

  // Sort frames by index (no guarantee that it's ordered even in a folder).
  std::sort(input_frames.begin(), input_frames.end());

  // Parse them in order until index is no longer incremental.
  durations->clear();
  WP2::FileList new_list;
  uint32_t last_index = 0;
  for (const InputFrame& input_frame : input_frames) {
    CHECK_TRUE(last_index == 0 || input_frame.index > last_index,
               "Error! Duplicate frame index %u at '%s'", input_frame.index,
               input_frame.file_path->c_str());
    new_list.push_back(*input_frame.file_path);
    durations->push_back(input_frame.duration);
    last_index = input_frame.index;
  }
  std::swap(new_list, *frames_paths);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Gets files to encode from input arguments.
WP2_NO_DISCARD WP2Status GetFilesToEncode(
    const WP2::FileList& paths, bool recursive_input, WP2::LogLevel log_level,
    bool input_frames, std::vector<uint32_t>* const frame_durations,
    WP2::FileList* const files) {
  for (const std::string& path : paths) {
    if (WP2::IsDirectory(path)) {
      if (recursive_input) {
        WP2::FileList files_in_dir;
        CHECK_TRUE(WP2::GetFilesIn(path, &files_in_dir, /*recursive=*/true),
                   "Error! Opening directory %s failed.", path.c_str());

        // Only keep files with a handled extension.
        for (const std::string& file : files_in_dir) {
          const WP2::FileFormat format =
              WP2::GetFormatFromExtension(file.c_str());
          if (format != WP2::FileFormat::UNSUPPORTED &&
              format != WP2::FileFormat::BMP) {
            files->push_back(file);
          } else if (log_level >= WP2::LogLevel::VERBOSE) {
            fprintf(stderr, "Ignoring file %s.\n", file.c_str());
          }
        }
      } else if (log_level >= WP2::LogLevel::DEFAULT) {
        fprintf(stderr, "Ignoring directory %s because -r was not specified.\n",
                path.c_str());
      }
    } else {
      // No extension checking for explicit input files.
      files->push_back(path);
    }
  }

  if (input_frames) {
    return GetFramesToEncode(log_level, frame_durations, files);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Erases or replaces parts of 'metadata' depending on 'params'.
WP2_NO_DISCARD
WP2Status ForceMetadata(const Params& params, WP2::Metadata* const metadata) {
  if (params.no_metadata) metadata->Clear();
  if (!params.forced_metadata.iccp.IsEmpty()) {
    WP2_CHECK_STATUS(metadata->iccp.CopyFrom(params.forced_metadata.iccp.bytes,
                                             params.forced_metadata.iccp.size));
  }
  if (!params.forced_metadata.xmp.IsEmpty()) {
    WP2_CHECK_STATUS(metadata->xmp.CopyFrom(params.forced_metadata.xmp.bytes,
                                            params.forced_metadata.xmp.size));
  }
  if (!params.forced_metadata.exif.IsEmpty()) {
    WP2_CHECK_STATUS(metadata->exif.CopyFrom(params.forced_metadata.exif.bytes,
                                             params.forced_metadata.exif.size));
  }
  return WP2_STATUS_OK;
}

// Crops the 'buffer' depending on 'params'.
WP2_NO_DISCARD
WP2Status ApplyModifications(const char* const file, const Params& params,
                             WP2::ArgbBuffer* const buffer) {
  if (params.crop) {
    CHECK_STATUS(buffer->SetView(*buffer, params.crop_area),
                 "Error! Cropping operation of '%s' failed.", file);
  }
  return WP2_STATUS_OK;
}

// Encodes 'frames' into 'writer'.
WP2_NO_DISCARD
WP2Status CompressAnimation(const WP2::FileList& frames,
                            const std::vector<uint32_t>& frame_durations,
                            const Params& params,
                            std::vector<WP2::ArgbBuffer>* const refs,
                            WP2::ArgbBuffer* const rgb_buffer,
                            WP2::Writer* const writer) {
  assert(frames.size() == frame_durations.size());
  WP2::AnimationEncoder animation_encoder;
  WP2::Metadata metadata;

  // Read and add all frames.
  double start = GetStopwatchTime();
  for (size_t i = 0; i < frames.size(); ++i) {
    const char* const file_path = frames[i].c_str();
    WP2::Data data;
    CHECK_STATUS(IoUtilReadFile(file_path, &data),
                 "Error! Cannot open input file '%s'", file_path);
    CHECK_STATUS(WP2::ReadImage(data.bytes, data.size, rgb_buffer,
                                WP2::FileFormat::AUTO, params.log_level),
                 "Error! Cannot read input file '%s'", file_path);
    if (i == 0) {
      swap(metadata, rgb_buffer->metadata_);  // save for later call
    } else if (!rgb_buffer->metadata_.IsEmpty()) {
      fprintf(stderr, "Warning: metadata from frame #%u will be ignored!\n",
              (uint32_t)(i + 1));
    }
    WP2_CHECK_STATUS(ApplyModifications(file_path, params, rgb_buffer));

    if (refs != nullptr) {
      refs->emplace_back(rgb_buffer->format());
      CHECK_STATUS(OrientateBuffer(params.config.decoding_orientation,
                                   *rgb_buffer, &refs->back()),
                   "Error! Image copy failed.");
    }

    CHECK_STATUS(animation_encoder.AddFrame(*rgb_buffer, frame_durations[i]),
                 "Error! Cannot add input frame '%s'", file_path);
  }
  if (params.log_level >= WP2::LogLevel::VERBOSE) {
    const double elapsed = GetStopwatchTime() - start;
    fprintf(stderr, "Time to read all input frames: %.3fs\n", elapsed);
  }
  WP2_CHECK_STATUS(ForceMetadata(params, &metadata));

  // Encode animation.
  start = GetStopwatchTime();
  CHECK_STATUS(animation_encoder.Encode(writer, params.config,
                                        params.loop_forever, metadata),
               "Error! Cannot encode animation as WP2");
  if (params.log_level >= WP2::LogLevel::VERBOSE) {
    const double encode_time = GetStopwatchTime() - start;
    fprintf(stderr, "Time to encode animation: %.3fs\n", encode_time);
  }
  return WP2_STATUS_OK;
}

// Encodes 'file' into 'writer'.
WP2_NO_DISCARD
WP2Status CompressImage(const char* const file_path, const Params& params,
                        std::vector<WP2::ArgbBuffer>* const refs,
                        WP2::ArgbBuffer* const buffer,
                        WP2::Writer* const writer) {
  // Read input image.
  double start = GetStopwatchTime();
  WP2::ImageReader image_reader(file_path, buffer, WP2::FileFormat::AUTO,
                                params.log_level);
  // 'buffer' must not be modified between calls to ReadFrame(), so we use a
  // proxy.
  WP2::ArgbBuffer frame(buffer->format());
  WP2::Metadata metadata;
  bool is_last_frame = false;
  uint32_t duration_ms = 0;
  WP2::AnimationEncoder animation_encoder;

  for (uint32_t i = 0; !is_last_frame; ++i) {
    CHECK_TRUE(i < WP2::kMaxNumFrames,
               "Error! There are too many frames in '%s'", file_path);
    CHECK_STATUS(image_reader.ReadFrame(&is_last_frame, &duration_ms),
                 "Error! Cannot read input '%s'", file_path);
    CHECK_STATUS(frame.SetView(*buffer), "Error! Cannot set view.");
    if (i == 0) {
      swap(metadata, buffer->metadata_);  // save for later call
    } else if (!buffer->metadata_.IsEmpty()) {
      fprintf(stderr, "Warning: metadata from frame #%u will be ignored!\n",
              i + 1);
    }
    WP2_CHECK_STATUS(ApplyModifications(file_path, params, &frame));

    if (refs != nullptr) {
      refs->emplace_back(frame.format());
      CHECK_STATUS(OrientateBuffer(params.config.decoding_orientation, frame,
                                   &refs->back()),
                   "Error! Image copy failed.");
    }

    // Infinite duration means it's a still image.
    if (duration_ms != WP2::ImageReader::kInfiniteDuration) {
      CHECK_STATUS(animation_encoder.AddFrame(frame, duration_ms),
                   "Error! Cannot add input frame %u from '%s'", i, file_path);
    } else {
      assert(is_last_frame);
    }
  }
  if (params.log_level >= WP2::LogLevel::VERBOSE) {
    const double elapsed = GetStopwatchTime() - start;
    fprintf(stderr, "Time to read input image: %.3fs\n", elapsed);
  }
  WP2_CHECK_STATUS(ForceMetadata(params, &metadata));

  // Encode image.
  start = GetStopwatchTime();
  if (duration_ms == WP2::ImageReader::kInfiniteDuration) {
    swap(metadata, frame.metadata_);
    CHECK_STATUS(WP2::Encode(frame, writer, params.config, params.pic_hint),
                 "Error! Cannot encode '%s' as WP2", file_path);
  } else {
    const bool loop_forever = params.loop_forever_was_specified
                                  ? params.loop_forever
                                  : (image_reader.GetLoopCount() != 1);
    CHECK_STATUS(
        animation_encoder.Encode(writer, params.config, loop_forever, metadata),
        "Error! Cannot encode '%s' as WP2", file_path);
  }
  if (params.log_level >= WP2::LogLevel::VERBOSE) {
    const double encode_time = GetStopwatchTime() - start;
    fprintf(stderr, "Time to encode picture: %.3fs\n", encode_time);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

uint32_t GetWidth(const Params& params, const WP2::ArgbBuffer& argb_buffer) {
  return params.crop ? params.crop_area.width : argb_buffer.width();
}
uint32_t GetHeight(const Params& params, const WP2::ArgbBuffer& argb_buffer) {
  return params.crop ? params.crop_area.height : argb_buffer.height();
}

void PrintSize(const std::string& out_file, size_t out_size, float num_pixels,
               bool short_output) {
  const float bpp = 8.f * out_size / num_pixels;
  printf("output size:  %u (%.2f bpp)", (uint32_t)out_size, bpp);
  if (out_file.empty()) {
    printf("    [not saved]\n");
  } else if (short_output) {
    printf("    [saved]\n");
  } else {
    printf("    [%s]\n", out_file.c_str());
  }
}

WP2_NO_DISCARD
WP2Status SaveImage(const WP2::MemoryWriter& memory_writer,
                    const Params& params, const std::string& out_file,
                    uint32_t width, uint32_t height) {
  if (!out_file.empty()) {
    const WP2Status status =
        WP2::IoUtilWriteFile(memory_writer.mem_, memory_writer.size_,
                             out_file.c_str(), params.allow_overwrite);
    CHECK_STATUS(
        status, "Error while saving file '%s'%s", out_file.c_str(),
        (status == WP2_STATUS_BAD_WRITE)
            ? ""
            : " (hint: use -force flag to overwrite an existing file)");
  }

  if (params.bit_trace > 0) {
    WP2_CHECK_STATUS(WP2::CollectBitTraces(
        memory_writer.mem_, memory_writer.size_, params.bit_trace,
        params.bit_trace_level, params.short_output, params.show_histograms));
  }

  if (params.log_level >= WP2::LogLevel::DEFAULT && params.print_info) {
    printf("%s",
           WP2::PrintSummary(memory_writer.mem_, memory_writer.size_).c_str());
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace

#define NEED_ARGS(nb)                                                   \
  do {                                                                  \
    CHECK_TRUE(c + (nb) < argc,                                         \
               "Expected at least %d arguments after option %s!", (nb), \
               argv[c]);                                                \
  } while (false)

int main(int argc, char* argv[]) {
  CHECK_TRUE(WP2CheckVersion(), "Error! Library version mismatch!");

  WP2::FileList in_paths;
  std::vector<uint32_t> frame_durations;  // It's an animation if at least one.
  std::string out_path;
  std::string decoded_file;
  const char* preview_file = nullptr;
  WP2::Data preview;

  WP2::DecoderConfig dec_config;
  WP2::DecoderInfo dec_info;
  dec_config.info = &dec_info;
  WP2::CSPTransform transform;
  dec_info.csp = &transform;

  bool recursive_input = false;
  bool input_frames = false;
  bool inplace_output = false;
  Params params = {};
  WP2::EncoderInfo enc_info;
  bool report_progress = false;

  if (argc == 1) {
    HelpShort();
    return 0;
  }

  for (int c = 1; c < argc; ++c) {
    bool parse_error = false;
    if (!strcmp(argv[c], "-h") || !strcmp(argv[c], "-help")) {
      HelpLong();
      return 0;
    } else if (!strcmp(argv[c], "-o")) {
      NEED_ARGS(1);
      CHECK_TRUE(out_path.empty() && !inplace_output,
                 "Error! Output was already specified.");
      out_path = argv[++c];
    } else if (!strcmp(argv[c], "-d")) {
      NEED_ARGS(1);
      decoded_file = argv[++c];
    } else if (!strcmp(argv[c], "-r")) {
      recursive_input = true;
    } else if (!strcmp(argv[c], "-frames")) {
      CHECK_TRUE(!inplace_output,
                 "Error! -frames is not compatible with -inplace.");
      input_frames = true;
    } else if (!strcmp(argv[c], "-inplace")) {
      CHECK_TRUE(!input_frames,
                 "Error! -inplace is not compatible with -frames.");
      CHECK_TRUE(out_path.empty(), "Error! Output was already specified.");
      inplace_output = true;
    } else if (!strcmp(argv[c], "-force")) {
      params.allow_overwrite = true;
    } else if (!strcmp(argv[c], "-short")) {
      params.short_output = true;
    } else if (!strcmp(argv[c], "-bt") || !strcmp(argv[c], "-BT")) {
      params.bit_trace = !strcmp(argv[c], "-bt") ? 1 : 2;
      if (c + 1 < argc) {
        if (isdigit(argv[c + 1][0])) {
          params.bit_trace_level = ExUtilGetUInt(argv[++c], &parse_error);
        } else {
          params.bit_trace_level = 0;
        }
      }
    } else if (!strcmp(argv[c], "-histos")) {
      params.show_histograms = true;
    } else if ((!strcmp(argv[c], "-z") || !strcmp(argv[c], "-effort"))) {
      NEED_ARGS(1);
      params.config.effort = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-av1")) {
      params.config.use_av1 = true;
    } else if (!strcmp(argv[c], "-tile_shape")) {
      NEED_ARGS(1);
      params.config.tile_shape =
          (WP2::TileShape)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-q")) {
      NEED_ARGS(1);
      params.config.quality = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-quants")) {
      NEED_ARGS(1);
      ExUtilGetFloats(argv[++c], params.config.segment_factors,
                      WP2::kMaxNumSegments, &parse_error);
    } else if (!strcmp(argv[c], "-alt_tuning")) {
      NEED_ARGS(1);
      ++c;
      if (!strcmp(argv[c], "on")) {
        params.config.enable_alt_tuning = true;
      } else if (!strcmp(argv[c], "off")) {
        params.config.enable_alt_tuning = false;
      } else {
        fprintf(stderr, "Unsupported -alt_tuning %s\n", argv[c]);
        parse_error = true;
      }
    } else if (!strcmp(argv[c], "-alpha_q")) {
      NEED_ARGS(1);
      params.config.alpha_quality = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-keep_unmultiplied")) {
      params.config.keep_unmultiplied = true;
    } else if (!strcmp(argv[c], "-csp")) {
      NEED_ARGS(1);
      params.config.csp_type = (WP2::Csp)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-uv_mode")) {
      NEED_ARGS(1);
      params.config.uv_mode =
          (WP2::EncoderConfig::UVMode)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-pm")) {
      NEED_ARGS(1);
      params.config.partition_method =
          (WP2::PartitionMethod)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-ps")) {
      NEED_ARGS(1);
      params.config.partition_set =
          (WP2::PartitionSet)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-partition_snapping")) {
      params.config.partition_snapping = true;
    } else if (!strcmp(argv[c], "-lossless")) {
      params.config.quality = 100;
    } else if (!strcmp(argv[c], "-transfer")) {
      NEED_ARGS(1);
      const int tfunction = ExUtilGetInt(argv[++c], &parse_error);
      params.config.transfer_function = (WP2::TransferFunction)tfunction;
    } else if (!strcmp(argv[c], "-create_preview")) {
      params.config.create_preview = true;
    } else if (!strcmp(argv[c], "-preview")) {
      NEED_ARGS(1);
      preview_file = argv[++c];
    } else if (!strcmp(argv[c], "-target_size")) {
      NEED_ARGS(1);
      params.config.target_size = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-target_psnr")) {
      NEED_ARGS(1);
      params.config.target_psnr = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-orientation")) {
      NEED_ARGS(1);
      params.config.decoding_orientation =
          (WP2::Orientation)(ExUtilGetUInt(argv[++c], &parse_error) & 7);
    } else if (!strcmp(argv[c], "-no_loop")) {
      params.loop_forever = false;
      params.loop_forever_was_specified = true;
    } else if (!strcmp(argv[c], "-loop_forever")) {
      params.loop_forever = true;
      params.loop_forever_was_specified = true;
    } else if (!strcmp(argv[c], "-crop")) {
      NEED_ARGS(4);
      params.crop = true;
      params.crop_area.x = (uint32_t)ExUtilGetInt(argv[++c], &parse_error);
      params.crop_area.y = (uint32_t)ExUtilGetInt(argv[++c], &parse_error);
      params.crop_area.width = (uint32_t)ExUtilGetInt(argv[++c], &parse_error);
      params.crop_area.height = (uint32_t)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-segments")) {
      NEED_ARGS(1);
      params.config.segments = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-segment_mode")) {
      NEED_ARGS(1);
      ++c;
      if (!strcmp(argv[c], "auto")) {
        params.config.segment_id_mode = WP2::EncoderConfig::SEGMENT_ID_AUTO;
      } else if (!strcmp(argv[c], "explicit")) {
        params.config.segment_id_mode = WP2::EncoderConfig::SEGMENT_ID_EXPLICIT;
      } else if (!strcmp(argv[c], "implicit")) {
        params.config.segment_id_mode = WP2::EncoderConfig::SEGMENT_ID_IMPLICIT;
      } else {
        fprintf(stderr, "Unsupported segment mode %s\n", argv[c]);
        parse_error = true;
      }
    } else if (!strcmp(argv[c], "-sns")) {
      NEED_ARGS(1);
      params.config.sns = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-diffusion")) {
      NEED_ARGS(1);
      params.config.error_diffusion = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-rnd_mtx")) {
      params.config.use_random_matrix = true;
    } else if (!strcmp(argv[c], "-grain")) {
      params.config.store_grain = true;
    } else if (!strcmp(argv[c], "-perceptual")) {
      params.config.tune_perceptual = true;
    } else if (!strcmp(argv[c], "-no_perceptual")) {
      params.config.tune_perceptual = false;
    } else if (!strcmp(argv[c], "-mt")) {
      if (c + 1 < argc &&
          ExUtilTryGetInt(argv[c + 1], &params.config.thread_level)) {
        ++c;
      } else {
        params.config.thread_level = kDefaultThreadLevel;
      }
    } else if (!strcmp(argv[c], "-low_memory")) {
      params.config.low_memory = true;
    } else if (!strcmp(argv[c], "-nofilter")) {
      dec_config.enable_deblocking_filter = false;
      dec_config.enable_directional_filter = false;
      dec_config.enable_restoration_filter = false;
    } else if (!strcmp(argv[c], "-notransforms")) {
      enc_info.disable_transforms = true;
      params.config.info = &enc_info;
    } else if (!strcmp(argv[c], "-nopreds")) {
      enc_info.disable_preds = true;
      params.config.info = &enc_info;
    } else if (!strcmp(argv[c], "-nosplit_tf")) {
      enc_info.disable_split_tf = true;
      params.config.info = &enc_info;
    } else if (!strcmp(argv[c], "-pass")) {
      NEED_ARGS(1);
      params.config.pass = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-pre")) {
      NEED_ARGS(1);
      params.config.preprocessing = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-nometadata")) {
      params.no_metadata = true;
    } else if (!strcmp(argv[c], "-metadata")) {
      NEED_ARGS(2);
      WP2::Data* dst = nullptr;
      ++c;
      if (!strcmp(argv[c], "iccp")) dst = &params.forced_metadata.iccp;
      if (!strcmp(argv[c], "xmp")) dst = &params.forced_metadata.xmp;
      if (!strcmp(argv[c], "exif")) dst = &params.forced_metadata.exif;
      CHECK_TRUE(dst != nullptr, "Error! unknown metadata type '%s'.\n",
                 argv[c]);
      ++c;
      CHECK_STATUS(IoUtilReadFile(argv[c], dst),
                   "Error! Can't read data for '%s' metadata.", argv[c - 1]);
    } else if (!strcmp(argv[c], "-progress")) {
      report_progress = true;
    } else if (!strcmp(argv[c], "-quiet")) {
      params.log_level = WP2::LogLevel::QUIET;
    } else if (!strcmp(argv[c], "-info")) {
      params.print_info = true;
    } else if (!strcmp(argv[c], "-hint")) {
      NEED_ARGS(1);
      ++c;
      if (!strcmp(argv[c], "default")) {
        params.pic_hint = WP2::HINT_NONE;
      } else if (!strcmp(argv[c], "photo")) {
        params.pic_hint = WP2::HINT_PHOTO;
      } else if (!strcmp(argv[c], "picture")) {
        params.pic_hint = WP2::HINT_PICTURE;
      } else if (!strcmp(argv[c], "drawing")) {
        params.pic_hint = WP2::HINT_DRAWING;
      } else if (!strcmp(argv[c], "icon")) {
        params.pic_hint = WP2::HINT_ICON;
      } else if (!strcmp(argv[c], "text")) {
        params.pic_hint = WP2::HINT_TEXT;
      } else {
        fprintf(stderr, "Error! Unrecognized picture hint: %s\n", argv[c]);
        parse_error = true;
      }
    } else if (!strcmp(argv[c], "-v")) {
      params.log_level = WP2::LogLevel::VERBOSE;
    } else if (!strcmp(argv[c], "-neural")) {
      NEED_ARGS(1);
      // The -neural argument takes a path to a directory that holds enc/decoder
      // graphdefs in subdirectories, one for each quality level. Specifically,
      // the full path to a graphdef is formed as:
      //
      // path = sprintf("%s/%02d/encoder.pbbin", base_path, quality)
      //
      // where:
      //  base_path = argument to -neural,
      //  the subdirectory is e.g., "02" if quality == 2,
      //  and encoder.pbbin & decoder.pbbin are hard-coded file names.
      params.config.use_neural_compression = 1;
      params.config.graphdef_path = argv[++c];
    } else if (!strcmp(argv[c], "--")) {
      CHECK_TRUE(frame_durations.empty(),
                 "Error! Can't specify another input when -f is specified.");
      if (c + 1 < argc) in_paths.emplace_back(argv[++c]);
    } else if (!strcmp(argv[c], "-f")) {
      CHECK_TRUE(in_paths.empty(),
                 "Error! Can't specify another input when -f is specified.");
      while (c + 2 < argc) {  // Parse as long as it's not another option.
        if (!strcmp(argv[c + 1], "--")) {
          NEED_ARGS(3);
          ++c;
        } else if (argv[c + 1][0] == '-') {
          break;
        }
        in_paths.emplace_back(argv[c + 1]);
        frame_durations.push_back(ExUtilGetUInt(argv[c + 2], &parse_error));
        c += 2;
      }
      CHECK_TRUE(!in_paths.empty(), "Error! Missing frame after -f.");
    } else if (argv[c][0] == '-') {
      bool must_stop;
      WP2::MetricType metric_type = WP2::NUM_METRIC_TYPES;
      int skip;
      if (ProgramOptions::ParseSystemOptions(argv[c], &must_stop)) {
        if (must_stop) return 0;
      } else if (ProgramOptions::ParseMetricOptions(argv[c], &metric_type)) {
        params.print_distortion = (int)metric_type;
      } else if (ProgramOptions::ParseMemoryOptions(argv + c, argc - c, skip)) {
        c += skip - 1;
      } else {
        fprintf(stderr, "Error! Unknown option '%s'\n", argv[c]);
        HelpLong();
        return 1;
      }
    } else {
      CHECK_TRUE(frame_durations.empty(),
                 "Error! Can't specify another input when -f is specified.");
      in_paths.emplace_back(argv[c]);
    }

    if (parse_error) {
      HelpLong();
      return 1;
    }
  }

  // Parse input directories to have the full list of files.
  WP2::FileList in_files;
  CHECK_STATUS(GetFilesToEncode(in_paths, recursive_input, params.log_level,
                                input_frames, &frame_durations, &in_files),
               "No input file specified!");

  CHECK_TRUE(preview_file == nullptr || (in_files.size() == 1),
             "Can only use -preview option with a single input image.");
  if (preview_file != nullptr) {
    CHECK_STATUS(IoUtilReadFile(preview_file, &preview),
                 "Error! Cannot open preview file '%s'", preview_file);
    params.config.preview_size = preview.size;
    params.config.preview = preview.bytes;
  }

  // Don't compute distortion in quiet mode.
  if (params.log_level == WP2::LogLevel::QUIET) params.print_distortion = -1;
  if (params.log_level == WP2::LogLevel::QUIET) report_progress = false;

  // Check for unsupported command line options for lossless mode and log
  // warning for such options.
  if (params.log_level >= WP2::LogLevel::DEFAULT &&
      params.config.quality > 95) {
    if (params.config.target_size > 0 || params.config.target_psnr > 0) {
      fprintf(stderr,
              "Encoding for specified size or PSNR is not supported "
              "for ~lossless encoding. Ignoring such option(s)!\n");
    }
  }
  // If a target size or PSNR was given, but somehow the -pass option was
  // omitted, force a reasonable value.
  if (params.config.target_size > 0 || params.config.target_psnr > 0) {
    if (params.config.pass == 1) params.config.pass = 6;
  }

  CHECK_TRUE(params.config.IsValid(),
             "Error! Invalid configuration. "
             "Some parameters are erroneous.");

  const bool is_animation = !frame_durations.empty();
  const bool output_to_directory =
      (inplace_output || WP2::IsDirectory(out_path));
  // Destination is explicitly specified, allow overwriting it.
  if (!output_to_directory) params.allow_overwrite = true;

  // Reuse these to do fewer memory allocations.
  WP2::MemoryWriter writer;
  WP2::Counter counter;

  // Decide whether to keep samples unmultiplied or to discard color information
  // in translucent pixels.
  if (params.config.quality == 100 && params.config.alpha_quality == 100) {
    // The user probably expects true lossless encoding.
    params.config.keep_unmultiplied = true;
  }
  // The action to premultiply could be left to the encoder but doing it now may
  // save one conversion.
  WP2::ArgbBuffer rgb_buffer(params.config.keep_unmultiplied ? WP2_ARGB_32
                                                             : WP2_Argb_32);

  if (is_animation) {
    CHECK_TRUE(!output_to_directory,
               "Error! Animation output file must be explicitly specified.");

    const std::string& out_file = out_path;
    const bool write_to_memory =
        (!out_file.empty() || params.print_distortion >= 0 ||
         params.print_info || params.bit_trace > 0);

    WP2::MultiProgressPrinter progress_printer;
    if (report_progress) {
      progress_printer.Init(/*num_jobs=*/1);
      params.config.progress_hook = progress_printer.GetProgressHook(0);
    }

    size_t out_size;
    if (write_to_memory) {
      if (params.print_distortion >= 0) {
        std::vector<WP2::ArgbBuffer> refs;
        WP2_CHECK_STATUS(CompressAnimation(in_files, frame_durations, params,
                                           &refs, &rgb_buffer, &writer));

        WP2_CHECK_STATUS(PrintDistortion(
            refs, writer.mem_, writer.size_,
            (WP2::MetricType)params.print_distortion, params.short_output));
      } else {
        WP2_CHECK_STATUS(CompressAnimation(in_files, frame_durations, params,
                                           nullptr, &rgb_buffer, &writer));
      }

      out_size = writer.size_;
      CHECK_STATUS(
          SaveImage(writer, params, out_file, GetWidth(params, rgb_buffer),
                    GetHeight(params, rgb_buffer)),
          "Error saving image");
      writer.size_ = 0;
    } else {
      WP2_CHECK_STATUS(CompressAnimation(in_files, frame_durations, params,
                                         nullptr, &rgb_buffer, &counter));
      out_size = counter.size_;
      counter.size_ = 0;
    }

    if (params.log_level >= WP2::LogLevel::DEFAULT &&
        params.print_distortion < 0) {
      const uint32_t width = GetWidth(params, rgb_buffer);
      const uint32_t height = GetHeight(params, rgb_buffer);
      const float num_pixels = ((float)width * height * in_files.size());
      PrintSize(out_path, out_size, num_pixels, params.short_output);
    }
  } else {  // !is_animation
    CHECK_TRUE(
        (in_files.size() == 1) || output_to_directory || out_path.empty(),
        "Error! Several input images require the output to be an existing "
        "directory.");
    CHECK_TRUE(decoded_file.empty() || in_files.size() == 1,
               "Can only use -d option with a single input image.");

    WP2::MultiProgressPrinter progress_printer;
    if (report_progress) progress_printer.Init(in_files.size());
    uint32_t job_index = 0;

    // Compress each image.
    for (const std::string& in_file : in_files) {
      if (in_files.size() > 1 && params.log_level >= WP2::LogLevel::DEFAULT &&
          !report_progress) {
        printf("%s\n", in_file.c_str());
      }
      std::string out_file;
      if (inplace_output) {
        out_file =
            (WP2::RemoveFileExtension(in_file) + '.' + WP2::kFileExtension);
      } else if (output_to_directory) {
        out_file =
            WP2::InputFileToOutputDir(in_file, out_path, WP2::kFileExtension);
      } else {
        out_file = out_path;
      }

      if (report_progress) {
        params.config.progress_hook =
            progress_printer.GetProgressHook(job_index++);
      }

      const bool write_to_memory =
          (!out_file.empty() || params.print_distortion >= 0 ||
           params.print_info || params.bit_trace > 0 || !decoded_file.empty());

      size_t out_size;
      if (write_to_memory) {
        if (params.print_distortion >= 0) {
          std::vector<WP2::ArgbBuffer> refs;
          WP2_CHECK_STATUS(CompressImage(in_file.c_str(), params, &refs,
                                         &rgb_buffer, &writer));

          WP2_CHECK_STATUS(PrintDistortion(
              refs, writer.mem_, writer.size_,
              (WP2::MetricType)params.print_distortion, params.short_output));
        } else {
          WP2_CHECK_STATUS(CompressImage(in_file.c_str(), params, nullptr,
                                         &rgb_buffer, &writer));
        }

        out_size = writer.size_;
        CHECK_STATUS(
            SaveImage(writer, params, out_file, GetWidth(params, rgb_buffer),
                      GetHeight(params, rgb_buffer)),
            "Error saving image");
        if (!decoded_file.empty()) {
          WP2::ArgbBuffer argb;
          CHECK_STATUS(Decode(writer.mem_, writer.size_, &argb, dec_config),
                       "Could not read the compressed WP2 bitstream. "
                       "This should NOT happen!");
          CHECK_STATUS(
              WP2::SaveImage(argb, decoded_file.c_str(), params.allow_overwrite,
                             WP2::FileFormat::AUTO, nullptr, transform),
              "Could not save the decoded sample to file.");
        }
        writer.size_ = 0;
      } else {
        WP2_CHECK_STATUS(CompressImage(in_file.c_str(), params, nullptr,
                                       &rgb_buffer, &counter));
        out_size = counter.size_;
        counter.size_ = 0;
      }

      if (params.log_level >= WP2::LogLevel::DEFAULT &&
          params.print_distortion < 0 && !report_progress) {
        const uint32_t width = GetWidth(params, rgb_buffer);
        const uint32_t height = GetHeight(params, rgb_buffer);
        const float num_pixels = (float)width * height;
        PrintSize(out_file, out_size, num_pixels, params.short_output);
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
