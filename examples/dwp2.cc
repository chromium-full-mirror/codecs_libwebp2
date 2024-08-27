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
//  Command-line tool for decoding a WP2 image.
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
#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/utils/thread_utils.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/debug.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"

namespace {

using WP2::ExUtilGetUInt;
using WP2::ExUtilTryGetUInt;
using WP2::ProgramOptions;
using WP2::SPrintf;

constexpr uint32_t kDefaultThreadLevel = 20;  // When -mt is set.

//------------------------------------------------------------------------------

void Help() {
  ProgramOptions opt;
  opt.Add("Decodes the WP2 image file.");
  opt.Add("");
  opt.Add("Usage:");
  opt.Add("  dwp2 in_file [options] [-o out_file]");
  opt.Add("");
  opt.Add("Use the following options to force the conversion into:");
  opt.Add("-pam", "save the raw RGBA samples as a color PAM");
  opt.Add("-ppm", "save the raw RGB samples as a color PPM");
  opt.Add("-bmp", "save as uncompressed BMP format");
  opt.Add("-jxl", "save as lossless JPEG XL format");
  opt.Add("-tiff", "save as uncompressed TIFF format");
  opt.Add("-png", "save the raw RGBA samples as PNG (default)");
  opt.Add("-pgm", "save the raw Y/U/V/A samples as PGM (for debugging)");
  opt.Add("-webp", "save as lossless WebP format");
  opt.Add("");
  opt.Add("Other options are:");
  opt.Add("-mt [int]", SPrintf("enable multi-threading, with %u extra threads "
                               "if unspecified (0 to disable, default is %u)",
                               kDefaultThreadLevel,
                               WP2::DecoderConfig::kDefault.thread_level));
  opt.Add("-grain <int>", "grain amplitude (if available): [0=off .. 100]");
  opt.Add(
      "-[no_]dblk_filter",
      SPrintf("enable or disable deblocking filter (%s by default)",
              WP2::DecoderConfig::kDefault.enable_deblocking_filter ? "on"
                                                                    : "off"));
  opt.Add(
      "-[no_]drct_filter",
      SPrintf("enable or disable directional filter (%s by default)",
              WP2::DecoderConfig::kDefault.enable_directional_filter ? "on"
                                                                     : "off"));
  opt.Add(
      "-[no_]rstr_filter",
      SPrintf("enable or disable restoration filter (%s by default)",
              WP2::DecoderConfig::kDefault.enable_restoration_filter ? "on"
                                                                     : "off"));
  opt.Add("-r", "recurse into input directories");
  opt.Add("-inplace",
          "output files in the same directories as input files (replaces -o)");
  opt.Add("-frames",
          "output frames named \"frame[index]_[duration]ms\" to the directory "
          "specified with -o");
  opt.Add("-force",
          "overwrite destination if it exists (default=off unless out_file is "
          "explicit)");
  opt.Add("-info", "just print bitstream info and exit");
  opt.Add("-v", "verbose (e.g. print encoding/decoding times)");
  opt.Add("-progress", "display decoding progression");
  opt.Add("-quiet", "quiet mode, don't print anything");
#ifdef WP2_BITTRACE
  opt.Add("-bt / -BT", "set bit trace level to 1 / 2");
  opt.Add("-histos", "display histograms per symbol when -bt is set");
#endif
  opt.Add("");
  opt.AddMemoryOptionSection();
  opt.AddSystemOptionSection();
  opt.Print();
}

//------------------------------------------------------------------------------

struct DecodingSettings {
  // File writing.
  std::string out_path;
  bool inplace_output = false;
  bool output_frames_to_directory = false;
  bool output_to_directory = false;
  bool allow_overwrite = false;
  bool need_transform = false;
  WP2::FileFormat out_format = WP2::FileFormat::AUTO;

  // Standard output settings.
  bool verbose = false;
  bool quiet = false;
  bool print_info = false;
  bool short_output = false;
  bool report_progress = false;

  WP2::DecoderConfig config;
  const char* visual_debug = nullptr;

  // For WP2_BITTRACE.
  int bit_trace = 0;  // 0=none, 1=trace in bits, 2=trace in bytes
  uint32_t bit_trace_level = 0;
  bool show_histograms = false;
  bool count_blocks = false;

  bool NeedInfo() const {
    return (verbose || bit_trace || count_blocks || visual_debug != nullptr ||
            need_transform);
  }
};

//------------------------------------------------------------------------------

// Get images to decode from input arguments.
WP2Status GetFilesToDecode(const WP2::FileList& paths, bool recursive_input,
                           bool verbose, bool quiet,
                           WP2::FileList* const files_to_decode) {
  files_to_decode->clear();
  for (const std::string& path : paths) {
    if (WP2::IsDirectory(path)) {
      if (recursive_input) {
        WP2::FileList files_in_dir;
        CHECK_TRUE(WP2::GetFilesIn(path, &files_in_dir, /*recursive=*/true),
                   "Error! Opening directory %s failed.", path.c_str());

        // Only keep files with a wp2 extension.
        for (const std::string& file : files_in_dir) {
          WP2::Data data;
          if (WP2::IoUtilReadFile(file.c_str(), &data, /*max_num_bytes=*/12) !=
              WP2_STATUS_OK) {
            if (verbose) {
              printf("Could not read file %s: skipping.\n", file.c_str());
            }
          } else if (WP2::GuessImageFormat(data.bytes, data.size) !=
                     WP2::FileFormat::WP2) {
            if (verbose) {
              printf("File %s has no WP2 header: skipping.\n", file.c_str());
            }
          } else {
            files_to_decode->push_back(file);
          }
        }
      } else if (!quiet) {
        fprintf(stderr, "Ignoring directory %s because -r was not specified.\n",
                path.c_str());
      }
    } else {
      // No extension checking for explicit input files.
      files_to_decode->push_back(path);
    }
  }
  return WP2_STATUS_OK;
}

uint8_t GetNumDigits(size_t number) {
  uint8_t num_digits = 1;
  for (number /= 10; number > 0; ++num_digits) number /= 10;
  return num_digits;
}

std::string GetFrameFilename(size_t index, uint32_t duration_ms) {
  std::string index_str = std::to_string(index);
  // Add leading zeros.
  static const size_t max_num_digits = GetNumDigits(WP2::kMaxNumFrames - 1);
  index_str = std::string(max_num_digits - index_str.size(), '0') + index_str;
  return "frame" + index_str + "_" + std::to_string(duration_ms) + "ms";
}

//------------------------------------------------------------------------------

class ImageDecoder : public WP2::WorkerBase {
 public:
  ImageDecoder(const DecodingSettings& settings,
               const WP2::FileList& image_pool, size_t* const next_image_index,
               WP2::ThreadLock* const mutex,
               WP2::MultiProgressPrinter* const progress_printer)
      : settings_(settings),
        image_pool_(image_pool),
        next_image_index_(next_image_index),
        mutex_(mutex),
        progress_printer_(progress_printer) {}

 protected:
  WP2Status SaveImage(const std::string& in_path, const WP2::ArgbBuffer& buffer,
                      std::string* const standard_out,
                      std::string* const standard_err) {
    if (!settings_.quiet) {
      *standard_out +=
          SPrintf("Decoded %s. Dimensions: %d x %d. Transparency: %s.\n",
                  in_path.c_str(), buffer.width(), buffer.height(),
                  buffer.HasTransparency() ? "yes" : "no");
    }
    if (!settings_.out_path.empty()) {
      std::string final_out_path;
      if (settings_.output_to_directory) {
        const char* const extension =
            GetExtensionFromFormat(settings_.out_format);
        WP2_CHECK_OK(extension != nullptr, WP2_STATUS_UNSUPPORTED_FEATURE);
        if (settings_.inplace_output) {
          final_out_path = WP2::RemoveFileExtension(in_path) + '.' + extension;
        } else {
          final_out_path =
              WP2::InputFileToOutputDir(in_path, settings_.out_path, extension);
        }
      } else {
        final_out_path = settings_.out_path;
      }

      const double start = GetStopwatchTime();
      const WP2Status status = WP2::SaveImage(
          buffer, final_out_path.c_str(), settings_.allow_overwrite,
          settings_.out_format, nullptr, transform_);
      if (!settings_.quiet && status != WP2_STATUS_OK) {
        *standard_err +=
            SPrintf("Could not save to '%s'\n", final_out_path.c_str());
        if (status == WP2_STATUS_BAD_WRITE && !settings_.allow_overwrite) {
          *standard_err +=
              "(hint: use -force flag to overwrite an existing file)\n";
        }
      }
      WP2_CHECK_STATUS(status);

      if (!settings_.quiet) {
        *standard_out += SPrintf("Saved to '%s'\n", final_out_path.c_str());
      }
      if (settings_.verbose) {
        const double write_time = GetStopwatchTime() - start;
        *standard_out += SPrintf("Time to write output: %.3fs\n", write_time);
      }
    }
    return WP2_STATUS_OK;
  }

  WP2Status DecodeAndSaveImage(const std::string& in_path,
                               std::string* const standard_out,
                               std::string* const standard_err,
                               WP2::ProgressHook* const progress_hook) {
    // Most image formats besides WebP2 use unmultiplied samples.
    WP2::ArgbBuffer output_buffer(WP2_ARGB_32);

    WP2::Data data;
    WP2_CHECK_STATUS(WP2::IoUtilReadFile(in_path.c_str(), &data));

    if (settings_.print_info) {
      *standard_out +=
          WP2::PrintSummary(data.bytes, data.size, settings_.short_output);
      return WP2_STATUS_OK;
    }

    // Grab a local copy of the config, that we can modify.
    WP2::DecoderConfig config = settings_.config;
    config.progress_hook = progress_hook;
    WP2::DecoderInfo info;
    if (settings_.NeedInfo()) {
      info.visual_debug = settings_.visual_debug;
      if (settings_.count_blocks) info.store_blocks = true;
      if (settings_.need_transform) info.csp = &transform_;
      config.info = &info;
    }

    const WP2::ArgbBuffer& buffer_to_save =
        (settings_.visual_debug != nullptr) ? info.debug_output : output_buffer;

    if (settings_.output_frames_to_directory) {
      WP2::ArrayDecoder decoder(data.bytes, data.size, config, &output_buffer);
      size_t num_frames = 0;
      double start = GetStopwatchTime();
      uint32_t duration_ms;
      while (decoder.ReadFrame(&duration_ms)) {
        if (settings_.verbose) {
          const double decode_time = GetStopwatchTime() - start;
          *standard_out +=
              SPrintf("Time to decode picture: %.3fs\n", decode_time);
          start = GetStopwatchTime();
        }

        const std::string frame_filename =
            GetFrameFilename(num_frames, duration_ms);
        WP2_CHECK_STATUS(SaveImage(frame_filename, buffer_to_save, standard_out,
                                   standard_err));
        ++num_frames;
      }

      WP2_CHECK_STATUS(decoder.GetStatus());

      // Output a frame even if it is a still image.
      WP2::BitstreamFeatures bitstream;
      WP2_CHECK_STATUS(bitstream.Read(data.bytes, data.size));
      if (!bitstream.is_animation) {
        assert(num_frames == 1);
        const std::string frame_filename =
            GetFrameFilename(/*index=*/0, WP2::kMaxFrameDurationMs);
        WP2_CHECK_STATUS(SaveImage(frame_filename, buffer_to_save, standard_out,
                                   standard_err));
      }
    } else {
      const double start = GetStopwatchTime();
      const WP2Status status =
          WP2::Decode(data.bytes, data.size, &output_buffer, config);
      if (settings_.verbose) {
        const double decode_time = GetStopwatchTime() - start;
        *standard_out +=
            SPrintf("Time to decode picture: %.3fs\n", decode_time);
      }
      WP2_CHECK_STATUS(status);

      WP2_CHECK_STATUS(
          SaveImage(in_path, buffer_to_save, standard_out, standard_err));
    }

    if (!settings_.quiet && (settings_.bit_trace || settings_.count_blocks)) {
      if (image_pool_.size() > 1) {
        *standard_err +=
            "Bit traces and block count are printed only for single images.\n";
      } else {
#if !defined(WP2_BITTRACE)
        *standard_err +=
            "Bit traces and block count are not available without WP2_BITTRACE "
            "compile flag.\n";
#else
        if (settings_.bit_trace) {
          PrintBitTraces(info, data.size, true, settings_.bit_trace == 2,
                         settings_.show_histograms, settings_.short_output,
                         settings_.bit_trace_level);
        }
        if (settings_.count_blocks) {
          printf("num-blocks : %u\n", (uint32_t)info.blocks.size());
        }
#endif
      }
    }
    return WP2_STATUS_OK;
  }

  WP2Status Execute() override {
    while (true) {
      // Get the next image to decode from the shared pool, if any.
      if (mutex_ != nullptr) WP2_CHECK_STATUS(mutex_->Acquire());
      const size_t image_index = *next_image_index_;
      if (image_index >= image_pool_.size()) {
        // Nothing left to do.
        if (mutex_ != nullptr) mutex_->Release();
        break;
      }
      const std::string& in_path = image_pool_[image_index];
      ++*next_image_index_;
      if (mutex_ != nullptr) mutex_->Release();

      std::string standard_out, standard_err;
      const WP2Status status = DecodeAndSaveImage(
          in_path, &standard_out, &standard_err,
          settings_.report_progress
              ? progress_printer_->GetProgressHook(image_index)
              : nullptr);

      if (!settings_.quiet && !settings_.report_progress) {
        // Acquire mutex to be sure to print in order without mixing lines.
        if (mutex_ != nullptr) WP2_CHECK_STATUS(mutex_->Acquire());
        printf("%s", standard_out.c_str());
        fprintf(stderr, "%s", standard_err.c_str());
        if (mutex_ != nullptr) mutex_->Release();
      }
      WP2_CHECK_STATUS(status);
    }
    return WP2_STATUS_OK;
  }

 protected:
  // Shared settings, const so no need to thread-lock it.
  const DecodingSettings& settings_;

  // Shared pool of images to decode.
  const WP2::FileList& image_pool_;
  size_t* const next_image_index_ = nullptr;
  WP2::ThreadLock* const mutex_ = nullptr;  // Acquire/release if not null.

  WP2::MultiProgressPrinter* const progress_printer_;  // Thread-safe.
  WP2::CSPTransform transform_;                        // to capture csp
};

//------------------------------------------------------------------------------

}  // namespace

int main(int argc, char* argv[]) {
  CHECK_TRUE(WP2CheckVersion(), "Error! Library version mismatch!");

  WP2::FileList in_paths;
  bool recursive_input = false;
  DecodingSettings settings = {};
  uint32_t thread_level = 0;

  for (int c = 1; c < argc; ++c) {
    bool parse_error = false;
    if (!strcmp(argv[c], "-h") || !strcmp(argv[c], "-help")) {
      Help();
      return 0;
    } else if (!strcmp(argv[c], "-o") && c + 1 < argc) {
      CHECK_TRUE(settings.out_path.empty() && !settings.inplace_output,
                 "Error! Output was already specified.");
      settings.out_path = argv[++c];
    } else if (!strcmp(argv[c], "-png")) {
      settings.out_format = WP2::FileFormat::PNG;
    } else if (!strcmp(argv[c], "-pam")) {
      settings.out_format = WP2::FileFormat::PAM;
    } else if (!strcmp(argv[c], "-ppm")) {
      settings.out_format = WP2::FileFormat::PPM;
    } else if (!strcmp(argv[c], "-bmp")) {
      settings.out_format = WP2::FileFormat::BMP;
    } else if (!strcmp(argv[c], "-jxl")) {
      settings.out_format = WP2::FileFormat::JPEG_XL;
    } else if (!strcmp(argv[c], "-tiff")) {
      settings.out_format = WP2::FileFormat::TIFF;
    } else if (!strcmp(argv[c], "-pgm")) {
      settings.out_format = WP2::FileFormat::PGM;
    } else if (!strcmp(argv[c], "-webp")) {
      settings.out_format = WP2::FileFormat::WEBP;
    } else if (!strcmp(argv[c], "-mt")) {
      if (c + 1 < argc && ExUtilTryGetUInt(argv[c + 1], &thread_level)) {
        ++c;
      } else {
        thread_level = kDefaultThreadLevel;
      }
    } else if (!strcmp(argv[c], "-grain") && c + 1 < argc) {
      settings.config.grain_amplitude = ExUtilGetUInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-dblk_filter") ||
               !strcmp(argv[c], "-no_dblk_filter")) {
      settings.config.enable_deblocking_filter =
          !strcmp(argv[c], "-dblk_filter");
    } else if (!strcmp(argv[c], "-drct_filter") ||
               !strcmp(argv[c], "-no_drct_filter")) {
      settings.config.enable_directional_filter =
          !strcmp(argv[c], "-drct_filter");
    } else if (!strcmp(argv[c], "-rstr_filter") ||
               !strcmp(argv[c], "-no_rstr_filter")) {
      settings.config.enable_restoration_filter =
          !strcmp(argv[c], "-rstr_filter");
    } else if (!strcmp(argv[c], "-r")) {
      recursive_input = true;
    } else if (!strcmp(argv[c], "-inplace")) {
      CHECK_TRUE(!settings.output_frames_to_directory,
                 "Error! -inplace is not compatible with -frames.");
      CHECK_TRUE(settings.out_path.empty(),
                 "Error! Output was already specified.");
      settings.inplace_output = true;
    } else if (!strcmp(argv[c], "-frames")) {
      CHECK_TRUE(!settings.inplace_output,
                 "Error! -frames is not compatible with -inplace.");
      settings.output_frames_to_directory = true;
    } else if (!strcmp(argv[c], "-force")) {
      settings.allow_overwrite = true;
    } else if (!strcmp(argv[c], "-info")) {
      settings.print_info = true;
    } else if (!strcmp(argv[c], "-vdebug") && c + 1 < argc) {
      settings.visual_debug = argv[++c];
    } else if (!strcmp(argv[c], "-bt") || !strcmp(argv[c], "-BT")) {
      settings.bit_trace = !strcmp(argv[c], "-bt") ? 1 : 2;
      if (c + 1 < argc) {
        if (isdigit(argv[c + 1][0])) {
          settings.bit_trace_level = ExUtilGetUInt(argv[++c], &parse_error);
        } else {
          settings.bit_trace_level = 0;
        }
      }
    } else if (!strcmp(argv[c], "-histos")) {
      settings.show_histograms = true;
    } else if (!strcmp(argv[c], "-count_blocks")) {
      settings.count_blocks = true;
    } else if (!strcmp(argv[c], "-progress")) {
      settings.report_progress = true;
    } else if (!strcmp(argv[c], "-quiet")) {
      settings.quiet = true;
    } else if (!strcmp(argv[c], "-short")) {
      settings.short_output = true;
    } else if (!strcmp(argv[c], "-v")) {
      settings.verbose = true;
    } else if (!strcmp(argv[c], "--")) {
      if (c + 1 < argc) in_paths.emplace_back(argv[++c]);
      break;
    } else if (argv[c][0] == '-') {
      bool must_stop;
      int skip;
      if (ProgramOptions::ParseSystemOptions(argv[c], &must_stop)) {
        if (must_stop) return 0;
      } else if (ProgramOptions::ParseMemoryOptions(argv + c, argc - c, skip)) {
        c += skip - 1;
      } else {
        printf("Unknown option '%s'\n", argv[c]);
        Help();
        return 1;
      }
    } else {
      in_paths.emplace_back(argv[c]);
    }

    if (parse_error) {
      Help();
      return 1;
    }
  }

  WP2::FileList in_files;
  CHECK_STATUS(GetFilesToDecode(in_paths, recursive_input, settings.verbose,
                                settings.quiet, &in_files),
               "missing input file!!");

  if (settings.quiet) settings.verbose = false;
  if (settings.quiet) settings.report_progress = false;

  settings.output_to_directory =
      (settings.inplace_output || WP2::IsDirectory(settings.out_path));

  if (!settings.output_to_directory) {
    // Destination is explicitly specified, allow overwriting it.
    settings.allow_overwrite = true;
    if (settings.out_format == WP2::FileFormat::AUTO) {
      settings.out_format =
          WP2::GetFormatFromExtension(settings.out_path.c_str());
    }
  }
  if (settings.out_format == WP2::FileFormat::PGM) {
    settings.need_transform = true;
  }
  if (settings.output_frames_to_directory) {
    CHECK_TRUE(in_files.size() == 1,
               "Error! -frames requires a single input file.");
    CHECK_TRUE(settings.output_to_directory,
               "Error! -frames requires a single output directory.");
  } else {
    CHECK_TRUE(
        in_files.size() == 1 || settings.output_to_directory ||
            settings.out_path.empty(),
        "Error! Several input images require the output to be an existing "
        "directory.");
  }

  // Warning! This object is allocating underneath. Bad interference with -mem*!
  WP2::MultiProgressPrinter progress_printer;

  if (settings.report_progress) progress_printer.Init(in_files.size());

  // Use a pool of images to decode.
  size_t next_image_index = 0;
  WP2::ThreadLock mutex;

  // Create a pool of workers that will get jobs from 'in_files'.
  const size_t num_workers =
      std::min(1 + (size_t)thread_level, in_files.size());
  const bool use_mt_workers = num_workers > 1;

  // Enable multithreading at image level or at tile level but not both.
  // TODO(yguyon): This could be optimized depending on the number of tiles etc.
  settings.config.thread_level = use_mt_workers ? 0 : thread_level;

  std::vector<ImageDecoder> workers;
  workers.reserve(num_workers);

  WP2Status status = WP2_STATUS_OK;
  for (size_t i = 0; i < num_workers; ++i) {
    workers.emplace_back(settings, in_files, &next_image_index,
                         use_mt_workers ? &mutex : nullptr, &progress_printer);
    status = workers.back().Start(use_mt_workers);
    if (status != WP2_STATUS_OK) break;
  }

  for (ImageDecoder& worker : workers) {
    const WP2Status end_status = worker.End();
    if (status == WP2_STATUS_OK) status = end_status;
    // Still wait for others to finish in case of error.
  }
  CHECK_STATUS(status, "Error: %s", WP2GetStatusText(status));

  if (!settings.quiet && settings.out_path.empty()) {
    printf("Nothing written; use -o flag to save the result.\n");
  }
  return 0;
}

//------------------------------------------------------------------------------
