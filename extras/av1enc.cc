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
//  compressor/decompressor for testing AV1
//
//  ./av1enc input [-o output.png] [-d bitstream.av1] [-q xx ...]
//
// The -setenv option allows AV1 behavior modifications and comparisons.
// For example the following code can be added anywhere
// (such as in third_party/libaom/git_root/av1/decoder/decodeframe.c):
//     if (getenv("STUFF_ON")) { /* do stuff */ }
// And then this tool can be called:
//     ./av1enc src.png -O -setenv STUFF_ON -setenv STUFF_OFF
// This will generate 'src_dec_STUFF_ON.png' and 'src_dec_STUFF_OFF.png'
// This tool is not available on _WIN32.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "examples/example_utils.h"
#include "extras/aom_utils.h"
#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"

namespace {

//------------------------------------------------------------------------------

std::string AppendToFileName(const std::string& file_path,
                             const std::string& suffix) {
  return WP2::RemoveFileExtension(file_path) + suffix + "." +
         WP2::GetFileExtension(file_path);
}

//------------------------------------------------------------------------------

#if defined(WP2_HAVE_AOM_DBG)
// Names of most variables in struct 'insp_mi_data' from AV1 inspection.h
const char* const kAllMiDataNames[]{
    "mode",          "uv_mode",          "sb_type",        "skip",
    "segment_id",    "dual_filter_type", "filter[0]",      "filter[1]",
    "tx_type",       "tx_size",          "cdef_level",     "cdef_strength",
    "cfl_alpha_idx", "cfl_alpha_sign",   "current_qindex", "compound_type",
    "motion_mode",   "intrabc",          "palette",        "uv_palette"};
#endif

// Mapping from aom option names to aome_enc_control_id enum values.
constexpr std::pair<const char*, int> kAomOptions[] = {
    {"use-reference", 7},
    {"roi-map", 8},
    {"activemap", 9},
    {"scalemode", 11},
    {"spatial-layer-id", 12},
    {"cpuused", 13},
    {"enableautoaltref", 14},
    {"sharpness", 16},
    {"static-threshold", 17},
    {"get-last-quantizer", 19},
    {"get-last-quantizer-64", 20},
    {"arnr-maxframes", 21},
    {"arnr-strength", 22},
    {"tuning", 24},
    {"cq-level", 25},
    {"max-intra-bitrate-pct", 26},
    {"number-spatial-layers", 27},
    {"max-inter-bitrate-pct", 28},
    {"gf-cbr-boost-pct", 29},
    {"lossless", 31},
    {"row-mt", 32},
    {"tile-columns", 33},
    {"tile-rows", 34},
    {"enable-tpl-model", 35},
    {"enable-keyframe-filtering", 36},
    {"frame-parallel-decoding", 37},
    {"error-resilient-mode", 38},
    {"s-frame-mode", 39},
    {"aq-mode", 40},
    {"frame-periodic-boost", 41},
    {"noise-sensitivity", 42},
    {"tune-content", 43},
    {"cdf-update-mode", 44},
    {"color-primaries", 45},
    {"transfer-characteristics", 46},
    {"matrix-coefficients", 47},
    {"chroma-sample-position", 48},
    {"min-gf-interval", 49},
    {"max-gf-interval", 50},
    {"get-activemap", 51},
    {"color-range", 52},
    {"render-size", 53},
    {"target-seq-level-idx", 54},
    {"get-seq-level-idx", 55},
    {"superblock-size", 56},
    {"enableautobwdref", 57},
    {"enable-cdef", 58},
    {"enable-restoration", 59},
    {"force-video-mode", 60},
    {"enable-obmc", 61},
    {"disable-trellis-quant", 62},
    {"enable-qm", 63},
    {"qm-min", 64},
    {"qm-max", 65},
    {"qm-y", 66},
    {"qm-u", 67},
    {"qm-v", 68},
    {"num-tg", 70},
    {"mtu", 71},
    {"enable-rect-partitions", 73},
    {"enable-ab-partitions", 74},
    {"enable-1to4-partitions", 75},
    {"min-partition-size", 76},
    {"max-partition-size", 77},
    {"enable-intra-edge-filter", 78},
    {"enable-order-hint", 79},
    {"enable-tx64", 80},
    {"enable-flip-idtx", 81},
    {"enable-rect-tx", 82},
    {"enable-dist-wtd-comp", 83},
    {"enable-ref-frame-mvs", 84},
    {"allow-ref-frame-mvs", 85},
    {"enable-dual-filter", 86},
    {"enable-chroma-deltaq", 87},
    {"enable-masked-comp", 88},
    {"enable-onesided-comp", 89},
    {"enable-interintra-comp", 90},
    {"enable-smooth-interintra", 91},
    {"enable-diff-wtd-comp", 92},
    {"enable-interinter-wedge", 93},
    {"enable-interintra-wedge", 94},
    {"enable-global-motion", 95},
    {"enable-warped-motion", 96},
    {"allow-warped-motion", 97},
    {"enable-filter-intra", 98},
    {"enable-smooth-intra", 99},
    {"enable-paeth-intra", 100},
    {"enable-cfl-intra", 101},
    {"enable-superres", 102},
    {"enable-overlay", 103},
    {"enable-palette", 104},
    {"enable-intrabc", 105},
    {"enable-angle-delta", 106},
    {"deltaq-mode", 107},
    {"deltalf-mode", 108},
    {"single-tile-decoding", 109},
    {"enable-motion-vector-unit-test", 110},
    {"timing-info-type", 111},
    {"film-grain-test-vector", 112},
    {"film-grain-table", 113},
    {"denoise-noise-level", 114},
    {"denoise-block-size", 115},
    {"chroma-subsampling-x", 116},
    {"chroma-subsampling-y", 117},
    {"reduced-tx-type-set", 118},
    {"intra-dct-only", 119},
    {"inter-dct-only", 120},
    {"intra-default-tx-only", 121},
    {"quant-b-adapt", 122},
    {"gf-max-pyramid-height", 123},
    {"max-reference-frames", 124},
    {"reduced-reference-set", 125},
    {"coeff-cost-upd-freq", 126},
    {"mode-cost-upd-freq", 127},
    {"mv-cost-upd-freq", 128},
    {"tier-mask", 129},
    {"min-cr", 130},
    {"svc-layer-id", 131},
    {"svc-params", 132},
    {"svc-ref-frame-config", 133},
    {"vmaf-model-path", 134},
    {"enable-ext-tile-debug", 135},
    {"enable-sb-multipass-unit-test", 136},
    {"gf-min-pyramid-height", 137},
    {"vbr-corpus-complexity-lap", 138},
    {"get-baseline-gf-interval", 139},
    {"enable-dnl-denoising", 140},
    {"enable-diagonal-intra", 141},
    {"dv-cost-upd-freq", 142},
    {"partition-info-path", 143},
    {"external-partition", 144},
    {"enable-directional-intra", 145},
    {"enable-tx-size-search", 146},
};

//------------------------------------------------------------------------------

void Help() {
  WP2::ProgramOptions opt;
  opt.Add("Usage:");
  opt.Add("  av1enc [options] in_file [-o decoded_img] [-d encoded_av1]");
  opt.Add("");
  opt.Add("Options:");
  opt.Add("-q <float>", "quality factor [0..100]");
  opt.Add("-alpha_q <float>", "alpha quality factor [0..100]");
  opt.Add("-size <int>", "target size (predates -q)");
  opt.Add("-effort <int>", "compression effort [0..9]");
  opt.Add("-threads <int>", "number of threads used");
  opt.Add("-pass <int>", "number of passes [1..2]");
  opt.Add("-420", "use YUV420 encoding (default)");
  opt.Add("-444", "use YUV444 encoding");
  opt.Add("-tune <metric>", "choose metric for tuning. "
                            "One of: psnr, ssim, vmaf, butteraugli");
  opt.Add("-notransforms", "disable transforms other than DCT");
  opt.Add("-nopreds",
          "disable predictors other than DC (and some angles for now)");
  opt.Add("-nopalette", "disable palette");
  opt.Add("-noblock_copy", "disable block copy");
  opt.Add("-notrellis", "disable trellis (uses dropout instead)");
  opt.Add("-force_block_size <int>",
          "forces a square block size (4, 8, 16, 32, 64, 128");
  opt.AddMetricOptions();
#if defined(WP2_HAVE_AOM_DBG)
  opt.Add("-bt / -BT", "report bit use (if compiled)");
  opt.Add("-compact", "use compact trace for bt/BT");
  opt.Add("-b", "draw blocks (if compiled)");
  opt.Add("-B", "dump blocks in /tmp/partition.txt");
  opt.Add("-t", "draw luma transforms boundaries (if compiled)");
  opt.Add("-T", "dump luma transforms layout in /tmp/partition.txt");
  opt.Add("-mi_data [var_name]", "draw insp_mi_data (.var_name)");
  opt.Add("-mi_number", "draw mi value as a red square");
  opt.Add("-quant [y|u|v]", "draw/print quantization for the given channel");
#if !defined(_WIN32)
  opt.Add("-setenv <name> [value]",
          "set environment variable (one encoding per -setenv)");
#endif  // !_WIN32
#endif  // WP2_HAVE_AOM_DBG
  opt.Add("-o <file path>", "path to decoded image");
  opt.Add("-O", "same as -o [in_file]_dec.png");
  opt.Add("-d <file path>", "path to encoded AV1 image");
  opt.Add("-D", "same as -d [in_file].av1");
  opt.Add("-a option=value", "extra options for libaom");
  opt.Add("-quiet", "minimal output");
  opt.Add("-avif", "write as AVIF");
  opt.Add("-libgav1", "use libgav1 decoder");
  opt.Add("-h", "this help");
  opt.Print();
}

}  // namespace

//------------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
  std::string input_path;
  std::string decoded_path;
  std::string encoded_path;
  bool decoded_path_from_input = false, encoded_path_from_input = false;
  WP2::ParamsAV1 params;
  WP2::MetricType type = WP2::PSNR;
  std::vector<std::pair<std::string, std::string>> env_vars;
  std::vector<const char*> mi_data_names;
  std::vector<WP2::Rectangle> blocks;
  std::vector<WP2::Rectangle> transforms;
  bool store_blocks = false, store_transforms = false;
  bool count_blocks = false, count_transforms = false;
  bool print_quant = false;
  bool quiet = false;
  bool use_avif = false;
  for (int c = 1; c < argc; ++c) {
    if (!strcmp(argv[c], "-d") && c + 1 < argc) {
      encoded_path = argv[++c];
    } else if (!strcmp(argv[c], "-D")) {
      encoded_path_from_input = true;
    } else if (!strcmp(argv[c], "-o") && c + 1 < argc) {
      decoded_path = argv[++c];
    } else if (!strcmp(argv[c], "-O")) {
      decoded_path_from_input = true;
    } else if (!strcmp(argv[c], "-q") && c + 1 < argc) {
      params.quality = atof(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-alpha_q") && c + 1 < argc) {
      params.alpha_quality = atof(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-420")) {
      params.use_yuv444 = false;
    } else if (!strcmp(argv[c], "-444")) {
      params.use_yuv444 = true;
    } else if (!strcmp(argv[c], "-notransforms")) {
      params.disable_transforms = true;
    } else if (!strcmp(argv[c], "-nopreds")) {
      params.disable_predictors = true;
    } else if (!strcmp(argv[c], "-nopalette")) {
      params.disable_palette = true;
    } else if (!strcmp(argv[c], "-noblock_copy")) {
      params.disable_block_copy = true;
    } else if (!strcmp(argv[c], "-notrellis")) {
      params.disable_trellis = true;
    } else if (!strcmp(argv[c], "-force_block_size") && c + 1 < argc) {
      params.force_block_size = atoi(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-tune") && c + 1 < argc) {
      ++c;
      if (!strcmp(argv[c], "psnr") || !strcmp(argv[c], "PSNR")) {
        params.tuning = WP2::ParamsAV1::Tuning::kPsnr;
      } else if (!strcmp(argv[c], "ssim") || !strcmp(argv[c], "SSIM")) {
        params.tuning = WP2::ParamsAV1::Tuning::kSsim;
      } else if (!strcmp(argv[c], "vmaf")) {
        params.tuning = WP2::ParamsAV1::Tuning::kVmaf;
      } else if (!strcmp(argv[c], "butteraugli")) {
        params.tuning = WP2::ParamsAV1::Tuning::kButteraugli;
      } else {
        fprintf(stderr, "Ignored unhandled -tune '%s'\n", argv[c]);
      }
#if defined(WP2_HAVE_AOM_DBG)
    } else if (!strcmp(argv[c], "-bt")) {
      params.bittrace = 1;
    } else if (!strcmp(argv[c], "-BT")) {
      params.bittrace = 2;
    } else if (!strcmp(argv[c], "-compact")) {
      params.compact_trace = true;
    } else if (!strcmp(argv[c], "-b")) {
      params.draw_blocks = true;
    } else if (!strcmp(argv[c], "-B")) {
      store_blocks = true;
    } else if (!strcmp(argv[c], "-count_blocks")) {
      count_blocks = true;
    } else if (!strcmp(argv[c], "-t")) {
      params.draw_transforms = true;
    } else if (!strcmp(argv[c], "-T")) {
      store_transforms = true;
    } else if (!strcmp(argv[c], "-count_transforms")) {
      count_transforms = true;
    } else if (!strcmp(argv[c], "-mi_data")) {
      if (c + 1 < argc && argv[c + 1][0] != '-') {
        ++c;
        bool found = false;
        for (const char* n : kAllMiDataNames) found |= (!strcmp(argv[c], n));
        if (found) mi_data_names.emplace_back(argv[c]);
        if (!found) {
          fprintf(stderr, "Ignored unknown mi_data '%s'. Must be one of:\n",
                  argv[c]);
          for (const char* n : kAllMiDataNames) fprintf(stderr, "%s ", n);
          fprintf(stderr, "\n");
        }
      } else {
        for (const char* n : kAllMiDataNames) mi_data_names.emplace_back(n);
        mi_data_names.emplace_back(nullptr);  // Also output the regular img.
      }
    } else if (!strcmp(argv[c], "-mi_number")) {
      params.draw_mi_number = true;
    } else if (!strcmp(argv[c], "-quant") && c + 1 < argc) {
      ++c;
      print_quant = true;
      if (!strcmp(argv[c], "y")) {
        mi_data_names.emplace_back("quanty");
      } else if (!strcmp(argv[c], "u")) {
        mi_data_names.emplace_back("quantu");
      } else if (!strcmp(argv[c], "v")) {
        mi_data_names.emplace_back("quantv");
      } else {
        fprintf(stderr, "Unknown channel '%s'. Must be one of: y, u, v\n",
                argv[c]);
      }
#if !defined(_WIN32)
    } else if (!strcmp(argv[c], "-setenv") && c + 1 < argc) {
      const bool has_value = (c + 2 < argc && argv[c + 2][0] != '-');
      env_vars.emplace_back(argv[c + 1], has_value ? argv[c + 2] : "");
      c += (has_value ? 2 : 1);
#endif  // !_WIN32
#endif  // WP2_HAVE_AOM_DBG

    } else if (!strcmp(argv[c], "-size") && c + 1 < argc) {
      params.size = atoi(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-threads") && c + 1 < argc) {
      params.threads = atoi(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-effort") && c + 1 < argc) {
      params.effort = atoi(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-pass") && c + 1 < argc) {
      params.pass = atoi(argv[++c]);  // NOLINT
    } else if (!strcmp(argv[c], "-a") && c + 1 < argc) {
      const std::string param(argv[++c]);
      std::string option_name;
      int option_value;
      const int pos = param.find('=');
      if (pos == std::string::npos) {  // Assume value is 1.
        option_name = param;
        option_value = 1;
      } else {
        option_name = param.substr(0, pos);
        option_value = atoi(param.substr(pos + 1, std::string::npos).c_str());
      }
      int option_enum = -1;
      for (const auto& pair : kAomOptions) {
        if (!strcmp(option_name.c_str(), pair.first)) {
          option_enum = pair.second;
          break;
        }
      }
      if (option_enum == -1) {
        fprintf(stderr, "Ignored unknown AOM option '%s'. Must be one of:\n",
                option_name.c_str());
        for (const auto& pair : kAomOptions) {
          fprintf(stderr, "%s ", pair.first);
        }
        fprintf(stderr, "\n");
        continue;
      }
      params.extra_options[option_enum] = option_value;
    } else if (!strcmp(argv[c], "-quiet")) {
      quiet = true;
    } else if (!strcmp(argv[c], "-avif")) {
      use_avif = true;
    } else if (!strcmp(argv[c], "-libgav1")) {
      params.use_libgav1_decoder = true;
    } else if (!strcmp(argv[c], "-h")) {
      Help();
      return 0;
    } else if (WP2::ProgramOptions::ParseMetricOptions(argv[c], &type)) {
      continue;
    } else {
      input_path = argv[c];
    }
  }
  if (input_path.empty()) {
    fprintf(stderr, "Missing input filename!\n");
    Help();
    return 255;
  }
  if (decoded_path_from_input) {
    decoded_path = AppendToFileName(input_path, "_dec");
  }
  if (encoded_path_from_input) {
    encoded_path = WP2::RemoveFileExtension(input_path) + ".av1";
  }

  WP2::ArgbBuffer ref;
  if (WP2::ReadImage(input_path.c_str(), &ref) != WP2_STATUS_OK) {
    fprintf(stderr, "Error opening file: %s\n", input_path.c_str());
    return 1;
  }
  for (size_t i = 0; i < std::max((size_t)1, env_vars.size()); ++i) {
    for (size_t n = 0; n < std::max((size_t)1, mi_data_names.size()); ++n) {
      std::string suffix;
#if !defined(_WIN32)
      if (!env_vars.empty()) {
        setenv(env_vars[i].first.c_str(), env_vars[i].second.c_str(), 1);
        if (env_vars.size() > 1) {
          // Only append it to the file name if there are several env vars.
          suffix = "_" + env_vars[i].first;
          if (!env_vars[i].second.empty()) suffix += "_" + env_vars[i].second;
        }
      }
#endif  // !_WIN32
      if (!mi_data_names.empty()) {
        params.draw_mi_data = mi_data_names[n];
        if (mi_data_names[n] != nullptr) {
          suffix += "-";
          suffix += mi_data_names[n];
        }
      }

      std::string bits;
      WP2::ArgbBuffer decoded;
      double timing[2];
      WP2::QuantAV1 quant;

      if (use_avif) {
        // Disable statistics gathering.
        store_blocks = count_blocks = false;
        store_transforms = count_transforms = false;
        print_quant = false;
        if (WP2::CompressAVIF(ref, params, &decoded, &bits, timing) !=
            WP2_STATUS_OK) {
          fprintf(stderr, "Error compressing to AVIF\n");
          return 2;
        }
      } else if (WP2::CompressAV1(
              ref, params, &decoded, &bits, timing,
              (store_blocks || count_blocks) ? &blocks : nullptr,
              (store_transforms || count_transforms) ? &transforms : nullptr,
              print_quant ? &quant : nullptr) != WP2_STATUS_OK) {
        fprintf(stderr, "Error compressing to AV1\n");
        return 2;
      }
#if !defined(_WIN32)
      if (!env_vars.empty()) {
        unsetenv(env_vars[i].first.c_str());
      }
#endif  // !_WIN32

      // Align text with bit traces.
      if (count_blocks) {
        printf("num-blocks                     %u\n", (uint32_t)blocks.size());
      }
      if (count_transforms) {
        printf("num-transforms                 %u\n",
               (uint32_t)transforms.size());
      }
      if (print_quant) {
        for (uint32_t s = 0; s < 8; ++s) {
          printf("segment %d\n", s);
          printf("\tY quant, DC: %d AC: %d\n", quant.y_dequant[s][0],
                 quant.y_dequant[s][1]);
          printf("\tU quant, DC: %d AC: %d\n", quant.u_dequant[s][0],
                 quant.u_dequant[s][1]);
          printf("\tV quant, DC: %d AC: %d\n", quant.v_dequant[s][0],
                 quant.v_dequant[s][1]);
        }
      }

      // Don't display the distortion if there is debug info in the image.
      if (!params.draw_blocks && !params.draw_transforms &&
          params.draw_mi_data == nullptr) {
        float disto[5];
        if (ref.GetDistortionBlackOrWhiteBackground(decoded, type, disto) !=
            WP2_STATUS_OK) {
          fprintf(stderr, "Error while computing the distortion.\n");
          return 2;
        }
        printf("%u %f   (%f %f %f %f) ", (uint32_t)bits.size(), disto[4],
               disto[0], disto[1], disto[2], disto[3]);
        printf("[ %.2f bpp ] [ %d x %d ] [%.3f / %.3f secs]\n",
               8.f * bits.size() / (ref.width() * ref.height()), ref.width(),
               ref.height(), timing[0], timing[1]);
      }

      if (store_blocks || store_transforms) {
        const char path[] = "/tmp/partition.txt";
        // Transforms are contained in blocks so output only transforms if both.
        if (!WritePartition(store_transforms ? transforms : blocks, path)) {
          fprintf(stderr, "Error saving partition file (%s)\n", path);
          return 3;
        } else if (!quiet) {
          printf("Saved partition file to %s\n", path);
        }
      }

      if (!decoded_path.empty()) {
        const std::string path = AppendToFileName(decoded_path, suffix);
        if (WP2::SaveImage(decoded, path.c_str(), /*overwrite=*/true) !=
            WP2_STATUS_OK) {
          fprintf(stderr, "Error saving decoded image (%s)\n", path.c_str());
          return 3;
        } else if (!quiet) {
          printf("Saved decoded image %s\n", path.c_str());
        }
      }

      if (!encoded_path.empty() && !params.draw_blocks &&
          !params.draw_transforms && params.draw_mi_data == nullptr) {
        const std::string path = AppendToFileName(encoded_path, suffix);
        if (WP2::IoUtilWriteFile(bits, path.c_str(),
                                 /*overwrite=*/true) != WP2_STATUS_OK) {
          fprintf(stderr, "Error saving encoded file (%s)\n", path.c_str());
          return 3;
        } else if (!quiet) {
          printf("Saved encoded file: %s (%u bytes)\n", path.c_str(),
                 (uint32_t)bits.size());
        }
      }
    }
  }
  return 0;
}
