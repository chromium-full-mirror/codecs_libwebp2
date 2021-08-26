// Copyright 2021 Google LLC
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

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "./av1_obu.h"
#include "./avif.h"
#include "./container.h"

namespace {

//------------------------------------------------------------------------------

void Help() {
  std::cout << "Tool for converting between AVIF and minimal container formats."
            << std::endl;
  std::cout << "Usage:" << std::endl;
  std::cout << " convert <path> [options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl
            << " -o <path>" << std::endl
            << " --output <path> ... Output file" << std::endl
            << " --container ....... Outputs minimal image container"
            << std::endl
            << " --avif ............ Outputs AVIF" << std::endl
            << " --strip-obu ....... Trims input OBU sequence/frame headers"
            << std::endl
            << " --info ............ Prints details" << std::endl
            << " -h" << std::endl
            << " --help ............ Prints this" << std::endl;
}

//------------------------------------------------------------------------------

bool EndsWith(const std::string& str, const std::string& token) {
  return (str.size() >= token.size() &&
          str.compare(str.size() - token.size(), token.size(), token) == 0);
}

//------------------------------------------------------------------------------

enum class Format {
  kContainer,
  kAvif,
  kObu,
  kDeduceFromExtension
};

struct Params {
  std::string input_path;
  std::string output_path;
  Format output_format = Format::kDeduceFromExtension;
  bool strip_obu = false;
  container::EncoderConfig config;
  bool print_info = false;
};

// Converts the flags given to the binary to 'params' values.
// Returns true if the program should continue or false if the program should
// exit with 'error_code'.
bool ParseArguments(int argc, const char* argv[], Params& params,
                    int& error_code) {
  // argv[0] contains the name of the binary, skip it.
  for (int c = 1; c < argc; ++c) {
    if ((!std::strcmp(argv[c], "-o") || !std::strcmp(argv[c], "--output")) &&
        c + 1 < argc) {
      params.output_path = argv[++c];
    } else if (!std::strcmp(argv[c], "--container")) {
      params.output_format = Format::kContainer;
    } else if (!std::strcmp(argv[c], "--avif")) {
      params.output_format = Format::kAvif;
    } else if (!std::strcmp(argv[c], "--strip-obu")) {
      params.strip_obu = true;
    } else if (!std::strcmp(argv[c], "--info")) {
      params.print_info = true;
    } else if (!std::strcmp(argv[c], "-h") || !std::strcmp(argv[c], "--help")) {
      Help();
      error_code = 0;  // Not an error.
      return false;
    } else {
      if (argv[c][0] == '-') {
        std::cerr << "Error: Unknown flag " << argv[c] << std::endl;
        error_code = 1;
        return false;
      }
      if (!params.input_path.empty()) {
        std::cerr << "Error: Input is specified more than once ("
                  << params.input_path << ", " << argv[c] << ")" << std::endl;
        error_code = 1;
        return false;
      }
      params.input_path = argv[c];
    }
  }
  return true;
}

// Validates the 'params'. Reads the input file into the 'input' bytes.
// Returns true if the program should continue or false if the program should
// exit with 'error_code'.
bool ProcessArguments(Params& params, std::vector<char>& input,
                      int& error_code) {
  if (params.input_path.empty()) {
    std::cerr << "Error: Missing input filename" << std::endl;
    return false;
  }

  std::ifstream input_file(params.input_path, std::ios_base::binary);
  input = std::vector<char>(std::istreambuf_iterator<char>(input_file),
                            std::istreambuf_iterator<char>());

  if (input.empty()) {
    std::cerr << "Error: Unable to read any byte from " << params.input_path
              << std::endl;
    error_code = 1;
    return false;
  }

  if (!params.output_path.empty() &&
      params.output_format == Format::kDeduceFromExtension) {
    if (EndsWith(params.output_path, ".pic")) {  // Temporary file extension.
      params.output_format = Format::kContainer;
    } else if (EndsWith(params.output_path, ".avif")) {
      params.output_format = Format::kAvif;
    } else if (EndsWith(params.output_path, ".obu")) {
      params.output_format = Format::kObu;
    } else {
      std::cerr << "Error: Unable to deduce a supported output format from "
                   "the extension of "
                << params.output_path << std::endl;
      error_code = 1;
      return false;
    }
  }
  return true;
}

}  // namespace

//------------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
  Params params;
  std::vector<char> input;  // Input file content.

  int error_code;
  if (!ParseArguments(argc, argv, params, error_code)) return error_code;
  if (!ProcessArguments(params, input, error_code)) return error_code;
  const uint8_t* const input_data =
      reinterpret_cast<const uint8_t*>(input.data());

  container::Image image;  // Used to store the image features, not the pixels.
  uint32_t codec = 0;      // Input pixels format:
                           //   kContainerCodecObu
                           //   kContainerCodecAv1 (stripped OBU)
                           //   kContainerCodecWp2
  size_t offset_till_codec_bytes, num_codec_bytes;   // Encoded input pixels.
  container::SequenceHeaderObu sequence_header_obu;  // In case of OBU input.

  // Try reading the input as both formats.
  const bool input_is_container =
      container::UnwrapContainer(input_data, input.size(), image, codec,
                                 offset_till_codec_bytes, num_codec_bytes);
  const bool input_is_avif =
      !input_is_container &&
      container::UnwrapAvif(input_data, input.size(), image,
                            offset_till_codec_bytes, num_codec_bytes);
  const bool input_is_obu = !input_is_container && !input_is_avif &&
                            container::UnwrapObuSequenceHeader(
                                input_data, input.size(), sequence_header_obu);
  if (input_is_avif) {
    codec = container::kContainerCodecObu;
  } else if (input_is_obu) {
    codec = container::kContainerCodecObu;
    offset_till_codec_bytes = 0;
    num_codec_bytes = input.size();

    image.width = sequence_header_obu.width;
    image.height = sequence_header_obu.height;
    image.is_opaque = true;  // TODO(yguyon): Handle transparency
  }

  if (!input_is_container && !input_is_avif && !input_is_obu) {
    std::cerr << "Error: Unsupported input or output" << std::endl;
    return 1;
  }

  // Print info.
  if (input_is_container) {
    std::cout << "Minimal image container input (" << input.size() << " bytes)"
              << std::endl;
    if (params.print_info) {
      std::cout << container::ContainerHeaderToStr(input_data, input.size());
    }
  } else if (input_is_avif) {
    std::cout << "AVIF input (" << input.size() << " bytes)" << std::endl;
    if (params.print_info) {
      std::cout << container::BmffBoxesToStr(input_data, input.size());
    }
  }

  if (codec == container::kContainerCodecObu) {
    std::cout << "OBU input (" << num_codec_bytes << " bytes)" << std::endl;
    if (params.print_info) {
      std::cout << container::ObuUnitsToStr(
          input_data + offset_till_codec_bytes, num_codec_bytes);
    }
  }

  container::ObuFeatures features;

  // Strip the OBU.
  if (codec == container::kContainerCodecAv1 ||
      (params.strip_obu && codec == container::kContainerCodecObu)) {
    size_t offset_till_tiles_bytes, num_tiles_bytes;
    std::string log;
    const bool success =
        (codec == container::kContainerCodecAv1)
            ? container::UnwrapStrippedObu(
                  input_data + offset_till_codec_bytes, num_codec_bytes,
                  image.width, image.height, features, offset_till_tiles_bytes,
                  num_tiles_bytes, log)
            : container::UnwrapObu(input_data + offset_till_codec_bytes,
                                   num_codec_bytes, image.width, image.height,
                                   features, offset_till_tiles_bytes,
                                   num_tiles_bytes, log);

    if (params.print_info) std::cout << log;
    if (!success) {
      std::cerr << "Error: Unable to unwrap OBU input" << std::endl;
      return 1;
    }

    // Replace the codec bytes by the tile group bytes and the 'features'.
    codec = container::kContainerCodecAv1;
    offset_till_codec_bytes += offset_till_tiles_bytes;
    num_codec_bytes = num_tiles_bytes;
  }

  container::Data reconstructed_obu;
  std::string reconstructed_obu_log;
  const uint8_t* final_codec_bytes = input_data + offset_till_codec_bytes;
  size_t num_final_codec_bytes = num_codec_bytes;

  if (params.output_format != Format::kDeduceFromExtension &&
      codec == container::kContainerCodecAv1) {
    // Restore the OBU. Only kContainer output supports stripped OBU.
    const bool strip_obu =
        (params.strip_obu && params.output_format == Format::kContainer);
    const bool success =
        strip_obu ? container::WrapStrippedObu(
                        image.width, image.height, features,
                        input_data + offset_till_codec_bytes, num_codec_bytes,
                        reconstructed_obu, reconstructed_obu_log)
                  : container::WrapObu(image.width, image.height, features,
                                       input_data + offset_till_codec_bytes,
                                       num_codec_bytes, reconstructed_obu,
                                       reconstructed_obu_log);

    if (!success) {
      if (params.print_info) std::cout << reconstructed_obu_log;
      std::cerr << "Error: Unable to reconstruct OBU" << std::endl;
      return 1;
    }

    codec = strip_obu ? container::kContainerCodecAv1
                      : container::kContainerCodecObu;
    final_codec_bytes = reconstructed_obu.data();
    num_final_codec_bytes = reconstructed_obu.size();
  }

  if (codec == container::kContainerCodecObu) {
    std::string log =
        "OBU output (" + std::to_string(num_final_codec_bytes) + " bytes)\n";
    if (params.print_info) {
      log += container::ObuUnitsToStr(final_codec_bytes, num_final_codec_bytes);
    }
    reconstructed_obu_log = log + reconstructed_obu_log;
  }

  // Wrap the encoded pixels to the destination format.
  container::Data output_data;
  if (params.output_format == Format::kContainer) {
    container::WrapContainer(image, params.config, codec, final_codec_bytes,
                             num_final_codec_bytes, output_data);
    std::cout << std::endl
              << "Minimal image container output (" << output_data.size()
              << " bytes)" << std::endl;
    if (params.print_info) {
      std::cout << container::ContainerHeaderToStr(output_data.data(),
                                                   output_data.size())
                << reconstructed_obu_log;
    }
  } else if (params.output_format == Format::kAvif) {
    if (!container::WrapAvif(image, final_codec_bytes, num_final_codec_bytes,
                             output_data)) {
      std::cerr << "Error: Unable to create AVIF file from input" << std::endl;
      return 1;
    }
    std::cout << std::endl
              << "AVIF output (" << output_data.size() << " bytes)"
              << std::endl;
    if (params.print_info) {
      std::cout << container::BmffBoxesToStr(output_data.data(),
                                             output_data.size())
                << reconstructed_obu_log;
    }
  } else if (params.output_format == Format::kObu) {
    if (codec != container::kContainerCodecObu) {
      std::cerr << "Error: Unable to get OBU from non-AV1 input" << std::endl;
      return 1;
    }
    output_data = container::Data(final_codec_bytes,
                                  final_codec_bytes + num_final_codec_bytes);
    std::cout << std::endl
              << "OBU output (" << output_data.size() << " bytes)"
              << std::endl;
    if (params.print_info) {
      std::cout << reconstructed_obu_log;
    }
  }

  // Write the file.
  if (!params.output_path.empty()) {
    std::ofstream output_file(params.output_path, std::ios_base::binary);
    output_file.write(reinterpret_cast<const char*>(output_data.data()),
                      output_data.size());
  }
  return 0;
}
