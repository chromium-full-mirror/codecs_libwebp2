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

#include <cassert>
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

enum class Format { kContainer, kAvif, kDeduceFromExtension };

struct Params {
  std::string input_path;
  std::string output_path;
  Format output_format = Format::kDeduceFromExtension;
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
  uint32_t codec = 0;      // Input pixels format.
  size_t offset_till_codec_bytes, num_codec_bytes;  // Encoded input pixels.

  // Try reading the input as both formats.
  const bool input_is_container =
      container::UnwrapContainer(input_data, input.size(), image, codec,
                                 offset_till_codec_bytes, num_codec_bytes);
  const bool input_is_avif =
      !input_is_container &&
      container::UnwrapAvif(input_data, input.size(), image,
                            offset_till_codec_bytes, num_codec_bytes);
  if (input_is_avif) codec = container::kContainerCodecAv1;

  if (!input_is_container && !input_is_avif) {
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
  } else {
    assert(input_is_avif);
    std::cout << "AVIF input (" << input.size() << " bytes)" << std::endl;
    if (params.print_info) {
      std::cout << container::BmffBoxesToStr(input_data, input.size());
    }
  }

  if (params.print_info && codec == container::kContainerCodecAv1) {
    std::cout << "OBU input (" << num_codec_bytes << " bytes)" << std::endl;
    std::cout << container::ObuUnitsToStr(input_data + offset_till_codec_bytes,
                                          num_codec_bytes);
  }

  // Wrap the encoded pixels to the destination format.
  container::Data output_data;
  if (params.output_format == Format::kContainer) {
    container::WrapContainer(image, params.config, codec,
                             input_data + offset_till_codec_bytes,
                             num_codec_bytes, output_data);
  } else if (params.output_format == Format::kAvif) {
    if (!container::WrapAvif(image, input_data + offset_till_codec_bytes,
                             num_codec_bytes, output_data)) {
      std::cerr << "Error: Unable to create AVIF file from input" << std::endl;
      return 1;
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
