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
//  AOM / AV1 wrapper
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WP2_EXTRAS_AOM_UTILS_H_
#define WP2_EXTRAS_AOM_UTILS_H_

#include <map>
#include <string>
#include <vector>

#include "src/wp2/base.h"

namespace WP2 {

struct ParamsAV1 {
  float quality = 75.f;         // in [0..100]
  float alpha_quality = 100.f;  // in [0..100]
  uint32_t size = 0;            // target size if not zero (predates 'quality')
  uint32_t effort = 5;           // in [0..9]
  uint32_t threads = 1;         // 1 means no multi-threading
  uint32_t pass = 2;
  bool use_yuv444 = true;
  // if 0, CDEF and restoration filters are disabled (but deblocking is kept)
  float filter_strength = 50.f;  // in [0..100]
  std::map<int, int> extra_options;  // extra options for libaom
  enum class Tuning { kPsnr, kSsim, kVmaf, kButteraugli };
  Tuning tuning = Tuning::kPsnr;
  uint32_t bittrace = 0;         // 0=off, 1=bits, 2=bytes
  bool compact_trace = false;    // compact traces into fewer categories
  bool draw_blocks = false;      // for visual debug
  bool draw_transforms = false;
  const char* draw_mi_data = nullptr;
  bool draw_mi_number = false;  // draw value of mi data as red square (9 bits)
  bool use_libgav1_decoder = false;  // use libaom decoder if false

  // libaom only options
  bool disable_transforms = false;   // disable tf other than DCT (av1 only)
  // disable predictors other than DC (and some directional predictors for now)
  bool disable_predictors = false;
  bool disable_palette = false;     // disable palette predictor
  bool disable_block_copy = false;  // disable intra block copy
  bool disable_trellis = false;     // disable trellis (uses dropout instead)
  // force a given square block size (4, 8, 16, 32, 64, 128)
  uint32_t force_block_size = 0;

  std::string ToString() const;  // informative: prints the fields as a string
};

struct QuantAV1 {
  // Dequantization values per segment, one for DC, one for AC.
  int16_t y_dequant[8][2];
  int16_t u_dequant[8][2];
  int16_t v_dequant[8][2];
};

WP2Status CompressAV1(const ArgbBuffer& ref, const ParamsAV1& params,
                      ArgbBuffer* const decoded, std::string* const out,
                      double timing[2] = nullptr,
                      std::vector<WP2::Rectangle>* const blocks = nullptr,
                      std::vector<WP2::Rectangle>* const transforms = nullptr,
                      QuantAV1* const quant = nullptr);

WP2Status CompressAVIF(const ArgbBuffer& ref, const ParamsAV1& params,
                       ArgbBuffer* const decoded, std::string* const out,
                       double timing[2] = nullptr);

// Parses an option of libaom of the from "option-name=value", e.g. "aq-mode=1".
// Sets 'option_enum' to the corresponding enum value for use with
// 'aom_codec_control', and 'option_value' to the provided value.
// Returns true on success, false and prints a message to stderr on error
// (unknown option).
WP2_NO_DISCARD bool ParseLibaomOption(const std::string& str,
                                      int* const option_enum,
                                      int* const option_value);

}  // namespace WP2

#endif  // WP2_EXTRAS_AOM_UTILS_H_
