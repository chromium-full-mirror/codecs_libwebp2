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

#ifndef CONTAINER_AV1_OBU_H_
#define CONTAINER_AV1_OBU_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "./container.h"

namespace container {

//------------------------------------------------------------------------------

// Subset of the features stored in a Sequence Header OBU.
// These fields are necessary to encode a valid AVIF/MIAF header.
// See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
struct SequenceHeaderObu {
  uint32_t width;
  uint32_t height;
  uint8_t seq_profile;
  uint8_t seq_level_idx_0;
  bool high_bitdepth;
  bool twelve_bit;
  bool mono_chrome;
  uint8_t subsampling_x;
  uint8_t subsampling_y;
  uint8_t chroma_sample_position;
};

// Decodes the SequenceHeaderObu 'header' from an entire 'data' OBU bitstream.
// Used to get a few fields without decoding everything or integrity checking.
// Returns false in case of parsing error.
bool UnwrapObuSequenceHeader(const uint8_t data[], size_t data_size,
                             SequenceHeaderObu& header);

//------------------------------------------------------------------------------

// Most features stored in a Sequence Header OBU and a Frame Header OBU.
// Some fields are marked as "maybe not needed", meaning they might be omitted
// from this struct (forced to some value instead) without impacting the range
// of accepted AVIF files.
// See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
struct ObuFeatures {
  // Sequence header
  uint32_t seq_profile;      // 3 bits
  uint32_t seq_level_idx_0;  // 5 bits
  bool use_128x128_superblock;
  bool enable_filter_intra;
  bool enable_intra_edge_filter;
  bool enable_superres;
  bool enable_cdef;
  bool enable_restoration;
  // Color config
  bool high_bitdepth;
  bool twelve_bit;
  bool color_description_present_flag;
  uint32_t color_primaries;           // 8 bits
  uint32_t transfer_characteristics;  // 8 bits
  uint32_t matrix_coefficients;       // 8 bits
  bool color_range;
  uint32_t subsampling_x;           // 0 or 1
  uint32_t subsampling_y;           // 0 or 1
  uint32_t chroma_sample_position;  // 2 bits  Maybe not needed
  bool separate_uv_delta_q;         // Maybe not needed

  // Frame header
  bool disable_cdf_update;
  bool allow_screen_content_tools;  // Maybe not needed
  bool reduced_tx_set;
  // Superres
  bool use_superres;
  uint32_t coded_denom;  // 3 bits
  bool allow_intrabc;
  // Tile info
  bool uniform_tile_spacing_flag;  // Maybe not needed
  uint32_t tile_cols_log2;
  uint32_t tile_rows_log2;
  uint32_t width_in_sbs_minus_1[/*AV1_MAX_TILE_COLS=*/64];
  uint32_t height_in_sbs_minus_1[/*AV1_MAX_TILE_ROWS=*/64];
  uint32_t context_update_tile_id;
  uint32_t tile_size_bytes_minus_1;  // 2 bits
  // Quantization params
  uint32_t base_q_idx;  // 8 bits
  bool diff_uv_delta;
  uint32_t delta_qydc;  // 6 bits + sign
  uint32_t delta_qudc;  // 6 bits + sign
  uint32_t delta_quac;  // 6 bits + sign
  uint32_t delta_qvdc;  // 6 bits + sign
  uint32_t delta_qvac;  // 6 bits + sign
  bool using_qmatrix;
  uint32_t qm_y;  // 4 bits
  uint32_t qm_u;  // 4 bits
  uint32_t qm_v;  // 4 bits
  // Quantizer index delta parameters
  bool delta_q_present;
  uint32_t delta_q_res;  // 2 bits
  // Loop filter delta parameters
  bool delta_lf_present;
  uint32_t delta_lf_res;  // 2 bits
  bool delta_lf_multi;
  // Loop filter params
  uint32_t loop_filter_level[4];   // 6 bits each
  uint32_t loop_filter_sharpness;  // 3 bits
  bool loop_filter_delta_enabled;
  bool loop_filter_delta_update;
  bool update_ref_delta[/*TOTAL_REFS_PER_FRAME=*/8];
  uint32_t loop_filter_ref_deltas[/*TOTAL_REFS_PER_FRAME=*/8];  // 6 bits + sign
  bool update_mode_delta[2];
  uint32_t loop_filter_mode_deltas[2];  // 6 bits + sign
  // CDEF params
  uint32_t cdef_damping_minus_3;
  uint32_t cdef_bits;
  uint32_t cdef_y_pri_strength[1 << 3];   // 4 bits
  uint32_t cdef_y_sec_strength[1 << 3];   // 0, 1, 2 or 4
  uint32_t cdef_uv_pri_strength[1 << 3];  // 4 bits
  uint32_t cdef_uv_sec_strength[1 << 3];  // 2 bits
  // Loop restoration params
  uint32_t lr_type[3];     // 2 bits each
  uint32_t lr_unit_shift;  // 0, 1 or 2
  uint32_t lr_uv_shift;    // 0 or 1
  // TX mode
  uint32_t tx_mode;  // ONLY_4X4=0, TX_MODE_LARGEST=1 or TX_MODE_SELECT=2
};

// Wraps the input 'tiles_bytes' corresponding to an image of 'width*height'
// pixels and having the 'features' into an OBU sequence 'data'.
// 'tiles_bytes' must be a valid OBU tile group payload (no OBU header).
// Written fields are also displayed in the 'log', along with any error.
// Returns false in case of error.
bool WrapObu(uint32_t width, uint32_t height, const ObuFeatures& features,
             const uint8_t tiles_bytes[], size_t num_tiles_bytes, Data& data,
             std::string& log);

// Unwraps an OBU sequence 'data' corresponding to an image of 'width*height'
// pixels into OBU 'features' and locates the 'tile_bytes' within.
// Read fields are also displayed in the 'log', along with any error.
// Does not decode the pixels. Returns false in case of error.
bool UnwrapObu(const uint8_t data[], size_t data_size, uint32_t width,
               uint32_t height, ObuFeatures& features,
               size_t& offset_till_tiles_bytes, size_t& num_tiles_bytes,
               std::string& log);

//------------------------------------------------------------------------------

// Same as WrapObu() but does not produce a valid OBU sequence. It just
// serializes the 'features' necessary for unwrapping and discards the rest.
bool WrapStrippedObu(uint32_t width, uint32_t height,
                     const ObuFeatures& features, const uint8_t tiles_bytes[],
                     size_t num_tiles_bytes, Data& data, std::string& log);

// Same as UnwrapObu() but takes WrapStrippedObu() output as input.
bool UnwrapStrippedObu(const uint8_t data[], size_t data_size, uint32_t width,
                       uint32_t height, ObuFeatures& features,
                       size_t& offset_till_tiles_bytes, size_t& num_tiles_bytes,
                       std::string& log);

//------------------------------------------------------------------------------

// Returns a string containing the OBU layout of an AV1 payload (OBU sequence).
std::string ObuUnitsToStr(const uint8_t data[], size_t data_size);

//------------------------------------------------------------------------------

}  // namespace container

#endif  // CONTAINER_AV1_OBU_H_
