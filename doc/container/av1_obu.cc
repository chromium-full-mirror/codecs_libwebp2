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

#include "./av1_obu.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <sstream>
#include <string>

#include "./bit_packing.h"

namespace container {

//------------------------------------------------------------------------------

namespace {

// OBU type to name.
// See "AV1 Bitstream & Decoding Process Specification - 6.2.2."
static const char* kObuTypeStr[] = {"Reserved OBU",
                                    "OBU_SEQUENCE_HEADER",
                                    "OBU_TEMPORAL_DELIMITER",
                                    "OBU_FRAME_HEADER",
                                    "OBU_TILE_GROUP",
                                    "OBU_METADATA",
                                    "OBU_FRAME",
                                    "OBU_REDUNDANT_FRAME_HEADER",
                                    "OBU_TILE_LIST",
                                    "Reserved OBU",
                                    "Reserved OBU",
                                    "Reserved OBU",
                                    "Reserved OBU",
                                    "Reserved OBU",
                                    "Reserved OBU",
                                    "OBU_PADDING"};
static_assert(array_size(kObuTypeStr) == (1 << 4),
              "kObuTypeStr should match 4-bit long values.");

// Variable-length unsigned integer coding. Used to code an OBU size.
// Adapted from "AV1 Bitstream & Decoding Process Specification - 4.10.5."
void WriteLeb128(uint32_t value, Data& data) {
  for (int i = 0; i < 8; ++i) {
    uint8_t leb128_byte = value & 0x7F;
    value >>= 7;
    if (value > 0) leb128_byte |= 0x80;
    data.emplace_back(leb128_byte);
    if (value == 0) break;
  }
}
uint32_t ReadLeb128(BitUnpacker& bit_unpacker) {
  uint32_t value = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t leb128_byte = bit_unpacker.DecodeUInt(/*num_bits=*/8);  // f(8)
    value |= ((leb128_byte & 0x7F) << (i * 7));
    if (!(leb128_byte & 0x80)) break;
  }
  return value;
}

// Returns the number of bits necessary to store the 'value'.
// Equivalent to floor(log2(value))+1.
uint32_t NumBits(uint32_t value) {
  uint32_t num_bits = 1;
  while ((value >> num_bits) > 0) ++num_bits;
  return num_bits;
}

// See "AV1 Bitstream & Decoding Process Specification - 5.9.16."
uint32_t TileLog2(uint32_t blk_size, uint32_t target) {
  uint32_t k = 0;
  while ((blk_size << k) < target) ++k;
  return k;
}

//------------------------------------------------------------------------------
// The following functions implement the AV1 Bitstream & Decoding Process Spec.

void CodeObuHeader(uint32_t obu_type, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.3.2."
  obu.Force("obu_forbidden_bit", false);
  obu.Code("obu_type", /*num_bits=*/4, obu_type);
  obu.Force("obu_extension_flag", false);
  obu.Force("obu_has_size_field", true);
  obu.Force("obu_reserved_1bit", false);
  // Next is payload size followed by payload.
}

void CodeColorConfig(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.5.2."
  obu.Code("high_bitdepth", f.high_bitdepth);
  if (f.seq_profile == 2 && f.high_bitdepth) {
    obu.Code("twelve_bit", f.twelve_bit);
  } else {
    obu.Set("twelve_bit", f.twelve_bit, false);
  }
  if (f.seq_profile == 1) {
    // mono_chrome set to false
  } else {
    obu.Force("mono_chrome", false);
  }
  obu.Code("color_description_present_flag", f.color_description_present_flag);
  if (f.color_description_present_flag) {
    obu.Code("color_primaries", /*num_bits=*/8, f.color_primaries);
    obu.Code("transfer_characteristics", /*num_bits=*/8,
             f.transfer_characteristics);
    obu.Code("matrix_coefficients", /*num_bits=*/8, f.matrix_coefficients);
  } else {
    obu.Set("color_primaries", f.color_primaries, 0);
    obu.Set("transfer_characteristics", f.transfer_characteristics, 0);
    obu.Set("matrix_coefficients", f.matrix_coefficients, 0);
  }

  obu.Code("color_range", f.color_range);
  if (f.seq_profile == 0) {
    obu.Set("subsampling_x", f.subsampling_x, 1);
    obu.Set("subsampling_y", f.subsampling_y, 1);
  } else if (f.seq_profile == 1) {
    obu.Set("subsampling_x", f.subsampling_x, 0);
    obu.Set("subsampling_y", f.subsampling_y, 0);
  } else {
    if (f.twelve_bit) {
      obu.Code("subsampling_x", /*num_bits=*/1, f.subsampling_x);
      if (f.subsampling_x) {
        obu.Code("subsampling_y", /*num_bits=*/1, f.subsampling_y);
      } else {
        obu.Set("subsampling_y", f.subsampling_y, 0);
      }
    } else {
      obu.Set("subsampling_x", f.subsampling_x, 1);
      obu.Set("subsampling_y", f.subsampling_y, 0);
    }
  }
  if (f.subsampling_x && f.subsampling_y) {
    obu.Code("chroma_sample_position", /*num_bits=*/2,
             f.chroma_sample_position);
  }
  obu.Code("separate_uv_delta_q", f.separate_uv_delta_q);
}

void CodeSequenceHeaderObu(uint32_t width, uint32_t height, ObuFeatures& f,
                           BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
  obu.Code("seq_profile", /*num_bits=*/3, f.seq_profile);
  obu.Force("still_picture", true);
  obu.Force("reduced_still_picture_header", true);
  obu.Code("seq_level_idx_0", /*num_bits=*/5, f.seq_level_idx_0);

  const uint32_t width_mins_1 = width - 1, height_mins_1 = height - 1;
  obu.Force("frame_width_bits_minus_1", /*num_bits=*/4,
            NumBits(width_mins_1) - 1);
  obu.Force("frame_height_bits_minus_1", /*num_bits=*/4,
            NumBits(height_mins_1) - 1);
  obu.Force("width - 1", NumBits(width_mins_1), width - 1);
  obu.Force("height - 1", NumBits(height_mins_1), height - 1);

  obu.Code("use_128x128_superblock", f.use_128x128_superblock);
  obu.Code("enable_filter_intra", f.enable_filter_intra);
  obu.Code("enable_intra_edge_filter", f.enable_intra_edge_filter);
  obu.Code("enable_superres", f.enable_superres);
  obu.Code("enable_cdef", f.enable_cdef);
  obu.Code("enable_restoration", f.enable_restoration);

  CodeColorConfig(f, obu);

  obu.Force("film_grain_params_present", false);
}

//------------------------------------------------------------------------------

void CodeSuperresParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.8."
  if (f.enable_superres) {
    obu.Code("use_superres", f.use_superres);
  } else {
    obu.Set("use_superres", f.use_superres, false);
  }
  if (f.use_superres) {
    obu.Code("coded_denom", /*num_bits=*/3, f.coded_denom);
  } else {
    obu.Set("coded_denom", f.coded_denom, 0);
  }
  obu.Force("render_and_frame_size_different", false);
}

void CodeTileInfo(uint32_t width, uint32_t height, ObuFeatures& f,
                  BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.15."
  const uint32_t mi_cols = 2 * ((width + 7) >> 3);
  const uint32_t mi_rows = 2 * ((height + 7) >> 3);
  const uint32_t sb_cols = f.use_128x128_superblock
                               ? ((mi_cols + 31) >> 5)
                               : ((mi_cols + 15) >> 4);
  const uint32_t sb_rows = f.use_128x128_superblock
                               ? ((mi_rows + 31) >> 5)
                               : ((mi_rows + 15) >> 4);
  const uint32_t sb_shift = f.use_128x128_superblock ? 5 : 4;
  const uint32_t sb_size = sb_shift + 2;
  const uint32_t max_tile_width_sb = /*MAX_TILE_WIDTH=*/4096 >> sb_size;
  uint32_t max_tile_area_sb = (/*MAX_TILE_AREA=*/4096 * 2304) >> (2 * sb_size);
  const uint32_t min_log2_tile_cols = TileLog2(max_tile_width_sb, sb_cols);
  const uint32_t max_log2_tile_cols =
      TileLog2(1, std::min(sb_cols, /*MAX_TILE_COLS=*/64u));
  const uint32_t max_log2_tile_rows =
      TileLog2(1, std::min(sb_rows, /*MAX_TILE_ROWS=*/64u));
  const uint32_t min_log2_tiles = std::max(
      min_log2_tile_cols, TileLog2(max_tile_area_sb, sb_rows * sb_cols));

  obu.Code("uniform_tile_spacing_flag", f.uniform_tile_spacing_flag);
  if (f.uniform_tile_spacing_flag) {
    obu.CodeOnesTillEqual("tile_cols_log2", f.tile_cols_log2,
                          min_log2_tile_cols, max_log2_tile_cols);
    const uint32_t min_log2_tile_rows =
        (min_log2_tiles < f.tile_cols_log2)
            ? 0
            : (min_log2_tiles - f.tile_cols_log2);
    obu.CodeOnesTillEqual("tile_rows_log2", f.tile_rows_log2,
                          min_log2_tile_rows, max_log2_tile_rows);
  } else {
    uint32_t tile_cols = 0, widest_tile_sb = 0;
    for (uint32_t start_sb = 0; start_sb < sb_cols; ++tile_cols) {
      const uint32_t maxWidth = std::min(sb_cols - start_sb, max_tile_width_sb);
      obu.NonSymmetric(
          "width_in_sbs_minus_1[" + std::to_string(tile_cols) + "]",
          f.width_in_sbs_minus_1[tile_cols], maxWidth - 1);
      const uint32_t size_sb = f.width_in_sbs_minus_1[tile_cols] + 1;
      widest_tile_sb = std::max(size_sb, widest_tile_sb);
      start_sb += size_sb;
    }
    f.tile_cols_log2 = TileLog2(1, tile_cols);

    max_tile_area_sb = (min_log2_tiles > 0)
                           ? ((sb_rows * sb_cols) >> (min_log2_tiles + 1))
                           : sb_rows * sb_cols;
    const uint32_t max_tile_height_sb =
        std::max(max_tile_area_sb / widest_tile_sb, 1u);

    uint32_t tile_rows = 0;
    for (uint32_t start_sb = 0; start_sb < sb_rows; ++tile_rows) {
      const uint32_t maxHeight =
          std::min(sb_rows - start_sb, max_tile_height_sb);
      obu.NonSymmetric(
          "height_in_sbs_minus_1[" + std::to_string(tile_rows) + "]",
          f.height_in_sbs_minus_1[tile_rows], maxHeight - 1);
      const uint32_t size_sb = f.height_in_sbs_minus_1[tile_rows] + 1;
      start_sb += size_sb;
    }
    f.tile_rows_log2 = TileLog2(1, tile_rows);
  }

  if (f.tile_cols_log2 > 0 || f.tile_rows_log2 > 0) {
    const uint32_t tile_log2 = f.tile_rows_log2 + f.tile_cols_log2;
    obu.Code("context_update_tile_id", tile_log2, f.context_update_tile_id);
    obu.Code("tile_size_bytes_minus_1", /*num_bits=*/2,
             f.tile_size_bytes_minus_1);
  } else {
    obu.Set("context_update_tile_id", f.context_update_tile_id, 0);
    obu.Set("tile_size_bytes_minus_1", f.tile_size_bytes_minus_1, 0);
  }
}

void CodeQuantizationParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.12."
  obu.Code("base_q_idx", /*num_bits=*/8, f.base_q_idx);
  obu.CodeDelta("delta_qydc", /*num_bits=*/1 + 6, f.delta_qydc);

  // Assuming !mono_chrome.
  if (f.separate_uv_delta_q) {
    obu.Code("diff_uv_delta", f.diff_uv_delta);
  } else {
    obu.Set("diff_uv_delta", f.diff_uv_delta, false);
  }
  obu.CodeDelta("delta_qudc", /*num_bits=*/1 + 6, f.delta_qudc);
  obu.CodeDelta("delta_quac", /*num_bits=*/1 + 6, f.delta_quac);
  if (f.diff_uv_delta) {
    obu.CodeDelta("delta_qvdc", /*num_bits=*/1 + 6, f.delta_qvdc);
    obu.CodeDelta("delta_qvac", /*num_bits=*/1 + 6, f.delta_qvac);
  } else {
    obu.Set("delta_qvdc", f.delta_qvdc, f.delta_qudc);
    obu.Set("delta_qvac", f.delta_qvac, f.delta_quac);
  }

  obu.Code("using_qmatrix", f.using_qmatrix);
  if (f.using_qmatrix) {
    obu.Code("qm_y", /*num_bits=*/4, f.qm_y);
    obu.Code("qm_u", /*num_bits=*/4, f.qm_u);
    if (f.separate_uv_delta_q) {
      obu.Code("qm_v", /*num_bits=*/4, f.qm_v);
    } else {
      obu.Set("qm_v", f.qm_v, f.qm_u);
    }
  } else {
    obu.Set("qm_y", f.qm_y, 0);
    obu.Set("qm_u", f.qm_u, 0);
    obu.Set("qm_v", f.qm_v, 0);
  }
}

void CodeDeltaQParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.17."
  if (f.base_q_idx > 0) {
    obu.Code("delta_q_present", f.delta_q_present);
  } else {
    obu.Set("delta_q_present", f.delta_q_present, false);
  }
  if (f.delta_q_present) {
    obu.Code("delta_q_res", /*num_bits=*/2, f.delta_q_res);
  } else {
    obu.Set("delta_q_res", f.delta_q_res, 0);
  }
}

void CodeDeltaLfParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.18."
  if (f.delta_q_present && !f.allow_intrabc) {
    obu.Code("delta_lf_present", f.delta_lf_present);
  } else {
    obu.Set("delta_lf_present", f.delta_lf_present, false);
  }
  if (f.delta_lf_present) {
    obu.Code("delta_lf_res", /*num_bits=*/2, f.delta_lf_res);
    obu.Code("delta_lf_multi", f.delta_lf_multi);
  } else {
    obu.Set("delta_lf_res", f.delta_lf_res, 0);
    obu.Set("delta_lf_multi", f.delta_lf_multi, false);
  }
}

void CodeLoopFilterParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.11."
  obu.Code("loop_filter_level[0]", /*num_bits=*/6, f.loop_filter_level[0]);
  obu.Code("loop_filter_level[1]", /*num_bits=*/6, f.loop_filter_level[1]);
  // Assuming !mono_chrome.
  if ((f.loop_filter_level[0] | f.loop_filter_level[0]) > 0) {
    obu.Code("loop_filter_level[2]", /*num_bits=*/6, f.loop_filter_level[2]);
    obu.Code("loop_filter_level[3]", /*num_bits=*/6, f.loop_filter_level[3]);
  } else {
    obu.Set("loop_filter_level[2]", f.loop_filter_level[2], 0);
    obu.Set("loop_filter_level[3]", f.loop_filter_level[3], 0);
  }
  obu.Code("loop_filter_sharpness", /*num_bits=*/3, f.loop_filter_sharpness);
  obu.Code("loop_filter_delta_enabled", f.loop_filter_delta_enabled);
  if (f.loop_filter_delta_enabled) {
    obu.Code("loop_filter_delta_update", f.loop_filter_delta_update);
    if (f.loop_filter_delta_update) {
      for (uint32_t i = 0; i < /*TOTAL_REFS_PER_FRAME=*/8; i++) {
        obu.Code("update_ref_delta[" + std::to_string(i) + "]",
                 f.update_ref_delta[i]);
        if (f.update_ref_delta[i]) {
          obu.Code("loop_filter_ref_deltas[" + std::to_string(i) + "]",
                   /*num_bits=*/1 + 6, f.loop_filter_ref_deltas[i]);
        }
      }
      for (uint32_t i = 0; i < 2; i++) {
        obu.Code("update_mode_delta[" + std::to_string(i) + "]",
                 f.update_mode_delta[i]);
        if (f.update_mode_delta[i]) {
          obu.Code("loop_filter_mode_deltas[" + std::to_string(i) + "]",
                   /*num_bits=*/1 + 6, f.loop_filter_mode_deltas[i]);
        }
      }
    }
  }
}

void CodeCdefParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.19."
  obu.Code("cdef_damping_minus_3", /*num_bits=*/2, f.cdef_damping_minus_3);
  obu.Code("cdef_bits", /*num_bits=*/2, f.cdef_bits);
  for (uint32_t i = 0; i < (1 << f.cdef_bits); i++) {
    obu.Code("cdef_y_pri_strength[i]", /*num_bits=*/4,
             f.cdef_y_pri_strength[i]);
    obu.Code("cdef_y_sec_strength[i]", /*num_bits=*/2,
             f.cdef_y_sec_strength[i]);
    // Assuming !mono_chrome.
    obu.Code("cdef_uv_pri_strength[i]", /*num_bits=*/4,
             f.cdef_uv_pri_strength[i]);
    obu.Code("cdef_uv_sec_strength[i]", /*num_bits=*/2,
             f.cdef_uv_sec_strength[i]);
  }
}

void CodeLrParams(ObuFeatures& f, BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.20."
  bool uses_lr = false;
  bool uses_chroma_lr = false;
  // Assuming !mono_chrome.
  const uint32_t kNumPlanes = 3;
  for (uint32_t i = 0; i < kNumPlanes; i++) {
    obu.Code("lr_type[i]", /*num_bits=*/2, f.lr_type[i]);
    if (f.lr_type[i] != /*RESTORE_NONE=*/0) {
      uses_lr = true;
      if (i > 0) {
        uses_chroma_lr = true;
      }
    }
  }
  if (uses_lr) {
    obu.CodeOnesTillEqual("lr_unit_shift", f.lr_unit_shift,
                          /*min=*/(f.use_128x128_superblock ? 1 : 0),
                          /*max=*/2);
  } else {
    obu.Set("lr_unit_shift", f.lr_unit_shift, 0);
  }
  if (f.subsampling_x && f.subsampling_y && uses_chroma_lr) {
    obu.Code("lr_uv_shift", /*num_bits=*/1, f.lr_uv_shift);
  } else {
    obu.Set("lr_uv_shift", f.lr_uv_shift, 0);
  }
}

void CodeFrameHeaderObu(uint32_t width, uint32_t height, ObuFeatures& f,
                        BitCoder& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.9."
  obu.Code("disable_cdf_update", f.disable_cdf_update);
  obu.Code("allow_screen_content_tools", f.allow_screen_content_tools);

  // See "AV1 Bitstream & Decoding Process Specification - 5.9.5."
  CodeSuperresParams(f, obu);

  if (f.allow_screen_content_tools) {
    obu.Code("allow_intrabc", f.allow_intrabc);
  } else {
    obu.Set("allow_intrabc", f.allow_intrabc, false);
  }

  CodeTileInfo(width, height, f, obu);
  CodeQuantizationParams(f, obu);

  // See "AV1 Bitstream & Decoding Process Specification - 5.9.14."
  obu.Force("segmentation_enabled", false);

  CodeDeltaQParams(f, obu);
  CodeDeltaLfParams(f, obu);

  // Because !segmentation_enabled:
  bool coded_lossless = (f.base_q_idx == 0 && f.delta_qydc == 0 &&
                         f.delta_quac == 0 && f.delta_qudc == 0 &&
                         f.delta_qvac == 0 && f.delta_qvdc == 0);
  if (!coded_lossless && !f.allow_intrabc) {
    CodeLoopFilterParams(f, obu);
  }
  if (!coded_lossless && !f.allow_intrabc && f.enable_cdef) {
    CodeCdefParams(f, obu);
  }
  if (!coded_lossless && !f.allow_intrabc && f.enable_restoration) {
    CodeLrParams(f, obu);
  }
  // See "AV1 Bitstream & Decoding Process Specification - 5.9.20."
  if (!coded_lossless) {
    obu.Code("tx_mode", /*num_bits=*/1, f.tx_mode);
  } else {
    obu.Set("tx_mode", f.tx_mode, 0);  // ONLY_4X4
  }

  obu.Code("reduced_tx_set", f.reduced_tx_set);
}

void CodeByteAlignment(BitWriter& obu) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.3.5."
  const uint32_t unaligned_bits = obu.GetNumBits() % 8;
  if (unaligned_bits > 0) {
    obu.Force("byte_alignment", 8 - unaligned_bits, 0);
  }
}

//------------------------------------------------------------------------------

// Decodes the payload size of an OBU 'data' chunk.
// Returns false in case of parsing error.
bool ParseObuHeader(const uint8_t data[], size_t data_size, uint8_t& obu_type,
                    size_t& obu_header_size, size_t& obu_payload_size) {
  // See "AV1 Bitstream & Decoding Process Specification - 5.3.1."
  BitUnpacker obu(data, data_size);
  /*obu_forbidden_bit=*/obu.DecodeBool();
  obu_type = obu.DecodeUInt(4);
  const bool obu_extension_flag = obu.DecodeBool();
  const bool obu_has_size_field = obu.DecodeBool();
  /*obu_reserved_1bit=*/obu.DecodeBool();
  if (obu_extension_flag) {
    /*obu_extension_header=*/obu.DecodeUInt(8);
  }
  obu_payload_size = (obu_has_size_field ? ReadLeb128(obu)
                                         : (data_size - obu.GetNumReadBytes()));
  obu_header_size = obu.GetNumReadBytes();
  return (obu.Ok() && obu_header_size + obu_payload_size <= data_size);
}

//------------------------------------------------------------------------------

bool ParseSequenceHeaderObu(const uint8_t obu_payload[],
                            size_t obu_payload_size,
                            SequenceHeaderObu& header) {
  BitUnpacker obu(obu_payload, obu_payload_size);
  // See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
  header.seq_profile = obu.DecodeUInt(3);
  const bool still_picture = obu.DecodeBool();
  const bool reduced_still_picture_header = obu.DecodeBool();
  if (!still_picture || !reduced_still_picture_header) return false;
  header.seq_level_idx_0 = obu.DecodeUInt(5);
  const uint8_t frame_width_bits = 1 + obu.DecodeUInt(4);
  const uint8_t frame_height_bits = 1 + obu.DecodeUInt(4);
  header.width = 1 + obu.DecodeUInt(frame_width_bits);
  header.height = 1 + obu.DecodeUInt(frame_height_bits);
  /*use_128x128_superblock=*/obu.DecodeBool();
  /*enable_filter_intra=*/obu.DecodeBool();
  /*enable_intra_edge_filter=*/obu.DecodeBool();
  /*enable_superres=*/obu.DecodeBool();
  /*enable_cdef=*/obu.DecodeBool();
  /*enable_restoration=*/obu.DecodeBool();
  // color_config()
  header.high_bitdepth = obu.DecodeBool();
  header.twelve_bit = false;  // This is not in the spec.
  uint8_t bitdepth = 8;       // This is not in the spec.
  if (header.seq_profile == 2 && header.high_bitdepth) {
    header.twelve_bit = obu.DecodeBool();
    bitdepth = header.twelve_bit ? 12 : 10;
  } else if (header.seq_profile <= 2) {
    bitdepth = header.high_bitdepth ? 10 : 8;
  }
  header.mono_chrome = (header.seq_profile != 1) && obu.DecodeBool();
  const bool color_description_present_flag = obu.DecodeBool();
  const uint8_t color_primaries =
      color_description_present_flag ? obu.DecodeUInt(8) : /*CP_UNSPECIFIED=*/2;
  const uint8_t transfer_characteristics =
      color_description_present_flag ? obu.DecodeUInt(8) : /*TC_UNSPECIFIED=*/2;
  const uint8_t matrix_coefficients =
      color_description_present_flag ? obu.DecodeUInt(8) : /*MC_UNSPECIFIED=*/2;
  if (header.mono_chrome) {
    /*color_range=*/obu.DecodeUInt(1);
    header.subsampling_x = 1;
    header.subsampling_y = 1;
    header.chroma_sample_position = /*CSP_UNKNOWN=*/0;
  } else if (color_primaries == /*CP_BT_709=*/1 &&
             transfer_characteristics == /*TC_SRGB=*/13 &&
             matrix_coefficients == /*MC_IDENTITY=*/0) {
    header.subsampling_x = 1;
    header.subsampling_y = 1;
    header.chroma_sample_position = /*CSP_UNKNOWN=*/0;  // Not in spec.
  } else {
    /*color_range=*/obu.DecodeUInt(1);
    if (header.seq_profile == 0) {
      header.subsampling_x = 1;
      header.subsampling_y = 1;
    } else if (header.seq_profile == 1) {
      header.subsampling_x = 0;
      header.subsampling_y = 0;
    } else if (bitdepth == 12) {
      header.subsampling_x = obu.DecodeUInt(1);
      if (header.subsampling_x) {
        header.subsampling_y = obu.DecodeUInt(1);
      } else {
        header.subsampling_y = 0;
      }
    } else {
      header.subsampling_x = 1;
      header.subsampling_y = 0;
    }
    if (header.subsampling_x && header.subsampling_y) {
      header.chroma_sample_position = obu.DecodeUInt(2);
    } else {
      header.chroma_sample_position = /*CSP_UNKNOWN=*/0;  // Not in spec.
    }
  }

  /*film_grain_params_present=*/obu.DecodeBool();
  return obu.Ok();
}

}  // namespace

bool UnwrapObuSequenceHeader(const uint8_t data[], size_t data_size,
                             SequenceHeaderObu& header) {
  size_t position = 0;
  while (position < data_size) {
    uint8_t obu_type;
    size_t obu_header_size, obu_payload_size;
    if (!ParseObuHeader(data + position, data_size - position, obu_type,
                        obu_header_size, obu_payload_size)) {
      return false;
    }
    position += obu_header_size;
    if (obu_type == /*OBU_SEQUENCE_HEADER=*/1) {
      return ParseSequenceHeaderObu(data + position, obu_payload_size, header);
    }
    position += obu_payload_size;
  }
  return false;
}

//------------------------------------------------------------------------------

namespace {

// Uses the dtor to finalize the encoding of the header.
class ObuWriter : public BitWriter {
 public:
  ObuWriter(uint32_t obu_type, Data& output) : BitWriter(output) {
    // See "AV1 Bitstream & Decoding Process Specification - 5.3.2."
    CodeObuHeader(obu_type, *this);

    output.insert(output.end(), bit_packer_.GetBytes(),
                  bit_packer_.GetBytes() + bit_packer_.GetNumBytes());
    bit_packer_ = BitPacker();  // Reset for accurate payload size.
  }
  ~ObuWriter() override {
    // See "AV1 Bitstream & Decoding Process Specification - 5.3.4."
    if (bit_packer_.GetNumBits() % 8 != 0) {  // Padding till next whole byte.
      bit_packer_.EncodeBool(true);
      while (bit_packer_.GetNumBits() % 8 != 0) bit_packer_.EncodeBool(false);
    }

    // Write the size of the payload, that was unknown until now.
    // See "AV1 Bitstream & Decoding Process Specification - 5.3.1."
    WriteLeb128(bit_packer_.GetNumBytes(), output_);
  }
};

using ObuReader = BitReader;

}  // namespace

//------------------------------------------------------------------------------

bool WrapObu(uint32_t width, uint32_t height, const ObuFeatures& features,
             const uint8_t tiles_bytes[], size_t num_tiles_bytes, Data& data,
             std::string& log) {
  data.clear();
  ObuFeatures f;  // Copy 'features' for safety and to avoid a const_cast().
  std::memcpy(&f, &features, sizeof(f));

  {
    log += "\nOBU_SEQUENCE_HEADER\n";
    ObuWriter sequence_header(/*obu_type=*/1, data);
    CodeSequenceHeaderObu(width, height, f, sequence_header);
    log += sequence_header.GetLog();
    if (!sequence_header.Ok()) return false;
  }

  {
    // OBU_FRAME contains an OBU_FRAME_HEADER and an OBU_TILE_GROUP.
    // It is shorter than adding the two separately.
    // See "AV1 Bitstream & Decoding Process Specification - 5.10."
    log += "\nOBU_FRAME\n";
    ObuWriter frame_header(/*obu_type=*/6, data);
    CodeFrameHeaderObu(width, height, f, frame_header);
    CodeByteAlignment(frame_header);
    frame_header.AppendBytes("OBU_TILE_GROUP", tiles_bytes, num_tiles_bytes);
    log += frame_header.GetLog();
    if (!frame_header.Ok()) return false;
  }

  // Make sure the features were not modified by mistake.
  assert(std::memcmp(&f, &features, sizeof(f)) == 0);
  return true;
}

bool UnwrapObu(const uint8_t data[], size_t data_size, uint32_t width,
               uint32_t height, ObuFeatures& features,
               size_t& offset_till_tiles_bytes, size_t& num_tiles_bytes,
               std::string& log) {
  features = {};
  bool saw_sequence_header = false;
  bool saw_frame_header = false;
  bool saw_tile_group = false;

  size_t position = 0;  // In bytes within 'data' (OBU sequence).
  uint8_t obu_type = 0;
  while (position < data_size) {
    size_t obu_header_size, obu_payload_size;
    if (!ParseObuHeader(data + position, data_size - position, obu_type,
                        obu_header_size, obu_payload_size)) {
      return false;
    }
    position += obu_header_size;

    log += "\n";
    log += kObuTypeStr[obu_type];
    log += "\n";

    size_t position_in_obu = 0;  // In bytes within 'data+position' (OBU unit).
    bool parsed_obu = false;

    // See "AV1 Bitstream & Decoding Process Specification - 6.2.2."
    if (obu_type == /*OBU_SEQUENCE_HEADER=*/1) {
      saw_sequence_header = true;
      ObuReader sequence_header(data + position, obu_payload_size);
      CodeSequenceHeaderObu(width, height, features, sequence_header);
      log += sequence_header.GetLog();
      if (!sequence_header.Ok()) return false;
      position_in_obu += sequence_header.GetNumBytes();
      parsed_obu = true;
    } else if (obu_type == /*OBU_FRAME_HEADER=*/3 ||
               obu_type == /*OBU_FRAME=*/6) {
      if (saw_frame_header) {
        log += "/!\\ Encountered frame data twice\n";
        return false;
      }
      saw_frame_header = true;
      if (!saw_sequence_header) {
        log += "/!\\ Encountered frame data before OBU_SEQUENCE_HEADER\n";
        return false;
      }
      ObuReader frame_header(data + position, obu_payload_size);
      CodeFrameHeaderObu(width, height, features, frame_header);
      log += frame_header.GetLog();
      if (!frame_header.Ok()) return false;
      position_in_obu += frame_header.GetNumBytes();
      parsed_obu = true;
    }
    if (obu_type == /*OBU_TILE_GROUP=*/4 || obu_type == /*OBU_FRAME=*/6) {
      if (saw_tile_group) {
        log += "/!\\ Encountered tile group data twice\n";
        return false;
      }
      saw_tile_group = true;
      if (!saw_frame_header) {
        log += "/!\\ Encountered tile group data before frame data\n";
        return false;
      }

      offset_till_tiles_bytes = position + position_in_obu;
      num_tiles_bytes = obu_payload_size - position_in_obu;
      log += std::to_string(num_tiles_bytes) + " bytes of OBU_TILE_GROUP\n";
      parsed_obu = true;
    }
    if (!parsed_obu) log += "Skipped\n";
    position += obu_payload_size;
  }
  if (!saw_tile_group) {
    log += "/!\\ No tile group data found\n";
    return false;
  }
  return true;
}

//------------------------------------------------------------------------------
// Stripped OBU: OBU sequence fields without OBU headers nor fields with a known
//               or forced value.

namespace {

class StrippedObuWriter : public BitWriter {
 public:
  using BitWriter::BitWriter;

  // Do not write forced fields, because their value is known.
  void Force(const std::string& name, uint32_t num_bits,
             uint32_t value) override {
    uint32_t field = value;
    Set(name, field, value);  // For logging.
  }
};

class StrippedObuReader : public BitReader {
 public:
  using BitReader::BitReader;

  // Do not read forced fields, because their value is known.
  void Force(const std::string& name, uint32_t num_bits,
             uint32_t value) override {
    uint32_t field = value;
    Set(name, field, value);  // For logging.
  }
};

}  // namespace

bool WrapStrippedObu(uint32_t width, uint32_t height,
                     const ObuFeatures& features, const uint8_t tiles_bytes[],
                     size_t num_tiles_bytes, Data& data, std::string& log) {
  data.clear();
  ObuFeatures f;  // Copy 'features' for safety and to avoid a const_cast().
  std::memcpy(&f, &features, sizeof(f));  // Because of memcmp() below.

  {
    StrippedObuWriter stripped_obu(data);
    CodeSequenceHeaderObu(width, height, f, stripped_obu);
    CodeFrameHeaderObu(width, height, f, stripped_obu);
    CodeByteAlignment(stripped_obu);  // TODO(yguyon): Check it at unwrapping
    log += stripped_obu.GetLog();
    if (!stripped_obu.Ok()) return false;
  }

  {
    // TODO(yguyon): Maybe the minimum can be more than 1 byte
    EncodeVarInt(num_tiles_bytes, /*min=*/1u, kMaxVarInt, data);
    data.insert(data.end(), tiles_bytes, tiles_bytes + num_tiles_bytes);
  }

  // Make sure the features were not modified by mistake.
  assert(std::memcmp(&f, &features, sizeof(f)) == 0);
  return true;
}

bool UnwrapStrippedObu(const uint8_t data[], size_t data_size, uint32_t width,
                       uint32_t height, ObuFeatures& features,
                       size_t& offset_till_tiles_bytes, size_t& num_tiles_bytes,
                       std::string& log) {
  StrippedObuReader stripped_obu(data, data_size);
  CodeSequenceHeaderObu(width, height, features, stripped_obu);
  CodeFrameHeaderObu(width, height, features, stripped_obu);
  log += stripped_obu.GetLog();
  if (!stripped_obu.Ok()) return false;

  uint32_t data_pos = stripped_obu.GetNumBytes();
  uint32_t num_bytes;
  if (!DecodeVarInt(/*min=*/1u, kMaxVarInt, data, data_size, data_pos,
                    num_bytes)) {
    log += "Bad payload size (corrupted or truncated bitstream)\n";
    return false;
  } else if (data_pos + num_bytes != data_size) {
    log += "Bad payload size (corrupted or truncated bitstream, expected " +
           std::to_string(data_size - data_pos) + " instead of " +
           std::to_string(num_bytes) + ")\n";
    return false;
  }
  offset_till_tiles_bytes = data_pos;
  num_tiles_bytes = num_bytes;
  return true;
}

//------------------------------------------------------------------------------

std::string ObuUnitsToStr(const uint8_t data[], size_t data_size) {
  std::stringstream ss;

  size_t position = 0;
  while (position < data_size) {
    uint8_t obu_type;
    size_t obu_header_size, obu_payload_size;
    if (!ParseObuHeader(data + position, data_size - position, obu_type,
                        obu_header_size, obu_payload_size)) {
      ss << "  Bad OBU header at " << position << " ("
         << static_cast<int>(data[position]) << ")" << std::endl;
      return ss.str();
    }
    ss << "  " << kObuTypeStr[obu_type] << " (" << obu_header_size << "+"
       << obu_payload_size << " bytes)" << std::endl;
    position += obu_header_size;

    // Display the first few bytes as binary digits.
    ss << "   ";
    for (uint32_t i = 0; i < std::min(obu_payload_size, (size_t)10); ++i) {
      ss << ' ' << std::bitset<8>(data[position + i]);
    }
    if (obu_payload_size > 10) ss << " ...";  // Indicate missing bits.
    ss << std::endl;

    position += obu_payload_size;
  }
  return ss.str();
}

//------------------------------------------------------------------------------

}  // namespace container
