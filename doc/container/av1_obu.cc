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

// Variable-length unsigned integer decoding.
// Adapted from "AV1 Bitstream & Decoding Process Specification - 4.10.5."
uint32_t Leb128(BitUnpacker& bit_unpacker) {
  uint32_t value = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t leb128_byte = bit_unpacker.DecodeUInt(8);  // f(8)
    value |= ((leb128_byte & 0x7f) << (i * 7));
    if (!(leb128_byte & 0x80)) break;
  }
  return value;
}

// Decodes the payload size of an OBU 'data' chunk.
// Returns false in case of parsing error.
bool UnwrapOBU(const uint8_t data[], size_t data_size, uint8_t& obu_type,
               size_t& obu_header_size, size_t& obu_size) {
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
  obu_size =
      (obu_has_size_field ? Leb128(obu) : (data_size - obu.GetNumReadBytes()));
  obu_header_size = obu.GetNumReadBytes();
  return (obu.Ok() && obu_header_size + obu_size <= data_size);
}

}  // namespace

//------------------------------------------------------------------------------

bool UnwrapOBUSequenceHeader(const uint8_t data[], size_t data_size,
                             SequenceHeader& header) {
  size_t position = 0;
  uint8_t obu_type = 0;
  size_t obu_header_size, obu_size;
  while (position < data_size) {
    if (!UnwrapOBU(data + position, data_size - position, obu_type,
                   obu_header_size, obu_size)) {
      return false;
    }
    position += obu_header_size;
    if (obu_type == 1) break;  // Sequence Header OBU was found.
    position += obu_size;
  }
  if (obu_type != 1) return false;  // Impossible to find Sequence Header OBU.

  // See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
  BitUnpacker obu(data + position, obu_size);
  header.seq_profile = obu.DecodeUInt(3);
  const bool still_picture = obu.DecodeBool();
  const bool reduced_still_picture_header = obu.DecodeBool();
  if (!still_picture || !reduced_still_picture_header) return false;
  header.seq_level_idx_0 = obu.DecodeUInt(5);
  const uint8_t frame_width_bits = 1 + obu.DecodeUInt(4);
  const uint8_t frame_height_bits = 1 + obu.DecodeUInt(4);
  /*max_frame_width_minus_1=*/obu.DecodeUInt(frame_width_bits);
  /*max_frame_height_minus_1=*/obu.DecodeUInt(frame_height_bits);
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

//------------------------------------------------------------------------------

std::string ObuUnitsToStr(const uint8_t data[], size_t data_size) {
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

  std::stringstream ss;

  size_t position = 0;
  while (position < data_size) {
    uint8_t obu_type;
    size_t obu_header_size, obu_payload_size;
    if (!UnwrapOBU(data + position, data_size - position, obu_type,
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
