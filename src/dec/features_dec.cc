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
// Functions related to feature decoding.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cassert>
#include <cstddef>

#include "src/common/color_precision.h"
#include "src/common/constants.h"
#include "src/common/header_enc_dec.h"
#include "src/dec/preview/preview_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/ans_utils.h"
#include "src/utils/data_source.h"
#include "src/utils/orientation.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"

namespace WP2 {

//------------------------------------------------------------------------------
// ReadChunk() marks bytes as read but does not discard them.

// Extracts a chunk. 'chunk_data' is not owning the returned data and must be
// used or copied before reading from the 'data_source' again.
static WP2Status ReadChunk(DataSource* const data_source,
                           DataView* const chunk_data) {
  assert(chunk_data != nullptr);
  size_t chunk_size = 0;
  WP2_CHECK_STATUS(TryReadVarInt(data_source, 1, kChunkSizeMax, &chunk_size));
  WP2_CHECK_OK(data_source->TryReadNext(chunk_size, &chunk_data->bytes),
               WP2_STATUS_NOT_ENOUGH_DATA);
  chunk_data->size = chunk_size;
  return WP2_STATUS_OK;
}

// Reads the chunk size and set that many following bytes as read.
static WP2Status SkipChunk(DataSource* const data_source) {
  size_t chunk_size = 0;
  WP2_CHECK_STATUS(TryReadVarInt(data_source, 1, kChunkSizeMax, &chunk_size));
  data_source->MarkNumBytesAsRead(chunk_size);
  return WP2_STATUS_OK;
}

// Copies the bytes into 'chunk_data' unless it is null or not empty.
static WP2Status CopyChunk(DataSource* const data_source,
                           Data* const chunk_data) {
  if (chunk_data != nullptr && chunk_data->IsEmpty()) {
    DataView view;
    WP2_CHECK_STATUS(ReadChunk(data_source, &view));
    WP2_CHECK_STATUS(chunk_data->CopyFrom(view.bytes, view.size));
  } else {
    WP2_CHECK_STATUS(SkipChunk(data_source));
  }
  return WP2_STATUS_OK;
}

WP2Status DecodeTileChunkSize(const GlobalParams& params,
                              const Rectangle& tile_rect,
                              DataSource* const data_source,
                              size_t* const tile_chunk_size) {
  const uint64_t max_num_bytes = GetTileMaxNumBytes(params, tile_rect);
  WP2_CHECK_STATUS(
      TryReadVarInt(data_source, 1, max_num_bytes, tile_chunk_size));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// BitstreamFeatures definitions.

static uint32_t Load24(const uint8_t buf[]) {
  return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) | ((uint32_t)buf[2] << 16);
}

WP2Status BitstreamFeatures::InitInternal(int version, const uint8_t* data,
                                          size_t data_size) {
  WP2_CHECK_OK(!WP2_ABI_IS_INCOMPATIBLE(version, WP2_ABI_VERSION),
               WP2_STATUS_VERSION_MISMATCH);
  WP2_CHECK_OK(data != nullptr, WP2_STATUS_NULL_PARAMETER);
  WP2_CHECK_OK(data_size >= kHeaderMinSize, WP2_STATUS_NOT_ENOUGH_DATA);
  WP2_CHECK_OK(Load24(data) == kSignature, WP2_STATUS_BITSTREAM_ERROR);

  ExternalDataSource data_source(data, data_size);
  data_source.MarkNumBytesAsRead(3);  // For kSignature
  HeaderDec dec(&data_source, "image_features");
  raw_width = 1 + dec.ReadBits(kImageDimNumBits, "width_m1");
  raw_height = 1 + dec.ReadBits(kImageDimNumBits, "height_m1");

  orientation = (Orientation)dec.ReadBits(2, "orientation");
  width = RotateWidth(orientation, raw_width, raw_height);
  height = RotateHeight(orientation, raw_width, raw_height);
  is_opaque = !dec.ReadBits(1, "has_alpha");
  is_animation = !!dec.ReadBits(1, "is_animation");

  preview_color = ToRGB12b(dec.ReadBits(12, "preview_color"));
  quality_hint = dec.ReadBits(kQualityHintNumBits, "quality_hint");

  has_preview = !!dec.ReadBits(1, "has_preview");
  has_icc = !!dec.ReadBits(1, "has_icc");
  has_trailing_data = !!dec.ReadBits(1, "has_trailing_data");
  transfer_function =
      dec.ReadBits(1, "default_transfer_function") ? WP2_TF_ITU_R_BT2020_10BIT
                                                   : WP2_TF_UNSPECIFIED;

  const auto tile_shape = (TileShape)dec.ReadBits(kTileShapeBits, "tile_shape");
  tile_width = TileWidth(tile_shape, raw_width);
  tile_height = TileHeight(tile_shape, /*image_width=*/raw_width);
  WP2_CHECK_OK(dec.ReadBits(2, "filler") == 0x2u, WP2_STATUS_BITSTREAM_ERROR);

  if (is_animation) {
    loop_forever = !!dec.ReadBits(1, "loop");
    const bool custom_background = !!dec.ReadBits(1, "background");
    if (custom_background) {
      background_color.a =
          (uint8_t)(is_opaque ? kAlphaMax : dec.ReadBits(8, "background"));
      background_color.r = (uint16_t)dec.ReadBits(10, "background");
      background_color.g = (uint16_t)dec.ReadBits(10, "background");
      background_color.b = (uint16_t)dec.ReadBits(10, "background");
      WP2_CHECK_OK(CheckPremultiplied(background_color),
                   WP2_STATUS_BITSTREAM_ERROR);
    } else {
      background_color = kDefaultBackgroundColor;
      WP2_CHECK_OK(dec.ReadBits(6, "filler") == 0x2u,
                   WP2_STATUS_BITSTREAM_ERROR);
    }
  } else {
    loop_forever = true;
    background_color = kDefaultBackgroundColor;
  }
  if (transfer_function == WP2_TF_UNSPECIFIED) {
    transfer_function =
        (TransferFunction)(1 + dec.ReadBits(4, "transfer_function"));
    WP2_CHECK_OK(dec.ReadBits(4, "filler") == 0x0, WP2_STATUS_BITSTREAM_ERROR);
  }
  WP2_CHECK_STATUS(dec.GetStatus());

  header_size = data_source.GetNumReadBytes();
  assert(header_size >= kHeaderMinSize && header_size <= kHeaderMaxSize);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status DecodeHeader(DataSource* const data_source,
                       BitstreamFeatures* const features) {
  const uint8_t* data;
  // Try reading maybe more than necessary but do fewer TryGetNext() calls.
  // TODO(maryla): this could fail for a tiny image whose total size is less
  // than kHeaderMaxSize.
  WP2_CHECK_OK(data_source->TryGetNext(kHeaderMaxSize, &data),
               WP2_STATUS_NOT_ENOUGH_DATA);
  WP2_CHECK_STATUS(features->Read(data, kHeaderMaxSize));
  data_source->MarkNumBytesAsRead(features->header_size);
  return WP2_STATUS_OK;
}

WP2Status DecodeICC(DataSource* const data_source,
                    const BitstreamFeatures& features, Data* const iccp) {
  if (features.has_icc) {
    WP2_CHECK_STATUS(CopyChunk(data_source, iccp));
  }
  return WP2_STATUS_OK;
}

WP2Status SkipICC(DataSource* const data_source,
                  const BitstreamFeatures& features) {
  if (features.has_icc) WP2_CHECK_STATUS(SkipChunk(data_source));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status DecodeGLBL(DataSource* const data_source, const DecoderConfig& config,
                     const BitstreamFeatures& features,
                     GlobalParams* const gparams) {
  const size_t num_read_bytes = data_source->GetNumReadBytes();
  DataView chunk_glbl;
  const WP2Status status =
      ReadChunk(data_source, (gparams != nullptr) ? &chunk_glbl : nullptr);
  if (status != WP2_STATUS_OK) {
    // The data consumption must be atomic.
    data_source->UnmarkAllReadBytes();
    data_source->MarkNumBytesAsRead(num_read_bytes);
  }
  WP2_CHECK_STATUS(status);

  // if gparams is null, skip over the content in the bitstream
  if (gparams == nullptr) return WP2_STATUS_OK;

  ExternalDataSource src(chunk_glbl.bytes, chunk_glbl.size);
  ANSDec dec(&src);
  gparams->Reset();
  WP2_CHECK_STATUS(
      gparams->Read(features.quality_hint, !features.is_opaque, &dec));
#if defined(WP2_BITTRACE)
  if (config.info != nullptr) {
    for (const auto& it : dec.GetBitTraces()) {
      config.info->bit_traces[it.first].bits += it.second.bits;
      config.info->bit_traces[it.first].num_occurrences +=
          it.second.num_occurrences;
      config.info->bit_traces[it.first].type = it.second.type;
      for (const auto& p : it.second.histo) {
        config.info->bit_traces[it.first].histo[p.first] += p.second;
      }
    }
  }
#endif
  return gparams->ApplyDecoderConfig(config);
}

//------------------------------------------------------------------------------

WP2Status DecodePreview(DataSource* const data_source,
                        const BitstreamFeatures& features,
                        ArgbBuffer* const output_buffer) {
  if (features.has_preview) {
    if (output_buffer != nullptr) {
      DataView chunk;
      WP2_CHECK_STATUS(ReadChunk(data_source, &chunk));
      WP2_CHECK_STATUS(DecodePreview(chunk.bytes, chunk.size, output_buffer));
    } else {
      WP2_CHECK_STATUS(SkipChunk(data_source));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status SkipPreview(DataSource* const data_source,
                      const BitstreamFeatures& features) {
  if (features.has_preview) WP2_CHECK_STATUS(SkipChunk(data_source));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

static WP2Status HasMetadata(DataSource* const data_source,
                             const BitstreamFeatures& features,
                             bool* const has_xmp, bool* const has_exif) {
  if (features.has_trailing_data) {
    const uint8_t* data;
    WP2_CHECK_OK(data_source->TryReadNext(3, &data),
                 WP2_STATUS_NOT_ENOUGH_DATA);
    const uint32_t tag = Load24(data);
    WP2_CHECK_OK((tag & kTagMask) == kTagMask, WP2_STATUS_BITSTREAM_ERROR);
    const uint32_t content_bits = (tag & ~kTagMask) >> 16;
    *has_xmp = (content_bits & 1);
    *has_exif = (content_bits & 2);
  } else {
    *has_xmp = false;
    *has_exif = false;
  }
  return WP2_STATUS_OK;
}

WP2Status DecodeMetadata(DataSource* const data_source,
                         const BitstreamFeatures& features,
                         Data* const exif, Data* const xmp) {
  bool has_xmp, has_exif;
  WP2_CHECK_STATUS(HasMetadata(data_source, features, &has_xmp, &has_exif));
  if (has_xmp) {
    WP2_CHECK_STATUS(CopyChunk(data_source, xmp));
  }
  if (has_exif) {
    WP2_CHECK_STATUS(CopyChunk(data_source, exif));
  }
  return WP2_STATUS_OK;
}

WP2Status SkipMetadata(DataSource* const data_source,
                       const BitstreamFeatures& features) {
  bool has_xmp, has_exif;
  WP2_CHECK_STATUS(HasMetadata(data_source, features, &has_xmp, &has_exif));
  if (has_xmp) WP2_CHECK_STATUS(SkipChunk(data_source));
  if (has_exif) WP2_CHECK_STATUS(SkipChunk(data_source));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}   // namespace WP2
