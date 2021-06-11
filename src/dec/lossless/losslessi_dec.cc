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
// main entry for the decoder
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/dec/lossless/losslessi_dec.h"

#include <algorithm>
#include <string>

#include "src/common/color_precision.h"
#include "src/common/constants.h"
#include "src/common/lossless/color_cache.h"
#include "src/common/progress_watcher.h"
#include "src/common/symbols.h"
#include "src/common/vdebug.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/lossless/decl_dsp.h"
#include "src/dsp/math.h"
#include "src/utils/ans_utils.h"
#include "src/utils/plane.h"
#include "src/utils/vector.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

// -----------------------------------------------------------------------------

static const uint8_t kCodeToPlane[kCodeToPlaneCodes] = {
  0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
  0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
  0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
  0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
  0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
  0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
  0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
  0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
  0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
  0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
  0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
  0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70
};

//------------------------------------------------------------------------------

// Reads a prefix coded number where the prefix is stored as symbol 'sym' in
// cluster 'cluster'.
static inline WP2Status ReadLZ77PrefixCode(
    uint32_t sym, uint32_t cluster, uint32_t min, uint32_t max, WP2_OPT_LABEL,
    WP2::SymbolReader* const sr, WP2::ANSDec* const dec, uint32_t* const result,
    float* const cost) {
  const WP2::PrefixCode prefix_code_max(max - min, kLZ77PrefixCodeSize);

  // Read information about the prefix code representation.
  int32_t prefix;
  WP2_CHECK_STATUS(sr->ReadWithMax(sym, cluster, prefix_code_max.prefix, label,
                                   &prefix, cost));
  const uint32_t extra_bits_num =
      WP2::PrefixCode::NumExtraBits(prefix, kLZ77PrefixCodeSize);

  // Figure out the rest of the prefix code representation.
  uint32_t extra_bits_value = 0;
  const std::string label_bits = WP2SPrint("%s_bits", label);
  if (extra_bits_num == 0) {
    // keep extra_bits_value = 0
  } else if (prefix == (int32_t)prefix_code_max.prefix) {
    // In case we are in the upper part of the prefix code representation.
    const uint32_t new_range =
        max - min + 1 - WP2::PrefixCode::Merge(prefix, kLZ77PrefixCodeSize, 0);
    if (new_range <= WP2::kANSMaxRange) {
      extra_bits_value = dec->ReadRValue(new_range, label_bits.c_str());
      if (cost != nullptr) *cost += WP2Log2(new_range);
    } else {
      // Go recursively.
      WP2_CHECK_STATUS(ReadLZ77PrefixCode(sym, cluster, /*min=*/0,
                                          /*max=*/new_range - 1, label, sr, dec,
                                          &extra_bits_value, cost));
    }
  } else {
    if (extra_bits_num <= WP2::kANSMaxUniformBits) {
      extra_bits_value = dec->ReadUValue(extra_bits_num, label_bits.c_str());
    } else {
      extra_bits_value =
          dec->ReadUValue(WP2::kANSMaxUniformBits, label_bits.c_str());
      extra_bits_value |=
          dec->ReadUValue(extra_bits_num - WP2::kANSMaxUniformBits,
                          label_bits.c_str())
          << WP2::kANSMaxUniformBits;
    }
    if (cost != nullptr) *cost += extra_bits_num;
  }
  *result = min + WP2::PrefixCode::Merge(prefix, kLZ77PrefixCodeSize,
                                         extra_bits_value);
  return WP2_STATUS_OK;
}

static inline uint32_t PlaneCodeToDistance(uint32_t width,
                                           uint32_t plane_code) {
  if (plane_code >= kCodeToPlaneCodes) {
    return plane_code + 1 - kCodeToPlaneCodes;
  } else {
    const int dist_code = kCodeToPlane[plane_code];
    const int yoffset = dist_code >> 4;
    const int xoffset = 8 - (dist_code & 0xf);
    const uint32_t dist = yoffset * width + xoffset;
    return (dist >= 1) ? dist : 1;  // dist<1 can happen if width is very small
  }
}

//------------------------------------------------------------------------------

// Reads an image encode in a raw manner.
static void ReadRawImage(uint32_t width, uint32_t height,
                         const LosslessSymbolsInfo& symbols_info,
                         WP2::ANSDec* dec, int16_t* const argb) {
  // Cache the extrema.
  int16_t mins[4], maxs[4];
  for (Symbol sym : {kSymbolA, kSymbolR, kSymbolG, kSymbolB}) {
    if (symbols_info.IsUnused(sym)) continue;
    const uint32_t c = (int)sym - (int)kSymbolA;
    mins[c] = symbols_info.Min(sym, /*cluster=*/0);
    maxs[c] = symbols_info.Max(sym, /*cluster=*/0);
  }
  // Read the raw pixel values.
  for (uint32_t y = 0; y < height; ++y) {
    int16_t* const row = &argb[4 * width * y];
    for (uint32_t x = 0; x < width; ++x) {
      row[4 * x + 0] = symbols_info.IsUnused(kSymbolA)
                           ? WP2::kAlphaMax
                           : dec->ReadRange(mins[0], maxs[0],
                                            kSymbolNames[(int)kSymbolA + 0]);
      for (uint32_t c = 1; c < 4; ++c) {
        row[4 * x + c] =
            symbols_info.IsUnused((int)kSymbolA + c)
                ? 0
                : dec->ReadRange(WP2::DivBy255(mins[c] * row[4 * x + 0]),
                                 WP2::DivBy255(maxs[c] * row[4 * x + 0]),
                                 kSymbolNames[(int)kSymbolA + c]);
      }
    }
  }
}

//------------------------------------------------------------------------------

// Reads packed symbol depending on GREEN channel
WP2Status Decoder::ReadANSStats(uint32_t width, uint32_t height,
                                const LosslessSymbolsInfo& symbols_info) {
  WP2::ANSDebugPrefix prefix(dec_, "symbols");

  WP2::SymbolReader* const sr = &hdr_.sr_;
  WP2_CHECK_STATUS(sr->Init(symbols_info, dec_));
  WP2_CHECK_STATUS(sr->Allocate());
  const uint32_t num_clusters = symbols_info.NumClusters(0);
  for (uint32_t cluster = 0; cluster < num_clusters; ++cluster) {
    // Deal with the first symbol.
    WP2_CHECK_STATUS(sr->ReadHeader(kSymbolType, cluster, width * height,
                                    kSymbolNames[kSymbolType]));
    bool is_maybe_used[kSymbolTypeNum];
    sr->GetPotentialUsage(kSymbolType, cluster, is_maybe_used, kSymbolTypeNum);
    for (int s = 1; s < kSymbolNum; ++s) {
      if (symbols_info.IsUnused(s)) continue;
      const Symbol sym = (Symbol)(s);
      if (!IsSymbolUsed(sym, is_maybe_used)) continue;
      WP2_CHECK_STATUS(
          sr->ReadHeader(sym, cluster, width * height, kSymbolNames[s]));
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Emit rows without any scaling.
template <typename T>
uint32_t EmitRows(const int16_t* row_in, uint32_t mb_w, uint32_t mb_h,
                  bool has_alpha, bool premultiply, T* const out,
                  uint32_t out_stride) {
  int lines = mb_h;
  T* row_out = out;
  while (lines-- > 0) {
    // TODO(vrabaud) support multiple output color spaces.
    for (uint32_t i = 0; i < 4 * mb_w; i += 4, row_in += 4) {
      const uint8_t alpha = (uint8_t)row_in[0];
      if (!has_alpha) assert(alpha == WP2::kAlphaMax);
      row_out[i] = alpha;
      if (alpha > 0) {
        for (int j = 1; j <= 3; ++j) {
          row_out[i + j] = premultiply
                               ? static_cast<T>(WP2::DivBy255(
                                     static_cast<uint32_t>(row_in[j]) * alpha))
                               : static_cast<T>(row_in[j]);
        }
      } else {
        // To optimize encoding, the encoder can store arbitrary rgb values when
        // a = 0 (fully transparent pixel, see GetResidual in predictor_enc.cc).
        // To make sure we return proper pre-multiplied Argb, we sanitize these
        // values by setting rgb to 0.
        for (int j = 1; j <= 3; ++j) row_out[i + j] = 0;
      }
    }
    row_out = (T*)(((uint8_t*)row_out) + out_stride);
  }
  return mb_h;  // Num rows out == num rows in.
}

//------------------------------------------------------------------------------

static inline int GetMetaIndex(const WP2::Vector_s16& image, uint32_t width,
                               int bits, int x, int y) {
  if (bits == 0 || image.empty()) return 0;
  return image[4 * (width * (y >> bits) + (x >> bits)) + 2];
}

//------------------------------------------------------------------------------
// Main loop, with custom row-processing function

WP2Status Decoder::ApplyInverseTransforms(uint32_t num_rows,
                                          const int16_t* rows) {
  uint32_t num_transforms = 0;
  while (num_transforms < kPossibleTransformCombinationSize &&
         transforms_[num_transforms].type_ != TransformType::kNum) {
    ++num_transforms;
  }
  const uint32_t cache_pixs = tile_->rect.width * num_rows;

  if (num_transforms == 0) {
    // No transform called, hence just copy.
    std::copy(rows, rows + 4 * cache_pixs, argb_cache_);
    return WP2_STATUS_OK;
  }

  const uint32_t start_row = last_row_;
  const uint32_t end_row = start_row + num_rows;

  int16_t* const rows_out = (int16_t*)argb_cache_;
  // Inverse transforms.
  for (int ind = num_transforms - 1; ind >= 0; --ind) {
    const Transform& transform = transforms_[ind];
    if (transform.type_ == TransformType::kColorIndexing ||
        transform.type_ == TransformType::kGroup4) {
      // Check that all color map indices are valid here rather than in dsp.
      // TODO(yguyon): Remove pessimization.
      const int16_t color_map_size = (int16_t)transform.data_.size() / 4;
      const int16_t* const r_end = rows + 4 * num_rows * transform.width_pic_;
      for (const int16_t* r = rows; r < r_end; r += 4) {
        WP2_CHECK_OK(r[2] >= 0 && r[2] < color_map_size,
                     WP2_STATUS_BITSTREAM_ERROR);
      }
    }
    int16_t minima_range[4], maxima_range[4];
    if (transform.type_ == TransformType::kPredictor ||
        transform.type_ == TransformType::kPredictorWSub) {
      // Define the minimum/maximum values before applying the predictors.
      TransformType transforms_type[(uint32_t)TransformType::kNum];
      for (int i = 0; i < ind; ++i) transforms_type[i] = transforms_[i].type_;
      GetARGBRanges(transforms_type, hdr_.symbols_info_.SampleFormat(),
                    num_colors_, minima_range, maxima_range, ind);
    }
    int16_t* predicted = nullptr;
#ifndef WP2_REDUCE_BINARY_SIZE
    if (((transform.type_ == TransformType::kPredictor ||
          transform.type_ == TransformType::kPredictorWSub) &&
         VDMatch(config_, "lossless/prediction/raw")) ||
        (transform.type_ == TransformType::kCrossColor &&
         VDMatch(config_, "lossless/cross-color/raw"))) {
      predicted = (int16_t*)predicted_.GetRow16(start_row);
    }
#endif
    WP2_CHECK_REDUCED_STATUS(RegisterTransformedRowForVDebug(
        ind, num_rows, /*num_bits=*/8, rows, config_.info));
    InverseTransform(&transform, start_row, end_row, minima_range, maxima_range,
                     rows, gparams_->has_alpha_, rows_out, predicted);
    WP2_CHECK_REDUCED_STATUS(RegisterPredictedRowForVDebug(
        ind, num_rows, /*num_bits=*/8, predicted, config_.info));
    rows = rows_out;
  }
  // In case the selected transformed image is the first one and the reference
  // image is the original one.
  WP2_CHECK_REDUCED_STATUS(RegisterTransformedRowForVDebug(
      /*index=*/-1, num_rows, /*num_bits=*/8, rows, config_.info));
  return WP2_STATUS_OK;
}

WP2Status Decoder::ProcessRows(uint32_t row) {
  const int16_t* const rows =
      pixels_.data() + 4 * tile_->rect.width * last_row_;

  if (row > last_row_) {    // Emit output.
    WP2::ArgbBuffer* const buf = &tile_->rgb_output;
    const uint32_t num_rows = row - last_row_;
    // We can't process more than kNumARGBCacheRows at a time (that's the size
    // of argb_cache_), but we currently don't need more than that.
    assert(num_rows <= kNumARGBCacheRows);
    assert(!buf->IsEmpty());
    int16_t* const rows_data = argb_cache_;

    WP2_CHECK_STATUS(ApplyInverseTransforms(num_rows, rows));

    uint32_t num_rows_out;
    if (WP2Formatbpc(buf->format()) <= 8) {
      num_rows_out =
          EmitRows(rows_data, buf->width(), num_rows, gparams_->has_alpha_,
                   premultiply_, buf->GetRow8(last_out_row_), buf->stride());
    } else {
      num_rows_out =
          EmitRows(rows_data, buf->width(), num_rows, gparams_->has_alpha_,
                   premultiply_, buf->GetRow16(last_out_row_), buf->stride());
    }
    // Update 'last_out_row_'.
    last_out_row_ += num_rows_out;
    tile_->num_decoded_rows += num_rows_out;
    WP2_CHECK_STATUS(tile_->row_progress.AdvanceBy(num_rows_out));
  }

  // Update 'last_row_'.
  last_row_ = row;
  assert(last_row_ <= tile_->rect.height);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Helper functions for fast pattern copy.

static inline void CopyBlock(int16_t* dst, uint32_t dist, uint32_t length) {
  uint32_t length_written = 0;
  const int16_t* const dst_ini = dst - 4 * dist;
  int16_t* dst_end = dst;
  while (length_written < length) {
    // We cannot write more than what is left, and we cannot write further than
    // dst_end.
    const uint32_t length_to_write =
        std::min(length - length_written, (uint32_t)((dst_end - dst_ini) / 4));
    std::copy(dst_ini, dst_ini + 4 * length_to_write, dst_end);
    dst_end += 4 * length_to_write;
    length_written += length_to_write;
  }
}

//------------------------------------------------------------------------------

WP2Status Decoder::DecodeAndProcess(uint32_t width, uint32_t last_row,
                                    bool do_process,
                                    WP2::Vector_s16* const data_vec,
                                    WP2::Planef* const bits_per_pixel,
                                    WP2::DecoderInfo* const decoder_info,
                                    int16_t** src_out) {
  int16_t* const data = data_vec->data();
  uint32_t row = last_pixel_ / width;
  uint32_t col = last_pixel_ % width;
  int16_t* src = data + 4 * last_pixel_;
  const int16_t* last_cached = src;
  const int16_t* const src_end = &data_vec->back() + 1;  // End of data
  int16_t* const src_last =
      data + 4 * width * last_row;  // Last pixel to decode
  ColorCache* const color_cache =
      do_process ? hdr_.color_cache_.get() : nullptr;
  const uint32_t mask = hdr_.histogram_mask_;
  WP2::SymbolReader* const sr = &hdr_.sr_;

  uint32_t cluster =
      (src < src_last)
          ? GetMetaIndex(hdr_.histogram_image_, hdr_.histogram_xsize_,
                         hdr_.histogram_subsample_bits_, col, row)
          : 0;
  assert(last_row_ < last_row);
  assert(src_last <= src_end);

  const bool has_alpha = LosslessSymbolsInfo::HasAlpha(sr->symbols_info());
  const bool is_label_image =
      LosslessSymbolsInfo::IsLabelImage(sr->symbols_info());

  const bool get_cost = (bits_per_pixel != nullptr || decoder_info != nullptr);
  uint32_t i_pixel = 0;
  while (src < src_last) {
    float cost = 0.f;
#if defined(WP2_BITTRACE)
    float* const cost_ptr = get_cost ? &cost : nullptr;
#else
    float* const cost_ptr = nullptr;
#endif
    uint32_t distance = 0;
    uint32_t length = 0;  // Silence -Werror=maybe-uninitialized
    const auto type =
        (SymbolType) sr->Read(kSymbolType, cluster, "lossless_mode", cost_ptr);
    WP2_CHECK_STATUS(dec_->GetStatus());
    if (type == kSymbolTypeLiteral) {
      WP2::ANSDebugPrefix prefix(dec_, "literal");
      length = 1;
      if (has_alpha) {
        src[0] = sr->Read(kSymbolA, cluster, "A", cost_ptr);
      } else {
        // For transform images/entropy image, alpha is not used: any value
        // works. The default value only matters for real ARGB samples.
        src[0] = WP2::kAlphaMax;
      }
      if (!gparams_->has_alpha_) assert(src[0] == WP2::kAlphaMax);
      if (is_label_image) {
        src[1] = src[3] = 0;
      } else {
        src[1] = sr->Read(kSymbolR, cluster, "R", cost_ptr);
        src[3] = sr->Read(kSymbolB, cluster, "B", cost_ptr);
      }
      src[2] = sr->Read(kSymbolG, cluster, "G", cost_ptr);
      WP2_CHECK_STATUS(dec_->GetStatus());
    } else if (type == kSymbolTypeCopy) {
      WP2::ANSDebugPrefix prefix(dec_, "copy");

      const uint32_t pixel_ind = (src - data) / 4;
      const uint32_t max_plane_code = pixel_ind + kCodeToPlaneCodes - 1;
      uint32_t dist_code = 0;  // Silence -Werror=maybe-uninitialized
      WP2_CHECK_STATUS(ReadLZ77PrefixCode(kSymbolDist, cluster, /*min=*/0,
                                          max_plane_code, "dist", sr, dec_,
                                          &dist_code, cost_ptr));
      distance = PlaneCodeToDistance(width, dist_code);
      WP2_CHECK_OK(pixel_ind >= distance, WP2_STATUS_BITSTREAM_ERROR);

      length = 0;
      do {
        // Read in chunks of size kLZ77LengthEscape.
        const uint32_t min_length = (length == 0) ? kLZ77LengthMin : 0u;
        const uint32_t max_length = std::min(
            (uint32_t)((src_end - (src + 4 * length)) / 4), kLZ77LengthEscape);
        uint32_t length_tmp = 0;
        WP2_CHECK_STATUS(ReadLZ77PrefixCode(kSymbolLen, cluster, min_length,
                                            max_length, "len", sr, dec_,
                                            &length_tmp, cost_ptr));
        length += length_tmp;
        if (length_tmp != kLZ77LengthEscape) break;
      } while (true);

      WP2_CHECK_STATUS(dec_->GetStatus());
      CopyBlock(src, distance, length);
      if (!gparams_->has_alpha_) assert(src[0] == WP2::kAlphaMax);
    } else if (type == kSymbolTypeCacheIdx) {
      length = 1;
      assert(color_cache != nullptr);
      while (last_cached < src) {
        color_cache->Insert(last_cached, /*index_ptr=*/nullptr);
        last_cached += 4;
      }
      int32_t cache_index;
      WP2_CHECK_STATUS(sr->ReadWithMax(kSymbolCache, cluster,
                                       color_cache->IndexRange() - 1,
                                       "cache_idx", &cache_index, cost_ptr));
      color_cache->Lookup(cache_index, src);
      if (!gparams_->has_alpha_) assert(src[0] == WP2::kAlphaMax);
    } else {                                            // Not reached
      WP2_CHECK_OK(false, WP2_STATUS_BITSTREAM_ERROR);  // For error reporting.
    }
    if (get_cost) {
      WP2_CHECK_REDUCED_STATUS(RegisterSymbolForVDebug(
          type, /*is_group4=*/false, last_pixel_ + i_pixel, distance, length,
          cost, decoder_info));
      if (bits_per_pixel != nullptr) {
        const float bits = cost / length;
        for (uint32_t j = 0; j < length; ++j) {
          const uint32_t pixel = last_pixel_ + i_pixel + j;
          bits_per_pixel->Row(pixel / width)[pixel % width] = bits;
        }
      }
    }
    i_pixel += length;
    src += 4 * length;
    col += length;
    while (col >= width) {
      col -= width;
      ++row;
      if (do_process) {
        if (row <= last_row && (row % kNumARGBCacheRows == 0)) {
          WP2_CHECK_STATUS(ProcessRows(row));
        }
      }
    }

    // Only update when changing tile. Note we could use this test:
    // if "((((prev_col ^ col) | prev_row ^ row)) > mask)" -> tile changed
    // but that's actually slower and needs storing the previous col/row.
    if ((type == kSymbolTypeCopy || (col & mask) == 0) && src < src_end) {
      cluster = GetMetaIndex(hdr_.histogram_image_, hdr_.histogram_xsize_,
                             hdr_.histogram_subsample_bits_, col, row);
    }
  }

  // Finish filling the color cache.
  if (color_cache != nullptr) {
    while (last_cached < src) {
      color_cache->Insert(last_cached, /*index_ptr=*/nullptr);
      last_cached += 4;
    }
  }

  WP2_CHECK_STATUS(dec_->GetStatus());

  if (src_out != nullptr) *src_out = src;

  return WP2_STATUS_OK;
}

WP2Status Decoder::DecodeHelperImage(uint32_t width, uint32_t height,
                                     const LosslessSymbolsInfo& symbols_info,
                                     WP2::Vector_s16* const data_vec) {
  WP2_CHECK_ALLOC_OK(data_vec->resize(4 * (uint64_t)width * height));
  if (width * height == 1) {
    ReadRawImage(width, height, symbols_info, dec_, data_vec->data());
  } else {
    WP2_CHECK_STATUS(ReadANSStats(width, height, symbols_info));

    WP2_CHECK_STATUS(DecodeAndProcess(width, /*last_row=*/height,
                                      /*do_process=*/false, data_vec));
  }
  return WP2_STATUS_OK;
}

WP2Status Decoder::DecodeImageData(uint32_t width, uint32_t height,
                                   uint32_t last_row,
                                   WP2::Planef* const bits_per_pixel) {
  WP2_CHECK_ALLOC_OK(pixels_.resize(4 * (uint64_t)width * height));

  int16_t* src;
  if (use_group4_) {
    WP2_CHECK_STATUS(DecodeGroup4(width, last_row, bits_per_pixel, &src));
  } else {
    WP2_CHECK_STATUS(DecodeAndProcess(width, last_row, /*do_process=*/true,
                                      &pixels_, bits_per_pixel, config_.info,
                                      &src));
  }

  // Process the remaining rows corresponding to last row-block.
  const uint32_t row = (src - pixels_.data()) / 4 / width;
  WP2_CHECK_STATUS(ProcessRows(row > last_row ? last_row : row));

  WP2_CHECK_STATUS(dec_->GetStatus());

  // When decoding the main image, update the last pixel, otherwise reset.
  int16_t* const data = pixels_.data();
  last_pixel_ = (uint32_t)(src - data) / 4;  // end-of-scan marker
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Transform

static WP2Status ReadPaletteVerbatim(uint32_t num_colors,
                                     WP2SampleFormat format, bool has_alpha,
                                     bool is_grayscale, WP2::ANSDec* const dec,
                                     Transform* const transform) {
  WP2_CHECK_ALLOC_OK(transform->data_.resize(4 * num_colors));
  uint32_t channel_max[4];
  for (uint32_t c = 0; c < 4; ++c) {
    channel_max[c] = WP2::FormatMax(format, /*channel=*/c);
  }
  for (uint32_t i = 0; i < num_colors; ++i) {
    if (has_alpha) {
      transform->data_[4 * i + 0] =
          dec->ReadRValue(channel_max[0] + 1, "color_verbatim");
      for (uint32_t c = 1; c < 4; ++c) {
        const uint32_t max = WP2::DivRound(
            (uint32_t)(channel_max[c] * transform->data_[4 * i + 0]),
            channel_max[0]);
        transform->data_[4 * i + c] =
            (is_grayscale && c > 1)
                ? transform->data_[4 * i + 1]
                : dec->ReadRValue(max + 1, "color_verbatim");
      }
    } else {
      transform->data_[4 * i + 0] = channel_max[0];
      for (uint32_t c = 1; c < 4; ++c) {
        transform->data_[4 * i + c] =
            (is_grayscale && c > 1)
                ? transform->data_[4 * i + 1]
                : dec->ReadRValue(channel_max[c] + 1, "color_verbatim");
      }
    }
  }
  return dec->GetStatus();
}

WP2Status Decoder::ReadPalette(Transform* const transform) {
  transform->bits_ = 0;

  dec_->AddDebugPrefix("palette");
  // There are at least 2 colors in the palette.
  num_colors_ = ReadPrefixCode(
      /*min=*/2, MaxPaletteSize(transform->width_pic_ * transform->height_pic_),
      0, dec_, "num_colors");

  const bool is_grayscale = dec_->ReadBool("is_grayscale");
  const uint32_t num_channels = (is_grayscale ? 2 : 4);

  if (num_colors_ <= kSmallPaletteLimit) {
    WP2_CHECK_STATUS(ReadPaletteVerbatim(
        num_colors_, hdr_.symbols_info_.SampleFormat(), gparams_->has_alpha_,
        is_grayscale, dec_, transform));
    dec_->PopDebugPrefix();
    return WP2_STATUS_OK;
  }

  // Fill the sign array.
  WP2::Vector_u8 is_positive[4];
  for (WP2::Vector_u8& vec : is_positive) {
    WP2_CHECK_ALLOC_OK(vec.resize(num_colors_ - 1));
  }
  for (uint32_t c = 0; c < num_channels; ++c) {
    if (c == 0 && !gparams_->has_alpha_) continue;
    const bool is_monotonous = dec_->ReadBool("is_monotonous");
    const bool use_opposite_sign = dec_->ReadBool("use_opposite_sign");
    if (is_monotonous) {
      std::fill(is_positive[c].begin(), is_positive[c].end(),
                !use_opposite_sign);
    } else {
      dec_->AddDebugPrefix("transform_index_signs");
      WP2::ReadVector(dec_, 1, is_positive[c]);
      dec_->PopDebugPrefix();
      if (use_opposite_sign) {
        for (auto& is_pos : is_positive[c]) is_pos = !is_pos;
      }
    }
  }

  if (dec_->ReadBool("use_image")) {
    // Read the header of the palette image.
    LosslessSymbolsInfo symbols_info;
    symbols_info.Init(num_colors_, gparams_->has_alpha_,
                      hdr_.symbols_info_.SampleFormat());

    // Read the palette image.
    WP2_CHECK_STATUS(
        DecodeHelperImage(num_colors_, 1, symbols_info, &transform->data_));
    WP2_CHECK_STATUS(dec_->GetStatus());

    // Convert the diffs to proper colors.
    int16_t* const data = transform->data_.data();
    // data[0] stays unchanged and serves as a basis to adding successive
    // differences.
    if (!gparams_->has_alpha_) assert(data[0] == WP2::kAlphaMax);
    for (uint32_t i = 1; i < num_colors_; ++i) {
      for (uint32_t c = 0; c < num_channels; ++c) {
        if (c == 0 && !gparams_->has_alpha_) {
          assert(data[4 * i + c] == WP2::kAlphaMax);
          continue;
        }
        const int16_t alpha = data[4 * i + 0];
        int16_t channel_prev = data[4 * (i - 1) + c];
        if (c > 0 && channel_prev > alpha) channel_prev = alpha;
        const auto channel_diff = data[4 * i + c];
        if (is_positive[c][i - 1]) {
          data[4 * i + c] = channel_prev + channel_diff;
        } else {
          data[4 * i + c] = channel_prev - channel_diff;
        }
      }
    }
  } else {
    WP2_CHECK_ALLOC_OK(transform->data_.resize(4 * num_colors_));
    std::fill(transform->data_.begin(), transform->data_.end(), 0);
    // Read the channels one by one.
    const uint32_t alpha_max =
        WP2::FormatMax(hdr_.symbols_info_.SampleFormat(), 0);
    for (uint32_t c = 0; c < num_channels; ++c) {
      const uint32_t channel_max =
          WP2::FormatMax(hdr_.symbols_info_.SampleFormat(), c);
      if (c == 0 && !gparams_->has_alpha_) {
        for (uint32_t i = 0; i < num_colors_; ++i) {
          transform->data_[4 * i + 0] = channel_max;
        }
        continue;
      }
      const bool use_vector = dec_->ReadBool("use_vector");
      // Read the first channel.
      const uint32_t max_first = (c == 0) ? channel_max : transform->data_[0];
      int16_t channel = dec_->ReadRValue(max_first + 1, "first_channel");
      transform->data_[c] = channel;

      if (use_vector) {
        WP2::Vector_u16 diffs;
        WP2_CHECK_ALLOC_OK(diffs.resize(num_colors_ - 1));
        dec_->AddDebugPrefix("transform_index_vals");
        WP2::ReadVector(dec_, channel_max, diffs);
        dec_->PopDebugPrefix();
        for (uint32_t i = 1; i < num_colors_; ++i) {
          if (gparams_->has_alpha_) {
            const int16_t alpha = transform->data_[4 * i + 0];
            if (c > 0 && channel > alpha) channel = alpha;
          }
          channel = WP2::Clamp<int32_t>(channel +
              (is_positive[c][i - 1] ? diffs[i - 1] : -diffs[i - 1]),
              0, channel_max);
          transform->data_[4 * i + c] = channel;
        }
      } else {
        for (uint32_t i = 1; i < num_colors_; ++i) {
          const int16_t alpha = transform->data_[4 * i + 0];
          assert(alpha >= 0 && alpha <= (int16_t)alpha_max);
          if (gparams_->has_alpha_ && c > 0 && channel > alpha) channel = alpha;
          if (is_positive[c][i - 1]) {
            const uint32_t max =
                (c == 0) ? channel_max
                         : WP2::DivRound(channel_max * alpha, alpha_max);
            channel += dec_->ReadRValue(max + 1 - channel, "store_diff");
          } else {
            channel -= dec_->ReadRValue(channel + 1, "store_diff");
          }
          transform->data_[4 * i + c] = channel;
        }
      }
    }
  }
  // Duplicate the G/B channels if needed.
  if (is_grayscale) {
    for (uint32_t i = 0; i < num_colors_; ++i) {
      const uint32_t channel = transform->data_[4 * i + 1];
      transform->data_[4 * i + 2] = channel;
      transform->data_[4 * i + 3] = channel;
    }
  }
  dec_->PopDebugPrefix();
  return dec_->GetStatus();
}

WP2Status Decoder::ReadTransform(uint32_t width, uint32_t height,
                                 TransformType type,
                                 Transform* const transform) {
  transform->type_ = type;
  transform->width_pic_ = width;
  transform->height_pic_ = height;
  transform->width_ = 0;
  transform->height_ = 0;

  switch (type) {
    case TransformType::kPredictor:
    case TransformType::kPredictorWSub:
    case TransformType::kCrossColor: {
      if (type == TransformType::kCrossColor) {
        dec_->AddDebugPrefix("cross_color");
      } else {
        assert(type == TransformType::kPredictor ||
               type == TransformType::kPredictorWSub);
        dec_->AddDebugPrefix("predictor");
      }
      transform->bits_ = dec_->ReadRange(kTransformBitsMin, kTransformBitsMax,
                                         "transform_bits");

      // Read the transform header.
      transform->width_ =
          SubSampleSize(transform->width_pic_, transform->bits_);
      transform->height_ =
          SubSampleSize(transform->height_pic_, transform->bits_);
      LosslessSymbolsInfo symbols_info;
      const uint32_t num_pixels = transform->width_ * transform->height_;
      if (type == TransformType::kPredictor ||
          type == TransformType::kPredictorWSub) {
        symbols_info.InitAsPredictorImage(num_pixels);
      } else {
        symbols_info.InitAsCrossColorImage(num_pixels);
      }

      // Read the transform image.
      WP2_CHECK_STATUS(DecodeHelperImage(transform->width_, transform->height_,
                                         symbols_info, &transform->data_));
      if (type == TransformType::kPredictorWSub &&
          dec_->ReadBool("use_sub_modes")) {
        const WP2::ANSDebugPrefix prefix_sub_modes(dec_, "sub_modes");
        WP2_CHECK_STATUS(symbols_info.InitSubMode());

        WP2::SymbolReader sr;
        WP2_CHECK_STATUS(sr.Init(symbols_info, dec_));
        WP2_CHECK_STATUS(sr.Allocate());
        WP2_CHECK_STATUS(sr.ReadHeader(kSymbolSubMode,
                                       transform->width_ * transform->height_,
                                       "sub_mode"));
        // Read the sub-modes.
        for (uint32_t i = 0; i < 4 * transform->height_ * transform->width_;
             i += 4) {
          int16_t& mode = transform->data_[i + 2];
          if (kPredictorType[mode] != PredictorType::Angle) continue;
          // Decide between 45 degrees, 180, or anything else.
          const uint32_t cluster =
              (kPredictorIndex[mode] == 0)                         ? 0
              : (kPredictorIndex[mode] == kNumPredictorsAngle - 1) ? 1
                                                                   : 2;
          const int sub_mode = sr.Read(kSymbolSubMode, cluster, "sub_mode");
          // Store the signed sub-mode as signed so that it is positive and
          // does not create issues with future (potentially signed) shifts.
          // Also make surec 0 maps to 0 for when we do not use sub-modes.
          mode |= ((sub_mode >= 0 ? 2 * sub_mode : -2 * sub_mode - 1)
                   << kNumPredictorsBits);
        }
      }
      WP2_CHECK_STATUS(dec_->GetStatus());

      dec_->PopDebugPrefix();
      break;
    }
    case TransformType::kColorIndexing: {
      WP2_CHECK_STATUS(ReadPalette(transform));
      break;
    }
    case TransformType::kGroup4: {
      WP2_CHECK_STATUS(ReadPalette(transform));
      const uint32_t color_map_size = (uint32_t)transform->data_.size() / 4;
      group4_first_color_ = dec_->ReadRValue(color_map_size, "first_color");
      const bool use_move_to_front =
          (num_colors_ > 3) ? dec_->ReadBool("use_move_to_front") : false;
      WP2_CHECK_STATUS(mtf_.Init(use_move_to_front, num_colors_));

      WP2::SymbolsInfo info;
      InitGroup4(width, color_map_size, &info);

      // Read the headers.
      WP2::SymbolReader* const sr = &hdr_.sr_;
      WP2_CHECK_STATUS(sr->Init(info, dec_));
      WP2_CHECK_STATUS(sr->Allocate());

      {
        WP2::ANSDebugPrefix prefix(dec_, "group4");
        for (uint32_t s = 0; s < (uint32_t)kSymbolG4Num; ++s) {
          if (color_map_size <= 2 &&
              (s == kSymbolG4ColorChange || s == kSymbolG4NewColor)) {
            continue;
          }
          const uint32_t max_nnz = width * height;
          WP2_CHECK_STATUS(sr->ReadHeader(s, max_nnz, kSymbolGroup4Names[s]));
        }
      }
      break;
    }
    case TransformType::kSubtractGreen:
    case TransformType::kYCoCgR:
      break;
    default:
      assert(false);    // can't happen
      break;
  }

  return dec_->GetStatus();
}

// -----------------------------------------------------------------------------
// Metadata

void Metadata::Init(uint32_t num_pixels, bool has_alpha,
                    WP2SampleFormat format) {
  symbols_info_.Init(num_pixels, has_alpha, format);
  histogram_mask_ = 0;
  histogram_subsample_bits_ = 0;
  histogram_xsize_ = 0;
}

// -----------------------------------------------------------------------------
// Decoder

Decoder::Decoder() {
  DecLDspInit();  // Init critical function pointers.
}

void Decoder::Init(const WP2::DecoderConfig& config,
                   const WP2::GlobalParams& gparams, WP2::ANSDec* const dec,
                   WP2::Tile* const tile) {
  config_ = config;
  gparams_ = &gparams;
  dec_ = dec;
  last_out_row_ = 0;
  tile_ = tile;
  tile->num_decoded_rows = 0;
  hdr_.Init(tile->rect.GetArea(), gparams.has_alpha_,
            tile->rgb_output.format());
}

WP2Status Decoder::AllocateInternalBuffers(uint32_t final_width) {
  const uint64_t num_pixels = (uint64_t)tile_->rect.GetArea();
  // Scratch buffer corresponding to top-prediction row for transforming the
  // first row in the row-blocks. Not needed for paletted alpha.
  const uint64_t cache_top_pixels = final_width;
  // Scratch buffer for temporary BGRA storage. Not needed for paletted alpha.
  const uint64_t cache_pixels = (uint64_t)final_width * kNumARGBCacheRows;
  const uint64_t total_num_pixels =
      num_pixels + cache_top_pixels + cache_pixels;

  assert(tile_->rect.width <= final_width);
  argb_cache_ = nullptr;  // for sanity check
  WP2_CHECK_ALLOC_OK(pixels_.resize(4 * total_num_pixels));
  argb_cache_ = pixels_.data() + 4 * (num_pixels + cache_top_pixels);

  if (config_.info != nullptr && config_.info->bits_per_pixel != nullptr) {
    WP2_CHECK_STATUS(
        bits_per_pixel_.SetView(*config_.info->bits_per_pixel, tile_->rect));
  } else if (VDMatch(config_, "bits-per-pixel") ||
             VDMatch(config_, "lossless")) {
    WP2_CHECK_STATUS(
        bits_per_pixel_.Resize(tile_->rect.width, tile_->rect.height));
  }
  if (VDMatch(config_, "lossless")) {
    WP2_CHECK_STATUS(predicted_.SetFormat(WP2_Argb_38));
    WP2_CHECK_STATUS(predicted_.Resize(tile_->rect.width, tile_->rect.height));
  }

  return WP2_STATUS_OK;
}

WP2Status Decoder::DecodeHeader() {
  state_ = READ_HDR;

  const WP2::ANSDebugPrefix prefix1(dec_, "GlobalHeader");

  // Read the transforms.
  use_group4_ = false;
  dec_->AddDebugPrefix("transforms");
  const uint32_t index =
      dec_->ReadRValue(kPossibleTransformCombinationsNum, "index");
  const TransformType* transforms = kPossibleTransformCombinations[index];
  uint32_t num_transforms = 0;
  for (; num_transforms < kPossibleTransformCombinationSize; ++num_transforms) {
    if (kPossibleTransformCombinations[index][num_transforms] ==
        TransformType::kNum) {
      break;
    }
  }

  // De-activate all transforms in case the decoder is re-used.
  for (Transform& t : transforms_) t.type_ = TransformType::kNum;
  for (uint32_t i = 0; i < num_transforms; ++i) {
    TransformType type = transforms[i];
    WP2_CHECK_STATUS(ReadTransform(tile_->rect.width, tile_->rect.height, type,
                                   &transforms_[i]));
    if (type == TransformType::kGroup4) {
      use_group4_ = true;
      // Make sure it is the only used transform.
      assert(num_transforms == 1);
      dec_->PopDebugPrefix();
      return WP2_STATUS_OK;
    }
    WP2_CHECK_REDUCED_STATUS(
        RegisterTransformForVDebug(transforms_[i], config_.info));
  }
  premultiply_ = (gparams_->has_alpha_ ? dec_->ReadBool("premultiply") : false);

  // Modify the main image ranges if we only use the palette.
  if (num_transforms == 1 && transforms[0] == TransformType::kColorIndexing) {
    hdr_.symbols_info_.InitAsLabelImage(tile_->rect.GetArea(), num_colors_);
  } else if (num_transforms != 0) {
    int16_t minima_range[4], maxima_range[4];
    // Find the ranges in which the ARGB values are.
    GetARGBRanges(transforms, hdr_.symbols_info_.SampleFormat(), num_colors_,
                  minima_range, maxima_range);
    // Read the ARGB ranges. As the encoder tries to reduce the ranges of ARGB
    // values around zero, they are more efficiently stored with prefix coding.
    for (Symbol sym : {kSymbolA, kSymbolR, kSymbolG, kSymbolB}) {
      if (hdr_.symbols_info_.IsUnused(sym)) continue;
      const uint32_t channel = (int)sym - (int)kSymbolA;
      int16_t minimum, maximum;
      if (minima_range[channel] < 0 && dec_->ReadBool("min_is_negative")) {
        minimum = ReadPrefixCode(minima_range[channel], -1,
                                 kARGBRangePrefixSize, dec_, "minimum");
        if (dec_->ReadBool("max_is_negative")) {
          maximum = ReadPrefixCode(minimum, -1, kARGBRangePrefixSize, dec_,
                                   "maximum");
        } else {
          maximum = ReadPrefixCode(0, maxima_range[channel],
                                   kARGBRangePrefixSize, dec_, "maximum");
        }
      } else {
        minimum = ReadPrefixCode(0, maxima_range[channel], kARGBRangePrefixSize,
                                 dec_, "minimum");
        maximum = ReadPrefixCode(minimum, maxima_range[channel],
                                 kARGBRangePrefixSize, dec_, "maximum");
      }
      hdr_.symbols_info_.SetMinMax(sym, minimum, maximum);
    }
  }

  dec_->PopDebugPrefix();

  // Color cache
  hdr_.cache_config_.type = CacheType::kNone;
  hdr_.cache_config_.cache_bits = 0;
  // Color cache is disabled for 1x1 images.
  if (tile_->rect.GetArea() > 1 && dec_->ReadBool("color_cache")) {
    hdr_.cache_config_.type = CacheType::kHash;
    hdr_.cache_config_.cache_bits =
        dec_->ReadRange(1, kMaxCacheBits, "color_cache_bits");
    WP2_CHECK_STATUS(hdr_.color_cache_.Init(hdr_.cache_config_));
  }
  WP2_CHECK_STATUS(dec_->GetStatus());

  // Find the maximum subsampling that would result in at least 2 histograms.
  uint32_t max_sampling;
  bool more_than_one_hist = false;
  for (max_sampling = kHistogramBitsMax + 1;
       !more_than_one_hist && max_sampling-- > kHistogramBitsMin;) {
    more_than_one_hist = (SubSampleSize(tile_->rect.width, max_sampling) > 1 ||
                          SubSampleSize(tile_->rect.height, max_sampling) > 1);
  }

  // Read the entropy image if needed.
  uint32_t num_histo;
  if (more_than_one_hist && dec_->ReadBool("write_histogram_image")) {
    WP2::ANSDebugPrefix prefix_histogram_image(dec_, "histogram_image");
    const uint32_t histogram_precision =
        dec_->ReadRange(kHistogramBitsMin, max_sampling, "histogram_bits");
    const uint32_t histogram_width =
        SubSampleSize(tile_->rect.width, histogram_precision);
    const uint32_t histogram_height =
        SubSampleSize(tile_->rect.height, histogram_precision);
    assert(histogram_width > 1 || histogram_height > 1);  // more_than_one_hist

    num_histo = dec_->ReadRange(
        2, std::min(kMaxHistogramImageSize, histogram_width * histogram_height),
        "num_histograms_m2");

    // Read the entropy image probabilities.
    LosslessSymbolsInfo symbols_info;
    symbols_info.InitAsLabelImage(histogram_width * histogram_height,
                                  num_histo);

    // Read the entropy image.
    WP2_CHECK_STATUS(DecodeHelperImage(histogram_width, histogram_height,
                                       symbols_info, &hdr_.histogram_image_));

    hdr_.histogram_subsample_bits_ = histogram_precision;
  } else {
    num_histo = 1;
    hdr_.histogram_image_.clear();
  }
  WP2_CHECK_STATUS(dec_->GetStatus());

  // Finish setting up the color-cache
  if (hdr_.cache_config_.type != CacheType::kNone) {
    hdr_.symbols_info_.SetCacheRange(GetColorCacheRange(hdr_.cache_config_));
  }
  // Read the ANS probabilities.
  LosslessSymbolsInfo symbols_info;
  WP2_CHECK_STATUS(symbols_info.CopyFrom(hdr_.symbols_info_));
  symbols_info.SetNumClusters(num_histo);
  WP2_CHECK_STATUS(
      ReadANSStats(tile_->rect.width, tile_->rect.height, symbols_info));

  // Update information about the image.
  const int num_bits = hdr_.histogram_subsample_bits_;

  hdr_.histogram_xsize_ = SubSampleSize(tile_->rect.width, num_bits);
  hdr_.histogram_mask_ = (num_bits == 0) ? ~0 : (1 << num_bits) - 1;

  WP2_CHECK_REDUCED_STATUS(HeaderVDebug());

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status Decoder::DecodeImage() { return DecodeLines(tile_->rect.height); }

WP2Status Decoder::DecodeLines(uint32_t n_lines) {
  assert(last_row_ + n_lines <= tile_->rect.height);
  // Initialization.
  if (state_ != READ_DATA) {
    WP2_CHECK_STATUS(AllocateInternalBuffers(tile_->rect.width));
    state_ = READ_DATA;
  }

  WP2::Planef* const bits_per_pixel =
      bits_per_pixel_.IsEmpty() ? nullptr : &bits_per_pixel_;

  // Decode.
  WP2_CHECK_STATUS(DecodeImageData(tile_->rect.width, tile_->rect.height,
                                   last_row_ + n_lines, bits_per_pixel));
#if !defined(WP2_REDUCE_BINARY_SIZE)
  if (VDMatch(config_, "bits-per-pixel") && last_row_ == tile_->rect.height) {
    WP2::ArgbBuffer debug_output;
    WP2_CHECK_STATUS(
        debug_output.SetView(config_.info->debug_output, tile_->rect));
    WP2_CHECK_STATUS(bits_per_pixel_.ToGray(
        &debug_output, /*bit_depth=*/{5, /*is_signed=*/false}));
  }
#endif  // WP2_REDUCE_BINARY_SIZE

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2L
