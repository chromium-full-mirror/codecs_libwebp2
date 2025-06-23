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
#include <array>
#include <cassert>
#include <cstdint>
#include <string>

#include "src/common/color_precision.h"
#include "src/common/constants.h"
#include "src/common/global_params.h"
#include "src/common/lossless/color_cache.h"
#include "src/common/lossless/transforms.h"
#include "src/common/progress_watcher.h"
#include "src/common/symbols.h"
#include "src/common/vdebug.h"
#include "src/dec/symbols_dec.h"
#include "src/dsp/lossless/decl_dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/math.h"
#include "src/utils/ans.h"
#include "src/utils/ans_utils.h"
#include "src/utils/hash_map.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/debug.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

// -----------------------------------------------------------------------------

static const uint8_t kCodeToPlane[kCodeToPlaneCodes] = {
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a,
    0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04,
    0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
    0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
    0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
    0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
    0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e,
    0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
    0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
    0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70};

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
                  bool has_alpha, bool image_is_premultiplied,
                  bool tile_is_premultiplied, T* const out, uint32_t out_stride,
                  WP2SampleFormat out_format) {
  // If the whole image is expected to be unmultiplied, the lossless compression
  // algorithm cannot have used premultiplied samples.
  if (!image_is_premultiplied) assert(!tile_is_premultiplied);
  assert(out_format == WP2_Argb_32 || out_format == WP2_ARGB_32 ||
         out_format == WP2_Argb_38);
  if (has_alpha) {
    assert(WP2IsPremultiplied(out_format) == image_is_premultiplied);
  }
  const uint32_t max_value = WP2::FormatMax(out_format, 1);

  int lines = mb_h;
  T* row_out = out;
  while (lines-- > 0) {
    // TODO(vrabaud) support multiple output color spaces.
    for (uint32_t i = 0; i < 4 * mb_w; i += 4, row_in += 4) {
      const uint8_t alpha = static_cast<uint8_t>(row_in[0]);
      if (!has_alpha) assert(alpha == WP2::kAlphaMax);
      row_out[i] = alpha;
      if (!image_is_premultiplied || (alpha > 0 && tile_is_premultiplied) ||
          alpha == WP2::kAlphaMax) {
        // Keep unmultiplied, or keep premultiplied, or it does not matter
        // because alpha is fully opaque.
        for (int j = 1; j <= 3; ++j) {
          row_out[i + j] = static_cast<T>(row_in[j]);
        }
      } else if (alpha > 0 && !tile_is_premultiplied) {
        // Premultiply.
        for (int j = 1; j <= 3; ++j) {
          row_out[i + j] = static_cast<T>(
              std::min((row_in[j] * alpha + 127u) / 255u, max_value));
        }
      } else {
        // To optimize encoding, the encoder can store arbitrary rgb values when
        // a = 0 (fully transparent pixel, see GetResidual in predictor_enc.cc).
        // To make sure we return proper premultiplied Argb, we sanitize these
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
         transforms_[num_transforms].header_.type != TransformType::kNum) {
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

  int16_t* const rows_out = argb_cache_;
  // Inverse transforms.
  for (int ind = num_transforms - 1; ind >= 0; --ind) {
    const Transform& transform = transforms_[ind];
    // Check that all color map indices are valid here rather than in dsp.
    // The size(s) of the palette(s) could be deduced from the previous
    // transforms. Except values could still be negative which seems hard to
    // impose as a constraint.
    // The following needs to be checked in case of invalid images.
    if (transform.header_.type == TransformType::kColorIndexing) {
      const int16_t color_map_size = (int16_t)transform.data_.size() / 4;
      const int16_t* const r_end = rows + 4 * num_rows * transform.width_pic_;
      for (const int16_t* r = rows; r < r_end; r += 4) {
        WP2_CHECK_OK(r[2] >= 0 && r[2] < color_map_size,
                     WP2_STATUS_BITSTREAM_ERROR);
      }
    } else if (transform.header_.type == TransformType::kNormalizeChannels) {
      const int16_t* const r_end = rows + 4 * num_rows * transform.width_pic_;
      for (const int16_t* r = rows; r < r_end; r += 4) {
        for (uint32_t c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
          WP2_CHECK_OK(
              r[c] >= 0 &&
                  r[c] <= static_cast<int16_t>(
                              transform.header_.normalize_channels_max[c]),
              WP2_STATUS_BITSTREAM_ERROR);
        }
      }
    }
    std::array<int32_t, 4> minima_range, maxima_range;
    if (transform.header_.type == TransformType::kPredictor ||
        transform.header_.type == TransformType::kPredictorWSub) {
      // Define the minimum/maximum values before applying the predictors.
      WP2_CHECK_STATUS(GetARGBRanges(minima_range, maxima_range, ind));
    }
    int16_t* predicted = nullptr;
#ifndef WP2_REDUCE_BINARY_SIZE
    if (((transform.header_.type == TransformType::kPredictor ||
          transform.header_.type == TransformType::kPredictorWSub) &&
         VDMatch(config_, "lossless/prediction/raw")) ||
        (transform.header_.type == TransformType::kCrossColor &&
         VDMatch(config_, "lossless/cross-color/raw"))) {
      predicted = (int16_t*)predicted_.GetRow16(start_row);
    }
#endif
    WP2_CHECK_REDUCED_STATUS(RegisterTransformedRowForVDebug(
        ind, num_rows, /*num_bits=*/8, rows, config_.info));
    InverseTransform(&transform, start_row, end_row, minima_range.data(),
                     maxima_range.data(), rows, gparams_->has_alpha_, rows_out,
                     predicted);
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

  if (row > last_row_) {  // Emit output.
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
                   image_is_premultiplied_, tile_is_premultiplied_,
                   buf->GetRow8(last_out_row_), buf->stride(), buf->format());
    } else {
      num_rows_out =
          EmitRows(rows_data, buf->width(), num_rows, gparams_->has_alpha_,
                   image_is_premultiplied_, tile_is_premultiplied_,
                   buf->GetRow16(last_out_row_), buf->stride(), buf->format());
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
  const int16_t* last_cached_color = src;
  const int16_t* last_cached_segment = src;
  const int16_t* const src_end = &data_vec->back() + 1;  // End of data
  int16_t* const src_last =
      data + 4 * width * last_row;  // Last pixel to decode
  WP2::SymbolReader* const sr = &hdr_.sr_;
  WP2::SegmentCache* const segment_cache =
      (do_process && !sr->symbols_info().IsUnused(kSymbolSegmentCache))
          ? &hdr_.segment_cache_
          : nullptr;
  ColorCache* const color_cache =
      do_process ? hdr_.color_cache_.get() : nullptr;
  const uint32_t mask = hdr_.histogram_mask_;

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
        (SymbolType)sr->Read(kSymbolType, cluster, "lossless_mode", cost_ptr);
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
    } else if (type == kSymbolTypeColorCacheIdx) {
      length = 1;
      assert(color_cache != nullptr);
      while (last_cached_color < src) {
        color_cache->Insert(last_cached_color, /*index_ptr=*/nullptr);
        last_cached_color += 4;
      }
      int32_t cache_index;
      WP2_CHECK_STATUS(sr->ReadWithMax(
          kSymbolColorCache, cluster, color_cache->IndexRange() - 1,
          "color_cache_idx", &cache_index, cost_ptr));
      color_cache->Lookup(cache_index, src);
      if (!gparams_->has_alpha_) assert(src[0] == WP2::kAlphaMax);
    } else if (type == kSymbolTypeSegmentCacheIdx) {
      assert(segment_cache != nullptr);
      STATIC_ASSERT_ARRAY_SIZE(kSegmentSizes, 1);
      static_assert(kSegmentSizes[0] == 2, "Unsupported feature");
      for (uint32_t segment_size : kSegmentSizes) {
        uint32_t shift =
            std::min(static_cast<uint32_t>(last_cached_segment - data),
                     4 * (segment_size - 1));
        const int16_t* start_seg = last_cached_segment - shift;
        const int16_t* end_seg = start_seg + 4 * (segment_size - 1);
        ColorSegment seg;  // 'seg' comprehends [start_seg, end_seg]
        while (end_seg < src) {
          seg = {start_seg, segment_size};
          segment_cache->Insert(seg, /*index=*/nullptr);
          start_seg += 4;
          end_seg += 4;
        }
      }
      last_cached_segment = src;
      int32_t cache_index;
      ColorSegment seg;
      WP2_CHECK_STATUS(sr->ReadWithMax(
          kSymbolSegmentCache, cluster, segment_cache->IndexRange() - 1,
          "segment_cache_idx", &cache_index, cost_ptr));
      segment_cache->Lookup(cache_index, &seg);
      WP2_CHECK_OK(seg.data != nullptr, WP2_STATUS_BITSTREAM_ERROR);
      length =
          std::min(seg.num_pixels, static_cast<uint32_t>(src_end - src) / 4);
      std::copy(seg.data, seg.data + 4 * length, src);
      if (!gparams_->has_alpha_) assert(src[0] == WP2::kAlphaMax);
    } else {                                            // Not reached
      WP2_CHECK_OK(false, WP2_STATUS_BITSTREAM_ERROR);  // For error reporting.
    }
    if (get_cost) {
      WP2_CHECK_REDUCED_STATUS(RegisterSymbolForVDebug(
          type, last_pixel_ + i_pixel, distance, length, cost, decoder_info));
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
    if ((type == kSymbolTypeSegmentCacheIdx || type == kSymbolTypeCopy ||
         (col & mask) == 0) &&
        src < src_end) {
      cluster = GetMetaIndex(hdr_.histogram_image_, hdr_.histogram_xsize_,
                             hdr_.histogram_subsample_bits_, col, row);
    }
    if (src < src_end) {
      assert(static_cast<int>(cluster) ==
             GetMetaIndex(hdr_.histogram_image_, hdr_.histogram_xsize_,
                          hdr_.histogram_subsample_bits_, col, row));
    }
  }

  // Finish filling color cache and segment cache.
  if (color_cache != nullptr) {
    while (last_cached_color < src) {
      color_cache->Insert(last_cached_color, /*index_ptr=*/nullptr);
      last_cached_color += 4;
    }
  }
  if (segment_cache != nullptr) {
    for (uint32_t segment_size : kSegmentSizes) {
      uint32_t shift =
          std::min(static_cast<uint32_t>(last_cached_segment - data),
                   4 * (segment_size - 1));
      const int16_t* start_seg = last_cached_segment - shift;
      const int16_t* end_seg = start_seg + 4 * (segment_size - 1);
      ColorSegment seg;  // 'seg' comprehends [start_seg, end_seg]
      while (end_seg < src) {
        seg = {start_seg, segment_size};
        segment_cache->Insert(seg, /*index=*/nullptr);
        start_seg += 4;
        end_seg += 4;
      }
    }
    last_cached_segment = src;
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
  if (encoding_algorithm_ == EncodingAlgorithm::kGroup4) {
    WP2_CHECK_STATUS(DecodeGroup4(width, last_row, bits_per_pixel, &src));
  } else if (encoding_algorithm_ == EncodingAlgorithm::kLZW) {
    WP2_CHECK_STATUS(DecodeLZW(width, last_row, bits_per_pixel, &src));
  } else if (encoding_algorithm_ == EncodingAlgorithm::kSCP ||
             encoding_algorithm_ == EncodingAlgorithm::kCALIC) {
    WP2_CHECK_STATUS(
        DecodePlaneCodec(width, last_row, scp_global_, bits_per_pixel, &src));
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
                                     bool is_premultiplied, bool is_grayscale,
                                     WP2::ANSDec* const dec,
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
    } else {
      transform->data_[4 * i + 0] = channel_max[0];
    }
    for (uint32_t c = 1; c < 4; ++c) {
      const uint32_t max =
          has_alpha && is_premultiplied
              ? WP2::DivRound(
                    (uint32_t)(channel_max[c] * transform->data_[4 * i + 0]),
                    channel_max[0])
              : channel_max[c];
      transform->data_[4 * i + c] =
          (is_grayscale && c > 1) ? transform->data_[4 * i + 1]
                                  : dec->ReadRValue(max + 1, "color_verbatim");
    }
  }
  return dec->GetStatus();
}

WP2Status Decoder::ReadPalette(Transform* const transform) {
  if (gparams_->has_alpha_) {
    assert(WP2IsPremultiplied(hdr_.symbols_info_.SampleFormat()) ==
           tile_is_premultiplied_);
  }
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
        tile_is_premultiplied_, is_grayscale, dec_, transform));
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
      WP2::LoadVector(dec_, 1, is_positive[c]);
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
        if (c > 0 && channel_prev > alpha && tile_is_premultiplied_) {
          channel_prev = alpha;
        }
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
      const uint32_t max_first = (c == 0 || !tile_is_premultiplied_)
                                     ? channel_max
                                     : transform->data_[0];
      int16_t channel = dec_->ReadRValue(max_first + 1, "first_channel");
      transform->data_[c] = channel;

      if (use_vector) {
        WP2::Vector_u16 diffs;
        WP2_CHECK_ALLOC_OK(diffs.resize(num_colors_ - 1));
        dec_->AddDebugPrefix("transform_index_vals");
        WP2::LoadVector(dec_, channel_max, diffs);
        dec_->PopDebugPrefix();
        for (uint32_t i = 1; i < num_colors_; ++i) {
          if (gparams_->has_alpha_) {
            const int16_t alpha = transform->data_[4 * i + 0];
            if (c > 0 && channel > alpha && tile_is_premultiplied_) {
              channel = alpha;
            }
          }
          channel = WP2::Clamp<int32_t>(
              channel + (is_positive[c][i - 1] ? diffs[i - 1] : -diffs[i - 1]),
              0, channel_max);
          transform->data_[4 * i + c] = channel;
        }
      } else {
        for (uint32_t i = 1; i < num_colors_; ++i) {
          const int16_t alpha = transform->data_[4 * i + 0];
          assert(alpha >= 0 && alpha <= (int16_t)alpha_max);
          if (gparams_->has_alpha_ && c > 0 && channel > alpha &&
              tile_is_premultiplied_) {
            channel = alpha;
          }
          if (is_positive[c][i - 1]) {
            const uint32_t max =
                (c == 0 || !tile_is_premultiplied_)
                    ? channel_max
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
                                 TransformType type, int index,
                                 Transform* const transform) {
  transform->header_.type = type;
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
          // Also make sure 0 maps to 0 for when we do not use sub-modes.
          mode |= ((sub_mode >= 0 ? 2 * sub_mode : -2 * sub_mode - 1)
                   << kNumPredictorsBits);
        }
      }
      WP2_CHECK_STATUS(dec_->GetStatus());

      dec_->PopDebugPrefix();
      break;
    }
    case TransformType::kCrossColorGlobal: {
      WP2::ANSDebugPrefix prefix(dec_, "cross_color_global");
      uint32_t y_index, uv_index;
      const uint32_t pre_defined = dec_->ReadRValue(7, "pre_defined");
      if (pre_defined < kCommonCrossColorIndices.size()) {
        y_index = kCommonCrossColorIndices[pre_defined].first;
        uv_index = kCommonCrossColorIndices[pre_defined].second;
      } else {
        y_index = dec_->ReadRange(1, 9, "y_index");
        uv_index = dec_->ReadRange(1, 12, "uv_index");
      }
      transform->header_.cc_global_y_index = y_index;
      transform->header_.cc_global_uv_index = uv_index;
      break;
    }
    case TransformType::kColorIndexing: {
      WP2_CHECK_STATUS(ReadPalette(transform));
      transform->header_.indexing_num_colors = transform->data_.size() / 4;
      break;
    }
    case TransformType::kNormalizeChannels: {
      WP2::ANSDebugPrefix prefix(dec_, "normalize_channels");
      std::array<int32_t, 4> minima_range, maxima_range;
      WP2_CHECK_STATUS(GetARGBRanges(minima_range, maxima_range, index));
      for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
        transform->header_.normalize_channels_offset[c] =
            dec_->ReadRange(minima_range[c], maxima_range[c], "offset");
      }
      transform->header_.normalize_channels_has_palette =
          dec_->ReadBool("has_palette");
      if (transform->header_.normalize_channels_has_palette) {
        for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
          const uint32_t range =
              1 + maxima_range[c] -
              transform->header_.normalize_channels_offset[c];
          const uint32_t mapping_size = dec_->ReadRange(1, range, "size");
          transform->header_.normalize_channels_max[c] = mapping_size - 1;
          WP2_CHECK_ALLOC_OK(
              transform->normalize_channels_mapping_[c].resize(mapping_size));
          WP2_CHECK_STATUS(
              LoadMapping(dec_, mapping_size, range,
                          transform->normalize_channels_mapping_[c].data()));
        }
      } else {
        for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
          transform->header_.normalize_channels_max[c] = dec_->ReadRValue(
              maxima_range[c] -
                  transform->header_.normalize_channels_offset[c] + 1,
              "max");
        }
      }
      break;
    }
    case TransformType::kSubtractGreen:
      break;
    case TransformType::kNum:
      assert(false);  // can't happen
      break;
  }

  return dec_->GetStatus();
}

WP2Status Decoder::GetARGBRanges(std::array<int32_t, 4>& minima_range,
                                 std::array<int32_t, 4>& maxima_range,
                                 int size) {
  std::array<TransformHeader, kPossibleTransformCombinationSize> headers;
  for (int i = 0; i < size; ++i) headers[i] = transforms_[i].header_;

  WP2_CHECK_STATUS(::WP2L::GetARGBRanges(headers,
                                         hdr_.symbols_info_.SampleFormat(),
                                         minima_range, maxima_range, size));
  return WP2_STATUS_OK;
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
                   const WP2::GlobalParams& gparams,
                   bool image_is_premultiplied, WP2::ANSDec* const dec,
                   WP2::Tile* const tile) {
  config_ = config;
  gparams_ = &gparams;
  image_is_premultiplied_ = image_is_premultiplied;
  // Default value for tests. Overridden by DecodeHeader().
  tile_is_premultiplied_ = image_is_premultiplied;
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
  dec_->AddDebugPrefix("transforms");
  const uint32_t index = dec_->ReadRValue(kPossibleEncodingRecipesNum, "index");
  encoding_algorithm_ = kPossibleEncodingRecipes[index].algorithm;

  uint32_t num_transforms = kPossibleTransformCombinationSize;
  for (uint32_t i = 0; i < kPossibleTransformCombinationSize; ++i) {
    const TransformType type = kPossibleEncodingRecipes[index].transforms[i];
    Transform& transform = transforms_[i];
    if (type == TransformType::kNum) {
      num_transforms = std::min(num_transforms, i);
      // Deactivate all transforms in case the decoder is re-used.
      transform.header_.type = TransformType::kNum;
    } else {
      WP2_CHECK_STATUS(ReadTransform(tile_->rect.width, tile_->rect.height,
                                     type, i, &transform));
      WP2_CHECK_REDUCED_STATUS(
          RegisterTransformForVDebug(transform, config_.info));
    }
  }

  // Read headers for specific algorithms.
  if (encoding_algorithm_ == EncodingAlgorithm::kGroup4) {
    // Find the palette.
    Transform* transform = nullptr;
    for (uint32_t i = 0; i < num_transforms; ++i) {
      if (transforms_[i].header_.type == TransformType::kColorIndexing) {
        transform = &transforms_[i];
        break;
      }
    }
    assert(transform != nullptr);
    const uint32_t color_map_size = (uint32_t)transform->data_.size() / 4;
    group4_first_color_ = dec_->ReadRValue(color_map_size, "first_color");
    const bool use_move_to_front =
        (num_colors_ > 3) ? dec_->ReadBool("use_move_to_front") : false;
    WP2_CHECK_STATUS(mtf_.Init(use_move_to_front, num_colors_));

    WP2::SymbolsInfo info;
    InitGroup4(tile_->rect.width, color_map_size, &info);

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
        const uint32_t max_nnz = tile_->rect.GetArea();
        WP2_CHECK_STATUS(sr->ReadHeader(s, max_nnz, kSymbolGroup4Names[s]));
      }
    }
    dec_->PopDebugPrefix();
    return WP2_STATUS_OK;
  } else if (encoding_algorithm_ == EncodingAlgorithm::kLZW) {
    // TODO(vrabaud): hack. The palette is stored in a transform though
    // it is invalid (type == kNum).
    // Read the palette in a temporary transform.
    transforms_[0].width_pic_ = tile_->rect.width;
    transforms_[0].height_pic_ = tile_->rect.height;
    WP2_CHECK_STATUS(ReadPalette(&transforms_[0]));

    const uint32_t lzw_capacity =
        dec_->ReadRange(num_colors_ + 1, kLZWCapacityMax, "capacity");
    WP2_CHECK_ALLOC_OK(lzw_segments_.reserve(lzw_capacity + 1));

    const WP2::Vector_s16& palette = transforms_[0].data_;
    lzw_segments_.resize_no_check(num_colors_ + 1);
    lzw_segments_[0] = {nullptr, 0};  // reserved for clear code
    for (uint32_t i = 0; i < num_colors_; ++i) {
      lzw_segments_[i + 1] = {&palette[4 * i], 1};
    }
    dec_->PopDebugPrefix();
    return WP2_STATUS_OK;
  } else if (encoding_algorithm_ == EncodingAlgorithm::kSCP ||
             encoding_algorithm_ == EncodingAlgorithm::kCALIC) {
    for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
      scp_global_.max_tpv[c] =
          c == 0 ? WP2::kAlphaMax
                 : (1 << WP2Formatbpc(tile_->rgb_output.format())) - 1;
    }
    for (const Transform& t : transforms_) {
      if (t.header_.type != TransformType::kNormalizeChannels) continue;
      for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
        scp_global_.max_tpv[c] = t.header_.normalize_channels_max[c];
      }
    }
    // Read the other parameters.
    dec_->PopDebugPrefix();
    WP2_CHECK_STATUS(DecodePlaneCodecHeader(&scp_global_));
    return WP2_STATUS_OK;
  }

  if (gparams_->has_alpha_ && image_is_premultiplied_) {
    // The lossless encoder may have used unmultiplied samples for better
    // compression, even if the image is signaled as premultiplied overall.
    tile_is_premultiplied_ = dec_->ReadBool("is_tile_premultiplied");
  } else {
    // If the whole image is considered unmultiplied, the lossless compression
    // algorithm cannot have encoded premultiplied samples.
    tile_is_premultiplied_ = false;
  }

  // Modify the main image ranges if we only use the palette.
  if (num_transforms == 1 &&
      transforms_[0].header_.type == TransformType::kColorIndexing) {
    hdr_.symbols_info_.InitAsLabelImage(tile_->rect.GetArea(), num_colors_);
  } else if (num_transforms != 0) {
    std::array<int32_t, 4> minima_range, maxima_range;
    // Find the ranges in which the ARGB values are.
    WP2_CHECK_STATUS(GetARGBRanges(minima_range, maxima_range));
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

  hdr_.cache_config_.type = CacheType::kNone;
  hdr_.cache_config_.cache_bits = 0;
  // Color cache is disabled for 1x1 images.
  if (tile_->rect.GetArea() > 1 && dec_->ReadBool("color_cache")) {
    hdr_.cache_config_.type = CacheType::kHash;
    hdr_.cache_config_.cache_bits =
        dec_->ReadRange(1, kMaxColorCacheBits, "color_cache_bits");
    WP2_CHECK_STATUS(hdr_.color_cache_.Init(hdr_.cache_config_));
  }
  WP2_CHECK_STATUS(dec_->GetStatus());

  bool has_segment_cache = false;
  // Segment cache should only be enabled for images with area greater than 2
  // and when color cache is set.
  if (tile_->rect.GetArea() > 2 &&
      hdr_.cache_config_.type != CacheType::kNone) {
    has_segment_cache = dec_->ReadBool("segment_cache");
    WP2_CHECK_STATUS(hdr_.segment_cache_.Allocate(kMaxSegmentCacheBits));
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

  // Finish setting up the color cache and segment cache.
  if (hdr_.cache_config_.type != CacheType::kNone) {
    hdr_.symbols_info_.SetCacheRange(GetColorCacheRange(hdr_.cache_config_),
                                     has_segment_cache);
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
