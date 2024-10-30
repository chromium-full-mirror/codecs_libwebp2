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
// main entry for the lossless encoder.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/enc/lossless/losslessi_enc.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "src/common/constants.h"
#include "src/common/lossless/color_cache.h"
#include "src/common/progress_watcher.h"
#include "src/common/symbols.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/dsp/math.h"
#include "src/enc/lossless/backward_references_enc.h"
#include "src/enc/lossless/histogram_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/ans.h"
#include "src/utils/ans_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

using WP2::ProgressRange;

// -----------------------------------------------------------------------------

namespace {
void swap(EncodeInfo& a, EncodeInfo& b) {
  swap(a.line_tokens, b.line_tokens);
  swap(a.bits_per_pixel, b.bits_per_pixel);
}

// Writes a 'value' with prefix coding where the prefix is stored as symbol
// 'sym' in cluster 'cluster'.
void WriteLZ77PrefixCode(uint32_t sym, uint32_t cluster, int32_t value,
                         uint32_t min, uint32_t max, WP2_OPT_LABEL,
                         WP2::SymbolManager* const sw,
                         WP2::ANSEncBase* const enc, float* const cost) {
  // Write information about the prefix code representation.
  const WP2::PrefixCode prefix_code(value - min, kLZ77PrefixCodeSize);
  const WP2::PrefixCode prefix_code_max(max - min, kLZ77PrefixCodeSize);
  sw->ProcessWithCost(sym, cluster, prefix_code.prefix, prefix_code_max.prefix,
                      label, enc, cost);

  if (prefix_code.extra_bits_num == 0) return;

  // Write the rest of the representation.
  const std::string label_bits = WP2SPrint("%s_bits", label);
  if (prefix_code.prefix == prefix_code_max.prefix) {
    const uint32_t new_range =
        max - min + 1 -
        WP2::PrefixCode::Merge(prefix_code.prefix, kLZ77PrefixCodeSize, 0);
    if (new_range <= WP2::kANSMaxRange) {
      enc->PutRValue(prefix_code.extra_bits_value, new_range,
                     label_bits.c_str());
      if (cost != nullptr) *cost += WP2Log2(new_range);
    } else {
      // Go recursively.
      WriteLZ77PrefixCode(sym, cluster, prefix_code.extra_bits_value,
                          /*min=*/0, /*max=*/new_range - 1, label, sw, enc,
                          cost);
    }
  } else {
    if (prefix_code.extra_bits_num <= WP2::kANSMaxUniformBits) {
      enc->PutUValue(prefix_code.extra_bits_value, prefix_code.extra_bits_num,
                     label_bits.c_str());
    } else {
      enc->PutUValue(
          prefix_code.extra_bits_value & ((1u << WP2::kANSMaxUniformBits) - 1),
          WP2::kANSMaxUniformBits, label_bits.c_str());
      enc->PutUValue(prefix_code.extra_bits_value >> WP2::kANSMaxUniformBits,
                     prefix_code.extra_bits_num - WP2::kANSMaxUniformBits,
                     label_bits.c_str());
    }
    if (cost != nullptr) *cost += prefix_code.extra_bits_num;
  }
}
}  // namespace

// Counts the number of transforms for a combination of transforms from the
// official table.
uint32_t CrunchConfig::GetNumTransforms() const {
  uint32_t num_transforms = 0;
  while (num_transforms < kPossibleTransformCombinationSize &&
         recipe->transforms[num_transforms] != TransformType::kNum) {
    ++num_transforms;
  }
  return num_transforms;
}

bool CrunchConfig::ShouldBeCachedFor(const CrunchConfig& config) const {
  // First, make sure the encoding algorithm and pre-multiplication are the same
  // and that the set of transforms is a subset.
  if (recipe->algorithm != config.recipe->algorithm ||
      use_premultiplied != config.use_premultiplied ||
      !std::equal(recipe->transforms, recipe->transforms + GetNumTransforms(),
                  config.recipe->transforms)) {
    return false;
  }
  // If we have a palette, make sure the type is the same.
  for (uint32_t i = 0; i < GetNumTransforms(); ++i) {
    if (recipe->transforms[i] == TransformType::kColorIndexing &&
        palette_sorting_type != config.palette_sorting_type) {
      return false;
    }
  }
  return true;
}

// -----------------------------------------------------------------------------

// Writes an image in a raw manner.
static WP2Status WriteRawImage(const int16_t* const argb, uint32_t width,
                               uint32_t height,
                               const LosslessSymbolsInfo& symbols_info,
                               const ProgressRange& progress,
                               WP2::ANSEncBase* const enc,
                               EncodeInfo* const encode_info = nullptr) {
  ProgressRange progress_local(progress, 1.);
  const bool get_cost =
      encode_info != nullptr && !encode_info->bits_per_pixel.empty();

  // Cache the extrema.
  int16_t mins[4], maxs[4];
  for (Symbol sym : {kSymbolA, kSymbolR, kSymbolG, kSymbolB}) {
    if (symbols_info.IsUnused(sym)) continue;
    const uint32_t c = (int)sym - (int)kSymbolA;
    mins[c] = symbols_info.Min(sym, /*cluster=*/0);
    maxs[c] = symbols_info.Max(sym, /*cluster=*/0);
  }
  // Write the raw pixel values.
  for (uint32_t y = 0; y < height; ++y) {
    const int16_t* const row = &argb[4 * width * y];
    for (uint32_t x = 0; x < width; ++x) {
      // Deal with alpha.
      float cost;
      int16_t alpha;
      if (symbols_info.IsUnused(kSymbolA)) {
        cost = 0.f;
        alpha = WP2::kAlphaMax;
      } else {
        alpha = row[4 * x + 0];
        enc->PutRange(alpha, mins[0], maxs[0], kSymbolNames[(int)kSymbolA]);
        if (get_cost) cost = WP2Log2(maxs[0] - mins[0] + 1);
      }
      // Pre-multiply.
      for (uint32_t c = 1; c < 4; ++c) {
        if (symbols_info.IsUnused((int)kSymbolA + c)) continue;
        const int16_t min = WP2::DivBy255(mins[c] * alpha);
        const int16_t max = WP2::DivBy255(maxs[c] * alpha);
        enc->PutRange(WP2::DivBy255(row[4 * x + c] * alpha), min, max,
                      kSymbolNames[(int)kSymbolA + c]);
        if (get_cost) cost += WP2Log2(max - min + 1);
      }
      if (get_cost) encode_info->bits_per_pixel[y * width + x] = cost;
    }
    if (encode_info != nullptr && !encode_info->line_tokens.empty()) {
      encode_info->line_tokens[y] = enc->NumTokens();
    }
  }
  WP2_CHECK_STATUS(progress_local.AdvanceBy(1.));
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------

WP2Status StorePixels(uint32_t width, uint32_t height, uint32_t histo_bits,
                      const BackwardRefs& refs, const ProgressRange& progress,
                      const int16_t* const segments, WP2::ANSEncBase* const enc,
                      WP2::SymbolManager* const sw,
                      EncodeInfo* const encode_info) {
  const uint32_t histo_xsize =
      histo_bits ? SubSampleSize(width, histo_bits) : 1;
  const uint32_t tile_mask = (histo_bits == 0) ? 0 : ~((1u << histo_bits) - 1);
  // x and y trace the position in the image.
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t tile_x = x & tile_mask;
  uint32_t tile_y = y & tile_mask;
  uint32_t histogram_ix = (segments == nullptr) ? 0 : segments[4 * 0 + 2];

  const WP2::ProgressScale row_progress(progress, 1. / height);

  if (encode_info != nullptr) {
    assert(encode_info->line_tokens.empty() ||
           encode_info->line_tokens.size() == height);
    assert(encode_info->bits_per_pixel.empty() ||
           encode_info->bits_per_pixel.size() == width * height);
  }
  const bool get_cost =
      encode_info != nullptr && !encode_info->bits_per_pixel.empty();

  // Define unused symbols.
  const bool has_alpha = LosslessSymbolsInfo::HasAlpha(sw->symbols_info());
  const bool is_label_image =
      LosslessSymbolsInfo::IsLabelImage(sw->symbols_info());

  float cost;
  float* const cost_ptr = get_cost ? &cost : nullptr;
  for (RefsCursor c(refs); c.Ok(); c.Next()) {
    const PixelMode& v = *c.cur_pos_;
    if (v.GetMode() == kSymbolTypeDiscarded) continue;
    cost = 0.;
    if (histo_bits != 0u) {
      if ((tile_x != (x & tile_mask)) || (tile_y != (y & tile_mask))) {
        tile_x = x & tile_mask;
        tile_y = y & tile_mask;
        const uint32_t ind =
            (y >> histo_bits) * histo_xsize + (x >> histo_bits);
        histogram_ix = (segments == nullptr) ? ind : segments[4 * ind + 2];
      }
    }
    sw->ProcessWithCost(kSymbolType, histogram_ix, v.GetMode(), "lossless_mode",
                        enc, cost_ptr);
    switch (v.GetMode()) {
      case kSymbolTypeLiteral: {
        WP2::ANSDebugPrefix prefix(enc, "literal");
        if (has_alpha) {
          sw->ProcessWithCost(kSymbolA, histogram_ix, v.GetLiteral(0), "A", enc,
                              cost_ptr);
        }
        if (!is_label_image) {
          sw->ProcessWithCost(kSymbolR, histogram_ix, v.GetLiteral(1), "R", enc,
                              cost_ptr);
          sw->ProcessWithCost(kSymbolB, histogram_ix, v.GetLiteral(3), "B", enc,
                              cost_ptr);
        }
        sw->ProcessWithCost(kSymbolG, histogram_ix, v.GetLiteral(2), "G", enc,
                            cost_ptr);
        break;
      }
      case kSymbolTypeColorCacheIdx: {
        const uint32_t code = v.GetColorCacheIdx();
        sw->ProcessWithCost(kSymbolColorCache, histogram_ix, code,
                            v.GetColorCacheIdxRange() - 1, "color_cache_idx",
                            enc, cost_ptr);
        break;
      }
      case kSymbolTypeSegmentCacheIdx: {
        const uint32_t code = v.GetSegmentCacheIdx();
        sw->ProcessWithCost(kSymbolSegmentCache, histogram_ix, code,
                            v.GetSegmentCacheIdxRange() - 1,
                            "segment_cache_idx", enc, cost_ptr);
        break;
      }
      case kSymbolTypeCopy: {
        WP2::ANSDebugPrefix prefix(enc, "copy");
        const uint32_t pixel_ind = x + y * width;
        const uint32_t plane_code = OffsetToPlaneCode(width, v.GetOffset());
        const uint32_t max_plane_code = pixel_ind + kCodeToPlaneCodes - 1;
        WriteLZ77PrefixCode(kSymbolDist, histogram_ix, plane_code,
                            /*min=*/0, max_plane_code, "dist", sw, enc, &cost);
        uint32_t length_left = v.GetLength();
        uint32_t length_written = 0u;
        do {
          // Write in chunks of size at most kLZ77LengthEscape.
          // The first chunk has a size of at least kMinLZ77Length (otherwise
          // 0).
          const uint32_t min_length =
              (length_written == 0) ? kLZ77LengthMin : 0u;
          const uint32_t max_length = std::min(
              width * height - (pixel_ind + length_written), kLZ77LengthEscape);
          const uint32_t length_to_write = (length_left > kLZ77LengthEscape)
                                               ? kLZ77LengthEscape
                                               : length_left;
          WriteLZ77PrefixCode(kSymbolLen, histogram_ix, length_to_write,
                              min_length, max_length, "len", sw, enc, &cost);
          if (length_to_write < kLZ77LengthEscape) break;
          length_left -= length_to_write;
          length_written += length_to_write;
        } while (true);
        break;
      }
      default:
        assert(false);
    }
    if (get_cost) {
      const float bits_per_pixel = cost / v.GetLength();
      for (uint32_t i = 0; i < v.GetLength(); ++i) {
        encode_info->bits_per_pixel[y * width + x + i] = bits_per_pixel;
      }
    }
    x += v.GetLength();
    while (x >= width) {
      x -= width;
      if (encode_info != nullptr && !encode_info->line_tokens.empty()) {
        encode_info->line_tokens[y] = enc->NumTokens();
      }
      ++y;
      WP2_CHECK_STATUS(row_progress.AdvanceBy(1.));
    }
  }
  WP2_CHECK_STATUS(enc->GetStatus());
  return WP2_STATUS_OK;
}

WP2Status StorePixels(uint32_t width, uint32_t height, const BackwardRefs& refs,
                      WP2::ANSEncBase* const enc,
                      WP2::SymbolManager* const sw) {
  WP2_CHECK_STATUS(StorePixels(width, height, /*histo_bits=*/0, refs,
                               ProgressRange(), /*segments=*/nullptr, enc, sw));
  return WP2_STATUS_OK;
}

// Store the ANS statistics in the bitstream, so that they can be re-used
// when decoding the symbols.
// n_pixels is the number of pixels in the image, and use_palette a flag
// indicating whether the image uses a palette and therefore has no information
// in the A,R,B channels.
WP2Status WriteHeaders(const WP2::SymbolRecorder& recorder,
                       const LosslessSymbolsInfo& symbols_info,
                       uint32_t num_pixels, int effort,
                       WP2::ANSEncBase* const enc, WP2::ANSDictionaries* dicts,
                       WP2::SymbolWriter* const sw, float* const cost) {
  WP2::ANSDebugPrefix prefix(enc, "symbols");
  WP2_CHECK_STATUS(sw->Init(symbols_info, effort));
  WP2_CHECK_STATUS(sw->Allocate());
  float cost_tmp = 0.f;
  float* cost_tmp_ptr;
  if (cost == nullptr) {
    cost_tmp_ptr = nullptr;
  } else {
    *cost = 0.f;
    cost_tmp_ptr = &cost_tmp;
  }
  WP2::ANSDictionaries dict_tmp;
  WP2::ANSDictionaries* dict_tmp_ptr;
  if (dicts == nullptr) {
    // Even if dicts are not asked for, we need to compute them for kSymbolNames
    // so that they can be used in GetPotentialUsage.
    dicts = &dict_tmp;
    dict_tmp_ptr = nullptr;
  } else {
    dict_tmp_ptr = dicts;
  }

  // Iterate over all histograms and get the aggregate number of codes used.
  for (uint32_t i = 0; i < symbols_info.NumClusters(kSymbolType); ++i) {
    // Deal with the first symbol type.
    WP2_CHECK_STATUS(sw->WriteHeader(kSymbolType, i, num_pixels, recorder,
                                     kSymbolNames[kSymbolType], enc, dicts,
                                     cost_tmp_ptr));
    if (cost != nullptr) *cost += cost_tmp;
    bool is_maybe_used[kSymbolTypeNum];
    // TODO(vrabaud) For kSymbolType, it's either a dictionary, or a trivial
    // symbol (not a range) so we could limit the storage to those two methods.
    // We would get exact counts when decoding and we would therefore know
    // exactly what is used or not. That would lead to a very small gain in
    // compressibility.
    sw->GetPotentialUsage(kSymbolType, i, is_maybe_used, kSymbolTypeNum);

    // Figure out the number of elements and the max values in bits.
    for (int s = 1; s < kSymbolNum; ++s) {
      if (symbols_info.IsUnused(s)) continue;
      // Ignore symbols we don't even use.
      Symbol sym = (Symbol)s;
      if (!IsSymbolUsed(sym, is_maybe_used)) continue;
      WP2_CHECK_STATUS(sw->WriteHeader(sym, i, num_pixels, recorder,
                                       kSymbolNames[s], enc, dict_tmp_ptr,
                                       cost_tmp_ptr));
      if (cost != nullptr) *cost += cost_tmp;
    }
  }

  return WP2_STATUS_OK;
}

WP2Status EncodeHelperImage(WP2::ANSEncBase* const enc,
                            WP2::ANSDictionaries* const dicts,
                            const int16_t* const argb,
                            HashChain* const hash_chain,
                            BackwardRefsPool* const ref_pool, uint32_t width,
                            uint32_t height,
                            const LosslessSymbolsInfo& symbols_info, int effort,
                            const ProgressRange& progress) {
  if (width * height == 1) {
    WP2_CHECK_STATUS(
        WriteRawImage(argb, width, height, symbols_info, progress, enc));
    return WP2_STATUS_OK;
  }

  // Calculate backward references from ARGB image.
  WP2_CHECK_STATUS(hash_chain->Fill(effort, argb, width, height));
  BackwardRefsPool::RefsPtr refs = BackwardRefsPool::GetEmptyBackwardRefs();
  CacheConfig cache_config;
  WP2::SymbolRecorder symbol_recorder;
  WP2_CHECK_STATUS(symbol_recorder.Allocate(symbols_info, /*num_records=*/0));
  WP2_CHECK_STATUS(GetBackwardReferences(
      width, height, argb, effort, kLZ77Standard | kLZ77RLE,
      /*cache_bits_max=*/0, *hash_chain, symbols_info, &symbol_recorder,
      &cache_config, ref_pool, &refs));

  // Record symbol statistics.
  WP2::SymbolRecorder recorder;
  WP2::ANSEncNoop noop;
  WP2_CHECK_STATUS(recorder.Allocate(symbols_info, /*num_records=*/0));
  WP2_CHECK_STATUS(StorePixels(width, height, *refs, &noop, &recorder));

  // Store headers.
  WP2::SymbolWriter sw;
  WP2_CHECK_STATUS(WriteHeaders(recorder, symbols_info, width * height, effort,
                                enc, dicts, &sw));

  // Store actual literals.
  WP2_CHECK_STATUS(StorePixels(width, height, /*histo_bits=*/0, *refs, progress,
                               /*segments=*/nullptr, enc, &sw));
  return WP2_STATUS_OK;
}

WP2Status Encoder::EncodeImageInternal(
    const CrunchConfig& config, const ProgressRange& progress,
    const LosslessSymbolsInfo& symbols_info_init, float* const cost_best,
    WP2::ANSEnc* const enc_init, EncodeInfo* const encode_info,
    WP2::ANSDictionaries* const dicts_init) {
  // First, check whether Group4 is used or anything else.
  const TransformType* const end =
      config.recipe->transforms + kPossibleTransformCombinationSize;
  if (config.recipe->algorithm == EncodingAlgorithm::kGroup4) {
    WP2_CHECK_STATUS(Group4Encode(
        argb_buffer_, palette_.Size(), config.group4_use_move_to_front,
        config_.effort, progress, enc_init, encode_info));
    return WP2_STATUS_OK;
  }

  // Check if LZW is used.
  if (config.recipe->algorithm == EncodingAlgorithm::kLZW) {
    WP2_CHECK_STATUS(LZWEncode(argb_buffer_, palette_, config_.effort, enc_init,
                               encode_info));
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }

  // If using a color cache, do not have it bigger than the number of
  // colors.
  const bool use_palette = (std::find(config.recipe->transforms, end,
                                      TransformType::kColorIndexing) != end);
  const uint32_t width = argb_buffer_.width, height = argb_buffer_.height;
  const bool color_cache_is_disabled = (width * height == 1);
  const bool segment_cache_is_disabled = (width * height < 3);
  const uint32_t cache_bits_max =
      color_cache_is_disabled ? 0
      : use_palette ? std::min(1u + (uint32_t)WP2Log2Floor(palette_.Size()),
                               kMaxColorCacheBits)
                    : kMaxColorCacheBits;

  const int16_t* const argb = argb_buffer_.data;
  const uint32_t histogram_bits = config.histo_bits;
  const uint32_t histogram_width = SubSampleSize(width, histogram_bits);
  const uint32_t histogram_height = SubSampleSize(height, histogram_bits);
  const uint32_t num_segments_max = histogram_width * histogram_height;
  HistogramSet histogram_image;
  WP2::ANSEnc enc_best_tmp, enc_tmp;
  WP2::ANSDictionaries dicts_tmp;
  WP2::SymbolWriter sw;

  assert(width * height >= 1 && num_segments_max >= 1);
  assert(histogram_bits >= kHistogramBitsMin);
  assert(histogram_bits <= kHistogramBitsMax);

  LosslessSymbolsInfo symbols_info;
  WP2_CHECK_STATUS(symbols_info.CopyFrom(symbols_info_init));
  // Get the maximum range for the color cache symbol.
  uint32_t max_range = 0;
  for (uint32_t i = 0; i <= cache_bits_max; ++i) {
    const CacheConfig cache_config_tmp = {
        (i > 0) ? CacheType::kHash : CacheType::kNone, i, false};
    max_range = std::max(max_range, GetColorCacheRange(cache_config_tmp));
  }
  symbols_info.SetCacheRange(max_range, /*has_segment_cache=*/max_range != 0);
  symbols_info.SetNumClusters(1);
  // Create a SymbolRecorder that can record the maximum number of cache
  // symbols.
  WP2::SymbolRecorder symbol_recorder_one_segment;
  WP2_CHECK_STATUS(
      symbol_recorder_one_segment.Allocate(symbols_info, /*num_records=*/0));

  // Label image where only the green channel is filled.
  WP2::Vector_s16 segments;
  WP2_CHECK_ALLOC_OK(segments.resize(4 * num_segments_max));
  std::fill(segments.begin(), segments.end(), 0);
  WP2_CHECK_STATUS(hash_chain_.Fill(config_.effort, argb, width, height));
  for (uint32_t lz77s_idx = 0; lz77s_idx < config.lz77s_types_to_try_size;
       ++lz77s_idx) {
    const ProgressRange idx_progress(progress,
                                     1. / config.lz77s_types_to_try_size);
    const ProgressRange histo_symbols_progress(idx_progress, 0.2);
    const ProgressRange histo_image_progress(idx_progress, 0.2);
    const ProgressRange symbol_stats_progress(idx_progress, 0.2);
    const ProgressRange header_progress(idx_progress, 0.2);
    const ProgressRange literals_progress(idx_progress, 0.2);

    // Reset the bit writer or take the input one if we are at the last
    // configuration.
    WP2::ANSDictionaries* dicts;
    WP2::ANSEnc* enc;
    if (lz77s_idx + 1 < config.lz77s_types_to_try_size) {
      WP2_CHECK_STATUS(enc_tmp.Clone(*enc_init));
      enc = &enc_tmp;
      WP2_CHECK_STATUS(dicts_tmp.CopyFrom(*dicts_init));
      dicts = &dicts_tmp;
    } else {
      enc = enc_init;
      dicts = dicts_init;
    }

    BackwardRefsPool::RefsPtr refs_best =
        BackwardRefsPool::GetEmptyBackwardRefs();
    CacheConfig cache_config;
    WP2_CHECK_STATUS(GetBackwardReferences(
        width, height, argb, config_.effort,
        config.lz77s_types_to_try[lz77s_idx], cache_bits_max, hash_chain_,
        symbols_info, &symbol_recorder_one_segment, &cache_config, &ref_pool_,
        &refs_best));
    symbols_info.SetCacheRange(GetColorCacheRange(cache_config),
                               cache_config.use_segment_cache);
    WP2_CHECK_ALLOC_OK(
        histogram_image.Allocate(num_segments_max, symbols_info));

    // Build histogram image and symbols from backward references.
    WP2_CHECK_STATUS(GetHistoImageSymbols(
        width, height, *refs_best, config_.effort, histo_symbols_progress,
        histogram_bits, symbols_info, &histogram_image, segments.data()));

    // Write the global header
    enc->AddDebugPrefix("GlobalHeader");
    // Color Cache parameters.
    if (!color_cache_is_disabled &&
        enc->PutBool(cache_config.type != CacheType::kNone, "color_cache")) {
      assert(cache_config.cache_bits > 0);
      enc->PutRange(cache_config.cache_bits, 1, kMaxColorCacheBits,
                    "color_cache_bits");
    }

    // Segment Cache parameters.
    if (!segment_cache_is_disabled && cache_config.type != CacheType::kNone) {
      enc->PutBool(cache_config.use_segment_cache, "segment_cache");
    }

    // Find the maximum subsampling that would result in >= 2 histograms.
    uint32_t max_sampling;
    bool more_than_one_hist = false;
    for (max_sampling = kHistogramBitsMax + 1;
         !more_than_one_hist && max_sampling-- > kHistogramBitsMin;) {
      more_than_one_hist = (SubSampleSize(width, max_sampling) > 1 ||
                            SubSampleSize(height, max_sampling) > 1);
    }

    // Histogram image.
    const uint32_t num_segments = histogram_image.histograms_.size();
    if (num_segments > 1) assert(more_than_one_hist);
    const bool write_histogram_image =
        (more_than_one_hist &&
         enc->PutBool(num_segments > 1, "write_histogram_image"));
    if (write_histogram_image) {
      WP2::ANSDebugPrefix prefix_histogram_image(enc, "histogram_image");
      enc->PutRange(histogram_bits, kHistogramBitsMin, max_sampling,
                    "histogram_bits");
      enc->PutRange(num_segments, 2,
                    std::min(kMaxHistogramImageSize, num_segments_max),
                    "num_histograms_m2");
      LosslessSymbolsInfo symbols_info_tmp;
      symbols_info_tmp.InitAsLabelImage(histogram_width * histogram_height,
                                        num_segments);
      WP2_CHECK_STATUS(EncodeHelperImage(
          enc, dicts, segments.data(), &hash_chain_, &ref_pool_,
          histogram_width, histogram_height, symbols_info_tmp, config_.effort,
          histo_image_progress));
    } else {
      WP2_CHECK_STATUS(histo_image_progress.AdvanceBy(1.));
    }

    // Get the symbols stats.
    const uint32_t num_pixels = width * height;
    WP2::ANSEncNoop enc_noop;
    WP2::SymbolRecorder recorder;
    symbols_info.SetNumClusters(num_segments);
    WP2_CHECK_STATUS(recorder.Allocate(symbols_info, /*num_records=*/0));
    WP2_CHECK_STATUS(StorePixels(width, height, histogram_bits, *refs_best,
                                 symbol_stats_progress, segments.data(),
                                 &enc_noop, &recorder));

    // Store symbol headers.
    WP2_CHECK_STATUS(WriteHeaders(recorder, symbols_info, num_pixels,
                                  config_.effort, enc, dicts, &sw));
    WP2_CHECK_STATUS(header_progress.AdvanceBy(1.));
    enc->PopDebugPrefix();

    // Store actual literals.
    EncodeInfo current_encode_info;
    if (encode_info != nullptr) {
      WP2_CHECK_ALLOC_OK(current_encode_info.line_tokens.resize(
          encode_info->line_tokens.size()));
      WP2_CHECK_ALLOC_OK(current_encode_info.bits_per_pixel.resize(
          encode_info->bits_per_pixel.size()));
    }
    WP2_CHECK_STATUS(StorePixels(width, height, histogram_bits, *refs_best,
                                 literals_progress, segments.data(), enc, &sw,
                                 &current_encode_info));

    // Keep track of the smallest image so far.
    const float cost_tmp = enc->GetCost(*dicts);
    if (cost_tmp < *cost_best) {
      *cost_best = cost_tmp;
      WP2::swap(*enc, enc_best_tmp);
      if (encode_info != nullptr) {
        swap(*encode_info, current_encode_info);
      }
    }
  }
  // enc_best can be empty if we have not found anything better.
  WP2::swap(*enc_init, enc_best_tmp);

  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Transforms

void Encoder::ApplyYCoCgR() {
  RGB2YCoCgR(argb_buffer_.data, argb_buffer_.width * argb_buffer_.height);
}

void Encoder::ApplySubtractGreen() {
  SubtractGreenFromBlueAndRed(argb_buffer_.data,
                              argb_buffer_.width * argb_buffer_.height);
}

// Finds the best spatial predictor for each prediction block (of size
// 2^(enc->transform_bits_)), saves the residuals in enc->argb_ and writes the
// transform for each block in the form of an image.
WP2Status Encoder::ApplyPredictFilter(
    const int16_t min_values[4], const int16_t max_values[4],
    bool can_use_sub_modes, const ProgressRange& progress, uint32_t pred_bits,
    WP2::ANSEnc* const enc, WP2::ANSDictionaries* const dicts) {
  const ProgressRange residual_image_progress(progress, 0.5);
  const ProgressRange helper_image_progress(progress, 0.5);
  WP2::Vector_s8 sub_modes;
  WP2_CHECK_STATUS(CreateResidualImage(
      pred_bits, config_.keep_unmultiplied, can_use_sub_modes, min_values,
      max_values, config_.effort, residual_image_progress, &argb_buffer_,
      argb_scratch_, transform_data_, &sub_modes));

  const WP2::ANSDebugPrefix prefix(enc, "predictor");
  enc->PutRange(pred_bits, kTransformBitsMin, kTransformBitsMax,
                "transform_bits");
  const uint32_t transform_width = SubSampleSize(argb_buffer_.width, pred_bits);
  const uint32_t transform_height =
      SubSampleSize(argb_buffer_.height, pred_bits);
  LosslessSymbolsInfo symbols_info;
  symbols_info.InitAsPredictorImage(transform_width * transform_height);
  const WP2Status status = EncodeHelperImage(
      enc, dicts, transform_data_, (HashChain*)&hash_chain_,
      (BackwardRefsPool*)&ref_pool_, transform_width, transform_height,
      symbols_info, config_.effort, helper_image_progress);

  if (can_use_sub_modes && enc->PutBool(!sub_modes.empty(), "use_sub_modes")) {
    const WP2::ANSDebugPrefix prefix_sub_modes(enc, "sub_modes");
    // Also encode the sub-modes.
    WP2_CHECK_STATUS(WriteSubModes(transform_data_, /*needs_shift=*/false,
                                   sub_modes, config_.effort, enc, dicts));
  }
  return status;
}

// Finds the best cross channel transform for each prediction block (of size
// 2^(enc->transform_bits_)), saves the residuals in enc->argb_ and writes the
// transform for each block in the form of an image.
WP2Status Encoder::ApplyCrossColorFilter(const ProgressRange& progress,
                                         uint32_t ccolor_transform_bits,
                                         WP2::ANSEnc* const enc,
                                         WP2::ANSDictionaries* const dicts) {
  WP2_CHECK_STATUS(ColorSpaceTransform(ccolor_transform_bits, config_.effort,
                                       &argb_buffer_, transform_data_));
  const WP2::ANSDebugPrefix prefix(enc, "cross_color");
  enc->PutRange(ccolor_transform_bits, kTransformBitsMin, kTransformBitsMax,
                "transform_bits");
  const uint32_t transform_width =
      SubSampleSize(argb_buffer_.width, ccolor_transform_bits);
  const uint32_t transform_height =
      SubSampleSize(argb_buffer_.height, ccolor_transform_bits);
  LosslessSymbolsInfo symbols_info;
  symbols_info.InitAsCrossColorImage(transform_width * transform_height);
  WP2_CHECK_STATUS(EncodeHelperImage(
      enc, dicts, transform_data_, (HashChain*)&hash_chain_,
      (BackwardRefsPool*)&ref_pool_, transform_width, transform_height,
      symbols_info, config_.effort, progress));
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------

// Allocates the memory for argb (W x H) buffer, 2 rows of context for
// prediction and transform data.
// Flags influencing the memory allocated:
//  transform_bits_
//  use_predict_, use_cross_color_
WP2Status Encoder::AllocateTransformBuffer(
    const WP2::Vector<CrunchConfig>& configs) {
  const uint32_t width = pic_.width();
  const uint32_t height = pic_.height();
  const uint64_t image_size = width * height;
  // Figure out whether we use predict or cross color at least once over all the
  // crunch configurations.
  bool use_predict = false, use_cross_color = false;
  for (const CrunchConfig& config : configs) {
    const TransformType* end =
        config.recipe->transforms + kPossibleTransformCombinationSize;
    use_predict |= (std::find(config.recipe->transforms, end,
                              TransformType::kPredictor) != end ||
                    std::find(config.recipe->transforms, end,
                              TransformType::kPredictorWSub) != end);
    use_cross_color |= (std::find(config.recipe->transforms, end,
                                  TransformType::kCrossColor) != end);
  }

  // ResidualImage needs room for 2 scanlines of uint32 pixels with an extra
  // pixel in each, plus 2 regular scanlines of bytes.
  const uint64_t argb_scratch_size =
      use_predict ? (width + 1) * 2 +
                        (width * 2 + sizeof(uint32_t) - 1) / sizeof(uint32_t)
                  : 0;
  uint64_t transform_data_size = 0u;
  if (use_predict || use_cross_color) {
    // Find the maximal transform size needed.
    for (const CrunchConfig& c : configs) {
      transform_data_size =
          std::max(transform_data_size,
                   (uint64_t)SubSampleSize(width, c.transform_bits) *
                       SubSampleSize(height, c.transform_bits));
    }
  }
  const uint64_t mem_size =
      image_size + argb_scratch_size + transform_data_size;
  int16_t* mem;
  if (4 * mem_size > transform_mem_.size()) {
    WP2_CHECK_ALLOC_OK(transform_mem_.resize(4 * mem_size));
    mem = transform_mem_.data();
  } else {
    mem = transform_mem_.data();
  }
  argb_buffer_.width = width;
  argb_buffer_.height = height;
  argb_buffer_.data = mem;
  mem += 4 * image_size;
  argb_scratch_ = mem;
  mem += 4 * argb_scratch_size;
  transform_data_ = mem;

  return WP2_STATUS_OK;
}

WP2Status Encoder::MakeInputImageCopy(const CrunchConfig& config) {
  const uint32_t width = pic_.width();
  const uint32_t height = pic_.height();
  const bool can_premultiply = has_alpha_ &&
                               !WP2IsPremultiplied(pic_.format()) &&
                               !config_.keep_unmultiplied;
  if (!can_premultiply) assert(!config.use_premultiplied);
  argb_buffer_.has_alpha = has_alpha_;
  argb_buffer_.channel_bits = WP2Formatbpc(pic_.format());
  for (uint32_t y = 0; y < height; ++y) {
    int16_t* const dst_row = argb_buffer_.GetRow(y);
    if (WP2Formatbpc(pic_.format()) <= 8) {
      const uint8_t* const src_row = pic_.GetRow8(y);
      if (can_premultiply) {
        for (uint32_t x = 0; x < width; ++x) {
          const uint8_t a = src_row[4 * x + 0];
          if (a == 0) {
            // Zero out fully transparent pixels.
            std::fill(dst_row + 4 * x, dst_row + 4 * (x + 1), 0u);
          } else if (config.use_premultiplied) {
            dst_row[4 * x + 0] = a;
            for (uint32_t c = 1; c < 4; ++c) {
              dst_row[4 * x + c] = WP2::DivBy255(src_row[4 * x + c] * a);
            }
          } else {
            std::copy(src_row + 4 * x, src_row + 4 * (x + 1), dst_row + 4 * x);
          }
        }
      } else {
        // Simply copy pixels over.
        std::copy(src_row, src_row + 4 * width, dst_row);
      }
    } else {
      std::copy(pic_.GetRow16(y), pic_.GetRow16(y) + 4 * width, dst_row);
    }
  }
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------

// Note: Expects "enc->palette_" to be set properly.
WP2Status Encoder::MapImageFromPalette(const CrunchConfig& config) {
  const WP2Status status = palette_.Apply(pic_, argb_buffer_.data);
  argb_buffer_.has_alpha = has_alpha_;
  argb_buffer_.channel_bits = WP2Formatbpc(pic_.format());
  return status;
}

WP2Status Encoder::ApplyAndWriteTransformsImpl(
    const CrunchConfig& config, const CrunchConfig* const config_prev,
    const ProgressRange progress, LosslessSymbolsInfo* const symbols_info,
    WP2::ANSEnc* const enc, WP2::ANSDictionaries* const dicts) {
  // We only need to copy the original image if we are not building upon a
  // previous transformed image.
  bool need_copy = (config_prev == nullptr);

  // Get the number of transforms and previous transforms.
  const uint32_t num_transforms = config.GetNumTransforms();
  const uint32_t num_transforms_prev =
      (config_prev == nullptr) ? 0 : config_prev->GetNumTransforms();
  assert(config_prev == nullptr || num_transforms_prev < num_transforms);
  for (uint32_t i = 0; i < num_transforms; ++i) {
    if (config_prev != nullptr && i < num_transforms_prev) {
      // Skip transforms that are the same in the previous configuration.
      assert(config_prev->recipe->transforms[i] ==
             config.recipe->transforms[i]);
      continue;
    }
    const TransformType type = config.recipe->transforms[i];
    ProgressRange progress_local(progress,
                                 1. / (num_transforms - num_transforms_prev));
    switch (type) {
      case TransformType::kColorIndexing: {
        // Encode palette.
        ProgressRange palette_find(progress_local, 0.5);
        ProgressRange palette_write(progress_local, 0.5);
        WP2_CHECK_STATUS(palette_.FindBestMethod(pic_, palette_find,
                                                 config.palette_sorting_type,
                                                 config_.effort, this));
        WP2_CHECK_STATUS(palette_.Write(palette_write, enc, dicts, this));
        WP2_CHECK_STATUS(MapImageFromPalette(config));
        need_copy = false;
        break;
      }
      case TransformType::kYCoCgR:
      case TransformType::kSubtractGreen:
      case TransformType::kPredictor:
      case TransformType::kPredictorWSub:
      case TransformType::kCrossColor:
        if (need_copy) {
          WP2_CHECK_STATUS(MakeInputImageCopy(config));
          need_copy = false;
        }
        if (type == TransformType::kYCoCgR) {
          ApplyYCoCgR();
          WP2_CHECK_STATUS(progress_local.AdvanceBy(1.));
        } else if (type == TransformType::kSubtractGreen) {
          ApplySubtractGreen();
          WP2_CHECK_STATUS(progress_local.AdvanceBy(1.));
        } else if (type == TransformType::kPredictor ||
                   type == TransformType::kPredictorWSub) {
          // Define the minimum/maximum values before applying the predictors.
          int16_t minima_range[4], maxima_range[4];
          GetARGBRanges(config.recipe->transforms, symbols_info_.SampleFormat(),
                        palette_.Size(), minima_range, maxima_range,
                        /*num_transforms=*/i);
          WP2_CHECK_STATUS(ApplyPredictFilter(
              minima_range, maxima_range, config.HasSubModes(), progress_local,
              config.transform_bits, enc, dicts));
        } else {
          WP2_CHECK_STATUS(ApplyCrossColorFilter(
              progress_local, config.transform_bits, enc, dicts));
        }
        break;
      case TransformType::kNum:
        assert(false);
    }
  }
  if (need_copy) WP2_CHECK_STATUS(MakeInputImageCopy(config));
  return WP2_STATUS_OK;
}

// Applies the different transforms decided by the cruncher and writes them to
// the header.
WP2Status Encoder::ApplyAndWriteTransforms(
    const CrunchConfig& config, const CrunchConfig* const config_prev,
    bool cache_data, const WP2::ProgressRange progress,
    LosslessSymbolsInfo* const symbols_info, WP2::ANSEnc* const enc,
    WP2::ANSDictionaries* const dicts) {
  WP2_CHECK_STATUS(symbols_info->CopyFrom(symbols_info_));
  symbols_info->SetCacheRange(0, /*has_segment_cache=*/false);

  const WP2::ANSDebugPrefix prefix1(enc, "GlobalHeader");
  const WP2::ANSDebugPrefix prefix2(enc, "transforms");

  // Find the transform combination in the official set and store its index.
  for (uint32_t index = 0; index < kPossibleEncodingRecipesNum; ++index) {
    if (config.recipe->algorithm == kPossibleEncodingRecipes[index].algorithm &&
        std::equal(
            config.recipe->transforms,
            config.recipe->transforms + kPossibleTransformCombinationSize,
            kPossibleEncodingRecipes[index].transforms)) {
      enc->PutRValue(index, kPossibleEncodingRecipesNum, "index");
      break;
    }
  }

  if (config_prev != nullptr) {
    // If we have already computed transforms before, re-use cached data.
    WP2_CHECK_STATUS(enc->AppendTokens(cached_transforms_enc_, 0,
                                       std::numeric_limits<uint32_t>::max()));
    WP2::swap(cached_transforms_dict_, *dicts);
  }
  if (cache_data) {
    // First store to cached_transforms_core_ for future use.
    cached_transforms_enc_.Reset();
    cached_transforms_enc_.CopyDebugPrefix(*enc);
    WP2_CHECK_STATUS(
        ApplyAndWriteTransformsImpl(config, config_prev, progress, symbols_info,
                                    &cached_transforms_enc_, dicts));
    WP2_CHECK_STATUS(cached_transforms_dict_.CopyFrom(*dicts));
    // Then append cached_transforms_core_ to the output encoder.
    WP2_CHECK_STATUS(enc->AppendTokens(cached_transforms_enc_, 0,
                                       std::numeric_limits<uint32_t>::max()));
  } else {
    WP2_CHECK_STATUS(ApplyAndWriteTransformsImpl(config, config_prev, progress,
                                                 symbols_info, enc, dicts));
  }

  if (config.recipe->algorithm == EncodingAlgorithm::kGroup4) {
    // Green channel of first pixel.
    const int16_t first_color = argb_buffer_.data[2];
    assert(first_color >= 0);
    enc->PutRValue(first_color, palette_.Size(), "first_color");
    if (palette_.Size() > 3) {
      // MoveToFront is only relevant when there are more than 3 colors.
      // With 2 colors, any color change is necessarily the other color
      // (different from the current one). With 3 colors, we use a boolean
      // to encode whether the new color is the one expected (from the row
      // above), or the other one, and that is enough.
      enc->PutBool(config.group4_use_move_to_front, "use_move_to_front");
    } else {
      assert(!config.group4_use_move_to_front);
    }
    return WP2_STATUS_OK;
  }
  if (config.recipe->algorithm == EncodingAlgorithm::kLZW) {
    WP2_CHECK_STATUS(MakeInputImageCopy(config));
    ProgressRange progress_local(progress, 1.);
    WP2_CHECK_STATUS(palette_.Write(progress_local, enc, dicts, this));
    // 'lzw_capacity' must be at least number of colors.
    const uint32_t lzw_capacity = palette_.Size() + kLZWCapacity;
    assert(lzw_capacity <= kLZWCapacityMax);
    enc->PutRange(lzw_capacity, palette_.Size() + 1, kLZWCapacityMax,
                  "capacity");
    return WP2_STATUS_OK;
  }

  if (has_alpha_ && image_is_premultiplied_) {
    // The lossless encoder may use unmultiplied samples for better compression,
    // even if the image is signaled as premultiplied overall.
    enc->PutBool(WP2IsPremultiplied(pic_.format()) || config.use_premultiplied,
                 "is_tile_premultiplied");
  } else {
    // If it is signaled that the color information in the translucent pixels is
    // kept, no lossless compression algorithm can encode premultiplied samples.
    assert(!config.use_premultiplied);
  }

  if (config.recipe->transforms[0] == TransformType::kNum) {
    ProgressRange progress_local(progress, 1.);
    WP2_CHECK_STATUS(progress_local.AdvanceBy(1.));
  } else if (config.recipe->transforms[0] == TransformType::kColorIndexing &&
             config.recipe->transforms[1] == TransformType::kNum) {
    symbols_info->InitAsLabelImage(pic_.width() * pic_.height(),
                                   palette_.Size());
  } else {
    // Find the ranges of the ARGB values in the image.
    int16_t minima[4], maxima[4];
    for (uint32_t i = 0; i < 4; ++i) {
      minima[i] = maxima[i] = argb_buffer_.data[i];
    }
    for (uint32_t i = 0; i < 4 * pic_.width() * pic_.height(); i += 4) {
      for (uint32_t c = 0; c < 4; ++c) {
        const int16_t val = argb_buffer_.data[i + c];
        if (val < minima[c]) {
          minima[c] = val;
        } else if (val > maxima[c]) {
          maxima[c] = val;
        }
      }
    }
    // Find the ranges in which they should belong according to the transforms.
    int16_t minima_range[4], maxima_range[4];
    GetARGBRanges(config.recipe->transforms, symbols_info->SampleFormat(),
                  palette_.Size(), minima_range, maxima_range);
    for (uint32_t c = 0; c < 4; ++c) {
      assert(minima_range[c] <= minima[c]);
      assert(minima[c] <= maxima[c]);
      assert(minima[c] <= maxima_range[c]);
    }
    // Write the ARGB ranges.
    for (Symbol sym : {kSymbolA, kSymbolR, kSymbolG, kSymbolB}) {
      if (symbols_info->IsUnused(sym)) continue;
      const uint32_t channel = (int)sym - (int)kSymbolA;
      if (minima_range[channel] < 0 &&
          enc->PutBool(minima[channel] < 0, "min_is_negative")) {
        WritePrefixCode(minima[channel], minima_range[channel], -1,
                        kARGBRangePrefixSize, enc, "minimum");
        if (enc->PutBool(maxima[channel] < 0, "max_is_negative")) {
          WritePrefixCode(maxima[channel], minima[channel], -1,
                          kARGBRangePrefixSize, enc, "maximum");
        } else {
          WritePrefixCode(maxima[channel], 0, maxima_range[channel],
                          kARGBRangePrefixSize, enc, "maximum");
        }
      } else {
        WritePrefixCode(minima[channel], 0, maxima_range[channel],
                        kARGBRangePrefixSize, enc, "minimum");
        WritePrefixCode(maxima[channel], minima[channel], maxima_range[channel],
                        kARGBRangePrefixSize, enc, "maximum");
      }
      symbols_info->SetMinMax(sym, minima[channel], maxima[channel]);
    }
  }

  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Encoder

Encoder::Encoder(const WP2::EncoderConfig& config,
                 const WP2::ArgbBuffer& picture, bool has_alpha,
                 bool is_premultiplied)
    : config_(config),
      pic_(picture),
      has_alpha_(has_alpha),
      image_is_premultiplied_(is_premultiplied) {
  argb_scratch_ = nullptr;
  transform_data_ = nullptr;

  symbols_info_.Init(picture.width() * picture.height(), has_alpha_,
                     pic_.format());

  EncLDspInit();
}

WP2Status Encoder::Allocate() {
  const uint32_t num_pixels = pic_.width() * pic_.height();
  WP2_CHECK_ALLOC_OK(hash_chain_.Allocate(num_pixels));
  palette_.Init(symbols_info_.SampleFormat(), has_alpha_,
                config_.keep_unmultiplied);
  ref_pool_.Init(num_pixels);

  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Main call

WP2Status EncodeImage(const WP2::EncoderConfig& encoder_config,
                      const WP2::ArgbBuffer& picture, bool has_alpha,
                      bool image_is_premultiplied,
                      const ProgressRange& progress,
                      WP2::ANSEnc* const enc_init,
                      EncodeInfo* const encode_info) {
  assert(!picture.IsEmpty());
  assert(enc_init != nullptr);

  const ProgressRange analysis_progress(progress, 0.5);
  const ProgressRange full_encode_progress(progress, 0.5);

  WP2::Vector<CrunchConfig> crunch_configs;

  // Analyze image (entropy, num_palettes etc)
  Encoder encoder(encoder_config, picture, has_alpha, image_is_premultiplied);
  WP2_CHECK_STATUS(encoder.Allocate());
  WP2_CHECK_STATUS(
      EncoderAnalyze(analysis_progress, &encoder, &crunch_configs));
  WP2_CHECK_STATUS(encoder.AllocateTransformBuffer(crunch_configs));
  const uint32_t width = encoder.pic_.width();
  const uint32_t height = encoder.pic_.height();
  uint32_t best_size = std::numeric_limits<uint32_t>::max();
  float best_cost = std::numeric_limits<float>::max();
  WP2::ANSEnc enc, enc_best;
  bool use_previous_transforms = false;

  for (uint32_t i_config = 0; i_config < crunch_configs.size(); ++i_config) {
    const CrunchConfig& config = crunch_configs[i_config];
    const ProgressRange config_progress(full_encode_progress,
                                        1. / crunch_configs.size());
    const ProgressRange transforms_progress(config_progress, 0.5);
    const ProgressRange encode_progress(config_progress, 0.5);

    // Reset the bit writer.
    WP2_CHECK_STATUS(enc.Clone(*enc_init));
    WP2::ANSDictionaries dicts;

    // Reset any parameter in the encoder that is set in the previous iteration.
    LosslessSymbolsInfo symbols_info;
    encoder.ref_pool_.Reset();

    // Check whether we need to cache data by checking whether the next
    // configuration has the same first transforms.
    bool cache_transforms = false;
    if (i_config + 1 < crunch_configs.size() &&
        config.recipe->algorithm != EncodingAlgorithm::kGroup4 &&
        config.recipe->algorithm != EncodingAlgorithm::kLZW) {
      cache_transforms = config.ShouldBeCachedFor(crunch_configs[i_config + 1]);
    }
    WP2_CHECK_STATUS(encoder.ApplyAndWriteTransforms(
        config,
        use_previous_transforms ? &crunch_configs[i_config - 1] : nullptr,
        cache_transforms, transforms_progress, &symbols_info, &enc, &dicts));
    use_previous_transforms = cache_transforms;

    // -------------------------------------------------------------------------
    // Encode and write the transformed image.
    EncodeInfo current_encode_info;
    if (encode_info != nullptr) {
      WP2_CHECK_ALLOC_OK(current_encode_info.line_tokens.resize(height));
      WP2_CHECK_ALLOC_OK(
          current_encode_info.bits_per_pixel.resize(width * height));
    }

    WP2_CHECK_STATUS(encoder.EncodeImageInternal(config, encode_progress,
                                                 symbols_info, &best_cost, &enc,
                                                 &current_encode_info, &dicts));
    // If enc has been emptied, nothing better could be found.
    if (enc.NumTokens() == 0) continue;

    // If we are better than what we already have.
    WP2_CHECK_STATUS(enc.AssembleToBitstream());
    if (enc.GetBitstreamSize() < best_size) {
      best_size = enc.GetBitstreamSize();
      // Store the BitWriter.
      WP2::swap(enc, enc_best);
      if (encode_info != nullptr) {
        swap(*encode_info, current_encode_info);
      }
    }
  }
  WP2::swap(*enc_init, enc_best);

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

bool EncodeInfo::CopyFrom(const EncodeInfo& other) {
  return line_tokens.copy_from(other.line_tokens) &&
         bits_per_pixel.copy_from(other.bits_per_pixel);
}

//------------------------------------------------------------------------------

}  // namespace WP2L
