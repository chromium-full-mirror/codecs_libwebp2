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
#include <cstdlib>
#include <memory>
#include <string>

#include "src/common/color_precision.h"
#include "src/common/progress_watcher.h"
#include "src/common/symbols.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/enc/lossless/backward_references_enc.h"
#include "src/enc/lossless/histogram_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/ans.h"
#include "src/utils/ans_utils.h"
#include "src/utils/plane.h"
#include "src/utils/vector.h"

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
  const WP2::PrefixCode prefix_code(value - min, kLZZPrefixCodeSize);
  const WP2::PrefixCode prefix_code_max(max - min, kLZZPrefixCodeSize);
  sw->ProcessWithCost(sym, cluster, prefix_code.prefix, prefix_code_max.prefix,
                      label, enc, cost);

  if (prefix_code.extra_bits_num == 0) return;

  // Write the rest of the representation.
  const std::string label_bits = WP2SPrint("%s_bits", label);
  if (prefix_code.prefix == prefix_code_max.prefix) {
    const uint32_t new_range =
        max - min + 1 -
        WP2::PrefixCode::Merge(prefix_code.prefix, kLZZPrefixCodeSize, 0);
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
         transforms[num_transforms] != TransformType::kNum) {
    ++num_transforms;
  }
  return num_transforms;
}

bool CrunchConfig::ShouldBeCachedFor(const CrunchConfig& config) const {
  return (use_premultiplied == config.use_premultiplied) &&
         std::equal(transforms, transforms + GetNumTransforms(),
                    config.transforms);
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
  const uint32_t tile_mask = (histo_bits == 0) ? 0 : ~((1u << histo_bits)-1);
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
    const PixelMode& v = *c.cur_pos_;
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
      case kSymbolTypeCacheIdx: {
        const uint32_t code = v.GetCacheIdx();
        sw->ProcessWithCost(kSymbolCache, histogram_ix, code,
                            v.GetCacheIdxRange() - 1, "cache_idx", enc,
                            cost_ptr);
        break;
      }
      case kSymbolTypeCopy: {
        WP2::ANSDebugPrefix prefix(enc, "copy");
        const uint32_t pixel_ind = x + y * width;
        const uint32_t plane_code = DistanceToPlaneCode(width, v.GetDistance());
        const uint32_t max_plane_code = pixel_ind + kCodeToPlaneCodes - 1;
        WriteLZ77PrefixCode(kSymbolDist, histogram_ix, plane_code,
                            /*min=*/0, max_plane_code, "dist", sw, enc, &cost);
        const uint32_t max_length =
            std::min(width * height - pixel_ind, kMaxLZ77Length);
        WriteLZ77PrefixCode(kSymbolLen, histogram_ix, v.GetLength(),
                            kMinLZ77Length, max_length, "len", sw, enc, &cost);
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
  WP2_CHECK_STATUS(GetBackwardReferences(
      width, height, argb, effort, kLZ77Standard | kLZ77RLE,
      /*cache_bits_max=*/0, *hash_chain, symbols_info, &cache_config, ref_pool,
      &refs));

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
    const LosslessSymbolsInfo& symbols_info_init, WP2::ANSEnc* const enc,
    WP2::ANSDictionaries* const dicts, EncodeInfo* const encode_info) {
  // First, check whether we use Group4 or anything else.
  const TransformType* const end =
      config.transforms + kPossibleTransformCombinationSize;
  if (std::find(config.transforms, end, TransformType::kGroup4) != end) {
    WP2_CHECK_STATUS(Group4Encode(argb_buffer_, palette_.Size(),
                                  config.group4_use_move_to_front,
                                  config_.effort, progress, enc, encode_info));
    return WP2_STATUS_OK;
  }

  // If using a color cache, do not have it bigger than the number of
  // colors.
  const bool use_palette =
      (std::find(config.transforms, end, TransformType::kColorIndexing) != end);
  const uint32_t cache_bits_max =
      use_palette ? std::min(1u + (uint32_t)WP2Log2Floor(palette_.Size()),
                             kMaxCacheBits)
                  : kMaxCacheBits;

  const int16_t* const argb = argb_buffer_.data;
  const uint32_t width = argb_buffer_.width, height = argb_buffer_.height;
  const uint32_t histogram_bits = config.histo_bits;
  const uint32_t histogram_width = SubSampleSize(width, histogram_bits);
  const uint32_t histogram_height = SubSampleSize(height, histogram_bits);
  const uint32_t num_segments_max = histogram_width * histogram_height;
  HistogramSet histogram_image;
  WP2::ANSEnc enc_best;
  WP2::ANSDictionaries dicts_best;
  WP2::ANSEnc enc_init;
  WP2::ANSDictionaries dicts_init;
  float best_cost = std::numeric_limits<float>::max();
  WP2::SymbolWriter sw;

  assert(width * height >= 1 && num_segments_max >= 1);
  assert(histogram_bits >= kHistogramBitsMin);
  assert(histogram_bits <= kHistogramBitsMax);

  // Calculate backward references from ARGB image.
  WP2_CHECK_STATUS(enc_init.Clone(*enc));
  WP2_CHECK_STATUS(dicts_init.CopyFrom(*dicts));
  WP2_CHECK_STATUS(enc_best.Clone(*enc));
  WP2_CHECK_STATUS(dicts_best.CopyFrom(*dicts));
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

    LosslessSymbolsInfo symbols_info;
    WP2_CHECK_STATUS(symbols_info.CopyFrom(symbols_info_init));
    BackwardRefsPool::RefsPtr refs_best =
        BackwardRefsPool::GetEmptyBackwardRefs();
    CacheConfig cache_config;
    WP2_CHECK_STATUS(GetBackwardReferences(
        width, height, argb, config_.effort,
        config.lz77s_types_to_try[lz77s_idx], cache_bits_max, hash_chain_,
        symbols_info, &cache_config, &ref_pool_, &refs_best));
    symbols_info.SetCacheRange(GetColorCacheRange(cache_config));
    WP2_CHECK_ALLOC_OK(
        histogram_image.Allocate(num_segments_max, symbols_info));

    // Build histogram image and symbols from backward references.
    WP2_CHECK_STATUS(GetHistoImageSymbols(
        width, height, *refs_best, config_.effort, histo_symbols_progress,
        histogram_bits, symbols_info, &histogram_image, segments.data()));

    // Write the global header
    enc->AddDebugPrefix("GlobalHeader");
    // Color Cache parameters.
    if (enc->PutBool(cache_config.type != CacheType::kNone, "color_cache")) {
      assert(cache_config.cache_bits > 0);
      enc->PutRange(cache_config.cache_bits, 1, kMaxCacheBits,
                    "color_cache_bits");
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
    if (enc->GetCost(*dicts) < best_cost) {
      best_cost = enc->GetCost(*dicts);
      WP2::swap(*enc, enc_best);
      WP2::swap(*dicts, dicts_best);
      if (encode_info != nullptr) {
        swap(*encode_info, current_encode_info);
      }
    }
    // Reset the bit writer for the following iteration if any.
    if (lz77s_idx + 1 < config.lz77s_types_to_try_size) {
      WP2_CHECK_STATUS(enc->Clone(enc_init));
      dicts->DeepClear();
      WP2_CHECK_STATUS(dicts->CopyFrom(dicts_init));
    }
  }
  WP2::swap(*enc, enc_best);
  WP2::swap(*dicts, dicts_best);

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
WP2Status Encoder::ApplyPredictFilter(const int16_t min_values[4],
                                      const int16_t max_values[4],
                                      const ProgressRange& progress,
                                      uint32_t pred_bits,
                                      WP2::ANSEnc* const enc,
                                      WP2::ANSDictionaries* const dicts) {
  WP2_CHECK_STATUS(CreateResidualImage(pred_bits, min_values, max_values,
                                       config_.effort, &argb_buffer_,
                                       argb_scratch_, transform_data_));

  const WP2::ANSDebugPrefix prefix(enc, "predictor");
  enc->PutRange(pred_bits, kTransformBitsMin, kTransformBitsMax,
                "transform_bits");
  const uint32_t transform_width = SubSampleSize(argb_buffer_.width, pred_bits);
  const uint32_t transform_height =
      SubSampleSize(argb_buffer_.height, pred_bits);
  LosslessSymbolsInfo symbols_info;
  symbols_info.InitAsLabelImage(transform_width * transform_height,
                                kNumPredictors);
  const WP2Status status = EncodeHelperImage(
      enc, dicts, transform_data_, (HashChain*)&hash_chain_,
      (BackwardRefsPool*)&ref_pool_, transform_width, transform_height,
      symbols_info, config_.effort, progress);
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
        config.transforms + kPossibleTransformCombinationSize;
    use_predict |=
        (std::find(config.transforms, end, TransformType::kPredictor) != end);
    use_cross_color |=
        (std::find(config.transforms, end, TransformType::kCrossColor) != end);
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
  const bool can_premultiply =
      (has_alpha_ && !WP2IsPremultiplied(pic_.format()));
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

WP2Status Encoder::DoApplyAndWriteTransforms(
    const CrunchConfig& config, const CrunchConfig* const config_prev,
    const ProgressRange progress, LosslessSymbolsInfo* const symbols_info,
    WP2::ANSEnc* const enc, WP2::ANSDictionaries* const dicts,
    bool* const is_group4) {
  // We only need to copy the original image if we are not building upon a
  // previous transformed image.
  bool need_copy = (config_prev == nullptr);
  *is_group4 = false;

  // Get the number of transforms and previous transforms.
  const uint32_t num_transforms = config.GetNumTransforms();
  const uint32_t num_transforms_prev =
      (config_prev == nullptr) ? 0 : config_prev->GetNumTransforms();
  assert(config_prev == nullptr || num_transforms_prev < num_transforms);
  for (uint32_t i = 0; i < num_transforms; ++i) {
    if (config_prev != nullptr && i < num_transforms_prev) {
      // Skip transforms that are the same in the previous configuration.
      assert(config_prev->transforms[i] == config.transforms[i]);
      continue;
    }
    const TransformType type = config.transforms[i];
    ProgressRange progress_local(progress,
                                 1. / (num_transforms - num_transforms_prev));
    switch (type) {
      case TransformType::kColorIndexing: {
        // Encode palette.
        WP2_CHECK_STATUS(palette_.Write(progress_local, enc, dicts, this));
        WP2_CHECK_STATUS(MapImageFromPalette(config));
        need_copy = false;
        break;
      }
      case TransformType::kGroup4: {
        // Group4 cannot be used with other transforms.
        assert(num_transforms == 1);
        assert(palette_.Size() >= 2);
        WP2_CHECK_STATUS(palette_.Write(progress_local, enc, dicts, this));
        WP2_CHECK_STATUS(MapImageFromPalette(config));
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
        *is_group4 = true;
        return WP2_STATUS_OK;
      }
      case TransformType::kYCoCgR:
      case TransformType::kSubtractGreen:
      case TransformType::kPredictor:
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
        } else if (type == TransformType::kPredictor) {
          // Define the minimum/maximum values before applying the predictors.
          int16_t minima_range[4], maxima_range[4];
          GetARGBRanges(config.transforms, symbols_info_.SampleFormat(),
                        palette_.Size(), minima_range, maxima_range,
                        /*num_transforms=*/i);
          WP2_CHECK_STATUS(
              ApplyPredictFilter(minima_range, maxima_range, progress_local,
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
  symbols_info->SetCacheRange(0);

  const WP2::ANSDebugPrefix prefix1(enc, "GlobalHeader");
  const WP2::ANSDebugPrefix prefix2(enc, "transforms");

  // Find the transform combination in the official set and store its index.
  uint32_t index = 0;
  for (; index < kPossibleTransformCombinationsNum; ++index) {
    if (std::equal(config.transforms,
                   config.transforms + kPossibleTransformCombinationSize,
                   kPossibleTransformCombinations[index])) {
      break;
    }
  }

  enc->PutRValue(index, kPossibleTransformCombinationsNum, "index");

  if (config_prev != nullptr) {
    // If we have already computed transforms before, re-use cached data.
    WP2_CHECK_STATUS(enc->AppendTokens(cached_transforms_enc_, 0,
                                       std::numeric_limits<uint32_t>::max()));
    WP2::swap(cached_transforms_dict_, *dicts);
  }
  bool is_group4;
  if (cache_data) {
    // First store to cached_transforms_core_ for future use.
    cached_transforms_enc_.Clear();
    cached_transforms_enc_.CopyDebugPrefix(*enc);
    WP2_CHECK_STATUS(
        DoApplyAndWriteTransforms(config, config_prev, progress, symbols_info,
                                  &cached_transforms_enc_, dicts, &is_group4));
    WP2_CHECK_STATUS(cached_transforms_dict_.CopyFrom(*dicts));
    // Then append cached_transforms_core_ to the output encoder.
    WP2_CHECK_STATUS(enc->AppendTokens(cached_transforms_enc_, 0,
                                       std::numeric_limits<uint32_t>::max()));
  } else {
    WP2_CHECK_STATUS(DoApplyAndWriteTransforms(
        config, config_prev, progress, symbols_info, enc, dicts, &is_group4));
  }
  if (is_group4) return WP2_STATUS_OK;

  if (has_alpha_) {
    if (WP2IsPremultiplied(pic_.format())) {
      // If the format is already pre-multiplied, the decoder will not have to
      // do it.
      enc->PutBool(false, "premultiply");
    } else {
      // The decoder will have to pre-multiply if the transforms do not
      // pre-multiply, and conversely.
      enc->PutBool(!config.use_premultiplied, "premultiply");
    }
  }

  if (config.transforms[0] == TransformType::kNum) {
    ProgressRange progress_local(progress, 1.);
    WP2_CHECK_STATUS(progress_local.AdvanceBy(1.));
  } else if (config.transforms[0] == TransformType::kColorIndexing &&
             config.transforms[1] == TransformType::kNum) {
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
    GetARGBRanges(config.transforms, symbols_info->SampleFormat(),
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
                 const WP2::ArgbBuffer& picture, bool has_alpha)
    : config_(config), pic_(picture), has_alpha_(has_alpha) {
  argb_scratch_ = nullptr;
  transform_data_ = nullptr;

  symbols_info_.Init(picture.width() * picture.height(), has_alpha_,
                     pic_.format());

  EncLDspInit();
}

WP2Status Encoder::Allocate() {
  const uint32_t num_pixels = pic_.width() * pic_.height();
  WP2_CHECK_ALLOC_OK(hash_chain_.Allocate(num_pixels));
  palette_.Init(symbols_info_.SampleFormat(), has_alpha_);
  ref_pool_.Init(num_pixels);

  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Main call

WP2Status EncodeImage(const WP2::EncoderConfig& encoder_config,
                      const WP2::ArgbBuffer& picture, bool has_alpha,
                      const ProgressRange& progress,
                      WP2::ANSEnc* const enc_init,
                      EncodeInfo* const encode_info) {
  assert(!picture.IsEmpty());
  assert(enc_init != nullptr);

  const ProgressRange analysis_progress(progress, 0.5);
  const ProgressRange full_encode_progress(progress, 0.5);

  WP2::Vector<CrunchConfig> crunch_configs;

  // Analyze image (entropy, num_palettes etc)
  Encoder encoder(encoder_config, picture, has_alpha);
  WP2_CHECK_STATUS(encoder.Allocate());
  WP2_CHECK_STATUS(
      EncoderAnalyze(analysis_progress, &encoder, &crunch_configs));
  WP2_CHECK_STATUS(encoder.AllocateTransformBuffer(crunch_configs));
  const uint32_t width = encoder.pic_.width();
  const uint32_t height = encoder.pic_.height();
  uint32_t best_size = std::numeric_limits<uint32_t>::max();
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
        config.transforms[0] != TransformType::kGroup4) {
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
                                                 symbols_info, &enc, &dicts,
                                                 &current_encode_info));

    // If we are better than what we already have.
    WP2_CHECK_STATUS(enc.Assemble());
    if (enc.BufferSize() < best_size) {
      best_size = enc.BufferSize();
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
