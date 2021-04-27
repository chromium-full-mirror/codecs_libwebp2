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
// Palette computation and storage for WebP Lossless.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/enc/lossless/palette.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "src/common/color_precision.h"
#include "src/common/lossless/color_cache.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/utils/ans_utils.h"
#include "src/wp2/format_constants.h"

namespace WP2L {

using WP2::ProgressRange;

//------------------------------------------------------------------------------
// Palette.

// Computes a value that is related to the entropy created by the
// palette entry diff.
static inline uint32_t PaletteColorDistance(const int16_t col1[],
                                            const int16_t col2[],
                                            uint32_t weight_RGB) {
  assert(weight_RGB <= 256);
  const uint32_t weight_A = 256 - weight_RGB;
  const uint32_t score_A = std::abs(col1[0] - col2[0]);
  uint32_t score_RGB = 0;
  score_RGB += std::abs(col1[1] - col2[1]);
  score_RGB += std::abs(col1[2] - col2[2]);
  score_RGB += std::abs(col1[3] - col2[3]);
  return score_RGB * weight_RGB + score_A * weight_A;
}

// Find greedily always the closest color of the predicted color to minimize
// deltas in the palette. This reduces storage needs since the
// palette is stored with delta encoding.
static void GreedyMinimizeDeltas(uint32_t num_colors, int16_t colors[],
                                 uint32_t weight_RGB /* in [0..256] */) {
  int16_t predict[4] = {0};
  for (uint32_t i = 0; i < num_colors; ++i) {
    uint32_t best_ix = i;
    uint32_t best_score =
        PaletteColorDistance(&colors[4 * i], predict, weight_RGB);
    for (uint32_t k = i + 1; k < num_colors; ++k) {
      const uint32_t cur_score =
          PaletteColorDistance(&colors[4 * k], predict, weight_RGB);
      if (cur_score < best_score) {
        best_score = cur_score;
        best_ix = k;
      }
    }
    // swap color in, using 'predict' for tmp storage
    ColorCopy(&colors[4 * best_ix], predict);
    if (i != best_ix) {
      ColorCopy(&colors[4 * i], &colors[4 * best_ix]);
      ColorCopy(predict, &colors[4 * i]);
    }
  }
}

// The palette has been sorted by alpha. This function checks if the other
// components of the palette have a monotonic development with regards to
// position in the palette. If all have monotonic development, there is
// no benefit to re-organize them greedily. A monotonic development would be
// spotted in green-only situations (like lossy alpha) or gray-scale images.
static bool HasNonMonotonousDeltas(const int16_t* const colors,
                                   uint32_t num_colors) {
  uint8_t diff_pos_bits = 0x00, diff_neg_bits = 0x00;
  const int16_t* prev = &colors[4 * 0];
  const int16_t* cur =  &colors[4 * 1];
  while (cur != colors + 4 * num_colors) {
    for (uint32_t j = 1; j < 4; ++j) {
      const int16_t diff = (int16_t)cur[j] - (int16_t)prev[j];
      if (diff > 0) diff_pos_bits |= 1 << j;
      if (diff < 0) diff_neg_bits |= 1 << j;
    }
    if ((diff_pos_bits & diff_neg_bits) != 0) return true;
    prev = cur;
    cur += 4;
  }
  return false;
}

static bool LexicographicComparison(const int16_t* c1, const int16_t* c2) {
  for (uint32_t i = 0; i < 4; ++i) {
    if (c1[i] != c2[i]) return (c1[i] < c2[i]);
  }
  return false;
}

static bool LuminanceComparison(const int16_t* c1, const int16_t* c2) {
  // If different, choose alpha for comparison.
  if (c1[0] != c2[0]) return (c1[0] < c2[0]);
  // If not, choose basic luminance.
  const float luma1 = 0.299 * c1[1] + 0.587 * c1[2] + 0.114 * c1[3];
  const float luma2 = 0.299 * c2[1] + 0.587 * c2[2] + 0.114 * c2[3];
  return (luma1 < luma2);
}

// Sort colors in place.
static WP2Status SortColors(decltype(LexicographicComparison) comparator,
                            WP2::Vector_s16& colors_in) {
  const uint32_t num_colors = colors_in.size() / 4;
  WP2::VectorNoCtor<const int16_t*> colors;
  WP2_CHECK_ALLOC_OK(colors.resize(num_colors));
  // TODO(vrabaud) use a proper struct to store palette colors.
  for (uint32_t i = 0; i < num_colors; ++i) colors[i] = &colors_in[4 * i];
  std::sort(colors.begin(), colors.end(), comparator);

  // Write back.
  WP2::Vector_s16 colors_final;
  WP2_CHECK_ALLOC_OK(colors_final.resize(4 * num_colors));
  for (uint32_t i = 0; i < num_colors; ++i) {
    ColorCopy(colors[i], &colors_final[4 * i]);
  }
  std::copy(colors_final.begin(), colors_final.end(), colors_in.begin());
  return WP2_STATUS_OK;
}

template <typename T>
uint32_t CountColors(const WP2::ArgbBuffer& pic, const T* src,
                     ColorCacheMap* const color_cache,
                     const uint32_t max_palette_size) {
  assert(pic.format() == WP2_Argb_32 || pic.format() == WP2_Argb_38 ||
         pic.format() == WP2_ARGB_32);
  const bool premultiply = !WP2IsPremultiplied(pic.format());
  uint32_t num_colors = 0;
  // so we're sure that last_pix != argb[0]
  int16_t prev_pix[4] = {(int16_t)~src[0], 0, 0, 0};
  for (uint32_t y = 0; y < pic.height(); ++y) {
    for (uint32_t x = 0; x < pic.width(); ++x) {
      int16_t pix[4] = {(int16_t)src[4 * x + 0], (int16_t)src[4 * x + 1],
                        (int16_t)src[4 * x + 2], (int16_t)src[4 * x + 3]};
      if (premultiply) {
        for (uint32_t c = 1; c < 4; ++c) {
          pix[c] = WP2::DivBy255(pix[c] * pix[0]);
        }
      }
      if (ColorEq(pix, prev_pix)) continue;
      ColorCopy(pix, prev_pix);
      uint32_t value;
      if (!color_cache->Contains(pix, &value)) {
        color_cache->Insert(pix, num_colors);
        ++num_colors;
        // Stop here if we have too many colors.
        if (num_colors > max_palette_size) {
          return 0;  // Exact count not needed.
        }
      }
    }
    src = (const T*)((const uint8_t*)src + pic.stride());
  }
  return num_colors;
}

WP2Status Palette::AnalyzeAndCreate(const WP2::ArgbBuffer& pic) {
  WP2_CHECK_OK(pic.format() == WP2_Argb_32 || pic.format() == WP2_Argb_38 ||
                   pic.format() == WP2_ARGB_32,
               WP2_STATUS_INVALID_COLORSPACE);

  max_palette_size_ = MaxPaletteSize(pic.width() * pic.height());
  constexpr uint32_t kMaxDimension = 128u;
  full_optimize_ =
      (pic.width() < kMaxDimension && pic.height() < kMaxDimension);

  ColorCacheMap color_cache;
  WP2_CHECK_STATUS(color_cache.Allocate(max_palette_size_));

  if (WP2Formatbpc(pic.format()) <= 8) {
    num_colors_ =
        CountColors(pic, pic.GetRow8(0), &color_cache, max_palette_size_);
  } else {
    num_colors_ =
        CountColors(pic, pic.GetRow16(0), &color_cache, max_palette_size_);
  }

  if (num_colors_ <= 1 || num_colors_ > max_palette_size_) {
    num_colors_ = 0;  // disables the palette
    cost_ = std::numeric_limits<float>::max();
    return WP2_STATUS_OK;
  }

  // Gather back the palette.
  WP2_CHECK_ALLOC_OK(colors_.resize(4 * num_colors_));
  color_cache.GetPalette(colors_.data());

  is_grayscale_ = IsGrayScale();

  return WP2_STATUS_OK;
}

template <typename T>
void ApplyT(const WP2::ArgbBuffer& pic, const T* src,
            const ColorCacheMap& color_cache, int16_t* dst) {
  int16_t prev_pix[4] = {(int16_t)~src[0], (int16_t)~src[1], (int16_t)~src[2],
                         (int16_t)~src[3]};
  uint32_t prev_idx = 0;
  for (uint32_t y = 0; y < pic.height(); ++y) {
    for (uint32_t x = 0; x < pic.width(); ++x, dst += 4) {
      int16_t pix[4] = {(int16_t)src[4 * x + 0], (int16_t)src[4 * x + 1],
                        (int16_t)src[4 * x + 2], (int16_t)src[4 * x + 3]};
      if (!WP2IsPremultiplied(pic.format())) {
        for (uint32_t c = 1; c < 4; ++c) {
          pix[c] = WP2::DivBy255(pix[c] * pix[0]);
        }
      }
      if (!ColorEq(pix, prev_pix)) {
        if (!color_cache.Contains(pix, &prev_idx)) assert(false);
        ColorCopy(pix, prev_pix);
      }
      dst[0] = WP2::FormatMax(pic.format(), /*channel=*/0);
      dst[1] = 0;
      dst[2] = prev_idx;
      dst[3] = 0;
    }
    src = (const T*)((const uint8_t*)src + pic.stride());
  }
}

WP2Status Palette::Apply(const WP2::ArgbBuffer& pic, int16_t* dst) const {
  WP2_CHECK_OK(pic.format() == WP2_Argb_32 || pic.format() == WP2_Argb_38 ||
                   pic.format() == WP2_ARGB_32,
               WP2_STATUS_INVALID_COLORSPACE);

  // Fill the cache.
  ColorCacheMap color_cache;
  WP2_CHECK_STATUS(color_cache.Allocate(num_colors_));
  for (uint32_t i = 0; i < num_colors_; ++i) {
    color_cache.Insert(&colors_[4 * i], i);
  }

  if (WP2Formatbpc(pic.format()) <= 8) {
    ApplyT(pic, pic.GetRow8(0), color_cache, dst);
  } else {
    ApplyT(pic, pic.GetRow16(0), color_cache, dst);
  }

  return WP2_STATUS_OK;
}

bool Palette::IsGrayScale() const {
  for (uint32_t i = 0; i < num_colors_; ++i) {
    if (!has_alpha_) assert(colors_[4 * i + 0] == WP2::kAlphaMax);
    const auto r = colors_[4 * i + 1];
    const auto g = colors_[4 * i + 2];
    const auto b = colors_[4 * i + 3];
    if (r != g || g != b) return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Palette serialization.

class InternalParams {
 public:
  InternalParams(WP2SampleFormat format, bool has_alpha, uint32_t channel_max,
                 uint32_t num_channels, const WP2::Vector_s16& colors);
  // Allocates the needed space for analysis.
  WP2Status Init(const int16_t colors[]);
  // Stores the signs.
  WP2Status StoreSigns(WP2::ANSEncBase* const bw);
  // Used for List method. It modifies stats_.
  WP2Status WriteAsList(WP2::ANSEncBase* const enc);
  // Used for Image method.
  WP2Status WriteAsImage(const ProgressRange& progress,
                         WP2::ANSDictionaries* const dict,
                         Encoder* const encoder,
                         WP2::ANSEncBase* const enc) const;
  WP2Status WriteVerbatim(WP2::ANSEncBase* const enc) const;

 public:
  WP2SampleFormat format_;
  bool has_alpha_;
  uint32_t channel_max_;
  uint32_t num_channels_;
  uint32_t num_colors_;
  const int16_t* const colors_;  // only used for Verbatim

  // method-dependent:
  bool is_increasing_[4], is_decreasing_[4];
  // 'is_stored_as_above' checks whether a channel is > to the
  // previous color, or == but closer to channel_max_ than 0.
  WP2::Vector_u8 is_stored_as_above_[4];
  WP2::Vector_u16 channel_diffs_[4];
  WP2::VectorNoCtor<WP2::OptimizeArrayStorageStat> stats_;
  WP2::Vector_u8 opposite_;
};

WP2Status InternalParams::StoreSigns(WP2::ANSEncBase* const bw) {
  // Store the signs.
  for (uint32_t c = 0; c < num_channels_; ++c) {
    if (c == 0 && !has_alpha_) {
      assert(is_increasing_[c] && is_decreasing_[c]);
      continue;
    }
    // Store whether the color progression is monotonous.
    const bool is_monotonous = (is_increasing_[c] || is_decreasing_[c]);
    if (bw->PutBool(is_monotonous, "is_monotonous")) {
      // Store whether we oppose the sign.
      bw->PutBool(is_decreasing_[c] && !is_increasing_[c], "use_opposite_sign");
    } else {
      // Store the signs, the first value does not matter.
      const float cost =
          WP2::StoreVector(&is_stored_as_above_[c][1], num_colors_ - 1, 1,
                           stats_.data(), nullptr);
      // Try with storing the opposite.
      for (uint32_t i = 0; i < num_colors_ - 1; ++i) {
        opposite_[i] = (uint8_t)!is_stored_as_above_[c][i + 1];
      }
      const float cost_opposite = WP2::StoreVector(
          opposite_.data(), num_colors_ - 1, 1, stats_.data(), nullptr);
      const bool use_opposite_sign = (cost_opposite < cost);
      bw->PutBool(use_opposite_sign, "use_opposite_sign");
      bw->AddDebugPrefix("transform_index_signs");
      if (use_opposite_sign) {
        WP2::StoreVector(opposite_.data(), num_colors_ - 1, 1, stats_.data(),
                         bw);
      } else {
        WP2::StoreVector(&is_stored_as_above_[c][1], num_colors_ - 1, 1,
                         stats_.data(), bw);
      }
      bw->PopDebugPrefix();
    }
  }
  return bw->GetStatus();
}

WP2Status InternalParams::WriteVerbatim(WP2::ANSEncBase* const enc) const {
  assert(num_colors_ <= kSmallPaletteLimit);
  uint32_t channel_max[4];
  for (uint32_t c = 0; c < num_channels_; ++c) {
    channel_max[c] = WP2::FormatMax(format_, /*channel=*/c);
  }
  for (uint32_t i = 0; i < num_colors_; ++i) {
    if (has_alpha_) {
      enc->PutRValue(colors_[4 * i + 0], channel_max[0] + 1, "color_verbatim");
      for (uint32_t c = 1; c < num_channels_; ++c) {
        const uint32_t max = WP2::DivRound(
            (uint32_t)(channel_max[c] * colors_[4 * i + 0]), channel_max[0]);
        enc->PutRValue(colors_[4 * i + c], max + 1, "color_verbatim");
      }
    } else {
      for (uint32_t c = 1; c < num_channels_; ++c) {
        enc->PutRValue(colors_[4 * i + c], channel_max[c] + 1,
                       "color_verbatim");
      }
    }
  }
  WP2_CHECK_STATUS(enc->GetStatus());
  return WP2_STATUS_OK;
}

// Given the differences between channels 'channel_diffs_', and whether a
// channel is superior to the previous one 'is_stored_as_above_' (which deals
// with the special case of 0), try to store the diffs as:
//  - a random vector
//  - a list of values in range [0:value] or [value:channel_max] depending on
//    is_stored_as_above_
WP2Status InternalParams::WriteAsList(WP2::ANSEncBase* const enc) {
  // Store the channels one by one.
  for (uint32_t c = has_alpha_ ? 0 : 1; c < num_channels_; ++c) {
    const uint16_t channel_max = WP2::FormatMax(format_, c);
    const WP2::Vector_u16& diffs = channel_diffs_[c];
    uint16_t alpha_value = has_alpha_ ? channel_diffs_[0][0] : channel_max;
    uint16_t prev_channel = diffs[0];
    // Compute the cost when storing everything as ranges.
    float cost_range = WP2Log2(channel_max + 1);
    for (uint32_t i = 1; i < num_colors_; ++i) {
      // TODO(vrabaud): Check if we should stop here (when reaching the max
      //                and everything else is above).
      if (has_alpha_) {
        alpha_value += is_stored_as_above_[0][i] ?  channel_diffs_[0][i]
                                                 : -channel_diffs_[0][i];
        if (c > 0 && prev_channel > alpha_value) {
          prev_channel = alpha_value;
        }
      }
      if (is_stored_as_above_[c][i]) {
        const uint32_t max =
            (c == 0) ? channel_max : WP2::DivBy255(channel_max * alpha_value);
        cost_range += WP2Log2(max + 1 - prev_channel);
        assert(prev_channel <= max);
        assert((int)prev_channel + (int)diffs[i] <= (int)channel_max);
        prev_channel += diffs[i];
      } else {
        cost_range += WP2Log2(prev_channel + 1);
        assert(diffs[i] <= prev_channel);
        prev_channel -= diffs[i];
      }
    }
    // Try using StoreVector in case values are monotonic.
    const float cost_store_vector = StoreVector(
        &diffs[1], num_colors_ - 1, channel_max, stats_.data(), nullptr);

    // Pick the best of the two solutions.
    const bool use_vector = cost_store_vector < cost_range;
    enc->PutBool(use_vector, "use_vector");
    const uint32_t max_first = (c == 0) ? channel_max : channel_diffs_[0][0];
    enc->PutRValue(diffs[0], max_first + 1, "first_channel");
    if (use_vector) {
      enc->AddDebugPrefix("transform_index_vals");
      StoreVector(&diffs[1], num_colors_ - 1, channel_max, stats_.data(), enc);
      enc->PopDebugPrefix();
    } else {
      // Store everything else as ranges.
      alpha_value = has_alpha_ ? channel_diffs_[0][0] : channel_max;
      prev_channel = diffs[0];
      for (uint32_t i = 1; i < num_colors_; ++i) {
        if (has_alpha_) {
          alpha_value += (is_stored_as_above_[0][i]) ?  channel_diffs_[0][i]
                                                     : -channel_diffs_[0][i];
          if (c > 0 && prev_channel > alpha_value) prev_channel = alpha_value;
        }
        if (is_stored_as_above_[c][i]) {
          const uint32_t max =
              (c == 0) ? channel_max : WP2::DivBy255(channel_max * alpha_value);
          enc->PutRValue(diffs[i], (uint32_t)(max + 1 - prev_channel),
                         "store_diff");
          prev_channel += diffs[i];
        } else {
          enc->PutRValue(diffs[i], prev_channel + 1, "store_diff");
          prev_channel -= diffs[i];
        }
      }
    }
  }
  return WP2_STATUS_OK;
}

WP2Status InternalParams::WriteAsImage(const ProgressRange& progress,
                                       WP2::ANSDictionaries* const dict,
                                       Encoder* const encoder,
                                       WP2::ANSEncBase* const enc) const {
  // Compute the cost of storing the palette differences as an image.
  WP2::Vector_s16 tmp_palette;
  WP2_CHECK_ALLOC_OK(tmp_palette.resize(4 * num_colors_));
  std::fill(tmp_palette.begin(), tmp_palette.end(), 0);
  for (uint32_t c = 0; c < num_channels_; ++c) {
    for (uint32_t i = 0; i < num_colors_; ++i) {
      tmp_palette[4 * i + c] = channel_diffs_[c][i];
      if (c == 0 && !has_alpha_) {
        assert(tmp_palette[4 * i + c] ==
               (int16_t)((i > 0) ? 0u : WP2::kAlphaMax));
      }
    }
  }
  LosslessSymbolsInfo symbols_info;
  assert(format_ == encoder->symbols_info_.SampleFormat());
  symbols_info.Init(num_colors_, has_alpha_, format_);
  WP2_CHECK_STATUS(EncodeHelperImage(
      enc, dict, tmp_palette.data(), &encoder->hash_chain_, &encoder->ref_pool_,
      num_colors_, 1, symbols_info, /*effort=*/3, progress));
  return WP2_STATUS_OK;
}

InternalParams::InternalParams(WP2SampleFormat format, bool has_alpha,
                               uint32_t channel_max, uint32_t num_channels,
                               const WP2::Vector_s16& colors)
    : format_(format),
      has_alpha_(has_alpha),
      channel_max_(channel_max),
      num_channels_(num_channels),
      num_colors_(colors.size() / 4),
      colors_(colors.data()) {}

WP2Status InternalParams::Init(const int16_t colors[]) {
  for (auto& b : is_increasing_) b = true;
  for (auto& b : is_decreasing_) b = true;
  for (uint32_t i = 0; i < num_channels_; ++i) {
    WP2_CHECK_ALLOC_OK(is_stored_as_above_[i].resize(num_colors_));
  }
  WP2_CHECK_ALLOC_OK(stats_.resize(num_colors_ - 1));
  WP2_CHECK_ALLOC_OK(opposite_.resize(num_colors_ - 1));
  for (auto& d : channel_diffs_) WP2_CHECK_ALLOC_OK(d.resize(num_colors_));

  // Compute the differences between the channels.
  for (uint32_t c = 0; c < num_channels_; ++c) {
    int16_t channel_prev = colors[c];
    channel_diffs_[c][0] = channel_prev;
    // is_stored_as_above_[c][0] is unused.
    for (uint32_t i = 1; i < num_colors_; ++i) {
      if (has_alpha_) {
        const auto alpha = colors[4 * i + 0];
        if (c > 0 && channel_prev > alpha) channel_prev = alpha;
      }
      const int16_t channel = colors[4 * i + c];
      const int diff = (int)channel - (int)channel_prev;
      channel_diffs_[c][i] = (uint32_t)std::abs(diff);
      channel_prev = channel;
      is_increasing_[c] &= (diff >= 0);
      is_decreasing_[c] &= (diff <= 0);
      // Store it above if it is indeed above or at the same value than
      // before but closer to channel_max_.
      is_stored_as_above_[c][i] = (uint8_t)(
          (diff > 0) || (diff == 0 && (int)channel_max_ - channel < channel));
    }
    if (is_increasing_[c]) {
      std::fill(&is_stored_as_above_[c][0],
                &is_stored_as_above_[c][num_colors_], 1);
    } else if (is_decreasing_[c]) {
      std::fill(&is_stored_as_above_[c][0],
                &is_stored_as_above_[c][num_colors_], 0);
    }
  }
  return WP2_STATUS_OK;
}

WP2Status Palette::TryPalette(float cost_header, WP2::Vector_s16& colors,
                              const ProgressRange& progress,
                              Encoder* const encoder,
                              InternalParams* const params,
                              bool also_as_image) {
  bool is_better = false;
  WP2::ANSEncCounter bw;
  float cost;

  const ProgressRange image_progress(progress, 0.5);
  const ProgressRange list_progress(progress, 0.5);

  WP2_CHECK_STATUS(params->Init(&colors[0]));
  WP2_CHECK_STATUS(params->StoreSigns(&bw));
  const float cost_base = cost_header + bw.GetCost();

  // start by evaluating the kMethodImage as baseline.
  if (also_as_image) {
    bw.Reset();
    WP2::ANSDictionaries dict;
    WP2_CHECK_STATUS(params->WriteAsImage(image_progress, &dict, encoder, &bw));
    cost = cost_base + bw.GetCost(dict);
    if (cost < cost_) {
      cost_ = cost;
      method_ = PaletteStorageMethod::kMethodImage;
      is_better = true;
    }
  } else {
    WP2_CHECK_STATUS(image_progress.AdvanceBy(1.));
  }

  // then as kMethodList
  bw.Reset();
  WP2_CHECK_STATUS(params->WriteAsList(&bw));
  WP2_CHECK_STATUS(list_progress.AdvanceBy(1.));
  cost = cost_base + bw.GetCost();
  if (cost < cost_) {
    cost_ = cost;
    method_ = PaletteStorageMethod::kMethodList;
    is_better = true;
  }
  if (is_better) swap(colors, colors_);
  return WP2_STATUS_OK;
}

WP2Status Palette::FindBestMethod(const WP2::ArgbBuffer& pic,
                                  const ProgressRange& progress, Sorting type,
                                  int effort, Encoder* const encoder) {
  assert(type != Sorting::kUndefined);
  // Stop here if we have already sorted to the same type.
  if (type == sort_type_) {
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }
  method_ = PaletteStorageMethod::kMethodVerbatim;
  cost_ = std::numeric_limits<float>::max();
  if (num_colors_ < 2) {
    sort_type_ = Sorting::kLexicographic;
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }

  WP2::ANSEncCounter bw;
  // Write the number of colors.
  WP2::WritePrefixCode(num_colors_, 2, max_palette_size_, /*prefix_size=*/0,
                       &bw, "num_colors");

  // Check if values are grayscale.
  bw.PutBool(is_grayscale_, "is_grayscale");

  const uint16_t channel_max = encoder->symbols_info_.GetMaxRange(kSymbolR) - 1;
  const uint32_t num_channels = (is_grayscale_ ? 2 : 4);

  InternalParams params(format_, has_alpha_, channel_max, num_channels,
                        colors_);

  if (num_colors_ <= kSmallPaletteLimit) {
    sort_type_ = Sorting::kLexicographic;
    WP2_CHECK_STATUS(SortColors(LexicographicComparison, colors_));
    WP2_CHECK_STATUS(params.WriteVerbatim(&bw));
    method_ = PaletteStorageMethod::kMethodVerbatim;
    cost_ = bw.GetCost();
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }

  const float cost_header = bw.GetCost() + 1.f;  // +1 for the 'use_image' bit

  switch (type) {
    case Sorting::kLexicographic:
      sort_type_ = Sorting::kLexicographic;
      WP2_CHECK_STATUS(SortColors(LexicographicComparison, colors_));
      WP2_CHECK_STATUS(
          TryPalette(cost_header, colors_, progress, encoder, &params));
      break;
    case Sorting::kMinimalSize:
      WP2_CHECK_STATUS(
          SortMinimalSize(progress, effort, cost_header, encoder, &params));
      break;
    case Sorting::kLuminance:
      WP2_CHECK_STATUS(
          SortLuminance(progress, effort, cost_header, encoder, &params));
      break;
    case Sorting::kModifiedZeng:
      WP2_CHECK_STATUS(SortModifiedZeng(pic, progress, effort, cost_header,
                                        encoder, &params));
      break;
    case Sorting::kUndefined:
      assert(false);
  }
  return WP2_STATUS_OK;
}

// Class containing the symmetric matrix of co-occurrences. The diagonal is not
// used. Less RAM could be used due to the symmetry but we would lose in speed.
class CoOccurrenceMatrix {
 public:
  explicit CoOccurrenceMatrix(uint32_t num_colors) : num_colors_(num_colors) {}
  WP2Status Allocate() {
    WP2_CHECK_ALLOC_OK(mat_.resize(num_colors_ * num_colors_));
    std::fill(mat_.begin(), mat_.end(), 0);
    return WP2_STATUS_OK;
  }
  uint32_t operator()(uint32_t c1, uint32_t c2) const {
    return mat_[c1 * num_colors_ + c2];
  }
  // Update the co-occurrence except if c1 == c2.
  void Set(uint32_t c1, uint32_t c2) {
    if (c1 == c2) return;
    ++mat_[c1 * num_colors_ + c2];
    ++mat_[c2 * num_colors_ + c1];
  }
  void FindMax(uint16_t* c1, uint16_t* c2) {
    // Find the index that is most frequently located adjacent to other
    // (different) indexes.
    uint32_t best_sum = 0;
    for (uint32_t i = 0; i < num_colors_; ++i) {
      const uint32_t sum = std::accumulate(
          &mat_[i * num_colors_], &mat_[i * num_colors_] + num_colors_, 0u);
      if (sum > best_sum) {
        best_sum = sum;
        *c1 = i;
      }
    }
    // Find the index that is most frequently found adjacent to it.
    const uint32_t* const beg = &mat_[*c1 * num_colors_ + 0];
    *c2 = std::max_element(beg, beg + num_colors_) - beg;
    assert(*c1 != *c2);
  }

 private:
  WP2::Vector_u32 mat_;
  uint32_t num_colors_;
};

template <typename T>
WP2Status BuildCoOccurrence(const WP2::ArgbBuffer& pic, const T* src,
                            const ColorCacheMap& color_cache,
                            CoOccurrenceMatrix* cooccurrence) {
  WP2::Vector_u16 lines;
  WP2_CHECK_ALLOC_OK(lines.resize(2 * pic.width()));
  uint16_t* line_top = &lines[0];
  uint16_t* line_current = &lines[pic.width()];
  int16_t prev_pix[4] = {(int16_t)~src[0], (int16_t)~src[1], (int16_t)~src[2],
                         (int16_t)~src[3]};
  uint32_t prev_idx = 0;
  const T* const src_ini = src;
  (void)src_ini;
  for (uint32_t y = 0; y < pic.height(); ++y) {
    for (uint32_t x = 0; x < pic.width(); ++x) {
      int16_t pix[4] = {(int16_t)src[4 * x + 0], (int16_t)src[4 * x + 1],
                        (int16_t)src[4 * x + 2], (int16_t)src[4 * x + 3]};
      if (!WP2IsPremultiplied(pic.format())) {
        for (uint32_t c = 1; c < 4; ++c) {
          pix[c] = WP2::DivBy255(pix[c] * pix[0]);
        }
      }
      if (!ColorEq(pix, prev_pix)) {
        if (!color_cache.Contains(pix, &prev_idx)) assert(false);
        ColorCopy(pix, prev_pix);
      }
      line_current[x] = prev_idx;
      // 4-connectivity is what works best as mentioned in "On the relation
      // between Memon's and the modified Zeng's palette reordering methods".
      if (x > 0) cooccurrence->Set(prev_idx, line_current[x - 1]);
      if (y > 0) cooccurrence->Set(prev_idx, line_top[x]);
    }
    std::swap(line_top, line_current);
    src = (const T*)((const uint8_t*)src + pic.stride());
  }
  return WP2_STATUS_OK;
}

// Implements the modified Zeng method from "A Survey on Palette Reordering
// Methods for Improving the Compression of Color-Indexed Images" by Armando J.
// Pinho and Antonio J. R. Neves.
WP2Status Palette::SortModifiedZeng(const WP2::ArgbBuffer& pic,
                                    const ProgressRange& progress, int effort,
                                    float cost_header, Encoder* const encoder,
                                    InternalParams* const params) {
  // Fill the cache.
  ColorCacheMap color_cache;
  WP2_CHECK_STATUS(color_cache.Allocate(num_colors_));
  for (uint32_t i = 0; i < num_colors_; ++i) {
    color_cache.Insert(&colors_[4 * i], i);
  }

  // Build the co-occurrence matrix.
  CoOccurrenceMatrix cooccurrence(num_colors_);
  WP2_CHECK_STATUS(cooccurrence.Allocate());
  if (WP2Formatbpc(pic.format()) <= 8) {
    WP2_CHECK_STATUS(
        BuildCoOccurrence(pic, pic.GetRow8(0), color_cache, &cooccurrence));
  } else {
    WP2_CHECK_STATUS(
        BuildCoOccurrence(pic, pic.GetRow16(0), color_cache, &cooccurrence));
  }

  // Initialize the mapping list with the two best indices.
  WP2::Vector_u16 remapping;
  WP2_CHECK_ALLOC_OK(remapping.resize(num_colors_));
  cooccurrence.FindMax(&remapping[0], &remapping[1]);

  // Initialize the vector of sums and find the best one.
  struct Sum {
    uint16_t index;
    uint32_t sum;
  };
  WP2::VectorNoCtor<Sum> sums;
  // '-2' for excluding remapping[0] and remapping[1].
  WP2_CHECK_ALLOC_OK(sums.resize(num_colors_ - 2));
  Sum* best_sum;
  if (!sums.empty()) {
    // Initialize the sums with the first two remappings.
    best_sum = &sums[0];
    for (uint16_t i = 0, j = 0; i < num_colors_; ++i) {
      if (i == remapping[0] || i == remapping[1]) continue;
      sums[j].index = i;
      sums[j].sum =
          cooccurrence(i, remapping[0]) + cooccurrence(i, remapping[1]);
      if (sums[j].sum > best_sum->sum) best_sum = &sums[j];
      ++j;
    }
  }

  // We need to append and prepend to the list of remapping. To this end, we
  // actually define the nest start/end of the list as indices in a vector (with
  // a wrap around when the end is reached).
  uint32_t first = 0, last = 1;
  while (!sums.empty()) {
    const uint16_t best_index = best_sum->index;
    // Compute delta to know if we need to prepend or append the best index.
    int32_t delta = 0;
    const int32_t n = num_colors_ - sums.size();
    for (uint32_t ind = first, j = 0; (ind + j) % num_colors_ != last + 1;
         ++j) {
      const uint16_t l_j = remapping[(ind + j) % num_colors_];
      delta +=
          (n - 1 - 2 * (int32_t)j) * (int32_t)cooccurrence(best_index, l_j);
    }
    if (delta > 0) {
      first = (first == 0) ? num_colors_ - 1 : first - 1;
      remapping[first] = best_index;
    } else {
      ++last;
      remapping[last] = best_index;
    }
    // Remove best_sum from sums.
    *best_sum = sums.back();
    sums.resize_no_check(sums.size() - 1);
    // Update all the sums and find the best one.
    best_sum = &sums[0];
    for (Sum& sum : sums) {
      sum.sum += cooccurrence(best_index, sum.index);
      if (sum.sum > best_sum->sum) best_sum = &sum;
    }
  }
  assert((last + 1) % num_colors_ == first);

  // Re-map the palette.
  WP2::Vector_s16 colors;
  WP2_CHECK_ALLOC_OK(colors.resize(4 * num_colors_));
  for (uint32_t i = 0; i < num_colors_; ++i) {
    ColorCopy(&colors_[4 * remapping[(first + i) % num_colors_]],
              &colors[4 * i]);
  }
  WP2_CHECK_STATUS(progress.AdvanceBy(0.5));

  // Find the best method to store the palette.
  ProgressRange try_progress(progress, 0.5);
  WP2_CHECK_STATUS(
      TryPalette(cost_header, colors, try_progress, encoder, params));

  return WP2_STATUS_OK;
}

WP2Status Palette::SortLuminance(const ProgressRange& progress, int effort,
                                 float cost_header,
                                 Encoder* const encoder,
                                 InternalParams* const params) {
  sort_type_ = Sorting::kLuminance;
  WP2_CHECK_STATUS(SortColors(LuminanceComparison, colors_));
  WP2_CHECK_STATUS(progress.AdvanceBy(0.5));

  // Find the best method to store the palette.
  ProgressRange try_progress(progress, 0.5);
  cost_ = std::numeric_limits<float>::max();
  WP2_CHECK_STATUS(
      TryPalette(cost_header, colors_, try_progress, encoder, params));

  return WP2_STATUS_OK;
}

WP2Status Palette::SortMinimalSize(const ProgressRange& progress, int effort,
                                   float cost_header, Encoder* const encoder,
                                   InternalParams* const params) {
  sort_type_ = Sorting::kMinimalSize;
  // Advance progression by half the remaining range for each TryPalette() call.
  ProgressRange try_progress(progress, 0.5);
  WP2_CHECK_STATUS(SortColors(LexicographicComparison, colors_));
  WP2_CHECK_STATUS(
      TryPalette(cost_header, colors_, try_progress, encoder, params));

  const bool try_alt_order =
      (effort > 0 && HasNonMonotonousDeltas(&colors_[0], num_colors_));
  while (try_alt_order) {  // use "while" so we can "break" from the loop
    // prepare an alternate palette to try
    WP2::Vector_s16 alt_colors;
    WP2_CHECK_ALLOC_OK(alt_colors.copy_from(colors_));

    GreedyMinimizeDeltas(num_colors_, &alt_colors[0], 220);
    try_progress = ProgressRange(try_progress, 0.5);
    WP2_CHECK_STATUS(
        TryPalette(cost_header, alt_colors, try_progress, encoder, params));

    // without alpha, any other call to GreedyMinimizeDeltas() is a no-op
    if (!has_alpha_) break;

    if (effort > 2) {  // try few other sorts if time permits
      GreedyMinimizeDeltas(num_colors_, &alt_colors[0], 119);
      try_progress = ProgressRange(try_progress, 0.5);
      WP2_CHECK_STATUS(
          TryPalette(cost_header, alt_colors, try_progress, encoder, params));
    }
    if (effort > 3) {
      GreedyMinimizeDeltas(num_colors_, &alt_colors[0], 42);
      try_progress = ProgressRange(try_progress, 0.5);
      WP2_CHECK_STATUS(
          TryPalette(cost_header, alt_colors, try_progress, encoder, params));
    }
    if (effort > 4) {
      GreedyMinimizeDeltas(num_colors_, &alt_colors[0], 0);
      try_progress = ProgressRange(try_progress, 0.5);
      WP2_CHECK_STATUS(
          TryPalette(cost_header, alt_colors, try_progress, encoder, params));
    }
    // for slower efforts, we uniformly sample the range of weights
    // with increasing density.
    if (effort > 5 && full_optimize_) {
      // we use more sub-divisions with higher effort
      const int shift = std::min(effort - 4, 5);  // [1..5]
      // if 'effort' is high enough, or we already found that MethodImage
      // was better, try MethodImage again.
      const bool as_image =
          (effort > 6 || method_ == PaletteStorageMethod::kMethodImage);
      uint32_t base_w = 1, range = 256;
      while (true) {
        const uint32_t step = range >> shift;
        if (step == 0) break;
        uint32_t new_base_w = base_w;
        float cost_hi = cost_;
        float cost_saved = cost_;
        for (uint32_t dw = 0; dw < range; dw += step) {
          GreedyMinimizeDeltas(num_colors_, &alt_colors[0], base_w + dw);
          try_progress = ProgressRange(try_progress, 0.5);
          WP2_CHECK_STATUS(TryPalette(cost_header, alt_colors, try_progress,
                                      encoder, params, as_image));
          if (cost_ < cost_saved) {
            new_base_w = base_w + dw;
            cost_saved = cost_;
          }
        }
        if (cost_ == cost_hi) break;
        base_w = new_base_w;
        range = step;
      }
    }
    break;
  }

  WP2_CHECK_STATUS(try_progress.AdvanceBy(1.));  // Finish 'progress'.
  return WP2_STATUS_OK;
}

WP2Status Palette::Write(const ProgressRange& progress,
                         WP2::ANSEncBase* const enc,
                         WP2::ANSDictionaries* const dicts_in,
                         Encoder* const encoder) const {
  const WP2::ANSDebugPrefix prefix(enc, "palette");
  // Write the number of colors.
  WP2::WritePrefixCode(num_colors_, 2, max_palette_size_, /*prefix_size=*/0,
                       enc, "num_colors");

  // Check if values are grayscale.
  enc->PutBool(is_grayscale_, "is_grayscale");

  const uint32_t num_channels = (is_grayscale_ ? 2 : 4);
  const uint16_t channel_max = encoder->symbols_info_.GetMaxRange(kSymbolR) - 1;
  InternalParams params(format_, has_alpha_, channel_max, num_channels,
                        colors_);

  if (method_ == PaletteStorageMethod::kMethodVerbatim) {
    WP2_CHECK_STATUS(params.WriteVerbatim(enc));
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
    return WP2_STATUS_OK;
  }
  WP2_CHECK_STATUS(params.Init(&colors_[0]));
  WP2_CHECK_STATUS(params.StoreSigns(enc));
  const bool use_image = (method_ == PaletteStorageMethod::kMethodImage);
  if (enc->PutBool(use_image, "use_image")) {
    WP2::ANSDictionaries dict;
    WP2_CHECK_STATUS(params.WriteAsImage(progress, &dict, encoder, enc));
    WP2_CHECK_STATUS(dicts_in->AppendAndClear(&dict));
  } else if (method_ == PaletteStorageMethod::kMethodList) {
    WP2_CHECK_STATUS(params.WriteAsList(enc));
    WP2_CHECK_STATUS(progress.AdvanceBy(1.));
  } else {
    assert(false);
  }
  WP2_CHECK_STATUS(enc->GetStatus());
  return WP2_STATUS_OK;
}

}  // namespace WP2L
