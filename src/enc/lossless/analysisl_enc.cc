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

#include <algorithm>
#include <limits>

#include "src/common/color_precision.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/math.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/utils/hash.h"

namespace WP2L {

// -----------------------------------------------------------------------------
// Entropy analysis

namespace {

// These five modes are evaluated and their respective entropy is computed.
enum class EntropyMode {
  // No transform, colors are stored as is.
  kDirect,
  // Spatial prediction. We predict the next pixel (using one of many prediction
  // modes) and store the difference between the prediction and the actual value
  // (= "the residual").
  kSpatial,
  // Instead of storing red, green, and blue, we store red minus green, green,
  // and blue minus green (takes advantage of correlation between the three
  // channels).
  kSubGreen,
  // We use both kSpatial and kSubGreen a the same time.
  kSpatialSubGreen,
  // We create a palette of colors and store one palette index per pixel.
  kPalette,
  // Same as kPalette except that we then apply predictors on the indices
  // themselves.
  kPaletteSpatial,
  kNum
};

// Histogram types.
enum HistoIdx {
  kHistoAlpha = 0,
  kHistoAlphaPred,  // Alpha difference with previous pixel.
  kHistoGreen,
  kHistoGreenPred,  // Green difference with previous pixel.
  kHistoRed,
  kHistoRedPred,  // Red difference with previous pixel.
  kHistoBlue,
  kHistoBluePred,          // Blue difference with previous pixel.
  kHistoRedSubGreen,       // Red minus green.
  kHistoRedPredSubGreen,   // Red minus green difference with previous pixel.
  kHistoBlueSubGreen,      // Blue minus green.
  kHistoBluePredSubGreen,  // Blue minus green difference with previous pixel.
  kHistoPalette,
  kHistoNum
};

// Lightweight histogram, not owning data.
struct Histogram {
  // Adds v to 'data' by calling MakeIndex and clamping it to fit in 'len'.
  void AddSigned(int16_t v) {
    ++data[WP2::Clamp((int)MakeIndex(v), 0, len - 1)];
  }
  uint32_t* data;
  int len;
};

// Adds the red minus green, and blue minus green values from the given argb
// pixel 'p' to the given histograms 'r' and 'b'.
template <typename T>
void AddSingleSubGreen(const T* const p, Histogram* const r,
                       Histogram* const b) {
  const int green = p[2];  // The upper bits are masked away later.
  r->AddSigned(p[1] - green);
  b->AddSigned(p[3] - green);
}

// Adds the values from the given argb pixel 'p' to the given histograms 'a',
// 'r', 'g', and 'b'.
template <typename T>
void AddSingleSigned(const T* const p, Histogram* const a, Histogram* const r,
                     Histogram* const g, Histogram* const b) {
  a->AddSigned(p[0]);
  r->AddSigned(p[1]);
  g->AddSigned(p[2]);
  b->AddSigned(p[3]);
}
template <typename T>
void AddSingleUnsigned(const T* const p, Histogram* const a, Histogram* const r,
                       Histogram* const g, Histogram* const b) {
  assert(p[0] >= 0 && p[0] < a->len && p[1] >= 0 && p[1] < r->len);
  assert(p[2] >= 0 && p[2] < g->len && p[3] >= 0 && p[3] < b->len);
  ++a->data[p[0]];
  ++r->data[p[1]];
  ++g->data[p[2]];
  ++b->data[p[3]];
}

template <typename T>
void Premultiply(const T* const src, T* const dst) {
  const uint32_t a = src[0];
  if (a < 255) {
    dst[0] = src[0];
    dst[1] = WP2::DivBy255(src[1] * a);
    dst[2] = WP2::DivBy255(src[2] * a);
    dst[3] = WP2::DivBy255(src[3] * a);
  } else {
    std::copy(src, src + 4, dst);
  }
}

// Local struct containing the final config and local info for sorting.
struct EntropyConfig {
  CrunchConfig config;
  EntropyMode mode;
  float entropy;
};

// Find the corresponding combination of transforms in the reference table.
const TransformType* FindTransformationCombination(
    const TransformType* const transforms, uint32_t size) {
  assert(size <= kPossibleTransformCombinationSize);
  for (uint32_t i = 0; i < kPossibleTransformCombinationsNum; ++i) {
    if (std::equal(transforms, transforms + size,
                   kPossibleTransformCombinations[i]) &&
        (size == kPossibleTransformCombinationSize ||
         kPossibleTransformCombinations[i][size] == TransformType::kNum)) {
      return kPossibleTransformCombinations[i];
    }
  }
  assert(false);
  return nullptr;
}

// Find a set of TransformType's into the official list and returns a pointer to
// it.
template <typename... Ts>
const TransformType* FindTransformTypes(Ts... args) {
  const TransformType transforms[] = {args...};
  return FindTransformationCombination(transforms, sizeof...(args));
}

void AddConfig(EntropyMode mode, bool use_premultiplied,
               bool red_and_blue_always_zero, float entropy, int effort,
               WP2::Vector<EntropyConfig>* configs) {
  EntropyConfig c;
  c.mode = mode;
  c.entropy = entropy;
  c.config.use_premultiplied = use_premultiplied;

  TransformType transforms[kPossibleTransformCombinationSize];
  uint32_t size = 0;
  if (mode == EntropyMode::kPalette || mode == EntropyMode::kPaletteSpatial) {
    transforms[size++] = TransformType::kColorIndexing;
    red_and_blue_always_zero = true;
  }
  if (mode == EntropyMode::kSubGreen || mode == EntropyMode::kSpatialSubGreen) {
    transforms[size++] = TransformType::kSubtractGreen;
  }
  if (mode == EntropyMode::kSpatial || mode == EntropyMode::kSpatialSubGreen ||
      mode == EntropyMode::kPaletteSpatial) {
    transforms[size++] = TransformType::kPredictor;
    if (effort != 0 && !red_and_blue_always_zero) {
      transforms[size++] = TransformType::kCrossColor;
    }
  }
  c.config.transforms = FindTransformationCombination(transforms, size);
  configs->push_back_no_resize(c);
}

// Analyzes the distribution of colors on the image to estimate the best
// transform configurations.
// Returns at most 'num_configs_max' sorted from best to worst in 'configs'.
template <typename T>
WP2Status AnalyzeEntropyImpl(const T* argb, bool can_use_palette, int effort,
                             uint32_t transform_bits, const Encoder& encoder,
                             uint32_t num_configs_max,
                             WP2::Vector<EntropyConfig>* configs) {
  assert(num_configs_max >= 1);
  const WP2::ArgbBuffer& pic = encoder.pic_;
  const uint32_t stride = pic.stride(), width = pic.width(),
                 height = pic.height();
  const WP2SampleFormat format = pic.format();

  if (can_use_palette && encoder.palette_.Size() <= 16) {
    // In practice, small palettes are better than any other transform.
    WP2_CHECK_ALLOC_OK(configs->reserve(1));
    for (EntropyMode mode :
         {EntropyMode::kPalette, EntropyMode::kPaletteSpatial}) {
      AddConfig(mode, /*use_premultiplied=*/!encoder.config_.exact,
                /*red_and_blue_always_zero=*/true, /*entropy=*/0.f, effort,
                configs);
    }
    return WP2_STATUS_OK;
  }

  // Allocate histogram set with cache_bits = 0.
  const uint32_t hash_bits =
      can_use_palette ? WP2Log2Ceil_k(encoder.palette_.Size()) : 0;
  Histogram histo[kHistoNum];
  // Default histogram length for signed transforms.
  const uint32_t kDefaultLen =
      1 + std::max(MakeIndex(WP2::FormatMax(format, /*channel=*/1)),
                   MakeIndex(-(int16_t)WP2::FormatMax(format, /*channel=*/1)));
  // Define the lengths of the histograms.
  histo[kHistoAlpha].len = WP2::kAlphaMax + 1;
  histo[kHistoAlphaPred].len = kDefaultLen;
  histo[kHistoGreen].len = WP2::FormatMax(format, /*channel=*/2) + 1;
  histo[kHistoGreenPred].len = kDefaultLen;
  histo[kHistoRed].len = WP2::FormatMax(format, /*channel=*/1) + 1;
  histo[kHistoRedPred].len = kDefaultLen;
  histo[kHistoBlue].len = WP2::FormatMax(format, /*channel=*/3) + 1;
  histo[kHistoBluePred].len = kDefaultLen;
  histo[kHistoRedSubGreen].len = kDefaultLen;
  histo[kHistoRedPredSubGreen].len = kDefaultLen;
  histo[kHistoBlueSubGreen].len = kDefaultLen;
  histo[kHistoBluePredSubGreen].len = kDefaultLen;
  histo[kHistoPalette].len = (1 << hash_bits);
  // Define the vector that will contain all the histograms.
  uint32_t histo_len = 0;
  for (const Histogram& h : histo) histo_len += h.len;
  WP2::Vector_u32 histo_raw;
  WP2_CHECK_ALLOC_OK(histo_raw.resize(histo_len));
  // Define the pointers of the histograms.
  histo_len = 0;
  for (Histogram& h : histo) {
    h.data = &histo_raw[histo_len];
    histo_len += h.len;
  }

  const bool can_premultiply =
      (encoder.has_alpha_ && !WP2IsPremultiplied(format));
  WP2_CHECK_ALLOC_OK(configs->reserve(2 * (uint32_t)EntropyMode::kNum));
  for (bool premult : {false, true}) {
    if (!can_premultiply && premult) {
      // No need to test pre-multiply if the format is already pre-multiplied
      // or has no alpha.
      continue;
    }
    // Palette is always pre-multiplied.
    const bool analyze_palette =
        (can_use_palette && (premult || !can_premultiply));
    // Reset histograms.
    std::fill(histo_raw.begin(), histo_raw.end(), 0);

    // Update histograms.
    constexpr T kTransparent[4] = {0, 0, 0, 0};

    const T* prev_row = nullptr;
    const T* curr_row = argb;
    // Skip the first pixel.
    T pix_prev[4] = {curr_row[0], curr_row[1], curr_row[2], curr_row[3]};
    if (premult) {
      Premultiply(pix_prev, pix_prev);
    }
    T pix_premult_prev_row[4];
    T pix_premult[4];
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const T* pix = &curr_row[4 * x];

        const T* p;
        const T* p_prev = pix_prev;
        const T* p_prev_row;

        if (premult) {  // pixels premultiplied by us
          Premultiply(pix, pix_premult);
          if (y > 0) Premultiply(&prev_row[4 * x], pix_premult_prev_row);

          p = pix_premult;
          p_prev_row = (y > 0) ? pix_premult_prev_row : nullptr;
        } else {  // pixels as is
          p = pix;
          p_prev_row = (y > 0) ? &prev_row[4 * x] : nullptr;
          // Zero-out fully transparent pixels.
          if (p[0] == 0) p = kTransparent;
          if (p_prev[0] == 0) p_prev = kTransparent;
          if (p_prev_row != nullptr && p_prev_row[0] == 0) {
            p_prev_row = kTransparent;
          }
        }

        // Compute the difference between the two pixels.
        int16_t p_diff[4];
        for (uint32_t c = 0; c < 4; ++c) {
          p_diff[c] = (int16_t)p[c] - (int16_t)p_prev[c];
        }

        std::copy(p, p + 4, pix_prev);

        if ((p_diff[0] == 0 && p_diff[1] == 0 && p_diff[2] == 0 &&
             p_diff[3] == 0) ||
            (p_prev_row != nullptr &&
             memcmp(p, p_prev_row, 4 * sizeof(T)) == 0)) {
          continue;
        }

        AddSingleUnsigned(p, &histo[kHistoAlpha], &histo[kHistoRed],
                          &histo[kHistoGreen], &histo[kHistoBlue]);
        AddSingleSigned(p_diff, &histo[kHistoAlphaPred], &histo[kHistoRedPred],
                        &histo[kHistoGreenPred], &histo[kHistoBluePred]);
        AddSingleSubGreen(p, &histo[kHistoRedSubGreen],
                          &histo[kHistoBlueSubGreen]);
        AddSingleSubGreen(p_diff, &histo[kHistoRedPredSubGreen],
                          &histo[kHistoBluePredSubGreen]);

        // Palettes are always premultiplied. It doesn't make sense to look at
        // entropy for unmultiplied palettes.
        if (analyze_palette) {
          // Approximate the palette by the entropy of the multiplicative hash.
          const uint32_t hash = WP2::encoding::HashPix(p, hash_bits);
          assert((int)hash < histo[kHistoPalette].len);
          ++histo[kHistoPalette].data[hash];
        }
      }
      prev_row = curr_row;
      curr_row = (const T*)((const uint8_t*)curr_row + stride);
    }
    // Compute the entropy of the histograms.
    float entropy[kHistoNum];

    // Let's add one zero to the predicted histograms. The zeros are removed
    // too efficiently by the pix_diff == 0 comparison, at least one of the
    // zeros is likely to exist.
    ++histo[kHistoRedPredSubGreen].data[0];
    ++histo[kHistoBluePredSubGreen].data[0];
    ++histo[kHistoRedPred].data[0];
    ++histo[kHistoGreenPred].data[0];
    ++histo[kHistoBluePred].data[0];
    ++histo[kHistoAlphaPred].data[0];

    for (uint32_t j = 0; j < kHistoNum; ++j) {
      entropy[j] = WP2::ANSCountsCost(&histo[j].data[0], histo[j].len);
    }

    // Compute the entropy of the different entropy modes.
    AddConfig(EntropyMode::kDirect, premult,
              entropy[kHistoRed] + entropy[kHistoBlue] == 0.f,
              entropy[kHistoAlpha] + entropy[kHistoRed] + entropy[kHistoGreen] +
                  entropy[kHistoBlue],
              effort, configs);
    AddConfig(EntropyMode::kSpatial, premult,
              entropy[kHistoRedPred] + entropy[kHistoBluePred] == 0.f,
              entropy[kHistoAlphaPred] + entropy[kHistoRedPred] +
                  entropy[kHistoGreenPred] + entropy[kHistoBluePred],
              effort, configs);
    AddConfig(EntropyMode::kSubGreen, premult,
              entropy[kHistoRedSubGreen] + entropy[kHistoBlueSubGreen] == 0.f,
              entropy[kHistoAlpha] + entropy[kHistoRedSubGreen] +
                  entropy[kHistoGreen] + entropy[kHistoBlueSubGreen],
              effort, configs);
    AddConfig(
        EntropyMode::kSpatialSubGreen, premult,
        entropy[kHistoRedPredSubGreen] + entropy[kHistoBluePredSubGreen] == 0.f,
        entropy[kHistoAlphaPred] + entropy[kHistoRedPredSubGreen] +
            entropy[kHistoGreenPred] + entropy[kHistoBluePredSubGreen],
        effort, configs);

    // Palettes are always premultiplied. It doesn't make sense to look at
    // entropy for unmultiplied palettes, except if the format does not allow
    // it.
    if (analyze_palette) {
      AddConfig(EntropyMode::kPalette, premult,
                /*red_and_blue_always_zero=*/true, entropy[kHistoPalette],
                effort, configs);
    }
  }
  for (const EntropyConfig& c : *configs) {
    (void)c;
    if (!can_use_palette) assert(c.mode != EntropyMode::kPalette);
    if (!can_premultiply) assert(!c.config.use_premultiplied);
  }
  // Add extra bit costs.
  const uint32_t transform_image_size = SubSampleSize(width, transform_bits) *
                                        SubSampleSize(height, transform_bits);
  for (EntropyConfig& c : *configs) {
    for (uint32_t i = 0; i < c.config.GetNumTransforms(); ++i) {
      if (c.config.transforms[i] == TransformType::kPredictor) {
        // When including transforms, there is an overhead in bits from
        // storing them. This overhead is small but matters for small images.
        // For spatial, there are 14 transformations.
        c.entropy += transform_image_size * WP2Log2(14);
      } else if (c.config.transforms[i] == TransformType::kCrossColor) {
        // For color transforms, estimating the cross color transform entropy is
        // hard and easy to overshoot. Adding the cost of the transform image is
        // actually worsening results so we are not artificially adding it like
        // we do for kPredictor.
      } else if (c.config.transforms[i] == TransformType::kColorIndexing) {
        // For palettes, add the cost of storing the palette.
        c.entropy += encoder.palette_.GetCost();
      }
    }
  }
  // Sort the configs.
  std::stable_sort(configs->begin(), configs->end(),
                   [](const EntropyConfig& c1, const EntropyConfig& c2) {
                     return (c1.entropy < c2.entropy);
                   });
  // Only keep the top num_configs_max ones.
  configs->resize_no_check(
      std::min((uint32_t)configs->size(), num_configs_max));
  // In case we have a kColorIndexing transform, also add kColorIndexing +
  // kPredictor.
  for (const EntropyConfig& c : *configs) {
    if (c.config.transforms[0] == TransformType::kColorIndexing &&
        c.config.GetNumTransforms() == 1) {
      AddConfig(EntropyMode::kPaletteSpatial, c.config.use_premultiplied,
                /*red_and_blue_always_zero=*/true, c.entropy, effort, configs);
      break;
    }
  }

  return WP2_STATUS_OK;
}

WP2Status AnalyzeEntropy(bool can_use_palette, int effort,
                         uint32_t transform_bits, const Encoder& encoder,
                         uint32_t num_configs_max,
                         WP2::Vector<EntropyConfig>* configs) {
  if (WP2Formatbpc(encoder.pic_.format()) <= 8) {
    WP2_CHECK_STATUS(AnalyzeEntropyImpl(encoder.pic_.GetRow8(0),
                                        can_use_palette, effort, transform_bits,
                                        encoder, num_configs_max, configs));
  } else {
    WP2_CHECK_STATUS(AnalyzeEntropyImpl(encoder.pic_.GetRow16(0),
                                        can_use_palette, effort, transform_bits,
                                        encoder, num_configs_max, configs));
  }

  return WP2_STATUS_OK;
}

static int GetHistoBits(int effort, bool use_palette, int width, int height) {
  // Make tile size a function of encoding method (Range: 0 to 6).
  // TODO(vrabaud) use rounding. We leave it as is so that for the default
  // value of 5, it matches the old default value of 3.
  uint32_t histo_bits = (use_palette ? 9. : 7.) - effort * 6. / 9.;
  while (true) {
    const uint32_t huff_image_size =
        SubSampleSize(width, histo_bits) * SubSampleSize(height, histo_bits);
    if (huff_image_size <= kMaxHistogramImageSize) break;
    ++histo_bits;
  }
  return WP2::Clamp(histo_bits, kHistogramBitsMin, kHistogramBitsMax);
}

uint32_t GetTransformBits(uint32_t effort, uint32_t histo_bits) {
  const uint32_t max_transform_bits = (effort < 5) ? 6 : (effort > 5) ? 4 : 5;
  const uint32_t res =
      (histo_bits > max_transform_bits) ? max_transform_bits : histo_bits;
  assert(res <= kTransformBitsMax);
  return res;
}

}  // namespace

// Fills in 'crunch_configs' with a set of transforms and LZ77 variants to try
// based on heuristics. Also initializes the palette in enc->palette_ and
// sets red_and_blue_always_zero to true if the red and blue components are
// always zero after applying the best transform.
// TODO(maryla): the logic for red_and_blue_always_zero seems broken: it's only
// valid if there's only one transform (effort != 9)
WP2Status EncoderAnalyze(const WP2::ProgressRange& progress,
                         Encoder* const encoder,
                         WP2::Vector<CrunchConfig>* const configs) {
  const WP2::ArgbBuffer& pic = encoder->pic_;
  const int effort = encoder->config_.effort;
  uint32_t n_lz77s;
  assert(!pic.IsEmpty());

  const WP2::ProgressRange palette_progress(progress, 0.5);
  const WP2::ProgressRange entropy_progress(progress, 0.5);

  WP2_CHECK_STATUS(encoder->palette_.AnalyzeAndCreate(pic));
  WP2_CHECK_STATUS(encoder->palette_.FindBestMethod(
      pic, palette_progress, Palette::Sorting::kMinimalSize, effort, encoder));

  // Do not use a palette when there is only one color as ANS will optimize it
  // anyway.
  const bool can_use_palette = (encoder->palette_.Size() > 1);
  const bool can_premultiply =
      (!encoder->config_.exact && encoder->has_alpha_ &&
       !WP2IsPremultiplied(encoder->pic_.format()));

  // Try LZW within the same palette size range condition as Group4.
  if (encoder->palette_.Size() >= 2 && encoder->palette_.Size() < 256) {
    WP2_CHECK_ALLOC_OK(configs->resize(configs->size() + 1));
    configs->back().transforms = FindTransformTypes(TransformType::kLZW);
    configs->back().use_premultiplied = can_premultiply;
  }

  // Do not use Group4 for big palettes for CPU reasons and also because it is
  // less likely that Group4 will work.
  if (encoder->palette_.Size() >= 2 && encoder->palette_.Size() < 256) {
    // Try group4 for images with a palette.
    WP2_CHECK_ALLOC_OK(configs->resize(configs->size() + 1));
    configs->back().transforms = FindTransformTypes(TransformType::kGroup4);
    configs->back().group4_use_move_to_front = false;
    configs->back().use_premultiplied = false;

    if (encoder->palette_.Size() > 3 && effort > 0) {
      WP2_CHECK_ALLOC_OK(configs->resize(configs->size() + 1));
      // Copy the previous config.
      configs->back() = configs->at(configs->size() - 2);
      // Try using a MoveToFrontCache for images with more than 3 colors.
      configs->back().group4_use_move_to_front = true;
      configs->back().use_premultiplied = false;
    }
  }
  WP2::Vector<EntropyConfig> entropy_configs;
  WP2_CHECK_ALLOC_OK(
      entropy_configs.reserve(kPossibleTransformCombinationsNum));
  if (effort == 0) {
    if (configs->size() != 1) {
      // AnalyzeEntropy is somewhat slow. 'red_and_blue_always_zero' is unused
      // for effort == 0.
      configs->clear();
      AddConfig(can_use_palette ? EntropyMode::kPalette
                                : EntropyMode::kSpatialSubGreen,
                can_premultiply, /*red_and_blue_always_zero=*/false,
                /*entropy=*/0.f, effort, &entropy_configs);
    }
    n_lz77s = 1;
  } else {
    // Try out multiple LZ77 on images with few colors.
    n_lz77s = (can_use_palette && encoder->palette_.Size() <= 16) ? 2 : 1;
    // For effort 5, we ask for 2 so that in case the best one is Palette, we
    // still try a normal configuration because color cache can sometimes be
    // better than palette.
    const uint32_t num_configs_max = (effort < 5)    ? 1
                                     : (effort == 5) ? 2
                                     : (effort > 5 && effort < 9)
                                         ? effort - 4
                                         : std::numeric_limits<uint32_t>::max();
    // TODO(vrabaud) use proper bits for analysis: we use can_use_palette even
    //               though we do not use it for most transforms like spatial.
    //               Using it instead of setting it to false gives 0.1% better
    //               compression ...
    const uint32_t histo_bits =
        GetHistoBits(effort, can_use_palette, pic.width(), pic.height());
    const uint32_t transform_bits = GetTransformBits(effort, histo_bits);
    WP2_CHECK_STATUS(AnalyzeEntropy(can_use_palette, effort, transform_bits,
                                    *encoder, num_configs_max,
                                    &entropy_configs));
    if (effort == 5 && entropy_configs[0].mode != EntropyMode::kPalette) {
      // When we have palette as first transform, we also check the second one
      // as it can be better. If the first one is not palette, only focus on the
      // first one.
      entropy_configs.resize_no_check(1);
    }
  }
  WP2_CHECK_STATUS(entropy_progress.AdvanceBy(1.));
  // Remove the internal info.
  for (const EntropyConfig& c : entropy_configs) {
    WP2_CHECK_ALLOC_OK(configs->push_back(c.config));
  }
  // Use the minimal-size method for palette storage by default.
  uint32_t configs_size = configs->size();
  for (uint32_t i = 0; i < configs_size; ++i) {
    for (uint32_t t = 0; t < kPossibleTransformCombinationSize; ++t) {
      if ((*configs)[i].transforms[t] == TransformType::kColorIndexing) {
        (*configs)[i].palette_sorting_type = Palette::Sorting::kMinimalSize;
        if (effort == 9) {
          // Try more methods for palette sorting.
          WP2_CHECK_ALLOC_OK(configs->push_back((*configs)[i]));
          configs->back().palette_sorting_type = Palette::Sorting::kLuminance;
          WP2_CHECK_ALLOC_OK(configs->push_back((*configs)[i]));
          configs->back().palette_sorting_type =
              Palette::Sorting::kModifiedZeng;
        }
        break;
      }
    }
  }
  // Configs are ordered with the smallest entropies first to favor early exits
  // later. Still, we slightly change the order so that increasingly including
  // sets of transforms are clumped together to favor transform caching later.
  for (auto iter = configs->begin(); iter != configs->end();) {
    // Copy the current config: it is the best one so far (entropy-wise), and
    // we are looking for other configs with transform sequences
    // included/including its own ones (we have to copy the config as 'iter'
    // might be incremented).
    CrunchConfig config = *iter;
    if (config.GetNumTransforms() == 0) {
      ++iter;
      continue;
    }
    // Find configs with sets of transforms contained or containing the current
    // ones with increasing size.
    for (uint32_t i = 1; i <= kPossibleTransformCombinationSize; ++i) {
      // Look at the following configurations.
      for (auto iter_next = iter; iter_next != configs->end(); ++iter_next) {
        if (iter_next->GetNumTransforms() == i &&
            (config.ShouldBeCachedFor(*iter_next) ||
             iter_next->ShouldBeCachedFor(config))) {
          // If the set of transforms contains the currently studied one, have
          // it be the new studied one.
          if (i > config.GetNumTransforms()) config = *iter;
          std::swap(*iter, *iter_next);
          ++iter;
          break;
        }
      }
    }
  }
  // TODO(vrabaud): replace the decision to be based on an actual estimate
  // of entropy, or even spatial variance of entropy.
  for (CrunchConfig& c : *configs) {
    const bool use_palette = (c.GetNumTransforms() > 0 &&
                              c.transforms[0] == TransformType::kColorIndexing);
    c.histo_bits = GetHistoBits(effort, use_palette, pic.width(), pic.height());
    c.transform_bits = GetTransformBits(effort, c.histo_bits);
  }
  // Fill in the different LZ77s.
  assert(n_lz77s <= kCrunchLZ77Max);
  for (CrunchConfig& c : *configs) {
    for (uint32_t i = 0; i < n_lz77s; ++i) {
      c.lz77s_types_to_try[i] = (i == 0) ? kLZ77Standard | kLZ77RLE : kLZ77Box;
    }
    c.lz77s_types_to_try_size = n_lz77s;
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2L
