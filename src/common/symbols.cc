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
// Common code dealing with symbols.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/common/symbols.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/common/color_precision.h"
#include "src/common/constants.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/block_size_io.h"
#include "src/common/lossy/predictor.h"
#include "src/common/lossy/rnd_mtx.h"
#include "src/common/lossy/transforms.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/utils/ans.h"
#include "src/utils/ans_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

//------------------------------------------------------------------------------

namespace WP2 {

const SymbolsInfo::ClusterInfo SymbolsInfo::kDefaultClusterInfo = {};

WP2Status SymbolsInfo::CopyFrom(const SymbolsInfo& other) {
  num_symbols_ = other.num_symbols_;
  std::copy(other.symbols_info_, other.symbols_info_ + other.Size(),
            symbols_info_);
  WP2_CHECK_ALLOC_OK(cluster_info_.resize(other.cluster_info_.size()));
  std::copy(other.cluster_info_.begin(), other.cluster_info_.end(),
            cluster_info_.begin());
  next_cluster_info_index_ = other.next_cluster_info_index_;
  return WP2_STATUS_OK;
}

void SymbolsInfo::SetUnused(uint32_t sym, uint32_t num_clusters) {
  SetInfo(sym, /*min=*/kInvalidBound, /*max=*/kInvalidBound, num_clusters,
          StorageMethod::kUnused);
}

void SymbolsInfo::SetInfo(uint32_t sym, int32_t min, int32_t max,
                          uint32_t num_clusters, StorageMethod method,
                          bool probably_geometric) {
  assert(sym < kSymbolNumMax);
  // Used as a guarantee that Init() was called.
  assert(method == StorageMethod::kUnused || num_clusters > 0);
  num_symbols_ = std::max(num_symbols_, sym + 1);
  SymbolInfo* const info = GetSymbolInfo(sym);
  info->storage_method = method;
  info->num_clusters = num_clusters;
  info->probably_geometric = probably_geometric;
  SetMinMax(sym, min, max);
}

bool SymbolsInfo::SymbolInfo::MinMaxIsValid(int32_t new_min,
                                            int32_t new_max) const {
  if (new_min == kInvalidBound && new_max == kInvalidBound) {
    return true;
  }
  assert(new_min <= new_max);
  const int32_t new_range = new_max - new_min + 1;
  assert(new_range >= 0 && new_range < (1 << kANSMaxRangeBits));
  switch (storage_method) {
    case SymbolsInfo::StorageMethod::kAuto:
      return ((uint32_t)new_range <= kANSMaxRange);
    case SymbolsInfo::StorageMethod::kAdaptiveBit:
      return (new_range == 2);
    case SymbolsInfo::StorageMethod::kAdaptiveSym:
    case SymbolsInfo::StorageMethod::kAdaptiveWithAutoSpeed:
      return (new_range <= (int32_t)APROBA_MAX_SYMBOL);
    case SymbolsInfo::StorageMethod::kUnused:
      return true;
  }
  assert(false);
  return true;
}

WP2Status SymbolsInfo::SetMinMax(uint32_t sym, uint32_t cluster, int32_t min,
                                 int32_t max) {
  assert(sym < kSymbolNumMax);
  assert(cluster < GetSymbolInfo(sym)->num_clusters);
  assert(GetSymbolInfo(sym)->MinMaxIsValid(min, max));
  ClusterInfo* const info = GetOrMakeClusterInfo(sym, cluster);
  WP2_CHECK_ALLOC_OK(info != nullptr);
  info->min = min;
  info->max = max;
  return WP2_STATUS_OK;
}

void SymbolsInfo::SetMinMax(uint32_t sym, int32_t min, int32_t max) {
  SymbolInfo* const info = GetSymbolInfo(sym);
  // Check Init() was called.
  assert(IsUnused(sym) || info->num_clusters > 0);
  assert(info->MinMaxIsValid(min, max));
  info->min = min;
  info->max = max;
  for (uint32_t c = 0; c < info->num_clusters; ++c) {
    // Should not set a global range after setting per-cluster range.
    assert(GetClusterInfo(sym, c).min == kInvalidBound &&
           GetClusterInfo(sym, c).max == kInvalidBound);
  }
}

uint32_t SymbolsInfo::MaxRangeSum() const {
  uint32_t sum = 0;
  for (uint32_t sym = 0; sym < num_symbols_; ++sym) sum += GetMaxRange(sym);
  return sum;
}

uint32_t SymbolsInfo::RangeSum() const {
  uint32_t sum = 0;
  for (uint32_t sym = 0; sym < num_symbols_; ++sym) {
    for (uint32_t c = 0; c < NumClusters(sym); ++c) sum += Range(sym, c);
  }
  return sum;
}

uint32_t SymbolsInfo::GetMaxRange() const {
  const auto iter =
      std::max_element(symbols_info_, symbols_info_ + Size(),
                       [](const SymbolInfo& lhs, const SymbolInfo& rhs) {
                         return (lhs.max - lhs.min < rhs.max - rhs.min);
                       });
  return iter->max - iter->min + 1;
}

uint32_t SymbolsInfo::GetMaxRange(uint32_t sym) const {
  uint32_t max_range = 0;
  for (uint32_t c = 0; c < NumClusters(sym); ++c) {
    max_range = std::max(max_range, Range(sym, c));
  }
  return max_range;
}

WP2Status SymbolsInfo::SetInitialCDF(uint32_t sym, uint32_t cluster,
                                     const uint16_t* const cdf,
                                     uint16_t max_proba) {
  assert(IsAdaptiveMethod(sym));
  ClusterInfo* const info = GetOrMakeClusterInfo(sym, cluster);
  WP2_CHECK_ALLOC_OK(info != nullptr);
  info->cdf = cdf;
  info->max_proba = max_proba;
  return WP2_STATUS_OK;
}

void SymbolsInfo::GetInitialCDF(uint32_t sym, uint32_t cluster,
                                const uint16_t** const cdf,
                                uint16_t* const max_proba) const {
  assert(IsAdaptiveMethod(sym));
  *cdf = GetClusterInfo(sym, cluster).cdf;
  *max_proba = GetClusterInfo(sym, cluster).max_proba;
}

void SymbolsInfo::SetClusters(uint32_t sym, uint32_t num_clusters) {
  assert(sym < kSymbolNumMax);
  // Check Init() was called.
  assert(IsUnused(sym) || GetSymbolInfo(sym)->num_clusters > 0);
  assert(num_clusters > 0);
  GetSymbolInfo(sym)->num_clusters = num_clusters;
}

WP2Status SymbolsInfo::SetStartingProba(uint32_t sym, uint32_t cluster,
                                        uint32_t p0, uint32_t p1) {
  assert(p0 > 0);
  assert(p1 > 0);
  assert(Method(sym) == StorageMethod::kAdaptiveBit);
  ClusterInfo* const info = GetOrMakeClusterInfo(sym, cluster);
  WP2_CHECK_ALLOC_OK(info != nullptr);
  info->p0 = p0;
  info->p1 = p1;
  return WP2_STATUS_OK;
}

SymbolsInfo::ClusterInfo* SymbolsInfo::GetOrMakeClusterInfo(uint32_t sym,
                                                            uint32_t cluster) {
  SymbolInfo* const info = GetSymbolInfo(sym);
  assert(cluster < info->num_clusters);
  if (info->cluster_info_index == SymbolInfo::kNoPerClusterInfo) {
    info->cluster_info_index = next_cluster_info_index_;
    next_cluster_info_index_ += info->num_clusters;
    if (!cluster_info_.resize(next_cluster_info_index_)) {
      return nullptr;
    }
  }
  return &cluster_info_[info->cluster_info_index + cluster];
}

const SymbolsInfo::ClusterInfo& SymbolsInfo::GetClusterInfo(
    uint32_t sym, uint32_t cluster) const {
  assert(cluster < NumClusters(sym));
  const SymbolInfo& info = GetSymbolInfo(sym);
  if (info.cluster_info_index == SymbolInfo::kNoPerClusterInfo) {
    return kDefaultClusterInfo;
  }
  return cluster_info_[info.cluster_info_index + cluster];
}

//------------------------------------------------------------------------------

WP2Status SymbolsInfo::InitLossy(uint32_t num_segments,
                                 PartitionSet partition_set, bool has_alpha,
                                 bool use_aom_coeffs, bool use_splits) {
  const uint32_t num_channels = (has_alpha ? 4 : 3);

  SetInfo(kSymbolModeY, /*min=*/0, /*max=*/kYPredModeNum - 1,
          /*num_clusters=*/1, StorageMethod::kAdaptiveSym);
  SetInfo(kSymbolModeUV, /*min=*/0, /*max=*/kUVPredModeNum - 1,
          /*num_clusters=*/1, StorageMethod::kAdaptiveWithAutoSpeed);
  SetInfo(kSymbolSegmentId, /*min=*/0, /*max=*/num_segments - 1,
          /*num_clusters=*/1, StorageMethod::kAdaptiveWithAutoSpeed,
          /*probably_geometric=*/true);
  SetInfo(kSymbolResNumZeros, /*min=*/0, /*max=*/kResidualCons0Max,
          /*num_clusters=*/kNumResidualStats * num_channels,
          StorageMethod::kAuto);
  SetInfo(kSymbolBits0, /*min=*/0, /*max=*/kResidual1Max,
          /*num_clusters=*/kNumResidualStats * num_channels,
          StorageMethod::kAuto);
  SetInfo(kSymbolBits1, /*min=*/0, /*max=*/kMaxCoeffBits,
          /*num_clusters=*/kNumResidualStats * num_channels,
          StorageMethod::kAuto);
  // 1 bit is added by the fact that we subtract the predicted value from
  // the actual value, and 1 bit is added by the 2D transform.
  // Finally, one bit is added for the sign.
  SetInfo(kSymbolDC, /*min=*/0, /*max=*/2 * kMaxDcValue,
          /*num_clusters=*/num_channels, StorageMethod::kAuto,
          /*probably_geometric=*/true);
  SetInfo(kSymbolRndMtx0, /*min=*/0, /*max=*/kMaxNumRndMtx - 1,
          /*num_clusters=*/1, StorageMethod::kAuto);

  SetInfo(kSymbolTransform, /*min=*/0, /*max=*/kNumTransformPairs - 1,
          /*num_clusters=*/1, StorageMethod::kAdaptiveWithAutoSpeed);
  SetInfo(kSymbolBlockAlphaMode, /*min=*/0, /*max=*/2,
          /*num_clusters=*/1, StorageMethod::kAuto);
  SetInfo(kSymbolSplitTransform, /*min=*/0, /*max=*/1, /*num_clusters=*/1,
          StorageMethod::kAdaptiveWithAutoSpeed);
  SetInfo(kSymbolYuv420, /*min=*/0, /*max=*/1, /*num_clusters=*/1,
          StorageMethod::kAdaptiveWithAutoSpeed);
  SetInfo(kSymbolUseRandomMatrix, /*min=*/0, /*max=*/1,
          /*num_clusters=*/1, StorageMethod::kAdaptiveBit);

  constexpr static uint32_t kMaxAbsValueDeltaA =
      (1 << (SignalingCflPredictor::kIOBits - 1));
  SetInfo(kSymbolCflSlope,
          /*min=*/-kMaxAbsValueDeltaA, /*max=*/kMaxAbsValueDeltaA,
          /*num_clusters=*/1, StorageMethod::kAuto,
          /*probably_geometric=*/true);

  SetInfo(kSymbolHasCoeffs, /*min=*/0, /*max=*/1, /*num_clusters=*/num_channels,
          StorageMethod::kAdaptiveWithAutoSpeed);
  if (has_alpha) {
    SetInfo(kSymbolModeA, /*min=*/0, /*max=*/kAPredModeNum - 1,
            /*num_clusters=*/1, StorageMethod::kAdaptiveSym);
    SetInfo(kSymbolEncodingMethodA, /*min=*/0,
            /*max=*/(int)EncodingMethod::kNumMethod - 1, /*num_clusters=*/1,
            StorageMethod::kAdaptiveWithAutoSpeed);
  } else {
    SetUnused(kSymbolModeA, /*num_clusters=*/1);
    SetUnused(kSymbolEncodingMethodA, /*num_clusters=*/1);
  }
  SetInfo(kSymbolEncodingMethod, /*min=*/0,
          /*max=*/(int32_t)EncodingMethod::kNumMethod - 1,
          /*num_clusters=*/3 * num_segments,
          StorageMethod::kAdaptiveWithAutoSpeed);

  if (use_splits) {
    SetInfo(kSymbolBlockSize, /*min=*/0, /*max=*/kMaxSplitPatterns - 1,
            /*num_clusters=*/GetNumBlockSizes(partition_set),
            StorageMethod::kAdaptiveSym);
    const BlockSize* const sizes = GetBlockSizes(partition_set);
    uint32_t i = 0;
    for (const BlockSize* size = sizes; *size != BLK_LAST; ++size, ++i) {
      WP2_CHECK_STATUS(SetMinMax(
          kSymbolBlockSize, /*cluster=*/i,
          /*min=*/0, /*max=*/GetNumSplitPatterns(partition_set, *size) - 1));
    }
  } else {
    const uint32_t max_num_sizes = GetNumBlockSizes(partition_set, BLK_32x32);
    SetInfo(kSymbolBlockSize, /*min=*/0, /*max=*/max_num_sizes - 1,
            /*num_clusters=*/GetNumUniqueBounds(partition_set),
            (max_num_sizes > APROBA_MAX_SYMBOL) ? StorageMethod::kAuto
                                                : StorageMethod::kAdaptiveSym);
    for (uint32_t i = 0; i < GetNumUniqueBounds(partition_set); ++i) {
      const BlockSize bounds = GetUniqueBounds(partition_set, i);
      const uint32_t num_sizes = GetNumBlockSizes(partition_set, bounds);
      assert(num_sizes <= max_num_sizes);
      WP2_CHECK_STATUS(SetMinMax(kSymbolBlockSize, /*cluster=*/i,
                                 /*min=*/0, /*max=*/num_sizes - 1));
    }
  }

  SetInfo(kSymbolResidualUseBounds, /*min=*/0, /*max=*/1,
          /*num_clusters=*/kResidualClusters, StorageMethod::kAdaptiveBit);
  SetInfo(kSymbolResidualBound1IsX, /*min=*/0, /*max=*/1,
          /*num_clusters=*/kResidualClusters, StorageMethod::kAdaptiveBit);
  SetInfo(kSymbolResidualUseBound2, /*min=*/0, /*max=*/1,
          /*num_clusters=*/kResidualClusters, StorageMethod::kAdaptiveBit);

  // Non-AOM coeffs are always used for UV so they need to be initialized.
  for (uint32_t sym : kSymbolsForCoeffs) {
    // Y/UV, per residual method, per block size, per sector.
    const uint32_t num_clusters = kResidualClusters * ResidualIO::kNumSectors;
    SetInfo(sym, /*min=*/0, /*max=*/1, num_clusters,
            StorageMethod::kAdaptiveBit);
  }

  // Initialize coefficient probabilities.
  WP2_CHECK_STATUS(ResidualIO::InitializeInfo(has_alpha, use_aom_coeffs, this));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void WritePrefixCode(int32_t value, int32_t min, int32_t max,
                     uint32_t prefix_size, ANSEncBase* const enc,
                     WP2_OPT_LABEL) {
  ANSDebugPrefix ans_prefix(enc, label);

  const PrefixCode prefix_code(value - min, prefix_size);
  uint32_t range = max - min + 1;
  const PrefixCode max_prefix_code(range - 1, prefix_size);

  enc->PutRValue(prefix_code.prefix, max_prefix_code.prefix + 1, "prefix_code");
  if (prefix_code.extra_bits_num > 0) {
    // If the last bit interval is truncated, go with ranges.
    if (prefix_code.prefix == max_prefix_code.prefix) {
      assert(PrefixCode::Merge(prefix_code.prefix, prefix_size, 0) < range);
      range -= PrefixCode::Merge(prefix_code.prefix, prefix_size, 0);
      enc->PutRValue(prefix_code.extra_bits_value, range, "extra_bits_value");
    } else {
      enc->PutUValue(prefix_code.extra_bits_value, prefix_code.extra_bits_num,
                     "extra_bits_value");
    }
  }
}

int32_t ReadPrefixCode(int32_t min, int32_t max, uint32_t prefix_size,
                       ANSDec* const dec, WP2_OPT_LABEL) {
  ANSDebugPrefix ans_prefix(dec, label);

  uint32_t range = max - min + 1;
  const PrefixCode max_prefix_code(range - 1, prefix_size);

  const uint32_t prefix =
      dec->ReadRValue(max_prefix_code.prefix + 1, "prefix_code");
  const uint32_t extra_bits = PrefixCode::NumExtraBits(prefix, prefix_size);
  uint32_t extra_bits_value;
  if (extra_bits == 0) {
    extra_bits_value = 0;
  } else {
    // Use ranges if we are at the last interval.
    if (prefix == max_prefix_code.prefix) {
      assert(PrefixCode::Merge(prefix, prefix_size, 0) < range);
      range -= PrefixCode::Merge(prefix, prefix_size, 0);
      extra_bits_value = dec->ReadRValue(range, "extra_bits_value");
    } else {
      extra_bits_value = dec->ReadUValue(extra_bits, "extra_bits_value");
    }
  }
  return min + PrefixCode::Merge(prefix, prefix_size, extra_bits_value);
}

}  // namespace WP2

//------------------------------------------------------------------------------

namespace WP2L {

void InitGroup4(uint32_t width, uint32_t num_colors,
                WP2::SymbolsInfo* const info) {
  constexpr WP2::SymbolsInfo::StorageMethod storage_method =
      WP2::SymbolsInfo::StorageMethod::kAuto;
  info->SetInfo(kSymbolG4Type, /*min=*/0, /*max=*/2, /*num_clusters=*/1,
                storage_method);
  info->SetInfo(kSymbolG4HorizontalDist, /*min=*/0, /*max=*/width,
                /*num_clusters=*/2, storage_method);
  info->SetInfo(kSymbolG4VerticalDist, /*min=*/0, /*max=*/2 * kGroup4Window,
                /*num_clusters=*/1, storage_method);
  if (num_colors > 2) {
    info->SetInfo(kSymbolG4ColorChange, /*min=*/0, /*max=*/1,
                  /*num_clusters=*/1, storage_method);
    // Minus 2 because the new color cannot be the same as the current color or
    // the expected color.
    info->SetInfo(kSymbolG4NewColor, /*min=*/0, /*max=*/num_colors - 3,
                  /*num_clusters=*/1, storage_method);
  }
}

LosslessSymbolsInfo::LosslessSymbolsInfo() : sample_format_(WP2_FORMAT_NUM) {
  for (uint32_t i = 0; i < kSymbolNum; ++i) SetUnused(i);
}

void LosslessSymbolsInfo::Init(uint32_t num_pixels, bool has_alpha,
                               WP2SampleFormat sample_format) {
  num_pixels_ = num_pixels;
  sample_format_ = sample_format;
  constexpr StorageMethod storage_method = StorageMethod::kAuto;
  constexpr uint32_t num_clusters = 1u;
  SetInfo(kSymbolType, /*min=*/0, /*max=*/kSymbolTypeNum - 1, num_clusters,
          storage_method);
  if (has_alpha) {
    SetInfo(kSymbolA, /*min=*/0, /*max=*/WP2::FormatMax(sample_format, 0),
            num_clusters, storage_method);
  } else {
    SetUnused(kSymbolA);
  }
  SetInfo(kSymbolR, /*min=*/0, /*max=*/WP2::FormatMax(sample_format, 1),
          num_clusters, storage_method);
  SetInfo(kSymbolG, /*min=*/0, /*max=*/WP2::FormatMax(sample_format, 2),
          num_clusters, storage_method);
  SetInfo(kSymbolB, /*min=*/0, /*max=*/WP2::FormatMax(sample_format, 3),
          num_clusters, storage_method);
  SetUnused(kSymbolSubMode);

  InitNonARGB(num_pixels);
}

void LosslessSymbolsInfo::InitNonARGB(uint32_t num_pixels) {
  if (num_pixels == 1) {
    // Disable all symbol types for 1 x 1 images.
    SetInfo(kSymbolType, /*min=*/0, /*max=*/0, NumClusters(kSymbolType),
            StorageMethod::kAuto);
    SetUnused(kSymbolDist);
    SetUnused(kSymbolLen);
  } else {
    constexpr StorageMethod storage_method = StorageMethod::kAuto;
    constexpr uint32_t num_clusters = 1u;
    // Distance is at most num_pixels - 1 but we have to add
    // kCodeToPlaneCodes - 1 to convert to code.
    WP2::PrefixCode prefix_code_max(num_pixels - 1 + kCodeToPlaneCodes - 1,
                                    kLZ77PrefixCodeSize);
    SetInfo(kSymbolDist, /*min=*/0, prefix_code_max.prefix, num_clusters,
            storage_method);
    // Length is at most num_pixels - 1 as it has to fit in the image.
    // By format definition, it is also <= kLZ77LengthEscape.
    const uint32_t max_length = std::min(kLZ77LengthEscape, num_pixels - 1);
    // If we have a chance to escape, we might have a length of 0 as the size of
    // a second chunk.
    const uint32_t min_length =
        (max_length < kLZ77LengthEscape) ? kLZ77LengthMin : 0u;
    prefix_code_max =
        WP2::PrefixCode(max_length - min_length, kLZ77PrefixCodeSize);
    SetInfo(kSymbolLen, /*min=*/0, prefix_code_max.prefix, num_clusters,
            storage_method);
  }
  SetCacheRange(0, /*has_segment_cache=*/false);
}

void LosslessSymbolsInfo::InitAsLabelImage(uint32_t num_pixels,
                                           uint32_t num_labels) {
  num_pixels_ = num_pixels;
  SetNumClusters(1);
  SetUnused(kSymbolA);
  SetUnused(kSymbolR);
  SetInfo(kSymbolG, /*min=*/0, /*max=*/num_labels - 1, /*num_clusters=*/1,
          StorageMethod::kAuto);
  SetUnused(kSymbolB);
  InitNonARGB(num_pixels);
}

void LosslessSymbolsInfo::InitAsCrossColorImage(uint32_t num_pixels) {
  num_pixels_ = num_pixels;
  SetNumClusters(1);
  SetUnused(kSymbolA);
  SetInfo(kSymbolR, /*min=*/-(int32_t)kRedToBlueMax, /*max=*/kRedToBlueMax,
          /*num_clusters=*/1, StorageMethod::kAuto);
  SetInfo(kSymbolG, /*min=*/-(int32_t)kGreenToBlueMax,
          /*max=*/kGreenToBlueMax, /*num_clusters=*/1, StorageMethod::kAuto);
  SetInfo(kSymbolB, /*min=*/-(int32_t)kGreenToRedMax, /*max=*/kGreenToRedMax,
          /*num_clusters=*/1, StorageMethod::kAuto);
  InitNonARGB(num_pixels);
}

void LosslessSymbolsInfo::InitAsPredictorImage(uint32_t num_pixels) {
  InitAsLabelImage(num_pixels, kNumPredictors);
}

WP2Status LosslessSymbolsInfo::InitSubMode() {
  SetInfo(kSymbolSubMode, -3, 3, /*num_clusters=*/3,
          LosslessSymbolsInfo::StorageMethod::kAuto);
  // 45 degrees.
  WP2_CHECK_STATUS(
      SetMinMax(kSymbolSubMode, /*cluster=*/0, /*min=*/0, /*max=*/3));
  // 180 degrees.
  WP2_CHECK_STATUS(
      SetMinMax(kSymbolSubMode, /*cluster=*/1, /*min=*/-3, /*max=*/0));
  // Everything else.
  WP2_CHECK_STATUS(
      SetMinMax(kSymbolSubMode, /*cluster=*/2, /*min=*/-3, /*max=*/3));
  return WP2_STATUS_OK;
}

WP2Status LosslessSymbolsInfo::CopyFrom(const LosslessSymbolsInfo& other) {
  WP2_CHECK_STATUS(SymbolsInfo::CopyFrom(other));
  sample_format_ = other.sample_format_;
  num_pixels_ = other.num_pixels_;
  return WP2_STATUS_OK;
}

void LosslessSymbolsInfo::SetNumClusters(uint32_t num_clusters) {
  for (uint32_t sym = 0; sym < Size(); ++sym) SetClusters(sym, num_clusters);
}

void LosslessSymbolsInfo::SetCacheRange(uint32_t cache_range,
                                        bool has_segment_cache) {
  const uint32_t num_clusters = NumClusters(kSymbolType);
  if (cache_range == 0) {
    // Do not consider kSymbolColorCache or kSymbolSegmentCache if the cache is
    // unused.
    SetInfo(kSymbolType, /*min=*/0, /*max=*/IsUnused(kSymbolDist) ? 0 : 1,
            num_clusters, StorageMethod::kAuto);
    SetUnused(kSymbolColorCache);
    SetUnused(kSymbolSegmentCache);
    assert(!has_segment_cache);
  } else {
    assert(num_pixels_ > 1);
    SetInfo(kSymbolColorCache, /*min=*/0, /*max=*/cache_range - 1, num_clusters,
            StorageMethod::kAuto);
    if (has_segment_cache) {
      SetInfo(kSymbolType, /*min=*/0, /*max=*/kSymbolTypeNum - 1, num_clusters,
              StorageMethod::kAuto);
      SetInfo(kSymbolSegmentCache, /*min=*/0,
              /*max=*/(1 << kMaxSegmentCacheBits) - 1, num_clusters,
              StorageMethod::kAuto);
    } else {
      SetInfo(kSymbolType, /*min=*/0, /*max=*/kSymbolTypeNum - 2, num_clusters,
              StorageMethod::kAuto);
      SetUnused(kSymbolSegmentCache);
    }
  }
}

bool IsSymbolUsed(const Symbol sym, const bool is_maybe_used[kSymbolTypeNum]) {
  switch (sym) {
    case kSymbolType:
      // We always have symbols so the type is always useful.
      return true;
    case kSymbolA:
    case kSymbolR:
    case kSymbolG:
    case kSymbolB:
      return is_maybe_used[kSymbolTypeLiteral];
    case kSymbolColorCache:
      return is_maybe_used[kSymbolTypeColorCacheIdx];
    case kSymbolSegmentCache:
      return is_maybe_used[kSymbolTypeSegmentCacheIdx];
    case kSymbolDist:
    case kSymbolLen:
      return is_maybe_used[kSymbolTypeCopy];
    case kSymbolSubMode:
      return false;
    default:
      assert(false);
      return true;
  }
}

bool LosslessSymbolsInfo::HasAlpha(const SymbolsInfo& info) {
  return !info.IsUnused(kSymbolA);
}

bool LosslessSymbolsInfo::IsLabelImage(const SymbolsInfo& info) {
  const bool is_label_image = info.IsUnused(kSymbolR);
  if (is_label_image) {
    assert(info.IsUnused(kSymbolA) && info.IsUnused(kSymbolB));
  }
  return is_label_image;
}

// Updates the range of channel 'channel1' by taking the maximum
// addition/subtraction with channel 'channel2' up to a multiplicative
// coefficient 'coeff' with kColorTransformBits bits of fixed precision (like
// color transforms).
static inline void UpdateMinMax(uint32_t channel1, uint32_t channel2,
                                int16_t coeff, int32_t num_bits, bool do_clamp,
                                int16_t* const minima, int16_t* const maxima) {
  // Extend the interval with all possible deltas.
  int16_t minimum = minima[channel1];
  int16_t maximum = maxima[channel1];
  for (int16_t delta :
       {ColorTransformDelta((int32_t)coeff * minima[channel2]),
        ColorTransformDelta(-(int32_t)coeff * minima[channel2]),
        ColorTransformDelta((int32_t)coeff * maxima[channel2]),
        ColorTransformDelta(-(int32_t)coeff * maxima[channel2])}) {
    minimum = std::min(
        {(int)minimum, minima[channel1] - delta, minima[channel1] + delta});
    maximum = std::max(
        {(int)maximum, maxima[channel1] - delta, maxima[channel1] + delta});
  }
  minima[channel1] = minimum;
  maxima[channel1] = maximum;
  if (!do_clamp) return;
  const int16_t abs_value_max =
      (num_bits == 8) ? kMaxAbsResidual8 : kMaxAbsResidual10;
  static_assert(2 * kMaxAbsResidual8 + 1 <= WP2::kANSMaxRange,
                "invalid kMaxAbsResidual8 value");
  static_assert(2 * kMaxAbsResidual10 + 1 <= WP2::kANSMaxRange,
                "invalid kMaxAbsResidual10 value");
  minima[channel1] = std::max(minima[channel1], (int16_t)-abs_value_max);
  maxima[channel1] = std::min(maxima[channel1], abs_value_max);
}

// The written logic is good but it is overridden by setting bounds to [0,255]
// for kPredictor, kCrossColor and kSubtractGreen because they still use wrap
// arounds.
void GetARGBRanges(const TransformType* transforms, WP2SampleFormat format,
                   uint32_t num_colors, int16_t minima_range[4],
                   int16_t maxima_range[4], uint32_t num_transforms) {
  // Define the original range according to the pixel format.
  // Deal with alpha.
  minima_range[0] = 0;
  maxima_range[0] = WP2::kAlphaMax;
  const int32_t num_bits_alpha = WP2::kAlphaBits;
  // Deal with other channels.
  const int32_t num_bits_non_alpha = (format == WP2_Argb_38) ? 10 : 8;
  for (uint32_t c = 1; c < 4; ++c) {
    minima_range[c] = 0;
    maxima_range[c] = (1 << num_bits_non_alpha) - 1;
  }
  constexpr int16_t kCoeffOne = (1 << kColorTransformBits);
  bool do_clamp = true;
  for (uint32_t i = 0; i < num_transforms; ++i) {
    if (transforms[i] == TransformType::kNum) break;
    switch (transforms[i]) {
      case TransformType::kPredictor:
      case TransformType::kPredictorWSub:
        // The predictors are convex or clamped in the current bounds. Only the
        // black predictor can give results out of those bounds.
        // The ranges are updated to be the ranges of the residuals.
        for (uint32_t c = 0; c < 4; ++c) {
          UpdateMinMax(/*channel1=*/c, /*channel2=*/c, /*coeff=*/kCoeffOne,
                       (c == 0) ? num_bits_alpha : num_bits_non_alpha, do_clamp,
                       minima_range, maxima_range);
        }
        break;
      case TransformType::kCrossColor:
        // The color transforms add another rounded(channel * coeff >>
        // kColorTransformBits). First, red is modified by green.
        UpdateMinMax(/*channel1=*/1, /*channel2=*/2,
                     /*coeff=*/kGreenToRedMax, num_bits_non_alpha, do_clamp,
                     minima_range, maxima_range);
        // Then blue by green and red.
        UpdateMinMax(/*channel1=*/3, /*channel2=*/2,
                     /*coeff=*/kGreenToBlueMax, num_bits_non_alpha, do_clamp,
                     minima_range, maxima_range);
        UpdateMinMax(/*channel1=*/3, /*channel2=*/1,
                     /*coeff=*/kRedToBlueMax, num_bits_non_alpha, do_clamp,
                     minima_range, maxima_range);
        break;
      case TransformType::kYCoCgR:
        assert(i == 0);
        // Y range is the same as the original RGB.
        minima_range[1] = 0;
        maxima_range[1] = (1 << num_bits_non_alpha) - 1;
        // Switching to YCoCgR adds one bit to chroma.
        for (uint32_t c = 2; c < 4; ++c) {
          maxima_range[c] = (1 << num_bits_non_alpha) - 1;
          minima_range[c] = -maxima_range[c];
        }
        break;
      case TransformType::kSubtractGreen:
        // green is subtracted from red and blue
        UpdateMinMax(/*channel1=*/1, /*channel2=*/2, /*coeff=*/kCoeffOne,
                     num_bits_non_alpha, do_clamp, minima_range, maxima_range);
        UpdateMinMax(/*channel1=*/3, /*channel2=*/2, /*coeff=*/kCoeffOne,
                     num_bits_non_alpha, do_clamp, minima_range, maxima_range);
        break;
      case TransformType::kColorIndexing:
        assert(i == 0);
        // With color indexing, values can go up to the number of colors so do
        // not clamp anymore after that.
        do_clamp = false;
        // A is set to its maximum value.
        minima_range[0] = maxima_range[0] = WP2::kAlphaMax;
        // R and B are set to 0.
        minima_range[1] = minima_range[3] = 0;
        maxima_range[1] = maxima_range[3] = 0;
        // Only the color index in [0, num_colors-1] is stored in G.
        minima_range[2] = 0;
        maxima_range[2] = num_colors - 1;
        break;
      default:
        assert(false);
    }
  }
  for (uint32_t c = 0; c < 4; ++c) {
    assert(minima_range[c] <= maxima_range[c]);
  }
}

}  // namespace WP2L

//------------------------------------------------------------------------------
