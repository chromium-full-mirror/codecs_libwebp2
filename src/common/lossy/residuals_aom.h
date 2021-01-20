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
//   Common code for AOM residuals, branched from libgav1, with as little
//   changes as possible.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WP2_COMMON_LOSSY_RESIDUALS_AOM_H_
#define WP2_COMMON_LOSSY_RESIDUALS_AOM_H_

#include <array>

#include "src/common/lossy/residuals.h"
#include "src/common/symbols.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace libgav1 {

//------------------------------------------------------------------------------
// From libgav1/src/dsp/constants.h

constexpr uint32_t kQuantizedCoefficientBufferPadding = 4;
constexpr uint32_t kMaxTxBufferDim =
    (kMaxBlockSizePix + kQuantizedCoefficientBufferPadding);

// The plane types, called luma and chroma in the spec.
// Matches PlaneType in cdfs.inc
// TODO(maryla): should there be one separate for alpha?
enum PlaneType : uint8_t { kPlaneTypeYA, kPlaneTypeUV, kNumPlaneTypes };

constexpr PlaneType GetPlaneType(Channel channel) {
  return (channel == kUChannel || channel == kVChannel) ? kPlaneTypeUV
                                                        : kPlaneTypeYA;
}

//------------------------------------------------------------------------------
// From libgav1/src/utils/common.h

constexpr int DivideBy2(int n) { return n >> 1; }
constexpr int DivideBy4(int n) { return n >> 2; }
constexpr int DivideBy8(int n) { return n >> 3; }
constexpr int MultiplyBy4(int n) { return (4 * n); }

// Input: 1d array index |index|, which indexes into a 2d array of width
//     1 << |tx_width_log2|.
// Output: 1d array index which indexes into a 2d array of width
//     (1 << |tx_width_log2|) + kQuantizedCoefficientBufferPadding.
inline int PaddedIndex(int index, int tx_width_log2) {
  return index + MultiplyBy4(index >> tx_width_log2);
}

constexpr int kNumQuantizerBaseLevels = 2;

//------------------------------------------------------------------------------
// From libgav1/src/symbol_decoder_context.h

enum {
  kBooleanFieldCdfSize = 3,
  kNumSquareTransformSizes = 5,
  kAllZeroContexts = 13,
  kEobPtContexts = 2,
  kEobPt4SymbolCount = 3,
  kEobPt8SymbolCount = 4,
  kEobPt16SymbolCount = 5,
  kEobPt32SymbolCount = 6,
  kEobPt64SymbolCount = 7,
  kEobPt128SymbolCount = 8,
  kEobPt256SymbolCount = 9,
  kEobPt512SymbolCount = 10,
  kEobPt1024SymbolCount = 11,
  kEobExtraContexts = 9,
  kCoeffBaseEobContexts = 4,
  kCoeffBaseEobSymbolCount = 3,
  kCoeffBaseContexts = 42,
  kCoeffBaseSymbolCount = 4,
  kCoeffBaseRangeContexts = 21,
  kCoeffBaseRangeSymbolCount = 4
};  // anonymous enum

typedef int (*CoeffBaseContextFunc)(const uint8_t qlevels[], TrfSize tdim,
                                    int adjusted_tx_width_log2, uint16_t pos);
typedef int (*CoeffBaseRangeContextFunc)(
    const uint8_t qlevels[], int adjusted_tx_width_log2, int pos);

// Class specialized in reading transform residuals.
class AOMResidualIO {
 public:
  // Returns the cluster for a given symbol, given certain other parameters.
  // The overload GetClusterAOM1, GetClusterAOM2, GetClusterAOM3 are meant
  // for symbols using 1, 2 or 3 arguments.
  template <Symbol sym>
  static uint32_t GetClusterAOM1(uint32_t arg0) {
    static_assert(sym == kAOMEOBPT512 || sym == kAOMEOBPT1024, "wrong symbol");
    return arg0;
  }
  template <Symbol sym>
  static uint32_t GetClusterAOM2(uint32_t arg0, uint32_t arg1) {
    static_assert(sym == kAOMAllZero ||
                      sym == kAOMEOBPT4   || sym == kAOMEOBPT8 ||
                      sym == kAOMEOBPT16  ||
                      sym == kAOMEOBPT32  || sym == kAOMEOBPT64 ||
                      sym == kAOMEOBPT128 || sym == kAOMEOBPT256,
                  "wrong symbol");
    const std::array<uint32_t, 2> extent =
        (sym == kAOMAllZero)
            ? std::array<uint32_t, 2>{kNumSquareTransformSizes,
                                      kAllZeroContexts}
            : std::array<uint32_t, 2>{kNumPlaneTypes, kEobPtContexts};
    uint32_t cluster = arg1;
    uint32_t stride = extent[1];
    cluster += arg0 * stride;
    return cluster;
  }
  template <Symbol sym>
  static uint32_t GetClusterAOM3(uint32_t arg0, uint32_t arg1, uint32_t arg2) {
    static_assert(sym == kAOMEOBExtra || sym == kAOMCoeffBaseEOB ||
                      sym == kAOMCoeffBase || sym == kAOMCoeffBaseRange,
                  "wrong symbol");
    const std::array<uint32_t, 3> extent =
        (sym == kAOMEOBExtra)
            ? std::array<uint32_t, 3>{kNumSquareTransformSizes, kNumPlaneTypes,
                                      kEobExtraContexts}
            : (sym == kAOMCoeffBaseEOB)
                  ? std::array<uint32_t, 3>{kNumSquareTransformSizes,
                                            kNumPlaneTypes,
                                            kCoeffBaseEobContexts}
                  : (sym == kAOMCoeffBase)
                        ? std::array<uint32_t, 3>{kNumSquareTransformSizes,
                                                  kNumPlaneTypes,
                                                  kCoeffBaseContexts}
                        : std::array<uint32_t, 3>{kNumSquareTransformSizes,
                                                  kNumPlaneTypes,
                                                  kCoeffBaseRangeContexts};
    uint32_t cluster = arg2;
    uint32_t stride = extent[2];
    cluster += arg1 * stride;
    stride *= extent[1];
    cluster += arg0 * stride;
    return cluster;
  }

 public:
  // Lookup used to call the right variant of GetCoeffBase*() based on
  // the transform class (in [0..2])
  static CoeffBaseContextFunc
      GetCoeffsBaseContextFunc(uint32_t tx_class);
  static CoeffBaseRangeContextFunc
      GetCoeffBaseRangeContextFunc(uint32_t tx_class);

 protected:
  static int GetCoeffBaseContextEob(TrfSize tdim, int index);

  static int GetCoeffBaseRangeContextEob(int adjusted_tx_width_log2, int pos);
};

//------------------------------------------------------------------------------
// From libgav1/src/tile/tile.cc

// ith entry of this array is computed as:
// DivideBy2(TransformSizeToSquareTransformIndex(kTransformSizeSquareMin[i]) +
//           TransformSizeToSquareTransformIndex(kTransformSizeSquareMax[i]) +
//           1)
// with special value '0' for 2x? or ?x2 transforms.
constexpr uint8_t kTrfContext[] = {
    0, 0, 0,
    0, 0, 1, 1,
    0, 1, 1, 2, 2,
       1, 2, 2, 3,
          2, 3, 3
};
STATIC_ASSERT_ARRAY_SIZE(kTrfContext, TRF_LAST);

// Range above kNumQuantizerBaseLevels which the exponential golomb coding
// process is activated.
constexpr int kQuantizerCoefficientBaseRange = 12;
constexpr int kQuantizerCoefficientBaseRangeContextClamp =
    kQuantizerCoefficientBaseRange + kNumQuantizerBaseLevels + 1;
constexpr int kCoeffBaseRangeMaxIterations =
    kQuantizerCoefficientBaseRange / (kCoeffBaseRangeSymbolCount - 1);

}  // namespace libgav1
}  // namespace WP2

#endif /* WP2_COMMON_LOSSY_RESIDUALS_AOM_H_ */
