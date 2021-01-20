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
//   AOM residual decoding, branched from libgav1, with as little changes as
//   possible.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WP2_DEC_RESIDUALS_DEC_AOM_H_
#define WP2_DEC_RESIDUALS_DEC_AOM_H_

#include "src/common/lossy/block.h"
#include "src/common/lossy/residuals.h"
#include "src/common/lossy/residuals_aom.h"
#include "src/common/lossy/transforms.h"
#include "src/dec/symbols_dec.h"
#include "src/utils/ans.h"
#include "src/utils/utils.h"

namespace WP2 {
namespace libgav1 {

//------------------------------------------------------------------------------
// From libgav1/src/tile/tile.h

// Class specialized in reading transform residuals.
// This is a pure interface (only static methods to call directly).
class AOMResidualReader : public AOMResidualIO {
 public:
  // Reads residual coeffs, returns the number of non-zero coefficients read.
  static uint32_t ReadCoeffs(
      Channel channel, TrfSize tdim, TransformPair tx_type, bool first_is_dc,
      SymbolReader* const sr, ANSDec* const dec, int16_t* const coeffs);

 private:
  // Reads End-of-Block element.
  static int ReadEOB(TrfSize tdim, PlaneType plane_type,
                     int context, SymbolReader* const sr, ANSDec* const dec);

  // Part of 5.11.39.
  static void ReadSignAndApplyDequantization(
      int adjusted_tx_width_log2, int16_t* const coeffs, uint32_t max_value,
      SymbolReader* const sr, ANSDec* const dec);
  static int ReadCoeffBaseRange(int tx_size_context, int cdf_context,
                                int plane_type, SymbolReader* const sr);
};

}  // namespace libgav1
}  // namespace WP2

#endif /* WP2_DEC_RESIDUALS_DEC_AOM_H_ */
