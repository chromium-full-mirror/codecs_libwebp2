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
//   AOM residual encoding, reverse-engineered from libgav1 decoder.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WP2_ENC_RESIDUALS_ENC_AOM_H_
#define WP2_ENC_RESIDUALS_ENC_AOM_H_

#include "src/common/lossy/block.h"
#include "src/common/lossy/residuals_aom.h"
#include "src/common/lossy/transforms.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/ans.h"

namespace WP2 {
namespace libgav1 {

// Class writing AOM residuals.
// Each member function matches one in AOMResidualReader.
// This is a pure interface (only static methods to call directly).
class AOMResidualWriter : public AOMResidualIO {
 public:
  static void WriteCoeffs(const int16_t res[], Channel channel, TrfSize tdim,
                          TransformPair tx_type, bool first_is_dc,
                          SymbolManager* const sm, ANSEncBase* const enc);

 private:
  static void WriteEOB(
      int eob, TrfSize tdim, PlaneType plane_type, uint32_t context,
      SymbolManager* const sm, ANSEncBase* const enc);
  static void WriteSignAndApplyDequantization(
      int16_t res, uint32_t max_value,
      SymbolManager* const sm, ANSEncBase* const enc);
  static void WriteCoeffBaseRange(
      uint32_t level, uint32_t cdf_context, uint32_t plane_type,
      SymbolManager* const sm, ANSEncBase* const enc);
};

}  // namespace libgav1
}  // namespace WP2

#endif /* WP2_ENC_RESIDUALS_ENC_AOM_H_ */
