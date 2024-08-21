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
// WP2 lossless encoding.
//

#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/tile_enc.h"
#include "src/utils/ans_enc.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

namespace WP2 {

// -----------------------------------------------------------------------------

WP2Status TileEncoder::LosslessEncode(ANSEnc* const enc) {
  WP2_CHECK_STATUS(WP2L::EncodeImage(
      *config_, tile_->rgb_input, tiles_layout_->gparams->has_alpha_,
      tiles_layout_->image_is_premultiplied, tile_->progress, enc));
  return enc->GetStatus();
}

// -----------------------------------------------------------------------------

}  // namespace WP2
