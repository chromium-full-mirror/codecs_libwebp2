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
//  Predictor clustering
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include <cstdio>
#include <limits>

#include "src/common/lossy/block.h"
#include "src/enc/analysis.h"
#include "src/utils/front_mgr.h"
#include "src/utils/plane.h"

namespace WP2 {

//------------------------------------------------------------------------------

bool SetForcedPredictor(const Predictors& preds, const EncoderConfig& config,
                        uint32_t tile_pos_x, uint32_t tile_pos_y,
                        Channel channel, CodedBlock* const cb) {
  EncoderInfo* const info = config.info;
  if (info == nullptr || info->force_predictor.empty()) return false;

  for (const EncoderInfo::ForcedPredictor& forced : info->force_predictor) {
    if (forced.channel != channel || forced.x != (cb->x_pix() + tile_pos_x) ||
        forced.y != (cb->y_pix() + tile_pos_y)) {
      continue;
    }

    assert(!preds.empty());
    const Predictor* const pred =
        preds[std::min(forced.predictor_id, (uint8_t)(preds.size() - 1))];
    cb->SetPredictor(channel, pred);
    cb->pred_scores_[(channel == kVChannel) ? kUChannel : channel] = 1.f;
    return true;
  }

  return false;
}

//------------------------------------------------------------------------------

}  // namespace WP2
