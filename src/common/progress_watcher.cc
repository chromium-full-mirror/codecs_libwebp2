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
// ProgressWatcher implementation.
//
// Author: Yannis Guyon (yguyon@google.com)

#include "src/common/progress_watcher.h"

#include <cmath>

#include "src/common/constants.h"
#include "src/utils/utils.h"

namespace WP2 {

//------------------------------------------------------------------------------
// TODO(yguyon): Add no-op fast return when 'hook_' is null.

WP2Status ProgressWatcher::Set(float progress) {
  WP2_CHECK_STATUS(thread_lock_.Acquire());
  if (progress == 0.f) aborted_ = false;  // Rewinded.
  if (!aborted_) {
    assert(progress >= 0.f && progress <= 1.f);
    progress_ = progress;
    if (hook_ != nullptr) aborted_ = !hook_->OnUpdate(progress_);
  }
  thread_lock_.Release();
  return aborted_ ? WP2_STATUS_USER_ABORT : WP2_STATUS_OK;
}

WP2Status ProgressWatcher::Rewind() { return Set(0.f); }

WP2Status ProgressWatcher::Finish() {
  assert(progress_ < 1.f);
  assert(std::abs(progress_ + kProgressDecEnd - 1.f) < 0.001f);
  return Set(1.f);
}

WP2Status ProgressWatcher::AdvanceBy(float step) {
  WP2_CHECK_STATUS(thread_lock_.Acquire());
  if (!aborted_) {
    progress_ += scale_ * step;
    assert(progress_ >= 0.f && progress_ <= 1.f);
    if (hook_ != nullptr) aborted_ = !hook_->OnUpdate(progress_);
  }
  const bool aborted = aborted_;  // Prevent threaded data race.
  thread_lock_.Release();
  return aborted ? WP2_STATUS_USER_ABORT : WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

float GetProgressionBeforeFrames(bool has_preview, bool has_icc) {
  return kProgressDecHeader + (has_preview ? kProgressDecPreview : 0.f) +
         (has_icc ? kProgressDecICC : 0.f);
}

float GetProgressionAfterFrames(bool has_trailing_data) {
  return (has_trailing_data ? kProgressDecMetadata : 0.f) + kProgressDecEnd;
}

float GetFrameProgression(bool has_preview, bool has_icc,
                          bool has_trailing_data, uint32_t frame_index,
                          bool is_last) {
  // The portion of the [0:1] range that corresponds to ANMF and tile decoding.
  const float total_frames_progression =
      (1.f - (GetProgressionBeforeFrames(has_preview, has_icc) +
              GetProgressionAfterFrames(has_trailing_data)));
  const bool single_frame = (frame_index == 0 && is_last);
  if (single_frame) {
    return total_frames_progression;
  } else {
    // Each frame progression occupies half of what remains (so the first one
    // 1/2, the second 1/4, the third 1/8 etc.). The progress is not linear but
    // the number of frames is not known in advance and this has two advantages:
    //  - Animations with a few frames still get a decent progression portion
    //    per frame, compared to 1/kMaxNumFrames.
    //  - The last "missing" progression will be minimal and is actually
    //    included in the last frame.

    // The portion of the 'total_frames_progression' range that corresponds to
    // the single ANMF chunk and the tiles belonging to the 'frame_index'.
    const float frame_progression_among_others =
        std::pow(0.5f, frame_index + (is_last ? 0.f : 1.f));
    return total_frames_progression * frame_progression_among_others;
  }
}

float GetANMFProgression(bool has_preview, bool has_icc, bool has_trailing_data,
                         uint32_t frame_index, bool is_last) {
  return GetFrameProgression(has_preview, has_icc, has_trailing_data,
                             frame_index, is_last) *
         kProgressDecANMF;
}

float GetRowProgression(bool has_preview, bool has_icc, bool has_trailing_data,
                        bool is_animation, uint32_t num_tiles_x,
                        uint32_t frame_index, uint32_t frame_height,
                        bool is_last) {
  const uint32_t num_tile_rows = num_tiles_x * frame_height;
  assert(num_tile_rows > 0);
  return GetFrameProgression(has_preview, has_icc, has_trailing_data,
                             frame_index, is_last) *
         (is_animation ? (1.f - kProgressDecANMF) : 1.f) *
         (1.f / num_tile_rows);
}

float GetProgressAtFrame(bool has_preview, bool has_icc, bool has_trailing_data,
                         uint32_t frame_index) {
  const float progress_before_frames =
      GetProgressionBeforeFrames(has_preview, has_icc);
  if (frame_index == 0) {
    return progress_before_frames;
  } else {
    const float total_frames_progression =
        (1.f - progress_before_frames -
         GetProgressionAfterFrames(has_trailing_data));
    // Each time a frame is completed, it occupies half of the remaining
    // progression. Hence the same amount is left at the beginning of the next
    // frame.
    const float previous_frame_progression =
        GetFrameProgression(has_preview, has_icc, has_trailing_data,
                            frame_index - 1, /*is_last=*/false);
    return progress_before_frames +
           (total_frames_progression - previous_frame_progression);
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
