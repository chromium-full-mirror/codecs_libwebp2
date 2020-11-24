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
// TODO(yguyon): Add no-op fast return if 'hook_' is null
//               (useful to keep assertions always running for now)

WP2Status ProgressWatcher::Set(double progress) {
  WP2_CHECK_STATUS(thread_lock_.Acquire());
  if (progress == 0.) aborted_ = false;  // Rewinded.
  if (!aborted_) {
    assert(progress >= 0. && progress <= 1.);
    progress_ = progress;
    if (hook_ != nullptr) aborted_ = !hook_->OnUpdate(progress_);
  }
  thread_lock_.Release();
  return aborted_ ? WP2_STATUS_USER_ABORT : WP2_STATUS_OK;
}

WP2Status ProgressWatcher::Start() { return Set(0.); }

WP2Status ProgressWatcher::Finish() {
  assert(progress_ < 1.);
  // If the following assertion fails, it might be a precision error
  // (progression is incremented by amounts that are too small).
  // Try switching to doubles to see if the problem is fixed.
  assert(std::abs(progress_ + kProgressEnd - 1.) < 0.001);
  return Set(1.);
}

WP2Status ProgressWatcher::AdvanceBy(double step) {
  WP2_CHECK_STATUS(thread_lock_.Acquire());
  if (!aborted_) {
    progress_ += step;
    assert(progress_ >= 0. && progress_ <= 1.);
    if (hook_ != nullptr) aborted_ = !hook_->OnUpdate(progress_);
  }
  const bool aborted = aborted_;  // Prevent threaded data race.
  thread_lock_.Release();
  return aborted ? WP2_STATUS_USER_ABORT : WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

ProgressRange::ProgressRange(ProgressWatcher* const watcher, double range)
    : watcher_(watcher), range_(range) {}
ProgressRange::ProgressRange(const ProgressRange& parent, double range)
    : watcher_(parent.watcher_), range_(parent.range_ * range) {}

WP2Status ProgressRange::AdvanceBy(double step) const {
  if (watcher_ == nullptr) return WP2_STATUS_OK;
  return watcher_->AdvanceBy(step * range_);
}

//------------------------------------------------------------------------------

double GetFrameProgression(uint32_t frame_index, bool is_last) {
  if (frame_index == 0 && is_last) {
    return kProgressFrames;
  } else {
    // Each frame progression occupies half of what remains (so the first one
    // 1/2, the second 1/4, the third 1/8 etc.). The progress is not linear but
    // the number of frames is not known in advance and this has two advantages:
    //  - Animations with a few frames still get a decent progression portion
    //    per frame, compared to 1/kMaxNumFrames.
    //  - The last "missing" progression will be minimal and is actually
    //    included in the last frame.

    // The portion that corresponds to the single ANMF chunk and the tiles
    // belonging to the 'frame_index'.
    return kProgressFrames * std::pow(0.5, frame_index + (is_last ? 0. : 1.));
  }
}

double GetProgressAtFrame(uint32_t frame_index) {
  if (frame_index == 0) {
    return kProgressBeforeFrames;
  } else {
    // Each time a frame is completed, it occupies half of the remaining
    // progression. Hence the same amount is left at the beginning of the next
    // frame.
    const double previous_frame_progression =
        GetFrameProgression(frame_index - 1, /*is_last=*/false);
    return kProgressBeforeFrames +
           (kProgressFrames - previous_frame_progression);
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2
