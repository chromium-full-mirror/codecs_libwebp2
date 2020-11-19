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
// ProgressWatcher is handling parallel progress computation and user
// notification.
//
// Author: Yannis Guyon (yguyon@google.com)

#ifndef WP2_COMMON_PROGRESS_WATCHER_H_
#define WP2_COMMON_PROGRESS_WATCHER_H_

#include "src/utils/thread_utils.h"
#include "src/wp2/base.h"

namespace WP2 {

//------------------------------------------------------------------------------

// Handles progress computation concurrency and user notification.
// Thread-safe unless specified.
class ProgressWatcher {
 public:
  explicit ProgressWatcher(ProgressHook* const hook)
      : hook_(hook), progress_(0.f), aborted_(false) {}

  // Sets the 'progress_'. Must be in [0:1].
  WP2Status Set(float progress);
  // Advances the 'progress_' by 'step' units. Everything must remain in [0:1].
  WP2Status AdvanceBy(float step);

  // Rewind() goes to 0, Finish() goes to 1.
  WP2Status Start();
  WP2Status Finish();

  // Returns the progress in [0:1] range. Not thread-safe.
  float GetProgress() const { return progress_; }

 private:
  ProgressHook* const hook_;
  float progress_;
  ThreadLock thread_lock_;
  bool aborted_;  // True if 'hook_' returned false at least once.
};

class ProgressRange {
 public:
  // Sets the 'watcher' to be notified when AdvanceBy() is called.
  // The 'range' represents the portion of the progress booked by this instance.
  ProgressRange(ProgressWatcher* const watcher, float range);
  // Sub-range. Can be recursive.
  ProgressRange(const ProgressRange& parent, float range);
  // Can be used as a class member.
  ProgressRange() = default;
  ProgressRange(const ProgressRange&) = default;
  ProgressRange(ProgressRange&&) = default;
  ProgressRange& operator=(const ProgressRange&) = default;
  ProgressRange& operator=(ProgressRange&&) = default;

  // Advances the progress by 'step' units, in [0:1] mapped to the 'range_'.
  // It is const because the range is not modified, only the progress within.
  WP2Status AdvanceBy(float step) const;

 private:
  ProgressWatcher* watcher_ = nullptr;
  float range_ = 0.f;  // In [0:1].
};

// Can be used to scale AdvanceBy() steps rather than booking a portion of the
// progress. The concepts differ but the code is the same, although AdvanceBy()
// is not bounded to [0:1].
typedef ProgressRange ProgressScale;

//------------------------------------------------------------------------------
// Helper functions to compute the progression ranges.

// Returns the size of the progression range reserved to the decoding of the
// frame at 'frame_index'.
float GetFrameProgression(uint32_t frame_index, bool is_last);

// Returns the expected progress at the beginning of the given 'frame_index'
// (right before ANMF chunk).
float GetProgressAtFrame(uint32_t frame_index);

//------------------------------------------------------------------------------

}  // namespace WP2

#endif  /* WP2_COMMON_PROGRESS_WATCHER_H_ */
