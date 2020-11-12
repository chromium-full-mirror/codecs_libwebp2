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
class ProgressWatcher {
 public:
  explicit ProgressWatcher(ProgressHook* hook)
      : hook_(hook), progress_(0.f), scale_(1.f), aborted_(false) {}

  // Sets the 'progress_'. Should be in [0:1]. Ignores the 'scale_'.
  WP2Status Set(float progress);

  // Rewind() goes to 0, Finish() goes to 1. Ignores the 'scale_'.
  WP2Status Rewind();
  WP2Status Finish();

  // Sets the 'scale_' that will be used by the following AdvanceBy() calls.
  // e.g. scale is (1 / (width * height)) and progress advances by 1 per pixel.
  void SetAdvancementScale(float scale) { scale_ = scale; }
  WP2Status AdvanceBy(float step);

  // Returns the progress in [0:1] range.
  float GetProgress() const { return progress_; }

 private:
  ProgressHook* const hook_;
  float progress_;
  float scale_;
  ThreadLock thread_lock_;
  bool aborted_;  // True if 'hook_' returned false at least once.
};

//------------------------------------------------------------------------------
// Helper functions to compute the progression ranges reserved to header,
// frames, metadata.

// Returns the size of the progression range reserved to header, preview and ICC
// decoding.
float GetProgressionBeforeFrames(bool has_preview, bool has_icc);

// Returns the size of the progression range reserved to XMP and EXIF decoding.
float GetProgressionAfterFrames(bool has_trailing_data);

// Returns the size of the progression range reserved to the decoding of the
// frame at 'frame_index'.
float GetFrameProgression(bool has_preview, bool has_icc,
                          bool has_trailing_data, uint32_t frame_index,
                          bool is_last);

// Returns the size of the progression range reserved to the decoding of the
// ANMF chunk at 'frame_index'.
float GetANMFProgression(bool has_preview, bool has_icc, bool has_trailing_data,
                         uint32_t frame_index, bool is_last);

// Returns the size of the progression range reserved to the decoding of each
// row in each tile of the frame at 'frame_index'.
float GetRowProgression(bool has_preview, bool has_icc, bool has_trailing_data,
                        bool is_animation, uint32_t num_tiles_x,
                        uint32_t frame_index, uint32_t frame_height,
                        bool is_last);

// Returns the expected progress at the beginning of the given 'frame_index'
// (right before ANMF chunk).
float GetProgressAtFrame(bool has_preview, bool has_icc, bool has_trailing_data,
                         uint32_t frame_index);

//------------------------------------------------------------------------------

}  // namespace WP2

#endif  /* WP2_COMMON_PROGRESS_WATCHER_H_ */
