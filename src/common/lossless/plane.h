// Copyright 2025 Google LLC
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
// Classes to compress color planes independently.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef THIRD_PARTY_LIBWEBP2_PUBLIC_SRC_COMMON_LOSSLESS_PLANE_H_
#define THIRD_PARTY_LIBWEBP2_PUBLIC_SRC_COMMON_LOSSLESS_PLANE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

namespace WP2L {

////////////////////////////////////////////////////////////////////////////////
// Data structure that stores pixel information by planes, in order ARGB.

class ImagePlanes {
 public:
  WP2Status Create(size_t xsize, size_t ysize, bool has_alpha) {
    width_ = xsize;
    height_ = ysize;
    for (uint32_t c = has_alpha ? 0 : 1; c < 4; ++c) {
      WP2_CHECK_ALLOC_OK(planes_[c].resize(xsize * ysize));
    }
    return WP2_STATUS_OK;
  }
  int16_t* PlaneRow(uint32_t c, uint32_t y) {
    return planes_[c].data() + y * width_;
  }
  const int16_t* PlaneRow(uint32_t c, uint32_t y) const {
    return planes_[c].data() + y * width_;
  }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  bool has_alpha() const { return !planes_[0].empty(); }

 private:
  size_t width_;
  size_t height_;
  // Planes are in order ARGB.
  std::array<WP2::Vector_s16, 4> planes_;
};

////////////////////////////////////////////////////////////////////////////////
// Class for a generic plane compressor.

class PlaneCodec {
 public:
  enum class Method { kScpClassical, kScpJxl, kCalic, kNum };

  virtual ~PlaneCodec() = default;

  virtual WP2Status Init(uint32_t width, uint32_t height) {
    width_ = width;
    last_x_ = width - 1;
    height_ = height;

    WP2_CHECK_STATUS(Reset());

    return WP2_STATUS_OK;
  }

  virtual WP2Status Reset() {
    row_ = row_p_ = row_pp_ = nullptr;
    return WP2_STATUS_OK;
  }

  virtual inline void StartProcessingLine(uint32_t y, const int16_t* row) {
    y_ = y;
    if (y > 1) {
      row_pp_ = row_p_;
      row_p_ = row_;
      row_ = row;
    } else if (y == 1) {
      row_p_ = row_;
      row_ = row;
    } else {
      row_ = row;
    }
  }

 protected:
  // The current row of the image for prediction.
  const int16_t* WP2_RESTRICT row_;
  // Previous rows of the image for prediction.
  const int16_t *WP2_RESTRICT row_p_, *WP2_RESTRICT row_pp_;

  uint32_t y_;
  uint32_t last_x_;  // width - 1, for prediction borders

  uint32_t width_, height_;
};
}  // namespace WP2L

#endif  // THIRD_PARTY_LIBWEBP2_PUBLIC_SRC_COMMON_LOSSLESS_PLANE_H_
