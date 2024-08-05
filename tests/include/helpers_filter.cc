// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------

#include "tests/include/helpers_filter.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "src/common/global_params.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/segment.h"
#include "src/common/symbols.h"
#include "src/dec/filters/block_map_filter.h"
#include "src/dsp/math.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/random.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace testutil {

//------------------------------------------------------------------------------

YUVPlane CreatePlainPixels(uint32_t width, uint32_t height,
                           BitDepth bit_depth) {
  const int16_t min = bit_depth.min(), max = bit_depth.max();
  YUVPlane pixels;
  WP2_ASSERT_STATUS(pixels.Resize(width, height));
  // Testing full range.
  pixels.Fill(Ayuv38b{kAlphaMax, (int16_t)((min + max) / 2), min, max});
  return pixels;
}

void Noise(int32_t min_value, int32_t max_value, size_t seed, int32_t strength,
           Plane16* const plane) {
  UniformIntDistribution random(seed);
  for (uint32_t y = 0; y < plane->h_; ++y) {
    for (uint32_t x = 0; x < plane->w_; ++x) {
      const int32_t value = plane->At(x, y) + random.Get(-strength, strength);
      plane->At(x, y) = (int16_t)Clamp(value, min_value, max_value);
    }
  }
}

void Noise(size_t seed, int32_t strength, ArgbBuffer* const canvas) {
  UniformIntDistribution random(seed);
  assert(canvas->format() == WP2_Argb_32);

  uint8_t* pixel = canvas->GetRow8(0);
  for (uint32_t y = 0; y < canvas->height(); ++y, pixel += canvas->stride()) {
    for (uint32_t x = 0; x < canvas->width(); ++x) {
      const int32_t a = pixel[x * 4 + 0];
      uint8_t* const r = &pixel[x * 4 + 1];
      uint8_t* const g = &pixel[x * 4 + 2];
      uint8_t* const b = &pixel[x * 4 + 3];

      *r = (uint8_t)Clamp(*r + random.Get(-strength, strength), 0, a);
      *g = (uint8_t)Clamp(*g + random.Get(-strength, strength), 0, a);
      *b = (uint8_t)Clamp(*b + random.Get(-strength, strength), 0, a);
    }
  }
}

bool AreEqual(const Plane16& a, const Plane16& b) {
  if (a.w_ != b.w_ || a.h_ != b.h_) return false;
  for (uint32_t y = 0; y < a.h_; ++y) {
    for (uint32_t x = 0; x < a.w_; ++x) {
      if (a.At(x, y) != b.At(x, y)) return false;
    }
  }
  return true;
}

bool AreEqual(const YUVPlane& a, const YUVPlane& b) {
  return AreEqual(a.Y, b.Y) && AreEqual(a.U, b.U) && AreEqual(a.V, b.V) &&
         AreEqual(a.A, b.A);
}

bool AreEqual(const ArgbBuffer& a, const ArgbBuffer& b) {
  float psnr[5];
  if (a.GetDistortion(b, PSNR, psnr) != WP2_STATUS_OK) assert(false);
  return (psnr[4] == 99.f);
}

//------------------------------------------------------------------------------

Vector<Segment> CreateSegments(float quality, BitDepth bit_depth) {
  Vector<Segment> segments;
  if (!segments.resize(1)) abort();
  segments[0].SetYUVBounds(bit_depth.min(), bit_depth.max());
  segments[0].SetQuality(kQualityToQFactor(quality), kDefaultQuantOffset,
                         kDefaultQuantOffset);
  return segments;
}

void RegisterBlock(uint16_t x, uint16_t y, BlockSize dim, uint32_t segment_id,
                   double res_den, double min_bpp, bool with_lossy_alpha,
                   FilterBlockMap* const blocks) {
  CodedBlockBase cb;
  cb.SetDimDefault({x, y, dim});
  cb.is420_ = false;
  cb.id_ = (uint8_t)segment_id;
  cb.alpha_mode_ = with_lossy_alpha ? kBlockAlphaLossy : kBlockAlphaLossless;
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !with_lossy_alpha) continue;
    const uint32_t num_coeffs = cb.NumCoeffsPerTransform(c);
    const uint32_t nnz = std::lround(num_coeffs * res_den);
    for (uint32_t tf_i = 0; tf_i < cb.GetNumTransforms(c); ++tf_i) {
      std::fill(cb.coeffs_[c][tf_i], cb.coeffs_[c][tf_i] + nnz, 1);
      cb.num_coeffs_[c][tf_i] = nnz;
      std::fill(cb.coeffs_[c][tf_i] + nnz, cb.coeffs_[c][tf_i] + num_coeffs, 0);
    }
  }
  blocks->RegisterBlock(cb, std::lround(NumPix(dim) * min_bpp / 8));
}

void CreateBlocks(uint32_t tile_x, uint32_t tile_y,
                  const Vector<Segment>& segments, double res_den,
                  double min_bpp, BitDepth bit_depth, YUVPlane* const pixels,
                  FilterBlockMap* const blocks) {
  blocks->Init({tile_x, tile_y, pixels->Y.w_, pixels->Y.h_}, bit_depth,
               bit_depth.min(), bit_depth.max(), pixels);
  if (blocks->Allocate() != WP2_STATUS_OK) abort();

  const uint32_t seed = 7 * pixels->Y.w_ + pixels->Y.h_;
  UniformIntDistribution gen(seed);
  VectorNoCtor<Block> raw_blocks;
  CreatePartition(pixels->Y.w_, pixels->Y.h_, /*snapped=*/false, &gen,
                  &raw_blocks);
  uint32_t id = 0;
  for (const Block& blk : raw_blocks) {
    RegisterBlock(blk.x(), blk.y(), blk.dim(), id % segments.size(), res_den,
                  min_bpp, /*with_lossy_alpha=*/false, blocks);
    ++id;
  }
}

//------------------------------------------------------------------------------

}  // namespace testutil
}  // namespace WP2
