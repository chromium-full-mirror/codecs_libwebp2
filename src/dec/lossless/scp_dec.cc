// Copyright (c) the JPEG XL Project
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
// Forked from https://github.com/google/pik/blob/master/pik/lossless8.cc
// at 16268ef512a65b541c7b5e485468a7ed33bc13d8

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "src/common/lossless/scp.h"
#include "src/common/symbols.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/dec/symbols_dec.h"
#include "src/utils/ans.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

namespace WP2L {
namespace scp {

////////////////////////////////////////////////////////////////////////////////

// Decoder for the classical SCP.
class ClassicalDecState : public ClassicalState {
 public:
  WP2Status DecompressPlane(int planeToDecompress, const ScpGlobal& global,
                            WP2::SymbolReader& sr, ImagePlanes& img) {
    const int32_t min = global.min_tpv[planeToDecompress];
    const int32_t max = global.max_tpv[planeToDecompress];
    min_tpv_ = min << kPredExtraBits;
    max_tpv_ = max << kPredExtraBits;

    for (size_t y = 0; y < img.height(); ++y) {
      int16_t* row = img.PlaneRow(planeToDecompress, y);
      StartProcessingLine(y, row);

      // Set predictor pointer.
      const PredictionMode pred_mode =
          global.prediction_modes[planeToDecompress];
      auto func = y == 0 ? &ClassicalDecState::PredictY0
                  : pred_mode == PredictionMode::Regular
                      ? &ClassicalDecState::PredictRegular
                  : pred_mode == PredictionMode::West
                      ? &ClassicalDecState::PredictWest
                  : pred_mode == PredictionMode::North
                      ? &ClassicalDecState::PredictNorth
                      : nullptr;
      for (size_t x = 0; x < img.width(); ++x) {
        // Predict.
        uint8_t ctxt;
        int16_t prediction = (this->*func)(x, &ctxt);

        // Compute the true pixel value.
        ctxt >>= global.maxerr_shift;
        assert(0 <= ctxt && ctxt <= kNumContexts - 1);
        const int context = ctxt;
        int q = sr.Read(/*sym=*/planeToDecompress, context, "sym");
        row[x] = CalcTPVFromPredictionAndDistance(prediction, q, min, max);

        UpdateErrors</*USE_JXL=*/false>(x, row[x], q);
      }
    }

    return WP2_STATUS_OK;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Decoder for the JPEG XL SCP.
class JXLDecState : public JXLState {
 public:
  WP2Status DecompressPlane(int planeToDecompress, const ScpGlobal& global,
                            WP2::SymbolReader& sr, ImagePlanes& img) {
    const int32_t min = global.min_tpv[planeToDecompress];
    const int32_t max = global.max_tpv[planeToDecompress];
    min_tpv_ = min << kPredExtraBits;
    max_tpv_ = max << kPredExtraBits;

    for (size_t y = 0; y < img.height(); ++y) {
      int16_t* row = img.PlaneRow(planeToDecompress, y);
      StartProcessingLine(y, row);
      // Set predictor pointer.
      auto func = &JXLDecState::Predict;
      for (size_t x = 0; x < img.width(); ++x) {
        uint8_t ctxt;
        int16_t guess = (this->*func)(x, &ctxt);

        // Compute the true pixel value.
        int q = sr.Read(/*sym=*/planeToDecompress, ctxt, "sym");
#if 1
        q = q % 2 == 0 ? q / 2 : -(q + 1) / 2;
        row[x] = guess + q;
        // TODO(vrabaud): do not clamp !!! Those are wasted bits.
        row[x] = std::clamp((int32_t)row[x], min, max);
        assert(row[x] >= min && row[x] <= max);
#else
        row[x] = CalcTPVFromPredictionAndDistance(guess, q, min, max);
#endif

        int q2 = CalcDistanceFromPredictionAndTPV(guess, row[x], min, max);
        UpdateErrors</*USE_JXL=*/true>(x, row[x], q2);
      }
    }

    return WP2_STATUS_OK;
  }
};

// Main function to decompress a buffer using an SCP state.
template <typename STATE>
WP2Status Decompress(const ScpGlobal& global, uint32_t width, uint32_t height,
                     bool has_alpha, STATE& state, WP2::SymbolReader& sr,
                     int16_t* out) {
  ImagePlanes img;
  WP2_CHECK_STATUS(img.Create(width, height, has_alpha));

  for (uint32_t c = has_alpha ? 0 : 1; c < 4; ++c) {
    WP2_CHECK_STATUS(state.Reset());
    WP2_CHECK_STATUS(state.DecompressPlane(c, global, sr, img));
    // Copy back.
    for (size_t y = 0; y < height; ++y) {
      const int16_t* const __restrict__ rowImg = img.PlaneRow(c, y);
      for (size_t x = 0; x < width; ++x) {
        out[y * 4 * width + x * 4 + c] = rowImg[x];
      }
    }
  }

  return WP2_STATUS_OK;
}

}  // namespace scp

WP2Status Decoder::DecodeScpHeader(ScpGlobal* global) {
  std::array<int32_t, 4> minima_range, maxima_range;
  WP2_CHECK_STATUS(GetARGBRanges(minima_range, maxima_range));
  for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
    global->min_tpv[c] =
        dec_->ReadRange(minima_range[c], maxima_range[c], "min_tpv");
    global->max_tpv[c] =
        dec_->ReadRange(global->min_tpv[c], maxima_range[c], "max_tpv");
  }

  global->use_jxl = dec_->ReadBool("use_jxl");
  int num_contexts;
  if (global->use_jxl) {
    global->jxl_header_index =
        dec_->ReadRange(0, scp::kJxlHeaders.size(), "header_index");
    if (global->jxl_header_index == static_cast<int>(scp::kJxlHeaders.size())) {
      global->jxl_header.p1c = dec_->ReadRange(0, 40, "p1c");
      global->jxl_header.p2c = dec_->ReadRange(0, 40, "p2c");
      global->jxl_header.p3ca = dec_->ReadRange(0, 40, "p3ca");
      global->jxl_header.p3cb = dec_->ReadRange(0, 40, "p3cb");
      global->jxl_header.p3cc = dec_->ReadRange(0, 40, "p3cc");
      global->jxl_header.p3cd = dec_->ReadRange(0, 40, "p3cd");
      global->jxl_header.p3ce = dec_->ReadRange(0, 40, "p3ce");
      for (int i = 0; i < 4; ++i) {
        global->jxl_header.w[i] = dec_->ReadRange(0, 40, "w");
      }
    }
    num_contexts = scp::JXLState::GetNumContexts();
  } else {
    global->maxerr_shift = dec_->ReadRValue(5, "maxerrShift");
    num_contexts = scp::ClassicalState::GetNumContexts(global->maxerr_shift);
  }

  WP2::SymbolsInfo symbols_info;
  WP2_CHECK_STATUS(hdr_.sr_.Init(symbols_info, dec_));

  if (!global->use_jxl) {
    for (int c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
      global->prediction_modes[c] =
          static_cast<scp::PredictionMode>(dec_->ReadRValue(
              static_cast<int>(scp::PredictionMode::Num), "pred_mode"));
    }
  }

  for (uint32_t c = gparams_->has_alpha_ ? 0 : 1; c < 4; ++c) {
    symbols_info.SetInfo(/*sym=*/c, /*min=*/0, /*max=*/1024,
                         /*num_clusters=*/num_contexts,
                         WP2::SymbolsInfo::StorageMethod::kAuto);
  }

  WP2_CHECK_STATUS(hdr_.sr_.Init(symbols_info, hdr_.sr_.dec()));
  WP2_CHECK_STATUS(hdr_.sr_.Allocate());
  for (uint32_t s = 0; s < symbols_info.Size(); ++s) {
    for (uint32_t c = 0; c < symbols_info.NumClusters(s); ++c) {
      WP2_CHECK_STATUS(
          hdr_.sr_.ReadHeader(s, c, tile_->rect.GetArea(), "header"));
    }
  }
  return WP2_STATUS_OK;
}

WP2Status Decoder::DecodeScp(uint32_t width, uint32_t last_row,
                             const ScpGlobal& global,
                             WP2::Planef* bits_per_pixel, int16_t** src_out) {
  int16_t* const data = pixels_.data();

  if (last_pixel_ == 0) {
    // Decode everything the first time.
    const int height = tile_->rect.height;
    int16_t* src = data + 4 * last_pixel_;
    if (global.use_jxl) {
      scp::JXLDecState state;
      WP2_CHECK_STATUS(state.Init(width, height));
      if (global.jxl_header_index ==
          static_cast<int>(scp::kJxlHeaders.size())) {
        state.SetHeader(global.jxl_header);
      } else {
        state.SetHeaderIndex(global.jxl_header_index);
      }
      WP2_CHECK_STATUS(Decompress(global, width, height, gparams_->has_alpha_,
                                  state, hdr_.sr_, src));
    } else {
      scp::ClassicalDecState state;
      WP2_CHECK_STATUS(state.Init(width, height));
      WP2_CHECK_STATUS(Decompress(global, width, height, gparams_->has_alpha_,
                                  state, hdr_.sr_, src));
    }
    for (size_t y = 0; y < last_row; ++y) {
      WP2_CHECK_STATUS(ProcessRows(y));
    }
  }

  if (src_out != nullptr) *src_out = data + 4 * width * last_row;
  return WP2_STATUS_OK;
}

}  // namespace WP2L
