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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "src/common/lossless/scp.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/ans_enc.h"
#include "src/utils/ans_utils.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"

namespace WP2L {

namespace scp {

////////////////////////////////////////////////////////////////////////////////

struct Token {
  Token() = default;
  // c is the number of contexts.
  Token(uint32_t c, uint32_t v) : context(c), value(v) {}
  uint32_t context;
  int16_t value;
};

// What defines a plane encoding according to the SCP.
struct Encoding {
  void swap(Encoding& encoding) {
    tokens.swap(encoding.tokens);
    std::swap(mode, encoding.mode);
  }
  PredictionMode mode;  // classical only
  WP2::Vector<Token> tokens;
};

////////////////////////////////////////////////////////////////////////////////

// Encoder for the classical SCP.
// SetConfig must be called after Init to be properly usable.
class ClassicalEncState : public ClassicalState {
 public:
  struct Config {
    std::array<PredictionMode, 4> p_modes = {
        PredictionMode::Auto, PredictionMode::Auto, PredictionMode::Auto,
        PredictionMode::Auto};
    int maxerr_shift = -1;
  };

  void SetConfig(const Config& config, const ImagePlanes& img) {
    config_ = config;
    if (config_.maxerr_shift == -1) {
      const size_t area = img.width() * img.height();
      const int maxerr_shift = (area > 25600   ? 0
                                : area > 12800 ? 1
                                : area > 4000  ? 2
                                : area > 400   ? 3
                                               : 4);
      config_.maxerr_shift = maxerr_shift;
    }
  }

  WP2Status CompressPlane(const ImagePlanes& img, int planeToCompress,
                          int16_t min_tpv, int16_t max_tpv,
                          Encoding* encoding) {
    const int maxerr_shift = config_.maxerr_shift;
    encoding->tokens.clear();
    min_tpv_ = min_tpv << kPredExtraBits;
    max_tpv_ = max_tpv << kPredExtraBits;

    PredictionMode pred_mode;
    if (config_.p_modes[planeToCompress] == PredictionMode::Auto) {
      double fromN = 0, fromW = 0;
      for (size_t y = 1; y < img.height(); ++y) {
        const int16_t* const row = img.PlaneRow(planeToCompress, y);
        const int16_t* const row_p = img.PlaneRow(planeToCompress, y - 1);
        uint32_t fromNx = 0, fromWx = 0;
        for (size_t x = 1; x < img.width(); ++x) {
          int c = row[x];
          int N = row_p[x];
          int W = row[x - 1];
          N -= c;
          W -= c;
          fromNx += N * N;
          fromWx += W * W;
        }
        fromN += fromNx;
        fromW += fromWx;
      }
      pred_mode = PredictionMode::Regular;
      if (fromW * 5 < fromN * 4) pred_mode = PredictionMode::West;
      if (fromN * 5 < fromW * 4) pred_mode = PredictionMode::North;
    } else {
      pred_mode = config_.p_modes[planeToCompress];
    }

    for (size_t y = 0; y < img.height(); ++y) {
      const int16_t* row = img.PlaneRow(planeToCompress, y);
      StartProcessingLine(y, row);
      // Set predictor pointer.
      auto func = y == 0 ? &ClassicalEncState::PredictY0
                  : pred_mode == PredictionMode::Regular
                      ? &ClassicalEncState::PredictRegular
                  : pred_mode == PredictionMode::West
                      ? &ClassicalEncState::PredictWest
                  : pred_mode == PredictionMode::North
                      ? &ClassicalEncState::PredictNorth
                      : nullptr;
      for (size_t x = 0; x < img.width(); ++x) {
        uint8_t ctxt;
        const int16_t prediction = (this->*func)(x, &ctxt);
        ctxt >>= maxerr_shift;
        assert(0 <= ctxt && ctxt <= kNumContexts - 1);
        const int q = CalcDistanceFromPredictionAndTPV(prediction, row[x],
                                                       min_tpv, max_tpv);
        WP2_CHECK_ALLOC_OK(encoding->tokens.push_back(Token(ctxt, q)));

        UpdateErrors</*USE_JXL=*/false>(x, row[x], q);
      }
    }

    // output the prediction mode in an unused context
    encoding->mode = pred_mode;

    return WP2_STATUS_OK;
  }

  void EncodeHeader(WP2::ANSEncBase& enc) {
    enc.PutRValue(config_.maxerr_shift, 5, "maxerrShift");
  }

  int GetNumContexts() const {
    return ClassicalState::GetNumContexts(config_.maxerr_shift);
  }

 private:
  Config config_;
};

////////////////////////////////////////////////////////////////////////////////

// Encoder for the JPEG XL SCP.
// SetConfig must be called after Init to be properly usable.
class JXLEncState : public JXLState {
 public:
  struct Config {
    int jxl_header_index = -1;
    scp::JxlHeader jxl_header;
  };

  void SetConfig(const Config& config) {
    config_ = config;
    if (config_.jxl_header_index == static_cast<int>(kJxlHeaders.size())) {
      SetHeader(config_.jxl_header);
    } else {
      SetHeaderIndex(config_.jxl_header_index);
    }
  }

  WP2Status CompressPlane(const ImagePlanes& img, int planeToCompress,
                          int16_t min_tpv, int16_t max_tpv,
                          Encoding* encoding) {
    encoding->tokens.clear();
    min_tpv_ = min_tpv << kPredExtraBits;
    max_tpv_ = max_tpv << kPredExtraBits;

    for (size_t y = 0; y < img.height(); ++y) {
      const int16_t* row = img.PlaneRow(planeToCompress, y);
      StartProcessingLine(y, row);

      // Set predictor pointer.
      auto func = &JXLEncState::Predict;
      for (size_t x = 0; x < img.width(); ++x) {
        uint8_t ctxt;
        const int16_t guess = (this->*func)(x, &ctxt);
        // TODO(vrabaud): choose the best one.
#if 1
        const int16_t res = row[x] - guess;

        WP2_CHECK_ALLOC_OK(encoding->tokens.push_back(
            Token(ctxt, res >= 0 ? 2 * res : -2 * res - 1)));
#else
        WP2_CHECK_ALLOC_OK(encoding->tokens.push_back(
            Token(ctx, CalcDistanceFromPredictionAndTPV(guess, row[x], min_tpv,
                                                        max_tpv))));
#endif
        const int q =
            CalcDistanceFromPredictionAndTPV(guess, row[x], min_tpv, max_tpv);

        UpdateErrors</*USE_JXL=*/true>(x, row[x], q);
      }
    }

    return WP2_STATUS_OK;
  }

  void EncodeHeader(WP2::ANSEncBase& enc) {
    enc.PutRange(config_.jxl_header_index, 0, kJxlHeaders.size(),
                 "header_index");
    if (config_.jxl_header_index == static_cast<int>(kJxlHeaders.size())) {
      enc.PutRange(config_.jxl_header.p1c, 0, 40, "p1c");
      enc.PutRange(config_.jxl_header.p2c, 0, 40, "p2c");
      enc.PutRange(config_.jxl_header.p3ca, 0, 40, "p3ca");
      enc.PutRange(config_.jxl_header.p3cb, 0, 40, "p3cb");
      enc.PutRange(config_.jxl_header.p3cc, 0, 40, "p3cc");
      enc.PutRange(config_.jxl_header.p3cd, 0, 40, "p3cd");
      enc.PutRange(config_.jxl_header.p3ce, 0, 40, "p3ce");
      for (int i : config_.jxl_header.w) enc.PutRange(i, 0, 40, "w");
    }
  }

 private:
  Config config_;
};

// Main compression method for a whole image with values between min_tpv and
// max_tpv inclusive.
template <typename STATE>
WP2Status Compress(const scp::ImagePlanes& img,
                   const std::array<int16_t, 4>& min_tpv,
                   const std::array<int16_t, 4>& max_tpv, int effort,
                   WP2::ANSEncBase& enc, STATE& state,
                   WP2::ANSDictionaries& dicts, EncodeInfo* encode_info) {
  const int num_tokens_ori = encode_info != nullptr ? enc.NumTokens() : 0;

  enc.AddDebugPrefix("GlobalHeader");
  const bool use_jxl = std::is_same<STATE, JXLEncState>::value;
  enc.PutBool(use_jxl, "use_jxl");
  state.EncodeHeader(enc);

  // Process the different planes independently.
  std::array<Encoding, 4> encodings;
  for (int c = img.has_alpha() ? 0 : 1; c < 4; ++c) {
    WP2_CHECK_STATUS(state.Reset());
    WP2_CHECK_STATUS(
        state.CompressPlane(img, c, min_tpv[c], max_tpv[c], &encodings[c]));
  }

  // Store the prediction modes.
  if (!use_jxl) {
    for (int c = img.has_alpha() ? 0 : 1; c < 4; ++c) {
      enc.PutRValue(static_cast<uint32_t>(encodings[c].mode),
                    static_cast<uint32_t>(PredictionMode::Num), "pred_mode");
    }
  }

  // Define the symbols info.
  const int num_contexts = state.GetNumContexts();
  WP2::SymbolsInfo symbols_info;
  for (int c = img.has_alpha() ? 0 : 1; c < 4; ++c) {
    symbols_info.SetInfo(/*sym=*/c, /*min=*/0, /*max=*/1024,
                         /*num_clusters=*/num_contexts,
                         WP2::SymbolsInfo::StorageMethod::kAuto);
  }
  WP2::SymbolRecorder recorder;
  WP2_CHECK_STATUS(recorder.Allocate(symbols_info, /*num_records=*/0));

  WP2::ANSEncNoop noop_enc;
  for (uint32_t c = 0; c < 4; ++c) {
    for (const Token& token : encodings[c].tokens) {
      recorder.Process(/*sym=*/c, /*cluster=*/token.context, token.value,
                       /*label=*/nullptr, &noop_enc);
    }
  }

  // Recode the symbol headers.
  const uint32_t num_pixels = img.width() * img.height();
  WP2::SymbolWriter sw;
  WP2_CHECK_STATUS(sw.Init(symbols_info, effort));
  WP2_CHECK_STATUS(sw.Allocate());
  dicts.DeepClear();
  for (uint32_t s = 0; s < symbols_info.Size(); ++s) {
    for (uint32_t c = 0; c < symbols_info.NumClusters(s); ++c) {
      WP2_CHECK_STATUS(
          sw.WriteHeader(s, c, num_pixels, recorder, "header", &enc, &dicts));
    }
  }
  enc.PopDebugPrefix();

  // Store the actual symbols.
  for (uint32_t c = 0; c < 4; ++c) {
    for (const Token& token : encodings[c].tokens) {
      sw.Process(/*sym=*/c, /*cluster=*/token.context, token.value,
                 /*label=*/"sym", &enc);
    }
  }

  if (encode_info != nullptr && !encode_info->line_tokens.empty()) {
    // Each pixel has the same number of tokens.
    for (size_t y = 0; y < img.height(); ++y) {
      encode_info->line_tokens[y] =
          num_tokens_ori +
          (enc.NumTokens() - num_tokens_ori) * (y + 1) / img.height();
    }
  }

  return WP2_STATUS_OK;
}
}  // namespace scp

WP2Status SCPEncode(const Buffer_s16& img, int effort,
                    const std::array<int32_t, 4>& minima_range,
                    const std::array<int32_t, 4>& maxima_range,
                    WP2::ANSEnc& enc, EncodeInfo* encode_info) {
  if (img.channel_bits != 8) return WP2_STATUS_INVALID_PARAMETER;

  const size_t width = img.width;
  const size_t height = img.height;

  // The code modifies the image for palette so must copy for now.
  scp::ImagePlanes img_planes;
  WP2_CHECK_STATUS(img_planes.Create(width, height, img.has_alpha));
  for (int c = (img.has_alpha ? 0 : 1); c < 4; c++) {
    for (size_t y = 0; y < height; ++y) {
      auto* row = img_planes.PlaneRow(c, y);
      for (size_t x = 0; x < width; ++x) {
        row[x] = img.GetRow(y)[x * 4 + c];
      }
    }
  }
  std::array<int16_t, 4> min_tpv, max_tpv;
  FindExtrema(img.GetRow(0), width, height, img.has_alpha, min_tpv, max_tpv);
  enc.AddDebugPrefix("GlobalHeader");
  for (int c = img.has_alpha ? 0 : 1; c < 4; ++c) {
    enc.PutRange(min_tpv[c], minima_range[c], maxima_range[c], "min_tpv");
    enc.PutRange(max_tpv[c], min_tpv[c], maxima_range[c], "max_tpv");
  }
  enc.PopDebugPrefix();

  float cost_best = std::numeric_limits<float>::max();
  scp::JXLEncState::Config jxl_config_best, jxl_config;
  scp::ClassicalEncState::Config classical_config_best, classical_config;
  bool config_best_is_jxl = false;
  WP2::ANSDictionaries dicts;

  // Try JXL first.
  scp::JXLEncState jxl_state;
  WP2_CHECK_STATUS(jxl_state.Init(width, height));
  for (size_t jxl_header_index = 0; jxl_header_index < scp::kJxlHeaders.size();
       ++jxl_header_index) {
    jxl_config.jxl_header_index = jxl_header_index;
    jxl_state.SetConfig(jxl_config);
    WP2::ANSEncCounter enc_counter;
    WP2_CHECK_STATUS(Compress(img_planes, min_tpv, max_tpv, effort, enc_counter,
                              jxl_state, dicts,
                              /*encode_info=*/nullptr));

    const float cost = enc_counter.GetCost(dicts);
    if (cost < cost_best) {
      cost_best = cost;
      config_best_is_jxl = true;
      jxl_config_best = jxl_config;
    }
  }

  // TODO(vrabaud): enable somehow, that gives 0.5% compression improvement.
  // The range of parameters is too large: interesting sets need to be found.
#if 0
  int i = 0;
  // Only consider one every 1000th combination.
  size_t step = (size_t)std::pow(6, 4) * std::pow(4, 7) / 1000;
  for (int16_t w0 : {0, 4, 8, 12, 16, 20})
    for (int16_t w1 : {0, 4, 8, 12, 16, 20})
      for (int16_t w2 : {0, 4, 8, 12, 16, 20})
        for (int16_t w3 : {0, 4, 8, 12, 16, 20})
          for (int16_t p1c : {4, 8, 12, 16})
            for (int16_t p2c : {4, 8, 12, 16})
              for (int16_t p3ca : {0, 8, 16, 24})
                for (int16_t p3cb : {0, 8, 16, 24})
                  for (int16_t p3cc : {0, 8, 16, 24})
                    for (int16_t p3cd : {0, 8, 16, 24})
                      for (int16_t p3ce : {0, 8, 16, 24}) {
                        if (i++ % step != 0) continue;
                        jxl_config.jxl_header_index = scp::kJxlHeaders.size();
                        jxl_config.jxl_header.p1c = p1c;
                        jxl_config.jxl_header.p2c = p2c;
                        jxl_config.jxl_header.p3ca = p3ca;
                        jxl_config.jxl_header.p3cb = p3cb;
                        jxl_config.jxl_header.p3cc = p3cc;
                        jxl_config.jxl_header.p3cd = p3cd;
                        jxl_config.jxl_header.p3ce = p3ce;
                        jxl_config.jxl_header.w[0] = w0;
                        jxl_config.jxl_header.w[1] = w1;
                        jxl_config.jxl_header.w[2] = w2;
                        jxl_config.jxl_header.w[3] = w3;
                        jxl_state.SetConfig(jxl_config);

                        WP2::ANSEncCounter enc_counter;
                        WP2_CHECK_STATUS(Compress(img_planes, min_tpv, max_tpv,
                                                  effort, enc_counter,
                                                  jxl_state, dicts,
                                                  /*encode_info=*/nullptr));

                        const float cost = enc_counter.GetCost(dicts);
                        if (cost < cost_best) {
                          cost_best = cost;
                          config_best_is_jxl = true;
                          jxl_config_best = jxl_config;
                        }
                      }
#endif
  // Try the classical SCP.
  scp::ClassicalEncState classical_state;
  WP2_CHECK_STATUS(classical_state.Init(width, height));
  // Adding the other modes barely helps so commenting for now.
  static constexpr std::array<scp::PredictionMode, 3> kModes = {
      scp::PredictionMode::Regular /*, scp::PredictionMode::North,*/
      /*scp::PredictionMode::West*/};
  for (scp::PredictionMode pred_mode1 : kModes) {
    classical_config.p_modes[0] = pred_mode1;
    for (scp::PredictionMode pred_mode2 : kModes) {
      classical_config.p_modes[1] = pred_mode2;
      for (scp::PredictionMode pred_mode3 : kModes) {
        classical_config.p_modes[2] = pred_mode3;
        for (int maxerrShift : {-1 /*, 0, 1, 2, 3, 4*/}) {
          classical_config.maxerr_shift = maxerrShift;
          classical_state.SetConfig(classical_config, img_planes);

          WP2::ANSEncCounter enc_counter;
          WP2_CHECK_STATUS(Compress(img_planes, min_tpv, max_tpv, effort,
                                    enc_counter, classical_state, dicts,
                                    /*encode_info=*/nullptr));

          const float cost = enc_counter.GetCost(dicts);
          if (cost < cost_best) {
            cost_best = cost;
            config_best_is_jxl = false;
            classical_config_best = classical_config;
          }
        }
      }
    }
  }

  if (config_best_is_jxl) {
    scp::JXLEncState state;
    WP2_CHECK_STATUS(state.Init(width, height));
    state.SetConfig(jxl_config_best);
    WP2_CHECK_STATUS(Compress(img_planes, min_tpv, max_tpv, effort, enc, state,
                              dicts, encode_info));
  } else {
    scp::ClassicalEncState state;
    WP2_CHECK_STATUS(state.Init(width, height));
    state.SetConfig(classical_config_best, img_planes);
    WP2_CHECK_STATUS(Compress(img_planes, min_tpv, max_tpv, effort, enc, state,
                              dicts, encode_info));
  }

  return WP2_STATUS_OK;
}

}  // namespace WP2L
