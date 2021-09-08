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
//  Predictors
//
// Author: Skal (pascal.massimino@gmail.com)
//

#include "src/common/lossy/predictor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

#include "src/common/constants.h"
#include "src/common/lossy/block.h"
#include "src/dec/symbols_dec.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/block_enc.h"
#include "src/enc/symbols_enc.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"

#define LOG 0

namespace WP2 {

//------------------------------------------------------------------------------
// PredictorVector

void PredictorVector::reset() {
  for (auto& p : *this) delete p;  // we own the predictors
  Vector<Predictor*>::reset();
}

//------------------------------------------------------------------------------
// For debugging only.

std::string GetContextAndBlockPixelsStr(const int16_t context[],
                                        const int16_t context_right[],
                                        const int16_t context_left[],
                                        uint32_t w, uint32_t h,
                                        const int16_t block[], uint32_t step) {
  std::string str;
  str += WP2SPrint("     %4d | ", context[h]);
  for (uint32_t i = 0; i < w; ++i) str += WP2SPrint("%4d ", context[h + 1 + i]);
  str += WP2SPrint("| %4d", context[h + 1 + w]);
  uint32_t num_extra_dashes = 0;
  const uint32_t tr_size = ContextSize(kContextExtendRight, w, h);
  uint32_t tr_i = h + 1 + w + 1;
  if (context_right != nullptr) {
    str += " ";
    // Display the first extended right context values.
    for (uint32_t x = 0; x < 7 && tr_i < tr_size;
         ++x, ++tr_i, ++num_extra_dashes) {
      str += WP2SPrint("%4d ", context_right[tr_i]);
    }
  }
  str += "\n----------+-";
  for (uint32_t i = 0; i < w; ++i) str += "-----";
  str += "+-----";
  for (uint32_t i = 0; i < num_extra_dashes; ++i) str += "-----";
  str += "\n";
  for (uint32_t y = 0; y < h || (context_right != nullptr && tr_i < tr_size);
       ++y) {
    if (y < h) {
      if (context_left != nullptr) {
        str += WP2SPrint("%4d ", context_left[y]);
      } else {
        str += "     ";
      }
      str += WP2SPrint("%4d | ", context[h - 1 - y]);
      for (uint32_t x = 0; x < w; ++x) {
        str += WP2SPrint("%4d ", block[x + y * step]);
      }
      str += WP2SPrint("| %4d", context[h + 1 + w + 1 + y]);
    } else if (y == h) {
      str += WP2SPrint("  ^___| %*s", 12 + w * 5 + 6 - 8, " ");
    } else {
      str += WP2SPrint("%*s", 12 + w * 5 + 6, " ");
    }
    // Display remaining extended right context in small batches on the right.
    if (context_right != nullptr) {
      if (tr_i < tr_size) {
        str += "               ";
        for (uint32_t x = 0; x < 4 && tr_i < tr_size; ++x, ++tr_i) {
          str += WP2SPrint(" %4d", context_right[tr_i]);
        }
      }
    }
    str += "\n";
  }
  return str;
}

const int16_t kFakeContext[kContextSize] = {
    2,  4,  6,  8,   // left
    10,              // top left
    12, 14, 16, 18,  // top
    20,              // top-right
    22, 24, 26, 28   // right
};

//------------------------------------------------------------------------------

// Dummy predictor that always predicts zero.
class ZeroPredictor : public Predictor {
 public:
  void Predict(const CodedBlockBase& cb, Channel channel, bool split_tf,
               uint32_t tf_i, int16_t output[], uint32_t step) const override {
    const BlockSize split_size = GetSplitSize(cb.dim(), split_tf);
    const uint32_t split_w = BlockWidthPix(split_size);
    const uint32_t split_h = BlockHeightPix(split_size);

    for (uint32_t y = 0; y < split_h; ++y) {
      std::fill(output + y * step, output + y * step + split_w, 0);
    }
  }

  std::string GetName() const override { return "zero predictor"; }
  std::string GetFakePredStr() const override { return "zero predictor"; }
  std::string GetPredStr(const CodedBlockBase& cb, Channel channel,
                         bool split_tf, uint32_t tf_i) const override {
    return "zero predictor";
  }
};

//------------------------------------------------------------------------------

void ContextPredictor::Predict(const CodedBlockBase& cb, Channel channel,
                               bool split_tf, uint32_t tf_i, int16_t output[],
                               uint32_t step) const {
  if (tf_i > 0) assert(split_tf);
  const int16_t* const context =
      cb.GetContext(channel, split_tf, tf_i, context_type_);
  const BlockSize split_size = GetSplitSize(cb.dim(), split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  Predict(context, split_w, split_h, output, step);
}

std::string ContextPredictor::GetFakePredStr() const {
  return GetPredStr(kFakeContext, kFakeContext, kFakeContext, kContextSmall,
                    kPredWidth, kPredHeight);
}

std::string ContextPredictor::GetPredStr(const CodedBlockBase& cb,
                                         Channel channel, bool split_tf,
                                         uint32_t tf_i) const {
  const int16_t* const context =
      cb.GetContext(channel, split_tf, tf_i, kContextSmall);
  // Always display the top right context.
  const int16_t* const context_right =
      cb.GetContext(channel, split_tf, tf_i, kContextExtendRight);
  const int16_t* const context_left =
      cb.GetContext(channel, split_tf, tf_i, kContextExtendLeft);
  const BlockSize split_size = GetSplitSize(cb.dim(), split_tf);

  return GetPredStr(context, context_right, context_left, context_type_,
                    BlockWidthPix(split_size), BlockHeightPix(split_size));
}

std::string ContextPredictor::GetPredStr(const int16_t context[],
                                         const int16_t context_right[],
                                         const int16_t context_left[],
                                         ContextType context_type,
                                         uint32_t width,
                                         uint32_t height) const {
  int16_t output[kMaxBlockSizePix2];
  Predict((context_type == kContextExtendRight)  ? context_right
          : (context_type == kContextExtendLeft) ? context_left
                                                 : context,
          width, height, output, /*step=*/width);
  return GetContextAndBlockPixelsStr(context, context_right, context_left,
                                     width, height, output, /*step=*/width);
}

//------------------------------------------------------------------------------

// The DC predictor sets all pixels to the average of top and/or left context.
class DCPredictor : public ContextPredictor {
 public:
  enum Type { kAll, kLeft, kTop };
  explicit DCPredictor(Type type) : type_(type) {}

  std::string GetName() const override {
    const char* const kTypeStr[]{"DC predictor (avg of all context)",
                                 "DC predictor (avg of left context)",
                                 "DC predictor (avg of top context)"};
    return kTypeStr[type_];
  }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    // DC predictors will never predict values outside of the input range so we
    // don't need to pass in accurate min/max bounds.
    constexpr int16_t kMin = std::numeric_limits<int16_t>::min();
    constexpr int16_t kMax = std::numeric_limits<int16_t>::max();
    BasePredictors[BPRED_DC + type_](context, w, h, kMin, kMax, output, step);
  }

  const Type type_;
};

//------------------------------------------------------------------------------

// The MedianDC predictor sets all pixels to the median of top and/or left
// context.
class MedianDCPredictor : public ContextPredictor {
 public:
  enum Type { kAll, kLeft, kTop };
  explicit MedianDCPredictor(Type type) : type_(type) {}

  std::string GetName() const override {
    const char* const kTypeStr[]{"MedianDC predictor (median of all context)",
                                 "MedianDC predictor (median of left context)",
                                 "MedianDC predictor (median of top context)"};
    return kTypeStr[type_];
  }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    // MedianDC predictors will never predict values outside of the input range
    // so we don't need to pass in accurate min/max bounds.
    constexpr int16_t kMin = std::numeric_limits<int16_t>::min();
    constexpr int16_t kMax = std::numeric_limits<int16_t>::max();
    BasePredictors[BPRED_MEDIAN_DC + type_](context, w, h, kMin, kMax, output,
                                            step);
  }

  const Type type_;
};

//------------------------------------------------------------------------------

// TrueMotion predictor.
class TMPredictor : public ContextPredictor {
 public:
  TMPredictor(int16_t min_value, int16_t max_value)
      : min_value_(min_value), max_value_(max_value) {}

  std::string GetName() const override { return "TM predictor"; }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    BasePredictors[BPRED_TM](context, w, h, min_value_, max_value_, output,
                             step);
  }

 private:
  const int16_t min_value_;
  const int16_t max_value_;
};

//------------------------------------------------------------------------------

// Smooth predictor that interpolates pixels horizontally, vertically or a mix
// of both.
class SmoothPredictor : public ContextPredictor {
 public:
  enum class SmoothType {
    k2DSmooth,
    kVerticalSmooth,
    kHorizontalSmooth,
    kNumSmoothType
  };
  explicit SmoothPredictor(SmoothType type) : type_(type) {}

  std::string GetName() const override {
    const char* const kTypeStr[]{"2D", "vertical", "horizontal"};
    std::string name = "smooth predictor (";
    name += kTypeStr[(int)type_];
    name += ")";
    return name;
  }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    // Smooth predictors will never predict values outside of the input range so
    // we don't need to pass in accurate min/max bounds.
    constexpr int16_t kMin = std::numeric_limits<int16_t>::min();
    constexpr int16_t kMax = std::numeric_limits<int16_t>::max();
    const BasePredictor p = (type_ == SmoothType::k2DSmooth)
                                ? BPRED_SMOOTH
                                : (type_ == SmoothType::kVerticalSmooth)
                                      ? BPRED_SMOOTH_V
                                      : BPRED_SMOOTH_H;
    BasePredictors[p](context, w, h, kMin, kMax, output, step);
  }

  const SmoothType type_;
};

//------------------------------------------------------------------------------

static uint32_t GetMaxAngleDelta(Channel channel) {
  return (channel == kUChannel || channel == kVChannel)
             ? kDirectionalMaxAngleDeltaUV
             : kDirectionalMaxAngleDeltaYA;
}

// Angle predictor.
class AnglePredictor : public ContextPredictor {
 public:
  // Use AV1 convention to name angle predictors.
  enum class Type {
    // Angles must be in the same order as in Predictors::Pred
    D23_PRED,
    D45_PRED,
    D67_PRED,
    V_PRED,  // 90 degrees
    D113_PRED,
    D135_PRED,
    D157_PRED,
    H_PRED,  // 180 degrees
    D203_PRED,
    D225_PRED,
    Num
  };
  AnglePredictor(Type type, Channel channel, uint32_t sub_mode,
                 int16_t angle_delta, int16_t min_value, int16_t max_value)
      : min_value_(min_value),
        max_value_(max_value),
        sub_mode_(sub_mode),
        angle_idx_(AngleIdx(type, channel, angle_delta)) {
    context_type_ = GetContextType(angle_idx_);
    static_assert((uint32_t)Type::Num == kAnglePredNum,
                  "Incorrect kAnglePredNum");
  }

  // Returns the angle index for precalculated angles.
  // Index goes from 0 (12.86 degrees) to 69 (234.64 degrees).
  static uint8_t AngleIdx(Type type, Channel channel, int16_t angle_delta) {
    constexpr int16_t max_delta = kDirectionalMaxAngleDeltaYA;
    if (channel == kUChannel || channel == kVChannel) {
      // Note that the U/V prediction angles are not exactly even spaced.
      // But they map to already existing angles for Y/A.
      // TODO(skal): should the YA / UV angles be made independent?
      angle_delta *= 2;
    }
    const int idx = (int)type * (2 * max_delta + 1) + max_delta + angle_delta;
    assert(idx >= 0 && idx < (int)kNumDirectionalAngles);
    return (uint8_t)idx;
  }

  bool IsAngle(float* const angle) const override {
    if (angle != nullptr) *angle = 22.5f + 22.5f * ((int)angle_idx_ - 3) / 7.f;
    return true;
  }

  uint32_t WriteParams(const CodedBlockBase& cb, Channel channel,
                       SymbolManager* const sm,
                       ANSEncBase* const enc) const override {
    if (channel == kVChannel) return 0;
    (void)cb;
    (void)sm;
    // By storing the angle step, we are then able, at decoding time, to figure
    // out the sub-mode of an angle predictor.
    const uint32_t max_delta = GetMaxAngleDelta(channel);
    enc->PutRange(sub_mode_, 0, 2 * max_delta, "sub_mode");
    return sub_mode_;
  }

  uint32_t ReadParams(CodedBlockBase* const cb, Channel channel,
                      SymbolReader* const sm,
                      ANSDec* const dec) const override {
    if (channel == kVChannel) return 0;
    (void)sm;
    // TODO(vrabaud) explore using a symbol for the step, like in AV1.
    const uint32_t max_delta = GetMaxAngleDelta(channel);
    return dec->ReadRange(0, 2 * max_delta, "sub_mode");
  }

  std::string GetName() const override {
    float angle;
    if (!IsAngle(&angle)) assert(false);
    const int angle_deg = std::round(angle);
    return "angle predictor (" + std::to_string(angle_deg) +
           " degrees, idx=" + std::to_string((int)angle_idx_) + ")";
  }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    const uint32_t context_size = ContextSize(context_type_, w, h);
    for (uint32_t i = 0; i < context_size; ++i) {
      assert(context[i] >= min_value_ && context[i] <= max_value_);
    }
    // SimpleAnglePredictor is faster and currently gives better results than
    // BaseAnglePredictor.
    const uint32_t w_idx = TrfLog2[w];
    const uint32_t h_idx = TrfLog2[h];
    SimpleAnglePredictor(angle_idx_, context, w_idx, h_idx, output, step);
  }
  const int16_t min_value_;
  const int16_t max_value_;

 private:
  const uint32_t sub_mode_;   // used by encoder during WriteParams()
  uint32_t angle_idx_;        // global angle index, as calculated by AngleIdx()
};

//------------------------------------------------------------------------------

// Fuse predictor.
class FusePredictor : public ContextPredictor {
 public:
  FusePredictor(float strength, int16_t min_value, int16_t max_value)
      : strength_(strength), min_value_(min_value), max_value_(max_value) {
    PrecomputeLargeWeightTable(strength_, table_);
  }

  std::string GetName() const override {
    std::string name = "fuse predictor (strength: ";
    name += std::to_string(strength_);
    name += ")";
    return name;
  }

 protected:
  void Predict(const int16_t context[], uint32_t w, uint32_t h,
               int16_t output[], uint32_t step) const override {
    BaseFusePredictor(table_, context, w, h, output, step, min_value_,
                      max_value_);
  }

  const float strength_;
  LargeWeightTable table_;
  const int16_t min_value_;
  const int16_t max_value_;
};

//------------------------------------------------------------------------------

// Extrapolates a gradient from the bottom-left, top-left and top-right corners.
// This is sometimes more powerful than other predictors because it can guess
// values outside the ones in the context.
class GradientPredictor : public ContextPredictor {
 public:
  GradientPredictor(int16_t min_value, int16_t max_value)
      : min_value_(min_value), max_value_(max_value) {}

  std::string GetName() const override { return "gradient predictor"; }

 protected:
  void Predict(const int16_t c[], uint32_t w, uint32_t h, int16_t output[],
               uint32_t step) const override {
    // Average the three known (or extrapolated) corners.
    const int32_t bottom_left = DivRound(c[0] + c[1] + c[2], 3);
    const int32_t top_left =
        DivRound(c[h - 2] + c[h - 1] + c[h] + c[h + 1] + c[h + 2], 5);
    const int32_t top_right =
        DivRound(c[h + w - 1] + c[h + w] + c[h + w + 1], 3);
    // Extrapolate the middle value and from it, the bottom-right corner.
    const int32_t middle = DivRound(bottom_left + top_right, 2);
    const int32_t bottom_right =
        Clamp(middle - top_left + middle, min_value_, max_value_);

    // Create a gradient by bidimensional interpolation.
    const int32_t max_x = (int32_t)w - 1, max_y = (int32_t)h - 1;
    for (int32_t y = 0; y <= max_y; ++y) {
      const int32_t left =
          DivRound(top_left * (max_y - y) + bottom_left * y, max_y);
      const int32_t right =
          DivRound(top_right * (max_y - y) + bottom_right * y, max_y);
      for (int32_t x = 0; x <= max_x; ++x) {
        output[x] = DivRound(left * (max_x - x) + right * x, max_x);
        assert(output[x] >= min_value_ && output[x] <= max_value_);
      }
      output += step;
    }
  }

  const int32_t min_value_;
  const int32_t max_value_;
};

//------------------------------------------------------------------------------
// Alpha predictor.

void AdjustAlphaPrediction(const CodedBlockBase& cb, CSPTransform transform,
                           int16_t output[], uint32_t step);

// Predictor for the alpha plane, that wraps a normal predictor, and adjusts
// its prediction to take into account the fact RGB is premultiplied, therefore
// alpha >= max(r, g, b)
// This predictor assumes that YUV is encoded/reconstructed before alpha.
// The supplied Predictor is not owned, so we don't delete it.
class AlphaPredictor : public Predictor {
 public:
  // Takes ownership of 'pred'.
  AlphaPredictor(Predictor* const pred, const CSPTransform& transform)
      : pred_(pred), transform_(transform) {}

  ~AlphaPredictor() override { delete pred_; }

  void Predict(const CodedBlockBase& cb, Channel channel, bool split_tf,
               uint32_t tf_i, int16_t output[], uint32_t step) const override {
    assert(!split_tf);
    assert(channel == kAChannel);  // Should only be used for alpha.
    pred_->Predict(cb, channel, /*split_tf=*/false, /*tf_i=*/0, output, step);
    AdjustAlphaPrediction(cb, transform_, output, step);
  }

  std::string GetName() const override {
    return "alpha predictor based on " + pred_->GetName();
  }
  std::string GetFakePredStr() const override {
    printf("alpha predictor based on:\n");
    return pred_->GetFakePredStr();
  }
  std::string GetPredStr(const CodedBlockBase& cb, Channel channel,
                         bool split_tf, uint32_t tf_i) const override {
    assert(!split_tf);
    std::string str = pred_->GetPredStr(cb, channel, split_tf, tf_i);
    str += "==> adjusted by alpha predictor:\n";
    int16_t output[kMaxBlockSizePix2];
    Predict(cb, channel, /*split_tf=*/false, /*tf_i=*/0, output, cb.w_pix());
    const int16_t* const context = cb.GetContext(channel, kContextSmall);
    str += GetContextAndBlockPixelsStr(context, /*context_right=*/nullptr,
                                       /*context_left=*/nullptr, cb.w_pix(),
                                       cb.h_pix(), output, cb.w_pix());
    return str;
  }

  uint32_t mode() const override { return pred_->mode(); }
  bool ComputeParams(CodedBlock* const cb, Channel channel) const override {
    return pred_->ComputeParams(cb, channel);
  }
  uint32_t WriteParams(const CodedBlockBase& cb, Channel channel,
                       SymbolManager* const sm,
                       ANSEncBase* const enc) const override {
    return pred_->WriteParams(cb, channel, sm, enc);
  }
  uint32_t ReadParams(
      CodedBlockBase* const cb, Channel channel,
      SymbolReader* const sm, ANSDec* const dec) const override {
    return pred_->ReadParams(cb, channel, sm, dec);
  }
  bool DependsOnLuma() const override { return true; }
  bool IsAngle(float* const angle) const override {
    return pred_->IsAngle(angle);
  }

 protected:
  Predictor* const pred_;
  const CSPTransform& transform_;
};

//------------------------------------------------------------------------------

uint32_t Predictors::GetMaxMode() const {
  return preds_no_angle_.size() + kAnglePredNum - 1;
}

const Predictor* Predictors::GetPred(uint32_t mode, uint32_t sub_mode) const {
  const uint32_t index = main_modes_[mode];
  assert(index + sub_mode < preds_.size());
  assert(mode == GetMaxMode() || (index + sub_mode < main_modes_[mode + 1]));
  assert(preds_[index]->mode() == preds_[index + sub_mode]->mode());
  return preds_[index + sub_mode];
}

//------------------------------------------------------------------------------

Predictor* Predictors::RecordPredictor(
    Predictor* pred, uint32_t mode, const CSPTransform* const transform) {
  if (pred == nullptr) return nullptr;
  pred->SetMode(mode);
  if (channel_ == kAChannel) {
    AlphaPredictor* const alpha_pred =
        new (WP2Allocable::nothrow) AlphaPredictor(pred, *transform);
    if (alpha_pred == nullptr) {
      delete pred;
      return nullptr;
    }
    pred = alpha_pred;
  }
  if (!preds_.push_back(pred)) {
    delete pred;
    return nullptr;
  }
  return pred;
}

Predictor* MakeNonAnglePredictor(const Predictors::Pred predictor_type,
                                 int16_t min_value, int16_t max_value) {
  switch (predictor_type) {
    case Predictors::Pred::kDcAll:
      return new (WP2Allocable::nothrow) DCPredictor(DCPredictor::kAll);
    case Predictors::Pred::kDcLeft:
      return new (WP2Allocable::nothrow) DCPredictor(DCPredictor::kLeft);
    case Predictors::Pred::kDcTop:
      return new (WP2Allocable::nothrow) DCPredictor(DCPredictor::kTop);
    case Predictors::Pred::kMedianDcAll:
      return new (WP2Allocable::nothrow)
          MedianDCPredictor(MedianDCPredictor::kAll);
    case Predictors::Pred::kMedianDcLeft:
      return new (WP2Allocable::nothrow)
          MedianDCPredictor(MedianDCPredictor::kLeft);
    case Predictors::Pred::kMedianDcTop:
      return new (WP2Allocable::nothrow)
          MedianDCPredictor(MedianDCPredictor::kTop);
    case Predictors::Pred::kSmooth2D:
      return new (WP2Allocable::nothrow)
          SmoothPredictor(SmoothPredictor::SmoothType::k2DSmooth);
    case Predictors::Pred::kSmoothVertical:
      return new (WP2Allocable::nothrow)
          SmoothPredictor(SmoothPredictor::SmoothType::kVerticalSmooth);
    case Predictors::Pred::kSmoothHorizontal:
      return new (WP2Allocable::nothrow)
          SmoothPredictor(SmoothPredictor::SmoothType::kHorizontalSmooth);
    case Predictors::Pred::kTrueMotion:
      return new (WP2Allocable::nothrow) TMPredictor(min_value, max_value);
    case Predictors::Pred::kGradient:
      return new (WP2Allocable::nothrow)
          GradientPredictor(min_value, max_value);
    case Predictors::Pred::kCfl:
      return new (WP2Allocable::nothrow) CflPredictor(min_value, max_value);
    case Predictors::Pred::kSignalingCfl:
      return new (WP2Allocable::nothrow)
          SignalingCflPredictor(min_value, max_value);
    case Predictors::Pred::kZero:
      return new (WP2Allocable::nothrow) ZeroPredictor();
    default:
      // It will fail if you try to make an angle predictor with this function.
      assert(false);
  }
  return nullptr;
}

namespace {
bool IsAnglePredictor(const Predictors::Pred predictor_type) {
  switch (predictor_type) {
    case Predictors::Pred::kAngle23:
    case Predictors::Pred::kAngle45:
    case Predictors::Pred::kAngle67:
    case Predictors::Pred::kAngle90:
    case Predictors::Pred::kAngle113:
    case Predictors::Pred::kAngle135:
    case Predictors::Pred::kAngle157:
    case Predictors::Pred::kAngle180:
    case Predictors::Pred::kAngle203:
    case Predictors::Pred::kAngle225:
      return true;
    default:
      return false;
  }
}
}  // namespace

WP2Status Predictors::FillImpl(const Pred* const mapping, uint32_t num_modes,
                               int16_t min_value, int16_t max_value,
                               const CSPTransform* const transform) {
  const int max_delta = GetMaxAngleDelta(channel_);
  WP2_CHECK_ALLOC_OK(preds_.reserve(
      num_modes + (2 * max_delta) * (uint32_t)AnglePredictor::Type::Num));
  assert(num_modes >= (uint32_t)AnglePredictor::Type::Num);
  // DC or MedianDC all should be first (to make it easy to force DC or Median
  // DC for all blocks, which is the same as using the first predictor).
  assert(mapping[0] == Pred::kDcAll || mapping[0] == Pred::kMedianDcAll);
  WP2_CHECK_ALLOC_OK(
      preds_no_angle_.reserve(num_modes - (uint32_t)AnglePredictor::Type::Num));

  uint32_t angle_type = 0;
  uint32_t i = 0;
  for (; i < num_modes; ++i) {
    Pred p = mapping[i];
    WP2::Predictor* pred = nullptr;

    main_modes_[i] = preds_.size();  // for fast lookup in GetPred()

    if (IsAnglePredictor(p)) {
      for (int32_t sub_mode = 0; sub_mode <= 2 * max_delta; ++sub_mode) {
        // we map the sub-mode to a delta-angle around the main angle.
        // It mostly affects the RD-opt search order, since the coding
        // will be the same in the bitstream (we're using a Range).
        const int32_t angle_step =
            (sub_mode & 1) ? -((sub_mode + 1) >> 1) : (sub_mode >> 1);
        // We assume they're in the same order in ContextPredictor and in
        // AnglePredictor::Type
        pred = new (WP2Allocable::nothrow)
            AnglePredictor((AnglePredictor::Type)angle_type, channel_, sub_mode,
                           angle_step, min_value, max_value);
        pred = RecordPredictor(pred, i, transform);
        WP2_CHECK_ALLOC_OK(pred != nullptr);
        // Store the main angle.
        if (angle_step == 0) preds_main_angle_[angle_type] = pred;
      }
      ++angle_type;
    } else {
      pred = MakeNonAnglePredictor(p, min_value, max_value);
      pred = RecordPredictor(pred, i, transform);
      WP2_CHECK_ALLOC_OK(pred != nullptr);
      WP2_CHECK_ALLOC_OK(preds_no_angle_.push_back(pred));
    }
  }
  assert(i <= GetMaxMode() + 1);
  for (; i < ArraySize(main_modes_); ++i) {
    main_modes_[i] = 0;  // safety
  }

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status YPredictors::Fill(int16_t min_value, int16_t max_value) {
  constexpr Pred kYPreds[] = {Pred::kDcAll,
                              Pred::kDcLeft,
                              Pred::kDcTop,
                              Pred::kZero,
                              Pred::kSmooth2D,
                              Pred::kSmoothVertical,
                              Pred::kSmoothHorizontal,
                              Pred::kTrueMotion,
                              Pred::kGradient,
                              Pred::kAngle23,
                              Pred::kAngle45,
                              Pred::kAngle67,
                              Pred::kAngle90,
                              Pred::kAngle113,
                              Pred::kAngle135,
                              Pred::kAngle157,
                              Pred::kAngle180,
                              Pred::kAngle203,
                              Pred::kAngle225};

  STATIC_ASSERT_ARRAY_SIZE(kYPreds, kYPredModeNum);
  WP2_CHECK_STATUS(FillImpl(kYPreds, kYPredModeNum, min_value, max_value,
                            /*transform=*/nullptr));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status APredictors::Fill(int16_t min_value, int16_t max_value,
                            const CSPTransform* const transform) {
  constexpr Pred kAPreds[] = {Pred::kDcAll,          Pred::kSmooth2D,
                              Pred::kSmoothVertical, Pred::kSmoothHorizontal,
                              Pred::kTrueMotion,     Pred::kAngle23,
                              Pred::kAngle45,        Pred::kAngle67,
                              Pred::kAngle90,        Pred::kAngle113,
                              Pred::kAngle135,       Pred::kAngle157,
                              Pred::kAngle180,       Pred::kAngle203,
                              Pred::kAngle225,       Pred::kCfl,
                              Pred::kSignalingCfl,   Pred::kZero};

  assert(transform != nullptr);
  STATIC_ASSERT_ARRAY_SIZE(kAPreds, kAPredModeNum);
  WP2_CHECK_STATUS(
      FillImpl(kAPreds, kAPredModeNum, min_value, max_value, transform));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status UVPredictors::Fill(int16_t min_value, int16_t max_value) {
  constexpr Pred kUVPreds[] = {
      Pred::kDcAll,    Pred::kCfl,      Pred::kSignalingCfl, Pred::kSmooth2D,
      Pred::kAngle23,  Pred::kAngle45,  Pred::kAngle67,      Pred::kAngle90,
      Pred::kAngle113, Pred::kAngle135, Pred::kAngle157,     Pred::kAngle180,
      Pred::kAngle203, Pred::kAngle225};

  STATIC_ASSERT_ARRAY_SIZE(kUVPreds, kUVPredModeNum);
  WP2_CHECK_STATUS(FillImpl(kUVPreds, kUVPredModeNum, min_value, max_value,
                            /*transform=*/nullptr));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Chroma-From-Luma

CflPredictor::CflPredictor(int16_t min_value, int16_t max_value)
    : min_value_(min_value), max_value_(max_value) {}

void CflPredictor::LinearRegression(const int16_t* const luma,
                                    const int16_t* const chroma, uint32_t size,
                                    int32_t* const a, int32_t* const b) const {
  int32_t num_values = 0;
  int32_t l_sum = 0;
  int32_t uv_sum = 0;
  int32_t l_uv_sum = 0;
  int32_t l_l_sum = 0;
  for (uint32_t i = 0; i < size; ++i) {
    if (luma[i] == CodedBlock::kMissing) continue;
    ++num_values;
    const int32_t l = luma[i], uv = chroma[i];
    l_sum += l;
    uv_sum += uv;
    l_uv_sum += l * uv;
    l_l_sum += l * l;
  }
  if (num_values == 0) {
    *a = *b = 0;
    return;
  }
  const int64_t num = (int64_t)l_uv_sum * num_values - (int64_t)l_sum * uv_sum;
  const int64_t den = (int64_t)l_l_sum * num_values - (int64_t)l_sum * l_sum;
  int32_t A = 0;
  int32_t B = DivRound(LeftShift(uv_sum, kCflFracBits), num_values);
  if (num != 0 && den != 0) {
    A = DivRound(LeftShift(num, kCflFracBits), den);  // fits in 64b precision
    B -= DivRound<int64_t>((int64_t)A * l_sum, num_values);
  }
  B += (1 << kCflFracBits >> 1);  // include rounding constant

  // Tighten the scaling/offset value to avoid obvious overflows.
  // Warning! alpha-from-chroma heavily relies on saturating to [0..255],
  // and generate large 'B' constants. We must not clip 'B' with min/max_value_
  // but with the max codable range (kYuvMaxPrec = 9b) to retain this property.
  const int32_t max_range = 2 << (kCflFracBits + kYuvMaxPrec);
  if (std::abs(B) > max_range) {
    A = 0;
    B = DivRound(LeftShift(uv_sum, kCflFracBits), num_values);
  }
  *a = ClampToSigned(A, kCflABits);
  *b = ClampToSigned(B, kCflBBits);
}

void CflPredictor::ContextLinearRegression(Channel channel,
                                           const CodedBlockBase& cb,
                                           int32_t* const a,
                                           int32_t* const b) const {
  const int16_t* const luma_context =
      cb.GetContext(kYChannel, kContextSmallNoFillIn);
  const int16_t* const chroma_context =
      cb.GetContext(channel, kContextSmallNoFillIn);
  const uint32_t context_size =
      ContextSize(kContextSmall, cb.w_pix(), cb.h_pix());
  LinearRegression(luma_context, chroma_context, context_size, a, b);
}

void CflPredictor::BlockLinearRegression(Channel channel, const CodedBlock& cb,
                                         int32_t* const a,
                                         int32_t* const b) const {
  const Plane16& luma_plane = cb.out_.GetChannel(kYChannel);
  const Plane16& chroma_plane = cb.in_.GetChannel(channel);
  int16_t luma[kMaxBlockSizePix2], chroma[kMaxBlockSizePix2];

  for (uint32_t k = 0, y = 0; y < cb.h_pix(); ++y) {
    const int16_t* const y_src = luma_plane.Row(y);
    const int16_t* const c_src = chroma_plane.Row(y);
    for (uint32_t x = 0; x < cb.w_pix(); ++x, ++k) {
      luma[k] = y_src[x];
      chroma[k] = c_src[x];
    }
  }
  LinearRegression(luma, chroma, cb.h_pix() * cb.w_pix(), a, b);
}

void CflPredictor::DoPredict(const CodedBlockBase& cb, Channel channel,
                             uint32_t x_start, uint32_t y_start, uint32_t w,
                             uint32_t h, int16_t* output, uint32_t step) const {
  int32_t a, b;
  ContextLinearRegression(channel, cb, &a, &b);
  const Plane16& luma = cb.GetContextPlane(kYChannel);
  for (uint32_t y = 0; y < h; ++y) {
    const int16_t* const src = &luma.At(x_start, y_start + y);
    CflPredict(src, (int16_t)a, b, output, min_value_, max_value_, w);
    output += step;
  }
}

void CflPredictor::Predict(const CodedBlockBase& cb, Channel channel,
                           bool split_tf, uint32_t tf_i, int16_t output[],
                           uint32_t step) const {
  assert(!split_tf);
  assert(channel != kYChannel);
  DoPredict(cb, channel, /*x_start=*/0, /*y_start*/ 0, cb.w_pix(), cb.h_pix(),
            output, step);
}

std::string CflPredictor::GetName() const {
  return "chroma-from-luma predictor";
}

std::string CflPredictor::GetFakePredStr() const {
  return "(fake prediction not supported)\n";
}

void CflPredictor::DisplaySamples(const CodedBlockBase& cb, Channel channel,
                                  std::string* const str) const {
#if defined(WP2_BITTRACE)
  const Plane16& luma = cb.GetContextPlane(kYChannel);
  int16_t output[kMaxBlockSizePix2];
  Predict(cb, channel, /*split_tf=*/false, /*tf_i=*/0, output,
          kMaxBlockSizePix);

  // Display subset of values to fit screen.
  *str += WP2SPrint("       LUMA          |            chroma\n");
  for (uint32_t j = 0; j < kPredHeight; ++j) {
    for (uint32_t i = 0; i < kPredWidth; ++i) {
      *str += WP2SPrint("%4d ", luma.At(i, j));
    }
    *str += " | ";
    for (uint32_t i = 0; i < kPredWidth; ++i) {
      *str += WP2SPrint("%4d ", output[i + j * kMaxBlockSizePix]);
    }
    *str += "\n";
  }
  if (cb.w() > 1 || cb.h() > 1) *str += "(cropped)\n";
#endif
}

std::string CflPredictor::GetPredStr(const CodedBlockBase& cb, Channel channel,
                                     bool split_tf, uint32_t tf_i) const {
  assert(!split_tf);
  int32_t a, b;
  ContextLinearRegression(channel, cb, &a, &b);
  std::string str =
      WP2SPrint("Chroma = %.2f * Luma + %.2ff  [(%d * Luma + %2d) >> %d]\n",
                kCflNorm * a, kCflNorm * b, a, b, kCflFracBits);
  DisplaySamples(cb, channel, &str);
  return str;
}

//------------------------------------------------------------------------------

SignalingCflPredictor::SignalingCflPredictor(int16_t min_value,
                                             int16_t max_value)
    : CflPredictor(min_value, max_value) {}

void SignalingCflPredictor::GetParams(const CodedBlockBase& cb, Channel channel,
                                      uint32_t w, uint32_t h,
                                      int32_t* a, int32_t* b) const {
  const Plane16& luma = cb.GetContextPlane(kYChannel);
  int32_t predicted_a, predicted_b;
  ContextLinearRegression(channel, cb, &predicted_a, &predicted_b);
  const int32_t delta_a =
      ChangePrecision(cb.cfl_[channel - 1], kIOFracBits, kCflFracBits);

  int32_t luma_sum = 0;
  for (uint32_t y = 0; y < h; ++y) {
    const int16_t* const y_src = luma.Row(y);
    for (uint32_t x = 0; x < w; ++x) luma_sum += y_src[x];
  }
  predicted_b -= DivRound<int64_t>((int64_t)delta_a * luma_sum, w * h);
  predicted_a += delta_a;
  *a = ClampToSigned(predicted_a, kCflABits);
  *b = ClampToSigned(predicted_b, kCflBBits);
}

void SignalingCflPredictor::DoPredict(const CodedBlockBase& cb, Channel channel,
                                      uint32_t, uint32_t, uint32_t w,
                                      uint32_t h, int16_t* output,
                                      uint32_t step) const {
  const Plane16& luma = cb.GetContextPlane(kYChannel);
  int32_t a, b;
  GetParams(cb, channel, w, h, &a, &b);
  for (uint32_t y = 0; y < h; ++y) {
    const int16_t* const y_src = luma.Row(y);
    CflPredict(y_src, a, b, output, min_value_, max_value_, w);
    output += step;
  }
}

bool SignalingCflPredictor::ComputeParams(CodedBlock* const cb,
                                          Channel channel) const {
  int32_t predicted_a, best_a, predicted_b, best_b;
  ContextLinearRegression(channel, *cb, &predicted_a, &predicted_b);
  BlockLinearRegression(channel, *cb, &best_a, &best_b);
  best_a = ChangePrecision(best_a - predicted_a, kCflFracBits, kIOFracBits);
  cb->cfl_[channel - 1] = ClampToSigned(best_a, kIOBits);
  return (cb->cfl_[channel - 1] != 0);
}

uint32_t SignalingCflPredictor::WriteParams(const CodedBlockBase& cb,
                                            Channel channel,
                                            SymbolManager* const sm,
                                            ANSEncBase* const enc) const {
  sm->Process(kSymbolCflSlope, cb.cfl_[channel - 1], "cfl_slope", enc);
  return 0;
}

uint32_t SignalingCflPredictor::ReadParams(CodedBlockBase* const cb,
                                           Channel channel,
                                           SymbolReader* const sr,
                                           ANSDec* const dec) const {
  cb->cfl_[channel - 1] = sr->Read(kSymbolCflSlope, "cfl_slope");
  return 0;
}

std::string SignalingCflPredictor::GetName() const {
  return "signaling chroma-from-luma";
}

std::string SignalingCflPredictor::GetPredStr(const CodedBlockBase& cb,
                                              Channel channel, bool split_tf,
                                              uint32_t tf_i) const {
  assert(!split_tf);
  int32_t predicted_a, a, predicted_b, b;
  ContextLinearRegression(channel, cb, &predicted_a, &predicted_b);
  GetParams(cb, channel, cb.w(), cb.h_pix(), &a, &b);

  std::string str;
  str += WP2SPrint("Predicted a: %.2f, with correction: %.2f (param:%d)\n",
                   kCflNorm * predicted_a, kCflNorm * a, cb.cfl_[channel - 1]);
  str += WP2SPrint("Predicted b: %.2f, with correction: %.2f\n",
                   kCflNorm * predicted_b, kCflNorm * b);
  str += WP2SPrint("Chroma = %.2f * Luma + %.2ff  [(%d * Luma + %2d) >> %d]\n",
                   kCflNorm * a, kCflNorm * b, a, b, kCflFracBits);
  DisplaySamples(cb, channel, &str);
  return str;
}

//------------------------------------------------------------------------------

void AdjustAlphaPrediction(const CodedBlockBase& cb, CSPTransform transform,
                           int16_t output[], uint32_t step) {
  for (uint16_t j = 0; j < cb.h_pix(); ++j) {
    const int16_t* const y_src = cb.GetContextPlane(kYChannel).Row(j);
    const int16_t* const u_src = cb.GetContextPlane(kUChannel).Row(j);
    const int16_t* const v_src = cb.GetContextPlane(kVChannel).Row( j);
    for (uint16_t i = 0; i < cb.w_pix(); ++i) {
      int16_t r, g, b;
      transform.YuvToRgb8(y_src[i], u_src[i], v_src[i], &r, &g, &b);

      // Alpha is larger than the largest R, G or B value.
      // TODO(maryla): since this is based on reconstructed RGB, there might be
      // some noise. Add a safety margin at lower quality?
      const int16_t min_alpha_value = std::max({r, g, b, (int16_t)0});
      output[j * step + i] = std::max(output[j * step + i], min_alpha_value);
    }
  }
}

// TODO(vrabaud) Remove
WP2Status InitAlphaPredictors(const CSPTransform& transform,
                              APredictors* const preds) {
  const uint32_t kMinValue = 0;
  const uint32_t kMaxValue = kAlphaMax;

  preds->reset();
  WP2_CHECK_STATUS(preds->Fill(kMinValue, kMaxValue, &transform));
  assert(preds->size() == kAPredNum);

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

}  // namespace WP2
