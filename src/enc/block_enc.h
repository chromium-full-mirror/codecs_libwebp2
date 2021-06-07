// Copyright 2021 Google LLC
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
//   CodedBlock, for encoding
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WP2_ENC_BLOCK_ENC_H_
#define WP2_ENC_BLOCK_ENC_H_

#include "src/common/lossy/block.h"
#include "src/wp2/encode.h"

namespace WP2 {

class SymbolReader;
class SymbolCounter;
class UpdatingSymbolCounter;
class SymbolRecorder;

//------------------------------------------------------------------------------
// Context passed around and updated once we know the right parameters for each
// block. Currently quite empty but can be removed or extended when needed.

class BlockContext {
 public:
  // 'use_aom' defines the usage of AOM residuals.
  WP2Status Init(bool use_aom) {
    use_aom_ = use_aom;
    return WP2_STATUS_OK;
  }
  WP2Status CopyFrom(const BlockContext& other) {
    use_aom_ = other.use_aom_;
    return WP2_STATUS_OK;
  }
  void Reset() {}
  bool use_aom() const { return use_aom_; }

 private:
  bool use_aom_ = false;
};

//------------------------------------------------------------------------------

// Class containing several SymbolCounters for re-use (to avoid too many
// mallocs).
class Counters : public WP2Allocable {
 public:
  WP2Status Init(uint32_t effort, const SymbolRecorder& recorder);
  WP2Status CopyFrom(const Counters& other, const SymbolRecorder& recorder);
  const SymbolRecorder* recorder() const { return recorder_; }
  // Returns a pointer to the counter with symbols related to transforms.
  SymbolCounter* transform() { return transform_.get(); }
  // Returns a pointer to the counter with symbols related to predictors.
  SymbolCounter* predictor() { return predictor_.get(); }
  // Returns a pointer to the counter with symbols related to residuals.
  SymbolCounter* residuals() { return residuals_; }
  // Returns a pointer to the counter with symbols related to AOM residual.
  SymbolCounter* residuals_aom() { return residuals_aom_; }

 private:
  static constexpr uint32_t kSlowCounterEffortThreshold = 5;
  uint32_t effort_;
  std::unique_ptr<SymbolCounter> transform_;
  std::unique_ptr<SymbolCounter> predictor_;
  // Either the fast or slow version is used depending on 'effort'.
  std::unique_ptr<SymbolCounter> residuals_fast_;
  std::unique_ptr<UpdatingSymbolCounter> residuals_slow_;
  SymbolCounter* residuals_;
  std::unique_ptr<SymbolCounter> residuals_aom_fast_;
  std::unique_ptr<UpdatingSymbolCounter> residuals_aom_slow_;
  SymbolCounter* residuals_aom_;
  const SymbolRecorder* recorder_ = nullptr;
};

//------------------------------------------------------------------------------
// CodedBlock

// Class for encoder.
class CodedBlock : public CodedBlockBase {
 public:
  CodedBlock() = default;
  CodedBlock(const CodedBlock&) = delete;
  CodedBlock(CodedBlock&&) = default;
  ~CodedBlock() = default;

  // Set the views on the source and destination buffers based on 'blk_'.
  // Position and size must be set prior to calling these.
  void SetSrcInput(const YUVPlane& in);  // Used by GetDisto().

  // Assigns the best of the 'transforms' (based on full pred+tf+quant+recons
  // score) to the CodingParams of 'channel'. The final predictor must be
  // selected before calling FindBestTf().
  WP2Status FindBestTf(const EncoderConfig& config, const Rectangle& tile_rect,
                       const Segment& segment, const BlockContext& context,
                       Channel channel, uint32_t num_channels, bool reduced,
                       const Vector<TransformPair>& transforms,
                       Counters* const counters);

  // Assigns the best of the 'preds' (based on pred score only) to the
  // CodingParams of 'channel'. 'improvement' controls how much the best
  // score so far must be bestened during the search, on the range [0..256].
  void FindBestPred(const Predictors& preds, Channel channel,
                    uint32_t improvement = 256);
  // Assigns the best pair of 'preds/transforms' (based on full
  // pred+tf+quant+recons score) to the CodingParams of 'channel'.
  WP2Status FindBestPredTf(const EncoderConfig& config,
                           const Rectangle& tile_rect, const Predictors& preds,
                           const Segment& segment, const BlockContext& context,
                           Channel channel, uint32_t num_channels, bool reduced,
                           const Vector<TransformPair>& transforms,
                           Counters* const counters);
  // Selects whether to 'split_tf' or not based on residuals rate.
  WP2Status FindBestSplitTf(const EncoderConfig& config,
                            const Rectangle& tile_rect, const Segment& segment,
                            const BlockContext& context, Channel channel,
                            uint32_t num_channels, bool reduced,
                            Counters* const counters);
  // Calls FindBestPred/Tf() depending on 'config.effort' and 'context' then
  // Quantize(). All 'transforms' will be tested but only 'transforms_subset'
  // will be tried for each of the 'preds'.
  WP2Status OptimizeModesLuma(const EncoderConfig& config,
                              const Rectangle& tile_rect, bool has_alpha,
                              const Predictors& preds, const Segment& segment,
                              const BlockContext& context,
                              const Vector<TransformPair>& transforms,
                              const Vector<TransformPair>& transforms_subset,
                              Counters* const counters);

  // Decides whether to use yuv420 based on coeffs_[kUChannel or kVChannel].
  WP2Status DecideChromaSubsampling(const EncoderConfig& config,
                                    uint32_t tile_pos_x, uint32_t tile_pos_y,
                                    bool has_alpha, const QuantMtx& quant_u,
                                    const QuantMtx& quant_v,
                                    Counters* const counters);

  // Calls FindBestUVModes() and DecideChromaSubsampling() then Quantize().
  WP2Status OptimizeModesChroma(
    const EncoderConfig& config, const Rectangle& tile_rect, bool has_alpha,
    const FrontMgrBase& mgr, const UVPredictors& preds, const Segment& segment,
    const BlockContext& context, ChromaSubsampling chroma_subsampling,
    DCDiffusionMap* const dc_error_u, DCDiffusionMap* const dc_error_v,
    Counters* const counters);

  // Subtracts the 'prediction' from 'in_' and stores the residuals in 'res'.
  void GetResiduals(Channel channel, uint32_t tf_i, const Plane16& prediction,
                    BlockCoeffs16* const res) const;

  // Applies the TransformPair to 'res', quantizes the 'coeffs_' and
  // reconstructs the pixels to 'out_' using the 'prediction'.
  // 'tmp' is used for temporary computation. It can be the same as 'res' if
  // the caller doesn't mind losing the data.
  void TransformAndReconstruct(const EncoderConfig& config,
                               const BlockContext& context,
                               const Segment& segment, Channel channel,
                               uint32_t tf_i, bool reduced_transform,
                               const Plane16& prediction,
                               const BlockCoeffs16* const res,
                               BlockCoeffs16* const tmp,
                               Counters* const counters);

  // Predicts, quantizes and reconstructs channel 'channel'. If
  // 'reduced_transform' is true, 2x downsampling of residuals is performed.
  // Applies to...
  // ... the given sub transform 'tf_i'
  void Quantize(const EncoderConfig& config, const BlockContext& context,
                const Segment& segment, Channel channel, uint32_t tf_i,
                bool reduced_transform, Counters* const counters);
  // ... all sub transforms
  void QuantizeAll(const EncoderConfig& config, const BlockContext& context,
                   const Segment& segment, Channel channel,
                   bool reduced_transform, Counters* const counters);
  // ... all sub transforms except the first one (noop if there is only one)
  void QuantizeAllButFirst(const EncoderConfig& config,
                           const BlockContext& context, const Segment& segment,
                           Channel channel, bool reduced_transform,
                           Counters* const counters);

  // Reconstructs 'channel' from dequantized residuals 'res' and 'prediction'
  // to 'out_'. 'prediction' can also point to 'out_' or be empty.
  // Upon return, res[] contains the inverse-transformed coeffs.
  void Reconstruct(Channel channel, uint32_t tf_i, bool reduced_transform,
                   BlockCoeffs16* const res,
                   const Plane16& prediction = Plane16()) const;

  // Calls the base method and store some extra info
  // in 'blk'. Not a virtual.
  void ToBlockInfo(bool use_aom_coeffs, BlockInfo* const blk) const;

  // Returns the sum of square error between 'p_in_' and 'p_out_' for the pixels
  // inside the block (padded).
  uint32_t GetDisto(Channel channel) const;

  // Pixel transfer. 'yuv' is the full plane, not the block's sub-view.
  // No bound-check or clipping is performed.
  void ExtractFrom(const YUVPlane& yuv, Channel channel) const;

  // Returns an estimation of the number of bits necessary to encode the
  // predictor modes.
  WP2Status PredictorRate(Channel channel, SymbolCounter* const counter,
                          float* const rate) const;

  // Returns an estimation of the number of bits necessary to encode the
  // transform.
  WP2Status TransformRate(Channel channel,
                          SymbolCounter* const counter,
                          float* const rate) const;

  // Estimated cost in bits of storing residuals for the given channel.
  WP2Status ResidualRate(const BlockContext& context, Channel channel,
                         uint32_t num_channels, SymbolCounter* const counter,
                         SymbolCounter* const aom_counter, float* rate) const;

  // Pseudo-Visual debug: force the split_tf, tf or pred of the block.
  enum class SplitTf { kForcedUnsplit, kForcedSplit, kUnknown };
  SplitTf GetForcedSplitTf(const EncoderConfig& config,
                           const Rectangle& tile_rect) const;
  TransformPair GetForcedTransform(const EncoderConfig& config,
                                   const Rectangle& tile_rect) const;
  const Predictor* GetForcedPredictor(const EncoderConfig& config,
                                      const Rectangle& tile_rect,
                                      const Predictors& preds,
                                      Channel channel) const;

 public:
  //////////////// Methods for Visual Debug ////////////////

  // Visual debug: stores original residuals (before quantization) at block
  // position.
  void StoreOriginalResiduals(
      const EncoderConfig& config, uint32_t tile_pos_x, uint32_t tile_pos_y,
      int16_t original_res[kMaxNumTransformsPerBlock][kMaxBlockSizePix2],
      Plane16* const dst_plane) const;
  // Visual debug: stores prediction scores
  enum class SearchType {
    kTransform,
    kPredAndTransform,
    kSplit,
  };
  void StorePredictionScore(const EncoderConfig& config,
                            const Rectangle& tile_rect, Channel channel,
                            const Predictor& pred, TransformPair tf, float dist,
                            float lambda, float res_rate, float pred_rate,
                            float tf_rate, float score, bool is_best,
                            SearchType type) const;

  // Visual debug: stores information around the chroma subsampling decision.
  WP2Status Store420Scores(const EncoderConfig& config, uint32_t pos_x,
                           uint32_t pos_y, float lambda_u, float lambda_v,
                           bool reduced, uint32_t disto,
                           float rate_u, float rate_v);
  enum class Debug420Decision { k444EarlyExit, k444, k420 };
  WP2Status Store420Decision(const EncoderConfig& config, uint32_t pos_x,
                             uint32_t pos_y, Debug420Decision decision) const;

  // Visual debug: store lambda multiplier map.
  WP2Status StoreLambdaMult(const EncoderConfig& config,
                            uint32_t pos_x, uint32_t pos_y) const;

  // Visual debug: stores propagated error or new error for U/V error diffusion.
  void StoreErrorDiffusion(const EncoderConfig& config, uint32_t tile_x,
                           uint32_t tile_y, Plane16* const dst_plane) const;

  // Visual debug: stores raw (unquantized) residuals after ideal chroma from
  // luma prediction.
  void StoreBestCflResiduals(Channel channel, int16_t yuv_min, int16_t yuv_max,
                             Plane16* const dst_plane,
                             std::string* const debug_str) const;

  // Visual debug: stores the best possible prediction of the given channel,
  // assuming we could transmit the best slope/bias instead of deducing them
  // from context.
  void StoreBestCflPrediction(Channel channel, int16_t yuv_min, int16_t yuv_max,
                              Plane16* const dst_plane,
                              std::string* const debug_str) const;
  // Visual debug: stores the best slope for Chroma from luma ('a' in 'chroma =
  // a * luma + b'), after doing a linear regression between the output luma and
  // input chroma.
  void StoreBestCflSlope(Channel channel, int16_t yuv_min, int16_t yuv_max,
                         Plane16* const dst_plane,
                         std::string* const debug_str) const;
  // Visual debug: stores the best intercept for Chroma from luma ('b' in
  // 'chroma = a * luma + b'), after doing a linear regression between the
  // output luma and input chroma.
  void StoreBestCflIntercept(Channel channel, int16_t yuv_min, int16_t yuv_max,
                             Plane16* const dst_plane,
                             std::string* const debug_str) const;

 private:
  void CflLinearRegression(Channel channel, int16_t yuv_min, int16_t yuv_max,
                           float* const a, float* const b,
                           std::string* const debug_str) const;

 public:
  // Input and output samples must be views of buffers external to CodedBlock
  // pointing to the rectangle covered by 'blk_'. Except for 'predict_from_',
  // they will not be accessed outside of these bounds.
  YUVPlane in_;            // source samples (input), within blk_

  float pred_scores_[4] = {0};  // per channel, only available during encoding

  // Diffusion errors, indexed by Y/U/V channels.
  int16_t dc_error_[3] = {0};
  int16_t dc_error_next_[3] = {0};

  // Per-block lambda modulation. Larger multiplier will favor rate over
  // distortion for the block: should be assigned to highly textured blocks.
  float lambda_mult_ = 1.f;
};

}    // namespace WP2

#endif  /* WP2_ENC_BLOCK_H_ */
