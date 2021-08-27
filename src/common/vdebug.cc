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
//   Everything related to visual debug
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/common/vdebug.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <unordered_set>

#include "src/common/color_precision.h"
#include "src/common/lossy/block.h"
#include "src/common/lossy/block_size.h"
#include "src/common/lossy/predictor.h"
#include "src/dec/filters/alpha_filter.h"
#include "src/dec/filters/deblocking_filter.h"
#include "src/dec/filters/directional_filter.h"
#include "src/dec/filters/grain_filter.h"
#include "src/dec/filters/intertile_filter.h"
#include "src/dec/filters/restoration_filter.h"
#include "src/dec/tile_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless/dspl.h"
#include "src/dsp/math.h"
#include "src/enc/block_enc.h"
#include "src/enc/partitioning/partition_score_func_area.h"
#include "src/enc/partitioning/partition_score_func_block.h"
#include "src/enc/partitioning/partition_score_func_multi.h"
#include "src/enc/partitioning/partitioner.h"
#include "src/enc/partitioning/partitioner_exhaustive.h"
#include "src/enc/partitioning/partitioner_multi.h"
#include "src/enc/partitioning/partitioner_split_recurse.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"

//------------------------------------------------------------------------------
// Non-essential code starts here:

#if !defined(WP2_REDUCE_BINARY_SIZE)

namespace WP2 {

//------------------------------------------------------------------------------

void SyntaxWriter::PutRawPixels(const CodedBlockBase& cb,
                                const YUVPlane& pixels, ANSEnc* const enc) {
  assert(gparams_->transf_.GetYuvDepth().is_signed);
  const Block& block = cb.blk();
  enc->PutRValue(block.x(), SizeBlocks(tile_rect_.width), "DEEP_MATCH_X");
  enc->PutRValue(block.y(), SizeBlocks(tile_rect_.height), "DEEP_MATCH_Y");
  enc->PutRValue(block.w(), kMaxBlockSize + 1, "DEEP_MATCH_W");
  enc->PutRValue(block.h(), kMaxBlockSize + 1, "DEEP_MATCH_H");
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (pixels.GetChannel(channel).IsEmpty()) continue;
    const bool split_tf = enc->PutBool(cb.GetCodingParams(channel).split_tf,
                                       "DEEP_MATCH_SPLIT_TF");
    const uint32_t num_transforms =
        GetNumTransformsInBlock(block.dim(), split_tf);
    for (uint32_t tf_i = 0; tf_i < num_transforms; ++tf_i) {
      const Rectangle rect = block.LocalTfRect(split_tf, tf_i);
      for (uint32_t y = rect.y; y < rect.y + rect.height; ++y) {
        for (uint32_t x = rect.x; x < rect.x + rect.width; ++x) {
          enc->PutSUValue(pixels.GetChannel(channel).At(x, y),
                          gparams_->transf_.GetYuvDepth().num_bits,
                          kChannelStr[channel]);
        }
      }
    }
  }
}

void SyntaxReader::ReadAndCompareRawPixels(const CodedBlockBase& cb,
                                           const YUVPlane& pixels,
                                           ANSDec* const dec) {
  assert(gparams_->transf_.GetYuvDepth().is_signed);
  const uint32_t block_x = dec->ReadRValue(SizeBlocks(width_), "DEEP_MATCH_X");
  const uint32_t block_y = dec->ReadRValue(SizeBlocks(height_), "DEEP_MATCH_Y");
  const uint32_t block_w = dec->ReadRValue(kMaxBlockSize + 1, "DEEP_MATCH_W");
  const uint32_t block_h = dec->ReadRValue(kMaxBlockSize + 1, "DEEP_MATCH_H");
  const Block& block = cb.blk();
  if (block_x != block.x() || block_y != block.y()) assert(false);
  if (block_w != block.w() || block_h != block.h()) assert(false);
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (pixels.GetChannel(channel).IsEmpty()) continue;
    const bool split_tf = dec->ReadBool("DEEP_MATCH_SPLIT_TF");
    const uint32_t num_transforms =
        GetNumTransformsInBlock(block.dim(), split_tf);
    for (uint32_t tf_i = 0; tf_i < num_transforms; ++tf_i) {
      const Rectangle rect = block.LocalTfRect(split_tf, tf_i);
      for (uint32_t y = rect.y; y < rect.y + rect.height; ++y) {
        for (uint32_t x = rect.x; x < rect.x + rect.width; ++x) {
          const int16_t expected = dec->ReadSUValue(
              gparams_->transf_.GetYuvDepth().num_bits, kChannelStr[channel]);
          const int16_t actual = pixels.GetChannel(channel).At(x, y);
          if (expected != actual) assert(false);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Related to BitTrace

WP2Status RegisterBitTrace(const DecoderConfig& config, uint32_t num_bytes,
                           WP2_OPT_LABEL) {
  (void)config, (void)num_bytes, (void)label;
#if defined(WP2_BITTRACE)
  if (config.info != nullptr) {
    LabelStats* const stats = &config.info->bit_traces[label];
    stats->bits += num_bytes * 8.;
    stats->num_occurrences += 1;
    stats->type = LabelStats::Type::Symbol;
    stats->histo[0] += 1;
  }
#endif  // WP2_BITTRACE
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

#if defined(WP2_BITTRACE)
void CodedBlockBase::ToBlockInfo(bool use_aom_coeffs,
                                 BlockInfo* const blk) const {
  const CodingParams& y_params = GetCodingParams(kYChannel);

  blk->rect = AsRect();
  blk->segment_id = id_;
  blk->is420 = is420_;
  blk->tf_x = (uint8_t)y_params.tf_x();
  blk->tf_y = (uint8_t)y_params.tf_y();
  blk->y_context_is_constant = y_context_is_constant_;

  blk->y_pred = y_params.pred->mode();
  blk->has_lossy_alpha = HasLossyAlpha();
  if (HasLossyAlpha()) {
    const CodingParams& a_params = GetCodingParams(kAChannel);
    blk->a_pred = a_params.pred->mode();
  }
  blk->uv_pred = GetCodingParams(kUChannel).pred->mode();
  std::memset(blk->coeffs, 0, sizeof(blk->coeffs));
  std::fill(blk->split_tf, blk->split_tf + 4, false);
  std::fill(&blk->encoding_method[0][0], &blk->encoding_method[0][0] + 4 * 4,
            -1);
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !HasLossyAlpha()) continue;
    blk->split_tf[c] = GetCodingParams(c).split_tf;
    for (uint32_t i = 0; i < GetNumTransforms(c); ++i) {
      blk->encoding_method[c][i] = use_aom_coeffs
                                       ? (int)EncodingMethod::kNumMethod + 1
                                       : (int8_t)method_[c][i];
      std::copy(coeffs_[c][i], coeffs_[c][i] + NumCoeffsPerTransform(c),
                blk->coeffs[c][i]);
    }
  }
  blk->bits = 0;  // unknown for now

  // encoder part is reset:
  std::memset(blk->pred_scores, 0, sizeof(blk->pred_scores));
}

void CodedBlock::ToBlockInfo(bool use_aom_coeffs, BlockInfo* const blk) const {
  CodedBlockBase::ToBlockInfo(use_aom_coeffs, blk);

  // copy the encoder-specific part
  std::memcpy(blk->pred_scores, pred_scores_, sizeof(pred_scores_));
}

#endif  // WP2_BITTRACE

CodedBlock::SplitTf CodedBlock::GetForcedSplitTf(
    const EncoderConfig& config, const Rectangle& tile_rect) const {
  EncoderInfo* const info = config.info;
  if (info == nullptr) return SplitTf::kUnknown;
  if (info->disable_split_tf) return SplitTf::kForcedUnsplit;

  for (const auto& forced : info->force_param) {
    if (forced.type != EncoderInfo::ForcedParam::Type::kSplitTf) continue;
    if (forced.x != (x_pix() + tile_rect.x) ||
        forced.y != (y_pix() + tile_rect.y)) {
      continue;
    }

    return forced.value ? SplitTf::kForcedSplit : SplitTf::kForcedUnsplit;
  }
  return SplitTf::kUnknown;
}

TransformPair CodedBlock::GetForcedTransform(const EncoderConfig& config,
                                             const Rectangle& tile_rect) const {
  EncoderInfo* const info = config.info;
  if (info == nullptr) return kUnknownTf;
  if (info->disable_transforms) return kDctDct;

  for (const auto& forced : info->force_param) {
    if (forced.type != EncoderInfo::ForcedParam::Type::kTransform) continue;
    if (forced.x != (x_pix() + tile_rect.x) ||
        forced.y != (y_pix() + tile_rect.y)) {
      continue;
    }

    if (forced.value >= kNumTransformPairs) return kUnknownTf;
    return (TransformPair)forced.value;
  }
  return kUnknownTf;
}

const Predictor* CodedBlock::GetForcedPredictor(const EncoderConfig& config,
                                                const Rectangle& tile_rect,
                                                const Predictors& preds,
                                                Channel channel) const {
  EncoderInfo* const info = config.info;
  if (info == nullptr) return nullptr;
  if (info->disable_preds) return preds[0];

  for (const auto& forced : info->force_param) {
    if (forced.type != EncoderInfo::ForcedParam::Type::kPredictor) continue;
    if (forced.channel != channel ||
        forced.x != (x_pix() + tile_rect.x) ||
        forced.y != (y_pix() + tile_rect.y)) {
      continue;
    }

    assert(!preds.empty());
    const Predictor* const pred =
        preds[std::min(forced.value, (uint32_t)(preds.size() - 1))];
    return pred;
  }
  return nullptr;
}

//------------------------------------------------------------------------------

static const int kFontW = 8, kFontH = 8;   // font size
static constexpr uint64_t kFonts[96] = {  // Bitmap'd 8x8 font packed as u64
  0x0000000000000000, 0x0008000808080800, 0x0000000000242400,
  0x00247e24247e2400, 0x107c507c147c1000, 0x0062640810264600,
  0x005c225408140800, 0x0000000000081000, 0x0020101010102000,
  0x0004080808080400, 0x0028107c10280000, 0x0010107c10100000,
  0x0810100000000000, 0x0000007c00000000, 0x0018180000000000,
  0x0004081020400000, 0x003c464a52623c00, 0x007c101010141800,
  0x007e023c40423c00, 0x003c424030423c00, 0x00107e1214181000,
  0x003c42403e027e00, 0x003c42423e023c00, 0x0008081020407e00,
  0x003c42423c423c00, 0x003c407c42423c00, 0x0008000008000000,
  0x0408080000080000, 0x0020100810200000, 0x00007c007c000000,
  0x0008102010080000, 0x0010001020423c00, 0x003c027a6a523c00,
  0x0042427e42423c00, 0x003e42423e423e00, 0x003c420202423c00,
  0x001e224242221e00, 0x007e02023e027e00, 0x000202023e027e00,
  0x003c427202423c00, 0x004242427e424200, 0x007c101010107c00,
  0x003c424240404000, 0x004222120e122200, 0x007e020202020200,
  0x004242425a664200, 0x004262524a464200, 0x003c424242423c00,
  0x0002023e42423e00, 0x003c524a42423c00, 0x0042223e42423e00,
  0x003c42403c023c00, 0x0008080808087f00, 0x003c424242424200,
  0x0018244242424200, 0x00245a4242424200, 0x0042241818244200,
  0x0008080814224100, 0x007e040810207e00, 0x0070101010107000,
  0x0020100804020000, 0x000e080808080e00, 0x000808082a1c0800,
  0xff00000000000000, 0x007e04041e443800, 0x003c223c201c0000,
  0x003c44443c040400, 0x0038040404380000, 0x003c22223c202000,
  0x003c021e221c0000, 0x0008080818083000, 0x1c203c22223c0000,
  0x002222221e020200, 0x001c08080c000800, 0x1824202020002000,
  0x0024140c0c140400, 0x0030080808080800, 0x002a2a2a2a160000,
  0x00222222221e0000, 0x001c2222221c0000, 0x02021e22221e0000,
  0x60203c22223c0000, 0x0004040404380000, 0x001e201c021c0000,
  0x00300808081c0800, 0x001c222222220000, 0x0008141422220000,
  0x00142a2a2a220000, 0x0022140814220000, 0x1c203c2222220000,
  0x003e0408103e0000, 0x007010100c107000, 0x0010101010101000,
  0x000e080830080e00, 0x0000000000142800, 0x3c4299858599423c
};

static constexpr uint8_t kFontsSize[96] = {
  5, 4, 6, 7, 7, 7, 7, 5, 6, 4, 7, 7, 5, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  4, 4, 6, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 6, 4, 6, 8, 7, 6, 7, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6,
  6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 5, 6, 6, 8
};

void Print(const std::string& msg, int x, int y,
           Argb32b color, ArgbBuffer* const out) {
  assert(WP2FormatBpp(out->format()) == 4);
  const int x_start = x;  // for '\n'
  for (uint8_t c : msg) {
    if (c == '\n') {
      x = x_start;
      y += kFontH;
      continue;
    }
    // translate to printable character
    c = ((c >= 32 && c < 96 + 32) ? c : '?') - 32;
    uint64_t bits = kFonts[c];
    const int w = kFontsSize[c];
    const uint64_t mask = (1u << w) - 1;
    const int max_Y = std::min(y + kFontH, (int)out->height());
    for (int Y = y; bits && (Y < max_Y); ++Y, bits >>= kFontW) {
      if (Y >= 0 && (bits & mask)) {
        for (int i = 0; i < w; ++i) {
          const int X = x + i;
          if (X < 0 || X >= (int)out->width()) continue;
          if (bits & (1u << i)) ToUInt8(color, out->GetRow8(Y) + X * 4);
        }
      }
    }
    x += w;
  }
}

//------------------------------------------------------------------------------

// Declared in segment.h.
const Argb32b kSegmentColors[kMaxNumSegments]{
    {0xff, 0x30, 0xf0, 0xd0},  // Teal
    {0xff, 0x3d, 0x70, 0xf0},  // Blue
    {0xff, 0xb3, 0x30, 0xf0},  // Purple
    {0xff, 0xe4, 0x30, 0xf0},  // Pink
    {0xff, 0xf0, 0x30, 0x50},  // Red
    {0xff, 0xf0, 0x8c, 0x4d},  // Orange
    {0xff, 0xf0, 0xd0, 0x30},  // Yellow
    {0xff, 0x20, 0xf0, 0x60},  // Green
};

//------------------------------------------------------------------------------

// Sets pixel at (x, y) with 'color' if it fits in 'debug_output' boundaries.
static void MarkPixelForVDebug(int32_t x, int32_t y, Argb32b color,
                               ArgbBuffer* const debug_output) {
  assert(debug_output->format() == WP2_Argb_32);
  if (x >= 0 && x < (int32_t)debug_output->width() &&  // Within bounds.
      y >= 0 && y < (int32_t)debug_output->height()) {
    ToUInt8(color, debug_output->GetRow8(y) + x * 4);
  }
}

// Sets pixels with 'color' (or respectively 'bg_color' if not fully
// transparent), in the rectangle 'rect', for non-space characters in
// 'mask' (or respectively spaces).
static void DrawPixelArray(const Rectangle& rect,
                           Argb32b color, Argb32b bg_color,
                           const std::vector<std::string>& mask, uint32_t scale,
                           ArgbBuffer* const debug_output) {
  for (uint32_t l = 0; l < mask.size(); ++l) {
    for (uint32_t ls = 0; ls < scale; ++ls) {
      const uint32_t final_y = rect.y + l * scale + ls;
      if (final_y >= rect.y + rect.height) break;

      for (uint32_t c = 0; mask[l][c] != '\0'; ++c) {
        for (uint32_t cs = 0; cs < scale; ++cs) {
          const uint32_t final_x = rect.x + c * scale + cs;
          if (final_x >= rect.x + rect.width) break;

          uint8_t* const pixel = debug_output->GetRow8(final_y) + final_x * 4;
          if (mask[l][c] != ' ') {
            ToUInt8(color, pixel);
          } else if (bg_color.a > 0) {
            ToUInt8(bg_color, pixel);
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

void VDDrawUndefinedPattern(ArgbBuffer* const debug_output) {
  // Easily recognizable striped background.
  for (uint32_t y = 0; y < debug_output->height(); ++y) {
    for (uint32_t x = 0; x < debug_output->width(); ++x) {
      ToUInt8(((x % 10) == (y % 10)) ? Argb32b{0xff, 0x22, 0x22, 0x22}
                                     : Argb32b{0xff, 0x00, 0x00, 0x00},
              debug_output->GetRow8(y) + x * 4);
    }
  }

  // Display "undefined" if the 'debug_output' content is not overwritten.
  DrawPixelArray({debug_output->width() / 3,
                  debug_output->height() / 3,
                  debug_output->width() - debug_output->width() / 3,
                  debug_output->height() - debug_output->height() / 3},
                 Argb32b{0xff, 0x55, 0x55, 0x55}, Argb32b{0, 0, 0, 0},
                 {
                     "            #        # #             #",
                     "#  # # #   ##  ##   #    # #   ##   ##",
                     "#  # ## # # # #### ### # ## # #### # #",
                     "# ## #  # # # #     #  # #  # #    # #",
                     " # # #  #  ##  ##   #  # #  #  ##   ##",
                 },
                 /*scale=*/Clamp(debug_output->width() / 128u, 1u, 4u),
                 debug_output);
}

//------------------------------------------------------------------------------

// Returns true if the 'str' contains the 'token'. Tokens are separated by '/'.
template <>
bool VDMatch<const char*>(const char* const& in, const char token[]) {
  if (in != nullptr && token != nullptr) {
    const uint32_t token_length = std::strlen(token);
    if (token_length == 0) return true;

    for (const char* pos = in; (pos = std::strstr(pos, token)) != nullptr;
         pos += token_length) {
      if ((pos == in || pos[-1] == '/') &&  // Beginning of a token.
          (pos[token_length] == '\0' || pos[token_length] == '/')) {  // End.
        return true;  // A complete token is matched.
      }
    }
  }
  return false;
}
template <>
bool VDMatch<std::string>(const std::string& in, const char token[]) {
  return VDMatch<const char*>(in.c_str(), token);
}

// Returns true if an 'index' exists as a token in 'str' and outputs it.
static bool VDIndex(const char str[], uint32_t* const index) {
  if (str != nullptr) {
    // For each 'token' in 'str'.
    for (const char* token = str; *token != '\0'; ++token) {
      if (*token == '/') continue;  // Empty token.
      bool keep_token = true;
      uint32_t token_value = 0;
      static constexpr uint32_t max_value =
          std::numeric_limits<decltype(token_value)>::max() / 10;

      // For each character in 'token'.
      for (const char* pos = token; *pos != '\0'; ++pos) {
        if (*pos == '/') break;  // Token ended.
        if (*pos < '0' || *pos > '9') keep_token = false;
        if (keep_token && token_value > max_value) keep_token = false;
        if (keep_token) token_value = token_value * 10 + (*pos - '0');
      }
      if (keep_token) {
        *index = token_value;
        return true;
      }
    }
  }
  return false;
}

float VDGetParam(const EncoderConfig& config, float default_value) {
  float value;
  return (config.info != nullptr && config.info->visual_debug != nullptr &&
          std::sscanf(config.info->visual_debug, "p=%f", &value) == 1)
             ? value
             : default_value;
}

//------------------------------------------------------------------------------

// Maps a value in [0; num_values) to the [-512;512] range.
static int16_t ToPixel10(uint32_t value, uint32_t num_values) {
  if (num_values <= 1) return 0;
  return (int16_t)(DivRound(value * 1024u, num_values - 1u) - 512);
}

// Maps a value in [0; num_values) to the [0;255] range.
static int16_t ToPixel8(uint32_t value, uint32_t num_values) {
  if (num_values <= 1) return 0;
  return (uint8_t)DivRound(value * 255u, num_values - 1u);
}

static const Argb32b ColorMult(Argb32b color, float mutiplier) {
  return {color.a, (uint8_t)(color.r * mutiplier),
          (uint8_t)(color.g * mutiplier), (uint8_t)(color.b * mutiplier)};
}

//------------------------------------------------------------------------------

void CodedBlockBase::StoreTransform(const DecoderConfig& config,
                                    uint32_t tile_x, uint32_t tile_y,
                                    ArgbBuffer* const debug_output) const {
  const CodingParams& params = GetCodingParams(kYChannel);
  if (VDMatch(config, "xy")) {
  } else {
  }
  const WP2TransformType tf_x = kTfX[params.tf];
  const WP2TransformType tf_y = kTfY[params.tf];

  const bool all_zero = !HasCoeffs(kYChannel);
  static constexpr std::array<Argb32b, kNumTransforms> kColors = {
      {{0xff, 0xff, 0xff, 0xff},  // DCT is white.
       {0xff, 0xff, 0, 0xff},     // Adst is pink.
       {0xff, 0, 0xff, 0xff},     // Hadamaart is
       {0xff, 0xff, 0xff, 0}}};   // Identity is yellow.

  const Argb32b x_color = ColorMult(kColors[tf_x], all_zero ? 0.3f : 1.f);
  const Argb32b y_color = ColorMult(kColors[tf_y], all_zero ? 0.3f : 1.f);

  const Rectangle r_px = AsRect().ClipWith(debug_output->AsRect());

  const uint32_t w = r_px.width;
  const uint32_t h = r_px.height;
  if (VDMatch(config, "x") || (tf_x == tf_y)) {
    debug_output->Fill(r_px, x_color);
  } else if (VDMatch(config, "y")) {
    debug_output->Fill(r_px, y_color);
  } else {
    // Two different transforms. Show them as a checkerboard.
    const Argb32b x_color_dark = ColorMult(x_color, 0.8f);
    const Argb32b y_color_dark = ColorMult(y_color, 0.8f);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        const Rectangle pixel = {x_pix() + x, y_pix() + y, 1, 1};
        if (((x >> 1) % 2) == ((y >> 1) % 2)) {
          debug_output->Fill(pixel, (y % 2) ? x_color : x_color_dark);
        } else {
          debug_output->Fill(pixel, (x % 2) ? y_color : y_color_dark);
        }
      }
    }
  }

  if (VDSelected(tile_x, tile_y, r_px, config)) {
    debug_output->DrawRect(r_px, Argb32b{0xff, 0xff, 0x00, 0x00});
    std::string& str = config.info->selection_info;
    str += WP2SPrint("Block at %u, %u px (w %u, h %u)\n",
                     x_pix() + tile_x, y_pix() + tile_y, w_pix(), h_pix());
    if (all_zero) str += "All zero\n";
    str += WP2SPrint("Transform x: %s\n", WP2TransformNames[params.tf_x()]);
    str += WP2SPrint("Transform y: %s\n", WP2TransformNames[params.tf_y()]);
  }
}

// If the block is currently selected, outputs the pixel values from 'pixels'
// (in local block coordinates) to the debug selection_info.
static void OutputPixelValuesWithContext(const DecoderConfig& config,
                                         uint32_t tile_x, uint32_t tile_y,
                                         const CodedBlockBase& cb,
                                         const Plane16& pixels) {
  const Channel channel = VDChannel(config);
  const bool split_tf = cb.GetCodingParams(channel).split_tf;
  for (uint32_t tf_i = 0; tf_i < cb.GetNumTransforms(channel); ++tf_i) {
    const Rectangle rect = cb.blk().TfRect(split_tf, tf_i);
    if (VDSelected(tile_x, tile_y, rect, config)) {
      // Print pixels with the same formatting as "prediction/modes".
      std::string& str = config.info->selection_info;
      str += WP2SPrint("Block at %u, %u px (w %u, h %u) tf %d\n",
                       tile_x + cb.x_pix(), tile_y + cb.y_pix(), cb.w_pix(),
                       cb.h_pix(), tf_i);
      // To align with output from DisplayPredictionMode().
      str += WP2SPrint("\n\n");
      const int16_t* const context =
          cb.GetContext(channel, split_tf, tf_i, kContextSmall);
      const int16_t* const context_right =
          cb.GetContext(channel, split_tf, tf_i, kContextExtendRight);
      const int16_t* const context_left =
          cb.GetContext(channel, split_tf, tf_i, kContextExtendLeft);
      str += GetContextAndBlockPixelsStr(
          context, context_right, context_left, rect.width, rect.height,
          pixels.Row(rect.y - cb.y_pix()) + (rect.x - cb.x_pix()),
          pixels.Step());
    }
  }
}

void CodedBlockBase::StoreResiduals(const DecoderConfig& config,
                                    uint32_t tile_x, uint32_t tile_y,
                                    const QuantMtx& quant, Channel channel,
                                    Plane16* const dst_plane) const {
  const CodedBlockBase::CodingParams& params = GetCodingParams(channel);
  const BlockSize split_size = GetSplitSize(dim(), params.split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  uint32_t tf_i = 0;
  std::string& str = config.info->selection_info;
  for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
      const int16_t* const res = coeffs_[channel][tf_i];
      const bool selected = VDSelected(
          tile_x, tile_y,
          {x_pix() + split_x, y_pix() + split_y, split_w, split_h}, config);
      for (uint32_t y = 0; y < split_h; ++y) {
        for (uint32_t x = 0; x < split_w; ++x) {
          dst_plane->At(x_pix() + split_x + x, y_pix() + split_y + y) =
              res[x + split_w * y];
          if (selected) str += WP2SPrint("%4d ", res[x + split_w * y]);
        }
        if (selected) str += "\n";
      }
      ++tf_i;
    }
  }
}

void CodedBlock::StoreOriginalResiduals(
    const EncoderConfig& config, uint32_t tile_pos_x, uint32_t tile_pos_y,
    int16_t original_res[kMaxNumTransformsPerBlock][kMaxBlockSizePix2],
    Plane16* const dst_plane) const {
  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(VDChannel(config)).split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  uint32_t tf_i = 0;
  std::string& str = config.info->selection_info;
  for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
      const bool selected = VDSelected(
          tile_pos_x, tile_pos_y,
          {x_pix() + split_x, y_pix() + split_y, split_w, split_h}, config);
      for (uint32_t k = 0, y = 0; y < split_h; ++y) {
        if (y_pix() + split_y + y >= dst_plane->h_) break;
        for (uint32_t x = 0; x < split_w; ++x, ++k) {
          if (x_pix() + split_x + x >= dst_plane->w_) break;
          const int16_t value = original_res[tf_i][k];
          dst_plane->At(x_pix() + split_x + x, y_pix() + split_y + y) = value;
          if (selected) str += WP2SPrint("%4d ", value);
        }
        if (selected) str += "\n";
      }
      ++tf_i;
    }
  }
}

static Argb32b CtxtToColor(int16_t context) {
  const uint8_t v = Clamp(DivRound((context + 512) * 255, 1023), 0x00, 0xff);
  return Argb32b{0xff, v, v, v};
}

// Draws the context used for the prediction of 'cb'.
static void DisplayPredContext(const CodedBlockBase& cb, Channel channel,
                               bool split_tf, uint32_t tf_i,
                               bool draw_red_border,
                               ArgbBuffer* const debug_output) {
  const Rectangle raw_rect = cb.blk().TfRect(split_tf, tf_i);
  const uint32_t x = raw_rect.x, y = raw_rect.y, w = raw_rect.width,
                 h = raw_rect.height;
  const Rectangle rect =
      cb.blk().TfRect(split_tf, tf_i).ClipWith(debug_output->AsRect());

  if (draw_red_border) {  // Encircle the left, top, right context lines.
    const uint32_t bx = SafeSub(x, 2u), by = SafeSub(y, 2u);
    const Argb32b red = {0xff, 0xff, 0, 0};
    debug_output->DrawRect({bx, by, x - bx + 1, y + rect.height * 2 - by}, red);
    debug_output->DrawRect({bx, by, x + rect.width - bx, y - by + 1}, red);
    debug_output->DrawRect({x + w - 1, by, 3, y + rect.height - by}, red);
    debug_output->DrawRect(
        {x + w - 1, by,
         2 + ContextSize(kContextExtendRight, w, h) - (h + 1 + w), 3},
        red);
  }

  const int16_t* const context =
      cb.GetContext(channel, split_tf, tf_i, kContextSmall);
  if (x >= 1) {
    for (uint32_t j = 0; j < rect.height; ++j) {  // Left
      debug_output->Fill({x - 1, y + j, 1, 1}, CtxtToColor(context[h - j - 1]));
    }
    if (y >= 1) {  // Top-left
      debug_output->Fill({x - 1, y - 1, 1, 1}, CtxtToColor(context[h]));
    }
  }
  if (y >= 1) {
    for (uint32_t i = 0; i < rect.width; ++i) {  // Top
      debug_output->Fill({x + i, y - 1, 1, 1}, CtxtToColor(context[h + 1 + i]));
    }
    if (x + w < debug_output->width()) {  // Top-right
      debug_output->Fill({x + w, y - 1, 1, 1}, CtxtToColor(context[h + 1 + w]));
    }
  }
  if (x + w < debug_output->width()) {
    for (uint32_t j = 0; j < rect.height; ++j) {  // Right
      debug_output->Fill({x + w, y + j, 1, 1},
                         CtxtToColor(context[h + 1 + w + 1 + j]));
    }
  }

  // Extended right
  const int16_t* const tr_context =
      cb.GetContext(channel, split_tf, tf_i, kContextExtendRight);
  const uint32_t tr_size = ContextSize(kContextExtendRight, w, h);
  for (uint32_t i = h + 1 + w + 1, tr_x = x + w + 1; i < tr_size; ++i, ++tr_x) {
    debug_output->Fill({tr_x, y - 1, 1, 1}, CtxtToColor(tr_context[i]));
  }
  // Extended left
  const int16_t* const left_context =
      cb.GetContext(channel, split_tf, tf_i, kContextExtendLeft);
  for (uint32_t j = h; j < 2 * h; ++j) {
    if (y + j + 1 >= debug_output->height()) break;
    debug_output->Fill({x - 1, y + j, 1, 1},
                       CtxtToColor(left_context[2 * h - j - 1]));
  }
}

static constexpr const char* kSearchTypeNames[] = {"tf", "pred&tf", "split"};

void CodedBlock::StorePredictionScore(
    const EncoderConfig& config, const Rectangle& tile_rect, Channel channel,
    const Predictor& pred, TransformPair tf, float dist, float lambda,
    float res_rate, float pred_rate, float tf_rate, float score, bool is_best,
    SearchType type) const {
  if (!VDMatch(config, "prediction-scores")) return;
  if (VDChannel(config) != channel) return;

  // Check the formula did not change.
  assert(std::abs(dist + lambda * (res_rate + pred_rate + tf_rate) - score) <
         0.1f);

  ArgbBuffer debug_output;  // Tile view.
  WP2_ASSERT_STATUS(debug_output.SetView(config.info->debug_output, tile_rect));

  const Rectangle rect = AsRect().ClipWith(debug_output.AsRect());
  const bool selected = VDSelected(tile_rect.x, tile_rect.y, rect, config);

  debug_output.Fill(rect, Argb32b{0xff, 0x00, 0x22, 0x00});
  if (selected) {
    debug_output.DrawRect(rect, Argb32b{0xff, 0xff, 0x33, 0x33});

    // Print at least all better predictors. Skip worse ones if too many.
    std::string& str = config.info->selection_info;
    if (is_best || std::count(str.begin(), str.end(), '\n') < 30) {
      str += WP2SPrint("Score %9.1f = dist %9.1f", score, dist);
      str += WP2SPrint(" + lambda %9.1f * (res %7.4f + pred %7.4f + tf %7.4f)",
                       lambda, res_rate, pred_rate, tf_rate);
      str += WP2SPrint(" %s (%s - %s/%s x%u) (%s)\n", is_best ? "best" : "    ",
                       pred.GetName().c_str(), WP2TransformNames[kTfX[tf]],
                       WP2TransformNames[kTfY[tf]], GetNumTransforms(channel),
                       kSearchTypeNames[(int)type]);
    } else {
      // Print in console instead.
      printf(
          "Block at (%d %d) Score %9.1f = dist %9.1f + lambda %9.1f * (res "
          "%7.4f + pred %7.4f + "
          "tf %7.4f)  %s (%s - %s/%s x%u) (%s)\n",
          x_pix(), y_pix(), score, dist, lambda, res_rate, pred_rate, tf_rate,
          is_best ? "best" : "    ", pred.GetName().c_str(),
          WP2TransformNames[kTfX[tf]], WP2TransformNames[kTfY[tf]],
          GetNumTransforms(channel), kSearchTypeNames[(int)type]);
    }
  }

  for (uint32_t tf_i = 0; tf_i < GetNumTransforms(channel); ++tf_i) {
    const bool split_tf = GetCodingParams(channel).split_tf;
    const Rectangle raw_tf_rect = blk().TfRect(split_tf, tf_i);
    const Rectangle tf_rect = raw_tf_rect.ClipWith(debug_output.AsRect());
    const bool tf_selected =
        VDSelected(tile_rect.x, tile_rect.y, tf_rect, config);

    DisplayPredContext(*this, channel, split_tf, tf_i,
                       /*draw_red_border=*/tf_selected, &debug_output);
  }
}

static void DrawAnglePredictor(const WP2::Rectangle& rect, float angle_deg,
                               WP2::ArgbBuffer* const debug_output) {
  if (angle_deg == std::numeric_limits<float>::max()) return;
  const float angle = (M_PI / 180.f) * angle_deg;
  const float cos = std::cos(angle);
  const float sin = std::sin(angle);
  const uint32_t pixel_depth = WP2FormatBpp(debug_output->format());
  const uint32_t length =
      std::sqrt(rect.width * rect.width + rect.height * rect.height) / 2;
  for (float i = 0.f; i < length; i += 0.5f) {
    const int y = std::round(rect.y + (rect.height - 1) / 2.f - i * sin);
    const int x = std::round(rect.x + (rect.width - 1) / 2.f + i * cos);
    if (!rect.Contains(x, y)) break;
    const uint8_t color[4] = {255, 255, 0, 0};
    std::memcpy(debug_output->GetPosition(x, y), color, pixel_depth);
  }
}

static constexpr Argb32b kNonAnglePredictorColors[] = {
    {0xff, 0x30, 0xf0, 0xd0},  // Teal
    {0xff, 0x3d, 0x70, 0xf0},  // Blue
    {0xff, 0xb3, 0x30, 0xf0},  // Purple
    {0xff, 0xe4, 0x30, 0xf0},  // Pink
    {0xff, 0xf0, 0x30, 0x50},  // Red
    {0xff, 0xf0, 0x8c, 0x4d},  // Orange
    {0xff, 0xf0, 0xd0, 0x30},  // Yellow
    {0xff, 0x20, 0xf0, 0x60},  // Green
    {0xff, 0x20, 0x6b, 0x00},  // Dark green
};
// If this fails, add some more colors to the array above.
static_assert(ArraySize(kNonAnglePredictorColors) >= kYBasePredNum,
              "Not enough colors");
static_assert(ArraySize(kNonAnglePredictorColors) >= kABasePredNum,
              "Not enough colors");
static_assert(ArraySize(kNonAnglePredictorColors) >= kUVBasePredNum,
              "Not enough colors");

// Prints the selected block prediction to 'config.info->selection_info' and
// fills either the 'raw_prediction' (values) or the 'debug_output' (mode).
static void DisplayPredictionMode(
    const DecoderConfig& config, const Rectangle& tile_rect,
    const CodedBlockBase& cb, const Predictors& preds, const Predictor& p,
    Channel channel, uint32_t tf_i, const Rectangle& block_rect, bool selected,
    Plane16* const raw_prediction, ArgbBuffer* const debug_output) {
  const bool split_tf = cb.GetCodingParams(channel).split_tf;
  if (raw_prediction != nullptr) {
    Plane16 dst_view;
    WP2_ASSERT_STATUS(
        dst_view.SetView(*raw_prediction, cb.blk().TfRect(split_tf, tf_i)));
    const Predictor& pred = *cb.GetCodingParams(channel).pred;
    pred.Predict(cb, channel, split_tf, tf_i, dst_view.Row(0), dst_view.Step());
  } else {
    assert(debug_output != nullptr);
    float angle;
    if (p.IsAngle(&angle)) {
      // Map angle to light gray value in [128; 255]
      const uint8_t gray = ToPixel8(p.mode(), kAnglePredNum) / 2 + 128;
      debug_output->Fill(block_rect, Argb32b{0xff, gray, gray, gray});
      DrawAnglePredictor(block_rect, angle, debug_output);
    } else {
      // Find this index of this predictor out of the non-angle predictors.
      uint32_t i = 0;
      bool found = false;
      for (const Predictor* pred : preds) {
        if (pred->IsAngle(nullptr)) continue;
        if (pred->mode() == p.mode()) {
          found = true;
          break;
        }
        ++i;
      }
      (void)found;
      assert(found);
      assert(i < ArraySize(kNonAnglePredictorColors));
      debug_output->Fill(block_rect, kNonAnglePredictorColors[i]);
    }
    if (selected) {
      debug_output->DrawRect(block_rect, Argb32b{0xff, 0xff, 0x33, 0x33});
    }
    // draw split borders
    if (split_tf) {
      const BlockSize split_size = GetSplitSize(cb.dim(), true);
      const uint32_t split_w = BlockWidthPix(split_size);
      const uint32_t split_h = BlockHeightPix(split_size);
      for (uint32_t split_y = 0; split_y < cb.h_pix(); split_y += split_h) {
        for (uint32_t split_x = 0; split_x < cb.w_pix(); split_x += split_w) {
          constexpr Argb32b kBlack = Argb32b{0xff, 0x00, 0x00, 0x00};
          debug_output->Fill(
              {block_rect.x + split_x, block_rect.y + split_y, 1, split_h},
              kBlack);
          debug_output->Fill(
              {block_rect.x + split_x, block_rect.y + split_y, split_w, 1},
              kBlack);
        }
      }
    }
  }

  if (selected) {
    std::string& str = config.info->selection_info;
    str += WP2SPrint("Prediction block at %u, %u px (w %u, h %u)",
                     block_rect.x + tile_rect.x, block_rect.y + tile_rect.y,
                     block_rect.width, block_rect.height, tf_i);
    if (split_tf) {
      str += WP2SPrint(" tf %d", tf_i);
    }
    str += "\n";

    str += WP2SPrint(" %u: %s\n", p.mode(), p.GetName().c_str());

    if (!VDMatch(config, "short")) {
      const std::string prediction = p.GetPredStr(cb, channel, split_tf, tf_i);
      str += "Prediction:\n" + prediction;
    }
  }
}

void CodedBlockBase::StorePredictionModes(
    const DecoderConfig& config, const Rectangle& tile_rect, Channel channel,
    uint32_t tf_i, const Predictors& preds, Plane16* const raw_prediction,
    ArgbBuffer* const debug_output) const {
  const bool split_tf = GetCodingParams(channel).split_tf;

  const Rectangle raw_rect = blk().TfRect(split_tf, tf_i);
  const Rectangle r_px{
      raw_rect.x, raw_rect.y,
      std::min(raw_rect.width, tile_rect.width - raw_rect.x),
      std::min(raw_rect.height, tile_rect.height - raw_rect.y)};

  const bool selected = VDSelected(tile_rect.x, tile_rect.y, r_px, config);

  const CodingParams& params = GetCodingParams(channel);
  DisplayPredictionMode(config, tile_rect, *this, preds, *params.pred, channel,
                        tf_i, r_px, selected, raw_prediction, debug_output);

  if (selected && debug_output != nullptr) {
    DisplayPredContext(*this, channel, split_tf, tf_i, /*draw_red_border=*/true,
                       debug_output);
  }
}

void CodedBlockBase::AppendOriginalPixels(
    const DecoderConfig& config, uint32_t tile_x, uint32_t tile_y,
    const CSPTransform& csp_tranform, ArgbBuffer* const debug_output) const {
  if (!(VDMatch(config, "y") || VDMatch(config, "u") || VDMatch(config, "v") ||
        VDMatch(config, "a"))) {
    return;
  }
  if (VDSelected(AsRect(tile_x, tile_y), config)) {
    const Channel channel = VDChannel(config);
    YUVPlane original_yuv;  // 'debug_output' should be the original RGB pixels.
    WP2_ASSERT_STATUS(
        original_yuv.Import(*debug_output, /*import_alpha=*/true, csp_tranform,
                            /*resize_if_needed=*/true, /*pad=*/kPredWidth));
    // Use a dummy CodedBlockBase to get the 'context' out of 'original_yuv'.
    CodedBlockBase cb;
    cb.SetDimDefault(blk(), /*full_left_ctx=*/true);
    if (cb.x_pix() + cb.w_pix() < original_yuv.Y.w_) {
      cb.right_occupancy_ = cb.y() + cb.h();  // Full right context.
    }
    ContextCache context_cache;
    cb.SetContextInput(original_yuv, &context_cache);

    YUVPlane view;
    WP2_ASSERT_STATUS(view.SetView(original_yuv, AsRect()));
    OutputPixelValuesWithContext(config, tile_x, tile_y, cb,
                                 view.GetChannel(channel));
  }
}

void CodedBlockBase::AppendCompressedPixels(
    const DecoderConfig& config, uint32_t tile_x, uint32_t tile_y,
    ArgbBuffer* const debug_output) const {
  if (VDMatch(config, "r") || VDMatch(config, "g") || VDMatch(config, "b")) {
    return;
  }
  const Channel channel = VDChannel(config);
  if (channel == kAChannel && !HasLossyAlpha()) return;

  if (VDSelected(AsRect(tile_x, tile_y), config)) {
    OutputPixelValuesWithContext(config, tile_x, tile_y, *this,
                                 out_.GetChannel(channel));
    config.info->selection_info +=
        "Compressed values are before applying post processing filters\n";
  }
}

void SyntaxReader::StoreBitCost(const DecoderConfig& config,
                                uint32_t tile_x, uint32_t tile_y,
                                const Block& block,
                                Plane16* const dst_plane) const {
  (void)config, (void)tile_x, (void)tile_y, (void)block, (void)dst_plane;
#if defined(WP2_BITTRACE)
  const Rectangle& rect = block.AsRect();
  const std::string prefix =
      VDMatch(config, "overall") ? "" : kCoeffsStr[VDChannel(config)];

  double num_bits = 0.;
  for (const auto& bt : dec_->GetBitTracesCustom()) {
    if (bt.first.size() >= prefix.size() &&
        std::strncmp(bt.first.c_str(), prefix.c_str(), prefix.size()) == 0) {
      num_bits += bt.second.bits;
    }
  }

  // Max value is 6bpp (which is a lot!). Use 8b fixed point precision.
  const int16_t value = ToPixel10(
      (uint32_t)std::lround(num_bits * 256. / rect.GetArea()), 6u << 8);
  dst_plane->Fill(rect, value);

  if (VDSelected(tile_x, tile_y, rect, config)) {
    dst_plane->DrawRect(rect, 127);
    std::string& str = config.info->selection_info;
    str += WP2SPrint("\nBlock at %u, %u px\n\n", rect.x, rect.y);
    str += WP2SPrint("%f bits over %u pixels\n",
                     num_bits, (uint32_t)rect.GetArea());
    str += WP2SPrint(" = %f bpp\n", num_bits / rect.GetArea());
  }
#endif  // defined(WP2_BITTRACE)
}

void CodedBlockBase::StoreCoeffMethod(const DecoderConfig& config,
                                      Plane16* const dst_plane) const {
  const Channel channel = VDChannel(config);
  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(channel).split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  uint32_t tf_i = 0;
  for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
      dst_plane->Fill(
          {x_pix() + split_x, y_pix() + split_y, split_w - 1, split_h - 1},
          ToPixel10((uint32_t)method_[channel][tf_i],
                    (uint32_t)EncodingMethod::kNumMethod + 1));
      ++tf_i;
    }
  }
}

void CodedBlockBase::StoreHasCoeffs(const DecoderConfig& config,
                                    Plane16* const dst_plane) const {
  const Channel channel = VDChannel(config);
  const BlockSize split_size =
      GetSplitSize(dim(), GetCodingParams(channel).split_tf);
  const uint32_t split_w = BlockWidthPix(split_size);
  const uint32_t split_h = BlockHeightPix(split_size);
  uint32_t tf_i = 0;
  for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
    for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
      dst_plane->Fill(
          {x_pix() + split_x, y_pix() + split_y, split_w - 1, split_h - 1},
          ToPixel10((uint32_t)num_coeffs_[channel][tf_i] > 0, 2));
      ++tf_i;
    }
  }
}

//------------------------------------------------------------------------------
// Cfl

static constexpr float CflNorm = 1.f / (1 << kCflFracBits);

void CodedBlockBase::StoreCflSlope(Channel channel, int16_t yuv_min,
                                   int16_t yuv_max, Plane16* const dst_plane,
                                   std::string* const debug_str) const {
  int32_t a, b;
  CflPredictor cfl_predictor(yuv_min, yuv_max);
  cfl_predictor.ContextLinearRegression(channel, *this, &a, &b);
  dst_plane->Fill(blk_.AsRect(), (int16_t)std::lround(a * 100.f * CflNorm));

  if (debug_str != nullptr) {
    *debug_str += WP2SPrint("Chroma = %.2f * luma + %.2f\n",
                            CflNorm * a, CflNorm * b);
  }
}

void CodedBlockBase::StoreCflIntercept(Channel channel, int16_t yuv_min,
                                       int16_t yuv_max,
                                       Plane16* const dst_plane,
                                       std::string* const debug_str) const {
  int32_t a, b;
  CflPredictor cfl_predictor(yuv_min, yuv_max);
  cfl_predictor.ContextLinearRegression(channel, *this, &a, &b);
  dst_plane->Fill(blk_.AsRect(), (int16_t)std::lround(0.5f * CflNorm * b));

  if (debug_str != nullptr) {
    *debug_str = WP2SPrint("Chroma = %.2f * luma + %.2f\n",
                           CflNorm * a, CflNorm * b);
  }
}

// Similar to CflPredictor::LinearRegression, but uses the actual block
// instead of the block context.

void CodedBlock::CflLinearRegression(Channel channel,
                                     int16_t yuv_min, int16_t yuv_max,
                                     float* const a, float* const b,
                                     std::string* const debug_str) const {
  int32_t a_int, b_int;
  CflPredictor cfl_predictor(yuv_min, yuv_max);
  cfl_predictor.BlockLinearRegression(channel, *this, &a_int, &b_int);
  *a = CflNorm * a_int;
  *b = CflNorm * b_int;

  if (debug_str != nullptr) {
    *debug_str += WP2SPrint("Chroma = %.2f * luma + %.2f\n", *a, *b);
  }
}

void CodedBlock::StoreBestCflSlope(Channel channel, int16_t yuv_min,
                                   int16_t yuv_max,
                                   Plane16* const dst_plane,
                                   std::string* const debug_str) const {
  float a, b;
  CflLinearRegression(channel, yuv_min, yuv_max, &a, &b, debug_str);
  dst_plane->Fill(AsRect(), (int16_t)std::lround(a * 100.f));
}


void CodedBlock::StoreBestCflIntercept(Channel channel, int16_t yuv_min,
                                       int16_t yuv_max,
                                       Plane16* const dst_plane,
                                       std::string* const debug_str) const {
  float a, b;
  CflLinearRegression(channel, yuv_min, yuv_max, &a, &b, debug_str);
  dst_plane->Fill(AsRect(), (int16_t)std::lround(b / 2.f));
}

void CodedBlock::StoreBestCflPrediction(Channel channel, int16_t yuv_min,
                                        int16_t yuv_max,
                                        Plane16* const dst_plane,
                                        std::string* const debug_str) const {
  const Plane16& luma = out_.GetChannel(kYChannel);
  const Rectangle rect = AsRect().ClipWith(dst_plane->AsRect());

  float a, b;
  CflLinearRegression(channel, yuv_min, yuv_max, &a, &b, debug_str);

  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      const int32_t v = std::lround(a * luma.At(x, y) + b);
      dst_plane->At(rect.x + x, rect.y + y) = (int16_t)v;
    }
  }
}

void CodedBlock::StoreBestCflResiduals(Channel channel, int16_t yuv_min,
                                       int16_t yuv_max,
                                       Plane16* const dst_plane,
                                       std::string* const debug_str) const {
  const Plane16& luma = out_.GetChannel(kYChannel);
  const Plane16& chroma = in_.GetChannel(channel);
  const Rectangle rect = AsRect().ClipWith(dst_plane->AsRect());

  float a, b;
  CflLinearRegression(channel, yuv_min, yuv_max, &a, &b, debug_str);

  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      const int32_t v = std::lround(a * luma.At(x, y) + b);
      dst_plane->At(rect.x + x, rect.y + y) = (int16_t)(chroma.At(x, y) - v);
    }
  }
}

//------------------------------------------------------------------------------

void CodedBlock::StoreErrorDiffusion(const EncoderConfig& config,
                                     uint32_t tile_x, uint32_t tile_y,
                                     Plane16* const dst_plane) const {
  if (!VDMatch(config, "error-diffusion")) return;
  const Channel channel = VDChannel(config);

  if (VDSelected(AsRect(tile_x, tile_y), config)) {
    config.info->selection_info += WP2SPrint(
               "Block at %4u, %4u (%2u x %2u px)\n", tile_x + x_pix(),
               tile_y + y_pix(), w_pix(), h_pix());
    config.info->selection_info += WP2SPrint(
               "Propagated error = %d | New error = %d\n", dc_error_[channel],
               dc_error_next_[channel]);
  }

  const int16_t error = VDMatch(config, "propagated-error")
                            ? dc_error_[channel]
                            : dc_error_next_[channel];
  dst_plane->Fill(AsRect(), error / 2);
}

//------------------------------------------------------------------------------

void Block::Draw(YUVPlane* const yuv) const {
  const uint32_t x0 = x_pix();
  const uint32_t y0 = y_pix();
  for (uint32_t i = 0; i < w_pix(); ++i) {
    yuv->Y.At(x0 + i, y0) = 0x1ff;
    yuv->U.At(x0 + i, y0) = 0x100;
    yuv->V.At(x0 + i, y0) = 0x100;
  }
  for (uint32_t j = 0; j < h_pix(); ++j) {
    yuv->Y.At(x0, y0 + j) = 0x1ff;
    yuv->U.At(x0, y0 + j) = 0x100;
    yuv->V.At(x0, y0 + j) = 0x100;
  }
}

//------------------------------------------------------------------------------

static std::string PrintQuantSteps(const uint16_t steps[4], const char name[]) {
  return WP2SPrint(" %s quant_steps, DC: %d AC: %d %d %d %d\n", name, steps[0],
                   steps[1], steps[2], steps[3], steps[4]);
}

static std::string PrintSegmentInfo(const Segment& segment, bool is_alpha) {
  std::string str;
  if (segment.use_quality_factor_) {
    const float q = 1.f * segment.quality_factor_ / kQFactorMax;
    str += WP2SPrint(" quality_factor_: %d  (q ~= %.1f)\n",
                     segment.quality_factor_, kMaxLossyQuality * (1.f - q));
  }
  if (is_alpha) {
    str += PrintQuantSteps(segment.quant_steps_[kAChannel], "A");
  } else {
    str += PrintQuantSteps(segment.quant_steps_[kYChannel], "Y");
    str += PrintQuantSteps(segment.quant_steps_[kUChannel], "U");
    str += PrintQuantSteps(segment.quant_steps_[kVChannel], "V");
  }
  return str;
}

WP2Status CodedBlockBase::Draw(const DecoderConfig& config, const Tile& tile,
                               const GlobalParams& gparams,
                               ArgbBuffer* const debug_output) const {
  assert(VDMatch(config, "blocks"));
  const Rectangle rect = AsRect().ClipWith(debug_output->AsRect());
  const uint32_t x = rect.x;
  const uint32_t y = rect.y;
  const uint32_t w = rect.width;  // Border
  const uint32_t h = rect.height;
  const uint32_t mid_w = std::min(w - 2, debug_output->width() - (x + 1));
  const uint32_t mid_h = std::min(h - 2, debug_output->height() - (y + 1));
  const Rectangle sub_rect = {x + 1, y + 1, mid_w, mid_h};

  {    // Copy the decoded content in the block's sub-area.
    YUVPlane tmp_yuv;
    WP2_CHECK_STATUS(tmp_yuv.SetView(tile.yuv_output, rect));
    ArgbBuffer tmp_argb;
    WP2_CHECK_STATUS(tmp_argb.SetView(*debug_output, rect));
    WP2_CHECK_STATUS(tmp_yuv.Export(gparams.transf_, /*resize_if_needed=*/false,
                                    &tmp_argb));
  }

  if (VDMatch(config, "partition")) {
    if (VDMatch(config, "split-tf")) {
      const BlockSize split_size =
          GetSplitSize(dim(), GetCodingParams(kYChannel).split_tf);
      const uint32_t split_w = BlockWidthPix(split_size);
      const uint32_t split_h = BlockHeightPix(split_size);
      for (uint32_t split_y = 0; split_y < h_pix(); split_y += split_h) {
        for (uint32_t split_x = 0; split_x < w_pix(); split_x += split_w) {
          debug_output->Fill({x + split_x, y + split_y, 1, split_h},
                             Argb32b{0xff, 0x00, 0x30, 0xf0});
          debug_output->Fill({x + split_x, y + split_y, split_w, 1},
                             Argb32b{0xff, 0x00, 0x40, 0xf0});
        }
      }
    }
    debug_output->DrawRect(rect, Argb32b{0xff, 0xff, 0xff, 0x30}, 0x03);
  } else if (VDMatch(config, "segment-ids") ||
             VDMatch(config, "quantization")) {
    const bool is_alpha =
        VDMatch(config, "quantization") && (VDChannel(config) == kAChannel);
    if (is_alpha && !HasLossyAlpha()) {
      debug_output->Fill(rect, Argb32b{255, 0, 0, 0});
      return WP2_STATUS_OK;
    }
    const Segment& segment = gparams.segments_[is_alpha ? 0 : id_];

    Argb32b color;
    if (VDMatch(config, "segment-ids")) {
      color = kSegmentColors[id_];
      debug_output->Fill(sub_rect, color);
    } else {
      // showing quantization
      uint16_t quant[kNumQuantZones];
      segment.GetQuantSteps(quant, VDChannel(config));
      // Quant expressed in [0 .. 1]
      const float normalized_quant = (float)quant[0] / kQFactorMax;

      // Gamma it not really necessary but just to show I studied colors.
      constexpr float kGamma = 2.2f;
      const float red = std::pow(normalized_quant, 1.f / kGamma);
      const float green = std::pow((1.f - normalized_quant), 1.f / kGamma);
      // Max out saturation.
      const float mult = std::min(1.f / red, 1.f / green) * 255.f;

      color = Argb32b{255u,
                      (uint8_t)std::min((uint32_t)(red * mult), 255u),
                      (uint8_t)std::min((uint32_t)(green * mult), 255u),
                      0u};
      debug_output->Fill(rect, color);
    }

    if (VDSelected(tile.rect.x, tile.rect.y, rect, config)) {
      std::string& str = config.info->selection_info;
      str += WP2SPrint("\nSegment id: %u\n", id_);
      str += PrintSegmentInfo(segment, is_alpha);

      debug_output->Fill(rect, Argb32b{255, 0, 0, 0});
      debug_output->Fill(sub_rect, color);
    }
  } else if (VDMatch(config, "is420")) {
    debug_output->DrawRect(rect, is420_ ? Argb32b{0xff, 0x11, 0xff, 0x11}
                                        : Argb32b{0xff, 0xdd, 0xdd, 0xdd});
    if (is420_) {
      DrawPixelArray(sub_rect,
                     Argb32b{0xff, 0xaa, 0xff, 0xaa},
                     Argb32b{0xff, 0x00, 0x00, 0x00},
                     {
                         "             ",
                         "   #  #   #  ",
                         "  ## # # # # ",
                         " # #   # # # ",
                         " ###  #  # # ",
                         "   # ###  #  ",
                         "             ",
                     },
                     1, debug_output);
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status CodedBlock::Store420Scores(const EncoderConfig& config,
                                     uint32_t pos_x, uint32_t pos_y,
                                     float lambda_u, float lambda_v,
                                     bool reduced, uint32_t disto,
                                     float rate_u, float rate_v) {
  if (!VDMatch(config, "is420-scores")) return WP2_STATUS_OK;
  if (VDSelected(AsRect(pos_x, pos_y), config)) {
    config.info->selection_info += WP2SPrint(
        "%s score: %f = disto %d + lambda_u %f * rate_u %f + lambda_v "
        "%f * rate_v %f\n",
        reduced ? "420" : "444",
        disto + lambda_u * rate_u + lambda_v * rate_v, disto,
        lambda_u, rate_u, lambda_v, rate_v);
  }
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::Store420Decision(const EncoderConfig& config,
                                       uint32_t pos_x, uint32_t pos_y,
                                       Debug420Decision decision) const {
  if (!VDMatch(config, "is420-scores")) return WP2_STATUS_OK;

  if (VDSelected(AsRect(pos_x, pos_y), config)) {
    config.info->selection_info +=
        (decision == Debug420Decision::k444EarlyExit) ?
                  "Decision: 444 early exit (DC only)\n" :
        (decision == Debug420Decision::k444) ? "Decision: 444\n"
                                             : "Decision: 420\n";
  }

  ArgbBuffer& debug_output = config.info->debug_output;
  const Argb32b kColors[3] = {Argb32b{0xff, 0xff, 0x11, 0xff},
                              Argb32b{0xff, 0xff, 0x11, 0x11},
                              Argb32b{0xff, 0x11, 0xff, 0x11}};
  const Rectangle rect = AsRect().ClipWith(debug_output.AsRect());
  debug_output.Fill(rect, kColors[(int)decision]);
  debug_output.DrawRect(rect, Argb32b{0xff, 0x00, 0x00, 0x00}, 0x03);
  return WP2_STATUS_OK;
}

WP2Status CodedBlock::StoreLambdaMult(const EncoderConfig& config,
                                      uint32_t pos_x, uint32_t pos_y) const {
  if (!VDMatch(config, "lambda-mult")) return WP2_STATUS_OK;
  const bool selected = VDSelected(AsRect(pos_x, pos_y), config);

  if (selected) {
    config.info->selection_info += WP2SPrint("lambda-mult: %.3f", lambda_mult_);
  }

  ArgbBuffer& debug_output = config.info->debug_output;
  const Rectangle rect = AsRect(pos_x, pos_y).ClipWith(debug_output.AsRect());
  const uint8_t gray = (uint8_t)Clamp((lambda_mult_ - 1.0f) * 500.f + 128.f,
                                      0.f, 255.f);
  debug_output.Fill(rect, Argb32b{0xff, gray, gray, gray});
  if (selected) {
    debug_output.DrawRect(rect, Argb32b{0xff, 0xc0, 0x30, 0x00});
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void Tile::Draw(const DecoderConfig& config) {
  Argb32b color{0xff, 0xbb, 0xbb, 0x66};

  if (VDSelected(rect, config)) {
    color.r = color.g = color.b = 0xff;  // Highlight the tile's border.
    std::string& str = config.info->selection_info;
    str += WP2SPrint("\nTile at %u, %u px (%u x %u)\n", rect.x, rect.y,
                     rect.width, rect.height);
    assert(chunk_size_is_known);
    str += WP2SPrint("Tile size: %u bytes\n", chunk_size);
  }

  config.info->debug_output.DrawRect(rect, color);
}

//------------------------------------------------------------------------------

// Copies pixels in 'area'.
static void CopyArea(const YUVPlane& from, const Rectangle& area,
                     YUVPlane* const to) {
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    if (c == kAChannel && !from.HasAlpha()) continue;

    const Plane16& from_plane = from.GetChannel(c);
    Plane16* const to_plane = &to->GetChannel(c);

    for (uint32_t y = 0; y < area.height; ++y) {
      for (uint32_t x = 0; x < area.width; ++x) {
        const uint32_t px = area.x + x, py = area.y + y;
        to_plane->At(px, py) = from_plane.At(px, py);
      }
    }
  }
}

// Saves the unfiltered 'canvas' to compare with the filtered ones.
static void SavePixelsForVDebugDiff(const YUVPlane& canvas, uint32_t num_rows,
                                    YUVPlane* const unfiltered_pixels,
                                    uint32_t* const num_debug_rows) {
  if (unfiltered_pixels->Y.w_ != canvas.Y.w_ ||
      unfiltered_pixels->Y.h_ != canvas.Y.h_) {
    WP2_ASSERT_STATUS(unfiltered_pixels->Resize(canvas.Y.w_, canvas.Y.h_,
                                                /*pad=*/1, canvas.HasAlpha()));
  }

  // Copy the unfiltered available rows.
  assert(*num_debug_rows <= num_rows && num_rows <= canvas.Y.h_);
  CopyArea(canvas,
           {0, *num_debug_rows, canvas.Y.w_, num_rows - *num_debug_rows},
           unfiltered_pixels);
  *num_debug_rows = num_rows;
}

// Set pixels of 'debug_output' in 'rect' as the difference between the pixels
// of 'unfiltered_pixels' and 'filtered_pixels', scaled by 'multiplier'.
static void ApplyVDebugDiff(const YUVPlane& unfiltered_pixels,
                            const YUVPlane& filtered_pixels,
                            int32_t yuv_max_value, int32_t multiplier,
                            const Rectangle& rect,
                            ArgbBuffer* const debug_output) {
  assert(unfiltered_pixels.GetWidth() == filtered_pixels.GetWidth());
  assert(unfiltered_pixels.GetHeight() == filtered_pixels.GetHeight());
  assert(unfiltered_pixels.GetWidth() >= rect.width);
  assert(unfiltered_pixels.GetHeight() >= rect.height);
  YUVPlane diff_pixels;
  WP2_ASSERT_STATUS(diff_pixels.Resize(rect.width, rect.height));

  for (Channel c : {kYChannel, kUChannel, kVChannel}) {
    const Plane16& unfiltered = unfiltered_pixels.GetChannel(c);
    const Plane16& filtered = filtered_pixels.GetChannel(c);
    Plane16* const diff = &diff_pixels.GetChannel(c);
    for (uint32_t y = 0; y < rect.height; ++y) {
      for (uint32_t x = 0; x < rect.width; ++x) {
        diff->At(x, y) =
            Clamp((filtered.At(x, y) - unfiltered.At(x, y)) * multiplier,
                  -yuv_max_value, yuv_max_value);
      }
    }
  }
  // TODO(yguyon): Use the actual CSPTransform
  ArgbBuffer debug_output_view(debug_output->format());
  WP2_ASSERT_STATUS(debug_output_view.SetView(*debug_output, rect));
  WP2_ASSERT_STATUS(diff_pixels.Export(
      CSPTransform(), /*resize_if_needed=*/false, &debug_output_view));
}
static void ApplyVDebugAlphaDiff(const YUVPlane& unfiltered_pixels,
                                 const YUVPlane& filtered_pixels,
                                 const Rectangle& rect,
                                 ArgbBuffer* const debug_output) {
  if (!unfiltered_pixels.HasAlpha()) return;
  assert(unfiltered_pixels.GetWidth() == filtered_pixels.GetWidth());
  assert(unfiltered_pixels.GetHeight() == filtered_pixels.GetHeight());
  assert(unfiltered_pixels.GetWidth() >= rect.width);
  assert(unfiltered_pixels.GetHeight() >= rect.height);

  Channel c = kAChannel;
  const Plane16& unfiltered = unfiltered_pixels.GetChannel(c);
  const Plane16& filtered = filtered_pixels.GetChannel(c);
  const int32_t kMultiplier = 4;  // Multiplier to make things more visible.
  for (uint32_t y = 0; y < rect.height; ++y) {
    uint8_t* row = debug_output->GetRow8(rect.y + y);
    for (uint32_t x = 0; x < rect.width; ++x) {
      const int32_t diff = filtered.At(x, y) - unfiltered.At(x, y);
      const int32_t v = diff * kMultiplier * 255 / (int32_t)kAlphaMax + 128;
      const uint32_t abs_x = rect.x + x;
      row[4 * abs_x + 0] = 255;
      row[4 * abs_x + 1] = row[4 * abs_x + 2] = row[4 * abs_x + 3] =
          Clamp<int32_t>(v, 0, 255);
    }
  }
}

//------------------------------------------------------------------------------
// Filters

// Simple tool for gathering and displaying min, max and average values.
class MinMaxAvg {
 public:
  void Add(double value) {
    if (num_values_ == 0) min_ = max_ = sum_ = value;
    min_ = std::min(min_, value), max_ = std::max(max_, value), sum_ += value;
    ++num_values_;
  }

  uint32_t GetNumValues() const { return num_values_; }
  double GetMin() const { return min_; }
  double GetMax() const { return max_; }
  double GetAvg() const { return (num_values_ > 0) ? sum_ / num_values_ : 0; }

  std::string ToString() const {
    return WP2SPrint("min: %.3f, avg: %.3f, avg: %.3f (%u)",
                     min_, GetAvg(), max_, num_values_);
  }

 private:
  uint32_t num_values_ = 0;
  double sum_ = 0, min_ = 0, max_ = 0;
};


void FilterBlockMap::ApplyVDebug(const DecoderConfig& config,
                                 ArgbBuffer* const debug_output) {
  const int vd = VDMatch(config, "bpp") ? -1 : VDChannel(config);

  std::unordered_set<uint32_t> done;  // NOLINT (absl::flat_hash_set)
  for (uint32_t by = 0; by < tile_rect_.height / kMinBlockSizePix; ++by) {
    for (uint32_t bx = 0; bx < tile_rect_.width / kMinBlockSizePix; ++bx) {
      const FilterBlockMap::BlockFeatures& block_features =
          features_[by * max_num_blocks_x_ + bx];
      if (!done.insert(block_features.index).second) continue;

      const Rectangle rect =
          Rectangle(bx * kMinBlockSizePix, by * kMinBlockSizePix,
                    BlockWidthPix(block_features.size),
                    BlockHeightPix(block_features.size))
          .ClipWith(debug_output->AsRect());

      const uint8_t color =
          (vd == -1) ? block_features.min_bpp : block_features.res_den[vd];
      debug_output->Fill(rect, Argb32b{255, color, 0, 0});
      debug_output->DrawRect(rect, Argb32b{255, 128, 128, 128}, 0x03);

      if (VDSelected(tile_rect_.x, tile_rect_.y, rect, config)) {
        debug_output->DrawRect(rect, Argb32b{255, 255, 255, 255});

        std::string& str = config.info->selection_info;
        str += WP2SPrint("\nBlock at %u, %u px\n\n", rect.x, rect.y);

        str += WP2SPrint("Min bits-per-pixel: %f\n",
                         block_features.min_bpp / 128.);
        for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
          str += WP2SPrint("Residual density (%s): %f\n", kChannelStr[c],
                           block_features.res_den[c] / 255.);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

void DeblockingFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff")) {
    SavePixelsForVDebugDiff(*blocks_.pixels_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  } else if (VDMatch(config_, "strength")) {
    if (num_rows > 0 && num_debug_rows_ == 0) {
      // Registered pixels won't fill the whole canvas so fill it with black.
      config_.info->debug_output.Fill(
          blocks_.tile_rect_, Argb32b{0xff, 0x00, 0x00, 0x00});
    }
    num_debug_rows_ = num_rows;
  }
}

static bool IsHorizontal(const DecoderConfig& config) {
  if (VDMatch(config, "horizontal")) return true;
  assert(VDMatch(config, "vertical"));
  return false;
}

void DeblockingFilter::RegisterPixelsForVDebug(
    const DecoderConfig& config, uint32_t q0_x, uint32_t q0_y,
    uint32_t max_half, uint32_t filtered_half, bool horizontal_filtering,
    Channel channel, uint32_t strength, uint32_t sharpness,
    bool before_deblocking, bool deblocked, uint32_t num_bits,
    const int16_t* const q0, uint32_t step) {
  if (VDMatch(config, "strength") && VDChannel(config) == channel &&
      IsHorizontal(config) == horizontal_filtering) {
    ArgbBuffer* const debug_output = &config.info->debug_output;

    if (!before_deblocking) {
      // Strength heatmap on R, inverted sharpness heatmap on G.
      const uint32_t r = DivRound(strength * 255, kDblkMaxStrength);
      const uint32_t g =
          DivRound((kDblkMaxSharpness - sharpness) * 255, kDblkMaxSharpness);

      // Display gradients in filtered direction.
      for (uint32_t i = filtered_half; i >= 1; --i) {
        debug_output->Fill(
            horizontal_filtering ? Rectangle{q0_x - i, q0_y, i * 2, 1}
                                 : Rectangle{q0_x, q0_y - i, 1, i * 2},
            deblocked ? Argb32b{255, (uint8_t)(r / i), (uint8_t)(g / i), 0}
                      : Argb32b{255, (uint8_t)(200 / i), (uint8_t)(200 / i),
                                (uint8_t)(255 / i)});
      }
    }

    const Rectangle rect =
        horizontal_filtering
            ? Rectangle{q0_x - filtered_half, q0_y, 2 * filtered_half, 1}
            : Rectangle{q0_x, q0_y - filtered_half, 1, 2 * filtered_half};
    if (VDSelected(rect, config)) {
      debug_output->DrawRect(rect, Argb32b{255, 255, 255, 255});

      std::string& str = config.info->selection_info;
      if (before_deblocking) {
        str += WP2SPrint("\nEdge near %u, %u px (%s)\n\n", q0_x, q0_y,
                         kChannelStr[channel]);

        str += WP2SPrint("  Half-length: %u\n", filtered_half);
        str += WP2SPrint("  Strength: %u\n", strength);
        str += WP2SPrint("  Sharpness: %u\n", sharpness);
        str += WP2SPrint("  Deblock threshold: %d\n\n",
                         DeblockThresholdFromSharpness(sharpness, num_bits));

        for (int i = -(int32_t)max_half; i < (int32_t)max_half; ++i) {
          if (i == 0) str += " |";
          str += WP2SPrint(" %4d", q0[i * (int32_t)step]);
        }
        str += "\n";
      } else {  // !before_deblocking
        for (int i = -(int32_t)max_half; i < (int32_t)max_half; ++i) {
          if (i == 0) str += " |";
          str += WP2SPrint(" %4d", q0[i * (int32_t)step]);
        }
        str += "\n\n";
        str += WP2SPrint("  Deblocked: %s\n", deblocked ? "yes" : "no");
      }
    }
  }
}

void DeblockingFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff") && num_rows == blocks_.tile_rect_.height) {
    if (VDMatch(config_, "a")) {
      ApplyVDebugAlphaDiff(unfiltered_pixels_, *blocks_.pixels_,
                           blocks_.tile_rect_, &config_.info->debug_output);
    } else {
      ApplyVDebugDiff(unfiltered_pixels_, *blocks_.pixels_,
                      blocks_.bit_depth_.max(), 16, blocks_.tile_rect_,
                      &config_.info->debug_output);
    }
  }
}


//------------------------------------------------------------------------------

void GrainFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff")) {
    SavePixelsForVDebugDiff(*blocks_.pixels_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  }
}

void GrainFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff") && num_rows == blocks_.tile_rect_.height) {
    ApplyVDebugDiff(unfiltered_pixels_, *blocks_.pixels_,
                    blocks_.bit_depth_.max(), 2, blocks_.tile_rect_,
                    &config_.info->debug_output);
  }
}

//------------------------------------------------------------------------------

void DirectionalFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff")) {
    SavePixelsForVDebugDiff(*blocks_.pixels_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  }
}

void DirectionalFilter::RegisterPixelsForVDebug(uint32_t from_x, uint32_t to_x,
                                                uint32_t from_y, uint32_t to_y,
                                                uint32_t primary_strength,
                                                uint32_t secondary_strength,
                                                uint32_t direction,
                                                uint32_t variance) {
  ArgbBuffer* const debug_output = &config_.info->debug_output;
  const Rectangle rect{blocks_.tile_rect_.x + from_x,
                       blocks_.tile_rect_.y + from_y,
                       to_x - from_x + 1,
                       to_y - from_y + 1};

  if (VDMatch(config_, "strength")) {
    const uint8_t primary_color =
        (uint8_t)DivRound(primary_strength * 255, kMaxPriStr);
    const uint8_t secondary_color =
        (uint8_t)DivRound(secondary_strength * 255, kMaxSecStr);

    debug_output->Fill(
        rect, Argb32b{255, primary_color, secondary_color, 0});
  } else if (VDMatch(config_, "variance")) {
    const uint8_t color =
        (uint8_t)Clamp(DivRound(WP2Log2Floor(variance >> 6) * 255, 16), 0, 255);

    debug_output->Fill(rect, Argb32b{255, 0, 0, color});
  } else if (VDMatch(config_, "direction")) {
    assert(direction < kDrctFltNumDirs);
    // Light checkerboard to distinguish areas where the direction is computed.
    debug_output->Fill(rect, (((from_x + from_y) / kDrctFltSize) & 1u)
                                     ? Argb32b{255, 40, 40, 40}
                                     : Argb32b{255, 0, 0, 0});

    // Draw the oriented tap pattern in the middle of the block.
    for (int32_t dist = -(int32_t)kDrctFltTapDist;
         dist <= (int32_t)kDrctFltTapDist; ++dist) {
      const int32_t cx =
          blocks_.tile_rect_.x + from_x + (int32_t)kDrctFltSize / 2;
      const int32_t cy =
          blocks_.tile_rect_.y + from_y + (int32_t)kDrctFltSize / 2;
      MarkPixelForVDebug(cx, cy, {255, 255, 255, 255}, debug_output);
      if (dist != 0) {
        const int32_t s = (dist < 0) ? -1 : 1, d = std::abs(dist) - 1;
        // Primary taps.
        MarkPixelForVDebug(cx + s * kDrctFltTapPos[direction][d][0],
                           cy + s * kDrctFltTapPos[direction][d][1],
                           {255, 255, 255, 127}, debug_output);
        // Secondary taps.
        MarkPixelForVDebug(
            cx + s * kDrctFltTapPos[(direction + 2u) % kDrctFltNumDirs][d][0],
            cy + s * kDrctFltTapPos[(direction + 2u) % kDrctFltNumDirs][d][1],
            {255, 127, 127, 255}, debug_output);
        MarkPixelForVDebug(
            cx + s * kDrctFltTapPos[(direction + 6u) % kDrctFltNumDirs][d][0],
            cy + s * kDrctFltTapPos[(direction + 6u) % kDrctFltNumDirs][d][1],
            {255, 127, 127, 255}, debug_output);
      }
    }
  }
}

void DirectionalFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff") && num_rows == blocks_.tile_rect_.height) {
    ApplyVDebugDiff(unfiltered_pixels_, *blocks_.pixels_,
                    blocks_.bit_depth_.max(), 16, blocks_.tile_rect_,
                    &config_.info->debug_output);
  }
}

//------------------------------------------------------------------------------

void RestorationFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff")) {
    SavePixelsForVDebugDiff(*blocks_.pixels_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  }
}

void RestorationFilter::RegisterPixelsForVDebug(uint32_t from_x, uint32_t to_x,
                                                uint32_t from_y,
                                                uint32_t to_y) {
  if (VDMatch(config_, "strength")) {
    ArgbBuffer* const debug_output = &config_.info->debug_output;
    const Rectangle rect{blocks_.tile_rect_.x + from_x,
                         blocks_.tile_rect_.y + from_y,
                         to_x - from_x + 1,
                         to_y - from_y + 1};
    debug_output->Fill(rect, Argb32b{255, 128, 128, 128});

    if (VDSelected(rect, config_)) {
      MinMaxAvg min_max_avg_strength;
      for (uint32_t y = from_y; y <= to_y; ++y) {
        for (uint32_t x = from_x; x <= to_x; ++x) {
          const uint8_t strength =
              filter_strength_map_[(y - from_y) * blocks_.tile_rect_.width +
                                   (x - from_x)];
          const uint8_t color = (uint8_t)DivRound(strength * 255, 63);
          MarkPixelForVDebug(blocks_.tile_rect_.x + x, blocks_.tile_rect_.y + y,
                             Argb32b{255, color, 0, 0}, debug_output);
          min_max_avg_strength.Add(strength / 63.);
        }
      }

      // Highlight the border.
      debug_output->Fill(rect, Argb32b{255, 255, 255, 255});
      std::string& str = config_.info->selection_info;
      str += WP2SPrint("\nStrength %s\n\n",
                       min_max_avg_strength.ToString().c_str());

      const uint32_t area = from_y / kWieFltHeight;
      for (Channel c : {kYChannel, kUChannel, kVChannel}) {
        str += WP2SPrint("Channel %s\n", kChannelStr[c]);
        for (uint32_t h_or_v : {0, 1}) {
          int32_t tap_weights[kWieFltNumTaps];
          std::copy(params_.half_tap_weights[c][area][h_or_v],
                    params_.half_tap_weights[c][area][h_or_v] + kWieFltTapDist,
                    tap_weights);
          WienerHalfToFullWgts(tap_weights, tap_weights);
          str += (h_or_v == 0) ? "  Horizontal:" : "  Vertical:  ";
          for (int32_t weight : tap_weights) {
            str += WP2SPrint(
                " %6.3f", (double)weight / (1u << (kWieFltNumBitsTapWgts - 1)));
          }
          str += '\n';
        }
      }
    }
  }
}

void RestorationFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff") && num_rows == blocks_.tile_rect_.height) {
    ApplyVDebugDiff(unfiltered_pixels_, *blocks_.pixels_,
                    blocks_.bit_depth_.max(), 16, blocks_.tile_rect_,
                    &config_.info->debug_output);
  }
}

//------------------------------------------------------------------------------

void IntertileFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(*config_, "diff")) {
    SavePixelsForVDebugDiff(canvas_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  } else if (VDMatch(*config_, "strength")) {
    if (num_rows > 0 && num_debug_rows_ == 0) {
      // Registered pixels won't fill the whole canvas so fill it with black.
      config_->info->debug_output.Fill(Argb32b{0xff, 0x00, 0x00, 0x00});
    }
    num_debug_rows_ = num_rows;
  }
}

void IntertileFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(*config_, "diff") && (num_rows == canvas_.Y.h_)) {
    const FilterBlockMap& blocks = tiles_layout_->GetTileAt(0, 0).block_map;
    ApplyVDebugDiff(unfiltered_pixels_, canvas_, blocks.bit_depth_.max(), 16,
                    {0, 0, canvas_.Y.w_, canvas_.Y.h_},
                    &config_->info->debug_output);
  }
}

//------------------------------------------------------------------------------

void AlphaFilter::SavePixelsForVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff")) {
    SavePixelsForVDebugDiff(*blocks_.pixels_, num_rows, &unfiltered_pixels_,
                            &num_debug_rows_);
  }
}

void AlphaFilter::ApplyVDebug(uint32_t num_rows) {
  if (VDMatch(config_, "diff") && num_rows == blocks_.tile_rect_.height) {
    ApplyVDebugAlphaDiff(unfiltered_pixels_, *blocks_.pixels_,
                         blocks_.tile_rect_, &config_.info->debug_output);
  }
}

//------------------------------------------------------------------------------

static void ApplyVDebugDiff(const ArgbBuffer& unfiltered_pixels,
                            const ArgbBuffer& filtered_pixels,
                            int32_t multiplier,
                            ArgbBuffer* const debug_output) {
  assert(unfiltered_pixels.format() == filtered_pixels.format());
  assert(unfiltered_pixels.width() == filtered_pixels.width());
  assert(unfiltered_pixels.height() == filtered_pixels.height());
  assert(unfiltered_pixels.format() == debug_output->format());
  assert(unfiltered_pixels.width() == debug_output->width());
  assert(unfiltered_pixels.height() == debug_output->height());
  const uint32_t num_channels = WP2FormatBpp(debug_output->format());
  uint32_t alpha_channel;
  if (!WP2FormatHasAlpha(debug_output->format(), &alpha_channel)) {
    alpha_channel = num_channels;
  }

  for (uint32_t y = 0; y < unfiltered_pixels.height(); ++y) {
    for (uint32_t x = 0; x < unfiltered_pixels.width(); ++x) {
      for (uint32_t c = 0; c < num_channels; ++c) {
        const int32_t unfiltered =
            (unfiltered_pixels.GetRow8(y))[x * num_channels + c];
        const int32_t filtered =
            (filtered_pixels.GetRow8(y))[x * num_channels + c];

        int32_t diff = (filtered - unfiltered) * multiplier;
        if (c == alpha_channel) {
          diff = 255 - std::abs(diff);  // Opaque by default.
        } else {
          diff = 128 + diff;  // Grey by default.
        }

        uint8_t* const pixel = debug_output->GetRow8(y) + x * num_channels + c;
        *pixel = (uint8_t)Clamp(diff, 0, 255);
      }
    }
  }
}

// Sets all R, G and B channels to the value of a single 'channel'.
static void ToGray(const ArgbBuffer& from, uint32_t channel,
                   ArgbBuffer* const to) {
  assert(from.width() == to->width() && from.height() == to->height());
  assert(from.format() == WP2_Argb_32 && to->format() == WP2_Argb_32);
  uint32_t kAlpha = 0;
  for (uint32_t y = 0; y < to->height(); ++y) {
    const uint8_t* const from_row = from.GetRow8(y);
    uint8_t* const to_row = to->GetRow8(y);
    for (uint32_t x = 0; x < to->width(); ++x) {
      const uint8_t* const from_pixel = from_row + x * 4;
      uint8_t* const to_pixel = to_row + x * 4;
      uint8_t gray = from_pixel[channel];
      if (channel != kAlpha) {  // Unpremultiply by alpha.
        uint8_t alpha = from_pixel[kAlpha];
        assert(gray <= alpha);
        if (alpha < 255) {
          gray = (alpha > 0) ? ((gray * 255 + alpha / 2) / alpha) : 0;
        }
      }
      to_pixel[0] = 255;
      to_pixel[1] = to_pixel[2] = to_pixel[3] = gray;
    }
  }
}

// Spreads the values of the 'channel' of 'src' into at most 'histogram_size'
// buckets. Returns the number of buckets used.
static uint32_t GetHistogram(const ArgbBuffer& src, uint32_t channel,
                             uint32_t max_num_buckets, uint32_t histogram[],
                             uint8_t* const min_value,
                             uint8_t* const max_value) {
  assert(max_num_buckets > 0);
  const uint32_t num_channels = WP2FormatBpp(src.format());
  const uint8_t* row = src.GetRow8(0) + channel;
  uint8_t min = row[0], max = row[0];
  for (uint32_t y = 0; y < src.height(); ++y) {
    for (uint32_t x = 0; x < src.width(); ++x) {
      const uint8_t value = row[x * num_channels];
      min = std::min(min, value);
      max = std::max(max, value);
    }
    row += src.stride();
  }
  if (min_value != nullptr) *min_value = min;
  if (max_value != nullptr) *max_value = max;

  std::fill(histogram, histogram + max_num_buckets, 0u);
  const uint32_t range = std::max(1, max - min);
  max_num_buckets = std::min(range, max_num_buckets);
  const uint32_t max_bucket = max_num_buckets - 1;

  row = src.GetRow8(0) + channel;
  for (uint32_t y = 0; y < src.height(); ++y) {
    for (uint32_t x = 0; x < src.width(); ++x) {
      const uint8_t value = row[x * num_channels];
      const uint32_t bucket =
          DivRound((uint32_t)(value - min) * max_bucket, range);
      ++histogram[bucket];
    }
    row += src.stride();
  }
  return max_num_buckets;
}

// Handles VisualDebug for "compressed" and "original". Declared in wp2_dec_i.h.
void ApplyVDebugBeforeAfter(const DecoderConfig& config,
                            const CSPTransform& csp_tranform,
                            const Tile& tile,
                            ArgbBuffer* const debug_output) {
  if (VDMatch(config, "original/diff")) {
    ArgbBuffer argb;
    WP2_ASSERT_STATUS(
        argb.Resize(debug_output->width(), debug_output->height()));
    YUVPlane non_padded;
    WP2_ASSERT_STATUS(non_padded.SetView(
        tile.yuv_output, {0, 0, tile.rect.width, tile.rect.height}));
    WP2_ASSERT_STATUS(
        non_padded.Export(csp_tranform, /*resize_if_needed=*/false, &argb));
    ApplyVDebugDiff(*debug_output, argb, 2, debug_output);
  } else if (VDMatch(config, "y") || VDMatch(config, "u") ||
             VDMatch(config, "v")) {
    if (VDMatch(config, "original")) {
      csp_tranform.Apply(debug_output, csp_tranform.GetYuvDepth(), 0, 0,
                         debug_output->width(), debug_output->height());
      ToGray(*debug_output, VDChannel(config) + 1, debug_output);
    } else if (VDMatch(config, "compressed")) {
      const Plane16& plane = tile.yuv_output.GetChannel(VDChannel(config));
      WP2_ASSERT_STATUS(
          plane.ToGray(debug_output, csp_tranform.GetYuvDepth()));
    }
  } else {
    const bool a = VDMatch(config, "a"), r = VDMatch(config, "r");
    const bool g = VDMatch(config, "g"), b = VDMatch(config, "b");
    if (a || r || g || b) {
      const uint32_t channel = a ? 0 : r ? 1 : g ? 2 : 3;
      if (VDMatch(config, "original")) {
        ToGray(*debug_output, channel, debug_output);
      } else if (VDMatch(config, "compressed")) {
        ArgbBuffer argb;
        WP2_ASSERT_STATUS(
            argb.Resize(debug_output->width(), debug_output->height()));
        YUVPlane non_padded_yuv;
        WP2_ASSERT_STATUS(non_padded_yuv.SetView(
            tile.yuv_output, {0, 0, tile.rect.width, tile.rect.height}));
        WP2_ASSERT_STATUS(non_padded_yuv.Export(
            csp_tranform, /*resize_if_needed=*/false, &argb));
        ToGray(argb, channel, debug_output);
      }
    }
  }

  if (VDMatch(config, "histogram") && VDSelected(tile.rect, config)) {
    // A gray image of either Y,U,V,R,G or B should already be in
    // 'debug_output'. The histogram will be based on that.
    ArgbBuffer selected_area(debug_output->format());
    Rectangle rect = config.info->selection.ClipWith(tile.rect);
    assert(rect.GetArea() != 0);
    rect.x -= tile.rect.x;
    rect.y -= tile.rect.y;
    WP2_ASSERT_STATUS(selected_area.SetView(*debug_output, rect));

    std::array<uint32_t, 32> histogram{0};
    uint8_t min_value, max_value;
    const uint32_t num_buckets =
        GetHistogram(selected_area, /*channel=*/1, histogram.size(),
                     histogram.data(), &min_value, &max_value);

    debug_output->DrawRect(rect, Argb32b{255, 255, 128, 128});
    std::string& str = config.info->selection_info;
    str += WP2SPrint("\nArea %u x %u at %u, %u\n", rect.width, rect.height,
                     rect.x + tile.rect.x, rect.y + tile.rect.y);
    str += WP2SPrint("\nHistogram [%u, %u]\n", min_value, max_value);
    for (uint32_t bucket = 0; bucket < num_buckets; ++bucket) {
      const float percent = 100.f * histogram[bucket] / rect.GetArea();
      str += WP2SPrint("  Bucket %2u: %5.2f%% (%8u)  ", bucket, percent,
                           histogram[bucket]);
      str += std::string(std::lround(percent), '|');
      str += "\n";
    }

    std::array<uint32_t, histogram.size()> clusters{0};
    std::array<uint32_t, kMaxNumSegments> centers{0};
    const uint32_t num_clusters = ClusterHistogram(
        histogram.data(), num_buckets, /*max_clusters=*/kMaxNumSegments,
        clusters.data(), centers.data());
    str += WP2SPrint("\nNum clusters: %u\n", num_clusters);
  }
}

//------------------------------------------------------------------------------
// Encoder-related

static constexpr BitDepth kYuvMaxBitDepth = {kYuvMaxPrec + 1,
                                             /*is_signed=*/true};

// Fills the background of the current tile with luma and display some info.
WP2Status PartitionScoreFunc::ClearVDebug() const {
  if (VDMatch(*config_, "encoder/partition")) {
    ArgbBuffer debug_output;
    WP2_CHECK_STATUS(
        debug_output.SetView(config_->info->debug_output, tile_rect_));
    if (VDMatch(*config_, "method")) {
      const PartitionMethod pm =
          FinalPartitionMethod(*config_, tile_rect_.width, tile_rect_.height);
      if (VDSelected(tile_rect_, *config_)) {
        debug_output.Fill(Argb32b{255, 255, 255, 255});
        std::string& str = config_->info->selection_info;
        str += WP2SPrint("\nTile at %4u, %4u (%2u x %2u px): %s\n",
                         tile_rect_.x, tile_rect_.y, tile_rect_.width,
                         tile_rect_.height, kPartitionMethodString[pm]);
      } else {
        WP2_CHECK_STATUS(src_->Y.ToGray(&debug_output, kYuvMaxBitDepth));
      }
      debug_output.DrawRect(
          debug_output.AsRect(),
          Argb32b{255, (uint8_t)(pm * 255 / NUM_PARTITION_METHODS),
                  (uint8_t)(100 + pm * 100 / NUM_PARTITION_METHODS),
                  (uint8_t)(50 + pm * 50 / NUM_PARTITION_METHODS)},
          0x03);
    } else if (VDMatch(*config_, "score")) {
      WP2_CHECK_STATUS(src_->Y.ToGray(&debug_output, kYuvMaxBitDepth));
    }
  }
  return WP2_STATUS_OK;
}

bool PartitionScoreFunc::VDebugBlockSelected(const Block& block) const {
  // Top left corner of the block must be selected.
  return VDSelected(
      tile_rect_.x, tile_rect_.y,
      {block.x_pix(), block.y_pix(), kMinBlockSizePix, kMinBlockSizePix},
      *config_);
}

// Should be called after ClearVDebug() and for each block score computed.
WP2Status PartitionScoreFunc::RegisterScoreForVDebug(
    const Block& block, float score, bool force_selected,
    bool ending_new_line) const {
  if (!VDMatch(*config_, "encoder/partition/score") ||
      (!force_selected && !VDebugBlockSelected(block))) {
    return WP2_STATUS_OK;
  }
  if (config_->info != nullptr) {
    // Highlight the block with a color dependent on its size.
    const uint8_t color_w = (uint8_t)(191 + block.w() * 64 / kMaxBlockSize);
    const uint8_t color_h = (uint8_t)(191 + block.h() * 64 / kMaxBlockSize);
    config_->info->debug_output.Fill(
        {tile_rect_.x + block.x_pix() + 1, tile_rect_.y + block.y_pix() + 1,
         block.w_pix() - 2, block.h_pix() - 2},
        Argb32b{255, color_w, color_h, 0});

    std::string& str = config_->info->selection_info;
    str += WP2SPrint("Block at %4u, %4u (%2u x %2u px), score: %9.3f",
                     tile_rect_.x + block.x_pix(),
                     tile_rect_.y + block.y_pix(),
                     block.w_pix(), block.h_pix(), score);
    if (ending_new_line) str += "\n";
  }
  return WP2_STATUS_OK;
}

// Should be called after ClearVDebug() and for each block actually encoded.
WP2Status Partitioner::RegisterOrderForVDebug(
    uint32_t pass, uint32_t pass_index,
    const Block& block, uint32_t block_index, uint32_t max_num_blocks) const {
  if (VDMatch(*config_, "encoder/partition")) {
    ArgbBuffer* const debug_output = &config_->info->debug_output;
    const Rectangle rect =
      block.AsRect(tile_rect_.x, tile_rect_.y).ClipWith(debug_output->AsRect());

    if (VDMatch(*config_, "pass") || VDMatch(*config_, "order")) {
      Argb32b color;
      if (VDMatch(*config_, "pass")) {
        color.a = 255;
        color.r = color.g = color.b =
            (uint8_t)(255 - Clamp(pass_index * 12u, 0u, 255u));
      } else {
        color.a = 255;
        color.r = color.g = color.b =
            (uint8_t)(55u + DivRound(200u * block_index, max_num_blocks));
      }
      debug_output->Fill(rect, color);
      uint32_t selected_pass_index;
      if (VDMatch(*config_, "pass")) {
        if (VDIndex(config_->info->visual_debug, &selected_pass_index) &&
            pass_index == selected_pass_index) {
          debug_output->DrawRect(rect, Argb32b{255, 0, 0, 255}, 0x03);
        } else if (VDMatch(*config_, "all")) {
          const Argb32b kPassColor[] = {{255, 141, 202, 175},
                                        {255, 140, 73, 192},
                                        {255, 205, 104, 150},
                                        {255, 133, 148, 196}};
          STATIC_ASSERT_ARRAY_SIZE(kPassColor,
                                   (int)MultiScoreFunc::Pass::Any + 1);
          debug_output->DrawRect(rect, kPassColor[pass], 0x03);
        }
      }

      if (VDSelected(rect, *config_)) {
        // Highlight the border.
        debug_output->DrawRect(rect, Argb32b{255, 255, 0, 0});

        std::string& str = config_->info->selection_info;
        str += WP2SPrint(
            "Block at %u, %u (%u x %u px), pass: %u, order: %u\n",
            rect.x, rect.y, rect.width, rect.height, pass_index, block_index);
      }
    } else if (VDMatch(*config_, "score")) {
      // To have a reference, borders of actual encoded blocks are drawn.
      // Selected blocks are colored in RegisterScoreForVDebug().
      debug_output->DrawRect(rect, Argb32b{255, 64, 128, 64});
    }
  }
  return WP2_STATUS_OK;
}

bool SplitRecursePartitioner::VDebugBlockSelected(const Block& block) const {
  // Top left corner of the block must be selected.
  return VDSelected(
      tile_rect_.x, tile_rect_.y,
      {block.x_pix(), block.y_pix(), kMinBlockSizePix, kMinBlockSizePix},
      *config_);
}

WP2Status SplitRecursePartitioner::VDebugRecursionLevel(
    const Block& block, uint32_t recursion_level) const {
  if (!VDMatch(*config_, "encoder/partition/score")) {
    return WP2_STATUS_OK;
  }

  if (config_->info != nullptr) {
    std::string& str = config_->info->selection_info;
    for (uint32_t i = 0; i < recursion_level; ++i) {
      str += WP2SPrint(" ");
    }
  }
  return WP2_STATUS_OK;
}

WP2Status SplitRecursePartitioner::RegisterScoreForVDebug(
    const Block& block, uint32_t recursion_level, const float scores[4],
    float extra_rate, float total_score, bool is_best) const {
  if (!VDMatch(*config_, "encoder/partition/score")) {
    return WP2_STATUS_OK;
  }

  if (config_->info != nullptr) {
    WP2_CHECK_STATUS(VDebugRecursionLevel(block, recursion_level));

    std::string& str = config_->info->selection_info;
    str += WP2SPrint("Block at %4u, %4u (%2u x %2u px), score: %9.3f",
                     tile_rect_.x + block.x_pix(), tile_rect_.y + block.y_pix(),
                     block.w_pix(), block.h_pix(), total_score);

    str += WP2SPrint(" %s ", is_best ? "best" : "    ");
    str += WP2SPrint("  4 blks, recursive scores: ");
    for (uint32_t i = 0; i < 4; ++i) {
      str += WP2SPrint(" %.3f ", scores[i]);
    }
    str += WP2SPrint(" +r%.3f", extra_rate);
    str += "\n";
  }

  return WP2_STATUS_OK;
}

WP2Status BlockScoreFunc::RegisterScoreForVDebug(
    const Block blocks[], uint32_t num_blocks, const float rate[4],
    const float disto[4], float extra_rate, float total_rate, float total_disto,
    float score, bool is_best, bool force_selected) {
  Rectangle hull = blocks[0].rect();  // Print surrounding block if possible.
  for (uint32_t i = 1; i < num_blocks; ++i) {
    hull = hull.MergeWith(blocks[i].rect());
  }

  const Block hull_block(hull.x, hull.y, GetBlockSize(hull.width, hull.height));
  if (!VDMatch(*config_, "encoder/partition/score") ||
      (!force_selected && !VDebugBlockSelected(hull_block))) {
    return WP2_STATUS_OK;
  }

  WP2_CHECK_STATUS(PartitionScoreFunc::RegisterScoreForVDebug(
      hull_block, score, force_selected, /*ending_new_line=*/false));

  if (config_->info != nullptr) {
    std::string& str = config_->info->selection_info;
    str += WP2SPrint(" %s rate %3.2f disto %6.2f", is_best ? "best" : "    ",
                     total_rate, total_disto);
    if (num_blocks > 1) {  // Print sub-blocks stats.
     str += WP2SPrint("  %u blks: ", num_blocks);
     for (uint32_t i = 0; i < num_blocks; ++i) {
        str += WP2SPrint(" %ux%u(r%.2f,d%.2f)",
                         blocks[i].w_pix(), blocks[i].h_pix(),
                         rate[i], disto[i]);
     }
    }
    str += WP2SPrint(" +r%.3f", extra_rate);
    str += "\n";
  }
  return WP2_STATUS_OK;
}

WP2Status AreaScoreFunc::RegisterScoreForVDebug(
    BlockSize grid_size, const Vector<CodedBlock>& area_blocks, float score,
    float disto, float rate) const {
  // Only display something when the current 'area_' is selected.
  if (!VDMatch(*config_, "encoder/partition/score") ||
      !VDSelected(tile_rect_.x, tile_rect_.y, area_, *config_)) {
    return WP2_STATUS_OK;
  }

  std::string& str = config_->info->selection_info;
  if (grid_size == BLK_LAST) {  // Meaning default partitioning (happens once).
    str += WP2SPrint("\nArea at %4u, %4u (%2u x %2u px)\n",
                     tile_rect_.x + area_.x, tile_rect_.y + area_.y,
                     area_.width, area_.height);
    str += WP2SPrint("\nscore: %9.3f = disto %7.3f + rate %7.3f\n",
                     score, disto, rate);
    str += "  (default)\n";

    ArgbBuffer debug_output;
    WP2_ASSERT_STATUS(
        debug_output.SetView(config_->info->debug_output, tile_rect_));
    WP2_ASSERT_STATUS(buffer_.Export(
        gparams_->transf_, /*resize_if_needed=*/false, &debug_output));
    debug_output.DrawRect(area_, Argb32b{255, 255, 0, 0});
    for (const CodedBlock& cb : area_blocks) {
      debug_output.DrawRect(cb.AsRect(), Argb32b{255, 128, 255, 0}, 0x03);
    }
  } else {  // Grid partitioning (happens for each of several block sizes).
    str += WP2SPrint("\nscore: %9.3f = disto %7.3f + rate %7.3f\n",
                     score, disto, rate);
    str += WP2SPrint("  (grid of %2u x %2u px)\n",
                     BlockWidthPix(grid_size), BlockHeightPix(grid_size));
  }
  return WP2_STATUS_OK;
}

WP2Status SubAreaScoreFunc::RegisterScoreForVDebug(
    const Block& block, const Vector<CodedBlock>& area_remaining_blocks,
    float score, float disto, float rate) const {
  WP2_CHECK_STATUS(PartitionScoreFunc::RegisterScoreForVDebug(block, score));
  if (config_->info != nullptr) {
    std::string& str = config_->info->selection_info;
    str += WP2SPrint("  disto %7.3f + rate %7.3f", disto, rate);
    if (default_block_.dim() == BLK_LAST) str += " (default)";
    str += "\n";
  }
  return WP2_STATUS_OK;
}

WP2Status TileScoreFunc::RegisterScoreForVDebug(
    const char label[], const Block& block, float score) const {
  if (VDMatch(*config_, "encoder/partition/score")) {
    std::string& str = config_->info->selection_info;
    str +=
        WP2SPrint("\n%s: score %.6f, %4u blocks, %4u-byte tile", label, score,
                  (uint32_t)blocks_.size(),
                  (enc_tiles_layout_.tiles.front().data.size > 0)
                      ? enc_tiles_layout_.tiles.front().data.size
                      : enc_tiles_layout_.tiles.front().enc.GetBitstreamSize());
    if (block.dim() != BLK_LAST) {
      str += WP2SPrint(", on block at %3u,%3u (%2ux%2u)",
                       block.x_pix(), block.y_pix(),
                       block.w_pix(), block.h_pix());
    }
    str += WP2SPrint(", tile distortion %.4f dB", distortion_[4]);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status ExhaustivePartitioner::RegisterScoreForVDebug(
    float best_partition_score, uint64_t best_partition_size,
    uint64_t num_iterations) const {
  if (VDMatch(*config_, "encoder/partition/score")) {
    std::string& str = config_->info->selection_info;
    str += WP2SPrint("\n\nNum iterations: %u\n", (uint32_t)num_iterations);
    str += WP2SPrint("Best partition score: %8.3f size: %4u\n",
                     best_partition_score, (uint32_t)best_partition_size);
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status MultiPassPartitioner::RegisterPassForVDebug(
    MultiScoreFunc::Pass pass, BlockSize block_size,
    uint32_t num_chosen_blocks) const {
  if (!VDMatch(*config_, "encoder/partition/pass")) return WP2_STATUS_OK;
  if (!VDSelected(tile_rect_, *config_)) return WP2_STATUS_OK;
  const char* const kPassStr[] = {"FlatLumaAlpha", "NarrowStdDev", "Direction",
                                  "Any"};
  STATIC_ASSERT_ARRAY_SIZE(kPassStr, (int)MultiScoreFunc::Pass::Any + 1);
  std::string& str = config_->info->selection_info;
  str += WP2SPrint("Pass: %2u - %s - %2ux%2u - %4u blocks\n", pass_index_,
                   kPassStr[(int)pass], BlockWidthPix(block_size),
                   BlockHeightPix(block_size), num_chosen_blocks);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

// Displays the normalized value of the selected VDebug menu.
void MultiScoreFunc::DrawValueThresholdColor(
    uint32_t cell_width, uint32_t cell_height, const Block& block,
    ArgbBuffer* const debug_output) const {
  float value = 0.f;
  if (VDMatch(*config_, "luma-alpha-gradient")) {
    value = GetLumaAlphaGradient(block) / GetLumaAlphaGradientThreshold(block);
  } else if (VDMatch(*config_, "narrow-std-dev")) {
    value = GetStdDevRange(block) / GetStdDevRangeThreshold(block);
  } else if (VDMatch(*config_, "direction")) {
    value = GetDirection(block) / GetDirectionThreshold(block);
  } else {
    assert(false);
  }
  // Display as blue if passing threshold, otherwise as red. Dark blue means
  // close to the threshold, dark red means far from it.
  const float color[4] = {255.f, (value <= 1.f) ? 0.f : 55.f + 200.f / value,
                          0.f,
                          (value <= 1.f) ? 55.f + 200.f * (1.f - value) : 0.f};

  for (uint32_t sub_y = 0; sub_y < cell_height; ++sub_y) {
    if (block.y_pix() + sub_y >= debug_output->height()) break;
    uint8_t* const row = debug_output->GetRow8(block.y_pix() + sub_y);
    for (uint32_t sub_x = 0; sub_x < cell_width; ++sub_x) {
      if (block.x_pix() + sub_x >= debug_output->width()) break;
      uint8_t* const pixel = &row[(block.x_pix() + sub_x) * 4];
      for (uint32_t c : {1, 2, 3}) {
        pixel[c] = (uint8_t)std::lround((pixel[c] + color[c]) * 0.5f);
      }
    }
  }
}

// Displays the selected block border and info.
WP2Status MultiScoreFunc::DrawSelection(const Block& block,
                                        const Rectangle& block_rect,
                                        ArgbBuffer* const debug_output) const {
  Rectangle border = {SafeSub(block_rect.x, 1u), SafeSub(block_rect.y, 1u), 0,
                      0};
  border.width =
      std::min(block_rect.x + block_rect.width + 1u, debug_output->width()) -
      border.x;
  border.height =
      std::min(block_rect.y + block_rect.height + 1u, debug_output->height()) -
      border.y;
  debug_output->DrawRect(border, Argb32b{255, 255, 255, 0});

  if (config_->info == nullptr) return WP2_STATUS_OK;

  std::string& str = config_->info->selection_info;
  str += WP2SPrint("\nBlock at %4u, %4u (%2u x %2u px)\n",
                   tile_rect_.x + block.x_pix(), tile_rect_.y + block.y_pix(),
                   block.w_pix(), block.h_pix());
  const uint8_t segment_id = AssignSegmentId(
      *config_, *gparams_, {tile_rect_.x, tile_rect_.y, src_->Y.w_, src_->Y.h_},
      block);
  str += WP2SPrint("Segment id: %u\n", segment_id);
  str += WP2SPrint(
      "YA gradient: %5.3f = %5.2f / %5.2f\n",
      GetLumaAlphaGradient(block) / GetLumaAlphaGradientThreshold(block),
      GetLumaAlphaGradient(block), GetLumaAlphaGradientThreshold(block));
  str += WP2SPrint("Std dev:     %5.3f = %5.2f / %5.2f\n",
                   GetStdDevRange(block) / GetStdDevRangeThreshold(block),
                   GetStdDevRange(block), GetStdDevRangeThreshold(block));
  str += WP2SPrint("Direction:   %5.3f = %6.1f / %6.1f\n",
                   GetDirection(block) / GetDirectionThreshold(block),
                   GetDirection(block), GetDirectionThreshold(block));
  return WP2_STATUS_OK;
}

// Displays the orientations computed by MultiScoreFunc::ComputeDirection().
static void DrawDirection(const Rectangle& rect, uint32_t direction,
                          Argb32b color, ArgbBuffer* const debug_output) {
  // 4x4 mask per direction
  constexpr uint8_t kPatterns[kDrctFltNumDirs][4][4] = {
      {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 0, 0}},
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
      {{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}},
      {{0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}},
      {{0, 0, 1, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}}};
  assert(direction < 8 && rect.width <= 4 && rect.height <= 4);

  for (uint32_t y = 0; y < rect.height; ++y) {
    for (uint32_t x = 0; x < rect.width; ++x) {
      if (kPatterns[direction][y][x] == 1) {
        ToUInt8(color, debug_output->GetRow8(rect.y + y) +
                           (rect.x + x) * WP2FormatBpp(debug_output->format()));
      }
    }
  }
}

WP2Status MultiScoreFunc::DrawVDebug() const {
  if (!VDMatch(*config_, "encoder/partition/multi")) return WP2_STATUS_OK;
  ArgbBuffer debug_output;
  WP2_CHECK_STATUS(
      debug_output.SetView(config_->info->debug_output, tile_rect_));

  // Display the general orientation of all 4x4 blocks in the grid.
  if (VDMatch(*config_, "direction-4x4")) {
    // Display the luma source as background.
    WP2_CHECK_STATUS(src_->Y.ToGray(&debug_output, kYuvMaxBitDepth));
    for (uint32_t y = 0; y + 4 <= debug_output.height(); y += 4) {
      for (uint32_t x = 0; x + 4 <= debug_output.width(); x += 4) {
        const Rectangle block_rect = {x, y,
                                      std::min(4u, debug_output.width() - x),
                                      std::min(4u, debug_output.height() - y)};
        const uint32_t i = x / 4 + y / 4 * num_block_cols_;
        if (direction_certainty_[i] > 0) {  // Do not draw random directions.
          const uint8_t c =
              Clamp(direction_certainty_[i] * 255u / 3u, 0u, 255u);
          const Argb32b color = {255, c, (uint8_t)(c / 2), (uint8_t)(255 - c)};
          DrawDirection(block_rect, direction_[i], color, &debug_output);
        }
        if (VDSelected(tile_rect_.x, tile_rect_.y, block_rect, *config_)) {
          debug_output.DrawRect(block_rect, Argb32b{255, 0, 255, 0});
          const uint32_t angle_deg =
              (10 - (int32_t)direction_[i]) * 180 / kDrctFltNumDirs;
          std::string& str = config_->info->selection_info;
          str += WP2SPrint("Block at %4u, %4u: \n", x, y);
          str += WP2SPrint("  angle %u degrees\n", angle_deg);
          str += WP2SPrint("  certainty %u/3\n", direction_certainty_[i]);
        }
      }
    }
    return WP2_STATUS_OK;
  }

  // Original luma.
  WP2_CHECK_STATUS(src_->Y.ToGray(&debug_output, kYuvMaxBitDepth));

  Block block(0, 0,
              VDMatch(*config_, "32x32")   ? BLK_32x32
              : VDMatch(*config_, "16x16") ? BLK_16x16
              : VDMatch(*config_, "8x8")   ? BLK_8x8
                                           : BLK_4x4);
  const uint32_t cell_width =
      config_->partition_snapping ? block.w_pix() : kMinBlockSizePix;
  const uint32_t cell_height =
      config_->partition_snapping ? block.h_pix() : kMinBlockSizePix;
  Block selected_block = Block();

  // Display all blocks.
  for (uint32_t y = 0; y + block.h_pix() <= debug_output.height();
       y += cell_height) {
    for (uint32_t x = 0; x + block.w_pix() <= debug_output.width();
         x += cell_width) {
      block.SetXY(x / kMinBlockSizePix, y / kMinBlockSizePix);
      const Rectangle cell_rect =
          Rectangle(block.x_pix(), block.y_pix(), cell_width, cell_height)
              .ClipWith(debug_output.AsRect());
      DrawValueThresholdColor(cell_width, cell_height, block, &debug_output);
      if (VDSelected(tile_rect_.x, tile_rect_.y, cell_rect, *config_)) {
        selected_block = block;
      }
    }
  }

  // Display selected block on top.
  if (selected_block.dim() != BLK_LAST) {
    const Rectangle block_rect =
        selected_block.AsRect().ClipWith(debug_output.AsRect());
    WP2_CHECK_STATUS(DrawSelection(selected_block, block_rect, &debug_output));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

void GrainVDebug(const EncoderConfig& config, uint32_t x_pix, uint32_t y_pix,
                 Channel channel, FeatureMap::NoiseStr& noise_info,
                 int32_t* const coeffs) {
  if (!VDMatch(config, "encoder/grain") || VDMatch(config, "segment_y") ||
      VDMatch(config, "segment_uv")) {
    return;
  }
  if (channel != VDChannel(config)) return;
  const bool is_strength = VDMatch(config, "strength");
  ArgbBuffer* const debug_output = &config.info->debug_output;
  const Rectangle rect =
      Rectangle(x_pix, y_pix, kMinBlockSizePix, kMinBlockSizePix)
          .ClipWith(debug_output->AsRect());
  const uint16_t v = is_strength ? noise_info.strength[(int)channel]
                                 : noise_info.freq[(int)channel];
  const uint16_t num_values = is_strength ? 101 : 7;  // [0-100] or [0-6]
  const uint8_t gray = ToPixel8(v, num_values);
  debug_output->Fill(rect, Argb32b{255, gray, gray, gray});

  if (VDSelected(rect, config)) {
    std::string& str = config.info->selection_info;

    str += WP2SPrint("%u, %u px strength: %d freq: %d\n", x_pix, y_pix,
                     noise_info.strength[(int)channel],
                     noise_info.freq[(int)channel]);

    uint32_t k = 0;
    for (uint32_t j = 0; j < kMinBlockSizePix; ++j) {
      for (uint32_t i = 0; i < kMinBlockSizePix; ++i, ++k) {
        if (i == 0 && j == 0) {
          str += WP2SPrint("  X ");
        } else {
          str += WP2SPrint("%3d ", coeffs[k]);
        }
      }
      str += WP2SPrint("\n");
    }
    debug_output->DrawRect(rect, Argb32b{255, 255, 0, 0});
  }
}

void SegmentGrainVDebug(const EncoderConfig& config, const CodedBlock& cb,
                        uint32_t tile_pos_x, uint32_t tile_pos_y,
                        const Segment& segment) {
  if (!VDMatch(config, "encoder/grain/segments_y") &&
      !VDMatch(config, "encoder/grain/segments_uv")) {
    return;
  }

  ArgbBuffer* const debug_output = &config.info->debug_output;
  const bool is_strength = VDMatch(config, "strength");
  const bool is_y = VDMatch(config, "segments_y");
  const GrainParams& grain = segment.grain_;
  const uint16_t v = is_strength ? (is_y ? grain.y_ : grain.uv_)
                                 : (is_y ? grain.freq_y_ : grain.freq_uv_);
  const uint16_t num_values = is_strength ? 16 : 7;  // [0-15] or [0-6]
  const uint8_t gray = ToPixel8(v, num_values);
  const Rectangle& rect =
      cb.AsRect(tile_pos_x, tile_pos_y).ClipWith(debug_output->AsRect());
  debug_output->Fill(rect, Argb32b{255, gray, gray, gray});
  debug_output->Fill({tile_pos_x + cb.x_pix() + cb.w_pix() - 1,
                      tile_pos_y + cb.y_pix() + cb.h_pix() - 1, 1, 1},
                     kSegmentColors[cb.id_]);

  if (VDSelected(cb.AsRect(tile_pos_x, tile_pos_y), config)) {
    std::string& str = config.info->selection_info;
    str +=
        WP2SPrint("Segment %d Y strength %d feq %d | UV strength %d feq %d\n",
                  cb.id_, grain.y_, grain.freq_y_, grain.uv_, grain.freq_uv_);
  }
}

//------------------------------------------------------------------------------

void SegmentationGridVDebug(const EncoderConfig& config, const Rectangle& rect,
                            uint32_t tile_pos_x, uint32_t tile_pos_y,
                            float value) {
  const uint8_t gray = (uint8_t)std::lround(value * 10);
  ArgbBuffer* const debug_output = &config.info->debug_output;
  const Rectangle rect2 = { tile_pos_x + rect.x, tile_pos_y + rect.y,
                            rect.width, rect.height };
  if (VDSelected(rect2, config)) {
    config.info->debug_output.Fill(rect2, Argb32b{255, 128, 128, 0});

    std::string& str = config.info->selection_info;
    str += WP2SPrint("\nRect at %4u, %4u (%2u x %2u px), value: %9.3f\n",
                     tile_pos_x + rect.x, tile_pos_y + rect.y, rect.width,
                     rect.height, value);
  } else {
    config.info->debug_output.Fill(
      rect2.ClipWith(debug_output->AsRect()),
      Argb32b{255, gray, gray, gray});
  }
}

void SegmentationBlockVDebug(const EncoderConfig& config, const CodedBlock& cb,
                             uint32_t tile_pos_x, uint32_t tile_pos_y,
                             float score, const Segment& segment) {
  SegmentationGridVDebug(config, cb.AsRect(), tile_pos_x, tile_pos_y, score);

  config.info->debug_output.Fill(
      {tile_pos_x + cb.x_pix() + cb.w_pix() - 1,
       tile_pos_y + cb.y_pix() + cb.h_pix() - 1, 1, 1},
      kSegmentColors[cb.id_]);

  if (VDSelected(cb.AsRect(tile_pos_x, tile_pos_y), config)) {
    std::string& str = config.info->selection_info;
    str += WP2SPrint("Segment %d risk %f\n", cb.id_, segment.risk_);
  }
}

//------------------------------------------------------------------------------

}  // namespace WP2

//------------------------------------------------------------------------------
// Lossless decoder

namespace WP2L {

static bool VDSelectedLossless(uint32_t pos, uint32_t length,
                               const WP2::DecoderInfo& decoder_info,
                               const WP2::Rectangle& tile_rect) {
  const WP2::Rectangle& selection = decoder_info.selection;
  if (selection.width == 0 || selection.height == 0) return false;

  const uint32_t x = selection.x;
  const uint32_t y = selection.y;
  if (!tile_rect.Contains(x, y)) return false;
  const uint32_t selected_pos =
      (y - tile_rect.y) * tile_rect.width + (x - tile_rect.x);
  return (selected_pos >= pos && selected_pos < pos + length);
}

// Fills a copy-segment starting at 'pos' with length 'length' with 'color'.
static void Fill(uint32_t pos, uint32_t length, const WP2::Rectangle& tile_rect,
                 WP2::Argb32b color, WP2::ArgbBuffer* out) {
  const uint32_t tile_width = tile_rect.width;
  const uint32_t start_x = tile_rect.x + pos % tile_width;
  const uint32_t start_y = tile_rect.y + pos / tile_width;
  const uint32_t end_y = tile_rect.y + (pos + length) / tile_width;
  if (start_y == end_y) {
    out->Fill({start_x, start_y, length, 1}, color);
    return;
  }
  out->Fill({start_x, start_y, tile_width - pos % tile_width, 1}, color);
  if (end_y - start_y > 1) {
    out->Fill({tile_rect.x, start_y + 1, tile_width, end_y - start_y - 1},
              color);
  }
  out->Fill({tile_rect.x, end_y, (pos + length) % tile_width, 1}, color);
}

static const char* const kSymbolTypeNames[] = {"Literal", "Copy", "Cache"};
STATIC_ASSERT_ARRAY_SIZE(kSymbolTypeNames, kSymbolTypeNum);
static const char* const kGroup4ModeNames[] = {
    "Group4 Vertical", "Group4 Horizontal", "Group4 Pass"};
static const WP2::Argb32b kSymbolColors[2][3] = {
    {{255, 255, 0, 0}, {255, 0, 255, 0}, {255, 0, 0, 255}},       // SymbolType
    {{255, 255, 255, 0}, {255, 0, 255, 255}, {255, 255, 0, 255}}  // Group4
};

WP2Status Decoder::RegisterSymbolForVDebug(
    int symbol_type, bool is_group4, uint32_t pos, uint32_t distance,
    uint32_t length, float cost, WP2::DecoderInfo* const decoder_info) {
  assert(length > 0);
  if (decoder_info == nullptr) return WP2_STATUS_OK;
  if (!WP2::VDMatch(decoder_info->visual_debug, "bits-per-pixel/overall") &&
      !WP2::VDMatch(decoder_info->visual_debug, "lossless/symbols") &&
      !WP2::VDMatch(decoder_info->visual_debug, "lossless/transformed")) {
    return WP2_STATUS_OK;
  }

  if (WP2::VDMatch(decoder_info->visual_debug, "lossless/symbols")) {
    // Only show once per tile.
    if (pos == 0 && VDSelected(tile_->rect, config_)) {
      std::string& str = decoder_info->selection_info;
      str += WP2SPrint("Tile at (%d, %d)\n", tile_->rect.x, tile_->rect.y);
      if (use_group4_) {
        str += "Group 4\n";
        str += WP2SPrint("Move-to-front cache: %s\n",
                         mtf_.enabled() ? "enabled" : "disabled");
      } else {
        uint32_t cache_bits = hdr_.cache_config_.cache_bits;
        if (cache_bits > 0) {
          str += WP2SPrint("Color cache: %d bits\n", cache_bits);
        } else {
          str += "No color cache\n";
        }
      }
    }

    Fill(pos, length, tile_->rect, kSymbolColors[is_group4][symbol_type],
         &decoder_info->debug_output);
  }

  if (VDSelectedLossless(pos, length, *decoder_info, tile_->rect)) {
    if (WP2::VDMatch(decoder_info->visual_debug, "lossless/symbols")) {
      Fill(pos, length, tile_->rect, {255, 255, 255, 255},
           &decoder_info->debug_output);
    }
    const uint32_t tile_width = tile_->rect.width;
    std::string& str = decoder_info->selection_info;
    str += WP2SPrint("Pixel (%d, %d) Symbol at (%u, %u)\n",
                     decoder_info->selection.x, decoder_info->selection.y,
                     tile_->rect.x + pos % tile_width,
                     tile_->rect.y + pos / tile_width);
    str += WP2SPrint("Type %s ", is_group4 ? kGroup4ModeNames[symbol_type]
                                           : kSymbolTypeNames[symbol_type]);
    if (is_group4) {
      str += WP2SPrint("length %d ", length);
    } else if (symbol_type == kSymbolTypeCopy) {
      str += WP2SPrint("from (%d, %d) length %d ",
                       tile_->rect.x + (pos - distance) % tile_width,
                       tile_->rect.y + (pos - distance) / tile_width, length);
      // Do not display the length for single pixels.
    }
    str += WP2SPrint("cost %f bpp %f\n", cost, cost / length);
  }
  return WP2_STATUS_OK;
}

static const char* const kTransformNames[] = {
    "Predictor",      "Predictor with sub", "CrossColor", "YCoCgR",
    "SubstractGreen", "ColorIndexing",      "Group4"};
STATIC_ASSERT_ARRAY_SIZE(kTransformNames, (uint32_t)TransformType::kNum);
static const char* const kPredictorNames[] = {"Black",
                                              "180",
                                              "90",
                                              "45",
                                              "135",
                                              "Avg L,T,TR",
                                              "157",
                                              "Avg L,T",
                                              "113",
                                              "67",
                                              "Avg all",
                                              "Select L,TL,T",
                                              "L+T-TL",
                                              "(L+T)/2+((L+T)/2-TL)/2",
                                              "Clamped L+T-TL"};
STATIC_ASSERT_ARRAY_SIZE(kPredictorNames, (uint32_t)kNumPredictors);
static constexpr int kPredictorAngle[] = {0,   180, 90, 45, 135, 0, 157, 0,
                                          113, 67,  0,  0,  0,   0, 0};
STATIC_ASSERT_ARRAY_SIZE(kPredictorAngle, (uint32_t)kNumPredictors);

static void DisplayTileInfo(const WP2::DecoderConfig& config, uint32_t last_row,
                            const WP2::Rectangle& tile_rect,
                            WP2::DecoderInfo* const decoder_info) {
  if (last_row == 0 && VDSelected(tile_rect, config)) {
    std::string& str = decoder_info->selection_info;
    str += WP2SPrint("Tile at (%d, %d)\n", tile_rect.x, tile_rect.y);
  }
}

// Display an image buffer by modifying it according to the parameters (scale,
// channel shuffling).
static void DisplayBuffer(uint32_t last_row, uint32_t num_rows,
                          const WP2::Rectangle& tile,
                          const Transform& transform, const int16_t* const rows,
                          uint32_t num_bits, float offset, float scale,
                          bool discard_alpha, bool duplicate_green,
                          WP2::DecoderInfo* const decoder_info) {
  for (uint32_t y = 0; y < num_rows; ++y) {
    const uint32_t abs_y = tile.y + last_row + y;
    uint8_t* const row = decoder_info->debug_output.GetRow8(abs_y);
    for (uint32_t x = 0; x < tile.width; ++x) {
      const uint32_t abs_x = tile.x + x;
      const uint32_t offset_coord = (y * tile.width + x) * 4;
      if (decoder_info->selection.Contains(abs_x, abs_y)) {
        std::string& str = decoder_info->selection_info;
        // Display block info.
        if (transform.type_ == TransformType::kPredictor ||
            transform.type_ == TransformType::kPredictorWSub) {
          const uint32_t x_sub = x >> transform.bits_;
          const uint32_t y_sub = y >> transform.bits_;
          const uint32_t transform_width = transform.width_;
          const uint16_t predictor =
              transform.data_[y_sub * 4 * transform_width + 4 * x_sub + 2];
          assert(predictor < kNumPredictors);
          str += WP2SPrint("Block Predictor: %d %s\n", predictor,
                           kPredictorNames[predictor]);
        }

        str += WP2SPrint("Pixel (%d, %d) argb (%d, %d, %d, %d)\n",
                         decoder_info->selection.x, decoder_info->selection.y,
                         rows[offset_coord + 0], rows[offset_coord + 1],
                         rows[offset_coord + 2], rows[offset_coord + 3]);
      }
      // Display the pixel by scaling it according to the type.
      uint32_t c_min = 0, c_max = 3;
      if (discard_alpha) {
        row[abs_x * 4 + 0] = 255;
        c_min = 1;
      }
      if (duplicate_green) c_min = c_max = 2;
      for (uint32_t c = c_min; c <= c_max; ++c) {
        row[abs_x * 4 + c] = WP2::Clamp(
            (int)std::round(offset +
                            scale * WP2::RightShiftRound(rows[offset_coord + c],
                                                         num_bits - 8)),
            0, 255);
      }
      if (duplicate_green) {
        row[abs_x * 4 + 1] = row[abs_x * 4 + 3] = row[abs_x * 4 + 2];
      }
    }
  }
}

WP2Status Decoder::RegisterTransformedRowForVDebug(
    int index, uint32_t num_rows, uint32_t num_bits, const int16_t* const rows,
    WP2::DecoderInfo* const decoder_info) {
  if (decoder_info == nullptr) return WP2_STATUS_OK;
  assert(index >= -1 && index < (int)kPossibleTransformCombinationSize);
  const bool next_is_predictor =
      (index + 1 < (int)kPossibleTransformCombinationSize &&
       (transforms_[index + 1].type_ == TransformType::kPredictor ||
        transforms_[index + 1].type_ == TransformType::kPredictorWSub));
  if (next_is_predictor && WP2::VDMatch(decoder_info->visual_debug,
                                        "lossless/prediction/reference")) {
    // This case is when displaying the reference image before a transform.
  } else if (index >= 0) {
    const std::string menu = WP2SPrint("lossless/transformed/%d", index);
    if (!WP2::VDMatch(decoder_info->visual_debug, menu.c_str())) {
      return WP2_STATUS_OK;
    }
  } else {
    return WP2_STATUS_OK;
  }
  const Transform& transform = transforms_[index];

  // Only show once per tile.
  DisplayTileInfo(config_, last_row_, tile_->rect, decoder_info);
  if (last_row_ == 0 && VDSelected(tile_->rect, config_)) {
    std::string& str = decoder_info->selection_info;
    for (uint32_t i = 0; i < kPossibleTransformCombinationSize; ++i) {
      const Transform& transform_i = transforms_[i];
      if (transform_i.type_ == TransformType::kNum) break;
      // Highlight the used transform.
      if (i == (uint32_t)index) str += "=> ";
      str += WP2SPrint("Transform #%d: %s", i,
                       kTransformNames[(uint32_t)transform_i.type_]);
      if (transform_i.type_ == TransformType::kCrossColor ||
          transform_i.type_ == TransformType::kPredictor ||
          transform_i.type_ == TransformType::kPredictorWSub) {
        str += WP2SPrint(" Bits: %d", transform_i.bits_);
      } else if (transform_i.type_ == TransformType::kColorIndexing ||
                 transform_i.type_ == TransformType::kGroup4) {
        str += WP2SPrint(" Num colors: %d", num_colors_);
      }
      str += "\n";
    }
    str += "\n";
  }

  // Display the result of the transform.
  if (transform.type_ == TransformType::kSubtractGreen) {
    DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                  /*offset=*/0, /*scale=*/1.f, /*discard_alpha=*/false,
                  /*duplicate_green=*/false, decoder_info);
  } else if (transform.type_ == TransformType::kPredictor ||
             transform.type_ == TransformType::kPredictorWSub ||
             transform.type_ == TransformType::kCrossColor) {
    DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                  /*offset=*/128.f, /*scale=*/10.f, /*discard_alpha=*/true,
                  /*duplicate_green=*/false, decoder_info);
  } else if (transform.type_ == TransformType::kColorIndexing) {
    if (num_colors_ <= 200) {
      DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                    /*offset=*/55.f, /*scale=*/200.f / (num_colors_ - 1),
                    /*discard_alpha=*/true, /*duplicate_green=*/true,
                    decoder_info);
    } else if (num_colors_ <= 256) {
      DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                    /*offset=*/256 - num_colors_, /*scale=*/1.f,
                    /*discard_alpha=*/true, /*duplicate_green=*/true,
                    decoder_info);
    } else {
      DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                    /*offset=*/0.f, /*scale=*/1.f,
                    /*discard_alpha=*/true, /*duplicate_green=*/true,
                    decoder_info);
    }
  } else if (transform.type_ == TransformType::kYCoCgR) {
    DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                  /*offset=*/128.f, /*scale=*/1.f,
                  /*discard_alpha=*/false, /*duplicate_green=*/false,
                  decoder_info);
  }

  if (VDSelected(tile_->rect, config_)) {
    decoder_info->debug_output.DrawRect(tile_->rect,
                                        WP2::Argb32b{255, 255, 0, 0});
  }

  return WP2_STATUS_OK;
}

WP2Status Decoder::RegisterPredictedRowForVDebug(
    uint32_t index, uint32_t num_rows, uint32_t num_bits,
    const int16_t* const rows, WP2::DecoderInfo* const decoder_info) {
  if (decoder_info == nullptr || rows == nullptr) return WP2_STATUS_OK;
  assert(index >= 0 && index < kPossibleTransformCombinationSize);
  if (!WP2::VDMatch(decoder_info->visual_debug, "lossless/prediction/raw") &&
      !WP2::VDMatch(decoder_info->visual_debug, "lossless/cross-color/raw")) {
    return WP2_STATUS_OK;
  }
  const Transform& transform = transforms_[index];

  // Only show once per tile.
  DisplayTileInfo(config_, last_row_, tile_->rect, decoder_info);

  // Display the prediction of the transform.
  if (transform.type_ == TransformType::kPredictor ||
      transform.type_ == TransformType::kPredictorWSub) {
    DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                  /*offset=*/0, /*scale=*/1.f, /*discard_alpha=*/false,
                  /*duplicate_green=*/false, decoder_info);
  } else {
    // Offset because it is easier to visualize but keep the original scale.
    DisplayBuffer(last_row_, num_rows, tile_->rect, transform, rows, num_bits,
                  /*offset=*/128, /*scale=*/1.f, /*discard_alpha=*/true,
                  /*duplicate_green=*/false, decoder_info);
  }

  if (VDSelected(tile_->rect, config_)) {
    decoder_info->debug_output.DrawRect(tile_->rect,
                                        WP2::Argb32b{255, 255, 0, 0});
  }

  return WP2_STATUS_OK;
}

WP2Status Decoder::RegisterTransformForVDebug(
    const Transform& transform, WP2::DecoderInfo* const decoder_info) {
  if (decoder_info == nullptr) return WP2_STATUS_OK;
  if (!(WP2::VDMatch(decoder_info->visual_debug, "lossless/prediction/modes") &&
        (transform.type_ == TransformType::kPredictor ||
         transform.type_ == TransformType::kPredictorWSub)) &&
      !(WP2::VDMatch(decoder_info->visual_debug,
                     "lossless/cross-color/transform") &&
        transform.type_ == TransformType::kCrossColor)) {
    return WP2_STATUS_OK;
  }

  DisplayTileInfo(config_, last_row_, tile_->rect, decoder_info);

  // Display the transform.
  for (uint32_t y_sub = 0; y_sub < transform.height_; ++y_sub) {
    const uint32_t tsize = (1 << transform.bits_);
    const uint32_t block_y = tile_->rect.y + y_sub * tsize;
    for (uint32_t x_sub = 0; x_sub < transform.width_; ++x_sub) {
      const uint32_t block_x = tile_->rect.x + x_sub * tsize;
      const int16_t* const data =
          &transform.data_[y_sub * 4 * transform.width_ + 4 * x_sub];
      const WP2::Rectangle block_rect =
          WP2::Rectangle(block_x, block_y, tsize, tsize).ClipWith(tile_->rect);
      WP2::Argb32b color;
      if (transform.type_ == TransformType::kPredictor ||
          transform.type_ == TransformType::kPredictorWSub) {
        // Display predictors.
        const uint16_t predictor = data[2];
        const uint8_t c = WP2::ToPixel8(predictor, kNumPredictors);
        color = {255, c, c, c};
        decoder_info->debug_output.Fill(block_rect, color);

        if (kPredictorAngle[predictor] != 0) {
          DrawAnglePredictor(block_rect, kPredictorAngle[predictor],
                             &decoder_info->debug_output);
        }
      } else if (transform.type_ == TransformType::kCrossColor) {
        // Display cross-color.
        color = {255, (uint8_t)WP2::Clamp(128 + data[1], 0, 255),
                 (uint8_t)WP2::Clamp(128 + data[2], 0, 255),
                 (uint8_t)WP2::Clamp(128 + data[3], 0, 255)};
        decoder_info->debug_output.Fill(block_rect, color);
      }

      // Display block info.
      if (decoder_info->selection.Intersects(block_rect)) {
        std::string& str = decoder_info->selection_info;
        if (transform.type_ == TransformType::kPredictor ||
            transform.type_ == TransformType::kPredictorWSub) {
          str += WP2SPrint("Block Predictor: %d %s\n", data[2],
                           kPredictorNames[data[2]]);
        } else if (transform.type_ == TransformType::kCrossColor) {
          str += WP2SPrint("Block cross-color: GtoR: %d GtoB: %d RtoB: %d\n",
                           data[1], data[2], data[3]);
        }
      }
    }
  }

  if (VDSelected(tile_->rect, config_)) {
    decoder_info->debug_output.DrawRect(tile_->rect,
                                        WP2::Argb32b{255, 255, 0, 0});
  }
  return WP2_STATUS_OK;
}

WP2Status Decoder::HeaderVDebug() {
  WP2::DecoderInfo* const decoder_info = config_.info;
  if (decoder_info == nullptr) return WP2_STATUS_OK;

  if (!WP2::VDMatch(decoder_info->visual_debug, "lossless/clusters")) {
    return WP2_STATUS_OK;
  }

  const uint32_t bits = hdr_.histogram_subsample_bits_;
  std::string& str = decoder_info->selection_info;
  const uint32_t num_clusters = hdr_.sr_.symbols_info().NumClusters(0);
  const WP2::Rectangle& tile_rect = tile_->rect;
  if (VDSelected(tile_rect, config_)) {
    str += WP2SPrint("Tile at (%d, %d): %d cluster(s) %d histo_bits\n",
                     tile_rect.x, tile_rect.y, num_clusters, bits);
  }

  WP2::ArgbBuffer* const debug_output = &decoder_info->debug_output;
  if (num_clusters == 1) {
    debug_output->Fill(tile_rect, WP2::Argb32b{255, 0, 0, 0});
  } else {
    const uint32_t num_blocks_x = SubSampleSize(tile_rect.width, bits);
    const uint32_t num_blocks_y = SubSampleSize(tile_rect.height, bits);
    for (uint32_t y = 0; y < num_blocks_y; ++y) {
      for (uint32_t x = 0; x < num_blocks_x; ++x) {
        const WP2::Rectangle rect{
            tile_rect.x + (x << bits), tile_rect.y + (y << bits),
            std::min(1u << bits, tile_rect.width - (x << bits)),
            std::min(1u << bits, tile_rect.height - (y << bits))};
        const uint32_t cluster =
            hdr_.histogram_image_[4 * (num_blocks_x * y + x) + 2];
        const uint8_t gray = WP2::ToPixel8(cluster, num_clusters);
        debug_output->Fill(rect, WP2::Argb32b{255, gray, gray, gray});
        if (VDSelected(rect, config_)) {
          str += WP2SPrint("Rect (%d, %d) %d x %d : cluster %d\n",
                           rect.x, rect.y, rect.width, rect.height, cluster);
          debug_output->DrawRect(rect, WP2::Argb32b{255, 255, 0, 0});
        }
      }
    }
  }

  if (VDSelected(tile_rect, config_)) {
    debug_output->DrawRect(tile_rect, WP2::Argb32b{255, 0, 255, 0});
  }
  return WP2_STATUS_OK;
}

}  // namespace WP2L

#else

//------------------------------------------------------------------------------
// Code defined to compile:

namespace WP2 {

CodedBlock::SplitTf CodedBlock::GetForcedSplitTf(const EncoderConfig&, uint32_t,
                                                 uint32_t) const {
  return SplitTf::kUnknown;
}

TransformPair CodedBlock::GetForcedTransform(const EncoderConfig&, uint32_t,
                                             uint32_t) const {
  return kUnknownTf;
}

const Predictor* CodedBlock::GetForcedPredictor(const EncoderConfig&, uint32_t,
                                                uint32_t, const Predictors&,
                                                Channel) const {
  return nullptr;
}

}  // namespace WP2

#endif  // WP2_REDUCE_BINARY_SIZE
