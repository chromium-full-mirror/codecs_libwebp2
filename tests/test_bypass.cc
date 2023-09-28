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

#include <cmath>
#include <cstdint>
#include <tuple>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/common/global_params.h"
#include "src/dec/tile_dec.h"
#include "src/enc/analysis.h"
#include "src/enc/tile_enc.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

class BypassTest
    : public testing::TestWithParam<std::tuple<
          const char*, bool, bool, WP2SampleFormat, WP2SampleFormat>> {};

// Encode decode raw pixels of a tile of maximum dimensions.
TEST_P(BypassTest, MaxChunkSize) {
  const char* const file_name = std::get<0>(GetParam());
  const bool use_lossy = std::get<1>(GetParam());
  const bool force_alpha = std::get<2>(GetParam());
  const float quality = use_lossy ? 50.f : 100.f;
  const uint32_t width = (uint32_t)std::lround(std::sqrt(kMaxTilePixels));
  const WP2SampleFormat src_format = std::get<3>(GetParam());
  const WP2SampleFormat dst_format = std::get<4>(GetParam());

  GlobalParams gparams;
  FeatureMap feature_map;
  gparams.features_ = &feature_map;
  EncTile enc_tile;
  MemoryWriter writer;
  {  // Encode
    ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath(file_name).c_str(),
                            &enc_tile.rgb_input));
    if (force_alpha) enc_tile.rgb_input.GetRow8(0)[0] = 42;
    if (src_format != enc_tile.rgb_input.format()) {
      ArgbBuffer tmp(src_format);
      using std::swap;
      swap(tmp, enc_tile.rgb_input);
      ASSERT_WP2_OK(enc_tile.rgb_input.ConvertFrom(tmp));
    }
    enc_tile.rect = {0, 0, width, width};
    ASSERT_EQ(enc_tile.rect.GetArea(), kMaxTilePixels);
    ASSERT_WP2_OK(
        enc_tile.rgb_input.SetView(enc_tile.rgb_input, enc_tile.rect));

    CSPTransform transf;
    if (use_lossy) {
      transf.InitYCoCg();
      ASSERT_WP2_OK(enc_tile.yuv_input.Import(
          enc_tile.rgb_input,
          /*import_alpha=*/enc_tile.rgb_input.HasTransparency(), transf,
          /*resize_if_needed=*/true, /*pad=*/kPredWidth));
    }
    EncoderConfig config;
    config.quality = quality;

    ASSERT_WP2_OK(GlobalAnalysis(enc_tile.rgb_input, enc_tile.yuv_input, transf,
                                 config, &gparams));
    ASSERT_WP2_OK(BypassTileEnc(gparams, &enc_tile, &writer));
  }

#if !defined(WP2_ENC_DEC_MATCH)
  if ((use_lossy || WP2Formatbpc(src_format) == 10) && gparams.has_alpha_) {
    EXPECT_EQ(writer.size_, kMaxTileChunkSize);
  } else {
    EXPECT_LT(writer.size_, kMaxTileChunkSize);
  }
#endif

  {  // Decode
    BitstreamFeatures features;
    features.rgb_bit_depth = WP2Formatbpc(src_format);
    TilesLayout tiles;
    tiles.num_tiles_x = tiles.num_tiles_y = 1;
    tiles.tile_width = tiles.tile_height = width;
    Tile dec_tile;
    dec_tile.chunk_size_is_known = true;
    dec_tile.chunk_size = writer.size_;
    dec_tile.input = &dec_tile.private_input;
    dec_tile.private_input.Update(writer.mem_, writer.size_);
    dec_tile.rect = enc_tile.rect;
    dec_tile.output_is_yuv = use_lossy;
    if (dec_tile.output_is_yuv) {
      ASSERT_WP2_OK(
          dec_tile.yuv_output.Resize(dec_tile.rect.width, dec_tile.rect.height,
                                     /*pad=*/kPredWidth, gparams.has_alpha_));
    } else {
      ASSERT_WP2_OK(dec_tile.rgb_output.SetFormat(dst_format));
      ASSERT_WP2_OK(dec_tile.rgb_output.Resize(dec_tile.rect.width,
                                               dec_tile.rect.height));
    }
    TileDecoder dec;
    dec.features_ = &features;
    dec.config_ = &DecoderConfig::kDefault;
    dec.gparams_ = &gparams;
    dec.tiles_layout_ = &tiles;
    dec.tile_ = &dec_tile;
    dec.self_assigning_ = false;
#if defined(WP2_ENC_DEC_MATCH)
    ASSERT_WP2_OK(dec.BypassTileDec());
#else
    // Make sure bypass is chosen.
    ASSERT_WP2_OK(dec.Execute());
#endif

    if (use_lossy) {
      ASSERT_TRUE(testutil::Compare(enc_tile.yuv_input, dec_tile.yuv_output,
                                    gparams.transf_.GetYuvDepth(),
                                    file_name));
    } else {
      ArgbBuffer converted_rgb_output(enc_tile.rgb_input.format());
      ASSERT_WP2_OK(converted_rgb_output.ConvertFrom(dec_tile.rgb_output));
      ASSERT_TRUE(testutil::Compare(enc_tile.rgb_input, converted_rgb_output,
                                    file_name));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BypassTestInstantiation, BypassTest,
    testing::Combine(testing::Values("taiwan.png"),
                     testing::Values(true, false),  // Lossy / lossless
                     testing::Values(true, false),  // Force alpha
                     testing::Values(WP2_Argb_32, WP2_Argb_38),  // input
                     testing::Values(WP2_Argb_32, WP2_Argb_38)   // output
                     ));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
