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
// AV1 lossy encoding.

#include <cassert>

#include "src/common/av1_common.h"
#include "src/enc/tile_enc.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"

#if defined(WP2_HAVE_AOM)

#include "aom/aom_encoder.h"
#include "aom/aomcx.h"
#include "av1/encoder/encodeframe.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Converts 'rgb' into 'img' (YCbCr).
WP2Status CreateImage(const ArgbBuffer& rgb, aom_image_t* const img) {
  assert(!rgb.IsEmpty());
  WP2_CHECK_OK(!rgb.HasTransparency(), WP2_STATUS_UNSUPPORTED_FEATURE);

  YUVPlane ycbcr;
  WP2_CHECK_STATUS(ycbcr.Resize(rgb.width(), rgb.height()));
  // RGB->YCbCr matrix taken from http://www.mir.com/DMG/ycbcr.html (x 1<<12),
  // [0:255]->[16:235/240] range reduction included.
  constexpr int16_t kRGBToYCbCrMatrix[] = {1052, 2065, 401,   -607, -1192,
                                           1799, 1799, -1506, -293};
  constexpr uint32_t kRGBToYCbCrShift = 12;
  for (uint32_t y = 0; y < rgb.height(); ++y) {
    const uint8_t* const row = rgb.GetRow8(y);
    for (uint32_t x = 0; x < rgb.width(); ++x) {
      const uint8_t* argb = &row[x * WP2FormatBpp(rgb.format())];
      Multiply(argb[1], argb[2], argb[3], kRGBToYCbCrMatrix, kRGBToYCbCrShift,
               &ycbcr.Y.At(x, y), &ycbcr.U.At(x, y), &ycbcr.V.At(x, y));
    }
  }
  const bool is_monochrome = ycbcr.IsMonochrome();
  const aom_img_fmt_t aom_format =
      (is_monochrome ? AOM_IMG_FMT_I420 : AOM_IMG_FMT_I444);

  WP2_CHECK_ALLOC_OK(aom_img_alloc(img, aom_format, ycbcr.GetWidth(),
                                   ycbcr.GetHeight(), /*align=*/16) != NULL);
  ycbcr.Y.To(img->planes[AOM_PLANE_Y], img->stride[AOM_PLANE_Y], 16);
  if (is_monochrome) {
    img->monochrome = 1;
  } else {
    ycbcr.U.To(img->planes[AOM_PLANE_U], img->stride[AOM_PLANE_U], 128);
    ycbcr.V.To(img->planes[AOM_PLANE_V], img->stride[AOM_PLANE_V], 128);
  }
  return WP2_STATUS_OK;
}

// Initializes the 'codec'.
WP2Status CreateCodec(uint32_t w, uint32_t h, bool is_monochrome,
                      const EncoderConfig& config,
                      aom_codec_ctx_t* const codec) {
  const aom_codec_iface_t* const itf = aom_codec_av1_cx();
  WP2_CHECK_OK(itf != nullptr, WP2_STATUS_VERSION_MISMATCH);

  aom_codec_enc_cfg_t cfg;
  WP2_CHECK_STATUS(MapAomStatus(aom_codec_enc_config_default(itf, &cfg, 0)));
  cfg.g_w = w;
  cfg.g_h = h;
  cfg.g_limit = 1;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 25;
  cfg.g_bit_depth = AOM_BITS_8;
  cfg.g_usage =
      (config.effort == 0) ? AOM_USAGE_REALTIME : AOM_USAGE_GOOD_QUALITY;
  cfg.g_profile = (is_monochrome ? 0 : 1);  // yuv444 is 1, 0 otherwise.
  cfg.g_threads = 0;  // One thread per tile.
  cfg.rc_end_usage = AOM_Q;
  cfg.g_pass = AOM_RC_ONE_PASS;
  cfg.rc_twopass_stats_in = {NULL, 0};
  cfg.monochrome = is_monochrome;

  WP2_CHECK_STATUS(MapAomStatus(aom_codec_enc_init(codec, itf, &cfg, 0)));
  return WP2_STATUS_OK;
}

// Appends the bitstream chunks stored in 'codec' to 'enc'.
WP2Status ConcatenatePackets(aom_codec_ctx_t* const codec, Data* const enc,
                             bool* const got_packets) {
  *got_packets = false;
  aom_codec_iter_t iter = NULL;
  const aom_codec_cx_pkt_t* pkt = NULL;
  while ((pkt = aom_codec_get_cx_data(codec, &iter)) != NULL) {
    *got_packets = true;
    if (pkt->kind == AOM_CODEC_CX_FRAME_PKT) {
      WP2_CHECK_STATUS(
          enc->Append((const uint8_t*)pkt->data.frame.buf, pkt->data.frame.sz));
    }
  }
  return WP2_STATUS_OK;
}

}  // namespace

//------------------------------------------------------------------------------

WP2Status TileEncoder::Av1Encode(Data* const enc) {
  assert(enc->IsEmpty());

  // YCbCr input
  aom_image_t img;
  WP2_CHECK_STATUS(CreateImage(tile_->rgb_input, &img));
  Cleaner<aom_image_t> img_cleaner(&img, aom_img_free);

  // Codec instanciation
  aom_codec_ctx_t codec;
  WP2_CHECK_STATUS(CreateCodec(tile_->rgb_input.width(),
                               tile_->rgb_input.height(), img.monochrome,
                               *config_, &codec));
  Cleaner<aom_codec_ctx_t, aom_codec_err_t> codec_cleaner(&codec,
                                                          aom_codec_destroy);

  // Encoding settings
  const int level = std::lround(63 - config_->quality * 63 / kMaxLossyQuality);
  WP2_CHECK_STATUS(
      MapAomStatus(aom_codec_control(&codec, AOME_SET_CQ_LEVEL, level)));
  const int kTuneForSsim = 1;
  WP2_CHECK_STATUS(
      MapAomStatus(aom_codec_control(&codec, AOME_SET_TUNING, kTuneForSsim)));
  const int speed = 8 - DivRound(config_->effort * 8, kMaxEffort);
  WP2_CHECK_STATUS(
      MapAomStatus(aom_codec_control(&codec, AOME_SET_CPUUSED, speed)));

  // Encoding
  constexpr aom_codec_pts_t kPts = 0;  // Frame timestamp in animation.
  constexpr uint32_t kDuration = 1;    // Frame duration in milliseconds.
  WP2_CHECK_STATUS(MapAomStatus(
      aom_codec_encode(&codec, &img, kPts, kDuration, AOM_EFLAG_FORCE_KF)));

  // Flush
  bool got_paquets;
  do {
    WP2_CHECK_STATUS(MapAomStatus(
        aom_codec_encode(&codec, NULL, kPts, kDuration, AOM_EFLAG_FORCE_KF)));
    WP2_CHECK_STATUS(ConcatenatePackets(&codec, enc, &got_paquets));
  } while (got_paquets);
  WP2_CHECK_STATUS(MapAomStatus(codec_cleaner.Destruct()));

  // Wrap up
  WP2_CHECK_STATUS(tile_->progress.AdvanceBy(1.f));
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------

}  // namespace WP2

#else
namespace WP2 {
WP2Status TileEncoder::Av1Encode(Data* const) {
  fprintf(stderr, "Error: libwebp2 was not built with libaom.\n");
  return WP2_STATUS_UNSUPPORTED_FEATURE;
}
}  // namespace WP2
#endif  // defined(WP2_HAVE_AOM)
