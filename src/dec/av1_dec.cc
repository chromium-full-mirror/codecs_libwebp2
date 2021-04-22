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
// AV1 lossy decoding.

#include <cassert>

#include "src/common/av1_common.h"
#include "src/common/global_params.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/csp.h"
#include "src/utils/utils.h"
#include "src/wp2/decode.h"

#if defined(WP2_HAVE_AOM)

#include "aom/aom_decoder.h"
#include "aom/aomdx.h"

namespace WP2 {

//------------------------------------------------------------------------------

WP2Status Av1Decode(const BitstreamFeatures& features,
                    const DecoderConfig& config, const DataView& dec,
                    Tile* const tile) {
  // Codec instanciation
  const aom_codec_iface_t* const itf = aom_codec_av1_dx();
  WP2_CHECK_OK(itf != nullptr, WP2_STATUS_VERSION_MISMATCH);

  const aom_codec_dec_cfg cfg = {/*threads=*/1, tile->rect.width,
                                 tile->rect.height, /*allow_lowbitdepth=*/1};
  aom_codec_ctx_t codec;
  WP2_CHECK_STATUS(MapAomStatus(aom_codec_dec_init(&codec, itf, &cfg, 0)));
  Cleaner<aom_codec_ctx_t, aom_codec_err_t> codec_cleaner(&codec,
                                                          aom_codec_destroy);
  // Decoding settings
  WP2_CHECK_STATUS(
      MapAomStatus(aom_codec_control(&codec, AV1_SET_SKIP_LOOP_FILTER)));

  // Decoding
  WP2_CHECK_STATUS(MapAomStatus(
      aom_codec_decode(&codec, dec.bytes, dec.size, /*user_priv=*/NULL)));
  aom_codec_iter_t iter = NULL;
  aom_image_t* img = aom_codec_get_frame(&codec, &iter);
  WP2_CHECK_OK(img != NULL, WP2_STATUS_BITSTREAM_ERROR);
  YUVPlane ycbcr;
  WP2_CHECK_STATUS(ycbcr.Resize(tile->rect.width, tile->rect.height, /*pad=*/1,
                                !features.is_opaque));
  ycbcr.Y.From(img->planes[AOM_PLANE_Y], img->stride[AOM_PLANE_Y], -16);
  ycbcr.U.From(img->planes[AOM_PLANE_U], img->stride[AOM_PLANE_U], -128);
  ycbcr.V.From(img->planes[AOM_PLANE_V], img->stride[AOM_PLANE_V], -128);
  if (!ycbcr.A.IsEmpty()) {
    // Alpha should be handled but in the meantime it is set to opaque.
    ycbcr.A.Fill(kAlphaMax);
  }
  WP2_CHECK_STATUS(MapAomStatus(codec_cleaner.Destruct()));

  // Wrap up
  // YCbCr->RGB matrix from http://www.mir.com/DMG/ycbcr.html (x 1<<12),
  // [16:235/240]->[0:255] range increase included.
  constexpr int16_t kYCbCrToRGBMatrix[9] = {4769,  0,    6537, 4769, -1605,
                                            -3330, 4769, 8263, 0};
  constexpr uint32_t kYCbCrToRGBShift = 12;
  tile->output_is_yuv = false;
  WP2_CHECK_STATUS(ycbcr.Export(CSPMtx(kYCbCrToRGBMatrix, kYCbCrToRGBShift),
                                /*resize_if_needed=*/false, &tile->rgb_output));
  WP2_CHECK_STATUS(tile->row_progress.AdvanceBy(tile->rect.height));
  return WP2_STATUS_OK;
}

// -----------------------------------------------------------------------------

}  // namespace WP2

#else
namespace WP2 {
WP2Status Av1Decode(const BitstreamFeatures&, const DecoderConfig&,
                    const DataView&, Tile* const) {
  fprintf(stderr, "Error: libwebp2 was not built with libaom.\n");
  return WP2_STATUS_UNSUPPORTED_FEATURE;
}
}  // namespace WP2
#endif  // defined(WP2_HAVE_AOM)
