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

// Compare YUVPlane and ArgbBuffer distortion computation.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "examples/example_utils.h"
#include "extras/ccsp_imageio.h"
#include "extras/extras.h"
#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/imageio_util.h"
#include "include/helpers.h"
#include "include/helpers_filter.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Returns true if 'ccsp' and 'rgb' contain the same samples.
testing::AssertionResult AreEqual(const YUVPlane& ccsp, const ArgbBuffer& rgb) {
  if (ccsp.GetWidth() != rgb.width() || ccsp.GetHeight() != rgb.height() ||
      ccsp.IsDownsampled()) {
    return testing::AssertionFailure();
  }

  for (uint32_t y = 0; y < ccsp.GetHeight(); ++y) {
    const uint8_t* const rgb_row = rgb.GetRow8(y);
    for (uint32_t x = 0; x < ccsp.GetWidth(); ++x) {
      const int32_t ccsp_a = ccsp.HasAlpha() ? ccsp.A.At(x, y) : kAlphaMax;
      const int32_t rgb_a = rgb_row[x * 4 + 0];
      const int32_t ccsp_r = ccsp.Y.At(x, y), rgb_r = rgb_row[x * 4 + 1];
      const int32_t ccsp_g = ccsp.U.At(x, y), rgb_g = rgb_row[x * 4 + 2];
      const int32_t ccsp_b = ccsp.V.At(x, y), rgb_b = rgb_row[x * 4 + 3];
      if (ccsp_a != rgb_a || ccsp_r != rgb_r || ccsp_g != rgb_g ||
          ccsp_b != rgb_b) {
        return testing::AssertionFailure()
               << " diff at " << x << ", " << y << ": " << ccsp_a << ", "
               << ccsp_r << ", " << ccsp_g << ", " << ccsp_b << " vs " << rgb_a
               << ", " << rgb_r << ", " << rgb_g << ", " << rgb_b;
      }
    }
  }
  return testing::AssertionSuccess();
}

//------------------------------------------------------------------------------

// Tests GetDistortion for all format/metric combinations.
TEST(GetDistortion, HandlesAllFormats) {
  for (uint32_t f = 0; f < WP2_Argb_38; ++f) {
    SCOPED_TRACE(SPrintf("format %d", f));
    const WP2SampleFormat format = (WP2SampleFormat)f;
    for (uint32_t metric = 0; metric < NUM_METRIC_TYPES; ++metric) {
      SCOPED_TRACE(SPrintf("metric %d", metric));
      ArgbBuffer orig(format);
      const uint32_t kWidth = 10, kHeight = 10;  // Big enough for PSNRHVS
      const uint8_t kOrigValue = 100;
      std::vector<uint8_t> data_orig(
          kWidth * kHeight * WP2FormatBpp(format) + 1, kOrigValue);
      const uint32_t stride = kWidth * WP2FormatBpp(format);
      ASSERT_WP2_OK(orig.SetExternal(kWidth, kHeight, &data_orig[0], stride));

      std::vector<uint8_t> data_copy = data_orig;
      // Make some changes that should have no impact.
      const uint8_t kChangedValue = kOrigValue / 2;
      // Change outside of the image.
      data_copy[data_copy.size() - 1] = kChangedValue;
      // Change nonexistent alpha channel.
      if (format == WP2_XRGB_32) data_copy[0] = kChangedValue;
      if (format == WP2_RGBX_32 || format == WP2_BGRX_32) {
        data_copy[3] = kChangedValue;
      }

      ArgbBuffer copy(orig.format());
      ASSERT_WP2_OK(copy.SetExternal(kWidth, kHeight, &data_copy[0], stride));

      float disto[5];
      const WP2Status status =
          copy.GetDistortion(orig, (MetricType)metric, disto);
      // Some format/metric combinations are not supported.
      if (status == WP2_STATUS_UNSUPPORTED_FEATURE ||
          status == WP2_STATUS_INVALID_PARAMETER) {
        continue;
      }

      ASSERT_WP2_OK(status);
      uint32_t alpha_channel_index;
      const bool has_alpha = WP2FormatHasAlpha(format, &alpha_channel_index);
      for (uint32_t c = 0; c < 4; ++c) {
        if (!has_alpha && c == alpha_channel_index) continue;
        EXPECT_EQ(disto[c], 99);
      }
      EXPECT_EQ(disto[4], 99);  // total disto == 99

      // Make a change that SHOULD cause some distortion.
      const uint32_t kChangedChannelIdx = 1;
      data_copy[kChangedChannelIdx] = kChangedValue;
      ASSERT_WP2_OK(copy.GetDistortion(orig, (MetricType)metric, disto));
      if (metric == LSIM) {
        // LSIM should still find 99 if only one pixel is changed (because it
        // matches with surrounding pixels).
        EXPECT_EQ(disto[4], 99);
        continue;
      }
      // Other metrics should find a difference.
      EXPECT_LT(disto[4], 99);  // total disto < 99

      // For argb metrics, only the changed channel should have a disto < 99.
      const bool is_yuv_metric =
          metric == PSNRHVS || metric == PSNR_YUV || metric == SSIM_YUV;
      if (is_yuv_metric) continue;

      for (uint32_t c = 0; c < 4; ++c) {
        if (!has_alpha && c == alpha_channel_index) continue;
        if (c == kChangedChannelIdx) {
          EXPECT_LT(disto[c], 99);
        } else {
          EXPECT_EQ(disto[c], 99);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

// Verify that distortion results are not bit depth-dependent.
TEST(DistoTest, BitDepth) {
  YUVPlane original_8b, distorted_8b;
  for (Channel c : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    Plane16& original_plane = original_8b.GetChannel(c);
    Plane16& distorted_plane = distorted_8b.GetChannel(c);
    ASSERT_WP2_OK(original_plane.Resize(512, 512));
    original_plane.Fill((c == kAChannel) ? 128 : 0);
    const int32_t min = (c == kAChannel) ? -128 : 0;
    const int32_t max = (c == kAChannel) ? 127 : 0;
    testutil::Noise(min, max, /*seed=*/c, /*strength=*/128, &original_plane);
    ASSERT_WP2_OK(
        distorted_plane.Copy(original_plane, /*resize_if_needed=*/true));
    testutil::Noise(min, max, /*seed=*/c, /*strength=*/5, &distorted_plane);
  }

  for (uint32_t bit_depth : {9, 10, 11, 12}) {  // Sign included.
    YUVPlane original, distorted;
    ASSERT_WP2_OK(original.Copy(original_8b, /*resize_if_needed=*/true));
    ASSERT_WP2_OK(distorted.Copy(distorted_8b, /*resize_if_needed=*/true));
    const int32_t m = 1 << (bit_depth - 8);
    const int32_t k8bTo12bMatrix[] = {m, 0, 0, 0, m, 0, 0, 0, m};
    ASSERT_WP2_OK(original.Apply(k8bTo12bMatrix, /*shift=*/0));
    ASSERT_WP2_OK(distorted.Apply(k8bTo12bMatrix, /*shift=*/0));

    for (MetricType metric : {PSNR, SSIM}) {
      float disto_8b[5], disto[5];
      ASSERT_WP2_OK(distorted_8b.GetDistortion(
          original_8b, /*bit_depth=*/{8, /*is_signed=*/true}, metric,
          disto_8b));
      ASSERT_WP2_OK(distorted.GetDistortion(
          original, {bit_depth, /*is_signed=*/true}, metric, disto));
      for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(disto_8b[i], disto[i], 1e-6) << "Channel " << i;
      }
    }
  }
}

//------------------------------------------------------------------------------

class DistoTest
    : public testing::TestWithParam<std::tuple<std::string, MetricType>> {};

// Compare RGB distortion for YUVPlane and ArgbBuffer structs. It should be
// equal for opaque images and might slightly differ otherwise (because some
// random color noise might be clamped during YUVPlane::Extract() as the
// samples are alpha-premultiplied).
TEST_P(DistoTest, ComparedToRGB) {
  const std::string file_path =
      testutil::GetTestDataPath(std::get<0>(GetParam()));
  const MetricType metric = std::get<1>(GetParam());

  // Load image into 'original_ccsp' and apply noise to get 'distorted_ccsp'.
  YUVPlane original_ccsp, distorted_ccsp;
  CSPMtx ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(file_path.c_str(), &original_ccsp, &ccsp_to_rgb));
  ASSERT_WP2_OK(distorted_ccsp.Copy(original_ccsp, /*resize_if_needed=*/true));
  // Distort alpha first then clamp distorted RGB samples.
  testutil::Noise(0, 255, /*seed=*/kAChannel, /*strength=*/5,
                  &distorted_ccsp.A);
  for (Channel channel : {kYChannel, kUChannel, kVChannel}) {
    Plane16* const plane = &distorted_ccsp.GetChannel(channel);
    testutil::Noise(0, 255, /*seed=*/channel, /*strength=*/5, plane);
    if (distorted_ccsp.HasAlpha()) {
      assert(!distorted_ccsp.IsDownsampled());
      for (uint32_t y = 0; y < plane->h_; ++y) {
        for (uint32_t x = 0; x < plane->w_; ++x) {
          plane->At(x, y) =
              std::min(plane->At(x, y), distorted_ccsp.A.At(x, y));
        }
      }
    }
  }

  // Copy both to ArgbBuffer (identity matrix, no shift).
  ASSERT_EQ(ccsp_to_rgb, CSPMtx({1, 0, 0, 0, 1, 0, 0, 0, 1}, 0));
  ArgbBuffer original_rgb, distorted_rgb;
  ASSERT_WP2_OK(original_ccsp.Export(ccsp_to_rgb, /*resize_if_needed=*/true,
                                     &original_rgb, &SamplingTaps::kUpSmooth));
  ASSERT_WP2_OK(distorted_ccsp.Export(ccsp_to_rgb, /*resize_if_needed=*/true,
                                      &distorted_rgb,
                                      &SamplingTaps::kUpSmooth));

  // Make sure data did not change during the copy.
  ASSERT_TRUE(AreEqual(original_ccsp, original_rgb));
  ASSERT_TRUE(AreEqual(distorted_ccsp, distorted_rgb));

  // Compute distortion with YUVPlane and ArgbBuffer.
  float disto_ccsp[5], disto_rgb[5];
  ASSERT_WP2_OK(distorted_ccsp.GetDistortion(
      original_ccsp, /*bit_depth=*/{8, /*is_signed=*/false}, metric,
      disto_ccsp));
  ASSERT_WP2_OK(distorted_rgb.GetDistortion(original_rgb, metric, disto_rgb));

  // Same metric for each channel.
  for (uint32_t i = 0; i < 5; ++i) {
    // The number of channels impacts the number of samples, hence a different
    // total distortion.
    const float error_margin = (original_ccsp.HasAlpha() && i == 4) ? 1.f : 0.f;
    EXPECT_NEAR(disto_ccsp[i], disto_rgb[i], error_margin) << i;
  }
}

// Compare distortion for down/upsampled chroma.
TEST_P(DistoTest, ChromaSubsampling) {
  const std::string file_path =
      testutil::GetTestDataPath(std::get<0>(GetParam()));
  const MetricType metric = std::get<1>(GetParam());

  // Load image into 'original_ccsp' and apply noise to get 'distorted_ccsp'.
  YUVPlane original_420, distorted_420;
  CSPMtx ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(file_path.c_str(), &original_420, &ccsp_to_rgb));
  if (!original_420.IsDownsampled()) {
    ASSERT_WP2_OK(original_420.Downsample());
  }

  ASSERT_WP2_OK(distorted_420.Copy(original_420, /*resize_if_needed=*/true));
  for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
    testutil::Noise(0, 255, /*seed=*/channel, /*strength=*/5,
                    &distorted_420.GetChannel(channel));
  }

  // 420 -> 444 with no interpolation for closer distortion.
  YUVPlane original_444, distorted_444;
  const bool upsamplable =
      (original_420.GetWidth() > 1 || original_420.GetHeight() > 1);
  ASSERT_WP2_OK(
      original_444.UpsampleFrom(original_420, SamplingTaps::kUpNearest));
  ASSERT_WP2_OK(
      distorted_444.UpsampleFrom(distorted_420, SamplingTaps::kUpNearest));

  // Compare 420 and 444 distortions.
  BitDepth bit_depth;
  ASSERT_WP2_OK(ReadBitDepth(file_path.c_str(), &bit_depth));
  float disto_420[5], disto_444[5];
  ASSERT_WP2_OK(
      distorted_420.GetDistortion(original_420, bit_depth, metric, disto_420));
  ASSERT_WP2_OK(
      distorted_444.GetDistortion(original_444, bit_depth, metric, disto_444));
  for (uint32_t i = 0; i < 5; ++i) {
    // SSIM uses some padding so the distortion differs when dimensions change.
    // TODO(yguyon): Modify SSIM to take that into account?
    const float error_margin =
        (metric == SSIM && upsamplable && i >= 2) ? 4.5f : 0.f;
    EXPECT_NEAR(disto_420[i], disto_444[i], error_margin) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    DistoTestInstantiation, DistoTest,
    testing::Combine(testing::Values("source0.pgm", "source0.ppm",
                                     "source3.jpg", "source1_1x48.png",
                                     "source1_64x1.png", "source1_1x1.png"),
                     testing::Values(PSNR, SSIM)));

//------------------------------------------------------------------------------
// Encode 'original_ccsp', either by converting it to RGB first or not.

WP2Status EncodeInRGB(const YUVPlane& original_ccsp, const CSPMtx& ccsp_to_rgb,
                      const EncoderConfig& config, Writer* const writer) {
  ArgbBuffer original_rgb;
  WP2_CHECK_STATUS(original_ccsp.Export(ccsp_to_rgb, /*resize_if_needed=*/true,
                                        &original_rgb,
                                        &SamplingTaps::kUpSmooth));
  WP2_CHECK_STATUS(Encode(original_rgb, writer, config));
  return WP2_STATUS_OK;
}

WP2Status EncodeInCCSP(const YUVPlane& original_ccsp, const CSPMtx& ccsp_to_rgb,
                       const EncoderConfig& config, Writer* const writer) {
  YUVPlane upsampled;
  if (original_ccsp.IsDownsampled()) {
    WP2_CHECK_STATUS(upsampled.UpsampleFrom(original_ccsp));
  } else {
    WP2_CHECK_STATUS(upsampled.SetView(
        original_ccsp,
        {0, 0, original_ccsp.GetWidth(), original_ccsp.GetHeight()}));
  }
  WP2_CHECK_STATUS(
      Encode(upsampled.GetWidth(), upsampled.GetHeight(), upsampled.Y.Row(0),
             upsampled.Y.Step(), upsampled.U.Row(0), upsampled.U.Step(),
             upsampled.V.Row(0), upsampled.V.Step(),
             upsampled.HasAlpha() ? upsampled.A.Row(0) : nullptr,
             upsampled.A.Step(), /*c_premultiplied_by_a=*/true,
             ccsp_to_rgb.mtx(), ccsp_to_rgb.shift, writer, config));
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Decode to RGB then convert or decode directly to 'decoded_ccsp'.

WP2Status DecodeInRGB(const uint8_t* const data, size_t data_size,
                      FileFormat format, BitDepth bit_depth, bool has_alpha,
                      bool is_downsampled, YUVPlane* const decoded_ccsp) {
  ArgbBuffer decoded_rgb;
  WP2_CHECK_STATUS(Decode(data, data_size, &decoded_rgb));

  if (IsCustomColorSpace(format)) {
    WP2_CHECK_STATUS(ToYCbCr(
        decoded_rgb, bit_depth,
        is_downsampled ? &SamplingTaps::kDownSharp : nullptr, decoded_ccsp));
  } else {
    const CSPMtx identity({1, 0, 0, 0, 1, 0, 0, 0, 1}, bit_depth.num_bits - 8);
    WP2_CHECK_STATUS(decoded_ccsp->Import(decoded_rgb, has_alpha, identity,
                                          /*resize_if_needed=*/true));
    if (is_downsampled) WP2_CHECK_ALLOC_OK(decoded_ccsp->Downsample());
  }
  return WP2_STATUS_OK;
}

WP2Status DecodeInCCSP(const uint8_t* const data, size_t data_size,
                       FileFormat format, BitDepth bit_depth, bool has_alpha,
                       bool is_downsampled, YUVPlane* const decoded_ccsp) {
  // Select output color space and bit depth.
  int16_t rgb_to_ccsp_matrix[9] = {};
  uint32_t rgb_to_ccsp_shift;
  if (IsCustomColorSpace(format)) {
    std::copy(kRGBToYCbCrMatrix, kRGBToYCbCrMatrix + 9, rgb_to_ccsp_matrix);
    rgb_to_ccsp_shift = kRGBToYCbCrShift - (bit_depth.num_bits - 8);
  } else {
    const int16_t v = 1 << (bit_depth.num_bits - 8);
    SetArray<int16_t>(rgb_to_ccsp_matrix, {v, 0, 0, 0, v, 0, 0, 0, v});
    rgb_to_ccsp_shift = 0;
  }

  BitstreamFeatures bitstream;
  WP2_CHECK_STATUS(bitstream.Read(data, data_size));

  // Allocate output and decode.
  WP2_CHECK_STATUS(decoded_ccsp->Resize(bitstream.width, bitstream.height,
                                        /*pad=*/1, has_alpha));
  WP2_CHECK_STATUS(Decode(
      data, data_size, rgb_to_ccsp_matrix, rgb_to_ccsp_shift,
      decoded_ccsp->Y.Row(0), decoded_ccsp->Y.Step(), decoded_ccsp->Y.Size(),
      decoded_ccsp->U.Row(0), decoded_ccsp->U.Step(), decoded_ccsp->U.Size(),
      decoded_ccsp->V.Row(0), decoded_ccsp->V.Step(), decoded_ccsp->V.Size(),
      decoded_ccsp->HasAlpha() ? decoded_ccsp->A.Row(0) : nullptr,
      decoded_ccsp->A.Step(), decoded_ccsp->A.Size()));
  if (is_downsampled) WP2_CHECK_ALLOC_OK(decoded_ccsp->Downsample());
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

enum class WhatToDo {
  kEncodeDecodeSameWay,
  kEncodeBothAsRGB,
  kEncodeBothAsCCSP,
  kDecodeBothAsRGB,
  kDecodeBothAsCCSP
};

class MorePreciseCCSPTest
    : public testing::TestWithParam<
          std::tuple<const char*, float, WhatToDo, MetricType>> {};

// Make sure encoding then decoding from and to custom color space samples does
// not bring more distortion than doing it in RGB.
TEST_P(MorePreciseCCSPTest, EncodeDecode) {
  const char* const file_name = std::get<0>(GetParam());
  const float quality = std::get<1>(GetParam());
  const WhatToDo what_to_do = std::get<2>(GetParam());
  const MetricType metric = std::get<3>(GetParam());

  Data data;
  ASSERT_WP2_OK(
      IoUtilReadFile(testutil::GetTestDataPath(file_name).c_str(), &data));
  const FileFormat format = GuessImageFormat(data.bytes, data.size);
  BitDepth bit_depth;
  ASSERT_WP2_OK(ReadBitDepth(data.bytes, data.size, &bit_depth));

  YUVPlane original_ccsp;
  CSPMtx ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(data.bytes, data.size, &original_ccsp, &ccsp_to_rgb));

  EncoderConfig config = EncoderConfig::kDefault;
  config.quality = quality;
  float disto_rgb[5], disto_ccsp[5];

  // RGB
  {
    MemoryWriter writer;
    if (what_to_do == WhatToDo::kEncodeBothAsCCSP) {
      ASSERT_WP2_OK(EncodeInCCSP(original_ccsp, ccsp_to_rgb, config, &writer));
    } else {
      ASSERT_WP2_OK(EncodeInRGB(original_ccsp, ccsp_to_rgb, config, &writer));
    }
    YUVPlane decoded_ccsp;
    if (what_to_do == WhatToDo::kDecodeBothAsCCSP) {
      ASSERT_WP2_OK(DecodeInCCSP(writer.mem_, writer.size_, format, bit_depth,
                                 original_ccsp.HasAlpha(),
                                 original_ccsp.IsDownsampled(), &decoded_ccsp));
    } else {
      ASSERT_WP2_OK(DecodeInRGB(writer.mem_, writer.size_, format, bit_depth,
                                original_ccsp.HasAlpha(),
                                original_ccsp.IsDownsampled(), &decoded_ccsp));
    }
    ASSERT_WP2_OK(decoded_ccsp.GetDistortion(original_ccsp, bit_depth, metric,
                                             disto_rgb));
  }

  // Custom Color Space
  {
    MemoryWriter writer;
    if (what_to_do == WhatToDo::kEncodeBothAsRGB) {
      ASSERT_WP2_OK(EncodeInRGB(original_ccsp, ccsp_to_rgb, config, &writer));
    } else {
      ASSERT_WP2_OK(EncodeInCCSP(original_ccsp, ccsp_to_rgb, config, &writer));
    }
    YUVPlane decoded_ccsp;
    if (what_to_do == WhatToDo::kDecodeBothAsRGB) {
      ASSERT_WP2_OK(DecodeInRGB(writer.mem_, writer.size_, format, bit_depth,
                                original_ccsp.HasAlpha(),
                                original_ccsp.IsDownsampled(), &decoded_ccsp));
    } else {
      ASSERT_WP2_OK(DecodeInCCSP(writer.mem_, writer.size_, format, bit_depth,
                                 original_ccsp.HasAlpha(),
                                 original_ccsp.IsDownsampled(), &decoded_ccsp));
    }
    ASSERT_WP2_OK(decoded_ccsp.GetDistortion(original_ccsp, bit_depth, metric,
                                             disto_ccsp));
  }

  // Custom color space should not be (way) more distorted than RGB.
  // Some loss is expected because of the lack of clamping by alpha for the CCSP
  // pipeline (even if actually handling RGB samples).
  const float error_margin =
      (IsCustomColorSpace(format) || original_ccsp.HasAlpha()) ? 1.5f : 0.01f;
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(disto_rgb[i], disto_ccsp[i], error_margin)
        << i << " in " << std::endl
        << disto_rgb[0] << ", " << disto_rgb[1] << ", " << disto_rgb[2] << ", "
        << disto_rgb[3] << ", " << disto_rgb[4] << " vs " << std::endl
        << disto_ccsp[0] << ", " << disto_ccsp[1] << ", " << disto_ccsp[2]
        << ", " << disto_ccsp[3] << ", " << disto_ccsp[4];
  }

  if (quality > (float)kMaxLossyQuality) {
    for (uint32_t i = 0; i < 5; ++i) {
      if (IsCustomColorSpace(format)) {
        // Lossless custom color space should never be more distorted than RGB.
        EXPECT_LE(disto_rgb[i], disto_ccsp[i]) << i;
      } else {
        // Lossless RGB or alpha input should give the exact same distortion.
        EXPECT_EQ(disto_rgb[i], disto_ccsp[i]) << i;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    MorePreciseCCSPTestInstantiationSmall, MorePreciseCCSPTest,
    testing::Combine(testing::Values("source1_1x1.png", "source1_64x48.png"),
                     testing::Values(0.f, 95.f, 100.f),
                     testing::Values(WhatToDo::kEncodeBothAsCCSP,
                                     WhatToDo::kDecodeBothAsCCSP),
                     testing::Values(PSNR, SSIM)));

// This one might take ages so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_MorePreciseCCSPTestInstantiation, MorePreciseCCSPTest,
    testing::Combine(
        testing::Values("source1_1x1.png", "source1_64x48.png", "source1.png",
                        "source3.jpg", "ccsp/source3_C444p8.y4m",
                        "ccsp/source3_C444p12.y4m", "ccsp/source3_C420p8.y4m",
                        "ccsp/source3_C420p12.y4m"),
        testing::Values(0.f, 75.f, 95.f, 100.f),
        testing::Values(WhatToDo::kEncodeDecodeSameWay,
                        WhatToDo::kEncodeBothAsCCSP, WhatToDo::kEncodeBothAsRGB,
                        WhatToDo::kDecodeBothAsCCSP,
                        WhatToDo::kDecodeBothAsRGB),
        testing::Values(PSNR, SSIM)));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
