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

// Test ImageIo decoding a RGB image into a custom color space and vice versa.

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>

#include "extras/ccsp_imageio.h"
#include "extras/extras.h"
#include "imageio/file_format.h"
#include "imageio/image_dec.h"
#include "imageio/imageio_util.h"
#include "include/helpers.h"
#include "src/utils/csp.h"
#include "src/utils/plane.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

typedef std::tuple<const char*, const char*, float> Param;

class ComparisonTest : public testing::TestWithParam<Param> {};

TEST_P(ComparisonTest, RGBAndYUV) {
  const std::string rgb_file_name = std::get<0>(GetParam());
  const std::string ccsp_file_name = std::get<1>(GetParam());
  const float expected_distortion = std::get<2>(GetParam());

  ArgbBuffer rgb;
  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(rgb_file_name).c_str(), &rgb));

  YUVPlane ccsp;
  CSPMtx ccsp_to_rgb = {};
  Metadata metadata;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath(ccsp_file_name).c_str(),
                          &ccsp, &ccsp_to_rgb, &metadata));

  ArgbBuffer converted;
  ASSERT_WP2_OK(ccsp.Export(ccsp_to_rgb, /*resize_if_needed=*/true, &converted,
                            /*upsample_if_needed=*/&SamplingTaps::kUpSmooth));

  EXPECT_TRUE(testutil::Compare(rgb, converted,
                                rgb_file_name + "/" + ccsp_file_name,
                                expected_distortion));
  if (rgb_file_name == ccsp_file_name) {
    EXPECT_TRUE(testutil::HasSameData(rgb.metadata_.iccp, metadata.iccp));
    EXPECT_TRUE(testutil::HasSameData(rgb.metadata_.xmp, metadata.xmp));
    EXPECT_TRUE(testutil::HasSameData(rgb.metadata_.exif, metadata.exif));
  }
}

// Command used to generate y4m files (with yuv420p, yuv444p12le etc.):
//   ffmpeg -i source3.jpg -pix_fmt yuv444p10le -strict -1 source3_C444p10.y4m
INSTANTIATE_TEST_SUITE_P(
    ComparisonTestInstantiation, ComparisonTest,
    testing::Values(
        // Identical files should give the exact same result.
        Param{"source1.png", "source1.png", 99.f},
        // Files converted from RGB to YUV should be better with more bits.
        Param{"source3.jpg", "ccsp/source3_C444p8.y4m", 50.f},
        Param{"source3.jpg", "ccsp/source3_C444p10.y4m", 55.f},
        Param{"source3.jpg", "ccsp/source3_C444p12.y4m", 59.f},
        // Downsampling loss outweights more precision bits.
        Param{"source3.jpg", "ccsp/source3_C420p8.y4m", 35.f},
        Param{"source3.jpg", "ccsp/source3_C420p10.y4m", 35.f},
        Param{"source3.jpg", "ccsp/source3_C420p12.y4m", 35.f}));

//------------------------------------------------------------------------------

class DownsampledTest : public testing::TestWithParam<Param> {};

TEST_P(DownsampledTest, CCSP) {
  const std::string file_name = std::get<0>(GetParam());
  const std::string downsampled_file_name = std::get<1>(GetParam());
  const std::string file_path = testutil::GetTestDataPath(file_name);
  const std::string downsampled_file_path =
      testutil::GetTestDataPath(downsampled_file_name);
  const float expected_distortion = std::get<2>(GetParam());

  YUVPlane ccsp, downsampled_ccsp;
  CSPMtx ccsp_to_rgb = {}, downsampled_ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(file_path.c_str(), &ccsp, &ccsp_to_rgb));
  ASSERT_WP2_OK(ReadImage(downsampled_file_path.c_str(), &downsampled_ccsp,
                          &downsampled_ccsp_to_rgb));
  ASSERT_FALSE(ccsp.IsDownsampled());
  ASSERT_TRUE(downsampled_ccsp.IsDownsampled());
  ASSERT_WP2_OK(downsampled_ccsp.Upsample());
  ASSERT_FALSE(downsampled_ccsp.IsDownsampled());

  BitDepth bit_depth, downsampled_bit_depth;
  ASSERT_WP2_OK(ReadBitDepth(file_path.c_str(), &bit_depth));
  ASSERT_WP2_OK(
      ReadBitDepth(downsampled_file_path.c_str(), &downsampled_bit_depth));
  ASSERT_EQ(bit_depth, downsampled_bit_depth);

  EXPECT_TRUE(testutil::Compare(ccsp, downsampled_ccsp, bit_depth,
                                file_name + "/" + downsampled_file_name,
                                expected_distortion));
  float disto[5];
  ASSERT_WP2_OK(downsampled_ccsp.GetDistortion(ccsp, bit_depth, PSNR, disto));
  EXPECT_EQ(disto[0], 99.f);  // Luma and alpha are not impacted by
  EXPECT_EQ(disto[1], 99.f);  // downsampling, only chroma.
}

INSTANTIATE_TEST_SUITE_P(
    DownsampledTestInstantiation, DownsampledTest,
    testing::Values(
        // Distortion is around 41 dB no matter the bit depth.
        Param{"ccsp/source3_C444p8.y4m", "ccsp/source3_C420p8.y4m", 40.f},
        Param{"ccsp/source3_C444p10.y4m", "ccsp/source3_C420p10.y4m", 40.f},
        Param{"ccsp/source3_C444p12.y4m", "ccsp/source3_C420p12.y4m", 40.f}));

//------------------------------------------------------------------------------

class Y4MTest : public testing::TestWithParam<const char*> {};

TEST_P(Y4MTest, ToYCbCr) {
  const char* const file_name = GetParam();
  Data data;
  ASSERT_WP2_OK(
      IoUtilReadFile(testutil::GetTestDataPath(file_name).c_str(), &data));
  ASSERT_EQ(GuessImageFormat(data.bytes, std::min(data.size, (size_t)64)),
            FileFormat::Y4M_444);

  BitDepth bit_depth;
  ASSERT_WP2_OK(ReadBitDepth(data.bytes, data.size, &bit_depth));

  YUVPlane ccsp;
  CSPMtx ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(data.bytes, data.size, &ccsp, &ccsp_to_rgb));

  ASSERT_GE(ccsp_to_rgb.shift, CSPTransform::kMtxShift);
  const BitDepth file_num_bits = {
      CCSPImageReader::kMinBitDepth +
          (ccsp_to_rgb.shift - CSPTransform::kMtxShift),
      /*is_signed=*/false};
  ASSERT_FALSE(ccsp.IsDownsampled());
  ASSERT_EQ(bit_depth, file_num_bits);

  // As 'ccsp' is already in YCbCr (Y4M_444), converting it should not be lossy.
  YUVPlane ycbcr;
  ASSERT_WP2_OK(ToYCbCr(ccsp, ccsp_to_rgb, file_num_bits,
                        /*downsample=*/nullptr, &ycbcr));
  EXPECT_TRUE(testutil::Compare(ccsp, ycbcr, bit_depth, file_name));
}

INSTANTIATE_TEST_SUITE_P(Y4MTestInstantiation, Y4MTest,
                         testing::Values("ccsp/source3_C444p8.y4m",
                                         "ccsp/source3_C444p10.y4m",
                                         "ccsp/source3_C444p12.y4m"));

//------------------------------------------------------------------------------

class MatrixConversionTest
    : public testing::TestWithParam<
          std::tuple<const char*, bool, BitDepth, const SamplingTaps*>> {};

TEST_P(MatrixConversionTest, Comparison) {
  const char* const file_name = std::get<0>(GetParam());
  const bool through_ycocg = std::get<1>(GetParam());
  const BitDepth ycbcr_bit_depth = std::get<2>(GetParam());
  const SamplingTaps* const downsample = std::get<3>(GetParam());

  Data data;
  ASSERT_WP2_OK(
      IoUtilReadFile(testutil::GetTestDataPath(file_name).c_str(), &data));
  const FileFormat file_format =
      GuessImageFormat(data.bytes, std::min(data.size, (size_t)64));
  ASSERT_NE(file_format, FileFormat::UNSUPPORTED);

  BitDepth bit_depth;
  ASSERT_WP2_OK(ReadBitDepth(data.bytes, data.size, &bit_depth));

  YUVPlane ccsp;
  CSPMtx ccsp_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(data.bytes, data.size, &ccsp, &ccsp_to_rgb));
  if (ccsp.IsDownsampled()) {
    ASSERT_WP2_OK(ccsp.Upsample());
  }

  ArgbBuffer rgb;
  ASSERT_WP2_OK(ccsp.Export(ccsp_to_rgb, /*resize_if_needed=*/true, &rgb));

  YUVPlane ycbcr_from_ccsp;
  if (through_ycocg) {
    ASSERT_WP2_OK(ccsp.Apply(ccsp_to_rgb.mtx(), ccsp_to_rgb.shift));
    ASSERT_WP2_OK(ccsp.Apply(kRGBToYCoCgMatrix, kRGBToYCoCgShift));
    // Try converting to YCbCr with a matrix which is not the identity neither
    // the inverse (YCbCr>RGB), so why not YCoCg.
    ASSERT_WP2_OK(ToYCbCr(ccsp, CSPMtx(kYCoCgToRGBMatrix, kYCoCgToRGBShift),
                          ycbcr_bit_depth, downsample, &ycbcr_from_ccsp));
  } else {
    ASSERT_WP2_OK(ToYCbCr(ccsp, ccsp_to_rgb, ycbcr_bit_depth, downsample,
                          &ycbcr_from_ccsp));
  }
  YUVPlane ycbcr_from_rgb;
  ASSERT_WP2_OK(ToYCbCr(rgb, ycbcr_bit_depth, downsample, &ycbcr_from_rgb));

  float expected_distortion = 99.f;
  if (IsCustomColorSpace(file_format)) {
    // Some precision error might happen when reading CCSP directly into RGB.
    expected_distortion = 50.f;
    // The error margin increases with the output precision.
    expected_distortion -= 5.f * (ycbcr_bit_depth.num_bits - 8);
  }
  EXPECT_TRUE(testutil::Compare(ycbcr_from_ccsp, ycbcr_from_rgb, bit_depth,
                                file_name, expected_distortion));
}

INSTANTIATE_TEST_SUITE_P(
    MatrixConversionTestInstantiation, MatrixConversionTest,
    testing::Combine(testing::Values("source1_1x1.png",
                                     "ccsp/source3_C444p10.y4m"),
                     testing::Values(false, true),  // through_ycocg
                     testing::Values(BitDepth{8, /*is_signed=*/false},
                                     BitDepth{10, /*is_signed=*/false}),
                     testing::Values(&SamplingTaps::kDownSharp,
                                     &SamplingTaps::kDownAvg, nullptr)));

// Disabled to reduce the number of test cases.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_MoreMatrixConversionTestInstantiation, MatrixConversionTest,
    testing::Combine(testing::Values("alpha_ramp.pam", "source1_64x48.png",
                                     "source3.jpg", "ccsp/source3_C444p8.y4m",
                                     "ccsp/source3_C444p12.y4m",
                                     "alpha_ramp.bmp"),
                     testing::Values(false, true),  // through_ycocg
                     testing::Values(BitDepth{8, /*is_signed=*/false},
                                     BitDepth{10, /*is_signed=*/false},
                                     BitDepth{12, /*is_signed=*/false}),
                     testing::Values(&SamplingTaps::kDownSharp,
                                     &SamplingTaps::kDownAvg, nullptr)));

//------------------------------------------------------------------------------

// Returns OK if reading file_name both in ARGB and 'format' lead to the same
// samples.
WP2Status TestChannelOrder(const char* file_name, WP2SampleFormat format) {
  const std::string path = testutil::GetTestDataPath(file_name);
  ArgbBuffer argb(WP2_ARGB_32);
  WP2_CHECK_STATUS(ReadImage(path.c_str(), &argb));
  ArgbBuffer image(format);
  WP2_CHECK_STATUS(ReadImage(path.c_str(), &image));
  ArgbBuffer image_converted_to_argb(argb.format());
  WP2_CHECK_STATUS(image_converted_to_argb.ConvertFrom(image));
  WP2_CHECK_OK(testutil::Compare(image_converted_to_argb, argb, path),
               WP2_STATUS_INVALID_PARAMETER);
  return WP2_STATUS_OK;
}

TEST(ReadImageTest, ChannelOrder) {
  // WP2ConversionFunction() only returns conversion functions for all formats
  // from and to WP2_ARGB_32 or WP2_Argb_32 and between 32-bit formats. It also
  // supports plain copy, and ImageReaderPNG reads PNG samples in RGB(A) order.

  // Opaque
  EXPECT_WP2_OK(TestChannelOrder("taiwan.png", WP2_ARGB_32));
  EXPECT_WP2_OK(TestChannelOrder("taiwan.png", WP2_Argb_32));
  EXPECT_WP2_OK(TestChannelOrder("taiwan.png", WP2_RGB_24));

  // Alpha
  EXPECT_WP2_OK(TestChannelOrder("source1.png", WP2_ARGB_32));
  EXPECT_WP2_OK(TestChannelOrder("source1.png", WP2_RGBA_32));
  EXPECT_WP2_OK(TestChannelOrder("source1.png", WP2_BGRA_32));

  // RGBA -> premultiplied/opaque Argb -> ARGB implies quality loss.
  for (WP2SampleFormat format : {WP2_Argb_32, WP2_rgbA_32, WP2_bgrA_32,
                                 WP2_XRGB_32, WP2_RGBX_32, WP2_BGRX_32}) {
    EXPECT_EQ(TestChannelOrder("source1.png", format),
              WP2_STATUS_INVALID_PARAMETER);
  }

  // 24-bit to something else except WP2_Argb_32 and WP2_ARGB_32 is unsupported.
  for (WP2SampleFormat format :
       {WP2_RGBA_32, WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32,
        WP2_BGR_24, WP2_Argb_38}) {
    EXPECT_EQ(TestChannelOrder("taiwan.png", format),
              WP2_STATUS_INVALID_COLORSPACE);
  }

  // Expect all other combinations to fail.
  for (WP2SampleFormat format : {WP2_RGB_24, WP2_BGR_24, WP2_Argb_38}) {
    EXPECT_EQ(TestChannelOrder("source1.png", format),
              WP2_STATUS_INVALID_COLORSPACE);
  }
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
