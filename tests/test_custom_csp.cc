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

// Custom color space conversion test.

#include <tuple>
#include <vector>

#include "extras/ccsp_imageio.h"
#include "extras/extras.h"
#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/dsp/math.h"
#include "src/utils/csp.h"
#include "src/utils/random.h"
#include "src/wp2/base.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Matrices and associated shifts from extras.h
enum class Matrix { kRGB8, kRGB10, kYCoCg, kYpUV };
constexpr CSPMtx kRGBToCCSPMatrices[] = {{kRGBToRGB8Matrix, kRGBToRGB8Shift},
                                         {kRGBToRGB10Matrix, kRGBToRGB10Shift},
                                         {kRGBToYCoCgMatrix, kRGBToYCoCgShift},
                                         {kRGBToYpUVMatrix, kRGBToYpUVShift}};
constexpr CSPMtx kCCSPToRGBShifts[] = {{kRGB8ToRGBMatrix, kRGB8ToRGBShift},
                                       {kRGB10ToRGBMatrix, kRGB10ToRGBShift},
                                       {kYCoCgToRGBMatrix, kYCoCgToRGBShift},
                                       {kYpUVToRGBMatrix, kYpUVToRGBShift}};

constexpr const BitDepth kCCSPBitDepth[] = {{8, /*is_signed=*/false},
                                            {10, /*is_signed=*/false},
                                            {9, /*is_signed=*/true},
                                            {10, /*is_signed=*/true}};

//------------------------------------------------------------------------------

class CspMatrixTest : public ::testing::TestWithParam<std::tuple<Csp, Matrix>> {
  void SetUp() override { WP2CSPConverterInit(); }
};

TEST_P(CspMatrixTest, All) {
  const Csp csp_type = std::get<0>(GetParam());
  const Matrix matrix = std::get<1>(GetParam());
  const CSPMtx& rgb_to_ccsp = kRGBToCCSPMatrices[(int)matrix];
  const CSPMtx& ccsp_to_rgb = kCCSPToRGBShifts[(int)matrix];

  // YCoCg is the only lossless conversion from and to RGB. For the others, the
  // 'CSPTransform::im_' is not exactly the inverse matrix of 'm_' and thus
  // having more precision bits for the custom color space is not enough to keep
  // the exact same channel values because there is a temporary RGB conversion.
  const int16_t error_margin = (csp_type == Csp::kYCoCg) ? 0 : 1;

  CSPTransform transf;
  ASSERT_WP2_OK(transf.Init(csp_type));

  // Test most Argb combinations. It is easier than testing YUV combinations
  // that lead to valid Argb.
  constexpr int32_t kStep = 7;
  constexpr int16_t in_a = kAlphaMax;
  for (int16_t r = 0; r < in_a + kStep; r += kStep) {
    for (int16_t g = 0; g < in_a + kStep; g += kStep) {
      for (int16_t b = 0; b < in_a + kStep; b += kStep) {
        // Make sure valid extremes are tested.
        const int16_t in_r = std::min(r, in_a);
        const int16_t in_g = std::min(g, in_a);
        const int16_t in_b = std::min(b, in_a);

        int16_t in_y, in_u, in_v;
        transf.Rgb8ToYuv(in_r, in_g, in_b, &in_y, &in_u, &in_v);

        int16_t c0, c1, c2;
        const uint32_t width = 1, height = 1, step = 1;
        ASSERT_WP2_OK(transf.YuvToCustom(width, height, &in_y, step, &in_u,
                                         step, &in_v, step, rgb_to_ccsp, &c0,
                                         step, &c1, step, &c2, step));
        if (matrix == Matrix::kRGB8 || matrix == Matrix::kRGB10) {
          ASSERT_NEAR(RightShiftRound(c0, ccsp_to_rgb.shift), in_r,
                      error_margin);
          ASSERT_NEAR(RightShiftRound(c1, ccsp_to_rgb.shift), in_g,
                      error_margin);
          ASSERT_NEAR(RightShiftRound(c2, ccsp_to_rgb.shift), in_b,
                      error_margin);
        } else {
          ASSERT_LE(std::abs(c0), (1 << 11) - 1);  // Reasonable range.
          ASSERT_LE(std::abs(c1), (1 << 11) - 1);
          ASSERT_LE(std::abs(c2), (1 << 11) - 1);
        }

        int16_t out_y, out_u, out_v;
        ASSERT_WP2_OK(transf.CustomToYuv(width, height, &c0, step, &c1, step,
                                         &c2, step, ccsp_to_rgb, &out_y, step,
                                         &out_u, step, &out_v, step));

        ASSERT_NEAR(out_y, in_y, error_margin);
        ASSERT_NEAR(out_u, in_u, error_margin);
        ASSERT_NEAR(out_v, in_v, error_margin);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CspMatrixTestInstantiation, CspMatrixTest,
    ::testing::Combine(::testing::Values(Csp::kYCoCg, Csp::kYCbCr, Csp::kYIQ),
                       ::testing::Values(Matrix::kRGB8, Matrix::kRGB10,
                                         Matrix::kYCoCg, Matrix::kYpUV)));

//------------------------------------------------------------------------------
// Make sure the loss from the conversion RGB -> YUV -> RGB is low enough.

class YUVImportExportTest
    : public ::testing::TestWithParam<std::tuple<WP2SampleFormat, Csp, bool>> {
  void SetUp() override { WP2CSPConverterInit(); }
};

TEST_P(YUVImportExportTest, All) {
  const WP2SampleFormat format = std::get<0>(GetParam());  // input and output
  const Csp csp_type = std::get<1>(GetParam());  // temporary colorspace
  const bool import_alpha = std::get<2>(GetParam());

  CSPTransform transf;
  ASSERT_WP2_OK(transf.Init(csp_type));
  ArgbBuffer buffer(format), converted(format);
  ASSERT_WP2_OK(buffer.Resize(1, 1));
  ASSERT_WP2_OK(converted.Resize(1, 1));
  YUVPlane planes;
  ASSERT_WP2_OK(planes.Resize(1, 1, /*pad=*/1, import_alpha));

  // Test most Argb/ARGB combinations including extremes.
  constexpr int32_t kStep = 23;
  for (int16_t a = import_alpha ? 0 : kAlphaMax; a < (int32_t)kAlphaMax + kStep;
       a += kStep) {
    const int16_t in_a = std::min(a, (int16_t)kAlphaMax);

    const bool has_premultiplication_loss =
        (!WP2IsPremultiplied(format) && in_a < (int32_t)kAlphaMax);
    if (has_premultiplication_loss && in_a == 0.f) {
      // No point in checking further: r,g,b will be 0,0,0 no matter the input.
      continue;
    }

    int16_t in_max = WP2IsPremultiplied(format) ? in_a : kRgbMax;
    if (WP2Formatbpc(format) != 8) {
      assert(WP2Formatbpc(format) == 10);
      // For 10b, the maximum value is 1023 for RGB but 255 for A. Shifting
      // by 2 would limit to 1020, hence the division to keep the full range.
      in_max = DivRound<int32_t>(in_max * 1023, 255);
    }

    for (int16_t r = 0; r < in_max + kStep; r += kStep) {
      for (int16_t g = 0; g < in_max + kStep; g += kStep) {
        for (int16_t b = 0; b < in_max + kStep; b += kStep) {
          const int16_t in_r = std::min(r, in_max);
          const int16_t in_g = std::min(g, in_max);
          const int16_t in_b = std::min(b, in_max);

          // Set 8- or 10-bit input.
          if (WP2Formatbpc(format) == 8) {
            buffer.GetRow8(0)[0] = in_a;
            buffer.GetRow8(0)[1] = in_r;
            buffer.GetRow8(0)[2] = in_g;
            buffer.GetRow8(0)[3] = in_b;
          } else {
            buffer.GetRow16(0)[0] = in_a;
            buffer.GetRow16(0)[1] = in_r;
            buffer.GetRow16(0)[2] = in_g;
            buffer.GetRow16(0)[3] = in_b;
          }

          // Convert to then from YUV.
          ASSERT_WP2_OK(planes.Import(buffer, import_alpha, transf,
                                      /*resize_if_needed=*/false));
          ASSERT_WP2_OK(
              planes.Export(transf, /*resize_if_needed=*/false, &converted));

          // Only kYCoCg <-> Argb 8b is lossless.
          const float expected_distortion = has_premultiplication_loss   ? 25.f
                                            : (WP2Formatbpc(format) > 8) ? 40.f
                                            : (csp_type != Csp::kYCoCg)  ? 50.f
                                                                         : 99.f;
          ASSERT_TRUE(
              testutil::Compare(buffer, converted, "1x1", expected_distortion));
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    YUVImportExportTestInstantiation, YUVImportExportTest,
    ::testing::Combine(::testing::Values(WP2_Argb_32, WP2_ARGB_32, WP2_Argb_38),
                       ::testing::Values(Csp::kYCoCg, Csp::kYCbCr, Csp::kYIQ),
                       ::testing::Values(true, false)  // import_alpha
                       ));

//------------------------------------------------------------------------------

class BackAndForthTest
    : public ::testing::TestWithParam<std::tuple<const char*, Matrix>> {
  void SetUp() override { WP2CSPConverterInit(); }
};

TEST_P(BackAndForthTest, Comparison) {
  const char* const file_name = std::get<0>(GetParam());
  const Matrix matrix = std::get<1>(GetParam());
  const CSPMtx& rgb_to_ccsp = kRGBToCCSPMatrices[(int)matrix];
  const CSPMtx& ccsp_to_rgb = kCCSPToRGBShifts[(int)matrix];
  const BitDepth ccsp_bit_depth = kCCSPBitDepth[(int)matrix];

  YUVPlane original;
  CSPMtx file_to_rgb = {};
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath(file_name).c_str(),
                          &original, &file_to_rgb));
  if (original.IsDownsampled()) {
    ASSERT_WP2_OK(original.Upsample());
  }

  YUVPlane rgb;
  ASSERT_WP2_OK(rgb.Copy(original, /*resize_if_needed=*/true));
  const uint32_t kept_precision = std::min(file_to_rgb.shift, 1u);
  ASSERT_WP2_OK(
      rgb.Apply(file_to_rgb.mtx(), file_to_rgb.shift - kept_precision));

  YUVPlane converted;
  ASSERT_WP2_OK(converted.Copy(rgb, /*resize_if_needed=*/true));
  ASSERT_WP2_OK(converted.Apply(rgb_to_ccsp.mtx(), rgb_to_ccsp.shift));

  {
    YUVPlane bypass_rgb;
    ASSERT_WP2_OK(bypass_rgb.Copy(original, /*resize_if_needed=*/true));
    int32_t file_to_ccsp_matrix[9];
    Multiply(rgb_to_ccsp.mtx(), file_to_rgb.mtx(), file_to_ccsp_matrix);
    ASSERT_WP2_OK((bypass_rgb.Apply<int32_t, int64_t>(
        file_to_ccsp_matrix,
        /*shift=*/file_to_rgb.shift - kept_precision + rgb_to_ccsp.shift)));
    // The loss can only come from a high precision input matrix that forces a
    // right shift or a clamping of the samples (int16_t) during the RGB step.
    const float expected_distortion =
        (file_to_rgb.shift - kept_precision == 0) ? 99.f : 50.f;
    EXPECT_TRUE(testutil::Compare(converted, bypass_rgb, ccsp_bit_depth,
                                  file_name, expected_distortion));
  }

  {
    YUVPlane back_to_rgb;
    ASSERT_WP2_OK(back_to_rgb.Copy(converted, /*resize_if_needed=*/true));
    ASSERT_WP2_OK(back_to_rgb.Apply(ccsp_to_rgb.mtx(), ccsp_to_rgb.shift));
    // Almost no loss thanks to the 'kept_precision' above.
    EXPECT_TRUE(testutil::Compare(rgb, back_to_rgb, ccsp_bit_depth, file_name,
                                  /*expected_distortion=*/90.f));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BackAndForthTestInstantiation, BackAndForthTest,
    ::testing::Combine(::testing::Values("alpha_ramp.pam", "source1_1x1.png",
                                         "source1_64x48.png", "source3.jpg",
                                         "ccsp/source3_C420p8.y4m",
                                         "ccsp/source3_C420p10.y4m",
                                         "ccsp/source3_C420p12.y4m",
                                         "ccsp/source3_C444p8.y4m",
                                         "ccsp/source3_C444p10.y4m",
                                         "ccsp/source3_C444p12.y4m"),
                       ::testing::Values(Matrix::kRGB8, Matrix::kRGB10,
                                         Matrix::kYCoCg, Matrix::kYpUV)));

//------------------------------------------------------------------------------

TEST(CspCustomTest, Simple) {
  const std::string& file_name = "composited.png";

  // Make sure Csp::kCustom is going to be used and not the default Csp::kYCoCg.
  ArgbBuffer img;
  ASSERT_WP2_OK(ReadImage(testutil::GetTestDataPath(file_name).c_str(), &img));
  CSPTransform transf;
  ASSERT_WP2_OK(transf.Init(Csp::kCustom, img));
  ASSERT_EQ(transf.GetType(), Csp::kCustom);

  EncoderConfig config = EncoderConfig::kDefault;
  config.csp_type = Csp::kCustom;
  ASSERT_TRUE(testutil::EncodeDecodeCompare(file_name, config));
}

//------------------------------------------------------------------------------
// Golden-checking test that can be used as a CPU usage benchmark.

class SpeedTest : public ::testing::TestWithParam<WP2CPUInfo> {
  void SetUp() override {
    WP2DspReset();
    WP2GetCPUInfo = GetParam();
    WP2CSPConverterInit();
  }

 protected:
  static constexpr uint32_t kIter = 100u;
  static constexpr uint32_t kNumTest = 10000u / kIter;
  static constexpr uint32_t kMaxLen = 16384u;
  static constexpr int32_t kMaxAvg = (1u << 10) - 1;
  static constexpr int32_t kMaxYUV = (1u << 10) - 1;
  static constexpr int32_t kMaxMtx = 1u << 14;
  std::vector<int16_t> yuv_;
  int16_t avg_[3], mtx_[9];

  void Init(UniformIntDistribution* const random) {
    yuv_.resize(3 * kMaxLen);
    for (auto& v : yuv_) {
      // from time to time, inject some extreme values
      const int rnd = random->Get(0, 30);
      v = (rnd == 0) ?  kMaxYUV
        : (rnd == 1) ? -kMaxYUV
        : random->Get<int16_t>(-kMaxYUV, kMaxYUV);
    }
  }

  // Initializes the side data used for conversion.
  void FillMtx(UniformIntDistribution* const random) {
    for (auto& i : avg_) i = random->Get(-kMaxAvg, kMaxAvg);
    for (auto& i : mtx_) i = random->Get(-kMaxMtx, kMaxMtx);
  }

  // Tests the 'func' a bunch of times and returns its SillyCRC.
  uint32_t DoTest(WP2AyuvToArgbFunc func,
                  UniformIntDistribution* const random) {
    Init(random);
    const uint32_t bytes_per_pixel =
        (func == WP2AyuvToArgb38 ? WP2FormatBpp(WP2_Argb_38)
                                 : WP2FormatBpp(WP2_Argb_32));
    std::vector<uint8_t> out(bytes_per_pixel * kMaxLen, 0);
    uint32_t crc = 0;
    for (uint32_t test = 0; test < kNumTest; ++test) {
      const uint32_t len = (test == 0) ? kMaxLen : random->Get(1u, kMaxLen);
      const int16_t* const y = &yuv_[random->Get(0u, kMaxLen - len)];
      const int16_t* const u = &yuv_[random->Get(0u, kMaxLen - len)];
      const int16_t* const v = &yuv_[random->Get(0u, kMaxLen - len)];
      const int16_t* const a = &yuv_[random->Get(0u, kMaxLen - len)];
      FillMtx(random);
      for (uint32_t iter = 0; iter < kIter; ++iter) {
        func(y, u, v, a, avg_, mtx_, &out[0], len);
      }
      for (auto i : out) crc = testutil::SillyCRC(crc, i);
    }
    return crc;
  }

  // Same as above but with no alpha.
  uint32_t DoTest(WP2YuvToArgbFunc func, UniformIntDistribution* const random) {
    Init(random);
    const uint32_t bytes_per_pixel =
        (func == WP2YuvToArgb38 ? WP2FormatBpp(WP2_Argb_38)
                                : WP2FormatBpp(WP2_Argb_32));
    std::vector<uint8_t> out(bytes_per_pixel * kMaxLen, 0);
    uint32_t crc = 0;
    for (uint32_t test = 0; test < kNumTest; ++test) {
      const uint32_t len = (test == 0) ? kMaxLen : random->Get(1u, kMaxLen);
      const int16_t* const y = &yuv_[random->Get(0u, kMaxLen - len)];
      const int16_t* const u = &yuv_[random->Get(0u, kMaxLen - len)];
      const int16_t* const v = &yuv_[random->Get(0u, kMaxLen - len)];
      FillMtx(random);
      for (uint32_t iter = 0; iter < kIter; ++iter) {
        func(y, u, v, avg_, mtx_, &out[0], len);
      }
      for (auto i : out) crc = testutil::SillyCRC(crc, i);
    }
    return crc;
  }
};

// Same as above but for custom-format output instead of (only) RGB.
TEST_P(SpeedTest, YuvToCustom) {
  UniformIntDistribution random(431412);
  Init(&random);
  std::vector<int16_t> out(3 * kMaxLen, 0);
  uint32_t crc = 0;
  for (uint32_t test = 0; test < kNumTest; ++test) {
    const uint32_t len = (test == 0) ? kMaxLen : random.Get(1u, kMaxLen);
    const int16_t* const y = &yuv_[random.Get(0u, kMaxLen - len)];
    const int16_t* const u = &yuv_[random.Get(0u, kMaxLen - len)];
    const int16_t* const v = &yuv_[random.Get(0u, kMaxLen - len)];
    FillMtx(&random);
    for (uint32_t iter = 0; iter < kIter; ++iter) {
      WP2YuvToCustom(y, u, v, avg_, mtx_,
                     &out[0 * kMaxLen], &out[1 * kMaxLen], &out[2 * kMaxLen],
                     len);
    }
    for (auto i : out) crc = testutil::SillyCRC(crc, i);
  }
  EXPECT_EQ(crc, 1929850794u);
}

TEST_P(SpeedTest, AyuvToArgb32) {
  UniformIntDistribution random(/*seed=*/48623);
  EXPECT_EQ(DoTest(WP2AyuvToArgb32, &random), 3017869648u);
}

TEST_P(SpeedTest, AyuvToARGB32) {
  UniformIntDistribution random(/*seed=*/8724);
  EXPECT_EQ(DoTest(WP2AyuvToARGB32, &random), 3144549024u);
}

TEST_P(SpeedTest, AyuvToXRGB32) {
  UniformIntDistribution random(/*seed=*/1532);
  EXPECT_EQ(DoTest(WP2AyuvToXRGB32, &random), 952933099u);
}

TEST_P(SpeedTest, YuvToArgb32) {
  UniformIntDistribution random(/*seed=*/48623);
  EXPECT_EQ(DoTest(WP2YuvToArgb32, &random), 2953853455u);
}

TEST_P(SpeedTest, AyuvToArgb38) {
  UniformIntDistribution random(/*seed=*/48623);
  EXPECT_EQ(DoTest(WP2AyuvToArgb32, &random), 3017869648u);
}

TEST_P(SpeedTest, YuvToArgb38) {
  UniformIntDistribution random(/*seed=*/48623);
  EXPECT_EQ(DoTest(WP2YuvToArgb32, &random), 2953853455u);
}

INSTANTIATE_TEST_SUITE_P(SpeedTestInstantiation, SpeedTest,
                         ::testing::ValuesIn(testutil::kWP2CpuInfos));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
