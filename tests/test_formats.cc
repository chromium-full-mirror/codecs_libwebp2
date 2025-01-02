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

// Test premultiplied, unpremultiplied, reordered and/or opaque decoding output.
// There is no loss during format conversion because the source ('original') is
// premultiplied before encoding and the 'premul <-> unpremul <-> premul'
// operation is stable.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include "imageio/image_dec.h"
#include "include/helpers.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/utils/random.h"
#include "src/utils/utils.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Reference implementation, without the tricks used in argb_converter.cc
void Premultiply(uint8_t* argb, uint32_t width) {
  for (uint32_t i = 0; i < width; ++i, argb += 4) {
    const uint32_t alpha = argb[0];
    if (alpha < 255u) {
      argb[1] = (uint8_t)((argb[1] * alpha + 127u) / 255u);
      argb[2] = (uint8_t)((argb[2] * alpha + 127u) / 255u);
      argb[3] = (uint8_t)((argb[3] * alpha + 127u) / 255u);
    }
  }
}

// Copies alpha from 'src' to 'dst' (must both be WP2_Argb_32).
WP2Status CopyAlpha(const ArgbBuffer& src, ArgbBuffer* const dst,
                    bool premultiply) {
  WP2_CHECK_OK(src.format() == WP2_Argb_32 || src.format() == WP2_ARGB_32,
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(dst->format() == WP2_Argb_32 || src.format() == WP2_ARGB_32,
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(src.width() == dst->width(), WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(src.height() == dst->height(), WP2_STATUS_INVALID_PARAMETER);

  for (uint32_t y = 0; y < src.height(); ++y) {
    const uint8_t* const src_row = src.GetRow8(y);
    uint8_t* const dst_row = dst->GetRow8(y);
    for (uint32_t x = 0; x < src.width(); ++x) dst_row[x * 4] = src_row[x * 4];
    if (premultiply) Premultiply(dst_row, src.width());
  }

  return WP2_STATUS_OK;
}

// Go back to Argb32 for comparison and copy alpha if it was lost.
WP2Status ConvertOutputBackToOriginalFormat(
    const ArgbBuffer& original, const ArgbBuffer& output, bool decoded,
    ArgbBuffer* const output_in_original_format) {
  WP2ArgbConverterInit();

  WP2_CHECK_OK(original.format() == output_in_original_format->format(),
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_STATUS(output_in_original_format->ConvertFrom(output));
  const bool orig_premultiplied = WP2IsPremultiplied(original.format());
  if (!WP2FormatHasAlpha(output.format())) {
    WP2_CHECK_STATUS(
        CopyAlpha(original, output_in_original_format, orig_premultiplied));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Basic format conversion only tests

TEST(FormatTest, ConvertSimple) {
  WP2ArgbConverterInit();

  for (WP2SampleFormat orig_format : {WP2_Argb_32, WP2_ARGB_32}) {
    ArgbBuffer original(orig_format);
    ASSERT_WP2_OK(original.Resize(1, 1));
    uint8_t* const a = original.GetRow8(0) + 0;
    uint8_t* const r = original.GetRow8(0) + 1;
    uint8_t* const g = original.GetRow8(0) + 2;
    uint8_t* const b = original.GetRow8(0) + 3;
    *a = 0xAA;
    *r = 0x55;
    *g = 0xAA;
    *b = 0x00;

    for (WP2SampleFormat format = WP2_Argb_32; format < WP2_FORMAT_NUM;
         format = (WP2SampleFormat)(format + 1)) {
      ArgbBuffer converted(format);
      ASSERT_WP2_OK(converted.ConvertFrom(original));

      ArgbBuffer reconverted(original.format());
      ASSERT_WP2_OK(ConvertOutputBackToOriginalFormat(original, converted,
                                                      false, &reconverted));

      const float expected_distortion =
          !WP2IsPremultiplied(orig_format) && WP2IsPremultiplied(format) ? 25.f
          : WP2Formatbpc(orig_format) > WP2Formatbpc(format)             ? 90.f
                                                                         : 99.f;
      ASSERT_TRUE(
          testutil::Compare(original, reconverted, "1x1", expected_distortion))
          << "format: " << orig_format << " to: " << format;
    }
  }
}

TEST(FormatTest, ConvertImage) {
  WP2ArgbConverterInit();

  for (WP2SampleFormat orig_format : {WP2_Argb_32, WP2_ARGB_32}) {
    const std::string& file_name = "source1_64x48.png";
    ArgbBuffer original(orig_format);
    ASSERT_WP2_OK(
        ReadImage(testutil::GetTestDataPath(file_name).c_str(), &original));

    for (WP2SampleFormat format = WP2_Argb_32; format < WP2_FORMAT_NUM;
         format = (WP2SampleFormat)(format + 1)) {
      ArgbBuffer converted(format);
      ASSERT_WP2_OK(converted.ConvertFrom(original));

      ArgbBuffer reconverted(original.format());
      ASSERT_WP2_OK(ConvertOutputBackToOriginalFormat(original, converted,
                                                      false, &reconverted));

      const float expected_distortion =
          !WP2IsPremultiplied(orig_format) && WP2IsPremultiplied(format) ? 25.f
          : WP2Formatbpc(orig_format) > WP2Formatbpc(format)             ? 90.f
                                                                         : 99.f;
      ASSERT_TRUE(testutil::Compare(original, reconverted, file_name,
                                    expected_distortion))
          << "format: " << orig_format << " to: " << format;
    }
  }
}

//------------------------------------------------------------------------------
// Encode and decode tests

typedef std::tuple<std::vector<const char*>, WP2SampleFormat, float> Param;
class FormatTest : public testing::TestWithParam<Param> {};

//------------------------------------------------------------------------------

TEST_P(FormatTest, EncodeDecodeImage) {
  const char* src_file_name = std::get<0>(GetParam()).front();
  const WP2SampleFormat output_format = std::get<1>(GetParam());
  // TODO(yguyon): Handle WP2_Argb_38 output for the lossless pipeline
  const float quality =
      (output_format == WP2_Argb_38) ? 40.f : std::get<2>(GetParam());

  ArgbBuffer src;
  MemoryWriter data;
  ASSERT_WP2_OK(testutil::CompressImage(src_file_name, &data, &src, quality,
                                        /*effort=*/2));

  ArgbBuffer output(output_format);
  ASSERT_WP2_OK(Decode(data.mem_, data.size_, &output));

  ArgbBuffer output_original_format(src.format());
  ASSERT_WP2_OK(ConvertOutputBackToOriginalFormat(src, output, true,
                                                  &output_original_format));
  ASSERT_TRUE(testutil::Compare(src, output_original_format, src_file_name,
                                testutil::GetExpectedDistortion(quality)))
      << output_format;
}

//------------------------------------------------------------------------------

TEST_P(FormatTest, EncodeDecodeAnimation) {
  const std::vector<const char*>& file_names = std::get<0>(GetParam());
  const std::vector<uint32_t> durations_ms(file_names.size(), 100u);
  const WP2SampleFormat output_format = std::get<1>(GetParam());
  const float quality = std::get<2>(GetParam());
  const uint32_t effort = 0, thread_level = 0;

  std::vector<ArgbBuffer> frames;
  MemoryWriter encoded_data;
  ASSERT_WP2_OK(testutil::CompressAnimation(file_names, durations_ms,
                                            &encoded_data, &frames, quality,
                                            effort, thread_level));

  ArgbBuffer decoded_frame(output_format);
  ArrayDecoder decoder(encoded_data.mem_, encoded_data.size_,
                       DecoderConfig::kDefault, &decoded_frame);

  for (size_t i = 0; i < frames.size(); ++i) {
    ASSERT_TRUE(decoder.ReadFrame());

    ArgbBuffer decoded_frame_original_format(frames[i].format());
    ASSERT_WP2_OK(ConvertOutputBackToOriginalFormat(
        frames[i], decoded_frame, true, &decoded_frame_original_format));
    EXPECT_TRUE(testutil::Compare(frames[i], decoded_frame_original_format,
                                  file_names[i],
                                  testutil::GetExpectedDistortion(quality)));
  }
}

//------------------------------------------------------------------------------

constexpr float kMinExpectedSimilarityForPreview = 10.f;
constexpr float kMinExpectedSimilarityForPreviewColor = 5.f;

TEST_P(FormatTest, EncodeDecodePreview) {
  const char* file_name = std::get<0>(GetParam()).front();
  const WP2SampleFormat output_format = std::get<1>(GetParam());
  const float quality = std::get<2>(GetParam());

  ArgbBuffer original_image;
  ASSERT_WP2_OK(
      ReadImage(testutil::GetTestDataPath(file_name).c_str(), &original_image));

  EncoderConfig config;
  config.quality = quality;
  config.effort = 2;
  config.create_preview = true;
  MemoryWriter memory_writer;
  EXPECT_WP2_OK(Encode(original_image, &memory_writer, config));

  ArgbBuffer decompressed_preview(output_format);
  ASSERT_WP2_OK(ExtractPreview(memory_writer.mem_, memory_writer.size_,
                               &decompressed_preview));

  ArgbBuffer decompressed_preview_original_format(original_image.format());
  ASSERT_WP2_OK(ConvertOutputBackToOriginalFormat(
      original_image, decompressed_preview, true,
      &decompressed_preview_original_format));
  float disto[5];
  ASSERT_WP2_OK(decompressed_preview_original_format.GetDistortion(
      original_image, PSNR, disto));
  EXPECT_GE(disto[4], config.create_preview
                          ? kMinExpectedSimilarityForPreview
                          : kMinExpectedSimilarityForPreviewColor);
}

//------------------------------------------------------------------------------
// Input images (frames) are expected to be of equal size inside the same Param.

// Simple straightforward tests.
INSTANTIATE_TEST_SUITE_P(
    FormatTestInstantiation, FormatTest,
    testing::Values(
        Param({"source1_64x48.png"}, WP2_ARGB_32, /*quality=*/100.0f),
        Param({"source1_64x48.png"}, WP2_rgbA_32, /*quality=*/40.0f),
        Param({"source1_64x48.png"}, WP2_Argb_38, /*quality=*/100.0f),
        Param({"source1_64x48.png"}, WP2_Argb_38, /*quality=*/40.0f)));

// Test all lossy/lossless bypass paths for each output format.
INSTANTIATE_TEST_SUITE_P(
    BypassFormatTestInstantiation, FormatTest,
    testing::Combine(testing::Values(std::vector<const char*>{
                         "source1_1x1.png"}),
                     testing::Range(WP2_Argb_32, WP2_FORMAT_NUM),
                     /*quality=*/testing::Values(10.f, 100.f)));

// This one takes a while to run so it is disabled.
// Can still be run with flag --test_arg=--gunit_also_run_disabled_tests
INSTANTIATE_TEST_SUITE_P(
    DISABLED_HeavyFormatTestInstantiation, FormatTest,
    testing::Combine(
        testing::Values(
            std::vector<const char*>{"source1_64x48.png"},
            std::vector<const char*>{"alpha_ramp.png", "alpha_ramp.lossy.webp",
                                     "alpha_ramp.webp"},
            std::vector<const char*>{"source0.pgm", "source1.png",
                                     "source2.tiff", "source3.jpg"}),
        testing::Range(WP2_Argb_32, WP2_FORMAT_NUM),
        testing::Values(0.f, 70.f, 96.f)));

//------------------------------------------------------------------------------
// Exhaustive test (convert all combinations of alpha, red, green, blue values)

// Demultiply color 'v' by alpha 'a'.
uint32_t Unmult(uint32_t a, uint32_t v) {
  return (a == 0) ? 0 : DivRound(v * 255, a);
}

class ExhaustiveTest
    : public testing::TestWithParam<
          std::tuple<testutil::WP2CPUInfoStruct, bool, WP2SampleFormat>> {
  void SetUp() override {
    WP2DspReset();
    WP2GetCPUInfo = std::get<0>(GetParam()).cpu_info;
    WP2ArgbConverterInit();

    const WP2SampleFormat tmp_format = std::get<2>(GetParam());
    src_is_premul_ = std::get<1>(GetParam());
    tmp_is_premul_ = WP2IsPremultiplied(tmp_format);
    tmp_has_alpha_ = WP2FormatHasAlpha(tmp_format);
    tmp_is_38b_ = tmp_format == WP2_Argb_38;
    tmp_is_64b_ = WP2Formatbpc(tmp_format) == 16;
    convert_to_tmp_ = src_is_premul_ ? WP2ArgbConvertTo[tmp_format]
                                     : WP2ARGBConvertTo[tmp_format];
    convert_from_tmp_ = src_is_premul_ ? WP2ArgbConvertFrom[tmp_format]
                                       : WP2ARGBConvertFrom[tmp_format];
  }

 protected:
  // Returns the expected output of 'v' given alpha 'a' and input/temp formats.
  uint32_t GetExpectedValue(uint32_t a, uint32_t v) const {
    if (src_is_premul_) {
      if (tmp_has_alpha_) return v;
      if (tmp_is_38b_) return std::min(Unmult(a, v), 1023u);
      if (tmp_is_64b_) return std::min(Unmult(a, v), 65535u);
      return (tmp_has_alpha_ ? v : Unmult(a, v));
    }
    if (tmp_is_premul_) {
      // Temp is premultiplied but not src so there is some loss.
      if (tmp_is_38b_) {
        const uint32_t a_10b = (a * 1023 + 127) / 255;
        const uint32_t v_premul_38b = (a_10b * v + 127) / 255;
        return Unmult(a_10b, v_premul_38b);
      }
      if (tmp_is_64b_) {
        const uint32_t a_16b = (a * 65535 + 127) / 255;
        const uint32_t v_premul_64b = (a_16b * v + 127) / 255;
        return Unmult(a_16b, v_premul_64b);
      }
      return Unmult(a, (a * v + 127) / 255);
    }
    return v;
  }

  bool src_is_premul_, tmp_is_premul_, tmp_has_alpha_, tmp_is_38b_, tmp_is_64b_;
  WP2ArgbConverterF convert_to_tmp_, convert_from_tmp_;
};

TEST_P(ExhaustiveTest, TestArgbConvertToFrom) {
  for (uint32_t a = 0; a <= kAlphaMax; ++a) {
    // Red, green and blue channels are not correlated so a single loop is used
    // instead of the kAlphaMax * kRgbMax * kRgbMax * kRgbMax possibilities.
    // They have distinct values to test channel permutations (BGR etc.).
    const uint32_t max = src_is_premul_ ? a : 255u;
    for (uint32_t r = 0, g = max / 3, b = max * 2 / 3; r <= max;
         ++r, g = (g + 1) % (max + 1), b = (b + 1) % (max + 1)) {
      const uint32_t exp_a = tmp_has_alpha_ ? a : kAlphaMax;
      const uint32_t exp_r = GetExpectedValue(a, r);
      const uint32_t exp_g = GetExpectedValue(a, g);
      const uint32_t exp_b = GetExpectedValue(a, b);

      uint8_t src[4] = {(uint8_t)a, (uint8_t)r, (uint8_t)g, (uint8_t)b};
      uint8_t tmp[8], dst[4];
      uint16_t* const tmp16 = (uint16_t*)tmp;  // For 38-bit Argb
      convert_to_tmp_(src, /*width=*/1, tmp);
      convert_from_tmp_(tmp, /*width=*/1, dst);
      ASSERT_EQ(dst[0], exp_a) << a;
      ASSERT_TRUE(dst[1] == exp_r && dst[2] == exp_g && dst[3] == exp_b)
          << "in:    " << a << "," << r << "," << g << "," << b << "\n"
          << "tmp:   " << (uint32_t)tmp[0] << "," << (uint32_t)tmp[1] << ","
          << (uint32_t)tmp[2] << "," << (uint32_t)tmp[3] << "\n"
          << "tmp16: " << (uint32_t)tmp16[0] << "," << (uint32_t)tmp16[1] << ","
          << (uint32_t)tmp16[2] << "," << (uint32_t)tmp16[3] << "\n"
          << "out:   " << (uint32_t)dst[0] << "," << (uint32_t)dst[1] << ","
          << (uint32_t)dst[2] << "," << (uint32_t)dst[3] << "\n"
          << "exp:   " << exp_a << "," << exp_r << "," << exp_g << "," << exp_b;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ExhaustiveTestInstantiation, ExhaustiveTest,
    testing::Combine(testing::ValuesIn(testutil::kWP2CpuInfoStructs),
                     testing::Values(false, true),  // src_is_premul_
                     testing::Range(WP2_Argb_32, WP2_FORMAT_NUM)  // tmp_format
                     ));

//------------------------------------------------------------------------------
// Speed test

static struct FmtTestCase {
  WP2SampleFormat fmt;
  const char* name;
  size_t seed;
  bool has_alpha;
} kFmtTestCases[] = {
    {WP2_Argb_32, "WP2_Argb_32", 412412, true},
    {WP2_ARGB_32, "WP2_ARGB_32", 655473, true},
    {WP2_XRGB_32, "WP2_XRGB_32", 653134, false},
    {WP2_rgbA_32, "WP2_rgbA_32", 812463, true},
    {WP2_RGBA_32, "WP2_RGBA_32", 547271, true},
    {WP2_RGBX_32, "WP2_RGBX_32", 872353, false},
    {WP2_bgrA_32, "WP2_bgrA_32", 274353, true},
    {WP2_BGRA_32, "WP2_BGRA_32", 897534, true},
    {WP2_BGRX_32, "WP2_BGRX_32", 414125, false},
    {WP2_RGB_24, "WP2_RGB_24", 414125, false},
    {WP2_BGR_24, "WP2_BGR_24", 414125, false},
};

typedef std::tuple<FmtTestCase, testutil::WP2CPUInfoStruct> SpeedTestParam;

class SpeedTest : public testing::TestWithParam<SpeedTestParam> {
  void SetUp() override {
    WP2DspReset();
    WP2GetCPUInfo = std::get<1>(GetParam()).cpu_info;
    WP2ArgbConverterInit();
    WP2PSNRInit();
    test_ = std::get<0>(GetParam());
    bpp_ = WP2FormatBpp(test_.fmt);
    convert_to_ = WP2ArgbConvertTo[test_.fmt];
    convert_from_ = WP2ArgbConvertFrom[test_.fmt];
  }

 protected:
#if defined(NDEBUG)
  static constexpr uint32_t kNumTest = 1000u;
#else
  static constexpr uint32_t kNumTest = 100u;
#endif
  static constexpr uint32_t kMaxWidth = 1024u;
  static constexpr uint32_t kMaxHeight = 64u;
  static constexpr int32_t kMaxSize = kMaxWidth * kMaxHeight;
  std::vector<uint8_t> argb_in_, tmp_, argb_out_;
  FmtTestCase test_;
  uint32_t bpp_;
  bool no_alpha_;
  WP2ArgbConverterF convert_to_, convert_from_;

  void Init(UniformIntDistribution* const random) {
    argb_in_.resize(4 * kMaxSize, 0);
    for (uint32_t i = 0; i < kMaxSize; ++i) {
      const uint8_t A = test_.has_alpha ? random->Get(0u, 255u) : 0xffu;
      argb_in_[4 * i + 0] = A;
      argb_in_[4 * i + 1] = (A > 0) ? random->Get<uint32_t>(0u, A) : 0u;
      argb_in_[4 * i + 2] = (A > 0) ? random->Get<uint32_t>(0u, A) : 0u;
      argb_in_[4 * i + 3] = (A > 0) ? random->Get<uint32_t>(0u, A) : 0u;
    }
    argb_out_ = argb_in_;
    tmp_.resize(bpp_ * kMaxWidth);
  }

  bool DoTest() {
    UniformIntDistribution random(test_.seed);
    Init(&random);
    for (uint32_t test = 0; test < kNumTest; ++test) {
      const uint32_t width = 1 + random.Get(0u, kMaxWidth - 1u);
      const size_t stride = 4 * width;
      for (uint32_t y = 0; y < kMaxHeight; ++y) {
        convert_to_(&argb_in_[y * stride], width, &tmp_[0]);
        convert_from_(&tmp_[0], width, &argb_out_[y * stride]);
      }
    }
    if (WP2SumSquaredError8u(&argb_in_[0], &argb_out_[0], kMaxSize * 4)) {
      for (int i = 0; i < 12; ++i) {
        printf("[%d %d %d]", argb_in_[i], tmp_[i], argb_out_[i]);
      }
      printf("\n");
      return false;
    }
    return true;
  }
};

TEST_P(SpeedTest, TestBackForth) { EXPECT_TRUE(DoTest()); }

INSTANTIATE_TEST_SUITE_P(
    SpeedTestInstantiation, SpeedTest,
    testing::Combine(testing::ValuesIn(kFmtTestCases),
                     testing::ValuesIn(testutil::kWP2CpuInfoStructs)));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
