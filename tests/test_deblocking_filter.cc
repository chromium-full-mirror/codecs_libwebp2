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
// -----------------------------------------------------------------------------
//
// Test the deblocking filter.

#include <limits>
#include <tuple>

#include "include/helpers.h"
#include "include/helpers_filter.h"
#include "src/common/global_params.h"
#include "src/dec/filters/deblocking_filter.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

// Constants used for the tests below.
constexpr uint32_t kMaxNumBlocksX = 8;
constexpr uint32_t kMaxNumBlocksY = 8;
constexpr uint32_t kMaxPixelsWidth = kMaxNumBlocksX * kMinBlockSizePix;
constexpr uint32_t kMaxPixelsHeight = kMaxNumBlocksY * kMinBlockSizePix;
// Minus pixels to exercise cropped blocks (bottom/right).
constexpr uint32_t kPixelsWidth = kMaxPixelsWidth - 1u;
constexpr uint32_t kPixelsHeight = kMaxPixelsHeight - (kMinBlockSizePix - 1u);

// Creates areas with close values, otherwise the filter will not be applied.
// If with_alpha is true, adds a fully opaque (255) alpha plane.
YUVPlane CreateBlockedPixels(BitDepth bit_depth, bool with_alpha = false) {
  YUVPlane pixels;
  WP2_ASSERT_STATUS(
      pixels.Resize(kPixelsWidth, kPixelsHeight, /*pad=*/1, with_alpha));
  const uint32_t half_x = kMaxPixelsWidth / 2;
  const uint32_t half_y = kMaxPixelsHeight / 2;
  // Testing horizontal edge.
  pixels.Y.Fill({0, 0, half_x, kMaxPixelsHeight}, 42);
  pixels.Y.Fill({half_x, 0, half_x, kMaxPixelsHeight}, 38);
  // Testing vertical edge.
  pixels.U.Fill({0, 0, kMaxPixelsWidth, half_y}, bit_depth.min());
  pixels.U.Fill({0, half_y, kMaxPixelsWidth, half_y}, bit_depth.min() + 4);
  // Testing both.
  pixels.V.Fill({0, 0, half_x, half_y}, bit_depth.max() - 7);
  pixels.V.Fill({half_x, 0, half_x, half_y}, bit_depth.max());
  pixels.V.Fill({0, half_y, half_x, half_y}, bit_depth.max() - 12);
  pixels.V.Fill({half_x, half_y, half_x, half_y}, bit_depth.max() - 5);

  if (with_alpha) pixels.A.Fill(kAlphaMax);

  return pixels;
}

FilterBlockMap CreateAlignedBlocks(const Vector<Segment>& segments,
                                   double res_den, double min_bpp,
                                   BitDepth bit_depth, YUVPlane* const pixels) {
  assert((pixels->Y.w_ + kMinBlockSizePix - 1u) / kMinBlockSizePix == 8u);
  assert((pixels->Y.h_ + kMinBlockSizePix - 1u) / kMinBlockSizePix == 8u);
  FilterBlockMap blocks;
  blocks.Init(/*tile_rect=*/{0, 0, pixels->Y.w_, pixels->Y.h_},
              bit_depth, bit_depth.min(), bit_depth.max(), pixels);
  EXPECT_WP2_OK(blocks.Allocate());
  const uint32_t s = segments.size();
  uint32_t id = 0;  // Segment id
  testutil::RegisterBlock(0, 0, BLK_16x8, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(4, 0, BLK_16x8, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(0, 2, BLK_8x16, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(2, 2, BLK_16x16, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(6, 2, BLK_8x4, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(6, 3, BLK_4x4, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/false, &blocks);
  testutil::RegisterBlock(7, 3, BLK_4x4, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(6, 4, BLK_8x16, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(0, 6, BLK_8x8, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/true, &blocks);
  testutil::RegisterBlock(2, 6, BLK_16x8, id++ % s, res_den, min_bpp,
                          /*with_lossy_alpha=*/false, &blocks);
  return blocks;
}

// Applies the deblocking filter on 'pixels' with 'filter_strength'.
void Deblock(BitDepth bit_depth, uint32_t quality_hint,
             YUVPlane* const pixels) {
  const Vector<Segment> segments =
      testutil::CreateSegments(quality_hint, bit_depth);
  const double residual_density = 1. * quality_hint / kMaxLossyQualityHint;
  const double min_bpp = 4. * quality_hint / kMaxLossyQualityHint;
  // The created block edges should match at least some of the boundaries of
  // the areas defined in CreateBlockedPixels().
  const FilterBlockMap blocks = CreateAlignedBlocks(segments, residual_density,
                                                    min_bpp, bit_depth, pixels);

  DecoderConfig config = DecoderConfig::kDefault;
  config.enable_deblocking_filter = true;
  GlobalParams gparams;
  gparams.quality_hint_ = quality_hint;
  DeblockingFilter deblocking_filter(config, gparams, blocks);
  ASSERT_WP2_OK(deblocking_filter.Allocate());

  uint32_t num_deblocked_rows = 0;
  for (uint32_t num_decoded_yuv_rows = 0; num_decoded_yuv_rows <= pixels->Y.h_;
       ++num_decoded_yuv_rows) {
    const uint32_t n = deblocking_filter.Deblock(num_decoded_yuv_rows);
    ASSERT_GE(n, num_deblocked_rows);
    num_deblocked_rows = n;
  }
  ASSERT_EQ(num_deblocked_rows, pixels->Y.h_);
}

//------------------------------------------------------------------------------

// Param: num_precision_bits
class DeblockingFilterTest : public testing::TestWithParam<BitDepth> {};

// Tests that no pixel is modified if each plane has one unique color.
TEST_P(DeblockingFilterTest, FilteringPlain) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels =
      testutil::CreatePlainPixels(kPixelsWidth, kPixelsHeight, bit_depth);
  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth, /*quality_hint=*/0, &pixels);

  EXPECT_TRUE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_TRUE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_TRUE(testutil::AreEqual(pixels.V, reference.V));
}

// Tests that some pixels are deblocked for horizontal and vertical edges.
TEST_P(DeblockingFilterTest, MaxFilteringBlocked) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels = CreateBlockedPixels(bit_depth, /*with_alpha=*/false);
  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth, /*quality_hint=*/0, &pixels);

  EXPECT_FALSE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_FALSE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_FALSE(testutil::AreEqual(pixels.V, reference.V));
}

// Tests that some pixels are deblocked for horizontal and vertical edges
// when the alpha plane is constant.
TEST_P(DeblockingFilterTest, MaxFilteringBlockedWithPlainAlpha) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels = CreateBlockedPixels(bit_depth, /*with_alpha=*/true);
  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth, /*quality_hint=*/0, &pixels);

  EXPECT_FALSE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_FALSE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_FALSE(testutil::AreEqual(pixels.V, reference.V));
  EXPECT_TRUE(testutil::AreEqual(pixels.A, reference.A));
}

// Tests that some pixels are deblocked for horizontal and vertical edges
// when the alpha plane has low variance. Alpha plane should also be deblocked.
TEST_P(DeblockingFilterTest, MaxFilteringBlockedWithLowVarianceAlpha) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels = CreateBlockedPixels(bit_depth, /*with_alpha=*/true);
  // The rest is 255.
  pixels.A.Fill({0, 0, kMaxPixelsWidth / 2, kMaxPixelsWidth / 2}, 250);
  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth, /*quality_hint=*/0, &pixels);

  EXPECT_FALSE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_FALSE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_FALSE(testutil::AreEqual(pixels.V, reference.V));
  EXPECT_FALSE(testutil::AreEqual(pixels.A, reference.A));
}

// Tests that pixels are not deblocked when alpha plane has high variance.
TEST_P(DeblockingFilterTest, MaxFilteringBlockedWithHighVarianceAlpha) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels = CreateBlockedPixels(bit_depth, /*with_alpha=*/true);
  const uint32_t half_x = kMaxPixelsWidth / 2;
  const uint32_t half_y = kMaxPixelsHeight / 2;
  // Testing horizontal edge.
  pixels.A.Fill({0, 0, half_x, half_y}, 50);
  pixels.A.Fill({half_x, half_y, half_x, half_y}, 50);

  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth, /*quality_hint=*/0, &pixels);

  EXPECT_TRUE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_TRUE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_TRUE(testutil::AreEqual(pixels.V, reference.V));
  EXPECT_TRUE(testutil::AreEqual(pixels.A, reference.A));
}

// Tests that no pixel is modified if it's the highest quality.
TEST_P(DeblockingFilterTest, NoFiltering) {
  const BitDepth bit_depth = GetParam();

  YUVPlane pixels = CreateBlockedPixels(bit_depth);
  YUVPlane reference;
  ASSERT_WP2_OK(reference.Copy(pixels, /*resize_if_needed=*/true));

  Deblock(bit_depth,
          /*quality_hint=*/kMaxLossyQualityHint, &pixels);

  EXPECT_TRUE(testutil::AreEqual(pixels.Y, reference.Y));
  EXPECT_TRUE(testutil::AreEqual(pixels.U, reference.U));
  EXPECT_TRUE(testutil::AreEqual(pixels.V, reference.V));
}

INSTANTIATE_TEST_SUITE_P(DeblockingFilterTestInstantiation,
                         DeblockingFilterTest,
                         testing::Values(BitDepth{8, /*is_signed=*/true},
                                         BitDepth{10, /*is_signed=*/true},
                                         BitDepth{12, /*is_signed=*/true}));

//------------------------------------------------------------------------------

// Tests that PSNR is good even if there is only deblocking.
TEST(DeblockingFilterTest, Simple) {
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 40;  // Expect compression artifacts.
  encoder_config.effort = 2;    // Speed up the test.

  DecoderConfig decoder_config = DecoderConfig::kDefault;
  decoder_config.enable_deblocking_filter = true;
  decoder_config.enable_directional_filter = false;
  decoder_config.enable_restoration_filter = false;

  ASSERT_TRUE(testutil::EncodeDecodeCompare("source4.webp", encoder_config,
                                            decoder_config));
}

// Tests that PSNR is better with the deblocking filter than without.
TEST(DeblockingFilterTest, BetterPSNR) {
  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.quality = 50;  // Expect compression artifacts.
  encoder_config.effort = 2;    // Speed up the test.

  DecoderConfig decoder_config_no_filter = DecoderConfig::kDefault;
  decoder_config_no_filter.enable_deblocking_filter = false;

  DecoderConfig decoder_config_filter = decoder_config_no_filter;
  decoder_config_filter.enable_deblocking_filter = true;

  float disto_diff[5];
  ASSERT_WP2_OK(testutil::GetDistortionDiff(
      "source1.png", {25, 1, 150, 200}, encoder_config,
      decoder_config_no_filter, decoder_config_filter, disto_diff));
  EXPECT_GT(disto_diff[4], 0.f);
}

//------------------------------------------------------------------------------

TEST(HelpersFilterTest, MinMaxValueSigned) {
  constexpr BitDepth kBd8 = {8, /*is_signed=*/false};
  EXPECT_EQ(kBd8.min(), std::numeric_limits<uint8_t>::min());
  EXPECT_EQ(kBd8.max(), std::numeric_limits<uint8_t>::max());

  constexpr BitDepth kBd10 = {10, /*is_signed=*/true};
  EXPECT_EQ(kBd10.min(), -512);
  EXPECT_EQ(kBd10.max(), 511);

  constexpr BitDepth kBd16 = {16, /*is_signed=*/true};
  EXPECT_EQ(kBd16.min(), std::numeric_limits<int16_t>::min());
  EXPECT_EQ(kBd16.max(), std::numeric_limits<int16_t>::max());
}

//------------------------------------------------------------------------------

class FilterTestCpuInfo : public testing::TestWithParam<WP2CPUInfo> {
  void SetUp() override {
    WP2DspReset();
    WP2GetCPUInfo = GetParam();
    DblkFilterInit();
  }
};

TEST_P(FilterTestCpuInfo, TestMeasureLength) {
  UniformIntDistribution gen(52452);
  const uint32_t kNumTests = 10000;
  int16_t A[32 * kDblCacheStep], *B = A + 8;
  uint32_t crc = 323452;
  for (uint32_t test = 0; test < kNumTests; ++test) {
    for (auto& a : A) a = gen.Get(-1024, 1024);
    for (uint32_t half = 1; half <= kDblkMaxHalf; ++half) {
      for (uint32_t s = 0; s <= kDblkMaxSharpness; ++s) {
        uint8_t out[32];
        MeasureFlatLengths(s, half, B, kDblCacheStep, out, 32);
        for (uint32_t o : out) crc = testutil::SillyCRC(crc, o);
      }
    }
  }
  EXPECT_EQ(crc, 1059920269u);
}

TEST_P(FilterTestCpuInfo, TestWouldDeblock) {
  UniformIntDistribution gen(86214);
  const uint32_t kNumTests = 1000;
  int16_t A[kDblCacheStep], * const B = A + 8;
  uint32_t crc = 1;
  for (uint32_t test = 0; test < kNumTests; ++test) {
    for (auto& a : A) a = gen.Get(-1024, 1024);
    for (uint32_t half = 1; half <= kDblkMaxHalf; ++half) {
      for (uint32_t t = 0; t <= 220; ++t) {
        crc = testutil::SillyCRC(crc, WouldDeblockLine(t, half, B));
      }
    }
  }
  EXPECT_EQ(crc, 2237354673u);
}

TEST_P(FilterTestCpuInfo, TestDeblockLines) {
  UniformIntDistribution gen(86214);
  const uint32_t kNumTests = 120000;
  int16_t A[kDblCacheStep], * const B = A + 8, C[kDblCacheStep];
  uint32_t crc = 1;
  for (int h = 1; h <= 8; ++h) {
    for (uint32_t bits : {8, 10}) {
      const int32_t min = (bits == 8) ?   0 : -1024;
      const int32_t max = (bits == 8) ? 255 : 1024;
      for (auto& c : C) c = gen.Get(min, max);
      for (uint32_t test = 0; test < kNumTests; ++test) {
        const uint32_t fstrength = 1 + (test % kDblkMaxStrength);
        memcpy(A, C, sizeof(A));
        for (uint32_t s = 0; s <= kDblkMaxSharpness; ++s) {
          const uint32_t thresh = DeblockThresholdFromSharpness(s, bits);
          DeblockLine(fstrength, thresh, h, min, max, B);
        }
      }
      for (int x = -h; x < h; ++x) {
        crc = testutil::SillyCRC(crc, B[x]);
      }
    }
  }
  EXPECT_EQ(crc, 813221631u);
}

TEST_P(FilterTestCpuInfo, TestCopyTransposed) {
  const uint32_t kNumTests = 10000;
  for (int32_t W = 1; W <= 32; ++W) {
    for (int32_t half = 1; half <= 8; ++half) {
      const int32_t kStep = 37;
      int16_t src0[kStep * kDblCacheStep], *src = src0 + 8 * kStep;
      int16_t cache[32 * kDblCacheStep], *dst = cache + 8;
      for (int32_t i = 0, x = 0; x < W; ++x) {
        for (int32_t y = -half; y < half; ++y, ++i) {
          src[y * kStep + x] = i;
        }
      }
      for (uint32_t n = 0; n < kNumTests; ++n) {
        FilterCopyIn(src, kStep, dst, W, half);
        FilterCopyOut(dst, src, kStep, W, half);
      }
      for (int32_t i = 0, x = 0; x < W; ++x) {
        for (int32_t y = -half; y < half; ++y, ++i) {
          EXPECT_EQ(src[y * kStep + x], i);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(FilterTestCpuInfoInstantiation, FilterTestCpuInfo,
                         testing::ValuesIn(testutil::kWP2CpuInfos));

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
