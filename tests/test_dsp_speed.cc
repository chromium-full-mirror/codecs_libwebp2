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

// Test speed of some DSP functions

#include <memory>

#include "examples/example_utils.h"
#include "include/helpers.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless/encl_dsp.h"
#include "src/utils/plane.h"
#include "src/utils/random.h"

namespace WP2 {
namespace {

class SpeedTestCpuInfo : public testing::TestWithParam<WP2CPUInfo> {
  void SetUp() override {
    WP2DspReset();
    WP2GetCPUInfo = GetParam();
    WP2DecDspInit();
    WP2EncDspInit();
    WP2SSIMInit();
    WP2PSNRInit();
    WP2L::EncLDspInit();
  }
};

//------------------------------------------------------------------------------

TEST_P(SpeedTestCpuInfo, TestAddSub) {
  constexpr uint32_t kMaxLen = 32 * 32;
  int16_t src[kMaxLen], pred[kMaxLen], dst[kMaxLen], res[kMaxLen];
  constexpr uint32_t kNumTests = 10;
  constexpr uint32_t kNumLoops = 10000;
  PseudoRNG rng;
  for (uint32_t t = 0; t < kNumTests; ++t) {
    const uint32_t len = (rng.GetUnsigned(kMaxLen - 1) + 1) & ~3;
    const int32_t min = -(1 + rng.GetUnsigned(1023u));
    const int32_t max = (1 + rng.GetUnsigned(1023u));
    testutil::PrecalculatedRandom<kMaxLen * 3, int16_t> rnd16(-2048, 2048);
    for (uint32_t l = 0; l < kNumLoops; ++l) {
      rnd16.Fill(src, len);
      rnd16.Fill(pred, len);
      WP2::SubtractRow(src, pred, res, len);
      WP2::AddRow(pred, res, min, max, dst, len);
    }
    uint32_t nb_errors = 0;
    for (uint32_t i = 0; i < len; ++i) {
      if (src[i] <= min) {
        nb_errors += (dst[i] != min);
      } else if (src[i] >= max) {
        nb_errors += (dst[i] != max);
      } else {
        nb_errors += (src[i] != dst[i]);
      }
    }
    EXPECT_EQ(nb_errors, 0u) << " rate: " << (100 * nb_errors / len) << "%";
  }
}

TEST_P(SpeedTestCpuInfo, TestAddSubBlock) {
  constexpr uint32_t kMaxLen = 32 * 32;
  int16_t src[kMaxLen], pred[kMaxLen], dst[kMaxLen], res[kMaxLen];
  constexpr uint32_t kNumTests = 100;
  constexpr uint32_t kNumLoops = 200;
  PseudoRNG rng;
  for (bool add_eq : {false, true}) {
    for (uint32_t t = 0; t < kNumTests; ++t) {
      const uint32_t height = 1 + rng.GetUnsigned(32 - 1);
      const int32_t min = -(1 + rng.GetUnsigned(1023u));
      const int32_t max = (1 + rng.GetUnsigned(1023u));
      testutil::PrecalculatedRandom<kMaxLen * 3, int16_t> rnd16(-2048, 2048);
      for (uint32_t l = 0; l < kNumLoops; ++l) {
        rnd16.Fill(src, kMaxLen);
        rnd16.Fill(pred, kMaxLen);
        // reverse order, from large to small:
        for (uint32_t idx : {3, 2, 1, 0}) {
          const uint32_t len = (4 << idx);
          WP2::SubtractBlock[idx](src, 32, pred, 32, res, len, height);
          if (add_eq) {
            // 32b->16b conversion, with different stride ('len' -> '32')
            for (uint32_t j = 0; j < height; ++j) {
              for (uint32_t i = 0; i < len; ++i) {
                dst[i + j * 32] = (int16_t)res[i + len * j];
              }
            }
            WP2::AddBlockEq[idx](pred, 32, dst, 32, min, max, height);
          } else {
            WP2::AddBlock[idx](pred, 32, res, len, min, max, dst, 32, height);
          }
        }
      }
      uint32_t nb_errors = 0;
      for (uint32_t i = 0; i < height * 32; ++i) {
        if (src[i] <= min) {
          nb_errors += (dst[i] != min);
        } else if (src[i] >= max) {
          nb_errors += (dst[i] != max);
        } else {
          nb_errors += (src[i] != dst[i]);
        }
      }
      EXPECT_EQ(nb_errors, 0u)
          << " rate: " << (100 * nb_errors / kMaxLen) << "%";
    }
  }
}

//------------------------------------------------------------------------------

TEST_P(SpeedTestCpuInfo, TestBufferAdd) {
  constexpr uint32_t kMaxLen = 1 << 14;
  Vector_u32 A, B, C, D;
  ASSERT_TRUE(A.resize(kMaxLen) && B.resize(kMaxLen));
  ASSERT_TRUE(C.resize(kMaxLen) && D.resize(kMaxLen));
  constexpr uint32_t kNumTests = 10;
  constexpr uint32_t kNumLoops = 500;
  PseudoRNG rng;
  uint32_t CRC = 412325;
  for (auto& c : C) c = 0;
  for (auto& d : D) d = 0;
  for (uint32_t t = 0; t < kNumTests; ++t) {
    testutil::PrecalculatedRandom<kMaxLen * 2, uint32_t> rnd32(0, 1 << 12);
    const uint32_t proba = rng.GetUnsigned(255);
    uint32_t nzero = rng.GetUnsigned(532);  // shouldn't matter
    uint32_t* nzero_ptr = &nzero;
    if (rng.GetUnsigned(10) < 5) nzero_ptr = nullptr;
    for (uint32_t i = 0; i < rnd32.Size(); ++i) {
      if (rng.GetUnsigned(1000) < proba) rnd32[i] = 0;
    }
    for (uint32_t l = 0; l < kNumLoops; ++l) {
      const uint32_t len = kMaxLen / 2 + rng.GetUnsigned(kMaxLen / 2 - 1);
      rnd32.Fill(&A[0], len);
      rnd32.Fill(&B[0], len);
      WP2L::BufferAdd(&A[0], &B[0], len, &C[0], nzero_ptr);
      WP2L::BufferAdd(&A[0], &D[0], len, &D[0], nzero_ptr);
      CRC = testutil::SillyCRC(CRC, nzero);
    }
  }
  for (auto& c : C) CRC = testutil::SillyCRC(CRC, c);
  for (auto& d : D) CRC = testutil::SillyCRC(CRC, d);
  ASSERT_EQ(468228884u, CRC);
}

//------------------------------------------------------------------------------

// Generates random data in [min_range:max_range[ and distorts it by 'max_diff'.
// Checks that the distortion result matches the 'expected_average_score'.
template <typename T>
void DoTestSsim(BitDepth bit_depth, int max_diff,
                double expected_average_score) {
  constexpr uint32_t kMaxDim = 300;
  constexpr uint32_t kNumTests = 5;
  constexpr uint32_t kNumLoops = 7;
  PseudoRNG rng;
  double average_score = 0.;
  for (uint32_t t = 0; t < kNumTests; ++t) {
    const uint32_t width = 1 + rng.GetUnsigned(kMaxDim);
    const uint32_t height = 1 + rng.GetUnsigned(kMaxDim);
    testutil::PrecalculatedRandom<kMaxDim * 7, T> rnd(bit_depth.min(),
                                                      bit_depth.max() + 1);
    double score;
    Plane<T> plane1, plane2;
    ASSERT_WP2_OK(plane1.Resize(width, height));
    EXPECT_EQ(plane1.GetSSIM(plane2, bit_depth, &score),
              WP2_STATUS_BAD_DIMENSION);
    ASSERT_WP2_OK(plane2.Resize(width, height));
    for (uint32_t y = 0; y < height; ++y) {
      T* const row1 = plane1.Row(y);
      T* const row2 = plane2.Row(y);
      rnd.Fill(row1, width);
      rnd.Fill(row2, width);
      for (uint32_t x = 0; x < width; ++x) {
        row2[x] = Clamp<int>(row2[x], row1[x] - max_diff, row1[x] + max_diff);
      }
    }
    for (uint32_t l = 0; l < kNumLoops; ++l) {
      ASSERT_WP2_OK(plane1.GetSSIM(plane2, bit_depth, &score));
      if (max_diff == 0) {
        EXPECT_EQ(score, 1. * width * height);
      } else {
        EXPECT_LT(score, 1. * width * height);
      }
    }
    average_score += score / (width * height * kNumTests);
  }
  EXPECT_NEAR(average_score, expected_average_score, 1e-16);
}

TEST_P(SpeedTestCpuInfo, TestSsim8bits) {
  constexpr BitDepth kBitDepth = {8, /*is_signed=*/false};
  DoTestSsim<uint8_t>(kBitDepth, /*max_diff=*/0,
                      /*expected_average_score=*/1.);
  DoTestSsim<uint8_t>(kBitDepth, /*max_diff=*/1,
                      /*expected_average_score=*/0.99990578289445553);
  DoTestSsim<uint8_t>(kBitDepth, /*max_diff=*/50,
                      /*expected_average_score=*/0.77854488910920416);
  DoTestSsim<uint8_t>(kBitDepth, /*max_diff=*/256,
                      /*expected_average_score=*/0.078342300386812946);
}

TEST_P(SpeedTestCpuInfo, TestSsim10bits) {
  constexpr BitDepth kBitDepth = {10, /*is_signed=*/true};
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/0,
                      /*expected_average_score=*/1.);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/1,
                      /*expected_average_score=*/0.9999983977231518);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/50,
                      /*expected_average_score=*/0.99596831375426853);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/1024,
                      /*expected_average_score=*/0.20170015131280025);
}

TEST_P(SpeedTestCpuInfo, TestSsim12bits) {
  constexpr BitDepth kBitDepth = {12, /*is_signed=*/true};
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/0,
                      /*expected_average_score=*/1.);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/1,
                      /*expected_average_score=*/0.99999989986046256);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/50,
                      /*expected_average_score=*/0.99974890122298021);
  DoTestSsim<int16_t>(kBitDepth, /*max_diff=*/4096,
                      /*expected_average_score=*/0.20169647034266905);
}

//------------------------------------------------------------------------------

template <typename T>
void DoTestPSNR(int min_range, int max_range, uint64_t result) {
  constexpr uint32_t kMaxDim = 500;
  constexpr uint32_t kNumTests = 10;
  constexpr uint32_t kNumLoops = 50;
  PseudoRNG rng;
  uint64_t total_score = 0;
  for (uint32_t t = 0; t < kNumTests; ++t) {
    const uint32_t width = 1 + rng.GetUnsigned(kMaxDim);
    const uint32_t height = 1 + rng.GetUnsigned(kMaxDim);
    testutil::PrecalculatedRandom<kMaxDim * 7, T> rnd(min_range, max_range);
    uint64_t score;
    Plane<T> plane1, plane2, plane3;
    ASSERT_WP2_OK(plane1.Resize(width, height));
    EXPECT_EQ(plane1.GetSSE(plane2, &score), WP2_STATUS_BAD_DIMENSION);
    ASSERT_WP2_OK(plane2.Resize(width, height));
    ASSERT_WP2_OK(plane3.Resize(width, height));
    plane3.Fill(0);
    for (uint32_t y = 0; y < height; ++y) {
      T* const row1 = plane1.Row(y);
      T* const row2 = plane2.Row(y);
      rnd.Fill(row1, width);
      for (uint32_t x = 0; x < width; ++x) row2[x] = row1[x] ^ 1;
    }
    for (uint32_t l = 0; l < kNumLoops; ++l) {
      ASSERT_WP2_OK(plane1.GetSSE(plane1, &score));
      EXPECT_EQ(score, 0u);
      ASSERT_WP2_OK(plane1.GetSSE(plane2, &score));
      EXPECT_EQ(score, width * height);  // diff is '1' for each pixel
      ASSERT_WP2_OK(plane1.GetSSE(plane3, &score));
    }
    total_score += score;
  }
  EXPECT_EQ(total_score, result);
}

TEST_P(SpeedTestCpuInfo, TestPSNR_16b) {
  DoTestPSNR<int16_t>(-2048, 2048, 6630626819665llu);
}

TEST_P(SpeedTestCpuInfo, TestPSNR_8b) {
  DoTestPSNR<uint8_t>(0, 256, 15200787724llu);
}

//------------------------------------------------------------------------------

TEST_P(SpeedTestCpuInfo, TestCflPrediction) {
  WP2::PredictionInit();
  PseudoRNG rng;
  constexpr uint32_t kMaxSize = 51353;
  constexpr uint32_t kNumTests = 60;
  constexpr uint32_t kNumLoops = 500;
  for (uint32_t t = 0; t < kNumTests; ++t) {
    int16_t in[kMaxSize], out[kMaxSize] = {0};
    uint32_t len = (t < 16) ? 4 * t : 1 + rng.GetUnsigned(kMaxSize - 1);
    len = (len & ~3);
    const int32_t min_value = -rng.GetUnsigned(511);
    const int32_t max_value = +rng.GetUnsigned(511);
    const int32_t a = -rng.Get(-1020, 1240);
    const int32_t b = +rng.Get(-512, 511);
    for (auto& v : in) v = rng.Get(-511, 511);
    for (uint32_t l = 0; l < kNumLoops; ++l) {
      CflPredict(in, a, b, out, min_value, max_value, len);
    }
    for (uint32_t l = 0; l < len; ++l) {
      ASSERT_EQ(out[l], CflScale(in[l], a, b, min_value, max_value))
          << "[" << in[l] << " * a=" << a << " + b=" << b << " = " << out[l]
          << "  clipped to [" << min_value << ", " << max_value << "]";
    }
    for (uint32_t l = len; l < kMaxSize; ++l) ASSERT_EQ(out[l], 0);
  }
}

//------------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(SpeedTestInstantiation, SpeedTestCpuInfo,
                         testing::ValuesIn(testutil::kWP2CpuInfos));

}  // namespace
}  // namespace WP2
