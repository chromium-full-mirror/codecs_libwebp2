// Copyright (c) the JPEG XL Project
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <tuple>

#include "imageio/image_dec.h"
#include "src/dec/lossless/losslessi_dec.h"
#include "src/enc/lossless/losslessi_enc.h"
#include "src/utils/utils.h"
#include "src/utils/vector.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"
#include "tests/include/helpers.h"

namespace WP2 {
namespace {

// Limit makes it choose fewer values so per-channel palette will be used
WP2::ArgbBuffer GenerateImage(size_t xsize, size_t ysize, bool has_alpha,
                              bool limit[4]) {
  WP2::ArgbBuffer image(WP2_ARGB_32);
  EXPECT_EQ(image.Resize(xsize, ysize), WP2_STATUS_OK);

  for (size_t y = 0; y < ysize; ++y) {
    auto* row = image.GetRow8(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[4 * x + 0] = has_alpha ? x + y : 255;
      if (limit[0]) row[4 * x + 0] &= ~3u;
      for (int c = 1; c < 4; c++) {
        auto* ptr = &row[4 * x + c];
        *ptr = x + y * c;
        if (limit[c]) *ptr &= ~3u;
      }
    }
  }

  return image;
}

// Limit makes it choose fewer values so per-channel palette will be used
void TestRoundTrip(const WP2::ArgbBuffer& image,
                   WP2L::EncodingAlgorithm algorithm) {
  WP2::Vector_u8 bytes;

  MemoryWriter memory_writer;
  EncoderConfig encoder_config;
  encoder_config.quality = 100;
  encoder_config.keep_unmultiplied = true;
  encoder_config.lossless_algorithm = algorithm;
  WP2_ASSERT_STATUS(Encode(image, &memory_writer, encoder_config));

  DecoderConfig decoder_config;
  ArgbBuffer decompressed(WP2_ARGB_32);
  WP2_ASSERT_STATUS(Decode(memory_writer.mem_, memory_writer.size_,
                           &decompressed, decoder_config));

  EXPECT_TRUE(testutil::Compare(image, decompressed, /*file_name=*/"pattern"));
}

class LosslessTestGen
    : public testing::TestWithParam<
          std::tuple<int, bool, std::array<int, 2>, WP2L::EncodingAlgorithm>> {
};

TEST_P(LosslessTestGen, RoundTripData) {
  const size_t val = std::get<0>(GetParam());
  const bool has_alpha = std::get<1>(GetParam());
  const size_t xsize = std::get<2>(GetParam())[0];
  const size_t ysize = std::get<2>(GetParam())[1];
  const WP2L::EncodingAlgorithm algorithm = std::get<3>(GetParam());
  bool limit[4] = {false};
  for (int i = 0; i < 4; ++i) limit[i] = static_cast<bool>(val & (1 << i));
  const auto image = GenerateImage(xsize, ysize, has_alpha, limit);
  TestRoundTrip(image, algorithm);
}

INSTANTIATE_TEST_SUITE_P(
    LosslessTestInstantiation, LosslessTestGen,
    testing::Combine(testing::Range(0, 1), testing::Values(false, true),
                     testing::Values(std::array<int, 2>{1, 1},
                                     std::array<int, 2>{3, 5},
                                     std::array<int, 2>{16, 16},
                                     std::array<int, 2>{256, 256},
                                     std::array<int, 2>{5, 800}),
                     testing::Values(WP2L::EncodingAlgorithm::kSCP,
                                     WP2L::EncodingAlgorithm::kCALIC)));

class LosslessTestFile : public testing::TestWithParam<
                             std::tuple<std::string, WP2L::EncodingAlgorithm>> {
};

TEST_P(LosslessTestFile, RoundTripFile) {
  const std::string file = std::get<0>(GetParam());
  const WP2L::EncodingAlgorithm algorithm = std::get<1>(GetParam());

  WP2::ArgbBuffer image(WP2_ARGB_32);
  ASSERT_EQ(WP2::ReadImage(testutil::GetTestDataPath(file).c_str(), &image),
            WP2_STATUS_OK)
      << testutil::GetTestDataPath(file);

  TestRoundTrip(image, algorithm);
}

INSTANTIATE_TEST_SUITE_P(
    LosslessTestInstantiation, LosslessTestFile,
    testing::Combine(testing::Values("background.png", "logo.png",
                                     "source1_64x48.png"),
                     testing::Values(WP2L::EncodingAlgorithm::kSCP,
                                     WP2L::EncodingAlgorithm::kCALIC)));

////////////////////////////////////////////////////////////////////////////////

TEST(Distance, Roundtrip) {
  constexpr int16_t kMin = -10, kMax = -10;
  for (bool use_opposite_error : {false, true}) {
    for (int16_t v = kMin; v <= kMax; ++v) {
      for (int16_t p = kMin; p <= kMax; ++p) {
        const int16_t d = WP2L::CalcDistanceFromPredictionAndTPV(
            p, use_opposite_error, v, kMin, kMax);
        const int16_t v_predicted = WP2L::CalcTPVFromPredictionAndDistance(
            p, use_opposite_error, d, kMin, kMax);
        EXPECT_EQ(v, v_predicted);
      }
    }
  }
}

}  // namespace
}  // namespace WP2
