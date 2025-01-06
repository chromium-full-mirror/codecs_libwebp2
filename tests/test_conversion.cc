// Copyright 2024 Google LLC
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "include/helpers.h"
#include "src/dsp/dsp.h"
#include "src/utils/random.h"
#include "src/wp2/base.h"
#include "gtest/gtest.h"

namespace WP2 {
namespace {

TEST(RgbConversionTest, FromTo) {
  WP2ArgbConverterInit();

  UniformIntDistribution rand(0);

  for (int i_from = 0; i_from < static_cast<int>(WP2_FORMAT_NUM); ++i_from) {
    const WP2SampleFormat from = static_cast<WP2SampleFormat>(i_from);
    ArgbBuffer src(from);
    ASSERT_WP2_OK(src.Resize(10, 10));
    ArgbBuffer roundtrip(from);
    ASSERT_WP2_OK(roundtrip.Resize(src.width(), src.height()));

    uint32_t src_alpha_channel;
    WP2FormatHasAlpha(from, &src_alpha_channel);

    for (int i_to = 0; i_to < static_cast<int>(WP2_FORMAT_NUM); ++i_to) {
      const WP2SampleFormat to = static_cast<WP2SampleFormat>(i_to);

      // Fill src with random values.
      for (uint32_t y = 0; y < src.height(); ++y) {
        for (uint32_t x = 0; x < src.width(); ++x) {
          std::array<uint16_t, 4> values;

          if (WP2FormatHasAlpha(from)) {
            if (!WP2FormatHasAlpha(to)) {
              for (uint32_t i = 0; i < 4u; ++i) {
                if (i == src_alpha_channel) {
                  // Make sure input is opaque because it is discarded.
                  values[i] = (1 << WP2Formatbpalpha(from)) - 1;
                } else {
                  values[i] = rand.Get(0, (1 << WP2Formatbpc(from)) - 1);
                }
              }
            } else if (WP2IsPremultiplied(from)) {
              // Make sure channels are smaller than alpha.
              values[src_alpha_channel] =
                  rand.Get(0, (1 << WP2Formatbpalpha(from)) - 1);
              for (uint32_t i = 0; i < 4u; ++i) {
                if (i != src_alpha_channel) {
                  values[i] = rand.Get(
                      0, values[src_alpha_channel]
                             << (WP2Formatbpc(from) - WP2Formatbpalpha(from)));
                }
              }
            } else {
              // Get random values in range.
              for (uint32_t i = 0; i < 4u; ++i) {
                if (i == src_alpha_channel) {
                  values[i] = rand.Get(0, (1 << WP2Formatbpalpha(from)) - 1);
                } else {
                  values[i] = rand.Get(0, (1 << WP2Formatbpc(from)) - 1);
                }
              }
            }
          } else if (src_alpha_channel < WP2FormatNumChannels(from)) {
            for (uint32_t i = 0; i < 4u; ++i) {
              if (i == src_alpha_channel) {
                // There is no alpha but we max it out.
                values[i] = (1 << WP2Formatbpalpha(from)) - 1;
              } else {
                values[i] = rand.Get(0, (1 << WP2Formatbpc(from)) - 1);
              }
            }
          } else {
            // Get random values in range. Alpha does not matter.
            for (uint32_t i = 0; i < 4u; ++i) {
              values[i] = rand.Get(0, (1 << WP2Formatbpc(from)) - 1);
            }
          }

          // Copy the values to src.
          if (WP2FormatBpp(from) <= 4) {
            std::copy(values.data(), values.data() + WP2FormatNumChannels(from),
                      src.GetPosition(x, y));
          } else {
            std::copy(values.data(), values.data() + WP2FormatNumChannels(from),
                      reinterpret_cast<uint16_t*>(src.GetPosition(x, y)));
          }
        }
      }

      // Perform the roundtrip.
      ArgbBuffer dst(to);
      ASSERT_WP2_OK(dst.ConvertFrom(src));
      ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));

      if (WP2Formatbpc(from) <= WP2Formatbpc(to) &&
          WP2Formatbpalpha(from) <= WP2Formatbpalpha(to) &&
          WP2IsPremultiplied(from) == WP2IsPremultiplied(to)) {
        // No information is lost so the matching should be exact.
        ASSERT_TRUE(testutil::Compare(src, roundtrip, "all_combinations"))
            << "from " << from << " to " << to;
        continue;
      }
      // Compare but with a tolerance.
      for (uint32_t y = 0; y < src.height(); ++y) {
        for (uint32_t x = 0; x < src.width(); ++x) {
          std::array<uint16_t, 4> src_pixel, roundtrip_pixel;

          // Create some log string.
          std::string log = "src:";
          for (uint32_t c = 0; c < WP2FormatNumChannels(from); ++c) {
            if (c == src_alpha_channel && !WP2FormatHasAlpha(from)) continue;
            src_pixel[c] =
                WP2FormatBpp(from) <= 4
                    ? src.GetPosition(x, y)[c]
                    : reinterpret_cast<uint16_t*>(src.GetPosition(x, y))[c];
            log += " " + std::to_string(src_pixel[c]);
          }
          log += ", dst:";
          for (uint32_t c = 0; c < WP2FormatNumChannels(to); ++c) {
            uint32_t dst_alpha_channel;
            if (!WP2FormatHasAlpha(to, &dst_alpha_channel) &&
                c == dst_alpha_channel) {
              continue;
            }
            log += " " + std::to_string(WP2FormatBpp(to) <= 4
                                            ? dst.GetPosition(x, y)[c]
                                            : reinterpret_cast<uint16_t*>(
                                                  dst.GetPosition(x, y))[c]);
          }
          log += ", roundtrip:";
          for (uint32_t c = 0; c < WP2FormatNumChannels(from); ++c) {
            if (c == src_alpha_channel && !WP2FormatHasAlpha(from)) continue;
            roundtrip_pixel[c] = WP2FormatBpp(from) <= 4
                                     ? roundtrip.GetPosition(x, y)[c]
                                     : reinterpret_cast<uint16_t*>(
                                           roundtrip.GetPosition(x, y))[c];
            log += " " + std::to_string(roundtrip_pixel[c]);
          }
          for (uint32_t c = 0; c < WP2FormatNumChannels(from); ++c) {
            if (c == src_alpha_channel && !WP2FormatHasAlpha(from)) continue;
            // Compute the comparison tolerance.
            int tolerance;
            if (c == src_alpha_channel) {
              if (WP2Formatbpalpha(from) <= WP2Formatbpalpha(to)) {
                // We expand the alpha so the roundtrip is perfect.
                tolerance = 0;
              } else {
                // We lose some information.
                tolerance = 1u
                            << (WP2Formatbpalpha(from) - WP2Formatbpalpha(to));
              }
            } else {
              tolerance = WP2Formatbpc(from) > WP2Formatbpc(to)
                              ? 1u << (WP2Formatbpc(from) - WP2Formatbpc(to))
                              : 1;
              if (WP2FormatHasAlpha(from)) {
                const uint32_t src_max = (1u << WP2Formatbpc(from)) - 1u;
                const uint32_t src_alpha = src_pixel[src_alpha_channel];
                tolerance = tolerance * std::max(1u, src_max / (src_alpha + 1));
                if (WP2FormatHasAlpha(to)) {
                  const uint32_t dst_max = (1u << WP2Formatbpc(to)) - 1u;
                  const uint32_t dst_alpha =
                      WP2Formatbpalpha(from) > WP2Formatbpalpha(to)
                          ? src_pixel[src_alpha_channel] >>
                                (WP2Formatbpalpha(from) - WP2Formatbpalpha(to))
                      : WP2Formatbpalpha(from) < WP2Formatbpalpha(to)
                          ? src_pixel[src_alpha_channel]
                                << (WP2Formatbpalpha(to) -
                                    WP2Formatbpalpha(from))
                          : src_alpha;
                  tolerance =
                      tolerance * std::max(1u, dst_max / (dst_alpha + 1));
                }
              }
            }
            // Compare the pixels.
            ASSERT_NEAR(roundtrip_pixel[c], src_pixel[c], tolerance)
                << " (channel " << c << ") from " << from << " to " << to
                << " (" << log << ")";
          }
        }
      }
    }
  }
}

TEST(RgbConversionTest, From32bTo32b) {
  WP2ArgbConverterInit();

  for (WP2SampleFormat from :
       {WP2_Argb_32, WP2_ARGB_32, WP2_XRGB_32, WP2_rgbA_32, WP2_RGBA_32,
        WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24,
        WP2_BGR_24}) {
    ArgbBuffer src(from);
    ASSERT_WP2_OK(src.Resize(1, 1));
    ArgbBuffer roundtrip(from);
    ASSERT_WP2_OK(roundtrip.Resize(1, 1));

    for (WP2SampleFormat to :
         {WP2_Argb_32, WP2_ARGB_32, WP2_XRGB_32, WP2_rgbA_32, WP2_RGBA_32,
          WP2_RGBX_32, WP2_bgrA_32, WP2_BGRA_32, WP2_BGRX_32, WP2_RGB_24,
          WP2_BGR_24}) {
      uint8_t values[] = {123, 61, 200, 211};  // Arbitrary.

      uint32_t alpha_channel_index;
      if (WP2FormatHasAlpha(from, &alpha_channel_index)) {
        if (!WP2FormatHasAlpha(to)) {
          // Make sure input is opaque.
          values[alpha_channel_index] = 255;
        } else if (WP2IsPremultiplied(from)) {
          // Make sure alpha is at least as big as the values of the other
          // channels.
          uint8_t* max_element = std::max_element(values, values + 4);
          std::swap(*max_element, values[alpha_channel_index]);
        }
      } else if (alpha_channel_index < WP2FormatNumChannels(from)) {
        values[alpha_channel_index] = 255;
      }

      std::copy(values, values + WP2FormatNumChannels(from), src.GetRow8(0));

      ArgbBuffer dst(to);
      ASSERT_WP2_OK(dst.ConvertFrom(src));
      ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));

      const int tolerance =
          WP2IsPremultiplied(from) == WP2IsPremultiplied(to) ? 0 : 1;
      for (uint32_t i = 0; i < WP2FormatNumChannels(from); ++i) {
        ASSERT_NEAR(roundtrip.GetRow8(0)[i], values[i], tolerance)
            << " (channel " << i << ") from " << from << " to " << to << " ("
            << static_cast<int>(dst.GetRow8(0)[i]) << ")";
      }
    }
  }
}

TEST(RgbConversionTest, From64bTo64b) {
  constexpr WP2SampleFormat kFormats64b[] = {
      WP2_Argb_64, WP2_ARGB_64, WP2_rgbA_64, WP2_RGBA_64,
      WP2_bgrA_64, WP2_BGRA_64, WP2_RGB_48};
  WP2ArgbConverterInit();

  for (WP2SampleFormat from : kFormats64b) {
    ArgbBuffer src(from);
    ASSERT_WP2_OK(src.Resize(1, 1));
    ArgbBuffer roundtrip(from);
    ASSERT_WP2_OK(roundtrip.Resize(1, 1));

    for (WP2SampleFormat to : kFormats64b) {
      uint16_t values[] = {31611, 15677, 51400, 54227};  // Arbitrary.

      uint32_t alpha_channel_index = 0;
      if (WP2FormatHasAlpha(from, &alpha_channel_index) &&
          WP2IsPremultiplied(from)) {
        // Make sure alpha is at least as big as the values of the other
        // channels.
        uint16_t* max_element = std::max_element(values, values + 4);
        std::swap(*max_element, values[alpha_channel_index]);
      }

      std::copy(values, values + WP2FormatNumChannels(from), src.GetRow16(0));

      ArgbBuffer dst(to);
      ASSERT_WP2_OK(dst.ConvertFrom(src));
      ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));

      if (WP2FormatHasAlpha(from) && !WP2FormatHasAlpha(to)) {
        for (uint32_t i = 0; i < WP2FormatNumChannels(from); ++i) {
          if (i == alpha_channel_index) {
            // Alpha was discarded.
            ASSERT_EQ(roundtrip.GetRow16(0)[i], 65535);
          } else if (WP2IsPremultiplied(from)) {
            // Alpha was discarded. Values were unmultiplied.
            const uint32_t src_alpha = src.GetRow16(0)[alpha_channel_index];
            ASSERT_EQ(roundtrip.GetRow16(0)[i],
                      (values[i] * 65535u + src_alpha / 2u) / src_alpha);
          } else {
            ASSERT_EQ(roundtrip.GetRow16(0)[i], values[i]);
          }
        }
      } else {
        const int tolerance =
            WP2IsPremultiplied(from) == WP2IsPremultiplied(to) ? 0 : 1;
        for (uint32_t i = 0; i < WP2FormatNumChannels(from); ++i) {
          ASSERT_NEAR(roundtrip.GetRow16(0)[i], values[i], tolerance);
        }
      }
    }
  }
}

TEST(RgbConversionTest, From32bTo64b) {
  WP2ArgbConverterInit();

  ArgbBuffer src(WP2_ARGB_32);
  ASSERT_WP2_OK(src.Resize(256, 256));
  for (uint32_t y = 0; y < src.height(); ++y) {
    for (uint32_t x = 0; x < src.width(); ++x) {
      src.GetRow8(y)[x * 4] = x;
      src.GetRow8(y)[x * 4 + 1] = y;
      src.GetRow8(y)[x * 4 + 2] = y;
      src.GetRow8(y)[x * 4 + 3] = y;
    }
  }

  ArgbBuffer dst(WP2_ARGB_64);
  ASSERT_WP2_OK(dst.ConvertFrom(src));
  ArgbBuffer roundtrip(WP2_ARGB_32);
  ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));
  ASSERT_TRUE(testutil::Compare(src, roundtrip, "all_alpha_rgb_combinations"));
}

TEST(RgbConversionTest, From32bTo64bPremultiplied) {
  WP2ArgbConverterInit();

  ArgbBuffer src(WP2_Argb_32);
  ASSERT_WP2_OK(src.Resize(256, 256));
  for (uint32_t y = 0; y < src.height(); ++y) {
    for (uint32_t x = 0; x < src.width(); ++x) {
      src.GetRow8(y)[x * 4] = x;
      src.GetRow8(y)[x * 4 + 1] = std::min(x, y);
      src.GetRow8(y)[x * 4 + 2] = std::min(x, y);
      src.GetRow8(y)[x * 4 + 3] = std::min(x, y);
    }
  }

  ArgbBuffer dst(WP2_Argb_64);
  ASSERT_WP2_OK(dst.ConvertFrom(src));
  ArgbBuffer roundtrip(WP2_Argb_32);
  ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));
  ASSERT_TRUE(testutil::Compare(src, roundtrip, "all_alpha_rgb_combinations"));
}

TEST(RgbConversionTest, From64bTo32b) {
  WP2ArgbConverterInit();

  // Try some alpha values but not all, otherwise it takes too long.
  std::vector<uint16_t> alpha_values(550);
  std::iota(alpha_values.begin(), alpha_values.end(), 0);
  alpha_values.push_back(1000);
  alpha_values.push_back(65533);
  alpha_values.push_back(65534);
  alpha_values.push_back(65535);

  for (uint16_t alpha : alpha_values) {
    ArgbBuffer src(WP2_ARGB_64);
    ASSERT_WP2_OK(src.Resize(256, 256));
    // Test all values of RGB.
    for (uint32_t y = 0; y < src.height(); ++y) {
      for (uint32_t x = 0; x < src.width(); ++x) {
        src.GetRow16(y)[x * 4] = alpha;
        src.GetRow16(y)[x * 4 + 1] = x;
        src.GetRow16(y)[x * 4 + 2] = y;
        src.GetRow16(y)[x * 4 + 3] = y * 256 + x;
      }
    }

    ArgbBuffer dst(WP2_ARGB_32);
    ASSERT_WP2_OK(dst.ConvertFrom(src));
    ArgbBuffer roundtrip(WP2_ARGB_64);
    ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));
    ASSERT_TRUE(testutil::Compare(src, roundtrip, "all_alpha_rgb_combinations",
                                  /*expected_distortion=*/40.f));

    for (uint32_t y = 0; y < src.height(); ++y) {
      for (uint32_t x = 0; x < src.width(); ++x) {
        for (uint32_t c = 0; c < 4; ++c) {
          ASSERT_NEAR(src.GetRow16(y)[x * 4 + c],
                      roundtrip.GetRow16(y)[x * 4 + c], 128)
              << " (channel " << c << " at " << x << ", " << y << ")";
        }
      }
    }
  }
}

TEST(RgbConversionTest, From64bTo32bPremultiplied) {
  WP2ArgbConverterInit();

  // Try some alpha values but not all, otherwise it takes too long.
  std::vector<uint16_t> alpha_values(550);
  std::iota(alpha_values.begin(), alpha_values.end(), 0);
  alpha_values.push_back(1000);
  alpha_values.push_back(65533);
  alpha_values.push_back(65534);
  alpha_values.push_back(65535);

  for (uint16_t alpha : alpha_values) {
    ArgbBuffer src(WP2_Argb_64);
    ASSERT_WP2_OK(src.Resize(256, 256));
    // Test all values of RGB.
    for (uint32_t y = 0; y < src.height(); ++y) {
      for (uint32_t x = 0; x < src.width(); ++x) {
        src.GetRow16(y)[x * 4] = alpha;
        src.GetRow16(y)[x * 4 + 1] = std::min<uint16_t>(x, alpha);
        src.GetRow16(y)[x * 4 + 2] = std::min<uint16_t>(y, alpha);
        src.GetRow16(y)[x * 4 + 3] = std::min<uint16_t>(y * 256 + x, alpha);
      }
    }

    ArgbBuffer dst(WP2_Argb_32);
    ASSERT_WP2_OK(dst.ConvertFrom(src));
    ArgbBuffer roundtrip(WP2_Argb_64);
    ASSERT_WP2_OK(roundtrip.ConvertFrom(dst));
    ASSERT_TRUE(testutil::Compare(src, roundtrip, "all_alpha_rgb_combinations",
                                  /*expected_distortion=*/50.f));
  }
}

}  // namespace
}  // namespace WP2
