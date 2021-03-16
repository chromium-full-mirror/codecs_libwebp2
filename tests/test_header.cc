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

// Test different headers (~encoder configs)

#include "include/helpers.h"
#include "src/common/constants.h"
#include "src/common/header_enc_dec.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/math.h"
#include "src/enc/anim/anim_enc.h"
#include "src/utils/data_source.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace {

//------------------------------------------------------------------------------

TEST(HeaderTest, VarInt) {
  uint8_t dst[kMaxVarIntLength];
  uint32_t decoded_value;

  // 0 bit written, 0 bit read.
  {
    ASSERT_EQ(WriteVarInt(/*value=*/0, /*min_value=*/0, /*max_value=*/0, dst),
              0u);
    ExternalDataSource data_source(dst, 0);
    ASSERT_WP2_OK(TryReadVarInt(&data_source, /*min_value=*/0, /*max_value=*/0,
                                &decoded_value));
    ASSERT_EQ(decoded_value, 0u);
    ASSERT_EQ(data_source.GetNumReadBytes(), 0u);
  }

  constexpr uint32_t kMin = 0;
  ASSERT_EQ(WriteVarInt(/*value=*/0, kMin, /*max_value=*/1, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/1, kMin, /*max_value=*/1, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/127, kMin, /*max_value=*/255, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/128, kMin, /*max_value=*/255, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/127, kMin, /*max_value=*/256, dst), 1u);
  ASSERT_EQ(WriteVarInt(/*value=*/128, kMin, /*max_value=*/256, dst), 2u);
  ASSERT_EQ(WriteVarInt(/*value=*/255, kMin, /*max_value=*/256, dst), 2u);
  ASSERT_EQ(WriteVarInt(/*value=*/256, kMin, /*max_value=*/256, dst), 2u);

  // clang-format off
  const uint32_t kValues[] =
      {0, 1, 2, 3, 5, 8, 126, 127, 128, 129, 254, 255, 256, 257, 260768,
       (kMaxVarInt >> 7) - 1, (kMaxVarInt >> 7), (kMaxVarInt >> 7) + 1,
       kMaxVarInt - 2, kMaxVarInt - 1, kMaxVarInt};
  // clang-format on

  for (size_t min_value : kValues) {
    for (size_t max_value : kValues) {
      if (min_value > max_value) break;
      for (size_t value : kValues) {
        if (value < min_value || value > max_value) break;

        const uint32_t num_bytes =
            WriteVarInt(value, min_value, max_value, dst);
        ASSERT_GE(num_bytes, (min_value == max_value) ? 0u : 1u);
        ASSERT_LE(num_bytes, GetMaxVarIntLength(min_value, max_value));

        {
          ExternalDataSource data_source(dst, num_bytes);
          ASSERT_WP2_OK(TryReadVarInt(&data_source, min_value, max_value,
                                      &decoded_value));
          ASSERT_EQ(decoded_value, value);
        }
        for (uint32_t not_enough = 0; not_enough < num_bytes; ++not_enough) {
          ExternalDataSource data_source(dst, not_enough);
          ASSERT_EQ(
              TryReadVarInt(&data_source, min_value, max_value, &decoded_value),
              WP2_STATUS_NOT_ENOUGH_DATA);
        }
      }
    }
  }

  // Bitstream error.
  {
    const uint32_t num_bytes =
        WriteVarInt(/*value=*/200, /*min_value=*/10, /*max_value=*/202, dst);
    ExternalDataSource data_source(dst, num_bytes);
    ASSERT_EQ(TryReadVarInt(&data_source, /*min_value=*/10, /*max_value=*/198,
                            &decoded_value),
              WP2_STATUS_BITSTREAM_ERROR);
  }
}

//------------------------------------------------------------------------------

TEST(HeaderTest, BitPackerUnpacker) {
  const uint32_t kSize = 500;
  uint8_t buf[kSize];
  BitPacker enc(buf, kSize, "test");
  ExternalDataSource data_source(buf, sizeof(buf));
  BitUnpacker dec(&data_source, "test");

  for (bool encoding : {true, false}) {
    UniformIntDistribution random(/*seed=*/1664);
    int bit_pos = 0;
    while (encoding ? enc.Ok() : (dec.GetStatus() == WP2_STATUS_OK)) {
      if ((bit_pos & 7) != 0 || random.FlipACoin()) {
        // Encode raw bits.
        const uint32_t num_bits = random.Get(1u, 24u);
        if (bit_pos + num_bits > kSize * 8) break;

        const bool is_signed = random.FlipACoin();
        if (is_signed) {
          const int32_t max_abs_v = (1 << (num_bits - 1u));
          const int32_t v = random.Get(-max_abs_v, max_abs_v - 1);
          if (encoding) {
            enc.PutSBits(v, num_bits, "sbit");
          } else {
            EXPECT_EQ(dec.ReadSBits(num_bits, "sbit"), v) << "at " << bit_pos;
          }
        } else {
          const uint32_t v = random.Get(0u, (1u << num_bits) - 1u);
          if (encoding) {
            enc.PutBits(v, num_bits, "bit");
          } else {
            EXPECT_EQ(dec.ReadBits(num_bits, "bit"), v) << "at " << bit_pos;
          }
        }
        bit_pos += num_bits;
      } else {
        const size_t num_bytes_before =
            encoding ? enc.Used() : data_source.GetNumReadBytes();
        if (random.FlipACoin()) {
          // Encode variable integer.
          // TODO(yguyon): Handle UniformIntDistribution::Get(0, kMaxVarInt)
          const size_t max_value = random.Get<size_t>(0, 1000000);
          const size_t min_value = random.Get<size_t>(0, max_value);
          const size_t value = random.Get<size_t>(min_value, max_value);
          if (bit_pos + GetMaxVarIntLength(min_value, max_value) * 8 >
              kSize * 8) {
            break;
          }
          if (encoding) {
            enc.PutVarUInt(value, min_value, max_value, "uint");
          } else {
            EXPECT_EQ(dec.ReadVarUInt(min_value, max_value, "uint"), value)
                << "at " << bit_pos;
          }
          const size_t num_bytes_after =
              encoding ? enc.Used() : data_source.GetNumReadBytes();
          bit_pos += (num_bytes_after - num_bytes_before) * 8;
        } else {
          // Encode rectangle.
          const uint32_t canvas_width = random.Get(1u, 1000000u);
          const uint32_t canvas_height = random.Get(1u, 1000000u);
          const uint32_t width = random.Get(1u, canvas_width);
          const uint32_t height = random.Get(1u, canvas_height);
          const uint32_t x = random.Get(0u, canvas_width - width);
          const uint32_t y = random.Get(0u, canvas_height - height);
          const Rectangle rect = {x, y, width, height};
          if (bit_pos + 4 * GetMaxVarIntLength(0u, 1000000u) * 8 > kSize * 8) {
            break;
          }
          if (encoding) {
            enc.PutRect(rect, canvas_width, canvas_height, "rect");
          } else {
            EXPECT_EQ(dec.ReadRect(canvas_width, canvas_height, "rect"), rect)
                << "at " << bit_pos;
          }
        }
        const size_t num_bytes_after =
            encoding ? enc.Used() : data_source.GetNumReadBytes();
        bit_pos += (num_bytes_after - num_bytes_before) * 8;
      }
    }

    if (encoding) {
      ASSERT_TRUE(enc.Ok()) << "Error encoding header.";
    } else {
      ASSERT_WP2_OK(dec.GetStatus()) << "Error decoding header.";
    }
  }
}

TEST(HeaderTest, BitUnpackerPrefetchBeforeReading) {
  uint8_t buf[] = {1, 2, 3};
  ExternalDataSource data_source(buf, sizeof(buf));
  BitUnpacker dec(&data_source, "test");

  for (uint32_t num_bits = 0; num_bits <= sizeof(buf) * 8; ++num_bits) {
    EXPECT_TRUE(dec.Prefetch(num_bits));
  }
  EXPECT_FALSE(dec.Prefetch(sizeof(buf) * 8 + 1));
  ASSERT_EQ(dec.ReadBits(1, "debug_label"), 1u);
  ASSERT_EQ(dec.ReadBits(1, "debug_label"), 0u);
}

TEST(HeaderTest, BitUnpackerPrefetchAfterReading) {
  uint8_t buf[] = {1, 2, 3};
  ExternalDataSource data_source(buf, sizeof(buf));
  BitUnpacker dec(&data_source, "test");

  ASSERT_EQ(dec.ReadBits(1, "debug_label"), 1u);
  ASSERT_EQ(dec.ReadBits(1, "debug_label"), 0u);
  ASSERT_WP2_OK(dec.GetStatus());
  EXPECT_TRUE(dec.Prefetch(sizeof(buf) * 8 - 2));
  EXPECT_FALSE(dec.Prefetch(sizeof(buf) * 8 - 1));
}

//------------------------------------------------------------------------------

TEST(HeaderTest, MinHeaderSize) {
  ArgbBuffer image;
  ASSERT_WP2_OK(image.Resize(1, 1));
  image.Fill(Argb32b{255, 0, 0, 0});

  MemoryWriter memory_writer;
  ASSERT_WP2_OK(Encode(image, &memory_writer));

  BitstreamFeatures features;
  ASSERT_WP2_OK(features.Read(memory_writer.mem_, memory_writer.size_));
  EXPECT_EQ(features.header_size, kHeaderMinSize);
}

TEST(HeaderTest, MaxHeaderSize) {
  const uint8_t num_frames = 2;
  const uint32_t duration_ms = 100u;
  AnimationEncoder encoder;
  for (uint8_t frame_index = 0; frame_index < num_frames; ++frame_index) {
    ArgbBuffer frame;
    ASSERT_WP2_OK(frame.Resize(1, 1));
    const Argb32b color{255, frame_index, frame_index, frame_index};
    frame.Fill(color);
    ASSERT_WP2_OK(encoder.AddFrame(frame, duration_ms));
  }

  EncoderConfig encoder_config = EncoderConfig::kDefault;
  encoder_config.transfer_function = WP2_TF_ITU_R_BT709;  // Any besides BT2020
  MemoryWriter memory_writer;
  ASSERT_WP2_OK(encoder.Encode(&memory_writer, encoder_config));

  BitstreamFeatures features;
  ASSERT_WP2_OK(features.Read(memory_writer.mem_, memory_writer.size_));
  // TODO(yguyon): Compare with exactly 'kHeaderMaxSize' when it's possible to
  //               reach it (with a 'kBackgroundCustom' color)
  EXPECT_EQ(features.header_size, kHeaderMaxSize - 4);
}

//------------------------------------------------------------------------------

TEST(HeaderTest, TestAlpha) {
  for (bool is_opaque : {true, false}) {
    ArgbBuffer image;
    ASSERT_WP2_OK(image.Resize(1, 1));
    image.Fill(Argb32b{(uint8_t)(is_opaque ? 255 : 128), 0, 0, 0});

    MemoryWriter memory_writer;
    ASSERT_WP2_OK(Encode(image, &memory_writer));

    BitstreamFeatures features;
    ASSERT_WP2_OK(features.Read(memory_writer.mem_, memory_writer.size_));
    EXPECT_EQ(features.is_opaque, is_opaque);
  }
}

//------------------------------------------------------------------------------

// Nearly-exhaustive ANMF encode/decode test.
TEST(HeaderTest, ANMF) {
  constexpr uint32_t kDims[] = {1, 2, kImageDimMax};
  constexpr uint32_t kDurations[] = {1, kMaxFrameDurationMs};
  for (uint32_t w : kDims) {
    for (uint32_t h : kDims) {
      BitstreamFeatures fake_features;
      fake_features.is_animation = true;
      fake_features.raw_width = w;
      fake_features.raw_height = h;
      for (uint32_t duration_ms : kDurations) {
        for (uint32_t window_width :
             {1u, std::min(2u, w), DivCeil(w, 3u), std::max(1u, w - 1u), w}) {
          for (uint32_t window_height :
               {1u, std::min(2u, h), DivCeil(h, 3u), std::max(1u, h - 1u), h}) {
            const uint32_t max_x = w - window_width, max_y = h - window_height;
            for (uint32_t window_x :
                 {0u, std::min(1u, max_x), max_x / 2u, max_x}) {
              for (uint32_t window_y :
                   {0u, std::min(1u, max_y), max_y / 2u, max_y}) {
                const Rectangle window = {window_x, window_y, window_width,
                                          window_height};
                for (bool dispose : {false, true}) {
                  for (bool blend : {false, true}) {
                    for (bool is_last : {false, true}) {
                      MemoryWriter writer;
                      ASSERT_WP2_OK(WriteANMF(w, h, duration_ms, window,
                                              dispose, blend, is_last,
                                              &writer));
                      ASSERT_GE(writer.size_, kANMFMinHeaderSize);
                      ASSERT_LE(writer.size_, kANMFMaxHeaderSize);
                      ExternalDataSource data_source(writer.mem_, writer.size_);
                      const uint32_t frame_index =
                          (dispose ? 0u : kMaxNumFrames - 1u);
                      AnimationFrame frame;
                      ASSERT_WP2_OK(DecodeANMF(DecoderConfig::kDefault,
                                               &data_source, fake_features,
                                               frame_index, &frame));
                      ASSERT_EQ(frame.dispose, dispose);
                      ASSERT_EQ(frame.blend, blend);
                      ASSERT_EQ(frame.duration_ms, duration_ms);
                      ASSERT_EQ(frame.window, window);
                      ASSERT_EQ(frame.is_last, is_last);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(HeaderTest, ANMFMin) {
  MemoryWriter writer;
  ASSERT_WP2_OK(WriteANMF(/*image_width=*/1, /*image_height=*/1, /*duration=*/0,
                          /*window=*/{0, 0, 1, 1}, /*dispose=*/false,
                          /*blend=*/false, /*is_last=*/false, &writer));
  ASSERT_EQ(writer.size_, kANMFMinHeaderSize);
}

TEST(HeaderTest, ANMFMax) {
  MemoryWriter writer;
  ASSERT_WP2_OK(WriteANMF(
      /*image_width=*/kImageDimMax, /*image_height=*/kImageDimMax,
      /*duration=*/kMaxFrameDurationMs, /*window=*/
      {kImageDimMax / 2 - 1, kImageDimMax / 2 - 1, kImageDimMax / 2,
       kImageDimMax / 2},
      /*dispose=*/true, /*blend=*/true, /*is_last=*/true, &writer));
  ASSERT_EQ(writer.size_, kANMFMaxHeaderSize);
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
