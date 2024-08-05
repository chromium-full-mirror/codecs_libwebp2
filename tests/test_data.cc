// Copyright 2021 Google LLC
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>

#include "include/helpers.h"
#include "src/wp2/base.h"

namespace WP2 {
namespace {

using TestArray = std::array<uint8_t, 128>;

//------------------------------------------------------------------------------

TEST(DataTest, Simple) {
  Data data;
  EXPECT_TRUE(data.IsEmpty());

  ASSERT_WP2_OK(data.Resize(1, /*keep_bytes=*/true));
  EXPECT_FALSE(data.IsEmpty());

  ASSERT_WP2_OK(data.Resize(0, /*keep_bytes=*/true));
  EXPECT_TRUE(data.IsEmpty());

  TestArray bytes;
  std::iota(bytes.begin(), bytes.end(), 0);
  ASSERT_WP2_OK(data.CopyFrom(bytes.data(), bytes.size()));
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), data.bytes));

  ASSERT_WP2_OK(data.Append(bytes.data(), bytes.size()));
  EXPECT_TRUE(
      std::equal(bytes.begin(), bytes.end(), data.bytes + bytes.size()));
}

//------------------------------------------------------------------------------

TEST(WriterTest, MemoryWriter) {
  TestArray bytes;
  std::iota(bytes.begin(), bytes.end(), 0);

  // Basic write and comparison.
  MemoryWriter writer;
  ASSERT_TRUE(writer.Append(bytes.data(), bytes.size()));
  ASSERT_EQ(writer.size_, bytes.size());
  EXPECT_GE(writer.max_size_, writer.size_);  // Allocated vs written memory.
  ASSERT_TRUE(std::equal(bytes.begin(), bytes.end(), writer.mem_));

  // Should not change anything.
  ASSERT_TRUE(writer.Append(nullptr, 0));
  ASSERT_TRUE(writer.Reserve(0));
  ASSERT_TRUE(std::equal(bytes.begin(), bytes.end(), writer.mem_));

  // Custom Reserve()/Append() amounts.
  writer.Reset();
  ASSERT_TRUE(writer.Reserve(1));
  ASSERT_TRUE(writer.Append(bytes.data(), 2));
  ASSERT_TRUE(writer.Reserve(10000));
  ASSERT_TRUE(writer.Append(bytes.data() + 2, bytes.size() - 2));
  ASSERT_EQ(writer.size_, bytes.size());
  EXPECT_GE(writer.max_size_, writer.size_);  // Allocated vs written memory.
  ASSERT_TRUE(std::equal(bytes.begin(), bytes.end(), writer.mem_));
}

TEST(WriterTest, StringWriter) {
  TestArray bytes;
  std::iota(bytes.begin(), bytes.end(), 0);

  // Write and compare.
  std::string str;
  StringWriter writer(&str);
  ASSERT_TRUE(writer.Reserve(1));
  ASSERT_TRUE(writer.Append(bytes.data(), 2));
  ASSERT_TRUE(writer.Reserve(10000));
  ASSERT_TRUE(writer.Append(bytes.data() + 2, bytes.size() - 2));
  ASSERT_EQ(str.size(), bytes.size());
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), str.begin()));

  // Should not change anything.
  ASSERT_TRUE(writer.Append(nullptr, 0));
  ASSERT_TRUE(writer.Reserve(0));
  ASSERT_TRUE(std::equal(bytes.begin(), bytes.end(), str.begin()));
}

TEST(WriterTest, StringWriterNull) {
  StringWriter string_writer(nullptr);
  EXPECT_FALSE(string_writer.Reserve(1));
  EXPECT_FALSE(string_writer.Append(nullptr, 1));
}

TEST(WriterTest, DataWriter) {
  TestArray bytes;
  std::iota(bytes.begin(), bytes.end(), 0);

  // Write and compare.
  Data data;
  DataWriter writer(&data);
  ASSERT_TRUE(writer.Reserve(1));
  ASSERT_TRUE(writer.Append(bytes.data(), 2));
  ASSERT_TRUE(writer.Reserve(10000));
  ASSERT_TRUE(writer.Append(bytes.data() + 2, bytes.size() - 2));
  ASSERT_EQ(writer.size(), bytes.size());
  ASSERT_GE(data.size, bytes.size());  // Allocated vs written memory.
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), data.bytes));

  // Should not change anything.
  ASSERT_TRUE(writer.Append(nullptr, 0));
  ASSERT_TRUE(writer.Reserve(0));
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), data.bytes));
}

TEST(WriterTest, DataWriterNull) {
  DataWriter data_writer(nullptr);
  EXPECT_FALSE(data_writer.Reserve(1));
  EXPECT_FALSE(data_writer.Append(nullptr, 1));
}

TEST(WriterTest, Counter) {
  TestArray bytes;
  std::iota(bytes.begin(), bytes.end(), 0);

  // Write and compare.
  Counter writer;
  ASSERT_TRUE(writer.Reserve(1));
  ASSERT_TRUE(writer.Append(bytes.data(), 2));
  ASSERT_TRUE(writer.Reserve(10000));
  ASSERT_TRUE(writer.Append(bytes.data() + 2, bytes.size() - 2));
  ASSERT_EQ(writer.size_, bytes.size());

  // Should not change anything.
  ASSERT_TRUE(writer.Append(nullptr, 0));
  ASSERT_TRUE(writer.Reserve(0));
  ASSERT_EQ(writer.size_, bytes.size());
}

//------------------------------------------------------------------------------

}  // namespace
}  // namespace WP2
