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
//
// Writer implementations
//
// Author: Skal (pascal.massimino@gmail.com)

#include <algorithm>
#include <cassert>
#include <cstring>  // for memcpy()
#include <string>

#include "src/utils/utils.h"
#include "src/wp2/encode.h"

namespace WP2 {

//------------------------------------------------------------------------------

bool MemoryWriter::Reserve(size_t size) {
  const uint64_t next_size = (uint64_t)size_ + size;
  if (next_size <= max_size_) return true;
  const uint64_t kMinSize = 8192u;
  const uint64_t next_max_size =
      std::max({kMinSize, 2u * (uint64_t)max_size_, next_size});
  uint8_t* const new_mem = (uint8_t*)WP2Malloc(next_max_size, 1);
  if (new_mem == nullptr) return false;
  if (size_ > 0) memcpy(new_mem, mem_, size_);
  WP2Free(mem_);
  mem_ = new_mem;
  // down-cast is ok, thanks to WP2Malloc
  max_size_ = (size_t)next_max_size;
  return true;
}

bool MemoryWriter::Append(const void* data, size_t data_size) {
  if (data_size > 0) {
    if (!Reserve(data_size)) return false;
    memcpy(mem_ + size_, data, data_size);
    size_ += data_size;
  }
  return true;
}

void MemoryWriter::Reset() {
  WP2Free(mem_);
  mem_ = nullptr;
  size_ = 0;
  max_size_ = 0;
}

//------------------------------------------------------------------------------

bool StringWriter::Reserve(size_t size) {
  if (str_ != nullptr) str_->reserve(str_->size() + size);
  return true;
}

bool StringWriter::Append(const void* data, size_t data_size) {
  if (str_ != nullptr && data_size > 0) {
    str_->append((const char*)data, data_size);
  }
  return true;
}

//------------------------------------------------------------------------------

bool Counter::Append(const void* data, size_t data_size) {
  (void)data;
  total_size_ += data_size;
  return true;
}

//------------------------------------------------------------------------------

}  // namespace WP2
