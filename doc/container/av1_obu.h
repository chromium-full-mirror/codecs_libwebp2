// Copyright 2021 Google LLC
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

#ifndef CONTAINER_AV1_OBU_H_
#define CONTAINER_AV1_OBU_H_

#include <cstddef>
#include <cstdint>

namespace container {

//------------------------------------------------------------------------------

// Subset of the features stored in a Sequence Header OBU.
// See "AV1 Bitstream & Decoding Process Specification - 5.5.1."
struct SequenceHeader {
  uint8_t seq_profile;
  uint8_t seq_level_idx_0;
  bool high_bitdepth;
  bool twelve_bit;
  bool mono_chrome;
  uint8_t subsampling_x;
  uint8_t subsampling_y;
  uint8_t chroma_sample_position;
};

// Decodes the content of a Sequence Header OBU 'data' chunk.
// Returns false in case of parsing error.
bool UnwrapOBUSequenceHeader(const uint8_t data[], size_t data_size,
                             SequenceHeader& header);

//------------------------------------------------------------------------------

}  // namespace container

#endif  // CONTAINER_AV1_OBU_H_
