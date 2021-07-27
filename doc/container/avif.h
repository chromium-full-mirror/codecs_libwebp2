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

#ifndef CONTAINER_AVIF_H_
#define CONTAINER_AVIF_H_

#include <cstddef>
#include <cstdint>

#include "./container.h"

namespace container {

//------------------------------------------------------------------------------

// Wraps the input 'image' characteristics and compressed 'codec_bytes' into
// AVIF 'data'. The 'codec_bytes' must be a valid AV1-OBU payload.
// Returns false in case of error.
bool WrapAvif(const Image& image, const uint8_t codec_bytes[],
              size_t num_codec_bytes, Data& data);

// Parses the AVIF 'data' into 'image' characteristics and returns the position
// of the compressed codec_bytes. Does not decode the pixels.
// Returns false in case of error.
bool UnwrapAvif(const uint8_t data[], size_t data_size, Image& image,
                size_t& offset_till_codec_bytes, size_t& num_codec_bytes);

//------------------------------------------------------------------------------

// Displays the layout of an ISO Base Media File Format file to standard output.
void PrintBmffBoxes(const uint8_t data[], size_t data_size);

//------------------------------------------------------------------------------

}  // namespace container

#endif  // CONTAINER_AVIF_H_
