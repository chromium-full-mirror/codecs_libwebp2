// Copyright 2019 Google LLC
// Copyright 2018 The Abseil Authors.
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
// Pixel hash functions, copied from abseil's city hash.

#ifndef WP2_UTILS_HASH_H_
#define WP2_UTILS_HASH_H_

#include "src/dsp/dsp.h"

//------------------------------------------------------------------------------
// Copied from abseil city.cc computing city hash.
// The char buffer has been replaced by a uint8_t one and styling changed.
// The endianness is checked with WP2_WORDS_BIGENDIAN (big-endian if defined,
// little-endian otherwise).

namespace WP2 {

namespace internal {
static inline uint32_t Fetch32(const uint8_t* const p) {
#ifdef WP2_WORDS_BIGENDIAN
  return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | (p[3] << 0);
#else
  return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | (p[0] << 0);
#endif
}

// Magic numbers for 32-bit hashing.  Copied from Murmur3.
static constexpr uint32_t c1 = 0xcc9e2d51;
static constexpr uint32_t c2 = 0x1b873593;

static inline uint32_t Rotate32(uint32_t val, int shift) {
  // Avoid shifting by 32: doing so yields an undefined result.
  return (shift == 0) ? val : ((val >> shift) | (val << (32 - shift)));
}

// A 32-bit to 32-bit integer hash copied from Murmur3.
static inline uint32_t fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

static inline uint32_t Mur(uint32_t a, uint32_t h) {
  // Helper from Murmur3 for combining two 32-bit values.
  a *= c1;
  a = Rotate32(a, 17);
  a *= c2;
  h ^= a;
  h = Rotate32(h, 19);
  return h * 5 + 0xe6546b64;
}

static inline uint32_t Hash32Len13to24(const uint8_t* const s, uint32_t len,
                                       uint32_t seed = 0) {
  uint32_t a = Fetch32(s - 4 + (len >> 1));
  uint32_t b = Fetch32(s + 4);
  uint32_t c = Fetch32(s + len - 8);
  uint32_t d = Fetch32(s + (len >> 1));
  uint32_t e = Fetch32(s);
  uint32_t f = Fetch32(s + len - 4);
  uint32_t h = d * c1 + len + seed;
  a = Rotate32(a, 12) + f;
  h = Mur(c, h) + a;
  a = Rotate32(a, 3) + c;
  h = Mur(e, h) + a;
  a = Rotate32(a + f, 12) + d;
  h = Mur(b ^ seed, h) + a;
  return fmix(h);
}

static inline uint32_t Hash32Len0to4(const uint8_t* const s, uint32_t len) {
  uint32_t b = 0;
  uint32_t c = 9;
  for (uint32_t i = 0; i < len; i++) {
    int8_t v = s[i];
    b = b * c1 + v;
    c ^= b;
  }
  return fmix(Mur(b, Mur(len, c)));
}

static inline uint32_t Hash32Len5to12(const uint8_t* const s, uint32_t len,
                                      uint32_t seed = 0) {
  uint32_t a = len, b = len * 5, c = 9, d = b + seed;
  a += Fetch32(s);
  b += Fetch32(s + len - 4);
  c += Fetch32(s + ((len >> 1) & 4));
  return fmix(seed ^ Mur(c, Mur(b, Mur(a, d))));
}
}  // namespace internal

#ifdef WP2_WORDS_BIGENDIAN
#define SPLIT(X) (uint8_t)((uint16_t)argb[X] >> 8), (uint8_t)(argb[X] & 0xff)
#else
#define SPLIT(X) (uint8_t)(argb[X] & 0xff), (uint8_t)((uint16_t)argb[X] >> 8)
#endif

// Defines a 'hash_bits'-bit hash value of a pixel.
// TODO(vrabaud) try CPU-lighter hashes.
static inline uint32_t HashPix(const uint8_t* const argb, uint32_t hash_bits) {
  return internal::Hash32Len0to4(argb, 4) & ((1u << hash_bits) - 1);
}
static inline uint32_t HashPix(const int16_t* const argb, uint32_t hash_bits) {
  const uint8_t s[8] = {SPLIT(0), SPLIT(1), SPLIT(2), SPLIT(3)};
  return internal::Hash32Len5to12(s, 8) & ((1u << hash_bits) - 1);
}
static inline uint32_t HashPix(const uint16_t* const argb, uint32_t hash_bits) {
  const uint8_t s[8] = {SPLIT(0), SPLIT(1), SPLIT(2), SPLIT(3)};
  return internal::Hash32Len5to12(s, 8) & ((1u << hash_bits) - 1);
}
// Defines a 'hash_bits'-bit hash value of a pai of pixels.
static inline uint32_t HashPixPair(const int16_t argb[8], uint32_t hash_bits) {
  const uint8_t s[16] = {SPLIT(0), SPLIT(1), SPLIT(2), SPLIT(3),
                         SPLIT(4), SPLIT(5), SPLIT(6), SPLIT(7)};
  return internal::Hash32Len13to24(s, 16) & ((1u << hash_bits) - 1);
}

#undef SPLIT

}  // namespace WP2

#endif  // WP2_UTILS_HASH_H_
