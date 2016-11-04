#ifndef INCLUDE_SIPHASHX4_H
#define INCLUDE_SIPHASHX4_H

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "highwayhash/sip_tree_hash.h"

#ifdef __AVX2__
#include <cstring>  // memcpy

#include "highwayhash/sip_hash.h"
#include "highwayhash/vec2.h"

namespace highwayhash {
namespace {

// Paper: https://www.131002.net/siphash/siphash.pdf
// SSE41 implementation: https://goo.gl/80GBSD
// Tree hash extension: http://dx.doi.org/10.4236/jis.2014.53010

// The hash state is updated by injecting 4x8-byte packets;
// XORing together all state vectors yields 32 bytes that are
// reduced to 64 bits via 8-byte SipHash.

const int kPacketSize = 32;
const int kNumLanes = kPacketSize / sizeof(uint64);

// 32 bytes key. Parameters are hardwired to c=2, d=4 [rounds].
class SipTreeHashState {
 public:
  explicit INLINE SipTreeHashState(const siphash_keys &keys) {
    const V4x64U init(keys.k1^0x7465646279746573ull, keys.k0^0x6c7967656e657261ull,
                      keys.k1^0x646f72616e646f6dull, keys.k0^0x736f6d6570736575ull);
    v0 = V4x64U(_mm256_permute4x64_epi64(init, 0x00));
    v1 = V4x64U(_mm256_permute4x64_epi64(init, 0x55));
    v2 = V4x64U(_mm256_permute4x64_epi64(init, 0xAA));
    v3 = V4x64U(_mm256_permute4x64_epi64(init, 0xFF));
  }

  INLINE void Update(const V4x64U& packet) {
    v3 ^= packet;
    Compress<2>();
    v0 ^= packet;
  }

  INLINE V4x64U Finalize() {
    // Mix in bits to avoid leaking the key if all packets were zero.
    v2 ^= V4x64U(0xFF);
    Compress<4>();
#ifdef HASHONLY
    return ((v0 ^ v1) ^ (v2 ^ v3)) & V4x64U(1ull);
#else
    return (v0 ^ v1) ^ (v2 ^ v3);
#endif
  }

 private:
  static INLINE V4x64U RotateLeft16(const V4x64U& v) {
    const V4x64U control(0x0D0C0B0A09080F0EULL, 0x0504030201000706ULL,
                         0x0D0C0B0A09080F0EULL, 0x0504030201000706ULL);
    return V4x64U(_mm256_shuffle_epi8(v, control));
  }

  // Rotates each 64-bit element of "v" left by N bits.
  template <uint64 bits>
  static INLINE V4x64U RotateLeft(const V4x64U& v) {
    const V4x64U left = v << bits;
    const V4x64U right = v >> (64 - bits);
    return left | right;
  }

  static INLINE V4x64U Rotate32(const V4x64U& v) {
    return V4x64U(_mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
  }

  template <size_t rounds>
  INLINE void Compress() {
    // Loop is faster than unrolling!
    for (size_t i = 0; i < rounds; ++i) {
      // ARX network: add, rotate, exclusive-or.
      v0 += v1;
      v2 += v3;
      v1 = RotateLeft<13>(v1);
      v3 = RotateLeft16(v3);
      v1 ^= v0;
      v3 ^= v2;

      v0 = Rotate32(v0);

      v2 += v1;
      v0 += v3;
      v1 = RotateLeft<17>(v1);
      v3 = RotateLeft<21>(v3);
      v1 ^= v2;
      v3 ^= v0;

      v2 = Rotate32(v2);
    }
  }

  V4x64U v0;
  V4x64U v1;
  V4x64U v2;
  V4x64U v3;
};

}  // namespace

INLINE void siphashx4(const siphash_keys *keys, const uint64 * const indices, uint64 * const hashes) {
  SipTreeHashState state(*keys);
  state.Update(Load256(indices));
  Store(state.Finalize(), hashes);
}

}  // namespace highwayhash

using highwayhash::uint64;
using highwayhash::SipTreeHash;

#endif  // #ifdef __AVX2__

#endif // ifdef INCLUDE_SIPHASHX4_H
