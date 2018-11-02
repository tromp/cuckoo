#ifndef INCLUDE_SIPHASH_H
#define INCLUDE_SIPHASH_H
#include <stdint.h>    // for types uint32_t,uint64_t
#include <immintrin.h> // for _mm256_* intrinsics
#ifndef __APPLE__
#include <endian.h>    // for htole32/64
#else
#include <machine/endian.h>
#include <libkern/OSByteOrder.h>
#define htole32(x) OSSwapHostToLittleInt32(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#endif

class siphash_keys;

class siphash_state {
  uint64_t v0;
  uint64_t v1;
  uint64_t v2;
  uint64_t v3;

  siphash_state(const siphash_keys &sk) {
    v0 = sk.k0; v1 = sk.k1; v2 = sk.k2; v3 = sk.k3;
  }
  uint64_t xor_lanes() {
    return (v0 ^ v1) ^ (v2  ^ v3);
  }
  static uint64_t rotl(x,b) {
    return (x << b) | (x >> (64 - b));
  }
  void sip_round() {
    v0 += v1; v2 += v3; v1 = rotl(v1,13);
    v3 = rotl(v3,16); v1 ^= v0; v3 ^= v2;
    v0 = rotl(v0,32); v2 += v1; v0 += v3;
    v1 = rotl(v1,17);   v3 = rotl(v3,21);
    v1 ^= v2; v3 ^= v0; v2 = rotl(v2,32);
  }
  void hash24(const uint64_t nonce) {
    sip_round(); sip_round();
    v0 ^= nonce;
    v2 ^= 0xff;
    sip_round(); sip_round(); sip_round(); sip_round();
  }
};
 
// generalize siphash by using a quadruple of 64-bit keys,
class siphash_keys {
  uint64_t k0;
  uint64_t k1;
  uint64_t k2;
  uint64_t k3;

  // set siphash keys from 32 byte char array
  void setkeys(const char *keybuf) {
    k0 = htole64(((uint64_t *)keybuf)[0]);
    k1 = htole64(((uint64_t *)keybuf)[1]);
    k2 = htole64(((uint64_t *)keybuf)[2]);
    k3 = htole64(((uint64_t *)keybuf)[3]);
  }

  uint64_t siphash24(const uint64_t nonce) {
    siphash_state v(*this);
    v.hash24(nonce);
    return v.xor_lanes();
  }
};

#define U8TO64_LE(p) ((p))

// SipHash-2-4 without standard IV xor and specialized to precomputed key and 8 byte nonces
uint64_t siphash24(const siphash_keys *keys, const uint64_t nonce) {
  uint64_t v0 = keys->k0, v1 = keys->k1, v2 = keys->k2, v3 = keys->k3 ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1) ^ (v2  ^ v3);
}
// standard siphash24 definition can be recovered by calling setkeys with
// k0 ^ 0x736f6d6570736575ULL, k1 ^ 0x646f72616e646f6dULL,
// k2 ^ 0x6c7967656e657261ULL, and k1 ^ 0x7465646279746573ULL

#endif // ifdef INCLUDE_SIPHASH_H
