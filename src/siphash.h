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

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

// siphash uses a pair of 64-bit keys,
typedef struct {
  u64 k0;
  u64 k1;
  u64 k2;
  u64 k3;
} siphash_keys;
 
#define U8TO64_LE(p) ((p))

// set doubled (128->256 bits) siphash keys from 32 byte char array
void setkeys(siphash_keys *keys, const char *keybuf) {
  keys->k0 = htole64(((u64 *)keybuf)[0]);
  keys->k1 = htole64(((u64 *)keybuf)[1]);
  keys->k2 = htole64(((u64 *)keybuf)[2]);
  keys->k3 = htole64(((u64 *)keybuf)[3]);
}

#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  } while(0)
 
// SipHash-2-4 without standard IV xor and specialized to precomputed key and 8 byte nonces
u64 siphash24(const siphash_keys *keys, const u64 nonce) {
  u64 v0 = keys->k0, v1 = keys->k1, v2 = keys->k2, v3 = keys->k3 ^ nonce;
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
