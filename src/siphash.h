#ifndef INCLUDE_SIPHASH_H
#define INCLUDE_SIPHASH_H
#include <stdint.h> // for types uint32_t,uint64_t
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"
#include <immintrin.h>

// length of header hashed into siphash key
#ifndef HEADERLEN
#define HEADERLEN 80
#endif

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

// siphash uses a pair of 64-bit keys,
typedef struct {
  u64 k0;
  u64 k1;
} siphash_keys;
 
#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))

#ifndef SHA256
#define SHA256(d, n, md) do { \
    SHA256_CTX c; \
    SHA256_Init(&c); \
    SHA256_Update(&c, d, n); \
    SHA256_Final(md, &c); \
  } while (0)
#endif
 
// derive siphash key from fixed length header
void setheader(siphash_keys *keys, const char *header) {
  unsigned char hdrkey[32];
  SHA256((unsigned char *)header, HEADERLEN, hdrkey);
  keys->k0 = U8TO64_LE(hdrkey);
  keys->k1 = U8TO64_LE(hdrkey+8);
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
 
// SipHash-2-4 specialized to precomputed key and 8 byte nonces
u64 siphash24(const siphash_keys *keys, const u64 nonce) {
  u64 v0 = keys->k0 ^ 0x736f6d6570736575ULL, v1 = keys->k1 ^ 0x646f72616e646f6dULL,
      v2 = keys->k0 ^ 0x6c7967656e657261ULL, v3 = keys->k1 ^ 0x7465646279746573ULL ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1) ^ (v2  ^ v3);
}

#define ADD(a, b) _mm256_add_epi64(a, b)
#define XOR(a, b) _mm256_xor_si256(a, b)
#define ROTATE16 _mm256_set_epi64x(0x0D0C0B0A09080F0EULL,0x0504030201000706ULL, \
                                   0x0D0C0B0A09080F0EULL, 0x0504030201000706ULL)
#define ROT13(x) _mm256_or_si256(_mm256_slli_epi64(x,13),_mm256_srli_epi64(x,51))
#define ROT16(x) _mm256_shuffle_epi8((x), ROTATE16)
#define ROT17(x) _mm256_or_si256(_mm256_slli_epi64(x,17),_mm256_srli_epi64(x,47))
#define ROT21(x) _mm256_or_si256(_mm256_slli_epi64(x,21),_mm256_srli_epi64(x,43))
#define ROT32(x) _mm256_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))

#define SIPROUNDX4 \
  do { \
    v0 = ADD(v0,v1); v2 = ADD(v2,v3); v1 = ROT13(v1); \
    v3 = ROT16(v3);  v1 = XOR(v1,v0); v3 = XOR(v3,v2); \
    v0 = ROT32(v0);  v2 = ADD(v2,v1); v0 = ADD(v0,v3); \
    v1 = ROT17(v1);                   v3 = ROT21(v3); \
    v1 = XOR(v1,v2); v3 = XOR(v3,v0); v2 = ROT32(v2); \
  } while(0)
 
// 4-way sipHash-2-4 specialized to precomputed key and 8 byte nonces
void siphash24x4(const siphash_keys *keys, const u64 *indices, u64 * hashes) {
  const __m256i packet = _mm256_load_si256((__m256i *)indices);
  const __m256i init = _mm256_set_epi64x(
    keys->k1^0x7465646279746573ULL,
    keys->k0^0x6c7967656e657261ULL,
    keys->k1^0x646f72616e646f6dULL,
    keys->k0^0x736f6d6570736575ULL);
  __m256i v3 = _mm256_permute4x64_epi64(init, 0xFF);
  __m256i v0 = _mm256_permute4x64_epi64(init, 0x00);
  __m256i v1 = _mm256_permute4x64_epi64(init, 0x55);
  __m256i v2 = _mm256_permute4x64_epi64(init, 0xAA);

  v3 = XOR(v3,packet);
  SIPROUNDX4; SIPROUNDX4;
  v0 = XOR(v0,packet);
  v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
  SIPROUNDX4; SIPROUNDX4; SIPROUNDX4; SIPROUNDX4;
  _mm256_store_si256((__m256i *)hashes, XOR(XOR(v0,v1),XOR(v2,v3)));
}

#define SIPROUNDX8 \
  do { \
    v0 = ADD(v0,v1); v4 = ADD(v4,v5); \
    v2 = ADD(v2,v3); v6 = ADD(v6,v7); \
    v1 = ROT13(v1);  v5 = ROT13(v5); \
    v3 = ROT16(v3);  v7 = ROT16(v7); \
    v1 = XOR(v1,v0); v5 = XOR(v5,v4); \
    v3 = XOR(v3,v2); v7 = XOR(v7,v6); \
    v0 = ROT32(v0);  v4 = ROT32(v4); \
    v2 = ADD(v2,v1); v6 = ADD(v6,v5); \
    v0 = ADD(v0,v3); v4 = ADD(v4,v7); \
    v1 = ROT17(v1);  v5 = ROT17(v5); \
    v3 = ROT21(v3);  v7 = ROT21(v7); \
    v1 = XOR(v1,v2); v5 = XOR(v5,v6); \
    v3 = XOR(v3,v0); v7 = XOR(v7,v4); \
    v2 = ROT32(v2);  v6 = ROT32(v6); \
  } while(0)
 
// 8-way sipHash-2-4 specialized to precomputed key and 8 byte nonces
void siphash24x8(const siphash_keys *keys, const u64 *indices, u64 * hashes) {
  const __m256i init = _mm256_set_epi64x(
    keys->k1^0x7465646279746573ull,
    keys->k0^0x6c7967656e657261ull,
    keys->k1^0x646f72616e646f6dull,
    keys->k0^0x736f6d6570736575ull);
  const __m256i packet0 = _mm256_load_si256((__m256i *)indices);
  const __m256i packet4 = _mm256_load_si256((__m256i *)(indices+4));
  __m256i v3 = _mm256_permute4x64_epi64(init, 0xFF);
  __m256i v0 = _mm256_permute4x64_epi64(init, 0x00);
  __m256i v1 = _mm256_permute4x64_epi64(init, 0x55);
  __m256i v2 = _mm256_permute4x64_epi64(init, 0xAA);
  __m256i v7 = _mm256_permute4x64_epi64(init, 0xFF);
  __m256i v4 = _mm256_permute4x64_epi64(init, 0x00);
  __m256i v5 = _mm256_permute4x64_epi64(init, 0x55);
  __m256i v6 = _mm256_permute4x64_epi64(init, 0xAA);

  v3 = XOR(v3,packet0); v7 = XOR(v7,packet4);
  SIPROUNDX8; SIPROUNDX8;
  v0 = XOR(v0,packet0); v4 = XOR(v4,packet4);
  v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
  v6 = XOR(v6,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
  SIPROUNDX8; SIPROUNDX8; SIPROUNDX8; SIPROUNDX8;
  _mm256_store_si256((__m256i *)hashes, XOR(XOR(v0,v1),XOR(v2,v3)));
  _mm256_store_si256((__m256i *)(hashes+4), XOR(XOR(v4,v5),XOR(v6,v7)));
}
#endif // ifdef INCLUDE_SIPHASH_H
