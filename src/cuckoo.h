// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2015 John Tromp

#include <stdint.h>
#include <string.h>
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"

// proof-of-work parameters
#ifndef SIZESHIFT 
#define SIZESHIFT 25
#endif
#ifndef PROOFSIZE
#define PROOFSIZE 42
#endif

#define SIZE (1UL<<SIZESHIFT)
#define HALFSIZE (SIZE/2)
#define NODEMASK (HALFSIZE-1)

typedef uint32_t u32;
typedef uint64_t u64;

typedef struct {
  u64 v[4];
} siphash_ctx;
 
#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))

#ifndef SHA256
#define SHA256(d, n, md) do { \
    SHA256_CTX c; \
    SHA256_Init(&c); \
    SHA256_Update(&c, (uint8_t *)d, n); \
    SHA256_Final((uint8_t *)md, &c); \
  } while (0)
#endif
 
// derive siphash key from header
void setheader(siphash_ctx *ctx, const char *header) {
  unsigned char hdrkey[32];
  SHA256((unsigned char *)header, strlen(header), hdrkey);
  u64 k0 = U8TO64_LE(hdrkey);
  u64 k1 = U8TO64_LE(hdrkey+8);
  ctx->v[0] = k0 ^ 0x736f6d6570736575ULL;
  ctx->v[1] = k1 ^ 0x646f72616e646f6dULL;
  ctx->v[2] = k0 ^ 0x6c7967656e657261ULL;
  ctx->v[3] = k1 ^ 0x7465646279746573ULL;
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
u64 siphash24(siphash_ctx *ctx, u64 nonce) {
  u64 v0 = ctx->v[0], v1 = ctx->v[1], v2 = ctx->v[2], v3 = ctx->v[3] ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2  ^ v3;
}

// generate edge endpoint in cuckoo graph without partition bit
u64 _sipnode(siphash_ctx *ctx, u64 nonce, u32 uorv) {
  return (siphash24(ctx, 2*nonce + uorv) & NODEMASK);
}

// generate edge endpoint in cuckoo graph
u64 sipnode(siphash_ctx *ctx, u64 nonce, u32 uorv) {
  return (siphash24(ctx, 2*nonce + uorv) & NODEMASK) << 1 | uorv;
}

// verify that (ascending) nonces, all less than easiness, form a cycle in header-generated graph
int verify(u64 nonces[PROOFSIZE], const char *header, u64 easiness) {
  siphash_ctx ctx;
  setheader(&ctx, header);
  u64 uvs[2*PROOFSIZE];
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= easiness || (n && nonces[n] <= nonces[n-1]))
      return 0;
    uvs[2*n  ] = sipnode(&ctx, nonces[n], 0);
    uvs[2*n+1] = sipnode(&ctx, nonces[n], 1);
  }
  u32 i = 0;
  for (u32 n = PROOFSIZE; n; ) { // follow cycle for n more steps
    u32 j = i;
    for (u32 k = i&1; k < 2*PROOFSIZE; k += 2) // find unique other j with same parity and uvs[j]
      if (k != i && uvs[k] == uvs[i]) {
        if (j != i)
          return 0; // more than 2 occurences
        j = k;
    }
    if (j == i)
      return 0; // no other occurence
    i = j^1;
    if (--n && i == 0) // don't return to 0 too soon
      return 0;
  }
  return i == 0;
}
