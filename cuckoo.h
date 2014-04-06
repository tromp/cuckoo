// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

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
typedef u64 nonce_t;
#ifdef HUGEFAST
typedef u32 node_t;
#else
typedef u64 node_t;
#endif

typedef struct {
  u64 v[4];
} siphash_ctx;
 
#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))
 
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

void sipedge(siphash_ctx *ctx, nonce_t nonce, node_t *pu, node_t *pv) {
  *pu = siphash24(ctx, 2*nonce  ) & NODEMASK;
  *pv = siphash24(ctx, 2*nonce+1) & NODEMASK;
}

// generate edge endpoint in cuckoo graph
node_t sipedge_u(siphash_ctx *ctx, nonce_t nonce) {
  return siphash24(ctx, 2*nonce  ) & NODEMASK;
}

// generate edge endpoint in cuckoo graph
node_t sipedge_v(siphash_ctx *ctx, nonce_t nonce) {
  return siphash24(ctx, 2*nonce+1) & NODEMASK;
}

// verify that (ascending) nonces, all less than easiness, form a cycle in header-generated graph
int verify(nonce_t nonces[PROOFSIZE], const char *header, u64 easiness) {
  siphash_ctx ctx;
  setheader(&ctx, header);
  node_t us[PROOFSIZE], vs[PROOFSIZE];
  unsigned i = 0, n;
  for (n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= easiness || (n && nonces[n] <= nonces[n-1]))
      return 0;
    sipedge(&ctx, nonces[n], &us[n], &vs[n]);
  }
  do {  // follow cycle until we return to i==0; n edges left to visit
    unsigned j = i;
    for (unsigned k = 0; k < PROOFSIZE; k++) // find unique other j with same vs[j]
      if (k != i && vs[k] == vs[i]) {
        if (j != i)
          return 0;
        j = k;
    }
    if (j == i)
      return 0;
    i = j;
    for (unsigned k = 0; k < PROOFSIZE; k++) // find unique other i with same us[i]
      if (k != j && us[k] == us[j]) {
        if (i != j)
          return 0;
        i = k;
    }
    if (i == j)
      return 0;
    n -= 2;
  } while (i);
  return n == 0;
}
