// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"

// proof-of-work parameters
#ifndef SIZESHIFT 
// the main parameter is the 2log of the graph size,
// which is the size in bits of the node identifiers
#define SIZESHIFT 25
#endif
#ifndef PROOFSIZE
// the next most important parameter is (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
#endif

// the graph size / number of nodes
#define SIZE (1ULL<<SIZESHIFT)
// number of nodes in one partition (eg. all even nodes)
#define HALFSIZE (SIZE/2)
// used to mask siphash output
#define NODEMASK (HALFSIZE-1)

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

// siphash uses a state of four 64-bit words,
typedef union {
  u64 v[4];
// or four 32-bit-word-pairs for the benefit of CUDA funnel shifter
#ifdef __CUDACC__
  uint2 v2[4];
#endif
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
    SHA256_Update(&c, d, n); \
    SHA256_Final(md, &c); \
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

enum verify_code { POW_OK, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};

// verify that nonces are ascending and form a cycle in header-generated graph
int verify(u64 nonces[PROOFSIZE], const char *header) {
  siphash_ctx ctx;
  setheader(&ctx, header);
  u64 uvs[2*PROOFSIZE];
  u64 xor0=0,xor1=0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= HALFSIZE)
      return POW_TOO_BIG;
    if (n && nonces[n] <= nonces[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(&ctx, nonces[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(&ctx, nonces[n], 1);
  }
  if (xor0|xor1)                        // matching endpoints imply zero xors
    return POW_NON_MATCHING;
  u32 n = 0, i = 0;
  do { // follow cycle
    u32 j = i;                          // indicate matching endpoint not yet found
    for (u32 k = i&1; k < 2*PROOFSIZE; k += 2) {
      if (uvs[k] == uvs[i] && k != i) { // find unique other edge endpoint j identical to i
        if (j != i)
          return POW_BRANCH;            // not so unique
        j = k;
      }
    }
    if (j == i)
      return POW_DEAD_END;              // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}
