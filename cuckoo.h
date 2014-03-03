// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <openssl/sha.h>

// proof-of-work parameters
#ifndef SIZEMULT 
#define SIZEMULT 1
#endif
#ifndef SIZESHIFT 
#define SIZESHIFT 20
#endif
#ifndef EASINESS 
#define EASINESS (SIZE/2)
#endif
#ifndef PROOFSIZE 
#define PROOFSIZE 42
#endif
// algorithm parameters
#ifndef NTHREADS
#define NTHREADS 1
#endif

#define SIZE (SIZEMULT*((unsigned)1<<SIZESHIFT))
// relatively prime partition sizes
#define PARTU (SIZE/2+1)
#define PARTV (SIZE/2-1)

typedef uint64_t u64;
 
#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
 
#define SIPROUND \
  do { \
    v0 += v1; v1=ROTL(v1,13); v1 ^= v0; v0=ROTL(v0,32); \
    v2 += v3; v3=ROTL(v3,16); v3 ^= v2; \
    v0 += v3; v3=ROTL(v3,21); v3 ^= v0; \
    v2 += v1; v1=ROTL(v1,17); v1 ^= v2; v2=ROTL(v2,32); \
  } while(0)
 
// SipHash-2-4 specialized to precomputed key and 4 byte nonces
u64 siphash24( unsigned nonce, u64 v0, u64 v1, u64 v2, u64 v3) {
  u64 b = ( ( u64 )4 ) << 56 | nonce;
  v3 ^= b;
  SIPROUND; SIPROUND;
  v0 ^= b;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2  ^ v3;
}

u64 v0 = 0x736f6d6570736575ULL, v1 = 0x646f72616e646f6dULL,
    v2 = 0x6c7967656e657261ULL, v3 = 0x7465646279746573ULL;

#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))
 
// derive siphash key from header
void setheader(const char *header) {
  unsigned char hdrkey[32];
  SHA256((unsigned char *)header, strlen(header), hdrkey);
  u64 k0 = U8TO64_LE( hdrkey ); u64 k1 = U8TO64_LE( hdrkey + 8 );
  v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;
}

u64 siphash(unsigned nonce) {
  return siphash24(nonce, v0, v1, v2, v3);
}

// generate edge in cuckoo graph
void sipedge(unsigned nonce, unsigned *pu, unsigned *pv) {
  u64 sip = siphash(nonce);
  *pu = 1 +         (unsigned)(sip % PARTU);
  *pv = 1 + PARTU + (unsigned)(sip % PARTV);
}
