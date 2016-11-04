// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"
#include "siphash.h"

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

// length of header (including nonce) hashed into siphash key
#ifndef HEADERLEN
#define HEADERLEN 80
#endif

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

// generate edge endpoint in cuckoo graph without partition bit
u64  __attribute__ ((noinline)) _sipnode(siphash_keys *keys, u64 nonce, u32 uorv) {
  return siphash24(keys, 2*nonce + uorv) & NODEMASK;
}

// generate edge endpoint in cuckoo graph
u64 sipnode(siphash_keys *keys, u64 nonce, u32 uorv) {
  return _sipnode(keys, nonce, uorv) << 1 | uorv;
}

enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
const char *errstr[] = { "OK", "wrong header length", "proof too big", "proof too small", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// verify that nonces are ascending and form a cycle in header-generated graph
int verify(u64 nonces[PROOFSIZE], const char *headernonce, const u32 headerlen) {
  if (headerlen != HEADERLEN)
    return POW_HEADER_LENGTH;
  siphash_keys keys;
  setheader(&keys, headernonce);
  u64 uvs[2*PROOFSIZE];
  u64 xor0=0,xor1=0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= HALFSIZE)
      return POW_TOO_BIG;
    if (n && nonces[n] <= nonces[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(&keys, nonces[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(&keys, nonces[n], 1);
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
    } if (j == i) return POW_DEAD_END;              // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}
