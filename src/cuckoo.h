// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <openssl/sha.h>
#include "siphash.h"

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 27
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
#endif

// number of edges
#define NEDGES (1ULL<<EDGEBITS)
// static const u64 NEDGES = 1ULL<<EDGEBITS;
// used to mask siphash output
#define EDGEMASK (NEDGES-1)
// static const u64 EDGEMASK = (1ULL<<EDGEBITS)-1; // NEDGES-1;
// the graph size / number of nodes
#define NNODES (2ULL<<EDGEBITS)
// static const u64 NNODES = 2ULL<<EDGEBITS;
// used to mask nodes
static const u64 NODEMASK = NNODES-1;

// generate edge endpoint in cuckoo graph without partition bit
u64 _sipnode(siphash_keys *keys, u64 nonce, u32 uorv) {
  return siphash24(keys, 2*nonce + uorv) & EDGEMASK;
}

// generate edge endpoint in cuckoo graph
u64 sipnode(siphash_keys *keys, u64 nonce, u32 uorv) {
  return _sipnode(keys, nonce, uorv) << 1 | uorv;
}

enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
const char *errstr[] = { "OK", "wrong header length", "nonce too big", "nonces not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// length of header hashed into siphash key
#ifndef HEADERLEN
#define HEADERLEN 80
#endif

void setheader(const char *header, const u32 headerlen, siphash_keys *keys) {
  char hdrkey[32];
  SHA256((unsigned char *)header, HEADERLEN, (unsigned char *)hdrkey);
  setkeys(keys, hdrkey);
}

// verify that nonces are ascending and form a cycle in header-generated graph
int verify(u64 nonces[PROOFSIZE], const char *header, const u32 headerlen) {
  if (headerlen != HEADERLEN)
    return POW_HEADER_LENGTH;
  siphash_keys keys;
  setheader(header, headerlen, &keys);
  u64 uvs[2*PROOFSIZE];
  u64 xor0=0,xor1=0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= NEDGES)
      return POW_TOO_BIG;
    if (n && nonces[n] <= nonces[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(&keys, nonces[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(&keys, nonces[n], 1);
  }
  if (xor0|xor1)              // matching endpoints imply zero xors
    return POW_NON_MATCHING;
  u32 n = 0, i = 0, j;
  do {                        // follow cycle
    for (u32 k = j = i; (k = (k+2) % (2*PROOFSIZE)) != i; ) {
      if (uvs[k] == uvs[i]) { // find other edge endpoint identical to one at i
        if (j != i)           // already found one before
          return POW_BRANCH;
        j = k;
      }
    }
    if (j == i) return POW_DEAD_END;  // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);           // must cycle back to start or we would have found branch
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}
