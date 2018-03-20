// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include "blake2.h"
#include "siphash.h"

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 29
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
#endif

#if EDGEBITS > 32
typedef u64 edge_t;
#else
typedef u32 edge_t;
#endif
#if EDGEBITS > 31
typedef u64 node_t;
#else
typedef u32 node_t;
#endif

// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK ((edge_t)NEDGES - 1)

// generate edge endpoint in cuckoo graph without partition bit
node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv) {
  return siphash24(keys, 2*edge + uorv) & EDGEMASK;
}

enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
const char *errstr[] = { "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// verify that edges are ascending and form a cycle in header-generated graph
int verify(edge_t edges[PROOFSIZE], siphash_keys *keys) {
  node_t uvs[2*PROOFSIZE];
  node_t xor0 = 0, xor1  =0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (edges[n] > EDGEMASK)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(keys, edges[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(keys, edges[n], 1);
  }
  if (xor0|xor1)              // optional check for obviously bad proofs
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

// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys) {
  char hdrkey[32];
  // SHA256((unsigned char *)header, headerlen, (unsigned char *)hdrkey);
  blake2b((void *)hdrkey, sizeof(hdrkey), (const void *)header, headerlen, 0, 0);
#ifdef SIPHASH_COMPAT
  u64 *k = (u64 *)hdrkey;
  u64 k0 = k[0];
  u64 k1 = k[1];
  printf("k0 k1 %lx %lx\n", k0, k1);
  k[0] = k0 ^ 0x736f6d6570736575ULL;
  k[1] = k1 ^ 0x646f72616e646f6dULL;
  k[2] = k0 ^ 0x6c7967656e657261ULL;
  k[3] = k1 ^ 0x7465646279746573ULL;
#endif
  setkeys(keys, hdrkey);
}

// edge endpoint in cuckoo graph with partition bit
edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}
