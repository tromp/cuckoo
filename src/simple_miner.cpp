// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "cuckoo.h"

// assume EDGEBITS < 31
#define NNODES (2 * NEDGES)
#define NCUCKOO NNODES

#include "cyclebase.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <set>

typedef unsigned char u8;

class cuckoo_ctx {
  static const u32 CUCKOO_NIL = ~0;
public:
  siphash_keys sip_keys;
  edge_t easiness;
  cyclebase cb;

  cuckoo_ctx(const char* header, const u32 headerlen, const u32 nonce, edge_t easy_ness) {
    easiness = easy_ness;
    cb.alloc();
    assert(cb.cuckoo != 0);
  }

  ~cuckoo_ctx() {
    cb.freemem();
  }

  u64 bytes() {
    return (u64)(1+NNODES) * sizeof(node_t);
  }

  void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &sip_keys);
    cb.reset();
  }

  void cycle_base() {
    for (node_t nonce = 0; nonce < easiness; nonce++) {
      node_t u = sipnode(&sip_keys, nonce, 0);
      node_t v = sipnode(&sip_keys, nonce, 1);
  #ifdef SHOW
      for (unsigned j=1; j<NNODES; j++)
        if (!cb.cuckoo[j]) printf("%2d:   ",j);
        else               printf("%2d:%02d ",j,cb.cuckoo[j]);
      printf(" %x (%d,%d)\n", nonce,2*u,2*v+1);
  #endif
      cb.addedge(u, v);
    }
  }
};

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

int main(int argc, char **argv) {
  char header[HEADERLEN];
  memset(header, 0, HEADERLEN);
  int c, easipct = 50;
  u32 nonce = 0;
  u32 range = 1;
  struct timeval time0, time1;
  u32 timems;

  while ((c = getopt (argc, argv, "e:h:n:r:")) != -1) {
    switch (c) {
      case 'e':
        easipct = atoi(optarg);
        break;
      case 'h':
        memcpy(header, optarg, strlen(optarg));
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with %d%% edges, ", easipct);
  u64 easiness = easipct * (u64)NNODES / 100;
  cuckoo_ctx ctx(header, sizeof(header), nonce, easiness);
  u64 bytes = ctx.bytes();
  int unit;
  for (unit=0; bytes >= 10240; bytes>>=10,unit++) ;
  printf("using %d%cB memory at %lx.\n", bytes, " KMGT"[unit], (u64)ctx.cb.cuckoo);

  for (u32 r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.sip_keys.k0, ctx.sip_keys.k1, ctx.sip_keys.k2, ctx.sip_keys.k3);
    ctx.cycle_base();
    ctx.cb.cycles();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);
  }
}
