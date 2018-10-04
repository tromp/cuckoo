// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include <stdint.h>
#include <string.h>
#include "cuckatoo.h"
#include "../crypto/siphash.cuh"
#include "graph.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <set>

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

typedef uint64_t u64; // save some typing

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef REDUCE_NONCES
// reduce number of edges this much under MAXEDGES
// so that number of nodepairs will remain below MAXEDGES as well
#define REDUCE_NONCES 7/8
#endif

#ifndef IDXSHIFT
#define IDXSHIFT (PART_BITS + 8)
#endif
#define MAXEDGES (NEDGES >> IDXSHIFT)

const static u32 PART_MASK = (1 << PART_BITS) - 1;
const static u32 NONPART_BITS = EDGEBITS - PART_BITS;
const static u32 NONPART_MASK = (1U << NONPART_BITS) - 1U;

#define NODEBITS (EDGEBITS + 1)
#define NNODES (2 * NEDGES)
#define NODEMASK (NNODES-1)

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  u32 *bits;
  __device__ void reset(word_t n) {
    bits[n/32] |= 1 << (n%32);
  }
  __device__ bool test(word_t n) const {
    return !((bits[n/32] >> (n%32)) & 1);
  }
  __device__ u32 block(word_t n) const {
    return ~bits[n/32];
  }
};

class biitmap {
public:
  u32 *bits;
  __device__ void set(word_t n) {
    atomicOr(&bits[n/32], 1 << (n%32));
  }
  __device__ bool test(word_t n) const {
    return (bits[n/32] >> (n%32)) & 1;
  }
};

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

class cuckoo_ctx {
public:
  siphash_keys sip_keys;
  shrinkingset alive;
  biitmap nonleaf;
  graph<word_t> cg;
  int nthreads;

  cuckoo_ctx(const u32 n_threads) : cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
    nthreads = n_threads;
  }
  void setheadernonce(char* headernonce, const u32 nonce) {
    ((u32 *)headernonce)[HEADERLEN/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, HEADERLEN, &sip_keys);
  }
};

__global__ void count_node_deg(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  biitmap &nonleaf = ctx->nonleaf;
  siphash_keys sip_keys = ctx->sip_keys; // local copy sip context; 2.5% speed gain
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (word_t block = id*32; block < NEDGES; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);
    for (word_t nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffs(alive32);
      nonce += ffs; alive32 >>= ffs;
      word_t u = dipnode(sip_keys, nonce, uorv);
      if ((u >> NONPART_BITS) == part) {
        nonleaf.set(u & NONPART_MASK);
      }
    }
  }
}

__global__ void kill_leaf_edges(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  biitmap &nonleaf = ctx->nonleaf;
  siphash_keys sip_keys = ctx->sip_keys;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (word_t block = id*32; block < NEDGES; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);
    for (word_t nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffs(alive32);
      nonce += ffs; alive32 >>= ffs;
      word_t u = dipnode(sip_keys, nonce, uorv);
      if ((u >> NONPART_BITS) == part && !nonleaf.test((u & NONPART_MASK) ^ 1)) {
        alive.reset(nonce);
      }
    }
  }
}

typedef std::pair<word_t,word_t> edge;

#include <unistd.h>

int main(int argc, char **argv) {
  int nthreads = 16384;
  int trims   = 32;
  int tpb = 0;
  int nonce = 0;
  int range = 1;
  const char *header = "";
  int c;
  while ((c = getopt (argc, argv, "h:n:m:r:t:p:")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        trims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
      case 'p':
        tpb = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
    }
  }
  if (!tpb) // if not set, then default threads per block to roughly square root of threads
    for (tpb = 1; tpb*tpb < nthreads; tpb *= 2) ;

  printf("Looking for %d-cycle on cuckatoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads %d per block\n", trims, nthreads, tpb);

  cuckoo_ctx ctx(nthreads);

  char headernonce[HEADERLEN];
  u32 hdrlen = strlen(header);
  memcpy(headernonce, header, hdrlen);
  memset(headernonce+hdrlen, 0, sizeof(headernonce)-hdrlen);

  u64 edgeBytes = NEDGES/8, nodeBytes = (NEDGES>>PART_BITS)/8;
  checkCudaErrors(cudaMalloc((void**)&ctx.alive.bits, edgeBytes));
  checkCudaErrors(cudaMalloc((void**)&ctx.nonleaf.bits, nodeBytes));

  int edgeUnit=0, nodeUnit=0;
  u64 eb = edgeBytes, nb = nodeBytes;
  for (; eb >= 1024; eb>>=10) edgeUnit++;
  for (; nb >= 1024; nb>>=10) nodeUnit++;
  printf("Using %d%cB edge and %d%cB node memory.\n",
     (int)eb, " KMGT"[edgeUnit], (int)nb, " KMGT"[nodeUnit]);

  cuckoo_ctx *device_ctx;
  checkCudaErrors(cudaMalloc((void**)&device_ctx, sizeof(cuckoo_ctx)));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  for (int r = 0; r < range; r++) {
    cudaEventRecord(start, NULL);
    checkCudaErrors(cudaMemset(ctx.alive.bits, 0, edgeBytes));
    ctx.setheadernonce(headernonce, nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.sip_keys.k0, ctx.sip_keys.k1, ctx.sip_keys.k2, ctx.sip_keys.k3);
    cudaMemcpy(device_ctx, &ctx, sizeof(cuckoo_ctx), cudaMemcpyHostToDevice);
    for (u32 round=0; round < trims; round++) {
      for (u32 uorv = 0; uorv < 2; uorv++) {
        for (u32 part = 0; part <= PART_MASK; part++) {
          checkCudaErrors(cudaMemset(ctx.nonleaf.bits, 0, nodeBytes));
          count_node_deg<<<nthreads/tpb,tpb >>>(device_ctx, uorv, part);
          kill_leaf_edges<<<nthreads/tpb,tpb >>>(device_ctx, uorv, part);
        }
      }
    }
  
    u64 *bits;
    bits = (u64 *)calloc(NEDGES/64, sizeof(u64));
    assert(bits != 0);
    cudaMemcpy(bits, ctx.alive.bits, (NEDGES/64) * sizeof(u64), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    u32 size = 0;
    for (int i = 0; i < NEDGES/64; i++)
      size += __builtin_popcountll(~bits[i]);
    printf("%d trims completed in %.3f seconds final size %d\n", trims, duration / 1000.0f, size);
  
    if (size >= MAXEDGES) {
      printf("overloaded! exiting...");
      exit(0);
    }
  
    ctx.cg.reset();
    for (word_t block = 0; block < NEDGES; block += 64) {
      u64 alive64 = ~bits[block/64];
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        word_t u=sipnode(&ctx.sip_keys, nonce, 0), v=sipnode(&ctx.sip_keys, nonce, 1);
	ctx.cg.add_compress_edge(u, v);
        if (ffs & 64) break; // can't shift by 64
      }
    }
    for (u32 s=0; s < ctx.cg.nsols; s++) {
      printf("Solution");
      u32 j = 0, nalive = 0;
      for (word_t block = 0; block < NEDGES; block += 64) {
        u64 alive64 = ~bits[block/64];
        for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
          u32 ffs = __builtin_ffsll(alive64);
          nonce += ffs; alive64 >>= ffs;
          if (nalive++ == ctx.cg.sols[s][j]) {
            printf(" %x", ctx.cg.sols[s][j] = nonce);
            if (++j == PROOFSIZE)
              goto uncompressed;
          }
          if (ffs & 64) break; // can't shift by 64
        }
      }
  uncompressed:
      printf("\n");
      int pow_rc = verify(ctx.cg.sols[s], &ctx.sip_keys);
      if (pow_rc == POW_OK) {
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)ctx.cg.sols[s], sizeof(ctx.cg.sols[0]), 0, 0);
        for (int i=0; i<32; i++)
          printf("%02x", cyclehash[i]);
        printf("\n");
      } else {
        printf("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
  }
  checkCudaErrors(cudaFree(ctx.alive.bits));
  checkCudaErrors(cudaFree(ctx.nonleaf.bits));
  return 0;
}
