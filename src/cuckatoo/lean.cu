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
#include <sys/time.h> // gettimeofday
#include <set>

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint64_t u64; // save some typing

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef IDXSHIFT
#define IDXSHIFT (PART_BITS + 8)
#endif
#define MAXEDGES (NEDGES >> IDXSHIFT)

const static u64 edgeBytes = NEDGES/8;
const static u64 nodeBytes = (NEDGES>>PART_BITS)/8;
const static word_t PART_MASK = (1 << PART_BITS) - 1;
const static word_t NONPART_BITS = EDGEBITS - PART_BITS;
const static word_t NONPART_MASK = ((word_t)1 << NONPART_BITS) - 1;

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

struct trimparams {
  u16 ntrims;
  u16 blocks;
  u16 tpb;

  trimparams() {
    ntrims      = 128;
    blocks      = 128;
    tpb         = 128;
  }
};

__global__ void count_node_deg(siphash_keys &sipkeys, shrinkingset &alive, biitmap &nonleaf, u32 uorv, u32 part) {
  const int nthreads = blockDim.x * gridDim.x;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (word_t block = id*32; block < NEDGES; block += nthreads*32) {
    u32 alive32 = alive.block(block);
    for (word_t nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffs(alive32);
      nonce += ffs; alive32 >>= ffs;
      word_t u = dipnode(sipkeys, nonce, uorv);
      if ((u >> NONPART_BITS) == part) {
        nonleaf.set(u & NONPART_MASK);
      }
    }
  }
}

__global__ void kill_leaf_edges(siphash_keys &sipkeys, shrinkingset &alive, biitmap &nonleaf, u32 uorv, u32 part) {
  const int nthreads = blockDim.x * gridDim.x;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (word_t block = id*32; block < NEDGES; block += nthreads*32) {
    u32 alive32 = alive.block(block);
    for (word_t nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffs(alive32);
      nonce += ffs; alive32 >>= ffs;
      word_t u = dipnode(sipkeys, nonce, uorv);
      if ((u >> NONPART_BITS) == part && !nonleaf.test((u & NONPART_MASK) ^ 1)) {
        alive.reset(nonce);
      }
    }
  }
}

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  shrinkingset alive;
  biitmap nonleaf;
  siphash_keys sipkeys, *dipkeys;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
    checkCudaErrors(cudaMalloc((void**)&alive.bits, edgeBytes));
    checkCudaErrors(cudaMalloc((void**)&nonleaf.bits, nodeBytes));
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
  }
  ~edgetrimmer() {
    checkCudaErrors(cudaFree(nonleaf.bits));
    checkCudaErrors(cudaFree(alive.bits));
    checkCudaErrors(cudaFree(dipkeys));
    checkCudaErrors(cudaFree(dt));
    cudaDeviceReset();
  }
  void trim() {
    cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemset(alive.bits, 0, edgeBytes));
    for (u32 round=0; round < tp.ntrims; round++) {
      for (u32 part = 0; part <= PART_MASK; part++) {
        checkCudaErrors(cudaMemset(nonleaf.bits, 0, nodeBytes));
        count_node_deg<<<tp.blocks,tp.tpb>>>(*dipkeys, dt->alive, dt->nonleaf, round&1, part);
        kill_leaf_edges<<<tp.blocks,tp.tpb>>>(*dipkeys, dt->alive, dt->nonleaf, round&1, part);
      }
    }
  }
};

struct solver_ctx {
public:
  edgetrimmer trimmer;
  graph<word_t> cg;
  u64 *bits;
  proof sols[MAXSOLS];

  solver_ctx(const trimparams tp) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
    bits = new u64[NEDGES/64];
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer.sipkeys);
  }
  ~solver_ctx() {
    delete[] bits;
  }

  void findcycles() {
    cg.reset();
    for (word_t block = 0; block < NEDGES; block += 64) {
      u64 alive64 = ~bits[block/64];
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        word_t u=sipnode(&trimmer.sipkeys, nonce, 0), v=sipnode(&trimmer.sipkeys, nonce, 1);
	cg.add_compress_edge(u, v);
        if (ffs & 64) break; // can't shift by 64
      }
    }
  }

  int solve() {
    u32 timems,timems2;
    struct timeval time0, time1;
    gettimeofday(&time0, 0);

    trimmer.trim();

    cudaMemcpy(bits, trimmer.alive.bits, edgeBytes, cudaMemcpyDeviceToHost);
    u32 nedges = 0;
    for (int i = 0; i < NEDGES/64; i++)
      nedges += __builtin_popcountll(~bits[i]);
    if (nedges >= MAXEDGES) {
      printf("overloaded! exiting...");
      exit(0);
    }
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    gettimeofday(&time0, 0);

    findcycles();

    gettimeofday(&time1, 0);
    timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("%d trims %d ms %d edges %d ms total %d ms\n", trimmer.tp.ntrims, timems, nedges, timems2, timems+timems2);

    for (u32 s=0; s < cg.nsols; s++) {
      u32 j = 0, nalive = 0;
      for (word_t block = 0; block < NEDGES; block += 64) {
        u64 alive64 = ~bits[block/64];
        for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
          u32 ffs = __builtin_ffsll(alive64);
          nonce += ffs; alive64 >>= ffs;
          if (nalive++ == cg.sols[s][j]) {
            sols[s][j] = nonce;
            if (++j == PROOFSIZE)
              goto uncompressed;
          }
          if (ffs & 64) break; // can't shift by 64
        }
      }
      uncompressed: ;
    }
    return cg.nsols;
  }
};

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

typedef std::pair<word_t,word_t> edge;

#include <unistd.h>

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "sb:h:n:m:r:t:")) != -1) {
    switch (c) {
      case 's':
        printf("SYNOPSIS\n  lcuda%d [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-b blocks] [-t threads]\n", NODEBITS);
        printf("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -b %d -t %d\n", NODEBITS, device, tp.ntrims, nonce, range, tp.blocks, tp.tpb);
        exit(0);
      case 'd':
        device = atoi(optarg);
        break;
      case 'h':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        tp.ntrims = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'b':
        tp.blocks = atoi(optarg);
        break;
      case 't':
        tp.tpb = atoi(optarg);
        break;
    }
  }
  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  assert(device < nDevices);
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  assert(tp.tpb <= prop.maxThreadsPerBlock);
  u64 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 10240; dbytes>>=10,dunit++) ;
  printf("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
  cudaSetDevice(device);

  printf("Looking for %d-cycle on cuckatoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads %d per block\n", tp.ntrims, tp.blocks*tp.tpb, tp.tpb);

  solver_ctx ctx(tp);

  int edgeUnit=0, nodeUnit=0;
  u64 eb = edgeBytes, nb = nodeBytes;
  for (; eb >= 1024; eb>>=10) edgeUnit++;
  for (; nb >= 1024; nb>>=10) nodeUnit++;
  printf("Using %d%cB edge and %d%cB node memory.\n", (int)eb, " KMGT"[edgeUnit], (int)nb, " KMGT"[nodeUnit]);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.trimmer.sipkeys.k0, ctx.trimmer.sipkeys.k1, ctx.trimmer.sipkeys.k2, ctx.trimmer.sipkeys.k3);
    u32 nsols = ctx.solve();
    for (u32 s = 0; s < nsols; s++) {
      printf("Solution");
      for (u32 j = 0; j < PROOFSIZE; j++)
        printf(" %x", ctx.sols[s][j]);
      printf("\n");
      int pow_rc = verify(ctx.sols[s], &ctx.trimmer.sipkeys);
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
        sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);
  return 0;
}
