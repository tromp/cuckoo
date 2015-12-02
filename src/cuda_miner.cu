// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2015 John Tromp

// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html



#include <stdint.h>
#include <string.h>
#include <time.h>
#include "cuckoo.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if SIZESHIFT <= 32
typedef u32 nonce_t;
typedef u32 node_t;
#else
typedef u64 nonce_t;
typedef u64 node_t;
#endif
#include <openssl/sha.h>

__shared__ siphash_ctx shared_siphash_ctx;

// d(evice s)ipnode
__device__ node_t dipnode(nonce_t nce, u32 uorv) {
#if (__CUDA_ARCH__  < 320)
	u64 nonce = 2*nce + uorv;
	u64 v0 = shared_siphash_ctx.v[0], v1 = shared_siphash_ctx.v[1], v2 = shared_siphash_ctx.v[2], v3 = shared_siphash_ctx.v[3] ^ nonce;

	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return (v0 ^ v1 ^ v2  ^ v3) & NODEMASK;
#else
	uint2 nonce = vectorize((nce << 1) + uorv);
	uint2 v0, v1, v2, v3;
	v0 = shared_siphash_ctx.vv[0], v1 = shared_siphash_ctx.vv[1], v2 = shared_siphash_ctx.vv[2], v3 = shared_siphash_ctx.vv[3] ^ nonce;

	SIPROUND2; SIPROUND2;
	v0 ^= nonce;
	v2 ^= vectorize(0xff);
	SIPROUND2; SIPROUND2; SIPROUND2; SIPROUND2;
	return devectorize(v0 ^ v1 ^ v2  ^ v3) & NODEMASK;
#endif
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <set>

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two, making twice_set the
// same size as shrinkingset at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) == sizeof(twice_set), so
// CUCKOO_SIZE * sizeof(u64) == TWICE_WORDS * sizeof(u32)
// CUCKOO_SIZE * 2 == TWICE_WORDS
// (SIZE >> IDXSHIFT) * 2 == 2 * ONCE_BITS / 32
// SIZE >> IDXSHIFT == HALFSIZE >> PART_BITS >> 5
// IDXSHIFT == 1 + PART_BITS + 5
#define IDXSHIFT (PART_BITS + 6)
#endif
// grow with cube root of size, hardly affected by trimming
#define MAXPATHLEN (8 << (SIZESHIFT/3))

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  u32 *bits;
  __device__ void reset(nonce_t n) {
    bits[n/32] |= 1 << (n%32);
  }
  __device__ bool test(node_t n) const {
    return !((bits[n/32] >> (n%32)) & 1);
  }
  __device__ u32 block(node_t n) const {
    return ~bits[n/32];
  }
};

#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS (HALFSIZE >> PART_BITS)
#define TWICE_WORDS ((2 * ONCE_BITS) / 32)

class twice_set {
public:
  u32 *bits;
  __device__ void reset() {
    memset(bits, 0, TWICE_WORDS * sizeof(u32));
  }
  __device__ void set(node_t u) {
	  node_t idx = u / 16;
	  u32 bit = 1 << (2 * (u % 16));
	  //u32 old = atomicOr(&bits[idx], bit); // ~1% slower than 2 lines below
	  u32 old = bits[idx];
	  bits[idx] |= bit;
	  //
	  u32 bit2 = bit << 1;
	  if ((old & (bit2 | bit)) == bit) atomicOr(&bits[idx], bit2); //somehow this is faster
  } 
  __device__ u32 test(node_t u) const {
    return (bits[u/16] >> (2 * (u%16))) & 2;
  }
};

#define CUCKOO_SIZE (SIZE >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by SIZESHIFT
#define KEYBITS (64-SIZESHIFT)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

class cuckoo_hash {
public:
  u64 *cuckoo;

  cuckoo_hash() {
    cuckoo = (u64 *)calloc(CUCKOO_SIZE, sizeof(u64));
    assert(cuckoo != 0);
  }
  ~cuckoo_hash() {
    free(cuckoo);
  }
  void set(node_t u, node_t v) {
    u64 niew = (u64)u << SIZESHIFT | v;
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#ifdef ATOMIC
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if ((old >> SIZESHIFT) == (u & KEYMASK)) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
#else
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> SIZESHIFT) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
#endif
        return;
      }
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#ifdef ATOMIC
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
#else
      u64 cu = cuckoo[ui];
#endif
      if (!cu)
        return 0;
      if ((cu >> SIZESHIFT) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (node_t)(cu & (SIZE-1));
      }
    }
  }
};

class cuckoo_ctx {
public:
  siphash_ctx sip_ctx;
  shrinkingset alive;
  twice_set nonleaf;
  int nthreads;

  cuckoo_ctx(const char* header, u32 n_threads) {
    setheader(&sip_ctx, header);
    nthreads = n_threads;
  }
};

#define TPB 128

__global__ void
__launch_bounds__(TPB, 1)
count_node_deg(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  
  if (threadIdx.x == 0) {
	  shared_siphash_ctx.v[0] = ctx->sip_ctx.v[0];
	  shared_siphash_ctx.v[1] = ctx->sip_ctx.v[1];
	  shared_siphash_ctx.v[2] = ctx->sip_ctx.v[2];
	  shared_siphash_ctx.v[3] = ctx->sip_ctx.v[3];
  }
  __syncthreads();
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (nonce_t block = id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);

    for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
      if (alive32 & 1) {
        node_t u = dipnode(nonce, uorv);
        if ((u & PART_MASK) == part) {
          nonleaf.set(u >> PART_BITS);
        }
      }
    }
  }
}

__global__ void
__launch_bounds__(TPB, 1)
kill_leaf_edges(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  
  if (threadIdx.x == 0) {
	  shared_siphash_ctx.v[0] = ctx->sip_ctx.v[0];
	  shared_siphash_ctx.v[1] = ctx->sip_ctx.v[1];
	  shared_siphash_ctx.v[2] = ctx->sip_ctx.v[2];
	  shared_siphash_ctx.v[3] = ctx->sip_ctx.v[3];
  }
  __syncthreads();
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (nonce_t block = id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);
    for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
      if (alive32 & 1) {
        node_t u = dipnode(nonce, uorv);
        if ((u & PART_MASK) == part) {
          if (!nonleaf.test(u >> PART_BITS)) {
            alive.reset(nonce);
          }
        }
      }
    }
  }
}

u32 path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (nu == ~0)
        printf("maximum path length exceeded\n");
      else printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
      exit(0);
    }
    us[nu] = u;
  }
  return nu;
}

typedef std::pair<node_t,node_t> edge;

#ifndef WIN32
#include <unistd.h>
#else
#include "getopt/getopt.h"
#endif

int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 1 + (PART_BITS+3)*(PART_BITS+4)/2;
  const char *header = "";
  bool profiling = false;
  int c;
  while ((c = getopt (argc, argv, "h:m:n:t:p")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
      case 'n':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
	  case 'p':
		  profiling = true;
		  break;
    }
  }
  printf("Looking for %d-cycle on cuckoo%d(\"%s\") with 50%% edges, %d trims, %d threads\n",
               PROOFSIZE, SIZESHIFT, header, ntrims, nthreads);
  u64 edgeBytes = HALFSIZE/8, nodeBytes = TWICE_WORDS*sizeof(u32);

  cuckoo_ctx ctx(header, nthreads);
  checkCudaErrors(cudaMalloc((void**)&ctx.alive.bits, edgeBytes));
  checkCudaErrors(cudaMemset(ctx.alive.bits, 0, edgeBytes));
  checkCudaErrors(cudaMalloc((void**)&ctx.nonleaf.bits, nodeBytes));

  int edgeUnit=0, nodeUnit=0;
  u64 eb = edgeBytes, nb = nodeBytes;
  for (; eb >= 1024; eb>>=10) edgeUnit++;
  for (; nb >= 1024; nb>>=10) nodeUnit++;
  printf("Using %d%cB edge and %d%cB node memory.\n",
     (int)eb, " KMGT"[edgeUnit], (int)nb, " KMGT"[nodeUnit]);

  cuckoo_ctx *device_ctx;
  checkCudaErrors(cudaMalloc((void**)&device_ctx, sizeof(cuckoo_ctx)));
  cudaMemcpy(device_ctx, &ctx, sizeof(cuckoo_ctx), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;

  if (profiling) {  
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));
	  cudaEventRecord(start, nullptr);
  }

  for (u32 round=0; round < ntrims; round++) {
    for (u32 uorv = 0; uorv < 2; uorv++) {
      for (u32 part = 0; part <= PART_MASK; part++) {
        checkCudaErrors(cudaMemset(ctx.nonleaf.bits, 0, nodeBytes));
		count_node_deg << <nthreads / TPB, TPB >> >(device_ctx, uorv, part);
		kill_leaf_edges << <nthreads / TPB, TPB >> >(device_ctx, uorv, part);
		cudaDeviceSynchronize();
      }
    }
  }
  if (profiling) {
	  cudaEventRecord(stop, nullptr);
	  cudaEventSynchronize(stop);

	  float duration;
	  cudaEventElapsedTime(&duration, start, stop);
	  printf("%d rounds completed in %.2f seconds.\n", ntrims, duration / 1000.0f);
  }
  u32 *bits;
  bits = (u32 *)calloc(HALFSIZE/32, sizeof(u32));
  assert(bits != 0);
  cudaMemcpy(bits, ctx.alive.bits, (HALFSIZE/32) * sizeof(u32), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaFree(ctx.alive.bits));
  checkCudaErrors(cudaFree(ctx.nonleaf.bits));

  u32 cnt = 0;
  for (int i = 0; i < HALFSIZE/32; i++) {
    for (u32 b = ~bits[i]; b; b &= b-1)
      cnt++;
  }
  u32 load = (u32)(100L * cnt / CUCKOO_SIZE);
  printf("final load %d%%\n", load);

  if (load >= 90) {
    printf("overloaded! exiting...");
    exit(0);
  }

  cuckoo_hash &cuckoo = *(new cuckoo_hash());
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (nonce_t block = 0; block < HALFSIZE; block += 32) {
    for (nonce_t nonce = block; nonce < block+32 && nonce < HALFSIZE; nonce++) {
      if (!(bits[nonce/32] >> (nonce%32) & 1)) {
        node_t u0=sipnode(&ctx.sip_ctx, nonce, 0), v0=sipnode(&ctx.sip_ctx, nonce, 1);
        if (u0 == 0) // ignore vertex 0 so it can be used as nil for cuckoo[]
          continue;
        node_t u = cuckoo[us[0] = u0], v = cuckoo[vs[0] = v0];
        u32 nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("% 4d-cycle found at %d:%d%%\n", len, 0, (u32)(nonce*100L/HALFSIZE));
          if (len == PROOFSIZE) {
            printf("Solution");
            std::set<edge> cycle;
            u32 n;
            cycle.insert(edge(*us, *vs));
            while (nu--)
              cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
            while (nv--)
              cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
            for (nonce_t nce = n = 0; nce < HALFSIZE; nce++)
              if (!(bits[nce/32] >> (nce%32) & 1)) {
                edge e(sipnode(&ctx.sip_ctx, nce, 0), sipnode(&ctx.sip_ctx, nce, 1));
                if (cycle.find(e) != cycle.end()) {
                  printf(" %lx", nonce);
                  if (PROOFSIZE > 2)
                    cycle.erase(e);
                  n++;
                }
              }
            assert(n==PROOFSIZE);
            printf("\n");
          }
          continue;
        }
        if (nu < nv) {
          while (nu--)
            cuckoo.set(us[nu+1], us[nu]);
          cuckoo.set(u0, v0);
        } else {
          while (nv--)
            cuckoo.set(vs[nv+1], vs[nv]);
          cuckoo.set(v0, u0);
        }
      }
    }
  }
  return 0;
}
