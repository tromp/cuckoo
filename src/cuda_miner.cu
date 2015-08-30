// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2015 John Tromp

// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

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
#if SIZESHIFT < 32
typedef u32 nonce_t;
typedef u32 node_t;
#else
typedef u64 nonce_t;
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
 
// generate edge endpoint in cuckoo graph
__device__ node_t sipnode(siphash_ctx *ctx, nonce_t nce, u32 uorv) {
  u64 nonce = 2*nce + uorv;
  u64 v0 = ctx->v[0], v1 = ctx->v[1], v2 = ctx->v[2], v3 = ctx->v[3] ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1 ^ v2  ^ v3) & NODEMASK;
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
typedef u32 au32;
typedef u64 au64;
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
  au32 *bits;
  __device__ void reset() {
    memset(bits, 0, TWICE_WORDS * sizeof(au32));
  }
  __device__ void set(node_t u) {
    node_t idx = u/16;
    u32 bit = 1 << (2 * (u%16));
    u32 old = atomicOr(&bits[idx], bit);
    u32 bit2 = bit<<1;
    if ((old & (bit2|bit)) == bit) atomicOr(&bits[idx], bit2);
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
  au64 *cuckoo;

  cuckoo_hash() {
    assert(cuckoo = (au64 *)calloc(CUCKOO_SIZE, sizeof(au64)));
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

#define LOGNBUCKETS	0
#define NBUCKETS	1
#define BUCKETSHIFT	(SIZESHIFT-1)
#define NONCESHIFT	(SIZESHIFT-1 - PART_BITS)
#define NODEPARTMASK	(NODEMASK >> PART_BITS)
#define NONCETRUNC	(1L << (64 - NONCESHIFT))

__global__ void count_node_deg(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (nonce_t block = id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);
    for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
      if (alive32 & 1) {
        node_t u = sipnode(&ctx->sip_ctx, nonce, uorv);
        if ((u & PART_MASK) == part) {
          nonleaf.set(u >> PART_BITS);
        }
      }
    }
  }
}

__global__ void kill_leaf_edges(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (nonce_t block = id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    u32 alive32 = alive.block(block);
    for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
      if (alive32 & 1) {
        node_t u = sipnode(&ctx->sip_ctx, nonce, uorv);
        if ((u & PART_MASK) == part) {
          if (!nonleaf.test(u >> PART_BITS)) {
            alive.reset(nonce);
          }
        }
      }
    }
  }
}

#if 0
u32 path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (nu == ~0)
        printf("maximum path length exceeded\n");
      else printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu] = u;
  }
  return nu;
}

typedef std::pair<node_t,node_t> edge;

void solution(cuckoo_ctx *ctx, node_t *us, u32 nu, node_t *vs, u32 nv) {
  std::set<edge> cycle;
  u32 n;
  cycle.insert(edge(*us, *vs));
  while (nu--)
    cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
  while (nv--)
    cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  for (nonce_t nonce = n = 0; nonce < HALFSIZE; nonce++)
    if (ctx->alive.test(nonce)) {
      edge e(sipnode(&ctx->sip_ctx, nonce, 0), HALFSIZE+sipnode(&ctx->sip_ctx, nonce, 1));
      if (cycle.find(e) != cycle.end()) {
        printf(" %lx", (long)nonce);
        if (PROOFSIZE > 2)
          cycle.erase(e);
      }
    }
  assert(n==PROOFSIZE);
  printf("\n");
}
#endif

#include <unistd.h>

int main(int argc, char **argv) {
  int nthreads = 1;
  int maxsols  = 8;
  int ntrims   = 1 + (PART_BITS+3)*(PART_BITS+4)/2;
  const char *header = "";
  int c;
  while ((c = getopt (argc, argv, "h:m:n:t:")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
      case 'm':
        maxsols = atoi(optarg);
        break;
      case 'n':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
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

  for (u32 round=0; round < ntrims; round++) {
    for (u32 uorv = 0; uorv < 2; uorv++) {
      for (u32 part = 0; part <= PART_MASK; part++) {
        checkCudaErrors(cudaMemset(ctx.nonleaf.bits, 0, nodeBytes));
        count_node_deg<<<nthreads,1>>>(device_ctx,uorv,part);
        kill_leaf_edges<<<nthreads,1>>>(device_ctx,uorv,part);
        printf("round %d part %c%d\n", round, "UV"[uorv], part);
      }
    }
  }

  u32 *bits;
  assert(bits = (u32 *)calloc(HALFSIZE/32, sizeof(u32)));
  cudaMemcpy(bits, ctx.alive.bits, (HALFSIZE/32) * sizeof(u32), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaFree(ctx.alive.bits));
  checkCudaErrors(cudaFree(ctx.nonleaf.bits));

  u32 cnt = 0;
  for (nonce_t nonce = 0; nonce < HALFSIZE; nonce++)
    cnt += ((bits[nonce/32] >> (nonce%32)) & 1) ^ 1;

  u32 load = (u32)(100L * cnt / CUCKOO_SIZE);
  printf("final load %d%%\n", load);

#if 0
  if (tp->id == 0) {
    load = (u32)(100 * alive.count() / CUCKOO_SIZE);
    if (load >= 90) {
      printf("overloaded! exiting...");
      exit(0);
    }
    delete ctx->nonleaf;
    ctx->nonleaf = 0;
    ctx->cuckoo = new cuckoo_hash();
  }
  barrier(&ctx->barry);
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (nonce_t block = tp->id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    for (nonce_t nonce = block; nonce < block+32 && nonce < HALFSIZE; nonce++) {
      if (alive.test(nonce)) {
        node_t u0 = sipnode(ctx, nonce, 0), v0 = sipnode(ctx, nonce, 1) + HALFSIZE;  // make v's different from u's
        node_t u = cuckoo[us[0] = u0], v = cuckoo[vs[0] = v0];
        u32 nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100L/HALFSIZE));
          if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
            solution(ctx, us, nu, vs, nv);
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

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  for (int t = 0; t < nthreads; t++) {
    threads[t].id = t;
    threads[t].ctx = &ctx;
    assert(pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]) == 0);
  }
  for (int t = 0; t < nthreads; t++)
    assert(pthread_join(threads[t].thread, NULL) == 0);
  free(threads);

  for (unsigned s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %lx", (long)ctx.sols[s][i]);
    printf("\n");
  }
#endif
  return 0;
}
