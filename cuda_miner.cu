// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

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
typedef u64 nonce_t;
typedef u64 node_t;

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

__device__ void sipedge(siphash_ctx *ctx, nonce_t nonce, node_t *pu, node_t *pv) {
  *pu = sipnode(ctx, nonce, 0);
  *pv = sipnode(ctx, nonce, 1);
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
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
// L3 cache should exceed NBUCKETS buckets of BUCKETSIZE uint_64_t (0.5MB below)
#ifndef LOGNBUCKETS
#define LOGNBUCKETS	8
#endif
#ifndef BUCKETSIZE
#define BUCKETSIZE	256
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

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  std::vector<u32> bits;
  std::vector<u64> cnt;

  __device__ shrinkingset(u32 nthreads) {
    nonce_t nwords = HALFSIZE/32;
    bits.resize(nwords);
    cnt.resize(nthreads);
    cnt[0] = HALFSIZE;
  }
  __device__ u64 count() const {
    u64 sum = 0L;
    for (u32 i=0; i<cnt.size(); i++)
      sum += cnt[i];
    return sum;
  }
  __device__ void reset(nonce_t n, u32 thread) {
    bits[n/32] |= 1 << (n%32);
    cnt[thread]--;
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

  __device__ twice_set() {
    assert(bits = (au32 *)calloc(TWICE_WORDS, sizeof(au32)));
  }
  __device__ void reset() {
    memset(bits, 0, TWICE_WORDS*sizeof(au32));
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
  __device__ ~twice_set() {
    free(bits);
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
    u64 old, niew = (u64)u << SIZESHIFT | v;
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
  shrinkingset *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;
  nonce_t (*sols)[PROOFSIZE];
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  pthread_barrier_t barry;

  cuckoo_ctx(const char* header, u32 n_threads, u32 n_trims, u32 max_sols) {
    setheader(&sip_ctx, header);
    nthreads = n_threads;
    alive = new shrinkingset(nthreads);
    cuckoo = 0;
    nonleaf = new twice_set;
    ntrims = n_trims;
    assert(pthread_barrier_init(&barry, NULL, nthreads) == 0);
    assert(sols = (nonce_t (*)[PROOFSIZE])calloc(maxsols = max_sols, PROOFSIZE*sizeof(nonce_t)));
    nsols = 0;
  }
  ~cuckoo_ctx() {
    delete alive;
    if (nonleaf)
      delete nonleaf;
    if (cuckoo)
      delete cuckoo;
  }
};

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

#define NBUCKETS	(1 << LOGNBUCKETS)
#define BUCKETSHIFT	(SIZESHIFT-1 - LOGNBUCKETS)
#define NONCESHIFT	(SIZESHIFT-1 - PART_BITS)
#define NODEPARTMASK	(NODEMASK >> PART_BITS)
#define NONCETRUNC	(1L << (64 - NONCESHIFT))

void trim_edges(thread_ctx *tp, u32 round) {
  cuckoo_ctx *ctx = tp->ctx;
  u64 (* buckets)[BUCKETSIZE] = tp->buckets;
  shrinkingset *alive = ctx->alive;
  twice_set *nonleaf = ctx->nonleaf;
  u32 bucketsizes[NBUCKETS];

  for (u32 uorv = 0; uorv < 2; uorv++) {
    for (u32 part = 0; part <= PART_MASK; part++) {
      if (tp->id == 0)
        nonleaf->reset();
      barrier(&ctx->barry);
      for (u32 qkill = 0; qkill < 2; qkill++) {
        for (u32 b=0; b < NBUCKETS; b++)
          bucketsizes[b] = 0;
        for (nonce_t block = tp->id*32; block < HALFSIZE; block += ctx->nthreads*32) {
          u32 alive32 = alive->block(block); // GLOBAL 1 SEQ
          for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
            if (alive32 & 1) {
              node_t u = sipnode(&ctx->sip_ctx, nonce, uorv);
              if ((u & PART_MASK) == part) {
                u32 b = u >> BUCKETSHIFT;
                u32 *bsize = &bucketsizes[b];
                buckets[b][*bsize] = (nonce << NONCESHIFT) | (u >> PART_BITS);
                if (++*bsize == BUCKETSIZE) {
                  *bsize = 0;
                  for (u32 i=0; i<BUCKETSIZE; i++) {
                    u64 bi = buckets[b][i];
                    if (!qkill) {
                      nonleaf->set(bi & NODEPARTMASK); // GLOBAL 1 RND BUCKETSIZE-1 SEQ 
                    } else {
                      if (!nonleaf->test(bi & NODEPARTMASK)) { // GLOBAL 1 RND BUCKETSIZE-1 SEQ 
                        nonce_t n = (nonce & -NONCETRUNC) | (bi >> NONCESHIFT);
                        alive->reset(n <= nonce ? n : n-NONCETRUNC, tp->id); // GLOBAL SEQ 
                      }
                    }
                  }
                }
              }
            }
          }
        }
        for (u32 b=0; b < NBUCKETS; b++) {
          u32 ni = bucketsizes[b];
          for (u32 i=0; i<ni ; i++) {
            node_t bi = buckets[b][i];
            if (!qkill) {
              nonleaf->set(bi & NODEPARTMASK);
            } else {
              if (!nonleaf->test(bi & NODEPARTMASK)) {
                nonce_t n = (HALFSIZE & -NONCETRUNC) | (bi >> NONCESHIFT);
                alive->reset(n < HALFSIZE  ? n : n-NONCETRUNC, tp->id); // GLOBAL SEQ 
              }
            }
          }
        }
        barrier(&ctx->barry);
      }
      if (tp->id == 0) {
        u32 load = (u32)(100 * alive->count() / CUCKOO_SIZE);
        printf("round %d part %c%d load %d%%\n", round, "UV"[uorv], part, load);
      }
    }
  }
}

u32 path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (nu < 0)
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
  u32 soli = atomicAdd(&ctx->nsols, 1U);
  for (nonce_t nonce = n = 0; nonce < HALFSIZE; nonce++)
    if (ctx->alive->test(nonce)) {
      edge e(sipnode(&ctx->sip_ctx, nonce, 0), HALFSIZE+sipnode(&ctx->sip_ctx, nonce, 1));
      if (cycle.find(e) != cycle.end()) {
        ctx->sols[soli][n++] = nonce;
        if (PROOFSIZE > 2)
          cycle.erase(e);
      }
    }
  assert(n==PROOFSIZE);
}

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
  u64 edgeBytes = HALFSIZE/8, nodeBytes = TWICE_WORDS*4;
  int edgeUnit, nodeUnit;
  for (edgeUnit=0; edgeBytes >= 1024; edgeBytes>>=10,edgeUnit++) ;
  for (nodeUnit=0; nodeBytes >= 1024; nodeBytes>>=10,nodeUnit++) ;
  printf("Using %d%cB edge and %d%cB node memory.\n",
     (int)edgeBytes, " KMGT"[edgeUnit], (int)nodeBytes, " KMGT"[nodeUnit]);
  cuckoo_ctx ctx(header, nthreads, ntrims, maxsols);


  checkCudaErrors(cudaMalloc((void**)&ctx.cuckoo, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMemset(ctx.cuckoo, 0, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMalloc((void**)&ctx.cycles, maxcycles*sizeof(cycle)));

  cuckoo_ctx *device_ctx;
  checkCudaErrors(cudaMalloc((void**)&device_ctx, sizeof(cuckoo_ctx)));
  cudaMemcpy(device_ctx, &ctx, sizeof(cuckoo_ctx), cudaMemcpyHostToDevice);

  worker<<<nthreads,1>>>(device_ctx);
  // assert(tp->buckets = (u64 (*)[BUCKETSIZE])calloc(NBUCKETS * BUCKETSIZE, sizeof(u64)));

  shrinkingset *alive = ctx->alive;
  u32 load = 100 * HALFSIZE / CUCKOO_SIZE;
  if (tp->id == 0)
    printf("initial load %d%%\n", load);
  for (u32 round=1; round <= ctx->ntrims; round++)
    trim_edges(tp, round);

  cudaMemcpy(&ctx, device_ctx, sizeof(cuckoo_ctx), cudaMemcpyDeviceToHost);
  cycle *cycles;
  cycles = (cycle *)calloc(maxcycles, sizeof(cycle));
  cudaMemcpy(cycles, ctx.cycles, maxcycles*sizeof(cycle), cudaMemcpyDeviceToHost);

  for (int s = 0; s < ctx.ncycles; s++)
    printf("% 4d-cycle found at %d:%d%%\n", cycles[s].len, cycles[s].id, cycles[s].pct);

  checkCudaErrors(cudaFree(device_ctx));
  checkCudaErrors(cudaFree(ctx.cycles));
  checkCudaErrors(cudaFree(ctx.cuckoo));

#if 0
  if (tp->id == 0) {
    load = (u32)(100 * alive->count() / CUCKOO_SIZE);
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
      if (alive->test(nonce)) {
        node_t u0, v0;
        sipedge(&ctx->sip_ctx, nonce, &u0, &v0);
        v0 += HALFSIZE;  // make v's different from u's
        node_t u = cuckoo[u0], v = cuckoo[v0];
        us[0] = u0;
        vs[0] = v0;
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
#endif
  // free(tp->buckets);

#if 0
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
