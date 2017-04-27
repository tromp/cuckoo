// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp
// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html
// xenoncat demonstrated how bucket sorting avoids random memory access latency
// define SINGLECYCLING to run cycle finding single threaded which runs slower
// but avoids losing cycles to race conditions (not worth it in my testing)

#include "cuckoo.h"
#include "siphashxN.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <x86intrin.h>

#ifdef __APPLE__
#include "osx_barrier.h"
#endif

#include <assert.h>

#ifdef ATOMIC
#include <atomic>
typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;
#else
typedef u32 au32;
typedef u64 au64;
#endif

// algorithm/performance parameters

#ifndef EDGEHASH_BYTES
#define EDGEHASH_BYTES 4
#endif

#define EDGEHASH_BITS (8 * EDGEHASH_BYTES)

#define NODEBITS (EDGEBITS + 1)

#ifndef BUCKETBITS
#define BUCKETBITS 8
#endif

#define BIGHASHBITS (EDGEBITS - BUCKETBITS)
#define EDGEBITSLO (EDGEHASH_BITS - BIGHASHBITS)
#define EDGEBITSHI (EDGEBITS - EDGEBITSLO)

#if EDGEBITS < 32
typedef u32 edge_t;
typedef u32 node_t;
#else
typedef u64 edge_t;
typedef u64 node_t;
#endif

#define NEDGESLO (1 << EDGEBITSLO)
#define NEDGESHI (1 << EDGEBITSHI)

#ifndef NPREFETCH
// how many prefetches to queue up
// before accessing the memory
// must be a multiple of NSIPHASH
#define NPREFETCH 32
#endif

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) < sizeof(alive), so
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * EDGEHASH_BYTES
// CUCKOO_SIZE * 8             < NEDGES * 4
// (NNODES >> IDXSHIFT) * 2    < NEDGES
// IDXSHIFT                    > 2
#define IDXSHIFT 8
#endif
// grow with cube root of size, hardly affected by trimming
#define MAXPATHLEN (8 << (NODEBITS/3))

#define NBUCKETS (1 << BUCKETBITS)
#define BUCKETMASK (NBUCKETS - 1)
#define BUCKETSIZE0 (1 << (EDGEBITS-BUCKETBITS))
#ifndef OVERHEAD_FACTOR
// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// OVERHEAD_FACTOR is 1/eps, and should be at most sqrt(n*p/64)
// to give negligible bad odds of e^-64
#define OVERHEAD_FACTOR 128
#endif
#define BUCKETSIZE (BUCKETSIZE0 + BUCKETSIZE0 / OVERHEAD_FACTOR)
#define BUCKETBYTES (BUCKETSIZE * EDGEHASH_BYTES)

typedef uint8_t bucket[BUCKETBYTES];

#if EDGEBITS >= 25
typedef uint8_t bucketcnt;
#else
#error make typedef uint16_t bucketcnt;
#endif

typedef struct histogram {
  bucketcnt cnt[NBUCKETS];
} histogram;

void clear_histogram(histogram *hist) {
  memset(hist->cnt, 0, NBUCKETS*sizeof(bucketcnt));
}

void diff_histogram(histogram *hist, u32 *old, u32 *new) {
  for (u32 i=0; i < NBUCKETS; i++)
    hist->cnt[i] = new[i] - old[i];
}

u32 total_histogram(histogram *hist) {
  u32 sum = 0;
  for (u32 i=0; i < NBUCKETS; i++)
    sum += (u32)hist->cnt[i];
  return sum;
}

typedef struct histgroup {
  histogram groups[NEDGESHI];
  u32 oldbig[NBUCKETS];
} histgroup;

void record_histgroup(histgroup *hg, u32 *big) {
  memcpy(hg->oldbig, big, NBUCKETS * sizeof(u32));
}

void update_histgroup(histgroup *hg, u32 grp, u32 *big) {
  diff_histogram(&hg->groups[grp], hg->oldbig, big);
  record_histgroup(hg, big);
}

// maintains set of trimmable edges
typedef struct edgetrimmer {
  u32 nthreads;
  histgroup *hists;
  bucket *buckets;
} edgetrimmer;

edgetrimmer *new_edgetrimmer(const u32 nt) {
  edgetrimmer *et = (edgetrimmer *)malloc(sizeof(edgetrimmer));
  assert(et);
  et->nthreads = nt;
  et->hists = (histgroup *)malloc(et->nthreads * sizeof(histgroup));
  assert(et->hists);
  et->buckets = (bucket *)malloc(NBUCKETS * sizeof(bucket));
  assert(et->buckets);
  return et;
}

void destroy_edgetrimmer(edgetrimmer *et) {
  free(et->hists);
  free(et->buckets);
}

void init_sort(edgetrimmer *et, u32 id, u32 *big) {
  for (u32 i=0; i < NBUCKETS; i++)
    big[i] = i * BUCKETSIZE + id * BUCKETSIZE / et->nthreads;
  record_histgroup(&et->hists[0], big);
}

u32 total_sorted(edgetrimmer *et) {
  u32 sum = 0;
  for (u32 t=0; t < et->nthreads; t++)
    for (u32 ehi=0; ehi < NEDGESHI; ehi++)
      sum += total_histogram(&et->hists[t].groups[ehi]);
  return sum;
}

#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#if 0
const static u64 CUCKOO_MASK = CUCKOO_SIZE - 1;
// number of (least significant) key bits that survives leftshift by NODEBITS
const static u32 KEYBITS = 64-NODEBITS;
const static u64 KEYMASK = (1LL << KEYBITS) - 1;
const static u64 MAXDRIFT = 1LL << (KEYBITS - IDXSHIFT);

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash(void *recycle) {
    cuckoo = (au64 *)recycle;
    memset(cuckoo, 0, CUCKOO_SIZE*sizeof(au64));
  }
  void set(node_t u, node_t v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#if !defined(SINGLECYCLING) && defined(ATOMIC)
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if ((old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
        return;
      }
#else
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
        return;
      }
#endif
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#if !defined(SINGLECYCLING) && defined(ATOMIC)
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
#else
      u64 cu = cuckoo[ui];
#endif
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (node_t)(cu & (NNODES-1));
      }
    }
  }
};
#endif

typedef struct cuckoo_ctx {
  siphash_keys sip_keys;
  edgetrimmer *alive;
  // cuckoo_hash *cuckoo;
  edge_t (*sols)[PROOFSIZE];
  u32 nonce;
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  pthread_barrier_t barry;
} cuckoo_ctx;

cuckoo_ctx *new_cuckoo_ctx(u32 n_threads, u32 n_trims, u32 max_sols) {
  cuckoo_ctx *ctx = (cuckoo_ctx *)malloc(sizeof(cuckoo_ctx));
  assert(ctx);
  ctx->alive = new_edgetrimmer(ctx->nthreads = n_threads);
  // cuckoo = 0;
  ctx->ntrims = n_trims;
  int err = pthread_barrier_init(&ctx->barry, NULL, n_threads);
  assert(err == 0);
  ctx->sols = (edge_t (*)[PROOFSIZE])calloc(ctx->maxsols = max_sols, PROOFSIZE*sizeof(edge_t));
  assert(ctx->sols);
  ctx->nsols = 0;
  return ctx;
}

void setheadernonce(cuckoo_ctx *ctx, char* headernonce, const u32 len, const u32 nce) {
  // place nonce at end
  ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(ctx->nonce = nce);
  setheader(headernonce, len, &ctx->sip_keys);
  ctx->nsols = 0;
}

void destroy_cuckoo_ctx(cuckoo_ctx *ctx) {
  destroy_edgetrimmer(ctx->alive);
  // delete cuckoo;
  free(ctx);
}

#ifdef DUMMY
#define STORE(i,v,x,w) dummy += _mm256_extract_epi32(v,x)
#else
#define STORE(i,v,x,w) \
  z = _mm256_extract_epi32(v,x);\
  if (block+i > last[z] + NEDGESLO) { big0[big[z]] = 0; big[z]++; dummy++; }\
  last[z] = block+i;\
  big0[big[z]] = _mm256_extract_epi32(w,x);\
  big[z]++;
#endif

void trim0(cuckoo_ctx *ctx, const u32 id) {
  u32 big[NBUCKETS];
  u32 last[NBUCKETS];
  uint64_t rdtsc0, rdtsc1;
  
  rdtsc0 = __rdtsc();
  init_sort(ctx->alive, id, big);
  edge_t hi0 = id * NEDGESHI / ctx->nthreads, endhi = (id+1) * NEDGESHI / ctx->nthreads; 
  static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
  static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
  const __m256i vinit = _mm256_set_epi64x(
    ctx->sip_keys.k1^0x7465646279746573ULL,
    ctx->sip_keys.k0^0x6c7967656e657261ULL,
    ctx->sip_keys.k1^0x646f72616e646f6dULL,
    ctx->sip_keys.k0^0x736f6d6570736575ULL);
  edge_t hi0x = hi0 * NEDGESLO;
  __m256i v0, v1, v2, v3, v4, v5, v6, v7;
  __m256i vpacket0 = _mm256_set_epi64x(hi0x+7, hi0x+5, hi0x+3, hi0x+1);
  __m256i vpacket1 = _mm256_set_epi64x(hi0x+15, hi0x+13, hi0x+11, hi0x+9);
  static const __m256i vpacketinc = {16, 16, 16, 16};
  __m256i vhi0 = _mm256_set_epi64x(3<<BIGHASHBITS, 2<<BIGHASHBITS, 1<<BIGHASHBITS, 0);
  __m256i vhi1 = _mm256_set_epi64x(7<<BIGHASHBITS, 6<<BIGHASHBITS, 5<<BIGHASHBITS, 4<<BIGHASHBITS);
  static const __m256i vhiinc = {8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS};
  u32 z, *big0 = (u32 *)ctx->alive->buckets;
  u64 dummy = 0;
  for (u32 i=0; i < NBUCKETS; i++)
    last[i] = 0;
  // for (edge_t hi = hi0 ; hi < endhi; hi++) {
  edge_t block = hi0 * NEDGESLO, endblock = endhi * NEDGESLO;
  for (; block < endblock; block += NSIPHASH) {
      v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
      v0 = _mm256_permute4x64_epi64(vinit, 0x00);
      v1 = _mm256_permute4x64_epi64(vinit, 0x55);
      v2 = _mm256_permute4x64_epi64(vinit, 0xAA);
      v7 = _mm256_permute4x64_epi64(vinit, 0xFF);
      v4 = _mm256_permute4x64_epi64(vinit, 0x00);
      v5 = _mm256_permute4x64_epi64(vinit, 0x55);
      v6 = _mm256_permute4x64_epi64(vinit, 0xAA);

      v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
      SIPROUNDX8; SIPROUNDX8;
      v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
      v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
      v6 = XOR(v6,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
      SIPROUNDX8; SIPROUNDX8; SIPROUNDX8; SIPROUNDX8;
      v0 = XOR(XOR(v0,v1),XOR(v2,v3));
      v4 = XOR(XOR(v4,v5),XOR(v6,v7));

      vpacket0 = _mm256_add_epi64(vpacket0, vpacketinc);
      vpacket1 = _mm256_add_epi64(vpacket1, vpacketinc);
      v1 = v0 & vbucketmask;
      v5 = v4 & vbucketmask;
      v0 = _mm256_srli_epi64(v0 & vnodemask, BUCKETBITS) | vhi0;
      v4 = _mm256_srli_epi64(v4 & vnodemask, BUCKETBITS) | vhi1;
      vhi0 = _mm256_add_epi64(vhi0, vhiinc);
      vhi1 = _mm256_add_epi64(vhi1, vhiinc);

      STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
      STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
    }
    // update_histgroup(&ctx->alive->hists[id], hi-hi0, big);
  // }
  rdtsc1 = __rdtsc();
  printf("big0 %lx rdtsc: %u dummy %lu\n", (u64)big0, (rdtsc1-rdtsc0)>>20, dummy);
}

#if 0
  void kill_leaf_edges(const u32 id, const u32 uorv, const u32 part) {
    alignas(64) u64 indices[NPREFETCH];
    alignas(64) u64 hashes[NPREFETCH];
  
    memset(hashes, 0, NPREFETCH * sizeof(u64)); // allow many nonleaf->test(0) to reduce branching
    u32 nidx = 0;
    for (edge_t block = id*64; block < NEDGES; block += nthreads*64) {
      u64 alive64 = alive->block(block);
      for (edge_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        indices[nidx++] = 2*nonce + uorv;
        if (nidx % NSIPHASH == 0) {
          siphash24xN(&sip_keys, indices+nidx-NSIPHASH, hashes+nidx-NSIPHASH);
          nidx %= NPREFETCH;
          kill(hashes+nidx, indices+nidx, NSIPHASH, part, id);
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    const u32 pnsip = nidx & -NSIPHASH;
    if (pnsip != nidx) {
      siphash24xN(&sip_keys, indices+pnsip, hashes+pnsip);
    }
    kill(hashes, indices, nidx, part, id);
    const u32 nnsip = pnsip + NSIPHASH;
    kill(hashes+nnsip, indices+nnsip, NPREFETCH-nnsip, part, id);
  }

void solution(node_t *us, u32 nu, node_t *vs, u32 nv) {
    typedef std::pair<node_t,node_t> edge;
    std::set<edge> cycle;
    u32 n = 0;
    cycle.insert(edge(*us, *vs));
    while (nu--)
      cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
    while (nv--)
      cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  #ifdef ATOMIC
    u32 soli = std::atomic_fetch_add_explicit(&nsols, 1U, std::memory_order_relaxed);
  #else
    u32 soli = nsols++;
  #endif
    for (edge_t block = 0; block < NEDGES; block += 64) {
      u64 alive64 = 0; // alive->block(block);
      for (edge_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        edge e(sipnode(&sip_keys, nonce, 0), sipnode(&sip_keys, nonce, 1));
        if (cycle.find(e) != cycle.end()) {
          sols[soli][n++] = nonce;
  #ifdef SHOWSOL
          printf("e(%x)=(%x,%x)%c", nonce, e.first, e.second, n==PROOFSIZE?'\n':' ');
  #endif
          if (PROOFSIZE > 2)
            cycle.erase(e);
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    assert(n==PROOFSIZE);
  }
};
#endif

typedef struct {
  u32 id;
  pthread_t thread;
  cuckoo_ctx *ctx;
} thread_ctx;

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

#if 0
u32 path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (!~nu)
        printf("maximum path length exceeded\n");
      else printf("illegal %4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu++] = u;
  }
  return nu-1;
}
#endif

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  edgetrimmer *alive = ctx->alive;
  u32 load = 100LL * NEDGES / CUCKOO_SIZE;
  if (tp->id == 0)
    printf("initial load %d%%\n", load);
  trim0(ctx, tp->id);
#if 0
  for (u32 round=1; round <= ctx->ntrims; round++) {
    if (tp->id == 0) printf("round %2d partition loads", round);
    for (u32 uorv = 0; uorv < 2; uorv++) {
        barrier(&ctx->barry);
        // ctx->count_node_deg(tp->id,uorv,0);
        barrier(&ctx->barry);
        // ctx->kill_leaf_edges(tp->id,uorv,0);
        barrier(&ctx->barry);
        if (tp->id == 0) {
          u32 load = (u32)(100LL * alive->total() / CUCKOO_SIZE);
          printf(" %c%d %4d%%", "UV"[uorv], 0, load);
        }
    }
    if (tp->id == 0) printf("\n");
  }
  if (tp->id == 0) {
    load = (u32)(100LL * alive->total() / CUCKOO_SIZE);
    printf("nonce %d: %d trims completed  final load %d%%\n", ctx->nonce, ctx->ntrims, load);
    if (load >= 90) {
      printf("overloaded! exiting...");
      pthread_exit(NULL);
    }
    ctx->cuckoo = new cuckoo_hash(ctx->alive->buckets);
  }
#ifdef SINGLECYCLING
  else pthread_exit(NULL);
#else
  barrier(&ctx->barry);
#endif
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
#ifdef SINGLECYCLING
  for (edge_t block = 0; block < NEDGES; block += 64) {
#else
  for (edge_t block = tp->id*64; block < NEDGES; block += ctx->nthreads*64) {
#endif
    u64 alive64 = 0; // alive->block(block);
    for (edge_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
      u32 ffs = __builtin_ffsll(alive64);
      nonce += ffs; alive64 >>= ffs;
      node_t u0=sipnode(&ctx->sip_keys, nonce, 0), v0=sipnode(&ctx->sip_keys, nonce, 1);
      if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
        u32 nu = path(cuckoo, u0, us), nv = path(cuckoo, v0, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("%4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100LL/NEDGES));
          if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
            // ctx->solution(us, nu, vs, nv);
        } else if (nu < nv) {
          while (nu--)
            cuckoo.set(us[nu+1], us[nu]);
          cuckoo.set(u0, v0);
        } else {
          while (nv--)
            cuckoo.set(vs[nv+1], vs[nv]);
          cuckoo.set(v0, u0);
        }
      }
      if (ffs & 64) break; // can't shift by 64
    }
  }
#endif
  pthread_exit(NULL);
  return 0;
}
