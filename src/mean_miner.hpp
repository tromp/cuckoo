// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp
// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html
// xenoncat demonstrated how bucket sorting avoids random memory access latency
// define SINGLECYCLING to run cycle finding single threaded which runs slower
// but avoids losing cycles to race conditions (not worth it in my testing)

#include "cuckoo.hpp"
#include "siphashxN.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <x86intrin.h>

#ifdef __APPLE__
#include "osx_barrier.h"
#endif
#include <set>

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

#ifndef EDGEBITS
#define EDGEBITS 27
#endif

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

const static edge_t NEDGES   = 1 << EDGEBITS;
const static edge_t EDGEMASK = NEDGES-1;
const static edge_t NEDGESLO = 1 << EDGEBITSLO;
const static edge_t NEDGESHI = 1 << EDGEBITSHI;
const static edge_t NNODES   = 2 << EDGEBITS;

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) < sizeof(alive), so
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * EDGEHASH_BYTES
// CUCKOO_SIZE * 8             < NEDGES * 4
// (NNODES >> IDXSHIFT) * 2    < NEDGES
// IDXSHIFT                    > 2
#define IDXSHIFT 8
#endif
// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << (NODEBITS/3);

const static u32 NBUCKETS = 1 << BUCKETBITS;
const static u32 BUCKETMASK = NBUCKETS - 1;
const static u32 BUCKETSIZE0 = (1 << (EDGEBITS-BUCKETBITS));
#ifndef OVERHEAD_FACTOR
#define OVERHEAD_FACTOR 64
#endif
const static u32 BUCKETSIZE = BUCKETSIZE0 + BUCKETSIZE0 / OVERHEAD_FACTOR;
const static u32 BUCKETBYTES = BUCKETSIZE * EDGEHASH_BYTES; // beware overflow

#if EDGEBITS >= 25
typedef uint8_t bucketcnt;
#else
#error make typedef uint16_t bucketcnt;
#endif

class histogram {
public:
  bucketcnt cnt[NBUCKETS];

  void clear() {
    memset(cnt, 0, NBUCKETS*sizeof(bucketcnt));
  }
  void diff(u32 *old, u32 *niew) {
    for (u32 i=0; i < NBUCKETS; i++)
      cnt[i] = niew[i] - old[i];
  }
  u32 total() {
    u32 sum = 0;
    for (u32 i=0; i < NBUCKETS; i++)
      sum += (u32)cnt[i];
    return sum;
  }
};

class histgroup {
public:
  histogram groups[NEDGESHI];
  u32 oldbig[NBUCKETS];
  void record(u32 *big) {
    memcpy(oldbig, big, NBUCKETS * sizeof(u32));
  }
  void update(u32 grp, u32 *big) {
    groups[grp].diff(oldbig, big);
    record(big);
  }
};

// maintains set of trimmable edges
class edgetrimmer {
public:
  u32 nthreads;
  histgroup *hists;
  u32 *buckets;
  u32 *tbuckets;

  edgetrimmer(const u32 nt) {
    nthreads = nt;
    hists    = new histgroup[nthreads];
#if EDGEHASH_BYTES == 4
    buckets  = new u32[NBUCKETS*BUCKETSIZE];
    tbuckets = new u32[nthreads*BUCKETSIZE];
#else
#error change buckets type
#endif
  }
  ~edgetrimmer() {
    delete[] hists;
    delete[] buckets;
    delete[] tbuckets;
  }
  void init(u32 id, u32 *big) {
    for (u32 i=0; i < NBUCKETS; i++)
      big[i] = i * BUCKETSIZE + id * BUCKETSIZE / nthreads;
    hists[0].record(big);
  }
  u32 total() const {
    u32 sum = 0;
    for (u32 t=0; t < nthreads; t++)
      for (u32 ehi=0; ehi < NEDGESHI; ehi++)
        sum += hists[t].groups[ehi].total();
    return sum;
  }
};

const static u64 CUCKOO_SIZE = NNODES >> IDXSHIFT;
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

class cuckoo_ctx {
public:
  siphash_keys sip_keys;
  edgetrimmer *alive;
  cuckoo_hash *cuckoo;
  edge_t (*sols)[PROOFSIZE];
  u32 nonce;
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  pthread_barrier_t barry;

  cuckoo_ctx(u32 n_threads, u32 n_trims, u32 max_sols) {
    nthreads = n_threads;
    alive = new edgetrimmer(nthreads);
    cuckoo = 0;
    ntrims = n_trims;
    int err = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
    sols = (edge_t (*)[PROOFSIZE])calloc(maxsols = max_sols, PROOFSIZE*sizeof(edge_t));
    assert(sols != 0);
    nsols = 0;
  }
  void setheadernonce(char* headernonce, const u32 len, const u32 nce) {
    nonce = nce;
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &sip_keys);
    nsols = 0;
  }
  ~cuckoo_ctx() {
    delete alive;
    delete cuckoo;
  }

  void trimbig0(const u32 id) {
    uint64_t rdtsc0, rdtsc1;
    u32 big[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    alive->init(id, big);
    edge_t hi0 = id * NEDGESHI / nthreads, endhi = (id+1) * NEDGESHI / nthreads; 
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    edge_t hi0x = hi0 * NEDGESLO;
    __m256i v0, v1, v2, v3;
    __m256i vpacket = _mm256_set_epi64x(hi0x+7, hi0x+5, hi0x+3, hi0x+1);
    static const __m256i vpacketinc = {8, 8, 8, 8};
    __m256i vhi = _mm256_set_epi64x(3<<BIGHASHBITS, 2<<BIGHASHBITS, 1<<BIGHASHBITS, 0);
    static const __m256i vhiinc = {4<<BIGHASHBITS, 4<<BIGHASHBITS, 4<<BIGHASHBITS, 4<<BIGHASHBITS};
    u32 z, *big0 = alive->buckets;
    u64 dummy = 0;
    for (edge_t hi = hi0 ; hi < endhi; hi++) {
      for (edge_t block = 0; block < NEDGESLO; block += NSIPHASH) {
        v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
        v0 = _mm256_permute4x64_epi64(vinit, 0x00);
        v1 = _mm256_permute4x64_epi64(vinit, 0x55);
        v2 = _mm256_permute4x64_epi64(vinit, 0xAA);

        v3 = XOR(v3,vpacket);
        SIPROUNDX4; SIPROUNDX4;
        v0 = XOR(v0,vpacket);
        v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
        SIPROUNDX4; SIPROUNDX4; SIPROUNDX4; SIPROUNDX4;
        v0 = XOR(XOR(v0,v1),XOR(v2,v3));

        vpacket = _mm256_add_epi64(vpacket, vpacketinc);
        v1 = v0 & vbucketmask;
        v0 = _mm256_srli_epi64(v0 & vnodemask, BUCKETBITS) | vhi;
        vhi = _mm256_add_epi64(vhi, vhiinc);
#ifdef DUMMY
       dummy += _mm256_extract_epi32(v1, 0);
       dummy += _mm256_extract_epi32(v1, 2);
       dummy += _mm256_extract_epi32(v1, 4);
       dummy += _mm256_extract_epi32(v1, 6);
#else
        z = _mm256_extract_epi32(v1,0);
        big0[big[z]] = _mm256_extract_epi32(v0,0); big[z]++;
        z = _mm256_extract_epi32(v1,2);
        big0[big[z]] = _mm256_extract_epi32(v0,2); big[z]++;
        z = _mm256_extract_epi32(v1,4);
        big0[big[z]] = _mm256_extract_epi32(v0,4); big[z]++;
        z = _mm256_extract_epi32(v1,6);
        big0[big[z]] = _mm256_extract_epi32(v0,6); big[z]++;
#endif
      }
      alive->hists[id].update(hi-hi0, big);
    }
    rdtsc1 = __rdtsc();
    printf("trimbig0 rdtsc: %lu dummy %lu\n", rdtsc1-rdtsc0, dummy);
  }

  void trimbig(const u32 id, int uorv) {
    uint64_t rdtsc0, rdtsc1;
    u32 big[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    alive->init(id, big);
    edge_t hi0 = id * NEDGESHI / nthreads, endhi = (id+1) * NEDGESHI / nthreads; 
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    edge_t hi0x = hi0 * NEDGESLO;
    __m256i v0, v1, v2, v3;
    __m256i vpacket = _mm256_set_epi64x(hi0x+7, hi0x+5, hi0x+3, hi0x+1);
    static const __m256i vpacketinc = {8, 8, 8, 8};
    __m256i vhi = _mm256_set_epi64x(3<<BIGHASHBITS, 2<<BIGHASHBITS, 1<<BIGHASHBITS, 0);
    static const __m256i vhiinc = {4<<BIGHASHBITS, 4<<BIGHASHBITS, 4<<BIGHASHBITS, 4<<BIGHASHBITS};
    u32 z, *big0 = alive->buckets;
    u64 dummy = 0;
    for (edge_t hi = hi0 ; hi < endhi; hi++) {
      for (edge_t block = 0; block < NEDGESLO; block += NSIPHASH) {
        v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
        v0 = _mm256_permute4x64_epi64(vinit, 0x00);
        v1 = _mm256_permute4x64_epi64(vinit, 0x55);
        v2 = _mm256_permute4x64_epi64(vinit, 0xAA);

        v3 = XOR(v3,vpacket);
        SIPROUNDX4; SIPROUNDX4;
        v0 = XOR(v0,vpacket);
        v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
        SIPROUNDX4; SIPROUNDX4; SIPROUNDX4; SIPROUNDX4;
        v0 = XOR(XOR(v0,v1),XOR(v2,v3));

        vpacket = _mm256_add_epi64(vpacket, vpacketinc);
        v1 = v0 & vbucketmask;
        v0 = _mm256_srli_epi64(v0 & vnodemask, BUCKETBITS) | vhi;
        vhi = _mm256_add_epi64(vhi, vhiinc);
#ifdef DUMMY
       dummy += _mm256_extract_epi32(v1, 0);
       dummy += _mm256_extract_epi32(v1, 2);
       dummy += _mm256_extract_epi32(v1, 4);
       dummy += _mm256_extract_epi32(v1, 6);
#else
        z = _mm256_extract_epi32(v1,0);
        big0[big[z]] = _mm256_extract_epi32(v0,0); big[z]++;
        z = _mm256_extract_epi32(v1,2);
        big0[big[z]] = _mm256_extract_epi32(v0,2); big[z]++;
        z = _mm256_extract_epi32(v1,4);
        big0[big[z]] = _mm256_extract_epi32(v0,4); big[z]++;
        z = _mm256_extract_epi32(v1,6);
        big0[big[z]] = _mm256_extract_epi32(v0,6); big[z]++;
#endif
      }
      alive->hists[id].update(hi-hi0, big);
    }
    rdtsc1 = __rdtsc();
    printf("trimbig0 rdtsc: %lu dummy %lu\n", rdtsc1-rdtsc0, dummy);
  }

  void trimsmall(const u32 id) {
    uint64_t rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u32 *small0 = alive->tbuckets + bigbkt * BUCKETSIZE;
    u32 nedges = 0;
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    for (; bigbkt < endbkt; bigbkt++) {
      u32 *big0 = alive->tbuckets + bigbkt * BUCKETSIZE;
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * (BUCKETSIZE >> BUCKETBITS);
      for (u32 from = 0 ; from < nthreads; from++) {
        edge_t hi0 = from * NEDGESHI / nthreads,
             endhi = (from+1) * NEDGESHI / nthreads; 
        histogram *groups = alive->hists[id].groups;
        u32 *readbig = big0 + from * BUCKETSIZE / nthreads;
        for (edge_t hi = hi0 ; hi < endhi; hi++) {
          for (u32 cnt = groups[hi-hi0].cnt[bigbkt]; cnt--; readbig++) {
            edge_t e = *readbig;
            z = e & BUCKETMASK;
            small0[small[z]] =  | e >> BUCKETBITS;
            small[z]++;
            nedges++;
          }
        }
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimsmall rdtsc: %lu edges %d\n", rdtsc1-rdtsc0, nedges);
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
    u32 soli = nsols++;
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

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;
  edgetrimmer *alive = ctx->alive;

  ctx->trimbig0(tp->id);
  barrier(&ctx->barry);
  ctx->trimsmall(tp->id);
  for (u32 round=1; round <= ctx->ntrims; round++) {
    if (tp->id == 0) printf("round %2d loads", round);
    for (u32 uorv = 0; uorv < 2; uorv++) {
      barrier(&ctx->barry);
      ctx->trimbig(tp->id, uorv);
      barrier(&ctx->barry);
      ctx->trimsmall(tp->id);
      barrier(&ctx->barry);
      if (tp->id == 0) {
        u32 load = (u32)(100LL * alive->total() / NEDGES);
        printf(" %c%d %4d%%", "UV"[uorv], 0, load);
      }
    }
    if (tp->id == 0) printf("\n");
  }
  if (tp->id == 0) {
    u32 load = (u32)(100LL * alive->total() / NEDGES);
    printf("nonce %d: %d trims completed  final load %d%%\n", ctx->nonce, ctx->ntrims, load);
    ctx->cuckoo = new cuckoo_hash(ctx->alive->buckets);
  }
  /* else */ pthread_exit(NULL);
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (edge_t block = 0; block < NEDGES; block += 64) {
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
            ctx->solution(us, nu, vs, nv);
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
  pthread_exit(NULL);
  return 0;
}
