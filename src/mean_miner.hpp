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

#ifndef EDGEBITS
#define EDGEBITS 27
#endif

#define NODEBITS (EDGEBITS + 1)

#ifndef BUCKETBITS
#define BUCKETBITS 8
#endif

// more than one 42-cycle is already exceedingly rare
#ifndef MAXSOLS
#define MAXSOLS 4
#endif

#define BIGHASHBITS (EDGEBITS - BUCKETBITS)
#define EDGEBITSLO (32 - BIGHASHBITS)
#define EDGEBITSHI (EDGEBITS - EDGEBITSLO)
#define SMALLBITSHI (32 - BUCKETBITS)

#if EDGEBITS >= 32
#error not implemented
#endif

const static u32 NEDGES   = 1 << EDGEBITS;
const static u32 EDGEMASK = NEDGES-1;
const static u32 NEDGESLO = 1 << EDGEBITSLO;
const static u32 NEDGESHI = 1 << EDGEBITSHI;
const static u32 NNODES   = 2 << EDGEBITS;

const static u32 NBUCKETS = 1 << BUCKETBITS;
const static u32 BUCKETMASK = NBUCKETS - 1;
const static u32 BIGBUCKETSIZE0 = (1 << (EDGEBITS-BUCKETBITS));
// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps and should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.
// 1/64 is good for EDGEBITS-log(nthreads) >= 26 and BUCKETBIS == 8
#ifndef BIGEPS
#define BIGEPS 1/64
#endif
const static u32 BIGBUCKETSIZE = BIGBUCKETSIZE0 + BIGBUCKETSIZE0 * BIGEPS;
typedef u32 bigbucket[BIGBUCKETSIZE];

// 1/4 is good for EDGEBITS-log(nthreads) >= 26 and BUCKETBIS == 8
#ifndef SMALLEPS
#define SMALLEPS 1/4
#endif
const static u32 SMALLBUCKETSIZE0 = (1 << (EDGEBITS-2*BUCKETBITS));
const static u32 SMALLBUCKETSIZE = SMALLBUCKETSIZE0 + SMALLBUCKETSIZE0 * SMALLEPS;
typedef u32 smallbucket[SMALLBUCKETSIZE];

typedef u32 histogram[NBUCKETS];

class edgetrimmer;

typedef struct {
  u32 id;
  pthread_t thread;
  edgetrimmer *et;
} thread_ctx;

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  histogram *hists;
  bigbucket *buckets;
  smallbucket *tbuckets;
  u32 ntrims;
  u32 nthreads;
  thread_ctx *threads;
  pthread_barrier_t barry;

  edgetrimmer(const u32 n_threads, u32 n_trims) {
    nthreads = n_threads;
    ntrims = n_trims;
    hists    = new histogram[n_threads];
    buckets  = new bigbucket[NBUCKETS];
    tbuckets = new smallbucket[n_threads*NBUCKETS];
    threads  = new thread_ctx[nthreads];
    int err = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
  }
  ~edgetrimmer() {
    delete[] hists;
    delete[] buckets;
    delete[] tbuckets;
    delete[] threads;
  }
  u32 *init(u32 id) {
   u32 *hist = hists[id];
    for (u32 i=0; i < NBUCKETS; i++)
      hist[i] = i * BIGBUCKETSIZE + id * BIGBUCKETSIZE / nthreads;
    return hist;
  }
  u32 sharedbytes() {
    return sizeof(buckets);
  }
  u32 threadbytes() {
    return nthreads * sizeof(smallbucket);
  }
#define likely(x)   __builtin_expect(!!(x), 1)
#ifdef DUMMY
#define STORE(i,v,x,w) dummy += _mm256_extract_epi32(v,x)
#else
#define STORE(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (x || likely(zz)) {\
    z = _mm256_extract_epi32(v,x);\
    for (; last[z] + NEDGESLO <= block+i; last[z] += NEDGESLO)\
      big0[big[z]++] = 0;\
    last[z] = block+i;\
    big0[big[z]] = _mm256_extract_epi32(w,x);\
    big[z]++;\
  }
#endif
  void trimbig0(const u32 id) {
    uint64_t rdtsc0, rdtsc1;
    u32 last[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u32 *big = init(id);
    u32 hi0 = id * NEDGESHI / nthreads, endhi = (id+1) * NEDGESHI / nthreads; 
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    u32 hi0x = hi0 * NEDGESLO;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    __m256i vpacket0 = _mm256_set_epi64x(hi0x+7, hi0x+5, hi0x+3, hi0x+1);
    __m256i vpacket1 = _mm256_set_epi64x(hi0x+15, hi0x+13, hi0x+11, hi0x+9);
    static const __m256i vpacketinc = {16, 16, 16, 16};
    __m256i vhi0 = _mm256_set_epi64x(3<<BIGHASHBITS, 2<<BIGHASHBITS, 1<<BIGHASHBITS, 0);
    __m256i vhi1 = _mm256_set_epi64x(7<<BIGHASHBITS, 6<<BIGHASHBITS, 5<<BIGHASHBITS, 4<<BIGHASHBITS);
    static const __m256i vhiinc = {8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS};
    u32 z, zz, *big0 = buckets[0];
    u64 dummy = 0;
    memset(last, 0, NBUCKETS * sizeof(u32));
    u32 block = hi0 * NEDGESLO, endblock = endhi * NEDGESLO;
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
    rdtsc1 = __rdtsc();
    printf("trimbig0 rdtsc: %lu dummy %lu\n", rdtsc1-rdtsc0, dummy);
  }
  void trimbig(const u32 id, u32 uorv) {
    uorv += id;
  }
  void trimsmall(const u32 id) {
    uint64_t rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u32 *small0 = tbuckets[id];
    u32 nedges = 0;
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    for (; bigbkt < endbkt; bigbkt++) {
      u32 *big0 = buckets[bigbkt];
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 *readbig = big0 + from * BIGBUCKETSIZE / nthreads;
        u32 last = from * NEDGES / nthreads;
        u32 cnt = hists[id][bigbkt];
        for (; cnt--; readbig++) {
          u32 e = *readbig;
          if (!e) { last += NEDGESLO; continue; }
          last += ((e>>BIGHASHBITS) - last) & (2*NEDGESLO-1); // magic!
          u32 z = e & BUCKETMASK;
          small0[small[z]] = last << BIGHASHBITS | e >> BUCKETBITS;
          small[z]++;
          nedges++;
        }
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimsmall rdtsc: %lu edges %d\n", rdtsc1-rdtsc0, nedges);
  }
  void trim() {
    void *worker(void *vp);
    for (int t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].et = this;
      int err = pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]);
      assert(err == 0);
    }
    for (int t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }
  }
  u32 nedges() {
    return 0;
  }
};

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) < sizeof(alive), so
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * sizeof(u32)
// CUCKOO_SIZE * 8             < NEDGES * 4
// (NNODES >> IDXSHIFT) * 2    < NEDGES
// IDXSHIFT                    > 2
#define IDXSHIFT 8
#endif
// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << (NODEBITS/3);

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
  void set(u32 u, u32 v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (u32 ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
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
  u32 operator[](u32 u) const {
    for (u32 ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#if !defined(SINGLECYCLING) && defined(ATOMIC)
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
#else
      u64 cu = cuckoo[ui];
#endif
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (u32)(cu & (NNODES-1));
      }
    }
  }
};

class solver_ctx {
public:
  edgetrimmer *alive;
  cuckoo_hash *cuckoo;
  u32 sols[MAXSOLS][PROOFSIZE];
  u32 nsols;

  solver_ctx(u32 n_threads, u32 n_trims) {
    alive = new edgetrimmer(n_threads, n_trims);
    cuckoo = 0;
  }
  void setheadernonce(char* headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &alive->sip_keys);
    nsols = 0;
  }
  ~solver_ctx() {
    delete cuckoo;
    delete alive;
  }

  u32 sharedbytes() {
    return alive->sharedbytes();
  }

  u32 threadbytes() {
    return alive->threadbytes();
  }

  void solution(u32 *us, u32 nu, u32 *vs, u32 nv) {
    typedef std::pair<u32,u32> edge;
    std::set<edge> cycle;
    u32 n = 0;
    cycle.insert(edge(*us, *vs));
    while (nu--)
      cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
    while (nv--)
      cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
    u32 soli = nsols++;
    for (u32 block = 0; block < NEDGES; block += 64) {
      u64 alive64 = 0; // alive->block(block);
      for (u32 nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        edge e(sipnode(&alive->sip_keys, nonce, 0), sipnode(&alive->sip_keys, nonce, 1));
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

  u32 path(u32 u, u32 *us) {
    u32 nu;
    for (nu = 0; u; u = (*cuckoo)[u]) {
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
  
  int solve() {
    alive->trim();
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];
    for (u32 block = 0; block < NEDGES; block += 64) {
      u64 alive64 = 0; // alive->block(block);
      for (u32 nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        u32 u0=sipnode(&alive->sip_keys, nonce, 0), v0=sipnode(&alive->sip_keys, nonce, 1);
        if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
          u32 nu = path(u0, us), nv = path(v0, vs);
          if (us[nu] == vs[nv]) {
            u32 min = nu < nv ? nu : nv;
            for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
            u32 len = nu + nv + 1;
            printf("%4d-cycle found at %d%%\n", len, (u32)(nonce*100LL/NEDGES));
            if (len == PROOFSIZE && nsols < MAXSOLS)
              solution(us, nu, vs, nv);
          } else if (nu < nv) {
            while (nu--)
              cuckoo->set(us[nu+1], us[nu]);
            cuckoo->set(u0, v0);
          } else {
            while (nv--)
              cuckoo->set(vs[nv+1], vs[nv]);
            cuckoo->set(v0, u0);
          }
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    return nsols;
  }
};

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  edgetrimmer *et = tp->et;

  et->trimbig0(tp->id);
  barrier(&et->barry);
  et->trimsmall(tp->id);
  pthread_exit(NULL);
  for (u32 round=1; round <= et->ntrims; round++) {
    if (tp->id == 0) printf("round %2d loads", round);
    for (u32 uorv = 0; uorv < 2; uorv++) {
      barrier(&et->barry);
      et->trimbig(tp->id, uorv);
      barrier(&et->barry);
      et->trimsmall(tp->id);
      barrier(&et->barry);
      if (tp->id == 0) {
        u32 load = (u32)(100LL * et->nedges() / NEDGES);
        printf(" %c%d %4d%%", "UV"[uorv], 0, load);
      }
    }
    if (tp->id == 0) printf("\n");
  }
  if (tp->id == 0) {
    u32 load = (u32)(100LL * et->nedges() / NEDGES);
    printf("%d trims completed  final load %d%%\n", et->ntrims, load);
  }
  pthread_exit(NULL);
  return 0;
}
