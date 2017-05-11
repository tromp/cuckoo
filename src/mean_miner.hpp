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
#include <assert.h>
#include <set>
#ifdef __APPLE__
#include "osx_barrier.h"
#endif

// algorithm/performance parameters

#ifndef EDGEBITS
#define EDGEBITS 29
#endif

#ifndef BUCKETBITS
#define BUCKETBITS 7
#endif

#ifndef MAXSOLS
// more than one 42-cycle is already exceedingly rare
#define MAXSOLS 4
#endif

#ifndef BIG0SIZE
#define BIG0SIZE 4
#endif

#if BIG0SIZE == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

#ifndef SMALL0SIZE
#define SMALL0SIZE 5
#endif

#if EDGEBITS >= 32
#error promote u32 to u64 where necessary
#endif

const static u32 NEDGES         = 1 << EDGEBITS;
const static u32 EDGEMASK       = NEDGES-1;
const static u32 BIG0BITS       = BIG0SIZE * 8;
const static u32 BIGHASHBITS    = EDGEBITS - BUCKETBITS;
const static u32 EDGEBITSLO     = BIG0BITS - BIGHASHBITS;
const static u32 NEDGESLO       = 1 << EDGEBITSLO;
const static u32 NNODES         = 2 << EDGEBITS;
const static u32 NBUCKETS       = 1 << BUCKETBITS;
const static u32 BUCKETMASK     = NBUCKETS - 1;
const static u32 BIGBUCKETSIZE0 = (BIG0SIZE << BIGHASHBITS);
const static u32 DEGREEBITS = EDGEBITS - 2 * BUCKETBITS;
const static u32 NDEGREES = 1 << DEGREEBITS;

// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps and should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.

// 1/64 is good for EDGEBITS-log(nthreads) >= 26 and BUCKETBIS == 8
#ifndef BIGEPS
#define BIGEPS 1/64
#endif
const static u32 BIGBUCKETSIZE = BIGBUCKETSIZE0 + BIGBUCKETSIZE0 * BIGEPS;
typedef uint8_t u8;
typedef u8 bigbucket[BIGBUCKETSIZE];

// 1/4 is good for EDGEBITS-log(nthreads) >= 26 and BUCKETBIS == 8
#ifndef SMALLEPS
#define SMALLEPS 1/4
#endif
const static u32 SMALLBUCKETSIZE0 = NDEGREES * SMALL0SIZE;
const static u32 SMALLBUCKETSIZE = SMALLBUCKETSIZE0 + SMALLBUCKETSIZE0 * SMALLEPS ;
typedef u8 smallbucket[SMALLBUCKETSIZE];

typedef u32 indices[NBUCKETS];

// break circular reference with forward declaration
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
  indices *nodes;
  indices *edges;
  bigbucket *buckets;
  smallbucket *tbuckets;
  u32 ntrims;
  u32 nthreads;
  thread_ctx *threads;
  pthread_barrier_t barry;
  u32 spare;

  edgetrimmer(const u32 n_threads, u32 n_trims) {
    nthreads = n_threads;
    ntrims   = n_trims;
    buckets  = new bigbucket[NBUCKETS];
    tbuckets = new smallbucket[n_threads*NBUCKETS];
    nodes    = new indices[n_threads];
    edges    = new indices[n_threads];
    threads  = new thread_ctx[nthreads];
    int err  = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] nodes;
    delete[] edges;
    delete[] threads;
  }
  u32 start(u32 id, u32 bkt) {
    return bkt * BIGBUCKETSIZE + id * BIGBUCKETSIZE / nthreads;
  }
  u32 *init(u32 id) {
   u32 *hist = nodes[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      hist[bkt] = start(id,bkt);
    return hist;
  }
  u32 index(u32 id, u32 bkt) {
    return nodes[id][bkt];
  }
  u32 size(u32 id, u32 bkt) {
    return nodes[id][bkt] - start(id,bkt);
  }
  u32 sumsize(u32 id) {
    u32 sum = 0;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      sum += size(id, bkt);
    return sum;
  }
  void sortbig0(const u32 id, u32 uorv) {
    uint64_t rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NBUCKETS];
#endif
  
    rdtsc0 = __rdtsc();
    u32 *big = init(id);
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    u32 block = NEDGES*id/nthreads, endblock = NEDGES*(id+1)/nthreads; 
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    u32 z, b2 = 2 * block + uorv;
    __m256i vpacket0 = _mm256_set_epi64x(b2+6, b2+4, b2+2, b2+0);
    __m256i vpacket1 = _mm256_set_epi64x(b2+14, b2+12, b2+10, b2+8);
    static const __m256i vpacketinc = {16, 16, 16, 16};
    __m256i vhi0 = _mm256_set_epi64x(3<<BIGHASHBITS, 2<<BIGHASHBITS, 1<<BIGHASHBITS, 0);
    __m256i vhi1 = _mm256_set_epi64x(7<<BIGHASHBITS, 6<<BIGHASHBITS, 5<<BIGHASHBITS, 4<<BIGHASHBITS);
    static const __m256i vhiinc = {8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS, 8<<BIGHASHBITS};
    u8 *big0 = buckets[0];
#ifdef NEEDSYNC
    u32 zz;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      last[bkt] = block;
#endif
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

#define likely(x)   __builtin_expect((x)!=0, 1)
#define unlikely(x)   __builtin_expect((x), 0)
#ifndef NEEDSYNC
#define STORE(i,v,x,w) \
  z = _mm256_extract_epi32(v,x);\
  *(u64 *)(big0+big[z]) = _mm256_extract_epi64(w,i%4);\
  big[z] += BIG0SIZE;
#else
#define STORE(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    z = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[z] + NEDGESLO <= block+i); last[z] += NEDGESLO, big[z] += BIG0SIZE)\
      *(u32 *)(big0+big[z]) = 0;\
    *(u32 *)(big0+big[z]) = zz;\
    big[z] += BIG0SIZE;\
    last[z] = block+i;\
  }
#endif
      STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
      STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
    }
    for (u32 z=0; z < NBUCKETS; z++) {
      assert(big[z] < start(id+1, z));
#ifdef NEEDSYNC
      for (; last[z]<endblock-NEDGESLO; last[z]+=NEDGESLO) {
        *(u32 *)(big0+big[z]) = 0;
        big[z] += BIG0SIZE;
      }
#endif
    }
    rdtsc1 = __rdtsc();
    printf("sortbig0 rdtsc: %lu sumsize %x\n", rdtsc1-rdtsc0, sumsize(id));
  }
  void trimbig(const u32 id, u32 uorv) {
    uorv += id;
  }
  template<typename BIGTYPE, u32 BIGSIZE, u32 SMALLSIZE>
  void trimsmall(const u32 id) {
    uint64_t rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
    const u32 BIGBITS = BIGSIZE * 8;
    const u64 BIGSIZEMASK = (1LL << BIGBITS) - 1LL;
    const u32 SMALLBITS = SMALLSIZE * 8;
    const u32 NONDEGREEBITS = SMALLBITS - DEGREEBITS;
    const u64 NONDEGREEMASK = (1 << NONDEGREEBITS) - 1;
    const u64 SMALLSIZEMASK = (1LL << SMALLBITS) - 1LL;
    const u32 DEGREEMASK = NDEGREES - 1;
  
    rdtsc0 = __rdtsc();
    spare = -1;
    u8 *big0 = buckets[0];
    // 81/128 is barely enough for (1-1/e) fraction of edges
    u8 *bigi0 = big0 + (BIGBUCKETSIZE / nthreads) * 47/128;
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 nedges = 0;
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 lastread = from * NEDGES / nthreads;
        u8    *readbig = big0 + start(from, bigbkt);
        u8 *endreadbig = big0 + index(from, bigbkt);
        for (; readbig < endreadbig; readbig += BIGSIZE) {
          BIGTYPE e = *(BIGTYPE *)readbig;
          if (BIGSIZE > 4) {
            e &= BIGSIZEMASK;
          } else {
            if (unlikely(!e)) { lastread += NEDGESLO; continue; }
          }
          lastread += ((u32)(e>>BIGHASHBITS) - lastread) & (NEDGESLO-1);
          u32 z = e & BUCKETMASK;
          *(u64 *)(small0+small[z]) = ((u64)lastread << DEGREEBITS)
                                    | (e >> BUCKETBITS);
          small[z] += SMALLSIZE;
        }
        if (unlikely(lastread >> EDGEBITSLO !=
          ((from+1)*NEDGES/nthreads - 1) >> EDGEBITSLO))
        { printf("OOPS1: bkt %d lastread %x\n", bigbkt, lastread); exit(0); }
      }
      u8 *degs = (u8 *)big0 + start(0, bigbkt); // recycle!
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 *writebig0 = (u32 *)(bigi0 + start(from, bigbkt));
        u32 *writebig = writebig0;
        u32 smallbkt = from * NBUCKETS / nthreads;
        u32 endsmallbkt = (from+1) * NBUCKETS / nthreads;
        for (; smallbkt < endsmallbkt; smallbkt++) {
          memset(degs, 1, NDEGREES);
          u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE;
          u8 *endreadsmall = small0 + small[smallbkt];
          for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE)
            degs[*(u32 *)rdsmall & DEGREEMASK]--;
          u32 lastread = 0;
          for (; readsmall < endreadsmall; readsmall+=SMALLSIZE) {
            u64 z = *(u64 *)readsmall & SMALLSIZEMASK;
            lastread += ((z>>DEGREEBITS) - lastread) & NONDEGREEMASK;
            *writebig = lastread;
            writebig += degs[z & DEGREEMASK] >> 7;
          }
          if (unlikely(lastread>>NONDEGREEBITS != EDGEMASK>>NONDEGREEBITS))
            { printf("OOPS2: id %d big %d from %d small %d lastread %x\n", id, bigbkt, from, smallbkt, lastread); exit(0); }
        }
        u32 *writelim = (u32 *)(big0 + start(from+1, bigbkt));
        if (writelim-writebig < spare)
          spare = writelim - writebig;
        nedges += writebig - writebig0;
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimsmall rdtsc: %lu edges %d\n", rdtsc1-rdtsc0, nedges);
    printf("%u to spare\n", spare);
  }

  void trim() {
    void *etworker(void *vp);
    for (u32 t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].et = this;
      int err = pthread_create(&threads[t].thread, NULL, etworker, (void *)&threads[t]);
      assert(err == 0);
    }
    for (u32 t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }
  }
  u32 nedges() {
    return 0;
  }
  void barrier(pthread_barrier_t *barry) {
    int rc = pthread_barrier_wait(barry);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      printf("Could not wait on barrier\n");
      pthread_exit(NULL);
    }
  }
  void trimmer(u32 id) {
    sortbig0(id, 1);
    barrier(&barry);
#if BIG0SIZE > 4
    trimsmall<u64,BIG0SIZE,SMALL0SIZE>(id);
#else
    trimsmall<u32,BIG0SIZE,SMALL0SIZE>(id);
#endif
    for (u32 round=1; round <= ntrims; round++) {
      if (id == 0) printf("round %2d loads", round);
      for (u32 uorv = 0; uorv < 2; uorv++) {
        barrier(&barry);
        trimbig(id, uorv);
        barrier(&barry);
        trimsmall<u64,6,6>(id);
        barrier(&barry);
        if (id == 0) {
          u32 load = (u32)(100LL * nedges() / NEDGES);
          printf(" %c%d %4d%%", "UV"[uorv], 0, load);
        }
      }
      if (id == 0) printf("\n");
    }
    if (id == 0) {
      u32 load = (u32)(100LL * nedges() / NEDGES);
      printf("%d trims completed  final load %d%%\n", ntrims, load);
    }
  }
};

void *etworker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  tp->et->trimmer(tp->id);
  pthread_exit(NULL);
  return 0;
}

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) < sizeof(alive), so
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * sizeof(u32)
// CUCKOO_SIZE * 8             < NEDGES * 4
// (NNODES >> IDXSHIFT) * 2    < NEDGES
// IDXSHIFT                    > 2
#define IDXSHIFT 8
#endif

#define NODEBITS (EDGEBITS + 1)

// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << (NODEBITS/3);

const static u64 CUCKOO_SIZE = NNODES >> IDXSHIFT;
const static u64 CUCKOO_MASK = CUCKOO_SIZE - 1;
// number of (least significant) key bits that survives leftshift by NODEBITS
const static u32 KEYBITS = 64-NODEBITS;
const static u64 KEYMASK = (1LL << KEYBITS) - 1;
const static u64 MAXDRIFT = 1LL << (KEYBITS - IDXSHIFT);

#ifdef ATOMIC
#include <atomic>
typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;
#else
typedef u32 au32;
typedef u64 au64;
#endif

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
    return NBUCKETS * sizeof(bigbucket);
  }
  u32 threadbytes() {
    return NBUCKETS * sizeof(smallbucket) + sizeof(indices) + sizeof(thread_ctx);
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
