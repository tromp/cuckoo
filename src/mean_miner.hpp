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
#include <assert.h>
#include <set>
#ifdef __APPLE__
#include "osx_barrier.h"
#endif

// algorithm/performance parameters

// EDGEBITS/NEDGES/EDGEMASK/NNODES defined in cuckoo.h

// 8 seems optimal for 2-4 threads
#ifndef BUCKETBITS
#define BUCKETBITS 8
#endif

#ifndef MAXSOLS
// more than one 42-cycle is already exceedingly rare
#define MAXSOLS 4
#endif

// size in bytes of a big-bucket entry
#define BIGSIZE 5
// size in bytes of a small-bucket entry
#define SMALLSIZE 5

// initial entries could be smaller at percent or two slowdown
#ifndef BIG0SIZE
#define BIG0SIZE 5
#endif

// but they'll need syncing entries
#if BIG0SIZE == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

#if EDGEBITS >= 32
#error promote u32 to u64 where necessary
#endif

#if BIG0SIZE > 4
typedef u64 BIGTYPE;
#else
typedef u32 BIGTYPE;
#endif

// node bits have two groups of bucketbits (big and small) and a remaining group of degree bits
const static u32 DEGSMALLBITS   = EDGEBITS - BUCKETBITS;
const static u32 DEGSMALLMASK   = (1<<DEGSMALLBITS) - 1;
const static u32 NBUCKETS       = 1 << BUCKETBITS;
const static u32 BUCKETMASK     = NBUCKETS - 1;
const static u32 DEGREEBITS     = DEGSMALLBITS - BUCKETBITS;
const static u32 NDEGREES       = 1 << DEGREEBITS;
const static u32 DEGREEMASK     = NDEGREES - 1;
const static u32 DEGREEBITS1    = DEGREEBITS - 1;
const static u32 BIGBITS        = BIGSIZE * 8;
const static u64 BIGSIZEMASK    = (1ULL << BIGBITS) - 1ULL;
const static u32 DEG2SMALLBITS  = DEGSMALLBITS + DEGREEBITS;
const static u64 DEG2SMALLMASK  = (1ULL << DEG2SMALLBITS) - 1;
const static u32 SMALLPREFBITS  = BIGBITS - DEG2SMALLBITS;
const static u32 SMALLPREFMASK  = (1 << SMALLPREFBITS) - 1;
const static u32 DEG2BITS       = 2 * DEGREEBITS;
const static u32 DEG2MASK       = (1 << DEG2BITS) - 1;

const static u32 BIG0BITS       = BIG0SIZE * 8;
const static u64 BIG0SIZEMASK   = (1ULL << BIG0BITS) - 1ULL;
const static u32 EDGE0BITSLO    = BIG0BITS - DEGSMALLBITS;
const static u32 NEDGES0LO      = 1 << EDGE0BITSLO;
const static u32 BIGBUCKETSIZE0 = (BIG0SIZE << DEGSMALLBITS);

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
const static u32 SMALLBUCKETSIZE0 = NDEGREES * SMALLSIZE;
const static u32 SMALLBUCKETSIZE = SMALLBUCKETSIZE0 + SMALLBUCKETSIZE0 * SMALLEPS ;
typedef u8 smallbucket[SMALLBUCKETSIZE];

typedef u32 indices[NBUCKETS];

struct indexer {
  u8 *base;
  u32 nthreads;
  indices *index;

  indexer(const u32 n_threads, u8 *base) {
    nthreads = n_threads;
    index = new indices[nthreads];
    setbase(base);
  }
  void setbase(u8 *_base) {
    base = _base;
  }
  u32 start(const u32 id, const u32 bkt) const {
    return bkt * BIGBUCKETSIZE + id * BIGBUCKETSIZE / nthreads;
  }
  u32 end(const u32 id, const u32 bkt) const {
    return index[id][bkt];
  }
  u32 *init(const u32 id) {
   u32 *idx = index[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      idx[bkt] = start(id, bkt);
    return idx;
  }
  u32 size(u32 id, u32 bkt) const {
    return end(id, bkt) - start(id, bkt);
  }
  u32 sumsize(u32 id) const {
    u32 sum = 0;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      sum += size(id, bkt);
    return sum;
  }
  u32 sumsize() {
    u32 sum = 0;
    for (u32 id=0; id < nthreads; id++)
      sum += sumsize(id);
    return sum;
  }
  ~indexer() {
    delete[] index;
  }
};

// break circular reference with forward declaration
class edgetrimmer;

typedef struct {
  u32 id;
  pthread_t thread;
  edgetrimmer *et;
} thread_ctx;

#define likely(x)   __builtin_expect((x)!=0, 1)
#define unlikely(x)   __builtin_expect((x), 0)

static const u32 SENTINELSIZE = NSIPHASH * BIGSIZE + sizeof(u64); // sentinel for bigsort1;

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  indices *up;
  indices *down;
  indexer *idxA;
  indexer *idxB;
  bigbucket *buckets;
  smallbucket *tbuckets;
  u32 ntrims;
  u32 nthreads;
  thread_ctx *threads;
  pthread_barrier_t barry;

  void touch(u8 *p, u32 n) {
    for (u32 i=0; i<n; i+=4096)
      *(u32 *)(p+i) = 0;
  }
  edgetrimmer(const u32 n_threads, u32 n_trims) {
    nthreads = n_threads;
    ntrims   = n_trims;
    buckets  = new bigbucket[NBUCKETS];
    touch((u8 *)buckets,NBUCKETS*sizeof(bigbucket));
    tbuckets = new smallbucket[n_threads*NBUCKETS];
    touch((u8 *)tbuckets,n_threads*NBUCKETS*sizeof(smallbucket));
    up       = new indices[n_threads];
    down     = new indices[n_threads];
    idxA     = new indexer(n_threads, (u8 *)buckets);
    idxB     = new indexer(n_threads, (u8 *)buckets + sizeof(bigbucket)/4); // anywhere from 1% to 37% (about 1/e) is ok
    threads  = new thread_ctx[nthreads];
    int err  = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
    for (u32 id=0; id < nthreads; id++)
      for (u32 bkt=0; bkt < NBUCKETS; bkt++)
        memset(buckets[0] + bucketend(id,bkt), 0, SENTINELSIZE);
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] up;
    delete[] down;
    delete   idxA;
    delete   idxB;
    delete[] threads;
  }
  u32 bucketstart(u32 id, u32 bkt) {
    return bkt * BIGBUCKETSIZE + id * BIGBUCKETSIZE / nthreads;
  }
  u32 bucketend(u32 id, u32 bkt) {
    return bucketstart(id+1, bkt) - SENTINELSIZE;
  }
  u32 *upinit(u32 id) {
   u32 *ups = up[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      ups[bkt] = bucketstart(id,bkt);
    return ups;
  }
  u32 *downinit(u32 id) {
   u32 *dwn = down[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      dwn[bkt] = bucketend(id,bkt);
    return dwn;
  }
  u32 downsize(u32 id, u32 bkt) {
    return bucketend(id, bkt) - down[id][bkt];
  }
  u32 downsumsize() {
    u32 sum = 0;
    for (u32 id=0; id < nthreads; id++)
      for (u32 bkt=0; bkt < NBUCKETS; bkt++)
        sum += downsize(id, bkt);
    return sum;
  }
  u32 upsize(u32 id, u32 bkt) {
    return up[id][bkt] - bucketstart(id,bkt);
  }
  u32 upsumsize(u32 id) {
    u32 sum = 0;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      sum += upsize(id, bkt);
    return sum;
  }
  u32 upsumsize() {
    u32 sum = 0;
    for (u32 id=0; id < nthreads; id++)
      sum += upsumsize(id);
    return sum;
  }
  void genUnodes(indexer *dst, const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NBUCKETS];
#endif
  
    rdtsc0 = __rdtsc();
    u32 z, *big = dst->init(id);
    u8 *base = dst->base;
    u32 block = (NEDGES* id   /nthreads) & -8,
     endblock = (NEDGES*(id+1)/nthreads) & -8; 
#if NSIPHASH == 8
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    u32 b2 = 2 * block + uorv;
    __m256i vpacket0 = _mm256_set_epi64x(b2+6, b2+4, b2+2, b2+0);
    __m256i vpacket1 = _mm256_set_epi64x(b2+14, b2+12, b2+10, b2+8);
    static const __m256i vpacketinc = {16, 16, 16, 16};
    __m256i vhi0 = _mm256_set_epi64x(3<<DEGSMALLBITS, 2<<DEGSMALLBITS, 1<<DEGSMALLBITS, 0);
    __m256i vhi1 = _mm256_set_epi64x(7<<DEGSMALLBITS, 6<<DEGSMALLBITS, 5<<DEGSMALLBITS, 4<<DEGSMALLBITS);
    static const __m256i vhiinc = {8<<DEGSMALLBITS, 8<<DEGSMALLBITS, 8<<DEGSMALLBITS, 8<<DEGSMALLBITS};
#endif
#ifdef NEEDSYNC
    u32 zz;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      last[bkt] = block;
#endif
    for (; block < endblock; block += NSIPHASH) {
// bit        28..16     15..8      7..0
// node       degree  smallbkt    bigbkt
// bit     39/31..21     20..8      7..0
// write        edge    degree  smallbkt
#if NSIPHASH == 1
      u32 node = _sipnode(&sip_keys, block, uorv);
      z = node & BUCKETMASK;
      u64 zz = (u64)block << DEGSMALLBITS | (node & EDGEMASK) >> BUCKETBITS;
#ifndef NEEDSYNC
      *(u64 *)(base+big[z]) = zz;
      big[z] += BIG0SIZE;
#else
      if (zz) {
        for (; unlikely(last[z] + NEDGES0LO <= block); last[z] += NEDGES0LO, big[z] += BIG0SIZE)
          *(u32 *)(base+big[z]) = 0;
        *(u32 *)(base+big[z]) = zz;
        big[z] += BIG0SIZE;
        last[z] = block;
      }
#endif
#elif NSIPHASH == 8
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

#ifndef NEEDSYNC
#define STORE0(i,v,x,w) \
  z = _mm256_extract_epi32(v,x);\
  *(u64 *)(base+big[z]) = _mm256_extract_epi64(w,i%4);\
  big[z] += BIG0SIZE;
#else
#define STORE0(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    z = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[z] + NEDGES0LO <= block+i); last[z] += NEDGES0LO, big[z] += BIG0SIZE)\
      *(u32 *)(base+big[z]) = 0;\
    *(u32 *)(base+big[z]) = zz;\
    big[z] += BIG0SIZE;\
    last[z] = block+i;\
  }
#endif
      STORE0(0,v1,0,v0); STORE0(1,v1,2,v0); STORE0(2,v1,4,v0); STORE0(3,v1,6,v0);
      STORE0(4,v5,0,v4); STORE0(5,v5,2,v4); STORE0(6,v5,4,v4); STORE0(7,v5,6,v4);
#else
#error not implemented
#endif
    }
    for (u32 z=0; z < NBUCKETS; z++) {
      assert(dst->size(id, z) <= BIGBUCKETSIZE);
#ifdef NEEDSYNC
      for (; last[z]<endblock-NEDGES0LO; last[z]+=NEDGES0LO) {
        *(u32 *)(base+big[z]) = 0;
        big[z] += BIG0SIZE;
      }
#endif
    }
    rdtsc1 = __rdtsc();
    printf("genUnodes rdtsc: %lu sumsize %x\n", rdtsc1-rdtsc0, dst->sumsize(id));
  }
  void genVnodes(indexer *src, indexer *dst, const u32 id, const u32 uorv) {
    u64 e, rdtsc0, rdtsc1;
  
#if NSIPHASH == 8
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vdegmask = {DEGREEMASK, DEGREEMASK, DEGREEMASK, DEGREEMASK};
    static const __m256i vbucketmask = {BUCKETMASK, BUCKETMASK, BUCKETMASK, BUCKETMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    __m256i vsmall, vpacket0, vpacket1, vhi0, vhi1;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    u32 e0, e1, e2, e3;
    u64 re0, re1, re2, re3;
    static const u32 EVENNONDEGMASK = (BIGSIZEMASK >> DEGREEBITS1) & -2;
#endif

    rdtsc0 = __rdtsc();
    u32 z, *big = upinit(id);
    u8 *big0 = buckets[0];
    // contents of each big bucket is ordered by originating small bucket
    // moving along one small bucket at a time frees up room for filling big buckets
    for (u32 smallperfrom = (NBUCKETS + nthreads-1) / nthreads; smallperfrom--; ) {
      barrier();
      for (u32 bigbkt = 0; bigbkt < NBUCKETS; bigbkt++) {
        int64_t big1 = (int64_t)bigbkt << DEG2SMALLBITS;
#if NSIPHASH == 8
        __m256i vbig  = {big1, big1, big1, big1};
#endif
        // assert(down[id][bigbkt] > up[id][bigbkt]);
        u8 *readedge = src->base + src->start(id, bigbkt);
        u32 edge = 0;
#if NSIPHASH == 8
        u32 edge2 = 0, prevedge2;
        for (; ; readedge+=NSIPHASH*BIGSIZE) {
          prevedge2 = edge2;
// bit          39..13    12..0
// read           edge   degree
          re0 = *(u64 *)(readedge+0*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re0 & BIGSIZEMASK) >> DEGREEBITS, re0 & DEGREEMASK );
          v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
          e0 = edge2 += ((re0>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re1 = *(u64 *)(readedge+1*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re1 & BIGSIZEMASK) >> DEGREEBITS, re1 & DEGREEMASK );
          v0 = _mm256_permute4x64_epi64(vinit, 0x00);
          e1 = edge2 += ((re1>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re2 = *(u64 *)(readedge+2*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re2 & BIGSIZEMASK) >> DEGREEBITS, re2 & DEGREEMASK );
          v1 = _mm256_permute4x64_epi64(vinit, 0x55);
          e2 = edge2 += ((re2>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re3 = *(u64 *)(readedge+3*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re3 & BIGSIZEMASK) >> DEGREEBITS, re3 & DEGREEMASK );
          v2 = _mm256_permute4x64_epi64(vinit, 0xAA);
          e3 = edge2 += ((re3>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          vpacket0 = _mm256_set_epi64x(e3, e2, e1, e0);
          vhi0     = _mm256_set_epi64x(re3, re2, re1, re0);
          re0 = *(u64 *)(readedge+4*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re0 & BIGSIZEMASK) >> DEGREEBITS, re0 & DEGREEMASK );
          v7 = _mm256_permute4x64_epi64(vinit, 0xFF);
          e0 = edge2 += ((re0>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re1 = *(u64 *)(readedge+5*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re1 & BIGSIZEMASK) >> DEGREEBITS, re1 & DEGREEMASK );
          v4 = _mm256_permute4x64_epi64(vinit, 0x00);
          e1 = edge2 += ((re1>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re2 = *(u64 *)(readedge+6*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re2 & BIGSIZEMASK) >> DEGREEBITS, re2 & DEGREEMASK );
          v5 = _mm256_permute4x64_epi64(vinit, 0x55);
          e2 = edge2 += ((re2>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
          re3 = *(u64 *)(readedge+7*BIGSIZE);
// printf("edge27 %07x degree1  %04x\n", (re3 & BIGSIZEMASK) >> DEGREEBITS, re3 & DEGREEMASK );
          v6 = _mm256_permute4x64_epi64(vinit, 0xAA);
          e3 = edge2 += ((re3>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;

          if (edge2 >= 2*NEDGES) // less than 8 edges from current bucket
            break;

          vpacket1 = _mm256_set_epi64x(e3, e2, e1, e0);
          vhi1     = _mm256_set_epi64x(re3, re2, re1, re0);
    
          v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
          SIPROUNDX8; SIPROUNDX8;
          v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
          v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
          v6 = XOR(v6,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
          SIPROUNDX8; SIPROUNDX8; SIPROUNDX8; SIPROUNDX8;
          v0 = XOR(XOR(v0,v1),XOR(v2,v3));
          v4 = XOR(XOR(v4,v5),XOR(v6,v7));
    
          v1 = v0 & vbucketmask;
          v5 = v4 & vbucketmask;
          v0 = vbig | _mm256_slli_epi64(_mm256_srli_epi64(v0 & vnodemask, BUCKETBITS), DEGREEBITS) | (vhi0 & vdegmask);
          v4 = vbig | _mm256_slli_epi64(_mm256_srli_epi64(v4 & vnodemask, BUCKETBITS), DEGREEBITS) | (vhi1 & vdegmask);

#define STORE(i,v,x,w) \
z = _mm256_extract_epi32(v,x);\
*(u64 *)(big0+big[z]) = _mm256_extract_epi64(w,i%4);\
big[z] += BIGSIZE;

          STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
          STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
        }
        edge = prevedge2 / 2;
#endif
        for (; ; readedge += BIGSIZE) { // process up to 7 leftover edges
// bit         39..13     12..0
// read          edge   degree1 within big1
          u64 e = *(u64 *)readedge & BIGSIZEMASK;
          edge += ((e>>DEGREEBITS) - edge) & (BIGSIZEMASK >> DEGREEBITS);
          if (edge >= NEDGES) break; // reached end of small1 section
          u32 node = _sipnode(&sip_keys, edge, uorv);
          z = node & BUCKETMASK;
// bit        39..34    33..26     25..13     12..0
// write        big1    small0    degree0   degree1   within big0
          *(u64 *)(big0+big[z]) = big1 | ((u64)(node >> BUCKETBITS) << DEGREEBITS) | (e & DEGREEMASK);
// printf("%010lx   big1 %02x small0 %02x degree0 %04x degree1 %04x\n", *(u64 *)(big0+big[z]),   big1>>34, node>>21, (node>>8)&DEGREEMASK, e & DEGREEMASK );
          big[z] += BIGSIZE;
        }
        down[id][bigbkt] = readedge - big0;
      }
    }
    rdtsc1 = __rdtsc();
    printf("genVnodes rdtsc: %lu sumsize %d\n", rdtsc1-rdtsc0, upsumsize(id));
  }
  bool power2(u32 n) {
    return (n & (n-1)) == 0;
  }
  void trimedgesU(indexer *src, indexer *dst, const u32 id) {
    u64 rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 edge = from * NEDGES / nthreads;
        u8    *readbig = src->base + src->start(from, bigbkt);
        u8 *endreadbig = src->base + src->end(from, bigbkt);
        for (; readbig < endreadbig; readbig += BIG0SIZE) {
// bit     39/31..21     20..8      7..0
// read         edge   degree1    small1   within big1
          BIGTYPE e = *(BIGTYPE *)readbig;
#if BIG0SIZE > 4
          e &= BIG0SIZEMASK;
#else
          if (unlikely(!e)) { edge += NEDGES0LO; continue; }
#endif
          edge += ((u32)(e>>DEGSMALLBITS) - edge) & (NEDGES0LO-1);
          u32 z = e & BUCKETMASK;
// bit         39..13     12..0
// write         edge   degree1   within big1 small1
          *(u64 *)(small0+small[z]) = ((u64)edge << DEGREEBITS) | (e >> BUCKETBITS);
          small[z] += SMALLSIZE;
        }
        if (power2(nthreads) && unlikely(edge >> EDGE0BITSLO !=
          ((from+1)*NEDGES/nthreads - 1) >> EDGE0BITSLO))
        { printf("OOPS1: bkt %d edge %x vs %x\n", bigbkt, edge, (from+1)*(u32)NEDGES/nthreads-1); exit(0); }
      }
      u8 *degs = src->base + bucketstart(0, bigbkt); // recycle!
      for (u32 from = 0 ; from < nthreads; from++) {
        u8 *writedge = dst->base + dst->start(from, bigbkt);
        int smallbkt = from * NBUCKETS / nthreads;
        int endsmallbkt = (from+1) * NBUCKETS / nthreads;
        for (; smallbkt < endsmallbkt; smallbkt++) {
          memset(degs, 1, NDEGREES);
          u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
             *endreadsmall = small0 + small[smallbkt], *rdsmall;
          for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE)
            degs[*(u32 *)rdsmall & DEGREEMASK]--;
          for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE) {
// bit         39..13     12..0
// read          edge   degree1    within big1 small1
            u64 z = *(u64 *)rdsmall;
            *(u64 *)writedge = z;
            writedge += degs[z & DEGREEMASK] ? BIGSIZE : 0;
          }
        }
        dst->index[from][bigbkt] = writedge - dst->base;
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimedgesU rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  void trimup(const u32 id, u32 round) {
    u64 rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u8 *big0 = buckets[0];
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    u32 *big = downinit(id);
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 smallbig1 = (from * NBUCKETS / nthreads) << BUCKETBITS;
        u8    *readbig = big0 + bucketstart(from, bigbkt);
        u8 *endreadbig = big0 + up[from][bigbkt];
        for (; readbig < endreadbig; readbig += BIGSIZE) {
// bit        39..34    33..26     25..13     12..0
// read         big1    small0    degree0   degree1   within big0
          u64 e = *(u64 *)readbig & BIGSIZEMASK;
          smallbig1 += ((u32)(e>>DEG2SMALLBITS) - smallbig1) & SMALLPREFMASK;
          u32 z = (e >> DEG2BITS) & BUCKETMASK;
// if (round>1) printf("%010lx big1 %02x small0 %02x degree0 %04x degree1 %04x\n", e, (u32)(e>>DEG2SMALLBITS), z, (e>>DEGREEBITS) & DEGREEMASK, e & DEGREEMASK );
// bit        39..34    33..26     25..13     12..0
// write      small1      big1    degree0   degree1   within big0 small0
          *(u64 *)(small0+small[z]) = ((u64)smallbig1 << DEG2BITS) | (e & DEG2MASK);
          small[z] += SMALLSIZE;
        }
        if (power2(nthreads) && unlikely(smallbig1/NBUCKETS != (from+1)*NBUCKETS/nthreads - 1))
        { printf("OOPS2: id %d bkt %d from %d smallbig1 %x vs %x\n", id, bigbkt, from, smallbig1/NBUCKETS, (from+1)*NBUCKETS/nthreads - 1); }
      }
      u8 *degs = (u8 *)big0 + bucketstart(0, bigbkt); // recycle!
      for (u32 smallbkt = 0; smallbkt < NBUCKETS; smallbkt++) {
        memset(degs, 1, NDEGREES);
        u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
           *endreadsmall = small0 + small[smallbkt], *rdsmall;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SMALLSIZE)
          degs[(*(u32 *)rdsmall >> DEGREEBITS) & DEGREEMASK]--;
        u32 small1 = 0;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SMALLSIZE) {
// bit        39..34    33..26     25..13     12..0
// write      small1      big1    degree0   degree1   within big0 small0
          u64 e = *(u64 *)rdsmall; // & SMALLSIZEMASK;
          small1 += ((u32)(e>>DEG2SMALLBITS) - small1) & SMALLPREFMASK;
// bit        39..34    33..26     25..13     12..0
// write      small0      big1    degree0   degree1   within small1
          *(u64 *)(big0+big[small1]) = (((u64)smallbkt << DEG2SMALLBITS) | (e & DEG2SMALLMASK)) << (64 - BIGBITS);
          big[small1] -= degs[(e >> DEGREEBITS) & DEGREEMASK] ? BIGSIZE : 0; // backwards
        }
        if (power2(nthreads) && unlikely(small1>>SMALLPREFBITS != BUCKETMASK>>SMALLPREFBITS))
        { printf("OOPS3: bkt %d small1 %x vs %x\n", bigbkt, small1, BUCKETMASK); }
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimup rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  void trimdown(const u32 id, const u32 round) {
    u64 rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u8 *big0 = buckets[0];
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    u32 *big = upinit(id);
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 bigsmall0 = (from * NBUCKETS / nthreads) << BUCKETBITS;
        u8    *readbig = big0 + bucketend(from, bigbkt) + 3;
        u8 *endreadbig = big0 + down[from][bigbkt] + 3;
        for (; readbig > endreadbig; readbig -= BIGSIZE) {
// bit        39..34    33..26     25..13     12..0
// write      small0      big1    degree0   degree1   within small1
          u64 e = *(u64 *)readbig & BIGSIZEMASK;
          bigsmall0 += ((u32)(e>>DEG2SMALLBITS) - bigsmall0) & SMALLPREFMASK;
          u32 z = (e >> DEG2BITS) & BUCKETMASK;
// if (round>10) printf("%010lx small0 %02x big1 %02x degree0 %04x degree1 %04x\n", e, (u32)(e>>DEG2SMALLBITS), z, (e>>DEGREEBITS) & DEGREEMASK, e & DEGREEMASK );
// bit        39..34    33..26     25..13     12..0
// write        big0    small0    degree0   degree1   within small1 big1
          *(u64 *)(small0+small[z]) = ((u64)bigsmall0 << DEG2BITS) | (e & DEG2MASK);
          small[z] += SMALLSIZE;
        }
        if (power2(nthreads) && unlikely(bigsmall0/NBUCKETS != (from+1)*NBUCKETS/nthreads - 1))
        { printf("OOPS4: id %d bkt %d from %d bigsmall0 %x vs %x\n", id, bigbkt, from, bigsmall0/NBUCKETS, (from+1)*NBUCKETS/nthreads - 1); }
      }
      u8 *degs = (u8 *)big0 + bucketend(0, bigbkt) - NDEGREES; // recycle!
      for (u32 smallbkt = 0; smallbkt < NBUCKETS; smallbkt++) {
        memset(degs, 1, NDEGREES);
        u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
           *endreadsmall = small0 + small[smallbkt], *rdsmall;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SMALLSIZE)
          degs[(*(u32 *)rdsmall) & DEGREEMASK]--;
        u32 rebig0 = 0;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SMALLSIZE) {
// bit        39..34    33..26     25..13     12..0
// write        big0    small0    degree0   degree1   within small1 big1
          u64 e = *(u64 *)rdsmall & BIGSIZEMASK;
          rebig0 += ((u32)(e>>DEG2SMALLBITS) - rebig0) & SMALLPREFMASK;
// if (bigbkt==164 && smallbkt==215) printf("%010lx big0 %02x small0 %02x degree0 %04x degree1 %04x\n", e, (u32)(e>>DEG2SMALLBITS), (u32)(e>>DEG2BITS) & BUCKETBITS, (e>>DEGREEBITS) & DEGREEMASK, e & DEGREEMASK );
// bit        39..34    33..26     25..13     12..0
// read         big1    small0    degree0   degree1   within big0
          *(u64 *)(big0+big[rebig0]) = (((u64)smallbkt << DEG2SMALLBITS) | (e & DEG2SMALLMASK));
          big[rebig0] += degs[(e) & DEGREEMASK] ? BIGSIZE : 0;
        }
        if (power2(nthreads) && unlikely(rebig0>>SMALLPREFBITS != BUCKETMASK>>SMALLPREFBITS))
        { printf("OOPS5: bkt %d %d small1 %x vs %x\n", bigbkt, smallbkt, rebig0, BUCKETMASK); }
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimdown rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  void trim() {
    if (nthreads == 1) {
      trimmer(0);
      return;
    }
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
  void barrier() {
    int rc = pthread_barrier_wait(&barry);
    assert(rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD);
  }
  void trimmer(u32 id) {
    genUnodes(idxA, id, 1);
    barrier();
    trimedgesU(idxA, idxB, id);
    barrier();
    if (id == 0)
      printf("round 0 edges %d\n", idxB->sumsize()/BIGSIZE);
    barrier();
    genVnodes(idxB, idxA, id, 0);
    barrier();
    for (u32 round=1; round <= ntrims; round++) {
      if (id == 0) {
        printf("round %2d nodes %d\n", round, upsumsize()/BIGSIZE);
        for (u32 id=1; id < nthreads; id++)
          for (u32 bkt=0; bkt < NBUCKETS/8; bkt++)
            printf("%d %3d %d%c", id, bkt, upsize(id, bkt)/BIGSIZE, (bkt&3)==3 ? '\n' : ' ');
      }
      barrier();
      trimup(id, round);
      barrier();
      if (id == 0) {
        printf("round %2d nodes %d\n", round, downsumsize()/BIGSIZE);
        for (u32 id=1; id < nthreads; id++)
          for (u32 bkt=0; bkt < NBUCKETS/8; bkt++)
            printf("%d %3d %d%c", id, bkt, downsize(id, bkt)/BIGSIZE, (bkt&3)==3 ? '\n' : ' ');
      }
      barrier();
      trimdown(id, round);
      barrier();
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
// we want sizeof(cuckoo_hash) < sizeof(trimmer), so
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * sizeof(u32)
// CUCKOO_SIZE * 8             < NEDGES * 4
// (NNODES >> IDXSHIFT) * 2    < NEDGES
// IDXSHIFT                    > 2
#define IDXSHIFT 8
#endif

#define NODEBITS (EDGEBITS + 1)

// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << (NODEBITS/3);

const static u32 CUCKOO_SIZE = NEDGES >> (IDXSHIFT-1);
const static u32 CUCKOO_MASK = CUCKOO_SIZE - 1;
// number of (least significant) key bits that survives leftshift by NODEBITS
const static u32 KEYBITS = 64-NODEBITS;
const static u64 KEYMASK = (1LL << KEYBITS) - 1;
const static u64 MAXDRIFT = 1LL << (KEYBITS - IDXSHIFT);
const static u32 NODEMASK = 2 * NEDGES - 1;

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
        return (u32)(cu & NODEMASK);
      }
    }
  }
};

class solver_ctx {
public:
  edgetrimmer *trimmer;
  cuckoo_hash *cuckoo;
  u32 sols[MAXSOLS][PROOFSIZE];
  u32 nsols;

  solver_ctx(u32 n_threads, u32 n_trims) {
    trimmer = new edgetrimmer(n_threads, n_trims);
    cuckoo = 0;
  }
  void setheadernonce(char* headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer->sip_keys);
    nsols = 0;
  }
  ~solver_ctx() {
    delete cuckoo;
    delete trimmer;
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
      u64 alive64 = 0; // trimmer->block(block);
      for (u32 nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        edge e(sipnode(&trimmer->sip_keys, nonce, 0), sipnode(&trimmer->sip_keys, nonce, 1));
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
    trimmer->trim();
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];
    for (u32 block = 0; block < NEDGES; block += 64) {
      u64 alive64 = 0; // trimmer->block(block);
      for (u32 nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        u32 u0=sipnode(&trimmer->sip_keys, nonce, 0), v0=sipnode(&trimmer->sip_keys, nonce, 1);
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
