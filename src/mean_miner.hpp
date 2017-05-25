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

// break circular reference with forward declaration
class edgetrimmer;

typedef struct {
  u32 id;
  pthread_t thread;
  edgetrimmer *et;
} thread_ctx;

#define likely(x)   __builtin_expect((x)!=0, 1)
#define unlikely(x)   __builtin_expect((x), 0)

static const u32 SENTINELSIZE = NSIPHASH * BIGSIZE; // sentinel for bigsort1;

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  indices *up;
  indices *down;
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
    down    = new indices[n_threads];
    threads  = new thread_ctx[nthreads];
    int err  = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
    for (u32 id=0; id < nthreads; id++)
      for (u32 bkt=0; bkt < NBUCKETS; bkt++)
        memset(buckets[0] + nodeend(id,bkt), 0, SENTINELSIZE);
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] up;
    delete[] down;
    delete[] threads;
  }
  u32 nodestart(u32 id, u32 bkt) {
    return bkt * BIGBUCKETSIZE + id * BIGBUCKETSIZE / nthreads;
  }
  u32 nodeend(u32 id, u32 bkt) {
    return nodestart(id+1, bkt) - SENTINELSIZE;
  }
  u32 *nodeinit(u32 id) {
   u32 *nds = up[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      nds[bkt] = nodestart(id,bkt);
    return nds;
  }
  u32 *edgeinit(u32 id) {
   u32 *eds = down[id];
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      eds[bkt] = nodeend(id,bkt);
    return eds;
  }
  u32 edgesize(u32 id, u32 bkt) {
    return nodeend(id, bkt) - down[id][bkt];
  }
  u32 edgesumsize() {
    u32 sum = 0;
    for (u32 id=0; id < nthreads; id++)
      for (u32 bkt=0; bkt < NBUCKETS; bkt++)
        sum += edgesize(id, bkt);
    return sum;
  }
  u32 nodesize(u32 id, u32 bkt) {
    return up[id][bkt] - nodestart(id,bkt);
  }
  u32 nodesumsize(u32 id) {
    u32 sum = 0;
    for (u32 bkt=0; bkt < NBUCKETS; bkt++)
      sum += nodesize(id, bkt);
    return sum;
  }
  u32 nodesumsize() {
    u32 sum = 0;
    for (u32 id=0; id < nthreads; id++)
      sum += nodesumsize(id);
    return sum;
  }
  void sortbig0(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NBUCKETS];
#endif
  
    rdtsc0 = __rdtsc();
    u32 z, *big = nodeinit(id);
    u8 *big0 = buckets[0];
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
// store        edge    degree  smallbkt
#if NSIPHASH == 1
      u32 node = _sipnode(&sip_keys, block, uorv);
      z = node & BUCKETMASK;
      u64 zz = (u64)block << DEGSMALLBITS | (node & EDGEMASK) >> BUCKETBITS;
#ifndef NEEDSYNC
      *(u64 *)(big0+big[z]) = zz;
      big[z] += BIG0SIZE;
#else
      if (zz) {
        for (; unlikely(last[z] + NEDGES0LO <= block); last[z] += NEDGES0LO, big[z] += BIG0SIZE)
          *(u32 *)(big0+big[z]) = 0;
        *(u32 *)(big0+big[z]) = zz;
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
  *(u64 *)(big0+big[z]) = _mm256_extract_epi64(w,i%4);\
  big[z] += BIG0SIZE;
#else
#define STORE0(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    z = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[z] + NEDGES0LO <= block+i); last[z] += NEDGES0LO, big[z] += BIG0SIZE)\
      *(u32 *)(big0+big[z]) = 0;\
    *(u32 *)(big0+big[z]) = zz;\
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
      assert(nodesize(id, z) <= BIGBUCKETSIZE);
#ifdef NEEDSYNC
      for (; last[z]<endblock-NEDGES0LO; last[z]+=NEDGES0LO) {
        *(u32 *)(big0+big[z]) = 0;
        big[z] += BIG0SIZE;
      }
#endif
    }
    rdtsc1 = __rdtsc();
    printf("sortbig0 rdtsc: %lu sumsize %x\n", rdtsc1-rdtsc0, nodesumsize(id));
  }
  void sortbig1(const u32 id, const u32 uorv) {
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
    u32 z, *big = nodeinit(id);
    u8 *big0 = buckets[0];
    u32 endbkt = (id+1)*NBUCKETS/nthreads; 
    for (u32 small=0; small<NBUCKETS; small++) {
      int64_t small1 = (int64_t)small << 34;
#if NSIPHASH == 8
      __m256i vsmall  = {small1, small1, small1, small1};
#endif
      barrier();
      for (u32 bigbkt = id*NBUCKETS/nthreads; bigbkt < endbkt; bigbkt++) {
        for (u32 from = 0 ; from < nthreads; from++) {
          assert(down[from][bigbkt] > up[from][bigbkt]);
          u8 *readedge = big0 + down[from][bigbkt];
          u32 edge = 0;
#if NSIPHASH == 8
          u32 edge2 = 0, prevedge2;
          for (; ; readedge+=NSIPHASH*BIGSIZE) {
            prevedge2 = edge2;
// bit          39..13    12..0
// read           edge   degree
            re0 = *(u64 *)(readedge+0*BIGSIZE);
            v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
            e0 = edge2 += ((re0>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re1 = *(u64 *)(readedge+1*BIGSIZE);
            v0 = _mm256_permute4x64_epi64(vinit, 0x00);
            e1 = edge2 += ((re1>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re2 = *(u64 *)(readedge+2*BIGSIZE);
            v1 = _mm256_permute4x64_epi64(vinit, 0x55);
            e2 = edge2 += ((re2>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re3 = *(u64 *)(readedge+3*BIGSIZE);
            v2 = _mm256_permute4x64_epi64(vinit, 0xAA);
            e3 = edge2 += ((re3>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            vpacket0 = _mm256_set_epi64x(e3, e2, e1, e0);
            vhi0     = _mm256_set_epi64x(re3, re2, re1, re0);
            re0 = *(u64 *)(readedge+4*BIGSIZE);
            v7 = _mm256_permute4x64_epi64(vinit, 0xFF);
            e0 = edge2 += ((re0>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re1 = *(u64 *)(readedge+5*BIGSIZE);
            v4 = _mm256_permute4x64_epi64(vinit, 0x00);
            e1 = edge2 += ((re1>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re2 = *(u64 *)(readedge+6*BIGSIZE);
            v5 = _mm256_permute4x64_epi64(vinit, 0x55);
            e2 = edge2 += ((re2>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            re3 = *(u64 *)(readedge+7*BIGSIZE);
            v6 = _mm256_permute4x64_epi64(vinit, 0xAA);
            e3 = edge2 += ((re3>>DEGREEBITS1) - edge2) & EVENNONDEGMASK;
            if (edge2 >= 2*NEDGES) { edge = prevedge2/2; break; }
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
            v0 = vsmall | _mm256_slli_epi64(_mm256_srli_epi64(v0 & vnodemask, BUCKETBITS), 2*DEGREEBITS) | (vhi0 & vdegmask);
            v4 = vsmall | _mm256_slli_epi64(_mm256_srli_epi64(v4 & vnodemask, BUCKETBITS), 2*DEGREEBITS) | (vhi1 & vdegmask);

#define STORE(i,v,x,w) \
  z = _mm256_extract_epi32(v,x);\
  *(u64 *)(big0+big[z]) = _mm256_extract_epi64(w,i%4);\
  big[z] += BIGSIZE;

            STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
            STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
          }
#endif
          for (; ; readedge += BIGSIZE) {
  // bit         39..13     12..0
  // read          edge   degree1
            u64 e = *(u64 *)readedge & BIGSIZEMASK;
            edge += ((e>>DEGREEBITS) - edge) & (BIGSIZEMASK >> DEGREEBITS);
            if (edge >= NEDGES) break; // reached end of small1 section
            u32 node = _sipnode(&sip_keys, edge, uorv);
            z = node & BUCKETMASK;
  // bit        39..34    33..21     25..13     12..0
  // store      small1    small0    degree0   degree1   within big0
            *(u64 *)(big0+big[z]) = small1 | ((u64)((node & EDGEMASK) >> BUCKETBITS) << (2*DEGREEBITS)) | (e & DEGREEMASK);
            big[z] += BIGSIZE;
          }
          down[from][bigbkt] = readedge - big0;
        }
      }
    }
    rdtsc1 = __rdtsc();
    printf("sortbig1 rdtsc: %lu sumsize %d\n", rdtsc1-rdtsc0, nodesumsize(id));
  }
// bit        39..34    33..26     25..13     12..0
// store        big1    small1    degree0   degree1   within big0 small0
  bool power2(u32 n) {
    return (n & (n-1)) == 0;
  }
  void trimup0(const u32 id) {
    u64 rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u8 *big0 = buckets[0];
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 lastread = from * NEDGES / nthreads;
        u8    *readbig = big0 + nodestart(from, bigbkt);
        u8 *endreadbig = big0 + up[from][bigbkt];
        for (; readbig < endreadbig; readbig += BIG0SIZE) {
// bit     39/31..21     20..8      7..0
// read         edge    degree  smallbkt
          BIGTYPE e = *(BIGTYPE *)readbig;
#if BIG0SIZE > 4
          e &= BIG0SIZEMASK;
#else
          if (unlikely(!e)) { lastread += NEDGES0LO; continue; }
#endif
          lastread += ((u32)(e>>DEGSMALLBITS) - lastread) & (NEDGES0LO-1);
          u32 z = e & BUCKETMASK;
// bit         39..13     12..0
// write         edge    degree
          *(u64 *)(small0+small[z]) = ((u64)lastread << DEGREEBITS) | (e >> BUCKETBITS);
          small[z] += SMALLSIZE;
        }
        if (power2(nthreads) && unlikely(lastread >> EDGE0BITSLO !=
          ((from+1)*NEDGES/nthreads - 1) >> EDGE0BITSLO))
        { printf("OOPS1: bkt %d lastread %x vs %x\n", bigbkt, lastread, (from+1)*(u32)NEDGES/nthreads-1); exit(0); }
      }
      u8 *degs = (u8 *)big0 + nodestart(0, bigbkt); // recycle!
      for (u32 from = 0 ; from < nthreads; from++) {
        u8 *writedge = big0 + nodeend(from, bigbkt) - 8;
        int smallbkt0 = from * NBUCKETS / nthreads;
        int smallbkt = (from+1) * NBUCKETS / nthreads;
        for (; --smallbkt >= smallbkt0; ) { // backwards
          memset(degs, 1, NDEGREES);
          u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
             *endreadsmall = small0 + small[smallbkt], *rdsmall;
          for (rdsmall = endreadsmall; (rdsmall-=SMALLSIZE) >= readsmall; ) // backwards
            degs[*(u32 *)rdsmall & DEGREEMASK]--;
          for (rdsmall = endreadsmall; (rdsmall-=SMALLSIZE) >= readsmall; ) { // backwards
// bit         39..13     12..0
// read          edge    degree
            u64 z = *(u64 *)rdsmall; // & SMALLSIZEMASK;
// bit         63..37    36..24    23..0
// write         edge    degree   fodder
            *(u64 *)writedge = z << (8 * (sizeof(u64)-BIGSIZE));
            writedge -= degs[z & DEGREEMASK] ? BIGSIZE : 0; // backwards
          }
        }
        down[from][bigbkt] = (u8 *)writedge + 8 - big0;
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimup0 rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  void trimup(const u32 id) {
    u64 rdtsc0, rdtsc1;
    u32 small[NBUCKETS];
  
    rdtsc0 = __rdtsc();
    u8 *big0 = buckets[0];
    u8 *small0 = tbuckets[id*NBUCKETS];
    u32 bigbkt = id*NBUCKETS/nthreads, endbkt = (id+1)*NBUCKETS/nthreads; 
    u32 *big = edgeinit(id);
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NBUCKETS; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 lastread = 0;
        u8    *readbig = big0 + nodestart(from, bigbkt);
        u8 *endreadbig = big0 + up[from][bigbkt];
        for (; readbig < endreadbig; readbig += BIGSIZE) {
  // bit        39..34    33..21     25..13     12..0
  // read       small1    small0    degree0   degree1   within big0
          u64 e = *(u64 *)readbig;
          lastread += ((u32)(e>>DEGSMALLBITS) - lastread) & (NEDGES0LO-1);
          u32 z = e & BUCKETMASK;
          *(u64 *)(small0+small[z]) = ((u64)lastread << DEGREEBITS) | (e >> BUCKETBITS);
          small[z] += SMALLSIZE;
        }
        if (power2(nthreads) && unlikely(lastread >> EDGE0BITSLO !=
          ((from+1)*NEDGES/nthreads - 1) >> EDGE0BITSLO))
        { printf("OOPS1: bkt %d lastread %x vs %x\n", bigbkt, lastread, (from+1)*(u32)NEDGES/nthreads-1); exit(0); }
      }
      u8 *degs = (u8 *)big0 + nodestart(0, bigbkt); // recycle!
      for (u32 from = 0 ; from < nthreads; from++) {
        u8 *writedge = big0 + nodeend(from, bigbkt) - 8;
        int smallbkt0 = from * NBUCKETS / nthreads;
        int smallbkt = (from+1) * NBUCKETS / nthreads;
        for (; --smallbkt >= smallbkt0; ) { // backwards
          memset(degs, 1, NDEGREES);
          u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
             *endreadsmall = small0 + small[smallbkt], *rdsmall;
          for (rdsmall = endreadsmall; (rdsmall-=SMALLSIZE) >= readsmall; ) // backwards
            degs[*(u32 *)rdsmall & DEGREEMASK]--;
          for (rdsmall = endreadsmall; (rdsmall-=SMALLSIZE) >= readsmall; ) { // backwards
            u64 z = *(u64 *)rdsmall; // & SMALLSIZEMASK;
            *(u64 *)(big0+big[z]) = z;
            // << (8 * (sizeof(u64)-BIGSIZE));
          }
        }
      }
    }
    rdtsc1 = __rdtsc();
    printf("trimup rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  void trimdown(const u32 id) {
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
  void barrier() {
    int rc = pthread_barrier_wait(&barry);
    assert(rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD);
  }
  void trimmer(u32 id) {
    sortbig0(id, 1);
    barrier();
    trimup0(id);
    barrier();
    if (id == 0)
      printf("round 0 edges %d\n", edgesumsize()/BIGSIZE);
    barrier();
    sortbig1(id, 0);
    barrier();
    for (u32 round=1; round <= ntrims; round++) {
      if (id == 0) {
        printf("round %2d nodes %d\n", round, nodesumsize()/BIGSIZE);
        for (u32 id=0; id < nthreads; id++)
          for (u32 bkt=0; bkt < NBUCKETS/8; bkt++)
            printf("%d %3d %d%c", id, bkt, nodesize(id, bkt)/BIGSIZE, (bkt&3)==3 ? '\n' : ' ');
      }
      barrier();
      trimup(id);
      barrier();
      trimdown(id);
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
