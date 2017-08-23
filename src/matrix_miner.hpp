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

// EDGEBITS/NEDGES/EDGEMASK defined in cuckoo.h

// The node bits (for U nodes of odd parity) are logically split into 3 groups, eg
// 8 X bits (most significant), 8 Y bits, and 13 Z bits (least significant)
// edgebits   XXXXXXXX YYYYYYYY ZZZZZZZZZZZZZ
// bit%10     87654321 09876543 2109876543210
// bit/10     22222222 11111111 1110000000000
// In each trimming round, edges are first partitioned on X value,
// then on Y value, and finally repeated edges are identified within
// each XY partition by counting how many times each Z value occurs.

#ifndef XBITS
// 8 seems optimal for 2-4 threads
#define XBITS 8
#endif

#ifndef YBITS
#define YBITS XBITS
#endif

#ifndef MAXSOLS
// more than one 42-cycle is already exceedingly rare
#define MAXSOLS 4
#endif

// size in bytes of a big bucket entry
#define BIGSIZE 5
// size in bytes of a small bucket entry
#define SMALLSIZE 5

// initial entries could be smaller at percent or two slowdown
#ifndef BIGSIZE0
#if EDGEBITS < 30
#define BIGSIZE0 4
#else
#define BIGSIZE0 5
#endif
#endif
// but they'll need syncing entries
#if BIGSIZE0 == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

// size expanding round; must be even
#ifndef EXPANDROUND
#define EXPANDROUND 18
#endif
// size in bytes of a big bucket entry after EXPANDROUND rounds
#ifndef BIGGERSIZE
#define BIGGERSIZE 6
#endif

#if EDGEBITS >= 30
typedef u64 offset_t;
#else
typedef u32 offset_t;
#endif

#if BIGSIZE0 > 4
typedef u64 BIGTYPE;
#else
typedef u32 BIGTYPE;
#endif

// node bits have two groups of bucketbits (big and small) and a remaining group of degree bits
const static u32 NX        = 1 << XBITS;
const static u32 XMASK     = NX - 1;
const static u32 NY        = 1 << YBITS;
const static u32 YMASK     = NY - 1;
const static u32 XYBITS    = XBITS + YBITS;
const static u32 ZBITS     = EDGEBITS - XYBITS;
const static u32 NZ        = 1 << ZBITS;
const static u32 ZMASK     = NZ - 1;
const static u32 YZBITS    = YBITS + ZBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 YZMASK    = NYZ - 1;
const static u32 YZZBITS   = YZBITS + ZBITS;

const static u32 BIGSLOTBITS   = BIGSIZE * 8;
const static u32 SMALLSLOTBITS = SMALLSIZE * 8;
const static u64 BIGSLOTMASK   = (1ULL << BIGSLOTBITS) - 1ULL;
const static u64 SMALLSLOTMASK = (1ULL << SMALLSLOTBITS) - 1ULL;
const static u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const static u64 BIGSLOTMASK0  = (1ULL << BIGSLOTBITS0) - 1ULL;
const static u32 EDGEBITSLO   = BIGSLOTBITS0 - YZBITS;
const static u32 NEDGESLO     = 1 << EDGEBITSLO;

// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.

typedef uint8_t u8;
typedef uint16_t u16;

// 1/16 reduces odds of overflowing z bucket on 2^30 nodes to 2^16*e^-32 (less than 1 on a billion)
#ifndef BIGEPS
#define BIGEPS 5/64
#endif

// safely over 1-e(-1) trimming fraction
#ifndef TRIMFRAC256
#define TRIMFRAC256 184
#endif

const static u32 NTRIMMEDZ  = NZ * TRIMFRAC256 / 256;
typedef u8 zbucket8[NZ];
typedef u16 zbucket16[NTRIMMEDZ];
typedef u32 zbucket32[NTRIMMEDZ];

const static u32 ZBUCKETSLOTS = NZ + NZ * BIGEPS;
#ifdef SHOWCYCLE
const static u32 ZBUCKETSIZE = NTRIMMEDZ * (BIGSIZE + sizeof(u32));  // assumes EDGEBITS <= 32
#else
const static u32 ZBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE0; 
#endif
const static u32 TBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE; 

template<u32 BUCKETSIZE>
struct zbucket {
  u32 size;
  u8 bytes[BUCKETSIZE];
  zbucket() {
    size = 0;
  }
  u32 setsize(u8 * end) {
    size = end - bytes;
    assert(size <= BUCKETSIZE);
    return size;
  }
};

template<u32 BUCKETSIZE>
using yzbucket = zbucket<BUCKETSIZE>[NY];
template <u32 BUCKETSIZE>
using xyzbucket = yzbucket<BUCKETSIZE>[NX];

template<u32 BUCKETSIZE>
struct indexer {
  offset_t index[NX];

  indexer() {
  }
  void matrixv(const u32 y) {
    yzbucket<BUCKETSIZE> *foo = 0;
    for (u32 x = 0; x < NX; x++)
      index[x] = foo[x][y].bytes - (u8 *)foo;
  }
  offset_t storev(yzbucket<BUCKETSIZE> *buckets, const u32 y) {
    u8 *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 x = 0; x < NX; x++)
      sumsize += buckets[x][y].setsize(base+index[x]);
    return sumsize;
  }
  void matrixu(const u32 x) {
    yzbucket<BUCKETSIZE> *foo = 0;
    for (u32 y = 0; y < NY; y++)
      index[y] = foo[x][y].bytes - (u8 *)foo;
  }
  offset_t storeu(yzbucket<BUCKETSIZE> *buckets, const u32 x) {
    u8 *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 y = 0; y < NY; y++)
      sumsize += buckets[x][y].setsize(base+index[y]);
    return sumsize;
  }
  ~indexer() {
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

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  yzbucket<ZBUCKETSIZE> *buckets;
  yzbucket<TBUCKETSIZE> *tbuckets;
  zbucket32 *tedges;
  zbucket16 *tzs;
  zbucket8 *tdegs;
  u32 *tcounts;
  u32 ntrims;
  u32 nthreads;
  thread_ctx *threads;
  pthread_barrier_t barry;

  void touch(u8 *p, offset_t n) {
    for (offset_t i=0; i<n; i+=4096)
      *(u32 *)(p+i) = 0;
  }
  edgetrimmer(const u32 n_threads, u32 n_trims) {
    assert(sizeof(xyzbucket<ZBUCKETSIZE>) == NX * sizeof(yzbucket<ZBUCKETSIZE>));
    assert(sizeof(xyzbucket<TBUCKETSIZE>) == NX * sizeof(yzbucket<TBUCKETSIZE>));
    nthreads = n_threads;
    ntrims   = n_trims;
    buckets  = new yzbucket<ZBUCKETSIZE>[NX];
    touch((u8 *)buckets, sizeof(xyzbucket<ZBUCKETSIZE>));
    tbuckets = new yzbucket<TBUCKETSIZE>[nthreads];
    touch((u8 *)tbuckets, nthreads * sizeof(yzbucket<TBUCKETSIZE>));
    threads  = new thread_ctx[nthreads];
#ifdef SHOWCYCLE
    tedges = 0;
#else
    tedges = new zbucket32[nthreads];
#endif
    tdegs = new zbucket8[nthreads];
    tzs = new zbucket16[nthreads];
    tcounts = new u32[nthreads];
    int err  = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] threads;
    delete[] tedges;
    delete[] tdegs;
    delete[] tcounts;
    delete[] tzs;
  }
  u32 count() const {
    u32 cnt = 0;
    for (u32 t = 0; t < nthreads; t++)
      cnt += tcounts[t];
    return cnt;
  }
  void genUnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NX];;
#endif
  
    rdtsc0 = __rdtsc();
    u8 *base = (u8 *)buckets;
    indexer<ZBUCKETSIZE> dst;
    u32 starty = NY *  id    / nthreads;
    u32   endy = NY * (id+1) / nthreads;
    u32 edge = starty * NYZ, endedge = edge + NYZ;
#if NSIPHASH == 8
    static const __m256i vxmask = {XMASK, XMASK, XMASK, XMASK};
    static const __m256i vyzmask = {YZMASK, YZMASK, YZMASK, YZMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    u32 e2 = 2 * edge + uorv;
    __m256i vpacket0 = _mm256_set_epi64x(e2+6, e2+4, e2+2, e2+0);
    __m256i vpacket1 = _mm256_set_epi64x(e2+14, e2+12, e2+10, e2+8);
    static const __m256i vpacketinc = {16, 16, 16, 16};
    u64 e1 = edge;
    __m256i vhi0 = _mm256_set_epi64x((e1+3)*NYZ, (e1+2)*NYZ, (e1+1)*NYZ, (e1+0)*NYZ);
    __m256i vhi1 = _mm256_set_epi64x((e1+7)*NYZ, (e1+6)*NYZ, (e1+5)*NYZ, (e1+4)*NYZ);
    static const __m256i vhiinc = {8*NYZ, 8*NYZ, 8*NYZ, 8*NYZ};
#endif
    offset_t sumsize = 0;
    for (u32 my = starty; my < endy; my++, endedge += NYZ) {
      dst.matrixv(my);
#ifdef NEEDSYNC
      for (u32 x=0; x < NX; x++)
        last[x] = edge;
#endif
      for (; edge < endedge; edge += NSIPHASH) {
// bit        28..21     20..13    12..0
// node       XXXXXX     YYYYYY    ZZZZZ
#if NSIPHASH == 1
        u32 node = _sipnode(&sip_keys, edge, uorv);
        u32 ux = node >> YZBITS;
        BIGTYPE zz = (BIGTYPE)edge << YZBITS | (node & YZMASK);
#ifndef NEEDSYNC
// bit        39..21     20..13    12..0
// write        edge     YYYYYY    ZZZZZ
        *(BIGTYPE *)(base+dst.index[ux]) = zz;
        dst.index[ux] += BIGSIZE0;
#else
        if (zz) {
          for (; unlikely(last[ux] + NEDGESLO <= edge); last[ux] += NEDGESLO, dst.index[ux] += BIGSIZE0)
            *(u32 *)(base+dst.index[ux]) = 0;
          *(u32 *)(base+dst.index[ux]) = zz;
          dst.index[ux] += BIGSIZE0;
          last[ux] = edge;
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
        v1 = _mm256_srli_epi64(v0, YZBITS) & vxmask;
        v5 = _mm256_srli_epi64(v4, YZBITS) & vxmask;
        v0 = (v0 & vyzmask) | vhi0;
        v4 = (v4 & vyzmask) | vhi1;
        vhi0 = _mm256_add_epi64(vhi0, vhiinc);
        vhi1 = _mm256_add_epi64(vhi1, vhiinc);

        u32 ux;
#ifndef NEEDSYNC
#define STORE0(i,v,x,w) \
  ux = _mm256_extract_epi32(v,x);\
  *(u64 *)(base+dst.index[ux]) = _mm256_extract_epi64(w,i%4);\
  dst.index[ux] += BIGSIZE0;
#else
  u32 zz;
#define STORE0(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    ux = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[ux] + NEDGESLO <= edge+i); last[ux] += NEDGESLO, dst.index[ux] += BIGSIZE0)\
      *(u32 *)(base+dst.index[ux]) = 0;\
    *(u32 *)(base+dst.index[ux]) = zz;\
    dst.index[ux] += BIGSIZE0;\
    last[ux] = edge+i;\
  }
#endif
        STORE0(0,v1,0,v0); STORE0(1,v1,2,v0); STORE0(2,v1,4,v0); STORE0(3,v1,6,v0);
        STORE0(4,v5,0,v4); STORE0(5,v5,2,v4); STORE0(6,v5,4,v4); STORE0(7,v5,6,v4);
#else
#error not implemented
#endif
      }
#ifdef NEEDSYNC
      for (u32 ux=0; ux < NX; ux++) {
        for (; last[ux]<endedge-NEDGESLO; last[ux]+=NEDGESLO) {
          *(u32 *)(base+dst.index[ux]) = 0;
          dst.index[ux] += BIGSIZE0;
        }
      }
#endif
      sumsize += dst.storev(buckets, my);
    }
    rdtsc1 = __rdtsc();
    printf("genUnodes id %d size %u rdtsc: %lu\n", id, sumsize/BIGSIZE0, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/BIGSIZE0;
  }

  void genVnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
  
#if NSIPHASH == 8
    static const __m256i vxmask = {XMASK, XMASK, XMASK, XMASK};
    static const __m256i vyzmask = {YZMASK, YZMASK, YZMASK, YZMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    __m256i vpacket0, vpacket1, vhi0, vhi1;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
#endif
    static const u32 NONDEGBITS = std::min(BIGSLOTBITS, 2 * YZBITS) - ZBITS;
    static const u32 NONDEGMASK = (1 << NONDEGBITS) - 1;
    indexer<ZBUCKETSIZE> src, dst;
    indexer<TBUCKETSIZE> small;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 *base = (u8 *)buckets;
    u8 *small0 = (u8 *)tbuckets[id];
    u32 startux = NX *  id    / nthreads;
    u32   endux = NX * (id+1) / nthreads;
    for (u32 ux = startux; ux < endux; ux++) { // matrix x == ux
      small.matrixu(0);
      for (u32 my = 0 ; my < NY; my++) {
        u32 edge = my * NYZ;
        u8    *readbig = buckets[ux][my].bytes;
        u8 *endreadbig = readbig + buckets[ux][my].size;
// printf("id %d x %d y %d size %u read %d\n", id, ux, my, buckets[ux][my].size, readbig-base);
        for (; readbig < endreadbig; readbig += BIGSIZE0) {
// bit     39/31..21     20..13    12..0
// read         edge     UYYYYY    UZZZZ   within UX partition
          BIGTYPE e = *(BIGTYPE *)readbig;
#if BIGSIZE0 > 4
          e &= BIGSLOTMASK0;
#elif defined NEEDSYNC
          if (unlikely(!e)) { edge += NEDGESLO; continue; }
#endif
          edge += ((u32)(e >> YZBITS) - edge) & (NEDGESLO-1);
// if (ux==78 && my==243) printf("id %d ux %d my %d e %08x prefedge %x edge %x\n", id, ux, my, e, e >> YZBITS, edge);
          u32 uy = (e >> ZBITS) & YMASK;
// bit         39..13     12..0
// write         edge     UZZZZ   within UX UY partition
          *(u64 *)(small0+small.index[uy]) = ((u64)edge << ZBITS) | (e & ZMASK);
// printf("id %d ux %d y %d e %010lx e' %010x\n", id, ux, my, e, ((u64)edge << ZBITS) | (e >> YBITS));
          small.index[uy] += SMALLSIZE;
        }
        if (unlikely(edge >> EDGEBITSLO != ((my+1) * NYZ - 1) >> EDGEBITSLO))
        { printf("OOPS1: id %d ux %d y %d edge %x vs %x\n", id, ux, my, edge, (my+1)*NYZ-1); exit(0); }
      }
      u8 *degs = tdegs[id];
      small.storeu(tbuckets+id, 0);
      dst.matrixu(ux);
      for (u32 uy = 0 ; uy < NY; uy++) {
        memset(degs, 0xff, NZ);
        u8 *readsmall = tbuckets[id][uy].bytes, *endreadsmall = readsmall + tbuckets[id][uy].size;
// if (id==1) printf("id %d ux %d y %d size %u sumsize %u\n", id, ux, uy, tbuckets[id][uy].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u16 *zs = tzs[id];
#ifdef SHOWCYCLE
        u32 *edges0 = (u32 *)(buckets[ux][uy].bytes + NTRIMMEDZ * BIGSIZE);
#else
        u32 *edges0 = tedges[id];
#endif
        u32 *edges = edges0, edge = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE) {
// bit         39..13     12..0
// read          edge     UZZZZ    sorted by UY within UX partition
          u64 e = *(u64 *)rdsmall;
          edge += ((e >> ZBITS) - edge) & NONDEGMASK;
// if (id==0) printf("id %d ux %d y %d e %010lx pref %4x edge %x\n", id, ux, uy, e, e>>ZBITS, edge);
          *edges = edge;
          u32 z = e & ZMASK;
          *zs = z;
          u32 delta = degs[z] ? 1 : 0;
          edges += delta;
          zs    += delta;
        }
        if (unlikely(edge >> NONDEGBITS != EDGEMASK >> NONDEGBITS))
        { printf("OOPS2: id %d ux %d y %d edge %x vs %x\n", id, ux, uy, edge, EDGEMASK); exit(0); }
        assert(edges - edges0 < NTRIMMEDZ);
        u16 *readz = tzs[id];
        u32 *readedge = edges0;
        int64_t uy34 = (int64_t)uy << YZZBITS;
#if NSIPHASH == 8
        __m256i vuy34  = {uy34, uy34, uy34, uy34};
        __m256i vuorv  = {uorv, uorv, uorv, uorv};
        for (; readedge <= edges-NSIPHASH; readedge += NSIPHASH, readz += NSIPHASH) {
          v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
          v0 = _mm256_permute4x64_epi64(vinit, 0x00);
          v1 = _mm256_permute4x64_epi64(vinit, 0x55);
          v2 = _mm256_permute4x64_epi64(vinit, 0xAA);
          v7 = _mm256_permute4x64_epi64(vinit, 0xFF);
          v4 = _mm256_permute4x64_epi64(vinit, 0x00);
          v5 = _mm256_permute4x64_epi64(vinit, 0x55);
          v6 = _mm256_permute4x64_epi64(vinit, 0xAA);

          vpacket0 = _mm256_slli_epi64(_mm256_cvtepu32_epi64(*(__m128i*) readedge     ), 1) | vuorv;
          vhi0     = vuy34 | _mm256_slli_epi64(_mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)readz)), YZBITS);
          vpacket1 = _mm256_slli_epi64(_mm256_cvtepu32_epi64(*(__m128i*)(readedge + 4)), 1) | vuorv;
          vhi1     = vuy34 | _mm256_slli_epi64(_mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)(readz + 4))), YZBITS);

          v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
          SIPROUNDX8; SIPROUNDX8;
          v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
          v2 = XOR(v2,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
          v6 = XOR(v6,_mm256_broadcastq_epi64(_mm_cvtsi64_si128(0xff)));
          SIPROUNDX8; SIPROUNDX8; SIPROUNDX8; SIPROUNDX8;
          v0 = XOR(XOR(v0,v1),XOR(v2,v3));
          v4 = XOR(XOR(v4,v5),XOR(v6,v7));
    
          v1 = _mm256_srli_epi64(v0, YZBITS) & vxmask;
          v5 = _mm256_srli_epi64(v4, YZBITS) & vxmask;
          v0 = vhi0 | (v0 & vyzmask);
          v4 = vhi1 | (v4 & vyzmask);

          u32 vx;
#define STORE(i,v,x,w) \
vx = _mm256_extract_epi32(v,x);\
*(u64 *)(base+dst.index[vx]) = _mm256_extract_epi64(w,i%4);\
dst.index[vx] += BIGSIZE;
// printf("Id %d ux %d y %d edge %08x e' %010lx vx %d\n", id, ux, uy, readedge[i], _mm256_extract_epi64(w,i%4), vx);

          STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
          STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
        }
#endif
        for (; readedge < edges; readedge++, readz++) { // process up to 7 leftover edges if NSIPHASH==8
          u32 node = _sipnode(&sip_keys, *readedge, uorv);
          u32 vx = node >> YZBITS; // & XMASK;
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          *(u64 *)(base+dst.index[vx]) = uy34 | ((u64)*readz << YZBITS) | (node & YZMASK);
// printf("id %d ux %d y %d edge %08x e' %010lx vx %d\n", id, ux, uy, *readedge, uy34 | ((u64)(node & YZMASK) << ZBITS) | *readz, vx);
          dst.index[vx] += BIGSIZE;
        }
      }
      sumsize += dst.storeu(buckets, ux);
    }
    rdtsc1 = __rdtsc();
    printf("genVnodes id %d size %u rdtsc: %lu\n", id, sumsize/BIGSIZE, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/BIGSIZE;
  }

  template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
  void trimedges(const u32 id, u32 round) {
    const u32 SRCSLOTBITS = std::min(SRCSIZE * 8, 2 * YZBITS);
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 DSTSLOTBITS = std::min(DSTSIZE * 8, 2 * YZBITS);
    const u64 DSTSLOTMASK = (1ULL << DSTSLOTBITS) - 1ULL;
    const u32 DSTPREFBITS = DSTSLOTBITS - YZZBITS;
    const u32 DSTPREFMASK = (1 << DSTPREFBITS) - 1;
    u64 rdtsc0, rdtsc1;
    indexer<ZBUCKETSIZE> src, dst;
    indexer<TBUCKETSIZE> small;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 *base = (u8 *)buckets;
    u8 *small0 = (u8 *)tbuckets[id];
    u32 startvx = NY *  id    / nthreads;
    u32   endvx = NY * (id+1) / nthreads;
    for (u32 vx = startvx; vx < endvx; vx++) {
      small.matrixu(0);
                                TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      for (u32 ux = 0 ; ux < NX; ux++) {
        u32 uxyz = ux << YZBITS;
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u8 *readbig = zb.bytes, *endreadbig = readbig + zb.size;
// printf("id %d vx %d ux %d size %u\n", id, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          u64 e = *(u64 *)readbig & SRCSLOTMASK;
          uxyz += ((u32)(e >> YZBITS) - uxyz) & SRCPREFMASK;
// if (round==6) printf("id %d vx %d ux %d e %010lx suffUXYZ %05x suffUXY %03x UXYZ %08x UXY %04x mask %x\n", id, vx, ux, e, (u32)(e >> YZBITS), (u32)(e >> YZZBITS), uxyz, uxyz>>ZBITS, SRCPREFMASK);
          u32 vy = (e >> ZBITS) & YMASK;
// bit     41/39..34    33..26     25..13     12..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          *(u64 *)(small0+small.index[vy]) = ((u64)uxyz << ZBITS) | (e & ZMASK);
          uxyz &= ~ZMASK;
          small.index[vy] += DSTSIZE;
        }
        if (unlikely(uxyz >> YZBITS != ux))
        { printf("OOPS3: id %d vx %d ux %d UXY %x\n", id, vx, ux, uxyz); exit(0); }
      }
      u8 *degs = tdegs[id];
      small.storeu(tbuckets+id, 0);
      for (u32 vy = 0 ; vy < NY; vy++) {
        u64 vy34 = (u64)vy << YZZBITS;
        memset(degs, 0xff, NZ);
        u8    *readsmall = tbuckets[id][vy].bytes, *endreadsmall = readsmall + tbuckets[id][vy].size;
// printf("id %d vx %d vy %d size %u sumsize %u\n", id, vx, vy, tbuckets[id][vx].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u32 ux = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE) {
// bit     41/39..34    33..26     25..13     12..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          u64 e = *(u64 *)rdsmall & DSTSLOTMASK;
          ux += ((u32)(e >> YZZBITS) - ux) & DSTPREFMASK;
// printf("id %d vx %d vy %d e %010lx suffUX %02x UX %x mask %x\n", id, vx, vy, e, (u32)(e >> YZZBITS), ux, SRCPREFMASK);
// bit    41/39..34    33..21     20..13     12..0
// write     VYYYYY    VZZZZZ     UYYYYY     UZZZZ   within UX partition
          *(u64 *)(base+dst.index[ux]) = vy34 | ((e & ZMASK) << YZBITS) | ((e >> ZBITS) & YZMASK);
          dst.index[ux] += degs[e & ZMASK] ? DSTSIZE : 0;
        }
        if (unlikely(ux >> DSTPREFBITS != XMASK >> DSTPREFBITS))
        { printf("OOPS4: id %d vx %x ux %x vs %x\n", id, vx, ux, XMASK); }
      }
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    rdtsc1 = __rdtsc();
    printf("trimedges round %02d size %u rdtsc: %lu\n", round, sumsize/DSTSIZE, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/DSTSIZE;
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
    genUnodes(id, 0);
    barrier();
    genVnodes(id, 1);
    for (u32 round = 2; round < ntrims; round += 2) {
      barrier();
      if (round < EXPANDROUND)
        trimedges<BIGSIZE, BIGSIZE, true>(id, round);
      else if (round == EXPANDROUND)
        trimedges<BIGSIZE, BIGGERSIZE, true>(id, round);
      else
        trimedges<BIGGERSIZE, BIGGERSIZE, true>(id, round);
      barrier();
      if (round < EXPANDROUND)
        trimedges<BIGSIZE, BIGSIZE, false>(id, round+1);
      else
        trimedges<BIGGERSIZE, BIGGERSIZE, false>(id, round+1);
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
// we want sizeof(cuckoo_hash) < sizeof(bigbucket)
// CUCKOO_SIZE * sizeof(u64)   < NEDGES * BIGSIZE0 / NX
// CUCKOO_SIZE * 8             < NEDGES / 64
// NEDGES >> (IDXSHIFT-1)      < NEDGES >> 9
// IDXSHIFT                    >= 10
#define IDXSHIFT 10
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

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

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
  u64 sharedbytes() {
    return sizeof(xyzbucket<ZBUCKETSIZE>); //  + (trimmer->savededges ? sizeof(xyzbucket32) : 0);
  }
  u32 threadbytes() {
    return sizeof(thread_ctx) + sizeof(yzbucket<TBUCKETSIZE>) + sizeof(zbucket8) + sizeof(zbucket16) + sizeof(zbucket32);
  }
  u32 recoveredge(const u32 u, const u32 v) {
    printf(" (%x,%x)", u, v);
#ifdef SHOWCYCLE
    u32 *endreadedges = &trimmer->buckets[u/2>>YZBITS][((u/2>>ZBITS)&YMASK) + 1].size;
    u32 *readedges = endreadedges - NTRIMMEDZ;
    for (; readedges < endreadedges; readedges++) {
      u32 nonce = *readedges;
      if (sipnode(&trimmer->sip_keys, nonce, 1) == v && sipnode(&trimmer->sip_keys, nonce, 0) == u)
        return nonce;
    }
#endif
    return 0;
  }
  void solution(u32 *us, u32 nu, u32 *vs, u32 nv) {
    printf("Nodes");
    u32 soli = nsols;
#ifdef SHOWCYCLE
    nsols++;
#endif
    u32 n = 0;
    sols[soli][n++] = recoveredge(*us, *vs);
    while (nu--)
      sols[soli][n++] = recoveredge(us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
      sols[soli][n++] = recoveredge(vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    printf("\n");
#ifdef SHOWCYCLE
    qsort(sols[soli], PROOFSIZE, sizeof(u32), nonce_cmp);
#endif
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
  
  template <u32 SRCSIZE>
  void findcycles() {
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];
    u64 rdtsc0, rdtsc1;
  
    const u32 SRCSLOTBITS = std::min(SRCSIZE * 8, 2 * YZBITS);
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    rdtsc0 = __rdtsc();
    for (u32 vx = 0; vx < NX; vx++) {
      for (u32 ux = 0 ; ux < NX; ux++) {
        u32 uxyz = ux << YZBITS;
        zbucket<ZBUCKETSIZE> &zb = trimmer->buckets[ux][vx];
        u8 *readbig = zb.bytes, *endreadbig = readbig + zb.size;
// printf("id %d vx %d ux %d size %u\n", id, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        41..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          u64 e = *(u64 *)readbig & SRCSLOTMASK;
          uxyz += ((u32)(e >> YZBITS) - uxyz) & SRCPREFMASK;
          u32 vxyz = (vx << YZBITS) | (e & YZMASK);
          u32 u0 = uxyz << 1, v0 = (vxyz << 1) | 1;
          uxyz &= ~ZMASK;
          if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
            u32 nu = path(u0, us), nv = path(v0, vs);
// printf("vx %02x ux %02x e %010x uy %02x uz %04x vy %02x vz %04x nu %d nv %d\n", vx, ux, e, uy, uz, vy, vz, nu, nv);
            if (us[nu] == vs[nv]) {
              u32 min = nu < nv ? nu : nv;
              for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
              u32 len = nu + nv + 1;
              printf("%4d-cycle found\n", len);
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
        }
        if (unlikely(uxyz >> YZBITS != ux))
        { printf("OOPS3: vx %d ux %d UXY %x\n", vx, ux, uxyz); exit(0); }
      }
    }
    rdtsc1 = __rdtsc();
    printf("findcycles rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  int solve() {
    assert((u64)CUCKOO_SIZE * sizeof(u64) <= (u64)NEDGES * BIGSIZE0 / NX);
    trimmer->trim();
    u32 pctload = trimmer->count() * 100 / CUCKOO_SIZE;
    printf("cuckoo load %d%%\n", pctload);
    if (pctload > 90) {
      printf("overload!\n");
      exit(0);
    }
    cuckoo = new cuckoo_hash(trimmer->tbuckets);
    findcycles<BIGGERSIZE>();
    return nsols;
  }
};
