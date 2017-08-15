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
#define BIGSIZE0 4
#endif
// but they'll need syncing entries
#if BIGSIZE0 == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

// size expanding round; must be odd
#ifndef EXPANDROUND
#define EXPANDROUND 17
#endif
// size in bytes of a big bucket entry after EXPANDROUND rounds
#ifndef BIGGERSIZE
#define BIGGERSIZE 6
#endif

#if EDGEBITS >= 32
#error promote u32 to u64 where necessary
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
const static u32 NXY       = 1 << XYBITS;
const static u32 XYMASK    = NXY - 1;
const static u32 ZBITS     = EDGEBITS - XYBITS;
const static u32 NZ        = 1 << ZBITS;
const static u32 ZMASK     = NZ - 1;
const static u32 YZBITS    = YBITS + ZBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 YZMASK    = NYZ - 1;
const static u32 ZBITS1    = ZBITS - 1;
const static u32 YZZBITS   = YZBITS + ZBITS;
const static u64 NYZZ      = 1ULL << YZZBITS;
const static u64 YZZMASK   = NYZZ - 1;
const static u32 ZZBITS    = 2 * ZBITS;
const static u32 NZZ       = 1 << ZZBITS;
const static u32 ZZMASK    = NZZ - 1;

const static u32 BIGSLOTBITS   = BIGSIZE * 8;
const static u32 SMALLSLOTBITS = SMALLSIZE * 8;
const static u64 BIGSLOTMASK   = (1ULL << BIGSLOTBITS) - 1ULL;
const static u64 SMALLSLOTMASK = (1ULL << SMALLSLOTBITS) - 1ULL;
const static u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const static u64 BIGSLOTMASK0  = (1ULL << BIGSLOTBITS0) - 1ULL;
const static u32 EDGE0BITSLO   = BIGSLOTBITS0 - YZBITS;
const static u32 NEDGES0LO     = 1 << EDGE0BITSLO;

// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.

typedef uint8_t u8;
typedef uint16_t u16;

// 1/16 reduces odds of overflowing z bucket on 2^30 nodes to 2^16*e^-32 (less than 1 on a billion)
#ifndef BIGEPS
#define BIGEPS 1/16
#endif
const static u32 ZBUCKETSLOTS = NZ + NZ * BIGEPS;

template<u32 SLOTSIZE>
struct zbucket {
  const static u32 ZBUCKETSIZE = ZBUCKETSLOTS * SLOTSIZE; 
  u32 size;
  u8 bytes[ZBUCKETSIZE]; // conceptually, slot slots[ZBUCKETSLOTS], but need flexible slot size
  zbucket() {
    size = 0;
  }
  u32 setsize(u32 _size) {
    size = _size;
    assert(size <= ZBUCKETSIZE);
    return size;
  }
};

template<u32 SLOTSIZE>
using yzbucket = zbucket<SLOTSIZE>[NY];
template <u32 SLOTSIZE>
using xyzbucket = yzbucket<SLOTSIZE>[NX];

template<u32 SLOTSIZE>
struct indexer {
  const static u32 ZBUCKETSIZE = ZBUCKETSLOTS * SLOTSIZE; 
  u32 index[NX];

  indexer() {
  }
  void starty(const u32 y) {
    u32 byte_offset = y * sizeof(zbucket<SLOTSIZE>) + sizeof(u32);
    for (u32 x = 0; x < NX; x++, byte_offset += sizeof(yzbucket<SLOTSIZE>))
      index[x] = byte_offset;
  }
  u32 storey(yzbucket<SLOTSIZE> *buckets, const u32 y) {
    u32 sumsize = 0;
    u32 byte_offset = y * sizeof(zbucket<SLOTSIZE>) + sizeof(u32);
    for (u32 x = 0; x < NX; x++, byte_offset += sizeof(yzbucket<SLOTSIZE>)) {
      sumsize += buckets[x][y].setsize(index[x] - byte_offset);
      assert(buckets[x][y].size < ZBUCKETSIZE); // should just truncate if sufficiently unlikely
    }
    return sumsize;
  }
  void startx(const u32 x) {
    u32 byte_offset = x * sizeof(yzbucket<SLOTSIZE>) + sizeof(u32);
    for (u32 y = 0; y < NY; y++, byte_offset += sizeof(zbucket<SLOTSIZE>))
      index[y] = byte_offset;
  }
  u32 storex(yzbucket<SLOTSIZE> *buckets, const u32 x) {
    u32 sumsize = 0;
    u32 byte_offset = x * sizeof(yzbucket<SLOTSIZE>) + sizeof(u32);
    for (u32 y = 0; y < NY; y++, byte_offset += sizeof(zbucket<SLOTSIZE>)) {
      sumsize += buckets[x][y].setsize(index[y] - byte_offset);
      assert(buckets[x][y].size < ZBUCKETSIZE); // should just truncate if sufficiently unlikely
    }
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

const static u32 NTRIMMEDZ  = NZ * 3/4; // safely over 1-e(-1) trimming fraction
typedef u8 zbucket8[NZ];
typedef u16 zbucket16[NTRIMMEDZ];
typedef u32 zbucket32[NTRIMMEDZ];
typedef zbucket32 yzbucket32[NY];
typedef yzbucket32 xyzbucket32[NX];

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  yzbucket<BIGSIZE0> *buckets;
  yzbucket<BIGSIZE> *tbuckets;
  yzbucket32 *saveedges;
  zbucket32 *tedges;
  zbucket16 *tzs;
  zbucket8 *tdegs;
  u32 ntrims;
  u32 nthreads;
  thread_ctx *threads;
  pthread_barrier_t barry;

  void touch(u8 *p, u32 n) {
    for (u32 i=0; i<n; i+=4096)
      *(u32 *)(p+i) = 0;
  }
  edgetrimmer(const u32 n_threads, u32 n_trims, const bool showcycle) {
    assert(sizeof(xyzbucket<BIGSIZE0>) == NX * sizeof(yzbucket<BIGSIZE0>));
    assert(sizeof(xyzbucket<BIGSIZE>) == NX * sizeof(yzbucket<BIGSIZE>));
    nthreads = n_threads;
    ntrims   = n_trims;
    buckets  = new yzbucket<BIGSIZE0>[NX];
    touch((u8 *)buckets, sizeof(xyzbucket<BIGSIZE0>));
    tbuckets = new yzbucket<BIGSIZE>[nthreads];
    touch((u8 *)tbuckets, nthreads * sizeof(yzbucket<BIGSIZE>));
    threads  = new thread_ctx[nthreads];
    saveedges = showcycle ? new yzbucket32[NX] : 0;
    tedges = showcycle ? 0 : new zbucket32[nthreads];
    tdegs = new zbucket8[nthreads];
    tzs = new zbucket16[nthreads];
    int err  = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] threads;
    delete[] saveedges;
    delete[] tedges;
    delete[] tdegs;
    delete[] tzs;
  }
  void genUnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NX];;
#endif
  
    rdtsc0 = __rdtsc();
    u8 *base = (u8 *)buckets;
    indexer<BIGSIZE0> dst;
    u32 z;
    u32 starty = NY *  id    / nthreads;
    u32   endy = NY * (id+1) / nthreads;
    u32 edge = starty * NYZ, endedge = edge + NYZ;
#if NSIPHASH == 8
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {XMASK, XMASK, XMASK, XMASK};
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
    u32 sumsize = 0;
    for (u32 ey = starty; ey < endy; ey++, endedge += NYZ) {
      dst.starty(ey);
#ifdef NEEDSYNC
      for (u32 x=0; x < NX; x++)
        last[x] = edge;
#endif
      for (; edge < endedge; edge += NSIPHASH) {
// bit        28..21     20..13    12..0
// node       XXXXXX     YYYYYY    ZZZZZ
#if NSIPHASH == 1
        u32 node = _sipnode(&sip_keys, edge, uorv);
        x = node >> YZBITS;
        BIGTYPE zz = (BIGTYPE)edge << YZBITS | (node & YZBITS);
#ifndef NEEDSYNC
// bit        39..21     20..13    12..0
// write        edge     YYYYYY    ZZZZZ
        *(BIGTYPE *)(base+dst.index[z]) = zz;
        dst.index[z] += BIGSIZE0;
#else
        if (zz) {
          for (; unlikely(last[z] + NEDGES0LO <= block); last[z] += NEDGES0LO, dst.index[z] += BIGSIZE0)
            *(u32 *)(base+dst.index[z]) = 0;
          *(u32 *)(base+dst.index[z]) = zz;
          dst.index[z] += BIGSIZE0;
          last[z] = edge;
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
        v0 = _mm256_srli_epi64(v0 & vnodemask, XBITS) | vhi0;
        v4 = _mm256_srli_epi64(v4 & vnodemask, XBITS) | vhi1;
        vhi0 = _mm256_add_epi64(vhi0, vhiinc);
        vhi1 = _mm256_add_epi64(vhi1, vhiinc);

#ifndef NEEDSYNC
#define STORE0(i,v,x,w) \
  z = _mm256_extract_epi32(v,x);\
  *(u64 *)(base+dst.index[z]) = _mm256_extract_epi64(w,i%4);\
  dst.index[z] += BIGSIZE0;
#else
  u32 zz;
#define STORE0(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    z = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[z] + NEDGES0LO <= edge+i); last[z] += NEDGES0LO, dst.index[z] += BIGSIZE0)\
      *(u32 *)(base+dst.index[z]) = 0;\
    *(u32 *)(base+dst.index[z]) = zz;\
    dst.index[z] += BIGSIZE0;\
    last[z] = edge+i;\
  }
#endif
        STORE0(0,v1,0,v0); STORE0(1,v1,2,v0); STORE0(2,v1,4,v0); STORE0(3,v1,6,v0);
        STORE0(4,v5,0,v4); STORE0(5,v5,2,v4); STORE0(6,v5,4,v4); STORE0(7,v5,6,v4);
#else
#error not implemented
#endif
      }
#ifdef NEEDSYNC
      for (u32 z=0; z < NX; z++) {
        for (; last[z]<endedge-NEDGES0LO; last[z]+=NEDGES0LO) {
          *(u32 *)(base+dst.index[z]) = 0;
          dst.index[z] += BIGSIZE0;
        }
      }
#endif
      sumsize += dst.storey(buckets, ey);
    }
    rdtsc1 = __rdtsc();
    printf("genUnodes id %d rdtsc: %lu sumsize %x\n", id, rdtsc1-rdtsc0, sumsize);
  }

  void genVnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
  
#if NSIPHASH == 8
    static const __m256i vnodemask = {EDGEMASK, EDGEMASK, EDGEMASK, EDGEMASK};
    static const __m256i vbucketmask = {XMASK, XMASK, XMASK, XMASK};
    const __m256i vinit = _mm256_set_epi64x(
      sip_keys.k1^0x7465646279746573ULL,
      sip_keys.k0^0x6c7967656e657261ULL,
      sip_keys.k1^0x646f72616e646f6dULL,
      sip_keys.k0^0x736f6d6570736575ULL);
    __m256i vpacket0, vpacket1, vhi0, vhi1;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    static const u32 EVENNONDEGMASK = (BIGSLOTMASK >> ZBITS1) & -2;
#endif
    indexer<BIGSIZE0> src, dst;
    indexer<BIGSIZE> small;
  
    rdtsc0 = __rdtsc();
    u32 sumsize = 0;
    u8 *base = (u8 *)buckets;
    u8 *small0 = (u8 *)tbuckets[id];
    u32 startx = NX *  id    / nthreads;
    u32   endx = NX * (id+1) / nthreads;
    for (u32 x = startx; x < endx; x++) {
      small.startx(0);
      int64_t x34 = (int64_t)x << YZZBITS;
#if NSIPHASH == 8
      __m256i vx34  = {x34, x34, x34, x34};
#endif
      for (u32 y = 0 ; y < NY; y++) {
        u32 edge = y * NYZ;
        u8    *readbig = buckets[x][y].bytes;
        u8 *endreadbig = readbig + buckets[x][y].size;
// printf("id %d x %d y %d size %d read %d\n", id, x, y, buckets[x][y].size, readbig-base);
        for (; readbig < endreadbig; readbig += BIGSIZE0) {
// bit     39/31..21     20..13    12..0
// read         edge     UYYYYY    UZZZZ   within UX partition
          BIGTYPE e = *(BIGTYPE *)readbig;
#if BIGSIZE0 > 4
          e &= BIGSLOTMASK0;
#else
          if (unlikely(!e)) { edge += NEDGES0LO; continue; }
#endif
          edge += ((u32)(e>>YZBITS) - edge) & (NEDGES0LO-1);
// if (y==199) printf("id %d x %d y %d e %08x prefedge %x edge %x\n", id, x, y, e, e >> YZBITS, edge);
          u32 z = (e >> ZBITS) & YMASK;
// bit         39..13     12..0
// write         edge     UZZZZ   within UX UY partition
          *(u64 *)(small0+small.index[z]) = ((u64)edge << ZBITS) | (e & ZMASK);
// printf("id %d x %d y %d e %010lx e' %010x\n", id, x, y, e, ((u64)edge << ZBITS) | (e >> YBITS));
          small.index[z] += SMALLSIZE;
        }
        if (unlikely(edge >> EDGE0BITSLO != ((y+1) * NYZ - 1) >> EDGE0BITSLO))
        { printf("OOPS1: id %d x %d y %d edge %x vs %x\n", id, x, y, edge, (y+1)*NYZ-1); exit(0); }
      }
      u8 *degs = tdegs[id];
      small.storex(tbuckets, 0);
      for (u32 y = 0 ; y < NY; y++) {
        memset(degs, 0xff, NZ);
        u8    *readsmall = tbuckets[id][y].bytes, *endreadsmall = readsmall + tbuckets[id][y].size;
// printf("id %d x %d y %d size %d sumsize %d\n", id, x, y, tbuckets[id][y].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u16 *zs = tzs[id];
        u32 *edges0 = saveedges ? saveedges[x][y] : tedges[id], *edges = edges0;
        u32 edge2 = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE) {
// bit         39..13     12..0
// read          edge     UZZZZ    sorted by UY within UX partition
          u64 e = *(u64 *)rdsmall;
// printf("id %d x %d y %d e %010lx z %04x\n", id, x, y, e, (u32)e & ZMASK);
          edge2 += ((e>>ZBITS1) - edge2) & EVENNONDEGMASK;
          *edges = edge2;
          u32 z = e & ZMASK;
          *zs = z;
          u32 delta = degs[z] ? 1 : 0;
          edges += delta;
          zs    += delta;
          sumsize += delta;
        }
        u16 *readz = tzs[id];
        assert(edges - edges0 < NTRIMMEDZ);
        dst.startx(x);
#if NSIPHASH == 8
        for (u32 *readedge = edges0; readedge <= edges-NSIPHASH; readedge += NSIPHASH, readz += NSIPHASH) {
          v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
          v0 = _mm256_permute4x64_epi64(vinit, 0x00);
          v1 = _mm256_permute4x64_epi64(vinit, 0x55);
          v2 = _mm256_permute4x64_epi64(vinit, 0xAA);
          v7 = _mm256_permute4x64_epi64(vinit, 0xFF);
          v4 = _mm256_permute4x64_epi64(vinit, 0x00);
          v5 = _mm256_permute4x64_epi64(vinit, 0x55);
          v6 = _mm256_permute4x64_epi64(vinit, 0xAA);

          vpacket0 = _mm256_cvtepu32_epi64(*(__m128i*)readedge);
          vhi0     = _mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)readz));
          vpacket1 = _mm256_cvtepu32_epi64(*(__m128i*)(readedge + 4));
          vhi1     = _mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)(readz + 4)));

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
          v0 = vx34 | _mm256_slli_epi64(_mm256_srli_epi64(v0 & vnodemask, XBITS), ZBITS) | vhi0;
          v4 = vx34 | _mm256_slli_epi64(_mm256_srli_epi64(v4 & vnodemask, XBITS), ZBITS) | vhi1;

          u32 z;
#define STORE(i,v,x,w) \
z = _mm256_extract_epi32(v,x);\
*(u64 *)(base+dst.index[z]) = _mm256_extract_epi64(w,i%4);\
dst.index[z] += BIGSIZE0;

          STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
          STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
        }
#endif
      }
    }
    rdtsc1 = __rdtsc();
    printf("genVnodes id %d rdtsc: %lu sumsize %d\n", id, rdtsc1-rdtsc0, sumsize);
  }

#if 0
        }
        edge = prevedge2 / 2;
        for (; ; readedge += BIGSIZE) { // process up to 7 leftover edges
// bit         39..13     12..0
// read          edge     UZZZZ within UX partition
          u64 e = *(u64 *)readedge & BIGSLOTMASK;
          edge += ((e>>ZBITS) - edge) & (BIGSLOTMASK >> ZBITS);
          if (edge >= NEDGES) break; // reached end of UY section
          u32 node = _sipnode(&sip_keys, edge, uorv);
          z = node & XMASK;
// bit        39..34    33..26     25..13     12..0
// write      UXXXXX    VYYYYY     VZZZZZ     UZZZZ   within VX partition
          *(u64 *)(base+big[z]) = UX34 | ((u64)(node >> XBITS) << ZBITS) | (e & ZMASK);
          big[z] += BIGSIZE;
        }
        src->index[id][UX] = readedge - src->base;
      }
    }
    if (endUY-startUY < (NX + nthreads-1) / nthreads)
      barrier(); // make sure all threads have an equal number of barriers
    rdtsc1 = __rdtsc();
    printf("genVnodes id %d rdtsc: %lu sumsize %d\n", id, rdtsc1-rdtsc0, dst->sumsize(id));
  }

#if XBITS != YBITS
#error trimedges needs rewriting for XBITS != YBITS
#endif

  template <u32 ZSHIFT, u32 SRCSIZE, u32 DSTSIZE>
  void trimedges(indexer *src, indexer *dst, const u32 id, u32 round) {
    u64 rdtsc0, rdtsc1;
    u32 small[NY];
  
    const u32 SRCSLOTBITS = SRCSIZE * 8;
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 DSTSLOTBITS = DSTSIZE * 8;
    const u64 DSTSLOTMASK = (1ULL << DSTSLOTBITS) - 1ULL;
    const u32 DSTPREFBITS = DSTSLOTBITS - YZZBITS;
    const u32 DSTPREFMASK = (1 << DSTPREFBITS) - 1;
    rdtsc0 = __rdtsc();
    u8 *base = dst->base;
    u8 *small0 = tbuckets[id*NY];
    u32 bigbkt = id*NX/nthreads, endbkt = (id+1)*NX/nthreads; 
    u32 *big = dst->init(id);
    for (; bigbkt < endbkt; bigbkt++) {
      for (u32 i=0; i < NY; i++)
        small[i] = i * SMALLBUCKETSIZE;
      for (u32 from = 0 ; from < nthreads; from++) {
        u32 UYX = (from * NY / nthreads) << XBITS;
        u8    *readbig = src->base + src->start(from, bigbkt);
        u8 *endreadbig = src->base + src->curr(from, bigbkt);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        39..34    33..26     25..13     12..0
// read       UXXXXX    VYYYYY     VZZZZZ     UZZZZ   within VX partition
          u64 e = *(u64 *)readbig & SRCSLOTMASK;
// bit        47..42    41..34    33..26     25..13     12..0
// read       UYYYYY    UXXXXX    VYYYYY     VZZZZZ     UZZZZ   within VX partition
          UYX += ((u32)(e>>YZZBITS) - UYX) & SRCPREFMASK;
// if (SRCSIZE>5) printf("id %d bkt %d from %d e %012lx suffUYX %x UYX %x mask %x\n", id, bigbkt, from, e, e>>YZZBITS, UYX, SRCPREFMASK);
          u32 z = (e >> ZZBITS) & YMASK;
// bit        39..34    33..26     25..13     12..0
// write      UYYYYY    UXXXXX     VZZZZZ     UZZZZ   within VX VY partition
          *(u64 *)(small0+small[z]) = ((u64)UYX << ZZBITS) | (e & ZZMASK);
// bit        41..34    33..26     25..13     12..0
// write      UYYYYY    UXXXXX     VZZZZZ     UZZZZ   within VX VY partition
          small[z] += DSTSIZE;
          assert(small[z] < NY * SMALLBUCKETSIZE);
        }
        if (unlikely(UYX/NX != (from+1)*NY/nthreads - 1))
        { printf("OOPS2: id %d bkt %d from %d UYX %x vs %x\n", id, bigbkt, from, UYX/NX, (from+1)*NY/nthreads - 1); }
      }
      u8 *degs = src->base + src->start(0, bigbkt); // recycle!
      for (u32 smallbkt = 0; smallbkt < NY; smallbkt++) {
        memset(degs, 1, NZ);
        u8    *readsmall = small0 + smallbkt * SMALLBUCKETSIZE,
           *endreadsmall = small0 + small[smallbkt], *rdsmall;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE)
          degs[(*(u32 *)rdsmall >> ZSHIFT) & ZMASK]--;
        u32 UY = 0;
        for (rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE) {
// bit        39..34    33..26     25..13     12..0
// write      UYYYYY    UXXXXX     VZZZZZ     UZZZZ   within VX VY partition
// bit        41..34    33..26     25..13     12..0
          u64 e = *(u64 *)rdsmall & DSTSLOTMASK;
          UY += ((u32)(e>>YZZBITS) - UY) & DSTPREFMASK;
          // if (round>=20&&bigbkt==207&&smallbkt==214) printf("id %d small %02x  e %010lx UY %02x\n", id, smallbkt, e & SMALLSLOTMASK, (u32)(e>> YZZBITS) & SMLPREFMASK );
          assert(UY < NY);
// bit        39..34    33..26     25..13     12..0
// write      VYYYYY    UXXXXX     VZZZZZ     UZZZZ   within UY partition
          *(u64 *)(base+big[UY]) = ((u64)((bigbkt<<YBITS) | smallbkt) << YZZBITS) | (e & YZZMASK);
// bit        47..42    41..34    33..26     25..13     12..0
// read       VXXXXX    VYYYYY    UXXXXX     VZZZZZ     UZZZZ   within VX partition
          big[UY] += degs[(e >> ZSHIFT) & ZMASK] ? DSTSIZE : 0;
        }
        if (unlikely(UY>>DSTPREFBITS != YMASK>>DSTPREFBITS))
        { printf("OOPS3: id %d bkt %d %d UY %x vs %x\n", id, bigbkt, smallbkt, UY, YMASK); exit(0); }
      }
    }
    for (u32 bigbkt = 0; bigbkt < NX; bigbkt++)
      assert(dst->fits(id, bigbkt));
    rdtsc1 = __rdtsc();
    printf("trimedges rdtsc: %lu\n", rdtsc1-rdtsc0);
  }
#endif

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
    genUnodes(id, 1);
    barrier();
    if (id == 0)
      printf("round 0 edges %d\n", 0);
    barrier();
    genVnodes(id, 0);
    barrier();
#if 0
    for (u32 round=1; round <= ntrims; round++) {
      if (id == 0) {
        printf("round %2d edges %d\n", round, idx[(round&1)^1]->sumsize()/(round<=EXPANDROUND?BIGSIZE:BIGGERSIZE));
//        for (u32 id=0; id < nthreads; id++)
//          for (u32 bkt=0; bkt < NX/8; bkt++)
//            printf("%d %3d %d%c", id, bkt, idx[(round&1)^1]->size(id, bkt)/BIGSIZE, (bkt&3)==3 ? '\n' : ' ');
      }
      barrier();
      if (round & 1) {
        if (round < EXPANDROUND)
          trimedges<ZBITS, BIGSIZE, BIGSIZE>(idx[0], idx[1], id, round);
        else if (round == EXPANDROUND)
          trimedges<ZBITS, BIGSIZE, BIGGERSIZE>(idx[0], idx[1], id, round);
        else
          trimedges<ZBITS, BIGGERSIZE, BIGGERSIZE>(idx[0], idx[1], id, round);
      } else {
        if (round < EXPANDROUND)
          trimedges<0, BIGSIZE, BIGSIZE>(idx[1], idx[0], id, round);
        else
          trimedges<0, BIGGERSIZE, BIGGERSIZE>(idx[1], idx[0], id, round);
      }
      barrier();
    }
    if (id == 0)
      idx[1]->setbase((u8 *)buckets, 42); // restore former glory
#endif
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

class solver_ctx {
public:
  edgetrimmer *trimmer;
  cuckoo_hash *cuckoo;
  u32 sols[MAXSOLS][PROOFSIZE];
  u32 nsols;

  solver_ctx(u32 n_threads, u32 n_trims, const bool showcycle) {
    trimmer = new edgetrimmer(n_threads, n_trims, showcycle);
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
    return sizeof(xyzbucket<BIGSIZE0>) + (trimmer->saveedges ? sizeof(xyzbucket32) : 0);
  }
  u32 threadbytes() {
    return sizeof(thread_ctx) + sizeof(yzbucket<BIGSIZE>) + sizeof(zbucket8) + sizeof(zbucket16) + sizeof(zbucket32);
  }
  void solution(u32 *us, u32 nu, u32 *vs, u32 nv) {
    return; // not functional yet
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
  
#if 0
  template <u32 SRCSIZE>
  void findcycles(indexer *src) {
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];
    u64 rdtsc0, rdtsc1;
  
    const u32 SRCSLOTBITS = SRCSIZE * 8;
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    rdtsc0 = __rdtsc();
    for (u32 VX = 0; VX < NX; VX++) {
      for (u32 from = 0 ; from < src->nthreads; from++) {
        u32 UYX = (from * NY / src->nthreads) << XBITS;
        u8    *readbig = src->base + src->start(from, VX);
        u8 *endreadbig = src->base + src->curr(from, VX);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        47..42    41..34    33..26     25..13     12..0
// read       UYYYYY    UXXXXX    VYYYYY     VZZZZZ     UZZZZ   within VX partition
          u64 e = *(u64 *)readbig & SRCSLOTMASK;
          UYX += ((u32)(e>>YZZBITS) - UYX) & SRCPREFMASK;
          u32 u0 = (((u32)e & ZMASK) << XYBITS | UYX) << 1 | 1, v0 = (((e >> ZBITS) & YZMASK) << XBITS | VX) << 1;
          if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
            u32 nu = path(u0, us), nv = path(v0, vs);
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
        if (unlikely(UYX/NX != (from+1)*NY/src->nthreads - 1))
        { printf("OOPS4: bkt %d from %d UYX %x vs %x\n", VX, from, UYX/NX, (from+1)*NY/src->nthreads - 1); }
      }
    }
    rdtsc1 = __rdtsc();
    printf("findcycles rdtsc: %lu\n", rdtsc1-rdtsc0);
  }
#endif

  int solve() {
    assert(CUCKOO_SIZE * sizeof(u64) <= NEDGES * BIGSIZE0 / NX);
    trimmer->trim();
#if 0
    indexer *edges = trimmer->idx[0];
    u32 pctload = (edges->sumsize() / BIGGERSIZE) * 100 / CUCKOO_SIZE;
    printf("cuckoo load %d%%\n", pctload);
    if (pctload > 90) {
      printf("overload!\n");
      exit(0);
    }
    cuckoo = new cuckoo_hash(trimmer->tbuckets);
    findcycles<BIGGERSIZE>(edges);
#endif
    return nsols;
  }
};
