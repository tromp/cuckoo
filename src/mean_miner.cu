#include <stdint.h>
#include <string.h>
#include "cuckoo.h"

// d(evice s)ipnode
#if (__CUDA_ARCH__  >= 320) // redefine ROTL to use funnel shifter, 3% speed gain

static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __device__ __forceinline__ void operator^= (uint2 &a, uint2 b) { a.x ^= b.x, a.y ^= b.y; }
static __device__ __forceinline__ void operator+= (uint2 &a, uint2 b) {
  asm("{\n\tadd.cc.u32 %0,%2,%4;\n\taddc.u32 %1,%3,%5;\n\t}\n\t"
    : "=r"(a.x), "=r"(a.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}
#undef ROTL
__inline__ __device__ uint2 ROTL(const uint2 a, const int offset) {
  uint2 result;
  if (offset >= 32) {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
  } else {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
  }
  return result;
}
__device__ __forceinline__ uint2 vectorize(const uint64_t x) {
  uint2 result;
  asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.x), "=r"(result.y) : "l"(x));
  return result;
}
__device__ __forceinline__ uint64_t devectorize(uint2 x) {
  uint64_t result;
  asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(x.x), "r"(x.y));
  return result;
}
__device__ node_t dipnode(siphash_keys &keys, edge_t nce, u32 uorv) {
  uint2 nonce = vectorize(2*nce + uorv);
  uint2 v0 = vectorize(keys.k0 ^ 0x736f6d6570736575ULL),
        v1 = vectorize(keys.k1 ^ 0x646f72616e646f6dULL),
        v2 = vectorize(keys.k0 ^ 0x6c7967656e657261ULL),
        v3 = vectorize(keys.k1 ^ 0x7465646279746573ULL) ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= vectorize(0xff);
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return devectorize(v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

#else

__device__ node_t dipnode(siphash_keys &keys, edge_t nce, u32 uorv) {
  u64 nonce = 2*nce + uorv;
  u64 v0 = keys.k0 ^ 0x736f6d6570736575ULL, v1 = keys.k0 ^ 0x646f72616e646f6dULL,
      v2 = keys.k0 ^ 0x6c7967656e657261ULL, v3 = keys.k0 ^ 0x7465646279746573ULL ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <bitset>

// algorithm/performance parameters

// EDGEBITS/NEDGES/EDGEMASK defined in cuckoo.h

// The node bits are logically split into 3 groups:
// XBITS 'X' bits (most significant), YBITS 'Y' bits, and ZBITS 'Z' bits (least significant)
// Here we have the default XBITS=YBITS=7, ZBITS=15 summing to EDGEBITS=29
// nodebits   XXXXXXX YYYYYYY ZZZZZZZZZZZZZZZ
// bit%10     8765432 1098765 432109876543210
// bit/10     2222222 2111111 111110000000000

// The matrix solver stores all edges in a matrix of NX * NX buckets,
// where NX=2^XBITS is the number of possible values of the 'X' bits.
// Edge i between nodes ui = siphash24(2*i) and vi = siphash24(2*i+1)
// resides in the bucket at (uiX,viX)
// In each trimming round, either a matrix row or a matrix column (NX buckets)
// is bucket sorted on uY or vY respectively, and then within each bucket
// uZ or vZ values are counted and edges with a count of only one are eliminated,
// while remaining edges are bucket sorted back on vY or uY respectively.
// When sufficiently many edges have been eliminated, a pair of compression
// rounds remap surviving Z values in each X,Y bucket to fit into 15-YBITS bits,
// allowing the remaining rounds to avoid the sorting on Y and directly
// count YZ values in a cache friendly 32KB.

#ifndef XBITS
// 7 seems to give best performance
#define XBITS 7
#endif

#define YBITS XBITS

// size in bytes of a big bucket entry
#ifndef BIGSIZE
#if EDGEBITS <= 15
#define BIGSIZE 4
// no compression needed
#define COMPRESSROUND 0
#else
#define BIGSIZE 5
// YZ compression round; must be even
#ifndef COMPRESSROUND
#define COMPRESSROUND 14
#endif
#endif
#endif
// size in bytes of a small bucket entry
#define SMALLSIZE BIGSIZE

// initial entries could be smaller at percent or two slowdown
#ifndef BIGSIZE0
#if EDGEBITS < 30 && !defined SAVEEDGES
#define BIGSIZE0 4
#else
#define BIGSIZE0 BIGSIZE
#endif
#endif
// but they may need syncing entries
#if BIGSIZE0 == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

typedef uint8_t u8;
typedef uint16_t u16;

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
const static u32 NXY       = 1 << XYBITS;
const static u32 ZBITS     = EDGEBITS - XYBITS;
const static u32 NZ        = 1 << ZBITS;
const static u32 ZMASK     = NZ - 1;
const static u32 YZBITS    = YBITS + ZBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 YZMASK    = NYZ - 1;
const static u32 YZ1BITS   = 15;  // combined Y and compressed Z bits
const static u32 NYZ1      = 1 << YZ1BITS;
const static u32 YZ1MASK   = NYZ1 - 1;
const static u32 Z1BITS    = YZ1BITS - YBITS;
const static u32 NZ1       = 1 << Z1BITS;
const static u32 Z1MASK    = NZ1 - 1;
const static u32 YZ2BITS   = YZBITS < 11 ? YZBITS : 11;  // more compressed YZ bits
const static u32 NYZ2      = 1 << YZ2BITS;
const static u32 YZ2MASK   = NYZ2 - 1;
const static u32 Z2BITS    = YZ2BITS - YBITS;
const static u32 NZ2       = 1 << Z2BITS;
const static u32 Z2MASK    = NZ2 - 1;
const static u32 YZZBITS   = YZBITS + ZBITS;
const static u32 YZZ1BITS  = YZ1BITS + ZBITS;

const static u32 BIGSLOTBITS   = BIGSIZE * 8;
const static u32 SMALLSLOTBITS = SMALLSIZE * 8;
const static u64 BIGSLOTMASK   = (1ULL << BIGSLOTBITS) - 1ULL;
const static u64 SMALLSLOTMASK = (1ULL << SMALLSLOTBITS) - 1ULL;
const static u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const static u64 BIGSLOTMASK0  = (1ULL << BIGSLOTBITS0) - 1ULL;
const static u32 NONYZBITS     = BIGSLOTBITS0 - YZBITS;
const static u32 NNONYZ        = 1 << NONYZBITS;

// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.

// 1/32 reduces odds of overflowing z bucket on 2^30 nodes to 2^14*e^-32
// (less than 1 in a billion) in theory. not so in practice (fails first at matrix30 -n 1549)
#ifndef BIGEPS
#define BIGEPS 3/64
#endif

// 176/256 is safely over 1-e(-1) ~ 0.63 trimming fraction
#ifndef TRIMFRAC256
#define TRIMFRAC256 176
#endif

const static u32 NTRIMMEDZ  = NZ * TRIMFRAC256 / 256;

const static u32 ZBUCKETSLOTS = NZ + NZ * BIGEPS;
#ifdef SAVEEDGES
const static u32 ZBUCKETSIZE = NTRIMMEDZ * (BIGSIZE + sizeof(u32));  // assumes EDGEBITS <= 32
#else
const static u32 ZBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE0;
#endif
const static u32 TBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE;

template<u32 BUCKETSIZE>
struct zbucket {
  u32 size;
  const static u32 RENAMESIZE = 2*NZ2 + 2*(COMPRESSROUND ? NZ1 : 0);
  union {
    u8 bytes[BUCKETSIZE];
    struct {
#ifdef SAVEEDGES
      u32 words[BUCKETSIZE/sizeof(u32) - RENAMESIZE - NTRIMMEDZ];
#else
      u32 words[BUCKETSIZE/sizeof(u32) - RENAMESIZE];
#endif
      u32 renameu1[NZ2];
      u32 renamev1[NZ2];
      u32 renameu[COMPRESSROUND ? NZ1 : 0];
      u32 renamev[COMPRESSROUND ? NZ1 : 0];
#ifdef SAVEEDGES
      u32 edges[NTRIMMEDZ];
#endif
    };
  };
  __device__ u32 setsize(u8 const *end) {
    size = end - bytes;
    assert(size <= BUCKETSIZE);
    return size;
  }
};

// template<u32 BUCKETSIZE>
// using yzbucket = zbucket<BUCKETSIZE>[NY];
// template <u32 BUCKETSIZE>
// using matrix = yzbucket<BUCKETSIZE>[NX];

template<u32 BUCKETSIZE>
struct indexer {
  offset_t index[NX];

  __device__ void matrixv(const u32 y) {
    const zbucket<BUCKETSIZE>[NY] *foo = 0;
    for (u32 x = 0; x < NX; x++)
      index[x] = foo[x][y].bytes - (u8 *)foo;
  }
  __device__ offset_t storev(zbucket<BUCKETSIZE> (*buckets)[NY], const u32 y) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 x = 0; x < NX; x++)
      sumsize += buckets[x][y].setsize(base+index[x]);
    return sumsize;
  }
  __device__ void matrixu(const u32 x) {
    const zbucket<BUCKETSIZE>[NY] *foo = 0;
    for (u32 y = 0; y < NY; y++)
      index[y] = foo[x][y].bytes - (u8 *)foo;
  }
  __device__ offset_t storeu(zbucket<BUCKETSIZE> (*buckets)[NY], const u32 x) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 y = 0; y < NY; y++)
      sumsize += buckets[x][y].setsize(base+index[y]);
    return sumsize;
  }
};

#define likely(x)   __builtin_expect((x)!=0, 1)
#define unlikely(x) __builtin_expect((x), 0)

class edgetrimmer; // avoid circular references

typedef struct {
  u32 id;
  edgetrimmer *et;
} thread_ctx;

typedef u8 zbucket8[2*NYZ1];
typedef u16 zbucket16[NTRIMMEDZ];
typedef u32 zbucket32[NTRIMMEDZ];

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  edgetrimmer *dt;
  zbucket<ZBUCKETSIZE> (*buckets)[NY];
  zbucket<TBUCKETSIZE> (*tbuckets)[NY];
  zbucket32 *tedges;
  zbucket16 *tzs;
  zbucket8 *tdegs;
  offset_t *tcounts;
  u32 ntrims;
  u32 nblocks;
  u32 threadsperblock;
  bool showall;

  // __device__ void touch(u8 *p, const offset_t n) {
  //   for (offset_t i=0; i<n; i+=4096)
  //     *(u32 *)(p+i) = 0;
  // }
  edgetrimmer(const u32 n_blocks, const u32 n_trims, const bool show_all) {
    nblocks = n_blocks;
    threadsperblock = 1;
    ntrims   = n_trims;
    showall = show_all;
    checkCudaErrors(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    // buckets  = new yzbucket<ZBUCKETSIZE>[NX];
    checkCudaErrors(cudaMalloc((void**)&buckets, sizeof(zbucket<ZBUCKETSIZE>[NX][NY])));
    // touch((u8 *)buckets, sizeof(matrix<ZBUCKETSIZE>));
    // tbuckets = new yzbucket<TBUCKETSIZE>[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tbuckets, sizeof(zbucket<TBUCKETSIZE>[NY])));
    // touch((u8 *)tbuckets, nthreads * sizeof(yzbucket<TBUCKETSIZE>));
#ifdef SAVEEDGES
    tedges  = 0;
#else
    // tedges  = new zbucket32[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tedges, sizeof(zbucket<TBUCKETSIZE>[NY])));
#endif
    // tdegs   = new zbucket8[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tdegs, nblocks * sizeof(zbucket8)));
    // tzs     = new zbucket16[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tzs, nblocks * sizeof(zbucket16)));
    // tcounts = new offset_t[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tcounts, nblocks * sizeof(offset_t)));
  }
  ~edgetrimmer() {
    // delete[] buckets;
    checkCudaErrors(cudaFree(buckets));
    // delete[] tbuckets;
    checkCudaErrors(cudaFree(tbuckets));
    // delete[] tedges;
    checkCudaErrors(cudaFree(tedges));
    // delete[] tdegs;
    checkCudaErrors(cudaFree(tdegs));
    // delete[] tzs;
    checkCudaErrors(cudaFree(tzs));
    // delete[] tcounts;
    checkCudaErrors(cudaFree(tcounts));
  }
  __device__ offset_t count() const {
    offset_t cnt = 0;
    for (u32 t = 0; t < nblocks; t++)
      cnt += tcounts[t];
    return cnt;
  }

  __device__ void genUnodes(const u32 id, const u32 uorv) {
#ifdef NEEDSYNC
    u32 last[NX];;
#endif

    u8 const *base = (u8 *)buckets;
    indexer<ZBUCKETSIZE> dst;
    const u32 starty = NY *  id    / nblocks;
    const u32   endy = NY * (id+1) / nblocks;
    u32 edge = starty << YZBITS, endedge = edge + NYZ;
    offset_t sumsize = 0;
    for (u32 my = starty; my < endy; my++, endedge += NYZ) {
      dst.matrixv(my);
#ifdef NEEDSYNC
      for (u32 x=0; x < NX; x++)
        last[x] = edge;
#endif
      for (; edge < endedge; edge += 1) {
// bit        28..21     20..13    12..0
// node       XXXXXX     YYYYYY    ZZZZZ
        const u32 node = _sipnode(&sip_keys, edge, uorv);
        const u32 ux = node >> YZBITS;
        const BIGTYPE zz = (BIGTYPE)edge << YZBITS | (node & YZMASK);
#ifndef NEEDSYNC
// bit        39..21     20..13    12..0
// write        edge     YYYYYY    ZZZZZ
        *(BIGTYPE *)(base+dst.index[ux]) = zz;
        dst.index[ux] += BIGSIZE0;
#else
        if (zz) {
          for (; unlikely(last[ux] + NNONYZ <= edge); last[ux] += NNONYZ, dst.index[ux] += BIGSIZE0)
            *(u32 *)(base+dst.index[ux]) = 0;
          *(u32 *)(base+dst.index[ux]) = zz;
          dst.index[ux] += BIGSIZE0;
          last[ux] = edge;
        }
#endif
      }
#ifdef NEEDSYNC
      for (u32 ux=0; ux < NX; ux++) {
        for (; last[ux]<endedge-NNONYZ; last[ux]+=NNONYZ) {
          *(u32 *)(base+dst.index[ux]) = 0;
          dst.index[ux] += BIGSIZE0;
        }
      }
#endif
      sumsize += dst.storev(buckets, my);
    }
    if (!id) printf("genUnodes round %2d size %u\n", uorv, sumsize/BIGSIZE0);
    tcounts[id] = sumsize/BIGSIZE0;
  }

  __device__ void genVnodes(const u32 id, const u32 uorv) {

    static const u32 NONDEGBITS = (BIGSLOTBITS < 2 * YZBITS ? BIGSLOTBITS : 2 * YZBITS) - ZBITS;
    static const u32 NONDEGMASK = (1 << NONDEGBITS) - 1;
    indexer<ZBUCKETSIZE> dst;
    indexer<TBUCKETSIZE> small;

    offset_t sumsize = 0;
    u8 const *base = (u8 *)buckets;
    u8 const *small0 = (u8 *)tbuckets[id];
    const u32 startux = NX *  id    / nblocks;
    const u32   endux = NX * (id+1) / nblocks;
    for (u32 ux = startux; ux < endux; ux++) { // matrix x == ux
      small.matrixu(0);
      for (u32 my = 0 ; my < NY; my++) {
        u32 edge = my << YZBITS;
        u8    *readbig = buckets[ux][my].bytes;
        u8 const *endreadbig = readbig + buckets[ux][my].size;
// printf("id %d x %d y %d size %u read %d\n", id, ux, my, buckets[ux][my].size, readbig-base);
        for (; readbig < endreadbig; readbig += BIGSIZE0) {
// bit     39/31..21     20..13    12..0
// read         edge     UYYYYY    UZZZZ   within UX partition
          BIGTYPE e = *(BIGTYPE *)readbig;
#if BIGSIZE0 > 4
          e &= BIGSLOTMASK0;
#elif defined NEEDSYNC
          if (unlikely(!e)) { edge += NNONYZ; continue; }
#endif
          edge += ((u32)(e >> YZBITS) - edge) & (NNONYZ-1);
// if (ux==78 && my==243) printf("id %d ux %d my %d e %08x prefedge %x edge %x\n", id, ux, my, e, e >> YZBITS, edge);
          const u32 uy = (e >> ZBITS) & YMASK;
// bit         39..13     12..0
// write         edge     UZZZZ   within UX UY partition
          *(u64 *)(small0+small.index[uy]) = ((u64)edge << ZBITS) | (e & ZMASK);
// printf("id %d ux %d y %d e %010lx e' %010x\n", id, ux, my, e, ((u64)edge << ZBITS) | (e >> YBITS));
          small.index[uy] += SMALLSIZE;
        }
        if (unlikely(edge >> NONYZBITS != (((my+1) << YZBITS) - 1) >> NONYZBITS))
        { printf("OOPS1: id %d ux %d y %d edge %x vs %x\n", id, ux, my, edge, ((my+1)<<YZBITS)-1); exit(0); }
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
#ifdef SAVEEDGES
        u32 *edges0 = (u32 *)(buckets[ux][uy].bytes + NTRIMMEDZ * BIGSIZE);
#else
        u32 *edges0 = tedges[id];
#endif
        u32 *edges = edges0, edge = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE) {
// bit         39..13     12..0
// read          edge     UZZZZ    sorted by UY within UX partition
          const u64 e = *(u64 *)rdsmall;
          edge += ((e >> ZBITS) - edge) & NONDEGMASK;
// if (id==0) printf("id %d ux %d uy %d e %010lx pref %4x edge %x mask %x\n", id, ux, uy, e, e>>ZBITS, edge, NONDEGMASK);
          *edges = edge;
          const u32 z = e & ZMASK;
          *zs = z;
          const u32 delta = degs[z] ? 1 : 0;
          edges += delta;
          zs    += delta;
        }
        if (unlikely(edge >> NONDEGBITS != EDGEMASK >> NONDEGBITS))
        { printf("OOPS2: id %d ux %d uy %d edge %x vs %x\n", id, ux, uy, edge, EDGEMASK); exit(0); }
        assert(edges - edges0 < NTRIMMEDZ);
        const u16 *readz = tzs[id];
        const u32 *readedge = edges0;
        int64_t uy34 = (int64_t)uy << YZZBITS;
        for (; readedge < edges; readedge++, readz++) {
          const u32 node = _sipnode(&sip_keys, *readedge, uorv);
          const u32 vx = node >> YZBITS; // & XMASK;
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          *(u64 *)(base+dst.index[vx]) = uy34 | ((u64)*readz << YZBITS) | (node & YZMASK);
// printf("id %d ux %d y %d edge %08x e' %010lx vx %d\n", id, ux, uy, *readedge, uy34 | ((u64)(node & YZMASK) << ZBITS) | *readz, vx);
          dst.index[vx] += BIGSIZE;
        }
      }
      sumsize += dst.storeu(buckets, ux);
    }
    if (!id) printf("genVnodes round %2d size %u\n", uorv, sumsize/BIGSIZE);
    tcounts[id] = sumsize/BIGSIZE;
  }

  void trim() {
    __global__ void _genUnodes(edgetrimmer *et, const u32 uorv);
    __global__ void _genVnodes(edgetrimmer *et, const u32 uorv);

    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    _genUnodes<<<nblocks,threadsperblock>>>(dt, 0);
    _genVnodes<<<nblocks,threadsperblock>>>(dt, 1);
#if 0
    for (u32 round = 2; round < ntrims; round += 2) {
      if (round < COMPRESSROUND) {
        if (round < EXPANDROUND)
          trimedges<BIGSIZE, BIGSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
        else if (round == EXPANDROUND)
          trimedges<BIGSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
        else trimedges<BIGGERSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
      } else if (round==COMPRESSROUND) {
        trimrename<BIGGERSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
      } else trimedges1<true><<<nblocks,threadsperblock>>>(dt, round);
      if (round < COMPRESSROUND) {
        if (round+1 < EXPANDROUND)
          trimedges<BIGSIZE, BIGSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
        else if (round+1 == EXPANDROUND)
          trimedges<BIGSIZE, BIGGERSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
        else trimedges<BIGGERSIZE, BIGGERSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
      } else if (round==COMPRESSROUND) {
        trimrename<BIGGERSIZE, sizeof(u32), false><<<nblocks,threadsperblock>>>(dt, round+1);
      } else trimedges1<false><<<nblocks,threadsperblock>>>(dt, round+1);
    }
#endif
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    printf("%d trims completed in %.3f seconds\n", ntrims, duration / 1000.0f);
  }
};

__global__ void _genUnodes(edgetrimmer *et, const u32 uorv) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  et->genUnodes(id, uorv);
}

__global__ void _genVnodes(edgetrimmer *et, const u32 uorv) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  et->genVnodes(id, uorv);
}

#define NODEBITS (EDGEBITS + 1)

// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << ((NODEBITS+2)/3);

const static u32 CUCKOO_SIZE = 2 * NX * NYZ2;

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

typedef u32 proof[PROOFSIZE];

class solver_ctx {
public:
  edgetrimmer *trimmer;
  zbucket<ZBUCKETSIZE> (*buckets)[NY];
  u32 *cuckoo;
  bool showcycle;
  proof cycleus;
  proof cyclevs;
  std::bitset<NXY> uxymap;
  std::vector<u32> sols; // concatanation of all proof's indices

  solver_ctx(const u32 n_threads, const u32 n_trims, bool allrounds, bool show_cycle) {
    trimmer = new edgetrimmer(n_threads, n_trims, allrounds);
    showcycle = show_cycle;
    cuckoo = 0;
  }
  void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer->sip_keys);
    sols.clear();
  }
  ~solver_ctx() {
    delete cuckoo;
    delete trimmer;
  }
  u64 sharedbytes() const {
    return sizeof(zbucket<ZBUCKETSIZE>[NX][NY]);
  }
  u32 threadbytes() const {
    return sizeof(thread_ctx) + sizeof(zbucket<TBUCKETSIZE>[NY]) + sizeof(zbucket8) + sizeof(zbucket16) + sizeof(zbucket32);
  }

  void recordedge(const u32 i, const u32 u2, const u32 v2) {
    const u32 u1 = u2/2;
    const u32 ux = u1 >> YZ2BITS;
    u32 uyz = buckets[ux][(u1 >> Z2BITS) & YMASK].renameu1[u1 & Z2MASK];
    assert(uyz < NYZ1);
    const u32 v1 = v2/2;
    const u32 vx = v1 >> YZ2BITS;
    u32 vyz = buckets[(v1 >> Z2BITS) & YMASK][vx].renamev1[v1 & Z2MASK];
    assert(vyz < NYZ1);
#if COMPRESSROUND > 0
    uyz = buckets[ux][uyz >> Z1BITS].renameu[uyz & Z1MASK];
    vyz = buckets[vyz >> Z1BITS][vx].renamev[vyz & Z1MASK];
#endif
    const u32 u = ((ux << YZBITS) | uyz) << 1;
    const u32 v = ((vx << YZBITS) | vyz) << 1 | 1;
    printf(" (%x,%x)", u, v);
#ifdef SAVEEDGES
    u32 *readedges = buckets[ux][uyz >> ZBITS].edges, *endreadedges = readedges + NTRIMMEDZ;
    for (; readedges < endreadedges; readedges++) {
      u32 edge = *readedges;
      if (sipnode(&trimmer->sip_keys, edge, 1) == v && sipnode(&trimmer->sip_keys, edge, 0) == u) {
        sols.push_back(edge);
        return;
      }
    }
    assert(0);
#else
    cycleus[i] = u/2;
    cyclevs[i] = v/2;
    uxymap[u/2 >> ZBITS] = 1;
#endif
  }

  void solution(const u32 *us, u32 nu, const u32 *vs, u32 nv) {
    printf("Nodes");
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
    while (nu--)
      recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
      recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    printf("\n");
    if (showcycle) {
#ifndef SAVEEDGES
      void *matchworker(void *vp);

      sols.resize(sols.size() + PROOFSIZE);
      match_ctx *threads = new match_ctx[trimmer->nthreads];
      for (u32 t = 0; t < trimmer->nthreads; t++) {
        threads[t].id = t;
        threads[t].solver = this;
        int err = pthread_create(&threads[t].thread, NULL, matchworker, (void *)&threads[t]);
        assert(err == 0);
      }
      for (u32 t = 0; t < trimmer->nthreads; t++) {
        int err = pthread_join(threads[t].thread, NULL);
        assert(err == 0);
      }
#endif
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), nonce_cmp);
    }
  }

  static const u32 CUCKOO_NIL = ~0;

  u32 path(u32 u, u32 *us) const {
    u32 nu, u0 = u;
    for (nu = 0; u != CUCKOO_NIL; u = cuckoo[u]) {
      if (nu >= MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (!~nu)
          printf("maximum path length exceeded\n");
        else printf("illegal %4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
        exit(0);
      }
      us[nu++] = u;
    }
    return nu-1;
  }

  void findcycles() {
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];

    for (u32 vx = 0; vx < NX; vx++) {
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = buckets[ux][vx];
        u32 *readbig = (u32 *)zb.bytes, *endreadbig = (u32 *)((u8 *)readbig + zb.size);
// printf("id %d vx %d ux %d size %u\n", id, vx, ux, zb.size/4);
        for (; readbig < endreadbig; readbig++) {
// bit        29..22    21..15     14..7     6..0
// write      UYYYYY    UZZZZ'     VYYYY     VZZ'   within VX partition
          const u32 e = *readbig;
          const u32 uxyz = (ux << YZ1BITS) | (e >> YZ1BITS);
          const u32 vxyz = (vx << YZ1BITS) | (e & YZ1MASK);
          const u32 u0 = uxyz << 1, v0 = (vxyz << 1) | 1;
          if (u0 != CUCKOO_NIL) {
            u32 nu = path(u0, us), nv = path(v0, vs);
// printf("vx %02x ux %02x e %08x uxyz %06x vxyz %06x u0 %x v0 %x nu %d nv %d\n", vx, ux, e, uxyz, vxyz, u0, v0, nu, nv);
            if (us[nu] == vs[nv]) {
              const u32 min = nu < nv ? nu : nv;
              for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
              const u32 len = nu + nv + 1;
              printf("%4d-cycle found\n", len);
              if (len == PROOFSIZE)
                solution(us, nu, vs, nv);
            } else if (nu < nv) {
              while (nu--)
                cuckoo[us[nu+1]] = us[nu];
              cuckoo[u0] = v0;
            } else {
              while (nv--)
                cuckoo[vs[nv+1]] = vs[nv];
              cuckoo[v0] = u0;
            }
          }
        }
      }
    }
    printf("findcycles\n");
  }

  int solve() {
    assert((u64)CUCKOO_SIZE * sizeof(u32) <= trimmer->nblocks * sizeof(zbucket<TBUCKETSIZE>[NY]));
    trimmer->trim();
    buckets = new zbucket<ZBUCKETSIZE>[NX][NY];
    cudaMemcpy(buckets, trimmer->buckets, sizeof(zbucket<ZBUCKETSIZE>[NX][NY]), cudaMemcpyDeviceToHost);
    u32 *cuckoo = new u32[CUCKOO_SIZE];
    memset(cuckoo, (int)CUCKOO_NIL, CUCKOO_SIZE * sizeof(u32));
    findcycles();
    delete[] cuckoo;
    delete[] buckets;
    return sols.size() / PROOFSIZE;
  }
};

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

int main(int argc, char **argv) {
  int nblocks = 64;
  int ntrims = 96;
  int tpb = 0;
  int nonce = 0;
  int range = 1;
  char header[HEADERLEN];
  u32 len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "h:n:m:r:t:p:x:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len == sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 't':
        nblocks = atoi(optarg);
        break;
      case 'p':
        tpb = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
    }
  }
  // if (!tpb) // if not set, then default threads per block to roughly square root of threads
    // for (tpb = 1; tpb*tpb < nthreads; tpb *= 2) ;

  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads %d per block\n", ntrims, nblocks, tpb);

  solver_ctx ctx(nblocks, ntrims);

  u64 sbytes = ctx.sharedbytes();
  u32 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory at %lx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer->buckets);
  printf("%dx%d%cB thread memory at %lx,\n", nblocks, tbytes, " KMGT"[tunit], (u64)ctx.trimmer->tbuckets);
  printf("and %d buckets.\n", NX);

  checkCudaErrors(cudaMalloc((void**)&ctx.alive.bits, edgeBytes));
  checkCudaErrors(cudaMalloc((void**)&ctx.nonleaf.bits, nodeBytes));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  for (int r = 0; r < range; r++) {
    cudaEventRecord(start, NULL);
    checkCudaErrors(cudaMemset(ctx.alive.bits, 0, edgeBytes));
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 %lx %lx\n", nonce+r, ctx.trimmer->sip_keys.k0, ctx.trimmer->sip_keys.k1);
    u32 nsols = ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (u32 i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
      printf("\n");
    }
    sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);

  checkCudaErrors(cudaFree(ctx.alive.bits));
  checkCudaErrors(cudaFree(ctx.nonleaf.bits));
  return 0;
}
