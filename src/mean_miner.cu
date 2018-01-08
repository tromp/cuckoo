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
  uint2 v0 = vectorize(keys.k0), v1 = vectorize(keys.k1), v2 = vectorize(keys.k2), v3 = vectorize(keys.k3) ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= vectorize(0xff);
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return devectorize(v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

#else

__device__ node_t dipnode(siphash_keys &keys, edge_t nce, u32 uorv) {
  u64 nonce = 2*nce + uorv;
  u64 v0 = keys.k0, v1 = keys.k0, v2 = keys.k2, v3 = keys.k3^ nonce;
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
#define BIGSIZE 5
#endif

// YZ compression round; must be even
#ifndef COMPRESSROUND
#define COMPRESSROUND 14
#endif
// size in bytes of a small bucket entry
#define SMALLSIZE BIGSIZE

// initial entries could be smaller at percent or two slowdown
#ifndef BIGSIZE0
#define BIGSIZE0 BIGSIZE
#endif

typedef uint8_t u8;
typedef uint16_t u16;

#if EDGEBITS >= 30
typedef u64 offset_t;
#else
typedef u32 offset_t;
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
const static u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const static u32 NONYZBITS     = BIGSLOTBITS0 - YZBITS;
const static u32 NNONYZ        = 1 << NONYZBITS;

const static u32 Z2BUCKETSIZE = NYZ2 >> 3;

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
const static u32 ZBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE0;
const static u32 TBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE;

template<u32 BUCKETSIZE, u32 NRENAME, u32 NRENAME1, u32 NSAVEDEDGES>
struct zbucket {
  u32 size;
  const static u32 RENAMESIZE = 2*NZ2 + 2*NZ1;
  union {
    u8 bytes[BUCKETSIZE];
    struct {
      u32 words[BUCKETSIZE/sizeof(u32) - 2*NRENAME1 - 2*NRENAME - NSAVEDEDGES];
      u32 renameu1[NRENAME1];
      u32 renamev1[NRENAME1];
      u32 renameu[NRENAME];
      u32 renamev[NRENAME];
      u32 edges[NSAVEDEDGES];
    };
  };
  __device__ u32 setsize(u8 const *end) {
    size = end - bytes;
    assert(size <= BUCKETSIZE);
    return size;
  }
};

const static u32 TWICE_WORDS = ((2 * NZ) / 32);

class twice_set {
public:
  u32 bits[TWICE_WORDS];
  __device__ void reset() {
    memset(bits, 0, TWICE_WORDS * sizeof(u32));
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
};

// template<u32 BUCKETSIZE>
// using yzbucket = zbucket<BUCKETSIZE>[NY];
// template <u32 BUCKETSIZE>
// using matrix = yzbucket<BUCKETSIZE>[NX];

template<u32 BUCKETSIZE, u32 NR, u32 NR1, u32 NSE>
struct indexer {
  offset_t index[NX];

  __device__ void matrixv(const u32 y) {
    const zbucket<BUCKETSIZE, NR, NR1, NSE> (*foo)[NY] = 0;
    for (u32 x = 0; x < NX; x++)
      index[x] = foo[x][y].bytes - (u8 *)foo;
  }
  __device__ offset_t storev(zbucket<BUCKETSIZE, NR, NR1, NSE> (*buckets)[NY], const u32 y) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 x = 0; x < NX; x++)
      sumsize += buckets[x][y].setsize(base+index[x]);
    return sumsize;
  }
  __device__ void matrixu(const u32 x) {
    const zbucket<BUCKETSIZE, NR, NR1, NSE> (*foo)[NY] = 0;
    for (u32 y = 0; y < NY; y++)
      index[y] = foo[x][y].bytes - (u8 *)foo;
  }
  __device__ offset_t storeu(zbucket<BUCKETSIZE, NR, NR1, NSE> (*buckets)[NY], const u32 x) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 y = 0; y < NY; y++)
      sumsize += buckets[x][y].setsize(base+index[y]);
    return sumsize;
  }
};

#define likely(x)   ((x)!=0)
#define unlikely(x) (x)

class edgetrimmer; // avoid circular references

typedef struct {
  u32 id;
  edgetrimmer *et;
} thread_ctx;

typedef u8 zbucket8[NYZ1];
typedef u8 zbucket82[NYZ1*2];
typedef u16 zbucket16[NTRIMMEDZ];
typedef u32 zbucket32[NTRIMMEDZ];

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  edgetrimmer *dt;
  zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> (*buckets)[NY];
  zbucket<TBUCKETSIZE,0,0,0> (*tbuckets)[NY];
  zbucket82 *tdegs;
  offset_t *tcounts;
  u32 ntrims;
  u32 nblocks;
  u32 threadsperblock;
  bool showall;
  u32 *uvnodes;
  proof sol;

  // __device__ void touch(u8 *p, const offset_t n) {
  //   for (offset_t i=0; i<n; i+=4096)
  //     *(u32 *)(p+i) = 0;
  // }
  edgetrimmer(const u32 n_blocks, const u32 tpb, const u32 n_trims, const bool show_all) {
    nblocks = n_blocks;
    threadsperblock = tpb;
    ntrims   = n_trims;
    showall = show_all;
    checkCudaErrors(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    // buckets  = new yzbucket<ZBUCKETSIZE>[NX];
    checkCudaErrors(cudaMalloc((void**)&buckets, sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>[NX][NY])));
    // touch((u8 *)buckets, sizeof(matrix<ZBUCKETSIZE>));
    // tbuckets = new yzbucket<TBUCKETSIZE>[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tbuckets, nblocks * sizeof(zbucket<TBUCKETSIZE,0,0,0>[NY])));
    // touch((u8 *)tbuckets, nthreads * sizeof(yzbucket<TBUCKETSIZE>));
    // tdegs   = new zbucket82[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tdegs, nblocks * sizeof(zbucket82)));
    // tcounts = new offset_t[nthreads];
    checkCudaErrors(cudaMalloc((void**)&tcounts, nblocks * sizeof(offset_t)));
    checkCudaErrors(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
  }
  ~edgetrimmer() {
    checkCudaErrors(cudaFree(buckets));
    checkCudaErrors(cudaFree(tbuckets));
    checkCudaErrors(cudaFree(tdegs));
    checkCudaErrors(cudaFree(tcounts));
    checkCudaErrors(cudaFree(uvnodes));
  }
  __device__ offset_t count() const {
    offset_t cnt = 0;
    for (u32 t = 0; t < nblocks; t++)
      cnt += tcounts[t];
    return cnt;
  }

  __device__ void write40(u8 *p64, const u64 x) {
    memcpy(p64, (u8 *)&x, BIGSIZE);
  }

  __device__ u16 read16(const u8 *p64) {
    u16 foo;
    memcpy((u8 *)&foo, p64, 2);
    return foo;
  }

  __device__ u64 read40(const u8 *p64) {
    u64 foo = 0;
    memcpy((u8 *)&foo, p64, BIGSIZE);
    return foo;
  }

  __device__ void genUnodes(const u32 uorv) {
    __shared__ indexer<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> dst;

    u8 * const base = (u8 *)buckets;
    u32 y          = NY *  blockIdx.x    / nblocks;
    const u32 endy = NY * (blockIdx.x+1) / nblocks;
    offset_t sumsize = 0;
    for (; y < endy; y++) {
      if (!threadIdx.x)
	dst.matrixv(y);
      __syncthreads();
      u32 edge     = y << YZBITS;
      const u32 endedge  = edge + NYZ;
      for (edge += threadIdx.x; edge < endedge; edge += blockDim.x) {
// bit        28..21     20..13    12..0
// node       XXXXXX     YYYYYY    ZZZZZ
        const u32 node = dipnode(sip_keys, edge, uorv);
        const u32 ux = node >> YZBITS;
        const u64 zz = (u64)edge << YZBITS | (node & YZMASK);
// bit        39..21     20..13    12..0
// write        edge     YYYYYY    ZZZZZ
        const u32 idx = atomicAdd(&dst.index[ux], BIGSIZE0);
        write40(base+idx, zz);
      }
      __syncthreads();
      if (!threadIdx.x)
        sumsize += dst.storev(buckets, y);
    }
    if (!blockIdx.x && !threadIdx.x) {
      printf("genUnodes round %2d size %u\n", uorv, sumsize/BIGSIZE0);
      //for (u32 j=0; j<65536; j++)
        //printf("%d %010lx\n", j, read40(base+4+j*BIGSIZE));
      tcounts[blockIdx.x] = sumsize/BIGSIZE0;
    }
  }

  __device__ void genVnodes(const u32 uorv) {
    static const u32 NONDEGBITS = (BIGSLOTBITS < 2 * YZBITS ? BIGSLOTBITS : 2 * YZBITS) - ZBITS;
    static const u32 NONDEGMASK = (1 << NONDEGBITS) - 1;
    __shared__ indexer<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> dst;
    __shared__ indexer<TBUCKETSIZE,0,0,0> small;
    __shared__ twice_set degs;

    offset_t sumsize = 0;
    u8 * const base = (u8 *)buckets;
    u8 * const small0 = (u8 *)tbuckets[blockIdx.x];
    u32          ux = NX *  blockIdx.x    / nblocks;
    const u32 endux = NX * (blockIdx.x+1) / nblocks;
    for (; ux < endux; ux++) {
      if (!threadIdx.x)
        small.matrixu(0);
      __syncthreads();
      for (u32 my = 0 ; my < NY; my++) {
        u32 edge = my << YZBITS;
        const u8           *readbig = buckets[ux][my].bytes;
        const u8 * const endreadbig = readbig + buckets[ux][my].size;
// printf("id %d x %d y %d size %u read %d\n", blockIdx.x, ux, my, buckets[ux][my].size, readbig-base);
        for (readbig += BIGSIZE0*threadIdx.x; readbig < endreadbig; readbig += BIGSIZE0*blockDim.x) {
// bit     39/31..22     21..15    14..0
// read         edge     UYYYYY    UZZZZ   within UX partition
          const u64 e = read40(readbig);
// u32 oldedge = edge;
	  const u32 lag = NNONYZ >> 2;
          edge += (((u32)(e >> YZBITS) - edge + lag) & (NNONYZ-1)) - lag;
// if (blockIdx.x==4 && edge>oldedge+4096) printf("oldedge %x edge %x delta %d\n",  oldedge, edge, oldedge+NNONYZ-edge);
// if (ux==78 && my==243) printf("id %d ux %d my %d e %08x prefedge %x edge %x\n", blockIdx.x, ux, my, e, e >> YZBITS, edge);
          const u32 uy = (e >> ZBITS) & YMASK;
// bit         39..15     14..0
// write         edge     UZZZZ   within UX UY partition
          const u32 idx = atomicAdd(&small.index[uy], SMALLSIZE);
          write40(small0+idx, ((u64)edge << ZBITS) | (e & ZMASK));;

// printf("id %d ux %d y %d e %010lx e' %010x\n", blockIdx.x, ux, my, e, ((u64)edge << ZBITS) | (e >> YBITS));
        }
        if (unlikely(edge >> NONYZBITS != (((my+1) << YZBITS) - 1) >> NONYZBITS))
        { printf("OOPS1: id %d ux %d y %d edge %x vs %x\n", blockIdx.x, ux, my, edge, ((my+1)<<YZBITS)-1); assert(0); }
      }
      if (!threadIdx.x) {
        small.storeu(tbuckets+blockIdx.x, 0);
        dst.matrixu(ux);
      }
      for (u32 uy = 0 ; uy < NY; uy++) {
        if (!threadIdx.x)
          degs.reset();
        __syncthreads();
        u8 *readsmall = tbuckets[blockIdx.x][uy].bytes, *endreadsmall = readsmall + tbuckets[blockIdx.x][uy].size;
// if (blockIdx.x==1) printf("id %d ux %d y %d size %u sumsize %u\n", blockIdx.x, ux, uy, tbuckets[blockIdx.x][uy].size/BIGSIZE, sumsize);
	readsmall += SMALLSIZE * threadIdx.x;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE*blockDim.x)
	  degs.set(read16(rdsmall) & ZMASK);
        __syncthreads();
        u32 edge = 0;
	u64 uy34 = (u64)uy << YZZBITS;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE*blockDim.x) {
// bit         39..13     12..0
// read          edge     UZZZZ    sorted by UY within UX partition
          const u64 e = read40(rdsmall);
// u32 oldedge = edge;
	  const u32 lag = NONDEGMASK >> 2;
          edge += (((e >> ZBITS) - edge + lag) & NONDEGMASK) - lag;
// if (blockIdx.x==4 && edge>oldedge+1000000) printf("oldedge %x edge %x delta %d\n",  oldedge, edge, oldedge+NONDEGMASK+1-edge);
// if (blockIdx.x==0) printf("id %d ux %d uy %d e %010lx pref %4x edge %x mask %x\n", blockIdx.x, ux, uy, e, e>>ZBITS, edge, NONDEGMASK);
	  const u32 z = e & ZMASK;
          if (degs.test(z)) {
            const u32 node = dipnode(sip_keys, edge, uorv);
            const u32 vx = node >> YZBITS; // & XMASK;
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
            const u32 idx = atomicAdd(&dst.index[vx], BIGSIZE);
            write40(base+idx, uy34 | ((u64)z << YZBITS) | (node & YZMASK));
// printf("id %d ux %d y %d edge %08x e' %010lx vx %d\n", blockIdx.x, ux, uy, *readedge, uy34 | ((u64)(node & YZMASK) << ZBITS) | *readz, vx);
	  }
        }
        if (unlikely(edge >> NONDEGBITS != EDGEMASK >> NONDEGBITS))
        { printf("OOPS2: id %d ux %d uy %d edge %x vs %x\n", blockIdx.x, ux, uy, edge, EDGEMASK); assert(0); }
      }
      __syncthreads();
      if (!threadIdx.x)
        sumsize += dst.storeu(buckets, ux);
    }
    if (!blockIdx.x) printf("genVnodes round %2d size %u\n", uorv, sumsize/BIGSIZE);
    if (!threadIdx.x)
      tcounts[blockIdx.x] = sumsize/BIGSIZE;
  }

#define mymin(a,b) ((a) < (b) ? (a) : (b))

  template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
  __device__ void trimedges(const u32 round) {
    const u32 SRCSLOTBITS = mymin(SRCSIZE * 8, 2 * YZBITS);
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 DSTSLOTBITS = mymin(DSTSIZE * 8, 2 * YZBITS);
    const u32 DSTPREFBITS = DSTSLOTBITS - YZZBITS;
    const u32 DSTPREFMASK = (1 << DSTPREFBITS) - 1;
    __shared__ indexer<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> dst;
    __shared__ indexer<TBUCKETSIZE,0,0,0> small;
    __shared__ twice_set degs;

    u8 * const base   = (u8 *)buckets;
    u8 * const small0 = (u8 *)tbuckets[blockIdx.x];
    u32 vx          = NY *  blockIdx.x    / nblocks;
    const u32 endvx = NY * (blockIdx.x+1) / nblocks;
    offset_t sumsize = 0;
    for (; vx < endvx; vx++) {
      if (!threadIdx.x)
        small.matrixu(0);
      for (u32 ux = 0; ux < NX; ux++) {
        __syncthreads();
        u32 uxyz = ux << YZBITS;
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        const u8 *readbig = zb.bytes;
        const u8 * const endreadbig = readbig + zb.size;
// if (!blockIdx.x && !threadIdx.x)
// printf("round %d vx %d ux %d size %u\n", round, vx, ux, pzb->size/SRCSIZE);
        for (readbig += SRCSIZE*threadIdx.x; readbig < endreadbig; readbig += SRCSIZE*blockDim.x) {
// bit     43/39..37    36..22     21..15     14..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          const u64 e = read40(readbig); // & SRCSLOTMASK;
// if (!blockIdx.x && !threadIdx.x && round==4 && ux+vx==0)
// printf("id %d vx %d ux %d e %010llx suffUXYZ %05x suffUXY %03x UXYZ %08x UXY %04x mask %x\n", blockIdx.x, vx, ux, e, (u32)(e >> YZBITS), (u32)(e >> YZZBITS), uxyz, uxyz>>ZBITS, SRCPREFMASK);

	  const u32 lag = SRCPREFMASK >> 2;
          uxyz += (((u32)(e >> YZBITS) - uxyz + lag) & SRCPREFMASK) - lag;
          const u32 vy = (e >> ZBITS) & YMASK;
// bit     43/39..37    36..30     29..15     14..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          const u32 idx = atomicAdd(&small.index[vy], DSTSIZE);
          write40(small0+idx, ((u64)uxyz << ZBITS) | (e & ZMASK));
          uxyz &= ~ZMASK;
        }
        if (unlikely(uxyz >> YZBITS != ux))
        { printf("OOPS3: id %d vx %d ux %d UXY %x\n", blockIdx.x, vx, ux, uxyz); break; }
      }
      if (!threadIdx.x) {
        small.storeu(tbuckets+blockIdx.x, 0);
        TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      }
      for (u32 vy = 0 ; vy < NY; vy++) {
        const u64 vy34 = (u64)vy << YZZBITS;
        if (!threadIdx.x)
          degs.reset();
        __syncthreads();
        u8    *readsmall = tbuckets[blockIdx.x][vy].bytes, *endreadsmall = readsmall + tbuckets[blockIdx.x][vy].size;
// printf("id %d vx %d vy %d size %u sumsize %u\n", blockIdx.x, vx, vy, tbuckets[blockIdx.x][vx].size/BIGSIZE, sumsize);
        readsmall += DSTSIZE * threadIdx.x;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE*blockDim.x)
	  degs.set(read16(rdsmall) & ZMASK);
        __syncthreads();
        u32 ux = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE*blockDim.x) {
// bit     41/39..34    33..26     25..13     12..0
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
// bit     45/39..37    36..30     29..15     14..0      with XBITS==YBITS==7
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          const u64 e = read40(rdsmall); //  & DSTSLOTMASK;
	  const u32 lag = DSTPREFMASK >> 2;
          ux += (((u32)(e >> YZZBITS) - ux + lag) & DSTPREFMASK) - lag;
// if (vx==0x21 && ux==0x77) printf("id %d.%d vx %d vy %d e %010lx suffUX %02x UX %x mask %x\n", blockIdx.x, threadIdx.x, vx, vy, e, (u32)(e >> YZZBITS), ux, DSTPREFMASK);
// bit    41/39..34    33..21     20..13     12..0
// write     VYYYYY    VZZZZZ     UYYYYY     UZZZZ   within UX partition
          if (degs.test(e & ZMASK)) {
            const u32 idx = atomicAdd(&dst.index[ux], DSTSIZE);
            write40(base+idx, vy34 | ((e & ZMASK) << YZBITS) | ((e >> ZBITS) & YZMASK));
	  }
        }
        if (unlikely(ux >> DSTPREFBITS != XMASK >> DSTPREFBITS))
        { printf("OOPS4: id %d.%d vx %x ux %x vs %x\n", blockIdx.x, threadIdx.x, vx, ux, XMASK); }
      }
      __syncthreads();
      if (!threadIdx.x)
        sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    // if (showall || !blockIdx.x && !(round & (round+1)))
    if (!blockIdx.x && !threadIdx.x)
      printf("trimedges id %d round %2d size %u\n", blockIdx.x, round, sumsize/DSTSIZE);
  }

  template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
  __device__ void trimrename(const u32 round) {
     if (threadIdx.x) return;
    const u32 SRCSLOTBITS = mymin(SRCSIZE * 8, (TRIMONV ? YZBITS : YZ1BITS) + YZBITS);
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 SRCPREFBITS2 = SRCSLOTBITS - YZZBITS;
    const u32 SRCPREFMASK2 = (1 << SRCPREFBITS2) - 1;
    indexer<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> dst;
    indexer<TBUCKETSIZE,0,0,0> small;
    u32 maxnnid = 0;

    offset_t sumsize = 0;
    u8 *base = (u8 *)buckets;
    u8 *small0 = (u8 *)tbuckets[blockIdx.x];
    const u32 startvx = NY *  blockIdx.x    / nblocks;
    const u32   endvx = NY * (blockIdx.x+1) / nblocks;
    for (u32 vx = startvx; vx < endvx; vx++) {
      small.matrixu(0);
      for (u32 ux = 0 ; ux < NX; ux++) {
        u32 uyz = 0;
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        const u8 *readbig = zb.bytes, *endreadbig = readbig + zb.size;
// printf("id %d vx %d ux %d size %u\n", blockIdx.x, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        39..37    36..22     21..15     14..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition  if TRIMONV
// bit            36...22     21..15     14..0
// write          VYYYZZ'     UYYYYY     UZZZZ   within UX partition  if !TRIMONV
          const u64 e = read40(readbig); //  & SRCSLOTMASK;
          if (TRIMONV)
            uyz += ((u32)(e >> YZBITS) - uyz) & SRCPREFMASK;
          else uyz = e >> YZBITS;
// if (round==32 && ux==25) printf("id %d vx %d ux %d e %010lx suffUXYZ %05x suffUXY %03x UXYZ %08x UXY %04x mask %x\n", blockIdx.x, vx, ux, e, (u32)(e >> YZBITS), (u32)(e >> YZZBITS), uxyz, uxyz>>ZBITS, SRCPREFMASK);
          const u32 vy = (e >> ZBITS) & YMASK;
// bit        39..37    36..30     29..15     14..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition  if TRIMONV
// bit            36...30     29...15     14..0
// write          VXXXXXX     VYYYZZ'     UZZZZ   within UX UY partition  if !TRIMONV
          write40(small0+small.index[vy], ((u64)(ux << (TRIMONV ? YZBITS : YZ1BITS) | uyz) << ZBITS) | (e & ZMASK));
// if (TRIMONV&&vx==75&&vy==83) printf("id %d vx %d vy %d e %010lx e15 %x ux %x\n", blockIdx.x, vx, vy, ((u64)uxyz << ZBITS) | (e & ZMASK), uxyz, uxyz>>YZBITS);
          if (TRIMONV)
            uyz &= ~ZMASK;
          small.index[vy] += SRCSIZE;
        }
      }
      u16 *degs = (u16 *)tdegs[blockIdx.x];
      small.storeu(tbuckets+blockIdx.x, 0);
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      u32 newnodeid = 0;
      u32 *renames = TRIMONV ? buckets[0][vx].renamev : buckets[vx][0].renameu;
      u32 *endrenames = renames + NZ1;
      for (u32 vy = 0 ; vy < NY; vy++) {
        memset(degs, 0xff, 2*NZ);
        u8    *readsmall = tbuckets[blockIdx.x][vy].bytes, *endreadsmall = readsmall + tbuckets[blockIdx.x][vy].size;
// printf("id %d vx %d vy %d size %u sumsize %u\n", blockIdx.x, vx, vy, tbuckets[blockIdx.x][vx].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SRCSIZE)
          degs[read16(rdsmall) & ZMASK]++;
        u32 ux = 0;
        u32 nrenames = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SRCSIZE) {
// bit        39..37    36..30     29..15     14..0
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition  if TRIMONV
// bit            36...30     29...15     14..0
// read           VXXXXXX     VYYYZZ'     UZZZZ   within UX UY partition  if !TRIMONV
          const u64 e = read40(rdsmall); //  & SRCSLOTMASK;
          if (TRIMONV)
            ux += ((u32)(e >> YZZBITS) - ux) & SRCPREFMASK2;
          else ux = e >> YZZ1BITS;
          const u32 vz = e & ZMASK;
          u16 vdeg = degs[vz];
// if (TRIMONV&&vx==75&&vy==83) printf("id %d vx %d vy %d e %010lx e37 %x ux %x vdeg %d nrenames %d\n", blockIdx.x, vx, vy, e, e>>YZZBITS, ux, vdeg, nrenames);
          if (vdeg) {
            if (vdeg < 32) {
              degs[vz] = vdeg = 32 + nrenames++;
              *renames++ = vy << ZBITS | vz;
              if (renames == endrenames) {
                endrenames += (TRIMONV ? sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>[NY]) : sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>)) / sizeof(u32);
                renames = endrenames - NZ1;
              }
            }
// bit       36..22     21..15     14..0
// write     VYYZZ'     UYYYYY     UZZZZ   within UX partition  if TRIMONV
            if (TRIMONV)
                 write40(base+dst.index[ux], ((u64)(newnodeid + vdeg-32) << YZBITS ) | ((e >> ZBITS) & YZMASK));
            else *(u32 *)(base+dst.index[ux]) = ((newnodeid + vdeg-32) << YZ1BITS) | ((e >> ZBITS) & YZ1MASK);
// if (vx==44&&vy==58) printf("  id %d vx %d vy %d newe %010lx\n", blockIdx.x, vx, vy, vy28 | ((vdeg) << YZBITS) | ((e >> ZBITS) & YZMASK));
            dst.index[ux] += DSTSIZE;
          }
        }
        newnodeid += nrenames;
        if (TRIMONV && unlikely(ux >> SRCPREFBITS2 != XMASK >> SRCPREFBITS2))
        { printf("OOPS6: id %d vx %d vy %d ux %x vs %x\n", blockIdx.x, vx, vy, ux, XMASK); break; }
      }
      if (newnodeid > maxnnid)
        maxnnid = newnodeid;
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    if (showall || !blockIdx.x) printf("trimrename id %d round %2d size %u maxnnid %d\n", blockIdx.x, round, (u32)sumsize/DSTSIZE, maxnnid);
    assert(maxnnid < NYZ1);
    tcounts[blockIdx.x] = sumsize/DSTSIZE;
  }

  template <bool TRIMONV>
  __device__ void trimedges1(const u32 round) {
     if (threadIdx.x) return;
    indexer<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> dst;

    offset_t sumsize = 0;
    u8 *degs = tdegs[blockIdx.x];
    u8 const *base = (u8 *)buckets;
    const u32 startvx = NY *  blockIdx.x    / nblocks;
    const u32   endvx = NY * (blockIdx.x+1) / nblocks;
    for (u32 vx = startvx; vx < endvx; vx++) {
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      memset(degs, 0xff, NYZ1);
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        // printf("id %d vx %d ux %d size %d\n", blockIdx.x, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig++)
          degs[*readbig & YZ1MASK]++;
      }
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        for (; readbig < endreadbig; readbig++) {
// bit       29..22    21..15     14..7     6..0
// read      UYYYYY    UZZZZ'     VYYYY     VZZ'   within VX partition
          const u32 e = *readbig;
          const u32 vyz = e & YZ1MASK;
          // printf("id %d vx %d ux %d e %08lx vyz %04x uyz %04x\n", blockIdx.x, vx, ux, e, vyz, e >> YZ1BITS);
// bit       29..22    21..15     14..7     6..0
// write     VYYYYY    VZZZZ'     UYYYY     UZZ'   within UX partition
	  if ((u64)(base+dst.index[ux]) & 3ULL) { printf("HOLY FUCK!\n"); }
          *(u32 *)(base+dst.index[ux]) = (vyz << YZ1BITS) | (e >> YZ1BITS);
          dst.index[ux] += degs[vyz] ? sizeof(u32) : 0;
        }
      }
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    if (showall || !blockIdx.x && !(round & (round+1)))
      printf("trimedges1 id %d round %2d size %u\n", blockIdx.x, round, sumsize/sizeof(u32));
    tcounts[blockIdx.x] = sumsize/sizeof(u32);
  }

  template <bool TRIMONV>
  __device__ void trimrename1(const u32 round) {
    if (blockDim.x && threadIdx.x) return;
    const u32 BS = TRIMONV ? ZBUCKETSIZE : Z2BUCKETSIZE;
    const u32 NR1 = TRIMONV ? NZ1 : 0;
    const u32 NR2 = TRIMONV ? NZ2 : 0;
    const u32 NSE = TRIMONV ? NTRIMMEDZ: 0;
    indexer<BS,NR1,NR2,NSE> dst;
    u32 maxnnid = 0;

    offset_t sumsize = 0;
    u16 *degs = (u16 *)tdegs[blockIdx.x];
    u8 const *base = TRIMONV ? (u8 *)buckets : (u8 *)tbuckets;
    zbucket<BS,NR1,NR2,NSE> (*sbuckets)[NY] = (zbucket<BS,NR1,NR2,NSE> (*)[NY])base;
    const u32 startvx = NY *  blockIdx.x    / nblocks;
    const u32   endvx = NY * (blockIdx.x+1) / nblocks;
    for (u32 vx = startvx; vx < endvx; vx++) {
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      memset(degs, 0xff, 2 * NYZ1); // sets each u16 entry to 0xffff
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        // printf("id %d vx %d ux %d size %d\n", blockIdx.x, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig++)
          degs[*readbig & YZ1MASK]++;
      }
      u32 newnodeid = 0;
      u32 *renames = TRIMONV ? buckets[0][vx].renamev1 : buckets[vx][0].renameu1;
      u32 *endrenames = renames + NZ2;
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        for (; readbig < endreadbig; readbig++) {
// bit       29...15     14...0
// read      UYYYZZ'     VYYZZ'   within VX partition
          const u32 e = *readbig;
          const u32 vyz = e & YZ1MASK;
          u16 vdeg = degs[vyz];
          if (vdeg) {
            if (vdeg < 32) {
              degs[vyz] = vdeg = 32 + newnodeid++;
              *renames++ = vyz;
              if (renames == endrenames) {
                endrenames += (TRIMONV ? sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>[NY]) : sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>)) / sizeof(u32);
                renames = endrenames - NZ2;
              }
            }
// bit       25...15     14...0
// write     VYYZZZ"     UYYZZ'   within UX partition
            *(u32 *)(base+dst.index[ux]) = ((vdeg - 32) << (TRIMONV ? YZ1BITS : YZ2BITS)) | (e >> YZ1BITS);
            dst.index[ux] += sizeof(u32);
          }
        }
      }
      if (newnodeid > maxnnid)
        maxnnid = newnodeid;
      sumsize += TRIMONV ? dst.storev(sbuckets, vx) : dst.storeu(sbuckets, vx);
    }
    if (showall || !blockIdx.x) printf("trimrename1 id %d round %2d size %d maxnnid %d\n", blockIdx.x, round, (u32)(sumsize/sizeof(u32)), maxnnid);
    assert(maxnnid < NYZ2);
    tcounts[blockIdx.x] = sumsize/sizeof(u32);
  }

  __device__ void recoveredges() {
    __shared__ u32 u, ux, uyz, v, vx, vyz;

    if (threadIdx.x == 0) {
      const u32 u1 = uvnodes[2*blockIdx.x], v1 = uvnodes[2*blockIdx.x+1];
      ux = u1 >> YZ2BITS;
      vx = v1 >> YZ2BITS;
      uyz = buckets[ux][(u1 >> Z2BITS) & YMASK].renameu1[u1 & Z2MASK];
      assert(uyz < NYZ1);
      vyz = buckets[(v1 >> Z2BITS) & YMASK][vx].renamev1[v1 & Z2MASK];
      assert(vyz < NYZ1);
#if COMPRESSROUND > 0
      uyz = buckets[ux][uyz >> Z1BITS].renameu[uyz & Z1MASK];
      vyz = buckets[vyz >> Z1BITS][vx].renamev[vyz & Z1MASK];
#endif
      u = (ux << YZBITS) | uyz;
      v = (vx << YZBITS) | vyz;
      uvnodes[2*blockIdx.x] = u;
      uvnodes[2*blockIdx.x+1] = v;
    }
    __syncthreads();
  }

  __device__ void recoveredges1() {
    __shared__ u32 uxymap[NXY/32];

    for (u32 i = threadIdx.x; i < PROOFSIZE; i += blockDim.x) {
      const u32 uxy = uvnodes[2*i] >> ZBITS;
      atomicOr(&uxymap[uxy/32], 1 << uxy%32);
    }
    __syncthreads();
    for (u32 edge = blockIdx.x * blockDim.x + threadIdx.x; edge < NEDGES; edge += gridDim.x * blockDim.x) {
      const u32 u = dipnode(sip_keys, edge, 0);
      const u32 uxy = u  >> ZBITS;
      if ((uxymap[uxy/32] >> uxy%32) & 1) {
	for (u32 j = 0; j < PROOFSIZE; j++) {
           if (uvnodes[2*j] == u && dipnode(sip_keys, edge, 1) == uvnodes[2*j+1]) {
             sol[j] = edge;
           }
        }
      }
    }
  }

  void trim();
};

__global__ void _genUnodes(edgetrimmer *et, const u32 uorv) {
  et->genUnodes(uorv);
}

__global__ void _genVnodes(edgetrimmer *et, const u32 uorv) {
  et->genVnodes(uorv);
}

template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
__global__ void _trimedges(edgetrimmer *et, const u32 round) {
  et->trimedges<SRCSIZE, DSTSIZE, TRIMONV>(round);
}

template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
__global__ void _trimrename(edgetrimmer *et, const u32 round) {
  et->trimrename<SRCSIZE, DSTSIZE, TRIMONV>(round);
}

template <bool TRIMONV>
__global__ void _trimedges1(edgetrimmer *et, const u32 round) {
  et->trimedges1<TRIMONV>(round);
}

template <bool TRIMONV>
__global__ void _trimrename1(edgetrimmer *et, const u32 round) {
  et->trimrename1<TRIMONV>(round);
}

__global__ void _recoveredges(edgetrimmer *et) {
  et->recoveredges();
}

__global__ void _recoveredges1(edgetrimmer *et) {
  et->recoveredges1();
}

#ifdef EXPANDROUND
#define BIGGERSIZE BIGSIZE+1
#else
#define BIGGERSIZE BIGSIZE
#define EXPANDROUND COMPRESSROUND
#endif

  void edgetrimmer::trim() {
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float duration;

    cudaEventRecord(start, NULL);
    _genUnodes<<<nblocks,threadsperblock>>>(dt, 0);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    printf("genUnodes completed in %.3f seconds\n", duration / 1000.0f);

    cudaEventRecord(start, NULL);
    cudaEventRecord(start, NULL);
    _genVnodes<<<nblocks,threadsperblock>>>(dt, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    printf("genVnodes completed in %.3f seconds\n", duration / 1000.0f);

    // printf("bailing out...\n"); return;
    for (u32 round = 2; round < ntrims-2; round += 2) {
      if (round < COMPRESSROUND) {
        if (round < EXPANDROUND)
          _trimedges<BIGSIZE, BIGSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
        else if (round == EXPANDROUND)
          _trimedges<BIGSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
        else _trimedges<BIGGERSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
      } else if (round==COMPRESSROUND) {
        _trimrename<BIGGERSIZE, BIGGERSIZE, true><<<nblocks,threadsperblock>>>(dt, round);
      } else _trimedges1<true><<<nblocks,threadsperblock>>>(dt, round);
      if (round < COMPRESSROUND) {
        if (round+1 < EXPANDROUND)
          _trimedges<BIGSIZE, BIGSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
        else if (round+1 == EXPANDROUND)
          _trimedges<BIGSIZE, BIGGERSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
        else _trimedges<BIGGERSIZE, BIGGERSIZE, false><<<nblocks,threadsperblock>>>(dt, round+1);
      } else if (round==COMPRESSROUND) {
        _trimrename<BIGGERSIZE, sizeof(u32), false><<<nblocks,threadsperblock>>>(dt, round+1);
      } else _trimedges1<false><<<nblocks,threadsperblock>>>(dt, round+1);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    _trimrename1<true ><<<nblocks,threadsperblock>>>(dt, ntrims-2);
    checkCudaErrors(cudaDeviceSynchronize());
    _trimrename1<false><<<nblocks,threadsperblock>>>(dt, ntrims-1);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    printf("%d trims completed in %.3f seconds\n", ntrims, duration / 1000.0f);
  }

#define NODEBITS (EDGEBITS + 1)

// grow with cube root of size, hardly affected by trimming
const static u32 MAXPATHLEN = 8 << ((NODEBITS+2)/3);

const static u32 CUCKOO_SIZE = 2 * NX * NYZ2;

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

class solver_ctx {
public:
  edgetrimmer *trimmer;
  zbucket<Z2BUCKETSIZE,0,0,0> (*buckets)[NY];
  u32 *cuckoo;
  bool showcycle;
  u32 uvnodes[2*PROOFSIZE];
  std::bitset<NXY> uxymap;
  std::vector<u32> sols; // concatanation of all proof's indices

  solver_ctx(const u32 n_threads, const u32 tpb, const u32 n_trims, bool allrounds, bool show_cycle) {
    trimmer = new edgetrimmer(n_threads, tpb, n_trims, allrounds);
    showcycle = show_cycle;
    cuckoo = 0;
  }
  void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer->sip_keys);
    sols.clear();
  }
  ~solver_ctx() {
    delete trimmer;
  }
  u32 sharedbytes() const {
    return sizeof(zbucket<ZBUCKETSIZE,NZ1,NZ2,NTRIMMEDZ>[NX][NY]);
  }
  u32 threadbytes() const {
    return sizeof(thread_ctx) + sizeof(zbucket<TBUCKETSIZE,0,0,0>[NY]) + sizeof(zbucket82) + sizeof(zbucket16) + sizeof(zbucket32);
  }

  void recordedge(const u32 i, const u32 u2, const u32 v2) {
    uvnodes[2*i]   = u2/2;
    uvnodes[2*i+1] = v2/2;
  }

  void solution(const u32 *us, u32 nu, const u32 *vs, u32 nv) {
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
    while (nu--)
      recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
    recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    assert(ni == PROOFSIZE);
    sols.resize(sols.size() + PROOFSIZE);
    cudaMemcpy(trimmer->uvnodes, uvnodes, sizeof(uvnodes), cudaMemcpyHostToDevice);
    _recoveredges<<<PROOFSIZE,1>>>(trimmer->dt);
    _recoveredges1<<<1024,PROOFSIZE>>>(trimmer->dt);
    cudaMemcpy(&sols[sols.size() - PROOFSIZE], trimmer->dt->sol, sizeof(trimmer->sol), cudaMemcpyDeviceToHost);
    qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), nonce_cmp);
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

//    for (u32 vx = 0; vx < NX; vx++) {
//      for (u32 ux = 0 ; ux < NX; ux++) {
//        printf("vx %02x ux %02x size %d\n", vx, ux, buckets[ux][vx].size/4);
//      }
//    }
    for (u32 vx = 0; vx < NX; vx++) {
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<Z2BUCKETSIZE,0,0,0> &zb = buckets[ux][vx];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        for (; readbig < endreadbig; readbig++) {
// bit        21..11     10...0
// write      UYYZZZ'    VYYZZ'   within VX partition
          const u32 e = *readbig;
          const u32 uxyz = (ux << YZ2BITS) | (e >> YZ2BITS);
          const u32 vxyz = (vx << YZ2BITS) | (e & YZ2MASK);
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
    trimmer->trim();
    buckets = new zbucket<Z2BUCKETSIZE,0,0,0>[NX][NY];
    printf("start cudaMemcpy\n");
    checkCudaErrors(cudaMemcpy(buckets, trimmer->tbuckets, sizeof(zbucket<Z2BUCKETSIZE,0,0,0>[NX][NY]), cudaMemcpyDeviceToHost));
    printf("end cudaMemcpy\n");
    cuckoo = new u32[CUCKOO_SIZE];
    memset(cuckoo, (int)CUCKOO_NIL, CUCKOO_SIZE * sizeof(u32));
    printf("start findcycles\n");
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
  int ntrims = 68;
  int tpb = 1;
  int nonce = 0;
  int range = 1;
  int device = 0;
  bool allrounds = false;
  bool showcycle = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "ad:h:n:m:r:t:p:x:")) != -1) {
    switch (c) {
      case 'a':
        allrounds = true;
        break;
      case 'd':
        device = atoi(optarg);
        break;
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
      case 's':
        showcycle = true;
        break;
      case 'r':
        range = atoi(optarg);
        break;
    }
  }

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(device < nDevices);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  u32 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 10240; dbytes>>=10,dunit++) ;
  printf("  CUDA Device name: %s\n", prop.name);
  printf("  Memory: %d%c MaxThreadsPerBlock: %d\n", dbytes, " KMGT"[dunit], prop.maxThreadsPerBlock);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  cudaSetDevice(device);

  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads %d per block\n", ntrims, nblocks, tpb);

  solver_ctx ctx(nblocks, tpb, ntrims, allrounds, showcycle);

  u32 sbytes = ctx.sharedbytes();
  u32 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory at %llx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer->buckets);
  printf("%dx%d%cB thread memory at %llx,\n", nblocks, tbytes, " KMGT"[tunit], (u64)ctx.trimmer->tbuckets);
  printf("and %d buckets.\n", NX);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 %llx %llx\n", nonce+r, ctx.trimmer->sip_keys.k0, ctx.trimmer->sip_keys.k1);
    u32 nsols = ctx.solve();

    for (unsigned s = 0; s < nsols; s++) {
      printf("Solution");
      u32* prf = &ctx.sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)prf[i]);
      printf("\n");
      int pow_rc = verify(prf, &ctx.trimmer->sip_keys);
      if (pow_rc == POW_OK) {
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          printf("%02x", cyclehash[i]);
        printf("\n");
      } else {
        printf("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);

  return 0;
}
