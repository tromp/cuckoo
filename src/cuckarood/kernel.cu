// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

//Includes for IntelliSense
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <stdio.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

#define EDGEBITS 29
#define NNODES1 NEDGES1
#define NNODES2 NEDGES2

#define EDGEMASK (NEDGES2 - 1)

#define BKTMASK4K (4096-1)
#define BKTGRAN 64

#define XBITS 6
const u32 NX = 1 << XBITS;
const u32 XMASK = NX - 1;
const u32 NX2 = NX * NX;
const u32 X2MASK = NX2 - 1;
const u32 YBITS = XBITS;
const u32 NY = 1 << YBITS;
const u32 YZBITS = EDGEBITS - XBITS;
const u32 ZBITS = YZBITS - YBITS;
const u32 NZ = 1 << ZBITS;
const u32 ZMASK = NZ - 1;

__device__ __forceinline__ uint2 ld_cs_u32_v2(const uint2 *p_src)
{
  uint2 n_result;
  asm("ld.global.cs.v2.u32 {%0,%1}, [%2];"  : "=r"(n_result.x), "=r"(n_result.y) : "l"(p_src));
  return n_result;
}

__device__ __forceinline__ void st_cg_u32_v2(uint2 *p_dest, const uint2 n_value)
{
  asm("st.global.cg.v2.u32 [%0], {%1, %2};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y));
}

__device__ __forceinline__ void st_cg_u32_v4(uint4 *p_dest, const uint4 n_value)
{
  asm("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y), "r"(n_value.z), "r"(n_value.w));
}

__device__ __forceinline__  void setbit(u32 * ecounters, const int bucket)
{
  const int word = bucket >> 5;
  const unsigned char bit = bucket & 0x1F;
  const u32 mask = 1 << bit;
  atomicOr(ecounters + word, mask);
}

__device__ __forceinline__  bool testbit(u32 * ecounters, const int bucket)
{
  const int word = bucket >> 5;
  const unsigned char bit = bucket & 0x1F;
  return (ecounters[word] >> bit) & 1;
}

__constant__ siphash_keys dipkeys;
__constant__ u64 recovery[42];

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND {\
  v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
  v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
  v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
  v1 = ROTL(v1,17);   v3 = ROTL(v3,25); \
  v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
}
#define SIPBLOCK(b) {\
  v3 ^= (b);\
  for (short r = 0; r < 2; r++)\
  SIPROUND;\
  v0 ^= (b);\
  v2 ^= 0xff;\
  for (short r = 0; r < 4; r++)\
  SIPROUND;\
}
#define DUMP(E, dir) {\
  u64 lookup = E;\
  uint2 edge1 = make_uint2( (lookup & NODE1MASK) << 1 | dir, (lookup >> 31) & (NODE1MASK << 1) | dir);\
  int bucket = (edge1.x >> ZBITS) & X2MASK;\
  u64 edge64 = (((u64)edge1.y) << 32) | edge1.x;\
  for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {\
    u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);\
        if (ret2 == ret) {\
      int block = bucket / (NX2 / NA);\
      int shift = (bktOutSize * NX2) * block;\
      int position = (min(bktOutSize - 2, (atomicAdd(indexes + bucket, 2))));\
      int idx = (shift+((bucket%(NX2 / NA)) * (bktOutSize) + position)) / 2;\
      buffer[idx] = make_uint4(ret, ret >> 32, edge1.x, edge1.y);\
      break;\
    }\
    ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, edge64);\
  }\
}

template<int tpb, int bktOutSize>
__global__  void FluffySeed4K(uint4 * __restrict__ buffer, u32 * __restrict__ indexes, const u32 offset)
{
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;

  assert(blockDim.x == tpb);
  assert(NEDGES2 % (NA * gridDim.x * tpb * EDGE_BLOCK_SIZE) == 0);
  const int nloops = NEDGES2 / NA / gridDim.x / tpb / EDGE_BLOCK_SIZE;
  ulonglong4 sipblockL[EDGE_BLOCK_SIZE/4 - 1];
  __shared__ unsigned long long magazine[NX2];

  uint64_t v0, v1, v2, v3;

  for (short i = 0; i < NX2/tpb; i++)
    magazine[lid + tpb * i] = 0;

  __syncthreads();

  for (short i = 0; i < nloops; i++) {
    u64 blockNonce = offset + (gid * nloops * EDGE_BLOCK_SIZE + i * EDGE_BLOCK_SIZE);

    v0 = dipkeys.k0;
    v1 = dipkeys.k1;
    v2 = dipkeys.k2;
    v3 = dipkeys.k3;

    // do one block of 64 edges
    for (short b = 0; b < EDGE_BLOCK_SIZE-4; b += 4) {
      SIPBLOCK(blockNonce + b);
      u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 1);
      u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 2);
      u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 3);
      u64 e4 = (v0 ^ v1) ^ (v2  ^ v3);
      sipblockL[b / 4] = make_ulonglong4(e1, e2, e3, e4);
    }

    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-4);
    u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-3);
    u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-2);
    u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-1);
    u64 last = (v0 ^ v1) ^ (v2  ^ v3);

    DUMP(last,      1);
    DUMP(e1 ^ last, 0);
    DUMP(e2 ^ last, 1);
    DUMP(e3 ^ last, 0);

    for (short s = 14; s >= 0; s--) {
      ulonglong4 edges = sipblockL[s];
      DUMP(edges.x ^ last, 0);
      DUMP(edges.y ^ last, 1);
      DUMP(edges.z ^ last, 0);
      DUMP(edges.w ^ last, 1);
    }
  }

  __syncthreads();

  for (int i = 0; i < NX2/tpb; i++) {
    int bucket = lid + (tpb * i);
    u64 edge = magazine[bucket];
    if (edge != 0) {
      int block = bucket / (NX2 / NA);
      int shift = (bktOutSize * NX2) * block;
      int position = (min(bktOutSize - 2, (atomicAdd(indexes + bucket, 2))));
      int idx = (shift + ((bucket % (NX2 / NA)) * bktOutSize + position)) / 2;
      buffer[idx] = make_uint4(edge, edge >> 32, 0, 0);
    }
  }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A1(uint2 * source, uint4 * destination, const u32 * sourceIndexes, u32 * destinationIndexes, const int offset)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/32];
  __shared__ unsigned long long magazine[NX2];
  int edgesInBucket[NA];

  assert(blockDim.x == tpb);
  assert(gridDim.x == NX2/NA);

  for (int i = 0; i < NZ/32/tpb; i++)
    ecounters[lid + i * tpb] = 0;

  for (int i = 0; i < NX2/tpb; i++)
    magazine[lid + tpb * i] = 0;

  for (int a = 0; a < NA; a++)
    edgesInBucket[a] = min(sourceIndexes[a * NX2 + offset + group], bktInSize);

  const int rowOffset = offset * NA;
  source += bktInSize * (rowOffset + group);

  __syncthreads();

  for (int a = 0; a < NA; a++) {
    const int delta = a * (NX2/NA) * bktInSize;
    int i;
    for (i = 0; i < edgesInBucket[a] / tpb; i++) {
      uint2 edge = source[delta + (i * tpb) + lid];
      if (edge.x == 0 && edge.y == 0) continue;
      setbit(ecounters, (edge.x & ZMASK));
    }
    int lindex = (i * tpb) + lid;
    if (lindex < edgesInBucket[a]) {
      uint2 edge = source[delta + lindex];
      if (edge.x == 0 && edge.y == 0) continue;
      setbit(ecounters, (edge.x & ZMASK));
    }
  }

  __syncthreads();

  for (int a = 0; a < NA; a++) {
    const int delta = a * (NX2/NA) * bktInSize;
    for (int i = 0; i < (edgesInBucket[a] + tpb-1) / tpb; i++) {
      const int lindex = (i * tpb) + lid;
      if (lindex < edgesInBucket[a]) {
        uint2 edge = source[delta + lindex];
        if (edge.x == 0 && edge.y == 0) continue;
        if (testbit(ecounters, (edge.x & ZMASK) ^ 1)) {
          int bucket = (edge.y >> ZBITS) & BKTMASK4K;
          u64 edge64 = (((u64)edge.y) << 32) | edge.x;
          for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {
            u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);
            if (ret2 == ret) {
              int bktIdx = min(atomicAdd(destinationIndexes + bucket + rowOffset, 2), bktOutSize - 2);
              destination[ ((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(ret >> 32, ret, edge.y, edge.x);
              break;
            }
            ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, edge64);
          }
        }
      }
    }
  }

  __syncthreads();

  for (int i = 0; i < NX2/tpb; i++) {
    int bucket = lid + (tpb * i);
    u64 edge = magazine[bucket];
    if (edge != 0) {
      int bktIdx = min(atomicAdd(destinationIndexes + bucket + rowOffset, 2), bktOutSize - 2);
      destination[((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(edge >> 32, edge, 0, 0);
    }
  }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A2(const uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes, const int round, int * aux)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/32];

  const int edgesInBucket = min(sourceIndexes[group], bktInSize);
  const int loops = (edgesInBucket + tpb-1) / tpb;

  if (!loops)
    return;

  const int offset = bktInSize * group;

  assert(blockDim.x == tpb);
  assert(gridDim.x == NX2);

  for (int i = 0; i < NZ/32/tpb; i++)
    ecounters[lid + (tpb * i)] = 0;

  __syncthreads();

  for (int i = 0; i < loops - 1; i++) {
    const int lindex = (i * tpb) + lid;
    uint2 edge = source[offset + lindex];
    setbit(ecounters, (edge.x & ZMASK));
  }

  uint2 edge1;
  int lindex = ((loops - 1) * tpb) + lid;
  if (lindex < edgesInBucket) {
    edge1 = ld_cs_u32_v2(&source[offset + lindex]);
    setbit(ecounters, (edge1.x & ZMASK));
  }

  __syncthreads();

  if (lindex < edgesInBucket) {
    if (testbit(ecounters, (edge1.x & ZMASK) ^ 1)) {
      const int bucket = (edge1.y >> ZBITS) & BKTMASK4K;
      const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
      st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge1.y, edge1.x));
    }
  }

  for (int i = loops - 2; i >= 0; i--) {
    const int lindex = (i * tpb) + lid;
    uint2 edge = source[offset + lindex];
    if (testbit(ecounters, (edge.x & ZMASK) ^ 1)) {
      const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
      const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
      st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
    }
  }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A3(uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/32];
  int edgesInBucket[NA];

  assert(blockDim.x == tpb);
  assert(gridDim.x == NX2);

  for (int i = 0; i < NZ/32/tpb; i++)
    ecounters[lid + (i * tpb)] = 0;

  for (int a = 0; a < NA; a++) {
    edgesInBucket[a] = min(sourceIndexes[group + a*NX2], bktInSize);
  }

  source += bktInSize * group;

  __syncthreads();

  for (int a = 0; a < NA; a++) {
    const int delta = a * bktInSize * NX2;
    for (int i = 0; i < (edgesInBucket[a] + tpb-1) / tpb; i++) {
      int lindex = (i * tpb) + lid;
      if (lindex < edgesInBucket[a]) {
        uint2 edge = ld_cs_u32_v2(&source[delta + lindex]);
        if (edge.x == 0 && edge.y == 0) continue;
        setbit(ecounters, (edge.x & ZMASK));
      }
    }
  }

  __syncthreads();

  for (int a = 0; a < NA; a++) {
    const int delta = a * bktInSize * NX2;
    for (int i = 0; i < (edgesInBucket[a] + tpb-1) / tpb; i++) {
      int lindex = (i * tpb) + lid;
      if (lindex < edgesInBucket[a]) {
        uint2 edge = ld_cs_u32_v2(&source[delta + lindex]);
        if (edge.x == 0 && edge.y == 0) continue;
        if (testbit(ecounters, (edge.x & ZMASK) ^ 1)) {
          const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
          const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
          st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
        }
      }
    }
  }
}

template<int bktInSize>
__global__  void FluffyTail(const uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  __shared__ int destIdx;

  assert(gridDim.x == NX2);

  const int nEdges = sourceIndexes[group];

  assert(blockDim.x >= nEdges);

  if (lid == 0)
    destIdx = atomicAdd(destinationIndexes, nEdges);

  __syncthreads();

  if (lid < nEdges)
    destination[destIdx + lid] = source[group * bktInSize + lid];
}

__global__  void FluffyRecovery(u32 * indexes)
{
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;

  __shared__ u32 nonces[PROOFSIZE];
  u64 sipblock[EDGE_BLOCK_SIZE];

  uint64_t v0;
  uint64_t v1;
  uint64_t v2;
  uint64_t v3;

  const int nloops = NEDGES2 / (gridDim.x * blockDim.x * EDGE_BLOCK_SIZE);
  if (lid < PROOFSIZE) nonces[lid] = 0;

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    u64 blockNonce = gid * nloops * EDGE_BLOCK_SIZE + i * EDGE_BLOCK_SIZE;

    v0 = dipkeys.k0;
    v1 = dipkeys.k1;
    v2 = dipkeys.k2;
    v3 = dipkeys.k3;

    for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++) {
      v3 ^= blockNonce + b;
      for (short r = 0; r < 2; r++)
        SIPROUND;
      v0 ^= blockNonce + b;
      v2 ^= 0xff;
      for (short r = 0; r < 4; r++)
        SIPROUND;

      sipblock[b] = (v0 ^ v1) ^ (v2  ^ v3);
    }

    const u64 last = sipblock[EDGE_BLOCK_SIZE - 1];
    sipblock[EDGE_BLOCK_SIZE - 1] = 0;

    for (short s = EDGE_BLOCK_SIZE; --s >= 0; ) {
      u32 dir = s & 1;
      u64 lookup = sipblock[s] ^ last;

      u64 u = (lookup & NODE1MASK) << 1 | dir;
      u64 v = (lookup >> 31) & (NODE1MASK << 1) | dir;

      u64 a = u | (v << 32);
      u64 b = v | (u << 32);

      for (int i = 0; i < PROOFSIZE; i++) {
        if (recovery[i] == a || recovery[i] == b)
          nonces[i] = blockNonce + s;
      }
    }
  }

  __syncthreads();

  if (lid < PROOFSIZE)
  {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}
