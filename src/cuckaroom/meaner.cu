// Cuckaroom Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2020 Wilke Trei, Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

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
#include <string.h>
#include <vector>
#include <assert.h>
#include "cuckaroom.hpp"
#include "graph.hpp"
#include "../crypto/blake2.h"

// number of invocations of seeding kernel
// values larger than 1 reduce pressure on TLB
// which appears to be an issue on GTX cards
#ifndef NTLB
#define NTLB 2
#endif
// number of slices in edge splitting kernel
// larger values reduce memory use
// which is proportional to 1+1/NMEM
#ifndef NMEM
#define NMEM 4
#endif
// after seeding, main memory buffer is a NMEM x NTLB matrix
// of blocks of buckets
#define NMEMTLB (NMEM * NTLB)

#define NODE1MASK NODEMASK
#include "../crypto/siphash.cuh"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

#ifndef BUCKBITS
// assumes at least 2^18 bits of shared mem (32 KB) on thread block
// #define BUCKBITS (EDGEBITS-18)
#define BUCKBITS 12
#endif

const u32 NB = 1 << BUCKBITS;
const u32 NB_NTLB  = NB / NTLB;
const u32 NB_NMEM  = NB / NMEM;
const u32 ZBITS = EDGEBITS - BUCKBITS;
const u32 ZBITS1 = ZBITS + 1;
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

__device__ __forceinline__ void setbit(u32 *bitmap, const int index)
{
  const u32 word = index / 32;
  const u32  bit = index % 32;
  const u32 mask = 1 << bit;
  atomicOr(&bitmap[word], mask);
}

__device__ __forceinline__ bool testbit(u32 *bitmap, const int index)
{
  const u32 word = index / 32;
  const u32  bit = index % 32;
  return (bitmap[word] >> bit) & 1;
}

__device__ __forceinline__ void resetbit(u64 *edgemap, const int slot)
{
  const u32 word = slot / 32;
  const u32  bit = slot % 32;
  const u32 mask = 1 << bit;
  atomicAnd((u32 *)edgemap + word, ~mask);
}

__device__ __forceinline__ bool testbit(const u64 edgemap, const int slot)
{
  return (edgemap >> slot) & 1;
}

__constant__ siphash_keys dipkeys;
__constant__ u64 recovery[42];

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND {\
  v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
  v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
  v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
  v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
  v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
}
#define SIPBLOCK(b) {\
  v3 ^= (b);\
  for (int r = 0; r < 2; r++)\
  SIPROUND;\
  v0 ^= (b);\
  v2 ^= 0xff;\
  for (int r = 0; r < 4; r++)\
  SIPROUND;\
}

#ifndef SEED_TPB
#define SEED_TPB 256
#endif
#ifndef SPLIT_TPB
#define SPLIT_TPB 1024
#endif
#ifndef TRIM_TPB
#define TRIM_TPB 1024
#endif
#ifndef UNSPLIT_TPB
#define UNSPLIT_TPB 256
#endif
#ifndef RELAY_TPB
#define RELAY_TPB 512
#endif
#ifndef TAIL_TPB
#define TAIL_TPB RELAY_TPB
#endif

template<int maxOut>
__global__ void Seed(uint4 * __restrict__ buffer, u32 * __restrict__ indexes, const u32 part)
{
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = gridDim.x * SEED_TPB;

  const int nloops = (NEDGES / NTLB / EDGE_BLOCK_SIZE - gid + nthreads-1) / nthreads;
  ulonglong4 sipblockL[EDGE_BLOCK_SIZE/4];
  uint64_t v0, v1, v2, v3;

  __shared__ unsigned long long magazine[NB];

  for (int i = 0; i < NB/SEED_TPB; i++)
    magazine[lid + SEED_TPB * i] = 0;

  __syncthreads();

  u64 offset = part * (NEDGES / NTLB);
  buffer  += part * (maxOut * NB_NMEM / 2); 
  indexes += part * NB;

  for (int i = 0; i < nloops; i++) {
    u64 blockNonce = offset + ((gid * nloops + i) * EDGE_BLOCK_SIZE);

    v0 = dipkeys.k0;
    v1 = dipkeys.k1;
    v2 = dipkeys.k2;
    v3 = dipkeys.k3;

    // do one block of 64 edges
    for (int b = 0; b < EDGE_BLOCK_SIZE; b += 4) {
      SIPBLOCK(blockNonce + b);
      u64 e0 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 1);
      u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 2);
      u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 3);
      u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
      sipblockL[b / 4] = make_ulonglong4(e0, e1, e2, e3);
    }

    u64 last = 0;

#define DUMP(E) {\
  u64 lookup = (E);\
  const uint2 edge1 = make_uint2(lookup & NODEMASK, (lookup >> 32) & NODEMASK);\
  int bucket = edge1.x >> ZBITS;\
  u64 edge64 = (((u64)edge1.y) << 32) | edge1.x;\
  for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {\
    u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);\
    if (ret2 == ret) {\
      const u32 position = atomicAdd(indexes + bucket, 2);\
      if (position >= maxOut) {\
        printf("Seed dropping edges %llx %llx\n", ret, edge64);\
        break;\
      }\
      int idx = (bucket + (bucket / NB_NMEM) * (NTLB-1) * NB_NMEM) * maxOut + position;\
      buffer[idx/2] = make_uint4(ret, ret >> 32, edge1.x, edge1.y);\
      break;\
    }\
    ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, edge64);\
  }\
}

    for (int s = EDGE_BLOCK_SIZE/4; s--; ) {
      ulonglong4 edges = sipblockL[s];
      DUMP(last ^= edges.w);
      DUMP(last ^= edges.z);
      DUMP(last ^= edges.y);
      DUMP(last ^= edges.x);
    }
  }

  __syncthreads();

  for (int i = 0; i < NB/SEED_TPB; i++) {
    int bucket = lid + (SEED_TPB * i);
    u64 edge = magazine[bucket];
    if (edge != 0) {
      const u32 position = atomicAdd(indexes + bucket, 2);
      if (position >= maxOut) {
        printf("Seed dropping edge %llx\n", edge);
        continue;
      }
      int idx = (bucket + (bucket / NB_NMEM) * (NTLB-1) * NB_NMEM) * maxOut + position;\
      buffer[idx/2] = make_uint4(edge, edge >> 32, 0, 0);
    }
  }
}

#ifndef EXTSHARED
#define EXTSHARED 0xc000
#endif

#ifndef CYCLE_V0
#define CYCLE_V0 0x59f7ada
#define CYCLE_V1 0x106a0753
#endif
#define BUCK0 (CYCLE_V0 >> ZBITS)
#define REST0 (CYCLE_V0 & ZMASK)
#define BUCK1 (CYCLE_V1 >> ZBITS)
#define REST1 (CYCLE_V1 & ZMASK)

template<int maxIn, int maxOut>
__global__ void EdgeSplit(const uint2 *src, uint2 *dst, const u32 *srcIdx, u32 *dstIdx, u64 *edgemap, const int part)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  extern __shared__ unsigned long long magazine[];
  u32 *emap = (u32 *)(magazine + NB);
  int nloops[NTLB];

  for (int i = 0; i < NB/SPLIT_TPB; i++) {
    magazine[lid + SPLIT_TPB * i] = 0;
    emap[lid + SPLIT_TPB * i] = 0;
  }

  const int offset = part * NB_NMEM;
  const int Group = offset + group;
  edgemap += NB * Group;

#if EXTSHARED >= 0xc004
  u32 *ourIdx = emap + NB;
#else
  u32 *ourIdx = (u32 *)edgemap;
#endif

  if (!lid)
    *ourIdx = 0;

  for (int a = 0; a < NTLB; a++)
    nloops[a] = (min(srcIdx[a * NB + Group], maxIn) - lid + SPLIT_TPB-1) / SPLIT_TPB;

  src += (offset * NTLB + group) * maxIn + lid;
  dst += part * NB * maxOut;
  uint2 *dstX = dst + (NB_NMEM + group) * NTLB * maxIn / 2;

  __syncthreads();

  const int rowOffset = offset * NMEM;
  for (int a = 0; a < NTLB; a++) {
    const u32 delta = a * NB_NMEM * maxIn;
    for (int i = 0; i < nloops[a]; i++) {
      uint2 edge = src[delta + i * SPLIT_TPB];
      if (edge.x == 0 && edge.y == 0) continue;
      int bucket = edge.y >> ZBITS;
      u64 edge64 = (((u64)edge.y) << 32) | edge.x;
      for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {
        u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);
        if (ret2 == ret) {
          const u32 slot = atomicAdd(emap + bucket, 2);
          if (slot >= 64) {
#ifdef VERBOSE
            printf("dropped edges %llx %llx\n", ret, edge64);
#endif
            break;
          }
          const u32 slots = (slot+1) << 6 | slot; // slot for edge, slot+1 for ret
          int bktIdx = atomicAdd(ourIdx, 1);
          dstX[bktIdx] = make_uint2(bucket << ZBITS | edge.x & ZMASK, slots << ZBITS | ret & ZMASK);
          bktIdx = atomicAdd(dstIdx + rowOffset + bucket, 1);
          if (bktIdx >= maxOut/2) {
#ifdef VERBOSE
            printf("dropped halfedge %llx\n", edge);
#endif
            break;
          }
          dst[bucket * maxOut/2 + bktIdx] = make_uint2(Group << ZBITS | edge.y & ZMASK, slots << ZBITS | (ret>>32) & ZMASK);
          break;
        }
        ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, edge64);
      }
    }
  }

  __syncthreads();

  for (int i = 0; i < NB/SPLIT_TPB; i++) {
    int bucket = lid + SPLIT_TPB * i;
    u64 edge = magazine[bucket];
    if (edge != 0) {
      const u32 slot = atomicAdd(emap + bucket, 1);
      if (slot >= 64) {
#ifdef VERBOSE
        printf("dropped edge %llx\n", edge);
#endif
        continue;
      }
      const u32 slots = slot << 6 | slot; // duplicate slot indicates singleton pair
      int bktIdx = atomicAdd(ourIdx, 1);
      dstX[bktIdx] = make_uint2(bucket << ZBITS | edge & ZMASK, slots << ZBITS | edge & ZMASK);
      bktIdx = atomicAdd(dstIdx + rowOffset + bucket, 1);
      if (bktIdx >= maxOut/2) {
#ifdef VERBOSE
        printf("dropped halfedge %llx\n", edge);
#endif
        continue;
      }
      dst[bucket * maxOut/2 + bktIdx] = make_uint2(Group << ZBITS | (edge>>32) & ZMASK, slots << ZBITS | (edge>>32) & ZMASK);
    }
  }

  __syncthreads();

  if (!lid) {
    u32 *dstXIdx = dstIdx + NMEM * NB;
    dstXIdx[Group] = *ourIdx;
  }

  for (int i = 0; i < NB/SPLIT_TPB; i++) {
    int idx = lid + SPLIT_TPB * i;
    edgemap[idx] = emap[idx] < 64 ? (1ULL << emap[idx]) - 1 : ~0ULL;
  }
}

__device__ __forceinline__ void addhalfedge(u32 *magazine, const u32 bucket, const u32 rest, const u32 slot, u32 *indices, uint2* dst) {
  u32 halfedge = slot << ZBITS | rest;
  for (u32 ret = atomicCAS(&magazine[bucket], 0, halfedge); ret; ) {
    u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);
    if (ret2 == ret) {
      int idx = atomicAdd(indices, 1);
      u32 slots = 64 * slot + (ret >> ZBITS);
      dst[idx] = make_uint2(bucket << ZBITS | ret & ZMASK, slots << ZBITS | rest);
      break;
    }
    ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, halfedge);
  }
}

__device__ __forceinline__ void flushhalfedges(u32 *magazine, const int lid, u32 *indices, uint2* dst) {
  for (int i = 0; i < NB/TRIM_TPB; i++) {
    int bucket = lid + TRIM_TPB * i;
    u32 halfedge = magazine[bucket];
    if (halfedge != 0) {
      int idx = atomicAdd(indices, 1);
      u32 slot = halfedge >> ZBITS;
      dst[idx] = make_uint2(bucket << ZBITS | halfedge & ZMASK, slot * 64 << ZBITS | halfedge);
    }
    magazine[bucket] = 0;
  }
}

template<int maxIn> // maxIn is size of small buckets in half-edges
__global__ void Round(uint2 *source, u32 *sourceIndexes, u64 *edgemap)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  int bktsize[NMEM];
  int npairs[NMEM];

  __shared__ u32 htnode[2 * NZ/32]; // head & tail node bits
  __shared__ u32 magazine[NB];

  for (int i = 0; i < 2 * NZ/32/TRIM_TPB; i++)
    htnode[lid + i * TRIM_TPB] = 0;

  for (int i = 0; i < NB/TRIM_TPB; i++)
    magazine[lid + i * TRIM_TPB] = 0;

  for (int a = 0; a < NMEM; a++) {
    bktsize[a] = sourceIndexes[a * NB + group];
    npairs[a] = (bktsize[a] + TRIM_TPB-1) / TRIM_TPB;
  }
  const u32 bsize = sourceIndexes[NMEM * NB + group];
  const u32 np = (bsize + TRIM_TPB-1) / TRIM_TPB;

  __syncthreads();

  if (lid <= NMEM)
    sourceIndexes[lid * NB + group] = 0;

  const u32 delta = (group / NB_NMEM) * NB * maxIn + (NB + (group % NB_NMEM) * NMEM) * maxIn/2;
  for (int i = 0; i < np; i++) {
    __syncthreads();
    u32 idx = lid + i * TRIM_TPB;
    if (idx >= bsize) continue;
    uint2 pair = source[delta + idx];
    u32 bucket = pair.x >> ZBITS;
    u32 slots = pair.y >> ZBITS;
    u32 slot0 = slots % 64, slot1 = slots / 64;
    u64 es = edgemap[group * NB + bucket];
    if (testbit(es, slot0))
      setbit(htnode, 2 * (pair.x & ZMASK));
    if (slot1 != slot0 && testbit(es, slot1))
      setbit(htnode, 2 * (pair.y & ZMASK));
  }

  for (int a = 0; a < NMEM; a++) {
    const u32 delta = a * NB * maxIn + group * maxIn/2; // / 2 for indexing pairs, * 2 for alternate buckets
    for (int i = 0; i < npairs[a]; i++) {
      __syncthreads();
      u32 idx = lid + i * TRIM_TPB;
      if (idx >= bktsize[a]) continue;
      uint2 pair = source[delta + idx];
      u32 bucket = pair.x >> ZBITS;
      u32 slots = pair.y >> ZBITS;
      u32 slot0 = slots % 64, slot1 = slots / 64;
      u64 es = edgemap[bucket * NB + group];
      if (testbit(es, slot0)) {
	if (testbit(htnode, 2 * (pair.x & ZMASK))) {
          addhalfedge(magazine, bucket, pair.x&ZMASK, slot0, sourceIndexes+a*NB+group, source+delta);
          setbit(htnode, 2 * (pair.x & ZMASK) + 1);
	} else resetbit(edgemap + bucket * NB + group, slot0);
      }
      if (slot1 != slot0 && testbit(es, slot1)) {
        if (testbit(htnode, 2 * (pair.y & ZMASK))) {
          addhalfedge(magazine, bucket, pair.y&ZMASK, slot1, sourceIndexes+a*NB+group, source+delta);
          setbit(htnode, 2 * (pair.y & ZMASK) + 1);
	} else resetbit(edgemap + bucket * NB + group, slot1);
      }
    }
    __syncthreads();

    flushhalfedges(magazine, lid, sourceIndexes+a*NB+group, source+delta);
  }

  for (int i = 0; i < np; i++) {
    __syncthreads();
    u32 idx = lid + i * TRIM_TPB;
    if (idx >= bsize) continue;
    uint2 pair = source[delta + idx];
    u32 bucket = pair.x >> ZBITS;
    u32 slots = pair.y >> ZBITS;
    u32 slot0 = slots % 64, slot1 = slots / 64;
    u64 es = edgemap[group * NB + bucket];
    if (testbit(es, slot0)) {
      if (testbit(htnode, 2 * (pair.x & ZMASK) + 1)) {
        addhalfedge(magazine, bucket, pair.x&ZMASK, slot0, sourceIndexes+NMEM*NB+group, source+delta);
      } else resetbit(edgemap + group * NB + bucket, slot0);
    }
    if (slot1 != slot0 && testbit(es, slot1)) {
      if (testbit(htnode, 2 * (pair.y & ZMASK) + 1)) {
        addhalfedge(magazine, bucket, pair.y&ZMASK, slot1, sourceIndexes+NMEM*NB+group, source+delta);
      } else resetbit(edgemap + group * NB + bucket, slot1);
    }
  }
  __syncthreads();

  flushhalfedges(magazine, lid, sourceIndexes+NMEM*NB+group, source+delta);
}

template<int maxIn> // maxIn is size of small buckets in half-edges
__global__ void EdgesMeet(uint2 *source, u32 *sourceIndexes, u32 *destIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  int bktsize[NMEM];
  int npairs[NMEM];

  for (int a = 0; a < NMEM; a++) {
    bktsize[a] = sourceIndexes[a * NB + group];
    npairs[a] = (bktsize[a] + TRIM_TPB-1) / TRIM_TPB;
  }

  __syncthreads();

  for (int a = 0; a < NMEM; a++) {
    const u32 delta = a * NB * maxIn + group * maxIn/2; // / 2 for indexing pairs, * 2 for alternate buckets
    const u32 delta2 = a * NB * maxIn + NB * maxIn/2;
    for (int i = 0; i < npairs[a]; i++) {
      __syncthreads();
      u32 idx = lid + i * TRIM_TPB;
      if (idx >= bktsize[a]) continue;
      uint2 pair = source[delta + idx];
      u32 bucket = pair.x >> ZBITS;
      pair.x = (group << ZBITS) | (pair.x & ZMASK);
      int i = atomicAdd(destIndexes + bucket, 1);
      const u32 dlt = (bucket % NB_NMEM) * NMEM * maxIn/2;
      source[delta2 + dlt + i] = pair;
    }
  }
}

template<int maxOut>
__device__ __forceinline__ void addedge(const u32 buckx, const u32 restx, const u32 bucky, const u32 resty, u32 *indices, uint2* dest, u32* dstidx) {
  int idx = atomicAdd(indices + bucky, 1);
  dest[    bucky *maxOut + idx] = make_uint2(restx << ZBITS1 | resty, buckx << ZBITS | restx); // extra +1 shift makes a 0 copybit
  int idx2 = atomicAdd(dstidx, 1);
  dest[(NB+buckx)*maxOut + idx2] = make_uint2(resty << ZBITS1 | restx, bucky << ZBITS | resty);
}

template<int maxIn, int maxOut> // maxIn is size of small buckets in half-edges
__global__ void UnsplitEdges(uint2 *source, u32 *srcIndexes, u32 *srcIndexes2, uint2 *dest, u32 *dstIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 lists[NB];
  __shared__ u32 nexts[NB];
  __shared__ u32 dstidx;

  for (int i = 0; i < NB/UNSPLIT_TPB; i++)
    lists[i * UNSPLIT_TPB + lid] = ~0;

  if (!lid)
    dstidx = 0;

  const u32 size1 = srcIndexes[group];
  const u32 np1 = (size1 + UNSPLIT_TPB-1) / UNSPLIT_TPB;
  const u32 size2 = srcIndexes2[group];
  const u32 np2 = (size2-size1 + UNSPLIT_TPB-1) / UNSPLIT_TPB;

  __syncthreads();

  const u32 delta = (group / NB_NMEM) * NB * maxIn + (NB + (group % NB_NMEM) * NMEM) * maxIn/2;
  source += delta;
  for (int i = 0; i < np1; i++) {
    const u32 index = lid + i * UNSPLIT_TPB;
    if (index >= size1) continue;
    const uint2 pair = source[index];
    const u32 bucket = pair.x >> ZBITS;
    nexts[index] = atomicExch(&lists[bucket], index);
  }

  __syncthreads();

  for (int i = 0; i < np2; i++) {
    const u32 index = size1 + lid + i * UNSPLIT_TPB;
    if (index >= size2) continue;
    const uint2 pair = source[index];
    const u32 bucket = pair.x >> ZBITS;
    const u32 slots = pair.y >> ZBITS;
    const u32 slot0 = slots % 64, slot1 = slots / 64;
    for (u32 idx = lists[bucket]; idx != ~0; idx = nexts[idx]) {
      const uint2 pair2 = source[idx];
      const u32 slots2 = pair2.y >> ZBITS;
      const u32 slot20 = slots2 % 64, slot21 = slots2 / 64;
      if (slot20 == slot0)
        addedge<maxOut>(group, pair2.x & ZMASK, bucket, pair.x & ZMASK, dstIndexes, dest, &dstidx);
      else if (slot20 == slot1)
        addedge<maxOut>(group, pair2.x & ZMASK, bucket, pair.y & ZMASK, dstIndexes, dest, &dstidx);
      if (slot21 != slot20) {
        if (slot21 == slot0)
          addedge<maxOut>(group, pair2.y & ZMASK, bucket, pair.x & ZMASK, dstIndexes, dest, &dstidx);
        else if (slot21 == slot1)
          addedge<maxOut>(group, pair2.y & ZMASK, bucket, pair.y & ZMASK, dstIndexes, dest, &dstidx);
      }
      // if (slots != slots2) printf("pair slots %d %d pair2 slots %d %d\n", slot0, slot1, slot20, slot21);
    }
  }

  __syncthreads();

  if (!lid)
    dstIndexes[NB + group] = dstidx;
}

#ifndef LISTBITS
#define LISTBITS 11
#endif

const u32 NLISTS  = 1 << LISTBITS;

#ifndef NNEXTS
#define NNEXTS NLISTS
#endif

template<int maxIn, int maxOut>
__global__ void Tag_Relay(const uint2 *source, uint2 *destination, const u32 *sourceIndexes, u32 *destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const u32 LISTMASK = NLISTS - 1;
  const u32 TAGMASK = (~1U) << ZBITS;
  const u32 COPYFLAG = NZ;

  __shared__ u32 lists0[NLISTS];
  __shared__ u32 nexts0[NNEXTS];
  __shared__ u32 lists1[NLISTS];
  __shared__ u32 nexts1[NNEXTS];

  const int nloops0 = (min(sourceIndexes[   group], NNEXTS) - lid + RELAY_TPB-1) / RELAY_TPB;
  const int nloops1 = (min(sourceIndexes[NB+group], NNEXTS) - lid + RELAY_TPB-1) / RELAY_TPB;

  source += group * maxIn;

  for (int i = 0; i < NLISTS/RELAY_TPB; i++)
    lists0[i * RELAY_TPB + lid] = lists1[i * RELAY_TPB + lid] = ~0;

  __syncthreads();

  for (int i = 0; i < nloops0; i++) {
    const u32 index = i * RELAY_TPB + lid;
    const uint2 edge = source[index];
    const u32 list = edge.x & LISTMASK;
    nexts0[index] = atomicExch(&lists0[list], index);
  }

  __syncthreads();

  for (int i = 0; i < nloops1; i++) {
    const u32 index = i * RELAY_TPB + lid;
    const uint2 edge = source[NB * maxIn + index];
    const u32 list = edge.x & LISTMASK;
    nexts1[index] = atomicExch(&lists1[list], index);
    if (edge.x & COPYFLAG) continue; // copies don't relay
    u32 bucket = edge.y >> ZBITS;
    u32 copybit = 0;
    for (u32 idx = lists0[list]; idx != ~0; idx = nexts0[idx]) {
      uint2 tagged = source[idx];
      // printf("compare with tagged %08x %08x xor %x\n", tagged.x, tagged.y, (tagged.x ^ edge.x) & ZMASK);
      if ((tagged.x ^ edge.x) & ZMASK) continue;
      u32 bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), maxOut - 1);
      // printf("relaying from tagged %08x %08x bktIdx %d\n", tagged.x, tagged.y, bktIdx);
      u32 tag = (tagged.x & TAGMASK) | copybit;
      destination[bucket * maxOut + bktIdx] = make_uint2(tag | (edge.y & ZMASK), (group << ZBITS) | (edge.x & ZMASK));
      copybit = COPYFLAG;
    }
  }

  __syncthreads();

  for (int i = 0; i < nloops0; i++) {
    const u32 index = i * RELAY_TPB + lid;
    const uint2 edge = source[index];
    const u32 list = edge.x & LISTMASK;
    if (edge.x & COPYFLAG) continue; // copies don't relay
    u32 bucket = edge.y >> ZBITS;
    u32 copybit = 0;
    for (u32 idx = lists1[list]; idx != ~0; idx = nexts1[idx]) {
      uint2 tagged = source[NB * maxIn + idx];
      if ((tagged.x ^ edge.x) & ZMASK) continue;
      u32 bktIdx = min(atomicAdd(destinationIndexes + NB + bucket, 1), maxOut - 1);
      // printf("relaying from tagged %08x %08x bktIdx %d\n", tagged.x, tagged.y, bktIdx);
      u32 tag = (tagged.x & TAGMASK) | copybit;
      destination[(NB + bucket) * maxOut + bktIdx] = make_uint2(tag | (edge.y & ZMASK), (group << ZBITS) | (edge.x & ZMASK));
      copybit = COPYFLAG;
    }
  }
}

template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const u32 *sourceIndexes, u32 *destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const u32 LISTMASK = NLISTS - 1;
  const u32 COPYFLAG = NZ;

  __shared__ u32 lists0[NLISTS];
  __shared__ u32 nexts0[NNEXTS];

  const int nloops0 = (min(sourceIndexes[   group], NNEXTS) - lid + TAIL_TPB-1) / TAIL_TPB;
  const int nloops1 = (min(sourceIndexes[NB+group], NNEXTS) - lid + TAIL_TPB-1) / TAIL_TPB;

  source += group * maxIn;

  for (int i = 0; i < NLISTS/TAIL_TPB; i++)
    lists0[i * TAIL_TPB + lid] = ~0;

  __syncthreads();

  for (int i = 0; i < nloops0; i++) {
    const u32 index = i * TAIL_TPB + lid;
    const uint2 edge = source[index];
    const u32 list = edge.x & LISTMASK;
    nexts0[index] = atomicExch(&lists0[list], index);
  }

  __syncthreads();

  for (int i = 0; i < nloops1; i++) {
    const u32 index = i * TAIL_TPB + lid;
    const uint2 edge = source[NB * maxIn + index];
    const u32 list = edge.x & LISTMASK;
    for (u32 idx = lists0[list]; idx != ~0; idx = nexts0[idx]) {
      uint2 tagged = source[idx];
      if ((tagged.x ^ edge.x) & ~COPYFLAG) continue;
      u32 bktIdx = atomicAdd(destinationIndexes, 1);
      destination[bktIdx] = make_uint2((group << ZBITS) | (edge.x & ZMASK), edge.y);
    }
  }
}

__global__ void Recovery(u32 * indexes)
{
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;

  __shared__ u32 nonces[PROOFSIZE];
  u64 sipblock[EDGE_BLOCK_SIZE];

  uint64_t v0;
  uint64_t v1;
  uint64_t v2;
  uint64_t v3;

  const int nloops = (NEDGES / EDGE_BLOCK_SIZE - gid + nthreads-1) / nthreads;
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
      for (int r = 0; r < 2; r++)
        SIPROUND;
      v0 ^= blockNonce + b;
      v2 ^= 0xff;
      for (int r = 0; r < 4; r++)
        SIPROUND;

      sipblock[b] = (v0 ^ v1) ^ (v2  ^ v3);
    }

    u64 last = 0;
    const u64 NODEMASK2 = NODEMASK | ((u64)NODEMASK << 32);
    u64 lookup;
    for (int s = EDGE_BLOCK_SIZE; --s >= 0; last = lookup) {
      lookup = sipblock[s] ^ last;
      for (int i = 0; i < PROOFSIZE; i++) {
        if (recovery[i] == (lookup & NODEMASK2))
          nonces[i] = blockNonce + s;
      }
    }
  }

  __syncthreads();

  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}
typedef uint8_t u8;
typedef uint16_t u16;

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#define IDXSHIFT 12
#endif

const u32 MAXEDGES = NEDGES >> IDXSHIFT;

#ifndef NEPS_A
#define NEPS_A 135
#endif
#ifndef NEPS_B
#define NEPS_B 88
#endif
#ifndef NEPS_C
#define NEPS_C 55
#endif
#define NEPS 128

const u32 EDGES_A = NZ * NEPS_A / NEPS;
const u32 EDGES_B = NZ * NEPS_B / NEPS;

const u32 ALL_EDGES_A = EDGES_A * NB;
const u32 ALL_EDGES_B = EDGES_B * NB;

#define checkCudaErrors_V(ans) ({if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return;})
#define checkCudaErrors_N(ans) ({if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return NULL;})
#define checkCudaErrors(ans) ({int retval = gpuAssert((ans), __FILE__, __LINE__); if (retval != cudaSuccess) return retval;})

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  int device_id;
  cudaGetDevice(&device_id);
  if (code != cudaSuccess) {
    snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d", device_id, cudaGetErrorString(code), file, line);
    cudaDeviceReset();
    if (abort) return code;
  }
  return code;
}

struct blockstpb {
  u16 blocks;
  u16 tpb;
};

struct trimparams {
  u16 ntrims;
  blockstpb seed;
  blockstpb trim0;
  blockstpb trim1;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    ntrims         =        15;
    seed.blocks    =   NB_NTLB;
    seed.tpb       =  SEED_TPB;
    trim0.blocks   =   NB_NMEM;
    trim0.tpb      =  TRIM_TPB;
    trim1.blocks   =   NB_NMEM;
    trim1.tpb      =  TRIM_TPB;
    trim.blocks    =        NB;
    trim.tpb       =  TRIM_TPB;
    tail.blocks    =        NB;
    tail.tpb       =  TAIL_TPB;;
    recover.blocks =      2048;
    recover.tpb    =       256;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  const size_t sizeA = ALL_EDGES_A * sizeof(uint2);
  const size_t sizeB = ALL_EDGES_B * sizeof(uint2);
  const size_t bufferSize = sizeA / NMEM + sizeA;
  const size_t indexesSize = NB * sizeof(u32);
  const size_t indexesSizeNTLB = NTLB * indexesSize;
  const size_t indexesSizeNMEM = NMEM * indexesSize;
  const size_t edgemapSize = 2 * NEDGES / 8; // 8 bits per byte; twice that for bucket (2^24 in all) size variance
  u8 *bufferA;
  u8 *bufferB;
  u32 *indexesA;
  u32 *indexesB;
  u64 *edgemap;
  u32 nedges;
  siphash_keys sipkeys;
  bool abort;
  bool initsuccess = false;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors_V(cudaMalloc((void**)&indexesA, indexesSizeNTLB));
    checkCudaErrors_V(cudaMalloc((void**)&indexesB, indexesSizeNMEM + indexesSize));
    checkCudaErrors_V(cudaMalloc((void**)&edgemap, edgemapSize));
    checkCudaErrors_V(cudaMalloc((void**)&bufferB, bufferSize));
    bufferA = bufferB + sizeA / NMEM;
    // print_log("allocated %lld bytes bufferB %llx endBuffer %llx\n", bufferSize, bufferB, bufferB+bufferSize);
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    initsuccess = true;
    cudaFuncSetAttribute(EdgeSplit<EDGES_A/NTLB, EDGES_A/NMEM>, cudaFuncAttributeMaxDynamicSharedMemorySize, EXTSHARED);
  }
  u64 globalbytes() const {
    return bufferSize + indexesSizeNTLB + indexesSizeNMEM + indexesSize + edgemapSize + sizeof(siphash_keys) + sizeof(edgetrimmer);
  }
  ~edgetrimmer() {
    checkCudaErrors_V(cudaFree(bufferB));
    checkCudaErrors_V(cudaFree(indexesA));
    checkCudaErrors_V(cudaFree(indexesB));
    checkCudaErrors_V(cudaFree(edgemap));
    checkCudaErrors_V(cudaFree(dt));
    cudaDeviceReset();
  }
  void indexcount(u32 round, const u32 *indexes) {
#ifdef VERBOSE
    u32 nedges[NB];
    for (int i = 0; i < NB; i++)
      nedges[i] = 0;
    cudaMemcpy(nedges, indexes, NB * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    u32 sum, max;
    for (int i = sum = max = 0; i < NB; i++) {
      sum += nedges[i];
      if (nedges[i] > max)
        max = nedges[i];
    }
    print_log("round %d edges avg %d max %d sum %d\n", round, sum/NB, max, sum);
#endif
  }
  void edgemapcount(const int round, const u64* edgemap) {
#ifdef VERBOSE
    u64* emap = (u64 *)calloc(NB*NB, sizeof(u64));
    assert(emap);
    cudaMemcpy(emap, edgemap, NB*NB*sizeof(u64), cudaMemcpyDeviceToHost);
    u32 cnt;
    for (int i = cnt = 0; i < NB*NB; i++) {
      cnt += __builtin_popcountl(emap[i]);
    }
    if (NEDGES-cnt < 1024)
      print_log("round %d edges %d - %d\n", round, NEDGES, NEDGES-cnt);
    else print_log("round %d edges %d = %2.2lf%%\n", round, cnt, 100.0*cnt/NEDGES);
    free(emap);
#endif
  }
  u32 trim() {
    u32 nedges = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
    cudaMemcpyToSymbol(dipkeys, &sipkeys, sizeof(sipkeys));

    cudaDeviceSynchronize();
    float durationA, durationB, durationC;
    cudaEventRecord(start, NULL);
  
    cudaMemset(indexesA, 0, indexesSizeNTLB);
    for (u32 i=0; i < NTLB; i++) {
      Seed<EDGES_A/NTLB><<<tp.seed.blocks, SEED_TPB>>>((uint4*)bufferA, indexesA, i);
      if (abort) return false;
    }
  
#ifdef VERBOSE
    print_log("%d x Seed<<<%d,%d>>>\n", NTLB, tp.seed.blocks, tp.seed.tpb); // 1024x512
    for (u32 i=0; i < NTLB; i++)
      indexcount(0, indexesA+i*NB);
#endif

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    cudaEventRecord(start, NULL);
  
#ifdef VERBOSE
    print_log("Seeding completed in %.0f ms\n", durationA);
    print_log("EdgeSplit<<<%d,%d>>>\n", NB_NMEM, SPLIT_TPB); // 1024x1024
#endif

    cudaMemset(indexesB, 0, indexesSizeNMEM + indexesSize);
    for (u32 i=0; i < NMEM; i++) {
      EdgeSplit<EDGES_A/NTLB, EDGES_A/NMEM><<<NB_NMEM, SPLIT_TPB, EXTSHARED>>>((uint2*)bufferA, (uint2*)bufferB, indexesA, indexesB, edgemap, i);
      if (abort) return false;
    }

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    cudaEventRecord(start, NULL);
  
#ifdef VERBOSE
    for (u32 i=0; i < NMEM; i++)
      indexcount(1, indexesB+i*NB);
    indexcount(1, indexesB+NMEM*NB);
    edgemapcount(1, edgemap); // .400
    print_log("EdgeSplit completed in %.0f ms\n", durationB);
    print_log("Round<<<%d,%d>>>\n", NB, TRIM_TPB); // 4096x1024
#endif

    Round<EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, edgemap);

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationC, start, stop);
    checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(stop));
  
    print_log("Round completed in %.0f ms\n", durationC);
    for (u32 i=0; i < NMEM; i++)
      indexcount(2, indexesB+i*NB);
    indexcount(2, indexesB+NMEM*NB);
    edgemapcount(2, edgemap);
    if (abort) return false;

#ifdef VERBOSE
    print_log("Round<><<<%d,%d>>>\n", NB, TRIM_TPB);
#endif

    for (int r = 3; r < tp.ntrims; r++) {
      Round<EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, edgemap);
      checkCudaErrors(cudaDeviceSynchronize());
      edgemapcount(r, edgemap);
    }

    cudaMemcpy((void *)indexesA, indexesB+NMEM*NB, indexesSize, cudaMemcpyDeviceToDevice);
    EdgesMeet<EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesA);

    uint2 *bufferC = (uint2 *)bufferA;
    cudaMemset(indexesB, 0, indexesSize); // UnsplitEdges zeroes second indexes set
    UnsplitEdges<EDGES_A/NMEM, EDGES_A/128><<<NB, UNSPLIT_TPB>>>((uint2*)bufferB, indexesB+NMEM*NB, indexesA, bufferC, indexesB);
    checkCudaErrors(cudaDeviceSynchronize());
    indexcount(tp.ntrims, indexesB);
    indexcount(tp.ntrims, indexesB+NB);

#ifdef VERBOSE
    print_log("UnsplitEdges<><<<%d,%d>>>\n", NB, UNSPLIT_TPB);
#endif

    static_assert(NTLB >= 2, "2 index sets need to fit in indexesA");
    static_assert(NMEM >= 2, "2 index sets need to fit in indexesB");

    for (int r = 0; r < PROOFSIZE/2 - 1; r++) {
      // printf("Relay round %d\n", r);
      if (r % 2 == 0) {
        cudaMemset(indexesA, 0, 2*indexesSize);
        Tag_Relay<EDGES_A/128, EDGES_A/128><<<NB, RELAY_TPB>>>(bufferC, bufferC+NB_NMEM*EDGES_A, indexesB, indexesA);
      } else {
        cudaMemset(indexesB, 0, 2*indexesSize);
        Tag_Relay<EDGES_A/128, EDGES_A/128><<<NB, RELAY_TPB>>>(bufferC+NB_NMEM*EDGES_A, bufferC, indexesA, indexesB);
      }
      checkCudaErrors(cudaDeviceSynchronize());
    }

#ifdef VERBOSE
    print_log("Tail<><<<%d,%d>>>\n", NB, TAIL_TPB);
#endif

    cudaMemset(indexesA, 0, sizeof(u32));
    if ((PROOFSIZE/2 - 1) % 2 == 0) {
      Tail<EDGES_A/128><<<NB, TAIL_TPB>>>(bufferC,                 (uint2 *)bufferB, indexesB, indexesB+2*NB);
    } else {
      Tail<EDGES_A/128><<<NB, TAIL_TPB>>>(bufferC+NB_NMEM*EDGES_A, (uint2 *)bufferB, indexesA, indexesB+2*NB);
    }

    cudaMemcpy(&nedges, indexesB+2*NB, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print_log("%d rounds %d edges\n", tp.ntrims, nedges);
    return nedges;
  }
};

struct solver_ctx {
  edgetrimmer trimmer;
  bool mutatenonce;
  uint2 *edges;
  graph<word_t> cg;
  uint2 soledges[PROOFSIZE];
  std::vector<u32> sols; // concatenation of all proof's indices

  solver_ctx(const trimparams tp, bool mutate_nonce) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAX_SOLS, IDXSHIFT) {
    edges   = new uint2[MAXEDGES];
    mutatenonce = mutate_nonce;
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    if (mutatenonce)
      ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer.sipkeys);
    sols.clear();
  }
  ~solver_ctx() {
    delete[] edges;
  }

  int findcycles(uint2 *edges, u32 nedges) {
    u32 ndupes = 0;
    cg.reset();
    for (u32 i = 0; i < nedges; i++) {
      ndupes += !cg.add_compress_edge(edges[i].x, edges[i].y);
    }
    for (u32 s = 0 ;s < cg.nsols; s++) {
#ifdef VERBOSE
      print_log("Solution");
#endif
      for (u32 j = 0; j < PROOFSIZE; j++) {
        soledges[j] = edges[cg.sols[s][j]];
#ifdef VERBOSE
        print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
#endif
      }
#ifdef VERBOSE
      print_log("\n");
#endif
      sols.resize(sols.size() + PROOFSIZE);
      cudaMemcpyToSymbol(recovery, soledges, sizeof(soledges));
      cudaMemset(trimmer.indexesA, 0, trimmer.indexesSize);
#ifdef VERBOSE
    print_log("Recovery<><<<%d,%d>>>\n", trimmer.tp.recover.blocks, trimmer.tp.recover.tpb);
#endif
      Recovery<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>((u32 *)trimmer.bufferA);
      cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer.bufferA, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
      checkCudaErrors(cudaDeviceSynchronize());
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg.nonce_cmp);
    }
    return ndupes;
  }

  int solve() {
    u64 time0, time1;
    u32 timems,timems2;

    trimmer.abort = false;
    time0 = timestamp();
    u32 nedges = trimmer.trim();
    if (!nedges)
      return 0;
    if (nedges > MAXEDGES) {
      print_log("OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
    }
    cudaMemcpy(edges, trimmer.bufferB, sizeof(uint2[nedges]), cudaMemcpyDeviceToHost);
    time1 = timestamp(); timems  = (time1 - time0) / 1000000;
    time0 = timestamp();
    const u32 ndupes = findcycles(edges, nedges);
    time1 = timestamp(); timems2 = (time1 - time0) / 1000000;
    print_log("%d trims %d ms %d edges %d dupes %d ms total %d ms\n", trimmer.tp.ntrims, timems, nedges, ndupes, timems2, timems+timems2);
    return sols.size() / PROOFSIZE;
  }

  void abort() {
    trimmer.abort = true;
  }
};

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

typedef solver_ctx SolverCtx;

CALL_CONVENTION int run_solver(SolverCtx* ctx,
                               char* header,
                               int header_length,
                               u32 nonce,
                               u32 range,
                               SolverSolutions *solutions,
                               SolverStats *stats
                               )
{
  u64 time0, time1;
  u32 timems;
  u32 sumnsols = 0;
  int device_id;
  if (stats != NULL) {
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, stats->device_id);
    stats->device_id = device_id;
    stats->edge_bits = EDGEBITS;
    strncpy(stats->device_name, props.name, MAX_NAME_LEN);
  }

  if (ctx == NULL || !ctx->trimmer.initsuccess){
    print_log("Error initialising trimmer. Aborting.\n");
    print_log("Reason: %s\n", LAST_ERROR_REASON);
    if (stats != NULL) {
       stats->has_errored = true;
       strncpy(stats->error_reason, LAST_ERROR_REASON, MAX_NAME_LEN);
    }
    return 0;
  }

  for (u32 r = 0; r < range; r++) {
    time0 = timestamp();
    ctx->setheadernonce(header, header_length, nonce + r);
    print_log("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx->trimmer.sipkeys.k0, ctx->trimmer.sipkeys.k1, ctx->trimmer.sipkeys.k2, ctx->trimmer.sipkeys.k3);
    u32 nsols = ctx->solve();
    time1 = timestamp();
    timems = (time1 - time0) / 1000000;
    print_log("Time: %d ms\n", timems);
    for (unsigned s = 0; s < nsols; s++) {
      print_log("Solution");
      u32* prf = &ctx->sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        print_log(" %jx", (uintmax_t)prf[i]);
      print_log("\n");
      if (solutions != NULL){
        solutions->edge_bits = EDGEBITS;
        solutions->num_sols++;
        solutions->sols[sumnsols+s].nonce = nonce + r;
        for (u32 i = 0; i < PROOFSIZE; i++) 
          solutions->sols[sumnsols+s].proof[i] = (u64) prf[i];
      }
      int pow_rc = verify(prf, ctx->trimmer.sipkeys);
      if (pow_rc == POW_OK) {
        print_log("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          print_log("%02x", cyclehash[i]);
        print_log("\n");
      } else {
        print_log("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
    if (stats != NULL) {
      stats->last_start_time = time0;
      stats->last_end_time = time1;
      stats->last_solution_time = time1 - time0;
    }
  }
  print_log("%d total solutions\n", sumnsols);
  return sumnsols > 0;
}

CALL_CONVENTION SolverCtx* create_solver_ctx(SolverParams* params) {
  trimparams tp;
  tp.ntrims = params->ntrims;
  tp.seed.blocks = params->genablocks;
  tp.seed.tpb = params->genatpb;
  tp.trim0.tpb = params->genbtpb;
  tp.trim.tpb = params->trimtpb;
  tp.tail.tpb = params->tailtpb;
  tp.recover.blocks = params->recoverblocks;
  tp.recover.tpb = params->recovertpb;

  cudaDeviceProp prop;
  checkCudaErrors_N(cudaGetDeviceProperties(&prop, params->device));

  assert(tp.seed.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim0.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
  // assert(tp.tailblocks <= prop.threadDims[0]);
  assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
  assert(tp.recover.tpb <= prop.maxThreadsPerBlock);

  cudaSetDevice(params->device);
  if (!params->cpuload)
    checkCudaErrors_N(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

  return new SolverCtx(tp, params->mutate_nonce);
}

CALL_CONVENTION void destroy_solver_ctx(SolverCtx* ctx) {
  delete ctx;
}

CALL_CONVENTION void stop_solver(SolverCtx* ctx) {
  ctx->abort();
}

CALL_CONVENTION void fill_default_params(SolverParams* params) {
  trimparams tp;
  params->device = 0;
  params->ntrims = tp.ntrims;
  params->genablocks = tp.seed.blocks;
  params->genatpb = tp.seed.tpb;
  params->genbtpb = tp.trim0.tpb;
  params->trimtpb = tp.trim.tpb;
  params->tailtpb = tp.tail.tpb;
  params->recoverblocks = tp.recover.blocks;
  params->recovertpb = tp.recover.tpb;
  params->cpuload = false;
}

static_assert(NB % SEED_TPB == 0, "SEED_TPB must divide NB");
static_assert(NB % SPLIT_TPB == 0, "SPLIT_TPB must divide NB");
static_assert(NLISTS % (RELAY_TPB) == 0, "RELAY_TPB must divide NLISTS"); // for Tag_Edges lists    init
static_assert(NZ % (32 * SPLIT_TPB) == 0, "SPLIT_TPB must divide NZ/32"); // for EdgeSplit htnode init
static_assert(NZ % (32 *  TRIM_TPB) == 0, "TRIM_TPB must divide NZ/32"); // for Round htnode init

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  // set defaults
  SolverParams params;
  fill_default_params(&params);

  memset(header, 0, sizeof(header));
  while ((c = getopt(argc, argv, "scd:h:m:n:r:U:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedblocks] [-Z recoverblocks] [-z recoverthreads]\n", EDGEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -Z %d -z %d\n", EDGEBITS, device, tp.ntrims, nonce, range, tp.seed.blocks, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'c':
        params.cpuload = false;
        break;
      case 'd':
        device = params.device = atoi(optarg);
        break;
      case 'h':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
        break;
      case 'm': // ntrims         =        15;
        params.ntrims = atoi(optarg);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'U': // seed.blocks    =      1024;
        params.genablocks = atoi(optarg);
        break;
      case 'Z': // recover.blocks =      2048;
        params.recoverblocks = atoi(optarg);
        break;
      case 'z': // recover.tpb    =       256;
        params.recovertpb = atoi(optarg);
        break;
    }
  }

  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  assert(device < nDevices);
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  u64 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 102040; dbytes>>=10,dunit++) ;
  print_log("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
  // cudaSetDevice(device);

  print_log("Looking for %d-cycle on cuckaroom%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d buckets, and %d trims.\n", NB, params.ntrims);

  assert(params.recovertpb >= PROOFSIZE);
  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 102400; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
