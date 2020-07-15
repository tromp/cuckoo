// Cuckarooz Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2020 Wilke Trei (Lolliedieb), Jiri Vadura (photon) and John Tromp
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
#include "cuckarooz.hpp"
#include "graph.hpp"
#include "../crypto/blake2.h"

// number of invocations of seeding kernel
// larger values reduce pressure on TLB
// which appears to be an issue on GTX cards
#define NTLB 4
// number of slices in edge splitting kernel
// larger values reduce memory use
// which is proportional to 1+1/NMEM
#define NMEM 4
// after seeding, main memory buffer is a NMEM x NTLB matrix
// of blocks of buckets
#define NMEMTLB (NMEM * NTLB)

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
const u32 NB2 = NB / 2;
const u32 NB_NTLB  = NB / NTLB;
const u32 NB_NMEM  = NB / NMEM;
const u32 NODEBITS = EDGEBITS+1;
const u32 ZBITS = NODEBITS - BUCKBITS;
const u32 NZ = 1 << ZBITS;
const u32 ZMASK = NZ - 1;
const u32 ZBITS1 = ZBITS + 1;

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
__constant__ u64 recovery[2*PROOFSIZE];

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
#define TRIM_TPB 512
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
// Seed<EDGES_A/NTLB><<<tp.seed.blocks, SEED_TPB>>>((uint4*)bufferA, indexesA, i);
{
  const int group = blockIdx.x;
  const int lid = threadIdx.x;
  const int gid = group * SEED_TPB + lid;
  const int nthreads = gridDim.x * SEED_TPB;

  const int nloops = (NEDGES / NTLB / EDGE_BLOCK_SIZE - gid + nthreads-1) / nthreads;
  ulonglong4 sipblockL[EDGE_BLOCK_SIZE/4];
  uint64_t v0, v1, v2, v3;

  __shared__ unsigned long long magazine[NB];

  for (int i = 0; i < NB/SEED_TPB; i++)
    magazine[lid + SEED_TPB * i] = 0;

  __syncthreads();

  u64 offset = part * (NEDGES / NTLB);
  buffer  += part * (maxOut * NB_NMEM / 2); // buffer indexed by edgepairs rather than edges
  indexes += part * NB;

  for (int i = 0; i < nloops; i++) {
    u64 blockNonce = offset + (gid * nloops + i) * EDGE_BLOCK_SIZE;

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

#define BBITS2 (BUCKBITS/2)
#define NBB2 (1 << BBITS2)
#define BB2MASK (NBB2 - 1)
#define BBITS21 (BBITS2 - 1)
#define NBB21 (1 << BBITS21)
#define BB21MASK (NBB21 - 1)

#define DUMP(E) {\
  u64 lookup = (E);\
  const uint2 edge1 = make_uint2(lookup & NODEMASK, (lookup >> 32) & NODEMASK);\
  const u32 uX = edge1.x >> (ZBITS + BBITS2);\
  assert(uX < NBB2);\
  const u32 vX = edge1.y >> (ZBITS + BBITS2);\
  assert(vX < NBB2);\
  const u32 uvXa = 2 * (uX >> BBITS21) + (vX >> BBITS21);\
  assert(uvXa < 4);\
  const u32 uvXb = (uX & BB21MASK) << BBITS21 | (vX & BB21MASK);\
  assert(uvXb < NB/4);\
  const u32 bucket = uvXa << (BUCKBITS-2) | uvXb;\
  assert(bucket < NB);\
  u64 edge64 = (((u64)edge1.y) << 32) | edge1.x;\
  for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {\
    u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);\
    if (ret2 == ret) {\
      const u32 position = atomicAdd(indexes + bucket, 2);\
      if (position >= maxOut) {\
        printf("Seed dropping edges %llx %llx\n", ret, edge64);\
        break;\
      }\
      const u32 idx = (uvXa * NTLB * NB_NMEM + uvXb) * maxOut + position;\
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
      const u32 idx = ((bucket/NB_NMEM) * NTLB * NB_NMEM + bucket % NB_NMEM) * maxOut + position;
      buffer[idx/2] = make_uint4(edge, edge >> 32, 0, 0);
    }
  }
}

template<int maxIn, int maxOut>
__global__ void EdgeSplit(const uint2 *src, uint2 *dst, const u32 *srcIdx, u32 *dstIdx, u32 *edgeslots, const int uvXa)
// EdgeSplit<EDGES_A/NTLB, EDGES_A/NMEM/2><<<NB_NMEM*4, SPLIT_TPB>>>((uint2*)bufferA, (uint2*)bufferB, indexesA, indexesB, edgeSlots, i);
{
  const int uvYa = blockIdx.x % 4;
  const int uvXb = blockIdx.x / 4;
  const int lid = threadIdx.x;

  __shared__ unsigned long long magazine[NB];
  int nloops[NTLB];

  for (int i = 0; i < NB/SPLIT_TPB; i++) {
    magazine[lid + SPLIT_TPB * i] = 0;
    // emap[lid + SPLIT_TPB * i] = 0;
  }

  const u32 offset = uvXa * NB_NMEM;
  const u32 uvX = offset + uvXb;
  const u32 uXb = uvXb >> BBITS21; // 5 bits
  const u32 vXb = uvXb & BB21MASK; // 5 bits
  const u32 uX = (uvXa/2) << BBITS21 | uXb; // 6 bits
  const u32 vX = (uvXa%2) << BBITS21 | vXb; // 6 bits
  edgeslots += (uX * NBB2 + vX) * NB;
  if (!lid) assert(uXb < 32);
  if (!lid) assert(vXb < 32);

  for (int a = 0; a < NTLB; a++)
    nloops[a] = (min(srcIdx[a * NB + uvX], maxIn) - lid + SPLIT_TPB-1) / SPLIT_TPB;

  src += (offset * NTLB + uvXb) * maxIn + lid;
  dst += uvXa * 2 * NB * maxOut;
  uint2 *ydst = dst + NB * maxOut;
  u32 *xdstIdx = dstIdx + uvXa * 2 * NB;
  u32 *ydstIdx = xdstIdx + NB;

  __syncthreads();

  for (int a = 0; a < NTLB; a++) {
    const u32 delta = a * NB_NMEM * maxIn;
    for (int i = 0; i < nloops[a]; i++) {
      uint2 edge = src[delta + i * SPLIT_TPB];
      if (edge.x == edge.y) continue;
      const u32 uXYa = edge.x >> (ZBITS-1); // 13 bits
      const u32 vXYa = edge.y >> (ZBITS-1); // 13 bits
      const u32 uXY = uXYa >> 1;
      const u32 vXY = vXYa >> 1;
      assert(uXY < NB);
      assert(vXY < NB);
      assert((uXY >> BBITS2) == ((uvXa/2) << BBITS21 | uXb)); // match 6 bits
      assert((vXY >> BBITS2) == ((uvXa%2) << BBITS21 | vXb)); // match 6 bits
      if (2 * (uXYa % 2) + (vXYa % 2) != uvYa) continue;
      const u32 bucket = (uXY % NBB2) * NBB2 + vXY % NBB2;
      assert(bucket < NB);
      u64 edge64 = (((u64)edge.y) << 32) | edge.x;
      for (u64 ret = atomicCAS(&magazine[bucket], 0, edge64); ret; ) {
        u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);
        if (ret2 == ret) {
          const u32 slot = atomicAdd(edgeslots + bucket, 2);
          if (slot >= 64) {
#ifdef VERBOSE
            printf("dropped edges %llx %llx\n", ret, edge64);
#endif
            break;
          } else if (slot >= 63) {
#ifdef VERBOSE
            printf("dropped edge %llx\n", ret);
#endif
            break;
	  }
          const u32 slots = (slot+1) << 6 | slot; // slot for edge, slot+1 for ret
          const u32 xidx = atomicAdd(xdstIdx + uXYa % NB, 1);
          if (xidx >= maxOut) {
#ifdef VERBOSE
            printf("dropped u-halfedge %llx\n", edge64);
#endif
            break;
          }
          dst[(uXYa % NB) * maxOut + xidx] = make_uint2((vXY << ZBITS) | (edge.x & ZMASK), slots << ZBITS | ret & ZMASK);
          const u32 yIdx = atomicAdd(ydstIdx + vXYa % NB, 1);
          if (yIdx >= maxOut) {
#ifdef VERBOSE
            printf("dropped v-halfedge %llx\n", edge);
#endif
            break;
          }
          ydst[(vXYa % NB) * maxOut + yIdx] = make_uint2((uXY << ZBITS) | (edge.y & ZMASK), slots << ZBITS | (ret>>32) & ZMASK);
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
      const u32 slot = atomicAdd(edgeslots + bucket, 1);
      if (slot >= 64) {
#ifdef VERBOSE
        printf("dropped edge %llx\n", edge);
#endif
        continue;
      }
      const u32 uXY = uX * NBB2 + bucket / NBB2;
      const u32 vXY = vX * NBB2 + bucket % NBB2;
      const u32 uXYa = uXY << 1 | (uvYa / 2);
      const u32 vXYa = vXY << 1 | (uvYa % 2);
      assert(((edge & 0xffffffff) >> (ZBITS-1)) == uXYa);
      assert((edge >> (32+ZBITS-1)) == vXYa);
      const u32 slots = slot << 6 | slot; // duplicate slot indicates singleton pair
      int bktIdx = atomicAdd(xdstIdx + uXYa % NB, 1);
      if (bktIdx >= maxOut) {
#ifdef VERBOSE
        printf("dropped u-halfedge %llx\n", edge);
#endif
        break;
      }
      dst[(uXYa % NB) * maxOut + bktIdx] = make_uint2((vXY << ZBITS) | (edge & ZMASK), slots << ZBITS | edge & ZMASK);
      bktIdx = atomicAdd(ydstIdx + vXYa % NB, 1);
      if (bktIdx >= maxOut) {
#ifdef VERBOSE
        printf("dropped v-halfedge %llx\n", edge);
#endif
        break;
      }
      ydst[(vXYa % NB) * maxOut + bktIdx] = make_uint2((uXY << ZBITS) | (edge>>32) & ZMASK, slots << ZBITS | (edge>>32) & ZMASK);
    }
  }
}

__global__  void setEdgeMap(u32 *edgeslots, u64 *edgemap)
//setEdgeMap<<<NB, SPLIT_TPB>>>(edgeSlots, edgeMap);
{
  const int uX = blockIdx.x / NBB2;
  const int vX = blockIdx.x % NBB2;
  const int lid = threadIdx.x;

  edgeslots += blockIdx.x * NB;
  for (int i = 0; i < NB/SPLIT_TPB; i++) {
    int bucket = lid + SPLIT_TPB * i;
    edgemap[(uX*NBB2 + bucket/NBB2) * NB + vX*NBB2 + bucket%NBB2] = edgeslots[bucket] < 64 ? (1ULL << edgeslots[bucket]) - 1 : ~0ULL;
  }
}

__device__ __forceinline__ void addhalfedge(u32 *magazine, const u32 bucket, const u32 rest, const u32 slot, u32 *indices, uint2* dst) {
// addhalfedge(magazine, bucket, pair.x&ZMASK, slot0, sourceIndexes+NMEM*NB+group, source+delta);
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

static const int NODEDEGWORDS = NZ / 32;

__device__ __forceinline__  void nodeSet(u32 *nodeDeg, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  u32 old = atomicOr(nodeDeg + word, mask) & mask;
  if (old)
    atomicOr(nodeDeg + NODEDEGWORDS/2 + word, mask);
}

__device__ __forceinline__  bool nodeTest(u32 *nodeDeg, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;

  return (nodeDeg[NODEDEGWORDS/2 + word] >> bit) & 1;
}

template<int maxIn> // maxIn is size of buckets in half-edge apirs
__global__  void Round(uint2 *source, u32 *sourceIndexes, u64 *edgemap)
// Round<EDGES_A/NMEM/2><<<2*NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, edgeMap);
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  int xsize[2], ysize[2];
  int nxpairs[2], nypairs[2];

  __shared__ u32 nodeDeg[NODEDEGWORDS]; // node degrees
  __shared__ u32 magazine[NB];

  for (int i = 0; i < NODEDEGWORDS/TRIM_TPB; i++)
    nodeDeg[lid + i * TRIM_TPB] = 0;

  for (int i = 0; i < NB/TRIM_TPB; i++)
    magazine[lid + i * TRIM_TPB] = 0;

  const u32 groupa = group / NB;
  if (!lid) assert(groupa < 2);
  const int groupb = group % NB;
  for (int ai = 0; ai < 2; ai++) {
    xsize[ai] = sourceIndexes[(2 * groupa + ai) * 2*NB + groupb];
    nxpairs[ai] = (xsize[ai] + TRIM_TPB-1) / TRIM_TPB;
    ysize[ai] = sourceIndexes[(2 * ai + groupa) * 2*NB + NB + groupb];
    nypairs[ai] = (ysize[ai] + TRIM_TPB-1) / TRIM_TPB;
  }

  __syncthreads();

  if (lid < NMEM) {
    if (lid / 2 == groupa)
      sourceIndexes[lid * 2*NB + groupb] = 0;
    if (lid % 2 == groupa)
      sourceIndexes[lid * 2*NB + NB + groupb] = 0;
  }

#if 1
  for (int ai = 0; ai < 2; ai++) {
    int a = 2 * groupa + ai;
    const u32 delta = (a * 2*NB + groupb) * maxIn;
    // if (!group && !lid && !a) printf("xsize[%d] = %d nxpairs[a] = %d\n", a, xsize[ai], nxpairs[ai]);
    for (int i = 0; i < nxpairs[ai]; i++) {
      __syncthreads();
      const u32 idx = lid + i * TRIM_TPB;
      if (idx >= xsize[ai]) continue; // not to be optimized away with every iteration synced
      const uint2 pair = source[delta + idx];
      const u32 bucket = pair.x >> ZBITS;
      assert(bucket < NB);
      const u32 slots = pair.y >> ZBITS;
      const u32 slot0 = slots % 64, slot1 = slots / 64;
      assert(group/2 < NB);
      const u64 es = edgemap[(group/2) * NB + bucket]; // group/2 * NB
      // if (!group && !lid && !a) printf("delta+idx %d bucket %x pair %x %x\n", delta+idx, bucket, pair.x, pair.y);
      if (testbit(es, slot0))
        nodeSet(nodeDeg, pair.x & ZMASK/2);
      if (slot1 != slot0 && testbit(es, slot1))
        nodeSet(nodeDeg, pair.y & ZMASK/2);
    }
  }
#endif

#if 1
  for (int ai = 0; ai < 2; ai++) {
    int a = 2 * ai + groupa;
    const u32 delta = (a * 2*NB + NB + groupb) * maxIn;
    for (int i = 0; i < nypairs[ai]; i++) {
      __syncthreads();
      const u32 idx = lid + i * TRIM_TPB;
      if (idx >= ysize[ai]) continue; // not to be optimized away with every iteration synced
      const uint2 pair = source[delta + idx];
      const u32 bucket = pair.x >> ZBITS;
      assert(bucket < NB);
      const u32 slots = pair.y >> ZBITS;
      const u32 slot0 = slots % 64, slot1 = slots / 64;
      const u64 es = edgemap[bucket * NB + group/2];
      if (testbit(es, slot0))
        nodeSet(nodeDeg, pair.x & ZMASK/2);
      if (slot1 != slot0 && testbit(es, slot1))
        nodeSet(nodeDeg, pair.y & ZMASK/2);
    }
  }
#endif

#if 1
  for (int ai = 0; ai < 2; ai++) {
    int a = 2 * groupa + ai;
    const u32 delta = (a * 2*NB + groupb) * maxIn;
    for (int i = 0; i < nxpairs[ai]; i++) {
      __syncthreads();
      const u32 idx = lid + i * TRIM_TPB;
      if (idx >= xsize[ai]) continue; // not to be optimized away with every iteration synced
      const uint2 pair = source[delta + idx];
      const u32 bucket = pair.x >> ZBITS;
      assert(bucket < NB);
      const u32 slots = pair.y >> ZBITS;
      const u32 slot0 = slots % 64, slot1 = slots / 64;
      const u64 es = edgemap[(group/2) * NB + bucket];
      if (testbit(es, slot0)) {
        if (nodeTest(nodeDeg, pair.x & ZMASK/2)) {
          addhalfedge(magazine, bucket, pair.x&ZMASK, slot0, sourceIndexes+a*2*NB+groupb, source+delta);
        } else {
          // if (!group && !bucket) printf("u-halfedge %08x slot %d dies\n", pair.x & ZMASK, slot0);
	  resetbit(edgemap + (group/2) * NB + bucket, slot0);
	}
      }
      if (slot1 != slot0 && testbit(es, slot1)) {
        if (nodeTest(nodeDeg, pair.y & ZMASK/2)) {
          addhalfedge(magazine, bucket, pair.y&ZMASK, slot1, sourceIndexes+a*2*NB+groupb, source+delta);
        } else {
          // if (!group && !bucket) printf("u-halfedge %08x slot %d dies\n", pair.y & ZMASK, slot1);
          resetbit(edgemap + (group/2) * NB + bucket, slot1);
	}
      }
    }
    __syncthreads();
    flushhalfedges(magazine, lid, sourceIndexes+a*2*NB+groupb, source+delta);
  }
#endif

#if 1
  for (int ai = 0; ai < 2; ai++) {
    int a = 2 * ai + groupa;
    const u32 delta = (a * 2*NB + NB + groupb) * maxIn;
    for (int i = 0; i < nypairs[ai]; i++) {
      __syncthreads();
      const u32 idx = lid + i * TRIM_TPB;
      if (idx >= ysize[ai]) continue; // not to be optimized away with every iteration synced
      const uint2 pair = source[delta + idx];
      const u32 bucket = pair.x >> ZBITS;
      assert(bucket < NB);
      const u32 slots = pair.y >> ZBITS;
      const u32 slot0 = slots % 64, slot1 = slots / 64;
      const u64 es = edgemap[bucket * NB + group/2];
      if (testbit(es, slot0)) {
	if (nodeTest(nodeDeg, pair.x & ZMASK/2)) {
          addhalfedge(magazine, bucket, pair.x&ZMASK, slot0, sourceIndexes+a*2*NB+NB+groupb, source+delta);
	} else {
          // if (!group && !bucket) printf("v-halfedge %08x slot %d dies\n", pair.x & ZMASK, slot0);
	  resetbit(edgemap + bucket * NB + group/2, slot0);
	}
      }
      if (slot1 != slot0 && testbit(es, slot1)) {
        if (nodeTest(nodeDeg, pair.y & ZMASK/2)) {
          addhalfedge(magazine, bucket, pair.y&ZMASK, slot1, sourceIndexes+a*2*NB+NB+groupb, source+delta);
	} else {
          // if (!group && !bucket) printf("v-halfedge %08x slot %d dies\n", pair.y & ZMASK, slot1);
	  resetbit(edgemap + bucket * NB + group/2, slot1);
	}
      }
    }
    __syncthreads();
    flushhalfedges(magazine, lid, sourceIndexes+a*2*NB+NB+groupb, source+delta);
  }
#endif
}

template<int maxIn>
__global__ void EdgesMeet(uint2 *source, u32 *sourceIndexes, u32 *destIndexes)
// EdgesMeet<EDGES_A/NMEM/2><<<4*NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesA);
{
  const int lid = threadIdx.x;
  const int groupa = blockIdx.x / NB;
  assert(groupa < 4);
  const int groupb = blockIdx.x % NB;

  int vsize = sourceIndexes[groupa * 2*NB + NB + groupb];
  int nvpairs = (vsize + TRIM_TPB-1) / TRIM_TPB;

  __syncthreads();

  const u32 deltaOut = groupa * 2*NB * maxIn;
  const u32 deltaIn = deltaOut + (NB + groupb) * maxIn;
  const int vX = (groupa%2) * NB2 + groupb/2;
  for (int i = 0; i < nvpairs; i++) {
    __syncthreads();
    u32 idx = lid + i * TRIM_TPB;
    if (idx >= vsize) continue;
    uint2 pair = source[deltaIn + idx];
    u32 uXb = (pair.x >> (ZBITS-1)) % NB;
    pair.x = (vX << ZBITS) | (pair.x & ZMASK);
    int i = atomicAdd(destIndexes + groupa * 2*NB + uXb, 1);
    source[deltaOut + uXb * maxIn + i] = pair;
  }
}

template<int maxIn, int  maxOut>
__global__ void EdgesMoveU(uint2 *source, u32 *indexes, u32 *dstIndexes)
// EdgesMoveU<EDGES_A/NMEM/2,2*EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesB+NMEM*2*NB);
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int aOut = group / NB_NMEM;
  const int uXa = aOut / 2;
  assert(uXa < 2);
  const int groupb = group % NB2;
  const int groupbb = group % NB_NMEM;

  const u32 deltaOut0 = (aOut * 2*NB + NB) * maxIn + groupbb * maxOut;
  u32 deltaOut = deltaOut0;
  for (int vXa = 0; vXa < 2; vXa++) {
    int aIn = 2 * uXa + vXa;
    for (int uYa = 0; uYa < 2; uYa++) {
      const int groupba = 2*groupb +uYa;
      int usize = indexes[aIn * 2*NB + groupba];
      int nupairs = (usize + TRIM_TPB-1) / TRIM_TPB;
      const u32 deltaIn = (aIn * 2*NB + groupba) * maxIn;
      for (int i = 0; i < nupairs; i++) {
        u32 idx = lid + i * TRIM_TPB;
        if (idx >= usize) continue;
        uint2 pair = source[deltaIn + idx];
        // if (group==0 && lid>=0) printf("pair %08x %08x\n", pair.x, pair.y);
        source[deltaOut + idx] = pair;
      }
      deltaOut += usize;
    }
  }
  if (!lid)
    dstIndexes[group] = deltaOut - deltaOut0;
}

template<int maxIn, int  maxOut>
__global__ void EdgesMoveV(uint2 *source, u32 *indexes, u32 *indexes2, u32 *dstIndexes)
// EdgesMoveV<EDGES_A/NMEM/2,2*EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesA, indexesB+NMEM*2*NB);
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int aOut = group / NB_NMEM;
  const int uXa = aOut / 2;
  assert(uXa < 2);
  const int groupb = group % NB2;
  const int groupbb = group % NB_NMEM;

  const u32 deltaOut0 = (aOut * 2*NB + NB) * maxIn + groupbb * maxOut;
  const u32 uOutsize = dstIndexes[group];
  u32 deltaOut = deltaOut0 + uOutsize;
  for (int vXa = 0; vXa < 2; vXa++) {
    int aIn = 2 * uXa + vXa;
    for (int uYa = 0; uYa < 2; uYa++) {
      const int groupba = 2*groupb +uYa;
      int usize = indexes[aIn * 2*NB + groupba];
      int vsize = indexes2[aIn * 2*NB + groupba] - usize;
      assert(vsize >= 0);
      int nvpairs = (vsize + TRIM_TPB-1) / TRIM_TPB;
      const u32 deltaIn = (aIn * 2*NB + groupba) * maxIn + usize;
      for (int i = 0; i < nvpairs; i++) {
        u32 idx = lid + i * TRIM_TPB;
        if (idx >= vsize) continue;
        uint2 pair = source[deltaIn + idx];
        // if (group==0 && lid>=0) printf("pair %08x %08x\n", pair.x, pair.y);
        source[deltaOut + idx] = pair;
      }
      deltaOut += vsize;
    }
  }
  if (!lid)
    dstIndexes[NB+group] = deltaOut - deltaOut0;
}

template<int maxOut>
__device__ __forceinline__ void addedge(const u32 buckx, const u32 restx, const u32 bucky, const u32 resty, u32 *indices, uint2* dest) {
  // if (restx==0x150f3 || resty==0x150f3) printf("x %x %x y %x %x\n", buckx, restx, bucky, resty);
  int idx  = atomicAdd(indices + bucky, 1);
  dest[bucky * maxOut + idx] = make_uint2(restx << ZBITS1 | resty, buckx << ZBITS | restx); // extra +1 shift makes a 0 copybit
  int idx2 = atomicAdd(indices + buckx, 1);
  dest[buckx * maxOut + idx2] = make_uint2(resty << ZBITS1 | restx, bucky << ZBITS | resty);
}

template<int maxIn, int maxOut> // maxIn is size of small buckets in half-edges
__global__ void UnsplitEdges(uint2 *source, u32 *srcIndexes, u32 *srcIndexes2, uint2 *dest, u32 *dstIndexes)
// UnsplitEdges<EDGES_A/NMEM, EDGES_A/64><<<NB, UNSPLIT_TPB>>>((uint2*)bufferB, indexesB+NMEM*NB, indexesA, bufferC, indexesB);
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 lists[NB];
  __shared__ u32 nexts[NB];

  for (int i = 0; i < NB/UNSPLIT_TPB; i++)
    lists[i * UNSPLIT_TPB + lid] = ~0;

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
    // if (group==0 && lid==0) printf("pair %08x %08x\n", pair.x, pair.y);
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
      // if (group==0 && lid==0) printf("pair %08x %08x pair2 %08x %08x\n", pair.x, pair.y, pair2.x, pair2.y);
      if (slot20 == slot0)
        addedge<maxOut>(group, pair2.x & ZMASK, bucket, pair.x & ZMASK, dstIndexes, dest);
      else if (slot20 == slot1)
        addedge<maxOut>(group, pair2.x & ZMASK, bucket, pair.y & ZMASK, dstIndexes, dest);
      if (slot21 != slot20) {
        if (slot21 == slot0)
          addedge<maxOut>(group, pair2.y & ZMASK, bucket, pair.x & ZMASK, dstIndexes, dest);
        else if (slot21 == slot1)
          addedge<maxOut>(group, pair2.y & ZMASK, bucket, pair.y & ZMASK, dstIndexes, dest);
      }
      // if (slots != slots2) printf("pair slots %d %d pair2 slots %d %d\n", slot0, slot1, slot20, slot21);
    }
  }
}

#ifndef LISTBITS
#define LISTBITS 12
#endif

const u32 NLISTS  = 1 << LISTBITS;

#ifndef NNEXTS
#define NNEXTS NLISTS
#endif

template<int maxIn, int maxOut>
__global__  void Tag_Relay(const uint2 *source, uint2 *destination, const u32 *sourceIndexes, u32 *destinationIndexes)
// Tag_Relay<EDGES_A/64, EDGES_A/64><<<NB, RELAY_TPB>>>(bufferC, bufferD, indexesB, indexesA);
// Tag_Relay<EDGES_A/64, EDGES_A/64><<<NB, RELAY_TPB>>>(bufferD, bufferC, indexesA, indexesB);
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const u32 LISTMASK = NLISTS - 1;
  const u32 TAGMASK = (~1U) << ZBITS;
  const u32 COPYFLAG = NZ;

  __shared__ u32 lists[NLISTS];
  __shared__ u32 nexts[NNEXTS];

  const int nloops = (min(sourceIndexes[group], NNEXTS) - lid + RELAY_TPB-1) / RELAY_TPB;

  source += group * maxIn;

  for (int i = 0; i < NLISTS/RELAY_TPB; i++)
    lists[i * RELAY_TPB + lid] = ~0;

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    const u32 index = i * RELAY_TPB + lid;
    const uint2 edge = source[index];
    // if (group==0 && lid>=0) printf("edge %08x %08x\n", edge.x, edge.y);
    const u32 list = edge.x & LISTMASK;
    nexts[index] = atomicExch(&lists[list], index);
  }

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    const u32 index = i * RELAY_TPB + lid;
    const uint2 edge = source[index];
    if (edge.x & COPYFLAG) continue; // copies don't relay
    const u32 list = edge.x & LISTMASK;
    // nexts1[index] = atomicExch(&lists1[list], index);
    u32 bucket = edge.y >> ZBITS;
    u32 copybit = 0;
    for (u32 idx = lists[list]; idx != ~0; idx = nexts[idx]) {
      uint2 tagged = source[idx];
      // printf("compare with tagged %08x %08x xor %x\n", tagged.x, tagged.y, (tagged.x ^ edge.x) & ZMASK);
      if (idx == index || (tagged.x ^ edge.x) & ZMASK || tagged.y == edge.y) continue;
      if (((group << ZBITS) | (edge.x & ZMASK)) == edge.y) { printf("OOPS! SELF\n"); continue; }
      u32 bktIdx = atomicAdd(destinationIndexes + bucket, 1); // , maxOut - 1);
      assert(bktIdx < maxOut);
      // printf("relaying from tagged %08x %08x bktIdx %d\n", tagged.x, tagged.y, bktIdx);
      u32 tag = (tagged.x & TAGMASK) | copybit;
      destination[bucket * maxOut + bktIdx] = make_uint2(tag | (edge.y & ZMASK), (group << ZBITS) | (edge.x & ZMASK));
      copybit = COPYFLAG;
    }
  }
}

template<int maxIn>
__global__  void Tail(const uint2 *source, uint2 *destination, const u32 *sourceIndexes, u32 *destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const u32 LISTMASK = NLISTS - 1;
  const u32 COPYFLAG = NZ;

  __shared__ u32 lists[NLISTS];
  __shared__ u32 nexts[NNEXTS];

  const int nloops = (min(sourceIndexes[group], NNEXTS) - lid + TAIL_TPB-1) / TAIL_TPB;

  source += group * maxIn;

  for (int i = 0; i < NLISTS/TAIL_TPB; i++)
    lists[i * TAIL_TPB + lid] = ~0;

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    const u32 index = i * TAIL_TPB + lid;
    const uint2 edge = source[index];
    const u32 list = edge.x & LISTMASK;
    nexts[index] = atomicExch(&lists[list], index);
  }

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    const u32 index = i * TAIL_TPB + lid;
    const uint2 edge = source[index];
    const u32 list = edge.x & LISTMASK;
    for (u32 idx = lists[list]; idx != ~0; idx = nexts[idx]) {
      uint2 tagged = source[idx];
      if (idx == index || (tagged.x ^ edge.x) & ~COPYFLAG) continue;
      u32 bktIdx = atomicAdd(destinationIndexes, 1);
      destination[bktIdx] = make_uint2((group << ZBITS) | (edge.x & ZMASK), edge.y);
    }
  }
}

__global__  void Recovery(u32 * indexes)
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
    assert(blockNonce < NEDGES);

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
        if (recovery[i] == (lookup & NODEMASK2) || recovery[PROOFSIZE+i] == (lookup & NODEMASK2))
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
#define NEPS_A 144
#endif
#ifndef NEPS_B
#define NEPS_B 92
#endif
#define NEPS 128

const u32 EDGES_A = (NEDGES / NB) * NEPS_A / NEPS;
const u32 EDGES_B = (NEDGES / NB) * NEPS_B / NEPS;

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
  const size_t edgeSlotsSize = NB * NB * sizeof(u32); // count slots in edgeMap
  const size_t edgeMapSize = 2 * NEDGES / 8; // 8 bits per byte; twice that for bucket (2^24 in all) size variance
  u8 *bufferA;
  u8 *bufferB;
  u8 *bufferA1;
  u32 *indexesA;
  u32 *indexesB;
  u32 *edgeSlots;
  u64 *edgeMap;
  u32 nedges;
  siphash_keys sipkeys;
  bool abort;
  bool initsuccess = false;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors_V(cudaMalloc((void**)&indexesA, 2*indexesSizeNTLB));
    checkCudaErrors_V(cudaMalloc((void**)&indexesB, 3*indexesSizeNMEM));
    checkCudaErrors_V(cudaMalloc((void**)&edgeSlots, edgeSlotsSize));
    checkCudaErrors_V(cudaMalloc((void**)&bufferB, bufferSize));
    bufferA = bufferB + sizeA / NMEM;
    assert(edgeMapSize < sizeA / NMEM);
    assert(edgeMapSize == NB * NB * sizeof(u64));
    edgeMap = (u64 *)(bufferB + bufferSize - edgeMapSize);
    bufferA1 = bufferB + bufferSize - sizeA / NMEM;
    print_log("allocated %lld bytes bufferB %llx endBuffer %llx\n", bufferSize, bufferB, bufferB+bufferSize);
    print_log("bufferA %llx bufferA1 %llx\n", bufferA, bufferA1);
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    initsuccess = true;
  }
  u64 globalbytes() const {
    return bufferSize + 2*indexesSizeNTLB + 3*indexesSizeNMEM + edgeSlotsSize + edgeMapSize + sizeof(siphash_keys) + sizeof(edgetrimmer);
  }
  ~edgetrimmer() {
    checkCudaErrors_V(cudaFree(bufferB));
    checkCudaErrors_V(cudaFree(indexesA));
    checkCudaErrors_V(cudaFree(indexesB));
    checkCudaErrors_V(cudaFree(edgeSlots));
    checkCudaErrors_V(cudaFree(dt));
    cudaDeviceReset();
  }
  void indexcount(u32 round, const u32 *indexes) {
#if VERBOSE
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
    if (abort) return false;

    assert((uint2*)bufferB + NB * EDGES_A/NMEM == (uint2 *)bufferA);
    cudaMemset(edgeSlots, 0, edgeSlotsSize);
    cudaMemset(indexesB, 0, 2*indexesSizeNMEM);
    for (u32 i=0; i < NMEM; i++) {
      EdgeSplit<EDGES_A/NTLB, EDGES_A/NMEM/2><<<NB_NMEM*4, SPLIT_TPB>>>((uint2*)bufferA, (uint2*)bufferB, indexesA, indexesB, edgeSlots, i);
      if (abort) return false;
    }

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
  
    setEdgeMap<<<NB, SPLIT_TPB>>>(edgeSlots, edgeMap);
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef VERBOSE
    for (u32 i=0; i < 2*NMEM; i++)
      indexcount(1, indexesB+i*NB);
    edgemapcount(1, edgeMap); // .400
    print_log("EdgeSplit completed in %.0f ms\n", durationB);
    print_log("Round<<<%d,%d>>>\n", NB, TRIM_TPB); // 4096x1024
#endif

    cudaEventRecord(start, NULL);
    Round<EDGES_A/NMEM/2><<<2*NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, edgeMap);

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationC, start, stop);
    checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(stop));
  
    print_log("Round completed in %.0f ms\n", durationC);
    for (u32 i=0; i < 2*NMEM; i++)
      indexcount(2, indexesB+i*NB);
    edgemapcount(2, edgeMap);
    if (abort) return false;

#ifdef VERBOSE
    print_log("Round<><<<%d,%d>>>\n", NB, TRIM_TPB);
#endif

    for (int r = 3; r < tp.ntrims; r++) {
      Round<EDGES_A/NMEM/2><<<2*NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, edgeMap);
      checkCudaErrors(cudaDeviceSynchronize());
      indexcount(r, indexesB);
      edgemapcount(r, edgeMap);
    }

    cudaMemcpy((void *)indexesA, indexesB, 2*indexesSizeNMEM, cudaMemcpyDeviceToDevice); // assumes NMEM == NTLB == 4
    EdgesMeet<EDGES_A/NMEM/2><<<4*NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesA);
    checkCudaErrors(cudaDeviceSynchronize());
    EdgesMoveU<EDGES_A/NMEM/2,2*EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesB+NMEM*2*NB);
    checkCudaErrors(cudaDeviceSynchronize());
    EdgesMoveV<EDGES_A/NMEM/2,2*EDGES_A/NMEM><<<NB, TRIM_TPB>>>((uint2*)bufferB, indexesB, indexesA, indexesB+NMEM*2*NB);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy((void *)indexesA, indexesB+NMEM*2*NB+NB, indexesSize, cudaMemcpyDeviceToDevice);

    indexcount(tp.ntrims, indexesB+NMEM*2*NB);
    indexcount(tp.ntrims, indexesA);

#ifdef VERBOSE
    print_log("Done moving\n");
#endif

    uint2 *bufferC = (uint2 *)bufferA;
    uint2 *bufferD = bufferC+NB_NMEM*EDGES_A;
    cudaMemset(indexesB, 0, indexesSize);
    UnsplitEdges<EDGES_A/NMEM, EDGES_A/64><<<NB, UNSPLIT_TPB>>>((uint2*)bufferB, indexesB+NMEM*2*NB, indexesA, bufferC, indexesB);
    checkCudaErrors(cudaDeviceSynchronize());
    indexcount(tp.ntrims, indexesB);
    indexcount(tp.ntrims, indexesB+NB);

#ifdef VERBOSE
    print_log("UnsplitEdges<><<<%d,%d>>>\n", NB, UNSPLIT_TPB);
#endif

    static_assert(NTLB == 4, "4 index sets need to fit in indexesA");
    static_assert(NMEM == 4, "4 index sets need to fit in indexesB");

    for (int r = 0; r < PROOFSIZE/2 - 1; r++) {
      if (r % 2 == 0) {
        cudaMemset(indexesA, 0, indexesSize);
        Tag_Relay<EDGES_A/64, EDGES_A/64><<<NB, RELAY_TPB>>>(bufferC, bufferD, indexesB, indexesA);
        indexcount(tp.ntrims, indexesA);
      } else {
        cudaMemset(indexesB, 0, indexesSize);
        Tag_Relay<EDGES_A/64, EDGES_A/64><<<NB, RELAY_TPB>>>(bufferD, bufferC, indexesA, indexesB);
        indexcount(tp.ntrims, indexesB);
      }
      checkCudaErrors(cudaDeviceSynchronize());
    }

#ifdef VERBOSE
    print_log("Tail<><<<%d,%d>>>\n", NB, TAIL_TPB);
#endif

    cudaMemset(indexesB+NB, 0, sizeof(u32));
    if ((PROOFSIZE/2 - 1) % 2 == 0) {
      Tail<EDGES_A/64><<<NB, TAIL_TPB>>>(bufferC, (uint2 *)bufferB, indexesB, indexesB+NB);
    } else {
      Tail<EDGES_A/64><<<NB, TAIL_TPB>>>(bufferD, (uint2 *)bufferB, indexesA, indexesB+NB);
    }

    cudaMemcpy(&nedges, indexesB+NB, sizeof(u32), cudaMemcpyDeviceToHost);
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
  uint2 soledges[2*PROOFSIZE];
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
        soledges[PROOFSIZE+j] = make_uint2(soledges[j].y, soledges[j].x);
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
      Recovery<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>((u32 *)trimmer.bufferA1);
      cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer.bufferA1, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
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

static_assert(BUCKBITS % 2 == 0, "BUCKBITS must be even");
static_assert(NB % SEED_TPB == 0, "SEED_TPB must divide NB");
static_assert(NB % SPLIT_TPB == 0, "SPLIT_TPB must divide NB");
static_assert(NLISTS % (RELAY_TPB) == 0, "RELAY_TPB must divide NLISTS"); // for Tag_Edges lists    init
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
      case 'm': // ntrims         =       15;
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

  print_log("Looking for %d-cycle on cuckarooz%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
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
