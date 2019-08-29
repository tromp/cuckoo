// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2019 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include "cuckarood.hpp"
#include "graph.hpp"
#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"
#include "kernel.cu"

typedef uint8_t u8;
typedef uint16_t u16;

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#define IDXSHIFT 12
#endif

const u32 MAXEDGES = NEDGES2 >> IDXSHIFT;

#ifndef NEPS_A
#define NEPS_A 134 // to match Photon's kernel.cu
#endif
#ifndef NEPS_B
#define NEPS_B 85 // to match Photon's kernel.cu
#endif
#define NEPS 128

const u32 EDGES_A = NZ * NEPS_A / NEPS;
const u32 EDGES_B = NZ * NEPS_B / NEPS;

const u32 ROW_EDGES_A = EDGES_A * NY;
const u32 ROW_EDGES_B = EDGES_B * NY;

// Number of Parts of BufferB, all but one of which will overlap BufferA
#ifndef NA
#define NA 4
#endif
#define NA2 (NA * NA)

__constant__ uint2 recoveredges[PROOFSIZE];
__constant__ uint2 e0 = {0,0};

__device__ u64 dipblock(const siphash_keys &keys, const word_t edge, u64 *buf) {
  diphash_state<25> shs(keys);
  word_t edge0 = edge & ~EDGE_BLOCK_MASK;
  u32 i;
  for (i=0; i < EDGE_BLOCK_MASK; i++) {
    shs.hash24(edge0 + i);
    buf[i] = shs.xor_lanes();
  }
  shs.hash24(edge0 + i);
  buf[i] = 0;
  return shs.xor_lanes();
}

__device__ u32 endpoint(uint2 nodes, int uorv) {
  return uorv ? nodes.y : nodes.x;
}

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

template<int maxOut>
__global__ void SeedA(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, u32 * __restrict__ indexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = gridDim.x * dim;
  const int FLUSHA2 = 2*FLUSHA;

  __shared__ uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);
  __shared__ int counters[NX];
  u64 buf[EDGE_BLOCK_SIZE];

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  __syncthreads();

  const int col = group % NX;
  const int loops = NEDGES2 / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++) {
      u64 edge = buf[e] ^ last;
      u32 dir = e & 1;
      u32 node0 = (edge        &  NODE1MASK) << 1 | dir;
      u32 node1 = (edge >> 31) & (NODE1MASK << 1) | dir;
      int row = node0 >> YZBITS;
      int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
      tmp[row][counter] = make_uint2(node0, node1);
      __syncthreads();
      if (counter == FLUSHA-1) {
        int localIdx = min(FLUSHA2, counters[row]);
        int newCount = localIdx % FLUSHA;
        int nflush = localIdx - newCount;
        u32 grp = row * NX + col;
        int cnt = min((int)atomicAdd(indexes + grp, nflush), (int)(maxOut - nflush));
        for (int i = 0; i < nflush; i += TMPPERLL4)
          buffer[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
        for (int t = 0; t < newCount; t++) {
          tmp[row][t] = tmp[row][t + nflush];
        }
        counters[row] = newCount;
      }
      __syncthreads();
    }
  }
  uint2 zero = make_uint2(0, 0);
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    u32 grp = row * NX + col;
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      buffer[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
    }
  }
}

template <typename Edge> __device__ bool null(Edge e);

__device__ bool null(u32 nonce) {
  return nonce == 0;
}

__device__ bool null(uint2 nodes) {
  return nodes.x == 0 && nodes.y == 0;
}

#ifndef FLUSHB
#define FLUSHB 8
#endif

template<int maxOut>
__global__ void SeedB(const uint2 * __restrict__ source, ulonglong4 * __restrict__ destination, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int FLUSHB2 = 2 * FLUSHB;

  __shared__ uint2 tmp[NX][FLUSHB2];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);
  __shared__ int counters[NX];

  for (int col = lid; col < NX; col += dim)
    counters[col] = 0;
  __syncthreads();
  const int row = group / NX;
  const int bucketEdges = min((int)srcIdx[group], (int)maxOut);
  const int loops = (bucketEdges + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    int col;
    int counter = 0;
    const int edgeIndex = loop * dim + lid;
    if (edgeIndex < bucketEdges) {
      const int index = group * maxOut + edgeIndex;
      uint2 edge = __ldg(&source[index]);
      if (!null(edge)) {
        u32 node1 = edge.x;
        col = (node1 >> ZBITS) & XMASK;
        counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1)); // assuming COLS_LIMIT_LOSSES checked
        tmp[col][counter] = edge;
        }
    }
    __syncthreads();
    if (counter == FLUSHB-1) {
      int localIdx = min(FLUSHB2, counters[col]);
      int newCount = localIdx % FLUSHB;
      int nflush = localIdx - newCount;
      u32 grp = row * NX + col;
#ifdef SYNCBUG
      if (grp==0x2d6) printf("group %x size %d lid %d nflush %d\n", group, bucketEdges, lid, nflush);
#endif
      int cnt = min((int)atomicAdd(dstIdx + grp, nflush), (int)(maxOut - nflush));
      for (int i = 0; i < nflush; i += TMPPERLL4)
        destination[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
      for (int t = 0; t < newCount; t++) {
        tmp[col][t] = tmp[col][t + nflush];
      }
      counters[col] = newCount;
    }
    __syncthreads(); 
  }
  uint2 zero = make_uint2(0, 0);
  for (int col = lid; col < NX; col += dim) {
    int localIdx = min(FLUSHB2, counters[col]);
    u32 grp = row * NX + col;
#ifdef SYNCBUG
    if (group==0x2f2 && grp==0x2d6) printf("group %x size %d lid %d localIdx %d\n", group, bucketEdges, lid, localIdx);
#endif
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[col][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(dstIdx + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      destination[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
    }
  }
}

__device__ __forceinline__  void bitmapset(u32 *ebitmap, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;
  atomicOr(ebitmap + word, mask);
}

__device__ __forceinline__  bool bitmaptest(u32 *ebitmap, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  return (ebitmap[word] >> bit) & 1;
}

template<int NP, int maxIn, int maxOut>
__global__ void Round(const int round, const uint2 * __restrict__ src, uint2 * __restrict__ dst, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int BITMAPWORDS = NZ / 32;

  __shared__ u32 ebitmap[BITMAPWORDS];

  for (int i = lid; i < BITMAPWORDS; i += dim)
    ebitmap[i] = 0;
  __syncthreads();

  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group], maxIn);
    // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
    const int loops = (edgesInBucket + dim-1) / dim;

    for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
        const int index = maxIn * group + lindex;
        uint2 edge = __ldg(&src[index]);
        if (null(edge)) continue;
        u32 node = endpoint(edge, round&1);
        bitmapset(ebitmap, node & ZMASK);
      }
    }
  }

  __syncthreads();

  src -= NP * NX2 * maxIn; srcIdx -= NP * NX2;
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group], maxIn);
    const int loops = (edgesInBucket + dim-1) / dim;
    for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
        const int index = maxIn * group + lindex;
        uint2 edge = __ldg(&src[index]);
        if (null(edge)) continue;
        u32 node0 = endpoint(edge, round&1);
        if (bitmaptest(ebitmap, (node0 & ZMASK) ^ 1)) {
          u32 node1 = endpoint(edge, (round&1)^1);
          const int bucket = node1 >> ZBITS;
          const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
          dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
        }
      }
    }
  }
}

template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const u32 *srcIdx, u32 *dstIdx) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  int myEdges = srcIdx[group];
  __shared__ int destIdx;

  if (lid == 0)
    destIdx = atomicAdd(dstIdx, myEdges);
  __syncthreads();
  for (int i = lid; i < myEdges; i += dim)
    destination[destIdx + lid] = source[group * maxIn + lid];
}

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

__global__ void Recovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int loops = NEDGES2 / nthreads;
  __shared__ u32 nonces[PROOFSIZE];
  u64 buf[EDGE_BLOCK_SIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
      u64 edge = buf[i] ^ last;
      u32 dir = i & 1;
      u32 u = (edge & NODE1MASK) << 1 | dir;
      u32 v = ((edge >> 32) & NODE1MASK) << 1 | dir;
      for (int p = 0; p < PROOFSIZE; p++) { //YO
        if (recoveredges[p].x == u && recoveredges[p].y == v) {
          nonces[p] = nonce0 + i;
        }
      }
    }
  }
  __syncthreads();
  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}

struct blockstpb {
  u16 blocks;
  u16 tpb;
};

#ifndef SEED_TPB
#define SEED_TPB 256
#endif
#ifndef TRIM0_TPB
#define TRIM0_TPB 1024
#endif
#ifndef TRIM1_TPB
#define TRIM1_TPB 512
#endif
#ifndef TRIM_TPB
#define TRIM_TPB 512
#endif

struct trimparams {
  u16 ntrims;
  blockstpb seed;
  blockstpb trim0;
  blockstpb trim1;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    ntrims         =       458;
    seed.blocks    =        64;
    seed.tpb       =  SEED_TPB;
    trim0.blocks   =    NX2/NA;
    trim0.tpb      = TRIM0_TPB;
    trim1.blocks   =    NX2/NA;
    trim1.tpb      = TRIM1_TPB;
    trim.blocks    =       NX2;
    trim.tpb       =  TRIM_TPB;
    tail.blocks    =       NX2;
    tail.tpb       =       256;
    recover.blocks =      2048;
    recover.tpb    =       256;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  size_t sizeA, sizeB;
  const size_t indexesSize = NX2 * sizeof(u32);
  const size_t indexesSizeNA = NA * indexesSize;
  u8 *bufferA;
  u8 *bufferB;
  u8 *bufferA1;
  u32 *indexesA;
  u32 *indexesB;
  u32 nedges;
  u32 *uvnodes;
  siphash_keys sipkeys;
  bool abort;
  bool initsuccess = false;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors_V(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
    checkCudaErrors_V(cudaMalloc((void**)&indexesA, indexesSizeNA));
    checkCudaErrors_V(cudaMalloc((void**)&indexesB, indexesSizeNA));
    sizeA = ROW_EDGES_A * NX * sizeof(uint2);
    sizeB = ROW_EDGES_B * NX * sizeof(uint2);
    const size_t bufferSize = sizeA + sizeB / NA;
    checkCudaErrors_V(cudaMalloc((void**)&bufferB, bufferSize));
    bufferA = bufferB + sizeB / NA;
    bufferA1 = bufferB + sizeB;
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    initsuccess = true;
  }
  u64 globalbytes() const {
    return (sizeA+sizeB/NA) + (1+NA) * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer);
  }
  ~edgetrimmer() {
    checkCudaErrors_V(cudaFree(bufferB));
    checkCudaErrors_V(cudaFree(indexesA));
    checkCudaErrors_V(cudaFree(indexesB));
    checkCudaErrors_V(cudaFree(uvnodes));
    checkCudaErrors_V(cudaFree(dt));
    cudaDeviceReset();
  }
  u32 trim() {
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
    cudaMemcpyToSymbol(dipkeys, &sipkeys, sizeof(sipkeys));

    cudaDeviceSynchronize();
    float durationA, durationB;
    cudaEventRecord(start, NULL);
  

    cudaMemset(indexesA, 0, indexesSizeNA);
    for (u32 i=0; i < NA; i++) {
      FluffySeed4K<SEED_TPB, EDGES_A/NA><<<tp.seed.blocks, tp.seed.tpb>>>((uint4*)(bufferA+i*(sizeA/NA2)), indexesA+i*NX2, i*(NEDGES2/NA));
      if (abort) return false;
    }
  
#ifdef VERBOSE
    print_log("%d x Seed4K<<<%d,%d>>>\n", NA, tp.seed.blocks, tp.seed.tpb); // 1024x512
    cudaMemcpy(&nedges, indexesA, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print_log("round %d edges %d\n", 0, nedges);
#endif

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    cudaEventRecord(start, NULL);
  
    cudaMemset(indexesB, 0, indexesSizeNA);
    const u32 qB = sizeB/NA;
    const u32 qI = NX2 / NA;
    for (u32 i=0; i < NA; i++) {
      FluffyRound_A1<TRIM0_TPB, EDGES_A/NA, EDGES_B/NA><<<NX2/NA, TRIM0_TPB>>>((uint2*)bufferA, (uint4*)(bufferB+i*qB), indexesA, indexesB, i*qI);
      if (abort) return false;
    }

#ifdef VERBOSE
    print_log("Seeding completed in %.0f ms\n", durationA);
    print_log("Round_A1<<<%d,%d>>>\n", NX2/NA, TRIM0_TPB); // 1024x1024
    cudaMemcpy(&nedges, indexesB, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print_log("round %d edges %d\n", 1, nedges);
#endif

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(stop));
    // print_log("Round 0 completed in %.0f ms\n", durationB);
  
    cudaMemset(indexesA, 0, indexesSize);
    FluffyRound_A3<TRIM1_TPB, EDGES_B/NA, EDGES_B/2><<<NX2, TRIM1_TPB>>>((uint2*)bufferB, (uint2*)bufferA1, indexesB, indexesA);
    if (abort) return false;

#ifdef VERBOSE
    print_log("Round_A3<<<%d,%d>>>\n", NX2/NA, TRIM1_TPB); // 4096x1024
    cudaMemcpy(&nedges, indexesA, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print_log("round %d edges %d\n", 1, nedges);
#endif

    cudaMemset(indexesB, 0, indexesSize);
    FluffyRound_A2<TRIM_TPB, EDGES_B/2, EDGES_B/2><<<NX2, TRIM_TPB>>>((uint2*)bufferA1, (uint2*)bufferB, indexesA, indexesB, 2, 0);
    if (abort) return false;

#ifdef VERBOSE
    print_log("Round_A2<><<<%d,%d>>>\n", NX2, TRIM_TPB); // 4096x512
    cudaMemcpy(&nedges, indexesB, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print_log("round %d edges %d\n", 2, nedges);
#endif

    cudaMemset(indexesA, 0, indexesSize);
    FluffyRound_A2<TRIM_TPB, EDGES_B/2, EDGES_B/2><<<NX2, TRIM_TPB>>>((uint2*)bufferB, (uint2*)bufferA1, indexesB, indexesA, 3, 0);
    if (abort) return false;

    // cudaMemcpy(&nedges, indexesA, sizeof(u32), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // print_log("round %d edges %d\n", 3, nedges);

    cudaMemset(indexesB, 0, indexesSize);
    FluffyRound_A2<TRIM_TPB, EDGES_B/2, EDGES_B/2><<<NX2, TRIM_TPB>>>((uint2*)bufferA1, (uint2*)bufferB, indexesA, indexesB, 4, 0);
    if (abort) return false;

    // cudaMemcpy(&nedges, indexesB, sizeof(u32), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // print_log("round %d edges %d\n", 4, nedges);

    cudaMemset(indexesA, 0, indexesSize);
    FluffyRound_A2<TRIM_TPB, EDGES_B/2, EDGES_B/4><<<NX2, TRIM_TPB>>>((uint2*)bufferB, (uint2*)bufferA1, indexesB, indexesA, 5, 0);
    if (abort) return false;

    // cudaMemcpy(&nedges, indexesA, sizeof(u32), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // print_log("round %d edges %d\n", 5, nedges);

    cudaDeviceSynchronize();
  
    for (int round = 6; round < tp.ntrims; round += 2) {
      cudaMemset(indexesB, 0, indexesSize);
      FluffyRound_A2<TRIM_TPB, EDGES_B/4, EDGES_B/4><<<NX2, TRIM_TPB>>>((uint2*)bufferA1, (uint2*)bufferB, indexesA, indexesB, round, 0);
      if (abort) return false;

      cudaMemset(indexesA, 0, indexesSize);
      FluffyRound_A2<TRIM_TPB, EDGES_B/4, EDGES_B/4><<<NX2, TRIM_TPB>>>((uint2*)bufferB, (uint2*)bufferA1, indexesB, indexesA, round+1, 0);
      if (abort) return false;
    }
    
    cudaMemset(indexesB, 0, indexesSize);
    FluffyTail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((uint2*)bufferA1, (uint2*)bufferB, indexesA, indexesB);

    cudaMemcpy(&nedges, indexesB, sizeof(u32), cudaMemcpyDeviceToHost);
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
    cg.reset();
    for (u32 i = 0; i < nedges; i++)
      cg.add_compress_edge(edges[i].x, edges[i].y);
    for (u32 s = 0 ;s < cg.nsols; s++) {
      // print_log("Solution");
      for (u32 j = 0; j < PROOFSIZE; j++) {
        soledges[j] = edges[cg.sols[s][j]];
        // print_log(" (%x, %x)", soledges[j].x>>1, soledges[j].y>>1);
      }
      // print_log("\n");
      sols.resize(sols.size() + PROOFSIZE);
      cudaMemcpyToSymbol(recovery, soledges, sizeof(soledges));
      cudaMemset(trimmer.indexesA, 0, trimmer.indexesSize);
      FluffyRecovery<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>((u32 *)trimmer.indexesA);
      cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer.indexesA, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
      checkCudaErrors(cudaDeviceSynchronize());
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg.nonce_cmp);
    }
    return 0;
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
    findcycles(edges, nedges);
    time1 = timestamp(); timems2 = (time1 - time0) / 1000000;
    print_log("trim time %d ms findcycles edges %d time %d ms total %d ms\n", timems, nedges, timems2, timems+timems2);
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
  print_log("create_solver_ctx %d = %d\n", tp.recover.tpb, params->recovertpb);
  tp.recover.tpb = params->recovertpb;

  cudaDeviceProp prop;
  checkCudaErrors_N(cudaGetDeviceProperties(&prop, params->device));

  assert(tp.seed.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim0.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
  // assert(tp.tailblocks <= prop.threadDims[0]);
  assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
  assert(tp.recover.tpb <= prop.maxThreadsPerBlock);

  assert(tp.seed.blocks * tp.seed.tpb * EDGE_BLOCK_SIZE <= NEDGES2); // check THREADS_HAVE_EDGES
  assert(tp.recover.blocks * tp.recover.tpb * EDGE_BLOCK_SIZE <= NEDGES2); // check THREADS_HAVE_EDGES

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
  while ((c = getopt(argc, argv, "scd:h:m:n:r:U:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedblocks] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", EDGEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -y %d -Z %d -z %d\n", EDGEBITS, device, tp.ntrims, nonce, range, tp.seed.blocks, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
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
      case 'm': // ntrims         =       458;
        params.ntrims = atoi(optarg) & -2; // odd number of trimming rounds is treated same as 1 less anyway
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'U': // seed.blocks    =        64;
        params.genablocks = atoi(optarg);
        break;
      case 'y': // tail.tpb       =       256;
        params.tailtpb = atoi(optarg);
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

  print_log("Looking for %d-cycle on cuckarood%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, params.ntrims, NX);

  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 102400; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
