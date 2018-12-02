// Cuckatoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2019 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <sys/time.h> // gettimeofday
#include "cuckatoo.h"
#include "graph.hpp"
#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"

typedef uint8_t u8;
typedef uint16_t u16;

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#define IDXSHIFT 12
#endif

const static u32 MAXEDGES = NEDGES >> IDXSHIFT;

typedef uint64_t u64; // save some typing

#ifndef XBITS
// assumes at least 2^18 bits of shared mem (32 KB) on thread block
// #define XBITS ((EDGEBITS-18+1)/2)
// scrap that; too few buckets inhibits parallellism
#define XBITS 6
#endif

#define NODEBITS (EDGEBITS + 1)

const static u32 NX        = 1 << XBITS;
const static u32 XMASK     = NX - 1;
const static u32 NX2       = NX * NX;
const static u32 YBITS     = XBITS;
const static u32 NY        = 1 << YBITS;
const static u32 YZBITS    = EDGEBITS - XBITS;
const static u32 ZBITS     = YZBITS - YBITS;
const static u32 NZ        = 1 << ZBITS;
const static u32 ZMASK     = NZ - 1;

#ifndef EPS_A
#define EPS_A 133/128
#endif
#ifndef EPS_B
#define EPS_B 85/128
#endif

const static u32 EDGES_A = NZ * EPS_A;
const static u32 EDGES_B = NZ * EPS_B;

const static u32 ROW_EDGES_A = EDGES_A * NY;
const static u32 ROW_EDGES_B = EDGES_B * NX;

__constant__ uint2 recoveredges[PROOFSIZE];
__constant__ uint2 e0 = {0,0};

__device__ __forceinline__ ulonglong4 Pack8(const u32 e0, const u32 e1, const u32 e2, const u32 e3, const u32 e4, const u32 e5, const u32 e6, const u32 e7) {
  return make_ulonglong4((u64)e0<<32|e1, (u64)e2<<32|e3, (u64)e4<<32|e5, (u64)e6<<32|e7);
}

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

template<int maxOut, typename EdgeOut>
__global__ void SeedA(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, int * __restrict__ indexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = gridDim.x * dim;
  const int FLUSHA2 = 2*FLUSHA;

  __shared__ EdgeOut tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(EdgeOut);
  __shared__ int counters[NX];

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  __syncthreads();

  const int col = group % NX;
  const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int i = 0; i < loops; i++) {
    u32 nonce = gid * loops + i;
    u32 node1, node0 = dipnode(sipkeys, (u64)nonce, 0);
    if (sizeof(EdgeOut) == sizeof(uint2)) {
      node1 = dipnode(sipkeys, (u64)nonce, 1);
    }
    int row = node0 >> YZBITS;
    int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
    tmp[row][counter] = make_Edge(nonce, tmp[0][0], node0, node1);
    __syncthreads();
    if (counter == FLUSHA-1) {
      int localIdx = min(FLUSHA2, counters[row]);
      int newCount = localIdx % FLUSHA;
      int nflush = localIdx - newCount;
      int cnt = min((int)atomicAdd(indexes + row * NX + col, nflush), (int)(maxOut - nflush));
      for (int i = 0; i < nflush; i += TMPPERLL4) {
        buffer[((u64)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
      }
      for (int t = 0; t < newCount; t++) {
        tmp[row][t] = tmp[row][t + nflush];
      }
      counters[row] = newCount;
    }
    __syncthreads();
  }
  EdgeOut zero = make_Edge(0, tmp[0][0], 0, 0);
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(indexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      buffer[((u64)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
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

template<int maxOut, typename EdgeOut>
__global__ void SeedB(const siphash_keys &sipkeys, const EdgeOut * __restrict__ source, ulonglong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int FLUSHB2 = 2 * FLUSHB;

  __shared__ EdgeOut tmp[NX][FLUSHB2];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(EdgeOut);
  __shared__ int counters[NX];

  for (int col = lid; col < NX; col += dim)
    counters[col] = 0;
  __syncthreads();
  const int row = group / NX;
  const int bucketEdges = min((int)sourceIndexes[group], (int)maxOut);
  const int loops = (bucketEdges + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    int col;
    int counter = 0;
    const int edgeIndex = loop * dim + lid;
    if (edgeIndex < bucketEdges) {
      const int index = group * maxOut + edgeIndex;
      EdgeOut edge = __ldg(&source[index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(sipkeys, edge, 0);
      col = (node0 >> ZBITS) & XMASK;
      counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1)); // assuming COLS_LIMIT_LOSSES checked
      tmp[col][counter] = edge;
    }
    __syncthreads();
    if (counter == FLUSHB-1) {
      int localIdx = min(FLUSHB2, counters[col]);
      int newCount = localIdx % FLUSHB;
      int nflush = localIdx - newCount;
      int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, nflush), (int)(maxOut - nflush));
      for (int i = 0; i < nflush; i += TMPPERLL4)
        destination[((u64)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
      for (int t = 0; t < newCount; t++) {
        tmp[col][t] = tmp[col][t + nflush];
      }
      counters[col] = newCount;
    }
    __syncthreads(); 
  }
  EdgeOut zero = make_Edge(0, tmp[0][0], 0, 0);
  for (int col = lid; col < NX; col += dim) {
    int localIdx = min(FLUSHB2, counters[col]);
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[col][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      destination[((u64)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
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

__device__ uint2 make_Edge(const u32 nonce, const uint2 dummy, const u32 node0, const u32 node1) {
   return make_uint2(node0, node1);
}

__device__ uint2 make_Edge(const uint2 edge, const uint2 dummy, const u32 node0, const u32 node1) {
   return edge;
}

__device__ u32 make_Edge(const u32 nonce, const u32 dummy, const u32 node0, const u32 node1) {
   return nonce;
}

template <typename Edge> u32 __device__ endpoint(const siphash_keys &sipkeys, Edge e, int uorv);

__device__ u32 endpoint(const siphash_keys &sipkeys, u32 nonce, int uorv) {
  return dipnode(sipkeys, nonce, uorv);
}

__device__ u32 endpoint(const siphash_keys &sipkeys, uint2 nodes, int uorv) {
  return uorv ? nodes.y : nodes.x;
}

template<int maxIn, typename EdgeIn, int maxOut, typename EdgeOut>
__global__ void Round(const int round, const siphash_keys &sipkeys, const EdgeIn * __restrict__ source, EdgeOut * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const static int BITMAPWORDS = NZ / 32; // 32-bit words in bitmap

  __shared__ u32 ebitmap[BITMAPWORDS];

  for (int i = lid; i < BITMAPWORDS; i += dim)
    ebitmap[i] = 0;
  __syncthreads();
  const int edgesInBucket = min(sourceIndexes[group], maxIn);
  const int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      EdgeIn edge = __ldg(&source[index]);
      if (null(edge)) continue;
      u32 node = endpoint(sipkeys, edge, round&1);
      bitmapset(ebitmap, node & ZMASK);
    }
  }
  __syncthreads();
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      EdgeIn edge = __ldg(&source[index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(sipkeys, edge, round&1);
      if (bitmaptest(ebitmap, (node0 & ZMASK) ^ 1)) {
        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
        const int bucket = node1 >> ZBITS;
        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), maxOut - 1);
        destination[bucket * maxOut + bktIdx] = (round&1) ? make_Edge(edge, *destination, node1, node0)
                                                          : make_Edge(edge, *destination, node0, node1);
      }
    }
  }
}

template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const int *sourceIndexes, int *destinationIndexes) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  int myEdges = sourceIndexes[group];
  __shared__ int destIdx;

  if (lid == 0)
    destIdx = atomicAdd(destinationIndexes, myEdges);
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
    snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d\0", device_id, cudaGetErrorString(code), file, line);
    cudaDeviceReset();
    if (abort) return code;
  }
  return code;
}

__global__ void Recovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __shared__ u32 nonces[PROOFSIZE];
  
  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int i = 0; i < loops; i++) {
    u64 nonce = gid * loops + i;
    u64 u = dipnode(sipkeys, nonce, 0);
    u64 v = dipnode(sipkeys, nonce, 1);
    for (int i = 0; i < PROOFSIZE; i++) {
      if (recoveredges[i].x == u && recoveredges[i].y == v)
        nonces[i] = nonce;
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

struct trimparams {
  u16 expand;
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    expand              =    0;
    ntrims              =  176;
    genA.blocks         = 4096;
    genA.tpb            =  256;
    genB.blocks         =  NX2;
    genB.tpb            =  128;
    trim.blocks         =  NX2;
    trim.tpb            =  512;
    tail.blocks         =  NX2;
    tail.tpb            = 1024;
    recover.blocks      = 1024;
    recover.tpb         = 1024;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  size_t sizeA, sizeB;
  const size_t indexesSize = NX * NY * sizeof(u32);
  ulonglong4 *bufferA;
  ulonglong4 *bufferB;
  ulonglong4 *bufferAB;
  int *indexesE;
  int *indexesE2;
  u32 hostA[NX * NY];
  u32 *uvnodes;
  siphash_keys sipkeys, *dipkeys;
  bool abort;
  bool initsuccess = false;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors_V(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
    checkCudaErrors_V(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
    checkCudaErrors_V(cudaMalloc((void**)&indexesE, indexesSize));
    checkCudaErrors_V(cudaMalloc((void**)&indexesE2, indexesSize));
    sizeA = ROW_EDGES_A * NX * (tp.expand > 0 ? sizeof(u32) : sizeof(uint2));
    sizeB = ROW_EDGES_B * NX * (tp.expand > 1 ? sizeof(u32) : sizeof(uint2));
    const size_t bufferSize = sizeA + sizeB;
    checkCudaErrors_V(cudaMalloc((void**)&bufferA, bufferSize));
    bufferB  = bufferA + sizeA / sizeof(ulonglong4);
    bufferAB = bufferA + sizeB / sizeof(ulonglong4);
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    initsuccess = true;
  }
  u64 globalbytes() const {
    return (sizeA+sizeB) + 2 * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer);
  }
  ~edgetrimmer() {
    checkCudaErrors_V(cudaFree(bufferA));
    checkCudaErrors_V(cudaFree(indexesE2));
    checkCudaErrors_V(cudaFree(indexesE));
    checkCudaErrors_V(cudaFree(dipkeys));
    checkCudaErrors_V(cudaFree(uvnodes));
    checkCudaErrors_V(cudaFree(dt));
    cudaDeviceReset();
  }
  u32 trim() {
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
  
    cudaMemset(indexesE, 0, indexesSize);
    cudaMemset(indexesE2, 0, indexesSize);
    cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);
  
    cudaDeviceSynchronize();
    float durationA, durationB;
    cudaEventRecord(start, NULL);
  
    assert(tp.genA.blocks * tp.genA.tpb <= NEDGES); // check THREADS_HAVE_EDGES
    assert(tp.genA.tpb / NX <= FLUSHA); // check ROWS_LIMIT_LOSSES
    if (tp.expand == 0) {
      SeedA<EDGES_A, uint2><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
    } else
      SeedA<EDGES_A,   u32><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
  
    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    if (abort) return false;
    cudaEventRecord(start, NULL);
  
    const u32 halfA = sizeA/2 / sizeof(ulonglong4);
    const u32 halfE = NX2 / 2;
    assert(tp.genA.tpb / NX <= FLUSHA); // check COLS_LIMIT_LOSSES
    if (tp.expand == 0) {
      SeedB<EDGES_A, uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
      if (abort) return false;
      SeedB<EDGES_A, uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const uint2 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
    } else {
      SeedB<EDGES_A,   u32><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const   u32 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
      if (abort) return false;
      SeedB<EDGES_A,   u32><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const   u32 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
    }

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    print_log("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
    if (abort) return false;
  
    cudaMemset(indexesE, 0, indexesSize);

    if (tp.expand == 0)
      Round<EDGES_A, uint2, EDGES_B, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
    else if (tp.expand == 1)
      Round<EDGES_A,   u32, EDGES_B, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const   u32 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
    else // tp.expand == 2
      Round<EDGES_A,   u32, EDGES_B,   u32><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const   u32 *)bufferA, (  u32 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
    if (abort) return false;

    cudaMemset(indexesE2, 0, indexesSize);

    if (tp.expand < 2)
      Round<EDGES_B, uint2, EDGES_B/2, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(1, *dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296
    else
      Round<EDGES_B,   u32, EDGES_B/2, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(1, *dipkeys, (const   u32 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296

    if (abort) return false;
    cudaMemset(indexesE, 0, indexesSize);
    Round<EDGES_B/2, uint2, EDGES_A/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(2, *dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .176
    if (abort) return false;
    cudaMemset(indexesE2, 0, indexesSize);
    Round<EDGES_A/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(3, *dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .117
  
    cudaDeviceSynchronize();
  
    for (int round = 4; round < tp.ntrims; round += 2) {
      if (abort) return false;
      cudaMemset(indexesE, 0, indexesSize);
      Round<EDGES_B/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round, *dipkeys,  (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
      if (abort) return false;
      cudaMemset(indexesE2, 0, indexesSize);
      Round<EDGES_B/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round+1, *dipkeys,  (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2);
    }
    
    if (abort) return false;
    cudaMemset(indexesE, 0, indexesSize);
    cudaDeviceSynchronize();
  
    Tail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
    cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return hostA[0];
  }
};

struct solver_ctx {
  edgetrimmer trimmer;
  bool mutatenonce;
  uint2 *edges;
  graph<word_t> cg;
  uint2 soledges[PROOFSIZE];
  std::vector<u32> sols; // concatenation of all proof's indices

  solver_ctx(const trimparams tp, bool mutate_nonce) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
    edges   = new uint2[MAXEDGES];
    mutatenonce = mutate_nonce;
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    if (mutatenonce) {
      ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    }
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
	// print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
      }
      // print_log("\n");
      sols.resize(sols.size() + PROOFSIZE);
      cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges));
      cudaMemset(trimmer.indexesE2, 0, trimmer.indexesSize);
      Recovery<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>(*trimmer.dipkeys, trimmer.bufferA, (int *)trimmer.indexesE2);
      cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer.indexesE2, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
      checkCudaErrors(cudaDeviceSynchronize());
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg.nonce_cmp);
    }
    return 0;
  }

  int solve() {
    u32 timems,timems2;
    struct timeval time0, time1;

    trimmer.abort = false;
    gettimeofday(&time0, 0);
    u32 nedges = trimmer.trim();
    if (!nedges)
      return 0;
    if (nedges > MAXEDGES) {
      print_log("OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
    }
    cudaMemcpy(edges, trimmer.bufferB, nedges * 8, cudaMemcpyDeviceToHost);
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    gettimeofday(&time0, 0);
    findcycles(edges, nedges);
    gettimeofday(&time1, 0);
    timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    print_log("%d trims %d ms %d edges %d ms total %d ms\n", trimmer.tp.ntrims, timems, nedges, timems2, timems+timems2);
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
      int pow_rc = verify(prf, &ctx->trimmer.sipkeys);
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
  tp.expand = params->expand;
  tp.genA.blocks = params->genablocks;
  tp.genA.tpb = params->genatpb;
  tp.genB.tpb = params->genbtpb;
  tp.trim.tpb = params->trimtpb;
  tp.tail.tpb = params->tailtpb;
  tp.recover.blocks = params->recoverblocks;
  tp.recover.tpb = params->recovertpb;

  cudaDeviceProp prop;
  checkCudaErrors_N(cudaGetDeviceProperties(&prop, params->device));

  assert(tp.genA.tpb <= prop.maxThreadsPerBlock);
  assert(tp.genB.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
  // assert(tp.tailblocks <= prop.threadDims[0]);
  assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
  assert(tp.recover.tpb <= prop.maxThreadsPerBlock);

  cudaSetDevice(params->device);

  SolverCtx* ctx = new SolverCtx(tp, params->mutate_nonce);

  return ctx;
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
  params->expand = tp.expand;
  params->genablocks = tp.genA.blocks;
  params->genatpb = tp.genA.tpb;
  params->genbtpb = tp.genB.tpb;
  params->trimtpb = tp.trim.tpb;
  params->tailtpb = tp.tail.tpb;
  params->recoverblocks = tp.recover.blocks;
  params->recovertpb = tp.recover.tpb;
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
  while ((c = getopt(argc, argv, "sb:d:E:h:k:m:n:r:U:u:v:w:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-d device] [-E 0-2] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedAblocks] [-u seedAthreads] [-v seedBthreads] [-w Trimthreads] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", NODEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -E %d -h \"\" -m %d -n %d -r %d -U %d -u %d -v %d -w %d -y %d -Z %d -z %d\n", NODEBITS, device, tp.expand, tp.ntrims, nonce, range, tp.genA.blocks, tp.genA.tpb, tp.genB.tpb, tp.trim.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'd':
        params.device = atoi(optarg);
        break;
      case 'E':
        params.expand = atoi(optarg);
        assert(params.expand <= 2);
        break;
      case 'h':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        params.ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'U':
        params.genablocks = atoi(optarg);
        break;
      case 'u':
        params.genatpb = atoi(optarg);
        break;
      case 'v':
        params.genbtpb = atoi(optarg);
        break;
      case 'w':
        params.trimtpb = atoi(optarg);
        break;
      case 'y':
        params.tailtpb = atoi(optarg);
        break;
      case 'Z':
        params.recoverblocks = atoi(optarg);
        break;
      case 'z':
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
  for (dunit=0; dbytes >= 10240; dbytes>>=10,dunit++) ;
  print_log("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);

  print_log("Looking for %d-cycle on cuckatoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, NX);

  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 10240; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
