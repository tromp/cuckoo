// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2019 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <sys/time.h> // gettimeofday
#include "cuckaroo.hpp"
#include "graph.hpp"
#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint64_t u64; // save some typing

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

#ifndef XBITS
// assumes at least 2^17 bitpairs of shared mem (32 KB) on thread block
// #define XBITS ((EDGEBITS-17+1)/2)
// scrap that; too few buckets inhibits parallellism
#define XBITS 6
#endif

#define NODEBITS (EDGEBITS + 1)

const static u32 NX        = 1 << XBITS;
const static u32 NX2       = NX * NX;
const static u32 XMASK     = NX - 1;
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

__device__ u64 dipblock(const siphash_keys &keys, const word_t edge, u64 *buf) {
  diphash_state shs(keys);
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
__global__ void SeedA(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, int * __restrict__ indexes) {
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
  const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++) {
      u64 edge = buf[e] ^ last;
      u32 node0 = edge & EDGEMASK;
      u32 node1 = (edge >> 32) & EDGEMASK;
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
__global__ void SeedB(const uint2 * __restrict__ source, ulonglong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
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
  const int bucketEdges = min((int)sourceIndexes[group], (int)maxOut);
#ifdef SYNCBUG
  if (group==0x2f2 && lid==0) printf("group %03x size %d\n", group, bucketEdges);
#endif
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
      int cnt = min((int)atomicAdd(destinationIndexes + grp, nflush), (int)(maxOut - nflush));
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
      int cnt = min((int)atomicAdd(destinationIndexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      destination[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
    }
  }
}

__device__ __forceinline__  void Increase2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  u32 old = atomicOr(ecounters + word, mask) & mask;
  if (old)
    atomicOr(ecounters + word + NZ/32, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;

  return (ecounters[word + NZ/32] >> bit) & 1;
}

template<int maxIn, int maxOut>
__global__ void Round(const int round, const uint2 * __restrict__ source, uint2 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

  __shared__ u32 ecounters[COUNTERWORDS];

  for (int i = lid; i < COUNTERWORDS; i += dim)
    ecounters[i] = 0;
  __syncthreads();
  const int edgesInBucket = min(sourceIndexes[group], maxIn);
#ifdef SYNCBUG
  if (group==0xad6 && lid==0) printf("round %d group %03x edges %d\n", round, group, edgesInBucket);
#endif
  const int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = __ldg(&source[index]);
      if (null(edge)) continue;
      u32 node = endpoint(edge, round&1);
      Increase2bCounter(ecounters, node & ZMASK);
    }
  }
  __syncthreads();
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = __ldg(&source[index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(edge, round&1);
      if (Read2bCounter(ecounters, node0 & ZMASK)) {
        u32 node1 = endpoint(edge, (round&1)^1);
        const int bucket = node1 >> ZBITS;
        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), maxOut - 1);
        destination[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
      }
    }
  }
  // if (group==0&&lid==0) printf("round %d cnt(0,0) %d\n", round, sourceIndexes[0]);
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
  const int loops = NEDGES / nthreads;
  __shared__ u32 nonces[PROOFSIZE];
  u64 buf[EDGE_BLOCK_SIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
      u64 edge = buf[i] ^ last;
      u32 u = edge & EDGEMASK;
      u32 v = (edge >> 32) & EDGEMASK;
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

struct trimparams {
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    ntrims              =  176;
    genA.blocks         = 4096;
    genA.tpb            =  128;
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
    sizeA = ROW_EDGES_A * NX * sizeof(uint2);
    sizeB = ROW_EDGES_B * NX * sizeof(uint2);
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
  
    SeedA<EDGES_A><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
  
    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    if (abort) return false;
    cudaEventRecord(start, NULL);
  
    const u32 halfA = sizeA/2 / sizeof(ulonglong4);
    const u32 halfE = NX2 / 2;
    SeedB<EDGES_A><<<tp.genB.blocks/2, tp.genB.tpb>>>((const uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
    if (abort) return false;
    SeedB<EDGES_A><<<tp.genB.blocks/2, tp.genB.tpb>>>((const uint2 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    print_log("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
    if (abort) return false;
  
    cudaMemset(indexesE, 0, indexesSize);

    Round<EDGES_A, EDGES_B><<<tp.trim.blocks, tp.trim.tpb>>>(0, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
    cudaDeviceSynchronize();
    if (abort) return false;

    cudaMemset(indexesE2, 0, indexesSize);

    Round<EDGES_B, EDGES_B/2><<<tp.trim.blocks, tp.trim.tpb>>>(1, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296

    if (abort) return false;
    cudaMemset(indexesE, 0, indexesSize);
    Round<EDGES_B/2, EDGES_A/4><<<tp.trim.blocks, tp.trim.tpb>>>(2, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .176
    if (abort) return false;
    cudaMemset(indexesE2, 0, indexesSize);
    Round<EDGES_A/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(3, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .117
  
    cudaDeviceSynchronize();
  
    for (int round = 4; round < tp.ntrims; round += 2) {
      if (abort) return false;
      cudaMemset(indexesE, 0, indexesSize);
      Round<EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(round, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
      if (abort) return false;
      cudaMemset(indexesE2, 0, indexesSize);
      Round<EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(round+1, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2);
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
    print_log("%d edges after trimming\n", nedges);
    if (!nedges)
      return 0;
    if (nedges > MAXEDGES) {
      print_log("OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
      return 0;
    }
    cudaMemcpy(edges, trimmer.bufferB, sizeof(uint2[nedges]), cudaMemcpyDeviceToHost);
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    gettimeofday(&time0, 0);
    findcycles(edges, nedges);
    gettimeofday(&time1, 0);
    timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    print_log("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
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

  assert(tp.genA.blocks * tp.genA.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES
  assert(tp.recover.blocks * tp.recover.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES
  assert(tp.genA.tpb / NX <= FLUSHA); // check ROWS_LIMIT_LOSSES
  assert(tp.genB.tpb / NX <= FLUSHB); // check COLS_LIMIT_LOSSES

  cudaSetDevice(params->device);
  if (!params->cpuload)
    checkCudaErrors_N(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

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
  params->genablocks = tp.genA.blocks;
  params->genatpb = tp.genA.tpb;
  params->genbtpb = tp.genB.tpb;
  params->trimtpb = tp.trim.tpb;
  params->tailtpb = tp.tail.tpb;
  params->recoverblocks = tp.recover.blocks;
  params->recovertpb = tp.recover.tpb;
  params->cpuload = true;
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
  while ((c = getopt(argc, argv, "scb:d:h:k:m:n:r:U:u:v:w:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-E 0-2] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedAblocks] [-u seedAthreads] [-v seedBthreads] [-w Trimthreads] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", NODEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -u %d -v %d -w %d -y %d -Z %d -z %d\n", NODEBITS, device, tp.ntrims, nonce, range, tp.genA.blocks, tp.genA.tpb, tp.genB.tpb, tp.trim.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'c':
        params.cpuload = false;
        break;
      case 'd':
        params.device = atoi(optarg);
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
  // cudaSetDevice(device);

  print_log("Looking for %d-cycle on cuckaroo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, params.ntrims, NX);

  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 10240; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
