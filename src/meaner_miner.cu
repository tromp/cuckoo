// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura - photon
// This CUDA part of Theta optimized miner is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <sys/time.h> // gettimeofday
#include "cuckoo.h"
#include "siphash.cuh"
#include "blake2.h"

typedef uint8_t u8;
typedef uint16_t u16;

typedef u32 node_t;
typedef u64 nonce_t;

#ifndef XBITS
#define XBITS 6
#endif

#define NODEBITS (EDGEBITS + 1)
#define NNODES ((node_t)1 << NODEBITS)
#define NODEMASK (NNODES - 1)

#define YBITS XBITS
#define ZBITS (EDGEBITS - XBITS - YBITS)
const static u32 NX        = 1 << XBITS;
const static u32 NX2       = NX * NX;
const static u32 XMASK     = NX - 1;
const static u32 X2MASK    = NX2 - 1;
const static u32 NY        = 1 << YBITS;
const static u32 NZ        = 1 << ZBITS;

#define DUCK_SIZE_A 130LL
#define DUCK_SIZE_B 83LL

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#ifndef CTHREADS
#define CTHREADS 1024
#endif

#ifndef BKTGRAN
#define BKTGRAN 32
#endif

#define TMPDEPTH 16
#define TMPDEPTH1 (TMPDEPTH -1)

__device__ ulonglong4 Pack4edges(const uint2 e0, const  uint2 e1, const  uint2 e2, const  uint2 e3) {
  u64 r0 = ((u64)e0.y << 32) | (u64)e0.x;
  u64 r1 = ((u64)e1.y << 32) | (u64)e1.x;
  u64 r2 = ((u64)e2.y << 32) | (u64)e2.x;
  u64 r3 = ((u64)e3.y << 32) | (u64)e3.x;
  return make_ulonglong4(r0, r1, r2, r3);
}

__constant__ uint2 recoveredges[PROOFSIZE];
__constant__ uint2 e0 = {0,0};


__global__  void FluffySeed2A(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, int * __restrict__ indexes) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  __shared__ uint2 tmp[NX][TMPDEPTH];
  __shared__ int counters[NX];

  counters[lid] = 0;
  __syncthreads();
  const int loops = NEDGES / nthreads;
  for (int i = 0; i < loops; i++) {
    u64 nonce = gid * loops + i;
    uint2 hash;
    hash.x = dipnode(sipkeys, nonce, 0);
    hash.y = dipnode(sipkeys, nonce, 1);
    int bucket = hash.x & XMASK;
    __syncthreads();
    int counter = min((int)atomicAdd(counters + bucket, 1), (int)TMPDEPTH1);
    tmp[bucket][counter] = hash;
    __syncthreads();
    int localIdx = min(TMPDEPTH, counters[lid]);
    if (localIdx >= 8) {
      int newCount = localIdx - 8;
      counters[lid] = newCount;
      int cnt = min((int)atomicAdd(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));
      buffer[(lid * DUCK_A_EDGES_64 + cnt)     / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
      buffer[(lid * DUCK_A_EDGES_64 + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
      for (int t = 0; t < newCount; t++) {
        tmp[lid][t] = tmp[lid][t + 8];
      }
    }
  }
  __syncthreads();
  int localIdx = min(TMPDEPTH, counters[lid]);
  if (localIdx > 0) {
    int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
    buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][0],
      localIdx > 1 ? tmp[lid][1] : e0, localIdx > 2 ? tmp[lid][2] : e0, localIdx > 3 ? tmp[lid][3] : e0);
  }
  if (localIdx > 4) {
    int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
    buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][4],
      localIdx > 5 ? tmp[lid][5] : e0, localIdx > 6 ? tmp[lid][6] : e0, localIdx > 7 ? tmp[lid][7] : e0);
  }
}

__global__  void FluffySeed2B(const uint2 * __restrict__ source, ulonglong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  __shared__ uint2 tmp[NX][TMPDEPTH];
  __shared__ int counters[NX];

  counters[lid] = 0;
  __syncthreads();
  const int myBucket = group / BKTGRAN;
  const int microBlockNo = group % BKTGRAN;
  const int bucketEdges = min(sourceIndexes[myBucket], (int)DUCK_A_EDGES_64);
  const int microBlockEdgesCount = DUCK_A_EDGES_64 / BKTGRAN;
  const int loops = microBlockEdgesCount / NX;
  for (int i = 0; i < loops; i++) {
    int edgeIndex = microBlockNo * microBlockEdgesCount + NX*i + lid;
    if (edgeIndex < bucketEdges) {
      uint2 edge = source[myBucket*DUCK_A_EDGES_64 + edgeIndex];
      if (edge.x == 0 && edge.y == 0) continue;
      int bucket = (edge.x >> XBITS) & XMASK;
      __syncthreads();
      int counter = min((int)atomicAdd(counters + bucket, 1), (int)TMPDEPTH1);
      tmp[bucket][counter] = edge;
      __syncthreads();
      int localIdx = min(TMPDEPTH, counters[lid]);
      if (localIdx >= 8) {
        int newCount = localIdx - 8;
        counters[lid] = newCount;
        int cnt = min((int)atomicAdd(destinationIndexes + myBucket*64 + lid, 8), (int)(DUCK_A_EDGES - 8));
        destination[((myBucket * NX + lid) * DUCK_A_EDGES + cnt)     / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
        destination[((myBucket * NX + lid) * DUCK_A_EDGES + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
        for (int t = 0; t < newCount; t++) {
          tmp[lid][t] = tmp[lid][t + 8];
        }
      }
    }
  }
  __syncthreads();
  int localIdx = min(TMPDEPTH, counters[lid]);
  if (localIdx > 0) {
    int cnt = min((int)atomicAdd(destinationIndexes + myBucket*64 + lid, 4), (int)(DUCK_A_EDGES - 4));
    destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][0],
      localIdx > 1 ? tmp[lid][1] : e0, localIdx > 2 ? tmp[lid][2] : e0, localIdx > 3 ? tmp[lid][3] : e0);
  }
  if (localIdx > 4) {
    int cnt = min((int)atomicAdd(destinationIndexes + myBucket*64 + lid, 4), (int)(DUCK_A_EDGES - 4));
    destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][4],
      localIdx > 5 ? tmp[lid][5] : e0, localIdx > 6 ? tmp[lid][6] : e0, localIdx > 7 ? tmp[lid][7] : e0);
  }
}

__device__ __forceinline__  void Increase2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  u32 old = atomicOr(ecounters + word, mask) & mask;
  if (old)
    atomicOr(ecounters + word + 4096, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  return (ecounters[word + 4096] & mask) != 0;
}

template<int bktInSize, int bktOutSize>
__global__  void FluffyRound(const uint2 * __restrict__ source, uint2 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

  __shared__ u32 ecounters[COUNTERWORDS];

  const int edgesInBucket = min(sourceIndexes[group], bktInSize);
  const int loops = (edgesInBucket + CTHREADS-1) / CTHREADS;

  for (int i = 0; i < COUNTERWORDS / CTHREADS; i++)
    ecounters[lid + CTHREADS * i] = 0;
  __syncthreads();
  for (int i = 0; i < loops; i++) {
    const int lindex = i * CTHREADS + lid;
    if (lindex < edgesInBucket) {
      const int index = bktInSize * group + lindex;
      uint2 edge = __ldg(&source[index]);
      if (edge.x == 0 && edge.y == 0) continue;
      Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
    }
  }
  __syncthreads();
  for (int i = loops-1; i >= 0; i--) {
    const int lindex = i * CTHREADS + lid;
    if (lindex < edgesInBucket) {
      const int index = bktInSize * group + lindex;
      uint2 edge = __ldg(&source[index]);
      if (edge.x == 0 && edge.y == 0) continue;
      if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12)) {
        const int bucket = edge.y & X2MASK;
        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
        destination[bucket * bktOutSize + bktIdx] = make_uint2(edge.y, edge.x);
      }
    }
  }
}

__global__ void FluffyTail(const uint2 *source, uint2 *destination, const int *sourceIndexes, int *destinationIndexes) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  int myEdges = sourceIndexes[group];
  __shared__ int destIdx;

  if (lid == 0)
    destIdx = atomicAdd(destinationIndexes, myEdges);
  __syncthreads();
  if (lid < myEdges)
    destination[destIdx + lid] = source[group * DUCK_B_EDGES/4 + lid];
}

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__  void FluffyRecovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
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
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;
  u16 reportcount;
  u16 reportrounds;

  trimparams() {
    ntrims              =  176;
    genA.blocks         =  512;
    genA.tpb            =   NX;
    genB.blocks         = NX2*BKTGRAN/NX;
    genB.tpb            =   NX;
    trim.blocks         =  NX2;
    trim.tpb            = CTHREADS;
    tail.blocks         =  NX2;
    tail.tpb            = 1024; // needs to exceed #FINAL EDGES / NX2
    recover.blocks      = 1024;
    recover.tpb         = 1024;
    reportcount         =    1;
    reportrounds        =    0;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  const size_t bufferSize  = DUCK_A_EDGES_64 * NX * sizeof(uint2);
  const size_t bufferSize2 = DUCK_A_EDGES_64 * NX * sizeof(uint2);
  const size_t indexesSize = NX * NY * sizeof(u32);
  int *bufferA;
  int *bufferB;
  int *indexesE;
  int *indexesE2;
  u32 hostA[NX * NY];
  u32 *uvnodes;
  proof sol;
  siphash_keys sipkeys, *dipkeys;

  edgetrimmer(const trimparams _tp) {
    tp = _tp;
    checkCudaErrors(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    checkCudaErrors(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
    checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));

    checkCudaErrors(cudaMalloc((void**)&bufferA, bufferSize));
    checkCudaErrors(cudaMalloc((void**)&bufferB, bufferSize2));
    checkCudaErrors(cudaMalloc((void**)&indexesE, indexesSize));
    checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));
  }
  u64 sharedbytes() const {
    return bufferSize + bufferSize2 + 2 * indexesSize;
  }
  ~edgetrimmer() {
    cudaFree(bufferA);
    cudaFree(bufferB);
    cudaFree(indexesE);
    cudaFree(indexesE2);
    cudaDeviceReset();
  }
  u32 trim() {
    cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
    // cudaEvent_t start, stop;
    cudaEvent_t startall, stopall;
    checkCudaErrors(cudaEventCreate(&startall)); checkCudaErrors(cudaEventCreate(&stopall));
    // checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
  
    cudaMemset(indexesE, 0, indexesSize);
    cudaMemset(indexesE2, 0, indexesSize);
    cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);
  
    cudaDeviceSynchronize();
    // float durationA, durationB;
    // cudaEventRecord(start, NULL);
  
    FluffySeed2A<<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, (ulonglong4 *)bufferA, (int *)indexesE);
  
    checkCudaErrors(cudaDeviceSynchronize()); // cudaEventRecord(stop, NULL);
    // cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    // cudaEventRecord(start, NULL);
  
    FluffySeed2B<<<tp.genB.blocks, tp.genB.tpb>>>((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE, (int *)indexesE2);
    checkCudaErrors(cudaDeviceSynchronize()); // cudaEventRecord(stop, NULL);
    // cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    // printf("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
  
    cudaMemset(indexesE, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
    cudaMemset(indexesE2, 0, indexesSize);
    FluffyRound<DUCK_B_EDGES, DUCK_A_EDGES/2><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
    cudaMemset(indexesE, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES/2, DUCK_A_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
    cudaMemset(indexesE2, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
  
    cudaDeviceSynchronize();
  
    for (int i = 4; i < tp.ntrims; i+=2) {
      cudaMemset(indexesE, 0, indexesSize);
      FluffyRound<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
      cudaMemset(indexesE2, 0, indexesSize);
      FluffyRound<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
    }
    
    cudaMemset(indexesE, 0, indexesSize);
    cudaDeviceSynchronize();
  
    FluffyTail<<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
    cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return hostA[0];
  }
};

#define IDXSHIFT 10
#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by NODEBITS
#define KEYBITS (64-NODEBITS)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

class cuckoo_hash {
public:
  u64 *cuckoo;

  cuckoo_hash() {
    cuckoo = new u64[CUCKOO_SIZE];
  }
  ~cuckoo_hash() {
    delete[] cuckoo;
  }
  void set(node_t u, node_t v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
        return;
      }
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 cu = cuckoo[ui];
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (node_t)(cu & NODEMASK);
      }
    }
  }
};

const static u32 MAXPATHLEN = 8 << ((NODEBITS+2)/3);

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

const static u32 MAXEDGES = 0x20000;

struct solver_ctx {
  edgetrimmer *trimmer;
  uint2 *edges;
  cuckoo_hash *cuckoo;
  uint2 soledges[PROOFSIZE];
  std::vector<u32> sols; // concatenation of all proof's indices
  u32 us[MAXPATHLEN];
  u32 vs[MAXPATHLEN];

  solver_ctx(const trimparams tp) {
    trimmer = new edgetrimmer(tp);
    edges   = new uint2[MAXEDGES];
    cuckoo  = new cuckoo_hash();
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer->sipkeys);
    sols.clear();
  }
  ~solver_ctx() {
    delete cuckoo;
    delete[] edges;
    delete trimmer;
  }

  void recordedge(const u32 i, const u32 u2, const u32 v2) {
    soledges[i].x = u2/2;
    soledges[i].y = v2/2;
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
    cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges));
    cudaMemset(trimmer->indexesE2, 0, trimmer->indexesSize);
    FluffyRecovery<<<trimmer->tp.recover.blocks, trimmer->tp.recover.tpb>>>(*trimmer->dipkeys, (ulonglong4 *)trimmer->bufferA, (int *)trimmer->indexesE2);
    cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer->indexesE2, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());
    qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), nonce_cmp);
  }

  u32 path(u32 u, u32 *us) {
    u32 nu, u0 = u;
    for (nu = 0; u; u = (*cuckoo)[u]) {
      if (nu >= MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (~nu) {
          printf("illegal %4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
          exit(0);
        }
        printf("maximum path length exceeded\n");
        return 0; // happens once in a million runs or so; signal trouble
      }
      us[nu++] = u;
    }
    return nu;
  }

  void addedge(uint2 edge) {
    const u32 u0 = edge.x << 1, v0 = (edge.y << 1) | 1;
    if (u0) {
      u32 nu = path(u0, us), nv = path(v0, vs);
      if (!nu-- || !nv--)
        return; // drop edge causing trouble
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
          cuckoo->set(us[nu+1], us[nu]);
        cuckoo->set(u0, v0);
      } else {
        while (nv--)
          cuckoo->set(vs[nv+1], vs[nv]);
        cuckoo->set(v0, u0);
      }
    }
  }

  void findcycles(uint2 *edges, u32 nedges) {
    memset(cuckoo->cuckoo, 0, CUCKOO_SIZE * sizeof(u64));
    for (u32 i = 0; i < nedges; i++)
      addedge(edges[i]);
  }

  int solve() {
    u32 timems,timems2;
    struct timeval time0, time1;

    gettimeofday(&time0, 0);
    u32 nedges = trimmer->trim();
    assert(nedges <= MAXEDGES);
    cudaMemcpy(edges, trimmer->bufferA, nedges * 8, cudaMemcpyDeviceToHost);
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    gettimeofday(&time0, 0);
    findcycles(edges, nedges);
    gettimeofday(&time1, 0);
    timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
    return sols.size() / PROOFSIZE;
  }
};

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt(argc, argv, "sb:c:d:h:k:m:n:r:V:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        printf("SYNOPSIS\n  cuda30 [-d device] [-h hexheader] [-k rounds [-c count]] [-m trims] [-n nonce] [-r range] [-V blocks] [-y threads] [-Z blocks] [-z threads]\n");
        printf("DEFAULTS\n  cuda30 -d %d -h \"\" -k %d -c %d -m %d -n %d -r %d -V %d -y %d -Z %d -z %d\n", device, tp.reportrounds, tp.reportcount, tp.ntrims, nonce, range, tp.genA.blocks, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'd':
        device = atoi(optarg);
        break;
      case 'k':
        tp.reportrounds = atoi(optarg);
        break;
      case 'c':
        tp.reportcount = atoi(optarg);
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
        tp.ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'V':
        tp.genA.blocks = atoi(optarg);
        break;
      case 'y':
        tp.tail.tpb = atoi(optarg);
        break;
      case 'Z':
        tp.recover.blocks = atoi(optarg);
        break;
      case 'z':
        tp.recover.tpb = atoi(optarg);
        break;
    }
  }
  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  assert(device < nDevices);
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  assert(tp.genA.tpb <= prop.maxThreadsPerBlock);
  assert(tp.genB.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
  // assert(tp.tailblocks <= prop.threadDims[0]);
  assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
  assert(tp.recover.tpb <= prop.maxThreadsPerBlock);
  u64 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 10240; dbytes>>=10,dunit++) ;
  printf("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
  cudaSetDevice(device);

  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, NX);

  solver_ctx ctx(tp);

  // loop starts here
  // wait for header hashes, nonce+r
  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.trimmer->sipkeys.k0, ctx.trimmer->sipkeys.k1, ctx.trimmer->sipkeys.k2, ctx.trimmer->sipkeys.k3);
    u32 nsols = ctx.solve();
    for (unsigned s = 0; s < nsols; s++) {
      printf("Solution");
      u32* prf = &ctx.sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)prf[i]);
      printf("\n");
      int pow_rc = verify(prf, &ctx.trimmer->sipkeys);
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
