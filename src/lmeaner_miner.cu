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

__constant__ uint2 recoveredges[PROOFSIZE];
__constant__ uint2 e0 = {0,0};

__device__ __forceinline__ ulonglong4 Pack8(const u32 e0, const u32 e1, const u32 e2, const u32 e3, const u32 e4, const u32 e5, const u32 e6, const u32 e7) {
  return make_ulonglong4((u64)e0<<32|e1, (u64)e2<<32|e3, (u64)e4<<32|e5, (u64)e6<<32|e7);
}

__global__  void FluffySeed2A(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, int * __restrict__ indexes) {
  const int col = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int FLUSHA = 32;
  const int FLUSHA2 = 64;

  __shared__ u16 tmp[NX][FLUSHA2];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(tmp[0][0]);
  __shared__ int counters[NX];

  for (int r = lid; r < NX; r += dim)
    counters[r] = 0;
  __syncthreads();
  const int loops = (NEDGES / NX) / dim;

  for (int i = 0; i < loops; i++) {
    u64 nonce = col * (NEDGES / NX) + i * dim + lid;
    u32 hashx = dipnode(sipkeys, nonce, 0);
    int row = hashx & XMASK;
    int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1));
    tmp[row][counter] = (u16)nonce;
    __syncthreads();
    if (counter == FLUSHA-1) {
      int localIdx = min(FLUSHA2, counters[row]);
      int newCount = localIdx - FLUSHA;
      counters[row] = newCount;
      int cnt = min((int)atomicAdd(indexes + row * NX + col, FLUSHA), (int)(DUCK_A_EDGES - FLUSHA));
      for (int i = 0; i < FLUSHA; i += TMPPERLL4)
        buffer[((u64)(row * NX + col) * DUCK_A_EDGES + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
      for (int t = 0; t < newCount; t++) {
        tmp[row][t] = tmp[row][t + FLUSHA];
      }
    }
  }
  __syncthreads();
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = 0;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(indexes + row * NX + col, TMPPERLL4), (int)(DUCK_A_EDGES - TMPPERLL4));
      buffer[((u64)(row * NX + col) * DUCK_A_EDGES + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
    }
  }
}

__global__  void FluffySeed2B(const siphash_keys &sipkeys, const u16 * __restrict__ source, ulonglong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int FLUSHB = 32;
  const int FLUSHB2 = 2 * FLUSHB;

  __shared__ uint2 tmp[NX][FLUSHB2];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(tmp[0][0]);
  __shared__ int counters[NX];

  for (int r = lid; r < NX; r += dim)
    counters[r] = 0;
  __syncthreads();
  const int row = group / NX;
  const int bucketEdges = min((int)sourceIndexes[group], (int)(DUCK_A_EDGES));
  const int loops = (bucketEdges + dim-1) / dim;
  const u64 lag = 16384;
  u64 nonce = (group % NX) * (NEDGES / NX);
  for (int loop = 0; loop < loops; loop++) {
    int col;
    int counter = 0;
    const int edgeIndex = loop * dim + lid;
    if (edgeIndex < bucketEdges) {
      const int index = group * DUCK_A_EDGES + edgeIndex;
      u16 nonce16 = __ldg(&source[index]);
      if (loop < (loops * 15/16) || nonce16 != 0) { // fillers only at end
        nonce += (((u64)nonce16 - nonce + lag) & 0xffffULL) - lag;
        uint2 edge;
        edge.x = dipnode(sipkeys, nonce, 0);
        edge.y = dipnode(sipkeys, nonce, 1);
        col = (edge.x >> XBITS) & XMASK;
        counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1));
        tmp[col][counter] = edge;
      }
    }
    __syncthreads();
    if (counter == FLUSHB-1) {
      int localIdx = min(FLUSHB2, counters[col]);
      int newCount = localIdx - FLUSHB;
      int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, FLUSHB), (int)(DUCK_A_EDGES - FLUSHB));
      for (int i = 0; i < FLUSHB; i += TMPPERLL4)
        destination[((u64)(row * NX + col) * DUCK_A_EDGES + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
      for (int t = 0; t < newCount; t++) {
        tmp[col][t] = tmp[col][t + FLUSHB];
      }
      counters[col] = newCount;
      assert (newCount < FLUSHB);
    }
    __syncthreads(); 
  }
  if (((nonce - (group % NX) * (NEDGES / NX)) >> 16) != ((NEDGES-1)/NX)>>16) {
    printf("group %x lid %x nonce %08llx\n", group, lid, nonce);
    assert(0);
  }
  
  for (int col = lid; col < NX; col += dim) {
    int localIdx = min(FLUSHB2, counters[col]);
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[col][j] = e0;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, TMPPERLL4), (int)(DUCK_A_EDGES - TMPPERLL4));
      destination[((u64)(row * NX + col) * DUCK_A_EDGES + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
    }
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
__global__  void FluffyRound(const int round, const uint2 * __restrict__ source, uint2 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
  const int group = blockIdx.x;
  const int lid = threadIdx.x;
  const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

  __shared__ u32 ecounters[COUNTERWORDS];

  const int edgesInBucket = min(sourceIndexes[group], bktInSize);
  const int loops = (edgesInBucket + CTHREADS-1) / CTHREADS;

  for (int i = 0; i < COUNTERWORDS / CTHREADS; i++)
    ecounters[lid + CTHREADS * i] = 0; // IS SINGLE-INCREMENT FASTER?
  __syncthreads();
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * CTHREADS + lid;
    if (lindex < edgesInBucket) {
      const int index = bktInSize * group + lindex;
      uint2 edge = __ldg(&source[index]);
      if (edge.x == 0 && edge.y == 0) continue;
      Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
    }
  }
  __syncthreads();
  for (int loop = loops-1; loop >= 0; loop--) {
    const int lindex = loop * CTHREADS + lid;
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
    genA.blocks         =   NX;
    genA.tpb            = 2*NX;
    genB.blocks         =  NX2;
    genB.tpb            = 2*NX;
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
  const size_t bufferSize  = DUCK_B_EDGES_64 * NX * sizeof(uint2);
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
    cudaEvent_t start, stop;
    cudaEvent_t startall, stopall;
    checkCudaErrors(cudaEventCreate(&startall)); checkCudaErrors(cudaEventCreate(&stopall));
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
  
    cudaMemset(indexesE, 0, indexesSize);
    cudaMemset(indexesE2, 0, indexesSize);
    cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);
  
    cudaDeviceSynchronize();
    float durationA, durationB;
    cudaEventRecord(start, NULL);
  
    FluffySeed2A<<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, (ulonglong4 *)bufferA, (int *)indexesE);
  
    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    cudaEventRecord(start, NULL);
  
    FluffySeed2B<<<tp.genB.blocks, tp.genB.tpb>>>(*dipkeys, (const u16 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE, (int *)indexesE2);
    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    printf("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
  
    cudaMemset(indexesE, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES><<<tp.trim.blocks, tp.trim.tpb>>>(0, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
    cudaMemset(indexesE2, 0, indexesSize);
    FluffyRound<DUCK_B_EDGES, DUCK_A_EDGES/2><<<tp.trim.blocks, tp.trim.tpb>>>(1 ,(const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
    cudaMemset(indexesE, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES/2, DUCK_A_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>(2 ,(const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
    cudaMemset(indexesE2, 0, indexesSize);
    FluffyRound<DUCK_A_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>(3 ,(const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
  
    cudaDeviceSynchronize();
  
    for (int round = 4; round < tp.ntrims; round += 2) {
      cudaMemset(indexesE, 0, indexesSize);
      FluffyRound<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>(round, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
      cudaMemset(indexesE2, 0, indexesSize);
      FluffyRound<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>(round, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
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
        // if (len == 2) printf("edge %x %x\n", edge.x, edge.y);
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
  while ((c = getopt(argc, argv, "sb:c:d:h:k:m:n:r:V:v:w:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        printf("SYNOPSIS\n  cuda30 [-d device] [-h hexheader] [-k rounds [-c count]] [-m trims] [-n nonce] [-r range] [-V blocks] [-y threads] [-Z blocks] [-z threads]\n");
        printf("DEFAULTS\n  cuda30 -d %d -h \"\" -k %d -c %d -m %d -n %d -r %d -v %d -w %d -y %d -Z %d -z %d\n", device, tp.reportrounds, tp.reportcount, tp.ntrims, nonce, range, tp.genA.tpb, tp.genB.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
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
      case 'v':
        tp.genA.tpb = atoi(optarg);
        break;
      case 'w':
        tp.genB.tpb = atoi(optarg);
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
