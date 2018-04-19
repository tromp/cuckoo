// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura - photon
// This CUDA part of Theta optimized miner is covered by the FAIR MINING license 2.1.1

#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <xmmintrin.h>
#include <algorithm>
#include <stdio.h>
#include <stdint.h>
#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h> // gettimeofday

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u32 node_t;
typedef u64 nonce_t;

#ifndef XBITS
#define XBITS 6
#endif

#define EDGEBITS 29
// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)
#define NODEBITS (EDGEBITS + 1)
#define NNODES ((node_t)1 << NODEBITS)
#define NODEMASK (NNODES - 1)

#define YBITS XBITS
const static u32 NX        = 1 << XBITS;
const static u32 XMASK     = NX - 1;
const static u32 NY        = 1 << YBITS;
const static u32 YMASK     = NY - 1;
const static u32 XYBITS    = XBITS + YBITS;
const static u32 NXY       = 1 << XYBITS;
const static u32 ZBITS     = EDGEBITS - XYBITS;
const static u32 ZBYTES    = (ZBITS + 7) / 8;
const static u32 NZ        = 1 << ZBITS;
const static u32 ZMASK     = NZ - 1;
const static u32 YZBITS    = YBITS + ZBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 YZMASK    = NYZ - 1;

// ------ OPTIONS ---------------------------------------------------
// Do not change, reported as not working
#define VRAMSMALL 1

// ------------------------------------------------------------------


typedef struct uint10
{
	uint2 edges[5];
} uint10;

#define DUCK_SIZE_A 130LL
#define DUCK_SIZE_B 85LL

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#define CTHREADS 1024
#define BKTMASK4K (4096-1)

__constant__ u64 recovery[42];

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  } while(0)


__device__  node_t dipnode(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const  nonce_t nce, const  u32 uorv) {
	u64 nonce = 2 * nce + uorv;
	u64 v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

__device__ ulonglong4 Pack4edges(const uint2 e1, const  uint2 e2, const  uint2 e3, const  uint2 e4)
{
	u64 r1 = (((u64)e1.y << 32) | ((u64)e1.x));
	u64 r2 = (((u64)e2.y << 32) | ((u64)e2.x));
	u64 r3 = (((u64)e3.y << 32) | ((u64)e3.x));
	u64 r4 = (((u64)e4.y << 32) | ((u64)e4.x));
	return make_ulonglong4(r1, r2, r3, r4);
}

__global__  void FluffyRecovery(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, ulonglong4 * buffer, int * indexes)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

	__shared__ u32 nonces[42];

	if (lid < 42) nonces[lid] = 0;

	__syncthreads();

	for (int i = 0; i < 1024 * 4; i++)
	{
		u64 nonce = gid * (1024 * 4) + i;

		u64 u = dipnode(v0i, v1i, v2i, v3i, nonce, 0);
		u64 v = dipnode(v0i, v1i, v2i, v3i, nonce, 1);

		u64 a = u | (v << 32);
		u64 b = v | (u << 32);

		for (int i = 0; i < 42; i++)
		{
			if ((recovery[i] == a) || (recovery[i] == b))
				nonces[i] = nonce;
		}
	}

	__syncthreads();

	if (lid < 42)
	{
		if (nonces[lid] > 0)
			indexes[lid] = nonces[lid];
	}
}

__global__  void FluffySeed2A(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, ulonglong4 * buffer, int * indexes)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

	__shared__ uint2 tmp[NX][16];
	__shared__ int counters[NX];

	counters[lid] = 0;

	__syncthreads();

	u64 nonce    = (u64)gid * NEDGES / (blockDim.x * gridDim.x);
	u64 endnonce = (u64)(gid +1) * NEDGES / (blockDim.x * gridDim.x);
	for (; nonce < endnonce; nonce++)
	{
		uint2 hash;

		hash.x = dipnode(v0i, v1i, v2i, v3i, nonce, 0);

		int bucket = hash.x & XMASK;

		__syncthreads();

		int counter = min((int)atomicAdd(counters + bucket, 1), (int)15);

		hash.y = dipnode(v0i, v1i, v2i, v3i, nonce, 1);

		assert(hash.x | hash.y);

		tmp[bucket][counter] = hash;

		__syncthreads();

		{
			int localIdx = min(16, counters[lid]);
			assert(counters[lid] < 24);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				{
					int cnt = min((int)atomicAdd(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));

					{
						buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
						buffer[(lid * DUCK_A_EDGES_64 + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
					}
				}

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}

			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx >= 4)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}

}

#define BKTGRAN 32
__global__  void FluffySeed2B(const  uint2 * source, ulonglong4 * destination, const  int * sourceIndexes, int * destinationIndexes, int startBlock)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ uint2 tmp[64][16];
	__shared__ int counters[64];

	counters[lid] = 0;

	__syncthreads();

	const int offsetMem = startBlock * DUCK_A_EDGES_64;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / 64);

	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (64 * i) + lid;

		if (edgeIndex < bucketEdges)
		{
			uint2 edge = source[offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex];
			
			if (edge.x == 0 && edge.y == 0) continue;

			int bucket = (edge.x >> 6) & (64 - 1);

			__syncthreads();

			int counter = min((int)atomicAdd(counters + bucket, 1), (int)15);

			tmp[bucket][counter] = edge;

			__syncthreads();

			int localIdx = min(16, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				{
					int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));

					{
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
					}
				}

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}

			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx >= 4)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}
}

__device__ __forceinline__  void Increase2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	u32 old = atomicOr(ecounters + word, mask) & mask;

	if (old > 0)
		atomicOr(ecounters + word + 4096, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}

template<int bktInSize, int bktOutSize>
__global__   void FluffyRound(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	ecounters[lid] = 0;
	ecounters[lid + 1024] = 0;
	ecounters[lid + (1024 * 2)] = 0;
	ecounters[lid + (1024 * 3)] = 0;
	ecounters[lid + (1024 * 4)] = 0;
	ecounters[lid + (1024 * 5)] = 0;
	ecounters[lid + (1024 * 6)] = 0;
	ecounters[lid + (1024 * 7)] = 0;

	__syncthreads();

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	__syncthreads();

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
			}
		}
	}

}

template __global__ void FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);

__global__   void /*Magical*/FluffyTail/*Pony*/(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	int myEdges = sourceIndexes[group];
	__shared__ int destIdx;

	if (lid == 0)
		destIdx = atomicAdd(destinationIndexes, myEdges);

	__syncthreads();

	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[group * DUCK_B_EDGES + lid];
	}
}

static u32 hostB[2 * 260000];
static u64 h_mydata[42];

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

struct twostagetpb {
  u16 stage1tpb;
  u16 stage2tpb;
};

struct trimparams {
  u16 nblocks;
  u16 ntrims;
  u16 genUblocks;
  u16 genUtpb;
  twostagetpb genV;
  twostagetpb trim;
  twostagetpb rename[2];
  u16 trim3tpb;
  u16 rename3tpb;
  u16 reportcount;
  u16 reportrounds;

  trimparams() {
    ntrims              = 256;
    nblocks             = 128;
    genUblocks          = 512;
    genUtpb             =  64;
    genV.stage1tpb      =  32;
    genV.stage2tpb      = 128;
    trim.stage1tpb      =  32;
    trim.stage2tpb      = 128;
    rename[0].stage1tpb =  32;
    rename[0].stage2tpb =  64;
    rename[1].stage1tpb =  32;
    rename[1].stage2tpb = 128;
    trim3tpb            =  32;
    rename3tpb          =   8;
    reportcount         =   1;
    reportrounds        =   0;
  }
};

#define PROOFSIZE 42
typedef u32 proof[PROOFSIZE];

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
    cuckoo = (u64 *)calloc(CUCKOO_SIZE, sizeof(u64));
    assert(cuckoo != 0);
  }
  ~cuckoo_hash() {
    free(cuckoo);
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

#define MAXPATHLEN 8192

  u32 path(cuckoo_hash &cuckoo, u32 u, u32 *us) {
    u32 nu, u0 = u;
    for (nu = 0; u; u = cuckoo[u]) {
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

  void findcycles(u64* edges, u32 size) {
    cuckoo_hash &cuckoo = *(new cuckoo_hash());
    node_t us[MAXPATHLEN], vs[MAXPATHLEN];
    u32 sumsize = 0;
    for (u32 i = 0; i < size; i++) {
      u32 uxyz = edges[i] >> 32;  u32 vxyz = edges[i] & 0xffffffff;
      const u32 u0 = uxyz << 1, v0 = (vxyz << 1) | 1;
      if (u0) {
        u32 nu = path(cuckoo, u0, us), nv = path(cuckoo, v0, vs);
        if (!nu-- || !nv--)
          return; // drop edge causing trouble
        // printf("vx %02x ux %02x e %08x uxyz %06x vxyz %06x u0 %x v0 %x nu %d nv %d\n", vx, ux, e, uxyz, vxyz, u0, v0, nu, nv);
        if (us[nu] == vs[nv]) {
          const u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          const u32 len = nu + nv + 1;
          printf("%4d-cycle found\n", len);
          // if (len == PROOFSIZE)
            // solution(us, nu, vs, nv);
        } else if (nu < nv) {
          while (nu--)
            cuckoo.set(us[nu+1], us[nu]);
          cuckoo.set(u0, v0);
        } else {
          while (nv--)
            cuckoo.set(vs[nv+1], vs[nv]);
          cuckoo.set(v0, u0);
        }
      }
    }
  }

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len, timems,timems2;
  struct timeval time0, time1;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt(argc, argv, "sb:c:d:h:k:m:n:r:U:u:V:v:T:t:X:x:Y:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        printf("SYNOPSIS\n  cuda30 [-b blocks] [-d device] [-h hexheader] [-k rounds [-c count]] [-m trims] [-n nonce] [-r range] [-U blocks] [-u threads] [-V threads] [-v threads] [-T threads] [-t threads] [-X threads] [-x threads] [-Y threads] [-y threads] [-Z threads] [-z threads]\n");
        printf("DEFAULTS\n  cuda30 -b %d -d %d -h \"\" -k %d -c %d -m %d -n %d -r %d -U %d -u %d -V %d -v %d -T %d -t %d -X %d -x %d -Y %d -y %d -Z %d -z %d\n", tp.nblocks, device, tp.reportrounds, tp.reportcount, tp.ntrims, nonce, range, tp.genUblocks, tp.genUtpb, tp.genV.stage1tpb, tp.genV.stage2tpb, tp.trim.stage1tpb, tp.trim.stage2tpb, tp.rename[0].stage1tpb, tp.rename[0].stage2tpb, tp.rename[1].stage1tpb, tp.rename[1].stage2tpb, tp.trim3tpb, tp.rename3tpb);
        exit(0);
      case 'b':
        tp.nblocks = atoi(optarg);
        break;
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
      case 'U':
        tp.genUblocks = atoi(optarg);
        break;
      case 'u':
        tp.genUtpb = atoi(optarg);
        break;
      case 'V':
        tp.genV.stage1tpb = atoi(optarg);
        break;
      case 'v':
        tp.genV.stage2tpb = atoi(optarg);
        break;
      case 'T':
        tp.trim.stage1tpb = atoi(optarg);
        break;
      case 't':
        tp.trim.stage2tpb = atoi(optarg);
        break;
      case 'X':
        tp.rename[0].stage1tpb = atoi(optarg);
        break;
      case 'x':
        tp.rename[0].stage2tpb = atoi(optarg);
        break;
      case 'Y':
        tp.rename[1].stage1tpb = atoi(optarg);
        break;
      case 'y':
        tp.rename[1].stage2tpb = atoi(optarg);
        break;
      case 'Z':
        tp.trim3tpb = atoi(optarg);
        break;
      case 'z':
        tp.rename3tpb = atoi(optarg);
        break;
    }
  }
  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  assert(device < nDevices);
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  assert(tp.genUtpb <= prop.maxThreadsPerBlock);
  assert(tp.genV.stage1tpb <= prop.maxThreadsPerBlock);
  assert(tp.genV.stage2tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.stage1tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.stage2tpb <= prop.maxThreadsPerBlock);
  assert(tp.rename[0].stage1tpb <= prop.maxThreadsPerBlock);
  assert(tp.rename[0].stage2tpb <= prop.maxThreadsPerBlock);
  assert(tp.rename[1].stage1tpb <= prop.maxThreadsPerBlock);
  assert(tp.rename[1].stage2tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim3tpb <= prop.maxThreadsPerBlock);
  assert(tp.rename3tpb <= prop.maxThreadsPerBlock);
  u64 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 10240; dbytes>>=10,dunit++) ;
  printf("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
  cudaSetDevice(device);

  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, tp.nblocks);

	std::ofstream myfile;

	u64* buffer = new u64[150000];

	const size_t bufferSize = DUCK_SIZE_A * 1024 * 4096 * 8;
	const size_t bufferSize2 = DUCK_SIZE_B * 1024 * 4096 * 8;
	const size_t indexesSize = 128 * 128 * 4;

	const unsigned int edges = (1 << 29);

	int * bufferA;
	int * bufferB;
	int * indexesE;
	int * indexesE2;

	u32 hostA[256 * 256];

	cudaError_t cudaStatus;
	size_t free_device_mem = 0;
	size_t total_device_mem = 0;

	unsigned long long k0 = 0xa34c6a2bdaa03a14ULL;
	unsigned long long k1 = 0xd736650ae53eee9eULL;
	unsigned long long k2 = 0x9a22f05e3bffed5eULL;
	unsigned long long k3 = 0xb8d55478fa3a606dULL;

	checkCudaErrors( cudaMalloc((void**)&bufferA, bufferSize) );

	fprintf(stderr, "Allociating buffer 1\n");

	cudaMemGetInfo(&free_device_mem, &total_device_mem);

	//printf("Buffer A: Currently available amount of device memory: %zu bytes\n", free_device_mem);

	fprintf(stderr, "Allociating buffer 2\n");

	checkCudaErrors( cudaMalloc((void**)&bufferB, bufferSize2) );

	checkCudaErrors( cudaMalloc((void**)&indexesE, indexesSize) );
	checkCudaErrors( cudaMalloc((void**)&indexesE2, indexesSize) );

	cudaMemGetInfo(&free_device_mem, &total_device_mem);

	fprintf(stderr, "Currently available amount of device memory: %zu bytes\n", free_device_mem);

	fprintf(stderr, "CUDA device armed\n");

	// loop starts here
	// wait for header hashes, nonce+r

	  u32 sumnsols = 0;
    while (scanf("%llx %llx %llx %llx\n", &k0, &k1, &k2, &k3) == 4) {
    gettimeofday(&time0, 0);
    // ctx.setheadernonce(header, sizeof(header), nonce + r);
    // printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r,
       // ctx.trimmer->sip_keys.k0, ctx.trimmer->sip_keys.k1, ctx.trimmer->sip_keys.k2, ctx.trimmer->sip_keys.k3);

		{
			// comamnded to trim edges
			// parse k0 k1 k2 k3 nonce

			fprintf(stderr, "#a\n"); // ack
			fprintf(stderr, "Trimming: %llx %llx %llx %llx %llx\n", k0, k1, k2, k3, nonce); // ack
		}

		cudaMemset(indexesE, 0, indexesSize);
		cudaMemset(indexesE2, 0, indexesSize);

		cudaDeviceSynchronize();

#ifdef VRAMSMALL

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
    float duration;
    cudaEventRecord(start, NULL);

		FluffySeed2A << < tp.genUblocks, tp.genUtpb >> > (k0, k1, k2, k3, (ulonglong4 *)bufferA, (int *)indexesE2);

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&duration, start, stop);
    printf("FluffySeed2A completed in %.0f ms\n", duration);

		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE2, (int *)indexesE, 0);
		cudaMemcpy(bufferA, bufferB, bufferSize / 2, cudaMemcpyDeviceToDevice);

		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE2, (int *)indexesE, 32);
		checkCudaErrors( cudaMemcpy(&((char *)bufferA)[bufferSize / 2], bufferB, bufferSize / 2, cudaMemcpyDeviceToDevice) );


		cudaMemset(indexesE2, 0, indexesSize);
		FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);

#else
		FluffySeed2A << < tp.genUblocks, tp.genUtpb >> > (k0, k1, k2, k3, (ulonglong4 *)bufferA, (int *)indexesE);

		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE, (int *)indexesE2, 0);
		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, ((ulonglong4 *)bufferB) + (bufferSize2 / 8 / 4 / 2), (const int *)indexesE, (int *)indexesE2, 32);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "status memcpy: %s\n", cudaGetErrorString(cudaStatus));

		cudaMemset(indexesE, 0, indexesSize);
		FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
		cudaMemset(indexesE2, 0, indexesSize);
		FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
#endif

		cudaDeviceSynchronize();

		for (int i = 0; i < 80; i++)
		{
			cudaMemset(indexesE, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
			
		}
		
		cudaMemset(indexesE, 0, indexesSize);
		cudaDeviceSynchronize();

		FluffyTail << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
		cudaMemcpy(hostA, indexesE, 64 * 64 * 4, cudaMemcpyDeviceToHost);

		int pos = hostA[0];
		assert (pos > 0 && pos < 500000);
		cudaMemcpy(buffer, &((u64 *)bufferA)[0], pos * 8, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		fprintf(stderr, "Trimmed to: %d edges\n", pos);

    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("trim time: %d ms\n", timems);
    gettimeofday(&time0, 0);

		findcycles(buffer, pos);

    gettimeofday(&time1, 0);
    timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("findcycles time: %d ms total %d ms\n", timems2, timems+timems2);

	}

	delete buffer;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "status: %s\n", cudaGetErrorString(cudaStatus));


Error:
	fprintf(stderr, "CUDA terminating...\n");
	fprintf(stderr, "#x\n");
	cudaFree(bufferA);
	cudaFree(bufferB);
	cudaFree(indexesE);
	cudaFree(indexesE2);
	cudaDeviceReset();
	return 0;
}
