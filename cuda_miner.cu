// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
// #include <helper_cuda.h>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// proof-of-work parameters
#ifndef SIZEMULT 
#define SIZEMULT 1
#endif
#ifndef SIZESHIFT 
#define SIZESHIFT 20
#endif
#ifndef PROOFSIZE
#define PROOFSIZE 42
#endif

#define SIZE (SIZEMULT*((unsigned)1<<SIZESHIFT))
// relatively prime partition sizes, assuming SIZESHIFT >= 2
#define PARTU (SIZE/2+1)
#define PARTV (SIZE/2-1)

typedef uint64_t u64;
typedef struct {
  u64 v[4];
} siphash_ctx;
 
#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))
 
#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v1=ROTL(v1,13); v1 ^= v0; v0=ROTL(v0,32); \
    v2 += v3; v3=ROTL(v3,16); v3 ^= v2; \
    v0 += v3; v3=ROTL(v3,21); v3 ^= v0; \
    v2 += v1; v1=ROTL(v1,17); v1 ^= v2; v2=ROTL(v2,32); \
  } while(0)
 
// SipHash-2-4 specialized to precomputed key and 4 byte nonces
__device__ u64 siphash24(siphash_ctx *ctx, unsigned nonce) {
  u64 b = ( ( u64 )4 ) << 56 | nonce;
  u64 v0 = ctx->v[0], v1 = ctx->v[1], v2 = ctx->v[2], v3 = ctx->v[3] ^ b;
  SIPROUND; SIPROUND;
  v0 ^= b;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2  ^ v3;
}

// generate edge in cuckoo graph
__device__ void sipedge(siphash_ctx *ctx, unsigned nonce, unsigned *pu, unsigned *pv) {
  u64 sip = siphash24(ctx, nonce);
  *pu = 1 +         (unsigned)(sip % PARTU);
  *pv = 1 + PARTU + (unsigned)(sip % PARTV);
}

// algorithm parameters
#define MAXPATHLEN 1024
#ifndef PRESIP
#define PRESIP 0
#endif

typedef struct {
  siphash_ctx sip_ctx;
  unsigned easiness;
  unsigned *cuckoo;
  unsigned (*sols)[PROOFSIZE];
  unsigned maxsols;
  unsigned nsols;
  int nthreads;
} cuckoo_ctx;

__device__ int path(unsigned *cuckoo, unsigned u, unsigned *us) {
  int nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      --nu;
      break;
    }
    us[nu] = u;
  }
  return nu;
}

// largest number of u64's that fit in MAXPATHLEN-PROOFSIZE unsigned's
#define SOLMODU ((MAXPATHLEN-PROOFSIZE)/2)
#define SOLMODV (SOLMODU-1)

__device__ void storedge(u64 uv, u64 *usck, u64 *vsck) {
  int j, i = uv % SOLMODU;
  u64 uvi = usck[i]; 
  if (uvi) {
    if (vsck[j = uv % SOLMODV]) {
      vsck[uvi % SOLMODV] = uvi;
    } else {
      vsck[j] = uv;
      return;
    }
  } else usck[i] = uv;
}

__device__ void solution(cuckoo_ctx *ctx, unsigned *us, int nu, unsigned *vs, int nv) {
  u64 *usck = (u64 *)&us[PROOFSIZE], *vsck = (u64 *)&vs[PROOFSIZE];
  unsigned u, v, n;
  for (int i=0; i<SOLMODU; i++)
    usck[i] = vsck[i] = 0L;
  storedge((u64)*us<<32 | *vs, usck, vsck);
  while (nu--)
    storedge((u64)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
  while (nv--)
    storedge((u64)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
  for (unsigned nonce = n = 0; nonce < ctx->easiness; nonce++) {
    sipedge(&ctx->sip_ctx, nonce, &u, &v);
    u64 *c, uv = (u64)u<<32 | v;
    if (*(c = &usck[uv % SOLMODU]) == uv || *(c = &vsck[uv % SOLMODV]) == uv) {
      ctx->sols[ctx->nsols][n++] = nonce;
      *c = 0;
    }
  }
  if (n == PROOFSIZE)
    ctx->nsols++;
}

__global__ void worker(cuckoo_ctx *ctx) {
  unsigned *cuckoo = ctx->cuckoo;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned us[MAXPATHLEN], vs[MAXPATHLEN], uvpre[2*PRESIP], npre = 0; 
  for (unsigned nonce = id; nonce < ctx->easiness; nonce += ctx->nthreads) {
#if PRESIP==0
    sipedge(&ctx->sip_ctx, nonce, us, vs);
#else
    if (!npre)
      for (unsigned n = nonce; npre < PRESIP; npre++, n += ctx->nthreads)
        sipedge(&ctx->sip_ctx, n, &uvpre[2*npre], &uvpre[2*npre+1]);
    unsigned i = PRESIP - npre--;
    *us = uvpre[2*i];
    *vs = uvpre[2*i+1];
#endif
    unsigned u = cuckoo[*us], v = cuckoo[*vs];
    if (u == *vs || v == *us)
      continue; // ignore duplicate edges
    int nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
    if (us[nu] == vs[nv]) {
      int min = nu < nv ? nu : nv;
      for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
      int len = nu + nv + 1;
      // printf("% 4d-cycle found at %d:%d%%\n", len, id, (int)(nonce*100L/ctx->easiness));
      if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
        solution(ctx, us, nu, vs, nv);
      continue;
    }
    if (nu < nv) {
      while (nu--)
        cuckoo[us[nu+1]] = us[nu];
      cuckoo[*us] = *vs;
    } else {
      while (nv--)
        cuckoo[vs[nv+1]] = vs[nv];
      cuckoo[*vs] = *us;
    }
  }
  // ctx->sols[ctx->nsols][0] = 42;
  // ctx->sols[ctx->nsols][1] = 0;
  // for (int i=0; i<=SIZE; i++)
    // ctx->sols[ctx->nsols][1] |= cuckoo[i];
  // ctx->nsols++;
}

// derive siphash key from header
void setheader(siphash_ctx *ctx, char *header) {
  unsigned char hdrkey[32];
  SHA256((unsigned char *)header, strlen(header), hdrkey);
  u64 k0 = U8TO64_LE(hdrkey);
  u64 k1 = U8TO64_LE(hdrkey+8);
  ctx->v[0] = k0 ^ 0x736f6d6570736575ULL;
  ctx->v[1] = k1 ^ 0x646f72616e646f6dULL;
  ctx->v[2] = k0 ^ 0x6c7967656e657261ULL;
  ctx->v[3] = k1 ^ 0x7465646279746573ULL;
}

int main(int argc, char **argv) {
  assert(SIZE < 1L<<32);
  int nthreads = 1;
  int maxsols = 8;
  char *header = "";
  int c, easipct = 50;
  while ((c = getopt (argc, argv, "e:h:m:t:")) != -1) {
    switch (c) {
      case 'e':
        easipct = atoi(optarg);
        break;
      case 'h':
        header = optarg;
        break;
      case 'm':
        maxsols = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for %d-cycle on cuckoo%d%d(\"%s\") with %d%% edges and %d threads\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, easipct, nthreads);

  cuckoo_ctx ctx;
  setheader(&ctx.sip_ctx, header);
  ctx.easiness = (unsigned)(easipct * (u64)SIZE / 100);
  ctx.maxsols = maxsols;
  ctx.nsols = 0;
  ctx.nthreads = nthreads;
  checkCudaErrors(cudaMalloc((void**)&ctx.cuckoo, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMemset(ctx.cuckoo, 0, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMalloc((void**)&ctx.sols, maxsols*PROOFSIZE*sizeof(unsigned)));

  cuckoo_ctx *device_ctx;
  checkCudaErrors(cudaMalloc((void**)&device_ctx, sizeof(cuckoo_ctx)));
  cudaMemcpy(device_ctx, &ctx, sizeof(cuckoo_ctx), cudaMemcpyHostToDevice);

  worker<<<nthreads,1>>>(device_ctx);
  cudaMemcpy(&ctx, device_ctx, sizeof(cuckoo_ctx), cudaMemcpyDeviceToHost);
  unsigned (*sols)[PROOFSIZE];
  sols = (unsigned (*)[PROOFSIZE])calloc(maxsols, PROOFSIZE*sizeof(unsigned));
  cudaMemcpy(sols, ctx.sols, maxsols*PROOFSIZE*sizeof(unsigned), cudaMemcpyDeviceToHost);

  for (int s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %x", sols[s][i]);
    printf("\n");
  }
  checkCudaErrors(cudaFree(device_ctx));
  checkCudaErrors(cudaFree(ctx.sols));
  checkCudaErrors(cudaFree(ctx.cuckoo));
  return 0;
}
