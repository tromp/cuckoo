// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"
#include <unistd.h>
#include <stdio.h>
#include <assert.h>

// proof-of-work parameters
#ifndef SIZEMULT 
#define SIZEMULT 1
#endif
#ifndef SIZESHIFT 
#define SIZESHIFT 20
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
 
typedef struct {
  int len;
  int id;
  int pct;
} cycle;

typedef struct {
  siphash_ctx sip_ctx;
  unsigned easiness;
  unsigned *cuckoo;
  cycle *cycles;
  unsigned maxcycles;
  unsigned ncycles;
  int nthreads;
} cuckoo_ctx;

// algorithm parameters
#define MAXPATHLEN 4096

// generate edge in cuckoo graph
__device__ void sipedge(siphash_ctx *ctx, unsigned nonce, unsigned *pu, unsigned *pv) {
  u64 b = ( ( u64 )4 ) << 56 | nonce;
  u64 v0 = ctx->v[0], v1 = ctx->v[1], v2 = ctx->v[2], v3 = ctx->v[3] ^ b;
  SIPROUND; SIPROUND;
  v0 ^= b;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  u64 sip = v0 ^ v1 ^ v2  ^ v3;
  *pu = 1 +         (unsigned)(sip % PARTU);
  *pv = 1 + PARTU + (unsigned)(sip % PARTV);
}

__global__ void worker(cuckoo_ctx *ctx) {
  cuckoo_ctx local = *ctx;
  unsigned *cuckoo = local.cuckoo;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned us[MAXPATHLEN], vs[MAXPATHLEN];
  for (unsigned nonce = id; nonce < local.easiness; nonce += local.nthreads) {
    sipedge(&local.sip_ctx, nonce, us, vs);
    unsigned u = cuckoo[*us], v = cuckoo[*vs];
    if (u == *vs || v == *us)
      continue; // ignore duplicate edges
    int nu, nv;
    for (nu = 0; u; u = cuckoo[u]) {
      if (++nu >= MAXPATHLEN) return;
      us[nu] = u;
    }
    for (nv = 0; v; v = cuckoo[v]) {
      if (++nv >= MAXPATHLEN) return;
      vs[nv] = v;
    }
    if (us[nu] == vs[nv]) {
      int min = nu < nv ? nu : nv;
      for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
      cycle *c = &local.cycles[ctx->ncycles++];
      c->len = nu + nv + 1;
      c->id = id;
      c->pct = (int)(nonce*100L/local.easiness);
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

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main(int argc, char **argv) {
  assert(SIZE < 1L<<32);
  int nthreads = 1;
  int maxcycles = 16;
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
        maxcycles = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for cycles on cuckoo%d%d(\"%s\") with %d%% edges and %d threads\n",
               SIZEMULT, SIZESHIFT, header, easipct, nthreads);

  cuckoo_ctx ctx;
  setheader(&ctx.sip_ctx, header);
  ctx.easiness = (unsigned)(easipct * (u64)SIZE / 100);
  ctx.maxcycles = maxcycles;
  ctx.ncycles = 0;
  ctx.nthreads = nthreads;
  checkCudaErrors(cudaMalloc((void**)&ctx.cuckoo, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMemset(ctx.cuckoo, 0, (1+SIZE) * sizeof(unsigned)));
  checkCudaErrors(cudaMalloc((void**)&ctx.cycles, maxcycles*sizeof(cycle)));

  cuckoo_ctx *device_ctx;
  checkCudaErrors(cudaMalloc((void**)&device_ctx, sizeof(cuckoo_ctx)));
  cudaMemcpy(device_ctx, &ctx, sizeof(cuckoo_ctx), cudaMemcpyHostToDevice);

  worker<<<nthreads,1>>>(device_ctx);
  cudaMemcpy(&ctx, device_ctx, sizeof(cuckoo_ctx), cudaMemcpyDeviceToHost);
  cycle *cycles;
  cycles = (cycle *)calloc(maxcycles, sizeof(cycle));
  cudaMemcpy(cycles, ctx.cycles, maxcycles*sizeof(cycle), cudaMemcpyDeviceToHost);

  for (int s = 0; s < ctx.ncycles; s++)
    printf("% 4d-cycle found at %d:%d%%\n", cycles[s].len, cycles[s].id, cycles[s].pct);
  checkCudaErrors(cudaFree(device_ctx));
  checkCudaErrors(cudaFree(ctx.cycles));
  checkCudaErrors(cudaFree(ctx.cuckoo));
  return 0;
}
