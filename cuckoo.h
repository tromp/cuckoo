// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <openssl/sha.h> // if openssl absent, use #include "sha256.c"

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
// relatively prime partition sizes
#define PARTU (SIZE/2+1)
#define PARTV (SIZE/2-1)

typedef uint64_t u64;
typedef struct {
  u64 v[4];
} siphash_ctx;
 
#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v1=ROTL(v1,13); v1 ^= v0; v0=ROTL(v0,32); \
    v2 += v3; v3=ROTL(v3,16); v3 ^= v2; \
    v0 += v3; v3=ROTL(v3,21); v3 ^= v0; \
    v2 += v1; v1=ROTL(v1,17); v1 ^= v2; v2=ROTL(v2,32); \
  } while(0)
 
// SipHash-2-4 specialized to precomputed key and 4 byte nonces
u64 siphash24(siphash_ctx *ctx, unsigned nonce) {
  u64 b = ( ( u64 )4 ) << 56 | nonce;
  u64 v0 = ctx->v[0], v1 = ctx->v[1], v2 = ctx->v[2], v3 = ctx->v[3] ^ b;
  SIPROUND; SIPROUND;
  v0 ^= b;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2  ^ v3;
}

// generate edge in cuckoo graph
void sipedge(siphash_ctx *ctx, unsigned nonce, unsigned *pu, unsigned *pv) {
  u64 sip = siphash24(ctx, nonce);
  *pu = 1 +         (unsigned)(sip % PARTU);
  *pv = 1 + PARTU + (unsigned)(sip % PARTV);
}

#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))
 
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

typedef struct {
  siphash_ctx sip_ctx;
  unsigned easiness;
  unsigned *cuckoo;
  unsigned (*sols)[PROOFSIZE];
  unsigned maxsols;
  unsigned nsols;
  int nthreads;
  pthread_mutex_t setsol;
} cuckoo_ctx;

typedef struct {
  int id;
  pthread_t thread;
  cuckoo_ctx *ctx;
} thread_ctx;

// algorithm parameters
#define MAXPATHLEN 8192

int path(unsigned *cuckoo, unsigned u, unsigned *us) {
  int nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (nu < 0)
        printf("maximum path length exceeded\n");
      else printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu] = u;
  }
  return nu;
}

// largest odd number of u64's that fit in MAXPATHLEN-PROOFSIZE unsigned's
#define SOLMODU ((MAXPATHLEN-PROOFSIZE-2)/2 | 1)
#define SOLMODV (SOLMODU-1)

void storedge(u64 uv, u64 *usck, u64 *vsck) {
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

void solution(cuckoo_ctx *ctx, unsigned *us, int nu, unsigned *vs, int nv) {
  u64 *usck = (u64 *)&us[PROOFSIZE], *vsck = (u64 *)&vs[PROOFSIZE];
  unsigned u, v, n;
  for (int i=0; i<SOLMODU; i++)
    usck[i] = vsck[i] = 0L;
  storedge((u64)*us<<32 | *vs, usck, vsck);
  while (nu--)
    storedge((u64)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
  while (nv--)
    storedge((u64)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
  pthread_mutex_lock(&ctx->setsol);
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
  else printf("Only recovered %d nonces\n", n);
  pthread_mutex_unlock(&ctx->setsol);
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;
  unsigned *cuckoo = ctx->cuckoo;
  unsigned us[MAXPATHLEN], u, vs[MAXPATHLEN], v; 
  int nu, nv;
  for (unsigned nonce = tp->id; nonce < ctx->easiness; nonce += ctx->nthreads) {
    sipedge(&ctx->sip_ctx, nonce, us, vs);
    if ((u = cuckoo[*us]) == *vs || (v = cuckoo[*vs]) == *us)
      continue; // ignore duplicate edges
#ifdef SHOW
    for (int j=1; j<=SIZE; j++)
      if (!cuckoo[j]) printf("%2d:   ",j);
      else            printf("%2d:%02d ",j,cuckoo[j]);
    printf(" %x (%d,%d)\n", nonce,*us,*vs);
#endif
    if (us[nu = path(cuckoo, u, us)] == vs[nv = path(cuckoo, v, vs)]) {
      int min = nu < nv ? nu : nv;
      for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
      int len = nu + nv + 1;
      printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (int)(nonce*100L/ctx->easiness));
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
  pthread_exit(NULL);
}

int verify(unsigned nonces[PROOFSIZE], char *header, int easipct) {
  int easiness = (unsigned)(easipct * (u64)SIZE / 100);
  siphash_ctx ctx;
  setheader(&ctx, header);
  unsigned us[PROOFSIZE], vs[PROOFSIZE], i = 0, n;
  for (n = 0; n < PROOFSIZE; n++) {
    if (nonces[n] >= easiness || (n && nonces[n] <= nonces[n-1]))
      return 0;
    sipedge(&ctx, nonces[n], &us[n], &vs[n]);
  }
  do {  // follow cycle until we return to i==0; n edges left to visit
    int j = i;
    for (int k = 0; k < PROOFSIZE; k++) // find unique other j with same vs[j]
      if (k != i && vs[k] == vs[i]) {
        if (j != i)
          return 0;
        j = k;
    }
    if (j == i)
      return 0;
    i = j;
    for (int k = 0; k < PROOFSIZE; k++) // find unique other i with same us[i]
      if (k != j && us[k] == us[j]) {
        if (i != j)
          return 0;
        i = k;
    }
    if (i == j)
      return 0;
    n -= 2;
  } while (i);
  return n == 0;
}
