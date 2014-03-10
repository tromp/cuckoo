// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

// algorithm parameters
#define MAXPATHLEN 8192

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

// largest number of u64's that fit in MAXPATHLEN-PROOFSIZE unsigned's
#define SOLMODU ((MAXPATHLEN-PROOFSIZE)/2)
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
  assert(ctx.cuckoo = calloc(1+SIZE, sizeof(unsigned)));
  assert(ctx.sols = calloc(maxsols, PROOFSIZE*sizeof(unsigned)));
  ctx.maxsols = maxsols;
  ctx.nsols = 0;
  ctx.nthreads = nthreads;
  pthread_mutex_init(&ctx.setsol, NULL);

  thread_ctx *threads = calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  for (int t = 0; t < nthreads; t++) {
    threads[t].id = t;
    threads[t].ctx = &ctx;
    assert(pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]) == 0);
  }
  for (int t = 0; t < nthreads; t++)
    assert(pthread_join(threads[t].thread, NULL) == 0);
  for (int s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %x", ctx.sols[s][i]);
    printf("\n");
  }
  return 0;
}
