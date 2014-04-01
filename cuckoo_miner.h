// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
#include <stdio.h>
#include <pthread.h>

// algorithm parameters
#define MAXPATHLEN 6144
#ifndef PRESIP
#define PRESIP 1024
#endif

typedef struct {
  siphash_ctx sip_ctx;
  u64 easiness;
  u64 *cuckoo;
  u64 (*sols)[PROOFSIZE];
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

int path(u64 *cuckoo, u64 u, u64 *us) {
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

// largest number of u64's that fit in MAXPATHLEN-PROOFSIZE u64's
#define SOLMODU (MAXPATHLEN-PROOFSIZE)
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

void solution(cuckoo_ctx *ctx, u64 *us, int nu, u64 *vs, int nv) {
  u64 *usck = (u64 *)&us[PROOFSIZE], *vsck = (u64 *)&vs[PROOFSIZE];
  u64 u, v;
  unsigned n;
  for (int i=0; i<SOLMODU; i++)
    usck[i] = vsck[i] = 0L;
  storedge((u64)*us<<32 | *vs, usck, vsck);
  while (nu--)
    storedge((u64)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
  while (nv--)
    storedge((u64)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
  pthread_mutex_lock(&ctx->setsol);
  for (u64 nonce = n = 0; nonce < ctx->easiness; nonce++) {
    sipedge(&ctx->sip_ctx, nonce, &u, &v);
    u+=1; v+=1+PARTU;
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
  u64 *cuckoo = ctx->cuckoo;
  u64 us[MAXPATHLEN], vs[MAXPATHLEN], uvpre[2*PRESIP];
  unsigned npre = 0; 
  for (u64 nonce = tp->id; nonce < ctx->easiness; nonce += ctx->nthreads) {
#if PRESIP==0
    sipedge(&ctx->sip_ctx, nonce, us, vs);
#else
    if (!npre)
      for (u64 n = nonce; npre < PRESIP; npre++, n += ctx->nthreads)
        sipedge(&ctx->sip_ctx, n, &uvpre[2*npre], &uvpre[2*npre+1]);
    unsigned i = PRESIP - npre--;
    *us = uvpre[2*i];
    *vs = uvpre[2*i+1];
#endif
    *us+=1; *vs+=1+PARTU;
    u64 u = cuckoo[*us], v = cuckoo[*vs];
    if (u == *vs || v == *us)
      continue; // ignore duplicate edges
#ifdef SHOW
    for (int j=1; j<=SIZE; j++)
      if (!cuckoo[j]) printf("%2d:   ",j);
      else            printf("%2d:%02ld ",j,cuckoo[j]);
    printf(" %lx (%ld,%ld)\n", nonce,*us,*vs);
#endif
    int nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
    if (us[nu] == vs[nv]) {
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
