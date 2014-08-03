// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp
// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include "cuckoo.h"
#ifdef __APPLE__
#include "osx_barrier.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <vector>
#ifdef ATOMIC
#include <atomic>
typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;
#else
typedef u32 au32;
typedef u64 au64;
#endif
#include <set>

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition vertex set to save memory
#define PART_BITS 8
#endif

#ifndef IDXSHIFT
#define IDXSHIFT (PART_BITS - 4)
#endif
// grow with cube root of size, hardly affected by trimming
#define MAXPATHLEN (8 << (SIZESHIFT/3))

#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS (HALFSIZE >> PART_BITS)

#define CUCKOO_SIZE (SIZE >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by SIZESHIFT
#define KEYBITS (64-SIZESHIFT)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash() {
    assert(cuckoo = (au64 *)calloc(CUCKOO_SIZE, sizeof(au64)));
  }
  ~cuckoo_hash() {
    free(cuckoo);
  }
  void set(node_t u, node_t v) {
    u64 niew = (u64)u << SIZESHIFT | v;
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#ifdef ATOMIC
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if ((old >> SIZESHIFT) == (u & KEYMASK)) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
#else
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> SIZESHIFT) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
#endif
        return;
      }
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#ifdef ATOMIC
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
#else
      u64 cu = cuckoo[ui];
#endif
      if (!cu)
        return 0;
      if ((cu >> SIZESHIFT) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (node_t)(cu & (SIZE-1));
      }
    }
  }
};

class cuckoo_ctx {
public:
  siphash_ctx sip_ctx;
  shrinkingset *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;
  nonce_t (*sols)[PROOFSIZE];
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  pthread_barrier_t barry;

  cuckoo_ctx(const char* header, u32 n_threads, u32 max_sols) {
    setheader(&sip_ctx, header);
    nthreads = n_threads;
    alive = new shrinkingset(nthreads);
    cuckoo = 0;
    nonleaf = new twice_set;
    assert(pthread_barrier_init(&barry, NULL, nthreads) == 0);
    assert(sols = (nonce_t (*)[PROOFSIZE])calloc(maxsols = max_sols, PROOFSIZE*sizeof(nonce_t)));
    nsols = 0;
  }
  ~cuckoo_ctx() {
    delete alive;
    if (nonleaf)
      delete nonleaf;
    if (cuckoo)
      delete cuckoo;
  }
};

typedef struct {
  u32 id;
  pthread_t thread;
  cuckoo_ctx *ctx;
  u64 (* buckets)[BUCKETSIZE];
} thread_ctx;

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

u32 path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (!~nu)
        printf("maximum path length exceeded\n");
      else printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu] = u;
  }
  return nu;
}

typedef std::pair<node_t,node_t> edge;

void solution(cuckoo_ctx *ctx, node_t *us, u32 nu, node_t *vs, u32 nv) {
  std::set<edge> cycle;
  u32 n;
  cycle.insert(edge(*us, *vs));
  while (nu--)
    cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
  while (nv--)
    cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
#ifdef ATOMIC
  u32 soli = std::atomic_fetch_add_explicit(&ctx->nsols, 1U, std::memory_order_relaxed);
#else
  u32 soli = ctx->nsols++;
#endif
  for (nonce_t nonce = n = 0; nonce < HALFSIZE; nonce++)
    if (ctx->alive->test(nonce)) {
      edge e(sipnode(&ctx->sip_ctx, nonce, 0), HALFSIZE+sipnode(&ctx->sip_ctx, nonce, 1));
      if (cycle.find(e) != cycle.end()) {
        ctx->sols[soli][n++] = nonce;
        if (PROOFSIZE > 2)
          cycle.erase(e);
      }
    }
  assert(n==PROOFSIZE);
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  if (tp->id == 0)
    ctx->cuckoo = new cuckoo_hash();
  barrier(&ctx->barry);
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (nonce_t block = tp->id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    for (nonce_t nonce = block; nonce < block+32 && nonce < HALFSIZE; nonce++) {
      if (alive->test(nonce)) {
        node_t u0, v0;
        sipedge(&ctx->sip_ctx, nonce, &u0, &v0);
        v0 += HALFSIZE;  // make v's different from u's
        node_t u = cuckoo[us[0] = u0], v = cuckoo[vs[0] = v0];
        u32 nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100L/HALFSIZE));
          if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
            solution(ctx, us, nu, vs, nv);
          continue;
        }
        if (nu < nv) {
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
  free(tp->buckets);
  pthread_exit(NULL);
}
