// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp
// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include "cuckoo.h"
#include "osx_barrier.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <utility>
#include <bitset>
#include <atomic>
#include <set>
#include <assert.h>

// make EASINESS a multiple of 64 for bitset block access 
#define EASINESS (SIZE/2 & -64)
// algorithm parameters
#define MAXPATHLEN 6144
#ifndef PRESIP
#define PRESIP 1024
#endif
#ifndef PART_BITS
#define PART_BITS 1
#endif
#ifndef REDSHIFT
#define REDSHIFT 8
#endif
#define REDMASK ((1 << REDSHIFT) - 1)
#define CUCKOO_SIZE ((1+SIZE+REDMASK) >> REDSHIFT)
#define PART_MASK ((1<<PART_BITS)-1)
#define ONCE_SIZE ((PARTU + PART_MASK) >> PART_BITS)
#define TWICE_SIZE (2*ONCE_SIZE)

class twice_set {
public:
  std::bitset<TWICE_SIZE> both;

  void reset() {
    both.reset();
  }

  void set(node_t u) {
    if (both.test(u*=2))
      both.set(u+1);
    else both.set(u);
  }

  bool test(node_t u) {
    return both.test(2*u+1);
  }
};

typedef std::atomic<uint64_t> au64;

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash() {
    cuckoo = (au64 *)calloc(CUCKOO_SIZE, sizeof(au64));
    assert(cuckoo);
  }

  ~cuckoo_hash() {
    free(cuckoo);
  }

  void set(node_t u, node_t v) {
    node_t ui = u >> REDSHIFT;
    u64 old = 0, nuw = v << REDSHIFT | (u & REDMASK);;
    for (;;) {
      if (cuckoo[ui].compare_exchange_strong(old, nuw, std::memory_order_relaxed))
        return;
      if (((u^old) & REDMASK) == 0) {
        cuckoo[ui] = nuw;
        return;
      }
      ui = (ui+1) % CUCKOO_SIZE;
    }
  }

  node_t get(node_t u) {
    node_t ui = u >> REDSHIFT;
    for (;;) {
      u64 cu = cuckoo[ui];
      if (!cu)
        return 0;
      if (((u^cu) & REDMASK) == 0)
        return (node_t)(cu >> REDSHIFT);
      ui = (ui+1) % CUCKOO_SIZE;
    }
  }
};

class cuckoo_ctx {
public:
  siphash_ctx sip_ctx;
  node_t easiness;
  std::bitset<EASINESS> *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;
  nonce_t (*sols)[PROOFSIZE];
  unsigned maxsols;
  std::atomic<unsigned> nsols;
  int nthreads;
  int ntrims;
  pthread_barrier_t barry;
  // pthread_mutex_t setsol;

  cuckoo_ctx() {
    alive = new std::bitset<EASINESS>;
    nonleaf = new twice_set;
  }

  ~cuckoo_ctx() {
    delete alive;
    delete nonleaf;
  }
};

typedef struct {
  int id;
  pthread_t thread;
  cuckoo_ctx *ctx;
} thread_ctx;

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

#define FORALL_LIVE_NONCES(NONCE) \
  for (nonce_t block = tp->id*64; block < ctx->easiness; block += ctx->nthreads*64) {\
    for (nonce_t NONCE = block; NONCE < block+64; NONCE++) {\
      if (ctx->alive->test(NONCE))

void trim_edges(thread_ctx *tp, unsigned part) {
  cuckoo_ctx *ctx = tp->ctx;
  ctx->nonleaf->reset();
  FORALL_LIVE_NONCES(nonce) {
    node_t u = sipedgeu(&ctx->sip_ctx, nonce);
    if ((u & PART_MASK) == part)
      ctx->nonleaf->set(u >> PART_BITS);
  }}}
  barrier(&ctx->barry);
  FORALL_LIVE_NONCES(nonce) {
    node_t u = sipedgeu(&ctx->sip_ctx, nonce);
    if ((u & PART_MASK) == part && !ctx->nonleaf->test(u >> PART_BITS))
    ctx->alive->reset(nonce);
  }}}
  barrier(&ctx->barry);
  ctx->nonleaf->reset();
  FORALL_LIVE_NONCES(nonce) {
    node_t v = sipedgev(&ctx->sip_ctx, nonce);
    if ((v & PART_MASK) == part)
      ctx->nonleaf->set(v >> PART_BITS);
  }}}
  barrier(&ctx->barry);
  FORALL_LIVE_NONCES(nonce) {
    node_t v = sipedgev(&ctx->sip_ctx, nonce);
    if ((v & PART_MASK) == part && !ctx->nonleaf->test(v >> PART_BITS))
      ctx->alive->reset(nonce);
  }}}
  barrier(&ctx->barry);
}

int path(cuckoo_hash *cuckoo, node_t u, node_t *us) {
  int nu;
  for (nu = 0; u; u = cuckoo->get(u)) {
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

typedef std::pair<node_t,node_t> edge;

void solution(cuckoo_ctx *ctx, node_t *us, int nu, node_t *vs, int nv) {
  std::set<edge> cycle;
  node_t u, v;
  unsigned n;
  cycle.insert(edge(*us, *vs));
  while (nu--)
    cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
  while (nv--)
    cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  // pthread_mutex_lock(&ctx->setsol);
  unsigned soli = std::atomic_fetch_add_explicit(&ctx->nsols, 1U, std::memory_order_relaxed);
  for (nonce_t nonce = n = 0; nonce < ctx->easiness; nonce++) {
    if (ctx->alive->test(nonce)) {
      sipedge(&ctx->sip_ctx, nonce, &u, &v);
      edge e(u+1, v+1+PARTU);
      if (cycle.find(e) != cycle.end()) {
        ctx->sols[soli][n++] = nonce;
        cycle.erase(e);
      }
    }
  }
  // pthread_mutex_unlock(&ctx->setsol);
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  ctx->alive->set();
  for (int round=0; round < ctx->ntrims; round++) {
    u64 cnt = ctx->alive->count();
    if (tp->id == 0)
      printf("round %d: load %d%%\n", round, (int)(6400L*cnt/ctx->easiness));
    for (unsigned part = 0; part <= PART_MASK; part++)
      trim_edges(tp, part);
  }
  if (tp->id == 0) {
    delete ctx->nonleaf;
    ctx->cuckoo = new cuckoo_hash();
  }
  barrier(&ctx->barry);
  cuckoo_hash *cuckoo = ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  FORALL_LIVE_NONCES(nonce) {
    node_t u0, v0;
    sipedge(&ctx->sip_ctx, nonce, &u0, &v0);
    u0 += 1        ;  // make non-zero
    v0 += 1 + PARTU;  // make v's different from u's
    node_t u = cuckoo->get(u0), v = cuckoo->get(v0);
    if (u == v0 || v == u0)
      continue; // ignore duplicate edges
    us[0] = u0;
    vs[0] = v0;
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
        cuckoo->set(us[nu+1], us[nu]);
      cuckoo->set(u0, v0);
    } else {
      while (nv--)
        cuckoo->set(vs[nv+1], vs[nv]);
      cuckoo->set(v0, u0);
    }
  }}}
  pthread_exit(NULL);
}
