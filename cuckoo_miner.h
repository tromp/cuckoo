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
#include <atomic>
#include <set>

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two, making twice_set the
// same size as shrinkingset at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif
// L3 cache should exceed NBUCKETS buckets of BUCKETSIZE uint_64_t (0.5MB below)
#ifndef LOGNBUCKETS
#define LOGNBUCKETS	8
#endif
#ifndef BUCKETSIZE
#define BUCKETSIZE	256
#endif

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) == sizeof(twice_set), so
// CUCKOO_SIZE * sizeof(u64) == TWICE_WORDS * sizeof(u32)
// CUCKOO_SIZE * 2 == TWICE_WORDS
// (SIZE >> IDXSHIFT) * 2 == 2 * ONCE_BITS / 32
// SIZE >> IDXSHIFT == HALFSIZE >> PART_BITS >> 5
// IDXSHIFT == 1 + PART_BITS + 5
#define IDXSHIFT (PART_BITS + 6)
#endif
// grow with cube root of size, hardly affected by trimming
#define MAXPATHLEN (8 << (SIZESHIFT/3))

typedef uint32_t u32;

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  std::vector<u32> bits;
  std::vector<u64> cnt;

  shrinkingset(int nthreads) {
    nonce_t nwords = HALFSIZE/32;
    bits.resize(nwords);
    cnt.resize(nthreads);
    cnt[0] = HALFSIZE;
  }
  u64 count() const {
    u64 sum = 0L;
    for (unsigned i=0; i<cnt.size(); i++)
      sum += cnt[i];
    return sum;
  }
  void reset(nonce_t n, int thread) {
    bits[n/32] |= 1 << (n%32);
    cnt[thread]--;
  }
  bool test(node_t n) const {
    return !((bits[n/32] >> (n%32)) & 1);
  }
  u32 block(node_t n) const {
    return ~bits[n/32];
  }
};

#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS (HALFSIZE >> PART_BITS)
#define TWICE_WORDS ((2 * ONCE_BITS) / 32)

class twice_set {
public:
#ifdef ATOMIC
  typedef std::atomic<u32> au32;
  au32 *bits;

  twice_set() {
    assert(bits = (au32 *)calloc(TWICE_WORDS, sizeof(au32)));
  }
  void reset() {
    for (unsigned i=0; i<TWICE_WORDS; i++)
      bits[i].store(0, std::memory_order_relaxed);
  }
  void set(node_t u) {
    node_t idx = u/16;
    u32 bit = 1 << (2 * (u%16));
    u32 old = std::atomic_fetch_or_explicit(&bits[idx], bit   , std::memory_order_relaxed);
    if (old & bit) std::atomic_fetch_or_explicit(&bits[idx], bit<<1, std::memory_order_relaxed);
  }
  u32 test(node_t u) const {
    return (bits[u/16].load(std::memory_order_relaxed) >> (2 * (u%16))) & 2;
  }
#else
  u32 *bits;

  twice_set() {
    assert(bits = (u32 *)calloc(TWICE_WORDS, sizeof(u32)));
  }
  void reset() {
    for (unsigned i=0; i<TWICE_WORDS; i++)
      bits[i] = 0;
  }
  void set(node_t u) {
    node_t idx = u/16;
    u32 bit = 1 << (2 * (u%16));
    u32 old = bits[idx];
    bits[idx] = old | (bit + (old & bit));
  }
  u32 test(node_t u) const {
    return bits[u/16] >> (2 * (u%16)) & 2;
  }
#endif
  ~twice_set() {
    free(bits);
  }
};

#define CUCKOO_SIZE (SIZE >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by SIZESHIFT
#define KEYBITS (64-SIZESHIFT)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

class cuckoo_hash {
public:
  typedef std::atomic<u64> au64;
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
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if ((old >> SIZESHIFT) == (u & KEYMASK)) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
        return;
      }
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
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
  unsigned maxsols;
  std::atomic<unsigned> nsols;
  int nthreads;
  int ntrims;
  pthread_barrier_t barry;

  cuckoo_ctx(const char* header, int n_threads, int n_trims, int max_sols) {
    setheader(&sip_ctx, header);
    nthreads = n_threads;
    alive = new shrinkingset(nthreads);
    cuckoo = 0;
    nonleaf = new twice_set;
    ntrims = n_trims;
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
  int id;
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

#define NBUCKETS	(1 << LOGNBUCKETS)
#define BUCKETSHIFT	(SIZESHIFT-1 - LOGNBUCKETS)
#define NONCESHIFT	(SIZESHIFT-1 - PART_BITS)
#define NODEPARTMASK	(NODEMASK >> PART_BITS)
#define NONCETRUNC	(1L << (64 - NONCESHIFT))

void trim_edges(thread_ctx *tp, int round) {
  cuckoo_ctx *ctx = tp->ctx;
  u64 (* buckets)[BUCKETSIZE] = tp->buckets;
  shrinkingset *alive = ctx->alive;
  twice_set *nonleaf = ctx->nonleaf;
  u32 bucketsizes[NBUCKETS];

  for (int uorv = 0; uorv < 2; uorv++) {
    for (unsigned part = 0; part <= PART_MASK; part++) {
      if (tp->id == 0)
        nonleaf->reset();
      barrier(&ctx->barry);
      for (int qkill = 0; qkill < 2; qkill++) {
        for (int b=0; b < NBUCKETS; b++)
          bucketsizes[b] = 0;
        nonce_t end = qkill ? HALFSIZE*FRAC/100 : HALFSIZE;
        for (nonce_t block = tp->id*32; block < end; block += ctx->nthreads*32) {
          u32 alive32 = alive->block(block); // GLOBAL 1 SEQ
          for (nonce_t nonce = block; alive32; alive32>>=1, nonce++) {
            if (alive32 & 1) {
              node_t u = sipnode(&ctx->sip_ctx, nonce, uorv);
              if ((u & PART_MASK) == part) {
                u32 b = u >> BUCKETSHIFT;
                u32 *bsize = &bucketsizes[b];
                buckets[b][*bsize] = (nonce << NONCESHIFT) | (u >> PART_BITS);
                if (++*bsize == BUCKETSIZE) {
                  *bsize = 0;
                  for (int i=0; i<BUCKETSIZE; i++) {
                    u64 bi = buckets[b][i];
                    if (!qkill) {
                      nonleaf->set(bi & NODEPARTMASK); // GLOBAL 1 RND BUCKETSIZE-1 SEQ 
                    } else {
                      if (!nonleaf->test(bi & NODEPARTMASK)) { // GLOBAL 1 RND BUCKETSIZE-1 SEQ 
                        nonce_t n = (nonce & -NONCETRUNC) | (bi >> NONCESHIFT);
                        alive->reset(n <= nonce ? n : n-NONCETRUNC, tp->id); // GLOBAL SEQ 
                      }
                    }
                  }
                }
              }
            }
          }
        }
        for (int b=0; b < NBUCKETS; b++) {
          int ni = bucketsizes[b];
          for (int i=0; i<ni ; i++) {
            node_t bi = buckets[b][i];
            if (!qkill) {
              nonleaf->set(bi & NODEPARTMASK);
            } else {
              if (!nonleaf->test(bi & NODEPARTMASK)) {
                nonce_t n = (HALFSIZE & -NONCETRUNC) | (bi >> NONCESHIFT);
                alive->reset(n < HALFSIZE  ? n : n-NONCETRUNC, tp->id); // GLOBAL SEQ 
              }
            }
          }
        }
        barrier(&ctx->barry);
      }
      if (tp->id == 0) {
        int load = (int)(100 * alive->count() / CUCKOO_SIZE);
        printf("round %d part %c%d load %d%%\n", round, "UV"[uorv], part, load);
      }
    }
  }
}

int path(cuckoo_hash &cuckoo, node_t u, node_t *us) {
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

typedef std::pair<node_t,node_t> edge;

void solution(cuckoo_ctx *ctx, node_t *us, int nu, node_t *vs, int nv) {
  std::set<edge> cycle;
  unsigned n;
  cycle.insert(edge(*us, *vs));
  while (nu--)
    cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
  while (nv--)
    cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  unsigned soli = std::atomic_fetch_add_explicit(&ctx->nsols, 1U, std::memory_order_relaxed);
  for (nonce_t nonce = n = 0; nonce < HALFSIZE; nonce++)
    if (ctx->alive->test(nonce)) {
      edge e(sipnode(&ctx->sip_ctx, nonce, 0), HALFSIZE+sipnode(&ctx->sip_ctx, nonce, 1));
      if (cycle.find(e) != cycle.end()) {
        ctx->sols[soli][n++] = nonce;
        cycle.erase(e);
      }
    }
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  assert(tp->buckets = (u64 (*)[BUCKETSIZE])calloc(NBUCKETS * BUCKETSIZE, sizeof(u64)));
  cuckoo_ctx *ctx = tp->ctx;

  shrinkingset *alive = ctx->alive;
  int load = 100 * HALFSIZE / CUCKOO_SIZE;
  if (tp->id == 0)
    printf("initial load %d%%\n", load);
  for (int round=1; round <= ctx->ntrims; round++)
    trim_edges(tp, round);
  if (tp->id == 0) {
    load = (int)(100 * alive->count() / CUCKOO_SIZE);
    if (load >= 90) {
      printf("overloaded! exiting...");
      exit(0);
    }
    delete ctx->nonleaf;
    ctx->nonleaf = 0;
    ctx->cuckoo = new cuckoo_hash();
  }
  barrier(&ctx->barry);
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  node_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (nonce_t block = tp->id*32; block < HALFSIZE; block += ctx->nthreads*32) {
    for (nonce_t nonce = block; nonce < block+32 && nonce < HALFSIZE; nonce++) {
      if (alive->test(nonce)) {
        node_t u0, v0;
        sipedge(&ctx->sip_ctx, nonce, &u0, &v0);
        v0 += HALFSIZE;  // make v's different from u's
        node_t u = cuckoo[u0], v = cuckoo[v0];
        if (u == v0 || v == u0)
          continue; // ignore duplicate edges
        us[0] = u0;
        vs[0] = v0;
        int nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
        if (us[nu] == vs[nv]) {
          int min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          int len = nu + nv + 1;
          printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (int)(nonce*100L/HALFSIZE));
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
