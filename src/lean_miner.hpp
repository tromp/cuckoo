// Cuck(at)oo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2019 John Tromp
// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html
// The use of prefetching was suggested by Alexander Peslyak (aka Solar Designer)

#ifdef ATOMIC
#include <atomic>
#endif
#include "cuckatoo.h"
#include "siphashxN.h"
#include "graph.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#ifdef __APPLE__
#include "osx_barrier.h"
#endif
#include <assert.h>

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

typedef uint64_t u64; // save some typing

#ifdef ATOMIC
typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;
#else
typedef u32 au32;
typedef u64 au64;
#endif

// algorithm/performance parameters; assume EDGEBITS < 31

const static u32 NODEBITS = EDGEBITS + 1;
const static word_t NODEMASK = (EDGEMASK << 1) | (word_t)1;

#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two, making twice_set the
// same size as shrinkingset at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef NPREFETCH
// how many prefetches to queue up
// before accessing the memory
// must be a multiple of NSIPHASH
#define NPREFETCH 32
#endif

#ifndef IDXSHIFT
#define IDXSHIFT (PART_BITS + 9)
#endif
#define MAXEDGES (NEDGES >> IDXSHIFT)

const static u32 PART_MASK = (1 << PART_BITS) - 1;
const static u32 NONPART_BITS = EDGEBITS - PART_BITS;
const static u32 NONPART_MASK = (1U << NONPART_BITS) - 1U;

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  bitmap<u64> bmap;
  u64 *cnt;
  u32 nthreads;

  shrinkingset(const u32 nt) : bmap(NEDGES) {
    cnt  = new u64[nt];
    nthreads = nt;
  }
  ~shrinkingset() {
    delete[] cnt;
  }
  void clear() {
    bmap.clear();
    memset(cnt, 0, nthreads * sizeof(u64));
    cnt[0] = NEDGES;
  }
  u64 count() const {
    u64 sum = 0LL;
    for (u32 i=0; i<nthreads; i++)
      sum += cnt[i];
    return sum;
  }
  void reset(word_t n, u32 thread) {
    bmap.set(n);
    cnt[thread]--;
  }
  bool test(word_t n) const {
    return !bmap.test(n);
  }
  u64 block(word_t n) const {
    return ~bmap.block(n);
  }
};

const static u64 CUCKOO_SIZE = NEDGES >> (IDXSHIFT-1); // NNODES >> IDXSHIFT;
const static u64 CUCKOO_MASK = CUCKOO_SIZE - 1;
// number of (least significant) key bits that survives leftshift by NODEBITS
const static u32 KEYBITS = 64-NODEBITS;
const static u64 KEYMASK = (1LL << KEYBITS) - 1;
const static u64 MAXDRIFT = 1LL << (KEYBITS - IDXSHIFT);

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash(void *recycle) {
    cuckoo = (au64 *)recycle;
    memset(cuckoo, 0, CUCKOO_SIZE*sizeof(au64));
  }
  void set(word_t u, word_t v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (word_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
        return;
      }
    }
  }
  word_t operator[](word_t u) const {
    for (word_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 cu = cuckoo[ui];
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (word_t)(cu & NODEMASK);
      }
    }
  }
};

class cuckoo_ctx {
public:
  siphash_keys sip_keys;
  shrinkingset alive;
  bitmap<word_t> nonleaf;
  graph<word_t> cg;
  u32 nonce;
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  pthread_barrier_t barry;

  cuckoo_ctx(u32 n_threads, u32 n_trims, u32 max_sols) : alive(n_threads), nonleaf(NEDGES),
      cg(MAXEDGES, MAXSOLS, (void *)nonleaf.bits) {
    assert(cg.bytes() <= NEDGES/8); // check that graph cg can fit in share nonleaf's memory
    nthreads = n_threads;
    ntrims = n_trims;
    int err = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
    nsols = 0;
  }
  void setheadernonce(char* headernonce, const u32 len, const u32 nce) {
    nonce = nce;
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &sip_keys);
    alive.clear(); // set all edges to be alive
    nsols = 0;
  }
  ~cuckoo_ctx() {
  }
  void prefetch(const u64 *hashes, const u32 part) const {
    for (u32 i=0; i < NSIPHASH; i++) {
      u32 u = hashes[i] & EDGEMASK;
      if ((u >> NONPART_BITS) == part) {
        nonleaf.prefetch(u & NONPART_MASK);
      }
    }
  }
  void node_deg(const u64 *hashes, const u32 nsiphash, const u32 part) {
    for (u32 i=0; i < nsiphash; i++) {
      u32 u = hashes[i] & EDGEMASK;
      if ((u >> NONPART_BITS) == part) {
        nonleaf.set(u & NONPART_MASK);
      }
    }
  }
  void kill(const u64 *hashes, const u64 *indices, const u32 nsiphash, const u32 part, const u32 id) {
    for (u32 i=0; i < nsiphash; i++) {
      u32 u = hashes[i] & EDGEMASK;
      if ((u >> NONPART_BITS) == part && !nonleaf.test((u & NONPART_MASK) ^ 1)) {
        alive.reset(indices[i]/2, id);
      }
    }
  }
  void count_node_deg(const u32 id, const u32 uorv, const u32 part) {
    alignas(64) u64 indices[NSIPHASH];
    alignas(64) u64 hashes[NPREFETCH];
  
    memset(hashes, 0, NPREFETCH * sizeof(u64)); // allow many nonleaf.set(0) to reduce branching
    u32 nidx = 0;
    for (word_t block = id*64; block < NEDGES; block += nthreads*64) {
      u64 alive64 = alive.block(block);
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        indices[nidx++ % NSIPHASH] = 2*nonce + uorv;
        if (nidx % NSIPHASH == 0) {
          node_deg(hashes+nidx-NSIPHASH, NSIPHASH, part);
          siphash24xN(&sip_keys, indices, hashes+nidx-NSIPHASH);
          prefetch(hashes+nidx-NSIPHASH, part);
          nidx %= NPREFETCH;
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    node_deg(hashes, NPREFETCH, part);
    if (nidx % NSIPHASH != 0) {
      siphash24xN(&sip_keys, indices, hashes+(nidx&-NSIPHASH));
      node_deg(hashes+(nidx&-NSIPHASH), nidx%NSIPHASH, part);
    }
  }
  void kill_leaf_edges(const u32 id, const u32 uorv, const u32 part) {
    alignas(64) u64 indices[NPREFETCH];
    alignas(64) u64 hashes[NPREFETCH];
  
    for (int i=0; i < NPREFETCH; i++)
      hashes[i] = 1; // allow many nonleaf.test(0) to reduce branching
    u32 nidx = 0;
    for (word_t block = id*64; block < NEDGES; block += nthreads*64) {
      u64 alive64 = alive.block(block);
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        indices[nidx++] = 2*nonce + uorv;
        if (nidx % NSIPHASH == 0) {
          siphash24xN(&sip_keys, indices+nidx-NSIPHASH, hashes+nidx-NSIPHASH);
          prefetch(hashes+nidx-NSIPHASH, part);
          nidx %= NPREFETCH;
          kill(hashes+nidx, indices+nidx, NSIPHASH, part, id);
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    const u32 pnsip = nidx & -NSIPHASH;
    if (pnsip != nidx) {
      siphash24xN(&sip_keys, indices+pnsip, hashes+pnsip);
    }
    kill(hashes, indices, nidx, part, id);
    const u32 nnsip = pnsip + NSIPHASH;
    kill(hashes+nnsip, indices+nnsip, NPREFETCH-nnsip, part, id);
  }
};

typedef struct {
  u32 id;
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

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  shrinkingset &alive = ctx->alive;
  if (tp->id == 0)
    printf("initial size %d\n", NEDGES);
  u32 size = alive.count(); assert(size == NEDGES);
  for (u32 round=1; size > MAXEDGES; round++) {
    if (tp->id == 0) printf("round %2d partition sizes", round);
    for (u32 uorv = 0; uorv < 2; uorv++) {
      for (u32 part = 0; part <= PART_MASK; part++) {
        if (tp->id == 0)
          ctx->nonleaf.clear(); // clear all counts
        barrier(&ctx->barry);
        ctx->count_node_deg(tp->id,uorv,part);
        barrier(&ctx->barry);
        ctx->kill_leaf_edges(tp->id,uorv,part);
        barrier(&ctx->barry);
        if (tp->id == 0) {
          u32 size = alive.count();
          printf(" %c%d %d", "UV"[uorv], part, size);
        }
      }
    }
    if (tp->id == 0) printf("\n");
  }
  if (tp->id == 0) {
    u32 load = (u32)(100LL * alive.count() / CUCKOO_SIZE);
    printf("nonce %d: %d trims completed  final load %d%%\n", ctx->nonce, ctx->ntrims, load);
  }
  else pthread_exit(NULL);
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  word_t us[MAXPATHLEN], vs[MAXPATHLEN];
  for (word_t block = 0; block < NEDGES; block += 64) {
    u64 alive64 = alive.block(block);
    for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
      u32 ffs = __builtin_ffsll(alive64);
      nonce += ffs; alive64 >>= ffs;
      word_t u0=sipnode_(&ctx->sip_keys, nonce, 0), v0=sipnode_(&ctx->sip_keys, nonce, 1);
      if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
        u32 nu = path(cuckoo, u0, us), nv = path(cuckoo, v0, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("%4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100LL/NEDGES));
          if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
            ctx->solution(us, nu, vs, nv);
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
      if (ffs & 64) break; // can't shift by 64
    }
  }
  pthread_exit(NULL);
  return 0;
}
