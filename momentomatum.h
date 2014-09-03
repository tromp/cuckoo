// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp
// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#define PROOFSIZE 2
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

#ifndef SAVEMEM_BITS
#define SAVEMEM_BITS 6
#endif

#ifndef IDXSHIFT
#define IDXSHIFT SAVEMEM_BITS
#endif
#ifndef PART_BITS
// #bits used to partition vertex set to save memory
#define PART_BITS (IDXSHIFT+LOGPROOFSIZE)
#endif
#define NPARTS (1<<PART_BITS)

#define ONCE_BITS (HALFSIZE >> PART_BITS)
#define TWICE_WORDS ((2 * ONCE_BITS) / 32)

#define PART_SIZE (HALFSIZE >> PART_BITS)
#define NPARTS (1 << PART_BITS)
#define PART_MASK (NPARTS - 1)

class cuckoo_ctx {
public:
  siphash_ctx sip_ctx;
  au32 *uvmap;
  u32 nparts;
  nonce_t (*sols)[PROOFSIZE];
  u32 nthreads;
  pthread_barrier_t barry;

  cuckoo_ctx(const char* header, u32 n_threads, u32 n_parts) {
    setheader(&sip_ctx, header);
    nthreads = n_threads;
    nparts = n_parts;
    assert(uvmap = (au32 *)calloc(PART_SIZE, sizeof(au32)));
    assert(pthread_barrier_init(&barry, NULL, nthreads) == 0);
  }
  ~cuckoo_ctx() {
    delete uvmap;
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

typedef std::pair<node_t,node_t> edge;

void solution(cuckoo_ctx *ctx, node_t *us, u32 nu, node_t *vs, u32 nv) {
  std::set<edge> cycle;
  u32 n;
  cycle.insert(edge(*us, *vs));
  while (nu--)
    cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
  while (nv--)
    cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  printf("Solution: ");
  for (nonce_t nonce = n = 0; nonce < HALFSIZE; nonce++) {
    edge e(sipnode(&ctx->sip_ctx, nonce, 0), HALFSIZE+sipnode(&ctx->sip_ctx, nonce, 1));
    if (cycle.find(e) != cycle.end()) {
      printf("%x%c", nonce, ++n == PROOFSIZE?'\n':' ');
      if (PROOFSIZE > 2)
        cycle.erase(e);
    }
  }
  assert(n==PROOFSIZE);
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  for (node_t vpart=0; vpart < ctx->nparts; vpart++) {
    barrier(&ctx->barry);
    for (nonce_t nonce = tp->id; nonce < HALFSIZE; nonce += ctx->nthreads) {
      node_t u0 = sipnode(&ctx->sip_ctx, nonce, 0);
      node_t uv = uvmap[u >> PART_BITS];
      node_t v0 = sipnode(&ctx->sip_ctx, nonce, 1);
      if (uv != 0) {
        printf("Solution: %x %c\n", nonce, ++n == PROOFSIZE?'\n':' ');
        continue;
      }
      uvmap[u0] = v0;
    }
    barrier(&ctx->barry);
    if (tp->id == 0) {
      memset(uvmap, 0, PART_SIZE*sizeof(au32));
    }
  }
  pthread_exit(NULL);
}
