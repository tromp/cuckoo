// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp
// The edge-trimming time-memory trade-off is due to Dave Anderson:
// The use of prefetching was suggested by Alexander Peslyak (aka Solar Designer)
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include "cuckoo.h"

#ifdef SIMD
// Scalar and AVX2 implementations of siphash24 for 8B nonces
// gcc -march=native -std=gnu99 -O3 -g -Wall -Wextra siphash.c -o siphash -lcrypto

#include <x86intrin.h>

typedef __m256i ymm_t;
typedef __m128i xmm_t;

typedef struct {
  ymm_t v[4];
} ymm_siphash_ctx;

#define YMM_ROTATE_LEFT(vec, bits) \
  _mm256_or_si256(_mm256_slli_epi64(vec, bits), _mm256_srli_epi64(vec, (64 - bits)))

// swapping high and low is equivalent to rotating bits 32
#define YMM_SWAP_HIGH_LOW_32(vec) \
  _mm256_shuffle_epi32(vec, _MM_SHUFFLE(2, 3, 0, 1))

// Starting Byte Order: {0x0706050403020100, 0x0F0E0D0C0B0A0908}
static ymm_t ymmRotateLeft16 = {0x0504030201000706, 0x0D0C0B0A09080F0E,
                                0x0504030201000706, 0x0D0C0B0A09080F0E};
#define YMM_ROTATE_LEFT_16(vec) \
  _mm256_shuffle_epi8(vec, ymmRotateLeft16)

// commented out calculations for first round are done in advance
#define YMM_SIP_FIRST_ROUND(VEC)                     \
  VEC.v[2] = _mm256_add_epi64(VEC.v[2], VEC.v[3]); \
  VEC.v[3] = YMM_ROTATE_LEFT_16(VEC.v[3]);         \
  VEC.v[3] = _mm256_xor_si256(VEC.v[2], VEC.v[3]); \
  VEC.v[2] = _mm256_add_epi64(VEC.v[2], VEC.v[1]); \
  VEC.v[1] = YMM_ROTATE_LEFT(VEC.v[1], 17);        \
  VEC.v[0] = _mm256_add_epi64(VEC.v[0], VEC.v[3]); \
  VEC.v[3] = YMM_ROTATE_LEFT(VEC.v[3], 21);        \
  VEC.v[1] = _mm256_xor_si256(VEC.v[2], VEC.v[1]); \
  VEC.v[2] = YMM_SWAP_HIGH_LOW_32(VEC.v[2]);       \
  VEC.v[3] = _mm256_xor_si256(VEC.v[0], VEC.v[3])

// full round on 4x64-bit values in a 256-bit ymm_t vector
#define YMM_SIP_FULL_ROUND(VEC)      \
  VEC.v[0] = _mm256_add_epi64(VEC.v[0], VEC.v[1]); \
  VEC.v[1] = YMM_ROTATE_LEFT(VEC.v[1], 13);        \
  VEC.v[2] = _mm256_add_epi64(VEC.v[2], VEC.v[3]); \
  VEC.v[3] = YMM_ROTATE_LEFT_16(VEC.v[3]);         \
  VEC.v[1] = _mm256_xor_si256(VEC.v[0], VEC.v[1]); \
  VEC.v[0] = YMM_SWAP_HIGH_LOW_32(VEC.v[0]);       \
  VEC.v[3] = _mm256_xor_si256(VEC.v[2], VEC.v[3]); \
  VEC.v[2] = _mm256_add_epi64(VEC.v[2], VEC.v[1]); \
  VEC.v[1] = YMM_ROTATE_LEFT(VEC.v[1], 17);        \
  VEC.v[0] = _mm256_add_epi64(VEC.v[0], VEC.v[3]); \
  VEC.v[3] = YMM_ROTATE_LEFT(VEC.v[3], 21);        \
  VEC.v[1] = _mm256_xor_si256(VEC.v[2], VEC.v[1]); \
  VEC.v[2] = YMM_SWAP_HIGH_LOW_32(VEC.v[2]);       \
  VEC.v[3] = _mm256_xor_si256(VEC.v[0], VEC.v[3])

ymm_t ymm_siphash24(ymm_siphash_ctx VEC, ymm_t vecNonces) {
  ymm_t vec0xFF = {0xFF, 0xFF, 0xFF, 0xFF};

  VEC.v[3] = _mm256_xor_si256(VEC.v[3], vecNonces);
  YMM_SIP_FIRST_ROUND(VEC);
  YMM_SIP_FULL_ROUND(VEC);
  VEC.v[2] = _mm256_xor_si256(VEC.v[2], vec0xFF);
  VEC.v[0] = _mm256_xor_si256(VEC.v[0], vecNonces);
  YMM_SIP_FULL_ROUND(VEC);
  YMM_SIP_FULL_ROUND(VEC);
  YMM_SIP_FULL_ROUND(VEC);
  YMM_SIP_FULL_ROUND(VEC);

  VEC.v[0] = _mm256_xor_si256(VEC.v[0], VEC.v[1]);
  VEC.v[2] = _mm256_xor_si256(VEC.v[2], VEC.v[3]);
  VEC.v[0] = _mm256_xor_si256(VEC.v[0], VEC.v[2]);

  return VEC.v[0];
}

// Y2X routines work on 2 4x64-bit vectors simultaneously
#define Y2X_SIP_FIRST_ROUND(VEC0, VEC1) \
  VEC0.v[2] = _mm256_add_epi64(VEC0.v[2], VEC0.v[3]); \
  VEC1.v[2] = _mm256_add_epi64(VEC1.v[2], VEC1.v[3]); \
  VEC0.v[3] = YMM_ROTATE_LEFT_16(VEC0.v[3]);          \
  VEC1.v[3] = YMM_ROTATE_LEFT_16(VEC1.v[3]);          \
  VEC0.v[3] = _mm256_xor_si256(VEC0.v[2], VEC0.v[3]); \
  VEC1.v[3] = _mm256_xor_si256(VEC1.v[2], VEC1.v[3]); \
  VEC0.v[2] = _mm256_add_epi64(VEC0.v[2], VEC0.v[1]); \
  VEC1.v[2] = _mm256_add_epi64(VEC1.v[2], VEC1.v[1]); \
  VEC0.v[1] = YMM_ROTATE_LEFT(VEC0.v[1], 17);         \
  VEC1.v[1] = YMM_ROTATE_LEFT(VEC1.v[1], 17);         \
  VEC0.v[0] = _mm256_add_epi64(VEC0.v[0], VEC0.v[3]); \
  VEC1.v[0] = _mm256_add_epi64(VEC1.v[0], VEC1.v[3]); \
  VEC0.v[3] = YMM_ROTATE_LEFT(VEC0.v[3], 21);         \
  VEC1.v[3] = YMM_ROTATE_LEFT(VEC1.v[3], 21);         \
  VEC0.v[1] = _mm256_xor_si256(VEC0.v[2], VEC0.v[1]); \
  VEC1.v[1] = _mm256_xor_si256(VEC1.v[2], VEC1.v[1]); \
  VEC0.v[2] = YMM_SWAP_HIGH_LOW_32(VEC0.v[2]);        \
  VEC1.v[2] = YMM_SWAP_HIGH_LOW_32(VEC1.v[2]);        \
  VEC0.v[3] = _mm256_xor_si256(VEC0.v[0], VEC0.v[3]); \
  VEC1.v[3] = _mm256_xor_si256(VEC1.v[0], VEC1.v[3])

#define Y2X_SIP_FULL_ROUND(VEC0, VEC1)  \
  VEC0.v[0] = _mm256_add_epi64(VEC0.v[0], VEC0.v[1]); \
  VEC1.v[0] = _mm256_add_epi64(VEC1.v[0], VEC1.v[1]); \
  VEC0.v[1] = YMM_ROTATE_LEFT(VEC0.v[1], 13);         \
  VEC1.v[1] = YMM_ROTATE_LEFT(VEC1.v[1], 13);         \
  VEC0.v[2] = _mm256_add_epi64(VEC0.v[2], VEC0.v[3]); \
  VEC1.v[2] = _mm256_add_epi64(VEC1.v[2], VEC1.v[3]); \
  VEC0.v[3] = YMM_ROTATE_LEFT_16(VEC0.v[3]);          \
  VEC1.v[3] = YMM_ROTATE_LEFT_16(VEC1.v[3]);          \
  VEC0.v[1] = _mm256_xor_si256(VEC0.v[0], VEC0.v[1]); \
  VEC1.v[1] = _mm256_xor_si256(VEC1.v[0], VEC1.v[1]); \
  VEC0.v[0] = YMM_SWAP_HIGH_LOW_32(VEC0.v[0]);        \
  VEC1.v[0] = YMM_SWAP_HIGH_LOW_32(VEC1.v[0]);        \
  VEC0.v[3] = _mm256_xor_si256(VEC0.v[2], VEC0.v[3]); \
  VEC1.v[3] = _mm256_xor_si256(VEC1.v[2], VEC1.v[3]); \
  VEC0.v[2] = _mm256_add_epi64(VEC0.v[2], VEC0.v[1]); \
  VEC1.v[2] = _mm256_add_epi64(VEC1.v[2], VEC1.v[1]); \
  VEC0.v[1] = YMM_ROTATE_LEFT(VEC0.v[1], 17);         \
  VEC1.v[1] = YMM_ROTATE_LEFT(VEC1.v[1], 17);         \
  VEC0.v[0] = _mm256_add_epi64(VEC0.v[0], VEC0.v[3]); \
  VEC1.v[0] = _mm256_add_epi64(VEC1.v[0], VEC1.v[3]); \
  VEC0.v[3] = YMM_ROTATE_LEFT(VEC0.v[3], 21);         \
  VEC1.v[3] = YMM_ROTATE_LEFT(VEC1.v[3], 21);         \
  VEC0.v[1] = _mm256_xor_si256(VEC0.v[2], VEC0.v[1]); \
  VEC1.v[1] = _mm256_xor_si256(VEC1.v[2], VEC1.v[1]); \
  VEC0.v[2] = YMM_SWAP_HIGH_LOW_32(VEC0.v[2]);        \
  VEC1.v[2] = YMM_SWAP_HIGH_LOW_32(VEC1.v[2]);        \
  VEC0.v[3] = _mm256_xor_si256(VEC0.v[0], VEC0.v[3]); \
  VEC1.v[3] = _mm256_xor_si256(VEC1.v[0], VEC1.v[3])

void ymm_siphash24_2x(ymm_siphash_ctx *vec, u64 *sinput, u64 *soutput) {
  ymm_t vecNonce0 = _mm256_loadu_si256((ymm_t *)&sinput[0]);
  ymm_t vecNonce1 = _mm256_loadu_si256((ymm_t *)&sinput[4]);

  ymm_t vec0xFF = {0xFF, 0xFF, 0xFF, 0xFF};

  ymm_siphash_ctx vec0, vec1;
  vec0.v[0] = vec1.v[0] = vec->v[0];
  vec0.v[1] = vec1.v[1] = vec->v[1];
  vec0.v[2] = vec1.v[2] = vec->v[2];
  vec0.v[3] = _mm256_xor_si256(vec->v[3], vecNonce0);
  vec1.v[3] = _mm256_xor_si256(vec->v[3], vecNonce1);

  Y2X_SIP_FIRST_ROUND(vec0, vec1); Y2X_SIP_FULL_ROUND(vec0, vec1);

  vec0.v[2] = _mm256_xor_si256(vec0.v[2], vec0xFF);
  vec1.v[2] = _mm256_xor_si256(vec1.v[2], vec0xFF);

  vec0.v[0] = _mm256_xor_si256(vec0.v[0], vecNonce0);
  vec1.v[0] = _mm256_xor_si256(vec1.v[0], vecNonce1);

  Y2X_SIP_FULL_ROUND(vec0, vec1); Y2X_SIP_FULL_ROUND(vec0, vec1);
  Y2X_SIP_FULL_ROUND(vec0, vec1); Y2X_SIP_FULL_ROUND(vec0, vec1);

  vec0.v[0] = _mm256_xor_si256(vec0.v[0], vec0.v[1]);
  vec1.v[0] = _mm256_xor_si256(vec1.v[0], vec1.v[1]);

  vec0.v[2] = _mm256_xor_si256(vec0.v[2], vec0.v[3]);
  vec1.v[2] = _mm256_xor_si256(vec1.v[2], vec1.v[3]);

  vec0.v[0] = _mm256_xor_si256(vec0.v[0], vec0.v[2]);
  vec1.v[0] = _mm256_xor_si256(vec1.v[0], vec1.v[2]);

  _mm256_storeu_si256((ymm_t *)&soutput[0], vec0.v[0]);
  _mm256_storeu_si256((ymm_t *)&soutput[4], vec1.v[0]);
}

#define YMM_SET_ALL(x) _mm256_broadcastq_epi64(_mm_cvtsi64_si128(x))

// FUTURE: add alternative to calculate same nonce with 4 different headers (transpose vecs)
void ymm_sip_header(ymm_siphash_ctx *VEC, const char *header) {
  uint64_t generated[4];
  SHA256(header, strlen(header), &generated);

  // set all elements within a vector to same value
  VEC->v[0] = YMM_SET_ALL(generated[0] ^ 0x736f6d6570736575ULL);
  VEC->v[1] = YMM_SET_ALL(generated[1] ^ 0x646f72616e646f6dULL);
  VEC->v[2] = YMM_SET_ALL(generated[0] ^ 0x6c7967656e657261ULL);
  VEC->v[3] = YMM_SET_ALL(generated[1] ^ 0x7465646279746573ULL);

  // do the invariant start of the first sip round 
  VEC->v[0] = _mm256_add_epi64(VEC->v[0], VEC->v[1]);
  VEC->v[1] = _mm256_xor_si256(VEC->v[0], YMM_ROTATE_LEFT(VEC->v[1], 13));
  VEC->v[0] = YMM_SWAP_HIGH_LOW_32(VEC->v[0]);
}
#endif

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
#if SIZESHIFT <= 32
typedef u32 nonce_t;
typedef u32 node_t;
#else
typedef u64 nonce_t;
typedef u64 node_t;
#endif
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

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  std::vector<u64> bits;
  std::vector<u64> cnt;

  shrinkingset(u32 nthreads) {
    nonce_t nwords = HALFSIZE/64;
    bits.resize(nwords);
    cnt.resize(nthreads);
    cnt[0] = HALFSIZE;
  }
  u64 count() const {
    u64 sum = 0LL;
    for (u32 i=0; i<cnt.size(); i++)
      sum += cnt[i];
    return sum;
  }
  void reset(nonce_t n, u32 thread) {
    bits[n/64] |= 1LL << (n%64);
    cnt[thread]--;
  }
  bool test(node_t n) const {
    return !((bits[n/64] >> (n%64)) & 1LL);
  }
  u64 block(node_t n) const {
    return ~bits[n/64];
  }
};

#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS (HALFSIZE >> PART_BITS)
#define TWICE_WORDS ((2 * ONCE_BITS) / 32)

class twice_set {
public:
  au32 *bits;

  twice_set() {
    bits = (au32 *)calloc(TWICE_WORDS, sizeof(au32));
    assert(bits != 0);
  }
  void reset() {
    memset(bits, 0, TWICE_WORDS*sizeof(au32));
  }
  void prefetch(node_t u) const {
#ifdef PREFETCH
    __builtin_prefetch((const void *)(&bits[u/16]), /*READ=*/0, /*TEMPORAL=*/0);
#endif
  }
  void set(node_t u) {
    node_t idx = u/16;
    u32 bit = 1 << (2 * (u%16));
#ifdef ATOMIC
    u32 old = std::atomic_fetch_or_explicit(&bits[idx], bit, std::memory_order_relaxed);
    if (old & bit) std::atomic_fetch_or_explicit(&bits[idx], bit<<1, std::memory_order_relaxed);
  }
  u32 test(node_t u) const {
    return (bits[u/16].load(std::memory_order_relaxed) >> (2 * (u%16))) & 2;
  }
#else
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
#define KEYMASK ((1LL << KEYBITS) - 1)
#define MAXDRIFT (1LL << (KEYBITS - IDXSHIFT))

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash() {
    cuckoo = (au64 *)calloc(CUCKOO_SIZE, sizeof(au64));
    assert(cuckoo != 0);
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
#ifdef SIMD
  ymm_siphash_ctx ymm_sip_ctx;
#endif
  shrinkingset *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;
  nonce_t (*sols)[PROOFSIZE];
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  pthread_barrier_t barry;

  cuckoo_ctx(const char* header, u32 n_threads, u32 n_trims, u32 max_sols) {
    setheader(&sip_ctx, header);
#ifdef SIMD
    ymm_sip_header(&ymm_sip_ctx, header);
#endif
    nthreads = n_threads;
    alive = new shrinkingset(nthreads);
    cuckoo = 0;
    nonleaf = new twice_set;
    ntrims = n_trims;
    int err = pthread_barrier_init(&barry, NULL, nthreads);
    assert(err == 0);
    sols = (nonce_t (*)[PROOFSIZE])calloc(maxsols = max_sols, PROOFSIZE*sizeof(nonce_t));
    assert(sols != 0);
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
} thread_ctx;

void barrier(pthread_barrier_t *barry) {
  int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

#define NONCESHIFT	(SIZESHIFT-1 - PART_BITS)
#define NODEPARTMASK	(NODEMASK >> PART_BITS)
#define NONCETRUNC	(1LL << (64 - NONCESHIFT))

void count_node_deg(thread_ctx *tp, u32 uorv, u32 part) {
  cuckoo_ctx *ctx = tp->ctx;
  shrinkingset *alive = ctx->alive;
  twice_set *nonleaf = ctx->nonleaf;
  u64 buffer[64];
#ifdef SIMD
  u64 sinput[8], soutput[8];
#endif

  for (nonce_t block = tp->id*64; block < HALFSIZE; block += ctx->nthreads*64) {
    u32 bsize = 0;
#ifdef SIMD
    u32 nsin = 0;
#endif
    u64 alive64 = alive->block(block);
    for (nonce_t nonce = block; alive64; alive64>>=1, nonce++) if (alive64 & 1LL) {
#ifdef SIMD
      sinput[nsin++] = 2*nonce + uorv;
      if (nsin == 8) {
        ymm_siphash24_2x(&ctx->ymm_sip_ctx, sinput, soutput); // SINPUT AND SOUTPUT CLD BE SHARED?!
        for (u32 i=0; i<8; i++) {
          node_t u = soutput[i] & NODEMASK;
          if ((u & PART_MASK) == part) {
            buffer[bsize++] = u >> PART_BITS;
            nonleaf->prefetch(u >> PART_BITS);
          }
        }
        nsin = 0;
      }
    }
    if (nsin) {
      ymm_siphash24_2x(&ctx->ymm_sip_ctx, sinput, soutput);
      for (u32 i=0; i<nsin; i++) {
        node_t u = soutput[i] & NODEMASK;
        if ((u & PART_MASK) == part) {
          buffer[bsize++] = u >> PART_BITS;
          nonleaf->prefetch(u >> PART_BITS);
        }
#else
      node_t u = _sipnode(&ctx->sip_ctx, nonce, uorv);
      if ((u & PART_MASK) == part) {
        buffer[bsize++] = u >> PART_BITS;
        nonleaf->prefetch(u >> PART_BITS);
#endif
      }
    }
    for (u32 i=0; i<bsize; i++)
      nonleaf->set(buffer[i]);
  }
}

void kill_leaf_edges(thread_ctx *tp, u32 uorv, u32 part) {
  cuckoo_ctx *ctx = tp->ctx;
  shrinkingset *alive = ctx->alive;
  twice_set *nonleaf = ctx->nonleaf;
  u64 buffer[64];
#ifdef SIMD
  u64 nce[8], sinput[8], soutput[8];
#endif

  for (nonce_t block = tp->id*64; block < HALFSIZE; block += ctx->nthreads*64) {
    u32 bsize = 0;
#ifdef SIMD
    u32 nsin = 0;
#endif
    u64 alive64 = alive->block(block);
    for (nonce_t nonce = block; alive64; alive64>>=1, nonce++) if (alive64 & 1LL) {
#ifdef SIMD
      nce[nsin] = nonce;
      sinput[nsin++] = 2*nonce + uorv;
      if (nsin == 8) {
        ymm_siphash24_2x(&ctx->ymm_sip_ctx, sinput, soutput); // SINPUT AND SOUTPUT CLD BE SHARED?!
        for (u32 i=0; i<8; i++) {
          node_t u = soutput[i] & NODEMASK;
          if ((u & PART_MASK) == part) {
            buffer[bsize++] = ((u64)nce[i] << NONCESHIFT) | (u >> PART_BITS);
            nonleaf->prefetch(u >> PART_BITS);
          }
        }
        nsin = 0;
      }
    }
    if (nsin) {
      ymm_siphash24_2x(&ctx->ymm_sip_ctx, sinput, soutput);
      for (u32 i=0; i<nsin; i++) {
        node_t u = soutput[i] & NODEMASK;
        if ((u & PART_MASK) == part) {
          buffer[bsize++] = ((u64)nce[i] << NONCESHIFT) | (u >> PART_BITS);
          nonleaf->prefetch(u >> PART_BITS);
        }
#else
      node_t u = _sipnode(&ctx->sip_ctx, nonce, uorv);
      if ((u & PART_MASK) == part) {
        buffer[bsize++] = ((u64)nonce << NONCESHIFT) | (u >> PART_BITS);
        nonleaf->prefetch(u >> PART_BITS);
#endif
      }
    }
    for (u32 i=0; i<bsize; i++) {
      u64 bi = buffer[i];
      if (!nonleaf->test(bi & NODEPARTMASK)) {
        nonce_t n = block | (bi >> NONCESHIFT);
        alive->reset(n, tp->id);
      }
    }
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
      edge e(sipnode(&ctx->sip_ctx, nonce, 0), sipnode(&ctx->sip_ctx, nonce, 1));
      if (cycle.find(e) != cycle.end()) {
        ctx->sols[soli][n++] = nonce;
#ifdef SHOWSOL
        printf("e(%x)=(%x,%x)%c", nonce, e.first, e.second, n==PROOFSIZE?'\n':' ');
#endif
        if (PROOFSIZE > 2)
          cycle.erase(e);
      }
    }
  assert(n==PROOFSIZE);
}

void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  shrinkingset *alive = ctx->alive;
  u32 load = 100LL * HALFSIZE / CUCKOO_SIZE;
  if (tp->id == 0)
    printf("initial load %d%%\n", load);
  for (u32 round=1; round <= ctx->ntrims; round++) {
    for (u32 uorv = 0; uorv < 2; uorv++) {
      for (u32 part = 0; part <= PART_MASK; part++) {
        if (tp->id == 0)
          ctx->nonleaf->reset();
        barrier(&ctx->barry);
        count_node_deg(tp,uorv,part);
        barrier(&ctx->barry);
        kill_leaf_edges(tp,uorv,part);
        barrier(&ctx->barry);
        if (tp->id == 0) {
          u32 load = (u32)(100LL * alive->count() / CUCKOO_SIZE);
          printf("round %d part %c%d load %d%%\n", round, "UV"[uorv], part, load);
        }
      }
    }
  }
  if (tp->id == 0) {
    load = (u32)(100LL * alive->count() / CUCKOO_SIZE);
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
  for (nonce_t block = tp->id*64; block < HALFSIZE; block += ctx->nthreads*64) {
    for (nonce_t nonce = block; nonce < block+64 && nonce < HALFSIZE; nonce++) {
      if (alive->test(nonce)) {
        node_t u0=sipnode(&ctx->sip_ctx, nonce, 0), v0=sipnode(&ctx->sip_ctx, nonce, 1);
        if (u0 == 0) // ignore vertex 0 so it can be used as nil for cuckoo[]
          continue;
        node_t u = cuckoo[us[0] = u0], v = cuckoo[vs[0] = v0];
        u32 nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100LL/HALFSIZE));
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
  pthread_exit(NULL);
  return 0;
}
