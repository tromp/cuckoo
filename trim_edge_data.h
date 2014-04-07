// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp
// The edge=trimming time-memory trade-off is due to Dave Anderson:
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

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
#define CUCKOO_SIZE ((1+SIZE+(1<<IDXSHIFT)-1) >> IDXSHIFT)
#ifndef CLUMPSHIFT
// 2^CLUMPSHIFT should exceed maximum index drift (ui++) in cuckoo_hash
// SIZESHIFT-1 is limited to 64-KEYSHIFT
#define CLUMPSHIFT 9
#endif
#define KEYSHIFT (IDXSHIFT + CLUMPSHIFT)
#define KEYMASK ((1 << KEYSHIFT) - 1)
#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS ((HALFSIZE + PART_MASK) >> PART_BITS)
#define TWICE_WORDS ((2UL * ONCE_BITS + 31UL) / 32UL)

typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  std::vector<u32> bits;
  std::vector<u64> cnt;

  shrinkingset(nonce_t size, int nthreads) {
    bits.resize((size+31)/32);
    cnt.resize(nthreads);
    cnt[0] = size;
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

class twice_set {
public:
  au32 *bits;

  twice_set() {
    bits = (au32 *)calloc(TWICE_WORDS, sizeof(au32));
    assert(bits);
  }
  ~twice_set() {
    free(bits);
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
};

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
    node_t ui = u >> IDXSHIFT;
    u64 niew = (u64)v << KEYSHIFT | (u & KEYMASK);;
    for (;;) {
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if (((u^old) & KEYMASK) == 0) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
        return;
      }
      if (++ui == CUCKOO_SIZE)
        ui = 0;
    }
  }
  node_t operator[](node_t u) const {
    node_t ui = u >> IDXSHIFT;
    for (;;) {
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
      if (!cu)
        return 0;
      if (((u^cu) & KEYMASK) == 0)
        return (node_t)(cu >> KEYSHIFT);
      if (++ui == CUCKOO_SIZE)
        ui = 0;
    }
  }
};

class graph_data {
public:
  shrinkingset *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;

  graph_data(nonce_t easiness, int nthreads) {
    alive = new shrinkingset(easiness, nthreads);
    nonleaf = new twice_set;
  }
  ~graph_data() {
    delete alive;
    if (nonleaf)
      delete nonleaf;
    if (cuckoo)
      delete cuckoo;
  }
};

#define CUCKOOSET	cuckoo.set
#define CUCKOOHASH	cuckoo_hash &
