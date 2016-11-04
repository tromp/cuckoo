#include "cuckoo.h"
#include <stdio.h>
#include "highwayhash/siphashx4.h"

typedef highwayhash::uint64 ull64;

int main() {
#ifdef USE_AVX2
  alignas(64) u64 indices[4];
  alignas(64) u64 hashes[4];
  u32 nidx = 0;
#endif
  siphash_keys keys;
  u32 sum = 0;

  for (u64 nonce = 0; nonce < HALFSIZE; nonce++) {
#ifdef USE_AVX2
    indices[nidx++ % 4] = 2 * nonce;
    if (nidx % 4 == 0) {
      highwayhash::siphashx4(&keys, (ull64 *)indices, (ull64 *)hashes);
      sum ^= (hashes[0] ^ hashes[1] ^ hashes[2] ^ hashes[3]) & NODEMASK;
    }
#else
    sum ^= _sipnode(&keys, nonce, 0);
#endif
  }
  printf("sum %x\n", sum);
}
