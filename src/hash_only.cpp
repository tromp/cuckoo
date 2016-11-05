#include "siphash.h"
#include <stdio.h>

#include "highwayhash/siphashx4.h"
typedef highwayhash::uint64 ull64;

#ifndef WP
#define WP 4
#endif

int main() {
#ifdef USE_AVX2
  alignas(64) u64 indices[WP];
  alignas(64) u64 hashes[WP];
  u32 nidx = 0;
#endif
  siphash_keys keys = {34, 53};
  u32 sum = 0;

  for (u64 nonce = 0; nonce < (1ULL<<31); nonce++) {
#ifdef USE_AVX2
    indices[nidx++ % WP] = nonce;
    if (nidx % WP == 0) {
#if WP==4
      siphash24x4(&keys, indices, hashes);
      // highwayhash::siphashx4(&keys, (ull64 *)indices, (ull64 *)hashes);
#elif WP==8
      siphash24x8(&keys, indices, hashes);
#else
#error not implemented
#endif
      for (int i=0; i<WP; i++)
        sum ^= hashes[i];
    }
#else
    sum ^= siphash24(&keys, nonce);
    // sum ^= _sipnode(&keys, nonce,0);
#endif
  }
  printf("sum %x\n", sum);
}
