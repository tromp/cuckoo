#include "siphash.h"
#include <stdio.h>

int main() {
  alignas(64) u64 indices[NSIPHASH];
  alignas(64) u64 hashes[NSIPHASH];
  u32 nidx = 0;
  siphash_keys keys = {42, 54};
  u32 sum = 0;

  for (u64 nonce = 0; nonce < (1ULL<<31); nonce++) {
    indices[nidx++ % NSIPHASH] = nonce;
    if (nidx % NSIPHASH == 0) {
      siphash24xN(&keys, indices, hashes);
      for (int i=0; i < NSIPHASH; i++)
        sum ^= hashes[i];
    }
  }
  printf("sum %x\n", sum);
}
