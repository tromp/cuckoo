// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"

int main(int argc, char **argv) {
  char *header = argc >= 2 ? argv[1] : "";
  setheader(header);
  printf("Verifying size %d proof for cuckoo%d%d(\"%s\") of difficulty %d\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, EASINESS);
  int us[PROOFSIZE], vs[PROOFSIZE], i, nonce;
  for (int n = 0; n < PROOFSIZE; n++) {
    assert(scanf("%*d %x (%*d,%*d)\n", &nonce) == 1);
    assert(nonce < EASINESS);
    sipedge(nonce, &us[n], &vs[n]);
  }
  for (int n = i = 0; n < PROOFSIZE; n += 2) {
    assert(n == 0 || i != 0);
    int j = i; // find unique other j with same vs[j]
    for (int k = 0; k < PROOFSIZE; k++)
      if (k != i && vs[k] == vs[i]) {
        assert(j == i);
        j = k;
      }
    assert(j != i);
    i = j; // find unique other i with same us[i]
    for (int k = 0; k < PROOFSIZE; k++)
      if (k != j &&  us[k] == us[j]) {
        assert(i == j);
        i = k;
      }
    assert(i != j);
  }
  assert(i == 0);
  printf("Verified!\n");
  return 0;
}
