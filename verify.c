// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"

int main(int argc, char **argv) {
  char *header = argc >= 2 ? argv[1] : "";
  setheader(header);
  printf("Verifying size %d proof for cuckoo%d%d(\"%s\") with %d nodes and %d edges\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, SIZE, EASINESS);
  assert(scanf("Solution") == 0);
  unsigned us[PROOFSIZE], vs[PROOFSIZE], i = 0, n, nonce, sum;
  for (n = sum = 0; n < PROOFSIZE; n++) {
    assert(scanf(" %x", &nonce) == 1);
    assert(nonce < EASINESS);
    sum = (sum + nonce) % EASINESS;
    sipedge(nonce, &us[n], &vs[n]);
  }
  do {  // follow cycle until we return to i==0; n edges left to visit
    int j = i;
    for (int k = 0; k < PROOFSIZE; k++) // find unique other j with same vs[j]
      if (k != i && vs[k] == vs[i]) { assert(j == i); j = k; }
    assert(j != i);
    i = j;
    for (int k = 0; k < PROOFSIZE; k++) // find unique other i with same us[i]
      if (k != j && us[k] == us[j]) { assert(i == j); i = k; }
    assert(i != j);
    n -= 2;
  } while (i);
  assert(n == 0);
  printf("Verified with hash(%u)=%016llx\n", sum, siphash(sum));
  return 0;
}
