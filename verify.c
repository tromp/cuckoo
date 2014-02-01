// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"

int main(int argc, char **argv) {
  char *header = argc >= 2 ? argv[1] : "";
  setheader(header);
  printf("Verifying size %d proof for cuckoo%d%d(\"%s\") of difficulty %d\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, EASINESS);
  int us[PROOFSIZE], vs[PROOFSIZE], i, j, nonce;
  for (int n = 0; n < PROOFSIZE; n++) {
    assert(scanf("%*d %x (%*d,%*d)\n", &nonce) == 1);
    sipedge(nonce, &us[n], &vs[n]);
  }
  for (int n = i = 0; n < PROOFSIZE; n += 2) {
    assert(n == 0 || i != 0);
    for (j = 0; j < PROOFSIZE; j++)
      if (i != j &&  vs[i] == vs[j])
        break;
    assert(j < PROOFSIZE);
    for (i = 0; i < PROOFSIZE; i++)
      if (i != j &&  us[i] == us[j])
        break;
    assert(i < PROOFSIZE);
  }
  assert(i == 0);
  printf("Verified!\n");
  return 0;
}
