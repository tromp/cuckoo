// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int main(int argc, char **argv) {
  char *header = "";
  int c, easipct = 50;
  while ((c = getopt (argc, argv, "e:h:")) != -1) {
    switch (c) {
      case 'e':
        easipct = atoi(optarg);
        break;
          case 'h':
        header = optarg;
        break;
    }
  }
  printf("Verifying size %d proof for cuckoo%d%d(\"%s\") with %d%% edges\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, easipct);
  assert(scanf("Solution") == 0);
  unsigned nonces[PROOFSIZE];
  for (int n = 0; n < PROOFSIZE; n++)
    assert(scanf(" %x", &nonces[n]) == 1);
  int easiness = (unsigned)(easipct * (u64)SIZE / 100);
  int ok = verify(nonces, header, easiness);
  if (!ok) {
    printf("FAILED\n");
    exit(1);
  }
  printf("Verified with cyclehash ");
  unsigned char cyclehash[32];
  SHA256((unsigned char *)nonces, sizeof(nonces), cyclehash);
  for (int i=0; i<32; i++)
    printf("%02x", cyclehash[i]);
  printf("\n");
  return 0;
}
