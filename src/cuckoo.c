// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "cuckoo.h"
#include <inttypes.h> // for SCNx64 macro
#include <stdio.h>    // printf/scanf
#include <stdlib.h>   // exit
#include <unistd.h>   // getopt
#include <assert.h>   // d'uh

int main(int argc, char **argv) {
  const char *header = "";
  int c;
  while ((c = getopt (argc, argv, "h:")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
    }
  }
  printf("Verifying size %d proof for cuckoo%d(\"%s\")\n",
               PROOFSIZE, SIZESHIFT, header);
  int err = scanf("Solution");
  assert(err == 0);
  u64 nonces[PROOFSIZE];
  for (int n = 0; n < PROOFSIZE; n++) {
    int nscan = scanf(" %" SCNx64, &nonces[n]);
    assert(nscan == 1);
  }
  int pow_rc = verify(nonces, header);
  if (pow_rc != POW_OK) {
    printf("FAILED with code %d\n", pow_rc);
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
