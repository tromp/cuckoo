// Cuck(at)oo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2020 John Tromp

#include "cuckatoo.h"
#include <inttypes.h> // for SCNx64 macro
#include <stdio.h>    // printf/scanf
#include <stdlib.h>   // exit
#include <unistd.h>   // getopt
#include <assert.h>   // d'uh

// arbitrary length of header hashed into siphash key
#define HEADERLEN 246

int main(int argc, char **argv) {
  char headernonce[HEADERLEN];
  memset(headernonce, 0, HEADERLEN);
  int nonce = 0;
  int len, c;
  while ((c = getopt (argc, argv, "h:n:x:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(headernonce));
        memcpy(headernonce, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len == sizeof(headernonce)-sizeof(u64) || len == sizeof(headernonce));
        for (u32 i=0; i<len; i++) {
          sscanf(optarg+2*i, "%2hhx", headernonce+i);
        }
        break;
      case 'n':
        nonce = atoi(optarg);
        ((u32 *)headernonce)[HEADERLEN/sizeof(u32)-1] = htole32(nonce); // place nonce near end aligned at u32
        break;
    }
  }
  siphash_keys keys;
  setheader(headernonce, sizeof(headernonce), &keys);
  printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce, keys.k0, keys.k1, keys.k2, keys.k3);
  printf("Verifying size %d proof for cuckatoo%d(\"", PROOFSIZE, EDGEBITS);
  for (int i=0; i < HEADERLEN; i++)
    print_log("%02x", (unsigned char)headernonce[i]);
  if (nonce) print_log(",%d", nonce);
  print_log(")\n");

  word_t nonces[PROOFSIZE];
  uint64_t index;
#ifdef cuckoo_solution
  for (int nsols=0; scanf(" \"cuckoo_solution\": [") == 0; nsols++) {
    for (int n = 0; n < PROOFSIZE; n++) {
      int nscan = scanf(" %" SCNu64 ",", &index);
#else
  for (int nsols=0; scanf(" Solution") == 0; nsols++) {
    for (int n = 0; n < PROOFSIZE; n++) {
      int nscan = scanf(" %" SCNx64, &index);
#endif
      assert(nscan == 1);
      nonces[n] = index;
    }
    int pow_rc = verify(nonces, &keys);
    if (pow_rc == POW_OK) {
      printf("Verified with cyclehash ");
      unsigned char cyclehash[32];
      blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)nonces, sizeof(nonces), 0, 0);
      for (int i=0; i<32; i++)
        printf("%02x", cyclehash[i]);
      printf("\n");
    } else {
      printf("FAILED due to %s\n", errstr[pow_rc]);
    }
  }
  return 0;
}
