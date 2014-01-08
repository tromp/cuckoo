// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <openssl/sha.h>

// proof-of-work parameters
#ifndef SIZEMULT 
#define SIZEMULT 1
#endif
#ifndef SIZESHIFT 
#define SIZESHIFT 20
#endif
#ifndef EASYNESS 
#define EASYNESS (SIZE/2)
#endif
#ifndef PROOFSIZE 
#define PROOFSIZE 42
#endif

#define SIZE (SIZEMULT*(1<<SIZESHIFT))
// relatively prime partition sizes
#define PARTU (SIZE/2+1)
#define PARTV (SIZE/2-1)

// generate edge in cuckoo graph from hash(header++nonce)
void sha256edge(char *header, int nonce, int *pu, int *pv) {
  uint32_t hash[8];
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, header, strlen(header));
  SHA256_Update(&sha256, &nonce, sizeof(nonce));
  SHA256_Final((unsigned char *)hash, &sha256);
  uint64_t u64 = 0, v64 = 0;
  for (int i = 8; i--; ) {
    u64 = ((u64<<32) + hash[i]) % PARTU;
    v64 = ((v64<<32) + hash[i]) % PARTV;
  }
  *pu = 1 +         (int)u64;
  *pv = 1 + PARTU + (int)v64;
}
