// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "blake2/sse/blake2.h"

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
void blake2edge(char *header, int nonce, int *pu, int *pv) {
  uint32_t hash[8];

int blake2s( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen )
  blake2s_state S[1];
  if ( NULL == key ) keylen = 0; /* Fail here instead if keylen != 0 and key == NULL? */
  if( keylen > 0 ) {
    if( blake2s_init_key( S, outlen, key, keylen ) < 0 ) return -1;
  } else {
    if( blake2s_init( S, outlen ) < 0 ) return -1;
  }
  blake2s_update( S, ( uint8_t * )in, inlen );
  blake2s_final( S, out, outlen );

  int blake2s_init( blake2s_state *S, const uint8_t outlen );
  int blake2s_init_key( blake2s_state *S, const uint8_t outlen, const void *key, const uint8_t keylen );
  int blake2s_init_param( blake2s_state *S, const blake2s_param *P );
  int blake2s_update( blake2s_state *S, const uint8_t *in, uint64_t inlen );
  int blake2s_final( blake2s_state *S, uint8_t *out, uint8_t outlen );

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
