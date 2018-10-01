// Cuckatoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2019 John Tromp

#include "lean.hpp"
#include <unistd.h>
#include <sys/time.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80


int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 1 + (PART_BITS+3)*(PART_BITS+4)/2;
  int nonce = 0;
  int range = 1;
  char header[HEADERLEN];
  unsigned len;
  struct timeval time0, time1;
  u32 timems;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "h:n:r:t:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  printf("Looking for %d-cycle on cuckatoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with trimming to %d bits, %d threads\n", EDGEBITS-IDXSHIFT, nthreads);

  u64 EdgeBytes = NEDGES/8;
  int EdgeUnit;
  for (EdgeUnit=0; EdgeBytes >= 1024; EdgeBytes>>=10,EdgeUnit++) ;
  u64 NodeBytes = (NEDGES >> PART_BITS)/8;
  int NodeUnit;
  for (NodeUnit=0; NodeBytes >= 1024; NodeBytes>>=10,NodeUnit++) ;
  printf("Using %d%cB edge and %d%cB node memory, and %d-way siphash\n",
     (int)EdgeBytes, " KMGT"[EdgeUnit], (int)NodeBytes, " KMGT"[NodeUnit], NSIPHASH);

  thread_ctx *threads = new thread_ctx[nthreads];
  assert(threads);
  cuckoo_ctx ctx(nthreads, ntrims, MAXSOLS);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.sip_keys.k0, ctx.sip_keys.k1, ctx.sip_keys.k2, ctx.sip_keys.k3);
    for (int t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].ctx = &ctx;
      int err = pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]);
      assert(err == 0);
    }
    for (int t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);
    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
      printf("\n");
      int pow_rc = verify(ctx.sols[s], &ctx.sip_keys);
      if (pow_rc == POW_OK) {
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)ctx.cg.sols[s], sizeof(ctx.sols[0]), 0, 0);
        for (int i=0; i<32; i++)
          printf("%02x", cyclehash[i]);
        printf("\n");
      } else {
        printf("FAILED due to %s\n", errstr[pow_rc]);
      }
      sumnsols += ctx.nsols;
    }
  }
  delete[] threads;
  printf("%d total solutions\n", sumnsols);
  return 0;
}
