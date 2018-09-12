// Cuckatoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2019 John Tromp

#include "lean_miner.hpp"
#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80


int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 1 + (PART_BITS+3)*(PART_BITS+4)/2;
  int nonce = 0;
  int range = 1;
  char header[HEADERLEN];
  unsigned len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "h:m:n:r:t:")) != -1) {
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
      case 'm':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  printf("Looking for %d-cycle on cuckatoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with %d trims, %d threads\n", ntrims, nthreads);

  u64 Bytes = NEDGES/8;
  int Unit;
  for (Unit=0; Bytes >= 1024; Bytes>>=10,Unit++) ;
  printf("Using %d%cB edge and %d%cB node memory, and %d-way siphash\n",
     (int)Bytes, " KMGT"[Unit], (int)Bytes, " KMGT"[Unit], NSIPHASH);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  cuckoo_ctx ctx(nthreads, ntrims, MAXSOLS);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    ctx.setheadernonce(header, sizeof(header), nonce + r);
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
    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
      printf("\n");
    }
    sumnsols += ctx.nsols;
  }
  free(threads);
  printf("%d total solutions\n", sumnsols);
  return 0;
}
