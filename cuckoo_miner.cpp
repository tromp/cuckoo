// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo_miner.h"
#include <unistd.h>
#include <assert.h>

int main(int argc, char **argv) {
  assert(SIZE < 1L<<32);
  int nthreads = 1;
  int maxsols = 8;
  int ntrims = 8;
  const char *header = "";
  int c, easipct = 50;
  while ((c = getopt (argc, argv, "e:h:m:n:t:")) != -1) {
    switch (c) {
      case 'e':
        easipct = atoi(optarg);
        break;
      case 'h':
        header = optarg;
        break;
      case 'm':
        maxsols = atoi(optarg);
        break;
      case 'n':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for %d-cycle on cuckoo%d(\"%s\") with %d%% edges, %d trims, and %d threads\n",
               PROOFSIZE, SIZESHIFT, header, easipct, ntrims, nthreads);

  cuckoo_ctx ctx(header, easipct * (u64)SIZE / 100, nthreads, ntrims, maxsols);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  for (int t = 0; t < nthreads; t++) {
    threads[t].id = t;
    threads[t].ctx = &ctx;
    assert(pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]) == 0);
  }
  for (int t = 0; t < nthreads; t++)
    assert(pthread_join(threads[t].thread, NULL) == 0);
  for (unsigned s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %llx", ctx.sols[s][i]);
    printf("\n");
  }
  return 0;
}
