// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2015 John Tromp

#include "cuckoo_miner.h"
#include <unistd.h>

int main(int argc, char **argv) {
  int nthreads = 1;
  int maxsols  = 8;
  int ntrims   = 1 + (PART_BITS+3)*(PART_BITS+4)/2;
  const char *header = "";
  int c;
  while ((c = getopt (argc, argv, "h:m:n:t:")) != -1) {
    switch (c) {
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
  printf("Looking for %d-cycle on cuckoo%d(\"%s\") with 50%% edges, %d trims, %d threads\n",
               PROOFSIZE, SIZESHIFT, header, ntrims, nthreads);
  u64 edgeBytes = HALFSIZE/8, nodeBytes = TWICE_WORDS*sizeof(u32);
  int edgeUnit, nodeUnit;
  for (edgeUnit=0; edgeBytes >= 1024; edgeBytes>>=10,edgeUnit++) ;
  for (nodeUnit=0; nodeBytes >= 1024; nodeBytes>>=10,nodeUnit++) ;
  printf("Using %d%cB edge and %d%cB node memory.\n",
     (int)edgeBytes, " KMGT"[edgeUnit], (int)nodeBytes, " KMGT"[nodeUnit]);
  cuckoo_ctx ctx(header, nthreads, ntrims, maxsols);
  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
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
  free(threads);
  for (unsigned s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %lx", (long)ctx.sols[s][i]);
    printf("\n");
  }
  return 0;
}
