// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo_solve.h"
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int main(int argc, char **argv) {
  assert(SIZE < 1L<<32);
  int nthreads = 1;
  int maxsols = 8;
  char *header = "";
  int c, easipct = 50;
  while ((c = getopt (argc, argv, "e:h:m:t:")) != -1) {
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
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for %d-cycle on cuckoo%d%d(\"%s\") with %d%% edges and %d threads\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, easipct, nthreads);

  cuckoo_ctx ctx;
  setheader(&ctx.sip_ctx, header);
  ctx.easiness = (unsigned)(easipct * (u64)SIZE / 100);
  assert(ctx.cuckoo = calloc(1+SIZE, sizeof(unsigned)));
  assert(ctx.sols = calloc(maxsols, PROOFSIZE*sizeof(unsigned)));
  ctx.maxsols = maxsols;
  ctx.nsols = 0;
  ctx.nthreads = nthreads;
  pthread_mutex_init(&ctx.setsol, NULL);

  thread_ctx *threads = calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  for (int t = 0; t < nthreads; t++) {
    threads[t].id = t;
    threads[t].ctx = &ctx;
    assert(pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]) == 0);
  }
  for (int t = 0; t < nthreads; t++)
    assert(pthread_join(threads[t].thread, NULL) == 0);
  for (int s = 0; s < ctx.nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %x", ctx.sols[s][i]);
    printf("\n");
  }
  return 0;
}
