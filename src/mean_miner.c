// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "mean_miner.h"
#include <unistd.h>
#include <sys/time.h>

#define MAXSOLS 8

int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 1;
  int nonce = 0;
  int range = 1;
  struct timeval time0, time1;
  u64 rdtsc0, rdtsc1, timens;
  u32 timems;
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
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads\n", ntrims, nthreads);

  u64 bbytes = NBUCKETS * sizeof(bucket);
  u64 tbytes = nthreads * sizeof(histgroup);
  int bunit,tunit;
  for (bunit=0; bbytes >= 1024; bbytes>>=10,bunit++) ;
  for (tunit=0; tbytes >= 1024; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory, %d%cB thread memory, %d-way siphash, and %d-byte edgehash\n", (int)bbytes, " KMGT"[bunit], (int)tbytes, " KMGT"[tunit], NSIPHASH, EDGEHASH_BYTES);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  cuckoo_ctx *ctx = new_cuckoo_ctx(nthreads, ntrims, MAXSOLS);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    rdtsc0 = __rdtsc();
    setheadernonce(ctx, header, sizeof(header), nonce + r);
    printf("k0 k1 %lx %lx\n", ctx->sip_keys.k0, ctx->sip_keys.k1);
    for (int t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].ctx = ctx;
      int err = pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]);
      assert(err == 0);
    }
    for (int t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }
    gettimeofday(&time1, 0);
    rdtsc1 = __rdtsc();
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < ctx->nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx->sols[s][i]);
      printf("\n");
    }
    sumnsols += ctx->nsols;
  }
  free(threads);
  printf("%d total solutions\n", sumnsols);
  return 0;
}
