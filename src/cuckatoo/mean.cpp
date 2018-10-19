// Cuckatoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2019 John Tromp

#include "mean.hpp"
#include <unistd.h>
#include <sys/time.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

/*
 int device id
 int edge_bits
 string device_name
 bool has_errored
 int last_start_time
 int last_end_time
 int last_solution time
*/

CALL_CONVENTION int run_solver(char* header,
                               int header_length,
                               ParamMap params,
                               // Solution callback: header nonce, solution nonces
                               void (*sol_cb)(u64, word_t* nonces),
                               // Stat collection callback
                               // device_id, edge_bits, device_name, device_name_len, has_errored
                               // last_start_time, last_end_time, last_solution time
                               void (*stat_cb)(u32, u32, char*, u32, bool, u64, u64, u64)
                               )
{
  u32 nthreads = params.get("nthreads");
  u32 ntrims = params.get("ntrims");
  u32 nonce = params.get("nonce");
  u32 range = params.get("range");
  bool showcycle = params.get("showcycle");
  bool allrounds = params.get("allrounds");

  struct timeval time0, time1;
  u32 timems;
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges\n");

  solver_ctx ctx(nthreads, ntrims, allrounds, showcycle);

  u64 sbytes = ctx.sharedbytes();
  u32 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory at %lx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer.buckets);
  printf("%dx%d%cB thread memory at %lx,\n", nthreads, tbytes, " KMGT"[tunit], (u64)ctx.trimmer.tbuckets);
  printf("%d-way siphash, and %d buckets.\n", NSIPHASH, NX);

  u32 sumnsols = 0;

  for (u32 r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    ctx.setheadernonce(header, header_length, nonce + r);
    printf("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx.trimmer.sip_keys.k0, ctx.trimmer.sip_keys.k1, ctx.trimmer.sip_keys.k2, ctx.trimmer.sip_keys.k3);
    u32 nsols = ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < nsols; s++) {
      printf("Solution");
      word_t *prf = &ctx.sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)prf[i]);
      printf("\n");
      if (sol_cb != NULL){
        sol_cb(nonce + r, prf);
      }
      int pow_rc = verify(prf, &ctx.trimmer.sip_keys);
      if (pow_rc == POW_OK) {
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          printf("%02x", cyclehash[i]);
        printf("\n");
      } else {
        printf("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
    if (stat_cb != NULL) {
        /// TODO: better timer resolution
        char device_name[3] = {'C', 'P', 'U'};
        stat_cb(0, EDGEBITS, device_name, 3, false, time0.tv_usec, time1.tv_usec, time1.tv_usec-time0.tv_usec);
    }
  }
  printf("%d total solutions\n", sumnsols);
  return 0;
}

int main(int argc, char **argv) {
  u32 nthreads = 1;
  u32 ntrims = EDGEBITS > 30 ? 96 : 68;
  u32 nonce = 0;
  u32 range = 1;
#ifdef SAVEEDGES
  bool showcycle = 1;
#else
  bool showcycle = 0;
#endif
  char header[HEADERLEN];
  u32 len;
  bool allrounds = false;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "ah:m:n:r:st:x:")) != -1) {
    switch (c) {
      case 'a':
        allrounds = true;
        break;
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len == sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'm':
        ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 's':
        showcycle = true;
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }

	ParamMap params;
	params.set("allrounds", allrounds);
	params.set("nonce", nonce);
	params.set("range", range);
	params.set("ntrims", ntrims);
	params.set("showcycle", showcycle);
	params.set("nthreads", nthreads);

	run_solver(header, sizeof(header), params, NULL, NULL);
}
