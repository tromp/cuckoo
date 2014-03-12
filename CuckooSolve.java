// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

import Cuckoo;

class CuckooSolve {
  static final int MAXPATHLEN = 8192;
  Cuckoo graph;
  int easiness;
  int[] cuckoo;
  int[][] sols;
  int maxsols;
  int nsols;
  int nthreads;

  private int path(int u, int[] us) {
    int nu;
    for (nu = 0; u; u = cuckoo[u]) {
      if (++nu >= MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (nu < 0)
          System.out.println("maximum path length exceeded");
        else System.out.println("illegal " + (MAXPATHLEN-n) + "-cycle");
        Thread.currentThread().interrupt();
      }
      us[nu] = u;
    }
    return nu;
  }
  
  // largest number of long's that fit in MAXPATHLEN-PROOFSIZE unsigned's
  static final int SOLMODU = (MAXPATHLEN-PROOFSIZE)/2;
  static final int SOLMODV = SOLMODU-1;
  
  private void storedge(long uv, long[] usck, long[] vsck) {
    int j, i = uv % SOLMODU;
    long uvi = usck[i]; 
    if (uvi != 0) {
      if (vsck[j = uv % SOLMODV] != 0) {
        vsck[uvi % SOLMODV] = uvi;
      } else {
        vsck[j] = uv;
        return;
      }
    } else usck[i] = uv;
  }
  
  public synchronized void solution(int[] us, int nu, int[] vs, int nv) {
    long *usck = (long *)&us[PROOFSIZE], *vsck = (long *)&vs[PROOFSIZE];
    unsigned u, v, n;
    for (int i=0; i<SOLMODU; i++)
      usck[i] = vsck[i] = 0L;
    storedge((long)*us<<32 | *vs, usck, vsck);
    while (nu--)
      storedge((long)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
    while (nv--)
      storedge((long)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
    pthread_mutex_lock(&ctx->setsol);
    for (unsigned nonce = n = 0; nonce < ctx->easiness; nonce++) {
      sipedge(&ctx->sip_ctx, nonce, &u, &v);
      long *c, uv = (long)u<<32 | v;
      if (*(c = &usck[uv % SOLMODU]) == uv || *(c = &vsck[uv % SOLMODV]) == uv) {
        ctx->sols[ctx->nsols][n++] = nonce;
        *c = 0;
      }
    }
    if (n == PROOFSIZE)
      ctx->nsols++;
    else printf("Only recovered %d nonces\n", n);
    pthread_mutex_unlock(&ctx->setsol);
  }
  
  void *worker(void *vp) {
    thread_ctx *tp = (thread_ctx *)vp;
    cuckoo_ctx *ctx = tp->ctx;
    unsigned *cuckoo = ctx->cuckoo;
    unsigned us[MAXPATHLEN], u, vs[MAXPATHLEN], v; 
    int nu, nv;
    for (unsigned nonce = tp->id; nonce < ctx->easiness; nonce += ctx->nthreads) {
      sipedge(&ctx->sip_ctx, nonce, us, vs);
      if ((u = cuckoo[*us]) == *vs || (v = cuckoo[*vs]) == *us)
        continue; // ignore duplicate edges
  #ifdef SHOW
      for (int j=1; j<=SIZE; j++)
        if (!cuckoo[j]) printf("%2d:   ",j);
        else            printf("%2d:%02d ",j,cuckoo[j]);
      printf(" %x (%d,%d)\n", nonce,*us,*vs);
  #endif
      if (us[nu = path(cuckoo, u, us)] == vs[nv = path(cuckoo, v, vs)]) {
        int min = nu < nv ? nu : nv;
        for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
        int len = nu + nv + 1;
        printf("% 4d-cycle found at %d:%d%%\n", len, tp->id, (int)(nonce*100L/ctx->easiness));
        if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
          solution(ctx, us, nu, vs, nv);
        continue;
      }
      if (nu < nv) {
        while (nu--)
          cuckoo[us[nu+1]] = us[nu];
        cuckoo[*us] = *vs;
      } else {
        while (nv--)
          cuckoo[vs[nv+1]] = vs[nv];
        cuckoo[*vs] = *us;
      }
    }
    pthread_exit(NULL);
  }

  public static void main(String argv[]) {
    assert SIZE > 0;
    int nthreads = 1;
    int maxsols = 8;
    String header = "";
    int c, easipct = 50;
    for (i = 0; i < argv.length; i++) {
      if (argv[i].equals("-e")) {
        easipct = Integer.parseInt(argv[++i]);
      } else if (argv[i].equals("-h")) {
        header = argv[++i];
      if (argv[i].equals("-m")) {
        maxsols = Integer.parseInt(argv[++i]);
      }
    }
    assert easipct >= 0 && easipct <= 100;
    System.out.println("Looking for " + PROOFSIZE + "-cycle on cuckoo" + SIZEMULT + SIZESHIFT + "(\"" + header + "\") with " + easipct + "% edges and " + nthreads + " threads");
    cuckoo_ctx ctx;
    setheader(&ctx.sip_ctx, header);
    ctx.easiness = (unsigned)(easipct * (long)SIZE / 100);
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
}
