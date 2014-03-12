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
    else System.out.println("Only recovered " + n + " nonces");
  }
}

public class CuckooWorker implements Runnable {
  int id;
  CuckooSolve solve;

  public voic run() {
    int[] cuckoo = solve.cuckoo;
    int[] us = new int[MAXPATHLEN], vs = new int[MAXPATHLEN]
    int u, v, nu, nv;
    for (int nonce = id; nonce < solve.easiness; nonce += solve.nthreads) {
      solve.graph.sipedge(nonce, us, vs);
      if ((u = cuckoo[us[0]]) == vs[0] || (v = cuckoo[vs[0]]) == us[0])
        continue; // ignore duplicate edges
      if (us[nu = solve.path(cuckoo, u, us)] == vs[nv = solve.path(cuckoo, v, vs)]) {
        int min = nu < nv ? nu : nv;
        for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
        int len = nu + nv + 1;
        System.out.println(" " + len + "-cycle found at " + id + ":" + (int)(nonce*100L/ctx->easiness) + "%");
        if (len == PROOFSIZE && solve.nsols < solve.maxsols)
          solve.solution(us, nu, vs, nv);
        continue;
      }
      if (nu < nv) {
        while (nu--)
          cuckoo[us[nu+1]] = us[nu];
        cuckoo[us[0]] = vs[0];
      } else {
        while (nv--)
          cuckoo[vs[nv+1]] = vs[nv];
        cuckoo[vs[0]] = us[0];
      }
    }
    Thread.currentThread().interrupt();
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
      } else if (argv[i].equals("-m")) {
        maxsols = Integer.parseInt(argv[++i]);
      }
    }
    assert easipct >= 0 && easipct <= 100;
    System.out.println("Looking for " + PROOFSIZE + "-cycle on cuckoo" + SIZEMULT + SIZESHIFT + "(\"" + header + "\") with " + easipct + "% edges and " + nthreads + " threads");
    solve = new Solve;
    solve.graph.setheader(header.getBytes());
    solve.easiness = (unsigned)(easipct * (long)SIZE / 100);
    assert(solve.cuckoo = calloc(1+SIZE, sizeof(unsigned)));
    assert(solve.sols = calloc(maxsols, PROOFSIZE*sizeof(unsigned)));
    solve.maxsols = maxsols;
    solve.nsols = 0;
    solve.nthreads = nthreads;
    pthread_mutex_init(&solve.setsol, NULL);
  
    thread_solve *threads = calloc(nthreads, sizeof(thread_ctx));
    assert(threads);
    for (int t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].ctx = &ctx;
      assert(pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]) == 0);
    }
    for (int t = 0; t < nthreads; t++)
      assert(pthread_join(threads[t].thread, NULL) == 0);
    for (int s = 0; s < solve.nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %x", solve.sols[s][i]);
      printf("\n");
    }
    return 0;
  }
}
