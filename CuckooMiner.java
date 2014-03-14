// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

class CuckooSolve {
  static final int MAXPATHLEN = 8192;
  Cuckoo graph;
  int easiness;
  int[] cuckoo;
  int[][] sols;
  int nsols;
  int nthreads;

  public CuckooSolve(byte[] hdr, int en, int ms, int nt) {
    graph = new Cuckoo(hdr);
    easiness = en;
    sols = new int[ms][Cuckoo.PROOFSIZE];
    cuckoo = new int[1+Cuckoo.SIZE];
    assert cuckoo != null;
    nsols = 0;
    nthreads = nt;
  }

  public int path(int u, int[] us) {
    int nu;
    for (nu = 0; u != 0; u = cuckoo[u]) {
      if (++nu >= MAXPATHLEN) {
        while (nu-- != 0 && us[nu] != u) ;
        if (nu < 0)
          System.out.println("maximum path length exceeded");
        else System.out.println("illegal " + (MAXPATHLEN-nu) + "-cycle");
        Thread.currentThread().interrupt();
      }
      us[nu] = u;
    }
    return nu;
  }
  
  // largest number of long's that fit in MAXPATHLEN-Cuckoo.PROOFSIZE unsigned's
  static final int SOLMODU = (MAXPATHLEN-Cuckoo.PROOFSIZE)/2;
  static final int SOLMODV = SOLMODU-1;
  
  private void storedge(long uv, long[] usck, long[] vsck) {
    int j, i = (int)(uv % SOLMODU);
    long uvi = usck[i]; 
    if (uvi != 0) {
      if (vsck[j = (int)(uv % SOLMODV)] != 0) {
        vsck[(int)(uvi % SOLMODV)] = uvi;
      } else {
        vsck[j] = uv;
        return;
      }
    } else usck[i] = uv;
  }
  
  public synchronized void solution(int[] us, int nu, int[] vs, int nv) {
    long[] usck = new long[SOLMODU];
    long[] vsck = new long[SOLMODV];
    Edge e = new Edge();
    int c, n;
    storedge((long)us[0]<<32 | vs[0], usck, vsck);
    while (nu-- != 0)
      storedge((long)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
    while (nv-- != 0)
      storedge((long)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
    for (int nonce = n = 0; nonce < easiness; nonce++) {
      graph.sipedge(nonce, e);
      long uv = (long)e.u<<32 | e.v;
      if (usck[c = (int)(uv % SOLMODU)] == uv) {
        sols[nsols][n++] = nonce;
        usck[c] = 0;
      } else if (vsck[c = (int)(uv % SOLMODV)] == uv) {
        sols[nsols][n++] = nonce;
        vsck[c] = 0;
      }
    }
    if (n == Cuckoo.PROOFSIZE)
      nsols++;
    else System.out.println("Only recovered " + n + " nonces");
  }
}

public class CuckooMiner implements Runnable {
  int id;
  CuckooSolve solve;

  public CuckooMiner(int i, CuckooSolve cs) {
    id = i;
    solve = cs;
  }

  public void run() {
    int[] cuckoo = solve.cuckoo;
    int[] us = new int[CuckooSolve.MAXPATHLEN], vs = new int[CuckooSolve.MAXPATHLEN];
    int u, v, nu, nv;
    Edge e = new Edge();
    for (int nonce = id; nonce < solve.easiness; nonce += solve.nthreads) {
      solve.graph.sipedge(nonce, e);
      us[0] = e.u;
      vs[0] = e.v;
      if ((u = cuckoo[us[0]]) == vs[0] || (v = cuckoo[vs[0]]) == us[0])
        continue; // ignore duplicate edges
      if (us[nu = solve.path(u, us)] == vs[nv = solve.path(v, vs)]) {
        int min = nu < nv ? nu : nv;
        for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
        int len = nu + nv + 1;
        System.out.println(" " + len + "-cycle found at " + id + ":" + (int)(nonce*100L/solve.easiness) + "%");
        if (len == Cuckoo.PROOFSIZE && solve.nsols < solve.sols.length)
          solve.solution(us, nu, vs, nv);
        continue;
      }
      if (nu < nv) {
        while (nu-- != 0)
          cuckoo[us[nu+1]] = us[nu];
        cuckoo[us[0]] = vs[0];
      } else {
        while (nv-- != 0)
          cuckoo[vs[nv+1]] = vs[nv];
        cuckoo[vs[0]] = us[0];
      }
    }
    Thread.currentThread().interrupt();
  }

  public static void main(String argv[]) {
    assert Cuckoo.SIZE > 0;
    int nthreads = 1;
    int maxsols = 8;
    String header = "";
    int c, easipct = 50;
    for (int i = 0; i < argv.length; i++) {
      if (argv[i].equals("-e")) {
        easipct = Integer.parseInt(argv[++i]);
      } else if (argv[i].equals("-h")) {
        header = argv[++i];
      } else if (argv[i].equals("-m")) {
        maxsols = Integer.parseInt(argv[++i]);
      } else if (argv[i].equals("-t")) {
        nthreads = Integer.parseInt(argv[++i]);
      }
    }
    assert easipct >= 0 && easipct <= 100;
    System.out.println("Looking for " + Cuckoo.PROOFSIZE + "-cycle on cuckoo" + Cuckoo.SIZEMULT + Cuckoo.SIZESHIFT + "(\"" + header + "\") with " + easipct + "% edges and " + nthreads + " threads");
    CuckooSolve solve = new CuckooSolve(header.getBytes(), (int)(easipct * (long)Cuckoo.SIZE / 100), maxsols, nthreads);
  
    Thread[] threads = new Thread[nthreads];
    for (int t = 0; t < nthreads; t++) {
      threads[t] = new Thread(new CuckooMiner(t, solve));
      threads[t].start();
    }
    for (int t = 0; t < nthreads; t++) {
      try {
        threads[t].join();
      } catch (InterruptedException e) {
        System.out.println(e);
        System.exit(0);
      }
    }
    for (int s = 0; s < solve.nsols; s++) {
      System.out.print("Solution");
      for (int i = 0; i < Cuckoo.PROOFSIZE; i++)
        System.out.print(String.format(" %x", solve.sols[s][i]));
      System.out.println("");
    }
  }
}
