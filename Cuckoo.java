// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Scanner;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ByteOrder;

class Edge {
  int u;
  int v;
}

public class Cuckoo {
  public static final int SIZEMULT = 1;
  public static final int SIZESHIFT = 20;
  public static final int PROOFSIZE = 42;
  public static final int SIZE = SIZEMULT*(1<<SIZESHIFT);
  // relatively prime partition sizes, assuming SIZESHIFT >= 2
  public static final int PARTU  = SIZE/2+1;
  public static final int PARTV  = SIZE/2-1;

  long v[] = new long[4];

  public static long u8(byte b) {
    return (long)(b) & 0xff;
  }

  public static long u8to64(byte[] p, int i) {
    return u8(p[i  ])       | u8(p[i+1]) <<  8 |
           u8(p[i+2]) << 16 | u8(p[i+3]) << 24 |
           u8(p[i+4]) << 32 | u8(p[i+5]) << 40 |
           u8(p[i+6]) << 48 | u8(p[i+7]) << 56 ;
  }

  public Cuckoo(byte[] header) {
    byte[] hdrkey;
    try {
      hdrkey = MessageDigest.getInstance("SHA-256").digest(header);
      long k0 = u8to64(hdrkey, 0);
      long k1 = u8to64(hdrkey, 8);
      v[0] = k0 ^ 0x736f6d6570736575L;
      v[1] = k1 ^ 0x646f72616e646f6dL;
      v[2] = k0 ^ 0x6c7967656e657261L;
      v[3] = k1 ^ 0x7465646279746573L;
    } catch(NoSuchAlgorithmException e) {
    }
  }

  public long siphash24(int nonce) {
    long b = 4L << 56 | nonce;
    long v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3] ^ b;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    v0 ^= b;
    v2 ^= 0xff;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1;                    v2 += v3;
    v1 = (v1 << 13) | v1 >>> 51; v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0;                    v3 ^= v2;
    v0 = (v0 << 32) | v0 >>> 32; v2 += v1;
    v0 += v3;                    v1 = (v1 << 17) | v1 >>> 47;
    v3 = (v3 << 21) | v3 >>> 43; v1 ^= v2;
    v3 ^= v0;                    v2 = (v2 << 32) | v2 >>> 32;

    return v0 ^ v1 ^ v2 ^ v3;
  }

  // return u % m with a considered unsigned, assuming m > 0
  private static int remainder(long u, int m) {
    int i = (int)((u >>> 1) % m);
    int j = (i << 1) + (int)(u & 1);
    return j < m ? j : j - m;
  }

  // generate edge in cuckoo graph
  public void sipedge(int nonce, Edge e) {
    long sip = siphash24(nonce);
    e.u = 1 +         remainder(sip, PARTU);
    e.v = 1 + PARTU + remainder(sip, PARTV);
  }
  
  // verify that (ascending) nonces, all less than easiness, form a cycle in graph
  public Boolean verify(int[] nonces, int easiness) {
    Edge edges[] = new Edge[PROOFSIZE];
    int i = 0, n;
    for (n = 0; n < PROOFSIZE; n++) {
      if (nonces[n] >= easiness || (n != 0  && nonces[n] <= nonces[n-1]))
        return false;
      edges[n] = new Edge();
      sipedge(nonces[n], edges[n]);
    }
    do {  // follow cycle until we return to i==0; n edges left to visit
      int j = i;
      for (int k = 0; k < PROOFSIZE; k++) // find unique other j with same vs[j]
        if (k != i && edges[k].v == edges[i].v) {
          if (j != i)
            return false;
          j = k;
      }
      if (j == i)
        return false;
      i = j;
      for (int k = 0; k < PROOFSIZE; k++) // find unique other i with same us[i]
        if (k != j && edges[k].u == edges[j].u) {
          if (i != j)
            return false;
          i = k;
      }
      if (i == j)
        return false;
      n -= 2;
    } while (i != 0);
    return n == 0;
  }

  public static void main(String argv[]) {
    String header = "";
    int c, i, easipct = 50;
    for (i = 0; i < argv.length; i++) {
      if (argv[i].equals("-e")) {
        easipct = Integer.parseInt(argv[++i]);
      } else if (argv[i].equals("-h")) {
        header = argv[++i];
      }
    }
    System.out.println("Verifying size " + PROOFSIZE + " proof for cuckoo" + SIZEMULT + SIZESHIFT + "(\"" + header + "\") with " + easipct + "% edges");
    Scanner sc = new Scanner(System.in);
    sc.next();
    int nonces[] = new int[PROOFSIZE];
    for (int n = 0; n < PROOFSIZE; n++) {
      nonces[n] = Integer.parseInt(sc.next(), 16);
    }
    int easiness = (int)(easipct * (long)SIZE / 100L);
    Cuckoo cuckoo = new Cuckoo(header.getBytes());
    Boolean ok = cuckoo.verify(nonces, easiness);
    if (!ok) {
      System.out.println("FAILED");
      System.exit(1);
    }
    System.out.print("Verified with cyclehash ");
    ByteBuffer buf = ByteBuffer.allocate(PROOFSIZE*4);
    buf.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(nonces);
    byte[] cyclehash;
    try {
      cyclehash = MessageDigest.getInstance("SHA-256").digest(buf.array());
      for (i=0; i<32; i++)
        System.out.print(String.format("%02x",((int)cyclehash[i] & 0xff)));
      System.out.println("");
    } catch(NoSuchAlgorithmException e) {
    }
  }
}
