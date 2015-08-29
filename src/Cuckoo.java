// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2015 John Tromp

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Scanner;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class Edge {
  long u;
  long v;

  public Edge(long x, long y) {
    u = x;
    v = y;
  }

  public int hashCode() {
    return (int)(u^v);
  }

  public boolean equals(Object o) {
    Edge f = (Edge)o;
    return u == f.u && v == f.v;
  }
}

public class Cuckoo {
  public static final int SIZESHIFT = 20;
  public static final int PROOFSIZE = 42;
  public static final long SIZE = 1 << SIZESHIFT;
  public static final long HALFSIZE  = SIZE/2;
  public static final long NODEMASK  = HALFSIZE - 1;

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
      System.out.println(e);
      System.exit(1);
    }
  }

  public long siphash24(long nonce) {
    long v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3] ^ nonce;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    v0 ^= nonce; v2 ^= 0xff;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    v0 += v1; v2 += v3; v1 = (v1 << 13) | v1 >>> 51;
                        v3 = (v3 << 16) | v3 >>> 48;
    v1 ^= v0; v3 ^= v2; v0 = (v0 << 32) | v0 >>> 32;
    v2 += v1; v0 += v3; v1 = (v1 << 17) | v1 >>> 47;
                        v3 = (v3 << 21) | v3 >>> 43;
    v1 ^= v2; v3 ^= v0; v2 = (v2 << 32) | v2 >>> 32;

    return v0 ^ v1 ^ v2 ^ v3;
  }

  // generate edge in cuckoo graph
  public long sipnode(long nonce, int uorv) {
    return siphash24(2*nonce + uorv) & NODEMASK;
  }
  
  // generate edge in cuckoo graph
  public Edge sipedge(long nonce) {
    return new Edge(sipnode(nonce,0), sipnode(nonce,1));
  }
  
  // verify that (ascending) nonces, all less than easiness, form a cycle in graph
  public Boolean verify(long[] nonces, int easiness) {
    long us[] = new long[PROOFSIZE], vs[] = new long[PROOFSIZE];
    int i = 0, n;
    for (n = 0; n < PROOFSIZE; n++) {
      if (nonces[n] >= easiness || (n != 0  && nonces[n] <= nonces[n-1]))
        return false;
      us[n] = sipnode(nonces[n],0);
      vs[n] = sipnode(nonces[n],1);
    }
    do {  // follow cycle until we return to i==0; n edges left to visit
      int j = i;
      for (int k = 0; k < PROOFSIZE; k++) // find unique other j with same vs[j]
        if (k != i && vs[k] == vs[i]) {
          if (j != i)
            return false;
          j = k;
      }
      if (j == i)
        return false;
      i = j;
      for (int k = 0; k < PROOFSIZE; k++) // find unique other i with same us[i]
        if (k != j && us[k] == us[j]) {
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
    int i, easipct = 50;
    for (i = 0; i < argv.length; i++) {
      if (argv[i].equals("-e")) {
        easipct = Integer.parseInt(argv[++i]);
      } else if (argv[i].equals("-h")) {
        header = argv[++i];
      }
    }
    System.out.println("Verifying size " + PROOFSIZE + " proof for cuckoo" + SIZESHIFT + "(\"" + header + "\") with " + easipct + "% edges");
    Scanner sc = new Scanner(System.in);
    sc.next();
    long nonces[] = new long[PROOFSIZE];
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
    ByteBuffer buf = ByteBuffer.allocate(PROOFSIZE*8);
    buf.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(nonces);
    byte[] cyclehash;
    try {
      cyclehash = MessageDigest.getInstance("SHA-256").digest(buf.array());
      for (i=0; i<32; i++)
        System.out.print(String.format("%02x",((int)cyclehash[i] & 0xff)));
      System.out.println("");
    } catch(NoSuchAlgorithmException e) {
      System.out.println(e);
      System.exit(1);
    }
  }
}
