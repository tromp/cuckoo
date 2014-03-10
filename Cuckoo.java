// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

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

  public static long u8to64(byte[] p, int i) {
    return (long) p[i++]       | (long) p[i++] <<  8 |
           (long) p[i++] << 16 | (long) p[i++] << 24 |
           (long) p[i++] << 32 | (long) p[i++] << 40 |
           (long) p[i++] << 48 | (long) p[i++] << 56 ;
  }

  public void setheader(byte[] header) {
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

  // generate edge in cuckoo graph
  public void sipedge(int nonce, Edge e) {
    long sip = siphash24(nonce);
    e.u = 1 +         (int)(sip % PARTU);
    e.v = 1 + PARTU + (int)(sip % PARTV);
  }
  
  // verify that (ascending) nonces, all less than easiness, form a cycle in header-generated graph
  public Boolean verify(int[] nonces, byte[] header, int easiness) {
    setheader(header);
    Edge edges[] = new Edge[PROOFSIZE];
    int i = 0, n;
    for (n = 0; n < PROOFSIZE; n++) {
      if (nonces[n] >= easiness || (n != 0  && nonces[n] <= nonces[n-1]))
        return false;
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
    System.out.println("Hello, Cuckoo!");
  }
}
