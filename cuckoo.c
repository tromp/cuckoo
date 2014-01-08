// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
// algorithm parameters
#define MAXPATHLEN 65536

// used to simplify nonce recovery
#define CYCLE 0x80000000
int cuckoo[1+SIZE]; // global; conveniently initialized to zero

int main(int argc, char **argv) {
  // 6 largest sizes 131 928 529 330 729 132 not implemented
  assert(SIZE < (unsigned)CYCLE);
  char *header = argc >= 2 ? argv[1] : "";
  printf("Looking for %d-cycle on cuckoo%d%d(\"%s\") with %d edges\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, EASYNESS);
  int us[MAXPATHLEN], nu, u, vs[MAXPATHLEN], nv, v; 
  for (int nonce = 0; nonce < EASYNESS; nonce++) {
    sha256edge(header, nonce, us, vs);
    if ((u = cuckoo[*us]) == *vs || (v = cuckoo[*vs]) == *us)
      continue; // ignore duplicate edges
    for (nu = 0; u; u = cuckoo[u]) {
      assert(nu < MAXPATHLEN);
      us[++nu] = u;
    }
    for (nv = 0; v; v = cuckoo[v]) {
      assert(nv < MAXPATHLEN);
      vs[++nv] = v;
    }
#ifdef SHOW
    for (int j=1; j<=SIZE; j++)
      if (!cuckoo[j]) printf("%2d:   ",j);
      else            printf("%2d:%02d ",j,cuckoo[j]);
    printf(" %x (%d,%d)\n", nonce,*us,*vs);
#endif
    if (us[nu] == vs[nv]) {
      int min = nu < nv ? nu : nv;
      for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
      int len = nu + nv + 1;
      printf("% 4d-cycle found at %d%%\n", len, (int)(nonce*100L/EASYNESS));
      if (len != PROOFSIZE)
        continue;
      while (nu--)
        cuckoo[us[nu]] = CYCLE | us[nu+1];
      while (nv--)
        cuckoo[vs[nv+1]] = CYCLE | vs[nv];
      for (cuckoo[*vs] = CYCLE | *us; len ; nonce--) {
        sha256edge(header, nonce, &u, &v);
        int c;
        if (cuckoo[c=u] == (CYCLE|v) || cuckoo[c=v] == (CYCLE|u)) {
          printf("%2d %08x (%d,%d)\n", --len, nonce, u, v);
          cuckoo[c] &= ~CYCLE;
        }
      }
      break;
    }
    while (nu--)
      cuckoo[us[nu+1]] = us[nu];
    cuckoo[*us] = *vs;
  }
  return 0;
}
