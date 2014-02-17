// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
// algorithm parameters
#define MAXPATHLEN 8192

// used to simplify nonce recovery
#define CYCLE 0x80000000
unsigned *cuckoo, solus[MAXPATHLEN], solnu, solvs[MAXPATHLEN], solnv; 
pthread_t threads[NTHREADS];
pthread_mutex_t setsol = PTHREAD_MUTEX_INITIALIZER;

void *worker(void *tp) {
  int t = (pthread_t *)tp - threads;
  unsigned us[MAXPATHLEN], nu, u, vs[MAXPATHLEN], nv, v; 
  for (unsigned nonce = t; nonce < EASINESS; nonce += NTHREADS) {
    sipedge(nonce, us, vs);
    if ((u = cuckoo[*us]) == *vs || (v = cuckoo[*vs]) == *us)
      continue; // ignore duplicate edges
    for (nu = 0; u; u = cuckoo[u]) {
      assert(nu < MAXPATHLEN-1);
      us[++nu] = u;
    }
    for (nv = 0; v; v = cuckoo[v]) {
      assert(nv < MAXPATHLEN-1);
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
      printf("% 4d-cycle found at %d:%d%%\n", len, t, (int)(nonce*100L/EASINESS));
      if (len != PROOFSIZE)
        continue;
      pthread_mutex_lock(&setsol);
      for (solnu = nu, nu = 0; nu <= solnu; nu++)
        solus[nu] = us[nu];
      for (solnv = nv, nv = 0; nv <= solnv; nv++)
        solvs[nv] = vs[nv];
      pthread_mutex_unlock(&setsol);
      break;
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

int main(int argc, char **argv) {
  // 6 largest sizes 131 928 529 330 729 132 not implemented
  assert(SIZE < (unsigned)CYCLE);
  assert(cuckoo = calloc(1+SIZE, sizeof(unsigned)));
  char *header = argc >= 2 ? argv[1] : "";
  setheader(header);
  printf("Looking for %d-cycle on cuckoo%d%d(\"%s\") with %d edges\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, EASINESS);
  for (int t = 0; t < NTHREADS; t++)
    assert(pthread_create(&threads[t], NULL, worker, (void *)&threads[t]) == 0);
  for (int t = 0; t < NTHREADS; t++)
    assert(pthread_join(threads[t], NULL) == 0);
  if (solnu + solnv == 0)
    return 0;
  while (solnu--)
    cuckoo[solus[solnu]] = CYCLE | solus[solnu+1];
  while (solnv--)
    cuckoo[solvs[solnv+1]] = CYCLE | solvs[solnv];
  cuckoo[*solvs] = CYCLE | *solus;
  unsigned len = 0, u, v;
  for (unsigned nonce = 0; nonce < EASINESS ; nonce++) {
    sipedge(nonce, &u, &v);
    unsigned c;
    if (cuckoo[c=u] == (CYCLE|v) || cuckoo[c=v] == (CYCLE|u)) {
      printf("%2d %08x (%d,%d)\n", len++, nonce, u, v);
      cuckoo[c] &= ~CYCLE;
    }
  }
  return 0;
}
