// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2014 John Tromp

#include "cuckoo.h"
// algorithm parameters
#define MAXPATHLEN 8192
#define MAXSOLS 8

unsigned *cuckoo;

unsigned path(unsigned u, unsigned *us) {
  unsigned nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (++nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      assert(nu < MAXPATHLEN);
      printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu] = u;
  }
  return nu;
}

// largest odd number of u64's that fit in MAXPATHLEN-PROOFSIZE unsigned's
#define SOLMODU ((MAXPATHLEN-PROOFSIZE-2)/2 | 1)
#define SOLMODV (SOLMODU-1)

void storedge(u64 uv, u64 *usck, u64 *vsck) {
  int i = uv % SOLMODU;
  if (usck[i]) {
    i = uv % SOLMODV;
    assert(!vsck[i]);
    vsck[i] = uv;
  } else usck[i] = uv;
}

pthread_mutex_t setsol = PTHREAD_MUTEX_INITIALIZER;
unsigned sols[MAXSOLS][PROOFSIZE], nsols;

void solution(unsigned *us, unsigned nu, unsigned *vs, unsigned nv) {
  u64 *usck = (u64 *)&us[PROOFSIZE], *vsck = (u64 *)&vs[PROOFSIZE];
  unsigned u, v, n;
  for (int i=0; i<SOLMODU; i++)
    usck[i] = vsck[i] = 0L;
  storedge((u64)*us<<32 | *vs, usck, vsck);
  while (nu--)
    storedge((u64)us[(nu+1)&~1]<<32 | us[nu|1], usck, vsck); // u's in even position; v's in odd
  while (nv--)
    storedge((u64)vs[nv|1]<<32 | vs[(nv+1)&~1], usck, vsck); // u's in odd position; v's in even
  pthread_mutex_lock(&setsol);
  for (unsigned nonce = n = 0; nonce < EASINESS ; nonce++) {
    sipedge(nonce, &u, &v);
    u64 *c, uv = (u64)u<<32 | v;
    if (*(c = &usck[uv % SOLMODU]) == uv || *(c = &vsck[uv % SOLMODV]) == uv) {
      sols[nsols][n++] = nonce;
      *c = 0;
    }
  }
  nsols++;
  pthread_mutex_unlock(&setsol);
  assert(n == PROOFSIZE);
}

pthread_t threads[NTHREADS];

void *worker(void *tp) {
  int t = (pthread_t *)tp - threads;
  unsigned us[MAXPATHLEN], nu, u, vs[MAXPATHLEN], nv, v; 
  for (unsigned nonce = t; nonce < EASINESS; nonce += NTHREADS) {
    sipedge(nonce, us, vs);
    if ((u = cuckoo[*us]) == *vs || (v = cuckoo[*vs]) == *us)
      continue; // ignore duplicate edges
#ifdef SHOW
    for (int j=1; j<=SIZE; j++)
      if (!cuckoo[j]) printf("%2d:   ",j);
      else            printf("%2d:%02d ",j,cuckoo[j]);
    printf(" %x (%d,%d)\n", nonce,*us,*vs);
#endif
    if (us[nu = path(u, us)] == vs[nv = path(v, vs)]) {
      int min = nu < nv ? nu : nv;
      for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
      int len = nu + nv + 1;
      printf("% 4d-cycle found at %d:%d%%\n", len, t, (int)(nonce*100L/EASINESS));
      if (len == PROOFSIZE && nsols < MAXSOLS)
        solution(us, nu, vs, nv);
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

int main(int argc, char **argv) {
  assert(SIZE < 1L<<32);
  cuckoo = calloc(1+SIZE, sizeof(unsigned));
  assert(cuckoo);
  char *header = argc >= 2 ? argv[1] : "";
  setheader(header);
  printf("Looking for %d-cycle on cuckoo%d%d(\"%s\") with %u nodes and %u edges\n",
               PROOFSIZE, SIZEMULT, SIZESHIFT, header, SIZE, EASINESS);
  for (int t = 0; t < NTHREADS; t++)
    assert(pthread_create(&threads[t], NULL, worker, (void *)&threads[t]) == 0);
  for (int t = 0; t < NTHREADS; t++)
    assert(pthread_join(threads[t], NULL) == 0);
  for (int s = 0; s < nsols; s++) {
    printf("Solution");
    for (int i = 0; i < PROOFSIZE; i++)
      printf(" %x", sols[s][i]);
    printf("\n");
  }
  return 0;
}
