// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "cuckoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <set>

// assume EDGEBITS < 31
#define NNODES (2 * NEDGES)
#define MAXPATHLEN 8192

class cuckoo_ctx {
public:
  siphash_keys sip_keys;
  edge_t easiness;
  node_t *cuckoo;

  cuckoo_ctx(const char* header, const u32 headerlen, const u32 nonce, edge_t easy_ness) {
    easiness = easy_ness;
    cuckoo = (node_t *)calloc(1+NNODES, sizeof(node_t));
    assert(cuckoo != 0);
  }
  u64 bytes() {
    return (u64)(1+NNODES) * sizeof(node_t);
  }
  ~cuckoo_ctx() {
    free(cuckoo);
  }
  void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &sip_keys);
    memset(cuckoo, 0, (u64)(1+NNODES) * sizeof(node_t));
  }
  int path(node_t *cuckoo, node_t u, node_t *us) {
    int nu;
    for (nu = 0; u; u = cuckoo[u]) {
      if (++nu >= MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (nu < 0)
          printf("maximum path length exceeded\n");
        else printf("illegal % 4d-cycle\n", MAXPATHLEN-nu);
        exit(0);
      }
      us[nu] = u;
    }
    return nu;
  }
  
  typedef std::pair<node_t,node_t> edge;
  
  void solution(node_t *us, int nu, node_t *vs, int nv) {
    std::set<edge> cycle;
    unsigned n;
    cycle.insert(edge(*us, *vs));
    while (nu--)
      cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
    while (nv--)
      cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
    printf("Solution");
    for (edge_t nonce = n = 0; nonce < easiness; nonce++) {
      edge e(2*sipnode(&sip_keys, nonce, 0), 2*sipnode(&sip_keys, nonce, 1)+1);
      if (cycle.find(e) != cycle.end()) {
        printf(" %x", nonce);
        cycle.erase(e);
      }
    }
    printf("\n");
  }
  void solve() {
    node_t us[MAXPATHLEN], vs[MAXPATHLEN];
    for (node_t nonce = 0; nonce < easiness; nonce++) {
      node_t u0 = 2*sipnode(&sip_keys, nonce, 0);
      if (u0 == 0) continue; // reserve 0 as nil; v0 guaranteed non-zero
      node_t v0 = 2*sipnode(&sip_keys, nonce, 1)+1;
      node_t u = cuckoo[u0], v = cuckoo[v0];
      us[0] = u0;
      vs[0] = v0;
  #ifdef SHOW
      for (unsigned j=1; j<NNODES; j++)
        if (!cuckoo[j]) printf("%2d:   ",j);
        else           printf("%2d:%02d ",j,cuckoo[j]);
      printf(" %x (%d,%d)\n", nonce,*us,*vs);
  #endif
      int nu = path(cuckoo, u, us), nv = path(cuckoo, v, vs);
      if (us[nu] == vs[nv]) {
        int min = nu < nv ? nu : nv;
        for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
        int len = nu + nv + 1;
        printf("% 4d-cycle found at %d%%\n", len, (int)(nonce*100L/easiness));
        if (len == PROOFSIZE)
          solution(us, nu, vs, nv);
        continue;
      }
      if (nu < nv) {
        while (nu--)
          cuckoo[us[nu+1]] = us[nu];
        cuckoo[u0] = v0;
      } else {
        while (nv--)
          cuckoo[vs[nv+1]] = vs[nv];
        cuckoo[v0] = u0;
      }
    }
  }
};

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

int main(int argc, char **argv) {
  char header[HEADERLEN];
  memset(header, 0, HEADERLEN);
  int c, easipct = 50;
  u32 nonce = 0;
  u32 range = 1;
  struct timeval time0, time1;
  u32 timems;

  while ((c = getopt (argc, argv, "e:h:n:r:")) != -1) {
    switch (c) {
      case 'e':
        easipct = atoi(optarg);
        break;
      case 'h':
        memcpy(header, optarg, strlen(optarg));
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
    }
  }
  assert(easipct >= 0 && easipct <= 100);
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with %d%% edges, ", easipct);
  u64 easiness = easipct * (u64)NNODES / 100;
  cuckoo_ctx ctx(header, sizeof(header), nonce, easiness);
  u64 bytes = ctx.bytes();
  int unit;
  for (unit=0; bytes >= 10240; bytes>>=10,unit++) ;
  printf("using %d%cB memory at %lx.\n", bytes, " KMGT"[unit], (u64)ctx.cuckoo);

  for (u32 r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("nonce %d\n", nonce+r);
    ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);
  }
}
