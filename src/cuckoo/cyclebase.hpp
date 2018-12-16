#include <utility>
#include <stdio.h>
#include <assert.h>
#include <set>

#ifndef MAXCYCLES
#define MAXCYCLES 64 // single byte
#endif

struct edge {
  u32 u;
  u32 v;
  edge() : u(0), v(0) { }
  edge(u32 x, u32 y) : u(x), v(y) { }
};

struct cyclebase {
  // should avoid different values of MAXPATHLEN in different threads of one process
  static const u32 MAXPATHLEN = 16 << (EDGEBITS/3);

  int ncycles;
  word_t *cuckoo;
  u32 *pathcount;
  edge cycleedges[MAXCYCLES];
  u32 cyclelengths[MAXCYCLES];
  u32 prevcycle[MAXCYCLES];
  u32 us[MAXPATHLEN];
  u32 vs[MAXPATHLEN];

  void alloc() {
    cuckoo = (word_t *)calloc(NCUCKOO, sizeof(word_t));
    pathcount = (u32 *)calloc(NCUCKOO, sizeof(u32));
  }

  void freemem() { // not a destructor, as memory may have been allocated elsewhere, bypassing alloc()
    free(cuckoo);
    free(pathcount);
  }

  void reset() {
    resetcounts();
  }

  void resetcounts() {
    memset(cuckoo, -1, NCUCKOO * sizeof(word_t)); // for prevcycle nil
    memset(pathcount, 0, NCUCKOO * sizeof(u32));
    ncycles = 0;
  }

  int path(u32 u0, u32 *us) const {
    int nu;
    for (u32 u = us[nu = 0] = u0; pathcount[u]; ) {
      pathcount[u]++;
      u = cuckoo[u];
      if (++nu >= (int)MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (nu < 0)
          printf("maximum path length exceeded\n");
        else printf("illegal % 4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
        exit(0);
      }
      us[nu] = u;
    }
    return nu;
  }

  int pathjoin(u32 *us, int *pnu, u32 *vs, int *pnv) {
    int nu = *pnu, nv = *pnv;
    int min = nu < nv ? nu : nv;
    for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) min--;
    *pnu = nu; *pnv = nv;
    return min;
  }

  void addedge(u32 u0, u32 v0) {
    u32 u = u0 << 1, v = (v0 << 1) | 1;
    int nu = path(u, us), nv = path(v, vs);
    if (us[nu] == vs[nv]) {
      pathjoin(us, &nu, vs, &nv);
      int len = nu + nv + 1;
      printf("% 4d-cycle found\n", len);
      cycleedges[ncycles].u = u;
      cycleedges[ncycles].v = v;
      cyclelengths[ncycles++] = len;
      if (len == PROOFSIZE)
        solution(us, nu, vs, nv);
      assert(ncycles < MAXCYCLES);
    } else if (nu < nv) {
      pathcount[us[nu]]++;
      while (nu--)
        cuckoo[us[nu+1]] = us[nu];
      cuckoo[u] = v;
    } else {
      pathcount[vs[nv]]++;
      while (nv--)
        cuckoo[vs[nv+1]] = vs[nv];
      cuckoo[v] = u;
    }
  }

  void recordedge(const u32 i, const u32 u, const u32 v) {
    printf(" (%x,%x)", u, v);
  }

  void solution(u32 *us, int nu, u32 *vs, int nv) {
    printf("Nodes");
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
    while (nu--)
      recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
      recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    printf("\n");
#if 0
    for (u32 nonce = n = 0; nonce < NEDGES; nonce++) {
      edge e(2*sipnode(&sip_keys, nonce, 0), 2*sipnode(&sip_keys, nonce, 1)+1);
      if (cycle.find(e) != cycle.end()) {
        printf(" %x", nonce);
        cycle.erase(e);
      }
    }
    printf("\n");
#endif
  }

  int sharedlen(u32 *us, int nu, u32 *vs, int nv) {
    int len = 0;
    for (; nu-- && nv-- && us[nu] == vs[nv]; len++) ;
    return len;
  }

  void cycles() {
    int len, len2;
    word_t us2[MAXPATHLEN], vs2[MAXPATHLEN];
    for (int i=0; i < ncycles; i++) {
      word_t u = cycleedges[i].u, v = cycleedges[i].v;
      int   nu = path(u, us),    nv = path(v, vs);
      word_t root = us[nu]; assert(root == vs[nv]);
      int i2 = prevcycle[i] = cuckoo[root];
      cuckoo[root] = i;
      if (i2 < 0) continue;
      int rootdist = pathjoin(us, &nu, vs, &nv);
      do  {
        printf("chord found at cycleids %d %d\n", i2, i);
        word_t u2 = cycleedges[i2].u, v2 = cycleedges[i2].v;
        int nu2 = path(u2, us2), nv2 = path(v2, vs2);
        word_t root2 = us2[nu2]; assert(root2 == vs2[nv2] && root == root2);
        int rootdist2 = pathjoin(us2, &nu2, vs2, &nv2);
        if (us[nu] == us2[nu2]) {
          len  = sharedlen(us,nu,us2,nu2) + sharedlen(us,nu,vs2,nv2);
          len2 = sharedlen(vs,nv,us2,nu2) + sharedlen(vs,nv,vs2,nv2);
          if (len + len2 > 0) {
#if 0
            word_t ubranch = us[nu-len], vbranch = vs[nv-len2];
            addpath(ubranch, vbranch, len+len2);
            addpath(ubranch, vbranch, len+len2);
#endif
            printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*(len+len2), (int)(i*100L/ncycles));
          }
        } else {
          int rd = rootdist - rootdist2;
          if (rd < 0) {
            if (nu+rd > 0 && us2[nu2] == us[nu+rd]) {
              int len = sharedlen(us,nu+rd,us2,nu2) + sharedlen(us,nu+rd,vs2,nv2);
              if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
            } else if (nv+rd > 0 && vs2[nv2] == vs[nv+rd]) {
              int len = sharedlen(vs,nv+rd,us2,nu2) + sharedlen(vs,nv+rd,vs2,nv2);
              if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
            }
          } else if (rd > 0) {
            if (nu2-rd > 0 && us[nu] == us2[nu2-rd]) {
              int len = sharedlen(us2,nu2-rd,us,nu) + sharedlen(us2,nu2-rd,vs,nv);
              if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
            } else if (nv2-rd > 0 && vs[nv] == vs2[nv2-rd]) {
              int len = sharedlen(vs2,nv2-rd,us,nu) + sharedlen(vs2,nv2-rd,vs,nv);
              if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
            }
          } // else cyles are disjoint
        }
      } while ((i2 = prevcycle[i2]) >= 0);
    }
  }
};
