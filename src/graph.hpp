#include <stdio.h>
#include <assert.h>
#include "bitmap.hpp"

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

// cuck(at)oo graph with given two-power limit on number of edges (and on single partition nodes)
template <typename word_t>
class graph {
public:
  static const word_t NIL = ~0;

  struct link { // element of adjacency list
    word_t next;
    word_t to;
  };
  // typedef word_t proof[PROOFSIZE];

  word_t MAXEDGES;
  uint64_t MAXNODES;
  uint64_t nlinks; // aka halfedges, twice number of edges
  link *links;
  word_t *adjlist; // index into links array
  bitmap<u32> visited;
  word_t sols[MAXSOLS][PROOFSIZE];
  u32 nsols;

  graph(word_t maxedges) : visited(maxedges) {
    MAXEDGES = maxedges;
    MAXNODES = 2 * MAXEDGES;
    links = new link[MAXNODES];
    adjlist = new word_t[MAXNODES]; // index into links array
  }

  ~graph() {
    delete[] adjlist;
    delete[] links;
  }

  uint64_t bytes() {
    return MAXNODES * (sizeof(link) + sizeof(word_t)) + (MAXEDGES / 32) * sizeof(u32);
  }

  void reset() {
    memset(adjlist, ~0, sizeof(word_t[MAXNODES]));
    resetcounts();
  }

  void resetcounts() {
    nlinks = nsols = 0;
    // visited has entries set only during cycles() call
  }

  void add_edge(word_t u, word_t v) {
    v |= MAXEDGES; // distinguish partitions
    assert (u != NIL && v != NIL);
    word_t ulink = nlinks++;
    word_t vlink = nlinks++; // the two halfedges of an edge differ only in last bit
    links[ulink].next = adjlist[u];
    links[vlink].next = adjlist[v];
    links[adjlist[u] = ulink].to = u;
    links[adjlist[v] = vlink].to = v;
  }

  // remove lnk from u's adjacency list
  void remove_adj(word_t u, word_t lnk) {
    word_t *lp;
    for (lp = &adjlist[u]; *lp != lnk; lp = &links[*lp].next)
      assert(*lp != NIL);
    *lp = links[lnk].next;
    links[lnk].to = NIL; // mark lnk as detached
  }

  void remove_link(word_t lnk) {
    word_t u = links[lnk].to; // lnk is incident to node u
    // printf("remove_link(%d) to %d\n", lnk, u);
    if (u == NIL)
      return;
    remove_adj(u, lnk);
    if (adjlist[u] == NIL) {  // u has no more adjacencies
      for (word_t rl = adjlist[u ^ 1]; rl != NIL; rl = links[rl].next) {
        links[rl].to = NIL; // so no cycle through rl either
        remove_link(rl ^ 1);
      }
      adjlist[u ^ 1] = NIL;
    }
  }

  void cycles_with_link(u32 len, word_t u, word_t dest) {
    if (visited.test(u >> 1)) {
      // printf("already visited(%d >> 1)\n", u);
      return;
    }
    // printf("cycles with (%d,%d) length %d\n", dest, u, len);
    assert(u != NIL);
    if ((u ^ 1) == dest) {
      printf("  %d-cycle found\n", len);
      if (len == PROOFSIZE) {
        if (++nsols < MAXSOLS)
          memcpy(sols[nsols], sols[nsols-1], sizeof(sols[0]));
        else nsols--;
      }
      return;
    } else if (len == PROOFSIZE)
      return;
    word_t au1 = adjlist[u ^ 1];
    // printf("node %d has adjacency %d\n", u^1, links[au1 ^ 1].to);
    if (au1 != NIL) {
      visited.set(u >> 1);
      for (; au1 != NIL; au1 = links[au1].next) {
        sols[nsols][len] = au1/2;
        cycles_with_link(len+1, links[au1 ^ 1].to, dest);
      }
      visited.reset(u >> 1);
    }
  }

  u32 cycles() {
    while (nlinks) {
      nlinks -= 2;
      word_t u = links[nlinks].to, v = links[nlinks+1].to;
      if (u != NIL && v != NIL) {
        sols[nsols][0] = nlinks/2;
        cycles_with_link(1, u, v);
        remove_link(nlinks);
        remove_link(nlinks+1);
      }
    }
    return nsols;
  }

  void recordedge(const u32 i, const u32 u, const u32 v) {
    printf(" (%x,%x)", u, v);
  }

  void solution(u32 *us, int nu, u32 *vs, int nv) {
    printf("Nodes");
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
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
};
