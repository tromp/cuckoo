#include <stdio.h>
#include <assert.h>
#include "bitmap.hpp"
#include <new>

typedef word_t proof[PROOFSIZE];

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
  u32 MAXSOLS;
  proof *sols;
  u32 nsols;
  u32 nauxremovals;

  graph(word_t maxedges, u32 maxsols) : visited(maxedges) {
    MAXEDGES = maxedges;
    MAXNODES = 2 * MAXEDGES;
    MAXSOLS = maxsols;
    links = new link[MAXNODES];
    adjlist = new word_t[MAXNODES]; // index into links array
    sols = new proof[MAXSOLS];
  }

  graph(word_t maxedges, u32 maxsols, void *sharemem) : visited(maxedges) {
    MAXEDGES = maxedges;
    MAXNODES = 2 * MAXEDGES;
    MAXSOLS = maxsols;
    links = new (sharemem) link[MAXNODES];
    adjlist = new ((char*)sharemem + sizeof(link[MAXNODES])) word_t[MAXNODES]; // index into links array
    sols = new ((char*)sharemem + sizeof(link[MAXNODES]) + sizeof(word_t[MAXNODES])) proof[MAXSOLS];
  }

  ~graph() {
    delete[] adjlist;
    delete[] links;
    delete[] sols;
  }

  // total size of new-operated data, excludes visited bitmap of MAXEDGES bits
  uint64_t bytes() {
    return sizeof(link[MAXNODES]) + sizeof(word_t[MAXNODES]) + sizeof(proof[MAXSOLS]);
  }

  void reset() {
    memset(adjlist, NIL, sizeof(word_t[MAXNODES]));
    resetcounts();
  }

  void resetcounts() {
    nlinks = nsols = nauxremovals = 0;
    // visited has entries set only during cycles() call
  }

  void cycles_with_link(u32 len, word_t u, word_t dest) {
    if (visited.test(u >> 1))
      return;
    if ((u ^ 1) == dest) {
      printf("  %d-cycle found\n", len);
      if (len == PROOFSIZE) {
        if (++nsols < MAXSOLS)
          memcpy(sols[nsols], sols[nsols-1], sizeof(sols[0]));
        else nsols--;
      }
      return;
    }
    if (len == PROOFSIZE)
      return;
    word_t au1 = adjlist[u ^ 1];
    if (au1 != NIL) {
      visited.set(u >> 1);
      for (; au1 != NIL; au1 = links[au1].next) {
        sols[nsols][len] = au1/2;
        cycles_with_link(len+1, links[au1 ^ 1].to, dest);
      }
      visited.reset(u >> 1);
    }
  }

  void add_edge(word_t u, word_t v) {
    v |= MAXEDGES; // distinguish partitions
    if (adjlist[u ^ 1] != NIL && adjlist[v ^ 1] != NIL) { // possibly part of a cycle
      sols[nsols][0] = nlinks/2;
      cycles_with_link(1, u, v);
    }
    word_t ulink = nlinks++;
    word_t vlink = nlinks++; // the two halfedges of an edge differ only in last bit
    assert (vlink != NIL);   // don't want to confuse link with NIL
    links[ulink].next = adjlist[u];
    links[vlink].next = adjlist[v];
    links[adjlist[u] = ulink].to = u;
    links[adjlist[v] = vlink].to = v;
  }
};
