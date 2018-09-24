#include <stdio.h>
#include <assert.h>
#include "bitmap.hpp"
#include <new>

typedef word_t proof[PROOFSIZE];

// cuck(at)oo graph with given limit on number of edges (and on single partition nodes)
template <typename word_t>
class graph {
public:
  static const word_t NIL = ~(word_t)0;

  struct link { // element of adjacency list
    word_t next;
    word_t to;
  };
  // typedef word_t proof[PROOFSIZE];

  word_t MAXEDGES;
  word_t MAXNODES;
  word_t nlinks; // aka halfedges, twice number of edges
  link *links;
  word_t *adjlist; // index into links array
  bitmap<u32> visited;
  u32 MAXSOLS;
  proof *sols;
  u32 nsols;
  bool shared_mem;

  graph(word_t maxedges, word_t maxnodes, u32 maxsols) : visited(maxedges) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new word_t[2*MAXNODES]; // index into links array
    links   = new link[2*MAXEDGES];
    sols    = new proof[MAXSOLS];
    shared_mem = false;
    visited.clear();
  }

  ~graph() {
    if (!shared_mem) {
      delete[] adjlist;
      delete[] links;
    }
    delete[] sols;
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, void *sharemem) : visited(maxedges) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    char *bytes = (char *)sharemem;
    adjlist = new (bytes) word_t[2*MAXNODES]; // index into links array
    links   = new (bytes + sizeof(word_t[2*MAXNODES])) link[2*MAXEDGES];
    sols    = new  proof[MAXSOLS];
    shared_mem = true;
    visited.clear();
  }

  // total size of new-operated data, excludes sols and visited bitmap of MAXEDGES bits
  uint64_t bytes() {
    return sizeof(word_t[2*MAXNODES]) + sizeof(link[2*MAXEDGES]); //  + sizeof(proof[MAXSOLS]);
  }

  void reset() {
    memset(adjlist, (char)NIL, sizeof(word_t[2*MAXNODES]));
    resetcounts();
  }

  void resetcounts() {
    nlinks = nsols = 0;
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
    assert(u < MAXNODES);
    assert(v < MAXNODES);
    v += MAXNODES; // distinguish partitions
    if (adjlist[u ^ 1] != NIL && adjlist[v ^ 1] != NIL) { // possibly part of a cycle
      sols[nsols][0] = nlinks/2;
      assert(!visited.test(u >> 1));
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

  void nodecount() {
    word_t nu=0, nv=0;
    for (word_t i=0; i < MAXNODES; i+=2) {
      if (adjlist[i] != NIL || adjlist[i^1] != NIL) nu++;
      if (adjlist[MAXNODES+i] != NIL || adjlist[MAXNODES+(i^1)] != NIL) nv++;
    }
    printf("%d u %d v\n", nu, nv);
  }
};
