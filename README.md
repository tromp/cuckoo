cuckoo
======

Cuckoo Cycle is the first graph-theoretical Proof-Of-Work.
Keyed hash functions define arbitrarily large random graphs,
in which certain fixed-size subgraphs occur with suitably small probability.
Asking for a cycle, or a clique, is a close analogue of asking for
a chain or cluster of primes numbers, which were adopted as the
number-theoretic Proof-of-work in Primecoin and Riecoin, respectively.


Cuckoo Cycle represents a breakthrough in three important ways:

1) it performs only one very cheap siphash computation for each random accesses to memory,

2) its memory requirement can be set arbitrarily and appears to resist time-memory trade-offs.

3) verification of the proof of work is instant, requiring 2 sha256 and 42x2 siphash computations.

Runtime in Cuckoo Cycle is dominated by memory latency. This promotes the use
of commodity general-purpose hardware over custom designed single-purpose hardware.

Other features:

4) proofs take the form of a length 42 cycle in the Cuckoo graph.

5) it has a natural notion of (base) difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply.

6) running time new implementation on high end x86 is 1.6s/MB single-threaded,
   and 4.5mins/GB for 8 threads. it will take a superior GPU implementation to make
   Cuckoo Cycle requiring 1GB run in reasonable time.

7) it can optionally run without edge trimming, using about 21 times more memory, making
   1.65 random memory accesses per siphash, with a speedup that grows with size.
