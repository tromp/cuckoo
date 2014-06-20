cuckoo
======

Cuckoo Cycle is the first graph-theoretic proof-of-work.
Keyed hash functions define arbitrarily large random graphs,
in which certain fixed-size subgraphs occur with suitably small probability.
Asking for a cycle, or a clique, is a close analogue of asking for
a chain or cluster of primes numbers, which were adopted as the
number-theoretic proof-of-work in Primecoin and Riecoin, respectively.
The latest implementaton incorporates a huge memory savings proposed by Dave Andersen in

http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html


Cuckoo Cycle represents a breakthrough in three important ways:

1) it performs only one very cheap siphash computation for each random access to memory,

2) (intended) memory usage grows linearly with graph size, which can be set arbitrarily.
   there may be very limited opportunity to reduce memory usage without undue slowdown.

3) verification of the proof of work is instant, requiring 2 sha256 and 42x2 siphash computations.

Runtime in Cuckoo Cycle is dominated by memory latency (67%). This promotes the use
of commodity general-purpose hardware over custom designed single-purpose hardware.

Other features:

4) proofs take the form of a length 42 cycle in the Cuckoo graph.

5) it has a natural notion of (base) difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply.

6) running time on high end x86 is 9min/GB single-threaded, and 1min/GB for 20 threads.

Please read the latest version of the whitepaper for more details:

https://github.com/tromp/cuckoo/blob/master/cuckoo.pdf?raw=true
