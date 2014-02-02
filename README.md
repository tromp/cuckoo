cuckoo
======

Cuckoo Cycle is a new proof of work system with the following features

1) proofs take the form of a length 42 cycle in the Cuckoo graph,
   so that verification only requires computing 42 hashes.

2) the graph size can scale from 4 to 2^32 nodes,
   with 4 bytes needed per node, so memory use scales from 1KB to 16GB.
   Use of 4GB+ should make it somewhat resistent to botnets.

3) running time is roughly linear in memory, at under 24s/GB for the current
   implementation on high end x86.

4) no time-memory trade-off (TMTO) is known, and memory access patterns are the worst possible,
   making the algorithm constrained by memory latency.
 
5) it has a natural notion of difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply.

6) parallelization must allow many processing elements simultaneous random access to
   global shared memory, likely benefiting many other applications.
