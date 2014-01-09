cuckoo
======

Cuckoo Cycle is a new proof of work system with the following features

1) proofs take the form of a length 42 cycle in the Cuckoo graph,
   so that verification only requires computing 42 hashes.

2) the graph size (number of nodes) can scale from 1*2^10 to 7*2^29
   with 4 bytes needed per node, so memory use scales from 4KB to 14GB

3) runing time is roughly linear in memory, at under 1s/MB

4) there is no time-memory trade-off, and memory access patterns are the worst possible,
   making the algorithm constrained by memory latency
 
5) it has a natural notion of difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply
