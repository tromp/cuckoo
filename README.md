cuckoo
======

Mining is generally considered to be inherently power hungry but it need not be.
It’s a consequence of making the proof of work computationally intensive.
If computation is minimized in favor of random access to gigabytes of memory
(incurring long latencies), then mining will require large investments in RAM
but relatively little power.

Cuckoo Cycle represents a breakthrough in three important ways:

1) it performs only one very cheap siphash computation for one random accesses to memory,

2) its memory requirement can be set arbitrarily and appears to be time-memory trade-off resistant.

3) verification of the proof of work is instant, requiring 2 sha256 and 42 siphash computations.

Runtime in Cuckoo Cycle is completely dominated by memory latency. It promotes the use
of commodity general-purpose hardware over custom designed single-purpose hardware.

Other features:

4) proofs take the form of a length 42 cycle in the Cuckoo graph.

5) it has a natural notion of (base) difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply.

6) running time for the new implementation on high end x86 is 1.6s/MB single-threaded,
   and 4.5mins/GB for 8 threads. it will take a superior GPU implementation to make
   Cuckoo Cycle requiring 1GB run in reasonable time.

7) botnet resistance is out of the window:-(
