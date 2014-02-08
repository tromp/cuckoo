cuckoo
======

Mining is generally considered to be inherently power hungry but it need not be.
It’s a consequence of making the proof of work computationally intensive.
If computation is minimized in favor of random access to gigabytes of memory
(incurring long latencies), then mining will require large investments in RAM
but relatively little power.

Bitcoin mining is all computation and no memory.

Litecoin mining requires 128KB of memory per scrypt instance but only 1024
random accesses, with negligable latency.

Primecoin mining has a sieving and a modular exponentiation component, the latter of
which is pure computation, while the former requires a few megabytes of memory
(with non-random access).

Protoshares mining with Momentum requires 512MB but performs at least one complex SHA512
computation for each memory access, with a choice of algorithms some of which avoid random access.

The memory requirements above are not absolute but allow a trade-off between time and memory.

Cuckoo Cycle represents a breakthrough in three important ways:

1) it performs only one very cheap siphash computation for about 3.3 random accesses to memory,

2) its memory requirement can be set arbitrarily and doesn't allow for any time-memory trade-off.

3) verification of the proof of work is instant, requiring 1 sha256 and 42 siphash computations.

Runtime in Cuckoo Cycle is completely dominated by memory latency. It promotes the use
of commodity general-purpose hardware over custom designed single-purpose hardware.

Other features:

4) proofs take the form of a length 42 cycle in the Cuckoo graph

5) it has a natural notion of difficulty, namely the number of edges in the graph;
   above about 60% of size, a 42-cycle is almost guaranteed, but below 50% the probability
   starts to fall sharply.

6) running time is under 24s/GB for the current implementation on high end x86.
