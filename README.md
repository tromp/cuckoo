
Cuckoo Cycle
============
Whitepaper at https://github.com/tromp/cuckoo/blob/master/doc/cuckoo.pdf?raw=true

Cuckoo Cycle is the first graph-theoretic proof-of-work,
and by far the most memory bound, with memory latency
dominating the mining runtime, yet with instant verification.

Proofs take the form of a length 42 cycle in a bipartite graph
with N nodes and N/2 edges, with N scalable from millions to billions and beyond.

This makes verification trivial: compute the 42x2 edge endpoints
with one initialising sha256 and 84 very cheap siphash-2-4 hashes,
check that each endpoint occurs twice, and that you come back to the
starting point only after traversing 42 edges.

A final sha256 hash on the sorted 42 nonces can check whether the 42-cycle meets a difficulty target.

This is implemented in just 157 lines of C code (files src/cuckoo.h and src/cuckoo.c).

From this point of view, Cuckoo Cycle is a very simple PoW,
requiring hardly any code, time, or memory to verify.

Finding a 42-cycle, on the other hand, is far from trivial,
requiring considerable resources, and some luck
(for a given header, the odds of its graph having a 42-cycle are about 2.5%).

Where Satoshi Nakamoto aimed for "one-CPU-one-vote", Cuckoo Cycle aims for
1 memory bank + 1 virtual core = 1 vote
--------------

The algorithm implemented in cuckoo_miner.h runs in time linear in N.
(Note that running in sub-linear time is out of the question, as you could
only compute a fraction of all edges, and the odds of all 42 edges of a cycle
occurring in this fraction are astronomically small).

Memory-wise, it uses N/2 bits to maintain a subset of all edges (potential cycle edges)
and N additional bits (or N/2^k bits with corresponding slowdown)
to trim the subset in a series of edge trimming rounds.
This is the phase that takes the vast majority of (latency dominated) runtime.

Once the subset is small enough, an algorithm inspired by Cuckoo Hashing
is used to recognise all cycles, and recover those of the right length.

The runtime of a single proof attempt on a high end x86 is 9min/GB single-threaded, or 1min/GB for 20 threads.

I claim that this implementation is a reasonably optimal Cuckoo miner,
and that trading off memory for running time, as implemented in tomato_miner.h,
incurs at least one order of magnitude extra slowdown.
I'd further like to claim that GPUs cannot achieve speed parity with CPUs.

To that end, I offer the following bounties:

Speedup Bounty
--------------
$500 for an open source implementation that finds 42-cycles twice as fast, possibly using more memory.

Linear Time-Memory Trade-Off Bounty
--------------
$500 for an open source implementation that uses at most N/k bits while running up to 15 k times slower,
for any k>=2.

Both of these bounties require N ranging over {2^28,2^30,2^32} and #threads ranging over {1,2,4,8},
and further assume a high-end Intel Core i7 or Xeon and recent gcc compiler with regular flags as in my Makefile.

GPU Speed Parity Bounty
--------------
$500 for an open source implementation for a consumer GPU
that is as fast as an Intel Core i7 running a single-threaded instance on each of 8 cores.
Again with N ranging over {2^28,2^30,2^32}.

Note that there is already a cuda_miner.cu, my attempted port of the edge trimming part
of cuckoo_miner.c to CUDA, but while it seems to run ok for medium graph sizes,
it crashes my computer at larger sizes, and I haven't had the chance to debug the issue yet
(I prefer to let my computer work on solving 8x8 connect-4:-)
Anyway, this could make a good starting point.

These bounties are to expire at the end of 2015. They are admittedly modest in size, but then
claiming them might only require one or two insightful tweaks to my existing implementations.

I invite anyone who'd like to see my claims refuted to extend any of these bounties
with amounts of your crypto-currency of choice.

(I don't have any crypto-currencies to offer myself, but if need be, I might be able to convert
a bounty through a trusted 3rd party)

Happy bounty hunting!

-John
