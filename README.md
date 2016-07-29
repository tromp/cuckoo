
Cuckoo Cycle
============
Whitepaper at
https://github.com/tromp/cuckoo/blob/master/doc/cuckoo.pdf?raw=true

Blog article explaining Cuckoo Cycle at
http://cryptorials.io/beyond-hashcash-proof-work-theres-mining-hashing

This repo is Linux based. Microsoft Windows friendly code at
https://github.com/Genoil/cuckoo

Cuckoo Cycle is the first graph-theoretic proof-of-work, and by far the most
memory bound, with memory latency dominating the mining runtime, yet with
instant verification.

Proofs take the form of a length 42 cycle in a bipartite graph with N nodes and
N/2 edges, with N scalable from millions to billions and beyond.

This makes verification trivial: compute the 42x2 edge endpoints with one
initialising sha256 and 84 very cheap siphash-2-4 hashes, check that each
endpoint occurs twice, and that you come back to the starting point only after
traversing 42 edges (this also makes Cuckoo Cycle, unlike Hashcash, relatively
immune from Grover's quantum search algorithm).

A final sha256 hash on the sorted 42 nonces can check whether the 42-cycle
meets a difficulty target.

This is implemented in just 173 lines of C code (files src/cuckoo.h and
src/cuckoo.c).

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

Memory-wise, it uses N/2 bits to maintain a subset of all edges (potential
cycle edges) and N additional bits (or N/2^k bits with corresponding slowdown)
to trim the subset in a series of edge trimming rounds.
This is the phase that takes the vast majority of (latency dominated) runtime.

Once the subset is small enough, an algorithm inspired by Cuckoo Hashing
is used to recognise all cycles, and recover those of the right length.

The runtime of a single proof attempt on a high end x86 is 5.5min/GB
single-threaded, or 1.5min/GB for 8 threads.

I claim that this implementation is a reasonably optimal Cuckoo miner,
secondly, that trading off memory for running time,
as implemented in tomato_miner.h,
incurs at least one order of magnitude extra slowdown,
and finally, that cuda_miner.cu is a reasonably optimal GPU miner.
The latter runs about 4x-5x faster on an NVIDA GTX 980
than on an Intel Core-i7 CPU. 
To that end, I offer the following bounties:

CPU Speedup Bounty
--------------
$500 for an open source implementation that finds 42-cycles twice as fast,
possibly using more memory.

Linear Time-Memory Trade-Off Bounty
--------------
$500 for an open source implementation that uses at most N/k bits while finding
42-cycles up to 15 k times slower, for any k>=2.

Both of these bounties require N ranging over {2^28,2^30,2^32} and #threads
ranging over {1,2,4,8}, and further assume a high-end Intel Core i7 or Xeon and
recent gcc compiler with regular flags as in my Makefile.

GPU Speedup Bounty
--------------
$500 for an open source implementation for a consumer GPU combo
that finds 42-cycles twice as fast as cuda_miner.cu on comparable hardware.
Again with N ranging over {2^28,2^30,2^32}.

These bounties are admittedly modest in size,
but then claiming them might only require one or two insightful tweaks to
my existing implementations.

I invite anyone who'd like to see my claims tested to extend any of these
bounties.

Happy bounty hunting!

A less wasteful POW
--------------

A common criticism of POW is that it is immensely wasteful and serves no useful purpose.
In this regard, Cuckoo Cycle may be more useful in that it encourages the development
of energy efficient DRAM, which will offset the POW energy use with huge energy savings in
general purpose computing. Several suggestions for improved DRAM design can already be found
in the literature, e.g.
<a href="https://www.cs.utah.edu/~rajeev/pubs/isca10.pdf">Rethinking DRAM design and organization for energy-constrained multi-cores</a> and 
<a href="mbsullivan.info/attachments/papers/yoon2012dgms.pdf">The Dynamic Granularity Memory System</a>
but face a chicken-and-egg adoption problem due to required changes in memory controller design.
