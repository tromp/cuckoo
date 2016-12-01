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

This is implemented in just 173 lines of C code (files src/cuckoo.h and src/cuckoo.c).

From this point of view, Cuckoo Cycle is a very simple PoW,
requiring hardly any code, time, or memory to verify.

Finding a 42-cycle, on the other hand, is far from trivial,
requiring considerable resources, and some luck
(for a given header, the odds of its graph having a L-cycle are about 1 in L).

An indirectly useful Proof of Work
--------------
Global bitcoin mining consumes hundreds of megawatts, which many people have characterized
as a colossal waste. Meanwhile, datacenters worldwide consume thousands of megawatts,
an estimated 25-40% of which is spent on DRAM memory. Quoting from
<a href="https://www.cs.utah.edu/~rajeev/pubs/isca10.pdf">Rethinking DRAM design and organization for energy-constrained multi-cores</a>,
modern DRAM architectures are ill-suited for energy-efficient operation because
they are designed to fetch much more data than required, having long been optimized for cost-per-bit
rather than energy efficiency.
Thus there is enormous energy savings potential in accelerating the development of more efficient
DRAM designs. While this paper and others like
<a href="http://mbsullivan.info/attachments/papers/yoon2012dgms.pdf">The Dynamic Granularity Memory System</a>
have proposed several sensible and promising design improvements, memory manufacturers have
taken a wait-and-see approach, likely due to the need for more advanced memory controllers, which they don't develop
themselves, and uncertainty about market demand. However, a widely adopted PoW whose very bottleneck
is purely random accesses to billions of individual bits would provide such demand.
The world has little need for the extremely specialized SHA256 computation being efficient.
But it stands to benefit a lot from more energy efficient random access memories
(that, unlike SRAM, also remain very cost efficient).

ASICs
--------------
Solving sufficiently large Cuckoo Cycle instances requires the cooperation of
computing cores and memory chips.
The latter need to be optimized for price and energy use per randomly accessed bit.
While currently suboptimal from an energy-efficiency viewpoint, commodity mass production
makes DRAM chips the only cost effective ASICs for random bit access.
Since computing cores only need to be able to saturate the DRAM memory bandwidth,
they need to be optimized to a much lesser degree, thereby avoiding an ASICs arm race.
The most cost effective Cuckoo Cycle mining hardware should consist of a relatively
cheap and tiny many core memory controller that needs to be paired with commodity DRAM chips,
where the latter dominate both the hardware and energy cost (about 1 Watt per DRAM chip).

Cycle finding
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

Performance
--------------
The runtime of a single proof attempt on a 4GHz i7-4790K is 2.3 minutes
for a single-threaded cuckoo32stx8 using 768MB, or 1.1 minutes for 8 threads.
The 16x smaller cuckoo28stx8 takes only 7.4 seconds single-threaded.

I claim that this implementation is a reasonably optimal Cuckoo miner,
secondly, that trading off memory for running time,
as implemented in tomato_miner.h,
incurs at least one order of magnitude extra slowdown,
and finally, that cuda_miner.cu is a reasonably optimal GPU miner.
The latter runs about 4x faster on an NVIDA GTX 980
than on an Intel Core-i7 CPU. 
To that end, I offer the following bounties:

CPU Speedup Bounty
--------------
$2000 for an open source implementation that finds 42-cycles twice as fast,
possibly using more memory.

Linear Time-Memory Trade-Off Bounty
--------------
$2000 for an open source implementation that uses at most N/k bits while finding
42-cycles up to 10 k times slower, for any k>=2.

Both of these bounties require N ranging over {2^28,2^30,2^32} and #threads
ranging over {1,2,4,8}, and further assume a high-end Intel Core i7 or Xeon and
recent gcc compiler with regular flags as in my Makefile.

GPU Speedup Bounty
--------------
$1000 for an open source implementation for a consumer GPU combo
that finds 42-cycles twice as fast as cuda_miner.cu on comparable hardware.
Again with N ranging over {2^28,2^30,2^32}.

The Makefile defines corresponding targets cpubounty, tmtobounty, and gpubounty.
These bounties are admittedly modest in size,
but then claiming them might only require one or two insightful tweaks to
my existing implementations.

Half bounties
--------------
In order to minimize the risk of missing out on less drastic improvements,
I offer half of the above bounties for improvements within sqrt(2) of the target.

I invite anyone who'd like to see my claims tested to donate to
the Cuckoo Cycle Bounty Fund at

<a href="https://blockchain.info/address/1CnrpdKtfF3oAZmshyVC1EsRUa25nDuBvN">1CnrpdKtfF3oAZmshyVC1EsRUa25nDuBvN</a> (wallet balance as of Nov 1, 2016: 6.11 BTC)

I intend for the total bounty value to stay ahead of funding levels. Happy bounty hunting!

Cryptocurrencies using Cuckoo Cycle
--------------
<a href="https://github.com/ignopeverell/grin">Minimal implementation of the MimbleWimble protocol</a>
