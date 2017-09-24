Cuckoo Cycle
============
Newsflash: The Cuckoo Cycle <a href="https://github.com/tromp/cuckoo/blob/master/LICENSE.txt">LICENSE</a>
has been amended with a FAIR MINING clause. This hopefully allows coin developers to receive compensation
without needing to resort to unfair practices like premines (e.g. ICOs), instamines, or large mining taxes
(e.g. 10% of monetary base).

Whitepaper at
https://github.com/tromp/cuckoo/blob/master/doc/cuckoo.pdf?raw=true

Blog article explaining Cuckoo Cycle at
http://cryptorials.io/beyond-hashcash-proof-work-theres-mining-hashing

This repo is Linux based. Microsoft Windows friendly code at
https://github.com/Genoil/cuckoo

Cuckoo Cycle is the first graph-theoretic proof-of-work,
and the most memory bound, yet with instant verification.

Proofs take the form of a length 42 cycle in a bipartite graph with N nodes and
N/2 edges, with N scalable from millions to billions and beyond.

This makes verification trivial: compute the 42x2 edge endpoints with one
initialising sha256 and 84 very cheap siphash-2-4 hashes, check that each
endpoint occurs twice, and that you come back to the starting point only after
traversing 42 edges (this also makes Cuckoo Cycle, unlike Hashcash, relatively
immune from Grover's quantum search algorithm).

A final sha256 hash on the sorted 42 nonces can check whether the 42-cycle
meets a difficulty target.

This is implemented in under 200 lines of C code (files src/{siphash.h,cuckoo.h,cuckoo.c}).

From this point of view, Cuckoo Cycle is a very simple PoW,
requiring hardly any code, time, or memory to verify.

Finding a 42-cycle, on the other hand, is far from trivial,
requiring considerable resources, and some luck
(for a given header, the odds of its graph having a L-cycle are about 1 in L).

The memory efficient miner uses 3 bits per edge and is bottlenecked by
accessing random 2-bit counters, making it memory latency bound.  The roughly
4x faster latency avoiding miner, a rewrite from xenoncat's bounty winning solver,
uses 58 bits per edge and is bottlenecked by bucket sorting. making it memory bandwidth bound.
It is not clear which method is more energy efficient.

An indirectly useful Proof of Work
--------------

Traditional DRAM architectures are ill-suited for energy-efficient operation because
they are designed to fetch much more data than required, having long been optimized for cost-per-bit
rather than energy efficiency.
Thus there is enormous energy savings potential in accelerating the development of more efficient
DRAM designs. Papers like
<a href="https://www.cs.utah.edu/~rajeev/pubs/isca10.pdf">Rethinking DRAM design and organization for energy-constrained multi-cores</a> and
<a href="http://mbsullivan.info/attachments/papers/yoon2012dgms.pdf">The Dynamic Granularity Memory System</a>
have proposed several sensible and promising designs with large energy efficiency improvements,
which a latency bound proof of work could help realize.

ASICs
--------------
Cuckoo Cycle avoids the traditional ASICs arm race since it only takes a few dozen tiny siphash
computing cores to saturate the DRAM memory bandwidth, at which point
any further performance improvements will go to waste as the ASIC sits idle waiting for memory.
Using anything other than cheap commodity DRAM is going to be cost prohibitive.
Even the 10x faster SRAM is about 100x more expensive than DRAM, and thus not competitive.
The most cost effective Cuckoo Cycle mining hardware should consist of a relatively
cheap and tiny many core memory controller paired with commodity DRAM chips,
where the latter dominate both the hardware and energy cost (about 1 Watt per DRAM chip).

Cycle finding
--------------
The algorithm implemented in lean_miner.h runs in time linear in N.
(Note that running in sub-linear time is out of the question, as you could
only compute a fraction of all edges, and the odds of all 42 edges of a cycle
occurring in this fraction are astronomically small).

Memory-wise, it uses N/2 bits to maintain a subset of all edges (potential
cycle edges) and N additional bits (or 40N bits in the latency avoiding algorithm)
to trim the subset in a series of edge trimming rounds.
This is the phase that takes the vast majority of runtime.

Once the subset is small enough, an algorithm inspired by Cuckoo Hashing
is used to recognise all cycles, and recover those of the right length.

Performance
--------------

The runtime of a single proof attempt for a 2^30 node graph on a 4GHz i7-4790K is 10.5 seconds
with the single-threaded matrix solver, using 3200MB (or 2200MB with slower cycle recovery).
This reduces to 3.5 seconds with 4 threads (3x speedup).

Using an order of magnitude less memory (just under 200MB),
the cuckoo solver takes 32.8 seconds per proof attempt.
Its multi-threading performance is less impressive though,
with 2 threads still taking 25.6 seconds and 4 taking 20.5 seconds.

I claim that these implementations are reasonably optimal,
secondly, that trading off (less) memory for (more) running time,
incurs at least one order of magnitude extra slowdown,
and finally, that cuda_miner.cu is a reasonably optimal memory-efficient GPU miner.
The latter runs about 4x faster on an NVIDA GTX 980 than lean_miner on an Intel Core-i7 CPU.
To that end, I offer the following bounties:

CPU Speedup Bounties
--------------
$8000 for an open source implementation that finds 42-cycles twice as fast
as lean_miner, using no more than 1 byte per edge.

$8000 for an open source implementation that finds 42-cycles twice as fast
as mean_miner, regardless of memory use.

Linear Time-Memory Trade-Off Bounty
--------------
$8000 for an open source implementation that uses at most N/k bits while finding 42-cycles up to 10 k times slower, for any k>=2.

All of these bounties require N ranging over {2^28,2^30,2^32} and #threads
ranging over {1,2,4,8}, and further assume a high-end Intel Core i7 or Xeon and
recent gcc compiler with regular flags as in my Makefile.

GPU Speedup Bounty
--------------
$2000 for an open source implementation for a consumer GPU combo
that finds 42-cycles twice as fast as cuda_miner.cu on comparable hardware,
using no more than 1 byte per edge.
Again with N ranging over {2^28,2^30,2^32}.

The Makefile defines corresponding targets leancpubounty, meancpubounty, tmtobounty, and gpubounty.

Double and Half bounties
------------------------
Improvements by a factor of 4 will be rewarded with double the regular bounty.

In order to minimize the risk of missing out on less drastic improvements,
I further offer half the regular bounty for improvements by a factor of sqrt(2).

Anyone who'd like to see my claims tested is invited to donate to the Cuckoo Cycle Bounty Fund at

<a href="https://blockchain.info/address/1CnrpdKtfF3oAZmshyVC1EsRUa25nDuBvN">1CnrpdKtfF3oAZmshyVC1EsRUa25nDuBvN</a>

which has received generous donations by

<ul>
<li> Readers of the <a href="https://forum.z.cash/">Zcash Forum</a> </li>
<li> A $20k donation by <a href="https://www.genesis-mining.com/">Genesis Mining</a> </li>
<li> A 10BTC donation by <a href="https://www.simply-vc-co.ltd/?page_id=8">Simply VC</a> </li>
</ul>

I intend for the total bounty value to stay ahead of funding levels. Happy bounty hunting!

Projects using, or planning to use, Cuckoo Cycle
--------------
<UL>
<LI> <a href="https://github.com/ignopeverell/grin">Minimal implementation of the MimbleWimble protocol</a>
<LI> <a href="http://www.aeternity.com/">æternity - the oracle machine</a>
<LI> <a href="https://github.com/bitcoin/bips/blob/master/bip-0154.mediawiki">BIP 154: Rate Limiting via peer specified challenges; Bitcoin Peer Services</a>
<LI> <a href="http://www.raddi.net/">Raddi // radically decentralized discussion</a>
</UL>
