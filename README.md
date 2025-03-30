Cuck(at)oo Cycle
================

[Blog article explaining Cuckoo Cycle](https://www.cryptogurureview.com/what-is-the-cuckoo-cycle-a-deep-dive-into-this-clever-mining-algorithm)

[Whitepaper](doc/cuckoo.pdf?raw=true)

My [Grincon0 talk](https://www.youtube.com/watch?v=OsBsz8bKeN4) on Jan 28, 2019 in San Mateo.

My [Grincon1](https://grincon.org/) [talk](https://www.youtube.com/watch?list=PLvgCPbagiHgqYdVUj-ylqhsXOifWrExiq&v=CLiKX0nOsHE) on Nov 22, 2019 in Berlin.

Cuckoo Cycle is the first graph-theoretic proof-of-work, and the most memory bound, yet with instant verification.
Unlike Hashcash, Cuckoo Cycle is immune from quantum speedup by Grover's search algorithm.
Cuckatoo Cycle is a variation of Cuckoo Cycle that aims to simplify ASICs by reducing ternary counters to plain bits.

Simplest known PoW
------------------
With a 42-line [complete specification in C](doc/spec) and a 13-line
[mathematical description](doc/mathspec), Cuckatoo Cycle is less than half the
size of either [SHA256](https://en.wikipedia.org/wiki/SHA-2#Pseudocode),
[Blake2b](https://en.wikipedia.org/wiki/BLAKE_%28hash_function%29#Blake2b_algorithm), or
[SHA3 (Keccak)](https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c)
as used in Bitcoin, Equihash and ethash. Simplicity matters.

Cuckatoo proofs take the form of a length 42 node-pair-cycle in a bipartite graph with N+N nodes and
N edges, with N=2^n and n ranging from 10 up to 64.

In a node-pair-cycle, consecutive edges are not incident on a single node, but on a node-pair,
that is, two nodes differing in the last bit only.
From now on, whenever we say cycle, we'll mean a node-pair-cycle.

The graph is defined by the siphash-2-4 keyed hash function mapping an edge index
and partition side (0 or 1) to the edge endpoint on that side.
This makes verification trivial: hash the header to derive the siphash key,
then compute the 42x2 edge endpoints with 84 siphashes,
and that you come back to the starting edge only after traversing 42 node-pairs.

While trivially verifiable, finding a 42-cycle, on the other hand, is far from trivial,
requiring considerable resources, and some luck
(the odds of a random cuckoo graph having an L-cycle are approximately 1 in L).

Lean mining
-----------
The memory efficient miner uses 1 bit per edge and 1 bit per node in one partition
(or 1 bit per 2^k nodes with linear slowdown).
It is bottlenecked by accessing node bits completely at random, making it memory latency bound.
The core of this miner, where over 99% of time is spent, is also [relatively simple](doc/simplesolve).

Mean mining
-----------
The roughly 4x faster latency avoiding miner uses 33 bits per edge
and is bottlenecked by bucket sorting, making it memory bandwidth bound.

Dynamic Sizing
--------------
Instead of fixing n, a proof-of-work system could allow miners to work on any graph size they like,
above a certain minimum. Cycles in larger graphs are more valued as they take more effort to find.
We propose to scale the difficulty for a graph with N=2^n edges by 2^(n+1) * n,
which is the number of bits of siphash output that define the graph.

Upgrade by Soft Fork
--------------------
Dynamic Sizing allows for increasing the memory required for mining simply by raising the minimum allowed graph size.
Since this makes new valid blocks a subset of old valid blocks, it is not a hard fork but a soft fork, easily deployed
with majority miner support. Miner manufacturers are incentivized to support larger sizes as being more future proof.

Automatic Upgrades
------------------
A raise in minimum graph size could be triggered automatically if over the last so many blocks (on the scale of months), less than a certain fraction (a minority like 1/3 or 1/4) were solved at that minimum size. This locks in a new minimum which then activates within so many blocks (on the scale of weeks).

ASIC Friendly
-------------
The more efficient lean mining requires tons of SRAM, which is lacking on CPUs and GPUs, but easily implemented in ASICs,
either on a single chip, or for even larger graph sizes (cuckatoo33 requires 1 GB of SRAM), on multiple chips.

Cycle finding
--------------
The most efficient known way to find cycles in random bipartite graphs is
to start by eliminating edges that are not part of any cycle (over 99.9% of edges).
This preprocessing phase is called edge trimming and actually takes the vast majority of runtime.

The algorithm implemented in lean_miner.hpp runs in time linear in N.
(Note that running in sub-linear time is out of the question, as you could
only compute a fraction of all edges, and the odds of all 42 edges of a cycle
occurring in this fraction are astronomically small).
Memory-wise, it uses an N bit edge bitmap of potential cycle edges,
and an N bit node bitmap of nodes with incident edges.

The bandwidth bound algorithm implemented in mean_miner.hpp
uses (1+&Epsilon;) &times; N words to maintain bucket sorted bins of edges instead of a bitmap,
and only uses tiny node bitmaps within a bin to trim edges.

After edge trimming, an standard backtracking graph traversal
is run to recognise all cycles, and report those of the right length.

Performance
--------------
The runtime of a single proof attempt for a 2^29 edge graph on a 4GHz i7-4790K is 10.5 seconds
with the single-threaded mean solver, using 2200MB (or 3200MB with faster solution recovery).
This reduces to 3.5 seconds with 4 threads (3x speedup).

Using an order of magnitude less memory (128MB),
the lean solver takes 32.8 seconds per proof attempt.
Its multi-threading performance is less impressive though,
with 2 threads still taking 25.6 seconds and 4 taking 20.5 seconds.

I claim that siphash-2-4 is a safe choice of underlying hash function,
that these implementations are reasonably optimal,
that trading off (less) memory for (more) running time,
incurs at least one order of magnitude extra slowdown,
and finally, that mean_miner.cu is a reasonably optimal GPU miner.
The latter runs about 10x faster on an NVIDA 1080Ti than mean_miner.cpp on an Intel Core-i7 CPU.
In support of these claims, I offer the following bounties:

Linear Time-Memory Trade-Off Bounty
-----------------------------------
$10000 for an open source implementation that uses at most N/k bits while finding 42-cycles up to 10 k times slower, for any k>=2.

All of these bounties require n ranging over {27,29,31} and #threads
ranging over {1,2,4,8}, and further assume a high-end Intel Core i7 or Xeon and
recent gcc compiler with regular flags as in my Makefile.

GPU Cuckatoo32 Speedup Bounty
------------------
$10000 for an open source Cuckatoo32 solver that achieves 1.2 gps on an NVIDIA 4070Ti or 1 gps on an AMD RX 6900 XT with minimal cycle-loss.

Apple M1 Cuckatoo32 Ultra Bounty
-----------------
1 BTC for an open source Mac Studio (M1 Ultra) Cuckatoo32 miner achieving 0.5 gps with minimal cycle-loss.

Nicolas Flamel has <a href="https://github.com/NicolasFlamel1/Mac-Studio-M1-Ultra-Cuckatoo-Trimmer">attempted</a> this bounty, but found the M1 Ultra GPU topping out at 0.26 gps even while offloading all but the inital trimming round to the CPU.

Siphash Bounties
----------------
While both siphash-2-4 and siphash-1-3 pass the [smhasher](https://github.com/aappleby/smhasher)
test suite for non-cryptographic hash functions,
siphash-1-2, with 1 compression round and only 2 finalization rounds,
[fails](doc/SipHash12) quite badly in the Avalanche department.
We invite attacks on Cuckoo Cycle's dependence on its underlying hash function by offering

$5000 for an open source implementation that finds 42-cycles in graphs defined by siphash-1-2
twice as fast as lean_miner on graphs defined by siphash-2-4, using no more than 1 byte per edge.

$5000 for an open source implementation that finds 42-cycles in graphs defined by siphash-1-2
twice as fast as mean_miner on graphs defined by siphash-2-4, regardless of memory use.

These bounties are not subject to double and/or fractional payouts.

Happy bounty hunting!
 
How to build
--------------
<pre>
cd src
make
</pre>

Bounty contributors
-------------------

* [Zcash Forum](https://forum.z.cash/) participants
* [Genesis Mining](https://www.genesis-mining.com/)
* [Simply VC](https://www.simply-vc-co.ltd/?page_id=8)
* [Claymore](https://bitcointalk.org/index.php?topic=1670733.0)
* [Aeternity developers](http://www.aeternity.com/)

Projects using Cuckoo Cycle
--------------
* [Grin](https://grin.mw/)
* [æternity - the oracle machine](http://www.aeternity.com/)
* [CodeChain](https://codechain.io/)
* [BitCash](https://www.choosebitcash.com/)
* [Veres One](https://veres.one)
* [BIP 154: Rate Limiting via peer specified challenges; Bitcoin Peer Services](https://github.com/bitcoin/bips/blob/master/bip-0154.mediawiki)
* [Raddi // radically decentralized discussion](http://www.raddi.net/)
* [Cortex // AI on Blockchain](https://www.cortexlabs.ai/)

Projects reconsidering Cuckoo Cycle
--------------
* [Handshake](https://handshake.org) found [unreconcilable issues](https://handshake.org/files/handshake.txt)

![](img/logo.png?raw=true)
