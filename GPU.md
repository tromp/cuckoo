Progress report on mean_miner.cu, the CUDA Cuckoo Cycle solver for billion-node graphs
============

Since completing my rewrite of xenoncat's performance quadrupling CPU solver (winning a double bounty),
in the form of mean_miner.hpp, I've been slowly grinding away at porting that code to CUDA.

I consider myself an amateur GPU coder, having previously ported lean_miner.hpp to CUDA
(and having to pay a bounty to fellow Dutchman Genoil for improving performance by merely
tweaking the threads per block which I had naively fixed at 1), as well as my own
<a href="https://github.com/tromp/equihash">Equihash miner</a> submission to the
<a href="https://z.cash/blog/open-source-miner-winners.html">Zcash Open Source Miner Challenge</a>.
That Equihash CUDA miner achieved a paltry 27.2 Sol/s on an NVIDIA GTX 980,
matching the performance of my Equihash CPU solver. But it did serve as the basis for far more capable
rewrites such as this
<a href="https://github.com/nicehash/nheqminer/blob/master/cuda_djezo/equi_miner.cu">Nicehash miner</a>
by Leet Softdev (a.k.a. djezo), which achieves around 400 Sol/s on similar hardware.

Today, on Jan 30, 2018, I finished writing, optimizing, and tuning my CUDA solver, mean_miner.cu
Unlike my other, more generic, solvers, this one is written to target only billion-node (2^30 to be precise)
graphs, which is how Cuckoo Cycle will be deployed in the upcoming cryptocurrencies Aeternity and Grin.
The reason I recommended deployment at this size is that it's the largest one that a majority of GPUs
can handle. Indeed, with the default settings, the solver uses 3.95 GB. Throwing more memory at this solver
appears to do little good; I saw only about a 1% performance improvement.
Changing settings to allow it to run within 3GB however imposes a huge penalty, more than halving performance.

All my final tuning was done on an NVIDIA 1080Ti. The only other card I ran it on was the GTX 980Ti,
which achieves less than 1/3 the performance.

So how fast is this fastest known Cuckoo Cycle solver on the fastest known hardware?

First of all, we need to agre on a performance metric. Cuckoo Cycle and Equihash are examples of
asymmetric proofs-of-work. As explained in this
<a href="http://cryptorials.io/beyond-hashcash-proof-work-theres-mining-hashing">article</a>,
instead of just computing hash functions, they look for solutions to puzzles. Now, in the case of
Equihash, solutions are plentiful, on average nearly 2 solutions per random puzzle. So there it makes
sense to measure performance in solutions per second. With Cuckoo Cycle, the 42-cycles solutions
we're looking for are rather rare, so instead it makes more sense to measure performance as puzzles
per second. Since each puzzle is a billion-node graph, we arrive at graphs per second as the appropriate
performance measure. So, to return to our question:

How many graphs per second does the fastest solver achieve?

Less than one.

cuda_miner.cu takes about 1.03 seconds to search one graph on the NVIDIA 1080Ti.


having made it as fast as I know

https://www.reddit.com/r/Aeternity/comments/6vsot4/towards_a_more_egalitarian_pow_using_cuckoo_cycle/
https://moneromonitor.com/episodes/2017-09-26-Episode-014.html

Blog article explaining Cuckoo Cycle at







<UL>
<LI> <a href="https://github.com/mimblewimble/grin">Minimal implementation of the MimbleWimble protocol</a>
<LI> <a href="http://www.aeternity.com/">æternity - the oracle machine</a>
<LI> <a href="https://github.com/bitcoin/bips/blob/master/bip-0154.mediawiki">BIP 154: Rate Limiting via peer specified challenges; Bitcoin Peer Services</a>
<LI> <a href="http://www.raddi.net/">Raddi // radically decentralized discussion</a>
<LI> <a href="https://bitcointalk.org/index.php?topic=2360396">[ANN] *Aborted* Bitcoin Resilience: Cuckoo Cycle PoW Bitcoin Hardfork</a>
</UL>
