Tuning the GPU solver
============

Recall the solver options:

    SYNOPSIS
      cuda30 [-b blocks] [-d device] [-h hexheader] [-k rounds [-c count]] [-m trims] [-n nonce] [-r range] [-U blocks] [-u threads] [-V threads] [-v threads] [-T threads] [-t threads] [-X threads] [-x threads] [-Y threads] [-y threads] [-Z threads] [-z threads]
    DEFAULTS
      cuda30 -b 128 -d 0 -h "" -k 0 -c 1 -m 256 -n 0 -r 1 -U 128 -u 8 -V 32 -v 128 -T 32 -t 128 -X 32 -x 64 -Y 32 -y 128 -Z 32 -z 8

Let's look at each of these in turn.

-b blocks
------------
The number of threadblocks to use, which should be a 2-power between 1 and 128. Default is 64.
Each block requires its own 21MB memory buffer. Here's how memory (in MB) and graph times in (ms)
vary with different blocks on an NVIDIA 1080 Ti

_      |     1 |     2 |    4 |    8 |   16 |   32 |   64 |  128
-----  | ----- | ----- | ---- | ---- | ---- | ---- | ---- | ----
Memory |  2701 |  2722 | 2764 | 2849 | 3019 | 3358 | 4036 | 5392
Time   | 30510 | 15545 | 7912 | 4120 | 2230 | 1359 | 1012 |  968

So we can achieve subsecond times by using over 5GB. On the other hand,
a GPU with only 3GB is restricted to 16 thread blocks and is not even half as fast.

-d device
------------
This option lets us select among multiple CUDA devices. Default is 0.
Besides a 1080 Ti as device 0, I also have a plain 1080 available:

    $ ./cuda30 -d 1 | head -1
    GeForce GTX 1080 with 8114MB @ 256 bits x 5005MHz

-h hexheader
------------
This allows specification of arbitrary headers. Default "".
The given string is padded with nul bytes to a fixed header length of 80 bytes.
For example,

    $ ./cuda30 -h "DEADBEEF" | head -2
    GeForce GTX 1080 Ti with 10GB @ 352 bits x 5505MHz
    Looking for 42-cycle on cuckoo30("ޭ??",0) with 50% edges, 128*128 buckets, 256 trims, and 128 thread blocks.

    $ ./cuda30 -h "444541440A42454546" | head -3
    GeForce GTX 1080 Ti with 10GB @ 352 bits x 5505MHz
    Looking for 42-cycle on cuckoo30("DEAD
    BEEF",0) with 50% edges, 128*128 buckets, 256 trims, and 128 thread blocks.

-k rounds [-c counts]
------------
The number of rounds to report on, and the extent of edge counting in the reports. Default are 0 and 1.
For example:

    $ ./cuda30 -k 16 -r 2 | tail -27
    nonce 1 k0 k1 k2 k3 be6c0ae25622e409 ede28d78411671d4 74ffaa51c7aa70ac 2ab552193088c87a
    genUnodes size 32822 completed in 254 ms
    genVnodes size 20674 completed in 291 ms
    round 2 size 9713 completed in 145 ms
    round 3 size 4858 completed in 67 ms
    round 4 size 3896 completed in 41 ms
    round 5 size 2785 completed in 32 ms
    round 6 size 2037 completed in 23 ms
    round 7 size 1579 completed in 17 ms
    round 8 size 1254 completed in 14 ms
    round 9 size 1045 completed in 11 ms
    round 10 size 867 completed in 28 ms
    round 11 size 755 completed in 23 ms
    round 12 size 641 completed in 5 ms
    round 13 size 542 completed in 2 ms
    round 14 size 470 completed in 3 ms
    round 15 size 403 completed in 2 ms
    rounds 12 through 253 completed in 58 ms
    trimrename3 round 254 size 2 completed in 1 ms
    trimrename3 round 255 size 2 completed in 1 ms
       4-cycle found
     282-cycle found
    1006-cycle found
     390-cycle found
    findcycles completed on 29180 edges
    Time: 1010 ms
    0 total solutions

Each round report shows the number of remaining edges within the first count\*count buckets,
and the time taken by that round. Since Each bucket count requires a separate device to host transfer,
count all 128\*128 buckets is rather slow, as witnessed by the 3x longer Time below:

    $ ./cuda30 -k 16 -c 128 -r 2 | tail -27
    nonce 1 k0 k1 k2 k3 be6c0ae25622e409 ede28d78411671d4 74ffaa51c7aa70ac 2ab552193088c87a
    genUnodes size 536870912 completed in 254 ms
    genVnodes size 339368314 completed in 291 ms
    round 2 size 159013933 completed in 145 ms
    round 3 size 78421673 completed in 67 ms
    round 4 size 62684373 completed in 41 ms
    round 5 size 44919453 completed in 32 ms
    round 6 size 33847415 completed in 23 ms
    round 7 size 26456073 completed in 17 ms
    round 8 size 21268142 completed in 14 ms
    round 9 size 17481575 completed in 11 ms
    round 10 size 14630106 completed in 28 ms
    round 11 size 12429089 completed in 23 ms
    round 12 size 10694596 completed in 5 ms
    round 13 size 9301758 completed in 2 ms
    round 14 size 8167240 completed in 3 ms
    round 15 size 7230054 completed in 2 ms
    rounds 12 through 253 completed in 709 ms
    trimrename3 round 254 size 33514 completed in 1 ms
    trimrename3 round 255 size 33514 completed in 1 ms
       4-cycle found
     282-cycle found
    1006-cycle found
     390-cycle found
    findcycles completed on 29180 edges
    Time: 3356 ms
    0 total solutions

We see that genUnodes generates the "U" endpoints of all 2^29 = 536870912 edges in round 0.
In round 1, genVnodes finds that 197502598 edges do not share their U endpoint with another edge.
These dead-ends cannot be part of a cycle and can be safely dismissed. The remaining
The other 536870912-197502598 = 339368314 edges have their other "V" endpoint generated.
Successive trimming rounds, which alternate between checking shared V endpoints and shared U endpoints,
take less and less time as fewer edges remain.
The reason rounds 10 and 11 take longer is that in addition to trimming edges, they also rename the
endpoints to fit in fewer bits, which allows later rounds to run in a single stage.
Regardless of the value passed with -k, reporting always end with a summary of rounds 12..ntrims-3,
and the final two rounds which end up compactifying the storage of surviving edges for fast transfer
back to the host.

-m trims
------------
The number of trimming rounds. Default 256. Can be increased arbitrarily. At some point, there will be
no edges left to trim, as all remaining edges are already part of a cycle:

    $ ./cuda30 -m 1518 -k 2000 -c 128
    GeForce GTX 1080 Ti with 10GB @ 352 bits x 5505MHz
    Looking for 42-cycle on cuckoo30("",0) with 50% edges, 128*128 buckets, 2000 trims, and 128 thread blocks.
    Using 2680MB bucket memory and 21MB memory per thread block (5392MB total)
    nonce 0 k0 k1 k2 k3 a34c6a2bdaa03a14 d736650ae53eee9e 9a22f05e3bffed5e b8d55478fa3a606d
    genUnodes size 536870912 completed in 1428 ms
    genVnodes size 339368161 completed in 321 ms
    round 2 size 159003113 completed in 132 ms
    round 3 size 78416432 completed in 61 ms
    round 4 size 62685884 completed in 37 ms
    .
    .
    .
    round 1512 size 92 completed in 0 ms
    round 1513 size 90 completed in 0 ms
    round 1514 size 88 completed in 0 ms
    round 1515 size 88 completed in 0 ms
    rounds 12 through 1515 completed in 281145 ms
    trimrename3 round 1516 size 88 completed in 0 ms
    trimrename3 round 1517 size 88 completed in 0 ms
       4-cycle found
       2-cycle found
      16-cycle found
      26-cycle found
      40-cycle found
    findcycles completed on 88 edges
    Time: 284971 ms
    0 total solutions

Note that 88 is exactly the sum length of all 5 cycles.
Bad things happen we don't trim enough:

    $ ./cuda30 -m 200
    GeForce GTX 1080 Ti with 10GB @ 352 bits x 5505MHz
    Looking for 42-cycle on cuckoo30("",0) with 50% edges, 128*128 buckets, 200 trims, and 128 thread blocks.
    Using 2680MB bucket memory and 21MB memory per thread block (5392MB total)
    nonce 0 k0 k1 k2 k3 a34c6a2bdaa03a14 d736650ae53eee9e 9a22f05e3bffed5e b8d55478fa3a606d
    mean_miner.cu:125: void zbucket<BUCKETSIZE, NRENAME, NRENAME1>::setsize(const unsigned char *) [with BUCKETSIZE = 2048U, NRENAME = 0U, NRENAME1 = 0U]: block: [0,0,0], thread: [0,0,0] Assertion `size <= BUCKETSIZE` failed.
    GPUassert: device-side assert triggered mean_miner.cu 846
    
-n nonce
------------
The starting nonce. Default 0. This 32-bit number is stored in the final 4 bytes of the header,
just before hashing it with Blake2b to derive the siphash keys which determine the 2^30 node graph.

-r range
------------
The number of consecutive nonces to solve. Default 1. An actual miner would just keep incrementing
the nonce forever, until a solution that meets the difficulty threshold is found by any miner,
at which point a new header needs to be build.

-U blocks -u threads
------------
The number of threadblocks and threads per block to use in round 0. Default 128 and 8.
Since round 0, generating U nodes, is only writing into blocks, it can use
arbitrarily many threadblocks. Here's a table of genUnodes times with various
combinations of blocks and threads per block (tpb):

tpb\blocks |     4 |    8 |   16 |   32 |  64 | 128 | 256 | 512
---------- | ----- | ---- | ---- | ---- | --- | --- | --- | ---
 4  | 11274 | 5638 | 2819 | 1408 | 745 | 395 | 395 | 396
 8  |  5936 | 2949 | 1465 |  736 | 419 | 256 | 255 | 256
16  |  3416 | 1719 |  855 |  496 | 269 | 303 | 304 | 301
32  |  2088 | 1044 |  590 |  370 | 343 | 451 | 454 | 453
64  |  1347 |  714 |  471 |  353 | 446 | 450 | 450 | 441

-V threads
------------
The number of threads per block to use in stage 1 of round 1. Default 32.
Preferrably a 2-power. Here's the effect on genVnodes times:

tpb |    1 |   2 |   4 |   8 |  16 |  32 |  64 | 128 | 256 |   512
--- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | -----
 ms | 1124 | 707 | 517 | 414 | 326 | 285 | 291 | 304 | 336 | CRASH


The crashes have to do with only storing either 18 or 25 bits of the 29-bit edge index,
and having to recover the remaining bits from assumed ordering properties, which
are increasingly violated as we increase parallellism.

-v threads
------------
The number of threads per block to use in stage 2 of round 1. Default 128.
Need not be a 2-power. Here's the effect on genVnodes times:

tpb |    4 |   8 |  16 |  32 |  64 | 128 | 256 | 512 | 1024
--- | ---- | --- | --- | --- | --- | --- | --- | --- | ----
 ms | 1135 | 711 | 558 | 347 | 295 | 286 | 291 | 313 |  290

-T threads
------------
The number of threads per block to use in stage 1 of rounds 2-9. Default 32.
Preferrably a 2-power. Here's the effect on round 2 times:

tpb |   1 |   2 |   4 |   8 |  16 |  32 |    64
--- | --- | --- | --- | --- | --- | --- | -----
 ms | 668 | 403 | 277 | 196 | 139 | 131 | CRASH

These crashes are due to storing only 3 of the 7 UX bits and again assuming sufficient ordering
for recovery. We can avoid them by lowering EXPANDROUND, e.g. recompiling with -DEXPANDROUND=4,
resulting in a round 2 time of 132ms.

-t threads
------------
The number of threads per block to use in stage 2 of rounds 2-9. Default 128.
Need not be a 2-power. Here's the effect on round 2 times:

tpb |   4 |   8 |  16 |  32 |  64 |  96 | 128 |   192
--- | --- | --- | --- | --- | --- | --- | --- | -----
 ms | 596 | 380 | 247 | 180 | 147 | 136 | 131 | CRASH

Same problem with ux recovery, beyond 128 threads this time.

-X threads
------------
The number of threads per block to use in stage 1 of renaming round 10. Default 32.
Preferrably a 2-power. Here's the effect on round 10 times:

tpb |  1 |   2 |   4 |   8 |  16 |  32 |  64 | 128 | 256 | 512 | 1024
--- | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----
 ms | 59 |  45 |  39 |  32 |  31 |  31 |  31 |  31 |  32 |  32 |   31

-x threads
------------
The number of threads per block to use in stage 2 of renaming round 10. Default 64.
Need not be a 2-power. Here's the effect on round 10 times:

tpb |  4 |  8 | 16 | 32 | 64 |    96
--- | -- | -- | -- | -- | -- | -----
 ms | 84 | 50 | 31 | 31 | 31 | CRASH

Too many threads here cause a sparser use and eventual shortage of names.

-Y threads
------------
The number of threads per block to use in stage 1 of renaming round 11. Default 32.
Preferrably a 2-power. Here's the effect on round 11 times:

tpb |  1 |   2 |   4 |   8 |  16 |  32 |  64 | 128 | 256 | 512 | 1024
--- | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----
 ms | 50 |  39 |  34 |  30 |  27 |  27 |  27 |  27 |  27 |  28 |   27

-y threads
------------
The number of threads per block to use in stage 2 of renaming round 11. Default 128.
Need not be a 2-power. Here's the effect on round 11 times:

tpb |  4 |  8 | 16 | 32 | 64 | 128 | 192 | 256 |   384
--- | -- | -- | -- | -- | -- | --- | --- | --- | -----
 ms | 75 | 44 | 27 | 27 | 27 |  27 |  27 |  26 | CRASH

Same crashing as with -x.

-Z threads
------------
The number of threads per block to use in rounds 12..ntrims-3. Default 32.
Preferrably a 2-power. Here's the effect on round 12-237 times:

tpb |   4 |   8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024
--- | --- | --- | -- | -- | -- | --- | --- | --- | ----
 ms | 186 | 112 | 68 | 52 | 54 | 101 | 102 | 106 |   71

-z threads
------------
The number of threads per block to use in the final two renaming rounds. Default 8.
Preferrably a 2-power. Here's the lack of effect on round 238 times:

tpb | 1 | 2 | 4 | 8 | 16 |    32
--- | - | - | - | - | -- | -----
 ms | 0 | 0 | 0 | 0 |  0 | CRASH

Same crashing as with -x again.

Happy tuning!
