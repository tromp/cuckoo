.POSIX:
.SUFFIXES:

OPT ?= -O3

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. $(CPPFLAGS) -pthread
GPP ?= g++ $(GPP_ARCH_FLAGS) -std=c++11 $(FLAGS)
CFLAGS ?= -Wall -Wno-format -fomit-frame-pointer $(OPT)
GCC ?= gcc $(GCC_ARCH_FLAGS) -std=gnu11 $(CFLAGS)
LIBS ?= -L. -lblake2b

all : libblake2b.so test example lean28stx8 mean28x8 lean30stx8 mean30x8 lean32stx8 mean32x8 verify28 verify30 verify32 demo30x8

libblake2b.so:	blake2.h blake2-impl.h blake2b-ref.c
	$(GCC) -fPIC -shared -o libblake2b.so blake2b-ref.c
	echo "add build dir to LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on Mac OSX) if library not found"
	echo "or, if you have sudo rights, a better solution is to run"
	echo "sudo echo <build dir> > /etc/ld.so.conf.d/cuckoo.conf"
	echo "sudo ldconfig"

test:	lean20 verify20 Makefile
	./lean20 -n 38 | grep ^Sol | ./verify20 -n 38

simple4:	siphash.h cuckoo.h lean_miner.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DSHOW -DIDXSHIFT=0 -DPROOFSIZE=6 -DEDGEBITS=3 simple_miner.cpp $(LIBS)

simple30:	siphash.h cuckoo.h lean_miner.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple_miner.cpp $(LIBS)

lean6:	cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=6 -DLOGNBUCKETS=0 -DSHOWSOL -DEDGEBITS=5 lean_miner.cpp $(LIBS)

example:	simple4
	./simple4 -h 211

lean10:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=9 -DPROOFSIZE=8 -DSHOWSOL lean_miner.cpp $(LIBS)

lean15:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=14 lean_miner.cpp $(LIBS)

lean16:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=15 lean_miner.cpp $(LIBS)

lean20:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 lean_miner.cpp $(LIBS)

verify16:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=15 cuckoo.c $(LIBS)

verify20:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=19 cuckoo.c $(LIBS)

lean25:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=24 lean_miner.cpp $(LIBS)

mean16x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=0 -DNSIPHASH=8 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean16x1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DXBITS=0 -DNSIPHASH=1 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean16sx8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -mavx2 -DXBITS=0 -DNSIPHASH=8 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean16sx1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -DXBITS=0 -DNSIPHASH=1 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean20x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean_miner.cpp $(LIBS)

mean25x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=5 -DNSIPHASH=8 -DEDGEBITS=24 mean_miner.cpp $(LIBS)

mean28x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean28x1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=1 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean28sx8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean28sx1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES  -DXBITS=6 -DNSIPHASH=1 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean30x4:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean30x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean30x1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean30sx8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

# compatible with original siphash eg for reproducing xenoncat results
mean30sx8c:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSIPHASH_COMPAT -DSAVEEDGES -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean30sx1:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -DNSIPHASH=1 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

demo30x8:	mean30sx8
	./mean30sx8 -n 63

demo30x1:	mean30sx1
	./mean30sx1 -n 63

mean32x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DEXPANDROUND=8 -DCOMPRESSROUND=22 -DXBITS=8 -DNSIPHASH=8 -DEDGEBITS=31 mean_miner.cpp $(LIBS)

mean32sx8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DEXPANDROUND=8 -DCOMPRESSROUND=22 -DXBITS=8 -DNSIPHASH=8 -DEDGEBITS=31 mean_miner.cpp $(LIBS)

lean28:	cuckoo.h siphash.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28x8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28.1:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28st:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28stx4:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=4 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28stx8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28stx16:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=16 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28sc:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DSINGLECYCLING -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean28.2:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=2 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

speedup28:	mean28 Makefile
	for t in 1 2 4 6 8 12 16; do echo $$i; time ./mean28 -t $$t -r 10 2>&1; done > speedup28

cudaspeedup28:	cuda28 Makefile
	for t in 65536 32768 16384 8192 4096 2048 1024; do echo $$t; time ./cuda28 -t $$t -r 10; done;

lean30:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean30.1:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean30x2:	siphash.h siphashxN.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DATOMIC -DNSIPHASH=2 -DEDGEBITS=29 lean_miner.cpp $(LIBS)
	
lean30x4:	siphash.h siphashxN.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DATOMIC -DNSIPHASH=4 -DEDGEBITS=29 lean_miner.cpp $(LIBS)
	
lean30x8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean30stx8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 lean_miner.cpp $(LIBS)

speedup30:	cuckoo30 Makefile
	for t in 1 2 4 6 8; do echo $$i; do ./cuckoo30 -t $$t -r 10 2>&1; done > speedup30

tomato20:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 tomato_miner.cpp $(LIBS)

tomato25:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=24 tomato_miner.cpp $(LIBS)

tomato28:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=27 tomato_miner.cpp $(LIBS)

tomato30:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=29 tomato_miner.cpp $(LIBS)

tomato32:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=31 tomato_miner.cpp $(LIBS)

lcuda28:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -arch sm_35 lean_miner.cu $(LIBS)

lcuda30:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 lean_miner.cu $(LIBS)

lcuda32:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=31 -arch sm_35 lean_miner.cu $(LIBS)

cuda28:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -DCOMPRESSROUND=4 -DBIGEPS=1/16 -arch sm_35 mean_miner.cu $(LIBS)

cuda30:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 mean_miner.cu $(LIBS)

cuda32:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=31 -arch sm_35 mean_miner.cu $(LIBS)

cycle28:	cycle_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -arch sm_35 cycle_miner.cu $(LIBS)

cycle30:	cycle_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 cycle_miner.cu $(LIBS)

lean32:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean32.1:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean32x8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean32stx8:	siphash.h cuckoo.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=31 lean_miner.cpp $(LIBS)

leancpubounty:	lean28stx8 lean30stx8 lean32stx8 lean28x8 lean30x8 lean32x8
	for s in 28 30 32; do for t in 8 4 2; do time ./lean$${s}x8 -t $$t -r 10; done; time ./lean$${s}stx8 -r 10; done

meancpubounty:	mean28sx8 mean30sx8 mean32sx8
	for s in 28 30 32; do for t in 8 4 2 1; do time ./mean$${s}sx8 -t $$t -r 10; done; done

tmtobounty:	tomato28 tomato30 tomato32
	for s in 28 30 32; do for t in 8 4 2 1; do time ./tomato$$s -t $$t -r 10; done; done

gpubounty:	cuda30
	time ./cuda30 -r 100

hashonly: hash_only.cpp Makefile
	$(GPP) -o $@ hash_only.cpp $(LIBS)

hashonlyx4: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=4 hash_only.cpp $(LIBS)

hashonlyx8: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 hash_only.cpp $(LIBS)

hashonlyx16: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=16 hash_only.cpp $(LIBS)

verify28:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=27 cuckoo.c $(LIBS)

verify30:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=29 cuckoo.c $(LIBS)

verify32:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=31 cuckoo.c $(LIBS)

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

SimpleMiner.class:	Cuckoo.java SimpleMiner.java Makefile
	javac -O Cuckoo.java SimpleMiner.java

java:	Cuckoo.class SimpleMiner.class Makefile
	java SimpleMiner -h 261 | tail -1 | java Cuckoo -h 261
