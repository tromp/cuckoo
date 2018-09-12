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

all : libblake2b.so test example lean27stx8 mean27x8 lean29stx8 mean29x8 lean31stx8 mean31x8 verify27 verify29 verify31 demo29x8

libblake2b.so:	blake2.h blake2-impl.h blake2b-ref.c
	$(GCC) -fPIC -shared -o libblake2b.so blake2b-ref.c
	echo "add build dir to LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on Mac OSX) if library not found"
	echo "or, if you have sudo rights, a better solution is to run"
	echo "sudo echo <build dir> > /etc/ld.so.conf.d/cuckatoo.conf"
	echo "sudo ldconfig"

test:	lean19 verify19 Makefile
	./lean19 -n 38 | grep ^Sol | ./verify19 -n 38

simple3:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp graph.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DSHOW -DIDXSHIFT=0 -DPROOFSIZE=6 -DEDGEBITS=3 simple_miner.cpp $(LIBS)

simple15:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp graph.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=15 simple_miner.cpp $(LIBS)

simple29:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp graph.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple_miner.cpp $(LIBS)

lean5:	cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=6 -DLOGNBUCKETS=0 -DSHOWSOL -DEDGEBITS=5 lean_miner.cpp $(LIBS)

example:	simple3
	./simple3 -h 193

lean9:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=9 -DPROOFSIZE=8 -DSHOWSOL lean_miner.cpp $(LIBS)

lean14:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=14 lean_miner.cpp $(LIBS)

lean15:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=15 lean_miner.cpp $(LIBS)

lean19:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 lean_miner.cpp $(LIBS)

verify15:	siphash.h cuckatoo.h cuckatoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=15 cuckatoo.c $(LIBS)

verify19:	siphash.h cuckatoo.h cuckatoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=19 cuckatoo.c $(LIBS)

lean24:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=24 lean_miner.cpp $(LIBS)

mean15x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=0 -DNSIPHASH=8 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean15x1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DXBITS=0 -DNSIPHASH=1 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean15sx8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -mavx2 -DXBITS=0 -DNSIPHASH=8 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean15sx1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -DXBITS=0 -DNSIPHASH=1 -DEDGEBITS=15 mean_miner.cpp $(LIBS)

mean19x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean_miner.cpp $(LIBS)

mean24x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=5 -DNSIPHASH=8 -DEDGEBITS=24 mean_miner.cpp $(LIBS)

mean27x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean27x1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=1 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean27sx8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DXBITS=6 -DCOMPRESSROUND=10 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean27sx1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES  -DXBITS=6 -DNSIPHASH=1 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean29x4:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29x1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29sx8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29sx1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DSAVEEDGES -DNSIPHASH=1 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

demo29x8:	mean29sx8
	./mean29sx8 -n 63

demo29x1:	mean29sx1
	./mean29sx1 -n 63

mean31x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DEXPANDROUND=8 -DCOMPRESSROUND=22 -DXBITS=8 -DNSIPHASH=8 -DEDGEBITS=31 mean_miner.cpp $(LIBS)

mean31sx8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DEXPANDROUND=8 -DCOMPRESSROUND=22 -DXBITS=8 -DNSIPHASH=8 -DEDGEBITS=31 mean_miner.cpp $(LIBS)

lean27:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27x8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27.1:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27st:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27stx4:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=4 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27stx8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27stx16:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=16 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27sc:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DSINGLECYCLING -DEDGEBITS=27 lean_miner.cpp $(LIBS)

lean27.2:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=2 -DEDGEBITS=27 lean_miner.cpp $(LIBS)

speedup27:	mean27 Makefile
	for t in 1 2 4 6 8 12 16; do echo $$i; time ./mean27 -t $$t -r 10 2>&1; done > speedup27

lean29:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean29.1:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean29x2:	siphash.h siphashxN.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DATOMIC -DNSIPHASH=2 -DEDGEBITS=29 lean_miner.cpp $(LIBS)
	
lean29x4:	siphash.h siphashxN.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DATOMIC -DNSIPHASH=4 -DEDGEBITS=29 lean_miner.cpp $(LIBS)
	
lean29x8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean_miner.cpp $(LIBS)

lean29stx8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 lean_miner.cpp $(LIBS)

speedup29:	cuckoo30 Makefile
	for t in 1 2 4 6 8; do echo $$i; do ./cuckoo29 -t $$t -r 10 2>&1; done > speedup29

tomato19:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 tomato_miner.cpp $(LIBS)

tomato19:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=24 tomato_miner.cpp $(LIBS)

tomato27:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=27 tomato_miner.cpp $(LIBS)

tomato29:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=29 tomato_miner.cpp $(LIBS)

tomato31:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=31 tomato_miner.cpp $(LIBS)

lcuda27:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -arch sm_35 lean_miner.cu $(LIBS)

lcuda29:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 lean_miner.cu $(LIBS)

lcuda31:	siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=31 -arch sm_35 lean_miner.cu $(LIBS)

cuda27:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -arch sm_35 mean_miner.cu $(LIBS)

cuda29:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 mean_miner.cu $(LIBS)

cuda31:	siphash.cuh mean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=31 -arch sm_35 mean_miner.cu $(LIBS)

cycle27:	cycle_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=27 -arch sm_35 cycle_miner.cu $(LIBS)

cycle29:	cycle_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 cycle_miner.cu $(LIBS)

lean31:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean31.1:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DPART_BITS=1 -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean31x8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=31 lean_miner.cpp $(LIBS)

lean31stx8:	siphash.h cuckatoo.h  bitmap.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=31 lean_miner.cpp $(LIBS)

leancpubounty:	lean27stx8 lean29stx8 lean31stx8 lean27x8 lean29x8 lean31x8
	for s in 27 29 31; do for t in 8 4 2; do time ./lean$${s}x8 -t $$t -r 10; done; time ./lean$${s}stx8 -r 10; done

meancpubounty:	mean28sx7 mean29sx8 mean31sx8
	for s in 28 30 32; do for t in 8 4 2 1; do time ./mean$${s}sx8 -t $$t -r 10; done; done

tmtobounty:	tomato27 tomato29 tomato31
	for s in 28 30 32; do for t in 8 4 2 1; do time ./tomato$$s -t $$t -r 10; done; done

gpubounty:	cuda29
	time ./cuda29 -r 100

hashonly: hash_only.cpp Makefile
	$(GPP) -o $@ hash_only.cpp $(LIBS)

hashonlyx4: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=4 hash_only.cpp $(LIBS)

hashonlyx8: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 hash_only.cpp $(LIBS)

hashonlyx16: hash_only.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=16 hash_only.cpp $(LIBS)

verify27:	siphash.h cuckatoo.h cuckatoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=27 cuckatoo.c $(LIBS)

verify29:	siphash.h cuckatoo.h cuckatoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=29 cuckatoo.c $(LIBS)

verify31:	siphash.h cuckatoo.h cuckatoo.c Makefile
	$(GCC) -o $@ -DEDGEBITS=31 cuckatoo.c $(LIBS)

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

SimpleMiner.class:	Cuckoo.java SimpleMiner.java Makefile
	javac -O Cuckoo.java SimpleMiner.java

java:	Cuckoo.class SimpleMiner.class Makefile
	java SimpleMiner -h 261 | tail -1 | java Cuckoo -h 261
