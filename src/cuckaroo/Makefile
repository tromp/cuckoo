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
BLAKE_2B_SRC ?= ../crypto/blake2b-ref.c

all : simpletest leantest

simpletest:     simple19
	./simple19 -n 68

leantest:       lean19
	./lean19 -n 68

simple19:	../crypto/siphash.h cuckaroo.hpp  bitmap.hpp graph.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=19 simple.cpp $(BLAKE_2B_SRC)

simple29:	../crypto/siphash.h cuckaroo.hpp  bitmap.hpp graph.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple.cpp $(BLAKE_2B_SRC)

lean19:		../crypto/siphash.h cuckaroo.hpp  bitmap.hpp compress.hpp graph.hpp lean.hpp lean.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 lean.cpp $(BLAKE_2B_SRC)

lean29x8:	../threads/barrier.hpp ../crypto/siphash.h cuckaroo.hpp  bitmap.hpp compress.hpp graph.hpp lean.hpp lean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean.cpp $(BLAKE_2B_SRC)

mean19x8:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean.cpp $(BLAKE_2B_SRC)

mean29x4:	cuckaroo.hpp  bitmap.hpp graph.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean.cpp $(BLAKE_2B_SRC)

mean29x8:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean.cpp $(BLAKE_2B_SRC)

mean30x8:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEXPANDROUND=10 -DCOMPRESSROUND=22 -DEDGEBITS=30 mean.cpp $(BLAKE_2B_SRC)

mean29x8s:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DSAVEEDGES -DNSIPHASH=8 -DEDGEBITS=29 mean.cpp $(BLAKE_2B_SRC)

mean29x1:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean.cpp $(BLAKE_2B_SRC)

mean30x1:	cuckaroo.hpp  bitmap.hpp graph.hpp ../threads/barrier.hpp ../crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEXPANDROUND=10 -DCOMPRESSROUND=22 -DEDGEBITS=30 mean.cpp $(BLAKE_2B_SRC)

lcuda19:	../crypto/siphash.cuh lean.cu Makefile
	nvcc -o $@ -DEDGEBITS=19 -arch sm_35 lean.cu $(BLAKE_2B_SRC)

lcuda29:	../crypto/siphash.cuh lean.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 lean.cu $(BLAKE_2B_SRC)

lcuda30:	../crypto/siphash.cuh lean.cu Makefile
	nvcc -o $@ -DEDGEBITS=30 -arch sm_35 lean.cu $(BLAKE_2B_SRC)

lcuda32:	../crypto/siphash.cuh lean.cu Makefile
	nvcc -o $@ -DEDGEBITS=32 -arch sm_35 lean.cu $(BLAKE_2B_SRC)

cuda19:		../crypto/siphash.cuh mean.cu Makefile
	nvcc -o $@ -DEDGEBITS=19 -arch sm_35 mean.cu $(BLAKE_2B_SRC)

cuda29:		../crypto/siphash.cuh mean.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 mean.cu $(BLAKE_2B_SRC)
