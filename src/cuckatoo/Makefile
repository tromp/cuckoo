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
LIBS ?= -L.. -lblake2b

all : simple19 simple29 lean19 lean29 lcuda29 mean29 cuda29

simple19:	../siphash.h cuckatoo.h  ../bitmap.hpp graph.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=19 simple_miner.cpp $(LIBS)

simple29:	../siphash.h cuckatoo.h  ../bitmap.hpp graph.hpp simple_miner.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple_miner.cpp $(LIBS)

lean19:		../siphash.h cuckatoo.h  ../bitmap.hpp compress.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 lean_miner.cpp $(LIBS)

lean29x8:	../siphash.h cuckatoo.h  ../bitmap.hpp compress.hpp graph.hpp lean_miner.hpp lean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean_miner.cpp $(LIBS)

mean19x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean_miner.cpp $(LIBS)

mean29x4:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29x8:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean29x1:	cuckatoo.h  bitmap.hpp graph.hpp siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

lcuda29:	../siphash.cuh lean_miner.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 lean_miner.cu $(LIBS)
