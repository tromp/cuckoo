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
LIBS ?= -L../crypto -lblake2b

all : simpletest leantest

simpletest:     simple19
	./simple19 -n 38

leantest:       lean19
	./lean19 -n 38

simple19:	../crypto/siphash.h cuckoo.h cyclebase.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=19 simple.cpp $(LIBS)

simple29:	../crypto/siphash.h cuckoo.h cyclebase.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple.cpp $(LIBS)

lean19:		../crypto/siphash.h cuckoo.h  lean.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DEDGEBITS=19 lean.cpp $(LIBS)

lean29x8:	../crypto/siphash.h cuckoo.h  lean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean.cpp $(LIBS)

mean19x8:	cuckoo.h  crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean.cpp $(LIBS)

mean29x4:	cuckoo.h  crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean.cpp $(LIBS)

mean29x8:	cuckoo.h  crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean.cpp $(LIBS)

mean29x1:	cuckoo.h  crypto/siphash.h mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean.cpp $(LIBS)

lcuda29:	../crypto/siphash.cuh lean.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 lean.cu $(LIBS)

cuda29:		../crypto/siphash.cuh mean.cu Makefile
	nvcc -o $@ -DEDGEBITS=29 -arch sm_35 mean.cu $(LIBS)
