# -Wno-deprecated-declarations shuts up Apple OSX clang
OPT = -O3
DOPT = -DPREFETCH

# for Mac OSX with brew install openssl
CPPFLAGS = -I/usr/local/opt/openssl/include
LDFLAGS = -L/usr/local/opt/openssl/lib

FLAGS = -Wall -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) $(DOPT) -I. $(CPPFLAGS) $(LDFLAGS) -pthread 
GPP = g++ -march=native -m64 -std=c++11 $(FLAGS)
CFLAGS = -fomit-frame-pointer -mno-vzeroupper $(OPT) -pthread 
GCC = gcc -m64 -std=gnu11 $(CFLAGS)
LIBS = -lssl -lcrypto
# leave out -l crypto if using sha256.c instead of openssl
# leave out -std=c++11 -DATOMIC for older GCC versions lacking c++11 support


all : test example cuckoo28stx8 cuckoo30stx8 cuckoo32stx8 cuckoo28 cuckoo30 cuckoo32 verify28 verify30 verify32 demo28

test:	cuckoo20 verify20 Makefile
	./cuckoo20 -n 13 | grep ^Sol | ./verify20 -n 13

cuckoo4:	siphash.h cuckoo.h cuckoo_miner.hpp simple_miner.cpp Makefile
	$(GPP) -o cuckoo4 -DSHOW -DIDXSHIFT=0 -DPROOFSIZE=6 -DEDGEBITS=3 simple_miner.cpp $(LIBS)

cuckoo6:	cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo6 -DIDXSHIFT=0 -DPROOFSIZE=6 -DLOGNBUCKETS=0 -DSHOWSOL -DEDGEBITS=5 cuckoo_miner.cpp $(LIBS)

example:	cuckoo4
	./cuckoo4 -e 66 -h 3

cuckoo10:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo10 -DATOMIC -DEDGEBITS=9 -DPROOFSIZE=8 -DSHOWSOL cuckoo_miner.cpp $(LIBS)

cuckoo15:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo15 -DATOMIC -DEDGEBITS=14 cuckoo_miner.cpp $(LIBS)

cuckoo20:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo20 -DATOMIC -DEDGEBITS=19 cuckoo_miner.cpp $(LIBS)

verify20:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify20 -DEDGEBITS=19 cuckoo.c $(LIBS)

cuckoo25:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo25 -DATOMIC -DEDGEBITS=24 cuckoo_miner.cpp $(LIBS)

mean28x8:	cuckoo.h siphash.h mean_miner.h mean_miner.c Makefile
	$(GPP) -o mean28x8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean28x8b:	cuckoo.h siphash.h mean_miner.h mean_miner.c Makefile
	$(GPP) -o mean28x8b -DBIG0SIZE=5 -mavx2 -DNSIPHASH=8 -DEDGEBITS=27 mean_miner.cpp $(LIBS)

mean30x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o mean30x8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean30x8s:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o mean30x8s -DBIG0SIZE=4 -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean_miner.cpp $(LIBS)

mean32x8:	cuckoo.h siphash.h mean_miner.hpp mean_miner.cpp Makefile
	$(GPP) -o mean32x8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=31 mean_miner.cpp $(LIBS)

cuckoo28:	cuckoo.h siphash.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28 -DATOMIC -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28x8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28x8 -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28.1:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28.1 -DATOMIC -DPART_BITS=1 -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28st:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28st -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28stx4:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28stx4 -mavx2 -DNSIPHASH=4 -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28stx8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28stx8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

demo28:	cuckoo28stx8
	time ./cuckoo28stx8 -n 95

cuckoo28stx16:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28stx16 -mavx2 -DNSIPHASH=16 -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28sc:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28sc -DATOMIC -DSINGLECYCLING -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

cuckoo28.2:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28.2 -DATOMIC -DPART_BITS=2 -DEDGEBITS=27 cuckoo_miner.cpp $(LIBS)

sols28:	cuckoo28stx8
	for n in 962976 962982 962990 963011 963142 963146 963292 963317 963357 963499 963537 963577 963727 963757 963795 963867 963870 963893; do time ./cuckoo28stx8 -n $$n; done

speedup:	cuckoo28 Makefile
	for t in 1 2 4 6 8 12 16; do echo $$i; time ./cuckoo28 -t $$t -r 10 2>&1; done > speedup

cudaspeedup28:	cuda28 Makefile
	for t in 65536 32768 16384 8192 4096 2048 1024; do echo $$t; time ./cuda28 -t $$t -r 10; done;

cuckoo30:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30 -DATOMIC -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

cuckoo30.1:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.1 -DATOMIC -DPART_BITS=1 -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

cuckoo30x8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30x8 -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

cuckoo30stx8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30stx8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

cuckoo30.2:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.2 -DATOMIC -DPART_BITS=2 -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

cuckoo30.3:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.3 -DATOMIC -DPART_BITS=3 -DEDGEBITS=29 cuckoo_miner.cpp $(LIBS)

speedup30:	cuckoo30 Makefile
	for t in 1 2 4 6 8; do echo $$i; do ./cuckoo30 -t $$t -r 10 2>&1; done > speedup30

tomato20:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato20 -DATOMIC -DEDGEBITS=19 tomato_miner.cpp $(LIBS)

tomato25:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato25 -DATOMIC -DEDGEBITS=24 tomato_miner.cpp $(LIBS)

tomato28:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato28 -DATOMIC -DEDGEBITS=27 tomato_miner.cpp $(LIBS)

tomato30:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato30 -DATOMIC -DEDGEBITS=29 tomato_miner.cpp $(LIBS)

tomato32:	siphash.h cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato32 -DATOMIC -DEDGEBITS=31 tomato_miner.cpp $(LIBS)

cuda28:	cuda_miner.cu Makefile
	nvcc -o cuda28 -DEDGEBITS=27 -arch sm_35 cuda_miner.cu -lssl -lcrypto 

cuda30:	cuda_miner.cu Makefile
	nvcc -o cuda30 -DEDGEBITS=29 -arch sm_35 cuda_miner.cu -lssl -lcrypto

cuda32:	cuda_miner.cu Makefile
	nvcc -o cuda32 -DEDGEBITS=31 -arch sm_35 cuda_miner.cu -lssl -lcrypto

cycle28:	cycle_miner.cu Makefile
	nvcc -o cycle28 -DEDGEBITS=27 -arch sm_35 cycle_miner.cu -lssl -lcrypto 

cycle30:	cycle_miner.cu Makefile
	nvcc -o cycle30 -DEDGEBITS=29 -arch sm_35 cycle_miner.cu -lssl -lcrypto 

cuckoo32:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32 -DATOMIC -DEDGEBITS=31 cuckoo_miner.cpp $(LIBS)

cuckoo32.1:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32.1 -DATOMIC -DPART_BITS=1 -DEDGEBITS=31 cuckoo_miner.cpp $(LIBS)

cuckoo32x8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32x8 -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=31 cuckoo_miner.cpp $(LIBS)

cuckoo32stx8:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32stx8 -mavx2 -DNSIPHASH=8 -DEDGEBITS=31 cuckoo_miner.cpp $(LIBS)

cuckoo32.2:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32.2 -DATOMIC -DPART_BITS=2 -DEDGEBITS=31 cuckoo_miner.cpp $(LIBS)

cuckoo35:	siphash.h cuckoo.h cuckoo_miner.hpp cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo35 -DATOMIC -DEDGEBITS=34 cuckoo_miner.cpp $(LIBS)

cpubounty:	cuckoo28stx8 cuckoo30stx8 cuckoo32stx8 cuckoo28x8 cuckoo30x8 cuckoo32x8
	for s in 28 30 32; do for t in 8 4 2; do time ./cuckoo$${s}x8 -t $$t -r 10; done; time ./cuckoo$${s}stx8 -r 10; done

tmtobounty:	tomato28 tomato30 tomato32
	for s in 28 30 32; do for t in 8 4 2 1; do time ./tomato$$s -t $$t -r 10; done; done

gpubounty:	cuda28 cuda30 cuda32
	for s in 28 30 32; do time ./cuda$$s -r 10; done

hashonly: hash_only.cpp Makefile
	$(GPP) -o hashonly hash_only.cpp $(LIBS)

hashonlyx4: hash_only.cpp Makefile
	$(GPP) -o hashonlyx4 -mavx2 -DNSIPHASH=4 hash_only.cpp $(LIBS)

hashonlyx8: hash_only.cpp Makefile
	$(GPP) -o hashonlyx8 -mavx2 -DNSIPHASH=8 hash_only.cpp $(LIBS)

hashonlyx16: hash_only.cpp Makefile
	$(GPP) -o hashonlyx16 -mavx2 -DNSIPHASH=16 hash_only.cpp $(LIBS)

verify28:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify28 -DEDGEBITS=27 cuckoo.c $(LIBS)

verify30:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify30 -DEDGEBITS=29 cuckoo.c $(LIBS)

verify32:	siphash.h cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify32 -DEDGEBITS=31 cuckoo.c $(LIBS)

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

SimpleMiner.class:	Cuckoo.java SimpleMiner.java Makefile
	javac -O Cuckoo.java SimpleMiner.java

java:	Cuckoo.class SimpleMiner.class Makefile
	java SimpleMiner -h 85 | tail -1 | java Cuckoo -h 85
