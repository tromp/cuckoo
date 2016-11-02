# -Wno-deprecated-declarations shuts up Apple OSX clang
OPT = -O3
DOPT = -DPREFETCH
FLAGS = -Wall -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) $(DOPT) -pthread 
GPP = g++ -march=native -m64 -std=c++11 $(FLAGS)
LIBS = -lssl -lcrypto
# leave out -l crypto if using sha256.c instead of openssl
# leave out -std=c++11 -DATOMIC for older GCC versions lacking c++11 support

all : test example cuckoo28st cuckoo30st cuckoo32st cuckoo28 cuckoo30 cuckoo32 verify28 verify30 verify32

test:	cuckoo20 verify20 Makefile
	./cuckoo20 -n 13 | grep ^Sol | ./verify20 -n 13

cuckoo4:	cuckoo.h cuckoo_miner.h simple_miner.cpp Makefile
	$(GPP) -o cuckoo4 -DSHOW -DIDXSHIFT=0 -DPROOFSIZE=6 -DSIZESHIFT=4 simple_miner.cpp $(LIBS)

cuckoo6:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo6 -DIDXSHIFT=0 -DPROOFSIZE=6 -DLOGNBUCKETS=0 -DSHOWSOL -DSIZESHIFT=6 cuckoo_miner.cpp $(LIBS)

example:	cuckoo4
	./cuckoo4 -e 66 -h 3

cuckoo10:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo10 -DATOMIC -DSIZESHIFT=10 -DPROOFSIZE=8 -DSHOWSOL cuckoo_miner.cpp $(LIBS)

cuckoo15:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo15 -DATOMIC -DSIZESHIFT=15 cuckoo_miner.cpp $(LIBS)

cuckoo20:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo20 -DATOMIC -DSIZESHIFT=20 cuckoo_miner.cpp $(LIBS)

verify20:	cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify20 -DSIZESHIFT=20 cuckoo.c $(LIBS)

cuckoo25:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo25 -DATOMIC -DSIZESHIFT=25 cuckoo_miner.cpp $(LIBS)

momentum:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o momentum -DATOMIC -DSIZESHIFT=25 -DPROOFSIZE=2 cuckoo_miner.cpp $(LIBS)

momentomatum:	cuckoo.h momentomatum.h momentomatum.cpp Makefile
	$(GPP) -o momentomatum -DATOMIC -DSIZESHIFT=25 momentomatum.cpp $(LIBS)

cuckoo28:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28 -DATOMIC -DSIZESHIFT=28 cuckoo_miner.cpp $(LIBS)

cuckoo28.1:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28.1 -DATOMIC -DPART_BITS=1 -DSIZESHIFT=28 cuckoo_miner.cpp $(LIBS)

cuckoo28st:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28 -DSIZESHIFT=28 cuckoo_miner.cpp $(LIBS)

cuckoo28sc:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28sc -DATOMIC -DSINGLECYCLING -DSIZESHIFT=28 cuckoo_miner.cpp $(LIBS)

cuckoo28.2:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28.2 -DATOMIC -DPART_BITS=2 -DSIZESHIFT=28 cuckoo_miner.cpp $(LIBS)

speedup:	cuckoo28 Makefile
	for t in 1 2 4 6 8 12 16; do echo $$i; time ./cuckoo28 -t $$t -r 10 2>&1; done > speedup

cudaspeedup28:	cuda28 Makefile
	for t in 65536 32768 16384 8192 4096 2048 1024; do echo $$t; time ./cuda28 -t $$t -r 10; done;

cuckoo30:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30 -DATOMIC -DSIZESHIFT=30 cuckoo_miner.cpp $(LIBS)

cuckoo30.1:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.1 -DATOMIC -DPART_BITS=1 -DSIZESHIFT=30 cuckoo_miner.cpp $(LIBS)

cuckoo30st:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30 -DSIZESHIFT=30 cuckoo_miner.cpp $(LIBS)

cuckoo30.2:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.2 -DATOMIC -DPART_BITS=2 -DSIZESHIFT=30 cuckoo_miner.cpp $(LIBS)

cuckoo30.3:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30.3 -DATOMIC -DPART_BITS=3 -DSIZESHIFT=30 cuckoo_miner.cpp $(LIBS)

speedup30:	cuckoo30 Makefile
	for t in 1 2 4 6 8; do echo $$i; do ./cuckoo30 -t $$t -r 10 2>&1; done > speedup30

tomato20:	cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato20 -DATOMIC -DSIZESHIFT=20 tomato_miner.cpp $(LIBS)

tomato25:	cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato25 -DATOMIC -DSIZESHIFT=25 tomato_miner.cpp $(LIBS)

tomato28:	cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato28 -DATOMIC -DSIZESHIFT=28 tomato_miner.cpp $(LIBS)

tomato30:	cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato30 -DATOMIC -DSIZESHIFT=30 tomato_miner.cpp $(LIBS)

tomato32:	cuckoo.h tomato_miner.h tomato_miner.cpp Makefile
	$(GPP) -o tomato32 -DATOMIC -DSIZESHIFT=32 tomato_miner.cpp $(LIBS)

cuda28:	cuda_miner.cu Makefile
	nvcc -o cuda28 -DSIZESHIFT=28 -arch sm_35 cuda_miner.cu -lssl -lcrypto 

cuda30:	cuda_miner.cu Makefile
	nvcc -o cuda30 -DSIZESHIFT=30 -arch sm_35 cuda_miner.cu -lssl -lcrypto

cuda32:	cuda_miner.cu Makefile
	nvcc -o cuda32 -DSIZESHIFT=32 -arch sm_35 cuda_miner.cu -lssl -lcrypto

cycle28:	cycle_miner.cu Makefile
	nvcc -o cycle28 -DSIZESHIFT=28 -arch sm_35 cycle_miner.cu -lssl -lcrypto 

cycle30:	cycle_miner.cu Makefile
	nvcc -o cycle30 -DSIZESHIFT=30 -arch sm_35 cycle_miner.cu -lssl -lcrypto 

cuckoo32:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32 -DATOMIC -DSIZESHIFT=32 cuckoo_miner.cpp $(LIBS)

cuckoo32.1:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32.1 -DATOMIC -DPART_BITS=1 -DSIZESHIFT=32 cuckoo_miner.cpp $(LIBS)

cuckoo32st:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32 -DSIZESHIFT=32 cuckoo_miner.cpp $(LIBS)

cuckoo32.2:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo32.2 -DATOMIC -DPART_BITS=2 -DSIZESHIFT=32 cuckoo_miner.cpp $(LIBS)

cuckoo35:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo35 -DATOMIC -DSIZESHIFT=35 cuckoo_miner.cpp $(LIBS)

cpubounty:	cuckoo28 cuckoo30 cuckoo32
	for s in 28 30 32; do for t in 8 4 2 1; do time ./cuckoo$$s -t $$t -r 10; done; done

tmtobounty:	tomato28 tomato30 tomato32
	for s in 28 30 32; do for t in 8 4 2 1; do time ./tomato$$s -t $$t -r 10; done; done

gpubounty:	cuda28 cuda30 cuda32
	for s in 28 30 32; do time ./cuda$$s -r 10; done

verify28:	cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify28 -DSIZESHIFT=28 cuckoo.c $(LIBS)

verify30:	cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify30 -DSIZESHIFT=30 cuckoo.c $(LIBS)

verify32:	cuckoo.h cuckoo.c Makefile
	$(GPP) -o verify32 -DSIZESHIFT=32 cuckoo.c $(LIBS)

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

SimpleMiner.class:	Cuckoo.java SimpleMiner.java Makefile
	javac -O Cuckoo.java SimpleMiner.java

java:	Cuckoo.class SimpleMiner.class Makefile
	java SimpleMiner -h 85 | tail -1 | java Cuckoo -h 85
