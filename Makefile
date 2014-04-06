# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS = -Wall -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L -O3 -pthread -l crypto
# leave out -l crypto if using sha256.c instead of openssl
CC = cc -std=c99 $(FLAGS)
GPP = g++ -std=c++11 $(FLAGS)

cuckoo:		cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo -DSHOW -DIDXSHIFT=0 -DPROOFSIZE=6 -DSIZESHIFT=4 cuckoo_miner.cpp

example:	cuckoo
	./cuckoo -e 100 -h header -n 1

cuckoo10:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo10 -DSIZESHIFT=10 cuckoo_miner.cpp

cuckoo15:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo15 -DSIZESHIFT=15 cuckoo_miner.cpp

cuckoo20:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo20 -DSIZESHIFT=20 cuckoo_miner.cpp

fast20:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o fast20 -DHUGEFAST -DSIZESHIFT=20 cuckoo_miner.cpp

verify20:	cuckoo.h cuckoo.c Makefile
	$(CC) -o verify20 -DSIZESHIFT=20 cuckoo.c

test:	cuckoo20 verify20 Makefile
	./cuckoo20 -h 85 | tail -1 | ./verify20 -h 85

cuckoo25:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo25 -DSIZESHIFT=25 cuckoo_miner.cpp

fast25:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o fast25 -DHUGEFAST -DSIZESHIFT=25 cuckoo_miner.cpp

cuckoo28:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28 -DSIZESHIFT=28 cuckoo_miner.cpp

speedup:	cuckoo28 Makefile
	for i in {1..4}; do echo $$i; (time for j in {0..6}; do ./cuckoo28 -t $$i -h $$j; done) 2>&1; done > speedup

cuckoo30:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30 -DSIZESHIFT=30 cuckoo_miner.cpp

speedup30:	cuckoo30 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..9}; do ./cuckoo30 -t $$i -h $$j; done) 2>&1; done > speedup130

cuckoo30p0:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30p0 -DPRESIP=0 -DSIZESHIFT=30 cuckoo_miner.cpp

speedup30p0:	cuckoo30p0 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..9}; do ./cuckoo30p0 -t $$i -h $$j; done) 2>&1; done > speedup130p0

cuckoo32:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo729 -DSIZESHIFT=32 cuckoo_miner.cpp

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

CuckooMiner.class:	Cuckoo.java CuckooMiner.java Makefile
	javac -O Cuckoo.java CuckooMiner.java

java:	Cuckoo.class CuckooMiner.class Makefile
	java CuckooMiner -h 6 | tail -1 | java Cuckoo -h 6

cuda:	cuda_miner.cu Makefile
	nvcc -std=c++11 -o cuda -DSIZESHIFT=4 -arch sm_20 cuda_miner.cu -lcrypto
	./cuda -e 100 -h header

cuda28:	cuda_miner.cu Makefile
	nvcc -std=c++11 -o cuda128 -DSIZESHIFT=28 -arch sm_20 cuda_miner.cu -lcrypto

speedupcuda:	cuda28
	for i in 1 2 4 8 16 32 64 128 256 512; do echo $$i; (time for j in {0..6}; do ./cuda128 -t $$i -h $$j; done) 2>&1; done > speedupcuda

tar:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp osx_barrier.h Makefile
	tar -cf cuckoo.tar cuckoo.h cuckoo_miner.h cuckoo_miner.cpp osx_barrier.h Makefile
