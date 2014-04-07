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

tomato10:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato10 -DTRIMEDGES -DSIZESHIFT=10 cuckoo_miner.cpp

cuckoo15:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo15 -DSIZESHIFT=15 cuckoo_miner.cpp

tomato15:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato15 -DTRIMEDGES -DSIZESHIFT=15 cuckoo_miner.cpp

cuckoo20:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo20 -DSIZESHIFT=20 cuckoo_miner.cpp

tomato20:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato20 -DTRIMEDGES -DSIZESHIFT=20 cuckoo_miner.cpp

verify20:	cuckoo.h cuckoo.c Makefile
	$(CC) -o verify20 -DSIZESHIFT=20 cuckoo.c

test:	cuckoo20 verify20 Makefile
	./cuckoo20 -h 85 | tail -1 | ./verify20 -h 85

cuckoo25:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo25 -DSIZESHIFT=25 cuckoo_miner.cpp

tomato25:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato25 -DTRIMEDGES -DSIZESHIFT=25 cuckoo_miner.cpp

cuckoo28:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo28 -DSIZESHIFT=28 cuckoo_miner.cpp

tomato28:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato28 -DTRIMEDGES -DSIZESHIFT=28 cuckoo_miner.cpp

speedup:	cuckoo28 Makefile
	for i in {1..4}; do echo $$i; (time for j in {0..6}; do ./cuckoo28 -t $$i -h $$j; done) 2>&1; done > speedup

ketchup:	tomato28 Makefile
	for i in {1..4}; do echo $$i; (time for j in {0..6}; do ./tomato28 -t $$i -h $$j; done) 2>&1; done > ketchup

cuckoo30:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o cuckoo30 -DSIZESHIFT=30 cuckoo_miner.cpp

tomato30:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	$(GPP) -o tomato30 -DTRIMEDGES -DSIZESHIFT=30 cuckoo_miner.cpp

speedup30:	cuckoo30 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..9}; do ./cuckoo30 -t $$i -h $$j; done) 2>&1; done > speedup30

ketchup30:	tomato30 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..6}; do ./tomato30 -t $$i -h $$j; done) 2>&1; done > ketchup30

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
	nvcc -std=c++11 -o cuda28 -DSIZESHIFT=28 -arch sm_20 cuda_miner.cu -lcrypto

speedupcuda:	cuda28
	for i in 1 2 4 8 16 32 64 128 256 512; do echo $$i; (time for j in {0..6}; do ./cuda28 -t $$i -h $$j; done) 2>&1; done > speedupcuda

tar:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp osx_barrier.h trim_edge_data.h Makefile
	tar -cf cuckoo.tar cuckoo.h cuckoo_miner.h cuckoo_miner.cpp osx_barrier.h trim_edge_data.h Makefile
