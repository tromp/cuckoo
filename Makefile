# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS = -Wall -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L -O3 -pthread -l crypto
# leave out -l crypto if using sha256.c instead of openssl

cuckoo:		cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo -DSHOW -DPROOFSIZE=6 -DSIZESHIFT=4 cuckoo_miner.cpp ${FLAGS}

example:	cuckoo
	./cuckoo -e 100 -h header

cuckoo110:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo110 -DSIZESHIFT=10 cuckoo_miner.cpp ${FLAGS}

cuckoo115:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo115 -DSIZESHIFT=15 cuckoo_miner.cpp ${FLAGS}

cuckoo120:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo120 -DSIZESHIFT=20 cuckoo_miner.cpp ${FLAGS}

verify120:	cuckoo.h cuckoo.c Makefile
	cc  -o verify120 -DSIZESHIFT=20 cuckoo.c ${FLAGS}

test:	cuckoo120 verify120 Makefile
	./cuckoo120 -h 4 | tail -1 | ./verify120 -h 4

cuckoo125:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo125 -DSIZESHIFT=25 cuckoo_miner.cpp ${FLAGS}

cuckoo128:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo128 -DSIZESHIFT=28 cuckoo_miner.cpp ${FLAGS}

speedup:	cuckoo128 Makefile
	for i in {1..4}; do echo $$i; (time for j in {0..6}; do ./cuckoo128 -t $$i -h $$j; done) 2>&1; done > speedup

cuckoo130:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo130 -DSIZESHIFT=30 cuckoo_miner.cpp ${FLAGS}

speedup130:	cuckoo130 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..9}; do ./cuckoo130 -t $$i -h $$j; done) 2>&1; done > speedup130

cuckoo130p0:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo130p0 -DPRESIP=0 -DSIZESHIFT=30 cuckoo_miner.cpp ${FLAGS}

speedup130p0:	cuckoo130p0 Makefile
	for i in {1..64}; do echo $$i; (time for j in {0..9}; do ./cuckoo130p0 -t $$i -h $$j; done) 2>&1; done > speedup130p0

cuckoo729:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	g++ -std=c++11 -o cuckoo729 -DSIZESHIFT=29 cuckoo_miner.cpp ${FLAGS}

Cuckoo.class:	Cuckoo.java Makefile
	javac -O Cuckoo.java

CuckooMiner.class:	Cuckoo.java CuckooMiner.java Makefile
	javac -O Cuckoo.java CuckooMiner.java

java:	Cuckoo.class CuckooMiner.class Makefile
	java CuckooMiner -h 6 | tail -1 | java Cuckoo -h 6

cuda:	cuda_miner.cu Makefile
	nvcc -std=c++11 -o cuda -DSIZESHIFT=4 -arch sm_20 cuda_miner.cu -lcrypto
	./cuda -e 100 -h header

cuda128:	cuda_miner.cu Makefile
	nvcc -std=c++11 -o cuda128 -DSIZESHIFT=28 -arch sm_20 cuda_miner.cu -lcrypto

speedupcuda:	cuda128
	for i in 1 2 4 8 16 32 64 128 256 512; do echo $$i; (time for j in {0..6}; do ./cuda128 -t $$i -h $$j; done) 2>&1; done > speedupcuda

tar:	cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
	tar -cf cuckoo.tar cuckoo.h cuckoo_miner.h cuckoo_miner.cpp Makefile
