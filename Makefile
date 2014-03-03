# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS = -O3 -std=c99 -Wall -Wno-deprecated-declarations -pthread -l crypto

cuckoo:		cuckoo.h cuckoo.c Makefile
	cc -o cuckoo -DSHOW -DPROOFSIZE=6 -DSIZEMULT=1 -DSIZESHIFT=4 -DEASINESS=16 cuckoo.c ${FLAGS}

example:	cuckoo
	./cuckoo "header"

cuckoo110:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo110 -DSIZEMULT=1 -DSIZESHIFT=10 cuckoo.c ${FLAGS}

cuckoo115:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo115 -DSIZEMULT=1 -DSIZESHIFT=15 cuckoo.c ${FLAGS}

cuckoo120:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo120 -DSIZEMULT=1 -DSIZESHIFT=20 cuckoo.c ${FLAGS}

cuckoo120.4:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo120.4 -DNTHREADS=4 -DSIZEMULT=1 -DSIZESHIFT=20 cuckoo.c ${FLAGS}

cuckoo125:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo125 -DSIZEMULT=1 -DSIZESHIFT=25 cuckoo.c ${FLAGS}

cuckoo125.4:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo125.4 -DNTHREADS=4 -DSIZEMULT=1 -DSIZESHIFT=25 cuckoo.c ${FLAGS}

cuckoo130:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo130 -DSIZEMULT=1 -DSIZESHIFT=30 cuckoo.c ${FLAGS}

cuckoo130.4:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo130.4 -DNTHREADS=4 -DSIZEMULT=1 -DSIZESHIFT=30 cuckoo.c ${FLAGS}

cuckoo729.4:	cuckoo.h cuckoo.c Makefile
	cc -o cuckoo130.4 -DNTHREADS=4 -DSIZEMULT=7 -DSIZESHIFT=29 cuckoo.c ${FLAGS}

verify120:	cuckoo.h verify.c Makefile
	cc -o verify120 -DSIZEMULT=1 -DSIZESHIFT=20 verify.c ${FLAGS}

test:	cuckoo120 verify120 Makefile
	./cuckoo120 6 | tail -1 | ./verify120 6

speedup.25:	cuckoo.h cuckoo.c Makefile
	for i in {1..8}; do echo $$i; cc -o cuckoo.spd -DNTHREADS=$$i -DSIZEMULT=1 -DSIZESHIFT=25 cuckoo.c ${FLAGS}; time for j in {0..9}; do ./cuckoo.spd $$j; done; done

