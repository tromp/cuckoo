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

verify:	cuckoo.h verify.c Makefile
	cc -o verify verify.c ${FLAGS}

test:	cuckoo120 verify Makefile
	./cuckoo120 6 | tail -42 | ./verify 6
