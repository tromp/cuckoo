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
LIBS ?= -L. -lblake2b

all : libblake2b.so

libblake2b.so:	blake2.h blake2-impl.h blake2b-ref.c
	$(GCC) -fPIC -shared -o libblake2b.so blake2b-ref.c
	echo "add build dir to LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on Mac OSX) if library not found"
	echo "or, if you have sudo rights, a better solution is to run"
	echo "sudo echo <build dir> > /etc/ld.so.conf.d/cuckatoo.conf"
	echo "sudo ldconfig"
