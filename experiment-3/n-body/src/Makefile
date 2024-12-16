CC=mpiCC
CFLAGS= -O3
# GFLAGS= -D'DEBUG'
LIBS=-lmpi -lm

all: body.c
	$(CC) $(CFLAGS) $(GFLAGS) body.c -o nbody $(LIBS)

clean: 
	rm -f nbody
