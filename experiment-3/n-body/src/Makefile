CC=mpiCC
CFLAGS= -O3 -D'DEBUG'
LIBS=-lmpi -lm

all: body.c
	$(CC) $(CFLAGS) body.c -o nbody $(LIBS)

clean: 
	rm -f nbody
