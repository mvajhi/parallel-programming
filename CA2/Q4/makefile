CC=g++

CFLAGS=-c -mavx2 -mfma

P=/home/mvajhi/intel/oneapi/ipp/latest

INCDIR=-I$(P)/include

LIBDIR=-L$(P)/lib

LIBFILE=-lippcore

all: main

main: main.o 
	$(CC) main.o $(LIBDIR) $(LIBFILE) -o main.out

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCDIR) main.cpp

clean:
	rm -f *o main