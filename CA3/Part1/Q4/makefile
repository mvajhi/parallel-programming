CC=clang++

CFLAGS=-c -mavx2 -mfma -fopenmp
LIBFLAGS=-fopenmp

CST11=-std=c++11

all: main

main: main.o 
	$(CC) $(LIBFLAGS) $(CST11) main.o -o main.out

main.o: main.cpp
	$(CC) $(CFLAGS) $(CST11) main.cpp

clean:
	rm -f *o main