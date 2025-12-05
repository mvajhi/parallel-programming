CC=g++

CFLAGS=-c -std=c++17 -O3 -mavx2 -pthread

INCDIR=$(shell pkg-config --cflags opencv4)

LIBDIR=

LIBFILE=$(shell pkg-config --libs opencv4) -pthread

all: main

main: main.o 
	$(CC) main.o $(LIBDIR) $(LIBFILE) -o main.out

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCDIR) main.cpp

clean:
	rm -f *o main.out