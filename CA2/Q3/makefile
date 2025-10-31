CC=clang++

P=/home/mvajhi/intel/oneapi/ipp/latest

CFLAGS=-c `pkg-config --cflags opencv4` -mavx2 -mfma
LIBFLAGS=`pkg-config --libs opencv4`

INCDIR=-I/opt/intel/ipp/include -I/usr/local/Cellar/opencv/4.10.0_11/include/opencv4 -I$(P)/include

LIBDIR=-L/opt/intel/ipp/lib -L/usr/local/Cellar/opencv/4.10.0_11/lib -L$(P)/lib

LIBFILE=-lippcore

CST11=-std=c++11

all: main

main: main.o 
	$(CC) $(LIBFLAGS) $(LIBDIR) $(LIBFILE) $(CST11) main.o -o main.out

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCDIR) $(CST11) main.cpp

clean:
	rm -f *o main

