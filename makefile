CXX=g++
CXXFLAGS=-std=c++11 -Wall -march=native -g -fopenmp -fopenmp-simd 
CXXFLAGS+= -O3 -Ofast -march=native -malign-double -ftree-parallelize-loops=8 -std=c++2b 
CXXFLAGS+= -mavx -mavx2 -mfma -ffast-math -ftree-vectorize -fomit-frame-pointer
#-funroll-all-loops --param max-unroll-times=300
LDFLAGS= -lOpenCL #-lomp

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@  $(LDFLAGS)

clean:
	rm -f main