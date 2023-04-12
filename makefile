CXX=g++
CXXFLAGS=-std=c++11 -Wall -march=native -O3 -g -fopenmp -fopenmp-simd -funroll-all-loops --param max-unroll-times=300
LDFLAGS= -lOpenCL -lomp

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@  $(LDFLAGS)

clean:
	rm -f main