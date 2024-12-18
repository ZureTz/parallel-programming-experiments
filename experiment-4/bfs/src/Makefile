CC=gcc
CXX=g++
NVCC=/usr/local/cuda-12.4/bin/nvcc

CXXFLAGS= -O3 -std=c++11

all: main.cu bfsCUDA.cu bfsCPU.cpp graph.cpp
	$(NVCC) $(CXXFLAGS) main.cu bfsCPU.cpp graph.cpp -o bfs-exec

clean:
	rm -f bfs-exec
