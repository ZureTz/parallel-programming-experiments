CC=gcc
CXX=g++
NVCC=/usr/local/cuda-12.4/bin/nvcc

NVCCFLAGS= -O3 -std=c++11

all: main.cu knnCPU.cpp
	$(NVCC) $(NVCCFLAGS) main.cu knnCPU.cpp -o knn-exec

clean:
	rm -f knn-exec
