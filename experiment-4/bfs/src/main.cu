#include <cstdio>
#include <string>

#include <cuda.h>

#include "bfsCPU.h"
#include "bfsCUDA.cu"
#include "graph.h"
#include "hwtimer.h"

#define GPU_DEVICE 0

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
  bfsCPU(startVertex, G, distance, parent, visited);
}

void checkError(cudaError_t error, std::string msg) {
  if (error != cudaSuccess) {
    printf("%s: %d\n", msg.c_str(), error);
    exit(1);
  }
}

cudaDeviceProp deviceProp;

int *d_adjacencyList;
int *d_edgesOffset;
int *d_edgesSize;
int *d_distance;
int *d_parent;
int *d_currentQueue;
int *d_nextQueue;
int *d_degrees;
int *incrDegrees;

void initCuda(Graph &G) {
  // initialize CUDA
  checkError(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE),
             "cannot get device");
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  checkError(cudaSetDevice(GPU_DEVICE), "cannot set device");

  // copy memory to device
  checkError(cudaMalloc(&d_adjacencyList, G.numEdges * sizeof(int)),
             "cannot allocate d_adjacencyList");
  checkError(cudaMalloc(&d_edgesOffset, G.numVertices * sizeof(int)),
             "cannot allocate d_edgesOffset");
  checkError(cudaMalloc(&d_edgesSize, G.numVertices * sizeof(int)),
             "cannot allocate d_edgesSize");
  checkError(cudaMalloc(&d_distance, G.numVertices * sizeof(int)),
             "cannot allocate d_distance");
  checkError(cudaMalloc(&d_parent, G.numVertices * sizeof(int)),
             "cannot allocate d_parent");
  checkError(cudaMalloc(&d_currentQueue, G.numVertices * sizeof(int)),
             "cannot allocate d_currentQueue");
  checkError(cudaMalloc(&d_nextQueue, G.numVertices * sizeof(int)),
             "cannot allocate d_nextQueue");
  checkError(cudaMalloc(&d_degrees, G.numVertices * sizeof(int)),
             "cannot allocate d_degrees");
  checkError(cudaMallocHost((void **)&incrDegrees, sizeof(int) * G.numVertices),
             "cannot allocate memory");

  checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(),
                        G.numEdges * sizeof(int), cudaMemcpyHostToDevice),
             "cannot copy to d_adjacencyList");
  checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(),
                        G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
             "cannot copy to d_edgesOffset");
  checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(),
                        G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
             "cannot copy to d_edgesSize");
}

void finalizeCuda() {
  // free memory
  checkError(cudaFree(d_adjacencyList),
             "cannot free memory for d_adjacencyList");
  checkError(cudaFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
  checkError(cudaFree(d_edgesSize), "cannot free memory for d_edgesSize");
  checkError(cudaFree(d_distance), "cannot free memory for d_distance");
  checkError(cudaFree(d_parent), "cannot free memory for d_parent");
  checkError(cudaFree(d_currentQueue), "cannot free memory for d_parent");
  checkError(cudaFree(d_nextQueue), "cannot free memory for d_parent");
  checkError(cudaFreeHost(incrDegrees), "cannot free memory for incrDegrees");
}

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance,
                 Graph &G) {
  for (int i = 0; i < G.numVertices; i++) {
    if (distance[i] != expectedDistance[i]) {
      printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
      printf("Wrong output!\n");
      exit(1);
    }
  }

  printf("Output OK!\n\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance,
                       std::vector<int> &parent, Graph &G) {
  // initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  distance[startVertex] = 0;
  parent[startVertex] = 0;

  checkError(cudaMemcpy(d_distance, distance.data(),
                        G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
             "cannot copy to d)distance");
  checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cannot copy to d_parent");

  int firstElementQueue = startVertex;
  cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int),
             cudaMemcpyHostToDevice);
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent,
                     Graph &G) {
  // copy memory from device
  checkError(cudaMemcpy(distance.data(), d_distance,
                        G.numVertices * sizeof(int), cudaMemcpyDeviceToHost),
             "cannot copy d_distance to host");
  checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cannot copy d_parent to host");
}

void runCudaBfs(int startVertex, Graph &G, std::vector<int> &distance,
                std::vector<int> &parent) {
  initializeCudaBfs(startVertex, distance, parent, G);

  int blockSize = 128; // 每个线程块的线程数
  int *d_nextQueueSize;
  checkError(cudaMalloc(&d_nextQueueSize, sizeof(int)),
             "cannot allocate d_nextQueueSize");

  int currentQueueSize = 1; // 初始队列大小为1（起始节点）
  while (currentQueueSize > 0) {
    // 初始化下一层队列大小为0
    checkError(cudaMemset(d_nextQueueSize, 0, sizeof(int)),
               "cannot reset d_nextQueueSize");

    // 计算网格大小
    int gridSize = (currentQueueSize + blockSize / 32 - 1) / (blockSize / 32);

    // 启动 CUDA 核函数
    bfsKernel<<<gridSize, blockSize>>>(
        d_adjacencyList, d_edgesOffset, d_edgesSize, d_currentQueue,
        d_nextQueue, d_distance, d_parent, d_nextQueueSize, currentQueueSize);
    checkError(cudaDeviceSynchronize(), "kernel launch failed");

    // 读取下一层队列大小
    checkError(cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cannot copy nextQueueSize to host");

    // 交换队列指针
    std::swap(d_currentQueue, d_nextQueue);
  }

  cudaFree(d_nextQueueSize); // 释放临时变量
  finalizeCudaBfs(distance, parent, G);
}

__global__ void multiply_by_two(int *d_data, int n) {
  int idx = threadIdx.x;
  if (idx < n)
    d_data[idx] += 2;
}
void cuda_warmup() {
  int VECTOR_SIZE = 100;
  int h_data[100];
  int *d_data;
  for (int i = 0; i < VECTOR_SIZE; i++)
    h_data[i] = i + 1;
  cudaMalloc((void **)&d_data, VECTOR_SIZE * sizeof(int));
  cudaMemcpy(d_data, h_data, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  for (int i = 0; i < 1000; i++)
    multiply_by_two<<<1, VECTOR_SIZE>>>(d_data, VECTOR_SIZE);
  cudaMemcpy(h_data, d_data, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./bfs-exec <start vertex> <number of vertices> <number of "
           "edges>\n");
    exit(1);
  }

  // read graph from standard input
  Graph G;
  int startVertex = atoi(argv[1]);

  readGraph(G, argc, argv);

  printf("Number of vertices %d\n", G.numVertices);
  printf("Number of edges %d\n\n", G.numEdges);

  // vectors for results
  std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
  std::vector<bool> visited(G.numVertices, false);

  FILE *file = fopen("../data.txt", "a");

  HWTimer timer;

  printf("Starting sequential bfs.\n");

  const hwtime_t startTime = timer.get_time_ns();
  // run CPU sequential bfs
  runCpu(startVertex, G, distance, parent, visited);
  const hwtime_t endTime = timer.get_time_ns();

  const hwtime_t diff = endTime - startTime;
  const long diffInNS = diff.tv_nsec + ONE_S_TO_NS * diff.tv_sec;

  bool usingCUDA = false;
  fprintf(file, "%d %d %d %d %lu\n", usingCUDA, startVertex, G.numVertices,
          G.numEdges, diffInNS);

  printf("Elapsed time: %lu ns.\n\n", diffInNS);
  // save results from sequential bfs
  std::vector<int> expectedDistance(distance);
  std::vector<int> expectedParent(parent);

  // run CUDA simple parallel bfs
  printf("Starting parallel bfs.\n");

  initCuda(G);
  cuda_warmup();

  const hwtime_t startTimeCUDA = timer.get_time_ns();
  runCudaBfs(startVertex, G, distance, parent);
  const hwtime_t endTimeCUDA = timer.get_time_ns();
  const hwtime_t diffCUDA = endTimeCUDA - startTimeCUDA;
  const long diffCUDAInNS = diffCUDA.tv_nsec + ONE_S_TO_NS * diffCUDA.tv_sec;

  usingCUDA = true;
  fprintf(file, "%d %d %d %d %lu\n", usingCUDA, startVertex, G.numVertices,
          G.numEdges, diffCUDAInNS);

  printf("Elapsed time: %lu ns.\n\n", diffCUDAInNS);

  checkOutput(distance, expectedDistance, G);

  finalizeCuda();
  return 0;
}
