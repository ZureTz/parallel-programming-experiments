#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>

#include "base.h"
#include "hwtimer.h"
#include "knnCPU.h"
#include "knnCUDA.cu"

void knnInit(float *coords, float *newCoords, int *classes, int numSamples,
             int numClasses, int numNewSamples);
void genRandCoords(float *x, int numSamples);
void checkOutput(float *classes, float *gpuClasses, int numClasses,
                 int totalSamples);

void checkOutput(int *classes, int *gpuClasses, int numClasses,
                 int totalSamples) {
  int *numCpuClasses = (int *)malloc(sizeof(int) * numClasses);
  int *numGpuClasses = (int *)malloc(sizeof(int) * numClasses);

  for (int j = 0; j < numClasses; j++) {
    numCpuClasses[j] = 0;
    numGpuClasses[j] = 0;
  }

  for (int i = 0; i < totalSamples; i++) {
    for (int j = 0; j < numClasses; j++) {
      if (classes[i] == j)
        numCpuClasses[j] += 1;
      if (gpuClasses[i] == j)
        numGpuClasses[j] += 1;
    }
  }

  for (int j = 0; j < numClasses; j++) {
    if (numCpuClasses[j] != numGpuClasses[j]) {
      printf("Wrong output!\n");
      exit(1);
    }
  }

  printf("Output OK!\n\n");
}

void knnInit(float *coords, float *newCoords, int *classes, int numSamples,
             int numClasses, int numNewSamples) {
  for (int i = 0; i < numSamples; i++)
    classes[i] = rand() % numClasses;

  genRandCoords(coords, numSamples);
  genRandCoords(newCoords, numNewSamples);
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
void genRandCoords(float *x, int numSamples) {
  for (int i = 0; i < numSamples; i++)
    for (int j = 0; j < DIMENSION; j++)
      x[i * DIMENSION + j] =
          (float)rand() / (float)(RAND_MAX / POINTS_MAX) + (float)(POINTS_MIN);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("usage: ./knn_exec <k nearest neighbors> <number of classes> "
           "<number of existing samples> <number of new samples>\n");
    exit(1);
  }
  int device;
  cudaDeviceProp prop;

  // 获取当前使用的设备号
  cudaGetDevice(&device);

  // 获取设备的属性
  cudaGetDeviceProperties(&prop, device);

  // 打印每个维度支持的最大 Grid 尺寸
  printf("Max grid size:\n");
  printf("  X dimension: %d\n", prop.maxGridSize[0]);
  printf("  Y dimension: %d\n", prop.maxGridSize[1]);
  printf("  Z dimension: %d\n", prop.maxGridSize[2]);
  printf("Max shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
  hwtimer_t timer;
  initTimer(&timer);

  int k = atoi(argv[1]);             // number of k nearest neighbors
  int numClasses = atoi(argv[2]);    // number of classes
  int numSamples = atoi(argv[3]);    // number of existing samples
  int numNewSamples = atoi(argv[4]); // number of samples to classify
  int numTotalSamples = numSamples + numNewSamples; // total samples

  // array with a class for each sample
  int *classes;
  float *newCoords;
  float *coords;
  cudaMallocHost((void **)&classes, sizeof(int) * numTotalSamples);
  cudaMallocHost((void **)&newCoords,
                 sizeof(float) * numNewSamples * DIMENSION);
  cudaMallocHost((void **)&coords, sizeof(float) * numTotalSamples * DIMENSION);

  // gpu samples (initialized from cpu samples)
  int *gpuClasses = (int *)malloc(sizeof(int) * numTotalSamples);
  float *gpuNewCoords =
      (float *)malloc(sizeof(float) * numNewSamples * DIMENSION);
  float *gpuCoords =
      (float *)malloc(sizeof(float) * numTotalSamples * DIMENSION);

  srand(12345);

  printf("Starting initialization.\n");
  startTimer(&timer);
  knnInit(coords, newCoords, classes, numSamples, numClasses, numNewSamples);
  stopTimer(&timer);
  printf("Elapsed time: %lu ns.\n\n", getTimerNs(&timer));

  memcpy(gpuClasses, classes, sizeof(int) * numTotalSamples);
  memcpy(gpuNewCoords, newCoords, sizeof(float) * numNewSamples * DIMENSION);
  memcpy(gpuCoords, coords, sizeof(float) * numTotalSamples * DIMENSION);
  FILE *file = fopen("../data.txt", "a");
  printf("Starting sequential knn.\n");
  startTimer(&timer);
  knnSerial(coords, newCoords, classes, numClasses, numSamples, numNewSamples,
            k);
  stopTimer(&timer);
  fprintf(file, "%d %d %d %d %lu ", k, numClasses, numSamples, numNewSamples,
          getTimerNs(&timer));

  printf("Elapsed time: %lu ns.\n\n", getTimerNs(&timer));
  cuda_warmup();
  printf("Starting parallel knn.\n");
  startTimer(&timer);
  knnParallel(gpuCoords, gpuNewCoords, gpuClasses, numClasses, numSamples,
              numNewSamples, k);
  stopTimer(&timer);
  printf("Elapsed time: %lu ns.\n\n", getTimerNs(&timer));
  fprintf(file, "%lu\n", getTimerNs(&timer));
  checkOutput(classes, gpuClasses, numClasses, numTotalSamples);
  fclose(file);
  cudaFreeHost(classes);
  cudaFreeHost(newCoords);
  cudaFreeHost(coords);
  free(gpuClasses);
  free(gpuNewCoords);
  free(gpuCoords);

  return 0;
}
