#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "base.h"

void check_error(cudaError_t err, const char *msg);
void knnParallel(float *coords, float *newCoords, int *classes, int numClasses,
                 int numSamples, int numNewSamples, int k);

__global__ void kernelknn(float *coords, float *newCoords, int *classes,
                          int numClasses, int numSamples, int numNewSamples,
                          int k) {
  int bid = blockIdx.x;  // 每个 block 负责一个新样本点
  int tid = threadIdx.x; // 每个线程负责一个旧样本点
  __shared__ float distances[10000];
  __shared__ float sharedneighdist[400];
  __shared__ int sharedneighclass[400];
  float neighdist[20];
  int neighclass[20];
  for (int i = 0; i < k; i++) {
    neighdist[i] = FLT_MAX;
    neighclass[i] = -1;
  }
  for (int t = 0; t < numSamples / 10000; t++) {
    for (int i = tid; i < 10000; i += blockDim.x) {
      float distance = 0.0;
      for (int d = 0; d < DIMENSION; d++) {
        float diff = newCoords[bid * DIMENSION + d] -
                     coords[(i + t * 10000) * DIMENSION + d];
        distance += diff * diff;
      }
      distances[i] = distance;
    }
    __syncthreads();
    if (tid < 20) {
      for (int i = 0; i < 500; i++) {
        if (distances[i + tid * 500] < neighdist[k - 1]) {
          neighdist[k - 1] = distances[i + tid * 500];
          neighclass[k - 1] = classes[i + tid * 500 + t * 10000];
          for (int m = k - 1; m > 0 && neighdist[m] < neighdist[m - 1]; m--) {
            float tempdist = neighdist[m];
            neighdist[m] = neighdist[m - 1];
            neighdist[m - 1] = tempdist;
            int tempclass = neighclass[m];
            neighclass[m] = neighclass[m - 1];
            neighclass[m - 1] = tempclass;
          }
        }
      }
    }
    __syncthreads();
  }
  if (tid < 20) {
    for (int i = 0; i < k; i++) {
      sharedneighdist[tid * k + i] = neighdist[i];
      sharedneighclass[tid * k + i] = neighclass[i];
    }
  }
  __shared__ float minElements[20];
  __shared__ int minClasses[20];

  // 初始化最小元素和类别数组
  if (tid < k) {
    minElements[tid] = FLT_MAX;
    minClasses[tid] = -1; // 假设类别从-1开始
  }

  // 找到前K个最小的元素
  if (tid == 0)
    for (int i = 0; i < 20 * k; i++) {
      float currentDist = sharedneighdist[i];
      int currentClass = sharedneighclass[i];

      // 插入新元素
      for (int j = 0; j < k; j++) {
        if (currentDist < minElements[j]) {
          // 插入新元素，并保持顺序
          for (int l = k - 1; l > j; l--) {
            minElements[l] = minElements[l - 1];
            minClasses[l] = minClasses[l - 1];
          }
          minElements[j] = currentDist;
          minClasses[j] = currentClass;
          break;
        }
      }
    }
  __shared__ int classCounts[20];
  if (tid < numClasses)
    classCounts[tid] = 0;
  __syncthreads();
  if (tid < k)
    atomicAdd(&classCounts[minClasses[tid]], 1);
  if (tid == 0) {
    int maxCount = 0;
    int predictedClass = 0;
    for (int c = 0; c < numClasses; c++) {
      if (classCounts[c] > maxCount) {
        maxCount = classCounts[c];
        predictedClass = c;
      }
    }
    classes[numSamples + bid] = predictedClass;
  }
}

void knnParallel(float *coords, float *newCoords, int *classes, int numClasses,
                 int numSamples, int numNewSamples, int k) {
  float *d_coords;
  float *d_newCoords;
  int *d_classes;

  int totalSamples = numSamples + numNewSamples;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  check_error(cudaMalloc(&d_coords, numSamples * DIMENSION * sizeof(float)),
              "alloc d_coords");
  check_error(
      cudaMalloc(&d_newCoords, numNewSamples * DIMENSION * sizeof(float)),
      "alloc d_newCoords");
  check_error(cudaMalloc(&d_classes, totalSamples * sizeof(int)),
              "alloc d_classes");

  check_error(cudaMemcpy(d_coords, coords,
                         numSamples * DIMENSION * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy d_coords");
  check_error(cudaMemcpy(d_newCoords, newCoords,
                         numNewSamples * DIMENSION * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy d_newCoords");
  check_error(cudaMemcpy(d_classes, classes, totalSamples * sizeof(int),
                         cudaMemcpyHostToDevice),
              "copy d_classes");

  // 启动距离计算的 kernel
  int threadnum = 32 * 8;
  int blocknum = numNewSamples * (numSamples / numSamples);
  cudaEventRecord(start); // 记录开始时间
  kernelknn<<<blocknum, threadnum>>>(d_coords, d_newCoords, d_classes,
                                     numClasses, numSamples, numNewSamples, k);
  cudaDeviceSynchronize();
  cudaEventRecord(stop); // 记录结束时间

  cudaEventSynchronize(stop); // 确保事件完成

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间差
  long long nanoseconds = (long long)(milliseconds * 1000000); // 转换为纳秒
  // printf("Elapsed time: %lld ns.\n", nanoseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // 复制结果回主机
  check_error(cudaMemcpy(classes, d_classes, totalSamples * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "copy back classes");

  // 释放 GPU 内存
  cudaFree(d_coords);
  cudaFree(d_newCoords);
  cudaFree(d_classes);
}

void check_error(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s : error %d (%s)\n", msg, err, cudaGetErrorString(err));
    exit(err);
  }
}
