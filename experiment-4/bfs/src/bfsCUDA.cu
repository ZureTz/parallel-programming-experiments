#include <device_launch_parameters.h>

extern "C" {
// __global__ void bfsKernel(const int* adjacencyList, const int* edgesOffset,
// const int* edgesSize,
//                             const int* currentQueue, int* nextQueue, int*
//                             distance, int* parent, int* nextQueueSize, int
//                             currentQueueSize)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;    // 每个线程处理
//     currentQueue 中的一个节点 if (tid >= currentQueueSize) return; int node =
//     currentQueue[tid];           // 当前线程处理的节点 int nodeDistance =
//     distance[node]; int start = edgesOffset[node];          //
//     遍历该节点的邻居 int end = start + edgesSize[node]; for (int i = start; i
//     < end; ++i)
//     {
//         int neighbor = adjacencyList[i];
//         if (atomicCAS(&distance[neighbor], INT_MAX, nodeDistance + 1) ==
//         INT_MAX)
//         {
//             parent[neighbor] = node;        // 设置邻居的父节点
//             int index = atomicAdd(nextQueueSize, 1);
//             nextQueue[index] = neighbor;
//         }
//     }
// }
__global__ void bfsKernel(const int *adjacencyList, const int *edgesOffset,
                          const int *edgesSize, const int *currentQueue,
                          int *nextQueue, int *distance, int *parent,
                          int *nextQueueSize, int currentQueueSize) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int warpID = bid * 4 + (tid >> 5);
  int laneID = tid & 31;
  if (warpID >= currentQueueSize)
    return;
  int node = currentQueue[warpID]; // 当前线程处理的节点
  int nodeDistance = distance[node];
  int start = edgesOffset[node]; // 遍历该节点的邻居
  int end = start + edgesSize[node];
  for (int i = start + laneID; i < end; i += warpSize) {
    int neighbor = adjacencyList[i];
    if (atomicCAS(&distance[neighbor], INT_MAX, nodeDistance + 1) == INT_MAX) {
      parent[neighbor] = node; // 设置邻居的父节点
      int index = atomicAdd(nextQueueSize, 1);
      nextQueue[index] = neighbor;
    }
  }
}
}
