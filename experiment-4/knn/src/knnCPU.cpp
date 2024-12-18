#include <cfloat>
#include <cmath> // 如果需要 sqrt 可以包含此头文件
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "base.h"

typedef struct {
  float distance;
  int classe;
} Neighbor;

void knnSerial(float *coords, float *newCoords, int *classes, int numClasses,
               int numSamples, int numNewSamples, int k) {
  // 提前分配内存，避免重复分配
  float *distances = (float *)malloc(numSamples * sizeof(float));
  Neighbor *kNearest = (Neighbor *)malloc(k * sizeof(Neighbor));
  int *classCounts = (int *)malloc(numClasses * sizeof(int));

  for (int i = 0; i < numNewSamples; i++) {
    for (int j = 0; j < numSamples; j++) {
      float distance = 0.0;
      for (int d = 0; d < DIMENSION; d++) {
        float diff = newCoords[i * DIMENSION + d] - coords[j * DIMENSION + d];
        distance += diff * diff;
      }
      distances[j] = distance; // 保存距离
    }
    for (int j = 0; j < k; j++) {
      kNearest[j].distance = FLT_MAX;
      kNearest[j].classe = -1;
    }

    for (int j = 0; j < numSamples; j++) {
      if (distances[j] < kNearest[k - 1].distance) {
        kNearest[k - 1].distance = distances[j];
        kNearest[k - 1].classe = classes[j];
        for (int m = k - 1;
             m > 0 && kNearest[m].distance < kNearest[m - 1].distance; m--) {
          Neighbor temp = kNearest[m];
          kNearest[m] = kNearest[m - 1];
          kNearest[m - 1] = temp;
        }
      }
    }

    // 统计每个类别的出现次数
    memset(classCounts, 0, numClasses * sizeof(int)); // 清空类别计数
    for (int j = 0; j < k; j++) {
      if (kNearest[j].classe != -1)
        classCounts[kNearest[j].classe]++;
    }

    int maxCount = 0;
    int predictedClass = 0;
    for (int c = 0; c < numClasses; c++) {
      if (classCounts[c] > maxCount) {
        maxCount = classCounts[c];
        predictedClass = c;
      }
    }
    classes[numSamples + i] = predictedClass; // 保存预测的类别
  }

  // 释放动态分配的内存
  free(distances);
  free(kNearest);
  free(classCounts);
}

// void knnSerial(float* coords, float* newCoords, int* classes, int numClasses,
// int numSamples, int numNewSamples, int k)
// {
//     float* distances = (float*)malloc(sizeof(float) * numSamples);
//     int* neighborClasses = (int*)malloc(sizeof(int) * k);

//     for (int i = 0; i < numNewSamples; i++)
//     {
//         for (int j = 0; j < numSamples; j++)
//         {
//             float distance = 0.0;
//             for (int d = 0; d < DIMENSION; d++)
//             {
//                 float diff = newCoords[i * DIMENSION + d] - coords[j *
//                 DIMENSION + d]; distance += diff * diff;
//             }
//             distances[j] = sqrt(distance);      // 计算出欧几里得距离
//         }
//         for (int n = 0; n < k; n++)
//         {
//             float minDistance = FLT_MAX;
//             int minIndex = -1;
//             for (int j = 0; j < numSamples; j++)
//             {
//                 if (distances[j] < minDistance)
//                 {
//                     minDistance = distances[j];
//                     minIndex = j;
//                 }
//             }
//             neighborClasses[n] = classes[minIndex];
//             distances[minIndex] = FLT_MAX;  // 标记该样本已被选择
//         }
//         int* classCounts = (int*)calloc(numClasses, sizeof(int));
//         for (int n = 0; n < k; n++) {
//             classCounts[neighborClasses[n]]++;
//         }
//         int maxCount = 0;
//         int predictedClass = 0;
//         for (int c = 0; c < numClasses; c++) {
//             if (classCounts[c] > maxCount) {
//                 maxCount = classCounts[c];
//                 predictedClass = c;
//             }
//         }
//         classes[numSamples + i] = predictedClass;
//         free(classCounts);
//     }
//     free(distances);
//     free(neighborClasses);
// }