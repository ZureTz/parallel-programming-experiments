#include <float.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

// #define __USE_POSIX199506
#include "kmeans.h"

extern double wtime(void);
extern int num_omp_threads;

/*----< serial_clustering() >---------------------------------------------*/

// 计算欧几里得距离
float euclidean_distance(float *point1, float *point2, int nfeatures) {
  float sum = 0.0;
  for (int i = 0; i < nfeatures; i++) {
    sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
  }
  return sum;
}

// K-means 聚类算法 (串行版本)
float **serial_clustering(float **feature, // 输入点位置
                          int nfeatures,   // 数据点的维度
                          int npoints,     // 数据点数量
                          int nclusters,   // 簇数量
                          float threshold, // 停止条件
                          int *membership) // 输出: 每个数据点所属的簇
{
  float **clusters;          // 簇中心: [nclusters][nfeatures]
  int *new_centers_len;      // 每个簇中的成员数量
  float **new_centers;       // 新簇的坐标
  int delta = threshold + 1; // 记录重新分配的点数量
  int iter_threshold = npoints / 20;
  int iter = 0; // 迭代计数器

  clusters = (float **)malloc(nclusters * sizeof(float *));
  clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
  new_centers_len =
      (int *)calloc(nclusters, sizeof(int)); // 使用calloc自动初始化为0
  new_centers = (float **)malloc(nclusters * sizeof(float *));
  new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));

  for (int i = 1; i < nclusters; i++) {
    clusters[i] = clusters[i - 1] + nfeatures;
    new_centers[i] = new_centers[i - 1] + nfeatures;
  }

  // int *used = (int *)calloc(npoints, sizeof(int)); // 避免出现重复
  for (int i = 0; i < nclusters; i++) {
    // int n = rand() % npoints;
    int n = i;
    // while (used[n])
    //   n = rand() % npoints;

    // used[n] = 1; // 标记 n 已被选过
    for (int j = 0; j < nfeatures; j++)
      clusters[i][j] = feature[n][j]; // 从特征数组中复制数据点
  }

  // 初始化 membership 数组，表示每个点当前所属的簇
  for (int i = 0; i < npoints; i++)
    membership[i] = -1;
  while (delta > threshold && iter < iter_threshold) {
    delta = 0;
    for (int i = 0; i < npoints; i++) {
      int nearest_cluster = 0;
      float min_dist = FLT_MAX;
      for (int k = 0; k < nclusters; k++) // 找到离当前数据点最近的簇
      {
        float dist = euclidean_distance(feature[i], clusters[k], nfeatures);
        if (dist < min_dist) {
          min_dist = dist;
          nearest_cluster = k;
        }
      }

      if (membership[i] != nearest_cluster) // 如果点所属的簇改变了，更新 delta
      {
        delta++;
        membership[i] = nearest_cluster;
      }

      // 累加新簇的中心值
      new_centers_len[nearest_cluster]++;
      for (int j = 0; j < nfeatures; j++)
        new_centers[nearest_cluster][j] += feature[i][j];
    }

    // 更新簇中心
    for (int k = 0; k < nclusters; k++) {
      if (new_centers_len[k] > 0) {
        for (int j = 0; j < nfeatures; j++) {
          clusters[k][j] = new_centers[k][j] / new_centers_len[k];
          new_centers[k][j] = 0.0;
        }
        new_centers_len[k] = 0;
      }
    }
    iter++;
  }
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
  return clusters;
}

/*----< parallel_clustering() >---------------------------------------------*/
float **parallel_clustering(float **feature, /* in: [npoints][nfeatures] */
                            int nfeatures, int npoints, int nclusters,
                            float threshold,
                            int *membership) /* out: [npoints] */
{
  float **clusters;     // 簇中心: [nclusters][nfeatures]
  int *new_centers_len; // 每个簇中的成员数量
  float **new_centers;  // 新簇的坐标
  int delta = threshold + 1;
  int iter_threshold = npoints / 20;
  int iter = 0;
  omp_set_num_threads(num_omp_threads);
  // 分配内存
  clusters = (float **)malloc(nclusters * sizeof(float *));
  clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
  new_centers_len = (int *)calloc(nclusters, sizeof(int));
  new_centers = (float **)malloc(nclusters * sizeof(float *));
  new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
  int *used = (int *)calloc(npoints, sizeof(int));
  for (int i = 1; i < nclusters; i++) {
    clusters[i] = clusters[i - 1] + nfeatures;
    new_centers[i] = new_centers[i - 1] + nfeatures;
  }
  // 用于标记已选簇中心
  int n = 0, seed = time(NULL);
#pragma omp parallel shared(clusters, feature, used, nclusters) private(n, seed)
  {
// 每个线程使用不同的随机种子
#pragma omp for schedule(static)
    for (int i = 0; i < nclusters; i++) {
      // #pragma omp critical // 确保选点和更新 used 是原子操作
      // {
      n = i;
      // n = rand_r(&seed) % npoints;
      // while (used[n])
      //   n = rand_r(&seed) % npoints;
      // used[n] = 1; // 标记 n 已被选过
      // }
      for (int j = 0; j < nfeatures; j++)
        clusters[i][j] = feature[n][j]; // 选定初始簇中心
    }
  }
  free(used);

  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < npoints; i++)
    membership[i] = -1; // 初始化 membership 数组

  for (iter = 0; iter < iter_threshold && delta > threshold; iter++) {
    delta = 0;
#pragma omp parallel
    {
      int local_delta = 0;
      int *local_new_centers_len = (int *)calloc(nclusters, sizeof(int));
      float **local_new_centers = (float **)malloc(nclusters * sizeof(float *));
      local_new_centers[0] =
          (float *)calloc(nclusters * nfeatures, sizeof(float));
      for (int i = 1; i < nclusters; i++)
        local_new_centers[i] = local_new_centers[i - 1] + nfeatures;

#pragma omp for schedule(static) reduction(+ : delta)
      for (int i = 0; i < npoints; i++) {
        int nearest_cluster = 0;
        float min_dist = FLT_MAX;

        // 找到最近的簇
        for (int k = 0; k < nclusters; k++) {
          float dist = euclidean_distance(feature[i], clusters[k], nfeatures);
          if (dist < min_dist) {
            min_dist = dist;
            nearest_cluster = k;
          }
        }

        if (membership[i] != nearest_cluster) {
          local_delta++;
          membership[i] = nearest_cluster;
        }

        // 累加新簇的中心值
        local_new_centers_len[nearest_cluster]++;
        for (int j = 0; j < nfeatures; j++)
          local_new_centers[nearest_cluster][j] += feature[i][j];
      }

// 合并局部结果
#pragma omp critical
      {
        for (int k = 0; k < nclusters; k++) {
          new_centers_len[k] += local_new_centers_len[k];
          for (int j = 0; j < nfeatures; j++)
            new_centers[k][j] += local_new_centers[k][j];
        }
        delta += local_delta;
      }

      free(local_new_centers[0]);
      free(local_new_centers);
      free(local_new_centers_len);
#pragma omp barrier
    }

#pragma omp parallel for schedule(static)                                      \
    shared(nclusters, new_centers_len, clusters, nfeatures, new_centers)
    for (int k = 0; k < nclusters; k++) {
      if (new_centers_len[k] > 0) {
        for (int j = 0; j < nfeatures; j++) {
          clusters[k][j] = new_centers[k][j] / new_centers_len[k];
          new_centers[k][j] = 0.0;
        }
        new_centers_len[k] = 0;
      }
    }
  }
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
  return clusters;
}
