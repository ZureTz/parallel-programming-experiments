#include "bfsCPU.h"

void bfsCPU(int start, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
  std::queue<int> Q;
  distance[start] = 0;
  parent[start] = -1;
  visited[start] = true;
  Q.push(start); // 将起始顶点加入队列
  while (!Q.empty()) {
    int v = Q.front();
    Q.pop();
    // 遍历顶点v的所有相邻顶点
    for (int i = G.edgesOffset[v]; i < G.edgesOffset[v] + G.edgesSize[v]; ++i) {
      int w = G.adjacencyList[i];
      // 如果w没有被访问过
      if (!visited[w]) {
        visited[w] = true;
        distance[w] = distance[v] + 1; // 更新距离
        parent[w] = v;                 // 设置父节点
        Q.push(w);                   // 将顶点加入队列
      }
    }
  }
}
