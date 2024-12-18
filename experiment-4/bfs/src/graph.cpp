#include <cstdio>
#include <cstdlib>

#include "graph.h"

void readGraph(Graph &G, int argc, char **argv) {
  int m, n;
  // If no arguments then read graph from stdin
  bool fromStdin = argc <= 2;
  if (fromStdin) {
    if (scanf("%d %d", &n, &m) != 2) {
      fprintf(stderr, "Failed to scan! \n");
      exit(EXIT_FAILURE);
    }
  } else {
    srand(12345);
    n = atoi(argv[2]);
    m = atoi(argv[3]) / 2;
  }

  std::vector<std::vector<int>> adjacencyLists(n);
  for (int i = 0; i < m; i++) {
    int u, v;
    if (fromStdin) {
      if (scanf("%d %d", &u, &v) != 2) {
        fprintf(stderr, "Failed to scan! \n");
        exit(EXIT_FAILURE);
      }
      adjacencyLists[u].push_back(v);
    } else {
      u = rand() % n;
      v = rand() % n;
      adjacencyLists[u].push_back(v);
      adjacencyLists[v].push_back(u);
    }
  }

  for (int i = 0; i < n; i++) {
    G.edgesOffset.push_back(G.adjacencyList.size());
    G.edgesSize.push_back(adjacencyLists[i].size());
    for (auto &edge : adjacencyLists[i]) {
      G.adjacencyList.push_back(edge);
    }
  }

  G.numVertices = n;
  G.numEdges = G.adjacencyList.size();
}
