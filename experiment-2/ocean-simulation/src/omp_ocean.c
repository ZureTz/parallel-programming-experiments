#include <omp.h>

void ocean(int **grid, int dim, int timesteps, int threads) {
  omp_set_num_threads(threads);
  for (int t = 0; t < timesteps; t++) {
    if (t % 2 == 0) {
#pragma omp parallel for schedule(dynamic) shared(grid, dim)
      for (int i = 1; i < dim - 1; i++) {
        for (int j = (i + 1) % 2 + 1; j < dim - 1; j += 2)
          grid[i][j] = (grid[i][j] + grid[i - 1][j] + grid[i + 1][j] +
                        grid[i][j - 1] + grid[i][j + 1]) /
                       5;
      }
    } else {
#pragma omp parallel for schedule(dynamic) shared(grid, dim)
      for (int i = 1; i < dim - 1; i++) {
        for (int j = i % 2 + 1; j < dim - 1; j += 2)
          grid[i][j] = (grid[i][j] + grid[i - 1][j] + grid[i + 1][j] +
                        grid[i][j - 1] + grid[i][j + 1]) /
                       5;
      }
    }
  }
}
