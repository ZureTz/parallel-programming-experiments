void ocean(int **grid, int dim, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    if (t % 2 == 0) {
      for (int i = 1; i < dim - 1; i++) {
        int j = (i + 1) % 2 + 1;
        for (; j < dim - 1; j += 2)
          grid[i][j] = (grid[i][j] + grid[i - 1][j] + grid[i + 1][j] +
                        grid[i][j - 1] + grid[i][j + 1]) /
                       5;
      }
    } else {
      for (int i = 1; i < dim - 1; i++) {
        int j = i % 2 + 1;
        for (; j < dim - 1; j += 2)
          grid[i][j] = (grid[i][j] + grid[i - 1][j] + grid[i + 1][j] +
                        grid[i][j - 1] + grid[i][j + 1]) /
                       5;
      }
    }
  }
}
