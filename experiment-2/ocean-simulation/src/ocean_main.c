#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hwtimer.h"

extern inline void initTimer(hwtimer_t *timer);

/* Implement this function in serial_ocean and omp_ocean */
#ifdef PARALLEL
extern void ocean(int **grid, int dim, int timesteps, int threads);
#else
extern void ocean(int **grid, int dim, int timesteps);
#endif

void usage(char *argv0) {
#ifdef PARALLEL
  char *help = "Usage: %s -d <dimension> -t <timesteps> -n <threads>\n";
#else
  char *help = "Usage: %s -d <dimension> -t <timesteps>\n";
#endif
  fprintf(stderr, help, argv0);
  exit(-1);
}

void printGrid(int **grid, int dim);

int main(int argc, char *argv[]) {
  // Init timer
  hwtimer_t timer;
  initTimer(&timer);

  /********************Get the arguments correctly (start)
   * **************************/
  /*
  Three input Arguments to the program
  1. X Dimension of the grid
  2. Y dimension of the grid
  3. number of timesteps the algorithm is to be performed
  */
  int opt;
  extern char *optarg;
  int dim = 0, timesteps = 0, threads = 0;
#ifdef PARALLEL
  while ((opt = getopt(argc, argv, "d:t:n:?")) != EOF) {
    switch (opt) {
    case 'd':
      dim = atoi(optarg);
      break;
    case 't':
      timesteps = atoi(optarg);
      break;
    case 'n':
      threads = atoi(optarg);
      break;
    case '?':
      usage(argv[0]);
      break;
    default:
      usage(argv[0]);
      break;
    }
  }
  if ((dim == 0) || (timesteps == 0) || (threads == 0))
    usage(argv[0]);
#else
  while ((opt = getopt(argc, argv, "d:t:?")) != EOF) {
    switch (opt) {
    case 'd':
      dim = atoi(optarg);
      break;
    case 't':
      timesteps = atoi(optarg);
      break;
    case '?':
      usage(argv[0]);
      break;
    default:
      usage(argv[0]);
      break;
    }
  }
  if ((dim == 0) || (timesteps == 0))
    usage(argv[0]);
#endif
  ///////////////////////Get the arguments correctly (end)
  /////////////////////////////

  /*********************create the grid as required (start)
   * ************************/
  /*
  The grid needs to be allocated as per the input arguments and randomly
  initialized. Remember during allocation that we want to guarantee a contiguous
  block, hence the nasty pointer math.

  To test your code for correctness please comment this section of random
  initialization.
  */
  srand(1234);
  int **const grid = (int **)malloc(dim * sizeof(int *));
  int *temp = (int *)malloc(dim * dim * sizeof(int));
  for (int i = 0; i < dim; i++) {
    grid[i] = &temp[i * dim];
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      grid[i][j] = rand();
    }
  }
  ///////////////////////create the grid as required (end)
  /////////////////////////////

  // Start the time measurement here before the algorithm starts
  startTimer(&timer);

#ifdef PARALLEL
  ocean(grid, dim, timesteps, threads);
#else
  ocean(grid, dim, timesteps);
#endif

  stopTimer(&timer); // End the time measurement here since the algorithm ended

  const uint64_t runningTime = getTimerNs(&timer);

  FILE *dataFile = fopen("../data.txt", "a");
  if (dataFile == NULL) {
    printf("无法打开文件!\n");
    return 1;
  }

  // 按顺序写入 d, t, n, time 的值在同一行
  fprintf(dataFile, "%d %d %d %lu\n", dim, timesteps, threads, runningTime);

  // Do the time calculation
  printf("Total Execution time: %lu ns\n", runningTime);

  // Free the memory we allocated for grid
  free(temp);
  free(grid);

  return EXIT_SUCCESS;
}
