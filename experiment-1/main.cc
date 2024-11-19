#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <unistd.h>

#include "HRTimer.hh"
#include "Sorters.hh"

void print_usage(const char *prog_name) {
#ifdef PARALLEL
  printf("Usage:  %s -s <array size> -n <num threads> [ -r <seed> ]\n",
         prog_name);
#else
  printf("Usage:  %s -s <array size> [ -r <seed> ]\n", prog_name);
#endif
}

int main(int argc, char *const argv[]) {

  int array_size = 0;
  int nthreads = 0;

  // Default seed is based on time
  unsigned int seed = time(NULL);

  // Read arguments
#ifdef PARALLEL
  const char *shortopts = "n:s:r:h";
#else
  const char *shortopts = "s:r:h";
#endif

  char c;
  while ((c = getopt(argc, argv, shortopts)) != -1) {
    switch (c) {
#ifdef PARALLEL
    case 'n':
      nthreads = atoi(optarg);
      break;
#endif
    case 'h':
      print_usage(argv[0]);
      return 0;
    case 'r':
      seed = atoi(optarg);
      break;
    case 's':
      array_size = atoi(optarg);
      break;
    case '?':
      printf("Unrecognized option %c\n", optopt);
      print_usage(argv[0]);
      return 1;
    }
  }

  if (array_size <= 0) {
    fprintf(stderr, "Invalid array size %d. The size must be bigger than 0.\n",
            array_size);
    print_usage(argv[0]);
    return 1;
  }

#ifdef PARALLEL
  if (nthreads <= 0) {
    fprintf(stderr, "Missing required option -n\n");
    print_usage(argv[0]);
    return 1;
  }

  if (array_size < nthreads) {
    fprintf(stderr,
            "Array size (%d) cannot be smaller than thread numbers (%d).\n",
            array_size, nthreads);
    print_usage(argv[0]);
    return 1;
  }

#endif

  // Create array and then fill with random numbers

  uint64_t *array = new uint64_t[array_size];

  srand(seed);

  for (int i = 0; i < array_size; i++) {
    // Ranges from 0 to 2 ^ 64
    array[i] = (rand() & 0xffff);
    array[i] |= (rand() & 0xffff) << 16;
    array[i] |= static_cast<uint64_t>(rand() & 0xffff) << 32;
    array[i] |= static_cast<uint64_t>(rand() & 0xffff) << 48;
  }

  HRTimer timer;
  hrtime_t start, end;

#ifdef PARALLEL

#ifdef SHELL
  ParallelShellSorter sorter(nthreads);
#else
  ParallelRadixSorter sorter(nthreads);
#endif

#else

#ifdef SHELL
  ShellSorter sorter;
#else
  RadixSorter sorter;
#endif

#endif

  start = timer.get_time_ns();
  sorter.sort(array, array_size);
  end = timer.get_time_ns();

  // Check

  // for (int i = 0; i < array_size; i++) {
  //   printf("%lu ", array[i]);
  // }
  // putchar('\n');

  for (int i = 1; i < array_size; i++) {
    if (array[i - 1] > array[i]) {
      fprintf(stderr, "ERROR: Sort failed\n");
      return 1;
    }
  }

  printf("Sort time: %llu nanoseconds\n", end - start);

  // Free memory
  delete[] array;

  return 0;
}
