#include "Sorters.hh"

void ShellSorter::sort(uint64_t *array, int array_size) {
  for (int gap = array_size / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < array_size; i++) {
      uint64_t temp = array[i];
      int j;
      for (j = i; j >= gap && array[j - gap] > temp; j -= gap) {
        array[j] = array[j - gap]; // 后移元素
      }
      array[j] = temp;
    }
  }
}

void ParallelShellSorter::sort(uint64_t *array, int array_size) {
  
}
