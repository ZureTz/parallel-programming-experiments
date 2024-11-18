#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <pthread.h>

#include "Sorters.hh"

void partialShellSort(uint64_t *array, int begin, int end) {
  const int partialSize = end - begin;
  for (int gap = partialSize / 2; gap > 0; gap /= 2) {
    for (int i = begin + gap; i < end; i++) {
      uint64_t temp = array[i];
      int j;
      for (j = i; j >= begin + gap && array[j - gap] > temp; j -= gap) {
        // move back elements
        array[j] = array[j - gap];
      }
      array[j] = temp;
    }
  }
}
void ShellSorter::sort(uint64_t *array, int array_size) {
  // for (int gap = array_size / 2; gap > 0; gap /= 2) {
  //   for (int i = gap; i < array_size; i++) {
  //     uint64_t temp = array[i];
  //     int j;
  //     for (j = i; j >= gap && array[j - gap] > temp; j -= gap) {
  //       // move back elements
  //       array[j] = array[j - gap];
  //     }
  //     array[j] = temp;
  //   }
  // }
  partialShellSort(array, 0, array_size);
}

// Barrier, used to wait per thread to finish their separated Shell sort
pthread_barrier_t barrier;

void ParallelShellSorter::sort(uint64_t *array, int array_size) {
  // Init
  pthread_t *threads = new pthread_t[this->m_nthreads];
  // Init barrier
  pthread_barrier_init(&barrier, NULL, this->m_nthreads);
  // make threads running
  for (int i = 0; i < this->m_nthreads; i++) {
    // Init arguments
    pthread_create(
        &threads[i], NULL, this->thread_create_helper,
        (void *)new ParallelShellSorterArgs(this, i, array, array_size));
  }
  // Wait until finished
  for (int i = 0; i < this->m_nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  delete[] threads;
}

void *ParallelShellSorter::thread_body(void *arg) {
  // Preparation

  // Edit current tid
  ParallelShellSorterArgs *a = (ParallelShellSorterArgs *)arg;
  const int threadOrderID = a->tid;
  uint64_t *const array = a->array;
  const int arraySize = a->arraySize;
  // free memory
  delete a;

  // Get range from threadOrderID
  const int partialLength = arraySize / this->m_nthreads;
  const int begin = threadOrderID * partialLength;
  const int end =
      threadOrderID == this->m_nthreads - 1 ? arraySize : begin + partialLength;

  // Execute partial sort per thread (no mutex lock required)
  partialShellSort(array, begin, end);

  // Wait for all other threads to finish
  pthread_barrier_wait(&barrier);
  printf("Thread %d finished! Ranged from %d to %d\n", threadOrderID, begin,
         end);

  return NULL;
}
