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

// Must make sure it is the same array (must be continious)
void merge(uint64_t *array, int begin, int mid, int end) {}

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

// These two are for condition wait
pthread_mutex_t condMutex;
pthread_cond_t cond;

void ParallelShellSorter::sort(uint64_t *array, int array_size) {
  // Init
  pthread_t *threads = new pthread_t[this->m_nthreads];

  // Barrier count includes the main thread itself
  pthread_barrier_init(&barrier, NULL, this->m_nthreads + 1);

  // Running conditions shows the threads that can run in Merge process
  // if can run, runConditions[tid] == true
  // initialize too all false
  int *runConditions = new int[this->m_nthreads]();
  // mutex and cond

  // make threads running
  for (int i = 0; i < this->m_nthreads; i++) {
    // Init arguments
    pthread_create(&threads[i], NULL, this->thread_create_helper,
                   (void *)new ParallelShellSorterArgs(this, i, runConditions,
                                                       array, array_size));
  }
  // Wait for all of them pass the barrier
  pthread_barrier_wait(&barrier);
  printf("All threads finished sorting!");

  // Init conditions
  pthread_mutex_init(&condMutex, NULL);
  pthread_cond_init(&cond, NULL);

  // indicator *= 2 until reaching numOfThreads, exact same times as merge
  // operations
  for (int indicator = 2; indicator < this->m_nthreads; indicator *= 2) {
    // set barrier thread count
    pthread_barrier_init(&barrier, NULL, this->m_nthreads / indicator + 1 + 1);
    pthread_mutex_lock(&condMutex);
    // To let specific thread run
    for (int i = 0; i < this->m_nthreads; i += indicator) {
      runConditions[i] = 1;
    }
    // Broadcast Signal
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&condMutex);
    // Wait for all to finish
    pthread_barrier_wait(&barrier);
  }

  // Merge complete, singal all threads to stop
  pthread_mutex_lock(&condMutex);
  // To let specific thread run
  for (int i = 0; i < this->m_nthreads; i++) {
    runConditions[i] = 2;
  }
  // Broadcast Signal
  pthread_cond_broadcast(&cond);
  pthread_mutex_unlock(&condMutex);

  // Wait until finished
  for (int i = 0; i < this->m_nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  delete[] runConditions;
  delete[] threads;
}

void *ParallelShellSorter::thread_body(void *arg) {
  // Preparation

  // Edit current tid
  ParallelShellSorterArgs *a = (ParallelShellSorterArgs *)arg;
  const int threadOrderID = a->tid;
  int *const runConditions = a->runConditions;
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
  printf("Thread %d waiting!\n", threadOrderID);
  pthread_barrier_wait(&barrier);
  printf("Thread %d passed barrier! Ranged from %d to %d\n", threadOrderID,
         begin, end);

  // Begin condition wait, as specific thread needs to wake up
  int currentSegmentLength = partialLength;
  while (runConditions[threadOrderID] != 2) {
    pthread_mutex_lock(&condMutex);
    while (runConditions[threadOrderID] == 0) {
      pthread_cond_wait(&cond, &condMutex);
    }
    pthread_mutex_unlock(&condMutex);
    // Last segment
    if (begin + currentSegmentLength >= arraySize) {
      // skip
      currentSegmentLength *= 2;
      pthread_barrier_wait(&barrier);
      continue;
    }
    // Do merge in specific area
    merge(array, begin, begin + currentSegmentLength,
          begin + currentSegmentLength * 2 > arraySize
              ? arraySize
              : begin + currentSegmentLength * 2);
    currentSegmentLength *= 2;
    // Then Barrier Wait
    pthread_barrier_wait(&barrier);
  }

  return NULL;
}
