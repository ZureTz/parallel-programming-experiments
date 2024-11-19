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

  for (int i = begin; i < end - 1; i++) {
    if (array[i] > array[i + 1]) {
      printf("Sort failed! begin %d end %d\n", begin, end);
      break;
    }
  }
}

// Must make sure it is the same array (must be continious)
void merge(uint64_t *array, int begin, int mid, int end) {
  if (mid == end) {
    return;
  }

  const int segmentLength = end - begin;
  uint64_t *temp = new uint64_t[segmentLength]();

  int index1 = begin, index2 = mid, indexTemp = 0;
  while (index1 != mid && index2 != end) {
    if (array[index1] < array[index2]) {
      temp[indexTemp++] = array[index1++];
      continue;
    }
    // array[index1] >= array[index2]
    temp[indexTemp++] = array[index2++];
  }

  // One of the array merge finished, copy another in
  int notFinishedIndex = index1 == mid ? index2 : index1;
  int notFinishedIndexEnd = index1 == mid ? end : mid;
  while (notFinishedIndex != notFinishedIndexEnd) {
    temp[indexTemp++] = array[notFinishedIndex++];
  }

  // Copy back
  for (int i = 0; i < segmentLength; i++) {
    array[begin + i] = temp[i];
  }

  for (int i = 0; i < segmentLength - 1; i++) {
    if (temp[i] > temp[i + 1]) {
      printf("Merge failed! begin %d mid %d end %d\n", begin, mid, end);
      for (int i = 0; i < segmentLength; i++) {
        printf("%lu ", temp[i]);
      }
      putchar('\n');

      break;
    }
  }

  // Free memory
  delete[] temp;
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

  // Barrier count includes the main thread itself
  pthread_barrier_init(&barrier, NULL, this->m_nthreads + 1);

  // make threads running
  for (int i = 0; i < this->m_nthreads; i++) {
    // Init arguments
    pthread_create(
        &threads[i], NULL, this->thread_create_helper,
        (void *)new ParallelShellSorterArgs(this, i, array, array_size));
  }
  // Wait for all of them pass the barrier
  pthread_barrier_wait(&barrier);
  printf("All threads finished sorting!\n");
  // Wait until finished
  for (int i = 0; i < this->m_nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  // indicator *= 2 until reaching numOfThreads, exact same times as merge
  // operations
  int indicator = 0, lastUnMergedIndex = 0;
  for (indicator = 2; indicator / 2 < this->m_nthreads; indicator *= 2) {
    const int currentSubThreadsShouldRun = this->m_nthreads / indicator;
    // set barrier to thread count
    // plus 1 means the main thread
    pthread_barrier_init(&barrier, NULL, currentSubThreadsShouldRun + 1);

    // const int limit = this->m_nthreads / indicator;
    // for (int i = 0; i < limit; i++) {
    //   const int stepSize = (array_size / this->m_nthreads) * indicator / 2;
    //   const int mergeBegin = i * stepSize * 2;
    //   const int mergeMid = mergeBegin + stepSize;
    //   const int mergeEnd = mergeMid + stepSize;
    //   lastUnMergedIndex =
    //       lastUnMergedIndex < mergeEnd ? mergeEnd : lastUnMergedIndex;

    //   // Init arguments
    //   pthread_create(&threads[i], NULL, this->thread_create_helper,
    //                  (void *)new ParallelShellSorterArgs(this, i, array,
    //                                                      array_size, mergeBegin,
    //                                                      mergeMid, mergeEnd));
    // }

    // // Wait for all to finish
    // pthread_barrier_wait(&barrier);

    // // Wait until finished
    // for (int i = 0; i < limit; i++) {
    //   pthread_join(threads[i], NULL);
    // } 

    for (int i = 0; i < array_size; i++) {
      printf("%lu ", array[i]);
    }
    putchar('\n');
    putchar('\n');
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
  const int mergeBegin = a->mergeBegin, mergeMid = a->mergeMid,
            mergeEnd = a->mergeEnd;
  // free memory
  delete a;

  // Execute partial merge per thread (no mutex lock required)
  if (mergeMid && mergeEnd) {
    printf("Thread %d doing merge from %d mid %d to end %d\n", threadOrderID,
           mergeBegin, mergeMid, mergeEnd);
    merge(array, mergeBegin, mergeMid, mergeEnd);
    // Then Barrier Wait
    printf("Thread %d merge finished. Waiting...\n", threadOrderID);
    pthread_barrier_wait(&barrier);

    return NULL;
  }

  // Execute partial sort per thread (no mutex lock required)

  // Get range from threadOrderID
  const int partialLength = arraySize / this->m_nthreads;
  const int begin = threadOrderID * partialLength;
  const int end =
      threadOrderID == this->m_nthreads - 1 ? arraySize : begin + partialLength;

  partialShellSort(array, begin, end);

  // Wait for all other threads to finish
  printf("Thread %d sort finished, waiting!\n", threadOrderID);
  pthread_barrier_wait(&barrier);
  printf("Thread %d passed barrier! Ranged from %d to %d\n", threadOrderID,
         begin, end);

  return NULL;
}
