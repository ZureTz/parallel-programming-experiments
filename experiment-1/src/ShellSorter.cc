#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>

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
void merge(uint64_t *array, int begin, int mid, int end) {
  const int segmentLength = end - begin;
  uint64_t *temp = new uint64_t[segmentLength];

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

  // Free memory
  delete[] temp;
}

void ShellSorter::sort(uint64_t *array, int array_size) {
  for (int gap = array_size / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < array_size; i++) {
      uint64_t temp = array[i];
      int j;
      for (j = i; j >= gap && array[j - gap] > temp; j -= gap) {
        // move back elements
        array[j] = array[j - gap];
      }
      array[j] = temp;
    }
  }
}

// Barrier, used to wait per thread to finish their separated Shell sort
pthread_barrier_t barrier;

void ParallelShellSorter::sort(uint64_t *array, int array_size) {
  // Init
  pthread_t *threads = new pthread_t[this->m_nthreads];

  // Barrier count includes the main thread itself
  pthread_barrier_init(&barrier, NULL, this->m_nthreads + 1);

  const int sortedArraySegmentLength = array_size / this->m_nthreads;
  const int sortedArraySegmentLengthRemainder = array_size % this->m_nthreads;

  struct arraySegmentInfo {
    int begin;
    int end;
  };
  std::deque<arraySegmentInfo> segmentInfos;

  // make threads running
  for (int i = 0; i < this->m_nthreads; i++) {
    segmentInfos.push_back(arraySegmentInfo{
        .begin = i * sortedArraySegmentLength,
        .end = i == this->m_nthreads - 1 ? array_size
                                         : (i + 1) * sortedArraySegmentLength});
    // Init arguments
    pthread_create(
        &threads[i], NULL, this->thread_create_helper,
        (void *)new ParallelShellSorterArgs(this, i, array, array_size));
  }
  // Wait for all of them pass the barrier
  pthread_barrier_wait(&barrier);
  // printf("All threads finished sorting!\n");
  // Wait until finished
  for (int i = 0; i < this->m_nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  if (this->m_nthreads == 1) {
    // no need to merge anymore
    return;
  }

  while (segmentInfos.size() > 1) {
    int currentRoundSize = segmentInfos.size();
    // printf("currentRoundSize %d\n", currentRoundSize);
    int threadIndexID = 0;
    while (currentRoundSize > 1) {
      const arraySegmentInfo segment1 = segmentInfos.front();
      segmentInfos.pop_front();
      const arraySegmentInfo segment2 = segmentInfos.front();
      segmentInfos.pop_front();

      segmentInfos.push_back(
          arraySegmentInfo{.begin = segment1.begin, .end = segment2.end});

      // Init arguments
      pthread_create(&threads[threadIndexID], NULL, this->thread_create_helper,
                     (void *)new ParallelShellSorterArgs(
                         this, threadIndexID, array, array_size, segment1.begin,
                         segment2.begin, segment2.end));
      currentRoundSize -= 2;
      threadIndexID++;
    }

    if (currentRoundSize == 1) {
      arraySegmentInfo temp = segmentInfos.front();
      segmentInfos.pop_front();
      segmentInfos.push_back(temp);
    }

    for (int i = 0; i < threadIndexID; i++) {
      pthread_join(threads[i], NULL);
    }
    // for (int i = 0; i < array_size; i++) {
    //   printf("%lu ", array[i]);
    // }
    // putchar('\n');
    // putchar('\n');
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
    // printf("Thread %d doing merge from %d mid %d to end %d\n", threadOrderID,
    //        mergeBegin, mergeMid, mergeEnd);
    merge(array, mergeBegin, mergeMid, mergeEnd);
    // printf("Thread %d merge finished.\n", threadOrderID);

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
  // printf("Thread %d sort finished, waiting!\n", threadOrderID);
  pthread_barrier_wait(&barrier);
  // printf("Thread %d passed barrier! Ranged from %d to %d\n", threadOrderID,
  //        begin, end);

  return NULL;
}
