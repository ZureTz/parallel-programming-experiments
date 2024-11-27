#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>

#include "Sorters.hh"

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

void partialCountingSort(uint64_t *array, int begin, int end, uint64_t exp) {
  const int arraySegmentSize = end - begin;
  std::vector<uint64_t> output(arraySegmentSize); // 输出数组
  int count[10] = {0}; // 计数数组，用于统计每一位的数字出现的次数

  // 统计每一位的出现次数
  for (int i = begin; i < end; i++) {
    int digit = (array[i] / exp) % 10;
    count[digit]++;
  }

  // 将计数数组转化为实际的索引位置
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  // 从右到左遍历原数组，按照当前位的数字将元素放入输出数组
  for (int i = end - 1; i >= begin; i--) {
    int digit = (array[i] / exp) % 10;
    output[count[digit] - 1] = array[i];
    count[digit]--;
  }

  // 将排序后的结果拷贝回原数组
  for (int i = 0; i < arraySegmentSize; i++) {
    array[begin + i] = output[i];
  }
}

void partialRadixSort(uint64_t *array, int begin, int end) {
  // 找出数组中最大数的位数
  uint64_t max_value = *std::max_element(array + begin, array + end);
  // 从最低有效位开始进行排序，直到最高有效位
  const uint64_t max_exp = 10000000000000000000uLL;
  uint64_t exp = 0;
  for (exp = 1; max_value / exp > 0 && exp != max_exp; exp *= 10) {
    // printf("%" PRIu64 " "
    //        "%" PRIu64 " "
    //        "%" PRIu64 " %d\n",
    //        max_value, exp, max_value / exp, int((max_value / exp) % 10));
    partialCountingSort(array, begin, end, exp);
  }
  if (exp == max_exp) {
    partialCountingSort(array, begin, end, max_exp);
  }
}

void RadixSorter::sort(uint64_t *array, int array_size) {
  partialRadixSort(array, 0, array_size);
}

// 计数排序的实现，针对某一位（exp）进行排序
void RadixSorter::counting_sort(uint64_t *array, int array_size, uint64_t exp) {
  std::vector<uint64_t> output(array_size); // 输出数组
  int count[10] = {0}; // 计数数组，用于统计每一位的数字出现的次数

  // 统计每一位的出现次数
  for (int i = 0; i < array_size; i++) {
    int digit = (array[i] / exp) % 10;
    count[digit]++;
  }

  // 将计数数组转化为实际的索引位置
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  // 从右到左遍历原数组，按照当前位的数字将元素放入输出数组
  for (int i = array_size - 1; i >= 0; i--) {
    int digit = (array[i] / exp) % 10;
    output[count[digit] - 1] = array[i];
    count[digit]--;
  }

  // 将排序后的结果拷贝回原数组
  for (int i = 0; i < array_size; i++) {
    array[i] = output[i];
  }
}

// Barrier, used to wait per thread to finish their separated Shell sort
pthread_barrier_t barrier;

void ParallelRadixSorter::sort(uint64_t *array, int array_size) {
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
        (void *)new ParallelRadixSorterArgs(this, i, array, array_size));
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
                     (void *)new ParallelRadixSorterArgs(
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

void *ParallelRadixSorter::thread_body(void *arg) { // Preparation

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

  partialRadixSort(array, begin, end);

  // Wait for all other threads to finish
  // printf("Thread %d sort finished, waiting!\n", threadOrderID);
  pthread_barrier_wait(&barrier);
  // printf("Thread %d passed barrier! Ranged from %d to %d\n", threadOrderID,
  //        begin, end);

  return NULL;
}
