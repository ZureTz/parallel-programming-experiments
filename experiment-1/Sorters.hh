#ifndef SORTERS
#define SORTERS

#include <cstddef>
#include <cstdint>
#include <pthread.h>
#include <stdint.h>

class Sorter;

class SorterArgs {
public:
  SorterArgs(Sorter *s) : m_this(s) {}

  Sorter *m_this;
};

class Sorter {
public:
  virtual void sort(uint64_t *array, int array_size) {}

protected:
  virtual void *thread_body(void *args) { return NULL; }

  static void *thread_create_helper(void *args) {
    SorterArgs *a = (SorterArgs *)args;
    return a->m_this->thread_body(args);
  }
};

class ShellSorter : public Sorter {
public:
  ShellSorter() {}

  void sort(uint64_t *array, int array_size) override;
};

class ParallelShellSorter : public Sorter {
public:
  ParallelShellSorter(int nthreads) { m_nthreads = nthreads; }

  void sort(uint64_t *array, int array_size) override;

private:
  void *thread_body(void *arg) override;

private:
  int m_nthreads;
};

class ParallelShellSorterArgs : public SorterArgs {
public:
  ParallelShellSorterArgs(ParallelShellSorter *s, int tid, uint64_t *array,
                          int arraySize, int mergeBegin = 0, int mergeMid = 0,
                          int mergeEnd = 0)
      : SorterArgs(s), tid(tid), array(array), arraySize(arraySize),
        mergeBegin(mergeBegin), mergeMid(mergeMid), mergeEnd(mergeEnd) {}

  int tid;

  uint64_t *array;
  int arraySize;

  int mergeBegin, mergeMid, mergeEnd;
};

class RadixSorter : public Sorter {
public:
  RadixSorter() {}

  void sort(uint64_t *array, int array_size) override;
};

class ParallelRadixSorter : public Sorter {
public:
  ParallelRadixSorter(int nthreads) : m_nthreads(nthreads) {}
  void sort(uint64_t *array, int array_size) override;

private:
  void *thread_body(void *arg) override;

private:
  int m_nthreads;
};

class ParallelRadixSorterArgs : public SorterArgs {
public:
  ParallelRadixSorterArgs(ParallelRadixSorter *s, int _tid)
      : SorterArgs(s), tid(_tid) {}

  int tid;
};

#endif
