#ifndef TIMER_H
#define TIMER_H

#define _POSIX_C_SOURCE 200809L

#include <time.h>

const long ONE_S_TO_NS = 1000000000l;

struct timespec getRealTimeInNanoSeconds() {
  struct timespec currentTime;
  clock_gettime(CLOCK_REALTIME, &currentTime);
  return currentTime;
}

struct timespec subTime(const struct timespec t1, const struct timespec t2) {
  if (t1.tv_sec < t2.tv_sec) {
    const struct timespec negResult = subTime(t2, t1);
    return (struct timespec){.tv_sec = -negResult.tv_sec,
                             .tv_nsec = -negResult.tv_nsec};
  }

  struct timespec result = {.tv_sec = t1.tv_sec - t2.tv_sec,
                            .tv_nsec = t1.tv_nsec - t2.tv_nsec};
  if (result.tv_nsec < 0) {
    result.tv_sec -= 1;
    result.tv_nsec += ONE_S_TO_NS;
  }
  return result;
}

struct timespec addTime(const struct timespec t1, const struct timespec t2) {
  struct timespec result = {.tv_sec = t1.tv_sec + t2.tv_sec,
                            .tv_nsec = t1.tv_nsec + t2.tv_nsec};
  if (result.tv_nsec > ONE_S_TO_NS) {
    result.tv_sec += 1;
    result.tv_nsec -= ONE_S_TO_NS;
  }
  return result;
}

#endif
