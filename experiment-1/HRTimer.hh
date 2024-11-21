#ifndef HRTIMERS_H
#define HRTIMERS_H

#include <ctime>

typedef struct timespec hrtime_t;
constexpr long ONE_S_TO_NS = 1000000000l;

class HRTimer {
public:
  HRTimer() {}

  hrtime_t get_time_ns() const {
    hrtime_t currentTime;
    clock_gettime(CLOCK_REALTIME, &currentTime);
    return currentTime;
  }
};

inline hrtime_t operator-(const hrtime_t &t1, const hrtime_t &t2) {
  if (t1.tv_sec < t2.tv_sec) {
    const hrtime_t negResult = t2 - t1;
    return {.tv_sec = -negResult.tv_sec, .tv_nsec = -negResult.tv_nsec};
  }

  hrtime_t result = {.tv_sec = t1.tv_sec - t2.tv_sec,
                     .tv_nsec = t1.tv_nsec - t2.tv_nsec};
  if (result.tv_nsec < 0) {
    result.tv_sec -= 1;
    result.tv_nsec += ONE_S_TO_NS;
  }
  return result;
}

inline hrtime_t operator+(const hrtime_t &t1, const hrtime_t &t2) {
  hrtime_t result = {.tv_sec = t1.tv_sec + t2.tv_sec,
                     .tv_nsec = t1.tv_nsec + t2.tv_nsec};
  if (result.tv_nsec > ONE_S_TO_NS) {
    result.tv_sec += 1;
    result.tv_nsec -= ONE_S_TO_NS;
  }
  return result;
}

#endif
