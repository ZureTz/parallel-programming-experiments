#ifndef TIMERS_H
#define TIMERS_H

#include <ctime>

typedef struct timespec hwtime_t;
constexpr long ONE_S_TO_NS = 1000000000l;

class HWTimer {
public:
  HWTimer() {}

  hwtime_t get_time_ns() const {
    hwtime_t currentTime;
    clock_gettime(CLOCK_REALTIME, &currentTime);
    return currentTime;
  }
};

inline hwtime_t operator-(const hwtime_t &t1, const hwtime_t &t2) {
  if (t1.tv_sec < t2.tv_sec) {
    const hwtime_t negResult = t2 - t1;
    return {.tv_sec = -negResult.tv_sec, .tv_nsec = -negResult.tv_nsec};
  }

  hwtime_t result = {.tv_sec = t1.tv_sec - t2.tv_sec,
                     .tv_nsec = t1.tv_nsec - t2.tv_nsec};
  if (result.tv_nsec < 0) {
    result.tv_sec -= 1;
    result.tv_nsec += ONE_S_TO_NS;
  }
  return result;
}

inline hwtime_t operator+(const hwtime_t &t1, const hwtime_t &t2) {
  hwtime_t result = {.tv_sec = t1.tv_sec + t2.tv_sec,
                     .tv_nsec = t1.tv_nsec + t2.tv_nsec};
  if (result.tv_nsec > ONE_S_TO_NS) {
    result.tv_sec += 1;
    result.tv_nsec -= ONE_S_TO_NS;
  }
  return result;
}

#endif
