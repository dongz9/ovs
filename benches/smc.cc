#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "absl/container/flat_hash_map.h"

static constexpr size_t kSMCEntriesPerBucket = 8;
static constexpr size_t kSMCEntries = 1 << 20;
static constexpr size_t kSMCBucketCount = kSMCEntries / kSMCEntriesPerBucket;
static constexpr size_t kSMCMask = kSMCBucketCount - 1;
static constexpr size_t kSMCSignatureBits = 16;
static constexpr size_t kSMCProbes = 4;

struct SMCBucket {
  uint16_t sig[kSMCEntriesPerBucket];
  uint16_t v[kSMCEntriesPerBucket];
};

struct SMC {
  SMCBucket buckets[kSMCBucketCount];
};

void smc_init(SMC *smc) {
  for (size_t i = 0; i < kSMCBucketCount; ++i) {
    for (size_t j = 0; j < kSMCEntriesPerBucket; ++j) {
      smc->buckets[i].v[j] = UINT16_MAX;
    }
  }
}

size_t n_evict = 0;

static inline uint64_t fibonacci_hash(uint64_t hash) {
  return hash * 11400714819323198485llu;
}

void smc_insert(SMC *smc, uint32_t hash, uint16_t v) {
  size_t bucket_idx = hash & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  for (size_t probe = 0; probe < kSMCProbes; ++probe) {
    SMCBucket *bucket = &smc->buckets[bucket_idx];
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->sig[i] == sig) {
        bucket->v[i] = v;
        return;
      }
    }
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->v[i] == UINT16_MAX) {
        bucket->sig[i] = sig;
        bucket->v[i] = v;
        return;
      }
    }
    bucket_idx = (bucket_idx + probe + 1) & kSMCMask;
  }

  SMCBucket *bucket = &smc->buckets[hash & kSMCMask];
  size_t evict = rand() % kSMCEntriesPerBucket;
  bucket->sig[evict] = sig;
  bucket->v[evict] = v;
  ++n_evict;
}

uint16_t smc_lookup(SMC *smc, uint32_t hash) {
  size_t bucket_idx = hash & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  for (size_t probe = 0; probe < kSMCProbes; ++probe) {
    SMCBucket *bucket = &smc->buckets[bucket_idx];
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->sig[i] == sig) {
        return bucket->v[i];
      }
    }
    bucket_idx = (bucket_idx + probe + 1) & kSMCMask;
  }
  return UINT16_MAX;
}

inline uint64_t rdtsc(void) {
  uint32_t lo, hi;
  __asm__ __volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

static uint64_t get_clock_frequency() {
#ifdef CLOCK_MONOTONIC_RAW
#define NS_PER_SEC (1e9)

  struct timespec sleeptime;
  sleeptime.tv_sec = 0;
  sleeptime.tv_nsec = 5e8;
  struct timespec t_start, t_end;

  if (clock_gettime(CLOCK_MONOTONIC_RAW, &t_start) == 0) {
    uint64_t ns, end, start = rdtsc();
    nanosleep(&sleeptime, NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
    end = rdtsc();
    ns = (t_end.tv_sec - t_start.tv_sec) * NS_PER_SEC +
         (t_end.tv_nsec - t_start.tv_nsec);
    return (uint64_t)(end - start) / ((double)ns / NS_PER_SEC);
  }
#endif
  return 0;
}

static constexpr size_t kNumKeys = 1000000;
static constexpr size_t kNumQueries = 50000000;

SMC smc __attribute__((aligned(64)));
uint32_t keys[kNumKeys];
uint16_t values[kNumKeys];

int main() {
  srand(time(0));

  for (size_t i = 0; i < kNumKeys; ++i) {
    keys[i] = (uint32_t)rand();
    values[i] = (uint16_t)(rand() & 0xFFFE);
  }

  {
    smc_init(&smc);
    uint64_t start = rdtsc();
    size_t n_miss = 0, n_collision = 0;
    for (size_t i = 0; i < kNumQueries; ++i) {
      size_t idx = rand() % kNumKeys;
      uint16_t v = smc_lookup(&smc, keys[idx]);
      if (v == UINT16_MAX) {
        smc_insert(&smc, keys[idx], values[idx]);
        ++n_miss;
      } else if (v != values[idx]) {
        smc_insert(&smc, keys[idx], values[idx]);
        ++n_collision;
      }
    }
    uint64_t end = rdtsc();

    printf("[SMC] hit rate: %.2lf, miss: %lu, collision: %lu, evict: %lu, "
           "throughput: %.2lf "
           "Mpps\n",
           (double)(kNumQueries - n_miss - n_collision) / kNumQueries, n_miss,
           n_collision, n_evict,
           kNumQueries / ((double)(end - start) / get_clock_frequency()) /
               1000000);
  }

  // {
  //   absl::flat_hash_map<uint32_t, uint16_t> m;
  //   uint64_t start = rdtsc();
  //   size_t n_miss = 0;
  //   for (size_t i = 0; i < kNumQueries; ++i) {
  //     size_t idx = rand() % kNumKeys;
  //     auto it = m.find(keys[idx]);
  //     if (it == m.end()) {
  //       m.emplace(keys[idx], values[idx]);
  //       ++n_miss;
  //     }
  //   }
  //   uint64_t end = rdtsc();

  //   printf("<SwissTable> hit rate: %.2lf, throughput: %.2lf Mpps\n",
  //          (double)(kNumQueries - n_miss) / kNumQueries,
  //          kNumQueries / ((double)(end - start) / get_clock_frequency()) /
  //              1000000);
  // }

  return 0;
}
