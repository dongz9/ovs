#include <emmintrin.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

static constexpr size_t kSMCEntriesPerBucket = 4;
static constexpr size_t kSMCEntries = 1 << 20;
static constexpr size_t kSMCBucketCount = kSMCEntries / kSMCEntriesPerBucket;
static constexpr size_t kSMCMask = kSMCBucketCount - 1;
static constexpr size_t kSMCSignatureBits = 16;
static constexpr size_t kSMCProbes = 1;

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

size_t n_inserts = 0, n_replaces = 0;
size_t n_evicts = 0;

void smc_insert(SMC *smc, uint32_t hash, uint16_t v) {
  size_t bucket_idx = hash & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  ++n_inserts;
  for (size_t probe = 0; probe < kSMCProbes; ++probe) {
    SMCBucket *bucket = &smc->buckets[bucket_idx];
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->sig[i] == sig) {
        ++n_replaces;
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
  ++n_evicts;
}

void smc_insert_wide(SMC *smc, uint32_t hash, uint16_t v) {
  size_t bucket_idx = hash & kSMCMask;
  size_t alt_bucket_idx = (bucket_idx + 1) & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  auto match = _mm_set1_epi16(sig);
  auto sigs =
      _mm_set_epi64x(*(__int64_t *)(&smc->buckets[alt_bucket_idx & kSMCMask]),
                     *(__int64_t *)(&smc->buckets[bucket_idx]));

  auto bitmask = _mm_movemask_epi8(_mm_cmpeq_epi16(match, sigs));

  ++n_inserts;
  if (likely(bitmask == 0)) {
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (smc->buckets[bucket_idx].v[i] == UINT16_MAX) {
        smc->buckets[bucket_idx].sig[i] = sig;
        smc->buckets[bucket_idx].v[i] = v;
        return;
      }
    }
    for (size_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (smc->buckets[alt_bucket_idx].v[i] == UINT16_MAX) {
        smc->buckets[alt_bucket_idx].sig[i] = sig;
        smc->buckets[alt_bucket_idx].v[i] = v;
        return;
      }
    }
    size_t evict = rand() % kSMCEntriesPerBucket;
    smc->buckets[bucket_idx].sig[evict] = sig;
    smc->buckets[bucket_idx].v[evict] = v;
    ++n_evicts;
  } else {
    int idx = __builtin_ctz(bitmask & (-bitmask)) / 2;
    ++n_replaces;
    if (idx < 4) {
      smc->buckets[bucket_idx].sig[idx] = sig;
      smc->buckets[bucket_idx].v[idx] = v;
    } else {
      smc->buckets[alt_bucket_idx].sig[idx] = sig;
      smc->buckets[alt_bucket_idx].v[idx] = v;
    }
  }
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

uint16_t smc_lookup_wide(SMC *smc, uint32_t hash) {
  size_t bucket_idx = hash & kSMCMask;
  size_t alt_bucket_idx = (bucket_idx + 1) & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  auto match = _mm_set1_epi16(sig);
  auto sigs =
      _mm_set_epi64x(*(__int64_t *)(&smc->buckets[alt_bucket_idx & kSMCMask]),
                     *(__int64_t *)(&smc->buckets[bucket_idx]));

  auto bitmask = _mm_movemask_epi8(_mm_cmpeq_epi16(match, sigs));
  if (bitmask == 0) return UINT16_MAX;

  int idx = __builtin_ctz((bitmask & (-bitmask))) / 2;
  return (smc->buckets[alt_bucket_idx].v[idx & 3]) * (idx >> 2) +
         (smc->buckets[bucket_idx].v[idx & 3] * (1 - (idx >> 2)));
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
    uint64_t insert = 0;
    size_t n_misses = 0, n_collisions = 0;
    for (size_t i = 0; i < kNumQueries; ++i) {
      size_t idx = rand() % kNumKeys;
      uint16_t v = smc_lookup(&smc, keys[idx]);
      if (v == UINT16_MAX) {
        //uint64_t insert_start = rdtsc();
        smc_insert(&smc, keys[idx], values[idx]);
        //insert += rdtsc() - insert_start;
        ++n_misses;
      } else if (v != values[idx]) {
        //uint64_t insert_start = rdtsc();
        smc_insert(&smc, keys[idx], values[idx]);
        //insert += rdtsc() - insert_start;
        ++n_collisions;
      }
    }
    uint64_t end = rdtsc();

    printf("[SMC] hit rate: %.2lf, miss: %lu, collision: %lu, evict: %lu, "
           "throughput: %.2lf Mpps\n",
           (double)(kNumQueries - n_misses - n_collisions) / kNumQueries,
           n_misses, n_collisions, n_evicts,
           kNumQueries / ((double)(end - start) / get_clock_frequency()) /
               1000000);
    printf("# inserts = %lu, # replaces = %lu, # evicts = %lu\n", n_inserts,
           n_replaces, n_evicts);
    printf("avg insert time = %.2lf ns\n",
           (double)insert * 1000000000 / n_inserts / get_clock_frequency());
  }

  return 0;
}
