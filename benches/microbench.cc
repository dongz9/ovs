#include <iostream>
#include <algorithm>
#include <emmintrin.h>
#include <immintrin.h>
#include <functional>
#include <inttypes.h>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t *rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static pcg32_random_t pcg32_global = {.state = 0x853c49e6748fea9bULL,
                                      .inc = 0xda3e39cb94b95bdbULL};

uint32_t pcg32_random() { return pcg32_random_r(&pcg32_global); }

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

static constexpr uint32_t kSMCEntriesPerBucket = 4;
static constexpr uint32_t kSMCEntries = 1 << 22;
static constexpr uint32_t kSMCBucketCount = kSMCEntries / kSMCEntriesPerBucket;
static constexpr uint32_t kSMCMask = kSMCBucketCount - 1;
static constexpr uint32_t kSMCSignatureBits = 16;

int bucket_count = 1;
double theta = 0.99;
bool lru = false;
bool uniform = true;
bool optimal = false;
bool cuckoo = false;
int miss_penalty = 0;
bool avx = false;

static constexpr uint32_t kMaxNumKeys = 4000000;
static constexpr uint32_t kNumWarmups = 400000000;
static constexpr uint32_t kNumQueries = 400000000;

uint32_t n = kMaxNumKeys;
// uint32_t count[kMaxNumKeys];
uint32_t keys[kMaxNumKeys];
uint16_t values[kMaxNumKeys];

double alpha, zetan, eta;

struct SMCBucket {
  uint16_t sig[kSMCEntriesPerBucket];
  uint16_t v[kSMCEntriesPerBucket];
};

struct SMC {
  SMCBucket buckets[kSMCBucketCount];
};

static inline uint32_t smc_alt_bucket_index(const uint32_t hash) {
  uint32_t bucket_index = hash & kSMCMask;
  uint32_t tag = hash >> 12;
  return (bucket_index ^ ((tag + 1) * 0x5bd1e995)) & kSMCMask;
}

void smc_init(SMC *smc) {
  for (uint32_t i = 0; i < kSMCBucketCount; ++i) {
    for (uint32_t j = 0; j < kSMCEntriesPerBucket; ++j) {
      smc->buckets[i].v[j] = 0xFFFF;
    }
  }
}

void swap(uint16_t &x, uint16_t &y) {
  x ^= y;
  y ^= x;
  x ^= y;
}

uint32_t n_inserts = 0, n_replaces = 0;
uint32_t n_evicts = 0;

void smc_insert(SMC *smc, uint32_t hash, uint16_t v) {
  uint32_t bucket_idx = hash & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  if (cuckoo) {
    SMCBucket *bucket = &smc->buckets[bucket_idx];
    uint32_t alt_bucket_idx = smc_alt_bucket_index(hash);
    SMCBucket *alt_bucket = &smc->buckets[alt_bucket_idx];

    for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->sig[i] == sig) {
        ++n_replaces;
        bucket->v[i] = v;
        return;
      }
      if (bucket->v[i] == 0xFFFF) {
        bucket->sig[i] = sig;
        bucket->v[i] = v;
        return;
      }
    }
    for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (alt_bucket->sig[i] == sig) {
        ++n_replaces;
        alt_bucket->v[i] = v;
        return;
      }
      if (alt_bucket->v[i] == 0xFFFF) {
        alt_bucket->sig[i] = sig;
        alt_bucket->v[i] = v;
        return;
      }
    }

    uint32_t victim = pcg32_random() % (kSMCEntriesPerBucket * 2);

    if (victim & 1) {
      bucket->sig[victim >> 1] = sig;
      bucket->v[victim >> 1] = v;
    } else {
      alt_bucket->sig[victim >> 1] = sig;
      alt_bucket->v[victim >> 1] = v;
    }
  } else {
    for (uint32_t off = 0; off < bucket_count; ++off) {
      SMCBucket *bucket = &smc->buckets[(bucket_idx + off) % kSMCMask];
      for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
        if (bucket->sig[i] == sig) {
          ++n_replaces;
          bucket->v[i] = v;
          if (i > 0 && lru) {
            swap(bucket->sig[i - 1], bucket->sig[i]);
            swap(bucket->v[i - 1], bucket->v[i]);
          }
          return;
        }
      }
      for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
        if (bucket->v[i] == 0xFFFF) {
          bucket->sig[i] = sig;
          bucket->v[i] = v;
          return;
        }
      }
    }

    uint32_t off = pcg32_random() % bucket_count;
    SMCBucket *bucket = &smc->buckets[(bucket_idx + off) & kSMCMask];

    if (lru) {
      bucket->sig[kSMCEntriesPerBucket - 1] =
          bucket->sig[kSMCEntriesPerBucket - 2];
      bucket->v[kSMCEntriesPerBucket - 1] = bucket->v[kSMCEntriesPerBucket - 2];
      bucket->sig[kSMCEntriesPerBucket - 2] = sig;
      bucket->v[kSMCEntriesPerBucket - 2] = v;
    } else {
      uint32_t victim = pcg32_random() % kSMCEntriesPerBucket;
      bucket->sig[victim] = sig;
      bucket->v[victim] = v;
    }
  }
}

void smc_insert_wide(SMC *smc, uint32_t hash, uint16_t v) {
  uint32_t bucket_idx = hash & kSMCMask;
  uint32_t alt_bucket_idx = (bucket_idx + 1) & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  auto match = _mm_set1_epi16(sig);
  auto sigs =
      _mm_set_epi64x(*(__int64_t *)(&smc->buckets[alt_bucket_idx & kSMCMask]),
                     *(__int64_t *)(&smc->buckets[bucket_idx]));

  auto bitmask = _mm_movemask_epi8(_mm_cmpeq_epi16(match, sigs));

  ++n_inserts;
  if (likely(bitmask == 0)) {
    for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (smc->buckets[bucket_idx].v[i] == 0xFFFF) {
        smc->buckets[bucket_idx].sig[i] = sig;
        smc->buckets[bucket_idx].v[i] = v;
        return;
      }
    }
    for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (smc->buckets[alt_bucket_idx].v[i] == 0xFFFF) {
        smc->buckets[alt_bucket_idx].sig[i] = sig;
        smc->buckets[alt_bucket_idx].v[i] = v;
        return;
      }
    }
    uint32_t evict = rand() % kSMCEntriesPerBucket;
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
  uint32_t bucket_idx = hash & kSMCMask;
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  if (cuckoo) {
    SMCBucket *bucket = &smc->buckets[bucket_idx];
    uint32_t alt_bucket_idx = smc_alt_bucket_index(hash);
    SMCBucket *alt_bucket = &smc->buckets[alt_bucket_idx];

    for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
      if (bucket->sig[i] == sig) {
        uint16_t v = bucket->v[i];
        if (i > 0 && lru) {
          swap(bucket->sig[i - 1], bucket->sig[i]);
          swap(bucket->v[i - 1], bucket->v[i]);
        }
        return v;
      }
      if (alt_bucket->sig[i] == sig) {
        uint16_t v = alt_bucket->v[i];
        if (i > 0 && lru) {
          swap(alt_bucket->sig[i - 1], alt_bucket->sig[i]);
          swap(alt_bucket->v[i - 1], alt_bucket->v[i]);
        }
        return v;
      }
    }
  } else {
    for (uint32_t off = 0; off < bucket_count; ++off) {
      SMCBucket *bucket = &smc->buckets[(bucket_idx + off) & kSMCMask];
      for (uint32_t i = 0; i < kSMCEntriesPerBucket; ++i) {
        if (bucket->sig[i] == sig) {
          uint16_t v = bucket->v[i];
          if (i > 0 && lru) {
            swap(bucket->sig[i - 1], bucket->sig[i]);
            swap(bucket->v[i - 1], bucket->v[i]);
          }
          return v;
        }
      }
    }
  }

  return 0xFFFF;
}

uint64_t counter0,counter1,counter2,counter3;

uint16_t smc_lookup_wide(SMC *smc, uint32_t hash, uint32_t (&counter)[2]) {
  uint16_t sig = hash >> (32 - kSMCSignatureBits);

  uint32_t bucket_idx = hash & kSMCMask;
  if (cuckoo || unlikely(bucket_idx == kSMCMask - 1)) {
    uint32_t alt_bucket_idx = cuckoo ? smc_alt_bucket_index(hash) : (bucket_idx + 1) & kSMCMask;
    auto match = _mm_set1_epi16(sig);
    auto sigs =
        _mm_set_epi64x(*(__int64_t *)(&smc->buckets[alt_bucket_idx & kSMCMask]),
                       *(__int64_t *)(&smc->buckets[bucket_idx]));
    auto bitmask = _mm_movemask_epi8(_mm_cmpeq_epi16(match, sigs));
    if (bitmask == 0)
      return 0xFFFF;
    int idx = __builtin_ctz((bitmask & (-bitmask))) / 2;

    if (idx >> 2) {
      uint16_t v = smc->buckets[alt_bucket_idx].v[idx & 3];
      ++counter[idx&1];
      if (((idx & 3) > 0) && lru) {
        swap(smc->buckets[alt_bucket_idx].sig[(idx & 3) - 1],
             smc->buckets[alt_bucket_idx].sig[idx & 3]);
        swap(smc->buckets[alt_bucket_idx].v[(idx & 3) - 1],
             smc->buckets[alt_bucket_idx].v[idx & 3]);
      }
      return v;
    } else {
      uint16_t v = smc->buckets[bucket_idx].v[idx & 3];
      ++counter[idx&1];
      if (((idx & 3) > 0) && lru) {
        swap(smc->buckets[bucket_idx].sig[(idx & 3) - 1],
             smc->buckets[bucket_idx].sig[idx & 3]);
        swap(smc->buckets[bucket_idx].v[(idx & 3) - 1],
             smc->buckets[bucket_idx].v[idx & 3]);
      }
      return v;
    }
  } else {
    auto match =
        _mm256_set_epi16(0xFFFE, 0xFFFE, 0xFFFE, 0xFFFE, sig,
                        sig, sig, sig, 0xFFFE, 0xFFFE, 0xFFFE, 0xFFFE, sig, sig, sig, sig);
    auto sigs = _mm256_lddqu_si256((__m256i const *)(&smc->buckets[bucket_idx]));
    auto bitmask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(match, sigs));
    if (bitmask == 0) {
      return 0xFFFF;
    }
    int idx = __builtin_ctz(bitmask) / 2;
    if (idx < 4) {
      if (idx==0) ++counter0;
      else if (idx==1) ++counter1;
      else if (idx==2) ++counter2;
      else ++counter3;
      if (idx > 0 && lru) {
        swap(smc->buckets[bucket_idx].sig[idx-1],
             smc->buckets[bucket_idx].sig[idx]);
        swap(smc->buckets[bucket_idx].v[idx-1],
             smc->buckets[bucket_idx].v[idx]);
        return smc->buckets[bucket_idx].v[idx-1];      
      }
      return smc->buckets[bucket_idx].v[idx];
    } else {
      if (idx==8) ++counter0;
      else if (idx==9) ++counter1;
      else if (idx==10) ++counter2;
      else ++counter3;
      if (idx > 8 && lru) {
        swap(smc->buckets[bucket_idx+1].sig[idx-9],
             smc->buckets[bucket_idx+1].sig[idx-8]);
        swap(smc->buckets[bucket_idx+1].v[idx-9],
             smc->buckets[bucket_idx+1].v[idx-8]);
        return smc->buckets[bucket_idx+1].v[idx-9];
      }
      return smc->buckets[bucket_idx+1].v[idx-8];
    }
  }
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

void cycle_sleep(int num_cycles) {
  uint64_t start = rdtsc();
  while (rdtsc() - start < num_cycles)
    ;
}

SMC smc __attribute__((aligned(64)));

double zeta(uint32_t n, double theta) {
  double sum = 0;
  for (uint32_t i = 0; i < n; ++i)
    sum += 1 / pow(i + 1, theta);
  return sum;
}

std::random_device random_device;
std::mt19937 gen(random_device());
std::uniform_real_distribution<> distribution(0.0, 1.0);

uint32_t zipf() {
  double u = distribution(gen);
  double uz = u * zetan;
  if (uz < 1)
    return 0;
  if (uz < 1 + pow(0.5, theta))
    return 1;
  return (uint32_t)(n * pow(eta * u - eta + 1, alpha));
}

int main(int argc, char *argv[]) {
  int c;
  extern char *optarg;

  while ((c = getopt(argc, argv, "b:lzt:n:om:ca")) != -1) {
    switch (c) {
    case 'b':
      bucket_count = atoi(optarg);
      break;
    case 'l':
      lru = true;
      break;
    case 'z':
      uniform = false;
      break;
    case 't':
      theta = atof(optarg);
      break;
    case 'n':
      n = atoi(optarg);
      break;
    case 'o':
      optimal = true;
      break;
    case 'm':
      miss_penalty = atoi(optarg);
      break;
    case 'c':
      cuckoo = true;
      break;
    case 'a':
      avx = true;
      break;
    }
  }

  alpha = 1.0 / (1.0 - theta);
  zetan = zeta(n, theta);
  eta = (1 - pow(2.0 / n, 1.0 - theta)) / (1.0 - zeta(2, theta) / zetan);

  for (uint32_t i = 0; i < n; ++i) {
    keys[i] = (uint32_t)pcg32_random();
    values[i] = (uint16_t)(pcg32_random() % 0xFFFE);
  }

  // uint32_t sum = 0, uniq = 0;
  // for (uint32_t i = 0; i < kNumQueries; ++i) {
  //   uint32_t idx = zipf();
  //   while (idx >= n) idx = zipf();
  //   if (count[idx] == 0) ++uniq;
  //   count[idx]++;
  // }
  // for (uint32_t i = 0; i < kSMCEntries; ++i) {
  //   sum += count[i];
  // }
  // printf("%lf %lf\n", (double)sum / kNumQueries, (double)n / kSMCEntries);

  // size_t count = 0, count2 = 0;

  static uint32_t counter[2] = {0, 0};

  smc_init(&smc);
  for (uint32_t i = 0; i < kNumWarmups; ++i) {
    uint32_t idx;
    if (uniform) {
      idx = pcg32_random() % n;
    } else {
      idx = zipf();
      while (idx >= n) {
        idx = zipf();
      }
    }
    uint16_t v;
    if (avx) {
      v = smc_lookup_wide(&smc, keys[idx], counter);
    } else {
      v = smc_lookup(&smc, keys[idx]);
    }
    if (v == 0xFFFF || v != values[idx]) {
      smc_insert(&smc, keys[idx], values[idx]);
    }
  }

  uint64_t start = rdtsc();
  uint32_t n_misses = 0;
  uint64_t lookup_cycles = 0;
  uint64_t miss_cycles = 0;
  for (uint32_t i = 0; i < kNumQueries; ++i) {
    uint32_t idx;
    if (uniform) {
      idx = pcg32_random() % n;
    } else {
      idx = zipf();
      while (idx >= n) {
        idx = zipf();
      }
    }
    uint16_t v;
    if (avx) {
      // uint64_t lookup_start = rdtsc();
      v = smc_lookup_wide(&smc, keys[idx], counter);
      // lookup_cycles += rdtsc() - lookup_start;
    } else {
      // uint64_t lookup_start = rdtsc();
      v = smc_lookup(&smc, keys[idx]);
      // lookup_cycles += rdtsc() - lookup_start;
    }
    if (v == 0xFFFF || v != values[idx]) {
      // uint64_t miss_start = rdtsc();
      cycle_sleep(miss_penalty);
      smc_insert(&smc, keys[idx], values[idx]);
      // miss_cycles += rdtsc() - miss_start;
      ++n_misses;
    }
  }
  uint64_t end = rdtsc();

  uint32_t n_occupied = 0;
  for (uint32_t i = 0; i < kSMCBucketCount; ++i)
    for (uint32_t j = 0; j < kSMCEntriesPerBucket; ++j)
      if (smc.buckets[i].v[j] != 0xFFFF)
        ++n_occupied;

  printf("hit rate: %.8lf, miss: %u, replaces: %u, occupancy = %.8lf, tput = "
         "%.2lfMpps, cycles / lookup = %.2lf, cycles / miss = %.2lf\n",
         (double)(kNumQueries - n_misses) / kNumQueries, n_misses, n_replaces,
         (double)n_occupied / kSMCEntries,
         (double)kNumQueries / (end - start) * get_clock_frequency() / 1000000,
         (double)lookup_cycles / kNumQueries,
         (double)miss_cycles / n_misses);

  return 0;
}
