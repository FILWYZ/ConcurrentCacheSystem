# Cache System

This directory contains a small, header-only, thread-safe C++17 cache library.

## Policies

- `LRUCache`: evicts the least recently used entry.
- `ShardedLRUCache`: routes keys across independent LRU shards to reduce lock
  contention under concurrent workloads.
- `LFUCache`: evicts the least frequently used entry and uses LRU order to
  break frequency ties.
- `ShardedLFUCache`: distributes LFU state across independent shards and
  periodically ages frequencies so old hot keys cannot dominate forever.
- `ARCCache`: adapts automatically between recent and frequent access patterns
  using resident and ghost lists.
- `ShardedARCCache`: runs independent ARC instances across hash shards.

All policies provide average O(1) `put`, `get`, and eviction operations.
Every public operation is thread-safe; sequences of operations are not
transactional.

## API

```cpp
#include "LRU.h"

KamaCache::LRUCache<int, std::string> cache(100);
cache.put(1, "value");

std::string value;
if (cache.get(1, value)) {
    // Cache hit. get also updates the replacement policy.
}
```

For concurrent servers, select a shard count based on the expected worker
count:

```cpp
KamaCache::ShardedLRUCache<int, std::string> cache(100000, 32);
cache.put(1, "value");

const KamaCache::CacheStats stats = cache.stats();
```

Each shard has its own mutex and LRU list, so unrelated keys can be processed
in parallel. The configured capacity is divided exactly across shards. LRU
ordering is strict within each shard rather than global; heavily skewed hashes
can therefore leave capacity in other shards unused. Use a well-distributed
hash function and benchmark shard counts such as 8, 16, 32, or the number of
request-processing threads.

The concurrent LFU has the same sharding trade-off:

```cpp
KamaCache::ShardedLFUCache<int, std::string> cache(
    100000, // total capacity
    32,     // shard count
    1 << 20 // frequency aging threshold
);
```

LFU operations are average O(1). Frequency aging is an occasional O(n)
maintenance operation within one shard. Lower thresholds adapt faster to
workload changes but trigger aging more often.

ARC is available from `ARC/ARC.h`. Its `T1` and `T2` lists hold resident
entries, while `B1` and `B2` remember recently evicted keys and adjust the
replacement balance. See `ARC/README.md` for details.

`peek` reads without updating recency or frequency. `contains`, `size`,
`capacity`, `erase`, and `clear` are also available. `KLruCache` and
`KLfuCache` are compatibility aliases. `KHashLruCache` aliases the sharded LRU
implementation.

## Build and test

```sh
cmake -S Cache -B Cache/build
cmake --build Cache/build
ctest --test-dir Cache/build --output-on-failure
```

## Performance comparison

Build in Release mode and run the standalone benchmark:

```sh
cmake -S Cache -B Cache/build -DCMAKE_BUILD_TYPE=Release
cmake --build Cache/build --target cache_benchmark
./Cache/build/cache_benchmark
```

Optional arguments are:

```text
cache_benchmark [operations] [capacity] [threads] [shards]
```

The benchmark replays identical pre-generated traces against LRU, LFU, and
ARC. It reports throughput and hit rate for hotspot, scan, workload-shift, and
uniform-random patterns. Benchmark generation time is excluded. Use at least
several million operations for less timing noise; concurrent execution order is
naturally scheduler-dependent.
