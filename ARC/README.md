# ARC Cache

`ARCCache` implements Adaptive Replacement Cache using four LRU lists:

- `T1`: recently seen resident entries.
- `T2`: frequently reused resident entries.
- `B1`: ghost keys evicted from `T1`.
- `B2`: ghost keys evicted from `T2`.

Ghost lists store keys only. Hits in `B1` increase the target size of `T1`;
hits in `B2` decrease it. This lets ARC adapt between recency-heavy scans and
frequency-heavy hot sets without a manually selected LRU/LFU ratio.

```cpp
#include "ARC/ARC.h"

KamaCache::ARCCache<int, std::string> cache(1000);
cache.put(1, "value");

std::string value;
if (cache.get(1, value)) {
    // Resident hit.
}
```

For concurrent workloads, `ShardedARCCache` routes keys to independent ARC
instances:

```cpp
KamaCache::ShardedARCCache<int, std::string> cache(100000, 32);
```

Sharding reduces lock contention, but adaptation and replacement ordering are
local to each shard rather than global.
