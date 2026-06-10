#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "CachePolicy.h"
#include "LFU.h"

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct IdentityHash {
    std::size_t operator()(int key) const noexcept
    {
        return static_cast<std::size_t>(key);
    }
};

void testPolicyInterface()
{
    std::unique_ptr<KamaCache::CachePolicy<int, std::string>> cache =
        std::make_unique<KamaCache::LFUCache<int, std::string>>(2);
    cache->put(1, "one");
    std::string value;
    require(cache->get(1, value) && value == "one",
            "policy get should return the stored value");
    require(cache->erase(1), "policy erase should remove the key");
}

void testEvictionAndTieBreak()
{
    KamaCache::LFUCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);

    int value = 0;
    require(cache.get(1, value), "key 1 should be promoted");
    cache.put(3, 30);
    require(cache.contains(1), "more frequent key should survive");
    require(!cache.contains(2), "least frequent key should be evicted");

    KamaCache::LFUCache<int, int> tie(2);
    tie.put(1, 10);
    tie.put(2, 20);
    tie.put(3, 30);
    require(!tie.contains(1), "equal-frequency entries should use LRU tie-break");
}

void testUpdatePeekAndErase()
{
    KamaCache::LFUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(1, "ONE");
    cache.put(3, "three");

    std::string value;
    require(cache.peek(1, value) && value == "ONE", "update should replace the value");
    require(!cache.contains(2), "update should count as an access");
    require(cache.erase(1), "erase should remove a key");
    require(!cache.erase(1), "erase should report a missing key");
    cache.clear();
    require(cache.size() == 0, "clear should empty the cache");
}

void testFrequencyAging()
{
    KamaCache::LFUCache<int, int> cache(2, 4);
    cache.put(1, 10);
    cache.put(2, 20);

    int value = 0;
    for (int access = 0; access < 12; ++access) {
        require(cache.get(1, value), "hot key should survive repeated aging");
    }

    cache.put(3, 30);
    require(cache.contains(1), "aged hot key should still outrank a cold key");
    require(!cache.contains(2), "cold key should be evicted after aging");
}

void testZeroCapacity()
{
    KamaCache::LFUCache<int, int> cache(0);
    cache.put(1, 10);
    require(cache.size() == 0, "zero-capacity LFU should stay empty");
}

void testShardedCapacityRoutingAndStats()
{
    KamaCache::ShardedLFUCache<int, int, IdentityHash> cache(5, 3, 8);
    require(cache.capacity() == 5, "sharded LFU should preserve total capacity");
    require(cache.shardCount() == 3, "sharded LFU should preserve shard count");

    cache.put(0, 0);
    cache.put(3, 30);
    int value = 0;
    require(cache.get(0, value), "sharded LFU should record a hit");
    require(!cache.get(99, value), "sharded LFU should record a miss");
    cache.put(6, 60);
    require(cache.contains(0), "frequent key should survive local shard eviction");
    require(!cache.contains(3), "cold key should be evicted from its shard");

    const KamaCache::CacheStats stats = cache.stats();
    require(stats.hits == 1 && stats.misses == 1,
            "sharded LFU should track hits and misses");
    require(stats.evictions == 1, "sharded LFU should track evictions");
}

void testShardedConcurrency()
{
    constexpr int keyCount = 64;
    constexpr int threadCount = 12;
    constexpr int operationsPerThread = 4000;
    KamaCache::ShardedLFUCache<int, int, IdentityHash> cache(keyCount, 8, 128);

    for (int key = 0; key < keyCount; ++key) {
        cache.put(key, key);
    }

    std::atomic<bool> success{true};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread] {
            for (int operation = 0; operation < operationsPerThread; ++operation) {
                const int key = (operation + thread) % keyCount;
                if ((operation & 7) == 0) {
                    cache.put(key, operation);
                } else {
                    int value = 0;
                    if (!cache.get(key, value)) {
                        success.store(false, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    require(success.load(std::memory_order_relaxed),
            "sharded LFU should retain preloaded keys under contention");
    require(cache.size() == keyCount, "sharded LFU should preserve size");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

} // namespace

int main()
{
    const std::vector<TestCase> tests{
        {"policy interface", testPolicyInterface},
        {"eviction and tie-break", testEvictionAndTieBreak},
        {"update peek and erase", testUpdatePeekAndErase},
        {"frequency aging", testFrequencyAging},
        {"zero capacity", testZeroCapacity},
        {"sharded capacity routing and stats", testShardedCapacityRoutingAndStats},
        {"sharded concurrency", testShardedConcurrency},
    };

    std::size_t passed = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            ++passed;
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    std::cout << passed << '/' << tests.size() << " tests passed\n";
    return passed == tests.size() ? EXIT_SUCCESS : EXIT_FAILURE;
}
