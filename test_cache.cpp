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
#include "LRU.h"

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testPolicyInterface()
{
    std::unique_ptr<KamaCache::CachePolicy<int, std::string>> cache =
        std::make_unique<KamaCache::LRUCache<int, std::string>>(2);

    cache->put(1, "one");
    require(cache->get(1) == "one", "polymorphic get should return the stored value");
    require(cache->contains(1), "polymorphic contains should find the key");
    require(cache->erase(1), "polymorphic erase should remove the key");
    require(!cache->contains(1), "erased key should be absent");
}

void testLruEvictionAndUpdate()
{
    KamaCache::LRUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");

    std::string value;
    require(cache.get(1, value) && value == "one", "LRU get should hit key 1");
    cache.put(3, "three");
    require(!cache.contains(2), "LRU should evict the least recently used key");
    require(cache.contains(1) && cache.contains(3), "LRU should retain recent keys");

    cache.put(1, "ONE");
    require(cache.get(1, value) && value == "ONE", "LRU put should update an existing value");
    require(cache.size() == 2, "LRU update must not change size");
}

void testLruPeekAndManagement()
{
    KamaCache::LRUCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);

    int value = 0;
    require(cache.peek(1, value) && value == 10, "LRU peek should return a value");
    cache.put(3, 30);
    require(!cache.contains(1), "LRU peek must not refresh recency");
    require(cache.erase(2), "LRU erase should report success");
    require(!cache.erase(2), "LRU erase should report a missing key");
    cache.clear();
    require(cache.size() == 0, "LRU clear should empty the cache");
}

struct IdentityHash {
    std::size_t operator()(int key) const noexcept
    {
        return static_cast<std::size_t>(key);
    }
};

void testShardedLruCapacityAndRouting()
{
    KamaCache::ShardedLRUCache<int, int, IdentityHash> cache(5, 3);
    require(cache.capacity() == 5, "sharded LRU should preserve total capacity");
    require(cache.shardCount() == 3, "sharded LRU should preserve a valid shard count");

    cache.put(0, 0);
    cache.put(3, 30);
    cache.put(6, 60);
    require(!cache.contains(0), "a full shard should evict its local LRU key");
    require(cache.contains(3) && cache.contains(6),
            "a shard should retain its two most recent keys");

    cache.put(1, 10);
    cache.put(4, 40);
    cache.put(2, 20);
    require(cache.size() == cache.capacity(),
            "sum of shard capacities must equal configured capacity");
}

void testShardedLruStats()
{
    KamaCache::ShardedLRUCache<int, int, IdentityHash> cache(2, 1);
    cache.put(1, 10);

    int value = 0;
    require(cache.get(1, value) && value == 10, "sharded LRU should record a hit");
    require(!cache.get(2, value), "sharded LRU should record a miss");
    cache.put(2, 20);
    cache.put(3, 30);

    const KamaCache::CacheStats stats = cache.stats();
    require(stats.hits == 1, "sharded LRU hit count should be exact");
    require(stats.misses == 1, "sharded LRU miss count should be exact");
    require(stats.evictions == 1, "sharded LRU eviction count should be exact");
    require(stats.requests() == 2, "sharded LRU request count should be exact");
    require(stats.hitRate() == 0.5, "sharded LRU hit rate should be exact");

    cache.resetStats();
    const KamaCache::CacheStats reset = cache.stats();
    require(reset.requests() == 0 && reset.evictions == 0,
            "sharded LRU stats should be resettable");
}

void testShardedLruContention()
{
    constexpr int keyCount = 32;
    constexpr int threadCount = 12;
    constexpr int operationsPerThread = 5000;
    KamaCache::ShardedLRUCache<int, int, IdentityHash> cache(keyCount, 8);

    for (int key = 0; key < keyCount; ++key) {
        cache.put(key, key);
    }

    std::atomic<bool> success{true};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread] {
            for (int operation = 0; operation < operationsPerThread; ++operation) {
                const int key = (operation + thread) % keyCount;
                if ((operation & 3) == 0) {
                    cache.put(key, operation);
                    continue;
                }

                int value = 0;
                if (!cache.get(key, value)) {
                    success.store(false, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    require(success.load(std::memory_order_relaxed),
            "sharded LRU should not lose preloaded keys under contention");
    require(cache.size() == keyCount, "sharded LRU contention should preserve size");
}

void testLfuEviction()
{
    KamaCache::LFUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");

    std::string value;
    require(cache.get(1, value), "LFU get should hit key 1");
    cache.put(3, "three");
    require(cache.contains(1), "LFU should retain the more frequently used key");
    require(!cache.contains(2), "LFU should evict the least frequently used key");
    require(cache.contains(3), "LFU should insert the new key");
}

void testLfuLruTieBreak()
{
    KamaCache::LFUCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);

    require(!cache.contains(1), "LFU should use LRU order when frequencies tie");
    require(cache.contains(2) && cache.contains(3), "LFU tie-break should retain newer keys");
}

void testLfuUpdateAndManagement()
{
    KamaCache::LFUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(1, "ONE");
    cache.put(3, "three");

    std::string value;
    require(cache.peek(1, value) && value == "ONE", "LFU update should replace the value");
    require(!cache.contains(2), "LFU update should count as use");
    require(cache.erase(1), "LFU erase should remove an entry");
    require(cache.size() == 1, "LFU erase should decrease size");
    cache.clear();
    require(cache.size() == 0, "LFU clear should empty the cache");
}

void testLfuEraseMinimumFrequency()
{
    KamaCache::LFUCache<int, int> cache(3);
    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);

    int value = 0;
    require(cache.get(1, value), "LFU setup should promote key 1");
    require(cache.get(2, value), "LFU setup should promote key 2");
    require(cache.get(2, value), "LFU setup should promote key 2 twice");
    require(cache.erase(3), "LFU should erase the only minimum-frequency key");

    cache.put(4, 40);
    require(cache.get(4, value), "LFU should remain usable after erasing the minimum bucket");
    cache.put(5, 50);
    require(!cache.contains(1), "LFU should evict the new minimum-frequency key");
    require(cache.contains(2) && cache.contains(4) && cache.contains(5),
            "LFU should retain the correct keys after minimum-frequency repair");
}

template <typename Cache>
void testZeroCapacity(const std::string& name)
{
    Cache cache(0);
    cache.put(1, 1);
    require(cache.size() == 0, name + " with zero capacity must stay empty");

    int value = 99;
    require(!cache.get(1, value), name + " zero-capacity lookup should miss");
    require(value == 99, name + " miss should not overwrite the output value");
}

template <typename Cache>
void testConcurrentAccess(const std::string& name)
{
    constexpr int threadCount = 8;
    constexpr int keysPerThread = 250;
    Cache cache(threadCount * keysPerThread);
    std::atomic<bool> success{true};
    std::vector<std::thread> threads;

    for (int thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread] {
            const int first = thread * keysPerThread;
            for (int key = first; key < first + keysPerThread; ++key) {
                cache.put(key, key * 10);
                int value = 0;
                if (!cache.get(key, value) || value != key * 10) {
                    success.store(false, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    require(success.load(std::memory_order_relaxed), name + " concurrent read/write failed");
    require(cache.size() == threadCount * keysPerThread, name + " lost concurrent inserts");
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
        {"LRU eviction and update", testLruEvictionAndUpdate},
        {"LRU peek and management", testLruPeekAndManagement},
        {"sharded LRU capacity and routing", testShardedLruCapacityAndRouting},
        {"sharded LRU stats", testShardedLruStats},
        {"sharded LRU contention", testShardedLruContention},
        {"LFU eviction", testLfuEviction},
        {"LFU LRU tie-break", testLfuLruTieBreak},
        {"LFU update and management", testLfuUpdateAndManagement},
        {"LFU erase minimum frequency", testLfuEraseMinimumFrequency},
        {"LRU zero capacity", [] { testZeroCapacity<KamaCache::LRUCache<int, int>>("LRU"); }},
        {"LFU zero capacity", [] { testZeroCapacity<KamaCache::LFUCache<int, int>>("LFU"); }},
        {"LRU concurrent access",
         [] { testConcurrentAccess<KamaCache::LRUCache<int, int>>("LRU"); }},
        {"sharded LRU concurrent access",
         [] {
             testConcurrentAccess<KamaCache::ShardedLRUCache<int, int>>("sharded LRU");
         }},
        {"LFU concurrent access",
         [] { testConcurrentAccess<KamaCache::LFUCache<int, int>>("LFU"); }},
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
