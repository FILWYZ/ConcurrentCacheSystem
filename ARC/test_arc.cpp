#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ARC.h"

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
        std::make_unique<KamaCache::ARCCache<int, std::string>>(2);
    cache->put(1, "one");
    std::string value;
    require(cache->get(1, value) && value == "one",
            "ARC should support the common policy interface");
    require(cache->erase(1), "ARC erase should remove a resident key");
}

void testRecentAndFrequentPromotion()
{
    KamaCache::ARCCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);

    int value = 0;
    require(cache.get(1, value) && value == 10, "ARC should hit a resident key");
    const KamaCache::ARCState state = cache.state();
    require(state.recent == 1 && state.frequent == 1,
            "a repeated key should move from T1 to T2");

    cache.put(3, 30);
    require(cache.contains(1), "frequently accessed key should survive replacement");
    require(!cache.contains(2), "recent LRU key should move to its ghost list");
    require(cache.isGhost(2), "evicted recent key should be tracked in B1");
}

void testAdaptiveTarget()
{
    KamaCache::ARCCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);

    int value = 0;
    require(cache.get(1, value), "setup should promote key 1");
    cache.put(3, 30);
    require(cache.isGhost(2), "key 2 should enter B1");

    cache.put(2, 200);
    const std::size_t increasedTarget = cache.state().recentTarget;
    require(increasedTarget > 0, "B1 hit should increase the T1 target");
    require(cache.isGhost(1), "replacement should move key 1 to B2");

    cache.put(1, 100);
    require(cache.state().recentTarget < increasedTarget,
            "B2 hit should decrease the T1 target");
    require(cache.contains(1) && cache.contains(2),
            "ghost hits should restore entries into T2");
}

void testUpdatePeekClearAndZeroCapacity()
{
    KamaCache::ARCCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(1, "ONE");

    std::string value;
    require(cache.peek(1, value) && value == "ONE", "ARC update should replace the value");
    cache.clear();
    require(cache.size() == 0, "ARC clear should remove residents");

    KamaCache::ARCCache<int, int> empty(0);
    empty.put(1, 1);
    require(empty.size() == 0, "zero-capacity ARC should remain empty");
}

void testStats()
{
    KamaCache::ARCCache<int, int> cache(1);
    cache.put(1, 10);

    int value = 0;
    require(cache.get(1, value), "ARC should record a hit");
    require(!cache.get(2, value), "ARC should record a miss");
    cache.put(2, 20);

    const KamaCache::CacheStats stats = cache.stats();
    require(stats.hits == 1 && stats.misses == 1, "ARC hit and miss stats should be exact");
    require(stats.evictions == 1, "ARC eviction stats should be exact");
}

void testLongRunningInvariants()
{
    constexpr std::size_t capacity = 17;
    KamaCache::ARCCache<int, int> cache(capacity);
    std::uint32_t state = 0x12345678U;

    for (int operation = 0; operation < 20000; ++operation) {
        state = state * 1664525U + 1013904223U;
        const int key = static_cast<int>((state >> 8U) % 64U);

        if ((state & 3U) == 0U) {
            cache.erase(key);
        } else if ((state & 1U) == 0U) {
            cache.put(key, operation);
        } else {
            int value = 0;
            cache.get(key, value);
        }

        const KamaCache::ARCState arc = cache.state();
        require(arc.recent + arc.frequent == cache.size(),
                "ARC resident lists should match the resident map");
        require(cache.size() <= capacity, "ARC resident size should not exceed capacity");
        require(arc.recentGhost + arc.frequentGhost <= capacity,
                "ARC ghost history should not exceed capacity");
        require(arc.recentTarget <= capacity, "ARC adaptive target should stay bounded");
    }
}

void testShardedCapacityAndConcurrency()
{
    constexpr int keyCount = 64;
    constexpr int threadCount = 12;
    constexpr int operationsPerThread = 4000;
    KamaCache::ShardedARCCache<int, int, IdentityHash> cache(keyCount, 8);

    require(cache.capacity() == keyCount, "sharded ARC should preserve total capacity");
    require(cache.shardCount() == 8, "sharded ARC should preserve shard count");

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
            "sharded ARC should retain preloaded keys under contention");
    require(cache.size() == keyCount, "sharded ARC should preserve resident size");
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
        {"recent and frequent promotion", testRecentAndFrequentPromotion},
        {"adaptive target", testAdaptiveTarget},
        {"update peek clear and zero capacity", testUpdatePeekClearAndZeroCapacity},
        {"stats", testStats},
        {"long-running invariants", testLongRunningInvariants},
        {"sharded capacity and concurrency", testShardedCapacityAndConcurrency},
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
