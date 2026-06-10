#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ARC/ARC.h"
#include "LFU.h"
#include "LRU.h"

namespace {

using Clock = std::chrono::steady_clock;
using Key = std::uint64_t;
using Value = std::uint64_t;

struct BenchmarkResult {
    std::string policy;
    double seconds{0.0};
    double throughputMops{0.0};
    double hitRate{0.0};
    std::uint64_t hits{0};
    std::uint64_t misses{0};
};

class FastRandom {
public:
    explicit FastRandom(std::uint64_t seed)
        : state_(seed)
    {
    }

    std::uint64_t next()
    {
        std::uint64_t value = state_;
        value ^= value << 13U;
        value ^= value >> 7U;
        value ^= value << 17U;
        state_ = value;
        return value;
    }

private:
    std::uint64_t state_;
};

template <typename K, typename V, typename Hash = std::hash<K>>
class ExactShardedLRU {
public:
    ExactShardedLRU(std::size_t capacity, std::size_t shardCount)
        : shardCount_(std::max<std::size_t>(
              1, std::min(capacity == 0 ? 1 : capacity, shardCount)))
    {
        shards_.reserve(shardCount_);
        const std::size_t baseCapacity = capacity / shardCount_;
        const std::size_t extraCapacity = capacity % shardCount_;
        for (std::size_t index = 0; index < shardCount_; ++index) {
            shards_.push_back(std::make_unique<KamaCache::KLruCacheShard<K, V>>(
                baseCapacity + (index < extraCapacity ? 1U : 0U)));
        }
    }

    void put(const K& key, const V& value)
    {
        shardFor(key).put(key, value);
    }

    bool get(const K& key, V& value)
    {
        return shardFor(key).get(key, value);
    }

private:
    KamaCache::KLruCacheShard<K, V>& shardFor(const K& key)
    {
        return *shards_[hash_(key) % shardCount_];
    }

    const std::size_t shardCount_;
    Hash hash_;
    std::vector<std::unique_ptr<KamaCache::KLruCacheShard<K, V>>> shards_;
};

std::vector<Key> makeHotspotTrace(std::size_t operations, std::size_t capacity)
{
    std::vector<Key> trace;
    trace.reserve(operations);
    FastRandom random(0x91e10da5c79e7b1dULL);
    const std::size_t hotKeys = std::max<std::size_t>(1, capacity / 5);
    const std::size_t coldKeys = capacity * 20;

    for (std::size_t index = 0; index < operations; ++index) {
        const bool hot = random.next() % 100 < 80;
        trace.push_back(hot
            ? random.next() % hotKeys
            : hotKeys + random.next() % coldKeys);
    }
    return trace;
}

std::vector<Key> makeScanTrace(std::size_t operations, std::size_t capacity)
{
    std::vector<Key> trace;
    trace.reserve(operations);
    FastRandom random(0x4f1bbcdc6762c9adULL);
    const std::size_t hotKeys = std::max<std::size_t>(1, capacity / 4);
    std::uint64_t scanKey = capacity * 10;

    for (std::size_t index = 0; index < operations; ++index) {
        if (index % 10 < 7) {
            trace.push_back(random.next() % hotKeys);
        } else {
            trace.push_back(scanKey++);
        }
    }
    return trace;
}

std::vector<Key> makeShiftTrace(std::size_t operations, std::size_t capacity)
{
    std::vector<Key> trace;
    trace.reserve(operations);
    FastRandom random(0xd1b54a32d192ed03ULL);
    const std::size_t workingSet = std::max<std::size_t>(1, capacity / 2);
    const std::size_t phaseLength = std::max<std::size_t>(1, operations / 8);

    for (std::size_t index = 0; index < operations; ++index) {
        const std::size_t phase = index / phaseLength;
        const Key base = static_cast<Key>(phase * workingSet);
        trace.push_back(base + random.next() % workingSet);
    }
    return trace;
}

std::vector<Key> makeUniformTrace(std::size_t operations, std::size_t capacity)
{
    std::vector<Key> trace;
    trace.reserve(operations);
    FastRandom random(0x94d049bb133111ebULL);
    const std::size_t keySpace = capacity * 4;

    for (std::size_t index = 0; index < operations; ++index) {
        trace.push_back(random.next() % keySpace);
    }
    return trace;
}

template <typename Cache>
BenchmarkResult runSingle(
    const std::string& policy,
    Cache& cache,
    const std::vector<Key>& trace)
{
    std::uint64_t hits = 0;
    Value checksum = 0;
    const auto start = Clock::now();

    for (const Key key : trace) {
        Value value = 0;
        if (cache.get(key, value)) {
            ++hits;
            checksum ^= value;
        } else {
            cache.put(key, key);
        }
    }

    const double seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    if (checksum == std::numeric_limits<Value>::max()) {
        std::cerr << "";
    }

    const std::uint64_t misses = trace.size() - hits;
    return BenchmarkResult{
        policy,
        seconds,
        static_cast<double>(trace.size()) / seconds / 1'000'000.0,
        trace.empty() ? 0.0 : static_cast<double>(hits) / trace.size(),
        hits,
        misses,
    };
}

template <typename Cache>
BenchmarkResult runConcurrent(
    const std::string& policy,
    Cache& cache,
    const std::vector<Key>& trace,
    std::size_t threadCount)
{
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> checksum{0};
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (std::size_t thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread] {
            std::uint64_t localHits = 0;
            Value localChecksum = 0;

            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (std::size_t index = thread; index < trace.size(); index += threadCount) {
                const Key key = trace[index];
                Value value = 0;
                if (cache.get(key, value)) {
                    ++localHits;
                    localChecksum ^= value;
                } else {
                    cache.put(key, key);
                }
            }

            hits.fetch_add(localHits, std::memory_order_relaxed);
            checksum.fetch_xor(localChecksum, std::memory_order_relaxed);
        });
    }

    while (ready.load(std::memory_order_acquire) != threadCount) {
        std::this_thread::yield();
    }

    const auto startedAt = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }
    const double seconds =
        std::chrono::duration<double>(Clock::now() - startedAt).count();

    if (checksum.load(std::memory_order_relaxed)
        == std::numeric_limits<Value>::max()) {
        std::cerr << "";
    }

    const std::uint64_t hitCount = hits.load(std::memory_order_relaxed);
    const std::uint64_t misses = trace.size() - hitCount;
    return BenchmarkResult{
        policy,
        seconds,
        static_cast<double>(trace.size()) / seconds / 1'000'000.0,
        trace.empty() ? 0.0 : static_cast<double>(hitCount) / trace.size(),
        hitCount,
        misses,
    };
}

void printResults(
    const std::string& workload,
    const std::vector<BenchmarkResult>& results)
{
    std::cout << "\n" << workload << '\n';
    std::cout << std::left << std::setw(18) << "Policy"
              << std::right << std::setw(13) << "Mops/s"
              << std::setw(13) << "Hit rate"
              << std::setw(13) << "Seconds"
              << std::setw(14) << "Hits"
              << std::setw(14) << "Misses" << '\n';
    std::cout << std::string(85, '-') << '\n';

    for (const auto& result : results) {
        std::cout << std::left << std::setw(18) << result.policy
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(13) << result.throughputMops
                  << std::setw(12) << result.hitRate * 100.0 << "%"
                  << std::setw(13) << result.seconds
                  << std::setw(14) << result.hits
                  << std::setw(14) << result.misses << '\n';
    }
}

void runSingleSuite(
    const std::string& workload,
    const std::vector<Key>& trace,
    std::size_t capacity)
{
    KamaCache::KLruCacheShard<Key, Value> lru(capacity);
    KamaCache::LFUCache<Key, Value> lfu(capacity);
    KamaCache::ARCCache<Key, Value> arc(capacity);

    printResults(workload + " - single thread", {
        runSingle("LRU", lru, trace),
        runSingle("LFU", lfu, trace),
        runSingle("ARC", arc, trace),
    });
}

void runConcurrentSuite(
    const std::string& workload,
    const std::vector<Key>& trace,
    std::size_t capacity,
    std::size_t shardCount,
    std::size_t threadCount)
{
    ExactShardedLRU<Key, Value> lru(capacity, shardCount);
    KamaCache::ShardedLFUCache<Key, Value> lfu(capacity, shardCount);
    KamaCache::ShardedARCCache<Key, Value> arc(capacity, shardCount);

    printResults(
        workload + " - " + std::to_string(threadCount) + " threads",
        {
            runConcurrent("Sharded LRU", lru, trace, threadCount),
            runConcurrent("Sharded LFU", lfu, trace, threadCount),
            runConcurrent("Sharded ARC", arc, trace, threadCount),
        });
}

std::size_t parseSize(const char* text, const char* name)
{
    try {
        const std::size_t value = std::stoull(text);
        if (value == 0) {
            throw std::invalid_argument("zero");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::size_t operations =
            argc > 1 ? parseSize(argv[1], "operations") : 2'000'000;
        const std::size_t capacity =
            argc > 2 ? parseSize(argv[2], "capacity") : 4096;
        const std::size_t threadCount =
            argc > 3 ? parseSize(argv[3], "threads")
                     : std::max(1U, std::thread::hardware_concurrency());
        const std::size_t shardCount =
            argc > 4 ? parseSize(argv[4], "shards") : threadCount;

        std::cout << "Cache policy benchmark\n"
                  << "Operations: " << operations
                  << ", capacity: " << capacity
                  << ", threads: " << threadCount
                  << ", shards: " << shardCount << '\n';

        const std::vector<std::pair<std::string, std::vector<Key>>> workloads{
            {"Hotspot 80/20", makeHotspotTrace(operations, capacity)},
            {"Hot set + scan", makeScanTrace(operations, capacity)},
            {"Working-set shift", makeShiftTrace(operations, capacity)},
            {"Uniform random", makeUniformTrace(operations, capacity)},
        };

        for (const auto& workload : workloads) {
            runSingleSuite(workload.first, workload.second, capacity);
            runConcurrentSuite(
                workload.first,
                workload.second,
                capacity,
                shardCount,
                threadCount);
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n'
                  << "usage: cache_benchmark [operations] [capacity] [threads] [shards]\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
