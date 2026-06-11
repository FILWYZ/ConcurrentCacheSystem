#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_cache_benchmark.sh [headers_dir] [operations_per_scenario] [capacity] [threads] [shards]
# Example:
#   ./run_cache_benchmark.sh . 1000000 16384 8 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEADERS_DIR="${1:-$SCRIPT_DIR}"
OPS="${2:-1000000}"
CAPACITY="${3:-16384}"
THREADS="${4:-1}"
SHARDS="${5:-$THREADS}"

BUILD_DIR="$SCRIPT_DIR/.cache_benchmark_build"
INCLUDE_DIR="$BUILD_DIR/include"
SRC="$BUILD_DIR/cache_benchmark.cpp"
BIN="$BUILD_DIR/cache_benchmark"

mkdir -p "$INCLUDE_DIR"

for f in CachePolicy.h LRU.h LFU.h ARC.h; do
    if [[ ! -f "$HEADERS_DIR/$f" ]]; then
        echo "Missing header: $HEADERS_DIR/$f" >&2
        exit 1
    fi
done

cp "$HEADERS_DIR/CachePolicy.h" "$INCLUDE_DIR/CachePolicy.h"
cp "$HEADERS_DIR/LRU.h" "$INCLUDE_DIR/LRU.h"
cp "$HEADERS_DIR/LFU.h" "$INCLUDE_DIR/LFU.h"
# 兼容处理：确保基类头文件引用正常
if grep -q '"../CachePolicy.h"' "$HEADERS_DIR/ARC.h"; then
    sed 's|#include "../CachePolicy.h"|#include "CachePolicy.h"|' "$HEADERS_DIR/ARC.h" > "$INCLUDE_DIR/ARC.h"
else
    cp "$HEADERS_DIR/ARC.h" "$INCLUDE_DIR/ARC.h"
fi

cat > "$SRC" <<'CPP'
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "CachePolicy.h"
#include "LRU.h"
#include "LFU.h"
#include "ARC.h"

using Key = int;
using Value = int;

struct Operation {
    // 0 = get, 1 = put
    std::uint8_t type{};
    Key key{};
    Value value{};
};

struct BenchResult {
    std::string name;
    double milliseconds{};
    double mops{};
    std::uint64_t gets{};
    std::uint64_t puts{};
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    bool evictionsKnown{};
    std::size_t finalSize{};
};

struct LocalCounters {
    std::uint64_t gets{};
    std::uint64_t puts{};
    std::uint64_t hits{};
    std::uint64_t misses{};
};

// ============================================================================
// 大厂重构规范修复：统一将 KamaCache:: 调整为对齐后的 Cache:: 命名空间
// 并修正重构后的类名与接口调用方式
// ============================================================================

class LRUAdapter {
public:
    LRUAdapter(std::size_t capacity, std::size_t shards)
        : cache_(capacity, static_cast<int>(std::max<std::size_t>(1, shards))) {}

    bool get(const Key& key, Value& value) { return cache_.get(key, value); }
    void put(const Key& key, const Value& value) { cache_.put(key, value); }
    std::size_t size() const { return cache_.size(); }
    std::uint64_t evictions() const { return cache_.stats().evictions; }
    bool evictionsKnown() const { return true; }

private:
    Cache::HighConcurrencyCache<Key, Value> cache_; // 修复类名
};

class LRUKAdapter {
public:
    LRUKAdapter(std::size_t capacity, std::size_t shards)
        : cache_(capacity, capacity, 2, static_cast<int>(std::max<std::size_t>(1, shards))) {}

    bool get(const Key& key, Value& value) { return cache_.get(key, value); }
    void put(const Key& key, const Value& value) { cache_.put(key, value); }
    std::size_t size() const { return cache_.mainSize() + cache_.historySize(); }
    std::uint64_t evictions() const { return 0; }
    bool evictionsKnown() const { return false; }

private:
    Cache::LruKCache<Key, Value> cache_; // 修复类名
};

class LFUAdapter {
public:
    LFUAdapter(std::size_t capacity, std::size_t shards)
        : cache_(capacity, std::max<std::size_t>(1, shards)) {}

    bool get(const Key& key, Value& value) { return cache_.get(key, value); }
    void put(const Key& key, const Value& value) { cache_.put(key, value); }
    std::size_t size() const { return cache_.size(); }
    std::uint64_t evictions() const { return cache_.stats().evictions; }
    bool evictionsKnown() const { return true; }

private:
    Cache::ShardedLFUCache<Key, Value> cache_; // 修复命名空间
};

class ARCAdapter {
public:
    ARCAdapter(std::size_t capacity, std::size_t shards)
        : cache_(capacity, std::max<std::size_t>(1, shards)) {}

    bool get(const Key& key, Value& value) { return cache_.get(key, value); }
    void put(const Key& key, const Value& value) { cache_.put(key, value); }
    std::size_t size() const { return cache_.size(); }
    std::uint64_t evictions() const { return cache_.stats().evictions; } // 完美匹配无锁 stats 接口
    bool evictionsKnown() const { return true; }

private:
    Cache::ShardedARCCache<Key, Value> cache_; // 修复命名空间
};

std::vector<Operation> makeMixedWorkload(
    std::size_t operations,
    std::size_t keyspace,
    std::size_t hotKeys,
    double getRatio,
    double hotRatio,
    std::uint64_t seed)
{
    std::vector<Operation> workload;
    workload.reserve(operations);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int> hotDist(0, static_cast<int>(std::max<std::size_t>(1, hotKeys) - 1));
    std::uniform_int_distribution<int> coldDist(
        static_cast<int>(std::max<std::size_t>(1, hotKeys)),
        static_cast<int>(std::max<std::size_t>(hotKeys + 1, keyspace) - 1));

    for (std::size_t i = 0; i < operations; ++i) {
        const bool isGet = prob(rng) < getRatio;
        const bool isHot = prob(rng) < hotRatio;
        const int key = isHot ? hotDist(rng) : coldDist(rng);
        workload.push_back(Operation{
            static_cast<std::uint8_t>(isGet ? 0 : 1),
            key,
            key
        });
    }

    return workload;
}

std::vector<Operation> makeScanWorkload(
    std::size_t operations,
    std::size_t keyspace,
    double getRatio)
{
    std::vector<Operation> workload;
    workload.reserve(operations);

    for (std::size_t i = 0; i < operations; ++i) {
        const bool isGet = static_cast<double>(i % 1000) / 1000.0 < getRatio;
        const int key = static_cast<int>(i % keyspace);
        workload.push_back(Operation{
            static_cast<std::uint8_t>(isGet ? 0 : 1),
            key,
            key
        });
    }

    return workload;
}

template <typename Adapter>
BenchResult runBenchmark(
    const std::string& name,
    const std::vector<Operation>& workload,
    std::size_t capacity,
    std::size_t shards,
    std::size_t threadCount)
{
    Adapter cache(capacity, shards);

    // Prefill: 预热缓存使其达到可对比的边界状态
    for (std::size_t i = 0; i < capacity; ++i) {
        cache.put(static_cast<Key>(i), static_cast<Value>(i));
    }

    threadCount = std::max<std::size_t>(1, threadCount);
    std::vector<LocalCounters> counters(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t t = 0; t < threadCount; ++t) {
        const std::size_t begin = workload.size() * t / threadCount;
        const std::size_t end = workload.size() * (t + 1) / threadCount;

        workers.emplace_back([&, begin, end, t]() {
            Value value{};
            LocalCounters local;

            for (std::size_t i = begin; i < end; ++i) {
                const Operation& op = workload[i];

                if (op.type == 0) {
                    ++local.gets;
                    if (cache.get(op.key, value)) {
                        ++local.hits;
                    } else {
                        ++local.misses;
                        cache.put(op.key, op.value);
                        ++local.puts;
                    }
                } else {
                    cache.put(op.key, op.value);
                    ++local.puts;
                }
            }

            counters[t] = local;
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    const auto finish = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(finish - start).count();

    BenchResult result;
    result.name = name;
    result.milliseconds = ms;
    result.mops = workload.empty() || ms == 0.0
        ? 0.0
        : static_cast<double>(workload.size()) / ms / 1000.0;

    for (const auto& c : counters) {
        result.gets += c.gets;
        result.puts += c.puts;
        result.hits += c.hits;
        result.misses += c.misses;
    }

    result.evictions = cache.evictions();
    result.evictionsKnown = cache.evictionsKnown();
    result.finalSize = cache.size();
    return result;
}

void printResult(const BenchResult& r)
{
    const double hitRate = r.gets == 0 ? 0.0 : static_cast<double>(r.hits) * 100.0 / r.gets;
    const double missRate = r.gets == 0 ? 0.0 : static_cast<double>(r.misses) * 100.0 / r.gets;

    std::cout
        << std::left << std::setw(14) << r.name
        << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.milliseconds
        << std::setw(12) << std::fixed << std::setprecision(3) << r.mops
        << std::setw(12) << std::fixed << std::setprecision(2) << hitRate
        << std::setw(12) << std::fixed << std::setprecision(2) << missRate
        << std::setw(14) << r.puts
        << std::setw(14) << r.finalSize;

    if (r.evictionsKnown) {
        std::cout << std::setw(14) << r.evictions;
    } else {
        std::cout << std::setw(14) << "N/A";
    }

    std::cout << '\n';
}

void printHeader()
{
    std::cout
        << std::left << std::setw(14) << "cache"
        << std::right << std::setw(12) << "ms"
        << std::setw(12) << "Mops/s"
        << std::setw(12) << "hit%"
        << std::setw(12) << "miss%"
        << std::setw(14) << "puts"
        << std::setw(14) << "final_size"
        << std::setw(14) << "evictions"
        << '\n';

    std::cout << std::string(104, '-') << '\n';
}

void runScenario(
    const std::string& scenarioName,
    const std::vector<Operation>& workload,
    std::size_t capacity,
    std::size_t shards,
    std::size_t threads)
{
    std::cout << "\nScenario: " << scenarioName << '\n';
    printHeader();
    printResult(runBenchmark<LRUAdapter>("LRU", workload, capacity, shards, threads));
    printResult(runBenchmark<LRUKAdapter>("LRU-K(k=2)", workload, capacity, shards, threads));
    printResult(runBenchmark<LFUAdapter>("LFU", workload, capacity, shards, threads));
    printResult(runBenchmark<ARCAdapter>("ARC", workload, capacity, shards, threads));
}

int main(int argc, char** argv)
{
    const std::size_t operations = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 1000000;
    const std::size_t capacity = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 16384;
    const std::size_t threads = argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) : 1;
    const std::size_t shards = argc > 4 ? static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10)) : threads;

    const std::size_t keyspace = std::max<std::size_t>(capacity * 16, capacity + 1);
    const std::size_t hotKeys = std::max<std::size_t>(64, capacity / 8);

    std::cout << "cache benchmark\n"
              << "operations_per_scenario=" << operations
              << ", capacity=" << capacity
              << ", keyspace=" << keyspace
              << ", hot_keys=" << hotKeys
              << ", threads=" << threads
              << ", shards=" << shards
              << "\n";

    runScenario(
        "hot_read_95_get_90_hot",
        makeMixedWorkload(operations, keyspace, hotKeys, 0.95, 0.90, 42),
        capacity,
        shards,
        threads);

    runScenario(
        "mixed_80_get_80_hot",
        makeMixedWorkload(operations, keyspace, hotKeys, 0.80, 0.80, 43),
        capacity,
        shards,
        threads);

    runScenario(
        "scan_95_get",
        makeScanWorkload(operations, keyspace, 0.95),
        capacity,
        shards,
        threads);

    runScenario(
        "write_heavy_50_get_70_hot",
        makeMixedWorkload(operations, keyspace, hotKeys, 0.50, 0.70, 44),
        capacity,
        shards,
        threads);

    return 0;
}
CPP

CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX=g++
    elif command -v clang++ >/dev/null 2>&1; then
        CXX=clang++
    else
        echo "Neither g++ nor clang++ found." >&2
        exit 1
    fi
fi

"$CXX" -std=c++17 -O3 -DNDEBUG -pthread -I"$INCLUDE_DIR" "$SRC" -o "$BIN"
"$BIN" "$OPS" "$CAPACITY" "$THREADS" "$SHARDS"