#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CachePolicy.h"

namespace KamaCache {

// ============================================================================
// 1. 单分片 LFU 缓存
//    设计目标：
//    1. O(1) 查找
//    2. O(1) 频次提升
//    3. O(1) 淘汰最低频率节点
//    4. 同频率下按 LRU 淘汰
//
//    数据结构：
//    1. frequencyBuckets_:
//       frequency -> list<CacheNode>
//
//    2. nodeIndex_:
//       key -> list<CacheNode>::iterator
//
//    说明：
//    CacheNode 内部同时保存 key / value / frequency。
//    这样可以避免 value、frequency、bucket key 分散在多个容器中造成状态不一致。
// ============================================================================
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class alignas(64) LFUCache final : public CachePolicy<Key, Value> {
public:
    explicit LFUCache(
        std::size_t capacity,
        std::size_t maxFrequency = defaultMaxFrequency())
        : LFUCache(capacity, maxFrequency, Hash{}, KeyEqual{})
    {
    }

    LFUCache(
        std::size_t capacity,
        std::size_t maxFrequency,
        const Hash& hash,
        const KeyEqual& equal)
        : capacity_(capacity)
        , maxFrequency_(std::max<std::size_t>(2, maxFrequency))
        , nodeIndex_(0, hash, equal)
    {
        nodeIndex_.reserve(capacity_);
        frequencyBuckets_.reserve(capacity_);
    }

    ~LFUCache() override = default;

    LFUCache(const LFUCache&) = delete;
    LFUCache& operator=(const LFUCache&) = delete;

    LFUCache(LFUCache&&) = delete;
    LFUCache& operator=(LFUCache&&) = delete;

    void put(const Key& key, const Value& value) override
    {
        static_cast<void>(putAndReport(key, value));
    }

    template <typename K, typename V>
    CacheWriteResult putAndReport(K&& key, V&& value)
    {
        if (capacity_ == 0) {
            return CacheWriteResult::ignored;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto found = nodeIndex_.find(key);
        if (found != nodeIndex_.end()) {
            found->second->value = std::forward<V>(value);
            promote(found->second);
            return CacheWriteResult::updated;
        }

        const bool needsEviction = nodeIndex_.size() >= capacity_;

        auto& bucket = frequencyBuckets_[1];
        bucket.emplace_front(std::forward<K>(key), std::forward<V>(value), 1);

        auto insertedNode = bucket.begin();

        try {
            auto insertion = nodeIndex_.emplace(insertedNode->key, insertedNode);
            if (!insertion.second) {
                insertion.first->second->value = insertedNode->value;

                bucket.erase(insertedNode);
                if (bucket.empty()) {
                    frequencyBuckets_.erase(1);
                }

                promote(insertion.first->second);
                return CacheWriteResult::updated;
            }
        } catch (...) {
            bucket.erase(insertedNode);
            if (bucket.empty()) {
                frequencyBuckets_.erase(1);
            }
            throw;
        }

        if (needsEviction) {
            evictOneLocked();
        }

        minFrequency_ = 1;

        return needsEviction
            ? CacheWriteResult::insertedWithEviction
            : CacheWriteResult::inserted;
    }

    bool get(const Key& key, Value& value) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) {
            return false;
        }

        value = found->second->value;
        promote(found->second);

        return true;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) {
            return false;
        }

        value = found->second->value;
        return true;
    }

    bool erase(const Key& key) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) {
            return false;
        }

        const std::size_t removedFrequency = found->second->frequency;

        eraseNodeFromBucket(found->second);
        nodeIndex_.erase(found);

        if (nodeIndex_.empty()) {
            minFrequency_ = 0;
        } else if (
            removedFrequency == minFrequency_
            && frequencyBuckets_.find(removedFrequency) == frequencyBuckets_.end()) {
            recomputeMinFrequency();
        }

        return true;
    }

    void clear() override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        nodeIndex_.clear();
        frequencyBuckets_.clear();
        minFrequency_ = 0;
    }

    void purge()
    {
        clear();
    }

    [[nodiscard]] bool contains(const Key& key) const override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodeIndex_.find(key) != nodeIndex_.end();
    }

    [[nodiscard]] std::size_t size() const override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodeIndex_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept override
    {
        return capacity_;
    }

    [[nodiscard]] std::size_t maxFrequency() const noexcept
    {
        return maxFrequency_;
    }

    static constexpr std::size_t defaultMaxFrequency() noexcept
    {
        return 1U << 20U;
    }

private:
    struct CacheNode {
        Key key;
        Value value;
        std::size_t frequency;

        template <typename K, typename V>
        CacheNode(K&& k, V&& v, std::size_t f)
            : key(std::forward<K>(k))
            , value(std::forward<V>(v))
            , frequency(f)
        {
        }
    };

    using NodeList = std::list<CacheNode>;
    using NodeIterator = typename NodeList::iterator;

    using NodeIndex = std::unordered_map<Key, NodeIterator, Hash, KeyEqual>;
    using FrequencyBuckets = std::unordered_map<std::size_t, NodeList>;

private:
    void promote(NodeIterator node)
    {
        if (node->frequency >= maxFrequency_) {
            ageFrequenciesLocked();
        }

        const std::size_t oldFrequency = node->frequency;
        const std::size_t newFrequency = oldFrequency + 1;

        auto oldBucketIt = frequencyBuckets_.find(oldFrequency);
        auto& oldBucket = oldBucketIt->second;

        auto newBucketIt = frequencyBuckets_.try_emplace(newFrequency).first;
        auto& newBucket = newBucketIt->second;

        newBucket.splice(newBucket.begin(), oldBucket, node);
        node->frequency = newFrequency;

        if (oldBucket.empty()) {
            frequencyBuckets_.erase(oldFrequency);

            if (minFrequency_ == oldFrequency) {
                minFrequency_ = newFrequency;
            }
        }
    }

    void eraseNodeFromBucket(NodeIterator node)
    {
        const std::size_t frequency = node->frequency;

        auto bucketIt = frequencyBuckets_.find(frequency);
        bucketIt->second.erase(node);

        if (bucketIt->second.empty()) {
            frequencyBuckets_.erase(bucketIt);
        }
    }

    void evictOneLocked()
    {
        if (nodeIndex_.empty()) {
            minFrequency_ = 0;
            return;
        }

        auto bucketIt = frequencyBuckets_.find(minFrequency_);

        if (bucketIt == frequencyBuckets_.end() || bucketIt->second.empty()) {
            recomputeMinFrequency();
            bucketIt = frequencyBuckets_.find(minFrequency_);
        }

        auto& bucket = bucketIt->second;

        // 同频率下，链表头部是最近访问，尾部是最久未访问。
        auto victim = std::prev(bucket.end());

        nodeIndex_.erase(victim->key);
        bucket.erase(victim);

        if (bucket.empty()) {
            frequencyBuckets_.erase(bucketIt);
        }

        if (nodeIndex_.empty()) {
            minFrequency_ = 0;
        }
    }

    void ageFrequenciesLocked()
    {
        if (frequencyBuckets_.empty()) {
            minFrequency_ = 0;
            return;
        }

        std::vector<std::size_t> frequencies;
        frequencies.reserve(frequencyBuckets_.size());

        for (const auto& bucketPair : frequencyBuckets_) {
            frequencies.push_back(bucketPair.first);
        }

        std::sort(frequencies.begin(), frequencies.end());

        FrequencyBuckets agedBuckets;
        agedBuckets.reserve(frequencyBuckets_.size());

        // 先创建目标桶，降低迁移中途异常导致状态不一致的风险。
        for (const std::size_t frequency : frequencies) {
            const std::size_t agedFrequency =
                std::max<std::size_t>(1, frequency / 2);

            agedBuckets.try_emplace(agedFrequency);
        }

        for (const std::size_t frequency : frequencies) {
            auto sourceIt = frequencyBuckets_.find(frequency);
            if (sourceIt == frequencyBuckets_.end()) {
                continue;
            }

            const std::size_t agedFrequency =
                std::max<std::size_t>(1, frequency / 2);

            auto& source = sourceIt->second;
            auto& destination = agedBuckets.find(agedFrequency)->second;

            // 从旧桶尾部搬到新桶头部，可以保持原有 MRU -> LRU 的相对顺序。
            while (!source.empty()) {
                auto node = std::prev(source.end());

                node->frequency = agedFrequency;
                destination.splice(destination.begin(), source, node);
            }
        }

        frequencyBuckets_.swap(agedBuckets);
        recomputeMinFrequency();
    }

    void recomputeMinFrequency()
    {
        minFrequency_ = std::numeric_limits<std::size_t>::max();

        for (const auto& bucket : frequencyBuckets_) {
            minFrequency_ = std::min(minFrequency_, bucket.first);
        }

        if (frequencyBuckets_.empty()) {
            minFrequency_ = 0;
        }
    }

private:
    const std::size_t capacity_;
    const std::size_t maxFrequency_;

    std::size_t minFrequency_{0};

    NodeIndex nodeIndex_;
    FrequencyBuckets frequencyBuckets_;

    mutable std::mutex mutex_;
};

// ============================================================================
// 2. 分片 LFU 缓存
//    设计目标：
//    1. 将一把全局大锁拆成多把 shard 小锁
//    2. 每个 shard 内部是精确 LFU
//    3. 整体是分片近似 LFU，不是全局严格 LFU
//    4. 保留命中、未命中、淘汰统计
// ============================================================================
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class ShardedLFUCache final : public CachePolicy<Key, Value> {
public:
    explicit ShardedLFUCache(
        std::size_t capacity,
        std::size_t shardCount = std::thread::hardware_concurrency(),
        std::size_t maxFrequency =
            LFUCache<Key, Value, Hash, KeyEqual>::defaultMaxFrequency(),
        const Hash& hash = Hash{},
        const KeyEqual& equal = KeyEqual{})
        : capacity_(capacity)
        , hash_(hash)
        , shardCount_(normalizeShardCount(capacity, shardCount))
    {
        shards_.reserve(shardCount_);

        const std::size_t baseCapacity = capacity_ / shardCount_;
        const std::size_t extraCapacity = capacity_ % shardCount_;

        for (std::size_t index = 0; index < shardCount_; ++index) {
            const std::size_t shardCapacity =
                baseCapacity + (index < extraCapacity ? 1U : 0U);

            shards_.push_back(
                std::make_unique<Shard>(
                    shardCapacity,
                    maxFrequency,
                    hash,
                    equal));
        }
    }

    ~ShardedLFUCache() override = default;

    ShardedLFUCache(const ShardedLFUCache&) = delete;
    ShardedLFUCache& operator=(const ShardedLFUCache&) = delete;

    ShardedLFUCache(ShardedLFUCache&&) = delete;
    ShardedLFUCache& operator=(ShardedLFUCache&&) = delete;

    void put(const Key& key, const Value& value) override
    {
        const CacheWriteResult result = shardFor(key).putAndReport(key, value);

        if (result == CacheWriteResult::insertedWithEviction) {
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool get(const Key& key, Value& value) override
    {
        if (shardFor(key).get(key, value)) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override
    {
        return shardFor(key).peek(key, value);
    }

    bool erase(const Key& key) override
    {
        return shardFor(key).erase(key);
    }

    void clear() override
    {
        for (auto& shard : shards_) {
            shard->clear();
        }
    }

    void purge()
    {
        clear();
    }

    [[nodiscard]] bool contains(const Key& key) const override
    {
        return shardFor(key).contains(key);
    }

    [[nodiscard]] std::size_t size() const override
    {
        std::size_t total = 0;

        for (const auto& shard : shards_) {
            total += shard->size();
        }

        return total;
    }

    [[nodiscard]] std::size_t capacity() const noexcept override
    {
        return capacity_;
    }

    [[nodiscard]] std::size_t shardCount() const noexcept
    {
        return shardCount_;
    }

    [[nodiscard]] CacheStats stats() const noexcept override
    {
        return CacheStats{
            hits_.load(std::memory_order_relaxed),
            misses_.load(std::memory_order_relaxed),
            evictions_.load(std::memory_order_relaxed),
        };
    }

    void resetStats() noexcept override
    {
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
    }

private:
    using Shard = LFUCache<Key, Value, Hash, KeyEqual>;

private:
    static std::size_t normalizeShardCount(
        std::size_t capacity,
        std::size_t requestedShardCount) noexcept
    {
        if (capacity == 0) {
            return 1;
        }

        if (requestedShardCount == 0) {
            requestedShardCount = 1;
        }

        return std::min(capacity, requestedShardCount);
    }

    [[nodiscard]] std::size_t shardIndex(const Key& key) const
    {
        return hash_(key) % shardCount_;
    }

    Shard& shardFor(const Key& key)
    {
        return *shards_[shardIndex(key)];
    }

    const Shard& shardFor(const Key& key) const
    {
        return *shards_[shardIndex(key)];
    }

private:
    const std::size_t capacity_;
    const Hash hash_;
    const std::size_t shardCount_;

    std::vector<std::unique_ptr<Shard>> shards_;

    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> evictions_{0};
};

template <typename Key, typename Value>
using KLfuCache = LFUCache<Key, Value>;

template <typename Key, typename Value>
using KHashLfuCache = ShardedLFUCache<Key, Value>;

} // namespace KamaCache
