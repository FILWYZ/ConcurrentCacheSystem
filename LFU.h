#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map> // 引入 map 优化频次桶
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CachePolicy.h"

namespace Cache {

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class alignas(64) LFUCache final : public CachePolicy<Key, Value> {
public:
    struct CacheNode {
        Key key;
        Value value;
        std::size_t frequency;
        std::size_t lastTouchEpoch; 

        template <typename K, typename V>
        CacheNode(K&& k, V&& v, std::size_t f, std::size_t epoch)
            : key(std::forward<K>(k)), value(std::forward<V>(v)), frequency(f), lastTouchEpoch(epoch) {}
    };

    using NodeList = std::list<CacheNode>;
    using NodeIterator = typename NodeList::iterator;
    using NodeIndex = std::unordered_map<Key, NodeIterator, Hash, KeyEqual>;
    
    // 关键优化：使用基于红黑树的 std::map，依靠天然有序性，begin() 永远是最小频次桶
    using FrequencyBuckets = std::map<std::size_t, NodeList>;

public:
    explicit LFUCache(std::size_t capacity, std::size_t maxFrequency = defaultMaxFrequency())
        : LFUCache(capacity, maxFrequency, Hash{}, KeyEqual{}) {}

    LFUCache(std::size_t capacity, std::size_t maxFrequency, const Hash& hash, const KeyEqual& equal)
        : capacity_(capacity)
        , maxFrequency_(std::max<std::size_t>(2, maxFrequency))
        , nodeIndex_(0, hash, equal) {
        nodeIndex_.reserve(capacity_);
    }

    void put(const Key& key, const Value& value) override {
        static_cast<void>(putAndReport(key, value));
    }

    template <typename K, typename V>
    CacheWriteResult putAndReport(K&& key, V&& value) {
        if (capacity_ == 0) return CacheWriteResult::ignored;

        std::lock_guard<std::mutex> lock(mutex_);
        ++globalEpoch_; 

        auto found = nodeIndex_.find(key);
        if (found != nodeIndex_.end()) {
            found->second->value = std::forward<V>(value);
            applyLazyAgeingLocked(found->second);
            promote(found->second);
            return CacheWriteResult::updated;
        }

        const bool needsEviction = nodeIndex_.size() >= capacity_;
        if (needsEviction) evictOneLocked();

        auto& bucket = frequencyBuckets_[1];
        bucket.emplace_front(std::forward<K>(key), std::forward<V>(value), 1, globalEpoch_);
        auto insertedNode = bucket.begin();

        try {
            auto insertion = nodeIndex_.emplace(insertedNode->key, insertedNode);
            if (!insertion.second) {
                insertion.first->second->value = std::move(insertedNode->value);
                bucket.erase(insertedNode);
                if (bucket.empty()) frequencyBuckets_.erase(1);
                applyLazyAgeingLocked(insertion.first->second);
                promote(insertion.first->second);
                return CacheWriteResult::updated;
            }
        } catch (...) {
            bucket.erase(insertedNode);
            if (bucket.empty()) frequencyBuckets_.erase(1);
            throw;
        }

        return needsEviction ? CacheWriteResult::insertedWithEviction : CacheWriteResult::inserted;
    }

    bool get(const Key& key, Value& value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) return false;

        applyLazyAgeingLocked(found->second);
        value = found->second->value;
        promote(found->second);
        return true;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) return false;
        value = found->second->value;
        return true;
    }

    bool erase(const Key& key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = nodeIndex_.find(key);
        if (found == nodeIndex_.end()) return false;

        eraseNodeFromBucket(found->second);
        nodeIndex_.erase(found);
        return true;
    }

    void clear() override {
        std::lock_guard<std::mutex> lock(mutex_);
        nodeIndex_.clear();
        frequencyBuckets_.clear();
        globalEpoch_ = 0;
    }

    [[nodiscard]] bool contains(const Key& key) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodeIndex_.find(key) != nodeIndex_.end();
    }

    [[nodiscard]] std::size_t size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodeIndex_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
    [[nodiscard]] std::size_t maxFrequency() const noexcept { return maxFrequency_; }
    static constexpr std::size_t defaultMaxFrequency() noexcept { return 1U << 20U; }

private:
    void applyLazyAgeingLocked(NodeIterator node) {
        if (globalEpoch_ <= node->lastTouchEpoch) return;

        const std::size_t epochDiff = globalEpoch_ - node->lastTouchEpoch;
        const std::size_t ageingThreshold = capacity_ * 4;

        if (epochDiff >= ageingThreshold && node->frequency > 1) {
            const std::size_t oldFrequency = node->frequency;
            const std::size_t newFrequency = std::max<std::size_t>(1, oldFrequency / 2);

            if (oldFrequency != newFrequency) {
                auto oldBucketIt = frequencyBuckets_.find(oldFrequency);
                auto& oldBucket = oldBucketIt->second;

                auto newBucketIt = frequencyBuckets_.emplace(newFrequency, NodeList{}).first;
                auto& newBucket = newBucketIt->second;

                newBucket.splice(newBucket.begin(), oldBucket, node);
                node->frequency = newFrequency;

                if (oldBucket.empty()) frequencyBuckets_.erase(oldBucketIt);
            }
        }
        node->lastTouchEpoch = globalEpoch_;
    }

    void promote(NodeIterator node) {
        const std::size_t oldFrequency = node->frequency;
        if (oldFrequency >= maxFrequency_) {
            node->lastTouchEpoch = globalEpoch_;
            return;
        }

        const std::size_t newFrequency = oldFrequency + 1;
        auto oldBucketIt = frequencyBuckets_.find(oldFrequency);
        auto& oldBucket = oldBucketIt->second;

        auto newBucketIt = frequencyBuckets_.emplace(newFrequency, NodeList{}).first;
        auto& newBucket = newBucketIt->second;

        newBucket.splice(newBucket.begin(), oldBucket, node);
        node->frequency = newFrequency;
        node->lastTouchEpoch = globalEpoch_;

        if (oldBucket.empty()) frequencyBuckets_.erase(oldBucketIt);
    }

    void eraseNodeFromBucket(NodeIterator node) {
        const std::size_t frequency = node->frequency;
        auto bucketIt = frequencyBuckets_.find(frequency);
        if (bucketIt != frequencyBuckets_.end()) {
            bucketIt->second.erase(node);
            if (bucketIt->second.empty()) frequencyBuckets_.erase(bucketIt);
        }
    }

    void evictOneLocked() {
        if (frequencyBuckets_.empty()) return;

        // 核心优化：std::map 的 begin() 永远是当前活着的最小频次桶，彻底告别 O(N) 重算
        auto bucketIt = frequencyBuckets_.begin(); 
        auto& bucket = bucketIt->second;
        auto victim = std::prev(bucket.end());

        nodeIndex_.erase(victim->key);
        bucket.erase(victim);

        if (bucket.empty()) frequencyBuckets_.erase(bucketIt);
    }

private:
    const std::size_t capacity_;
    const std::size_t maxFrequency_;
    std::size_t globalEpoch_{0}; 

    NodeIndex nodeIndex_;
    FrequencyBuckets frequencyBuckets_;
    mutable std::mutex mutex_;
};

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
        std::size_t maxFrequency = LFUCache<Key, Value, Hash, KeyEqual>::defaultMaxFrequency(),
        const Hash& hash = Hash{},
        const KeyEqual& equal = KeyEqual{})
        : capacity_(capacity), hash_(hash), shardCount_(normalizeShardCount(capacity, shardCount)) {
        shards_.reserve(shardCount_);
        const std::size_t baseCapacity = capacity_ / shardCount_;
        const std::size_t extraCapacity = capacity_ % shardCount_;

        for (std::size_t index = 0; index < shardCount_; ++index) {
            const std::size_t shardCapacity = baseCapacity + (index < extraCapacity ? 1U : 0U);
            shards_.push_back(std::make_unique<Shard>(shardCapacity, maxFrequency, hash, equal));
        }
    }

    void put(const Key& key, const Value& value) override {
        if (shardFor(key).putAndReport(key, value) == CacheWriteResult::insertedWithEviction) {
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool get(const Key& key, Value& value) override {
        if (shardFor(key).get(key, value)) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override { return shardFor(key).peek(key, value); }
    bool erase(const Key& key) override { return shardFor(key).erase(key); }
    void clear() override { for (auto& shard : shards_) shard->clear(); }
    [[nodiscard]] bool contains(const Key& key) const override { return shardFor(key).contains(key); }
    
    [[nodiscard]] std::size_t size() const override {
        std::size_t total = 0;
        for (const auto& shard : shards_) total += shard->size();
        return total;
    }

    [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
    [[nodiscard]] std::size_t shardCount() const noexcept { return shardCount_; }

    [[nodiscard]] CacheStats stats() const noexcept override {
        return CacheStats{
            hits_.load(std::memory_order_relaxed),
            misses_.load(std::memory_order_relaxed),
            evictions_.load(std::memory_order_relaxed),
        };
    }

    void resetStats() noexcept override {
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
    }

private:
    using Shard = LFUCache<Key, Value, Hash, KeyEqual>;

    static std::size_t normalizeShardCount(std::size_t capacity, std::size_t requestedShardCount) noexcept {
        if (capacity == 0) return 1;
        if (requestedShardCount == 0) requestedShardCount = 1;
        return std::min(capacity, requestedShardCount);
    }

    [[nodiscard]] std::size_t shardIndex(const Key& key) const { return hash_(key) % shardCount_; }
    Shard& shardFor(const Key& key) { return *shards_[shardIndex(key)]; }
    const Shard& shardFor(const Key& key) const { return *shards_[shardIndex(key)]; }

private:
    const std::size_t capacity_;
    const Hash hash_;
    const std::size_t shardCount_;
    std::vector<std::unique_ptr<Shard>> shards_;

    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> evictions_{0};
};

} // namespace Cache