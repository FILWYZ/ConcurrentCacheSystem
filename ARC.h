#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CachePolicy.h"

namespace Cache {

struct ARCState {
    std::size_t recent{0};
    std::size_t frequent{0};
    std::size_t recentGhost{0};
    std::size_t frequentGhost{0};
    std::size_t recentTarget{0};

    [[nodiscard]] std::size_t resident() const noexcept { return recent + frequent; }
    [[nodiscard]] std::size_t ghost() const noexcept { return recentGhost + frequentGhost; }
    [[nodiscard]] std::size_t tracked() const noexcept { return resident() + ghost(); }
};

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class alignas(64) ARCCache final : public CachePolicy<Key, Value> {
public:
    enum class ListId { recent, frequent, recentGhost, frequentGhost };

    struct ResidentNode {
        Key key;
        Value value;
        ListId list;

        template <typename K, typename V>
        ResidentNode(K&& k, V&& v, ListId listId)
            : key(std::forward<K>(k)), value(std::forward<V>(v)), list(listId) {}
    };

    struct GhostNode {
        Key key;
        ListId list;

        template <typename K>
        GhostNode(K&& k, ListId listId) : key(std::forward<K>(k)), list(listId) {}
    };

    using ResidentList = std::list<ResidentNode>;
    using ResidentIterator = typename ResidentList::iterator;
    using ResidentIndex = std::unordered_map<Key, ResidentIterator, Hash, KeyEqual>;

    using GhostList = std::list<GhostNode>;
    using GhostIterator = typename GhostList::iterator;
    using GhostIndex = std::unordered_map<Key, GhostIterator, Hash, KeyEqual>;

public:
    explicit ARCCache(std::size_t capacity) : ARCCache(capacity, Hash{}, KeyEqual{}) {}

    ARCCache(std::size_t capacity, const Hash& hash, const KeyEqual& equal)
        : capacity_(capacity), residentIndex_(0, hash, equal), ghostIndex_(0, hash, equal) {
        residentIndex_.reserve(capacity_);
        ghostIndex_.reserve(capacity_);
    }

    void put(const Key& key, const Value& value) override {
        static_cast<void>(putAndReport(key, value));
    }

    template <typename K, typename V>
    CacheWriteResult putAndReport(K&& key, V&& value, std::uint64_t* outEvictions = nullptr) {
        if (capacity_ == 0) return CacheWriteResult::ignored;

        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t localEvictions = 0;

        auto resident = residentIndex_.find(key);
        if (resident != residentIndex_.end()) {
            resident->second->value = std::forward<V>(value);
            promoteToFrequent(resident->second);
            return CacheWriteResult::updated;
        }

        auto ghost = ghostIndex_.find(key);
        if (ghost != ghostIndex_.end()) {
            const Key stableKey = ghost->second->key;
            const bool frequentGhostHit = (ghost->second->list == ListId::frequentGhost);

            adaptTarget(frequentGhostHit);
            if (replace(frequentGhostHit)) ++localEvictions;

            insertResident(stableKey, std::forward<V>(value), ListId::frequent);

            auto ghostAfterReplace = ghostIndex_.find(stableKey);
            if (ghostAfterReplace != ghostIndex_.end()) eraseGhost(ghostAfterReplace);

            trimGhostHistory();
            if (outEvictions && localEvictions > 0) *outEvictions += localEvictions;
            return localEvictions > 0 ? CacheWriteResult::insertedWithEviction : CacheWriteResult::inserted;
        }

        const bool evicted = insertCold(std::forward<K>(key), std::forward<V>(value), localEvictions);
        trimGhostHistory();

        if (outEvictions && localEvictions > 0) *outEvictions += localEvictions;
        return evicted ? CacheWriteResult::insertedWithEviction : CacheWriteResult::inserted;
    }

    bool get(const Key& key, Value& value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = residentIndex_.find(key);
        if (found == residentIndex_.end()) return false;

        value = found->second->value;
        promoteToFrequent(found->second);
        return true;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = residentIndex_.find(key);
        if (found == residentIndex_.end()) return false;
        value = found->second->value;
        return true;
    }

    bool erase(const Key& key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto resident = residentIndex_.find(key);
        if (resident != residentIndex_.end()) {
            eraseResident(resident);
            return true;
        }
        auto ghost = ghostIndex_.find(key);
        if (ghost == ghostIndex_.end()) return false;
        eraseGhost(ghost);
        return true;
    }

    void clear() override {
        std::lock_guard<std::mutex> lock(mutex_);
        residentIndex_.clear();
        ghostIndex_.clear();
        recent_.clear();
        frequent_.clear();
        recentGhost_.clear();
        frequentGhost_.clear();
        recentTarget_ = 0;
    }

    [[nodiscard]] bool contains(const Key& key) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return residentIndex_.find(key) != residentIndex_.end();
    }

    [[nodiscard]] std::size_t size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return residentIndex_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }

private:
    ResidentList& residentListFor(ListId list) { return list == ListId::frequent ? frequent_ : recent_; }
    GhostList& ghostListFor(ListId list) { return list == ListId::frequentGhost ? frequentGhost_ : recentGhost_; }

    void promoteToFrequent(ResidentIterator node) {
        if (node->list == ListId::recent) {
            frequent_.splice(frequent_.begin(), recent_, node);
            node->list = ListId::frequent;
            return;
        }
        frequent_.splice(frequent_.begin(), frequent_, node);
    }

    void adaptTarget(bool frequentGhostHit) {
        if (frequentGhostHit) {
            const std::size_t divisor = std::max<std::size_t>(1, frequentGhost_.size());
            const std::size_t delta = std::max<std::size_t>(1, recentGhost_.size() / divisor);
            recentTarget_ = delta >= recentTarget_ ? 0 : recentTarget_ - delta;
            return;
        }
        const std::size_t divisor = std::max<std::size_t>(1, recentGhost_.size());
        const std::size_t delta = std::max<std::size_t>(1, frequentGhost_.size() / divisor);
        recentTarget_ = std::min(capacity_, recentTarget_ + delta);
    }

    bool replace(bool frequentGhostHit) {
        if (residentIndex_.empty()) return false;

        if (!recent_.empty() && 
           (recent_.size() > recentTarget_ || (frequentGhostHit && recent_.size() == recentTarget_))) {
            return moveResidentLruToGhost(ListId::recent, ListId::recentGhost);
        }
        if (!frequent_.empty()) return moveResidentLruToGhost(ListId::frequent, ListId::frequentGhost);
        if (!recent_.empty()) return moveResidentLruToGhost(ListId::recent, ListId::recentGhost);
        return false;
    }

    bool moveResidentLruToGhost(ListId residentList, ListId ghostList) {
        auto& source = residentListFor(residentList);
        if (source.empty()) return false;

        auto victim = std::prev(source.end());
        auto& destination = ghostListFor(ghostList);

        destination.emplace_front(victim->key, ghostList);
        auto ghostNode = destination.begin();

        try {
            ghostIndex_.emplace(ghostNode->key, ghostNode);
        } catch (...) {
            destination.pop_front();
            throw;
        }
        residentIndex_.erase(victim->key);
        source.erase(victim);
        return true;
    }

    template <typename K, typename V>
    bool insertCold(K&& key, V&& value, std::size_t& localEvictions) {
        bool evicted = false;
        const std::size_t recentSide = recent_.size() + recentGhost_.size();

        if (recentSide == capacity_) {
            if (recent_.size() < capacity_) {
                removeGhostLru(ListId::recentGhost);
                if (replace(false)) ++localEvictions;
                evicted = true;
            } else {
                evicted = removeResidentLru(ListId::recent);
                if (evicted) ++localEvictions;
            }
        } else {
            const std::size_t tracked = residentIndex_.size() + ghostIndex_.size();
            if (tracked >= capacity_) {
                if (tracked >= 2 * capacity_) {
                    if (!removeGhostLru(ListId::frequentGhost)) removeGhostLru(ListId::recentGhost);
                }
                if (replace(false)) ++localEvictions;
                evicted = true;
            }
        }
        insertResident(std::forward<K>(key), std::forward<V>(value), ListId::recent);
        return evicted;
    }

    template <typename K, typename V>
    void insertResident(K&& key, V&& value, ListId list) {
        auto& destination = residentListFor(list);
        destination.emplace_front(std::forward<K>(key), std::forward<V>(value), list);
        auto node = destination.begin();

        try {
            auto insertion = residentIndex_.emplace(node->key, node);
            if (!insertion.second) {
                insertion.first->second->value = std::move(node->value);
                destination.pop_front();
                promoteToFrequent(insertion.first->second);
            }
        } catch (...) {
            destination.pop_front();
            throw;
        }
    }

    bool removeResidentLru(ListId list) {
        auto& source = residentListFor(list);
        if (source.empty()) return false;
        auto victim = std::prev(source.end());
        residentIndex_.erase(victim->key);
        source.erase(victim);
        return true;
    }

    bool removeGhostLru(ListId list) {
        auto& source = ghostListFor(list);
        if (source.empty()) return false;
        auto victim = std::prev(source.end());
        ghostIndex_.erase(victim->key);
        source.erase(victim);
        return true;
    }

    void eraseResident(typename ResidentIndex::iterator resident) {
        auto node = resident->second;
        residentListFor(node->list).erase(node);
        residentIndex_.erase(resident);
    }

    void eraseGhost(typename GhostIndex::iterator ghost) {
        auto node = ghost->second;
        ghostListFor(node->list).erase(node);
        ghostIndex_.erase(ghost);
    }

    void trimGhostHistory() {
        while (ghostIndex_.size() > capacity_) {
            if (!removeGhostLru(ListId::frequentGhost)) removeGhostLru(ListId::recentGhost);
        }
    }

private:
    const std::size_t capacity_;
    std::size_t recentTarget_{0};

    ResidentList recent_;
    ResidentList frequent_;
    GhostList recentGhost_;
    GhostList frequentGhost_;

    ResidentIndex residentIndex_;
    GhostIndex ghostIndex_;
    mutable std::mutex mutex_;
};

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class ShardedARCCache final : public CachePolicy<Key, Value> {
public:
    explicit ShardedARCCache(
        std::size_t capacity,
        std::size_t shardCount = std::thread::hardware_concurrency(),
        const Hash& hash = Hash{},
        const KeyEqual& equal = KeyEqual{})
        : capacity_(capacity), hash_(hash), shardCount_(normalizeShardCount(capacity, shardCount)) {
        shards_.reserve(shardCount_);
        const std::size_t baseCapacity = capacity_ / shardCount_;
        const std::size_t extraCapacity = capacity_ % shardCount_;

        for (std::size_t index = 0; index < shardCount_; ++index) {
            const std::size_t shardCapacity = baseCapacity + (index < extraCapacity ? 1U : 0U);
            shards_.push_back(std::make_unique<Shard>(shardCapacity, hash, equal));
        }
    }

    void put(const Key& key, const Value& value) override {
        std::uint64_t evictionsAccumulator = 0;
        if (shardFor(key).putAndReport(key, value, &evictionsAccumulator) == CacheWriteResult::insertedWithEviction || evictionsAccumulator > 0) {
            evictions_.fetch_add(evictionsAccumulator > 0 ? evictionsAccumulator : 1, std::memory_order_relaxed);
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
    using Shard = ARCCache<Key, Value, Hash, KeyEqual>;

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