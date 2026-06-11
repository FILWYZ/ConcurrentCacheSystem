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

namespace KamaCache {

struct ARCState {
    std::size_t recent{0};
    std::size_t frequent{0};
    std::size_t recentGhost{0};
    std::size_t frequentGhost{0};
    std::size_t recentTarget{0};

    [[nodiscard]] std::size_t resident() const noexcept
    {
        return recent + frequent;
    }

    [[nodiscard]] std::size_t ghost() const noexcept
    {
        return recentGhost + frequentGhost;
    }

    [[nodiscard]] std::size_t tracked() const noexcept
    {
        return resident() + ghost();
    }
};

// ============================================================================
// 1. 单分片 ARC 缓存
//
//    ARC = Adaptive Replacement Cache
//
//    四个核心队列：
//    1. recent_        : T1，最近只访问过一次的真实缓存数据
//    2. frequent_      : T2，访问过至少两次的真实缓存数据
//    3. recentGhost_   : B1，recent_ 被淘汰后的 key 历史，不保存 value
//    4. frequentGhost_ : B2，frequent_ 被淘汰后的 key 历史，不保存 value
//
//    recentTarget_ 对应 ARC 论文中的 p：
//    - B1 命中：说明 recent 区太小，增大 recentTarget_
//    - B2 命中：说明 frequent 区太小，减小 recentTarget_
//
//    工程优化：
//    1. resident 节点使用 list<ResidentNode> 保存 key/value/listId
//    2. residentIndex_ 只保存 Key -> ResidentIterator
//    3. ghost 节点使用 list<GhostNode> 保存 key/listId
//    4. ghostIndex_ 只保存 Key -> GhostIterator
//    5. 节点迁移使用 list::splice，避免重复构造 key/value
//    6. 分片内一把 mutex，保证单 shard 状态一致
// ============================================================================
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
class alignas(64) ARCCache final : public CachePolicy<Key, Value> {
public:
    explicit ARCCache(std::size_t capacity)
        : ARCCache(capacity, Hash{}, KeyEqual{})
    {
    }

    ARCCache(std::size_t capacity, const Hash& hash, const KeyEqual& equal)
        : capacity_(capacity)
        , residentIndex_(0, hash, equal)
        , ghostIndex_(0, hash, equal)
    {
        residentIndex_.reserve(capacity_);
        ghostIndex_.reserve(capacity_);
    }

    ~ARCCache() override = default;

    ARCCache(const ARCCache&) = delete;
    ARCCache& operator=(const ARCCache&) = delete;

    ARCCache(ARCCache&&) = delete;
    ARCCache& operator=(ARCCache&&) = delete;

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

        auto resident = residentIndex_.find(key);
        if (resident != residentIndex_.end()) {
            resident->second->value = std::forward<V>(value);
            promoteToFrequent(resident->second);
            return CacheWriteResult::updated;
        }

        auto ghost = ghostIndex_.find(key);
        if (ghost != ghostIndex_.end()) {
            const Key stableKey = ghost->second->key;
            const bool frequentGhostHit =
                ghost->second->list == ListId::frequentGhost;

            adaptTarget(frequentGhostHit);
            const bool evicted = replace(frequentGhostHit);

            // replace() 可能向 ghostIndex_ 插入新元素并触发 rehash，
            // 因此这里必须重新查找，不能继续使用 replace() 之前的 ghost 迭代器。
            insertResident(stableKey, std::forward<V>(value), ListId::frequent);

            auto ghostAfterReplace = ghostIndex_.find(stableKey);
            if (ghostAfterReplace != ghostIndex_.end()) {
                eraseGhost(ghostAfterReplace);
            }

            trimGhostHistory();

            return evicted
                ? CacheWriteResult::insertedWithEviction
                : CacheWriteResult::inserted;
        }

        const bool evicted = insertCold(
            std::forward<K>(key),
            std::forward<V>(value));

        trimGhostHistory();

        return evicted
            ? CacheWriteResult::insertedWithEviction
            : CacheWriteResult::inserted;
    }

    bool get(const Key& key, Value& value) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto found = residentIndex_.find(key);
        if (found == residentIndex_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        value = found->second->value;
        promoteToFrequent(found->second);

        hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    using CachePolicy<Key, Value>::get;

    [[nodiscard]] bool peek(const Key& key, Value& value) const override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto found = residentIndex_.find(key);
        if (found == residentIndex_.end()) {
            return false;
        }

        value = found->second->value;
        return true;
    }

    bool erase(const Key& key) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto resident = residentIndex_.find(key);
        if (resident != residentIndex_.end()) {
            eraseResident(resident);
            return true;
        }

        auto ghost = ghostIndex_.find(key);
        if (ghost == ghostIndex_.end()) {
            return false;
        }

        eraseGhost(ghost);
        return true;
    }

    void clear() override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        residentIndex_.clear();
        ghostIndex_.clear();

        recent_.clear();
        frequent_.clear();
        recentGhost_.clear();
        frequentGhost_.clear();

        recentTarget_ = 0;
    }

    void purge()
    {
        clear();
    }

    [[nodiscard]] bool contains(const Key& key) const override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return residentIndex_.find(key) != residentIndex_.end();
    }

    [[nodiscard]] bool isGhost(const Key& key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return ghostIndex_.find(key) != ghostIndex_.end();
    }

    [[nodiscard]] std::size_t size() const override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return residentIndex_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept override
    {
        return capacity_;
    }

    [[nodiscard]] ARCState state() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        return ARCState{
            recent_.size(),
            frequent_.size(),
            recentGhost_.size(),
            frequentGhost_.size(),
            recentTarget_,
        };
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
    enum class ListId {
        recent,
        frequent,
        recentGhost,
        frequentGhost,
    };

    struct ResidentNode {
        Key key;
        Value value;
        ListId list;

        template <typename K, typename V>
        ResidentNode(K&& k, V&& v, ListId listId)
            : key(std::forward<K>(k))
            , value(std::forward<V>(v))
            , list(listId)
        {
        }
    };

    struct GhostNode {
        Key key;
        ListId list;

        template <typename K>
        GhostNode(K&& k, ListId listId)
            : key(std::forward<K>(k))
            , list(listId)
        {
        }
    };

    using ResidentList = std::list<ResidentNode>;
    using ResidentIterator = typename ResidentList::iterator;
    using ResidentIndex = std::unordered_map<Key, ResidentIterator, Hash, KeyEqual>;
    using ResidentIndexIterator = typename ResidentIndex::iterator;

    using GhostList = std::list<GhostNode>;
    using GhostIterator = typename GhostList::iterator;
    using GhostIndex = std::unordered_map<Key, GhostIterator, Hash, KeyEqual>;
    using GhostIndexIterator = typename GhostIndex::iterator;

private:
    ResidentList& residentListFor(ListId list)
    {
        return list == ListId::frequent ? frequent_ : recent_;
    }

    GhostList& ghostListFor(ListId list)
    {
        return list == ListId::frequentGhost ? frequentGhost_ : recentGhost_;
    }

    void promoteToFrequent(ResidentIterator node)
    {
        if (node->list == ListId::recent) {
            frequent_.splice(frequent_.begin(), recent_, node);
            node->list = ListId::frequent;
            return;
        }

        frequent_.splice(frequent_.begin(), frequent_, node);
    }

    void adaptTarget(bool frequentGhostHit)
    {
        if (frequentGhostHit) {
            const std::size_t divisor =
                std::max<std::size_t>(1, frequentGhost_.size());

            const std::size_t delta =
                std::max<std::size_t>(1, recentGhost_.size() / divisor);

            recentTarget_ = delta >= recentTarget_ ? 0 : recentTarget_ - delta;
            return;
        }

        const std::size_t divisor =
            std::max<std::size_t>(1, recentGhost_.size());

        const std::size_t delta =
            std::max<std::size_t>(1, frequentGhost_.size() / divisor);

        recentTarget_ = std::min(capacity_, recentTarget_ + delta);
    }

    bool replace(bool frequentGhostHit)
    {
        if (residentIndex_.empty()) {
            return false;
        }

        if (!recent_.empty()
            && (recent_.size() > recentTarget_
                || (frequentGhostHit && recent_.size() == recentTarget_))) {
            return moveResidentLruToGhost(ListId::recent, ListId::recentGhost);
        }

        if (!frequent_.empty()) {
            return moveResidentLruToGhost(ListId::frequent, ListId::frequentGhost);
        }

        if (!recent_.empty()) {
            return moveResidentLruToGhost(ListId::recent, ListId::recentGhost);
        }

        return false;
    }

    bool moveResidentLruToGhost(ListId residentList, ListId ghostList)
    {
        auto& source = residentListFor(residentList);
        if (source.empty()) {
            return false;
        }

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

        evictions_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    template <typename K, typename V>
    bool insertCold(K&& key, V&& value)
    {
        bool evicted = false;

        const std::size_t recentSide = recent_.size() + recentGhost_.size();

        if (recentSide == capacity_) {
            if (recent_.size() < capacity_) {
                removeGhostLru(ListId::recentGhost);
                evicted = replace(false);
            } else {
                evicted = removeResidentLru(ListId::recent);
            }
        } else {
            const std::size_t tracked =
                residentIndex_.size() + ghostIndex_.size();

            if (tracked >= capacity_) {
                if (tracked >= 2 * capacity_) {
                    if (!removeGhostLru(ListId::frequentGhost)) {
                        removeGhostLru(ListId::recentGhost);
                    }
                }

                evicted = replace(false);
            }
        }

        insertResident(std::forward<K>(key), std::forward<V>(value), ListId::recent);

        return evicted;
    }

    template <typename K, typename V>
    void insertResident(K&& key, V&& value, ListId list)
    {
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

    bool removeResidentLru(ListId list)
    {
        auto& source = residentListFor(list);
        if (source.empty()) {
            return false;
        }

        auto victim = std::prev(source.end());

        residentIndex_.erase(victim->key);
        source.erase(victim);

        evictions_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool removeGhostLru(ListId list)
    {
        auto& source = ghostListFor(list);
        if (source.empty()) {
            return false;
        }

        auto victim = std::prev(source.end());

        ghostIndex_.erase(victim->key);
        source.erase(victim);

        return true;
    }

    void eraseResident(ResidentIndexIterator resident)
    {
        auto node = resident->second;

        residentListFor(node->list).erase(node);
        residentIndex_.erase(resident);
    }

    void eraseGhost(GhostIndexIterator ghost)
    {
        auto node = ghost->second;

        ghostListFor(node->list).erase(node);
        ghostIndex_.erase(ghost);
    }

    void trimGhostHistory()
    {
        while (ghostIndex_.size() > capacity_) {
            if (!removeGhostLru(ListId::frequentGhost)) {
                removeGhostLru(ListId::recentGhost);
            }
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

    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> evictions_{0};
};

// ============================================================================
// 2. 分片 ARC 缓存
//
//    设计目标：
//    1. 将全局大锁拆成多把 shard 小锁
//    2. 每个 shard 内部是完整 ARC
//    3. 整体是分片近似 ARC，不是全局严格 ARC
//    4. 保留统一统计接口
// ============================================================================
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
                std::make_unique<Shard>(shardCapacity, hash, equal));
        }
    }

    ~ShardedARCCache() override = default;

    ShardedARCCache(const ShardedARCCache&) = delete;
    ShardedARCCache& operator=(const ShardedARCCache&) = delete;

    ShardedARCCache(ShardedARCCache&&) = delete;
    ShardedARCCache& operator=(ShardedARCCache&&) = delete;

    void put(const Key& key, const Value& value) override
    {
        shardFor(key).put(key, value);
    }

    bool get(const Key& key, Value& value) override
    {
        return shardFor(key).get(key, value);
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

    [[nodiscard]] bool isGhost(const Key& key) const
    {
        return shardFor(key).isGhost(key);
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
        CacheStats total;

        for (const auto& shard : shards_) {
            const CacheStats shardStats = shard->stats();

            total.hits += shardStats.hits;
            total.misses += shardStats.misses;
            total.evictions += shardStats.evictions;
        }

        return total;
    }

    void resetStats() noexcept override
    {
        for (auto& shard : shards_) {
            shard->resetStats();
        }
    }

    [[nodiscard]] std::vector<ARCState> shardStates() const
    {
        std::vector<ARCState> states;
        states.reserve(shards_.size());

        for (const auto& shard : shards_) {
            states.push_back(shard->state());
        }

        return states;
    }

private:
    using Shard = ARCCache<Key, Value, Hash, KeyEqual>;

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
};

template <typename Key, typename Value>
using KArcCache = ARCCache<Key, Value>;

template <typename Key, typename Value>
using KHashArcCache = ShardedARCCache<Key, Value>;

} // namespace KamaCache
