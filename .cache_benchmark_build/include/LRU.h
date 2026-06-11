#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KamaCache
{

// ============================================================================
// 1. 单分片 LRU 缓存
//    设计目标：O(1) 查找、O(1) 提权、O(1) 淘汰
//    数据结构：std::list + std::unordered_map<Key, list::iterator>
// ============================================================================
template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class alignas(64) KLruCacheShard
{
public:
    struct CacheNode {
        Key key;
        Value value;

        template<typename K, typename V>
        CacheNode(K&& k, V&& v)
            : key(std::forward<K>(k))
            , value(std::forward<V>(v))
        {}
    };

    using ListType = std::list<CacheNode>;
    using ListIter = typename ListType::iterator;
    using MapType  = std::unordered_map<Key, ListIter, Hash, KeyEqual>;

public:
    explicit KLruCacheShard(std::size_t capacity)
        : capacity_(capacity)
    {
        cacheMap_.reserve(capacity_);
    }

    ~KLruCacheShard() = default;

    KLruCacheShard(const KLruCacheShard&) = delete;
    KLruCacheShard& operator=(const KLruCacheShard&) = delete;

    KLruCacheShard(KLruCacheShard&&) = delete;
    KLruCacheShard& operator=(KLruCacheShard&&) = delete;

    bool get(const Key& key, Value& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cacheMap_.find(key);
        if (it == cacheMap_.end()) {
            return false;
        }

        // LRU 语义：命中后移动到头部
        cacheList_.splice(cacheList_.begin(), cacheList_, it->second);
        value = it->second->value;

        return true;
    }

    bool peek(const Key& key, Value& value) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cacheMap_.find(key);
        if (it == cacheMap_.end()) {
            return false;
        }

        // peek 只读 value，不改变 LRU 顺序
        value = it->second->value;
        return true;
    }

    template<typename V>
    bool updateIfExists(const Key& key, V&& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cacheMap_.find(key);
        if (it == cacheMap_.end()) {
            return false;
        }

        it->second->value = std::forward<V>(value);
        cacheList_.splice(cacheList_.begin(), cacheList_, it->second);

        return true;
    }

    template<typename K, typename V>
    void put(K&& key, V&& value)
    {
        if (capacity_ == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cacheMap_.find(key);
        if (it != cacheMap_.end()) {
            it->second->value = std::forward<V>(value);
            cacheList_.splice(cacheList_.begin(), cacheList_, it->second);
            return;
        }

        if (cacheList_.size() >= capacity_) {
            auto& backNode = cacheList_.back();
            cacheMap_.erase(backNode.key);
            cacheList_.pop_back();
        }

        cacheList_.emplace_front(std::forward<K>(key), std::forward<V>(value));

        try {
            cacheMap_.emplace(cacheList_.front().key, cacheList_.begin());
        } catch (...) {
            cacheList_.pop_front();
            throw;
        }
    }

    bool remove(const Key& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cacheMap_.find(key);
        if (it == cacheMap_.end()) {
            return false;
        }

        cacheList_.erase(it->second);
        cacheMap_.erase(it);

        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cacheList_.clear();
        cacheMap_.clear();
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cacheList_.size();
    }

    std::size_t capacity() const noexcept
    {
        return capacity_;
    }

private:
    const std::size_t capacity_;

    ListType cacheList_;
    MapType cacheMap_;

    mutable std::mutex mutex_;
};


// ============================================================================
// 2. 分片 LRU 缓存
//    设计目标：将一把大锁拆成多把小锁，降低高并发访问下的锁竞争
//    注意：这是分片加锁，不是 lock-free
// ============================================================================
template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class KHighConcurrencyCache
{
public:
    explicit KHighConcurrencyCache(std::size_t totalCapacity, int shardNum = 0)
        : shardNum_(normalizeShardNum(shardNum))
    {
        const std::size_t shardCapacity = ceilDiv(totalCapacity, shardNum_);

        shards_.reserve(shardNum_);

        for (std::size_t i = 0; i < shardNum_; ++i) {
            shards_.emplace_back(
                std::make_unique<ShardType>(shardCapacity)
            );
        }
    }

    KHighConcurrencyCache(const KHighConcurrencyCache&) = delete;
    KHighConcurrencyCache& operator=(const KHighConcurrencyCache&) = delete;

    template<typename K, typename V>
    void put(K&& key, V&& value)
    {
        const std::size_t index = getShardIndex(key);
        shards_[index]->put(std::forward<K>(key), std::forward<V>(value));
    }

    bool get(const Key& key, Value& value)
    {
        return shards_[getShardIndex(key)]->get(key, value);
    }

    bool peek(const Key& key, Value& value) const
    {
        return shards_[getShardIndex(key)]->peek(key, value);
    }

    template<typename V>
    bool updateIfExists(const Key& key, V&& value)
    {
        return shards_[getShardIndex(key)]->updateIfExists(
            key,
            std::forward<V>(value)
        );
    }

    bool remove(const Key& key)
    {
        return shards_[getShardIndex(key)]->remove(key);
    }

    void clear()
    {
        for (auto& shard : shards_) {
            shard->clear();
        }
    }

    std::size_t size() const
    {
        std::size_t total = 0;

        for (const auto& shard : shards_) {
            total += shard->size();
        }

        return total;
    }

    std::size_t shardNum() const noexcept
    {
        return shardNum_;
    }

private:
    using ShardType = KLruCacheShard<Key, Value, Hash, KeyEqual>;

    static std::size_t normalizeShardNum(int shardNum)
    {
        if (shardNum > 0) {
            return static_cast<std::size_t>(shardNum);
        }

        const unsigned int hardwareNum = std::thread::hardware_concurrency();

        if (hardwareNum == 0) {
            return 1;
        }

        return static_cast<std::size_t>(hardwareNum);
    }

    static std::size_t ceilDiv(std::size_t a, std::size_t b)
    {
        return (a + b - 1) / b;
    }

    std::size_t getShardIndex(const Key& key) const
    {
        return hash_(key) % shardNum_;
    }

private:
    const std::size_t shardNum_;

    Hash hash_;

    std::vector<std::unique_ptr<ShardType>> shards_;
};


// ============================================================================
// 3. LRU-K 历史区分片
//    设计目标：把 count 和 value 放在同一个节点里，避免两个容器之间的数据不一致
// ============================================================================
enum class KHistoryAccessResult
{
    Miss,
    HitHistory,
    Promoted
};

template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class KHistoryShard
{
public:
    struct HistoryNode {
        Key key;
        Value value;
        std::size_t count;

        template<typename K, typename V>
        HistoryNode(K&& k, V&& v, std::size_t c)
            : key(std::forward<K>(k))
            , value(std::forward<V>(v))
            , count(c)
        {}
    };

    using ListType = std::list<HistoryNode>;
    using ListIter = typename ListType::iterator;
    using MapType  = std::unordered_map<Key, ListIter, Hash, KeyEqual>;

public:
    explicit KHistoryShard(std::size_t capacity)
        : capacity_(capacity)
    {
        historyMap_.reserve(capacity_);
    }

    KHistoryShard(const KHistoryShard&) = delete;
    KHistoryShard& operator=(const KHistoryShard&) = delete;

    KHistoryShard(KHistoryShard&&) = delete;
    KHistoryShard& operator=(KHistoryShard&&) = delete;

    KHistoryAccessResult get(const Key& key, Value& value, std::size_t k)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = historyMap_.find(key);
        if (it == historyMap_.end()) {
            return KHistoryAccessResult::Miss;
        }

        HistoryNode& node = *(it->second);

        ++node.count;
        value = node.value;

        if (node.count >= k) {
            historyList_.erase(it->second);
            historyMap_.erase(it);
            return KHistoryAccessResult::Promoted;
        }

        historyList_.splice(historyList_.begin(), historyList_, it->second);
        return KHistoryAccessResult::HitHistory;
    }

    template<typename V>
    std::optional<Value> put(const Key& key, V&& value, std::size_t k)
    {
        if (capacity_ == 0) {
            return std::nullopt;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = historyMap_.find(key);
        if (it != historyMap_.end()) {
            HistoryNode& node = *(it->second);

            node.value = std::forward<V>(value);
            ++node.count;

            if (node.count >= k) {
                std::optional<Value> promotedValue = node.value;

                historyList_.erase(it->second);
                historyMap_.erase(it);

                return promotedValue;
            }

            historyList_.splice(historyList_.begin(), historyList_, it->second);
            return std::nullopt;
        }

        if (historyList_.size() >= capacity_) {
            auto& backNode = historyList_.back();
            historyMap_.erase(backNode.key);
            historyList_.pop_back();
        }

        historyList_.emplace_front(key, std::forward<V>(value), 1);

        try {
            historyMap_.emplace(historyList_.front().key, historyList_.begin());
        } catch (...) {
            historyList_.pop_front();
            throw;
        }

        return std::nullopt;
    }

    bool remove(const Key& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = historyMap_.find(key);
        if (it == historyMap_.end()) {
            return false;
        }

        historyList_.erase(it->second);
        historyMap_.erase(it);

        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        historyList_.clear();
        historyMap_.clear();
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return historyList_.size();
    }

private:
    const std::size_t capacity_;

    ListType historyList_;
    MapType historyMap_;

    mutable std::mutex mutex_;
};


// ============================================================================
// 4. 分片 LRU-K 历史缓存
// ============================================================================
template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class KShardedHistoryCache
{
public:
    explicit KShardedHistoryCache(std::size_t totalCapacity, int shardNum = 0)
        : shardNum_(normalizeShardNum(shardNum))
    {
        const std::size_t shardCapacity = ceilDiv(totalCapacity, shardNum_);

        shards_.reserve(shardNum_);

        for (std::size_t i = 0; i < shardNum_; ++i) {
            shards_.emplace_back(
                std::make_unique<ShardType>(shardCapacity)
            );
        }
    }

    KHistoryAccessResult get(const Key& key, Value& value, std::size_t k)
    {
        return shards_[getShardIndex(key)]->get(key, value, k);
    }

    template<typename V>
    std::optional<Value> put(const Key& key, V&& value, std::size_t k)
    {
        return shards_[getShardIndex(key)]->put(
            key,
            std::forward<V>(value),
            k
        );
    }

    bool remove(const Key& key)
    {
        return shards_[getShardIndex(key)]->remove(key);
    }

    void clear()
    {
        for (auto& shard : shards_) {
            shard->clear();
        }
    }

    std::size_t size() const
    {
        std::size_t total = 0;

        for (const auto& shard : shards_) {
            total += shard->size();
        }

        return total;
    }

private:
    using ShardType = KHistoryShard<Key, Value, Hash, KeyEqual>;

    static std::size_t normalizeShardNum(int shardNum)
    {
        if (shardNum > 0) {
            return static_cast<std::size_t>(shardNum);
        }

        const unsigned int hardwareNum = std::thread::hardware_concurrency();

        if (hardwareNum == 0) {
            return 1;
        }

        return static_cast<std::size_t>(hardwareNum);
    }

    static std::size_t ceilDiv(std::size_t a, std::size_t b)
    {
        return (a + b - 1) / b;
    }

    std::size_t getShardIndex(const Key& key) const
    {
        return hash_(key) % shardNum_;
    }

private:
    const std::size_t shardNum_;

    Hash hash_;

    std::vector<std::unique_ptr<ShardType>> shards_;
};


// ============================================================================
// 5. 升级版 LRU-K 缓存
//    设计目标：
//    1. 主缓存只放热点数据
//    2. 历史区记录未达到 K 次访问的数据
//    3. get / put 都参与访问计数
//    4. count 和 value 在同一个历史节点中维护，避免跨容器不一致
// ============================================================================
template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class KLruKCache
{
public:
    KLruKCache(
        std::size_t capacity,
        std::size_t historyCapacity,
        std::size_t k,
        int shardNum = 0
    )
        : k_(checkK(k))
        , mainCache_(capacity, shardNum)
        , historyCache_(historyCapacity, shardNum)
    {
        if (k_ > 1 && historyCapacity == 0) {
            throw std::invalid_argument(
                "historyCapacity must be greater than 0 when k > 1"
            );
        }
    }

    KLruKCache(const KLruKCache&) = delete;
    KLruKCache& operator=(const KLruKCache&) = delete;

    bool get(const Key& key, Value& value)
    {
        if (mainCache_.get(key, value)) {
            return true;
        }

        if (k_ == 1) {
            return false;
        }

        const KHistoryAccessResult result = historyCache_.get(key, value, k_);

        if (result == KHistoryAccessResult::Miss) {
            return false;
        }

        if (result == KHistoryAccessResult::Promoted) {
            mainCache_.put(key, value);
        }

        return true;
    }

    template<typename V>
    void put(const Key& key, V&& value)
    {
        if (mainCache_.updateIfExists(key, value)) {
            return;
        }

        if (k_ == 1) {
            mainCache_.put(key, std::forward<V>(value));
            return;
        }

        std::optional<Value> promotedValue =
            historyCache_.put(key, std::forward<V>(value), k_);

        if (promotedValue.has_value()) {
            mainCache_.put(key, std::move(*promotedValue));
        }
    }

    bool remove(const Key& key)
    {
        const bool removedFromMain = mainCache_.remove(key);
        const bool removedFromHistory = historyCache_.remove(key);

        return removedFromMain || removedFromHistory;
    }

    void clear()
    {
        mainCache_.clear();
        historyCache_.clear();
    }

    std::size_t mainSize() const
    {
        return mainCache_.size();
    }

    std::size_t historySize() const
    {
        return historyCache_.size();
    }

    std::size_t k() const noexcept
    {
        return k_;
    }

private:
    static std::size_t checkK(std::size_t k)
    {
        if (k == 0) {
            throw std::invalid_argument("k must be greater than 0");
        }

        return k;
    }

private:
    const std::size_t k_;

    KHighConcurrencyCache<Key, Value, Hash, KeyEqual> mainCache_;
    KShardedHistoryCache<Key, Value, Hash, KeyEqual> historyCache_;
};

} // namespace KamaCache