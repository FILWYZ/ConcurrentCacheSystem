#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace KamaCache {

// ============================================================================
// 1. 写入结果
// ============================================================================
enum class CacheWriteResult {
    ignored,               // 容量为 0，或策略决定忽略本次写入
    inserted,              // 新插入，未发生淘汰
    updated,               // key 已存在，更新 value
    insertedWithEviction,  // 新插入，并触发淘汰
};

// ============================================================================
// 2. 缓存统计信息
// ============================================================================
struct CacheStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t evictions{0};

    [[nodiscard]] std::uint64_t requests() const noexcept
    {
        return hits + misses;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return requests() == 0 && evictions == 0;
    }

    [[nodiscard]] double hitRate() const noexcept
    {
        const std::uint64_t total = requests();

        return total == 0
            ? 0.0
            : static_cast<double>(hits) / static_cast<double>(total);
    }

    [[nodiscard]] double missRate() const noexcept
    {
        const std::uint64_t total = requests();

        return total == 0
            ? 0.0
            : static_cast<double>(misses) / static_cast<double>(total);
    }

    void reset() noexcept
    {
        hits = 0;
        misses = 0;
        evictions = 0;
    }
};

// ============================================================================
// 3. 缓存策略基类
//
//    设计原则：
//    1. get(key, value) 是核心接口，用 bool 表达是否命中。
//    2. tryGet(key) 是安全便捷接口，用 optional 避免 miss 和默认值混淆。
//    3. getOrDefault(key, defaultValue) 用于业务允许默认值兜底的场景。
//    4. peek(key, value) 默认不提供真实实现，具体缓存策略可覆盖。
//    5. stats/resetStats 提供统一监控接口，具体缓存策略可覆盖。
// ============================================================================
template <typename Key, typename Value>
class CachePolicy {
public:
    virtual ~CachePolicy() = default;

    CachePolicy() = default;

    CachePolicy(const CachePolicy&) = delete;
    CachePolicy& operator=(const CachePolicy&) = delete;

    CachePolicy(CachePolicy&&) = delete;
    CachePolicy& operator=(CachePolicy&&) = delete;

    // ------------------------------------------------------------------------
    // 写入接口
    //
    // 注意：
    // 由于这是虚函数接口，不能做完美转发版本：
    // template <typename K, typename V> put(K&&, V&&)
    //
    // 如果具体缓存实现需要移动语义，可以在派生类中额外提供模板 put。
    // ------------------------------------------------------------------------
    virtual void put(const Key& key, const Value& value) = 0;

    // ------------------------------------------------------------------------
    // 查询接口
    //
    // 命中返回 true，并通过 value 输出结果。
    // 未命中返回 false。
    //
    // 对 LRU / LFU 来说，get 通常会改变缓存状态：
    // LRU：刷新访问顺序
    // LFU：增加访问频率
    // ------------------------------------------------------------------------
    virtual bool get(const Key& key, Value& value) = 0;

    // ------------------------------------------------------------------------
    // 安全便捷查询接口
    //
    // 推荐业务代码优先使用 tryGet，而不是 get(key) 返回默认值。
    // ------------------------------------------------------------------------
    [[nodiscard]] std::optional<Value> tryGet(const Key& key)
    {
        Value value{};

        if (!get(key, value)) {
            return std::nullopt;
        }

        return value;
    }

    // ------------------------------------------------------------------------
    // 默认值兜底查询接口
    //
    // 适合业务明确允许默认值的场景。
    // ------------------------------------------------------------------------
    [[nodiscard]] Value getOrDefault(const Key& key, Value defaultValue = Value{})
    {
        Value value{};

        if (!get(key, value)) {
            return defaultValue;
        }

        return value;
    }

    // ------------------------------------------------------------------------
    // 只读查看接口
    //
    // peek 和 get 的区别：
    // get 可能改变缓存状态；
    // peek 不应该改变缓存状态。
    //
    // 默认实现返回 false。
    // 支持 peek 的缓存策略应覆盖该方法。
    // ------------------------------------------------------------------------
    [[nodiscard]] virtual bool peek(const Key& key, Value& value) const
    {
        static_cast<void>(key);
        static_cast<void>(value);
        return false;
    }

    [[nodiscard]] std::optional<Value> tryPeek(const Key& key) const
    {
        Value value{};

        if (!peek(key, value)) {
            return std::nullopt;
        }

        return value;
    }

    // ------------------------------------------------------------------------
    // 删除与清理
    // ------------------------------------------------------------------------
    virtual bool erase(const Key& key) = 0;

    virtual void clear() = 0;

    void purge()
    {
        clear();
    }

    // ------------------------------------------------------------------------
    // 状态查询
    // ------------------------------------------------------------------------
    [[nodiscard]] virtual bool contains(const Key& key) const = 0;

    [[nodiscard]] virtual std::size_t size() const = 0;

    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;

    [[nodiscard]] bool empty() const
    {
        return size() == 0;
    }

    // ------------------------------------------------------------------------
    // 统计接口
    //
    // 普通缓存可以不覆盖，默认返回空统计。
    // 分片缓存、生产级缓存建议覆盖。
    // ------------------------------------------------------------------------
    [[nodiscard]] virtual CacheStats stats() const noexcept
    {
        return CacheStats{};
    }

    virtual void resetStats() noexcept
    {
    }
};

// ============================================================================
// 4. 兼容旧命名
// ============================================================================
template <typename Key, typename Value>
using KICachePolicy = CachePolicy<Key, Value>;

template <typename Key, typename Value>
using ICachePolicy = CachePolicy<Key, Value>;

} // namespace KamaCache