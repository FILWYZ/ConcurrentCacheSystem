# High-Performance Multi-Policy Concurrent Cache System
# 工业级高性能多策略并发缓存系统 (C++17)

本项目是一个基于 C++17 实现的高性能、线程安全、低锁内聚的生产级内存缓存库。系统不仅实现了传统缓存淘汰算法，更针对现代多核高并发场景进行了深度的**分片锁优化（Lock Sharding）**，并对经典算法在生产环境中的缺陷（如缓存污染、历史热点退化等）引入了**LRU-K 动态追溯**、**LFU 延迟老化（Lazy Ageing）**以及**ARC 自适应平衡**等大厂级硬核改进。

全库采用 Header-only 架构设计，无第三方依赖，开箱即用，具备极致的编译期优化性能。

---

## 🚀 核心架构与技术亮点

### 1. 高并发分片无锁/低锁设计 (Lock Sharding)
* **痛点**：传统缓存使用全局单锁（如 `std::mutex`），在多线程高并发读写时，锁竞争（Lock Contention）会导致严重的 CPU 上下文切换泥潭，吞吐量断层式下跌。
* **解耦方案**：系统引入 `ShardedCache` 封装层，通过高性能哈希函数将 Key 路由至不同的独立分片（Shard）。每个分片拥有独占的线程锁，将全局锁竞争稀释至 $1/N$（$N$ 为分片数），大幅提升多核并行的物理吞吐极限（Mops/s）。
* **伪共享防御**：核心单分片类采用 `alignas(64)` 进行**缓存行对齐**，彻底消灭多核多线程在频繁修改锁状态时的**伪共享 (False Sharing)** 硬件级性能陷阱。

### 2. 生产级算法矩阵与硬核改进
本项目拒绝学院派的简单实现，完全对齐大厂核心基础库的鲁棒性标准：
* **LRU (Least Recently Used)**：标准双向链表 + 哈希表组合，实现严格 $O(1)$ 的查找、提权与淘汰。
* **LRU-K (抗缓存污染)**：引入历史访问队列与多级晋升机制。通过追溯 Key 的第 $K$ 次访问时间（默认 $K=2$），精准过滤突发性的“全表扫描/冷数据涌入”，彻底解决传统 LRU 的**缓存污染**痛点。
* **LFU + Lazy Ageing (时效自适应)**：针对 LFU 长期运行下“早期极热数据在后期转冷，但因频次极高无法被淘汰”的**历史残留毒瘤缺陷**，引入了高效的**延迟老化机制**。在单分片内以 $O(1)$ 的时间复杂度平摊衰减历史频次，使其快速适应业务流量波峰的动态切换。
* **ARC (Adaptive Replacement Cache)**：动态自适应缓存。内部维持 `T1` (近期访问)、`T2` (频繁访问) 两个真实数据队列，以及 `B1`、`B2` 两个**幽灵历史队列 (Ghost List)**。通过反馈控制原理，在运行时无需人工调优即可自适应业务冷热突发波形。

### 3. 指标向上提权与 Lock-Free 监控
* 为防止监控打点（吞吐量、命中率、淘汰数）串行访问各分片引发**级联死锁**，系统采用“指标向上提权、全局共享”设计。
* 外部监控接口 `stats()` 采用 `std::atomic<uint64_t>` 和 `std::memory_order_relaxed` 读写，保证监控统计具备 **Lock-Free（无锁）** 的极致非阻塞性能。

---

## 🛠️ API 核心使用范例

### 1. 基础 LRU 缓存使用
```cpp
#include "LRU.h"
#include <string>
#include <iostream>

int main() {
    // 创建一个容量为 1000 的标准单线程安全 LRU 缓存
    Cache::HighConcurrencyCache<int, std::string> cache(1000, 1);
    
    // 写入
    cache.put(42, "Deep Answer");
    
    // 读取
    std::string value;
    if (cache.get(42, value)) {
        std::cout << "Hit! Value: " << value << std::endl;
    }
    return 0;
}