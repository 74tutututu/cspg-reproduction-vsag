# CSPG 复现总结

## 实现情况

在 VSAG 的 HGraph（HNSW）基础上实现了 CSPG (NeurIPS 2024) 四个核心算法：

- **Algorithm 1（Search）**：phase-1 在单分区 beam search（ef1），重置 visited，phase-2 跨全分区 beam search（ef2），routing vector 触发向其他分区的状态扩张。
- **Algorithm 2（Batch Build）**：采样 routing vectors，其余按 partition 分配，分别构建分区图。
- **Algorithm 3（Online Insert）**：routing vector 插入所有 m 个分区图；local vector 只插入所属分区。
- **Algorithm 4（Delete）**：从所有相关分区图中删除。

新增参数：
- `cspg_partition_graph_type`（`"odescent"` / `"nsw"`）：控制分区图构建方式，与全局 graph_type 解耦
- `cspg_max_routing_fanout`（search 参数，默认 0 = 不限）：限制每个 routing vector 触发的跨分区状态数
- `cspg_phase1_skip_base_descent`（search 参数，默认 false）：见下方 QPS 优化

---

## QPS 优化（2026-06）

精确诊断后定位到 gap 的两个来源并各个击破。

### 诊断：gap 的两个来源（efc=128，同 recall 对比）

| recall | baseline dist | CSPG dist | dist 比 | QPS 比 |
|--------|------|------|--------|--------|
| 0.93 | 865 | 989 | 1.14x | 0.69x |
| 0.96 | 1173 | 1338 | 1.14x | 0.74x |

- 多算 14% 距离（≈ phase-1 的固定 129 次）
- 每次距离贵 1.2–1.3x（多图 cache 开销）

### 优化一：frontier 二叉堆（已合入 `e49d656`）

phase-2 frontier 原为有序 vector，每次 enqueue 是 O(ef2) 内存搬移。改为二叉堆（O(log ef2)）。
+5% QPS，代码更简洁，功能测试不变。

### 优化二：`cspg_phase1_skip_base_descent`（已合入 `2cfb343`）

**根因**：开启 route descent 时，phase-1 已沿分层 route 图下降到近 query 入口，却又在 base 层
做一次完整 beam search 到收敛；phase-2 重置 visited 后**又重做一遍 base 层下降**。base 层被走了两次。

**修复**：route descent 停在 level 1，把入口直接交给 phase-2 做唯一一次 base 层搜索。

效果（1M SIFT，efc=128，20线程）：

| ef2 | recall | phase-1 dist | QPS 比（修复前→后） |
|-----|--------|--------------|---------------------|
| 40  | 0.93 | 129→33 | 0.69x → **0.72x** |
| 60  | 0.96 | 129→33 | 0.75x → **0.84x** |
| 100 | 0.98 | 129→33 | 0.66x → **0.74x** |
| 150 | 0.99 | 129→33 | 0.74x → **0.80x** |

**当前最好：recall=0.96 时达到 baseline 的 0.84x**（修复前 0.66x）。
减少 phase-1 同时把每次距离开销从 1.2–1.3x 降到约 1.09x（废弃的 phase-1 遍历有差的 cache 行为）。

### 结论：天花板与剩余 gap

剩余 ~0.84x 的 gap 是**结构性**的，无法在 SIFT-128 + HNSW 上消除：
- d=128 下分区缩小带来的路径压缩 ≈ 2%（理论上限，见下）
- 多图结构 + 跨分区跳转的 cache 开销（~1.09x）固有

这与论文自身理论一致（lim d→∞ speedup = α ≈ 1 for HNSW），且论文宣称的 1.5–2x 加速是
Vamana/HCNNG，**不是 HNSW**。在 SIFT-128 上 CSPG-HNSW 结构上无法超越 baseline；优化后已逼近
可达天花板。

---

## 正式实验结果

**实验设置**（对齐论文）：SIFT1M，1M 全量数据，10k queries，20 线程搜索，efc=300，M=32，m=2，λ=0.5，分区图类型 NSW，ef/ef2 扫描 {10,20,40,60,100,150,200,300}。

### QPS vs Recall@10 对比

| ef(baseline) / ef2(CSPG) | Baseline Recall | Baseline QPS | CSPG Recall | CSPG QPS | QPS 比 |
|--------------------------|-----------------|--------------|-------------|----------|--------|
| 10  | 0.729 | 156344 | 0.729 | 94470 | 0.60x |
| 20  | 0.857 | 115929 | 0.862 | 68528 | 0.59x |
| 40  | 0.942 |  72492 | 0.944 | 45316 | 0.63x |
| 60  | 0.970 |  50762 | 0.971 | 32445 | 0.64x |
| 100 | 0.989 |  33675 | 0.989 | 22552 | 0.67x |
| 150 | 0.996 |  23363 | 0.996 | 16238 | 0.70x |
| 200 | 0.998 |  17222 | 0.998 | 12631 | 0.73x |
| 300 | 0.999 |  13292 | 0.999 |  8256 | 0.62x |

**结论：CSPG-HNSW 在 SIFT1M 所有 recall 点均慢于 baseline，QPS 约为 baseline 的 60–73%。**

### 内部开销分析（ef2=20 为例）

| 来源 | dist_cmp | hops | dist/hop |
|------|----------|------|----------|
| CSPG phase-1 | 200.5 | 8.8 | 22.8 |
| CSPG phase-2 routing pops（69%） | 325.2 | 24.6 | 13.2 |
| CSPG phase-2 local pops（31%） | 283.6 | 10.9 | 25.9 |
| **CSPG 合计** | **809** | **44.4** | — |
| **baseline ef=20** | **589** | **34.2** | **17.2** |

关键观察：
- CSPG "真实探索" hops（phase-1 + local pops）= 8.8 + 10.9 = **19.7 hops**，少于 baseline 34.2 hops。
- 但 phase-1 独立耗费 200.5 dist_cmp（baseline 搜索全部才 589），加上 routing 扩张的 325.2 dist_cmp，总计 809 > 589。
- 理论上 CSPG 确实减少了有效探索步数（~19.7 vs 34.2），但两段额外开销（独立 phase-1 + routing 广播）把节省全部抵消。

---

## 关于"曾经达到过 1.1x"的澄清

历史记录里确实有一条 1.097x（`compare_1m_baseline_vs_auto_sparse.json`），但它**不是同 recall 的公平对比**：

| | 配置 | Recall | QPS |
|---|------|--------|-----|
| baseline | ef=60 | **0.9696** | 9042 |
| CSPG（旧） | ef2=55 | **0.9599** | 9922 |

CSPG 的 recall 低了约 1%，用更小的候选集换来了更高的 QPS。在同一 recall 点上 baseline 用更小的 ef 也能跑出同样甚至更高的 QPS——所以这个 1.1x 是 recall 错位造成的假象。

### 旧配置（ODescent d=24 + route_descent）的真实对比

重建旧配置索引，在 1M + 20线程下做同 recall 对比（baseline QPS 按 recall 线性插值）：

| ef2 | Recall | QPS | vs Baseline | vs 当前NSW |
|-----|--------|-----|-------------|-----------|
| 20  | 0.772 | 94341 | 0.66x | 1.10x |
| 40  | 0.879 | 63846 | 0.61x | 1.00x |
| 60  | 0.923 | 49018 | 0.60x | 0.96x |
| 100 | 0.960 | 31954 | 0.55x | 0.85x |
| 200 | 0.986 | 17232 | 0.47x | 0.71x |

两个结论：
1. **旧 ODescent 配置同样从未真正超过 baseline**（同 recall 下 0.47–0.66x）。所谓 1.1x 始终是 recall 错位的假象。
2. **当前 NSW 配置确实比旧 ODescent 配置慢**（高 recall 区慢 15–30%），这是真实回归。原因见下。

### 为什么当前配置比旧配置慢——三个真实变化

| 变更 | 提交 | 影响 |
|------|------|------|
| phase1 `route_descent` 改默认关闭（opt-in） | `8eb9b380` | phase-2 进入点变差，需更高 ef2 达到同 recall（注：本次回归对比里两边都开了 route_descent，已隔离此因素） |
| partition `max_degree` 从 auto-scale(24) 改为 full(32) | `36bb2e47` | 分区图更密：dist/hop 从 8.6 涨到 15.7（ef2=60），每跳成本接近翻倍 |
| 默认/测试分区图从 ODescent 改 NSW | `36bb2e47` | NSW 图更密、recall 更高，但每跳 dist_cmp 更多 |

**度数对比（ef2=60）**：
- 旧 ODescent d=24：dist/hop = **8.6**，recall=0.923
- 当前 NSW full：dist/hop = **15.7**，recall=0.971

当前配置牺牲了 QPS 换 recall。这两个改动当初是为"贴近论文"（论文要求分区图与 baseline 同 degree），但代价是 QPS 回归。

---

## ef_construction=128（论文设置）实验 —— 建图慢和 QPS 慢是两个独立问题

把分区图 `ef_construction` 从 300 调到论文的 128，重建 NSW 分区图（1M，16线程构建），并实测真实 degree。

### 实测 avg degree（用分析器，非推断）

| 配置 | partition local degree | routing degree | baseline 整图 |
|------|----------------------|----------------|--------------|
| baseline HNSW（max_degree=32） | — | — | **23.67** |
| CSPG NSW efc=300 | 21.73 | 24.43 | — |
| CSPG NSW efc=128 | 17.32 | 22.78 | — |
| 旧 ODescent d=24 | 10.46 | 10.50 | — |

CSPG NSW（efc=128/300）的分区图 degree ≈ 17–24，**与 baseline（23.67）基本对齐，符合论文要求**。旧 ODescent 的 10.5 才是真正偏稀疏。

### 建图时间

| 配置 | build time | vs baseline |
|------|-----------|-------------|
| baseline HNSW efc=128 | **39.7s** | 1.0x |
| CSPG NSW efc=128 | **420s** | **10.6x** |
| CSPG NSW efc=300 | 1393s | — |
| 论文 baseline HNSW | 33s | 1.0x |
| 论文 CSPG-HNSW | 50s | **1.5x** |

关键：
- 我们的 **baseline efc=128 = 39.7s，几乎等于论文的 33s** —— 说明 baseline 构建路径没问题。
- 但 CSPG NSW = 420s = **10.6x**，而论文 CSPG 只有 **1.5x**。
- 论文 CSPG 是用**和 baseline 完全相同的高效 HNSW 构建器**建 2 个分区图（总插入量 1.5n → 1.5x 时间）。
- 我们的 NSW 分区构建走了**独立的低效路径**（疑似没复用 baseline 的批量构建优化）。
- **结论：建图慢 10.6x 不是"图太密"造成的（degree 是对的），而是分区图构建代码路径效率低。**

### efc=128 下的 QPS（同 recall vs baseline efc=128）

| ef2 | CSPG Recall | CSPG QPS | baseline@同recall | Ratio |
|-----|-------------|----------|-------------------|-------|
| 20  | 0.842 | 73132 | 120119 | 0.61x |
| 40  | 0.930 | 45696 |  75606 | 0.60x |
| 60  | 0.962 | 33319 |  50242 | 0.66x |
| 100 | 0.984 | 24156 |  36432 | 0.66x |
| 200 | 0.996 | 13485 |  20417 | 0.66x |

**QPS 比仍是 0.60–0.66x，与 efc=300 完全相同。efc 不影响 QPS 差距。**  
QPS 慢的根因是 routing overhead（phase-2 中 69% hops 是 routing pop），这是 d=128 下 CSPG 的固有问题，与 degree、efc 都无关。

### 三个问题的最终归因

| 现象 | 根因 | 是否"图太密" |
|------|------|-------------|
| 建图慢 10.6x（vs 论文 1.5x） | NSW 分区构建走低效代码路径 | ❌ degree 是对的 |
| QPS 慢 0.6x | routing overhead（69% 无效 hops），d=128 路径压缩仅 2% | ❌ 与密度无关 |
| 旧"1.1x"假象 | recall 错位对比 | ❌ |

---

## 评测方法修正记录

### 之前的错误（100k 实验 recall 无效）

之前所有 100k 实验（recall≈0.13）均使用了 SIFT1M HDF5 文件的 1M ground truth。  
`build_base_limit: 100000` 只建了 100k 索引，但评测在对比 1M 邻居：  
recall≈0.13 = "13% 的真实 1M 邻居恰好落入 100k 子集"，与索引质量无关。

### 修正后

- Build：1M 全量，无 `build_base_limit`
- Ground truth：HDF5 文件原生 1M ground truth（一致）
- 搜索：20 线程（论文 24 线程，本机 22 核，取 20）
- Queries：10000（论文设置）
- ef 扫描：{10,20,40,60,100,150,200,300}（论文 [10,300]）

---

## 为何 CSPG-HNSW 在 SIFT1M 无加速——理论分析

CSPG 加速比公式（Section 5.4）：

```
Speedup = α × β
β = n^(1/d) log(n^(1/d))  /  (n')^(1/d) log(n'^(1/d))
n' = λn + n(1-λ)/m（分区大小）
```

对 SIFT1M（d=128，n=1M，m=2，λ=0.5，n'=750k）：
- β ≈ 1.023（路径长度减少仅 2.3%）
- 论文 lim(d→∞) Speedup = α（高维下 β→1，加速全靠 α）

α 来自 detour factor（w）和最小距离差（Δr）的改善，需要 baseline 算法本身有较大 detour。  
**HNSW 的 detour factor 已经较小（靠近 MSNET），α 接近 1。**  
**Vamana / HCNNG 的 avg_degree 更高、detour 更大，α 更大，论文 1.5x–2x 来自这两个算法。**

---

## 待确认的问题

1. **论文 CSPG-HNSW SIFT1M speedup 具体是多少？**  
   论文只说 Vamana/HCNNG ≥1.5x，未明确 HNSW 数字。从 Figure 3 估计约 1.1–1.3x。  
   我们测得 0.60–0.73x，差距原因：avg_degree 未对齐？还是 HNSW 本身 detour 就小？

2. **avg_degree 对齐的具体做法**  
   论文说 "slightly adjust parameters to ensure average degree is same"。  
   当前数据：我们 CSPG partition 图的 dist/hop ≈ 14–20，baseline ≈ 16–18，范围接近。  
   是否需要额外调整，或者 avg_degree 差距不是主因？

3. **论文构建时间 50s vs 我们 1393s**  
   论文 Table 1：CSPG-HNSW SIFT1M build = 50s（vs baseline 33s，约 1.5x）。  
   我们 NSW 分区构建 = 1393s（14x baseline）。  
   论文用的分区图类型究竟是什么？如果是 KNNG/ODescent 类型，50s 合理；NSW 不可能 50s。

4. **Routing state overhead（70%）是否在论文实现中存在？**  
   phase-2 中 69% 的 hops 是 routing pop（无效探索），这大幅抵消了路径压缩的收益。  
   论文原始代码（github.com/PUITAR/CSPG）中是否有对应开销？还是有规避机制？
