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
