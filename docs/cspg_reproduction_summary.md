# CSPG 复现总结

## 实现情况

在 VSAG 的 HGraph（HNSW）基础上实现了 CSPG (NeurIPS 2024) 四个核心算法：

- **Algorithm 1（Search）**：phase-1 在 G1 单分区 beam search（ef1），重置 visited，phase-2 在全分区 beam search（ef2），routing vector 触发向其他分区的状态扩张。
- **Algorithm 2（Batch Build）**：采样 routing vectors，其余按 partition 分配，分别构建分区图。
- **Algorithm 3（Online Insert）**：routing vector 插入所有 m 个分区图；local vector 只插入所属分区。
- **Algorithm 4（Delete）**：从所有相关分区图中删除。

新增参数：
- `cspg_partition_graph_type`（`"odescent"` / `"nsw"`）：控制分区图构建方式，解耦于全局 graph_type
- `cspg_max_routing_fanout`（search 参数，默认 0 = 不限制）：限制每个 routing vector 触发的跨分区状态数

---

## 实验结果

### 100k SIFT-128，efc=300，单线程搜索

| Config | Recall | QPS | hops | dist_cmp | routing_pops% |
|--------|--------|-----|------|----------|---------------|
| baseline ef=20 | 0.1386 | 15138 | 32.1 | 487 | — |
| baseline ef=30 | 0.1402 | 12000 | ~42 | ~590 | — |
| CSPG NSW ef2=15 | 0.1338 | 16340 | 33.8 | 501 | 70% |
| CSPG NSW ef2=20 | 0.1368 | 13507 | 41.2 | 584 | 69% |
| CSPG NSW ef2=25 | 0.1384 | 12711 | 48.2 | 660 | 69% |
| CSPG ODescent ef2=20 | ~0.12 | ~7500 | — | — | — |

### 100k SIFT-128，efc=128 对比（论文设置）

| Config | Recall | QPS | dist_cmp |
|--------|--------|-----|----------|
| baseline efc=128 ef=20 | 0.1372 | 16364 | 449 |
| CSPG NSW efc=128 ef2=15 | 0.1338 | 16340 | 501 |
| CSPG NSW efc=128 ef2=20 | 0.1368 | 13507 | 584 |

efc=128 下 CSPG ef2=15 与 baseline ef=20 QPS 几乎持平，但 recall 低 2.5%，dist_cmp 多 12%。

### 1M SIFT-128（NSW 分区，m=2，λ=0.5，16线程构建）

| Config | Recall | QPS | build time |
|--------|--------|-----|------------|
| baseline efc=300 ef=20 | 0.845 | 9497 | ~100s |
| CSPG NSW ef2=15 | 0.800 | 5305 | 1393s |
| CSPG NSW ef2=20 | 0.832 | 4623 | — |
| CSPG NSW ef2=30 | 0.854 | 3773 | — |

1M 规模下 CSPG NSW QPS 约为 baseline 的 56%，build time 约 14×。

---

## 观察到的核心问题

**问题 1：高维路径压缩量为零。**
CSPG 的理论加速来自分区后路径更短。对于 d=128、n=1M、分区后 n'=750k：
路径长度比 ≈ (n'/n)^(1/d) = (0.75)^(1/128) ≈ 0.9978，几乎没有压缩。
低维（如 d=4）才能获得明显路径减少。

**问题 2：Routing state 占据 70% 的 phase-2 hops。**
无论 m、lambda、分区图类型如何变化，routing vector 触发的状态扩张始终占 phase-2 hop 数的 69–70%。这使得有效搜索 hop 数被大量 routing pop 稀释。

**问题 3：NSW 分区图构建时间过长。**
1M 数据 NSW 分区构建需 1393s（≈23分钟），而 baseline HNSW 约 100s。
论文实验的构建时间数据需要参考。

---

## 想确认的问题

1. **CSPG 的加速条件**：论文里的实验用的是什么数据集和维度？SIFT-128 是否是论文里 CSPG 有加速的场景？还是论文主要在低维或特定分布上有效？

2. **Routing state 开销**：phase-2 中 70% 的 hop 是 routing pop，这是预期行为吗？论文里有没有对这部分的分析，或者实现上有什么规避方式？

3. **构建时间**：论文里 CSPG 的构建时间和 baseline 相比是什么关系？NSW 分区图 23分钟 vs baseline 100s，这个比例是否合理？

4. **ef_construction 设置**：论文用 efc=128，我们实验用了 efc=300 和 128，结果基本一致（CSPG 均未超过 baseline）。这个参数对结论影响不大，但想确认论文里的对比是否公平控制了这个变量。
