# 教师残差蒸馏线 · pro5k6 会话 handoff（2026-10-21）

> 会话坐标：`pro5kvrdma6`（8×RTX PRO 5000 72GB，公网）。
> 目标：用教师模型（Qwen3.6-35B-A3B）的中间层残差流蒸馏出**按位置对齐**的数据，
> 用于 n-gram 表（order 2-5）的初始化 / 聚类 / 负载均衡。
> 状态：**全部暂停**（8 卡空闲、所有 daemon 已停、数据与表都在盘上）。
> 本文所有数字都带来源（命令 / 文件路径 / 时间戳），未验证的一律标 `未验证`。

---

## 0. 一分钟版

1. **生产管线跑通了，产出了 0.94B 位置（16 个 shard）的教师残差聚合表** —— 80GB 的
   `teacher_table_o2to5_partial.npz`（4 worker part 合并 + 均值），聚合级校验已跑（§5.3）。
2. **发现真正的瓶颈不在 GPU**：每批 2.70s 里只有 0.67s 是前向，**2.03s 是 CPU 聚合**，
   根因是机器默认 192 个 torch 线程在 4 worker 争抢下互相踩（§6）。
   两个**逐位等价**的替代版已写好并验证：`distill_prod_v2.py`（1.26s，2.1×）、
   `distill_prod_v3.py`（0.79s，3.4×）。
3. **语料级 n-gram 计数已完成**（210 shard / 11.376B 位置，11 分钟）——这是回答
   「词表能开多大」的关键新数据（§9）：同样 256 阈值下，本语料可支撑 **11.4M 行**表
   （现在是 0.97M 行），阈值 1024 下 2.80M 行。
4. **投影器（残差压缩）已量化验证**：128 维 dense 投影 1.39ms/批、距离结构保真 0.998、
   recall@10 = 0.73、k-means ARI = 0.60，int8 化几乎无损（§8.3）。
5. **残差 dump 的代码已写但没验证、也没启动**（§10）：`distill_prod_v4/v5.py` +
   `dump_positions.py` + `validate_dump.py` 都在盘上，等价性/回读测试**没有留下证据**。
6. **事故**：16:22:59 w3 的 EngineCore 收到外部 SIGTERM 死亡（同期本机在跑我们自己的
   验证 harness）——优雅退出把数据保存到 16:22，**没有丢数据**，但这是必须记住的教训（§7.1）。
7. 待决策：存储架构（RAID0 vs 3 副本）、1B 残差预算怎么放、fp8 要不要评估、push 确认（§11）。

---

## 1. 这条线要解决什么

用户原话链（2026-10-21）：先蒸馏 **2B** 用于基础训练，最终 **10B**；
「预计只花几小时」；后续提问（未答完）：
「项目数据会比较大，可能要存 1B 个 token 的残差流，加上整个 128 cluster 的 ngram 词表，
空间要这么大，所有机器之间没有这么强的互联，所以数据应该怎么办？」
→ 本会话把**生产、瓶颈、语料统计、压缩可行性**四件事做到了可决策状态（§9/§10/§11）。

### 1.1 数据契约（已写进代码 docstring + 文档）

- **每个 context 行存的是：该 context 最后一次出现处、最后一个 token 的残差流**
  （=「读完该 context 之后」的状态，也就是把该行注入到位置 t 时 t 前向接收到的状态）。
- 表行集合 = `/data/assign_exact974k.npz`（`keys_*` 升序，`rows_* == arange(R)`）：
  order 2: 368,957 / 3: 408,220 / 4: 151,514 / 5: 45,642 → 合计 **974,333 行**。
- 每个位置捕获 **5 个块 × 2048 维**：`[d0, d10, d25, d40, final_normed]`
  （`OE_CAP_LAYERS=0,10,25,40`；d0 = wte 输出，final = pre-logits 的 RMSNorm 输出）。
- 位置对齐按**字节位置**（我们的 token t 取「累计字节结束 ≤ 我们的累计字节结束」的最后一个
  教师 token），而不是字符偏移重建 —— 对任意 UTF-8 都成立。

---

## 2. 硬件 / 集群坐标

| 项 | 值（本会话实测） |
|---|---|
| GPU | 8× RTX PRO 5000 72GB（Dawn/Blackwell 级），TP=2 一个 worker，4 worker 用满 8 卡 |
| 互连 | **无 NVLink**：`topo -m` 显示 0-3 同 NUMA0、4-7 同 NUMA1，跨 socket 是 SYS；TP=2 配对 (0,1)(2,3)(4,5)(6,7) 是正确选择 |
| CPU | AMD EPYC 9K65，**384 线程**（`avx512_bf16`，无 AMX） |
| RAM | 1450 GB（4 worker 各 ~40GB 累加器 + 读 part 时峰值 ~200GB，充裕） |
| 磁盘 | `/data` = `/dev/md0` **RAID0 7.0T**，当前用 1.2T / 剩 **5.4T** |
| 网络 | 公网可达（eth0/eth1 虚拟化 NIC，`speed=-1` 未知）；`vrdma_0` RDMA 设备存在，**本会话未测吞吐** |
| 语料 | `/data/climbmix_raw/shard_*.parquet` 210 个，每个 92MB，共 19.3 GB / **11.376B lab 位置** |

> 关于存储：RAID0 无冗余，任何一块盘挂掉整盘数据全没。这条线的产物
> （80GB 合并表 + 4×40GB part + 479GB 计数 + 未来的 TB 级残差 dump）目前**没有副本**（§11.1）。

---

## 3. 管线结构（生产）

```
climbmix_raw/shard_*.parquet (86016 docs/shard)
  └─ Packer：lossless 打包成 2048-token 整行（BOS 前缀，匹配 data/tokenized 约定）
       └─ lab token ids ──(tokenizer.pkl + tok_bytes.npz)──> 字节边界
            └─ vLLM 0.31.0 引擎（Qwen3.6-35B-A3B, TP=2, --enforce_eager）
                 ├─ oe_cap_patch.py hook：抓 4 个 decoder block 之后的残差流 + 最终 normed
                 └─ pooling 输出 (nq, 10240) fp32（sentinel 列 → 驱动的 denorm 还原）
                      ├─ 字节位置对齐 → q_star（我们的 token → 教师 token）
                      ├─ n-gram context key（context_keys_np）→ searchsorted 进 974k 表
                      └─ 累加：sum_L<d>_<n> += 残差行, cnt_<n> += 1（fp32 / int64）
```

- 引擎常驻（每 worker 只加载一次）；冷启动 9-11 分钟，页缓存热后 ~50s。
- 关键补丁：vLLM 的 `token_embed` pooler **硬编码 `PoolerNormalize`**，会把我们附加的每个
  捕获块做 L2 归一化（驱动侧曾看到 RMS 只有 1/√2048 ≈ 0.022，差 600×）。
  修法：附加一列全 1 sentinel，pooler 后变成 1/‖row‖，denorm 乘其倒数精确还原；
  `selftest` max rel err 5.04e-08。
- **CUDA graphs 与多层捕获不兼容**（graph runner 要求输出宽度 = hidden_size，
  捕获会附加 depths+1 个块 → 引擎在 graph init 阶段崩），因此固定 eager。

---

## 4. 验证证据（都在本会话做过，可复现）

| 检查 | 结果 | 来源 |
|---|---|---|
| HF 真值（CPU，真实文档 219 teacher token）hidden RMS | d0=0.0105174 / d10=0.051604 / d25=0.10792 / d39=0.340081 / final=2.4941 | `/tmp/hf_ref2.py` |
| 逐元素 cos（捕获 vs HF） | d0=**1.000000** / d10=0.999399 / d25=0.998062 / normed=0.998140 | 同上 |
| 生产端同文档诊断（同一文档过 vLLM） | d0=0.0105174 / d10=0.051635 / d25=0.108466 / normed=2.49382 | `/data/distill_r3.log` |
| 字节对齐 | `byte_end_match=True`，**99.9%** 位置对齐成功 | preflight |
| preflight | PASS（修掉「末行不够长」误报后：只检查非最后一行 `min(lens[:-1]) < seq`） | `/data/distill_prod.py` |
| 表命中率（生产稳态，4 worker 一致到 3 位） | n2≈**83.96%** n3≈**37.14%** n4≈**10.67%** n5≈**2.91%** | 各 part 的 `meta_json.hit_rate` |

> 语义确认：d0 = 教师词嵌入（Qwen3.6 无绝对位置加），所以 d0 的逐元素 cos 是 1.000000；
> 这同时证明深度映射正确（不是差一层）。

---

## 5. 生产结果（暂停时刻的快照）

### 5.1 覆盖面

| worker | shards | 位置数 | docs | rows | 起止 |
|---|---|---|---|---|---|
| w0 | 0,4,8,12,16 | 272.12M | 424,878 | 132,875 | 09:34 → 17:15 |
| w1 | 1,5,9,13 | 271.34M | 423,866 | 132,492 | 09:34 → 17:15 |
| w2 | 2,6,10,14,18 | 271.55M | 424,446 | 132,597 | 09:34 → 17:15 |
| w3 | 3,7 | 125.07M | 195,152 | 61,070 | 12:14 → **16:22 死亡** |
| **合计** | **16 / 210** | **0.940B** | 1,468,342 | 459,034 | — |

（每个 shard ≈ 54.4M 位置 / 1.5h；`meta_json` 逐 part 可查）

### 5.2 合并表

```
python3 /data/merge_distill_parts.py --in_dir /data/distill_out --tag w3r \
    --out /data/distill_out/teacher_table_o2to5_partial.npz --mean
→ wrote teacher_table_o2to5_partial.npz (79.83GB)  [4 parts, 17:20 完成]
   total project tokens = 0.940B over 1468342 docs
```

⚠️ 早期 `--max_docs` 测试遗留的 `part_w3_w3r.npz` 已改名 `oldtest_w3_w3r.npz`：
不改名时 `--include_ckpt` 会让**最终 part 遮蔽同 worker 的 ckpt**，静默丢掉 w3 的真实数据。

### 5.3 聚合级校验（`check_distill_aggregate.py`，17:2x）

| 深度 | HF 单文档参考 | 语料均值（cnt≥1 / ≥64） | 差 |
|---|---|---|---|
| L0 | 0.0105 | 0.01001 / 0.01001 | −5% |
| L10 | 0.0516 | 0.04419 / 0.04374 | −14% |
| L25 | 0.1079 | 0.08164 / 0.07143 | −24% |
| L40 | 0.340 | 0.2944 / 0.294 | −13% |
| Lfinal | 2.494 | 1.791 / 1.789 | −28% |

- 结构信号全部正常：四个 order 的**非零覆盖 99.5-99.9%**（n2 368,624/368,957 行有数据）、
  深度单调（0.010 → 0.044 → 0.082 → 0.294 → 1.79）、跨 cnt 阈值稳定（无排序/溢出异常）。
- **未闭环项**：语料级 RMS 比单文档参考低 5-28%。逐元素/单文档检查（§4）是精确匹配的，
  所以这不是映射错误，最可能是**文档构成差异**（参考文档是单篇英文样本，climbmix 混合
  web/code，逐文档均值方差很大）。要彻底闭环：抽 20 篇语料文档走一遍 per-doc 均值分布。
- 计数一致性：本表 sum(counts) = 788.9M + 348.8M + 100.2M + 27.4M = **1.265B 匹配对 /
  0.94B 位置** = 1.35 order/位置，与命中率之和 0.8396+0.3714+0.1067+0.0291 = 1.347 ✓ 自洽。

---

## 6. 性能解剖（本会话最有价值的发现）

### 6.1 每批 2.70s 的真实构成

| 项 | 耗时 | 证据 |
|---|---|---|
| vLLM 前向（16 行 = 32,768 位置，教师 ~28.2k token） | **0.67s**（≈24k teacher tok/s/GPU） | 每步计时 |
| CPU `accumulate`（4 个 order 的 index_add_ 写 974k×10,240 fp32） | **2.03s** | 每步计时 + py-spy |
| 合计 | 2.70s → 12.0k lab tok/s/worker | 生产日志 |

`py-spy record`（driver 进程，25-49 核常驻）显示 **84% 采样落在 `index_add_` 上**
（`distill_prod.py:257/266`）。

### 6.2 根因是线程数，不是算法

- 机器默认 **192 个 torch 线程**（`/proc/<pid>/environ` 确认），4 worker 共 768 线程挤 384 核。
- 隔离测量：n2 的 `index_add_` **362 ms @192 线程 vs 61 ms @32 线程**；竞争下中位数劣化 ~7×。
- 内在计算量下限 ~300ms → 剩下 1.7s 全是线程争抢。

### 6.3 两个已验证替代版（逐位等价）

| 版本 | 改动 | 实测步时 | 等价性证据 |
|---|---|---|---|
| `distill_prod_v2.py` | `torch.set_num_threads(32)` + 复用 `ones` | ~1.26s（2.1×） | `equiv_v2_1607.txt`：3 批 **PASS**，maxabs=0.000e+00 |
| `distill_prod_v3.py` | v2 + encode/accumulate **生产者-消费者重叠** | ~0.79s（3.4×） | `equiv_v3_report.txt`：**OVERALL PASS**，含「save while blocked」不丢不重 |

- v3 的失败路径也测了：producer 异常不死锁，exit non-zero。
- `Packer`/`llm.encode` 只在生产者线程调用（引擎单线程安全）；consumer 独占全部状态与 save。
- **未在真实引擎上验证**：v3 只在 mock 前向（sleep 0.67s）下测过；原计划让 w3 单独当金丝雀。

---

## 7. 事故与坑（都值得写进文档）

### 7.1 w3 被外部 SIGTERM 打死（16:22:59）— 没丢数据，但必须记住

- 现象：w3 driver 的 EngineCore 收到 `SIGTERM`（`/data/distill_r3.log` 16:22:59），
  engine 优雅退出，driver 捕获后 **保存 part + state 到 16:22**（文件 mtime 可证）。
- 病因**未定论**：时间窗与在本机跑的验证 harness（`equiv_v3.py` 16:16-16:26、
  `validate_dump.py` 16:57）重叠；已 grep 确认 v3/harness 里**没有 pkill/killpg**，
  dmesg 里也**没有 OOM**。最可能是 harness 的进程组/harness 导入的 vLLM 清理路径的副作用。
- 教训：**重 CPU / 重内存的验证 harness 不要在跑生产的同一台机器上跑**，
  或至少用独立 session 并把生产放在 `setsid` 外不可达的进程组里。
- 数据损失：0（优雅退出会 save）；但 w3 只完成 2 个 shard，是四臂里最慢的。

### 7.2 其它已修的坑

| 坑 | 现象 | 修法 |
|---|---|---|
| 每 shard 重载模型 | 11 分钟/shard，10 小时纯开销 | 常驻引擎 |
| 每文档一步（~560 token 行） | 575 lab tok/s/worker（10B 要 50 天） | 2048 整行打包 + `batch_rows 16` → 12k tok/s（9-17×） |
| pooler 归一化 | 块 RMS 小 600× | sentinel 列 + denorm（§3） |
| 过期 hook store | `UnboundLocalError` / 静默丢块 | 每次外层 forward 清 store；行数不符就返回纯输出并由驱动显式报错 |
| preflight 误报 | 「a full row came out short」 | 只检查非最后一行 |
| shell heredoc 吃引号 | patch 写入损坏 | 用 scp 传文件再追加 |

---

## 8. 并行工作（三个 workstream 在暂停前落地了多少）

> 这三个 scope 由 subagent 执行，**在我收回/取消时留下了产物但没有留下完整报告**。
> 下面严格区分「有证据」和「只有文件」。

### 8.1 残差 dump 集成 — 写成，**未验证**

- `/data/distill_prod_v4.py`（36.9KB, 16:47）、`/data/distill_prod_v5.py`（43.7KB, 16:47）
  = v2 / v3 + dump 集成，语法通过，含 `--dump_on/--dump_dir/--dump_tag/--dump_blocks/
  --dump_dims/--dump_full_shards`（默认 OFF）。
- `/data/dump_positions.py`（27.1KB, 17:11）：`DumpWriter`（memmap 直写、int8 对称量化、
  per-shard meta 带 sha1 与量化公式、`_load_sketch_class()` 会优先用 prj_sketch 并自带 fallback）。
- `/data/bench/validate_dump.py`（15KB, 16:57）：**写了但看不到运行证据**（`/data/distill_dump/` 空）。
- 结论：**必须补等价性（v4 vs v2）+ 回读验证**，才能开跑（§10.4）。

### 8.2 语料级 n-gram 计数 — **已完成**（本会话最有价值的新数据）

见 §9。

### 8.3 投影器（残差压缩）— **已完成量化验证**

`/data/prj_sketch.py`：`Sketch(in_dim, out_dim, nonzeros, seed)`，支持 `sparse/dense/fft`，
`info()/from_info/project/encode_i8/decode_i8`。

实测（28k×2048→128，本机负载下 7 次中位）：

| 实现 | 每批耗时 |
|---|---|
| **dense torch bf16 @32 线程** | **1.39 ms** |
| dense torch fp32 @32 | 12.1 ms |
| sparse numpy nz=8/16/32 | 83 / 88 / 133 ms |
| FFT | 218 ms |

保真（20k 真实残差向量，order 3，depth 25/40，`prj_summary.json`）：

| out_dim | 距离结构相关（dense） | recall@10（dense） | k-means ARI / purity（dense） |
|---|---|---|---|
| 32 | 0.9928 | 0.501 | 0.348 / 0.528 |
| 64 | 0.9964 | 0.629 | 0.463 / 0.606 |
| **128** | **0.9983** | **0.728** | **0.598 / 0.737** |
| 256 | 0.9991 | 0.798 | 0.607 / 0.753 |

- **int8 代价几乎为零**：dense od128 int8 的 ARI = 0.5929 vs fp32 0.5979（purity 0.734 vs 0.737）。
- 结论/建议：dump 用 **dense (torch bf16, 32 线程), out_dim=128, int8 对称量化**；
  1.39ms/批的开销相对 2000ms 聚合可忽略。若要更好聚类再考虑 256 维（磁盘翻倍）。

---

## 9. ★ 语料级 n-gram 计数（回答「词表能开多大」）

`/data/counts/`（`count_ngrams.py` + 96 进程，**210 shard / 11.376B 位置只用 11 分钟**；
合并 2180s）。`counts_*.npz` 存全部 distinct n-gram → count；`summary.json` 是直方图。

| order | distinct | ≥1024 | ≥256 | ≥64 | ≥32 | ≥2 |
|---|---|---|---|---|---|---|
| 2 | 27,570,210 | 761,572 | 1,930,024 | 4,523,002 | 6,649,874 | 22,095,422 |
| 3 | 693,077,059 | 1,225,785 | 4,727,547 | 16,615,162 | 30,212,569 | 324,825,624 |
| 4 | 2,908,437,697 | 599,433 | 3,299,216 | 16,000,753 | 34,241,338 | 882,501,014 |
| 5 | 5,670,244,522 | 214,100 | 1,485,228 | 9,037,560 | 21,643,235 | 1,125,103,556 |
| **行数合计** | — | **2,800,890** | **11,442,015** | **46,176,477** | **92,747,016** | — |

**关键推论（用于解释当前 974k 表）**：
本语料的匹配对总数 = 每位置 × 1.35 order ≈ **13.7B**，而 `assign_exact974k.npz` 的
counts 总和只有 1.357B → 反推该表是在 **≈1.0B 位置**的语料上按阈值 256 建的
（即训练用的那个 1B tokenized 语料，不是这 11.4B 原始语料）。
本语料在同样 256 阈值下 n2 就有 1.93M 行（是表里 36.9 万的 5.2×）。
→ 想要更大表时，**阈值必须随语料一起放大**，否则行数会失控；

**成本可行性**（3 个 dump 深度 × 2048 维 × fp32）：

| 阈值 | 行数 | 单份 sums | ×4 worker 常驻 | sums+mean 落盘 |
|---|---|---|---|---|
| 1024 | 2.80M | 69 GB | 276 GB ✓ | 138 GB |
| 256 | 11.44M | 281 GB | **1.13 TB ✗**（RAM 1.45TB 但只剩 ~300GB） | 562 GB |
| 64 | 46.2M | 1.14 TB | ✗✗ | 2.3 TB |

→ 现实选择：**阈值 1024（2.8M 行）**，或保持 256 阈值但只存 1 个深度 / fp16 累加。

**未完成**：把 `assign_exact974k.npz` 的 974k 行代入新计数做交叉核对（原计划 Deliverable 3，
被取消；这是下一步该做的第一件事，因为它是「换表」决策的前提）。
`/data/counts/` 目前占 **479 GB**（`counts_5.npz` 单文件 90GB），保留 `summary.json` 即可支撑
多数决策，npz 属可清理对象（需批准，§11.3）。

---

## 10. 残差 dump 设计（**未启动**，供下一段用）

### 10.1 预算（硬约束）

- 位置总数 11.376B；`/data` 剩 5.4T；要留 part（0.4T）+ 一个全保真 shard（0.67T）。
- 方案：每位置 **3 块（d10/d25/d40）× 128 维 int8（384B）+ uint32 teacher token id（4B）
  + fp16 per-块 scale（6B）≈ 392B** → 11.376B 位置 = **4.46 TB** ✓（d0 与 final 可从
  token id 和 d40 重建，无需存）。
- 加上 1 个 shard 的 fp16 全保真（3×2048×2B×54.4M = 669 GB）→ 总 ~5.1T，**紧但可行**。
  若不做全保真，余量立刻宽松。

### 10.2 文件布局（已按此实现于 `dump_positions.py`）

```
/data/distill_dump/<tag>/sketch_w<k>_s<shard>.i8.npy    (n_pos, 3, 128) int8
                        scale_w<k>_s<shard>.f16.npy     (n_pos, 3)
                        tid_w<k>_s<shard>.u32.npy       (n_pos,) teacher token id
                        meta_w<k>_s<shard>.json         # 量化公式/sha1/对齐约定/code md5
```

### 10.3 发现的必要性

用户观点（原话意思）：直接训练稀疏表再聚合会**欠训练、不一致**，
所以要先「超大 OE 词表训练 → 聚合」，或者直接把某些层的 residual 聚类后初始化+合并；
并且要**分层**（多层 OE 分别 offload 一部分层的 residual）。残差 dump 是这条路线的前提。

### 10.3b 与仓库既有 route (b) 的对接（**必须先读这条**）

仓库里**已经有一条 committed 的 route (b) 实现**（`8d6381f` / `a01c90e`，2026-10-19/20），
和本会话的教师蒸馏是**互补**关系，不要重复造：

| 既有工具 | 作用 | 契约 |
|---|---|---|
| `code/tools/w2_dump_residuals.py` | 用**我们自己的 nogram backbone** 跑一遍 train pool，在候选 OE 注入层 dump 每个 context 的**均值残差** | 输出 `<out_dir>/resid_L<l>.npz`：`keys_<n>`(int64 升序) / `sum_<n>`(K_n, **n_embd=768**) fp32 / `cnt_<n>` int64 / `meta_json` |
| `code/tools/w2_cluster_residuals.py` | 按**流量加权**两级 k-means（`w(c)=cnt_n[c]`），产出 per-layer assign + centroid init；不物化 N×R 矩阵，`sum(local K) == R_n` 精确对齐 | 读上面的 `resid_L*.npz`，输出 `--ngram_layer_assign` / `--ngram_layer_init` 用的 npz |
| `code/train.py` | 消费端已就位 | `--ngram_inject_layers 0,2,5`（0=input，l≥1=block l-1 输出后）、`--ngram_layer_assign "2:path,5:path"`、`--ngram_layer_init "2:path,5:path"`、`table_init_from_npz(model, npz, layer=l)`（train.py:2620） |

**对接的三个硬约束（决定这条蒸馏数据能不能直接用）**：

1. **宽度不匹配（最关键）**：教师 hidden = **2048**，实验室表的行宽 = **n_embd = 768**。
   所以 2048 维教师残差**不能**直接当 `table_<n>` 初始化。两条路：
   (a) 拟合一个固定的 2048→768 线性映射（在**同一批 context** 上配对「教师 depth-k 残差
   ↔ 实验室 layer-l 残差」做 ridge/CCA/Procrustes；几千个共享 context 就够），
   然后把教师的插值/外推残差映射进 768 维当 init；
   (b) **只在结构上使用教师数据**：聚类、负载均衡、哪些 context 该共用一行、
   每行的 centroid 序号——这些是**维度无关**的；具体向量仍由我们自己的 backbone
   （route b 的 `w2_dump_residuals.py`）提供。收益是零跨模型对齐风险。
2. **key 必须逐位一致**：`w2_dump_residuals.py` 用 `NanoGPT._ngram_context_keys_torch`
   （train.py:1199），本会话教师侧用 `ngram_freq.context_keys_np`（ngram_freq.py:328）。
   两者被文档声明为**逐位等价**（torch twin + `test_ngram_dict_cpu.py`），但**没有跨管线验证过**：
   教师侧是在我们自己 2048-token 打包行上算的，而 freq index 是
   「(sequence_len+1) 个 token 一个 chunk、context 不跨 chunk」的口径（ngram_freq.py:367）。
   行边界处的 shift 规则（`t < k` 时取自身）会产生**合成前缀**，边界不齐时最多影响每行
   n-1 = 4 个位置（0.2%）。**第一步就该在两三个真实 shard 上把两种 key 全量对比一次。**
3. **injection 点语义要对齐**：`--ngram_inject_layers` 的 `l` 是「block l-1 输出后」，
   而教师捕获的 `d0/d10/d25/d40` 是「第 k 个 decoder block 之后」。映射不是 1:1
   （40 层教师 vs 8 层实验室），要么按相对深度配对，要么只在**结构层面**用教师数据（见约束 1b）。

→ 建议的落地顺序：先做约束 2 的 key 校验（便宜、决定一切）→ 决定 1a 还是 1b →
再动 §10 的 dump 预算（如果走 1b，dump 的维度和磁盘需求都小一个量级）。



### 10.4 启动 dump 前必须先做的事（顺序）

1. 跑 `validate_dump.py`（+ 补 v4/v5 的等价性测试），确认 int8 回读误差 ~0.4%、
   token id 与 q_star 完全一致、meta sha1 可独立复算。
2. 用 1 个 shard 试跑，核对磁盘增量与耗时（目标 ≤0.15s/批）。
3. 才算力规划：全语料 4.46TB dump + 表聚合需要 **多久**（当前 12k tok/s/worker ×
   4 worker 走完 11.376B 位置 ≈ 66 小时；换成 v3 的 ~41k tok/s/worker ≈ 19 小时）。

---

## 11. 待用户决策

### 11.1 存储架构：RAID0 vs 3 副本（用户问「哪种好」）

- 现状：`/data` 是 **RAID0（无冗余）**，7.0T 里已用 1.2T，产物**无副本**。
- 建议：**分层**——热数据（正在写的 part / dump）放本机 RAID0 换速度，
  冷数据（合并表、计数 summary、checkpoint）**异步 rsync 到一处有冗余的存储**
  （Mac 侧 / 360 内网 / ophis）；不要指望单机 RAID0 承载 TB 级不可重建数据。
- 若要做 4.5TB 的 dump：要么先把 RAID0 扩到 ≥10T 或加挂第二块 RAID，
  要么把 dump 直接写到远端（RDMA/公网），本机只留中间态。

### 11.2 1B 残差流 + 128-cluster ngram 表的落地方案（用户问题）

「机器之间没有强互联」→ 三个可选方向，需你选：

1. **压缩后中心化**（本会话推荐的默认）：按 §10.1 存 int8 128 维 sketch（~392B/位置）
   = 4.5TB/全语料；单机可放，传输量降到 fp32 的 1/26。
2. **分片自治**：每台机器只存/聚合自己那份语料的残差，最后只交换**聚合后的表**
   （11.4M 行 × 3 块 × 2048 × fp32 = 281GB，跨机可传），不搬原始残差。
3. **按需重算**：不存残差，只在需要某个子集时重跑教师前向（用 8 卡 ~19-24h/全语料）。
   只有当访问模式是「一次性」时才划算。

### 11.3 其它

- `fp8` 加速：本机无量化权重；且我们蒸馏的目标就是残差流本身，换精度等于换目标值，
  必须重做 §4 的逐元素校验。等 CPU 修好后前向（0.67s）成为最大单项再评估。
- `/data/counts/` 479GB 的 npz 是否保留（建议留 summary + 按需重算）。
- 本会话 5 个 commit 未 push（§12.3）。

---

## 12. 文件与命令

### 12.1 远端（`pro5kvrdma6:/data/`）

| 文件 | 作用 | md5 |
|---|---|---|
| `distill_prod.py` | 生产 worker（原始版） | `dc75ccf9d98801ae4b15eb3c88fba984` |
| `distill_prod_v2.py` | +32 线程上限（**等价已验证**） | `e88be8ea8140c3636c443ffe8f194425` |
| `distill_prod_v3.py` | +生产者/消费者重叠（**等价已验证**） | `e402f2d0abf8b2db46a16d189f76ab41` |
| `distill_prod_v4.py` / `_v5.py` | = v2/v3 + dump 集成（**未验证**） | `c853f286de1e7a887fafc066158e7b89` / `8ae5341b2fb747f2f652dca3e33d0802` |
| `dump_positions.py` | DumpWriter（int8 sketch + meta） | `1762c21d29e03adde718e2f175a6a83e` |
| `prj_sketch.py` | 投影器（dense/sparse/fft，int8） | `5fa9b85915b0d38a5f686a3513328a57` |
| `merge_distill_parts.py` | 合并 part（`--include_ckpt --mean --parts`） | `ec1e2758a22310e35ceb259466a755c0` |
| `check_distill_aggregate.py` | 聚合级校验 | `ad280b57002c27223d78aad621688c1b` |
| `milestone_watch.py` | 里程碑监视（M1 校验 / M2 出表） | `fa2ea4532945fcb9dd644a1facf6412f` |
| `auto_restart_v2.py` | 受门控的自动切换到 v2/v3 | `2d80207b9b1043471363a21edca0022f` |
| `run_distill_v2.sh` / `run_w3_distill_prod.sh` | launcher（start/status/stop） | — |
| `oe_cap_patch.py` | vLLM 残差捕获补丁（hook 在 `qwen3_5.py` 末尾） | — |
| `counts/count_ngrams.py` + `counts/` | 语料级 n-gram 计数与直方图 | 见 `/data/counts/` |

以上远端文件已**同步回仓库** `code/tools/`（`distill_prod_v2/v3/v4/v5.py`、`dump_positions.py`、
`prj_sketch.py`、`count_ngrams.py`、`validate_dump.py`），计数直方图存为
`docs/notes/data/climbmix_ngram_counts_summary.json` —— 这些文件原本只存在于 pro5k6 一台机器上，
机器一重装就没了。

### 12.2 恢复生产（三条路，按推荐顺序）

```bash
# 0) 现在的暂停态确认
ssh pro5kvrdma6 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; \
                 pgrep -af "distill_prod|milestone_watch|auto_restart" || echo idle'

# ⚠️ 续跑前必读：shard 分配是 range(shard_start+worker, shard_end, num_workers)
#    （distill_prod.py:348），已覆盖的是
#      w0: 0,4,8,12,16 | w1: 1,5,9,13 | w2: 2,6,10,14,18 | w3: 3,7
#    → 缺口 = {11, 15, 17, 19} 加上 ≥20 的全部。直接从 20 开始会漏掉 11/15/17/19。

# A1) 先补 4 个缺口 shard（一次 9-shard 的小 pass：11..19，其中 12/13/14/16/18 会重算，
#     约 5 个 shard 的重复 ≈ 1.9h GPU，换准确性；tag 另起避免与 w3r 混）
ssh pro5kvrdma6 'cd /data && for w in 0 1 2 3; do
  g0=$((2*w)); g1=$((2*w+1))
  CUDA_VISIBLE_DEVICES=$g0,$g1 OE_CAP_LAYERS=0,10,25,40 setsid nohup python3 /data/distill_prod_v2.py \
    --worker $w --num_workers 4 --shard_start 11 --shard_end 20 \
    --token_budget 1e9 --tag w3fix --save_every_shards 1 \
    --enforce_eager --batch_rows 16 --max_num_batched_tokens 32768 \
    > /data/distill_r$w.log 2>&1 < /dev/null &
done'

# A2) 主续跑：≥20 的全部 shard，用已验证的快版（w0-w2 用 v2，w3 用 v3 当金丝雀，
#     这条线还没在真实引擎上验证过 v3）
ssh pro5kvrdma6 'cd /data && for w in 0 1 2 3; do
  g0=$((2*w)); g1=$((2*w+1))
  scr=/data/distill_prod_v2.py; [ $w = 3 ] && scr=/data/distill_prod_v3.py
  CUDA_VISIBLE_DEVICES=$g0,$g1 OE_CAP_LAYERS=0,10,25,40 setsid nohup python3 $scr \
    --worker $w --num_workers 4 --shard_start 20 --shard_end 210 \
    --token_budget 1.8e9 --tag w3s --save_every_shards 1 \
    --enforce_eager --batch_rows 16 --max_num_batched_tokens 32768 \
    > /data/distill_r$w.log 2>&1 < /dev/null &
done; sleep 600; for w in 0 1 2 3; do grep -o "rate=[0-9]* lab tok/s" /data/distill_r$w.log|tail -1; done'

# B) 带残差 dump 的版本（必须先过 §10.4 的验证闸门）
#    distill_prod_v4.py / v5.py + --dump_on 1 --dump_tag w3s --dump_dir /data/distill_dump

# C) 回滚到原始版（口径/复现用）
#    python3 /data/distill_prod.py ...（同参数，tag 另起）

# D) 全程结束后的合并（跨 tag 用 --parts 显式列举，见 merge_distill_parts.py）
#    python3 /data/merge_distill_parts.py --parts /data/distill_out/part_w0_w3r.npz,... \
#        --out /data/distill_out/teacher_table_full.npz --mean
```

### 12.3 本地仓库

本会话 commit（**均未 push**，P5 待批）：
`a3637ea`（milestone watcher + 步时解剖）、`cb947d2`（auto-restart + merge --parts）、
`800798e`（生产 worker/校验器/launcher 入库）、以及本文档的 commit。
改动文件：`code/tools/{distill_prod,distill_prod.py…}`、`code/tools/milestone_watch.py`、
`code/tools/auto_restart_v2.py`、`code/tools/merge_distill_parts.py`、
`code/tools/check_distill_aggregate.py`、`code/cluster/run_w3_distill_prod.sh`、
`docs/notes/data/teacher-distill-pro5k6.md`、本文档。

---

## 13. 时间线（2026-10-21，CST）

| 时刻 | 事件 |
|---|---|
| 09:34 | w0-w2 启动（tag `w3r`，`enforce_eager`，`batch_rows 16`） |
| 12:14 | w3 重启（加 `max_num_batched_tokens 32768`） |
| 12:3x-13:x | 步时解剖：2.70s = 0.67 GPU + 2.03 CPU；192 线程病确诊 |
| 13-15 | `accum_fast.py` 调优（V7 916ms）→ 产出 v2；等价性 PASS |
| 14:36-16:26 | v3 重叠版 + mock 等价性 harness（OVERALL PASS） |
| 15:08 | `milestone_watch.py` 部署（CPU-only 里程碑监视） |
| 15:44-15:48 | w0-w2 写第一个周期性 ckpt |
| 16:22:59 | **w3 被外部 SIGTERM 杀死**（优雅保存到 16:22） |
| 16:41-17:29 | 语料级 n-gram 计数（210 shard，11 分钟）+ 合并（2180s） |
| 16:43-17:11 | dump 集成（v4/v5）+ 投影器（`prj_sketch.py`）+ 各自的 bench |
| 17:14 | 用户指示「暂停所有实验」→ 停 daemon、SIGTERM 四个 worker（w0-w2 优雅保存） |
| 17:15 | 四 worker 全部退出，8 卡空闲 |
| 17:20 | 合并表 `teacher_table_o2to5_partial.npz`（79.83GB）完成 |
| 17:2x | 聚合级校验完成（§5.3） |
| 17:33 | 确认无残留进程、GPU 0 MiB、磁盘 1.2T/7.0T |

---

## 14. 下一步优先级（建议顺序）

1. **闭环两件事**（各 10-30 分钟）：
   (a) §5.3 的语料级 RMS 差异 → 抽 20 篇文档做 per-doc 均值分布；
   (b) §9 的交叉核对 → 974k 老表代入新计数（Spearman + 中位比），决定换表策略。
2. **过 dump 验证闸门**（§10.4）→ 然后决定是否跑全语料 dump（~19h 用 v3）。
3. **2B/10B 里程碑**：现有 0.94B 表可直接用于训练实验；要 2B 需续跑 ~20 个 shard
   （按 v3 速率 ~5h），要 10B 需全语料（~19h）。
4. **决策 §11.1/§11.2 的存储与落地方案**（影响 dump 是否值得在本机落盘）。
