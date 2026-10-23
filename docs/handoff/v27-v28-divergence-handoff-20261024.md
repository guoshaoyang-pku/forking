# Handoff：长训练发散攻坚 + v28/qkn/LR 三线判决 + 隔夜收割（2026-10-24 00:40）

SSOT：结论细节见 `docs/experiment-log.md` §143–§150（本 handoff 的所有数字都在 §150，带 run_id）；发散机制 brief 见 `docs/handoff/divergence-problem-brief-20261019.md` 的接续；图 `docs/figs/v27_divergence_20261023/fig_stab.png`。代码基线 `a22e23c`（v28 launcher 包），本地 7 个 commit 未 push（P5 待批）。

## §0 一分钟版本

1. **发散攻坚出了三条稳定线，强度排序清晰**：qk-norm（零发散 + 保收益 + 不伤 nogram，改动最小）> backbone LR 3e-4（50M dn56 长训救活、净收益 −27 保留，但速度减半）> v28 包（β1 0.9+clip+warmup+cosine：救活全部 17M OE 臂，但 nogram 在 17M s43 / 35M dn80 仍碎）。**clip-only 只推迟不消除（50M 尺度晚期发散确认）**。
2. **104M dn40 干净对首次全健康**（−41 mK @E3 末）：D/N 包络现在 dn2 +17（有害）/ dn20 −87/−59 / dn40 −41，净收益 D/N≈20 峰值、40 仍保持；tpp 规则复现，小-R 有害区（R/N=2 发散、R/N=4 反超 nogram）跨 size 复现。
3. **学生蒸馏**：去均值 4 臂健康跑到 ~20k 步（E3 末前），wave-3 clip1 臂正进入波 1 的 onset 区（8.5–10k）——**clip 能否救 raw 蒸馏臂几小时内出判决**；ctr vs nogram 的 E3 末配对要等 ctr 跑完。
4. **基础设施**：A100 舰队 335 任务 301 done / 17 running / 0 积压；360-1、360-2 SSH 全拒（200M 对、104M s44 对、s43 复本指纹核对全部压后）；ophis 50M 矩阵原始目录已在磁盘清理中删除（读数以登记簿为准），矩阵未重启；a100_t1 6 卡被其他租户占用（keephash 7 臂继续排队）。

## 1. 已确认结果（detail 见 §150 a–e）

| 结论 | 关键读数 | run |
|---|---|---|
| qkn 主设定保收益 | 50m dn20 s42：−35.7 @20345 步 DONE | `v27_50m_{exact,nogram}_r400k_dn20_tls1_s42_e3_qkn_fixed` |
| qkn 17M 保收益 | 17m s43：−53 @6917 步 DONE | `v27_17m_{exact,nogram}_r136k_dn20_tls1_s43_e3_qkn_fixed` |
| qkn 长训零发散 | 35m dn80 对 −63.8、50m dn56 对 −51.6，均过原 onset 2–3× | `*_p27b_qkn_fixed`（t1_2/t1_4/perm） |
| LR 3e-4 救活长训 | 50m dn56 对 −27，exact @46.1k 健康 | `v27_50m_*_dn56_tls1_s42_e3_p27b_lr0.0003_fixed` |
| clip-only 失败 | clip0.25 @46.2k train 5.51（13.2k 时还健康） | `v27_50m_exact_r400k_dn56_tls1_s42_e3_p27b_clip0.25_fixed` |
| v28 救 OE 臂 | 17m 两个必发散配置 E3 跑完：3.240 / 3.766 | `v27_17m_exact_r{136k s43,35k s42}_dn20_tls1_s42/43_e3_v28_fixed` |
| v28 nogram 仍碎 | 17m nogram s43 v28 4.162（s42 3.200）；35m dn80 nogram v28 5.512@24.5k | `v27_17m_nogram_r136k_dn20_tls1_s4{2,3}_e3_v28_fixed`、`v27_35m_nogram_r280k_dn80_tls0.5_s42_e3_p27b_v28_fixed` |
| 104M dn40 全健康 | −41.1 @84633 步 E3 末 | `v27_104m_{exact,nogram}_r832k_dn40_tls1_s42_e3_p27b_fixed` |
| 104M dn20 R 补点 | R/N=1 −14.4 / R/N=2 发散 7.36 / R/N=4 +13.2（劣） | `v27_104m_exact_r{104k,208k,400k}_dn20_tls1_s42_e3_fixed` |
| 104M dn2 OE 有害 | +17.1 | `v27_104m_{exact,nogram}_r832k_dn2_tls1_s42_e3_fixed` |

## 2. 在跑（收割点）

- **A100 V28 组 9 臂**（t1_2/t1_4/perm，21:53 起）：50m dn20 v28 四臂（exact s42 r400k / s43 r100k + nogram×2）60–80%；35m dn80 exact v28 2.751@20.4k 已过原 onset；50m dn56 exact v28 2.705@12.1k 逼近原 onset 13970——**过线即 v28 长训判决**。
- **S2_long**（t1_3）：exact lr3e-4 @46.1k → 57k 收尾；配对 nogram 已 97%。
- **Y3/Y4 qkn 长训对**：35m dn80（45–48k/57k）、50m dn56（25.5–30.9k）→ E3 末同机配对定稿。
- **pro5k 学生**：wave-3 clip1 四臂过 10k（raw 臂 clip 判决）；ctr 四臂跑到 E3 末（~20.3k+）→ ΔVal 表。

## 3. 机制假设状态（截至现在）

- **已证实（17M 探针 + 50M 发散统计）**：发散 = 单层注意力 logit 失控；OE 让第 1 层闲置 → 头停 BOS sink → sink 残差与 q/k 范数增长 → 恒定峰值 LR 下 logit 二次增长。
- **待验证**：跨 size 的直接 stab 探针证据（50m dn56 contract 臂无 stab 记录；v28 exact 带 STAB=100 的 50m dn56 臂到 onset 后可读）；「闲置层因果」的注入层位置实验未做。
- **契约偏离（β1 0.8 / 无 clip / 100 步 warmup / 恒定 LR）是「土壤」不是唯一根因**：β1 0.9 单独使 nogram 也发散（Y2）。

## 4. 基础设施与阻塞

- **360-1 / 360-2**：SSH 均 `Connection closed`。压后事项：200M s43 对、104M s44 对收割；3 个 s43 复本的 assignment 口径指纹核对（bigram 截断应 1000/392/146）。
- **a100_t1**：6 卡其他租户占用，fleet agent 无 claim；keephash 7 臂仍未启动（沿 keephash 会话登记，无新结果）。
- **ophis**：GPU0–3 空闲、4–7 其他租户；runs_fixed 只剩 10 个 104m 目录——50M 表-LR×选择策略矩阵原始目录已在 ENOSPC 清理中删除，读数以登记簿 §82 为准；矩阵重启未发生。
- **未 push**：`ddc981d..a22e23c` 7 个 commit（P5）。

## 5. 下一步 TODO（优先级）

1. **P0 收割判决**（几小时内自然到期）：v28 50m dn56 exact 过 13970 → v28 长训判决；wave-3 clip1 学生臂过 10k → raw 蒸馏 clip 判决；S2 exact lr3e-4 到 57k → LR 3e-4 E3 末配对定稿。
2. **P0 用户拍板契约修复**（P5）：qkn（推荐，最小改动零发散）/ LR 3e-4 / v28 全包。拍板前不再叠加新旋钮。
3. **P1 机制补证**：50m dn56 contract 复现臂补 stab 探针（或读 v28+STAB 臂的 stability.jsonl）；若确认跨 size 同机制，写 §151 机制小结并更新 fig_stab。
4. **P1 104M R-scan 补 s42 r832k / s43 r104k-400k**（dn20 R 峰的跨 seed 完整性；A100 空闲卡可排）。
5. **P2 360 恢复后**：收割 200M/104M-s44 对 + s43 复本指纹核对。
6. **P2 矩阵重启**：ophis GPU0–3 空闲，表-LR×选择策略矩阵用其他空盘重铺（用户已批「硬盘问题用其他的空的盘」）。
7. **P3 push 7 个 commit**（P5 待批）；experiment-lines.md 的 D/N 包络段落补 dn40/dn2 点。

## 6. 本次会话产物

- §150 登记（本文件 §1 的完整 run_id 表在其内）。
- 稳定性探针与 v28 包代码：`ddc981d`（--stability_interval）、`a457d3b`（--qk_norm/--adam_beta1 + 探针扩展）、`097481c`/`d1de981`（§149 判决）、`a22e23c`（v28 launcher 包）。
- `run_v27.sh` md5 `5b9bd62a` 已同步 A100 五机并核对；360-2 同步失败（连接拒绝）需恢复后重做。
