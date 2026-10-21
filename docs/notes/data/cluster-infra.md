# 集群算力与存储坐标

> 本文件由 `agents.md` §4 与 §5 offload（2026-08-26）。规则与可用算力摘要仍保留在
> `agents.md` §4；这里是**完整目录细则**，修改存储路径/配额时同步更新本文件与 agents.md §4 摘要。

## 可用算力（总览，权威）

| 集群 | 连接 | GPU | 公网 | 环境 | 状态 |
|---|---|---|---|---|---|
| **ophis-gpu**（主） | 直连 SSH `guoshaoyang@223.167.85.180:50002`，别名 `ophis-gpu` / `ophis_gpu` / `fcloud-223` | 8×H200 (141 GB) | ✅ | `uv` + torch 2.9.1 | ✅ 可用 |
| **360-1** | QConnect VPN → `10.234.161.2:22`，`ssh 360-1` | 8×H200 (143.7 GB) | ❌ | 系统 python3.10 + torch 2.13.0+cu130 + numpy 2.2.6 | ✅ 可用 |
| **360-2** | QConnect VPN → `10.234.161.3:22`，`ssh 360-2` | 8×H200 (143.7 GB) | ❌ | 同上 | ✅ 可用 |
| **a100_perm**（永久整机） | 直连 SSH `ubuntu@139.199.90.57:22`，`ssh a100_perm` | 8×A100-SXM4 (40 GB) | ✅ | Python 3.10.12 + torch 2.9.1+cu126 + numpy 2.2.6 | 已迁移；2026-09-25 验收见下 |
| **a100_t1**（Tione） | 直连 SSH `root@111.230.4.68:10925`，`ssh a100_t1` | 8×A100-SXM4 (40 GB) | ✅ | 同上，独立项目环境 | 已迁移；2026-09-25 验收见下 |

SSH 配置位于 `~/.ssh/config.d/`（主配置 `Include ~/.ssh/config.d/*.conf`），原计算节点在 `20-compute.conf`，A100 两台在 `35-a100.conf`。

360 系**无公网、且不能直连 ophis-gpu**（223.167.85.180 超时）。跨集群搬运走 **ophis-gpu → Mac → 360** 中转。

## 各集群的内容存储

### ophis-gpu（主集散地：全部权威实验数据在这里）

**仓库实体在 `/data4` 盘上**：`/data4/guoshaoyang/ngram-gap-lab/`（nvme4n1p1，7.0 TB）。
家目录 `/data/home/guoshaoyang` 在 `/data3` 盘上，其中 `~/ngram-gap-lab` 是指向 `/data4` 实体的符号链接——两条路径等价，引用时优先写 `/data4/...` 实体路径。

| 路径 | 容量 | 存什么 |
|---|---|---|
| `/data4/guoshaoyang/ngram-gap-lab/data/runs_fixed/` | 38 G，163 run | ★ **唯一权威 run 数据**（agents.md P4：只认 `_fixed` 后缀；含 8 个 `_fd` fast-diag run） |
| `/data4/guoshaoyang/ngram-gap-lab/data/runs_scaling/` | 44 G | epoch20 / scaling 系列，其中 56 个带 `_fixed` 后缀；另含 `logits_rank/` 分析产物与队列日志 |
| `/data4/guoshaoyang/ngram-gap-lab/data/runs_theory/` | 37 G | freeze / snap 理论干预 run（freezebb、input、nogram 的 snap+log） |
| `/data4/guoshaoyang/ngram-gap-lab/data/tokenized/` | 1.4 G | 标准 1x token shards |
| `/data4/guoshaoyang/ngram-gap-lab/data/tokenized_epoch20/` | 2.2 G | epoch20 长序列 shards |
| `/data4/guoshaoyang/ngram-gap-lab/data/freq_index*.npz` | 14 个文件，共 ~6.7 G | 频率索引：`freq_index.npz`（标准 1x）+ `freq_index_train{0.25x…8x}.npz` |
| `/data4/guoshaoyang/ngram-gap-lab/data/ngram5_minimal_order5/` | 9.3 G | ngram5 controlled 数据集（历史） |
| `data/runs/`、`runs_quarantine_epochbug/`、`runs_v50_1000_bak/`、`nglab2x_runs.tar.gz` | <1 G | ⛔ 作废/隔离/备份（`data/runs/` 受 freq-bin bug 污染，见 agents.md P4） |
| `/data3/guoshaoyang/ngram-gap-exp/` | — | **历史工作区**（OPHIS 时代）：旧 `train.py` / `lib.py`、`ngram5_data/`、`runs/ngram5/`、`toy/` |
| `/data3/guoshaoyang/ophis_gap_local_backup/` | — | 本地镜像备份 |
| `/data2/guoshaoyang` | 7.0 TB | 共享实验目录 |
| `/scratch/guoshaoyang`、`/tmp` | 438 GB 但**配额仅 15G soft / 20G hard** | ⚠️ **禁止写大文件**：`/tmp` 与 `/` 同分区，根分区配额已超 |

**合作者访问（yushanbin / zhaohaoran，2026-09-17 已开通）**：两人在 ophis-gpu 有账号（同在 `gpuusers` 组），GitHub `forking` 仓库已加 collaborator（有 push 权限）。集群侧已执行：`chmod 711 /data4/guoshaoyang`（穿越位，全世界可过但不可列目录；数据本体 644/755 对他们是**只读**，`runs_fixed` 无写权限）。共享可写目录 **`/data4/guoshaoyang/shared/`**：对两人 `setfacl rwx`（含 default ACL，新建文件自动继承），画图中间产物/成品统一放这里；他们自己的 `/data4/yushanbin/`、`/data4/zhaohaoran/` 也可用。撤销方式：`setfacl -x u:<user> /data4/guoshaoyang/shared` + `chmod 700 /data4/guoshaoyang`。

### 360-1 / 360-2

> 本节为历史记录，2026-09-17 盘点时 VPN 未连通、未复核；360 上的 run 产物按 P3/P4 回填后应回流到 ophis-gpu 的 `runs_fixed/` 才算权威。

| 路径 | 存什么 |
|---|---|
| `/data/home/guoshaoyang/ngram-gap-lab/` | **本仓库副本**（~1.4 GB）：`code/` + `data/tokenized/`（12 shard）+ `data/freq_index*.npz` |
| `/data/home/guoshaoyang/ngram-gap-lab/data/runs/` | run 产物 |
| `/data/home/guoshaoyang/ngram-gap-exp/toy/` | **toy 实验专用工作区**（360-2，~425 MB）：`ws/`（harness）、`data/`、`cache/`、各 `toy*_launch.sh` |
| `$ROOT/.inductor_cache` | torch inductor 编译缓存 |
| `/tmp` | ⚠️ **360-2 的 `/tmp` 只有 974 MB**，曾被 inductor 缓存写满。**禁止写 `/tmp`**，编译缓存一律走 `$ROOT/.inductor_cache` |

**2026-10-21 事故与规程（`/tmp` 满 → sshd 拒登）**

- **症状**：360-2 `/tmp`（独立 974 MB 卷）达 100%（907M used / 0 avail）后，`nc` 测 TCP 22 端口**仍可连**，但 SSH 在密钥交换阶段被远端关闭：`kex_exchange_identification: Connection closed by remote host` → **登录被拒，但训练进程不受影响**（GPU 上的 run 照跑，harvest 只是暂时取不到）。
- **定位**：`ssh 360-2 'df -h /tmp; du -sh /tmp/* 2>/dev/null | sort -h | tail -20'`。历史元凶在案是 inductor 缓存；我方本次在该机只写过 1 个 ~2 KB 脚本（且因 ENOSPC 只落了 0 字节）。
- **硬规则（所有 360 机器）**：禁止往 `/tmp` 写任何东西——脚本、base64 中间件、日志、转储都不行；`/` 根分区配额同样已超，也别写。
  - 本地脚本 → 远端：走管道，`ssh 360-2 'python3 -' < script.py`（或 `base64 -d | bash`），不留文件。
  - 远端临时目录：`$ROOT/.tmp/`（在 `/data` 上）；nohup 日志：`data/runs_fixed/_nohup_*.log`；编译缓存：`$ROOT/.inductor_cache`；需要 TMPDIR 的工具显式 `TMPDIR=$ROOT/.tmp`。
  - 写远端前的 preflight：`ssh 360-2 'df -h /tmp /'`（任一 >90% 先停下报人）。
- **安全清理步骤**（SSH 恢复后或走控制台/root 执行）：
  1. 只读盘点：`du -sh /tmp/* 2>/dev/null | sort -h | tail -30`；大文件：`find /tmp -maxdepth 1 -user guoshaoyang -size +10M -ls`。
  2. 查占用：`lsof +D /tmp 2>/dev/null | head -30` —— 有进程打开的文件**不许删**。
  3. **绝不 `rm -rf /tmp/*`**：`/tmp/.X11-unix`、`/tmp/.ICE-unix`、`/tmp/systemd-private-*`、`ssh-*`（agent socket）、他人账号的文件都在里面。
  4. 先干跑再删，只动**自己人且陈旧**的项：`find /tmp -maxdepth 1 -user guoshaoyang -mtime +0 -ls` 逐条确认后 `-delete`（或逐个 `rm`）。
  5. 若 sshd 因此拒登，登录侧无解 → 需控制台/root 清理；恢复后 `systemctl status sshd` 复核，再按 §4 强制检查重跑 harvest。

**分工约定**：
- 主线 nanoGPT 实验：ophis-gpu 或 360-1/360-2 均可（代码同步后口径一致）。
- toy / 合成数据实验：默认跑 **360-2**（集中算力）；ophis-gpu 不再启动新 toy run，历史 toy 数据保留在 `/data3/guoshaoyang/ngram-gap-exp/toy`。

## 跑实验前的强制检查

### A100 两台迁移坐标（2026-09-25 现场核验）

| 项目 | a100_perm | a100_t1 |
|---|---|---|
| 机器性质 | 用户确认的永久整机 | Tione 容器；实例生命周期未验证 |
| 系统 | Ubuntu 24.04.2，kernel 6.8.0-87 | Ubuntu 22.04.4，host kernel 5.4.119 |
| CPU / RAM | AMD EPYC 7K83，232 logical CPU，约 1.8 TiB RAM | 同型号 / 容量；cgroup CPU 配额 229 核 |
| GPU | 8×A100-SXM4-40GB，MIG disabled | 8×A100-SXM4-40GB，MIG disabled |
| NVIDIA driver | 580.178.04；nvidia-smi 显示 CUDA 13.0 | 535.183.06；nvidia-smi 显示 CUDA 12.4 |
| 系统 Toolkit | 未安装 nvcc；运行训练使用 wheel 自带 runtime | nvcc 12.4.131；训练仍使用 wheel 自带 runtime |
| 大盘 | `/data`、`/data2` 各约 7.0 TiB，已见 fstab 挂载项 | `/home/tione/guoshaoyang` 是 XFS 大盘的 `/notebook` 挂载，约 14 TiB；不把 `/run/ti` 的平台工具目录作为项目路径 |
| 项目 root | `/data/guoshaoyang/ngram-gap-lab` | `/home/tione/guoshaoyang/ngram-gap-lab` |
| Python | `$ROOT/.venv/bin/python`；uv 管理的 CPython 3.10.12 | 同左 |
| 训练依赖 | torch 2.9.1+cu126、CUDA runtime 12.6、numpy 2.2.6 | 与永久机完整 64 包锁文件相同 |

代码从本地已提交 `09ee8bb27119fac9fa5bcda4becb6c2e2c59c756` 导出 `code/` 与 `requirements.txt`（84 文件），两机 MD5/SHA-256 全部一致。保留部署快照，不把本地未提交代码混入远端。后续同步仍以本地已 commit 的版本为准。

标准输入从 ophis-gpu 的 `/data4/guoshaoyang/ngram-gap-lab` 迁移：`data/tokenized/shard_00001.bin` 至 `shard_00010.bin`、`shard_06542.bin` 和 `data/freq_index.npz`，共 12 文件，逐文件 SHA-256 已对源端核验。较大语料、历史结果与 checkpoint 不属于本次基础环境包。

进入项目后执行 `source ./env.sh`，固定解释器、输入、输出、uv/Hugging Face/Torch 缓存与临时目录在项目大盘下。两台机器均保留既有其他项目及其虚拟环境。

验收与锁文件见 [A100 onboarding 记录](a100-onboarding-20260925/README.md)。基础检查已覆盖两台共 16 张卡的 bf16 matmul backward、Flash SDPA backward、训练模块 import。正式使用前仍应检查 GPU 空闲；硬件验收不会预约卡。

**A100 容量边界**：40GB 不等于 H200 的 141GB。默认 `run_baseline.sh` 的 device batch 72 尚未通过 A100 容量验证。microbatch 变化还会影响当前训练 batch 的频率统计与 fixed-val batch 范围，不可仅按 total batch 相等就宣称完全同口径。正式 A100 运行应先登记并核验该差异；本次工程 smoke 只写 `provisioning/smoke/`，不进入 `data/runs_fixed/` 或论文证据。

Tione 数据目录位于大盘已核实，但实例销毁/重建后的保留策略尚未验证；重要产物完成后按登记回流 ophis-gpu。两机新增 16 张 A100，不与既有 H200 合并计算等效性能。

### 每次开跑

1. `nvidia-smi` 确认目标卡空闲，用 `CUDA_VISIBLE_DEVICES=<id>` 占卡；一个 Agent 只用自己登记的卡。
2. 同步代码到目标机，然后 `md5sum` 核对至少 `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`。
   - 权威源：本地 git 仓库已 commit 的版本，或 ophis-gpu `/data4/guoshaoyang/ngram-gap-lab`。
   - 教训（2026-08-06）：360 上曾残留旧 `train.py`（`f9388473`），与 ophis-gpu（`05bffab8`）的 val / freq-val 口径不同（旧版 freq-val 是移动窗口，新版固定 batch），导致同批实验口径不一致。
3. 改代码前先 commit，再同步到所有目标机。同一实验集跨机并行必须用同一份代码。

## Workspace（本地坐标）

| 用途 | 路径 | 状态 |
|---|---|---|
| **主开发仓库** | `/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab` | ✅ 当前唯一开发地。GitHub: `git@github.com:guoshaoyang-pku/forking.git`（仓库名 forking；`data/` 被 gitignore，代码+文档+绘图脚本走 git，数据本体在 ophis-gpu） |
| **发布博客仓库** | `/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io` | ✅ 主文档发布地。主页面 `blogs/ngram-gap-mechanism-guide/index.html` |
| 旧仓库（弃用） | `/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap` | ⛔ **已弃用**，只读溯源，不再开发。见 `deprecated-list.md` |
| 两因素模型参考 | `/Users/guoshaoyang/Documents/Codex/2026-08-21/xian-xi/outputs/ngram-repeat-gap-two-factor-model.html` | 📄 外部理论文档，待验证对象 |

**规则**：新文件、新代码、新实验一律落在 `ngram-gap-lab`。需要 OPHIS 里的东西就 `cp` 过来并在 `docs/_archive/docs/` 登记来源，不要跨仓库引用路径。
