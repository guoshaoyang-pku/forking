# 接管：先收齐长训，再验证注意力归一化

以 `ngram-gap-lab` 为本次交接主体。当前聊天目录 `forking_v2` 只关联学生蒸馏线，不混用训练契约。查询与键归一化（qk-norm）是优先候选。下一批只准备一对长训复本，完整预算通过后再扩展。**新训练未启动，主线契约未修改。**

## 已完成配对支持继续验证，不能推出长期零发散

本次直接回收原始 JSONL 与完成文件。统一使用 ΔVal = 加表 fixed-val − 无表 fixed-val，负值代表加表更好。mnat/token 是每 token 的千分之一 nat。末 60 个共同验证点取均值，最终单点另列，避免混用。时间点不当独立种子。

| 实验 | seed / 共同末窗 | 加表 / 无表窗口 loss | ΔVal 窗口 / 最终单点，mnat/token |
|---|---|---|---|
| 50M 档位，查询与键归一化，数据量/档位参数量=20 | 42 / 19760–20345 | 2.635713 / 2.671724 | −36.011 / −35.697 |
| 104M 档位，原设定，同数据比例 | 44 / 41730–42315 | 2.526305 / 2.548729 | −22.424 / −26.551 |

两对 `summary.json`、最终日志步数和计划步数一致。50M/104M 是内部模型档位，不是含表总参数量。原始证据与曲线来源：

- `outputs/takeover_20261024/live_a100/a100_perm/v27_50m_{exact,nogram}_r400k_dn20_tls1_s42_e3_qkn_fixed/`。
- `outputs/takeover_20261024/360-2/v27_104m_{exact,nogram}_r832k_dn20_tls1_s44_e3_fixed/`。
- `outputs/takeover_20261024/report_metrics.json` 保存统一窗的复算结果。脚本为 `docs/plot_scripts/takeover_20261024.py`，图为 `docs/figs/takeover_20261024/matched_validation.{svg,png}`。

50M 已完成对核过两臂种子、分片、有效 batch、验证形状、学习率日程与 qk_norm 配置。微批量 18/36 由梯度累积补齐，不能称逐位等价。360-2 下载文件逐一核过校验和。代码、数据及 assignment 的完整跨机来源检查仍是正式晋升前门槛。

## 长训当前读数只能支持继续观察

| 配对 | 共同 step / 目标 step | 同步数单点 ΔVal，mnat/token | 限定 |
|---|---:|---:|---|
| 50M dn56 qkn，seed42 | 26630 / 56966 | −69.899 | 同机，未完成 |
| 35M dn80 qkn，seed42 | 46850 / 56966 | −67.135 | 跨机暂估，未完成 |
| 50M dn56 组合修复，seed42 | 13640 / 56966 | −131.941 | 未完成，尚未到旧 onset 13970 |

这里 dn 是独立训练 token 数与模型档位参数量之比。完整标识符与原始路径见 `outputs/takeover_20261024/live_a100/compact_summary.json`。六条长训臂均缺完成文件。不能把“越过旧 onset（原先开始失稳的步数）”当完整训练验收。

交接记录 §150 的 −63.8、−51.6 和 −27 是不同步数末值相减，撤出公平收益表。qk-norm 的有限样本未发散，不等于长期发散概率为零，也不能仅凭健康就说“不伤无表基线”。本轮未重新访问借出节点上的低学习率任务，也未收割学生蒸馏线，因此不更新它们的完成判决。

## 学习率与机制归因必须拆开

`code/train.py` 的实际表学习率等于主干学习率乘 `table_lr_scale`。已有 `NGLAB_LR=0.0003` 实验同时减半了两者。它支持“联合降学习率能稳定部分配置”，不支持“只有主干学习率是原因”。若要隔离主干效应，再新增低主干学习率、表倍率加倍的实验，但不列入第一批。学习率减半没有吞吐减半的证据。

注意力 logit 失控先于 loss 恶化，以及归一化干预的结果，支持尺度失控解释。“表接管局部任务→第一层闲置→注意力集中到句首→失控”仍是机制假说。当前归一化带可学习仿射参数，也不构成 logit 永远有界的数学保证。组合修复 `v28` 同时改变动量、裁剪、预热与衰减，并已有无表臂失稳，暂不晋升。

旧 `parse_v27.py` 不识别 qkn/v28 后缀，配对键漏修复类型、表规模和倍率，并将观察区间末 10% 称为第三遍末尾。本次不使用该汇总。正式队列前修正解析器或继续用显式 run_id 配对，不把旧汇总作为放行依据。

## 第一批只准备两臂，后续按结果放行

| 顺序 | 工作 | 新增 token 预算 | 放行条件 |
|---|---|---:|---|
| 先收齐 | 已有归一化、低学习率、组合修复三遍训练；查收已排的50M dn20 seed43 | 0 | 双臂完成文件、相同步数及输入配置齐全；异常基线不计收益 |
| 第一批候选 | 50M dn56、每阶400k映射、表倍率1、qk-norm，seed43，加表/无表各一臂 | 约16.8B合计 | seed42长训先完整健康收尾；核验重复队列、代码和数据；确认候选契约 |
| 条件扩展 | 35M dn80、每阶280k映射、倍率0.5、qk-norm，seed43配对 | 约16.8B合计 | 前一批两臂健康，seed42/43各自ΔVal均<0 |
| 后置 | 104M表规模补种子，或单一更长预算锚点 | 单独登记 | 仅在仍影响主结论时择一，不并行铺全矩阵 |

已有50M dn20 seed43配对若未启动或因基础设施故障失败，才补跑，额外约6B token。科学发散保留原结果并阻止扩展，不以补跑替代。上述都是**候选队列**，不代表占卡或已注册开跑。全矩阵重启与学生蒸馏的契约更换均不在当前关键路径。

第一批复用已登记的 v27 实验族。它使用2–5阶精确映射、表 AdamW (0.9,0.95)、表裁剪1；主干 AdamW (0.8,0.95)、学习率6e-4、100步预热后恒定。它不同于 AGENTS 中历史1000步、双表 RMSProp/128×契约，不能重新套入旧默认。新实验只补种子，保留归一化候选。固定数据三遍，序列2048，有效batch147456 token，每10步固定验证288个片段。稳定性探针每100步写入。

第一批候选 run_id：

```text
v27_50m_exact_r400k_dn56_tls1_s43_e3_p27b_qkn_fixed
v27_50m_nogram_r400k_dn56_tls1_s43_e3_p27b_qkn_fixed
```

在 `a100_perm` 上核实可用的工作目录是 `/data2/guoshaoyang/ngram-gap-lab-v27`，位于 ext4 数据盘。下面是每臂命令模板。开跑前在 `experiment-log.md` 注册独立行、分配 GPU，核对代码与输入指纹，并确认 run_id 尚未存在。`NGLAB_QKN=0` 或 `NGLAB_V28=0` 仍会被 launcher 当开启，因此必须清除无关变量。

```bash
# 前置：已完成登记与放行，并设置本次分配的 GPU、ARM=exact 或 nogram。
: "${GPU:?指定已分配GPU}" "${ARM:?指定exact或nogram}"
TARGET_ROOT=/data2/guoshaoyang/ngram-gap-lab-v27
TARGET_PY="$TARGET_ROOT/.venv/bin/python"
if [ "$ARM" = exact ]; then DB=18; else DB=36; fi
env -u NGLAB_V28 -u NGLAB_B1 -u NGLAB_CLIP -u NGLAB_LR \
    -u NGLAB_SUBSET -u NGLAB_SUBSET_PASSES -u NGLAB_HEADS -u NGLAB_KK \
    NGLAB_ROOT="$TARGET_ROOT" NGLAB_PY="$TARGET_PY" \
    NGLAB_DATA_DIR_V27="$TARGET_ROOT/data/tokenized_v27" \
    NGLAB_FREQ_INDEX="$TARGET_ROOT/data/freq_index_o2to5_1b_f30.npz" \
    NGLAB_ASSIGN_DIR="$TARGET_ROOT/data/assignments" \
    NGLAB_DN=56 NGLAB_RK=400 NGLAB_TLS=1 NGLAB_EPOCHS=3 \
    NGLAB_DB="$DB" NGLAB_VDB=24 NGLAB_TAG=_p27b \
    NGLAB_QKN=1 NGLAB_STAB=100 \
    bash "$TARGET_ROOT/code/cluster/run_v27.sh" "$GPU" 50m "$ARM" 43
```

结果写到远端 `data/runs_fixed/<run_id>/`，该 run_id 已带 `_fixed`。不能覆盖已有结果。准确步数以 launcher 对数据块舍入后的预算为准，现有同配置为56,966步。GPU小时由首批各臂实测吞吐估计，不伪造固定ETA。

验收预先固定：使用完整预算末60个共同fixed-val点，同时报告两臂绝对loss、差值、每seed发散情况、原始未平滑曲线及探针。无表基线异常时整对不参与正收益判决。时间点不充当独立样本，不在只有两个seed时声称总体显著。

停止与扩展规则：非有限loss停止新启动的该臂并保存日志；单次尖峰只告警。沿用30条日志均值高于历史最低值0.10的规则作为诊断告警，不自动认定发散。训练与验证持续恶化需人工核对数据边界和探针。任一臂失败即停止下一批扩展，不自动改配置重试。既有任务本轮不启停。

## 保存与算力门槛已经更新

360-2 已恢复读取，但 `/data` 满盘，禁止据此直接派发新任务。200M seed43 两臂仍推进，无表臂明显失稳且都没有完成文件，不能计作健康配对。360-1 此次连接超时。现存assignment计数指纹与历史记录匹配，但不能证明历史进程加载版本。

`a100_perm` 此次可读且数据盘有空间，部分卡无显存占用，实际分配须开跑时重新检查。ophis低利用率卡仍有显存占用，不能当已预约空卡。`a100_t1`仍有六卡被占用。未操作其他租户进程。

本地 `data` 是断开的外置盘链接，本轮没有改动它。新证据保存到 `outputs/takeover_20261024/`。已补 `.gitignore` 的 `outputs/`、`checkpoints/`、`wandb/`、`artifacts/`、`*.safetensors`。已有追踪文件未发现超过50MiB，原有未追踪内容保持原样。

交接引用提交 `2fcbefb`。fetch后，当前HEAD相对已配置upstream为0落后/138领先，不能直接把“7个未push提交”当完整待推范围。未push、未改分支。正式改主线契约和push分别按仓库P5确认，不以此次接管准备替代这些决策。

来源：`docs/handoff/v27-v28-divergence-handoff-20261024.md`、`docs/experiment-log.md` §149–150、本轮原始日志与代码。复核日期：2026-10-24。
