# A100 夜间接管：只执行已登记清单

用户已授权四台 A100 彻夜执行有意义的实验。本轮清单是 `docs/plans/overnight-20261024.json`，逐臂登记在 `docs/experiment-log.md` §151。43 个候选包括长训换种子、六遍重复、学习率解耦和必要同机控制。全清单完成上限为212.46B token，预计约259 GPU小时，不承诺早晨全部完成。

**北京时间2026-10-24 08:00停止领取新任务。已启动任务完成其三遍或六遍预算。未领取任务保持待定，不自动延期。** 不改主线默认，不新增配置，不自动补跑科学发散。

01:28实机验收：32张卡均在训练，17个新臂在相隔176秒的快照中全部推进，26臂待领，其余15个旧训练保留。两个学生任务自然完成后，J42/J43整对已在`t1`的GPU0/7启动。该数字是部署验收记录，接手后必须重新执行下方状态命令。

## 接手后先跑这一条

```bash
cd /Users/guoshaoyang/Desktop/workdir/ngram-gap-lab
python3 code/cluster/overnight_status.py
```

完整机器可读状态保存在 `outputs/takeover_20261024/overnight_status.json`。`occupied` 只表示显存已占用，不等于有效训练。`progressed=x/y` 表示与上一快照可比较的新训练中有x条步数增长；第一轮无可比较证据时会是0/0。上一快照保存在同目录 `overnight_status.previous.json`。`finished_unreviewed` 只表示进程结束，未代表科学成功。不要使用旧 `parse_v27.py` 或默认访问借出节点的 `overnight_tick.sh`。

| 状态 | 固定动作 |
|---|---|
| `active` 且 step 增长，无告警 | 不改训练，让调度器续接。 |
| 有排队任务、空卡持续两次检查（间隔至少5分钟） | 检查对应 `agent_<host>.log`、空间及GPU锁。调度器缺失才用下方启动命令。 |
| `pending_without_agent` | 截止时间前按下方命令恢复唯一调度器。不要复制队列或删除claim。 |
| `starting` | 等待启动，最多10分钟。不能因暂时零显存启动第二份。 |
| `needs_process_check` / `no_progress_15min` | 核对 claim 中 wrapper_pid、boot_id、远端 cmdline 与日志。保留现场并报告，不盲目重跑。 |
| `failed`、非有限loss或完成文件不一致 | 保留失败，不改种子掩盖失败。只停止经PID及run_id核对的本次非有限任务。单次尖峰只告警。 |
| SSH失败、磁盘不足20GiB | 保留任务。不得删除已完成结果腾空间，不得换盘写入未核路径。 |
| 已到08:00 | 停止新领取。已启动的任务继续到注册终点。未领任务报告待定。 |

## 只允许恢复这四台的调度器

```bash
python3 code/cluster/start_overnight_20261024.py --host a100_perm
python3 code/cluster/start_overnight_20261024.py --host a100_t1_2
python3 code/cluster/start_overnight_20261024.py --host a100_t1_4
python3 code/cluster/start_overnight_20261024.py --host a100_t1
```

每条命令会检查截止时间、队列与源码指纹、旧调度器是否退出，已有新agent时不重复启动。主机级 flock 保证只有一个守护循环。GPU锁由训练wrapper持有，调度器重启不能覆盖活跃训练。`a100_t1_3` 已借出，禁止访问或向旧共享队列追加任务。

独立队列：`/home/cfs/guoshaoyang/nglab_queue_overnight_20261024`。只读入口为 `jobs.txt`、`claims/<key>/claim.json`、`rc`、`launcher.log`、`agent_<host>.log`、`supervisor_<host>.log`。host 名：永久机为 `perm`，其余等于SSH别名。`rc=0` 仍需核对 `summary.json`、最终step与loss。

旧主队列只drain了 `perm / a100_t1_2 / a100_t1_4` 的dispatcher。学生队列也在确认0条待领后drain。原训练全部保留，空出的卡由新队列接续。**不要移除旧STOP，也不要恢复学生旧dispatcher**，否则它会重新争抢全部GPU。若另一个会话需要追加学生任务，先协调GPU白名单并使用同版锁协议。

## 结果只按同机、同种子、同窗口配对

有新完成或失败项时，先刷新状态，再执行：

```bash
python3 code/cluster/collect_overnight_20261024.py
```

它只读取已终止任务与登记对照的小日志，按host/run_id保存到 `outputs/takeover_20261024/overnight_results/snapshots/<UTC时间>/`。终端会打印 `review.json` 路径。它核对校验和、登记预算、完成文件与配置，不下载权重、不改远端。没有终止任务时输出空清单是正常行为。

`manual_review` 表示证据缺失或不匹配，不跳过检查硬算。`validated_numeric_comparison` 只表示数值比较通过检查，仍要看两臂绝对loss与完整曲线，不能直接称科学成功。`comparison` 会区分“加表−无表”和“归一化干预”，所有比较都要求科学复核。未知或借出节点上的控制留为blocked，不访问。失败项原样登记，不能从汇总删除。

依据manifest的 `control_refs` 找对照，不自己猜。部分历史同名run在两台主机都有副本，键必须包含host。核对训练/验证分片、有效batch、种子、归一化、两组绝对学习率、表规模、映射、终点和源码指纹。

主指标是末60个**共同**fixed-val点的加表−无表均值，同时报告各臂绝对loss和最终单点。两臂均有完成文件且达到注册预算，才称完整结果。无表基线失稳时不计算“巨大收益”。科学发散也登记，不能只收存活臂。六遍与三遍的比较是预算干预，不是新增独立种子。

**已知例外不可自动豁免**：J42/J43已跑时发现DB18/36导致每遍pool差36,864 token（2B的0.0018432%），终点40,690/40,689步。两条保留作稳定性/探索证据，不计严格主收益，collector留为manual_review是正确行为。未领取的J39已改DB18，与J38同pool、同终点。不要为凑“通过”修改已跑配置或重新贴标签。

本地 `data` 是断开的外置盘链接。小型原始日志必须收割到 `outputs/takeover_20261024/`，按host/run_id分目录，保留校验和，不能写到源码目录或覆盖旧副本。代码/文档可提交，训练产物不得进入git。没有授权push或发布博客。

## 后续模型直接使用这段任务说明

> 继续本次A100夜间登记实验。先跑状态命令，严格按状态表处理。只恢复缺失的已授权调度器，不直接调用训练launcher，不删claim，不移动配对，不新增实验。完成或失败后收割原始证据并回填§151，比较时保留host与run_id。遇未覆盖的异常保留现场并报告，不自行改超参。08:00后停止新领取，已启动训练跑到注册终点。报告只说新完成、失败、恢复、持续空卡或需要决策的事项。

## 守护范围与限制

远端守护循环负责在08:00前恢复意外退出的调度器；每个训练只领一次，不自动重试。它不是主机重启服务。机器重启、SSH中断、外部占卡或科学失败不能用“保证不空卡”掩盖，按状态表处理。人工/聊天巡检只在完成、失败、持续空卡或需要决策时通知。

已创建本聊天的原生heartbeat `a100`，每20分钟巡检，有界至北京时间08:20。截止后只做最后查收并暂停，不继续开任务。远端接续已实际运行，不依赖本机聊天在线。heartbeat创建成功，首次自动唤醒尚未验收。

部署证据在 `outputs/takeover_20261024/`：`deploy_receipts.json`、四份`fingerprint_*.json`、`launch_audit.json`、`launch_progress_verification.json`、`watchdog_verification.json`。最后一项是t1_2调度器退出恢复的实测，5个原训练wrapper保持且继续推进，无重复任务。训练/调度代码版本为`44fc73d`，整对迁移版本为`57b51b1`。清单的status是登记状态，运行状态只认新鲜claim、进程与训练日志。

用户随后明确停止时，立即停止新领取；具体是否结束已跑训练按用户原话执行。禁止把此次彻夜授权扩展成无期限填卡。
