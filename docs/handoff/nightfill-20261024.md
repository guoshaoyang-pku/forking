# Nightfill58：恢复有效队列并补齐 Scaling 对照（2026-10-24）

用户本轮明确授权：优先恢复队列，直接排合适任务。root 于本轮注册58个新独立E3任务；这不是恢复已截止的旧夜间队列。禁止访问借出节点 a100_t1_3；不改旧STOP、manifest、claim、预算或运行中训练，不push，不删除产物。

## 科学目标与协议

沿用已审核的v27/cap24/capnext（§153），不是早期v5 RMSProp极简线。该延续已包含n=2/3/4/5独立exact top-R表、input注入、AdamW表(.9,.95)、table_grad_clip=1；主干AdamW(.8,.95)、LR6e-4、warmup_constant100、bf16无compile、learned absolute PE、8192词表。模型配置由固定run_v27来源定义。E3；DB18、总batch147456，fixed-val shard2的288chunks、VDB24，每10步验证；clean_v27数据/索引SHA逐文件预检。每项只改变其登记科学坐标，并复用同host同seed对照。

新增诊断（所有新臂相同）：独立uniform fixed-train probe16批=288blocks，间隔500步并按源码epoch/终点触发。它不消耗训练iterator、无全局随机采样，不改optimizer更新；online train更新前、fixed-val更新后的原口径继续作为主指标。fixed-probe比较必须核对其输入hash，不把旧无probe run纳入fixed-gap配对。

主判据：两臂预算完整后末60共同fixed-val点；同时报告同点online train gain、额外online Gap和末点。种子才是重复单位，时间点不能当seed。U扫描固定E3，因此数据与总计算同时变化，不是iso-compute。104M现有8臂在本次实时预检均有限且推进，仅是启动许可，不是完成结论。

## 队列与边界

冻结计划 docs/plans/nightfill-20261024.json；命令 docs/plans/nightfill-20261024.jobs。
58臂共126.3B名义tokens；a100_t1=16、a100_t1_4=26、perm=8、a100_t1_2=8。
新队列 /home/cfs/guoshaoyang/nglab_queue_nightfill_20261024。
新领取截止2026-10-25 16:56:48北京时间；已启动仅跑登记E3。无自动重试。
GPU门：连续300秒无compute PID且显存低于512MiB；查询间隔>90秒或GPU枚举不一致重置；继承同UID flock，锁内复查。旧队列同host任务必须全claimed才允许共存。source、数据、输出不存在和磁盘检查通过后才启动。

注意：run_v27以环境变量非空判断QKN，NGLAB_QKN=0仍开启。因此全部nonq任务彻底省略此变量，fleet清理继承NGLAB_*。旧未执行capnonq提案的0错误不影响已经完成的50M nonq实验。

操作入口（仓库根）：
python3 outputs/nightfill_20261024/nightfill_operator.py preflight
python3 outputs/nightfill_20261024/nightfill_operator.py start --host a100_t1
python3 outputs/nightfill_20261024/nightfill_operator.py status
python3 outputs/nightfill_20261024/nightfill_operator.py collect
start仅单host，另可perm/a100_t1_2/a100_t1_4。所有状态和小日志位于outputs/nightfill_20261024；实际生命周期以运行回执为准，冻结manifest的registered_not_launched不覆盖状态。

## 新登记逐臂

| ID | Host | run_id | Steps | 状态 |
|---|---|---|---:|---|
| R01 | a100_t1 | `v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed` | 6918 | planned |
| R02 | a100_t1 | `v27_17m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed` | 6918 | planned |
| R03 | a100_t1 | `v27_17m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed` | 6918 | planned |
| R04 | a100_t1 | `v27_17m_exact_r25k_dn20_tls1_s45_e3_nightfill24_fixed` | 6918 | planned |
| R05 | a100_t1 | `v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed` | 6918 | planned |
| R06 | a100_t1 | `v27_17m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed` | 6918 | planned |
| R07 | a100_t1 | `v27_17m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed` | 6918 | planned |
| R08 | a100_t1 | `v27_17m_exact_r25k_dn20_tls1_s46_e3_nightfill24_fixed` | 6918 | planned |
| R09 | a100_t1_4 | `v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed` | 6918 | planned |
| R10 | a100_t1_4 | `v27_17m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed` | 6918 | planned |
| R11 | a100_t1_4 | `v27_17m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed` | 6918 | planned |
| R12 | a100_t1_4 | `v27_17m_exact_r25k_dn20_tls1_s47_e3_nightfill24_fixed` | 6918 | planned |
| R13 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls1_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R14 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls1_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R15 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls1_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R16 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls1_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R17 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls1_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R18 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls1_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R19 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R20 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R21 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R22 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R23 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R24 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R25 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R26 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R27 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R28 | a100_t1_4 | `v27_35m_exact_r6k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R29 | a100_t1_4 | `v27_35m_exact_r12k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R30 | a100_t1_4 | `v27_35m_exact_r25k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed` | 14241 | planned |
| R31 | a100_t1 | `v27_17m_nogram_r136k_dn10_tls1_s45_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R32 | a100_t1 | `v27_17m_exact_r12k_dn10_tls1_s45_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R33 | a100_t1 | `v27_17m_nogram_r136k_dn40_tls1_s45_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R34 | a100_t1 | `v27_17m_exact_r12k_dn40_tls1_s45_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R35 | a100_t1 | `v27_17m_nogram_r136k_dn10_tls1_s46_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R36 | a100_t1 | `v27_17m_exact_r12k_dn10_tls1_s46_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R37 | a100_t1 | `v27_17m_nogram_r136k_dn40_tls1_s46_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R38 | a100_t1 | `v27_17m_exact_r12k_dn40_tls1_s46_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R39 | a100_t1_4 | `v27_17m_nogram_r136k_dn10_tls1_s47_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R40 | a100_t1_4 | `v27_17m_exact_r12k_dn10_tls1_s47_e3_nightfill24_qkn_fixed` | 3459 | planned |
| R41 | a100_t1_4 | `v27_17m_nogram_r136k_dn40_tls1_s47_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R42 | a100_t1_4 | `v27_17m_exact_r12k_dn40_tls1_s47_e3_nightfill24_qkn_fixed` | 13835 | planned |
| R43 | perm | `v27_50m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed` | 20345 | planned |
| R44 | perm | `v27_50m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed` | 20345 | planned |
| R45 | perm | `v27_50m_exact_r50k_dn20_tls1_s45_e3_nightfill24_fixed` | 20345 | planned |
| R46 | perm | `v27_50m_exact_r100k_dn20_tls1_s45_e3_nightfill24_fixed` | 20345 | planned |
| R47 | a100_t1_2 | `v27_50m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed` | 20345 | planned |
| R48 | a100_t1_2 | `v27_50m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed` | 20345 | planned |
| R49 | a100_t1_2 | `v27_50m_exact_r50k_dn20_tls1_s46_e3_nightfill24_fixed` | 20345 | planned |
| R50 | a100_t1_2 | `v27_50m_exact_r100k_dn20_tls1_s46_e3_nightfill24_fixed` | 20345 | planned |
| R51 | perm | `v27_50m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed` | 20345 | planned |
| R52 | perm | `v27_50m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed` | 20345 | planned |
| R53 | perm | `v27_50m_exact_r50k_dn20_tls1_s47_e3_nightfill24_fixed` | 20345 | planned |
| R54 | perm | `v27_50m_exact_r100k_dn20_tls1_s47_e3_nightfill24_fixed` | 20345 | planned |
| R55 | a100_t1_2 | `v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 42318 | planned |
| R56 | a100_t1_2 | `v27_104m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 42318 | planned |
| R57 | a100_t1_2 | `v27_104m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 42318 | planned |
| R58 | a100_t1_2 | `v27_104m_exact_r50k_dn20_tls1_s47_e3_nightfill24_qkn_fixed` | 42318 | planned |

## 每臂对照与交付

### R01 · v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=a100_t1；E3；U=339996672；R/阶=136k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_capnext24_qkn_fixed (normalization_intervention)。
完整环境与命令：冻结manifest jobs[0]及jobs文件对应claim_key=9d84883e91d2。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R02 · v27_17m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=a100_t1；E3；U=339996672；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r6k_dn20_tls1_s45_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[1]及jobs文件对应claim_key=d127d540083a。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R03 · v27_17m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=a100_t1；E3；U=339996672；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s45_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[2]及jobs文件对应claim_key=ac1f2482f7f1。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R04 · v27_17m_exact_r25k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=a100_t1；E3；U=339996672；R/阶=25k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r25k_dn20_tls1_s45_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[3]及jobs文件对应claim_key=19859434af7d。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r25k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R05 · v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1；E3；U=339996672；R/阶=136k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_capnext24_qkn_fixed (normalization_intervention)。
完整环境与命令：冻结manifest jobs[4]及jobs文件对应claim_key=01a76f1090b0。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R06 · v27_17m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1；E3；U=339996672；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r6k_dn20_tls1_s46_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[5]及jobs文件对应claim_key=e4daf384ee4f。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R07 · v27_17m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1；E3；U=339996672；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s46_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[6]及jobs文件对应claim_key=16d48b0a8071。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R08 · v27_17m_exact_r25k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1；E3；U=339996672；R/阶=25k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1:v27_17m_exact_r25k_dn20_tls1_s46_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[7]及jobs文件对应claim_key=922248b06d36。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r25k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R09 · v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=339996672；R/阶=136k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention)。
完整环境与命令：冻结manifest jobs[8]及jobs文件对应claim_key=a77e069e44a8。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R10 · v27_17m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=339996672；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1_4:v27_17m_exact_r6k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[9]及jobs文件对应claim_key=8b1ba55c7f54。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R11 · v27_17m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=339996672；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1_4:v27_17m_exact_r12k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[10]及jobs文件对应claim_key=6e4a434e7484。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R12 · v27_17m_exact_r25k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=339996672；R/阶=25k（nogram无表）；TLS=1；qkn=False。
问题：17M matched non-QKN replication; only qkn toggled relative to existing arm。
对照：a100_t1_4:v27_17m_exact_r25k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_nightfill24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[11]及jobs文件对应claim_key=1fb6e537090b。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r25k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数6918及finite；失败保留。


### R13 · v27_35m_exact_r6k_dn20_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[12]及jobs文件对应claim_key=04cac7a2a56e。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R14 · v27_35m_exact_r12k_dn20_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[13]及jobs文件对应claim_key=371ef1ee0ec4。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R15 · v27_35m_exact_r25k_dn20_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[14]及jobs文件对应claim_key=761e09fe8d78。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R16 · v27_35m_exact_r6k_dn20_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[15]及jobs文件对应claim_key=aaa8be53d3b9。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R17 · v27_35m_exact_r12k_dn20_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[16]及jobs文件对应claim_key=4e69af31bc32。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R18 · v27_35m_exact_r25k_dn20_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[17]及jobs文件对应claim_key=dcc5e0ab5b37。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R19 · v27_35m_exact_r6k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[18]及jobs文件对应claim_key=b32b66f9e02a。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R20 · v27_35m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[19]及jobs文件对应claim_key=576447043dfe。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R21 · v27_35m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=1；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[20]及jobs文件对应claim_key=af8a35a5ebdd。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R22 · v27_35m_exact_r6k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[21]及jobs文件对应claim_key=1931cabeb872。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R23 · v27_35m_exact_r12k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[22]及jobs文件对应claim_key=7063e04a0031。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R24 · v27_35m_exact_r25k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s45_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[23]及jobs文件对应claim_key=9c2e3da7657a。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls0.25_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R25 · v27_35m_exact_r6k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[24]及jobs文件对应claim_key=739a33d5f095。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R26 · v27_35m_exact_r12k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[25]及jobs文件对应claim_key=9cc245d5af9b。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R27 · v27_35m_exact_r25k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s46_e3_cap24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[26]及jobs文件对应claim_key=593eba889870。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls0.25_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R28 · v27_35m_exact_r6k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=6k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r6k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[27]及jobs文件对应claim_key=a544d167ca63。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r6k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R29 · v27_35m_exact_r12k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=12k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r12k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[28]及jobs文件对应claim_key=9d17f137a8b4。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r12k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R30 · v27_35m_exact_r25k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=699973632；R/阶=25k（nogram无表）；TLS=0.25；qkn=True。
问题：35M table-LR confound ablation; only table LR changes。
对照：a100_t1_4:v27_35m_exact_r25k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_lr_intervention); a100_t1_4:v27_35m_nogram_r280k_dn20_tls0.5_s47_e3_capnext24_qkn_fixed (table_effect_same_backbone_recipe)。
完整环境与命令：冻结manifest jobs[29]及jobs文件对应claim_key=edf79c4c13c7。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_35m_exact_r25k_dn20_tls0.25_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数14241及finite；失败保留。


### R31 · v27_17m_nogram_r136k_dn10_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1；E3；U=169979904；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[30]及jobs文件对应claim_key=f6848d8ec1dd。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn10_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R32 · v27_17m_exact_r12k_dn10_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1；E3；U=169979904；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s45_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1:v27_17m_nogram_r136k_dn10_tls1_s45_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[31]及jobs文件对应claim_key=96370bb3532f。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn10_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R33 · v27_17m_nogram_r136k_dn40_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1；E3；U=679993344；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s45_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[32]及jobs文件对应claim_key=2caac2e71584。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn40_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R34 · v27_17m_exact_r12k_dn40_tls1_s45_e3_nightfill24_qkn_fixed

Owner=root；seed=45；host=a100_t1；E3；U=679993344；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s45_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1:v27_17m_nogram_r136k_dn40_tls1_s45_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[33]及jobs文件对应claim_key=19cf6b39360a。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn40_tls1_s45_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R35 · v27_17m_nogram_r136k_dn10_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1；E3；U=169979904；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[34]及jobs文件对应claim_key=99be9ab82bc3。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn10_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R36 · v27_17m_exact_r12k_dn10_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1；E3；U=169979904；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s46_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1:v27_17m_nogram_r136k_dn10_tls1_s46_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[35]及jobs文件对应claim_key=3d66160982d9。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn10_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R37 · v27_17m_nogram_r136k_dn40_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1；E3；U=679993344；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_nogram_r136k_dn20_tls1_s46_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[36]及jobs文件对应claim_key=e414a9a20e06。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn40_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R38 · v27_17m_exact_r12k_dn40_tls1_s46_e3_nightfill24_qkn_fixed

Owner=root；seed=46；host=a100_t1；E3；U=679993344；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1:v27_17m_exact_r12k_dn20_tls1_s46_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1:v27_17m_nogram_r136k_dn40_tls1_s46_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[37]及jobs文件对应claim_key=128db1c2d965。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn40_tls1_s46_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R39 · v27_17m_nogram_r136k_dn10_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=169979904；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[38]及jobs文件对应claim_key=19a41af9acfd。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn10_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R40 · v27_17m_exact_r12k_dn10_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=169979904；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1_4:v27_17m_exact_r12k_dn20_tls1_s47_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1_4:v27_17m_nogram_r136k_dn10_tls1_s47_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[39]及jobs文件对应claim_key=3c4454338648。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn10_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数3459及finite；失败保留。


### R41 · v27_17m_nogram_r136k_dn40_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=679993344；R/阶=136k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1_4:v27_17m_nogram_r136k_dn20_tls1_s47_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20)。
完整环境与命令：冻结manifest jobs[40]及jobs文件对应claim_key=6017c0901298。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_nogram_r136k_dn40_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R42 · v27_17m_exact_r12k_dn40_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_4；E3；U=679993344；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Unique-data U scan at fixed E3 and R12k; total compute changes with U。
对照：a100_t1_4:v27_17m_exact_r12k_dn20_tls1_s47_e3_capnext24_qkn_fixed (data_budget_intervention_vs_dn20); a100_t1_4:v27_17m_nogram_r136k_dn40_tls1_s47_e3_nightfill24_qkn_fixed (table_effect_matched_data_budget)。
完整环境与命令：冻结manifest jobs[41]及jobs文件对应claim_key=601a6932cfc2。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_17m_exact_r12k_dn40_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数13835及finite；失败保留。


### R43 · v27_50m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=perm；E3；U=999972864；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r6k_dn20_tls1_s45_e3_cap24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s45_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[42]及jobs文件对应claim_key=0296c8e21b12。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r6k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R44 · v27_50m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=perm；E3；U=999972864；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r12k_dn20_tls1_s45_e3_cap24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s45_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[43]及jobs文件对应claim_key=9c8f96245405。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r12k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R45 · v27_50m_exact_r50k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=perm；E3；U=999972864；R/阶=50k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r50k_dn20_tls1_s45_e3_cap24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s45_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[44]及jobs文件对应claim_key=febcca15124b。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r50k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R46 · v27_50m_exact_r100k_dn20_tls1_s45_e3_nightfill24_fixed

Owner=root；seed=45；host=perm；E3；U=999972864；R/阶=100k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r100k_dn20_tls1_s45_e3_cap24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s45_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[45]及jobs文件对应claim_key=1499d4088d04。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r100k_dn20_tls1_s45_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R47 · v27_50m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1_2；E3；U=999972864；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：a100_t1_2:v27_50m_exact_r6k_dn20_tls1_s46_e3_cap24_qkn_fixed (normalization_intervention); a100_t1_2:v27_50m_nogram_r400k_dn20_tls1_s46_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[46]及jobs文件对应claim_key=dce66ec0e1c8。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_50m_exact_r6k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R48 · v27_50m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1_2；E3；U=999972864；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：a100_t1_2:v27_50m_exact_r12k_dn20_tls1_s46_e3_cap24_qkn_fixed (normalization_intervention); a100_t1_2:v27_50m_nogram_r400k_dn20_tls1_s46_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[47]及jobs文件对应claim_key=521e3a12421a。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_50m_exact_r12k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R49 · v27_50m_exact_r50k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1_2；E3；U=999972864；R/阶=50k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：a100_t1_2:v27_50m_exact_r50k_dn20_tls1_s46_e3_cap24_qkn_fixed (normalization_intervention); a100_t1_2:v27_50m_nogram_r400k_dn20_tls1_s46_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[48]及jobs文件对应claim_key=432f46f0c460。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_50m_exact_r50k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R50 · v27_50m_exact_r100k_dn20_tls1_s46_e3_nightfill24_fixed

Owner=root；seed=46；host=a100_t1_2；E3；U=999972864；R/阶=100k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：a100_t1_2:v27_50m_exact_r100k_dn20_tls1_s46_e3_cap24_qkn_fixed (normalization_intervention); a100_t1_2:v27_50m_nogram_r400k_dn20_tls1_s46_e3_cap24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[49]及jobs文件对应claim_key=c7d4b939cf82。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_50m_exact_r100k_dn20_tls1_s46_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R51 · v27_50m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=perm；E3；U=999972864；R/阶=6k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r6k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s47_e3_capnext24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[50]及jobs文件对应claim_key=cd5268b9b033。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r6k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R52 · v27_50m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=perm；E3；U=999972864；R/阶=12k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r12k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s47_e3_capnext24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[51]及jobs文件对应claim_key=db09f14a9ccb。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r12k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R53 · v27_50m_exact_r50k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=perm；E3；U=999972864；R/阶=50k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r50k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s47_e3_capnext24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[52]及jobs文件对应claim_key=1639a7ca0f4d。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r50k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R54 · v27_50m_exact_r100k_dn20_tls1_s47_e3_nightfill24_fixed

Owner=root；seed=47；host=perm；E3；U=999972864；R/阶=100k（nogram无表）；TLS=1；qkn=False。
问题：50M normalization x capacity; complete existing 3-seed grid。
对照：perm:v27_50m_exact_r100k_dn20_tls1_s47_e3_capnext24_qkn_fixed (normalization_intervention); perm:v27_50m_nogram_r400k_dn20_tls1_s47_e3_capnext24_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[53]及jobs文件对应claim_key=ae1254c42242。
产物：/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed/v27_50m_exact_r100k_dn20_tls1_s47_e3_nightfill24_fixed/；验收summary、config、训练/验证全曲线、SHA、步数20345及finite；失败保留。


### R55 · v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_2；E3；U=2079977472；R/阶=832k（nogram无表）；TLS=1；qkn=True。
问题：Third seed replication for current 104M scale boundary。
对照：该组新无表对照。
完整环境与命令：冻结manifest jobs[54]及jobs文件对应claim_key=325532048883。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数42318及finite；失败保留。


### R56 · v27_104m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_2；E3；U=2079977472；R/阶=12k（nogram无表）；TLS=1；qkn=True。
问题：Third seed replication for current 104M scale boundary。
对照：a100_t1_2:v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[55]及jobs文件对应claim_key=25c7f6378f60。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_104m_exact_r12k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数42318及finite；失败保留。


### R57 · v27_104m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_2；E3；U=2079977472；R/阶=25k（nogram无表）；TLS=1；qkn=True。
问题：Third seed replication for current 104M scale boundary。
对照：a100_t1_2:v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[56]及jobs文件对应claim_key=4fa737a65834。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_104m_exact_r25k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数42318及finite；失败保留。


### R58 · v27_104m_exact_r50k_dn20_tls1_s47_e3_nightfill24_qkn_fixed

Owner=root；seed=47；host=a100_t1_2；E3；U=2079977472；R/阶=50k（nogram无表）；TLS=1；qkn=True。
问题：Third seed replication for current 104M scale boundary。
对照：a100_t1_2:v27_104m_nogram_r832k_dn20_tls1_s47_e3_nightfill24_qkn_fixed (table_effect_same_normalization)。
完整环境与命令：冻结manifest jobs[57]及jobs文件对应claim_key=510099fdb69e。
产物：/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed/v27_104m_exact_r50k_dn20_tls1_s47_e3_nightfill24_qkn_fixed/；验收summary、config、训练/验证全曲线、SHA、步数42318及finite；失败保留。
