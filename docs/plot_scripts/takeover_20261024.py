#!/usr/bin/env python3
"""Recompute matched fixed-val windows and render the takeover brief from logs."""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = Path.home() / '.agents/skills/report-to-human'
sys.path.insert(0, str(SKILL / 'scripts'))
from panel_style import apply, C, save
import matplotlib.pyplot as plt

OUT = ROOT / 'docs/figs/takeover_20261024'
EVIDENCE = ROOT / 'outputs/takeover_20261024'
PAIRS = [
    ('50M 档位 · 查询与键归一化 · seed 42',
     EVIDENCE / 'live_a100/a100_perm',
     'v27_50m_{}_r400k_dn20_tls1_s42_e3_qkn_fixed'),
    ('104M 档位 · 原设定 · seed 44',
     EVIDENCE / '360-2',
     'v27_104m_{}_r832k_dn20_tls1_s44_e3_fixed'),
]

def read_vals(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r['step']:r['val_loss'] for r in rows if r.get('val_loss') is not None}

apply()
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), constrained_layout=True)
metrics = []
for ax, (label, base, pattern) in zip(axs, PAIRS):
    dirs = [base / pattern.format(a) for a in ('exact', 'nogram')]
    summaries = [json.loads((d/'summary.json').read_text()) for d in dirs]
    vals = [read_vals(d/'train_log.jsonl') for d in dirs]
    common = sorted(vals[0].keys() & vals[1].keys())
    assert summaries[0]['steps'] == summaries[1]['steps'] == common[-1]
    assert all(s['config']['max_steps'] == common[-1] for s in summaries)
    for key in ('seed', 'train_max_tokens', 'train_epochs'):
        assert summaries[0][key] == summaries[1][key], key
    for key in ('train_shards', 'val_shards', 'total_batch_size', 'data_seed',
                'val_batches', 'val_device_batch_size', 'lr_schedule', 'warmup_steps'):
        assert summaries[0]['config'][key] == summaries[1]['config'][key], key
    assert len(common) >= 60
    window = common[-60:]
    assert all(math.isfinite(v[s]) for v in vals for s in window)
    means = [sum(v[s] for s in window)/len(window) for v in vals]
    delta = (means[0]-means[1])*1000
    final_delta = (vals[0][common[-1]]-vals[1][common[-1]])*1000
    # Show the common last epoch only; the acceptance statistic uses 60 points.
    shown = [s for s in common if s >= common[-1]*2/3]
    for v, color, name in zip(vals, [C['orange'], C['blue']], ['加表', '无表基线']):
        ax.plot(shown, [v[s] for s in shown], color=color, label=name, linewidth=1.3)
    visible = [v[s] for v in vals for s in shown]
    lo, hi = min(visible), max(visible)
    ax.set_ylim(lo - (hi-lo)*.04, hi + (hi-lo)*.45)
    ax.axvspan(window[0], window[-1], color=C['soft'], alpha=.5)
    ax.set_ylabel('固定验证 loss')
    ax.set_xlabel('训练步数')
    ax.text(.02, .93, label, transform=ax.transAxes, va='top', fontsize=10)
    ax.text(.02, .78, f'末 60 个共同点：加表 − 无表 = {delta:.1f} mnat/token',
            transform=ax.transAxes, va='top', fontsize=10)
    ax.legend(loc='upper right', ncol=2)
    metrics.append(dict(label=label, run_ids=[d.name for d in dirs],
                        source=[str(d.relative_to(ROOT)) for d in dirs],
                        steps=common[-1], window=[window[0],window[-1]], n=60,
                        exact=means[0], nogram=means[1], delta_mnat=delta,
                        final_delta_mnat=final_delta))
save(fig, str(OUT/'matched_validation'))
(EVIDENCE/'report_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
svg='\n'.join(line.rstrip() for line in (OUT/'matched_validation.svg').read_text().splitlines())+'\n'
(OUT/'matched_validation.svg').write_text(svg)
svg=svg[svg.index('<svg'):].replace('<svg ', '<svg class="chart" role="img" aria-label="两对已完成训练的末轮固定验证曲线" ', 1)
q, r = metrics
status = json.loads((EVIDENCE/'overnight_status.json').read_text())
active = sum(j['state']=='active' for h in status['hosts'] for j in h.get('jobs', []))
queued = sum(j['state']=='queued' for h in status['hosts'] for j in h.get('jobs', []))
occupied = sum(g['memory_mib']>=1500 for h in status['hosts'] for g in h.get('gpus', []))
observed = datetime.fromtimestamp(status['observed_epoch']).strftime('%Y-%m-%d %H:%M')
headline = 'A100 已排满，夜间按登记队列续接' if occupied==32 else f'A100 已占用 {occupied} / 32 张卡，夜间按登记队列续接'
body = f'''
<h1>{headline}</h1>
<p class="lede">四台机器已占用 {occupied} 张卡，{active} 个新实验正在训练，{queued} 个登记实验等待释放容量；早晨 08:00 停止新领取，已启动实验跑完既定预算。</p>
<ul class="settings">
<li><svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4"/></svg>算力 <b>32 × A100</b></li>
<li><svg viewBox="0 0 24 24"><path d="M4 19l5-7 4 4 7-10"/></svg>训练 <b>固定数据三遍 / 六遍</b></li>
<li><svg viewBox="0 0 24 24"><path d="M6 3v18M18 3v18M6 12h12"/></svg>领取截止 <b>北京时间 08:00</b></li>
</ul>
<h2>今晚的实验分别检验稳定性、收益和学习率原因</h2>
<p>qk-norm（在注意力计算前归一化查询与键）仍是优先候选，是否改为默认契约，等待完整预算与跨种子结果。</p>
<table><thead><tr><th>实验</th><th>回答的问题</th><th>判定依据</th></tr></thead><tbody>
<tr><td>长训练增加种子</td><td>归一化的稳定性和加表收益能否复现？</td><td>两臂完整结束，同机、同种子比较</td></tr>
<tr><td>固定数据重复六遍</td><td>更晚是否发散，收益是否反转？</td><td>完整六遍结果，与三遍预算分开解释</td></tr>
<tr><td>拆开主干与表的学习率</td><td>哪一组学习率真正影响失稳？</td><td>补齐两组学习率的组合，使用同机控制</td></tr>
<tr><td>小表与原本健康的基线</td><td>稳定后小表是否仍有害，归一化是否伤基线？</td><td>检查绝对 loss，不用“未发散”代替性能</td></tr>
</tbody></table>
<details><summary>查看队列预算与接续边界</summary>
<p>共登记 43 个候选，完整跑完的上限是 212.46B token，估计约 259 A100 GPU 小时。这个数字不是已消耗量，也不承诺早晨全部完成。旧训练自然结束后，调度器领取本机下一个任务。已失败实验保留，不自动重跑。未领取任务在截止后保持待定。</p>
<p>四台机器的训练前缀、验证分片、频次索引、映射与训练代码已逐文件核对一致。六遍预算不当独立种子，历史任务加载了什么仍须依据各自记录。完整逐臂设置见实验登记第 151 节。</p>
<p>一对已启动的中间数据量后备实验，因微批量取整导致每遍数据池相差约 0.00184%，仅保留作稳定性与探索证据。尚未启动的同类配对已统一微批量，严格结果收割不会自动豁免这个差异。</p>
</details>
<h2>已完成的配对支持收益，长训结论仍待收齐</h2>
<p>查询与键归一化下，50M 档位的加表模型在末轮验证中优于无表基线。新收回的 104M 第三颗种子也保留收益。图中阴影是末 60 个共同验证点，mnat/token 表示每 token 的千分之一 nat。</p>
<figure>{svg}</figure>
<p>图中只有已经完成的配对。长训曾越过旧发散点，但这不能代替完整预算结果，也不能证明以后不会发散。</p>
<details><summary>查看配对来源与判断边界</summary>
<p>50M / 104M 是内部模型档位，不是含表总参数。两对均已核对完成文件、最终步数、种子、训练与验证分片及有效 batch。50M 的两臂微批量分别为 18 / 36，梯度累积保持相同有效 batch。今晚的四机指纹核验不能反向证明所有历史进程加载版本相同。</p>
<p>50M：seed 42，第 {q['window'][0]:,}–{q['window'][1]:,} 步，窗口差 {q['delta_mnat']:.3f}，最终单点差 {q['final_delta_mnat']:.3f} mnat/token。104M：seed 44，第 {r['window'][0]:,}–{r['window'][1]:,} 步，窗口差 {r['delta_mnat']:.3f}，最终单点差 {r['final_delta_mnat']:.3f} mnat/token。</p>
<p>实验标识符：<code>{q['run_ids'][0]}</code>（加表实验）与对应无表臂；<code>{r['run_ids'][0]}</code>（加表实验）与对应无表臂。每对只是一颗种子，时间点不当独立重复。</p>
</details>
<h2>三处口径修正会改变后续实验安排</h2>
<ul>
<li>交接中的长训收益使用了不同步数的末值，必须重算共同窗口。“不发散”也不能推出无表基线性能无损。</li>
<li>现有降学习率实验同时降低主干与表的实际学习率，不能单独归因于主干。学习率减半也不等于训练耗时翻倍。</li>
<li>注意力分数失控先于 loss 恶化有探针支持。“第一层闲置导致失控”仍是待验证机制。组合优化器修复已有无表臂失稳，暂不晋升。</li>
</ul>
<h2>接管步骤已经固定，调度器恢复通过实测</h2>
<p>后续模型先执行状态检查，再按手册的固定条件处理异常。调度器意外退出会自动恢复。一次实测保留了原训练进程，日志继续推进，没有重复领取。每 20 分钟的聊天巡检已登记，到 08:20 结束；远端接续不依赖聊天持续在线。</p>
<details><summary>查看交接动作与限制</summary>
<p>状态检查、结果收割、允许恢复的四台主机和停止条件都写在夜间接管手册。完成必须核对最终步数与有限 loss，收益必须用同机配对的末 60 个共同验证点。磁盘不足、持续无进展或科学失败需要留证，不改配置掩盖失败。</p>
<p>恢复测试只证明这次恢复成功，不等于整夜已经完成。聊天巡检已创建，尚未验证首次自动唤醒。主机重启、外部占卡和网络中断仍需按手册处理。</p>
</details>
<details><summary>查看算力与保存阻塞</summary><p>360-2 已恢复读取，但数据盘已满，不具备新任务放行条件。200M 配对的无表臂明显失稳，不能贡献收益。360-1 仍连接超时。ophis 的低利用率卡仍有进程占用，不能当已预约空卡。本地原始数据挂载不可用，新收割文件保存到已忽略的 outputs 目录。</p></details>
<footer>来源：实验登记、原始训练日志及完成文件；算力核验 {observed} 北京时间。图由接管绘图脚本直接重算。</footer>
'''
template=(SKILL/'assets/panel-template.html').read_text()
start=template.index('<main>')
end=template.index('</main>',start)+len('</main>')
html=template[:start]+'<main>\n'+body+'\n</main>'+template[end:]
html=html.replace('结论式标题：用平实的话写结论','A100 夜间实验与接管简报')
(ROOT/'docs/handoff/takeover-brief-20261024.html').write_text(html)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
