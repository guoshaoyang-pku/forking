#!/usr/bin/env python3
"""Recompute matched fixed-val windows and render the takeover brief from logs."""
import json
import math
import sys
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
svg=(OUT/'matched_validation.svg').read_text()
svg=svg[svg.index('<svg'):].replace('<svg ', '<svg class="chart" role="img" aria-label="两对已完成训练的末轮固定验证曲线" ', 1)
q, r = metrics
body = f'''
<h1>先收齐长训，再验证注意力归一化</h1>
<p class="lede">查询与键归一化是优先候选，下一批只准备一对长训复本，先通过完整预算验收再扩展。</p>
<ul class="settings">
<li><svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4"/></svg>模型档位 <b>50M / 104M</b></li>
<li><svg viewBox="0 0 24 24"><path d="M4 19l5-7 4 4 7-10"/></svg>训练 <b>固定数据重复三遍</b></li>
<li><svg viewBox="0 0 24 24"><path d="M6 3v18M18 3v18M6 12h12"/></svg>比较 <b>同种子、同步数</b></li>
</ul>
<h2>已完成的配对支持收益，长训结论仍待收齐</h2>
<p>qk-norm（在注意力计算前归一化查询与键）下，50M 档位的加表模型在末轮验证中优于无表基线。新收回的 104M 第三颗种子也保留收益。图中阴影是末 60 个共同验证点，mnat/token 表示每 token 的千分之一 nat。</p>
<figure>{svg}</figure>
<p>50M 长训归一化对目前只对齐到第 26,630 步，目标为 56,966 步。越过旧发散点支持继续验证，不能代替完整预算结果。</p>
<details><summary>查看配对来源与判断边界</summary>
<p>50M / 104M 是内部模型档位，不是含表总参数。两对均已核对完成文件、最终步数、种子、训练与验证分片及有效 batch。50M 的两臂微批量分别为 18 / 36，梯度累积保持相同有效 batch。跨机器文件指纹仍是正式晋升前的检查项。</p>
<p>50M：seed 42，第 {q['window'][0]:,}–{q['window'][1]:,} 步，窗口差 {q['delta_mnat']:.3f}，最终单点差 {q['final_delta_mnat']:.3f} mnat/token。104M：seed 44，第 {r['window'][0]:,}–{r['window'][1]:,} 步，窗口差 {r['delta_mnat']:.3f}，最终单点差 {r['final_delta_mnat']:.3f} mnat/token。</p>
<p>实验标识符：<code>{q['run_ids'][0]}</code>（加表实验）与对应无表臂；<code>{r['run_ids'][0]}</code>（加表实验）与对应无表臂。每对只是一颗种子，时间点不当独立重复。</p>
</details>
<h2>三处口径修正会改变后续实验安排</h2>
<ul>
<li>交接中的长训收益使用了不同步数的末值，必须重算共同窗口。“不发散”也不能推出无表基线性能无损。</li>
<li>现有降学习率实验同时降低主干与表的实际学习率，不能单独归因于主干。学习率减半也不等于训练耗时翻倍。</li>
<li>注意力分数失控先于 loss 恶化有探针支持。“第一层闲置导致失控”仍是待验证机制。组合优化器修复已有无表臂失稳，暂不晋升。</li>
</ul>
<h2>先补一个长训复本，再决定是否扩大</h2>
<table><thead><tr><th>顺序</th><th>执行内容</th><th>放行条件</th></tr></thead><tbody>
<tr><td>先收齐</td><td>现有归一化与降学习率配对，新增训练预算为零</td><td>完整三遍、同步数，排除异常基线</td></tr>
<tr><td>第一批</td><td>50M 长训，seed 43，加表与无表各一臂，共约 16.8B token</td><td>seed 42 长训先完成，核验输入与代码，确认队列无重复</td></tr>
<tr><td>再扩展</td><td>35M 长训另一颗种子的一对复本</td><td>上一批两臂健康，两个种子的验证差均为负</td></tr>
</tbody></table>
<p>具体命令与停止规则已写入接管记录。新训练尚未启动，归一化仍是候选契约。</p>
<details><summary>查看算力与保存阻塞</summary><p>360-2 已恢复读取，但数据盘已满，不具备新任务放行条件。200M 配对的无表臂明显失稳，不能贡献收益。360-1 仍连接超时。ophis 的低利用率卡仍有进程占用，不能当已预约空卡。本地原始数据挂载不可用，新收割文件保存到已忽略的 outputs 目录。</p></details>
<footer>来源：交接记录、原始训练日志及完成文件；2026-10-24。图由接管绘图脚本直接重算。</footer>
'''
template=(SKILL/'assets/panel-template.html').read_text()
start=template.index('<main>')
end=template.index('</main>',start)+len('</main>')
html=template[:start]+'<main>\n'+body+'\n</main>'+template[end:]
html=html.replace('结论式标题：用平实的话写结论','接管简报：先收齐长训，再验证注意力归一化')
(ROOT/'docs/handoff/takeover-brief-20261024.html').write_text(html)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
