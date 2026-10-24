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
review_path = sorted((EVIDENCE/'overnight_results/snapshots').glob('*/review.json'))[-1]
review = json.loads(review_path.read_text())
chosen = [c for c in review['comparisons'] if c['state']=='validated_numeric_comparison' and c.get('comparison')=='add_table_exact_minus_nogram' and (('_35m_' in c['treatment'] and '_dn80_' in c['treatment'] and '_qkn_' in c['treatment']) or '_e6_qkn_' in c['treatment'])]
fig2, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
labels = ['50M · 三遍 · seed 42']
deltas = [q['delta_mnat']]
for c in chosen:
    labels.append('50M · 六遍 · seed 42' if '_e6_' in c['treatment'] else ('35M · 长训练 · seed 43' if '_s43_' in c['treatment'] else '35M · 长训练 · seed 44'))
    deltas.append(c['mean_last_60']['delta_val']*1000)
ax.barh(labels, deltas, color=[C['orange'] if x>0 else C['blue'] for x in deltas])
ax.axvline(0,color=C['blue'],linewidth=.8)
ax.set_xlabel('加表 − 无表：末 60 个共同验证点（mnat/token，负值更好）')
ax.invert_yaxis()
for i,v in enumerate(deltas): ax.text(v+(2 if v>=0 else -2),i,f'{v:+.1f}',va='center',ha='left' if v>=0 else 'right')
ax.set_xlim(min(deltas)-24,max(deltas)+24)
save(fig2,str(OUT/'budget_comparison'))
newsvg='\n'.join(line.rstrip() for line in (OUT/'budget_comparison.svg').read_text().splitlines())+'\n'
(OUT/'budget_comparison.svg').write_text(newsvg)
newsvg=newsvg[newsvg.index('<svg'):].replace('<svg ', '<svg class="chart" role="img" aria-label="完整预算下的加表验证收益" ',1)
finished=sum(j['state']=='finished_unreviewed' for h in status['hosts'] for j in h.get('jobs',[]))
body = f'''<h1>发散有了可用修复，长期收益仍未解决</h1>
<p class="lede">原先的困难只解决了一部分：归一化支持稳定训练和多组收益，但同一设定训练六遍后，加表从受益转为落后，不能升级为通用默认方案。</p>
<ul class="settings"><li>模型 <b>17M / 35M / 50M 档位</b></li><li>比较 <b>同机、同种子、完整预算</b></li><li>指标 <b>末 60 个共同验证点</b></li></ul>
<h2>稳定跑完不等于长期保住收益</h2>
<p>qk-norm（在注意力计算前归一化查询与键）仍是优先修复候选。35M 长训练两颗新种子完整结束，加表仍更好。但 50M 同种子从三遍训练的约 −36.0，变成六遍训练的 +72.9 mnat/token（每 token 的千分之一 nat）。六遍加表训练 loss 更低、验证 loss 更高，符合过拟合表现，不能用“未发散”替代泛化收益。</p>
<figure>{newsvg}</figure>
<p>三遍与六遍是各自独立的预算实验，不能解释为同一条轨迹在中途切换。两者同一颗种子，不能代替跨种子结论。</p>
<h2>小表必然有害的判断已被反例推翻</h2>
<p>17M 小表在归一化后，seed 43 的完整配对改善 24.7 mnat/token。它反驳小表必然有害，但原有害区是否由优化失稳造成仍需进一步拆分，尚不能证明所有小表均有益。50M 无表基线在 seed 44 的归一化干预对照改善 26.8 mnat/token，仅支持这一设定没有受损。</p>
<h2>降低学习率不能作为统一救援方案</h2>
<p>35M 长训练中，同时降低主干和表学习率的加表臂末 60 点平均验证 loss 为 6.648，最终为 6.594。只降低主干、保持表学习率的配对则健康完成并保留收益。原先“主干学习率降低就解决”的说法不成立，主干与表的影响还需完整拆开。组合优化器修复仍有无表臂失稳，不能作为默认方案。</p>
<h2>先前悬而未决的 50M 长训练已完成</h2>
<p>50M 较大数据量、seed 42 的旧长训练现已核对完整 56,966 步：归一化配对的末 60 点验证 loss 为 2.525462 / 2.584698，加表改善 59.2 mnat/token。组合优化器修复（此前称 v28）的同机配对为 2.474061 / 2.562518，改善 88.5 mnat/token，两臂保持稳定。这两组旧预算的完成与收益问题已经解决。</p>
<p>两种修复运行在不同机器，不能用上述数字判定谁更优。组合优化器修复在其他设定仍有失败，不能由这次成功推广为通用方案。35M 旧种子的两臂跨机，保留描述性结果，不纳入严格同机收益。</p>
<h2>机制解释和剩余新长训练仍未完成</h2>
<p>注意力分数失控先于 loss 恶化有观察支持，但第一层状态如何导致失控仍未建立因果证据。50M 新种子的更大数据量配对尚未全部收齐。另有一对每遍数据池不一致，仅保留探索用途，不报告严格收益。</p>
<details><summary>保留此前已经核对的完成配对</summary><figure>{svg}</figure><p>50M 与 104M 是内部档位，并非含表总参数。这两组历史结果不替代新增六遍结果，也不证明所有规模和预算均受益。</p></details>
<h2>夜间接续已完成，继续运行不代表继续领取</h2>
<p>08:00 已停止领取新任务，夜间监控已暂停。最新检查有 {active} 个登记实验仍在训练，{queued} 个尚未领取，GPU 占用 {occupied}/32（包含旧训练）。已有训练按原预算收尾，未新增任务或延长预算。后续接管按手册检查状态、收割日志、验证配对。</p>
<details><summary>证据和接管边界</summary><p>收割校验通过仅表示文件与完成条件可验证，不表示科学成功。收益只使用完整同机配对的末 60 个共同验证点，不把这些时间点当独立种子。原始小日志和校验和保存在 outputs，逐臂设置与结果登记见实验日志第 151 节。当前结果来源：{review_path.parent.name}。</p></details>
<footer>来源：实验登记、训练日志与完成文件；状态核验 {observed} 北京时间。</footer>
'''
template=(SKILL/'assets/panel-template.html').read_text()
start=template.index('<main>')
end=template.index('</main>',start)+len('</main>')
html=template[:start]+'<main>\n'+body+'\n</main>'+template[end:]
html=html.replace('结论式标题：用平实的话写结论','A100 夜间实验与接管简报')
(ROOT/'docs/handoff/takeover-brief-20261024.html').write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
