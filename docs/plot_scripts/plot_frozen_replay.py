#!/usr/bin/env python3
"""Frozen-from-start vs no-gram: raw curves, fixed gaps and initial-adjusted excess."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from v5_style import apply_style

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'docs/appendices/frozen_replay_20260925'
OUT=ROOT/'docs/figs/frozen_replay_20260925'
COLOR={'frozen0':'#2d6f9f','nogram':'#686d73'}
LABEL={'frozen0':'Table frozen from update 1','nogram':'Pretrained no-gram'}

def load(partial):
    for r in json.loads((PACK/'source_manifest.json').read_text()):
        assert hashlib.sha256((PACK/'evidence'/r['path']).read_bytes()).hexdigest()==r['sha256']
    data={};ends=[];curves=[]
    for r in json.loads((PACK/'registered_runs.json').read_text()):
        rid=r['run_id'];p=PACK/'evidence/data/runs_fixed'/rid
        status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else dict(qc='running')
        done=status['qc']=='pass'
        if not partial:assert done,rid
        if not (p/'train_log.jsonl').exists():continue
        online=[json.loads(s) for s in (p/'train_log.jsonl').read_text().splitlines()]
        fixed=[json.loads(s) for s in (p/'fixed_train_loss.jsonl').read_text().splitlines()]
        if not online or not fixed:continue
        initial=json.loads((p/'initial_probe.json').read_text());fixed=[initial]+fixed
        u=r['pool_updates'];key=(u,r['rate'],r['arm'])
        for o in online:
            assert o['epoch']==(o['step']-1)//u+1
            assert abs(o['gap']-o['val_loss']+o['train_loss'])<1e-6
        for f in fixed:assert abs(f['fixed_gap']-f['fixed_val_loss']+f['fixed_train_loss'])<1e-6
        assert all(math.isfinite(x[k]) for array,keys in [(online,['train_loss','val_loss','gap']),
                   (fixed,['fixed_train_loss','fixed_val_loss','fixed_gap'])] for x in array for k in keys)
        if done:
            s=json.loads((p/'summary.json').read_text());c=s['config']
            assert s['steps']==1680 and s['seed']==42
            for k,v in dict(device_batch_size=18,total_batch_size=147456,epoch_batches=u*4,
                nanogpt_adam_lr=.0006 if r['rate']=='base' else .00015,lr_schedule='warmup_constant',
                warmup_steps=100,val_batches=16,dropout=0).items():assert c[k]==v,(rid,k)
            assert s['freeze_table']==(r['arm']=='frozen0') and not s['disable_ngram_tables']
            assert s['fixed_train_probe_sha256']==initial['probe_sha256']
            expected=sorted(set(range(10,1681,10))|{1680}|{v for b in range(u,1680,u) for v in [b,b+1]})
            assert [x['step'] for x in online]==expected and [x['step'] for x in fixed[1:]]==expected
            if r['arm']=='frozen0':
                a=json.loads((p/'freeze_audit.json').read_text())
                assert a['table_bitwise_equal'] and a['backbone_changed']
            sizes=json.loads((PACK/'remote_artifact_sizes.json').read_text())[rid]
            assert sizes['final_model.pt']>200000000 and sizes['probe_tokens.npz']>0
        data[key]=dict(online=online,fixed=fixed,initial=initial,done=done,run_id=rid)
        end=online[-1];f=fixed[-1]
        ends.append(dict(run_id=rid,pool_updates=u,rate=r['rate'],arm=r['arm'],seed=42,
            status=status['qc'],step=end['step'],passes=end['step']/u,train_loss=end['train_loss'],
            val_loss=end['val_loss'],online_gap=end['gap'],fixed_train_loss=f['fixed_train_loss'],
            fixed_gap=f['fixed_gap'],initial_fixed_gap=initial['fixed_gap'],
            fixed_growth=f['fixed_gap']-initial['fixed_gap']))
        for array,metrics in [(online,['train_loss','val_loss','gap']),
                              (fixed,['fixed_train_loss','fixed_val_loss','fixed_gap'])]:
            for x in array:
                for metric in metrics:curves.append(dict(run_id=rid,step=x['step'],metric=metric,value=x[metric]))
    comparisons=[]
    for u in [42,84]:
        for rate in ['base','q4']:
            if (u,rate,'frozen0') not in data or (u,rate,'nogram') not in data:continue
            a,b=data[u,rate,'frozen0'],data[u,rate,'nogram']
            assert a['initial']['probe_sha256']==b['initial']['probe_sha256']
            fo={x['step']:x for x in a['online']};no={x['step']:x for x in b['online']}
            ff={x['step']:x for x in a['fixed']};nf={x['step']:x for x in b['fixed']}
            common=sorted(fo.keys()&no.keys()&ff.keys()&nf.keys());end=common[-1]
            comparisons.append(dict(pool_updates=u,rate=rate,step=end,passes=end/u,seed=42,
                frozen_online_gap=fo[end]['gap'],nogram_online_gap=no[end]['gap'],
                online_excess=fo[end]['gap']-no[end]['gap'],
                frozen_fixed_gap=ff[end]['fixed_gap'],nogram_fixed_gap=nf[end]['fixed_gap'],
                fixed_excess=ff[end]['fixed_gap']-nf[end]['fixed_gap'],
                initial_excess=ff[0]['fixed_gap']-nf[0]['fixed_gap'],
                excess_growth=(ff[end]['fixed_gap']-ff[0]['fixed_gap'])-(nf[end]['fixed_gap']-nf[0]['fixed_gap'])))
    for name,records in [('endpoints',ends),('comparisons',comparisons),('curves',curves)]:
        if records:
            with (PACK/(name+('_partial' if partial else '')+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator=chr(10));w.writeheader();w.writerows(records)
    return data,comparisons

def save(fig,name):
    for ext in ['png','pdf','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight')
        if ext=='svg':p.write_text(chr(10).join(s.rstrip() for s in p.read_text().splitlines())+chr(10))
    plt.close(fig)

def report(comparisons):
    lines=['# Frozen-table post-training: completed replay-intensity grid','',
        'The table is frozen before downstream update 1; only the backbone learns. '
        'All eight runs use seed 42 and stop at the registered update 1680. '
        'All four architecture comparisons are retained, with no choice of a favorable endpoint.','',
        '| Unique target tokens/pass | LR | Passes | Frozen online gap | No-gram online gap | Online excess | Fixed excess | Initial-adjusted fixed-growth excess |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        assert r['step']==1680
        lines.append(f"| {r['pool_updates']*147456/1e6:.2f}M | {'.0006' if r['rate']=='base' else '.00015'} | {r['passes']:.0f} | {r['frozen_online_gap']:.6f} | {r['nogram_online_gap']:.6f} | {r['online_excess']:+.6f} | {r['fixed_excess']:+.6f} | {r['excess_growth']:+.6f} |")
    positives=sum(r['online_excess']>0 and r['excess_growth']>0 for r in comparisons)
    lines+=['',f'{positives}/4 tested settings have both positive online excess and positive initial-adjusted fixed-gap growth excess. '
            'This count is descriptive and is not a significance test. A large absolute frozen gap alone does not meet the requested excess-over-no-gram criterion.','',
            '## Raw endpoint losses','',
            '| Pool updates/pass | LR | Arm | Online train loss | Fixed validation loss | Fixed train loss | Fixed gap at initialization | Fixed gap at 1680 |',
            '|---:|---|---|---:|---:|---:|---:|---:|']
    for r in csv.DictReader((PACK/'endpoints.csv').open()):
        lines.append('| '+r['pool_updates']+' | '+r['rate']+' | '+r['arm']+' | '+' | '.join(f"{float(r[k]):.6f}" for k in ['train_loss','val_loss','fixed_train_loss','initial_fixed_gap','fixed_gap'])+' |')
    lines+=['','## What this tests','',
        'The 6.19M/12.39M-token pools are nested prefixes of the same-domain shard 21, which was held out from base-model optimization. '
        'Both use the same disjoint 288-row validation prefix of shard 22. The 1680-update budget is 247,726,080 target-token exposures for every run. '
        'Each architecture pair has identical training bytes, stream order, effective batch, LR, warmup and evaluation probes. '
        'The bases are the original input-trigram and separately pretrained no-gram, both trained on a 20× data pool for three passes. '
        'No artificial pretraining/no-replay claim is made.','',
        'Online gap is post-update fixed validation minus pre-update current training batch loss. Fixed gap uses the same parameter state '
        'on uniform 288-row training and validation probes. Initial-adjusted excess is '
        '[G_frozen(1680)−G_frozen(0)]−[G_nogram(1680)−G_nogram(0)]. '
        'The table is never trained on these downstream pools in the frozen arms.','',
        'The experiment is all-token continued LM, not response-only SFT or on-policy RL. '
        'Small-pool replay is a stress test. Results do not establish any universal absence or presence of frozen-table excess, '
        'and they do not isolate the mechanism from differing pretrained backbone quality. '
        'Changing only downstream seed cannot provide independent replication with fixed weights/data/dropout0.','',
        '## Prior freeze evidence and external practice','',
        'The older pretraining runs causalv5c_freeze_table_e1_128x{,_s43,_s44}_fixed freeze at step338 after joint training on the same replay pool. '
        'Their final online gaps at1000 are3.452124/3.425178/3.427931. All three summaries have fixed_train_probe_batches=0 and missing/NaN final_fixed_gap. '
        'Their small raw logs and hashes are archived under prior_pretraining/. They cannot supply a fixed-gap growth trajectory or be silently relabeled as post-training freeze-from-start.','',
        'Official Engram README and paper checked on2026-09-25: [repository](https://github.com/deepseek-ai/Engram), '
        '[paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf). '
        'The paper explicitly updates embedding parameters with Adam at5× LR in pretraining. Its static-memory terminology describes lookup architecture; '
        'it does not itself mean frozen parameters. This check did not establish a DeepSeek production post-training freeze policy. '
        'Freezing a feature encoder, freezing a memory table, and freezing the backbone for parameter-efficient fine-tuning are different protocols.','',
        '## Artifacts and acceptance','',
        'Code snapshot b744c0b; source/data hashes in code_hashes.json and data_manifest.json. '
        'Both full-model two-update smoke arms passed before formal launches. All final checkpoints and probe files remain remote. '
        'Each frozen final table must be bitwise identical to its initial table with a changed backbone; acceptance checks also cover config, '
        'cadence, epochs, finite arithmetic, probe identity and saved artifacts. No table norm proxy replaces the bitwise check.','',
        '[Full curves](../../figs/frozen_replay_20260925/fig_frozen_replay.png), '
        '[endpoint data](endpoints.csv), [paired contrasts](comparisons.csv), [raw curve data](curves.csv). '
        'PNG/PDF/SVG generated by docs/plot_scripts/plot_frozen_replay.py; small logs collected by collect_frozen_replay.py. '
        'Sources are SHA256-verified remote data/runs_fixed artifacts. The local external-SSD data symlink may be unavailable; '
        'all plot inputs are archived inside this packet.','']
    (PACK/'results.md').write_text(chr(10).join(lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    data,comparisons=load(a.partial);apply_style();OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,4,figsize=(14.3,11.8),sharex=True)
    fig.subplots_adjust(left=.07,right=.99,bottom=.10,top=.87,hspace=.27,wspace=.30)
    fig.suptitle('Frozen-table post-training: matched no-gram controls'+(' | RUNNING' if a.partial else ' | final'),y=.98,fontsize=14)
    fig.legend(handles=[Line2D([],[],color=COLOR[k],label=LABEL[k]) for k in COLOR]+
               [Line2D([],[],color='#333',label='Online train'),Line2D([],[],color='#333',ls='--',label='Fixed validation')],
               loc='upper center',bbox_to_anchor=(.5,.948),ncol=4)
    for j,(u,rate) in enumerate([(42,'base'),(42,'q4'),(84,'base'),(84,'q4')]):
        axes[0,j].set_title(f'{u*147456/1e6:.2f}M tokens/pass; LR '+('6e−4' if rate=='base' else '1.5e−4'),fontsize=10)
        for arm in ['frozen0','nogram']:
            if (u,rate,arm) not in data:continue
            d=data[u,rate,arm];o=d['online'];f=d['fixed'];x=[r['step'] for r in o];xf=[r['step'] for r in f]
            axes[0,j].plot(x,[r['train_loss'] for r in o],color=COLOR[arm],lw=1)
            axes[0,j].plot(x,[r['val_loss'] for r in o],color=COLOR[arm],lw=1.2,ls='--')
            axes[1,j].plot(x,[r['gap'] for r in o],color=COLOR[arm],lw=1)
            axes[2,j].plot(xf,[r['fixed_gap'] for r in f],color=COLOR[arm],lw=1.2)
        if (u,rate,'frozen0') in data and (u,rate,'nogram') in data:
            f={r['step']:r['fixed_gap'] for r in data[u,rate,'frozen0']['fixed']}
            n={r['step']:r['fixed_gap'] for r in data[u,rate,'nogram']['fixed']}
            x=sorted(f.keys()&n.keys());y=[(f[t]-n[t])-(f[0]-n[0]) for t in x]
            axes[3,j].plot(x,y,color=COLOR['frozen0'],lw=1.4)
            axes[3,j].axhline(0,color='#777',lw=.7)
        for i in range(4):
            axes[i,j].grid(True,axis='y');axes[i,j].grid(False,axis='x')
            axes[i,j].set_xlim(0,1680);axes[i,j].set_xticks([0,420,840,1260,1680])
        axes[3,j].set_xlabel('Optimizer update')
    for i,label in enumerate(['Loss (nats/token)','Online gap','Uniform fixed gap','Excess fixed-gap growth']):axes[i,0].set_ylabel(label)
    fig.text(.5,.024,'Table frozen before update 1; backbone trainable. Same pretrained bases as §69; same-domain pool never optimized in pretraining.'+chr(10)+
        '1680 updates: 40 passes over 6.19M tokens or 20 passes over 12.39M. Seed 42; 288-row fixed probes; raw curves.'+chr(10)+
        'Bottom: frozen-minus-no-gram fixed gap, minus initial difference. One pretrained/data realization; exploratory grid.',ha='center',fontsize=9)
    save(fig,'fig_frozen_replay'+('_partial' if a.partial else ''))
    if not a.partial:report(comparisons)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
