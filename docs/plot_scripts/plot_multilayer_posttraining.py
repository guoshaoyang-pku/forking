#!/usr/bin/env python3
"""Raw multilayer post-training curves; refuse final figures before all eight QC pass."""
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
PACK=ROOT/'docs/appendices/multilayer_posttraining_20260925'
OUT=ROOT/'docs/figs/multilayer_posttraining_20260925'
ARMS=['tableonly','module','trainable','nogram']
COLORS=dict(tableonly='#2d6f9f',module='#56a89a',trainable='#bf5b47',nogram='#686d73')
LABELS=dict(tableonly='Lookup frozen; gates trained',module='Lookup and gates frozen',
            trainable='Memory trainable',nogram='Pretrained no-gram')

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    for item in json.loads((PACK/'source_manifest.json').read_text()):
        assert hashlib.sha256((PACK/'evidence'/item['path']).read_bytes()).hexdigest()==item['sha256']
    data={};endpoints=[];comparisons=[]
    for source in ['id','math']:
        for arm in ARMS:
            rid=f'ptml_{source}_{arm}_L14_q4_s42_fixed'
            path=PACK/'evidence/data/runs_fixed'/rid
            status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {'qc':'pending'}
            if not a.partial:assert status['qc']=='pass',(rid,status)
            if not (path/'fixed_train_loss.jsonl').exists():continue
            online=[json.loads(s) for s in (path/'train_log.jsonl').read_text().splitlines()]
            fixed=[json.loads(s) for s in (path/'fixed_train_loss.jsonl').read_text().splitlines()]
            if not online or not fixed:continue
            initial=json.loads((path/'initial_probe.json').read_text())
            fixed=[initial]+fixed
            for r in online:
                assert abs(r['gap']-r['val_loss']+r['train_loss'])<1e-6
                assert r['epoch']==(r['step']-1)//333+1
            for r in fixed:assert abs(r['fixed_gap']-r['fixed_val_loss']+r['fixed_train_loss'])<1e-6
            assert all(math.isfinite(r[k]) for rows,keys in [(online,['train_loss','val_loss','gap']),
                       (fixed,['fixed_train_loss','fixed_val_loss','fixed_gap'])] for r in rows for k in keys)
            if status['qc']=='pass':
                s=json.loads((path/'summary.json').read_text());c=s['config']
                assert s['steps']==1665 and s['seed']==42
                assert c['epoch_batches']==333 and c['device_batch_size']==72
                assert c['nanogpt_adam_lr']==.00015 and c['table_lr_scale']==128
                assert s['freeze_table']==(arm=='module')
                assert s['freeze_embeddings_only']==(arm=='tableonly')
                assert s['inject_layers']==('' if arm=='nogram' else '1,4')
                assert s['fixed_train_probe_sha256']==initial['probe_sha256']
                if arm in ['tableonly','module']:
                    audit=json.loads((path/'freeze_audit.json').read_text())
                    assert audit['table_bitwise_equal'] and audit['backbone_changed']
            data[source,arm]=dict(online=online,fixed=fixed,initial=initial)
            o=online[-1];f=fixed[-1]
            endpoints.append(dict(run_id=rid,source=source,arm=arm,seed=42,status=status['qc'],step=o['step'],
                                  online_train=o['train_loss'],val=o['val_loss'],online_gap=o['gap'],
                                  fixed_gap=f['fixed_gap'],initial_fixed_gap=initial['fixed_gap']))
    if not data:raise SystemExit('No training curves available; no figure generated.')
    apply_style();OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(4,2,figsize=(10.6,11.0),sharex=True)
    fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.1,hspace=.24,wspace=.25)
    fig.suptitle('Multilayer post-training: frozen lookup vs no-gram'+(' | INCOMPLETE' if a.partial else ''),y=.985)
    handles=[Line2D([],[],color=COLORS[k],label=LABELS[k]) for k in ARMS]
    handles += [Line2D([],[],color='#333',label='Online train'),Line2D([],[],color='#333',ls='--',label='Fixed validation')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=3,fontsize=9)
    for j,source in enumerate(['id','math']):
        ax[0,j].set_title('ID: optimization-unseen shard' if source=='id' else 'Math: verified R1 Algebra')
        for arm in ARMS:
            if (source,arm) not in data:continue
            d=data[source,arm];o=d['online'];f=d['fixed'];x=[r['step'] for r in o];xf=[r['step'] for r in f]
            ax[0,j].plot(x,[r['train_loss'] for r in o],color=COLORS[arm],lw=1)
            ax[0,j].plot(x,[r['val_loss'] for r in o],color=COLORS[arm],ls='--',lw=1.2)
            ax[1,j].plot(x,[r['gap'] for r in o],color=COLORS[arm],lw=1.1)
            ax[2,j].plot(xf,[r['fixed_gap'] for r in f],color=COLORS[arm],lw=1.1)
            if arm!='nogram' and (source,'nogram') in data:
                base=data[source,'nogram']
                g={r['step']:r['fixed_gap'] for r in f};n={r['step']:r['fixed_gap'] for r in base['fixed']}
                xs=sorted(g.keys()&n.keys());ys=[g[t]-g[0]-n[t]+n[0] for t in xs]
                assert d['initial']['probe_sha256']==base['initial']['probe_sha256']
                ax[3,j].plot(xs,ys,color=COLORS[arm],lw=1.2)
                comparisons.append(dict(source=source,arm=arm,step=xs[-1],excess_fixed_growth=ys[-1]))
        for i in range(4):
            for t in [333,666,999,1332]:ax[i,j].axvline(t,color='#bbb',lw=.5,ls=':')
            ax[i,j].set_xlim(0,1665);ax[i,j].set_xticks([0,333,666,999,1332,1665])
        ax[3,j].axhline(0,color='#555',lw=.7);ax[3,j].set_xlabel('Optimizer update')
    for i,label in enumerate(['Loss (nats/token)','Online gap','Uniform fixed gap','Excess fixed-gap growth']):ax[i,0].set_ylabel(label)
    fig.text(.5,.022,'Blocks 1,4 (zero-based); dual clean tables; LR 1.5e−4; 333 updates/pass × 5; seed 42.'+chr(10)+
             'Bottom: fixed-gap growth minus independently pretrained no-gram growth. Raw curves; one base realization.'+chr(10)+
             'Static all-token continued LM; mathematics pool v2; not on-policy RLVR.',ha='center',fontsize=9)
    suffix='_partial' if a.partial else ''
    for ext in ['png','pdf','svg']:
        p=OUT/('fig_multilayer_posttraining'+suffix+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight')
        if ext=='svg':p.write_text(chr(10).join(s.rstrip() for s in p.read_text().splitlines())+chr(10))
    for name,rows in [('endpoints',endpoints),('comparisons',comparisons)]:
        if rows:
            with (PACK/(name+suffix+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
