#!/usr/bin/env python3
"""Raw continued-LM overlap/freeze curves; initialization and metrics explicit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from v5_style import apply_style
ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'docs/appendices/posttraining_overlap_20260925'
OUT=ROOT/'docs/figs/posttraining_overlap_20260925'
ARMS=[('trainable','Table trainable','#c4493d'),('frozen0','Frozen from start','#c58a0b'),('frozen1','Frozen after pass 1','#2d6f9f'),('nogram','No-gram','#686d73')]
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def load(partial):
    for r in json.loads((PACK/'source_manifest.json').read_text()):
        p=PACK/'runs'/r['run_id']/r['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    data={};ends=[];curves=[]
    for pool in ['unseen','seen']:
        probes=set()
        for arm,_,_ in ARMS:
            rid=f'ptov_{pool}_{arm}_q4_mb18_s42_fixed';p=PACK/'runs'/rid
            if not (p/'train_log.jsonl').exists():assert partial;continue
            online=lines(p/'train_log.jsonl');fixed=lines(p/'fixed_train_loss.jsonl')
            if not online or not fixed:assert partial;continue
            initial=json.loads((p/'initial_probe.json').read_text());probes.add(initial['probe_sha256'])
            done=(p/'status.json').exists() and json.loads((p/'status.json').read_text()).get('qc')=='pass'
            if not partial:assert done,rid
            if done:
                s=json.loads((p/'summary.json').read_text());c=s['config']
                assert s['steps']==1665 and s['seed']==42 and c['device_batch_size']==18
                assert c['total_batch_size']==147456 and c['epoch_batches']==1332
                assert c['nanogpt_adam_lr']==0.00015 and c['table_lr_scale']==128
                assert s['fixed_train_probe_sha256']==initial['probe_sha256']
                assert online[-1]['epoch']==5
                for r in online:assert r['epoch']==(r['step']-1)//333+1
            for r in online:assert abs(r['val_loss']-r['train_loss']-r['gap'])<1e-6
            for r in fixed:assert abs(r['fixed_val_loss']-r['fixed_train_loss']-r['fixed_gap'])<1e-6
            data[pool,arm]={'online':online,'fixed':[initial]+fixed,'done':done}
            final=online[-1];ff=fixed[-1];best=min(online,key=lambda r:r['val_loss'])
            row={'run_id':rid,'pool':pool,'arm':arm,'seed':42,'status':'done' if done else 'running','step':final['step'],'train_loss':final['train_loss'],'val_loss':final['val_loss'],'online_gap':final['gap'],'fixed_train_loss':ff['fixed_train_loss'],'fixed_gap':ff['fixed_gap'],'initial_fixed_gap':initial['fixed_gap'],'fixed_gap_growth':ff['fixed_gap']-initial['fixed_gap'],'best_logged_val':best['val_loss'],'best_val_step':best['step'],'val_above_best':final['val_loss']-best['val_loss'],'probe_sha256':initial['probe_sha256']}
            b=next((r for r in fixed if r['step']==333),None)
            if b:row.update(fixed_gap_at333=b['fixed_gap'],fixed_gap_growth_after333=ff['fixed_gap']-b['fixed_gap'],fixed_train_change_after333=ff['fixed_train_loss']-b['fixed_train_loss'],fixed_val_change_after333=ff['fixed_val_loss']-b['fixed_val_loss'])
            ends.append(row)
            for typ,rows in [('online',online),('fixed',[initial]+fixed)]:
                for r in rows:
                    for k in (['train_loss','val_loss','gap'] if typ=='online' else ['fixed_train_loss','fixed_val_loss','fixed_gap']):
                        curves.append({'run_id':rid,'pool':pool,'arm':arm,'seed':42,'step':r['step'],'metric':k,'value':r[k]})
        assert len(probes)==1,probes
    for name,rows in [('endpoints.csv',ends),('curves.csv',curves)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (PACK/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator=chr(10));w.writeheader();w.writerows(rows)
    return data,ends
def style(ax,gap=False):
    for e in range(1,6):ax.axvline(e,color='#b8bec4',lw=.6,ls=':',zorder=0)
    if gap:ax.axhline(0,color='#999',lw=.6)
    ax.grid(True,axis='y');ax.grid(False,axis='x');ax.set_xlim(0,5);ax.set_xticks(range(6))
def save(fig,name):
    for ext in ['png','pdf','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=200,bbox_inches='tight')
        if ext=='svg':p.write_text(chr(10).join(l.rstrip() for l in p.read_text().splitlines())+chr(10))
    plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    data,ends=load(a.partial);OUT.mkdir(parents=True,exist_ok=True);apply_style()
    handles=[Line2D([],[],color=c,label=label,lw=1.7) for _,label,c in ARMS]
    fig,axs=plt.subplots(3,2,figsize=(10.8,8.6),sharex=True)
    fig.subplots_adjust(left=.085,right=.99,bottom=.12,top=.82,hspace=.25,wspace=.23)
    fig.suptitle('Post-training replay: pretraining overlap and freeze timing'+(' (running)' if a.partial else ''),y=.985,fontsize=13)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.955),ncol=4)
    fig.text(.5,.882,'Top: fixed-train probe (solid), fixed validation (dashed). Two gap definitions stay separate.',ha='center',fontsize=9)
    for j,(pool,title) in enumerate([('unseen','Same-domain, not optimized in pretraining'),('seen','Exact training-prefix overlap with pretraining')]):
        axs[0,j].set_title(title,fontsize=10)
        for arm,_,c in ARMS:
            if (pool,arm) not in data:continue
            d=data[pool,arm];f=d['fixed'];o=d['online'];x=[r['step']/333 for r in f]
            axs[0,j].plot(x,[r['fixed_train_loss'] for r in f],color=c,lw=1.3)
            axs[0,j].plot(x,[r['fixed_val_loss'] for r in f],color=c,lw=1.3,ls='--')
            axs[1,j].plot([r['step']/333 for r in o],[r['gap'] for r in o],color=c,lw=1.2)
            axs[2,j].plot(x,[r['fixed_gap'] for r in f],color=c,lw=1.3)
        for i in range(3):style(axs[i,j],i>0)
        axs[2,j].set_xlabel('Downstream replay pass (333 optimizer updates)')
    axs[0,0].set_ylabel('Same-checkpoint loss (nats/token)')
    axs[1,0].set_ylabel('Online gap (nats/token)');axs[2,0].set_ylabel('Uniform fixed gap (nats/token)')
    footer='Seed 42; input trigram R=2²⁰; LR=1.5e−4; table LR=128×; A100 microbatch 18 × accumulation 4.'+chr(10)+'Fixed probe: 288 rows across each exact pool. After-pass-1 freeze precedes update 334. No smoothing.'
    fig.text(.5,.025,footer,ha='center',fontsize=8.8);save(fig,'fig_overlap_freeze_overview')
    for pool in ['unseen','seen']:
        fig,axs=plt.subplots(2,4,figsize=(12.2,5.9),sharex=True,sharey='row')
        fig.subplots_adjust(left=.065,right=.995,bottom=.18,top=.80,hspace=.28,wspace=.16)
        fig.suptitle(('Same-domain unseen training pool' if pool=='unseen' else 'Pretraining-seen training pool')+(' | running' if a.partial else ' | seed 42'),y=.985,fontsize=13)
        fig.legend(handles=[Line2D([],[],color='#2d6f9f',label='Online train'),Line2D([],[],color='#2d6f9f',ls='--',label='Fixed validation'),Line2D([],[],color='#888',label='Online gap'),Line2D([],[],color='#c4493d',label='Uniform fixed gap')],loc='upper center',bbox_to_anchor=(.5,.94),ncol=4)
        for j,(arm,title,_) in enumerate(ARMS):
            axs[0,j].set_title(title,fontsize=10)
            if (pool,arm) in data:
                d=data[pool,arm];o=d['online'];f=d['fixed'];x=[r['step']/333 for r in o]
                axs[0,j].plot(x,[r['train_loss'] for r in o],color='#2d6f9f',lw=1.2)
                axs[0,j].plot([r['step']/333 for r in f],[r['fixed_val_loss'] for r in f],color='#2d6f9f',ls='--',lw=1.2)
                axs[1,j].plot(x,[r['gap'] for r in o],color='#888',lw=1)
                axs[1,j].plot([r['step']/333 for r in f],[r['fixed_gap'] for r in f],color='#c4493d',lw=1.3)
            for i in range(2):style(axs[i,j],i==1)
            axs[1,j].set_xlabel('Replay pass')
        axs[0,0].set_ylabel('Loss (nats/token)');axs[1,0].set_ylabel('Gap (nats/token)')
        fig.text(.5,.045,'Same pretrained weights for table arms; separately pretrained no-gram. LR=1/4 standard; raw curves.',ha='center',fontsize=9)
        save(fig,'fig_'+pool+'_all_arms')
    print('Wrote',len(ends),'run summaries to',OUT)
if __name__=='__main__':main()
