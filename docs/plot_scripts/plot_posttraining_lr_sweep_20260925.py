#!/usr/bin/env python3
"""Matched-seed LR curves; online and fixed-probe gaps stay distinct."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v5_style import apply_style

ROOT=Path(__file__).resolve().parents[2]
PACKET=ROOT/'docs/appendices/posttraining_lr_sweep_20260925'
OLD=ROOT/'docs/appendices/posttraining_fixtok_20260925'
OUT=ROOT/'docs/figs/posttraining_lr_sweep_20260925'
SOURCES=[('id','ID'),('magicoder','Magicoder'),('openr1','OpenR1'),('rlvr','RLVR-MATH (static)')]
ARMS=[('trainable','Table trainable'),('frozen','Table frozen'),('nogram','No-gram')]
LRS=[('base',0.0006,'#686d73','6e-4 (original)'),('q4',0.00015,'#2d6f9f','1.5e-4 (1/4)'),('q10',0.00006,'#c58a0b','6e-5 (1/10)')]
STEPS=list(range(10,1661,10))+[1665]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load(partial):
    data={};end=[];curves=[];hashes={}
    oldmanifest={r['run_id']:r for r in json.loads((OLD/'source_manifest.json').read_text())}
    manifest={r['run_id']:r for r in json.loads((PACKET/'artifact_manifest.json').read_text())}
    baseline_path=PACKET/'baseline_uniform4.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else {'runs':[]}
    for source,_ in SOURCES:
        legacy={'openr1':'openr1_10shards','rlvr':'rlvr_math'}.get(source,source)
        for arm,_ in ARMS:
            for tag,lr,_,_ in LRS:
                if tag=='base':
                    rid=('sft_id_e5_nogram_s42' if arm=='nogram' else f'sft_id_tri_e5_freq10_{arm}_s42') if source=='id' else f'sft_{legacy}_fixtok_e5_{arm}_s42'
                    path=OLD/'runs'/rid
                    for fn,sha in oldmanifest[rid]['sha256'].items():
                        assert hashlib.sha256((path/fn).read_bytes()).hexdigest()==sha
                else:
                    rid=f'ptlr_{source}_{arm}_{tag}_s42_fixed';path=PACKET/'raw'/rid
                    if not (path/'summary.json').exists():
                        if partial:continue
                        raise RuntimeError('Missing '+rid)
                    assert json.loads((path/'status.json').read_text())['qc']=='pass',rid
                    for fn,meta in manifest[rid]['files'].items():
                        if fn not in ['final_model.pt','probe_tokens.npz']:
                            assert hashlib.sha256((path/fn).read_bytes()).hexdigest()==meta['sha256']
                    assert manifest[rid]['files']['final_model.pt']['bytes']>250000000
                    assert manifest[rid]['files']['probe_tokens.npz']['bytes']>4000000
                s=json.loads((path/'summary.json').read_text());c=s['config'];rows=read_jsonl(path/'train_log.jsonl')
                assert s['steps']==1665 and s['seed']==42 and [r['step'] for r in rows]==STEPS
                assert c['nanogpt_adam_lr']==lr and c['table_lr_scale']==128
                assert c['device_batch_size']==72 and c['total_batch_size']==147456 and c['epoch_batches']==333
                assert c['lr_schedule']=='warmup_constant' and c['warmup_steps']==100 and c['data_seed']==42
                assert c['enable_trigram_ve']==(arm!='nogram') and not c['enable_bigram_ve']
                assert s['freeze_table']==(arm=='frozen') and s['compute_dtype']=='bf16'
                assert c['val_batches']==4 and c['val_interval_steps']==10
                a={k:np.array([r[k] for r in rows]) for k in ['step','train_loss','val_loss','gap']}
                assert all(np.isfinite(v).all() for v in a.values())
                assert np.allclose(a['val_loss']-a['train_loss'],a['gap'],rtol=0,atol=1e-7)
                for k in ['train_loss','val_loss','gap']:assert abs(a[k][-1]-s['final_'+k])<1e-7
                if arm=='frozen':
                    norms=read_jsonl(path/'table_norm.jsonl')
                    for k in norms[0]:
                        if k!='step':assert len({r[k] for r in norms})==1
                row={'source':source,'arm':arm,'lr_tag':tag,'lr':lr,'run_id':rid,'seed':42,'step':1665,
                     'train_loss':s['final_train_loss'],'val_loss':s['final_val_loss'],'online_gap':s['final_gap']}
                if tag!='base':
                    assert s['fixed_train_probe_batches']==4 and s['train_probe_mode']=='uniform'
                    h=s['fixed_train_probe_sha256'];hashes.setdefault(source,set()).add(h)
                    fixed=read_jsonl(path/'fixed_train_loss.jsonl')
                    assert fixed[-1]['step']==1665 and set(STEPS).issubset({r['step'] for r in fixed})
                    for k in ['fixed_train_loss','fixed_val_loss','fixed_gap']:
                        a[k]=np.array([r[k] for r in fixed]);assert np.isfinite(a[k]).all()
                        row[k]=s['final_'+k]
                    assert np.allclose(a['fixed_val_loss']-a['fixed_train_loss'],a['fixed_gap'],rtol=0,atol=1e-7)
                    a['fixed_step']=np.array([r['step'] for r in fixed]);row['probe_sha256']=h
                    for r in fixed:curves.append({'source':source,'arm':arm,'lr_tag':tag,'run_id':rid,'step':r['step'],'metric':'fixed_gap','value':r['fixed_gap']})
                else:
                    b=next((x for x in baseline['runs'] if x['source']==legacy and x['arm']==arm),None)
                    if b:
                        assert abs(b['val_reproduction_error'])<0.005
                        row.update(fixed_train_loss=b['fixed_train'],fixed_val_loss=b['fixed_val'],fixed_gap=b['fixed_gap'],probe_sha256=b['probe_sha256'])
                        hashes.setdefault(source,set()).add(b['probe_sha256']);a['fixed_endpoint']=b['fixed_gap']
                data[source,arm,tag]=a;end.append(row)
                for r in rows:
                    for k in ['train_loss','val_loss','gap']:curves.append({'source':source,'arm':arm,'lr_tag':tag,'run_id':rid,'step':r['step'],'metric':k,'value':r[k]})
    assert all(len(h)==1 for h in hashes.values()),hashes
    for name,rows in [('endpoints.csv',end),('curves.csv',curves)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (PACKET/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return data,end


def style(ax,gap=False):
    for e in range(1,6):ax.axvline(e,color='#bbc0c5',lw=.5,ls=':',zorder=0)
    if gap:ax.axhline(0,color='#888',lw=.6,ls=':',zorder=0)
    ax.set_xlim(0,5);ax.set_xticks(range(6));ax.grid(True,axis='y');ax.grid(False,axis='x')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)


def save(fig,name):
    for ext in ['png','pdf','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight')
        if ext=='svg':p.write_text(chr(10).join(x.rstrip() for x in p.read_text().splitlines())+chr(10))


def plot(data,partial):
    OUT.mkdir(exist_ok=True,parents=True);apply_style()
    handles=[Line2D([],[],color=c,label=label,lw=1.4) for tag,_,c,label in LRS if any(k[2]==tag for k in data)]
    handles += [Line2D([],[],color='#222',ls='-',label='Online train'),Line2D([],[],color='#222',ls='--',label='Fixed validation')]
    with PdfPages(OUT/'all_source_lr_curves.pdf') as book:
        for source,label in SOURCES:
            fig,axs=plt.subplots(3,3,figsize=(10.6,8.2),sharex=True,sharey='row')
            fig.subplots_adjust(left=.085,right=.99,bottom=.10,top=.84,hspace=.22,wspace=.22)
            fig.suptitle(label+' | LR sweep | seed 42'+(' | completed runs only' if partial else ''),y=.99,fontsize=14)
            fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.52,.95),fontsize=9)
            for j,(arm,name) in enumerate(ARMS):
                axs[0,j].set_title(name)
                if partial:
                    missing=[tag for tag,_,_,_ in LRS if (source,arm,tag) not in data]
                    if missing:axs[0,j].text(.98,.98,'Pending: '+', '.join(missing),transform=axs[0,j].transAxes,ha='right',va='top',fontsize=8,color='#777')
                for tag,_,color,_ in LRS:
                    a=data.get((source,arm,tag))
                    if a is None:continue
                    x=a['step']/333
                    axs[0,j].plot(x,a['train_loss'],color=color,lw=1.1)
                    axs[0,j].plot(x,a['val_loss'],color=color,lw=1.1,ls='--')
                    axs[1,j].plot(x,a['gap'],color=color,lw=1.1)
                    if 'fixed_step' in a:axs[2,j].plot(a['fixed_step']/333,a['fixed_gap'],color=color,lw=1.1)
                    if 'fixed_endpoint' in a:axs[2,j].plot(5,a['fixed_endpoint'],'o',color=color,ms=5,clip_on=False)
                for i in range(3):style(axs[i,j],i>0)
                if not axs[2,j].lines or not any('fixed_step' in data.get((source,arm,tag),{}) or 'fixed_endpoint' in data.get((source,arm,tag),{}) for tag,_,_,_ in LRS):
                    axs[2,j].text(.5,.5,'Fixed-probe results pending',transform=axs[2,j].transAxes,ha='center',color='#777',fontsize=9)
                axs[2,j].set_xlabel('Replay pass')
            for i,label in enumerate(['Loss (nats/token)','Online gap (nats/token)','Fixed-probe gap (nats/token)']):axs[i,0].set_ylabel(label)
            fig.text(.5,.025,'Raw points; no smoothing. Fixed diagnostic uses 4 uniform train batches. Original-LR fixed value is endpoint-only.',ha='center',fontsize=8.5)
            save(fig,'lr_curves_'+source);book.savefig(fig,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(12,5.8),sharex=True)
    fig.subplots_adjust(left=.075,right=.99,bottom=.14,top=.82,wspace=.31,hspace=.32)
    colors={'trainable':'#2d6f9f','frozen':'#c58a0b','nogram':'#686d73'}
    fig.suptitle('LR sensitivity after five replay passes | seed 42',y=.99,fontsize=14)
    fig.legend(handles=[Line2D([],[],color=colors[a],marker='o',label=n) for a,n in ARMS],loc='upper center',bbox_to_anchor=(.5,.94),ncol=3)
    for j,(source,label) in enumerate(SOURCES):
        axs[0,j].set_title(label,fontsize=11)
        for arm,_ in ARMS:
            xx=[];gap=[];val=[]
            for tag,lr,_,_ in LRS:
                a=data.get((source,arm,tag))
                if a is None:continue
                xx.append(lr/0.0006);gap.append(a['gap'][-1]);val.append(a['val_loss'][-1])
            axs[0,j].plot(xx,gap,'o-',color=colors[arm],ms=4);axs[1,j].plot(xx,val,'o-',color=colors[arm],ms=4)
        for i in range(2):
            axs[i,j].set_xscale('log');axs[i,j].set_xticks([.1,.25,1],['1/10','1/4','1']);axs[i,j].grid(True,axis='y')
        axs[1,j].set_xlabel('LR / original LR')
    axs[0,0].set_ylabel('Online gap (nats/token)');axs[1,0].set_ylabel('Validation loss (nats/token)')
    fig.text(.5,.018,'Same initial checkpoints, data, total tokens and table/backbone LR ratio (128). Lower gap alone does not establish better generalization.',ha='center',fontsize=8.2)
    save(fig,'lr_endpoint_overview');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    data,rows=load(a.partial);plot(data,a.partial);print('Verified',len(rows),'endpoints;',OUT)


if __name__=='__main__':main()
