#!/usr/bin/env python3
"""Unified SFT-source/static-RLVR curves from frozen corrected-tokenizer logs."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v5_style import apply_style

SRC = ROOT / 'docs/appendices/posttraining_fixtok_20260925'
OUT = ROOT / 'docs/figs/posttraining_fixtok_20260925'
SOURCES = [('id', 'ID'), ('magicoder', 'Magicoder'), ('openr1_10shards', 'OpenR1'), ('rlvr_math', 'RLVR-MATH')]
ARMS = [('trainable', 'Table trainable', '#2d6f9f'), ('frozen', 'Table frozen', '#c58a0b'), ('nogram', 'No-gram', '#686d73')]
SOURCE_COLORS = ['#2d6f9f', '#c4493d', '#c58a0b', '#7b4fa6']
EXPECTED = list(range(10, 1661, 10)) + [1665]


def load():
    for manifest_name, directory in [('old_tokenizer_manifest.json', 'old_tokenizer_runs'),
                                     ('pretrain_reference_manifest.json', 'pretrain_reference')]:
        for run in json.loads((SRC / manifest_name).read_text()):
            for filename, expected in run['sha256'].items():
                assert hashlib.sha256((SRC / directory / run['run_id'] / filename).read_bytes()).hexdigest() == expected
    manifest = json.loads((SRC / 'source_manifest.json').read_text())
    sha = {r['run_id']:r['sha256'] for r in manifest}
    data = {}; endpoint = []; audit = []; curves = []
    for source, _ in SOURCES:
        data[source] = {}
        for arm, _, _ in ARMS:
            runs = []
            for seed in ([42] if source == 'id' else [42,43,44]):
                if source == 'id':
                    rid = 'sft_id_e5_nogram_s42' if arm == 'nogram' else f'sft_id_tri_e5_freq10_{arm}_s42'
                else:
                    rid = f'sft_{source}_fixtok_e5_{arm}_s{seed}'
                path = SRC / 'runs' / rid
                for fn, expected in sha[rid].items():
                    assert hashlib.sha256((path/fn).read_bytes()).hexdigest() == expected
                summary = json.loads((path/'summary.json').read_text())
                config = summary['config']
                rows = [json.loads(x) for x in (path/'train_log.jsonl').read_text().splitlines() if x.strip()]
                assert [r['step'] for r in rows] == EXPECTED
                assert summary['steps'] == config['max_steps'] == 1665 and summary['seed'] == seed
                assert config['epoch_batches'] == 333 and config['data_mode'] == 'fixed'
                assert config['val_interval_steps'] == 10 and config['val_batches'] == 4
                assert config['device_batch_size'] == 72 and config['total_batch_size'] == 147456
                assert config['dropout'] == 0 and summary['compute_dtype'] == 'bf16'
                assert config['nanogpt_adam_lr'] == 0.0006 and config['warmup_steps'] == 100
                assert config['lr_schedule'] == 'warmup_constant' and config['data_seed'] == 42
                assert summary['freeze_table'] == (arm == 'frozen')
                assert config['enable_trigram_ve'] == (arm != 'nogram')
                assert not config['enable_bigram_ve']
                assert not summary.get('fixed_train_probe_batches', 0)
                if source != 'id': assert 'fixtok' in config['data_dir']
                a = {k:np.array([r[k] for r in rows], float) for k in ['step','train_loss','val_loss','gap']}
                assert all(np.isfinite(v).all() for v in a.values())
                assert np.allclose(a['val_loss']-a['train_loss'],a['gap'],rtol=0,atol=1e-7)
                for key in ['train_loss','val_loss','gap']: assert abs(a[key][-1]-summary['final_'+key]) < 1e-7
                norms = [json.loads(x) for x in (path/'table_norm.jsonl').read_text().splitlines() if x.strip()]
                if arm == 'frozen':
                    for key in norms[0]:
                        if key != 'step': assert max(r[key] for r in norms) == min(r[key] for r in norms)
                runs.append(a)
                endpoint.append({'source':source,'arm':arm,'run_id':rid,'seed':seed,'step':1665,**{k:float(a[k][-1]) for k in ['train_loss','val_loss','gap']}})
                for row in rows: curves.append({'source':source,'arm':arm,'run_id':rid,'seed':seed,**{k:row[k] for k in ['step','train_loss','val_loss','gap']}})
                for epoch in range(1,6):
                    mask=(a['step']>(epoch-1)*333)&(a['step']<=epoch*333)
                    audit.append({'source':source,'arm':arm,'run_id':rid,'seed':seed,'epoch':epoch,'logged_points':int(mask.sum()),
                                  **{k+'_logged_mean':float(a[k][mask].mean()) for k in ['train_loss','val_loss','gap']}})
            data[source][arm] = {k:np.stack([a[k] for a in runs]) for k in runs[0]}
    for name, rows in [('endpoints.csv',endpoint),('epoch_logged_means.csv',audit),('all_curves.csv',curves)]:
        with (SRC/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(f'Validated {len(endpoint)} runs, {len(curves)} loss records and frozen table norms.')
    return data


def axes_style(ax, gap=False):
    for e in range(1,6): ax.axvline(e,color='#b8bec4',lw=.55,ls=':',zorder=0)
    if gap: ax.axhline(0,color='#80868b',lw=.65,ls=':',zorder=0)
    ax.set_xlim(0,5); ax.set_xticks(range(6)); ax.grid(True,axis='y'); ax.grid(False,axis='x')
    ax.tick_params(pad=2)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)


def draw(ax, x, array, color, ls='-', seeds=False):
    if seeds and len(array)>1:
        for row in array: ax.plot(x,row,color=color,lw=.55,alpha=.28,ls=ls)
    mean=array.mean(axis=0)
    if len(array)>1:
        sd=array.std(axis=0,ddof=1);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.10,lw=0)
    ax.plot(x,mean,color=color,lw=1.25,ls=ls)


def save(fig, name, pdf=True):
    for ext in (['png','svg','pdf'] if pdf else ['png','svg']):
        path = OUT/(name+'.'+ext)
        fig.savefig(path,dpi=220,bbox_inches='tight')
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def overview(data, sources, name, title, arms=ARMS, pdf=True, publication=False):
    single = len(sources) == 1
    if publication:
        # ICLR single-column width: keep the labels legible after
        # includegraphics[width=\linewidth] scaling.
        figsize = (5.5, 2.70)
        margins = dict(left=.080,right=.995,bottom=.185,top=.80,hspace=.14,wspace=.22)
    else:
        figsize = (4.6 if single else 2.25*len(sources)+.5,5.1 if single else 4.45)
        margins = dict(left=.17 if single else .09,right=.975,bottom=.17 if single else .14,top=.74 if single else .79,hspace=.22,wspace=.30)
    fig,ax=plt.subplots(2,len(sources),figsize=figsize,squeeze=False,sharex='col',gridspec_kw={'height_ratios':[1,.85]})
    fig.subplots_adjust(**margins)
    if title and not publication:
        fig.suptitle(title,fontsize=12,y=.995)
    handles=[Line2D([],[],color=c,lw=1.6,label=n) for _,n,c in arms]
    handles += [Line2D([],[],color='#222',ls='-',label='Online train'),Line2D([],[],color='#222',ls='--',label='Fixed validation')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.985 if publication else .935),ncol=4 if publication else (5 if len(sources)>1 else 2),fontsize=7.2 if publication else 8.5,columnspacing=1.1 if publication else 1.3,handlelength=2.0)
    for j,(source,label) in enumerate(sources):
        for arm,_,color in arms:
            a=data[source][arm];x=a['step'][0]/333
            draw(ax[0,j],x,a['train_loss'],color)
            draw(ax[0,j],x,a['val_loss'],color,'--')
            draw(ax[1,j],x,a['gap'],color)
        for i in range(2):axes_style(ax[i,j],i==1)
        ax[0,j].set_title(label,fontsize=8.2 if publication else 10)
        ax[1,j].set_xlabel('epoch' if publication else 'Replay pass',fontsize=8.0 if publication else None,labelpad=2 if publication else None)
        if publication:
            for axis in (ax[0,j],ax[1,j]):
                axis.tick_params(axis='both',labelsize=7.0,pad=1.5)
    ax[0,0].set_ylabel('loss (nats)',fontsize=8.0 if publication else None,labelpad=2)
    ax[1,0].set_ylabel('gap (nats)',fontsize=8.0 if publication else None,labelpad=2)
    if not publication:
        note = 'Raw logged points; mean ±1 sample SD. No smoothing.' if single else 'Raw logged points; mean and ±1 sample SD across downstream seeds. No smoothing.'
        fig.text(.5,.015,note,ha='center',fontsize=8)
    save(fig,name,pdf);plt.close(fig)


def atlas(data):
    with PdfPages(OUT/'posttraining_all_seed_curves.pdf') as book:
        for idx,(source,label) in enumerate(SOURCES):
            fig,ax=plt.subplots(2,3,figsize=(9.2,5.3),sharex='col',sharey='row',gridspec_kw={'height_ratios':[1,.85]})
            fig.subplots_adjust(left=.085,right=.985,bottom=.14,top=.78,hspace=.22,wspace=.22)
            fig.suptitle(label+' | correct tokenizer | '+('seed 42' if source=='id' else 'seeds 42, 43, 44'),fontsize=13,y=.985)
            handles=[Line2D([],[],color='#222',ls='-',label='Online train'),Line2D([],[],color='#222',ls='--',label='Fixed validation'),Line2D([],[],color='#222',lw=.6,alpha=.35,label='Individual seeds'),Line2D([],[],color='#222',lw=1.4,label='Mean (band: ±1 SD)')]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=4,fontsize=9)
            for j,(arm,name,_) in enumerate(ARMS):
                a=data[source][arm];x=a['step'][0]/333;c=SOURCE_COLORS[idx]
                draw(ax[0,j],x,a['train_loss'],c,seeds=True);draw(ax[0,j],x,a['val_loss'],c,'--',seeds=True)
                draw(ax[1,j],x,a['gap'],c,seeds=True)
                for i in range(2): axes_style(ax[i,j],i==1)
                ax[0,j].set_title(name);ax[1,j].set_xlabel('Replay pass')
            ax[0,0].set_ylabel('Loss (nats/token)');ax[1,0].set_ylabel('Online gap (nats/token)')
            fig.text(.5,.018,'Shared y scales across arms. Gap = fixed validation after update − current-batch train before update.',ha='center',fontsize=8.5)
            save(fig,'all_seeds_'+source,pdf=False);book.savefig(fig,bbox_inches='tight');plt.close(fig)
        # Every seed is also shown independently: no seed can disappear in an average.
        for idx,(source,label) in enumerate(SOURCES):
            for si,seed in enumerate([42] if source=='id' else [42,43,44]):
                fig,ax=plt.subplots(2,3,figsize=(9.2,4.9),sharex='col',sharey='row')
                fig.subplots_adjust(left=.085,right=.985,bottom=.15,top=.79,hspace=.23,wspace=.22)
                fig.suptitle(f'{label} | seed {seed} | raw logged curves',fontsize=13,y=.985)
                fig.legend(handles=[Line2D([],[],color='#222',ls='-',label='Online train'),Line2D([],[],color='#222',ls='--',label='Fixed validation')],loc='upper center',bbox_to_anchor=(.5,.93),ncol=2)
                for j,(arm,name,_) in enumerate(ARMS):
                    a=data[source][arm];x=a['step'][si]/333;c=SOURCE_COLORS[idx]
                    ax[0,j].plot(x,a['train_loss'][si],color=c,lw=1.2);ax[0,j].plot(x,a['val_loss'][si],color=c,lw=1.2,ls='--')
                    ax[1,j].plot(x,a['gap'][si],color=c,lw=1.2)
                    for i in range(2):axes_style(ax[i,j],i==1)
                    ax[0,j].set_title(name);ax[1,j].set_xlabel('Replay pass')
                ax[0,0].set_ylabel('Loss (nats/token)');ax[1,0].set_ylabel('Online gap (nats/token)')
                book.savefig(fig,bbox_inches='tight');plt.close(fig)
        for fig in diagnostic_figures(data):
            book.savefig(fig,bbox_inches='tight');plt.close(fig)


def diagnostic_figures(data):
    figures=[]
    # Old data are explicitly invalid, and are never averaged with corrected results.
    fig,ax=plt.subplots(2,3,figsize=(9.2,5.0),sharex=True,sharey='col')
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.79,hspace=.38,wspace=.25)
    fig.suptitle('Tokenizer correction: old vs corrected online gaps (seed 42)',fontsize=12,y=.985)
    fig.legend(handles=[Line2D([],[],color='#c4493d',ls='--',label='Old tokenizer (invalid)'),Line2D([],[],color='#2d6f9f',label='Correct tokenizer')],loc='upper center',bbox_to_anchor=(.5,.925),ncol=2)
    for j,(source,label) in enumerate(SOURCES[1:]):
        for i,arm in enumerate(['trainable','frozen']):
            rid = f'sft_rlvr_math_e5_{arm}_s42' if source=='rlvr_math' else f'sft_ood_{source}_e5_freq10_{arm}_s42'
            rows=[json.loads(x) for x in (SRC/'old_tokenizer_runs'/rid/'train_log.jsonl').read_text().splitlines() if x.strip()]
            ax[i,j].plot([r['step']/333 for r in rows],[r['gap'] for r in rows],color='#c4493d',ls='--',lw=1.0)
            a=data[source][arm];ax[i,j].plot(a['step'][0]/333,a['gap'][0],color='#2d6f9f',lw=1.2)
            axes_style(ax[i,j],True);ax[i,j].set_title(label+' | '+arm,fontsize=10)
            if i==1:ax[i,j].set_xlabel('Replay pass')
    for i in range(2):ax[i,0].set_ylabel('Online gap (nats/token)')
    fig.text(.5,.015,'Encoding also changes token packing; old/new differences are not a pure matched-text causal effect.',ha='center',fontsize=8)
    save(fig,'fig_tokenizer_before_after',pdf=False);figures.append(fig)
    checkpoint=json.loads((SRC/'checkpoint_audit.json').read_text())
    fig,ax=plt.subplots(1,4,figsize=(9.2,3.2));fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.70,wspace=.30)
    fig.suptitle('Same-checkpoint audit | seed 42 | step 1665',fontsize=12,y=.99)
    fig.legend(handles=[Line2D([],[],color='#222',marker='o',ls='',label='Logged online gap'),Line2D([],[],color='#222',marker='D',ls='',mfc='white',label='Sampled fixed-train gap')],loc='upper center',bbox_to_anchor=(.5,.90),ncol=2)
    for j,(source,label) in enumerate(SOURCES):
        for i,(arm,_,color) in enumerate(ARMS):
            r=next(x for x in checkpoint['runs'] if x['source']==source and x['arm']==arm)
            ax[j].plot([i-.12,i+.12],[r['logged_online_gap'],r['sampled_fixed_gap']],color=color,lw=.8)
            ax[j].plot(i-.12,r['logged_online_gap'],'o',color=color,ms=5)
            ax[j].plot(i+.12,r['sampled_fixed_gap'],'D',color=color,mfc='white',ms=5)
        ax[j].set_xticks(range(3),['Trainable','Frozen','No-gram'],rotation=25,ha='right');ax[j].set_title(label,fontsize=10);ax[j].axhline(0,color='#aaa',lw=.6)
    ax[0].set_ylabel('Gap (nats/token)')
    fig.text(.5,.015,'Fixed diagnostic: 33 train batches spanning the pool; same four validation batches. Separate metric.',ha='center',fontsize=8)
    save(fig,'fig_checkpoint_gap_audit',pdf=False);figures.append(fig)
    fig,ax=plt.subplots(1,2,figsize=(8,3.3));fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.74,wspace=.28)
    specs=[('causalv5c_none_128x_fixed','Table trainable','#2d6f9f'),('causalv5c_freeze_table_e1_128x_fixed','Freeze after pass 1','#c58a0b'),('nglab1x_nogram_v5_128x_freq10_fixed','No-gram reference','#686d73')]
    for rid,label,color in specs:
        p=SRC/'pretrain_reference'/rid;summary=json.loads((p/'summary.json').read_text());assert summary['seed']==42
        rows=[json.loads(x) for x in (p/'train_log.jsonl').read_text().splitlines() if x.strip()];rows=[r for r in rows if r['step']<=1000]
        x=[r['step'] for r in rows]
        ax[0].plot(x,[r['train_loss'] for r in rows],color=color,lw=1.2);ax[0].plot(x,[r['val_loss'] for r in rows],color=color,lw=1.2,ls='--')
        ax[1].plot(x,[r['gap'] for r in rows],color=color,lw=1.2,label=label)
    for a in ax:
        a.axvline(338,color='#999',ls=':',lw=.8);a.set_xlabel('Training step');a.set_xlim(0,1000)
    ax[0].set_ylabel('Loss (nats/token)');ax[1].set_ylabel('Online gap (nats/token)')
    fig.suptitle('Pretraining reference: freeze on the same corpus after pass 1',fontsize=12,y=.98)
    fig.legend(handles=ax[1].get_legend_handles_labels()[0],labels=ax[1].get_legend_handles_labels()[1],loc='upper center',bbox_to_anchor=(.5,.88),ncol=3)
    fig.text(.5,.018,'Seed 42; input bigram+trigram; train solid, fixed validation dashed. Distinct protocol from post-training.',ha='center',fontsize=8)
    save(fig,'fig_pretraining_freeze_reference',pdf=False);figures.append(fig)
    return figures


def main():
    OUT.mkdir(parents=True,exist_ok=True);apply_style();data=load()
    # Publication overview: keep the four source columns, but compare only
    # the two n-gram table policies.  The no-gram arm remains available in
    # the explicitly named diagnostic output below and is not part of this
    # SFT/RL comparison.
    overview(data,SOURCES,'fig_sft_rl_combined',None,arms=ARMS[:2],publication=True)
    # Preserve the historical basename as an alias with the same contents so
    # existing figure references cannot silently pick up the old no-gram panel.
    overview(data,SOURCES,'fig_posttraining_combined',None,arms=ARMS[:2],publication=True)
    overview(data,SOURCES[:3],'fig_sft_unified','SFT-source continued-LM stress tests')
    overview(data,SOURCES[3:],'fig_rlvr_unified','Static RLVR-MATH replay proxy')
    overview(data,SOURCES,'fig_nogram_audit','No-gram baseline: train, validation and gap',arms=ARMS[2:],pdf=False)
    atlas(data)
    print(OUT)


if __name__=='__main__':main()
