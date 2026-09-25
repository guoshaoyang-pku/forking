#!/usr/bin/env python3
"""Plot recovered fixed-head curves and independent full-pool checkpoint audit."""
from pathlib import Path
import csv
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v5_style import apply_style
SRC = ROOT / 'docs/appendices/freeze_fixed_gap_20260925'
OUT = ROOT / 'docs/figs/freeze_fixed_gap_20260925'


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=220, bbox_inches='tight')
    p = OUT/(name+'.svg')
    fig.savefig(p, bbox_inches='tight')
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def finish(ax, freeze):
    ax.axvline(freeze, color='#999', ls=':', lw=1)
    ax.grid(True, axis='y');ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True);apply_style()
    runs = {}
    csvrows = []
    for f in sorted(SRC.glob('*_exact_reduction.json')):
        for r in json.loads(f.read_text()):
            if 'missing' in r:continue
            runs[r['run_id']] = r
            for p in r['points']:
                csvrows.append({'run_id':r['run_id'],'seed':r['summary']['seed'],**p})
    with (SRC/'recovered_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows)
    fig,ax=plt.subplots(2,3,figsize=(11,6.2),sharex=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.79,hspace=.28,wspace=.30)
    fig.suptitle('Freeze after pass 1: online vs fixed-head diagnostics',fontsize=13,y=.99)
    fig.legend(handles=[Line2D([],[],color='#2d6f9f',label='Fixed-head train'),
                        Line2D([],[],color='#c4493d',ls='--',label='Fixed validation'),
                        Line2D([],[],color='#888',ls=':',label='Online train / gap'),
                        Line2D([],[],color='#2d6f9f',label='Fixed-head gap')],
               loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,fontsize=9)
    specs=[('causalv5c_none_128x_fixed','No freezing'),
           ('causalv5c_freeze_table_e1_128x_fixed','Freeze table at step 338'),
           ('causalv5c_freeze_backbone_e1_128x_fixed','Freeze backbone at step 338')]
    for j,(rid,title) in enumerate(specs):
        ps=runs[rid]['points'];x=[p['step'] for p in ps]
        for k,c,ls in [('train_fixed_head','#2d6f9f','-'),('val_fixed_diag','#c4493d','--'),('train_loss','#888',':')]:
            ax[0,j].plot(x,[p[k] for p in ps],color=c,ls=ls,lw=1.2)
        ax[1,j].plot(x,[p['fixed_head_gap'] for p in ps],color='#2d6f9f',lw=1.4)
        ax[1,j].plot(x,[p['gap'] for p in ps],color='#888',ls=':',lw=1.3)
        for i in range(2):finish(ax[i,j],338);ax[i,j].set_xlim(0,1000)
        ax[0,j].set_title(title,fontsize=10);ax[1,j].set_xlabel('Training step')
        ax[1,j].axhline(0,color='#aaa',lw=.6)
    ax[0,0].set_ylabel('Loss (nats/token)');ax[1,0].set_ylabel('Gap (nats/token)')
    fig.text(.5,.02,'Seed 42. Fixed-head = first 4 training batches (589,824 tokens); not a full-pool estimate. No smoothing.',ha='center',fontsize=9)
    save(fig,'fig_e1_online_vs_fixed_head')

    fig,ax=plt.subplots(1,2,figsize=(10,4.2));fig.subplots_adjust(left=.075,right=.985,bottom=.21,top=.73,wspace=.27)
    fig.suptitle('Freeze after pass 2: fixed-head gap continues to grow',fontsize=13,y=.99)
    arms=[('table','Freeze table','#c58a0b'),('backbone','Freeze backbone','#2d6f9f'),('both','Freeze both','#686d73')]
    for arm,label,c in arms:
        ps=runs[f'ffqv5_freeze_{arm}_e2_10ep_fixed']['points'];x=np.array([p['step'] for p in ps])/337
        ax[0].plot(x,[p['fixed_head_gap'] for p in ps],color=c,marker='o',ms=3,lw=1.3,label=label)
        ax[1].plot(x,[p['val_fixed_diag'] for p in ps],color=c,marker='o',ms=3,lw=1.3)
    for a in ax:finish(a,675/337);a.set_xlabel('Replay pass');a.set_xlim(.8,10.2);a.set_xticks(range(1,11))
    ax[0].set_ylabel('Fixed-head gap (nats/token)');ax[1].set_ylabel('Fixed validation loss (nats/token)')
    fig.legend(handles=ax[0].get_legend_handles_labels()[0],labels=ax[0].get_legend_handles_labels()[1],loc='upper center',bbox_to_anchor=(.5,.92),ncol=3)
    fig.text(.5,.035,'Seed 42; intervention at step 675. Dots are measured pass endpoints; lines only connect observations.',ha='center',fontsize=9)
    save(fig,'fig_e2_long_fixed_gap')

    full=json.loads((SRC/'full_train.json').read_text());assert full['complete']
    fig,ax=plt.subplots(1,2,figsize=(10,4.4));fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.73,wspace=.28)
    fig.suptitle('Freeze after pass 1: evaluate the entire training pool at step 1000',fontsize=12,y=.99)
    handles=[]
    for j,r in enumerate(full['runs']):
        assert len(r['train_batch_losses'])==337 and r['val_reproduction_error']==0
        c=['#2d6f9f','#c58a0b'][j];label=f'Seed {r["seed"]}'
        ax[0].plot(range(1,338),r['train_batch_losses'],color=c,lw=1,alpha=.9,label=label)
        ax[0].axhline(r['val_mean'],color=c,ls='--',lw=1)
        gaps=[r['fixed_head_gap'],r['fixed_full_gap'],r['online_gap']]
        ax[1].plot(np.arange(3)+(j-.5)*.10,gaps,'o-',color=c,lw=1,ms=5)
        handles.append(Line2D([],[],color=c,label=label))
    handles.append(Line2D([],[],color='#444',ls='--',label='Fixed validation'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3)
    ax[0].set_xlabel('Training batch position in the replay pool');ax[0].set_ylabel('Same-checkpoint loss (nats/token)')
    ax[0].set_xlim(1,337);ax[1].set_xticks(range(3),['Fixed head\n(first 4 batches)','Fixed full pool\n(all 337 batches)','Online\n(last logged point)'])
    ax[1].set_ylabel('Gap (nats/token)')
    for a in ax:a.grid(True,axis='y');a.grid(False,axis='x');a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
    fig.text(.5,.035,'Strict checkpoint load; BF16; original fixed validation reproduced exactly. No optimization or temporal smoothing.',ha='center',fontsize=9)
    save(fig,'fig_full_pool_position_audit')
    print(f'Wrote {len(csvrows)} recovered observations and 3 figures to {OUT}')


if __name__=='__main__':main()
