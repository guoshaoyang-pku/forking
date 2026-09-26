#!/usr/bin/env python3
"""Reproduce coverage audit from hash-verified compact inputs; no GPU or /tmp dependency.
Online gap: post-update fixed-val CE minus pre-update current-batch CE.
Fits are descriptive equal-bin OLS, F<=1000, >=300 tokens per side.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v5_style as style
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/appendices/coverage_takeover_20260925'
OUT=ROOT/'docs/figs/coverage_20260925'
EDGES=[1,2,3,4,5,6,9,13,21,31,51,76,101,151,201,301,501,751,1001]
STEPS=[670,1000,1340,2000]
COLORS=['#009E73','#0072B2','#CC79A7','#D55E00']

def write_csv(name,rows):
    with (BASE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def fit(x,y,w=None):
    w=np.ones(len(x)) if w is None else np.asarray(w)
    a,b=np.polyfit(x,y,1,w=np.sqrt(w)); residual=y-a*x-b
    rmse=np.sqrt(np.average(residual**2,weights=w))
    return dict(a=float(a),b=float(b),rmse=float(rmse),relative_rmse=float(rmse/np.average(abs(y),weights=w)),r2=float(1-np.average(residual**2,weights=w)/np.average((y-np.average(y,weights=w))**2,weights=w)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    z=np.load(BASE/'inputs/corpus_fval.npz')
    data=json.loads((BASE/'inputs/online_selected.json').read_text())
    manifest=json.loads((BASE/'source_manifest.json').read_text())
    for src in manifest.get('compact_inputs',[]):
        p=ROOT/src['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256'],str(p)
    # Verify inputs when full local logs exist; packed rows remain standalone.
    checked=0
    for src in manifest['sources']:
        p=ROOT/src['path']
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256'],str(p)
            checked+=1
    points=[]; stats=[]; corpus=[]; checks=[]
    for name,obj in data.items():
        cfg=obj['summary']['config']; seed=obj['summary']['seed']
        assert cfg['train_shards']==[1] and 1 not in cfg['val_shards']
        assert cfg['val_batches']==4 and cfg['val_interval_steps']==10
        assert cfg['lr_schedule']=='warmup_constant' and cfg['table_lr_scale']==128
        assert obj['summary']['steps']==2000
        arm='both' if '_both_' in name else 'nogram'
        for row in obj['rows']:
            for branch,g in [('bigram','bi'),('trigram','tri')]:
                # Scalar consistency includes F=0, but F=0 is never assigned a gap.
                tr=sum(v['mean_loss']*v['token_count'] for v in row['train'][branch].values())/sum(v['token_count'] for v in row['train'][branch].values())
                va=sum(v['mean_loss']*v['token_count'] for v in row['val'][branch].values())/sum(v['token_count'] for v in row['val'][branch].values())
                if row['step']==2000:
                    checks.append(abs(tr-obj['summary']['final_train_loss']))
                    checks.append(abs(va-obj['summary']['final_val_loss']))
                for lo,hi in zip(EDGES[:-1],EDGES[1:]):
                    label=str(lo) if hi==lo+1 else f'{lo}-{hi-1}'
                    if label=='751-1000':label='751-1k'
                    tv=row['train'][branch][label];vv=row['val'][branch][label]
                    n=z[f'probe_{g}_n'][lo:hi].sum();nov=z[f'probe_{g}_novel'][lo:hi].sum()
                    assert n==vv['token_count'],(name,branch,label,n,vv['token_count'])
                    if min(tv['token_count'],n)<300:continue
                    points.append(dict(run_id=name,seed=seed,arm=arm,step=row['step'],branch=branch,F_lo=lo,F_hi=hi-1,f=float(nov/n),gap=vv['mean_loss']-tv['mean_loss'],train_n=tv['token_count'],val_n=vv['token_count']))
    assert max(checks)<3e-6,max(checks)
    for branch,g in [('bigram','bi'),('trigram','tri')]:
        nF=z[f'gt_{g}_nF'];N1=z[f'gt_{g}_N1F']
        for lo,hi in zip(EDGES[:-1],EDGES[1:]):
            fs=np.arange(lo,hi);w=(fs+1)*nF[fs+1]
            # Train-only pooled ratio; neither weights nor calibration use val.
            gt=N1[fs+1].sum()/w.sum()
            n=z[f'full_{g}_n'][fs].sum();nov=z[f'full_{g}_novel'][fs].sum()
            probe=z[f'probe_{g}_novel'][fs].sum()/z[f'probe_{g}_n'][fs].sum()
            corpus.append(dict(branch=branch,F_lo=lo,F_hi=hi-1,F_mid=float(np.sqrt(lo*(hi-1))),heldout=float(nov/n),probe=float(probe),gt_shift_train_only=float(gt),ratio=float(gt/(nov/n)),full_n=float(n)))
    def array(branch,step,arm='both',seed=None):
        rows=[p for p in points if p['branch']==branch and p['step']==step and p['arm']==arm and (seed is None or p['seed']==seed)]
        grouped={}
        for p in rows:grouped.setdefault(p['F_lo'],[]).append(p)
        rows=[grouped[k] for k in sorted(grouped)]
        return (np.array([r[0]['f'] for r in rows]),np.array([np.mean([x['gap'] for x in r]) for r in rows]),np.array([np.std([x['gap'] for x in r],ddof=1) if len(r)>1 else 0 for r in rows]),np.array([r[0]['val_n'] for r in rows]),np.array([np.sqrt(r[0]['F_lo']*r[0]['F_hi']) for r in rows]))
    for branch in ['bigram','trigram']:
        for step in STEPS:
            for seed in [42,43,44,None]:
                x,y,sd,w,F=array(branch,step,seed=seed)
                for weighting in ['equal_bin','val_token']:
                    st=fit(x,y,None if weighting=='equal_bin' else w)
                    d=float(x@y/(x@x));prop=np.sqrt(np.mean((y-d*x)**2))
                    # Deterministic alternating frequency-bin holdout.
                    train=np.arange(len(x))%2==0;test=~train;co=np.polyfit(x[train],y[train],1)
                    hr=np.sqrt(np.mean((y[test]-np.polyval(co,x[test]))**2))
                    stats.append(dict(branch=branch,step=step,seed='mean_3' if seed is None else seed,weighting=weighting,n_bins=len(x),**st,proportional_rmse=float(prop),alternating_holdout_rmse=float(hr),max_seed_sd=float(sd.max())))
    write_csv('online_points.csv',points);write_csv('online_fits.csv',stats);write_csv('corpus_bins.csv',corpus)
    style.apply_style();plt.rcParams.update({'font.size':8.5,'axes.labelsize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8})
    fig,axes=plt.subplots(2,2,figsize=(7.3,4.6),sharex='col',gridspec_kw={'height_ratios':[3,1],'hspace':0.12,'wspace':0.27})
    for col,branch in enumerate(['bigram','trigram']):
        a,r=axes[:,col]
        for step,c in zip(STEPS,COLORS):
            x,y,sd,w,F=array(branch,step);s=fit(x,y);xx=np.linspace(x.min(),x.max(),100)
            a.errorbar(x,y,yerr=sd,fmt='o',ms=3,color=c,lw=.65,capsize=1.4,label=f'Step {step}')
            a.plot(xx,s['a']*xx+s['b'],color=c,lw=1)
            r.plot(x,y-s['a']*x-s['b'],'o-',ms=2.5,lw=.6,color=c)
        x,y,sd,_,_=array(branch,1000,'nogram');a.plot(x,y,'x',ms=3,color='#686d73',label='No-gram, step 1000')
        a.axhline(0,color='#999999',lw=.6);r.axhline(0,color='#999999',lw=.6)
        a.text(.04,.96,f'({chr(97+col)}) {branch} contexts',transform=a.transAxes,va='top',fontweight='bold')
        a.legend(loc='upper left',bbox_to_anchor=(0,0.9),ncol=1,handlelength=1.1)
        r.set_xlabel('Measured novel-continuation fraction $f$');r.set_ylabel('Residual')
        a.set_ylabel('Online gap (nats)' if col==0 else 'Online gap (nats)')
        a.set_ylim(-.6,11.2);r.set_ylim(-.6,.6)
    save(fig,'fig_coverage_online')
    fig,axes=plt.subplots(2,2,figsize=(7.3,4.3),sharex='col',gridspec_kw={'height_ratios':[3,1],'hspace':.12,'wspace':.26})
    for col,branch in enumerate(['bigram','trigram']):
        rows=[r for r in corpus if r['branch']==branch];F=np.array([r['F_mid'] for r in rows]);v=np.array([r['heldout'] for r in rows]);g=np.array([r['gt_shift_train_only'] for r in rows]);a,r=axes[:,col];c=style.BRANCH_COLORS[branch]
        a.plot(F,v,'o-',ms=3,color=c,label='Held-out shard: measured')
        a.plot(F,g,'s--',ms=3,color='#333333',label='Training counts: shifted GT')
        r.plot(F,g/v,'o-',ms=3,color=c);r.axhline(1,color='#777777',lw=.8,ls=':');r.set_ylim(.78,1.04)
        a.text(.0,1.03,f'({chr(97+col)}) {branch} contexts',transform=a.transAxes,va='bottom',fontweight='bold');a.legend(loc='lower left')
        a.set_xscale('log');a.set_ylim(0,.86);a.set_ylabel('Novel-continuation fraction');r.set_ylabel('GT / measured');r.set_xlabel('Training context count $F$')
    save(fig,'fig_coverage_gt')
    # Honest exact-F strata from the existing quantized-probability diagnostic.
    raw=json.loads((BASE/'inputs/perpos_decomp.json').read_text());strata=[];decomp=[]
    def ds(branch,arm,ep,b,s):
        def ar(side,key):return np.asarray(raw[f'{arm}|{ep}|{side}'][branch][key]).reshape(-1,2,4)[b][:,s]
        nt=ar('train','n').sum();lt=ar('train','nll').sum();nv=ar('val','n');lv=ar('val','nll');nn=nv[0].sum();ns=nv[1].sum()
        if min(nt,nn+ns)<200 or min(nn,ns)==0:return None
        tr=lt/nt;f=nn/(nn+ns);D=lv[0].sum()/nn-tr;S=lv[1].sum()/ns-tr;gap=lv.sum()/(nn+ns)-tr
        assert abs(gap-f*D-(1-f)*S)<1e-10
        return dict(f=float(f),D=float(D),S=float(S),gap=float(gap),novel_term=float(f*D),seen_term=float((1-f)*S),train_n=float(nt),val_n=float(nn+ns))
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8),sharex=True)
    for col,branch in enumerate(['bigram','trigram']):
        a=axes[col]
        for F,c in zip([2,3,4,5,6,7,8],plt.cm.viridis(np.linspace(.05,.9,7))):
            rr=[]
            for s in range(4):
                v=ds(branch,'both','e3',F,[s])
                if v:rr.append(v);strata.append(dict(branch=branch,arm='both',epoch='e3',F=F,stratum=s,**v))
            a.plot([v['f'] for v in rr],[v['gap'] for v in rr],'o-',ms=3,color=c,label=f'$F={F}$')
        a.set_xlabel('Measured novel fraction within singleton stratum');a.set_ylabel('Fixed-probe diagnostic gap (nats)');a.text(.04,.95,f'({chr(97+col)}) {branch}',transform=a.transAxes,va='top',fontweight='bold');a.legend(loc='lower right',ncol=2,fontsize=6.8)
    save(fig,'fig_coverage_exactF_strata');write_csv('exactF_strata.csv',strata)
    e=raw['meta']['edges']
    for branch in ['bigram','trigram']:
        for arm in ['both',branch]:
            for ep in ['e2','e3']:
                for i,lo in enumerate(e[:-1]):
                    if e[i+1]>1245:continue
                    v=ds(branch,arm,ep,i+1,[0,1,2,3])
                    if v:decomp.append(dict(branch=branch,arm=arm,epoch=ep,F_lo=lo,F_hi=e[i+1]-1,**v))
    write_csv('diagnostic_decomposition.csv',decomp)
    # Paper Fig3a/Fig5(a) is no longer drawn here: it lives in
    # docs/plot_scripts/plot_paper_fig345_row.py (draw_frequency_coverage),
    # which renders the same online three-seed means and equal-bin OLS linear
    # fit on the shared row canvas without error bars, so panels (a)-(c) of
    # the LaTeX figure share one text size.  See the packet README.

    # Reader-facing triptych: three linear comparisons, without audit panels.
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.5),gridspec_kw={'wspace':.35})
    a=axes[0]
    a.plot([0,.85],[0,.85],'--',color='#777777',lw=.8,label='Equality')
    for br,mk in [('bigram','o'),('trigram','^')]:
        rows=[r for r in corpus if r['branch']==br]
        a.plot([r['heldout'] for r in rows],[r['gt_shift_train_only'] for r in rows],
               mk,ms=3.3,color=style.BRANCH_COLORS[br],label=br)
    a.set(xlim=(0,.85),ylim=(0,.85),xlabel='Measured novel fraction',ylabel='GT estimate (training counts)')
    a.text(0,1.04,'(a) Coverage from counts',transform=a.transAxes,fontweight='bold',fontsize=8.5)
    a.legend(loc='upper left',fontsize=6.5,handlelength=1.2)
    for j,br in enumerate(['bigram','trigram'],1):
        a=axes[j]
        for step,c in zip(STEPS,COLORS):
            x,y,sd,w,F=array(br,step);s=fit(x,y)
            xx=np.linspace(x.min(),x.max(),100)
            a.plot(xx,s['a']*xx+s['b'],color=c,lw=.85)
            a.errorbar(x,y,yerr=sd,fmt='o',ms=2.4,color=c,lw=.6,capsize=1,label=str(step))
        x,y,sd,_,_=array(br,1000,'nogram')
        a.plot(x,y,'x',ms=2.6,color='#777777',label='No-gram, 1000')
        a.axhline(0,lw=.6,color='#999999')
        a.set(xlim=(.12,.84),ylim=(-.4,11.3 if j==1 else 7.8),
              xlabel='Measured novel fraction',ylabel='Online gap (nats)')
        a.text(0,1.04,f'({chr(97+j)}) {br.capitalize()} contexts',transform=a.transAxes,fontweight='bold',fontsize=8.5)
        a.legend(loc='upper left',fontsize=6.1,handlelength=.9,labelspacing=.25,title='Step',title_fontsize=6.5)
    save(fig,'fig_coverage_linear')
    qc=dict(verified_source_files=checked,max_scalar_loss_disagreement=max(checks),online_bins=18,fit_weight='equal bin OLS; val-token sensitivity also exported',window='F=1..1000 inclusive',seed_error_bars='sample SD across seeds 42,43,44; not corpus uncertainty',gt_bin_rule='sum N1(F+1) / sum (F+1)n(F+1); train-only weights, no calibration',perpos_probability_floor=raw['meta']['tiny'],perpos_max_zeros=max(v['n_zero_prob'] for k,v in raw.items() if k!='meta'))
    (BASE/'qc.json').write_text(json.dumps(qc,indent=2));print(json.dumps(qc,indent=2))
    for s in stats:
        if s['seed']=='mean_3' and s['weighting']=='equal_bin':print(s)
if __name__=='__main__':main()
