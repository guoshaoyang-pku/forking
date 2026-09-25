#!/usr/bin/env python3
"""Matched downstream comparisons with explicit initial-gap adjustment and controls."""
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

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'docs/appendices/posttraining_samecheckpoint_20260925'
OUT = ROOT / 'docs/figs/posttraining_samecheckpoint_20260925'
ARMS = ['trainable', 'frozen0', 'frozen1', 'nogram', 'disabled']
COLORS = dict(trainable='#c4493d', frozen0='#c58a0b', frozen1='#2d6f9f',
              nogram='#686d73', disabled='#7b4fa6')
LABELS = dict(trainable='Table trainable', frozen0='Frozen from start',
              frozen1='Frozen after pass 1', nogram='Pretrained no-gram',
              disabled='Table removed; same backbone')
POOLS = [('unseen', 'Same domain; no pretraining optimization'),
         ('seen', 'Exact overlap with pretraining data')]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_write(name, records):
    with (PACK / name).open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator=chr(10))
        writer.writeheader()
        writer.writerows(records)


def load(include_disabled):
    manifests = [('reference_manifest.json', 'references')]
    if include_disabled:
        manifests.append(('source_manifest.json', 'evidence'))
    for filename, folder in manifests:
        for record in json.loads((PACK / filename).read_text()):
            rel = record.get('path') or record['run_id'] + '/' + record['file']
            path = PACK / folder / rel
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
    data, endpoints, curves = {}, [], []
    remote_refs = {(r['run_id'],r['file']):r['sha256'] for r in
                   json.loads((PACK/'reference_remote_verification.json').read_text())}
    missing_canonical = 0
    expected = sorted(set(range(10, 1666, 10)) | {333,334,666,667,999,1000,1332,1333,1665})
    for pool, _ in POOLS:
        probe_hashes = set()
        for arm in ARMS if include_disabled else ARMS[:-1]:
            prefix = 'ptsc' if arm == 'disabled' else 'ptov'
            rid = f'{prefix}_{pool}_{arm}_q4_mb18_s42_fixed'
            path = PACK / ('evidence/data/runs_fixed' if arm == 'disabled' else 'references') / rid
            for name in ['summary.json','train_log.jsonl','fixed_train_loss.jsonl','initial_probe.json']:
                canonical = ROOT / 'data/runs_fixed' / rid / name
                if canonical.exists():
                    assert canonical.read_bytes() == (path/name).read_bytes(), str(canonical)
                else:
                    missing_canonical += 1
                    if arm != 'disabled':
                        assert hashlib.sha256((path/name).read_bytes()).hexdigest() == remote_refs[rid,name]
            summary = json.loads((path / 'summary.json').read_text())
            status = json.loads((path / 'status.json').read_text())
            initial = json.loads((path / 'initial_probe.json').read_text())
            online, fixed = rows(path / 'train_log.jsonl'), rows(path / 'fixed_train_loss.jsonl')
            assert status['qc'] == 'pass' and status['returncode'] == 0, rid
            assert summary['seed'] == 42 and summary['steps'] == 1665
            cfg = summary['config']
            for key, value in dict(device_batch_size=18,total_batch_size=147456,epoch_batches=1332,
                                   nanogpt_adam_lr=.00015,table_lr_scale=128,dropout=0,
                                   lr_schedule='warmup_constant',warmup_steps=100,
                                   val_batches=16,data_mode='fixed').items():
                assert cfg[key] == value, (rid,key,cfg[key])
            assert summary['fixed_train_probe_sha256'] == initial['probe_sha256']
            probe_hashes.add(initial['probe_sha256'])
            assert [r['step'] for r in online] == expected
            assert [r['step'] for r in fixed] == expected
            for o, f in zip(online, fixed):
                assert o['epoch'] == (o['step']-1)//333 + 1
                assert abs(o['gap'] - o['val_loss'] + o['train_loss']) < 1e-6
                assert abs(f['fixed_gap'] - f['fixed_val_loss'] + f['fixed_train_loss']) < 1e-6
                assert abs(o['val_loss']-f['fixed_val_loss']) < 1e-6
                assert all(math.isfinite(r[k]) for r, keys in [(o,['train_loss','val_loss','gap']),
                           (f,['fixed_train_loss','fixed_val_loss','fixed_gap'])] for k in keys)
            assert abs(summary['final_gap']-online[-1]['gap']) < 1e-6
            assert abs(summary['final_fixed_gap']-fixed[-1]['fixed_gap']) < 1e-6
            if arm == 'disabled':
                assert summary['disable_ngram_tables'] is True
                assert all(v == 0 for r in rows(path/'table_norm.jsonl') for k,v in r.items() if k != 'step')
                sizes = json.loads((PACK/'remote_artifact_sizes.json').read_text())[rid]
                assert sizes['final_model.pt'] > 200000000 and sizes['probe_tokens.npz'] > 0
            fixed = [initial] + fixed
            data[pool,arm] = dict(online=online, fixed=fixed, run_id=rid)
            b = next(r for r in fixed if r['step'] == 333)
            endpoints.append(dict(run_id=rid,pool=pool,arm=arm,seed=42,step=1665,
                online_train=online[-1]['train_loss'],val_loss=online[-1]['val_loss'],online_gap=online[-1]['gap'],
                fixed_train=fixed[-1]['fixed_train_loss'],fixed_gap=fixed[-1]['fixed_gap'],
                initial_train=initial['fixed_train_loss'],initial_val=initial['fixed_val_loss'],
                initial_fixed_gap=initial['fixed_gap'],fixed_growth=fixed[-1]['fixed_gap']-initial['fixed_gap'],
                fixed_gap_at333=b['fixed_gap'],fixed_growth_after333=fixed[-1]['fixed_gap']-b['fixed_gap'],
                fixed_train_change_after333=fixed[-1]['fixed_train_loss']-b['fixed_train_loss'],
                fixed_val_change_after333=fixed[-1]['fixed_val_loss']-b['fixed_val_loss']))
            for metric_rows, metrics in [(online,['train_loss','val_loss','gap']),
                                         (fixed,['fixed_train_loss','fixed_val_loss','fixed_gap'])]:
                for row in metric_rows:
                    for metric in metrics:
                        curves.append(dict(run_id=rid,pool=pool,arm=arm,step=row['step'],metric=metric,value=row[metric]))
        assert len(probe_hashes) == 1
    comparisons = []
    for pool, _ in POOLS:
        table = {r['arm']:r for r in endpoints if r['pool']==pool}
        for control in ['nogram','disabled'] if include_disabled else ['nogram']:
            base = table[control]
            for arm in ['trainable','frozen0','frozen1']:
                r = table[arm]
                comparisons.append(dict(pool=pool,arm=arm,control=control,run_id=r['run_id'],
                    control_run_id=base['run_id'],seed=42,step=1665,
                    online_excess=r['online_gap']-base['online_gap'],
                    online_ratio=r['online_gap']/base['online_gap'] if base['online_gap']>0 else None,
                    fixed_excess=r['fixed_gap']-base['fixed_gap'],
                    initial_fixed_excess=r['initial_fixed_gap']-base['initial_fixed_gap'],
                    excess_fixed_growth=r['fixed_growth']-base['fixed_growth'],
                    excess_fixed_growth_after333=r['fixed_growth_after333']-base['fixed_growth_after333']))
    csv_write('endpoints.csv', endpoints)
    csv_write('comparisons.csv', comparisons)
    csv_write('curves.csv', curves)
    if missing_canonical:
        print(f'{missing_canonical} local canonical files unavailable; using packet copies verified against remote runs_fixed SHA256.')
    return data, comparisons


def axis(ax, ylabel):
    for e in range(1,6):
        ax.axvline(e,color='#b8bec4',lw=.6,ls=':',zorder=0)
    ax.set(xlim=(0,5),xticks=range(6),xlabel='Downstream replay pass',ylabel=ylabel)
    ax.grid(True,axis='y')
    ax.grid(False,axis='x')


def save(fig, name):
    for ext in ['png','pdf','svg']:
        path = OUT / (name+'.'+ext)
        fig.savefig(path,dpi=200,bbox_inches='tight')
        if ext == 'svg':
            path.write_text(chr(10).join(s.rstrip() for s in path.read_text().splitlines())+chr(10))
    plt.close(fig)


def report(comparisons):
    endpoints=list(csv.DictReader((PACK/'endpoints.csv').open()))
    by={(r['pool'],r['arm']):r for r in endpoints}
    lines=['# Post-training excess over no-gram: final readout', '',
        'All numbers below use the predeclared step 1665 (five 333-update passes), seed 42. '
        'This is all-token continued language-model training from pretrained weights. '
        'It is not response-only instruction SFT or on-policy RLVR.', '',
        'The primary architecture baseline is the separately pretrained no-gram model. '
        'The table-removed control starts from the table model checkpoint with its table zeroed and frozen; '
        'it answers a supplementary memory-removal question and has a severely degraded starting function.', '',
        '## Primary comparison', '',
        '| Pool | Trainable online gap | No-gram online gap | Ratio | Online excess | Fixed excess | Excess fixed-gap growth |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['arm']=='trainable' and r['control']=='nogram':
            a,b=by[r['pool'],'trainable'],by[r['pool'],'nogram']
            lines.append(f"| {r['pool']} | {float(a['online_gap']):.6f} | {float(b['online_gap']):.6f} | {r['online_ratio']:.2f}× | {r['online_excess']:+.6f} | {r['fixed_excess']:+.6f} | {r['excess_fixed_growth']:+.6f} |")
    lines += ['', 'Online gap is fixed validation after the update minus the current training batch before the update. '
              'Fixed gap uses a uniform 288-row training probe and 288 fixed validation rows at identical parameters. '
              'Excess fixed-gap growth is [G_table(1665) − G_table(0)] − [G_control(1665) − G_control(0)]. '
              'Thus the large trainable effect is not explained by the initial fixed-gap difference alone.', '',
              'The unseen pool is same-domain shard 21, not used for pretraining optimization (it was a pretraining validation shard). '
              'The seen pool is an exact prefix of pretraining shard 1. Each contains 23,976 rows and 49,102,848 target tokens/pass. '
              'Both use the same disjoint validation prefix of shard 22. Pool identity also changes the text, so the difference '
              'between pools does not isolate familiarity as a causal variable.', '',
              '## All arms and absolute losses', '',
              '| Pool | Arm | Initial val | Final online train | Final val | Final online gap | Initial fixed gap | Final fixed gap |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in endpoints:
        lines.append('| '+r['pool']+' | '+r['arm']+' | '+' | '.join(f"{float(r[k]):.6f}" for k in ['initial_val','online_train','val_loss','online_gap','initial_fixed_gap','fixed_gap'])+' |')
    lines += ['', 'The independently pretrained no-gram baseline has lower absolute validation loss. '
              'A larger forking effect is an overfitting result, not an overall performance benefit of the table model. '
              'The table-removed arm starts with validation loss 10.54, so it must not replace the pretrained no-gram baseline.', '',
              '## Freeze interpretation', '',
              '| Pool | Arm | Fixed gap at step 333 | Growth after step 333 | Excess growth after step 333 over no-gram |',
              '|---|---|---:|---:|---:|']
    for pool,_ in POOLS:
        b=by[pool,'nogram']
        for arm in ARMS[:4]:
            r=by[pool,arm]
            lines.append(f"| {pool} | {arm} | {float(r['fixed_gap_at333']):.6f} | {float(r['fixed_growth_after333']):+.6f} | {float(r['fixed_growth_after333'])-float(b['fixed_growth_after333']):+.6f} |")
    lines += ['', 'The large trainable growth occurs mainly after the first downstream pass. '
              'Freezing after pass 1 does not stop all gap growth, but on unseen data its later growth is slightly below no-gram; '
              'on seen data the excess later growth is only about 0.056. '
              'Freezing from the start on seen data retains an endpoint excess that is almost entirely inherited from initialization '
              '(excess total fixed-gap growth is only about 0.006). '
              'These results do not support a large backbone-only excess under the frozen-table protocol.', '',
              '## Evidence scope and reproduction', '',
              'There is one pretrained seed/data realization per architecture, two specified downstream pools, and no independent '
              'replication of either pool. The effects above are large measured contrasts, not a multi-seed significance claim. '
              'Changing only the downstream seed with loaded weights, fixed stream and dropout 0 does not supply independent trials. '
              'No p-value or confidence interval is inferred from correlated logged steps.', '',
              'Backbone LR 0.00015, table LR ratio 128 (absolute 0.0192), warmup 100 then constant; input trigram clean R=2^20; '
              'A100 microbatch 18 × accumulation 4 preserves 147,456 target tokens/update. '
              'Both bases trained a 20× data pool for THREE passes. The result does not repair that base protocol into a no-replay experiment.', '',
              'Run IDs and all contrasts: [comparisons.csv](comparisons.csv); raw endpoint losses: [endpoints.csv](endpoints.csv); '
              'full curves: [curves.csv](curves.csv). Small source logs are archived in references/ and evidence/, with SHA256 manifests; '
              'source logs are remote data/runs_fixed/<run_id>/ artifacts. The local data/ link targets an external SSD '
              'that became unavailable during collection; packet hashes were checked directly against remote originals. '
              'The collector refreshes the local canonical cache when the drive is mounted. Final checkpoints stay on a100_perm.', '',
              'Figures: [primary excess](../../figs/posttraining_samecheckpoint_20260925/fig_primary_excess.png), '
              '[freeze contrasts](../../figs/posttraining_samecheckpoint_20260925/fig_freeze_excess.png), '
              '[memory removal](../../figs/posttraining_samecheckpoint_20260925/fig_memory_removal_control.png). '
              'Each is also available as PDF and SVG. Raw curves; no smoothing.', '',
              'From the repository root: python3 docs/plot_scripts/collect_posttraining_samecheckpoint.py; '
              '.venv/bin/python docs/plot_scripts/plot_posttraining_excess.py. '
              'Plotting validates source hashes, available canonical-byte agreement (otherwise verified remote hashes), registered config/cadence, finite loss arithmetic, '
              'probe identities, zero removed-table norms, QC status and remote final-artifact presence.', '']
    (PACK/'results.md').write_text(chr(10).join(lines))


def excess(data, pool, arm, control, mode):
    rows_a, rows_b = data[pool,arm][mode], data[pool,control][mode]
    assert [r['step'] for r in rows_a] == [r['step'] for r in rows_b]
    key = 'gap' if mode == 'online' else 'fixed_gap'
    ys = [a[key]-b[key] for a,b in zip(rows_a,rows_b)]
    return [r['step']/333 for r in rows_a], ys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--references-only',action='store_true')
    args=parser.parse_args()
    data, comparisons = load(not args.references_only)
    OUT.mkdir(parents=True,exist_ok=True)
    apply_style()
    fig, axes = plt.subplots(2,3,figsize=(12.4,6.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.15,top=.82,hspace=.40,wspace=.30)
    fig.suptitle('Post-training forking exceeds the pretrained no-gram baseline',y=.99,fontsize=14)
    handles=[Line2D([],[],color=COLORS[a],label=LABELS[a],lw=1.7) for a in ['trainable','nogram']]
    handles += [Line2D([],[],color='#333',label='Online train',lw=1.3),Line2D([],[],color='#333',ls='--',label='Fixed validation')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=4)
    for i,(pool,title) in enumerate(POOLS):
        for arm in ['trainable','nogram']:
            d=data[pool,arm];o=d['online'];x=[r['step']/333 for r in o]
            axes[i,0].plot(x,[r['train_loss'] for r in o],color=COLORS[arm],lw=1.2)
            axes[i,0].plot(x,[r['val_loss'] for r in o],color=COLORS[arm],lw=1.4,ls='--')
            axes[i,1].plot(x,[r['gap'] for r in o],color=COLORS[arm],lw=1.2)
        xs,ys=excess(data,pool,'trainable','nogram','fixed')
        adjusted=[y-ys[0] for y in ys]
        axes[i,2].plot(xs,adjusted,color=COLORS['trainable'],lw=1.7)
        axes[i,2].axhline(0,color='#888',lw=.7)
        axes[i,2].annotate(f'+{adjusted[-1]:.3f}',xy=(5,adjusted[-1]),xytext=(-7,-19),textcoords='offset points',ha='right',color=COLORS['trainable'],fontsize=12)
        for j,label in enumerate(['Loss (nats/token)','Online gap (nats/token)','Excess fixed-gap growth (nats/token)']):
            axis(axes[i,j],label)
        axes[i,0].set_title(title,fontsize=10)
        axes[i,1].set_title('Raw gap: validation − current batch',fontsize=10)
        axes[i,2].set_title('[G(table) − G(no-gram)] − initial difference',fontsize=10)
    fig.text(.5,.036,'Seed 42; 333 updates/pass; 49.10M target tokens/pass; backbone LR 1.5e−4; table LR 128×.'+chr(10)+
             'Right: fixed train/validation probes at identical parameters; 288 rows each. Single base/data realization; no smoothing.',ha='center',fontsize=9)
    save(fig,'fig_primary_excess')

    fig,axes=plt.subplots(3,2,figsize=(10.7,8.3),sharex=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.13,top=.84,hspace=.27,wspace=.27)
    fig.suptitle('Excess over no-gram depends strongly on whether table writes continue',y=.99,fontsize=13)
    fig.legend(handles=[Line2D([],[],color=COLORS[a],label=LABELS[a]) for a in ARMS[:3]],loc='upper center',bbox_to_anchor=(.5,.955),ncol=3)
    for j,(pool,title) in enumerate(POOLS):
        axes[0,j].set_title(title,fontsize=10)
        for arm in ARMS[:3]:
            for i,mode in [(0,'online'),(1,'fixed'),(2,'fixed')]:
                x,y=excess(data,pool,arm,'nogram',mode)
                if i==2:y=[v-y[0] for v in y]
                axes[i,j].plot(x,y,color=COLORS[arm],lw=1.3)
        for i,label in enumerate(['Online gap excess','Fixed gap excess','Excess fixed-gap growth']):
            axis(axes[i,j],label+' (nats/token)')
            axes[i,j].axhline(0,color='#777',lw=.7)
    fig.text(.5,.028,'Every curve subtracts the separately pretrained no-gram on the same downstream pool.'+chr(10)+
             'Bottom additionally subtracts the step-0 gap difference. Frozen after pass 1: writes stop before update 334. Seed 42.',ha='center',fontsize=9)
    save(fig,'fig_freeze_excess')

    if not args.references_only:
        fig,axes=plt.subplots(2,3,figsize=(12.4,6.5))
        fig.subplots_adjust(left=.065,right=.985,bottom=.15,top=.83,hspace=.4,wspace=.28)
        fig.suptitle('Removing a co-adapted table changes the starting function',y=.99,fontsize=14)
        fig.legend(handles=[Line2D([],[],color=COLORS[a],label=LABELS[a]) for a in ['trainable','nogram','disabled']],loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
        for i,(pool,title) in enumerate(POOLS):
            for arm in ['trainable','nogram','disabled']:
                d=data[pool,arm];f=d['fixed'];o=d['online'];x=[r['step']/333 for r in f]
                axes[i,0].plot(x,[r['fixed_train_loss'] for r in f],color=COLORS[arm],lw=1.2)
                axes[i,0].plot(x,[r['fixed_val_loss'] for r in f],color=COLORS[arm],ls='--',lw=1.3)
                axes[i,1].plot([r['step']/333 for r in o],[r['gap'] for r in o],color=COLORS[arm],lw=1.2)
                axes[i,2].plot(x,[r['fixed_gap']-f[0]['fixed_gap'] for r in f],color=COLORS[arm],lw=1.3)
            for j,label in enumerate(['Fixed-probe loss (nats/token)','Online gap (nats/token)','Fixed-gap growth from step 0']):
                axis(axes[i,j],label)
            axes[i,0].set_title(title,fontsize=10)
            axes[i,1].set_title('Raw online gap',fontsize=10)
            axes[i,2].set_title('Initial gap subtracted within each arm',fontsize=10)
        fig.text(.5,.03,'Left: fixed train solid; fixed validation dashed. Full initial loss range retained. Seed 42, 1665 updates.'+chr(10)+
                 'Table-removed arm starts with identical backbone weights to the table arms; pretrained no-gram is a distinct baseline.',ha='center',fontsize=9)
        save(fig,'fig_memory_removal_control')
        report(comparisons)
    print(json.dumps([r for r in comparisons if r['arm']=='trainable'],indent=2))


if __name__=='__main__':
    main()
