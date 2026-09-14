#!/usr/bin/env python3
"""Plot exact epoch-2 endpoints, retaining epoch 1 as a muted reference.

Source: experiment-log.md section 55, the registered runs_scaling/*_fixed
artifacts exported by collect_epoch20_snapshot.py. Missing endpoints stay
missing; no smoothing, fixed-train proxy, or gap increment is substituted.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT/'docs/appendices/epoch20_20260914/epoch2_snapshot.json'
OUT = ROOT/'docs/figs/main/fig_s1_epoch20_epoch2_live'
POINTS = [('0p125',42),('0p1667',56),('0p25',84),('0p3333',112),
          ('0p5',168),('0p6667',224),('0p75',253),('1p0',337),
          ('1p25',421),('1p5',506),('1p75',590),('2p0',674),
          ('2p5',842),('3p0',1011),('4p0',1348),('6p0',2022),
          ('8p0',2696),('10p0',3370),('15p0',5055),('20p0',6740)]
CST = timezone(timedelta(hours=8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_config(run, proc, epb):
    summary = run.get('summary')
    cfg = summary['config'] if summary else proc['config']
    expected = dict(seed=42, epoch_batches=epb, device_batch_size=72,
                    total_batch_size=147456, n_layer=8, n_head=6, n_embd=768,
                    sequence_len=2048, vocab_size=8192, trigram_clean_table=1048576,
                    bigram_clean_table=0, table_lr_scale=128, warmup_steps=100,
                    val_batches=4)
    for key, value in expected.items():
        require(float(cfg[key]) == value, f'{run["run_id"]}: {key} differs')
    require(cfg['table_optimizer'] == 'rmsprop' and cfg['lr_schedule'] == 'warmup_constant',
            'Optimizer/schedule mismatch')
    if summary:
        require(summary['compute_dtype'] == 'bf16' and not summary['torch_compile'], 'dtype/compile')
        require(cfg['enable_trigram_ve'] and not any(cfg[k] for k in
                ('enable_bigram_ve', 'enable_unigram_ve', 'enable_fourgram_ve')), 'branch mismatch')
        require(cfg['nanogpt_ngram_injection_position'] == 'input', 'injection mismatch')
        require(cfg['nanogpt_adam_lr'] == 0.0006 and cfg['table_betas'] == [0., 0.99], 'LR/betas')
        require(cfg['data_mode'] == 'fixed' and cfg['data_seed'] == 42, 'data mode/seed')
        train, val = cfg['train_shards'], cfg['val_shards']
    else:
        require(cfg['dtype'] == 'bf16' and cfg['injection_position'] == 'input', 'dtype/injection')
        require(cfg['enable_trigram'] == '1' and cfg['enable_bigram'] == '0', 'branch mismatch')
        require(float(cfg['lr']) == 0.0006 and cfg['table_betas'] == '0.0,0.99', 'LR/betas')
        train, val = ([int(x) for x in cfg[k].split(',')] for k in ('train_shards', 'val_shards'))
    require(train == list(range(21)) and val == [21,22,23], 'data split mismatch')
    require(cfg['data_dir'].endswith('/tokenized_epoch20'), 'data pool mismatch')


def extract(snapshot):
    runs = {r['run_id']: r for r in snapshot['runs']}
    processes = {p['run_id']: p for p in snapshot['processes']}
    captured = datetime.fromisoformat(snapshot['captured_at_utc']).astimezone(CST)
    records, statuses = [], []
    require(snapshot['code_sha256'] == snapshot['local_code_sha256'], 'remote/local source mismatch')
    pool_batches = sum(snapshot['shards'][f'shard_{i:05d}.bin']['batches'] for i in range(21))
    require(pool_batches >= 6740, 'insufficient training pool')
    for label, epb in POINTS:
        rid = f's1v5_128_ep20_tri_{label}xL4_3ep_v2_fixed'
        run = runs[rid]
        proc = processes.get(rid)
        require(run.get('summary') or proc, f'{rid}: neither complete nor running')
        audit_config(run, proc, epb)
        rows = run['rows']
        steps = [r['step'] for r in rows]
        require(steps == sorted(set(steps)), f'{rid}: repeated or unordered steps')
        by_step = {r['step']: r for r in rows}
        for r in rows:
            require(all(math.isfinite(r[k]) for k in ('gap','train_loss','val_loss')), f'{rid}: nonfinite loss')
            require(math.isclose(r['gap'], r['val_loss']-r['train_loss'], abs_tol=1e-9), f'{rid}: gap definition')
            require(r['epoch'] == (r['step']-1)//epb+1, f'{rid}: epoch semantics')
        last = rows[-1]
        needed = set(range(10, last['step']+1, 10)) | {e*epb for e in (1,2,3) if e*epb <= last['step']}
        require(needed <= set(by_step), f'{rid}: missing logged steps')
        done = bool(run.get('summary'))
        if done:
            require(last['step'] == 3*epb and run['summary']['steps'] == 3*epb, f'{rid}: not complete')
            require(math.isclose(last['gap'],run['summary']['final_gap'],abs_tol=1e-9), f'{rid}: summary gap')
        history = [r for r in rows if r['step'] >= last['step']-1000]
        speed = (last['elapsed_s']-history[0]['elapsed_s'])/(last['step']-history[0]['step'])
        stamp = datetime.fromtimestamp(run['files']['train_log.jsonl']['mtime'], tz=CST)
        def eta(step):
            return (stamp+timedelta(seconds=(step-last['step'])*speed)).isoformat() if step>last['step'] else None
        statuses.append(dict(run_id=rid, multiplier=float(label.replace('p','.')), epoch_batches=epb,
                             done=done, gpu=proc['gpu'] if proc else None, step=last['step'], target=3*epb,
                             seconds_per_step=speed, epoch2_eta=eta(2*epb), completion_eta=eta(3*epb),
                             log_age_seconds=(captured-stamp).total_seconds()))
        for epoch in (1,2,3):
            row = by_step.get(epoch*epb)
            records.append(dict(multiplier=float(label.replace('p','.')), actual_multiplier=epb/337,
                epoch_batches=epb, epoch=epoch, step=epoch*epb, seed=42, status='observed' if row else 'pending',
                train_loss=row['train_loss'] if row else '', val_loss=row['val_loss'] if row else '',
                gap=row['gap'] if row else '', run_id=rid, host=snapshot['host'],
                source=run['path']+'/train_log.jsonl', source_sha256=run['files']['train_log.jsonl']['sha256']))
    return records, statuses, captured, pool_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAP)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    records, statuses, captured, pool_batches = extract(snapshot)
    epoch1 = [r for r in records if r['epoch']==1]
    epoch2 = [r for r in records if r['epoch']==2]
    n = sum(r['status']=='observed' for r in epoch2)
    pending = [r for r in epoch2 if r['status']=='pending']
    S.apply_style()
    fig, ax = plt.subplots(figsize=(11.6,5.9))
    fig.subplots_adjust(left=.085,right=.975,top=.79,bottom=.25)
    for series, color, width, size, label in [
        (epoch1,'#bbc1c8',1.1,3.7,'Epoch 1 · background'),
        (epoch2,'#236b70',2.0,5.7,'Epoch 2 · observed boundary')]:
        ax.plot([r['actual_multiplier'] for r in series],
                [r['gap'] if r['status']=='observed' else float('nan') for r in series],
                color=color, linewidth=width, marker='o', markersize=size, label=label)
    # Availability markers sit below the axis, never at gap=0.
    if pending:
        ax.scatter([r['actual_multiplier'] for r in pending],[-.16]*len(pending),
                   transform=ax.get_xaxis_transform(), marker='v', color='#a2aab3', s=26, clip_on=False)
    ax.axhline(0,color='#acb4bf',lw=.7,zorder=0)
    ax.set_xscale('log',base=2)
    ticks=[.125,.25,.5,1,1.25,1.5,1.75,2,2.5,3,4,6,8,10,15,20]
    ax.set_xticks(ticks,[f'{x:g}' for x in ticks],rotation=45,ha='right')
    ax.set_xlim(.112,23)
    ax.set_ylim(-.20,2.08)
    ax.set_ylabel('Gap = fixed-val CE − online-train CE (nats)')
    ax.set_xlabel('Epoch length / L4   ·   L4 = 337 batches',labelpad=8)
    ax.grid(axis='x',visible=False)
    ax.legend(loc='upper right',fontsize=9.5)
    for r in epoch2:
        if r['status']=='observed' and r['multiplier'] in (.1667,.5,1,2,4,6,8,10,15,20):
            ax.annotate(f"{r['gap']:.3f}",(r['actual_multiplier'],r['gap']),
                        xytext=(0,10),textcoords='offset points',ha='center',fontsize=8.8,color='#236b70')
    fig.text(.085,.93,'Epoch-length scaling after two passes',fontsize=18,weight='medium',color='#20363e')
    fig.text(.085,.875,f'{n}/20 exact epoch-2 endpoints  ·  {captured:%Y-%m-%d %H:%M} CST',fontsize=11,color='#65737c')
    pending_text = ' / '.join(f"{r['multiplier']:g}×" for r in pending)
    fig.text(.085,.064,'Trigram-only · input injection · clean R = 2²⁰ · table LR = 128× · seed 42',fontsize=9,color='#53636c')
    fig.text(.085,.026,f'One shared data stream; x uses actual batches / 337. '
             +(f'Pending epoch 2: {pending_text}; grey triangles mark availability only.' if pending else 'All 20 epoch-2 points are observed.'),
             fontsize=8.7,color='#6e7a82')
    png, svg = S.save(fig,OUT.parent,OUT.name)
    # Matplotlib path data carries trailing spaces; keep generated text diff-clean.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with OUT.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    etas=[r['completion_eta'] for r in statuses if r['completion_eta']]
    evidence=dict(captured_at=captured.isoformat(),epoch2_observed=n,total=20,
                  done=sum(r['done'] for r in statuses),train_pool_batches=pool_batches,
                  source_snapshot=str(args.snapshot.relative_to(ROOT)),
                  completion_eta=max(etas) if etas else None,runs=statuses)
    (args.snapshot.parent/'epoch2_status.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(png);print(svg);print(OUT.with_suffix('.csv'))
    print(json.dumps({k:v for k,v in evidence.items() if k!='runs'},indent=2))
    for status in statuses:
        if not status['done']: print(json.dumps(status))


if __name__ == '__main__':
    main()
