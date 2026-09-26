#!/usr/bin/env python3
"""Reproduce Section 4's matched frequency fits and verify source records."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/appendices/coverage_takeover_20260925'
OUT = ROOT / 'docs/appendices/section4_close_20260926'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(x, y):
    assert min(x) > 0 and min(y) > 0
    lx, ly = np.log(x), np.log(y)
    design = np.column_stack([np.ones(len(lx)), lx])
    intercept, slope = np.linalg.lstsq(design, ly, rcond=None)[0]
    residual = ly - design @ np.array([intercept, slope])
    return dict(prefactor=float(np.exp(intercept)), exponent=float(-slope),
                r2_log=float(1 - residual @ residual / np.sum((ly - ly.mean()) ** 2)))


def main():
    inputs = json.loads((SOURCE / 'inputs/online_selected.json').read_text())
    source_rows = list(csv.DictReader((SOURCE / 'online_points.csv').open()))
    runs = [f'fig1v5_rp_both_s{seed}_fixed' for seed in (42, 43, 44)]
    configs = {}
    for seed, run in zip((42, 43, 44), runs):
        config = inputs[run]['summary']['config']
        expected = dict(seed=seed, nanogpt_ngram_injection_position='input',
                        bigram_clean_table=2**20, trigram_clean_table=2**20,
                        table_optimizer='rmsprop', table_betas=[0., .99],
                        table_lr_scale=128., nanogpt_adam_lr=.0006,
                        lr_schedule='warmup_constant', warmup_steps=100,
                        train_shards=[1], data_mode='fixed', data_seed=42,
                        val_batches=4, val_interval_steps=10)
        assert all(config[k] == v for k, v in expected.items()), run
        configs[run] = expected
    values, fits = [], {}
    for branch in ('bigram', 'trigram'):
        bins = {}
        for row in source_rows:
            if row['arm'] != 'both' or row['step'] != '1000' or row['branch'] != branch:
                continue
            lo, hi = int(row['F_lo']), int(row['F_hi'])
            assert 1 <= lo <= hi <= 1000
            assert min(int(row['train_n']), int(row['val_n'])) >= 300
            run = row['run_id']
            assert run in runs
            native_name = str(lo) if lo == hi else f'{lo}-{hi}'
            if hi == 1000:
                native_name = f'{lo}-1k'
            record = next(r for r in inputs[run]['rows'] if r['step'] == 1000)
            tr, va = (record[side][branch][native_name] for side in ('train', 'val'))
            assert abs(float(row['gap']) - (va['mean_loss'] - tr['mean_loss'])) < 1e-10
            bins.setdefault((lo, hi), []).append(row)
        assert len(bins) == 18
        branch_values = []
        for (lo, hi), rows in sorted(bins.items()):
            assert sorted(int(r['seed']) for r in rows) == [42, 43, 44]
            f = float(rows[0]['f'])
            assert all(float(r['f']) == f for r in rows)
            gaps = [float(r['gap']) for r in rows]
            value = dict(branch=branch, F_lo=lo, F_hi=hi, F=np.sqrt(lo*hi),
                         novel_fraction=f, online_gap_mean=float(np.mean(gaps)),
                         online_gap_sd=float(np.std(gaps, ddof=1)))
            branch_values.append(value)
        values.extend(branch_values)
        x = [v['F'] for v in branch_values]
        fits[branch] = dict(gap=fit(x, [v['online_gap_mean'] for v in branch_values]),
                            novelty=fit(x, [v['novel_fraction'] for v in branch_values]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'fit_values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    manifest = dict(
        step=1000, seeds=[42, 43, 44], run_ids=runs,
        metric='post-update fixed-val CE minus pre-update current-batch CE at the same logged step',
        protocol='All 18 native F=1..1000 bins; geometric midpoints; arithmetic mean gap across seeds; equal-bin OLS in log-log coordinates. No bins removed.',
        novelty='Direct counts in the same fixed validation pool; shared by all seeds.',
        inputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
            SOURCE / 'online_points.csv', SOURCE / 'source_manifest.json',
            SOURCE / 'inputs/online_selected.json', SOURCE / 'corpus_bins.csv']},
        verified_config=configs, fits=fits,
        plotted_values_sha256=digest(OUT / 'fit_values.csv'))
    (OUT / 'powerlaw_fits.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if '--update-paper-manifest' in sys.argv:
        paper_figure = ROOT / 'paper/figures/fig03a_frequency_coverage.pdf'
        lab_figure = ROOT / 'docs/figs/coverage_20260925/fig03a_frequency_coverage.pdf'
        assert digest(paper_figure) == digest(lab_figure)
        manifest['figure_sha256'] = digest(paper_figure)
        (ROOT / 'paper/supplement/frequency_powerlaw_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n')
        crosswalk = ROOT / 'paper/supplement/controlled_setting_manifest.json'
        data = json.loads(crosswalk.read_text())
        frequency = data['panels']['frequency']
        frequency.update(
            figure_sha256=digest(paper_figure),
            metric='online gap by native corpus-frequency bin; three-seed arithmetic means; no error bars',
            fit='equal-bin log-log OLS of gap versus F; 18 native F=1..1000 bins represented by geometric midpoints, at least 300 tokens per train/validation bin; step 1000 three-seed means',
            fit_values='docs/appendices/section4_close_20260926/powerlaw_fits.json',
            fit_values_sha256=digest(OUT / 'powerlaw_fits.json'),
            paper_fit_manifest='supplement/frequency_powerlaw_manifest.json')
        crosswalk.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(fits, indent=2))


if __name__ == '__main__':
    main()
