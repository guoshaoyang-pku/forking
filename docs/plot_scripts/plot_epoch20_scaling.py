#!/usr/bin/env python3
"""Figure 7-2: exact epoch 2 and 3 raw gaps over 20 epoch lengths.

Use the frozen section-55 snapshot and its existing source/config/metric
audit. Epoch 1 remains a pale reference. All three epochs must be complete.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import v5_style as S
from plot_epoch20_epoch2_live import SNAP, extract, require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/figs/main/fig_s1_epoch20_epoch2_epoch3'


def make_figure(records, captured):
    S.apply_style()
    fig, ax = plt.subplots(figsize=(11.6, 6.25))
    fig.subplots_adjust(left=.085, right=.975, top=.79, bottom=.235)
    styles = [
        (1, '#c4c9ce', 'o', 1.0, 3.8, 'Epoch 1 · background'),
        (2, '#236b70', 'o', 1.9, 5.4, 'Epoch 2'),
        (3, '#b6663d', 's', 2.1, 5.5, 'Epoch 3'),
    ]
    for epoch, color, marker, width, size, label in styles:
        rows = [r for r in records if r['epoch'] == epoch]
        ax.plot([r['actual_multiplier'] for r in rows],
                [r['gap'] for r in rows], color=color, marker=marker,
                linewidth=width, markersize=size, markeredgecolor='white',
                markeredgewidth=.55, label=label, zorder=epoch+2)
    ax.axhline(0, color='#acb4bf', linewidth=.7, zorder=0)
    ax.set_xscale('log', base=2)
    ticks = [.125, .25, .5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 10, 15, 20]
    ax.set_xticks(ticks, [f'{x:g}' for x in ticks], rotation=45, ha='right')
    ax.set_xlim(.112, 23)
    upper = max(r['gap'] for r in records)
    lower = min(0, min(r['gap'] for r in records))
    ax.set_ylim(lower-.16, upper+.38)
    ax.set_ylabel('Gap = fixed-val CE − online-train CE (nats)')
    ax.set_xlabel('Epoch length / L4   ·   L4 = 337 batches', labelpad=9)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', fontsize=10, labelspacing=.8)
    for r in records:
        if r['epoch'] == 3 and r['multiplier'] in (.1667, 1, 4, 10, 20):
            ax.annotate(f"{r['gap']:.3f}", (r['actual_multiplier'], r['gap']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color='#a0522e')
    fig.text(.085, .93, 'Gap across epoch lengths', fontsize=19,
             weight='medium', color='#20363e')
    fig.text(.085, .875,
             f'20 lengths · exact epoch 2 & 3 boundaries · {captured:%Y-%m-%d %H:%M} CST',
             fontsize=11, color='#65737c')
    fig.text(.085, .067,
             'Trigram-only · input injection · clean R = 2²⁰ · table LR = 128× · seed 42',
             fontsize=9, color='#53636c')
    fig.text(.085, .029,
             'Raw gap at step = epoch × epoch batches; one shared data stream; x = actual batches / 337.',
             fontsize=8.7, color='#6e7a82')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAP)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    records, statuses, captured, _ = extract(snapshot)
    require(len(records) == 60 and all(r['status'] == 'observed' for r in records),
            'Figure requires all 60 exact epoch endpoints')
    require(all(r['done'] for r in statuses), 'Figure requires completed runs')
    fig = make_figure(records, captured)
    png, svg = S.save(fig, OUT.parent, OUT.name)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with OUT.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    for path in (png, svg, OUT.with_suffix('.csv')):
        print(path)


if __name__ == '__main__':
    main()
