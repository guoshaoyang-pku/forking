#!/usr/bin/env python3
"""Write an evidence-linked status/results note from validated endpoints."""
import csv
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'docs/appendices/posttraining_lr_sweep_20260925'


def main():
    rows=list(csv.DictReader((P/'endpoints.csv').open()))
    index={(r['source'],r['arm'],r['lr_tag']):r for r in rows}
    count=sum(r['lr_tag']!='base' for r in rows)
    text=['# Correct-tokenizer post-training LR sweep','',f'Status: {count}/24 new runs passed artifact and metric QC. Seed 42 only. Original-LR comparison uses the matching seed-42 runs.',
          '', '## Protocol', '',
          'Four sources (ID, Magicoder, OpenR1, static RLVR-MATH) × three arms (trainable trigram table, frozen table, no-gram) × LR 1/4 or 1/10. Backbone LR 0.00015 / 0.00006; table/backbone ratio 128, giving trainable-table LR 0.0192 / 0.00768. Same pretrained checkpoints, correct-tokenizer packed data, fresh optimizer, 100-step warmup then constant, batch 72 × 2048, 1665 optimizer steps = five passes over 333 batches. This is continued all-token LM training, not response-only SFT or on-policy RL.',
          '', 'Each downstream pass has 49,102,848 target tokens; total 245,514,240. Validation has 589,824 target tokens. RLVR packing repeats the small source internally, so five packed passes correspond to approximately 42.7 source cycles. The known one-record train/val boundary crossing in Magicoder/OpenR1 is retained for matched historical comparison; no strict example-disjoint claim is made.',
          '', 'Primary online gap = fixed validation loss after update minus the current 72-example optimization batch loss before update. Diagnostic fixed gap = same-checkpoint validation loss minus 288 uniformly spaced training chunks. The probe is evaluated every 10 steps and logged boundary events. No interpolation or smoothing. An old-LR fixed endpoint is used only after evaluating the exact same uniform probe and checking its hash; the earlier 33-batch audit is not substituted.',
          '', '## Endpoint results (step 1665, nats/token)', '',
          '| Source | Arm | LR fraction | Train | Validation | Online gap | Fixed gap |',
          '|---|---|---:|---:|---:|---:|---:|']
    sources=[('id','ID'),('magicoder','Magicoder'),('openr1','OpenR1'),('rlvr','RLVR-MATH')]
    for source,label in sources:
        for arm in ['trainable','frozen','nogram']:
            for tag,fraction in [('base','1'),('q4','1/4'),('q10','1/10')]:
                r=index.get((source,arm,tag))
                if not r:continue
                fixed=f"{float(r['fixed_gap']):.4f}" if r.get('fixed_gap') else 'pending uniform4 audit'
                text.append(f"| {label} | {arm} | {fraction} | {float(r['train_loss']):.4f} | {float(r['val_loss']):.4f} | {float(r['online_gap']):.4f} | {fixed} |")
    text += ['', '## Changes relative to original LR', '',
             'Negative validation-loss change is an improvement. Negative gap change alone can also reflect slower training. Comparisons hold total token budget fixed, not training-loss convergence.', '',
             '| Source | Arm | LR fraction | Change in train | Change in validation | Change in online gap | Gap / original |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['lr_tag']=='base':continue
        b=index[r['source'],r['arm'],'base'];delta={k:float(r[k])-float(b[k]) for k in ['train_loss','val_loss','online_gap']}
        ratio=float(r['online_gap'])/float(b['online_gap'])
        text.append(f"| {r['source']} | {r['arm']} | {r['lr_tag']} | {delta['train_loss']:+.4f} | {delta['val_loss']:+.4f} | {delta['online_gap']:+.4f} | {ratio:.3f} |")
    text += ['', '## Excess gap over matched no-gram', '',
             '| Source | LR fraction | Trainable minus no-gram | Frozen minus no-gram |',
             '|---|---:|---:|---:|']
    for source,label in sources:
        for tag,fraction in [('base','1'),('q4','1/4'),('q10','1/10')]:
            if not all((source,a,tag) in index for a in ['trainable','frozen','nogram']):continue
            g={a:float(index[source,a,tag]['online_gap']) for a in ['trainable','frozen','nogram']}
            text.append(f"| {label} | {fraction} | {g['trainable']-g['nogram']:+.4f} | {g['frozen']-g['nogram']:+.4f} |")
    text += ['', '## Interpretation limits', '',
             'The LR reduction tests sensitivity to the size of optimization updates. It does not measure pretraining/downstream n-gram overlap, uniquely identify distribution shift, or establish that familiar n-grams suffice to induce a gap. The table and no-gram models start from different pretrained checkpoints with different baseline validation loss. All new jobs use the same H200 host/code/software; historical LR=1 ran on ophis with another torch version. Numerical and single-seed limits remain.',
             '', 'A100 batch72 no-gram smoke failed with OOM before the first update; it is retained as engineering evidence and excluded from scientific endpoints. Formal jobs preserve batch72 and use idle 360-2 H200s. Inputs live in /data/home/guoshaoyang/ngram-gap-lab/data; new outputs in /data/shared/guoshaoyang/ngram-gap-lab/posttraining_lr_20260925/data/runs_fixed. The shared location avoids the home-directory storage quota. No old artifacts are deleted.',
             '', '## Artifacts and reproduction', '',
             '- `assignments.json`: run IDs and settings; actual GPU/host/commands are in raw launch.json files.',
             '- `inputs_manifest.json`, `*-input-verification.json`, `code_hashes.json`: immutable input and code provenance.',
             '- `artifact_manifest.json`: remote paths, byte counts and SHA256, including final checkpoint and saved probe tokens. Large checkpoint files remain remote.',
             '- `raw/<run_id>/`: verified logs, summary, launch and completion status.',
             '- `endpoints.csv`, `curves.csv`: regenerated from verified sources.',
             '- `baseline_uniform4.json`: original-LR endpoints and initial-checkpoint diagnostic on exactly the same fixed probe, when available.',
             '- Figures: `docs/figs/posttraining_lr_sweep_20260925/`, including `all_source_lr_curves.pdf` and `lr_endpoint_overview.png`.',
             '', 'From repository root:', '', '```sh',
             'python3 docs/plot_scripts/collect_posttraining_lr_sweep_20260925.py',
             '.venv/bin/python docs/plot_scripts/plot_posttraining_lr_sweep_20260925.py'+(' --partial' if count<24 else ''),
             'python3 docs/plot_scripts/report_posttraining_lr_sweep_20260925.py', '```', '']
    (P/'README.md').write_text(chr(10).join(text))
    print(f'Report: {count}/24 new runs')


if __name__=='__main__':main()
