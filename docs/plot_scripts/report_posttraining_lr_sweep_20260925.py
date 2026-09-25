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
    baseline_path=P/'baseline_uniform4.json'
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text())
        text += ['', '## Initial-checkpoint fixed gap and growth', '',
                 'These use the identical four uniform training-probe batches and fixed validation set. Initial gap is a train/val sample-difficulty offset before downstream optimization. Growth is final fixed gap minus this initial value, a separate diagnostic from the primary online gap.', '',
                 '| Source | Arm | Initial fixed gap | Final growth at LR 1 | At LR 1/4 | At LR 1/10 |',
                 '|---|---|---:|---:|---:|---:|']
        for source,label in sources:
            legacy={'openr1':'openr1_10shards','rlvr':'rlvr_math'}.get(source,source)
            for arm in ['trainable','frozen','nogram']:
                initial=next(r['base_fixed_gap'] for r in baseline['runs'] if r['source']==legacy and r['arm']==arm)
                growth=[]
                for tag in ['base','q4','q10']:
                    row=index.get((source,arm,tag),{})
                    growth.append('%.4f' % (float(row['fixed_gap'])-initial) if row.get('fixed_gap') else 'pending')
                text.append('| %s | %s | %.4f | %s |' % (label,arm,initial,' | '.join(growth)))
    domain_path=P/'domain_overlap.json'
    if domain_path.exists():
        domain=json.loads(domain_path.read_text())
        text += ['', '## Domain-distribution diagnostic', '',
                 'Token histograms use the entire unique pretraining pool (993,853,440 target tokens) and each full downstream packed train pool. JS divergence is in nats (0 means identical unigram distributions). Exact n-gram coverage uses the same 4,096 uniformly sampled pretraining rows and 288 uniformly sampled downstream rows per source; it is a token-occurrence-weighted sampled lower bound on full-pretraining coverage, not a hash-table hit rate.', '',
                 '| Source | Unigram JS from pretraining | Sampled bigram coverage | Sampled trigram coverage |',
                 '|---|---:|---:|---:|']
        for row in domain['sources']:
            d=row['train']
            text.append(f"| {row['source']} | {d['unigram_js_from_pretrain']:.6f} | {100*d['sampled_pretrain_2gram_coverage']:.2f}% | {100*d['sampled_pretrain_3gram_coverage']:.2f}% |")
        text += ['', 'ID is much closer in unigram distribution; OOD nevertheless reuses almost exclusively previously seen vocabulary and many sampled contexts. This measures marginal distribution and sampled identity overlap, not conditional continuation similarity, causal explanation of the gap ranking, or suitability of the pretraining corpus. Artifacts: domain_overlap.json and domain_overlap_provenance.json; CPU-only script code/tools/audit_posttraining_domain_overlap.py.', '']
    coverage_path=P/'table_row_coverage.json'
    if coverage_path.exists():
        coverage=json.loads(coverage_path.read_text())
        text += ['', '## Hash-table address coverage', '',
                 'Using the actual input-trigram hash, the first %s pretraining rows visit all %s table addresses (within-row contexts only). A previously unseen exact context therefore need not address an unused parameter row: collisions share rows. This confirms address coverage, not exact-context familiarity, useful learned content, or a nonzero net gradient for every row. Source: table_row_coverage.json; script code/tools/audit_posttraining_table_row_coverage.py.' % (coverage['rows_scanned'],coverage['table_rows']), '']
    (P/'README.md').write_text(chr(10).join(text))
    print(f'Report: {count}/24 new runs')


if __name__=='__main__':main()
