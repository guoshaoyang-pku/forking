#!/usr/bin/env python3
"""Backfill only registered LR-sweep rows from recovered queue status."""
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'docs/appendices/posttraining_lr_sweep_20260925'


def backfill_registry(text, rows):
    """Replace exact registered rows, rejecting damaged or ambiguous input."""
    if not text.startswith('# ngram-gap-lab'):
        raise ValueError('Registry header is missing; refusing to write')
    marker='## §68 · Correct-tokenizer post-training LR reduction sweep'
    if text.count(marker)!=1:
        raise ValueError('Expected exactly one LR-sweep section')
    prefix,section=text.split(marker,1)
    lines=section.splitlines(keepends=True)
    ids=[row['run_id'] for row in rows]
    if len(ids)!=24 or len(set(ids))!=24:
        raise ValueError('Expected 24 distinct registered run IDs')
    for row in rows:
        key='| '+row['run_id']+' |'
        matches=[i for i,line in enumerate(lines) if line.startswith(key)]
        if len(matches)!=1:
            raise ValueError('Expected exactly one row for '+row['run_id'])
        if row['status']=='done':
            m=row['metrics']
            status='done; online gap %.6f; val %.6f; fixed gap %.6f @1665' % (m['final_gap'],m['final_val_loss'],m['final_fixed_gap'])
        else:
            status=row['status']
        location='360-2:'+str(row['gpu']) if row['gpu'] is not None else '360-2: see launch.json GPU UUID'
        line='| %s | %s | %s | %s |' % (row['run_id'],location,row['lr'],status)
        i=matches[0]
        lines[i]=line+('\n' if lines[i].endswith('\n') else '')
    section=marker+''.join(lines)
    if all(r['status']=='done' for r in rows):
        section=section.replace('(2026-09-25, running)','(2026-09-25, done)',1)
    result=prefix+section
    assert result.split(marker,1)[0]==prefix
    assert result.count('\n')==text.count('\n')
    return result


def main():
    remote='/data/shared/guoshaoyang/ngram-gap-lab/posttraining_lr_20260925/queue/state.json'
    state=json.loads(subprocess.check_output(['ssh','360-2','cat '+remote],text=True))
    (P/'live_state.json').write_text(json.dumps(state,indent=2)+chr(10))
    rows=json.loads((P/'assignments.json').read_text())
    complete={x['run_id']:x['status'] for x in state['completed']}
    for row in rows:
        rid=row['run_id']
        if rid in complete:
            row['status']='done' if complete[rid]['qc']=='pass' else 'stalled'
            launch=P/'raw'/rid/'launch.json'
            if launch.exists():
                cfg=json.loads(launch.read_text());row['gpu_uuid']=cfg['gpu'].split(',')[0];row['actual_host']=cfg['host']
            row['metrics']=complete[rid].get('metrics')
        else:
            row['status']='planned'
            for gpu,job in state['running'].items():
                if job['run_id']==rid:row.update(status='running',gpu=int(gpu))
    (P/'assignments.json').write_text(json.dumps(rows,indent=2)+chr(10))
    path=ROOT/'docs/experiment-log.md';text=path.read_text()
    done=sum(r['status']=='done' for r in rows)
    text=backfill_registry(text,rows)
    path.write_text(text)
    path=ROOT/'docs/experiment-lines.md';text=path.read_text()
    status='共24组seed42；%s/24 done，%s running；360-2 H200，保持原 batch。' % (done,len(state['running']))
    text=re.sub(r'共24组seed42；[^。]+。',status,text)
    path.write_text(text)
    print('Registry',done,'done,',len(state['running']),'running')


if __name__=='__main__':main()
