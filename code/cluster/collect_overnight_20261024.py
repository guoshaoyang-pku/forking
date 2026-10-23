#!/usr/bin/env python3
"""Collect terminal overnight runs and registered controls; never launch or retry.

Run overnight_status.py first. Evidence and conservative paired statistics go to
outputs/takeover_20261024/overnight_results/snapshots/<UTC timestamp>/review.json.
Only fixed-val comparisons with matching completion/configuration checks pass.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/takeover_20261024'
QUEUE = '/home/cfs/guoshaoyang/nglab_queue_overnight_20261024'
HOSTS = {'perm': ('a100_perm', '/data2/guoshaoyang/ngram-gap-lab-v27'),
         **{h: (h, '/home/tione/guoshaoyang/ngram-gap-lab')
            for h in ('a100_t1', 'a100_t1_2', 'a100_t1_4')}}
FILES = ('train_log.jsonl', 'summary.json', 'train_tail.txt')
FILE_LIMIT = 16 * 1024**2
HOST_LIMIT = 96 * 1024**2
CONFIG_KEYS = ('seed data_seed data_dir data_mode train_shards val_shards train_max_tokens '
               'total_batch_size val_batches val_device_batch_size val_interval_steps '
               'vocab_size n_layer n_head n_embd sequence_len dropout bias '
               'nanogpt_adam_lr adam_betas weight_decay lr_schedule warmup_steps '
               'warmup_start_lr_mult grad_clip norm_type pos_type untied_head ffn_type '
               'max_steps token_dtype').split()
SUMMARY_KEYS = ('seed train_epochs train_max_tokens train_pool_content_tokens compute_dtype '
                'torch_compile').split()
TABLE_KEYS = ('ngram_extra_orders ngram_assign_file ngram_clean_table ngram_inject_layers '
              'ngram_heads table_optimizer table_lr_scale table_betas table_grad_clip').split()
REMOTE = r'''
import base64,hashlib,json,shlex,sys,time
from pathlib import Path
p=json.loads(sys.argv[1]);root=Path(p['root']);q=Path(p['queue']);used=0;out=[]
for request in p['requests']:
    rid=request['run_id'];d=root/'data/runs_fixed'/rid
    rec={'host':p['host'],'run_id':rid,'files':{},'errors':[],'rc':None}
    try:
        if request.get('key'):
            claim=q/'claims'/request['key']
            if not (claim/'rc').is_file():
                rec['errors'].append('claim_not_terminal');out.append(rec);continue
            meta=json.loads((claim/'claim.json').read_text())
            env=dict(x.split('=',1) for x in shlex.split(meta['line'].split('|',1)[0]))
            if meta.get('host')!=p['host'] or env.get('FLEET_RUN_ID')!=rid:
                rec['errors'].append('claim_identity_mismatch');out.append(rec);continue
            rec['rc']=(claim/'rc').read_text().strip();rec['claim']=meta
        elif not (d/'summary.json').is_file():
            rec['errors'].append('control_not_complete');out.append(rec);continue
        tail=b''
        if (d/'train.log').is_file():
            with (d/'train.log').open('rb') as f:
                f.seek(max(0,f.seek(0,2)-16384));tail=f.read(16384)
        rec['done_line']=b'[nglab] DONE.' in tail
        if not request.get('key') and not rec['done_line']:
            rec['errors'].append('control_missing_DONE');out.append(rec);continue
        for name in p['files']:
            f=d/name
            if name=='train_tail.txt':data=tail
            elif not f.is_file():rec['errors'].append('missing:'+name);continue
            elif f.stat().st_size>p['file_limit']:
                rec['errors'].append('file_limit:'+name);continue
            else:
                before=f.stat()
                with f.open('rb') as stream:data=stream.read(p['file_limit']+1)
                after=f.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    rec['errors'].append('file_changed:'+name);continue
            if len(data)>p['file_limit'] or used+len(data)>p['host_limit']:
                rec['errors'].append('byte_limit:'+name);continue
            used+=len(data)
            rec['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                'base64':base64.b64encode(data).decode()}
    except (OSError,ValueError,KeyError) as error:rec['errors'].append(type(error).__name__+':'+str(error))
    out.append(rec)
print(json.dumps({'host':p['host'],'observed_epoch':time.time(),'runs':out}))
'''


def safe_id(run_id):
    return isinstance(run_id, str) and re.fullmatch(r'v27_[A-Za-z0-9_.]+_fixed', run_id)


def select_requests(manifest, status):
    jobs = {(j['host'], j['run_id']): j for j in manifest['jobs']}
    observed = {(h['host'], j['run_id']): j for h in status['hosts'] for j in h.get('jobs', [])}
    selected, edges = {}, set()
    for identity, job in jobs.items():
        row = observed.get(identity, {})
        if row.get('rc') is None:
            continue
        selected[identity] = {'run_id':identity[1], 'key':row.get('key')}
        if not row.get('key'):
            raise ValueError('Terminal status lacks claim key: '+identity[1])
        for ref in job.get('control_refs', []):
            other = (ref['host'], ref['run_id'])
            control = observed.get(other, {})
            selected[other] = {'run_id':other[1], 'key':control.get('key') if other in jobs else None}
            edges.add(tuple(sorted((identity, other))))
    for (host, rid), request in selected.items():
        if host not in HOSTS:
            request['blocked'] = 'host_not_allowlisted_no_access'
            continue
        if not safe_id(rid):raise ValueError('Invalid run identity')
        if request['key'] is not None and not re.fullmatch(r'[a-f0-9]{12}', request['key']):
            raise ValueError('Invalid claim key')
        if (host, rid) in jobs and not request['key']:
            request['blocked'] = 'registered_control_missing_status'
    return selected, sorted(edges)


def persist_run(record, snapshot):
    host, rid = record['host'], record['run_id']
    if host not in HOSTS or not safe_id(rid):raise ValueError('Invalid returned identity')
    directory = snapshot / host / rid
    directory.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, item in record.pop('files').items():
        if name not in FILES or item['bytes'] > FILE_LIMIT:raise ValueError('Unexpected/oversized file')
        data = base64.b64decode(item['base64'], validate=True)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Transfer checksum mismatch: '+name)
        path = directory / name
        path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Local checksum mismatch: '+name)
        receipts[name] = {'bytes':len(data), 'sha256':item['sha256']}
    record['files'] = receipts
    (directory / 'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def fetch_host(host, requests, snapshot):
    alias, root = HOSTS[host]
    payload = dict(host=host, root=root, queue=QUEUE, requests=requests, files=FILES,
                   file_limit=FILE_LIMIT, host_limit=HOST_LIMIT)
    cmd = 'python3 - '+shlex.quote(json.dumps(payload))
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',alias,cmd],
                            input=REMOTE, text=True, capture_output=True, timeout=90)
    if result.returncode:raise RuntimeError('SSH/remote collection failed: '+host)
    response = json.loads(result.stdout)
    expected = {r['run_id'] for r in requests}
    if response['host'] != host or {r['run_id'] for r in response['runs']} != expected:
        raise ValueError('Remote result identities mismatch')
    return [persist_run(r, snapshot) for r in response['runs']]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_run(record, snapshot, expected_job=None):
    result = dict(record, eligible=False)
    issues = list(record.get('errors', []))
    if record.get('rc') not in (None, '0'):issues.append('claim_failed_rc:'+str(record['rc']))
    if not record.get('done_line'):issues.append('missing_DONE')
    directory = snapshot / record['host'] / record['run_id']
    try:
        summary = json.loads((directory/'summary.json').read_text())
        rows = [json.loads(line) for line in (directory/'train_log.jsonl').read_text().splitlines() if line.strip()]
        config = summary['config']; target = config['max_steps']
        if expected_job is not None:
            epochs=expected_job['epochs'];tokens=expected_job['nominal_tokens']//epochs
            checks={'seed':expected_job['seed'],'train_epochs':epochs,'train_max_tokens':tokens}
            for key,value in checks.items():
                if summary.get(key)!=value:issues.append('registered_mismatch:'+key)
            for key,value in {'qk_norm':expected_job['qkn'],'nanogpt_adam_lr':expected_job['lr'],
                              'train_max_tokens':tokens}.items():
                if config.get(key)!=value:issues.append('registered_config_mismatch:'+key)
            if expected_job['arm']=='exact' and config.get('table_lr_scale')!=expected_job['tls']:
                issues.append('registered_config_mismatch:table_lr_scale')
            seq=config.get('sequence_len',0);batch=config.get('device_batch_size',0)
            total=config.get('total_batch_size',0)
            if not all(isinstance(v,int) and v>0 for v in (seq,batch,total)):
                issues.append('unknown_registered_step_calculation')
            else:
                pool=tokens//seq//batch*batch*seq
                expected_steps=-(-epochs*pool//total)
                if summary.get('train_pool_content_tokens')!=pool:issues.append('registered_pool_mismatch')
                if summary.get('steps')!=expected_steps or target!=expected_steps:
                    issues.append('registered_step_budget_mismatch')
        if not isinstance(target, int) or target <= 0:issues.append('invalid_target_steps')
        if summary.get('run_id') != record['run_id']:issues.append('run_id_mismatch')
        if not rows or rows[-1].get('step') != summary.get('steps') or summary.get('steps') != target:
            issues.append('incomplete_or_wrong_step')
        steps = [r.get('step') for r in rows]
        if any(not isinstance(s,int) for s in steps) or steps != sorted(set(steps)):
            issues.append('duplicate_or_unordered_steps')
        if any(not finite(r.get(k)) for r in rows for k in ('train_loss','val_loss')):
            issues.append('nonfinite_or_missing_loss')
        for key in ('train_loss', 'val_loss'):
            value = summary.get('final_'+key)
            if not finite(value):issues.append('nonfinite_summary:'+key)
            elif rows and (not finite(rows[-1].get(key)) or not math.isclose(value,rows[-1][key],rel_tol=0,abs_tol=1e-9)):
                issues.append('summary_log_mismatch:'+key)
        for key in ('seed','train_max_tokens'):
            if summary.get(key) != config.get(key):issues.append('summary_config_mismatch:'+key)
        seed_tag=re.search(r'_s(\d+)_e',record['run_id'])
        if not seed_tag or summary.get('seed') != int(seed_tag[1]):issues.append('run_id_seed_mismatch')
        arm = 'exact' if '_exact_' in record['run_id'] else 'nogram' if '_nogram_' in record['run_id'] else None
        if arm is None:issues.append('unknown_arm')
        if summary.get('ngram_orders') != ([2,3,4,5] if arm=='exact' else []):issues.append('arm_config_mismatch')
        if config.get('ngram_extra_orders') != summary.get('ngram_orders'):issues.append('summary_config_order_mismatch')
        result.update(_summary=summary, _rows={r['step']:r for r in rows}, arm=arm,
                      final_step=summary.get('steps'),
                      final_train=summary.get('final_train_loss') if finite(summary.get('final_train_loss')) else None,
                      final_val=summary.get('final_val_loss') if finite(summary.get('final_val_loss')) else None)
    except (OSError,ValueError,KeyError,TypeError) as error:issues.append('unreadable_evidence:'+type(error).__name__)
    result['issues'] = sorted(set(issues));result['eligible'] = not issues
    return result


def compare_runs(left, right):
    report = {'left': [left['host'],left['run_id']], 'right':[right['host'],right['run_id']],
              'state':'manual_review', 'scientific_review_required':True, 'issues':[]}
    if not left['eligible'] or not right['eligible']:
        report['issues'].append('both_runs_must_be_complete_finite_and_verified');return report
    a,b = left['_summary'],right['_summary'];ca,cb=a['config'],b['config']
    if left['host'] != right['host']:report['issues'].append('cross_host')
    for scope, keys, x, y in [('summary',SUMMARY_KEYS,a,b),('config',CONFIG_KEYS,ca,cb)]:
        for key in keys:
            if key not in x or key not in y:report['issues'].append('unknown:'+scope+'.'+key)
            elif x[key] != y[key]:report['issues'].append('mismatch:'+scope+'.'+key)
    if not isinstance(ca.get('qk_norm'),bool) or not isinstance(cb.get('qk_norm'),bool):
        report['issues'].append('unknown:qk_norm')
    different_arm = {left['arm'],right['arm']} == {'exact','nogram'}
    if different_arm and ca.get('qk_norm') == cb.get('qk_norm'):
        kind='add_table_exact_minus_nogram';treatment,control=(left,right) if left['arm']=='exact' else (right,left)
    elif left['arm']==right['arm'] and ca.get('qk_norm') != cb.get('qk_norm'):
        kind='qkn_intervention_same_arm_not_add_table';treatment,control=(left,right) if ca.get('qk_norm') else (right,left)
        for key in TABLE_KEYS:
            if key not in ca or key not in cb:report['issues'].append('unknown:config.'+key)
            elif ca[key] != cb[key]:report['issues'].append('mismatch:config.'+key)
    else:report['issues'].append('unsupported_or_multiple_interventions');return report
    if report['issues']:return report
    common = sorted(set(treatment['_rows']) & set(control['_rows']))
    if len(common)<60 or common[-1]!=a['steps']:
        report['issues'].append('need_60_common_points_and_same_final_step');return report
    def point(step):
        t,c=treatment['_rows'][step],control['_rows'][step]
        return dict(step=step,treatment_train=t['train_loss'],control_train=c['train_loss'],
                    treatment_val=t['val_loss'],control_val=c['val_loss'],
                    delta_val=t['val_loss']-c['val_loss'],delta_train=t['train_loss']-c['train_loss'])
    tail=[point(s) for s in common[-60:]]
    report.update(state='validated_numeric_comparison',comparison=kind,treatment=treatment['run_id'],
                  control=control['run_id'],final=point(common[-1]),last_60_common_points=tail,
                  mean_last_60={k:sum(p[k] for p in tail)/60 for k in tail[0] if k!='step'},
                  interpretation='Delta is treatment minus control; negative val favors treatment. '
                  'Finite loss does not imply stable training; inspect absolute losses and full curves '
                  'before claiming a benefit. Config/transport checks do not establish mechanism '
                  'or revalidate historical data fingerprints.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',type=Path,default=BASE/'overnight_status.json')
    args=parser.parse_args()
    if not args.status.is_file():parser.error('Run python3 code/cluster/overnight_status.py first')
    status=json.loads(args.status.read_text())
    if not 0 <= time.time()-status.get('observed_epoch',0) <= 600:
        parser.error('Status older than 10 minutes: run overnight_status.py first')
    manifest=json.loads((ROOT/'docs/plans/overnight-20261024.json').read_text())
    selected,edges=select_requests(manifest,status)
    snapshot=BASE/'overnight_results/snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    snapshot.mkdir(parents=True)
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        work={h:[r for (host,_),r in selected.items() if host==h and not r.get('blocked')] for h in HOSTS}
        futures={h:pool.submit(fetch_host,h,rs,snapshot) for h,rs in work.items() if rs}
        for host,future in futures.items():
            try:records.extend(future.result())
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                errors.append({'host':host,'error':str(error)})
    jobs={(j['host'],j['run_id']):j for j in manifest['jobs']}
    runs={(r['host'],r['run_id']):validate_run(r,snapshot,jobs.get((r['host'],r['run_id']))) for r in records}
    comparisons=[]
    for a,b in edges:
        if a in runs and b in runs:comparisons.append(compare_runs(runs[a],runs[b]))
        else:comparisons.append({'left':a,'right':b,'state':'manual_review',
                                 'scientific_review_required':True,'issues':['evidence_not_collected']})
    review=dict(status_observed_epoch=status['observed_epoch'],selected_runs=len(selected),errors=errors,
                runs=[{k:v for k,v in r.items() if not k.startswith('_')} for r in runs.values()],
                blocked_requests=[dict(host=h,**r) for (h,_),r in selected.items() if r.get('blocked')],
                comparisons=comparisons,scope='Only registered terminal claims and direct controls. '
                'No retries, weights, remote writes, or automatic scientific conclusions.')
    (snapshot/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved:',snapshot/'review.json')
    print('Runs:',len(runs),'valid numeric comparisons:',sum(p['state']=='validated_numeric_comparison' for p in comparisons),
          'host errors:',len(errors))


if __name__=='__main__':main()
