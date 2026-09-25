#!/usr/bin/env python3
"""Isolated, GPU-scoped post-training LR sweep; refuses artifact overwrite."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCES = {'id':('tokenized_epoch20','21','22,23'),
           'magicoder':('ood_tokenized_magicoder_fixtok','0','1'),
           'openr1':('ood_tokenized_openr1_10shards_fixtok','0','1'),
           'rlvr':('rlvr_tokenized/math_fixtok','0','1')}
LRS = {'q4':0.00015, 'q10':0.00006}


def command(root, data_root, output, source, arm, lr_tag, steps=1665, smoke=False):
    rid = f'ptlr_{source}_{arm}_{lr_tag}_s42' + ('_smoke10_fixed' if smoke else '_fixed')
    data,train,val = SOURCES[source]
    base = 's1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed' if arm == 'nogram' else 'sftbase_ep20_tri_ckpt_fixed'
    args = [sys.executable,'-u',str(root/'code/train.py'),
            '--run_id',rid,'--out_dir',str(output),'--data_dir',str(data_root/data),
            '--train_shards',train,'--val_shards',val,'--seed','42','--steps',str(steps),
            '--dtype','bf16','--deterministic','0','--injection_position','input',
            '--enable_unigram','0','--enable_bigram','0','--enable_trigram',str(int(arm!='nogram')),
            '--bigram_clean_table','0','--trigram_clean_table',str(1048576 if arm!='nogram' else 0),
            '--n_layer','8','--n_head','6','--n_embd','768','--vocab_size','8192','--sequence_len','2048',
            '--device_batch_size','72','--total_batch_size','147456',
            '--lr',str(LRS[lr_tag]),'--weight_decay','0.1',
            '--lr_schedule','warmup_constant','--warmup_steps','100','--warmup_start_lr_mult','0.25',
            '--table_optimizer','rmsprop','--table_betas','0.0,0.99','--table_lr_scale','128.0',
            '--epoch_batches','333','--val_interval','10','--val_batches','4',
            '--freq_eval_interval','10','--exact_freq_eval_interval','10','--table_norm_interval','10',
            '--fixed_train_probe','4','--train_probe_mode','uniform','--probe_eval_interval','10',
            '--init_model',str(data_root/'runs_scaling'/base/'final_model.pt')]
    if arm == 'frozen':args.append('--freeze_table')
    if not smoke:args.append('--save_final_model')
    return rid,args


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(summary_path, steps, arm, lr_tag):
    s=json.loads(summary_path.read_text());c=s['config']
    assert s['steps']==steps and s['seed']==42
    assert c['nanogpt_adam_lr']==LRS[lr_tag] and c['table_lr_scale']==128.0
    assert c['epoch_batches']==333 and c['device_batch_size']==72 and c['val_batches']==4
    assert c['data_mode']=='fixed' and c['data_seed']==42
    assert c['warmup_steps']==100 and c['lr_schedule']=='warmup_constant'
    assert c['enable_trigram_ve']==(arm!='nogram') and not c['enable_bigram_ve']
    assert s['freeze_table']==(arm=='frozen')
    assert s['fixed_train_probe_batches']==4 and s['train_probe_mode']=='uniform'
    for k in ['final_train_loss','final_val_loss','final_gap','final_fixed_train_loss','final_fixed_val_loss','final_fixed_gap']:
        assert math.isfinite(s[k]),k
    for fn in ['train_log.jsonl','fixed_train_loss.jsonl']:
        rows=[json.loads(l) for l in (summary_path.parent/fn).read_text().splitlines()]
        assert rows[-1]['step']==steps
        if fn=='train_log.jsonl':assert [x['step'] for x in rows]==list(range(10,steps+1,10))+([] if steps%10==0 else [steps])
    if arm=='frozen':
        norms=[json.loads(l) for l in (summary_path.parent/'table_norm.jsonl').read_text().splitlines()]
        for k in norms[0]:
            if k!='step':assert len({r[k] for r in norms})==1,k
    return {k:s[k] for k in ['final_train_loss','final_val_loss','final_gap','final_fixed_train_loss','final_fixed_val_loss','final_fixed_gap']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True);p.add_argument('--source',choices=SOURCES,required=True)
    p.add_argument('--arm',choices=['trainable','frozen','nogram'],required=True)
    p.add_argument('--lr-tag',choices=LRS,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();steps=10 if a.smoke else 1665
    rid,cmd=command(a.root,a.data_root,a.out_root,a.source,a.arm,a.lr_tag,steps,a.smoke)
    if a.dry_run:print(json.dumps({'run_id':rid,'command':cmd}));return
    import torch
    import numpy
    info=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    assert int(info.split(',')[-2].strip())<150, 'GPU is occupied: '+info
    for flag in ['--init_model','--data_dir']:
        assert Path(cmd[cmd.index(flag)+1]).exists()
    out=a.out_root/rid;out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
    provenance={'run_id':rid,'command':cmd,'gpu':info,'host':os.uname().nodename,
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'torch':torch.__version__,'numpy':numpy.__version__,
                'code_sha256':{str(f.relative_to(a.root)):sha(f) for f in [a.root/'code/train.py',a.root/'code/ngram_freq.py',Path(__file__).resolve()]}}
    (out/'launch.json').write_text(json.dumps(provenance,indent=2)+chr(10))
    with (out/'train.log').open('w') as f:
        process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        (out/'pid.txt').write_text(str(process.pid)+chr(10))
        rc=process.wait()
    status={'returncode':rc,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if rc==0:
        try:status['metrics']=verify(out/'summary.json',steps,a.arm,a.lr_tag);status['qc']='pass'
        except Exception as e:status['qc']='fail';status['error']=repr(e);rc=2
    else:status['qc']='failed_process'
    (out/'status.json').write_text(json.dumps(status,indent=2)+chr(10));print(json.dumps(status),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
