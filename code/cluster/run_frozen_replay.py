#!/usr/bin/env python3
"""Frozen from step zero vs no-gram; registered nested-pool replay/LR grid."""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ARMS = ['frozen0', 'nogram']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def command(a):
    steps = 2 if a.smoke else 1680
    rid = f'ptfr_u{a.pool_updates}_{a.arm}_{a.rate}_mb18_s{a.seed}' + ('_smoke2_fixed' if a.smoke else '_fixed')
    base = 's1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed' if a.arm == 'nogram' else 'sftbase_ep20_tri_ckpt_fixed'
    init = a.data_root/'runs_scaling'/base/'final_model.pt'
    vals = sorted(set(range(10, steps+1, 10)) | {steps} | ({v for b in range(a.pool_updates,steps,a.pool_updates) for v in [b,b+1]} if not a.smoke else set()))
    cmd = [sys.executable, '-u', str(a.root/'code/train.py'),
        '--run_id', rid, '--out_dir', str(a.out_root),
        '--data_dir', str(a.data_root/'frozen_replay_20260925'/f'u{a.pool_updates}'),
        '--train_shards','0','--val_shards','1','--seed',str(a.seed),
        '--steps',str(steps),'--dtype','bf16','--deterministic','0',
        '--injection_position','input','--enable_unigram','0','--enable_bigram','0',
        '--enable_trigram',str(int(a.arm!='nogram')),'--bigram_clean_table','0',
        '--trigram_clean_table',str(1048576 if a.arm!='nogram' else 0),
        '--n_layer','8','--n_head','6','--n_embd','768','--vocab_size','8192',
        '--sequence_len','2048','--device_batch_size','18','--total_batch_size','147456',
        '--lr',str(0.00015 if a.rate=='q4' else 0.0006),'--weight_decay','0.1','--lr_schedule','warmup_constant',
        '--warmup_steps','100','--warmup_start_lr_mult','0.25',
        '--table_optimizer','rmsprop','--table_betas','0.0,0.99','--table_lr_scale','128.0',
        '--epoch_batches',str(a.pool_updates*4),'--val_interval','10','--val_steps',','.join(map(str,vals)),
        '--val_batches','16','--freq_eval_batches','16','--freq_eval_interval','10',
        '--exact_freq_eval_interval','10','--table_norm_interval','10',
        '--fixed_train_probe','16','--train_probe_mode','uniform','--probe_eval_interval','10',
        '--init_model',str(init)]
    if a.arm=='frozen0': cmd += ['--freeze_table']
    if not a.smoke: cmd += ['--save_final_model']
    return rid, cmd, vals

def initial_probe(a, cmd, out):
    import numpy as np
    import torch
    sys.path.insert(0, str(a.root/'code'))
    from train import Config, NanoGPT, TokenizedShardDataset, evaluate_fixed_probe
    data = cmd[cmd.index('--data_dir')+1]
    cfg = Config(nanogpt_ngram_injection_position='input', enable_bigram_ve=False,
                 enable_trigram_ve=a.arm!='nogram', trigram_clean_table=1048576 if a.arm!='nogram' else 0)
    model = NanoGPT(cfg)
    state = torch.load(cmd[cmd.index('--init_model')+1], map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True); del state
    model.cuda().eval()
    train = TokenizedShardDataset(data,[0],2048,18,epoch_batches=a.pool_updates*4)
    val = TokenizedShardDataset(data,[1],2048,18)
    assert train.total_chunks()==a.pool_updates*72 and val.total_chunks()==288
    idx=[int(round(i*(a.pool_updates*72-1)/287)) for i in range(288)]
    batches=[]
    for start in range(0,288,18):
        pairs=[train.get_chunk(i) for i in idx[start:start+18]]
        batches.append((torch.stack([p[0] for p in pairs]).cuda(),torch.stack([p[1] for p in pairs]).cuda()))
    probe_hash=hashlib.sha256(b''.join(b[0].cpu().numpy().tobytes() for b in batches)).hexdigest()[:16]
    ft=evaluate_fixed_probe(model,batches,dtype='bf16')
    vi=val.iter_batches(torch.device('cuda'));vb=[next(vi) for _ in range(16)]
    fv=evaluate_fixed_probe(model,vb,dtype='bf16')
    (out/'initial_probe.json').write_text(json.dumps({'step':0,'fixed_train_loss':ft,'fixed_val_loss':fv,
        'fixed_gap':fv-ft,'probe_sha256':probe_hash,'train_rows':a.pool_updates*72,'val_rows':288},indent=2)+chr(10))
    del model,batches,vb,vi,train,val
    gc.collect();torch.cuda.empty_cache()

def verify(a,out,vals):
    s=json.loads((out/'summary.json').read_text());c=s['config']
    assert s['seed']==a.seed and s['steps']==vals[-1]
    assert c['device_batch_size']==18 and c['total_batch_size']==147456 and c['epoch_batches']==a.pool_updates*4
    assert c['nanogpt_adam_lr']==(0.00015 if a.rate=='q4' else 0.0006) and c['table_lr_scale']==128
    assert c['lr_schedule']=='warmup_constant' and c['warmup_steps']==100
    assert s['fixed_train_probe_batches']==16 and c['val_batches']==16
    for fn in ['train_log.jsonl','fixed_train_loss.jsonl']:
        rows=[json.loads(x) for x in (out/fn).read_text().splitlines()]
        assert [r['step'] for r in rows]==vals
        for r in rows:
            if fn.startswith('train_'): assert abs(r['gap']-r['val_loss']+r['train_loss'])<1e-6
            else: assert abs(r['fixed_gap']-r['fixed_val_loss']+r['fixed_train_loss'])<1e-6
    initial=json.loads((out/'initial_probe.json').read_text())
    assert s['fixed_train_probe_sha256']==initial['probe_sha256']
    events=s['intervention']['events']
    assert events==[]
    assert s['freeze_table']==(a.arm=='frozen0')
    norms=[json.loads(x) for x in (out/'table_norm.jsonl').read_text().splitlines()]
    if a.arm=='frozen0' and norms:
        after=norms
        for k in after[0] if after else []:
            if k!='step': assert len({r[k] for r in after})==1
    keys=['final_train_loss','final_val_loss','final_gap','final_fixed_train_loss','final_fixed_val_loss','final_fixed_gap']
    assert all(math.isfinite(s[k]) for k in keys)
    if not a.smoke: assert (out/'final_model.pt').stat().st_size>200000000 and (out/'probe_tokens.npz').exists()
    if not a.smoke and a.arm=='frozen0':
        import torch
        initial_state=torch.load(s['init_model'],map_location='cpu',mmap=True,weights_only=True)
        final_state=torch.load(out/'final_model.pt',map_location='cpu',mmap=True,weights_only=True)
        names=[k for k in initial_state if any(m in k for m in ('value_embeds','bigram_ves','trigram_ves','ve_gate','bigram_gate','trigram_gate'))]
        assert names and all(torch.equal(initial_state[k],final_state[k]) for k in names)
        assert any(not torch.equal(initial_state[k],final_state[k]) for k in initial_state if k not in names)
        (out/'freeze_audit.json').write_text(json.dumps(dict(table_bitwise_equal=True,backbone_changed=True,names=names),indent=2)+chr(10))
    return {k:s[k] for k in keys}

def main():
    p=argparse.ArgumentParser()
    for k in ['root','data-root','out-root']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pool-updates',type=int,choices=[42,84],required=True)
    p.add_argument('--rate',choices=['base','q4'],required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();rid,cmd,vals=command(a)
    if a.dry_run:print(json.dumps({'run_id':rid,'command':cmd}));return
    info=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    assert int(info.split(',')[-2])<150, 'GPU occupied: '+info
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
    out=a.out_root/rid;out.mkdir(parents=True,exist_ok=False)
    import torch
    launch={'run_id':rid,'command':cmd,'gpu':info,'host':os.uname().nodename,'torch':torch.__version__,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'code_hashes':{str(f.relative_to(a.root)):sha(f) for f in [a.root/'code/train.py',a.root/'code/ngram_freq.py',Path(__file__).resolve()]}}
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
    initial_probe(a,cmd,out)
    with (out/'train.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        (out/'pid.txt').write_text(str(child.pid)+chr(10));rc=child.wait()
    status={'returncode':rc,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if rc==0:
        try:status['metrics']=verify(a,out,vals);status['qc']='pass'
        except Exception as e:status.update(qc='fail',error=repr(e));rc=2
    else:status['qc']='failed_process'
    (out/'status.json').write_text(json.dumps(status,indent=2)+chr(10))
    print(json.dumps(status),flush=True);raise SystemExit(rc)

if __name__=='__main__':main()
