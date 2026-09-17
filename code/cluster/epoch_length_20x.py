#!/usr/bin/env python3
"""Run the registered 20-point epoch-length array on explicit GPUs."""
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

LABELS = ('0p125','0p1667','0p25','0p3333','0p5','0p6667','0p75','1p0','1p25','1p5','1p75','2p0','2p5','3p0','4p0','6p0','8p0','10p0','15p0','20p0')
EPB = (42,56,84,112,168,224,253,337,421,506,590,674,842,1011,1348,2022,2696,3370,5055,6740)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--dry-run',action='store_true'); p.add_argument('gpus',nargs='+'); a=p.parse_args()
    root=os.path.abspath(a.root); py=os.environ.get('NGLAB_PY',root+'/.venv/bin/python'); data=os.environ.get('NGLAB_DATA_DIR',root+'/data/tokenized_epoch20'); out=os.environ.get('NGLAB_OUT_DIR',root+'/data/runs_scaling')
    train=os.environ.get('NGLAB_EPOCH20_TRAIN_SHARDS',','.join(map(str,range(21)))); val=os.environ.get('NGLAB_EPOCH20_VAL_SHARDS','21,22,23')
    if not os.path.isdir(data): raise SystemExit('missing data dir: '+data)
    for sid in range(24):
        if not os.path.isfile(f'{data}/shard_{sid:05d}.bin'): raise SystemExit(f'missing shard {sid}')
    os.makedirs(out,exist_ok=True)
    def one(i):
        gpu=a.gpus[i%len(a.gpus)]; epb=EPB[i]; steps=3*epb; rid=f's1v5_128_ep20_tri_{LABELS[i]}xL4_3ep_v2'; result=f'{out}/{rid}_fixed'
        checkpoints = ','.join(map(str, sorted(set(range(10, steps + 1, 10)) | {epb, 2 * epb, steps})))
        cmd=[py,'-u',root+'/code/train.py','--run_id',rid+'_fixed','--out_dir',out,'--data_dir',data,'--train_shards',train,'--val_shards',val,'--seed','42','--steps',str(steps),'--dtype','bf16','--injection_position','input','--enable_unigram','0','--enable_bigram','0','--enable_trigram','1','--bigram_clean_table','0','--trigram_clean_table','1048576','--n_layer','8','--n_head','6','--n_embd','768','--vocab_size','8192','--sequence_len','2048','--device_batch_size','72','--total_batch_size','147456','--lr','0.0006','--lr_schedule','warmup_constant','--warmup_steps','100','--table_optimizer','rmsprop','--table_betas','0.0,0.99','--table_lr_scale','128.0','--epoch_batches',str(epb),'--val_interval','10','--val_steps',checkpoints,'--val_batches','4','--table_norm_interval','10','--fixed_train_probe','0']
        if os.path.exists(result): return rid,'blocked existing directory'
        if a.dry_run: return rid,'dry-run'
        os.makedirs(result); env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        with open(result+'/train.log','w') as log: rc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        return rid,('done' if rc==0 else f'failed rc={rc}')
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as ex:
        for f in as_completed([ex.submit(one,i) for i in range(len(EPB))]): print('[epoch20]',*f.result(),flush=True)
if __name__=='__main__': main()
