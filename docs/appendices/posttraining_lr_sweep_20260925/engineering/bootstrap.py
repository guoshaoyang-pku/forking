import json,os,subprocess,time
from pathlib import Path
root=Path('/data/home/guoshaoyang/ngram-gap-lab/provisioning/posttraining_lr_20260925/snapshot')
data=Path('/data/home/guoshaoyang/ngram-gap-lab/data')
base=Path('/data/shared/guoshaoyang/ngram-gap-lab/posttraining_lr_20260925')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
while True:
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
 free=[int(r.split(',')[0]) for r in rows if int(r.split(',')[1])<150]
 if free:break
 time.sleep(15)
gpu=free[0]
print('SMOKE GPU',gpu,flush=True)
cmd=['python3',str(root/'code/cluster/run_posttraining_lr_sweep.py'),'--root',str(root),'--data-root',str(data),'--out-root',str(base/'smoke'),'--source','magicoder','--arm','trainable','--lr-tag','q4','--gpu',str(gpu),'--smoke']
rc=subprocess.run(cmd,env=env).returncode
print('SMOKE EXIT',rc,flush=True)
if rc:
 if (base/'smoke/ptlr_magicoder_trainable_q4_s42_smoke10_fixed').exists():raise SystemExit(rc)
 time.sleep(15)
 os.execv('/usr/bin/python3',['python3','-u',str(Path(__file__))])
time.sleep(10)
cmd=['python3','-u',str(root/'code/cluster/continue_posttraining_lr_queue.py'),'--root',str(root),'--data-root',str(data),'--out-root',str(base/'data/runs_fixed'),'--queue-dir',str(base/'queue')]
raise SystemExit(subprocess.run(cmd,env=env).returncode)
