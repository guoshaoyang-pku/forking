#!/usr/bin/env python3
"""Finite idle-GPU queue: each source/arm smoke must pass before formal training."""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for key in ['root','data-root']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--admission-hours',type=float,default=8)
    a=p.parse_args()
    q=a.root/'queue';q.mkdir(exist_ok=False)
    pending=[(s,arm) for s in ['id','math'] for arm in ['tableonly','module','trainable','nogram']]
    active={};done=[];failed=[];started=time.monotonic()
    def write():
        state=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   pending=pending,active={str(g):dict(source=x['source'],arm=x['arm'],
                   phase=x['phase'],pid=x['process'].pid) for g,x in active.items()},done=done,failed=failed)
        tmp=q/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(q/'state.json')
    def launch(g,source,arm,smoke):
        phase='smoke' if smoke else 'formal'
        cmd=[sys.executable,str(a.root/'code/cluster/run_multilayer_posttraining.py'),
             '--root',str(a.root),'--data-root',str(a.data_root),
             '--pool-root',str(a.root/'data/pools'),
             '--out-root',str(a.root/('smoke' if smoke else 'data/runs_fixed')),
             '--source',source,'--arm',arm,'--gpu',str(g)]
        if smoke:cmd+=['--smoke']
        f=(q/f'{source}_{arm}_{phase}.log').open('x')
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);f.close()
        active[g]=dict(source=source,arm=arm,phase=phase,process=proc)
    while pending or active:
        for g,x in list(active.items()):
            rc=x['process'].poll()
            if rc is None:continue
            del active[g]
            item=dict(source=x['source'],arm=x['arm'],phase=x['phase'],returncode=rc)
            if rc!=0:failed.append(item)
            elif x['phase']=='smoke':
                status=a.root/'smoke'/f"ptml_{x['source']}_{x['arm']}_L14_q4_s42_smoke2_fixed"/'status.json'
                if json.loads(status.read_text()).get('qc')!='pass':
                    item['error']='smoke QC did not pass';failed.append(item)
                else:launch(g,x['source'],x['arm'],False)
            else:done.append(item)
        if pending and time.monotonic()-started<a.admission_hours*3600:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'],text=True).splitlines()
            for row in rows:
                g,mem,util=map(int,row.split(','))
                if pending and g not in active and mem<150 and util==0:
                    source,arm=pending.pop(0);launch(g,source,arm,True)
        write()
        if pending and not active and time.monotonic()-started>=a.admission_hours*3600:
            break
        if pending or active:time.sleep(20)
    write()

if __name__=='__main__':main()
