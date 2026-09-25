#!/usr/bin/env python3
"""Dispatch the registered 24 runs on idle H200s; never overwrite a run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data-root', 'out-root', 'queue-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    a.queue_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.queue_dir / 'dispatcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [(s, arm, lr) for s in ['id', 'magicoder', 'openr1', 'rlvr']
             for lr in ['q4', 'q10'] for arm in ['trainable', 'frozen', 'nogram']]
    running = {}
    completed = []
    pending = []
    for source, arm, lr in tasks:
        rid = f'ptlr_{source}_{arm}_{lr}_s42_fixed'
        out = a.out_root / rid
        if out.exists():
            status = out / 'status.json'
            if not status.exists():
                raise RuntimeError('Existing uncompleted run requires inspection: ' + str(out))
            completed.append({'run_id': rid, 'status': json.loads(status.read_text())})
        else:
            pending.append((source, arm, lr, rid,))
    allowed = set(map(int, a.gpus.split(',')))
    while pending or running:
        for gpu, (proc, log, rid) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                status_path = a.out_root / rid / 'status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {'returncode': proc.returncode, 'qc': 'launcher_failed'}
                completed.append({'run_id': rid, 'status': status})
                print('FINISHED', rid, json.dumps(status), flush=True)
                del running[gpu]
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        for line in memory.splitlines():
            gpu, used = map(int, line.split(','))
            if not pending or gpu not in allowed or gpu in running or used >= 150:
                continue
            source, arm, lr, rid = pending.pop(0)
            cmd = [sys.executable, str(a.root / 'code/cluster/run_posttraining_lr_sweep.py'),
                   '--root', str(a.root), '--data-root', str(a.data_root),
                   '--out-root', str(a.out_root), '--source', source, '--arm', arm,
                   '--lr-tag', lr, '--gpu', str(gpu)]
            log = (a.queue_dir / (rid + '.launcher.log')).open('x')
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            running[gpu] = proc, log, rid
            print('LAUNCHED', rid, 'GPU', gpu, 'PID', proc.pid, flush=True)
        state = {'pending': [t[-1] for t in pending],
                 'running': {str(g): {'run_id': t[2], 'pid': t[0].pid} for g, t in running.items()},
                 'completed': completed, 'updated_unix': time.time()}
        tmp = a.queue_dir / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + chr(10))
        tmp.replace(a.queue_dir / 'state.json')
        if pending or running:
            time.sleep(15)
    print('QUEUE COMPLETE', len(completed), flush=True)


if __name__ == '__main__':
    main()
