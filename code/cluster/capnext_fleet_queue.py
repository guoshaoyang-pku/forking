#!/usr/bin/env python3
"""New bounded E3 queue using existing claim/GPU locks, plus compute-PID gate."""
import subprocess
import fleet_queue as fleet

_memory_free = fleet.gpus_free

def unoccupied_gpus():
    memory = _memory_free()
    mapping = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    uuids = {uuid.strip():int(index) for index,uuid in (line.split(',') for line in mapping.splitlines())}
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    for line in processes.splitlines():
        if not line.strip():continue
        uuid = line.split(',')[0].strip()
        if uuid not in uuids:raise ValueError('Unknown active GPU UUID; refuse dispatch')
        memory[uuids[uuid]] = max(memory[uuids[uuid]],fleet.FREE_MIB)
    return memory

if __name__ == '__main__':
    fleet.gpus_free = unoccupied_gpus
    fleet.main()
