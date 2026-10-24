#!/usr/bin/env python3
"""Bounded queue adapter: inherited flock/claim semantics plus 5-minute idle admission."""
import json
from pathlib import Path
import subprocess
import sys
import time
import fleet_queue as fleet

_memory = fleet.gpus_free
_empty_since = {}
_last_sample = None
IDLE_SECONDS = 300
MAX_MEMORY_MIB = 512


def gate(memory, mapping, active, now):
    global _last_sample
    if len(set(mapping.values())) != len(mapping) or set(memory) != set(mapping.values()):
        _empty_since.clear()
        raise ValueError('Inconsistent GPU inventories; refuse dispatch')
    if _last_sample is not None and now - _last_sample > 90:
        _empty_since.clear()
    _last_sample = now
    if not active.issubset(mapping):
        _empty_since.clear()
        raise ValueError('Unknown compute GPU UUID; refuse dispatch')
    busy = {mapping[u] for u in active}
    out = {}
    for gpu, used in memory.items():
        free = gpu not in busy and used < MAX_MEMORY_MIB
        if free:
            _empty_since.setdefault(gpu, now)
        else:
            _empty_since.pop(gpu, None)
        admitted = free and now - _empty_since[gpu] >= IDLE_SECONDS
        out[gpu] = used if admitted else max(used, fleet.FREE_MIB)
    return out


def unoccupied_gpus():
    try:
        memory = _memory()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        mapping = {uuid.strip(): int(index) for index, uuid in
            (line.split(',') for line in rows.splitlines() if line.strip())}
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        active = {line.split(',')[0].strip() for line in rows.splitlines() if line.strip()}
        now = time.monotonic()
        result = gate(memory, mapping, active, now)
        q = Path(sys.argv[1])
        host = __import__('os').environ['FLEET_HOST']
        status = {'observed_unix': time.time(), 'host': host,
            'idle_seconds_required': IDLE_SECONDS, 'memory_limit_mib': MAX_MEMORY_MIB,
            'actual_memory_mib': memory, 'has_compute_pid': sorted(mapping[u] for u in active),
            'idle_seconds': {g: now-t for g,t in _empty_since.items()},
            'admitted_gpus': [g for g,v in result.items() if v < fleet.FREE_MIB]}
        p = q / ('admission_' + host + '.json')
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(p)
        return result
    except Exception:
        _empty_since.clear()
        raise


if __name__ == '__main__':
    fleet.gpus_free = unoccupied_gpus
    fleet.main()
