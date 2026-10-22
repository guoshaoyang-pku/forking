#!/usr/bin/env python3
"""Cross-host GPU queue for the A100 fleet (all hosts mount the same CFS).

jobs file (one job per line, '#' = comment):   ENV=a ENV2=b | model arm seed
A job is claimed by mkdir(<queue>/claims/<jobkey>) -- atomic on NFS, so every job
runs exactly once across hosts. Per-host agent: every POLL s, for each local GPU
with < FREE_MIB used and nothing started on it in the last GRACE s, claim the
first unclaimed job and start it with the host's launcher. On exit the wrapper
writes rc + host/gpu/timestamps into the claim dir. Appending lines to the jobs
file adds work at any time; touching <queue>/STOP_<host> drains the host.

usage: fleet_queue.py <queue_dir> <root> <launcher_rel> [--gpus 0,1,...]
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

POLL, GRACE, FREE_MIB = 60, 240, 1500


def gpus_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def jobs(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        envs, args = [x.strip() for x in line.split("|", 1)]
        key = hashlib.sha1(line.encode()).hexdigest()[:12]
        out.append((key, envs, args, line))
    return out


def main():
    qdir, root, launcher = sys.argv[1], sys.argv[2], sys.argv[3]
    only = None
    if "--gpus" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--gpus") + 1].split(",")}
    host = os.environ.get("FLEET_HOST", socket.gethostname())
    claims = os.path.join(qdir, "claims")
    os.makedirs(claims, exist_ok=True)
    log = open(os.path.join(qdir, f"agent_{host}.log"), "a", buffering=1)
    last = {}
    while True:
        if os.path.exists(os.path.join(qdir, f"STOP_{host}")):
            log.write(f"{time.strftime('%F %T')} STOP file, exiting\n")
            return
        try:
            free = gpus_free()
        except Exception as e:  # noqa: BLE001
            log.write(f"{time.strftime('%F %T')} nvidia-smi failed: {e}\n")
            time.sleep(POLL)
            continue
        for g, used in sorted(free.items()):
            if only is not None and g not in only:
                continue
            if used >= FREE_MIB or time.time() - last.get(g, 0) < GRACE:
                continue
            for key, envs, args, line in jobs(os.path.join(qdir, "jobs.txt")):
                cdir = os.path.join(claims, key)
                try:
                    os.mkdir(cdir)
                except FileExistsError:
                    continue
                meta = {"line": line, "host": host, "gpu": g, "start": time.strftime("%F %T")}
                json.dump(meta, open(os.path.join(cdir, "claim.json"), "w"))
                envd = dict(os.environ)
                envd.update(dict(kv.split("=", 1) for kv in envs.split() if "=" in kv))
                cmd = (f"cd {root} && bash {launcher} {g} {args} > {cdir}/launcher.log 2>&1; "
                       f"echo $? > {cdir}/rc; date '+%F %T' > {cdir}/end")
                subprocess.Popen(["bash", "-c", cmd], env=envd, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log.write(f"{time.strftime('%F %T')} gpu{g} <- {line} [{key}]\n")
                last[g] = time.time()
                break
            else:
                break  # nothing left to claim
        time.sleep(POLL)


if __name__ == "__main__":
    main()
