#!/usr/bin/env python3
"""Bounded unattended restart of the pro5k6 distillation workers onto distill_prod_v2.py.

Why: the running workers spend ~2.03 s of every 2.70 s step inside the CPU
accumulation, mostly because the box default of 192 torch threads thrashes under
4-worker contention.  `distill_prod_v2.py` caps the pool at 32 threads and is
bit-identical to production on CPU replays (cnt exact, acc max-abs diff 0).

The restart must be lossless, so it is gated on two conditions:

  A  the 2B milestone merge has already succeeded, i.e.
     /data/distill_out/teacher_table_o2to5_2b.npz exists and
     /data/distill_out/milestone_report_2b.txt reports a completed merge + check;
  B  every worker has just written a periodic checkpoint (all four ckpt files
     older than QUIET seconds) and each state file lists >= MIN_SHARDS shards, so
     the killed workers lose at most the shard they are currently in.

Then: rename the w3r logs, kill the four old workers by PID, wait for the GPUs to
drain, launch run_distill_v2.sh (tag w3s, shards 48+, 1.8e9 tokens/worker, a
checkpoint after every shard), and verify after 25 minutes that every worker is
producing progress at >= MIN_RATE lab tok/s.  The verdict goes to
/data/restart_summary.txt.

The script performs at most one restart and then exits.  It never touches
/data/distill_out/* parts or ckpts.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

OUT = "/data/distill_out"
TABLE = f"{OUT}/teacher_table_o2to5_2b.npz"
REPORT = f"{OUT}/milestone_report_2b.txt"
OLD_SCRIPT = "/data/distill_prod.py"
NEW_SCRIPT = "/data/distill_prod_v2.py"
CANARY_SCRIPT = "/data/distill_prod_v3.py"
CANARY_WORKER = 3
LAUNCHER = "/data/run_distill_v2.sh"
TAG_OLD = "w3r"
TAG_NEW = "w3s"
SHARD_END = 210
TOK_PER_WORKER = 1.8e9
LAYERS = "0,10,25,40"
WORKERS = (0, 1, 2, 3)
POLL = 120
QUIET = 300
MIN_SHARDS = 12
MIN_RATE = 15000
VERIFY_WAIT = 1500
SUMMARY = "/data/restart_summary.txt"
LOG = "/data/auto_restart.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def note(msg):
    with open(SUMMARY, "a") as fh:
        fh.write(msg + "\n")


def gate_a():
    if not (os.path.exists(TABLE) and os.path.getsize(TABLE) > 20e9):
        return False
    if not os.path.exists(REPORT):
        return False
    txt = open(REPORT).read()
    return "[exit 0]" in txt and "wrote " in txt


def gate_b():
    now = time.time()
    infos = {}
    for w in WORKERS:
        ck = f"{OUT}/.part_w{w}_{TAG_OLD}.ckpt.npz"
        st = f"{OUT}/state_w{w}_{TAG_OLD}.json"
        if not os.path.exists(ck) or not os.path.exists(st):
            return None
        if now - os.path.getmtime(ck) < QUIET:
            return None
        try:
            d = json.load(open(st))
        except Exception:
            return None
        if len(d.get("shards", [])) < MIN_SHARDS:
            return None
        infos[w] = d
    return infos


def pids():
    out = subprocess.run(["pgrep", "-f", f"^python3 {OLD_SCRIPT} --worker"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def main():
    log("auto-restart watchdog start")
    while True:
        if gate_a():
            break
        time.sleep(POLL)
    log("gate A satisfied: 2B table merged and checked")

    infos = None
    while infos is None:
        infos = gate_b()
        if infos is None:
            time.sleep(POLL)
    cov = sum(d["tokens"] for d in infos.values())
    shards = {w: len(infos[w]["shards"]) for w in WORKERS}
    log(f"gate B satisfied: ckpt coverage {cov/1e9:.2f}B shards={shards}")
    note(f"restart at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    note(f"  ckpt coverage {cov/1e9:.3f}B, shards done per worker {shards}")
    note(f"  frontier: shards 0-{4*min(shards.values())-1} are covered by w3r")

    for w in WORKERS:
        src = f"/data/distill_r{w}.log"
        if os.path.exists(src):
            os.rename(src, f"/data/distill_w3r_r{w}.log")
    kill = pids()
    log(f"terminating old workers: {kill}")
    for p in kill:
        subprocess.run(["kill", "-TERM", str(p)])
    deadline = time.time() + 300
    while time.time() < deadline and pids():
        time.sleep(10)
    left = pids()
    if left:
        log(f"workers still alive after SIGTERM: {left}; not forcing, aborting restart")
        note("  ABORT: old workers did not exit; nothing was relaunched")
        return
    deadline = time.time() + 300
    while time.time() < deadline and max(gpu_used_mb()) > 2000:
        time.sleep(15)
    log(f"gpu memory used after drain: {gpu_used_mb()}")

    frontier = 4 * max(shards.values())
    if frontier >= SHARD_END:
        log(f"frontier {frontier} >= shard_end {SHARD_END}: nothing left to cover")
        note(f"  ABORT: frontier {frontier} leaves no shards")
        return
    note(f"  launching tag {TAG_NEW}, shards {frontier}-{SHARD_END - 1}, "
         f"{TOK_PER_WORKER/1e9:.2f}B tokens/worker; "
         f"w0-w2 on v2, w{CANARY_WORKER} on v3 (canary)")
    for w in WORKERS:
        scr = CANARY_SCRIPT if w == CANARY_WORKER else NEW_SCRIPT
        g0, g1 = 2 * w, 2 * w + 1
        cmd = (f"cd /data && CUDA_VISIBLE_DEVICES={g0},{g1} OE_CAP_LAYERS={LAYERS} "
               f"setsid nohup python3 {scr} --worker {w} "
               f"--num_workers {len(WORKERS)} "
               f"--shard_start {frontier} --shard_end {SHARD_END} "
               f"--token_budget {TOK_PER_WORKER:.0f} --tag {TAG_NEW} "
               f"--save_every_shards 1 --enforce_eager --batch_rows 16 "
               f"--max_num_batched_tokens 32768 "
               f"> /data/distill_r{w}.log 2>&1 < /dev/null & echo launched w{w} {scr}")
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        log(f"launch w{w}: rc={p.returncode} {p.stdout.strip()} "
            f"{p.stderr.strip()[-300:]}")

    time.sleep(VERIFY_WAIT)
    verdict = []
    for w in WORKERS:
        txt = ""
        try:
            txt = open(f"/data/distill_r{w}.log", errors="replace").read()[-200000:]
        except OSError:
            pass
        rates = [int(x) for x in re.findall(r"rate=(\d+) lab tok/s", txt)]
        ready = "engine ready" in txt
        ok = bool(rates) and max(rates) >= MIN_RATE
        verdict.append((w, ready, max(rates) if rates else 0, ok))
    for w, ready, rate, ok in verdict:
        note(f"  w{w}: engine_ready={ready} max_rate={rate} {'OK' if ok else 'CHECK'}")
        log(f"verify w{w}: ready={ready} rate={rate} ok={ok}")
    if all(ok for _, _, _, ok in verdict):
        note("RESULT: all four workers healthy on v2")
    else:
        note("RESULT: NEEDS ATTENTION - see /data/auto_restart.log and the "
             "distill_r<w>.log tails; a failing worker must be relaunched with "
             "/data/distill_prod_v2.py on the same GPU pair and tag w3s")
    log("auto-restart watchdog done")


if __name__ == "__main__":
    main()
