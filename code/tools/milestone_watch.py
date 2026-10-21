#!/usr/bin/env python3
"""Bounded milestone watcher for the pro5k6 teacher-residual distillation run.

Runs on pro5k6, CPU-only (no GPU use), so it never competes with the workers.

Milestones
  M1  as soon as all four workers have a periodic checkpoint: merge -> check
      the aggregate against the HF depth references, keep the report, drop the
      merged copy (the per-worker parts stay authoritative).
  M2  once the live token total passes the trigger: merge -> check again and
      keep the merged table as a training-usable artifact.

Reports land in /data/distill_out/milestone_report_*.txt; the script exits after
M2.  It only ever reads the per-worker parts and writes into /data/distill_out.
"""
import json
import os
import re
import subprocess
import sys
import time

OUT = "/data/distill_out"
TAG = "w3r"
WORKERS = [0, 1, 2, 3]
LOG = "/data/milestone_watch.log"
MERGE = "/data/merge_distill_parts.py"
CHECK = "/data/check_distill_aggregate.py"
M2_TRIGGER = 2.5e9          # checkpoint coverage: project tokens summed over workers
RUN_START = 1792065600      # 2026-10-21 09:00 CST; older state files belong to dead runs
POLL = 300                  # seconds between polls
CKPT_QUIET = 300            # a checkpoint must be this old before we read it
RETRY_BACKOFF = 1800        # seconds to wait after a failed merge
FREE_M1_GB = 250
FREE_M2_GB = 400


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def ckpt_coverage():
    """Tokens actually inside the checkpoint files, from the per-worker state files.

    The state file is rewritten by the same save() call that writes the ckpt, so a
    fresh state file pins the coverage of that worker's checkpoint.
    """
    tot = 0
    per = {}
    for w in WORKERS:
        p = f"{OUT}/state_w{w}_{TAG}.json"
        if not os.path.exists(p) or os.path.getmtime(p) < RUN_START:
            return None
        with open(p) as fh:
            d = json.load(fh)
        per[w] = d.get("tokens", 0)
        tot += per[w]
    return tot, per


def live_tokens():
    """Cumulative project tokens per worker, read from the progress lines."""
    tot = 0.0
    per = {}
    for w in WORKERS:
        try:
            with open(f"/data/distill_r{w}.log", "rb") as fh:
                txt = fh.read()[-400000:].decode("utf-8", "replace")
        except OSError:
            return None
        m = re.findall(r"tok=([0-9.]+)M", txt)
        if not m:
            return None
        per[w] = float(m[-1]) * 1e6
        tot += per[w]
    return tot, per


def ckpt_ready():
    now = time.time()
    paths = []
    for w in WORKERS:
        p = f"{OUT}/.part_w{w}_{TAG}.ckpt.npz"
        if not os.path.exists(p) or now - os.path.getmtime(p) < CKPT_QUIET:
            return None
        paths.append(p)
    return paths


def free_gb():
    st = os.statvfs(OUT)
    return st.f_bavail * st.f_frsize / 1e9


def run(cmd, report):
    with open(report, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"[exit {r.returncode}]\n")
    return r.returncode


def main():
    log("watcher start")
    m1 = m2 = False
    next_try = 0.0
    while True:
        now = time.time()
        snap = live_tokens()
        if now >= next_try and not (m1 and m2):
            ck = ckpt_ready() if not m1 else None
            if not m1 and ck:
                if free_gb() < FREE_M1_GB:
                    log(f"M1 postponed: free={free_gb():.0f}GB")
                else:
                    log("M1: checkpoints ready -> merge + check")
                    rep = f"{OUT}/milestone_report_m1.txt"
                    tmp = f"{OUT}/_m1_merged.npz"
                    rc = run([sys.executable, MERGE, "--in_dir", OUT, "--tag", TAG,
                              "--out", tmp, "--include_ckpt", "--mean"], rep)
                    if rc == 0:
                        run([sys.executable, CHECK, tmp], rep)
                        os.remove(tmp)
                        m1 = True
                        log("M1 done")
                    else:
                        next_try = time.time() + RETRY_BACKOFF
                        log("M1 merge failed, backoff")
            cov = ckpt_coverage() if not m2 else None
            if not m2 and cov and cov[0] >= M2_TRIGGER:
                if free_gb() < FREE_M2_GB:
                    log(f"M2 postponed: free={free_gb():.0f}GB")
                else:
                    log(f"M2: ckpt coverage {cov[0]/1e9:.2f}B (live {snap[0]/1e9:.2f}B) -> merge + keep")
                    rep = f"{OUT}/milestone_report_2b.txt"
                    out = f"{OUT}/teacher_table_o2to5_2b.npz"
                    rc = run([sys.executable, MERGE, "--in_dir", OUT, "--tag", TAG,
                              "--out", out, "--include_ckpt", "--mean"], rep)
                    if rc == 0:
                        run([sys.executable, CHECK, out], rep)
                        m2 = True
                        log("M2 done")
                    else:
                        next_try = time.time() + RETRY_BACKOFF
                        log("M2 merge failed, backoff")
        if m1 and m2:
            log("all milestones done, watcher exit")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
