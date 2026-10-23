#!/usr/bin/env python3
"""v27 paired-delta table: every OE arm vs its same-seed nogram, read at E1/E2/E3 ends.

Input : .tmp/harvest/v27_a100/<host>/<run_id>/train_log.jsonl (+ summary.json)
Window: last WIN val rows (10-step logging => ~600 steps) of each epoch (jsonl 'epoch').
Sign  : dTrain = nog.train - arm.train, dVal = nog.val - arm.val (positive = OE wins),
        dGap = arm.gap - nog.gap (positive = OE adds forking). dVal = dTrain - dGap.
Units : mK = 1e-3 nat.
Flags : diverged  = final train window / min train window of last epoch >= 1.15
                    or final val window >= min val window of the run + 0.15
                    or E3 train window > E2 train window + 20 mK (partial blow-up)
        e1_dirty  = |dGap(E1)| > 25 mK (first pass should not fork; data-quality gate)
        dup_pool  = D > 2.385B on manifest v27a (shards 60/62/63 are byte copies of shard 1;
                    fixed in v27b, reruns tagged _p27b)
Pairing: same model, dn, seed, epochs, subset tag; nogram tls is a no-op (any tls).
         Prefer a nogram on the same host, then the same cluster family (a100 / h200);
         ophis vs 360-2 nogram differ by ~12 mK at E3 (104M dn20 s42), so cross-host is noisier.
Output: .tmp/harvest/v27_a100/paired.csv and a printed summary.
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/.tmp/harvest/v27_a100"
WIN = 60
NN = {"17m": 17e6, "35m": 35e6, "50m": 50e6, "104m": 104e6, "159m": 159e6, "200m": 200e6}
PAT = re.compile(
    r"v27_(?P<model>\d+m)_(?P<arm>exact|nogram|mh\d+k\d+k)_r(?P<rk>\d+)k_dn(?P<dn>[\d.]+)"
    r"_tls(?P<tls>[\d.]+)_s(?P<seed>\d+)_e(?P<ep>\d+)(?P<tag>_\S*?)?_fixed$")


def family(host):
    return "a100" if host.startswith("a100") else "h200"


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("val_loss") is not None and "step" in r:
                rows.append(r)
    return rows


def win(rows):
    rows = rows[-WIN:]
    n = len(rows)
    tr = sum(r["train_loss"] for r in rows) / n
    va = sum(r["val_loss"] for r in rows) / n
    return tr, va, va - tr


def epoch_windows(rows, ep_total):
    out = {}
    for e in range(1, ep_total + 1):
        er = [r for r in rows if r.get("epoch") == e]
        if len(er) >= WIN // 2:
            out[e] = win(er)
    return out


def collect():
    runs = []
    for d in glob.glob(os.path.join(ROOT, "*", "v27_*_fixed")):
        host = os.path.basename(os.path.dirname(d))
        rid = os.path.basename(d)
        m = PAT.match(rid)
        jl = os.path.join(d, "train_log.jsonl")
        if not m or not os.path.exists(jl):
            continue
        g = m.groupdict()
        rows = load(jl)
        if not rows:
            continue
        done = os.path.exists(os.path.join(d, "summary.json"))
        ep_total = int(g["ep"])
        # the final logged row carries epoch = ep_total + 1 (counter rolls at the last step)
        last_ep = min(rows[-1].get("epoch", 0), ep_total)
        complete_eps = ep_total if done else last_ep - 1
        ew = epoch_windows(rows, ep_total)
        ew = {e: v for e, v in ew.items() if e <= complete_eps}
        div = False
        if done and rows:
            last = [r for r in rows if r.get("epoch") == last_ep]
            if len(last) > WIN:
                trs = [sum(r["train_loss"] for r in last[i:i + WIN]) / WIN
                       for i in range(0, len(last) - WIN + 1, 10)]
                vas_all = [sum(r["val_loss"] for r in rows[i:i + WIN]) / WIN
                           for i in range(0, len(rows) - WIN + 1, 10)]
                if trs[-1] / min(trs) >= 1.15 or vas_all[-1] >= min(vas_all) + 0.15:
                    div = True
        # partial blow-up: healthy runs drop train loss E2->E3 (median -115 mK); a rise is not forking
        if ep_total >= 3 and 2 in ew and 3 in ew and ew[3][0] > ew[2][0] + 0.020:
            div = True
        runs.append(dict(host=host, fam=family(host), rid=rid, done=done, rows=len(rows),
                         last_step=rows[-1]["step"], last_ep=last_ep, ew=ew, div=div,
                         model=g["model"], arm=g["arm"], rk=int(g["rk"]),
                         dn=float(g["dn"]), tls=g["tls"], seed=int(g["seed"]),
                         ep=ep_total, tag=g["tag"] or ""))
    # de-duplicate the same run_id seen through several tione hosts (shared /home/tione):
    # keep the longer copy. The same run_id on different storages (e.g. a nogram re-run on
    # ophis and on 360-2) are distinct runs and are both kept.
    best = {}
    for r in runs:
        store = "tione" if r["host"].startswith("a100_t1") else (
            "ophis" if r["host"].startswith("ophis") else r["host"])
        k = (r["rid"], store)
        if k not in best or r["rows"] > best[k]["rows"]:
            best[k] = r
    return list(best.values())


def pair(runs):
    nog = {}
    for r in runs:
        if r["arm"] == "nogram":
            nog.setdefault((r["model"], r["dn"], r["seed"], r["ep"], r["tag"]), []).append(r)
    out = []
    for r in runs:
        if r["arm"] == "nogram":
            continue
        cands = nog.get((r["model"], r["dn"], r["seed"], r["ep"], r["tag"]), [])
        if not cands:
            continue
        cands = sorted(cands, key=lambda c: (c["host"] != r["host"], c["fam"] != r["fam"], -c["rows"]))
        n = cands[0]
        rec = dict(rid=r["rid"], host=r["host"], nog_host=n["host"], model=r["model"],
                   arm=r["arm"], rk=r["rk"], dn=r["dn"], tls=r["tls"], seed=r["seed"],
                   ep=r["ep"], tag=r["tag"], done=r["done"] and n["done"], diverged=r["div"],
                   cross_family=n["fam"] != r["fam"],
                   rn=r["rk"] * 1000 / (NN[r["model"]] / 1000),
                   tpp=r["dn"] * NN[r["model"]] / (r["rk"] * 1000))
        for e in range(1, r["ep"] + 1):
            if e in r["ew"] and e in n["ew"]:
                a, b = r["ew"][e], n["ew"][e]
                rec[f"dtrain{e}"] = round((b[0] - a[0]) * 1000, 2)
                rec[f"dval{e}"] = round((b[1] - a[1]) * 1000, 2)
                rec[f"dgap{e}"] = round((a[2] - b[2]) * 1000, 2)
                rec[f"val{e}"] = round(a[1], 5)
                rec[f"nogval{e}"] = round(b[1], 5)
                rec[f"noggap{e}"] = round(b[2] * 1000, 2)
        rec["e1_dirty"] = abs(rec.get("dgap1", 0)) > 25
        # manifest v27a pools > 2.385B tokens contain shards 60/62/63 = byte copies of shard 1
        rec["dup_pool"] = (r["dn"] * NN[r["model"]] > 2.385e9) and "_p27b" not in r["tag"]
        out.append(rec)
    return out


def main():
    runs = collect()
    paired = pair(runs)
    fields = ["rid", "host", "nog_host", "model", "arm", "rk", "rn", "dn", "tpp", "tls", "seed",
              "ep", "tag", "done", "diverged", "e1_dirty", "dup_pool", "cross_family"]
    for e in (1, 2, 3):
        fields += [f"dtrain{e}", f"dgap{e}", f"dval{e}", f"val{e}", f"nogval{e}", f"noggap{e}"]
    out = os.path.join(ROOT, "paired.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(paired, key=lambda r: (r["model"], r["tag"], r["dn"], r["rn"], r["tls"], r["seed"])):
            w.writerow(r)
    if "-q" in sys.argv:
        return
    print(f"{'model':5} {'tag':12} {'D/N':>5} {'R/N':>6} {'tls':>4} {'s':>3} | "
          f"{'dGap1':>6} {'dGap2':>6} {'dGap3':>6} | {'dTr3':>6} {'dVal3':>7} | flags")
    for r in sorted(paired, key=lambda r: (r["model"], r["tag"], r["tls"], r["dn"], r["rn"], r["seed"])):
        if "dval3" not in r and "dval1" not in r:
            continue
        fl = ("DIV " if r["diverged"] else "") + ("E1DIRTY " if r["e1_dirty"] else "") + \
             ("DUP " if r["dup_pool"] else "") + \
             ("xfam " if r["cross_family"] else "") + ("" if r["done"] else "run")
        g = lambda k: f"{r[k]:6.1f}" if k in r else "     -"
        print(f"{r['model']:5} {r['tag'] or '-':12} {r['dn']:5g} {r['rn']:6.2f} {r['tls']:>4} {r['seed']:>3} | "
              f"{g('dgap1')} {g('dgap2')} {g('dgap3')} | {g('dtrain3')} {r.get('dval3', float('nan')):7.1f} | {fl}")
    print(f"\n{len(runs)} runs, {len(paired)} paired -> {out}")


if __name__ == "__main__":
    main()
