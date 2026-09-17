#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import v5_style as S
ROOT=Path(__file__).resolve().parents[2]
SNAP=ROOT/"docs/appendices/epoch20_20260914/live_snapshot.json"
OUT=ROOT/"docs/figs/main/fig_s1_epoch20_epoch1_live"
def main():
 snap=json.loads(SNAP.read_text()); points=[]
 for r in snap["runs"]:
  rows=r.get("rows",[])
  if not rows: continue
  rid=r["run_id"]; label=rid.split("_tri_",1)[1].split("xL4",1)[0]
  ep1=[x for x in rows if x.get("epoch")==1]
  if not ep1: continue
  e=ep1[-1]; points.append((float(label.replace("p",".")),float(e["gap"]),int(e["step"]),rid))
 points.sort(); S.apply_style(); fig,ax=plt.subplots(figsize=(9.2,5.2))
 x=[p[0] for p in points]; y=[p[1] for p in points]
 ax.plot(x,y,color="#2d6f9f",marker="o",lw=1.5,label="epoch 1 boundary")
 for xi,yi,step,_ in points: ax.annotate(f"{yi:+.2f}",(xi,yi),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7.5)
 ax.axhline(0,color="#9aa3ad",lw=.8); ax.set_xscale("log",base=2)
 ax.set_xlabel("epoch length (xL4; L4 = 337 device batches)"); ax.set_ylabel("online gap = fixed val - online train")
 ax.set_title(f"V5 S1 epoch-length extension: latest epoch-1 gaps ({len(points)}/20 points)"); ax.grid(alpha=.22); ax.legend()
 ax.text(.02,.03,"Trigram-only; clean R=2^20; table LR 128x; seed 42; full-163 pool. 15x and 20x pending.",transform=ax.transAxes,fontsize=8,color="#475569")
 fig.tight_layout(); png,svg=S.save(fig,OUT.parent,OUT.name); print(png); print(svg)
 csv=OUT.with_suffix(".csv"); csv.write_text("multiplier,epoch,step,gap,run_id\n"+"\n".join(f"{x},1,{step},{gap:.10g},{rid}" for x,gap,step,rid in points)+"\n"); print(csv)
if __name__=="__main__": main()
