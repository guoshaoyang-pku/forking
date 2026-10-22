#!/usr/bin/env python3
# v27 pool duplicate scan (2026-10-23): 32-token window hashes every 97th sample, each shard vs all earlier shards in train_order.
# Found shards 60/62/63 = byte copies of shard 1 (manifest v27b drops them). Run on a host holding data/tokenized_v27.
import numpy as np, json
D="/data1/guoshaoyang/ngram-gap-lab-v27/data/tokenized_v27"
m=json.load(open(D+"/manifest.json")); order=m["train_order"]
rng=np.random.default_rng(0); MUL=rng.integers(1,2**63,size=32,dtype=np.uint64)|np.uint64(1)
offs=list(range(0,2017,128))
def wh(s):
    a=np.memmap(f"{D}/shard_{s:05d}.bin",dtype=np.uint16,mode="r"); n=len(a)//2049
    c=np.asarray(a[:n*2049]).reshape(n,2049)
    return np.stack([(c[:,o:o+32].astype(np.uint64)*MUL).sum(1) for o in offs],1), n
hs={}; cum=0
seen_owner={}  # hash -> first shard
res=[]
for i,s in enumerate(order):
    h,n=wh(s)
    flat=h.ravel()
    if seen_owner:
        keys=np.fromiter(seen_owner.keys(),dtype=np.uint64) if i<2 else None
    hit_owner=[seen_owner.get(int(x)) for x in flat[::97]]
    hits=[o for o in hit_owner if o is not None]
    frac=len(hits)/len(hit_owner)
    from collections import Counter
    top=Counter(hits).most_common(2)
    res.append((i,s,n,cum,frac,top))
    if frac>0.02: print(f"idx {i:3d} shard {s:3d} chunks {n:6d} start {cum/1e9:.3f}B step {cum/147456:.0f} dup-frac {frac:.3f} owners {top}", flush=True)
    for x in flat[::97]:
        seen_owner.setdefault(int(x), s)
    cum+=n*2048
print("scanned",len(order),"shards; total",cum/1e9,"B")
