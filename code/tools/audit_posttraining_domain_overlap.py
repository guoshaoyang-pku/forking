#!/usr/bin/env python3
"""CPU-only token-distribution and sampled exact-context overlap diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_rows(root,shards,indices):
    sizes=[(root/f'shard_{sid:05d}.bin').stat().st_size//2//2049 for sid in shards]
    offsets=np.cumsum([0]+sizes)
    result=[]
    for i,sid in enumerate(shards):
        local=indices[(indices>=offsets[i])&(indices<offsets[i+1])]-offsets[i]
        if len(local):
            buf=np.memmap(root/f'shard_{sid:05d}.bin',dtype=np.uint16,mode='r').reshape(-1,2049)
            result.append(np.asarray(buf[local],dtype=np.uint64))
    return np.concatenate(result)


def keys(rows,n):
    # Input contexts entirely inside each 2048-token input; exclude padding and target tail.
    x=rows[:,:2048];out=np.zeros((len(x),2048-n+1),dtype=np.uint64)
    for j in range(n):out=out*8192+x[:,j:j+2048-n+1]
    return out.ravel()


def js(p,q):
    p=p/p.sum();q=q/q.sum();m=(p+q)/2
    def kl(x):
        mask=x>0;return float(np.sum(x[mask]*np.log(x[mask]/m[mask])))
    return (kl(p)+kl(q))/2


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    base=a.root/'tokenized_epoch20';n=6740*72
    idx=np.rint(np.linspace(0,n-1,4096)).astype(np.int64)
    pre=load_rows(base,list(range(21)),idx)
    assert len(pre)==4096
    # Full unique pretraining pool target-token histogram, not repeated three times.
    hist=np.zeros(8192,dtype=np.int64);remaining=n
    for sid in range(21):
        buf=np.memmap(base/f'shard_{sid:05d}.bin',dtype=np.uint16,mode='r').reshape(-1,2049);take=min(remaining,len(buf))
        for start in range(0,take,2048):hist+=np.bincount(np.asarray(buf[start:min(start+2048,take),1:],dtype=np.int64).ravel(),minlength=8192)
        remaining-=take
        if not remaining:break
    assert remaining==0 and hist.sum()==993853440
    vocab={n:np.unique(keys(pre,n)) for n in [2,3]};del pre
    result={'pretrain_pool_target_tokens':int(hist.sum()),'pretrain_sample_rows':4096,'pretrain_sample_indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),
      'context_metric':'Exact n-gram identity, collision-free base8192 encoding; token-weighted occurrence coverage against 4096 uniformly sampled pretraining rows. This is sampled coverage, not full-pretraining support or table-row hits.',
      'unigram_metric':'JS divergence in nats, full pretraining pool vs full downstream packed train/val target-token histograms; distributional description, not causal evidence.',
      'pretrain_sample_distinct_contexts':{str(k):len(v) for k,v in vocab.items()},'sources':[]}
    specs=[('id','tokenized_epoch20',[21],[22,23]),('magicoder','ood_tokenized_magicoder_fixtok',[0],[1]),('openr1','ood_tokenized_openr1_10shards_fixtok',[0],[1]),('rlvr','rlvr_tokenized/math_fixtok',[0],[1])]
    for source,folder,train,val in specs:
        row={'source':source}
        for split,shards,count in [('train',train,23976),('val',val,288)]:
            rr=load_rows(a.root/folder,shards,np.arange(count));hh=np.bincount(rr[:,1:].astype(np.int64).ravel(),minlength=8192)
            probe=rr[np.rint(np.linspace(0,count-1,288)).astype(np.int64)]
            row[split]={'target_tokens':int(hh.sum()),'unigram_js_from_pretrain':js(hist,hh),'unigram_unseen_pretrain_fraction':float(hh[hist==0].sum()/hh.sum())}
            for order in [2,3]:
                kk=keys(probe,order);v=vocab[order];ii=np.searchsorted(v,kk);matched=(ii<len(v))&(v[np.minimum(ii,len(v)-1)]==kk)
                row[split][f'sampled_pretrain_{order}gram_coverage']=float(matched.mean())
            del rr
        result['sources'].append(row);print(json.dumps(row),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+chr(10))


if __name__=='__main__':main()
