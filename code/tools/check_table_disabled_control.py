#!/usr/bin/env python3
"""Verify exact backbone preservation and no-table forward equivalence."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import Config,NanoGPT
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=True,mmap=True)
    m=NanoGPT(Config(enable_bigram_ve=False,enable_trigram_ve=True,trigram_clean_table=1048576));m.load_state_dict(state,strict=True)
    markers=("value_embeds","bigram_ves","trigram_ves","ve_gate","bigram_gate","trigram_gate");zeroed=[]
    for name,param in m.named_parameters():
        if any(t in name for t in markers):
            with torch.no_grad():param.zero_()
            param.requires_grad_(False);zeroed.append(name)
    backbone={k:v for k,v in state.items() if not any(t in k for t in markers)}
    assert all(torch.equal(m.state_dict()[k],v) for k,v in backbone.items())
    n=NanoGPT(Config(enable_bigram_ve=False,enable_trigram_ve=False));n.load_state_dict(backbone,strict=True);m.eval();n.eval()
    x=torch.arange(64,dtype=torch.long).reshape(2,32)%8192
    with torch.inference_mode():err=(m(x)-n(x)).abs().max().item()
    assert err==0.0
    out={"backbone_exact":True,"zeroed_names":zeroed,"zeroed_params":sum(dict(m.named_parameters())[k].numel() for k in zeroed),"fp32_logits_max_error":err,"checkpoint_bytes":a.checkpoint.stat().st_size,"torch":torch.__version__}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out))
if __name__=="__main__":main()
