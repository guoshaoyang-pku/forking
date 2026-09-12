#!/usr/bin/env python3
"""Generate an evidence-oriented HTML/JSON index of every figure."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import html, json, re, subprocess

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "docs" / "figs"
SCRIPT_ROOT = ROOT / "docs" / "plot_scripts"
EXTS = {".png", ".svg", ".html", ".jpg", ".jpeg", ".webp"}
LABELS = {"main":"主线现象与干预", "theory":"理论、缺失质量与动力学", "epoch_scale":"epoch / 数据规模标度", "toy":"toy / 合成数据", "table_opt":"优化器与学习率消融", "short_epoch_b2":"历史 beta₂ / short-epoch", "root":"早期探针与 provenance"}
CURRENT = ("fig_v5_128x_", "fig_v5_s1_", "fig_v5_good_turing", "fig_v5_val_damage", "fig_v5_f_by_pass", "fig_v5_table_cosine", "fig_v5_dilution", "fig_v5_epoch_kernel", "fig_v5_netval", "fig_ls20ep_", "fig_ngram5_order5_", "fig_freq_", "fig_gap_vs_frequency")
OLD = ("fig_cmp_", "fig_probe_", "fig_uniform", "fig_online_", "fig_dtype_", "fig_fp32_", "fig_lrdiag_", "_existing", "_2x", "short_epoch_b2", "fig_table_opt_2x", "fig_beta2_ablation", "fig_beta2_curves")

def rel(p): return p.relative_to(ROOT).as_posix()
def category(p):
    r = p.relative_to(FIG_ROOT)
    return r.parts[0] if len(r.parts) > 1 else "root"

def document_refs():
    out = {}
    for d in list((ROOT / "docs").rglob("*.html")) + list((ROOT / "docs").rglob("*.md")):
        if d.name == "figure-index.html": continue
        text = d.read_text(errors="ignore")
        for m in re.finditer(r"(?:figs/|docs/figs/)([^\"'<> )]+)", text):
            key = m.group(1).lstrip("/")
            if key.startswith("figs/"): key = key[5:]
            out.setdefault(key, []).append(rel(d))
    return {k: sorted(set(v)) for k,v in out.items()}

def script_rows():
    out = []
    for p in sorted(SCRIPT_ROOT.glob("*.py")):
        text = p.read_text(errors="ignore")
        paths = sorted(set(re.findall(r"(?:data|tasks)/[A-Za-z0-9_./{}*,-]+", text)))
        clusters = []
        if re.search(r"ophis-gpu|ophis_gpu|/data3/", text): clusters.append("ophis-gpu")
        if re.search(r"360-1|3601|g3601", text): clusters.append("360-1")
        if re.search(r"360-2|3602|g3602", text): clusters.append("360-2")
        out.append({"path":rel(p), "name":p.name, "data_paths":paths[:40], "clusters":clusters or (["本地镜像 / 原始集群需按 run 核对"] if paths else ["本地离线 / 未声明集群"])})
    return out

def main():
    refs, scripts = document_refs(), script_rows()
    figures = []
    for p in sorted(x for x in FIG_ROOT.rglob("*") if x.is_file() and x.suffix.lower() in EXTS):
        key, name = p.relative_to(FIG_ROOT).as_posix(), p.name
        used = refs.get(key, [])
        if any(h in name for h in OLD) or category(p) in {"table_opt", "short_epoch_b2"}:
            status = "outdated"
        elif used or any(h in name for h in CURRENT):
            status = "referenced"
        else:
            status = "orphan-review"
        hits = []
        for s in scripts:
            st = (ROOT / s["path"]).read_text(errors="ignore")
            if name in st or p.stem in st: hits.append(s)
        figures.append({"path":rel(p), "file":name, "format":p.suffix[1:].lower(), "bytes":p.stat().st_size, "category":category(p), "status":status, "scripts":sorted({s["path"] for s in hits}), "data_paths":sorted({x for s in hits for x in s["data_paths"]}) or ["未在脚本中声明；查看对应 run 的 summary.json / JSONL"], "clusters":sorted({x for s in hits for x in s["clusters"]}) or ["本地 docs/figs；原始集群需按 run 核对"], "used_by":used})
    payload = {"generated_at":datetime.now().astimezone().isoformat(timespec="seconds"), "commit":subprocess.check_output(["git","rev-parse","--short","HEAD"], cwd=ROOT, text=True).strip(), "counts":dict(Counter(x["status"] for x in figures)), "figures":figures, "scripts":scripts, "category_labels":LABELS}
    (ROOT / "docs/figure-index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    cards=[]
    for x in figures:
        src=x["path"].replace("docs/", "", 1)
        thumb=("<iframe loading='lazy' src='"+html.escape(src)+"'></iframe>") if x["format"]=="html" else ("<img loading='lazy' src='"+html.escape(src)+"' alt='"+html.escape(x["file"])+"'>")
        chips="".join("<span class='chip "+html.escape(v if v in ('referenced','outdated','orphan-review') else '')+"'>"+html.escape(v)+"</span>" for v in [x["status"]]+x["clusters"])
        links=" · ".join("<a href='../"+html.escape(s)+"'>"+html.escape(Path(s).name)+"</a>" for s in x["scripts"]) or "未找到精确脚本引用"
        detail="<br>".join(html.escape(v) for v in x["data_paths"])
        search=html.escape(json.dumps(x, ensure_ascii=False))
        cards.append("<article class='card' data-search='"+search+"' data-cat='"+html.escape(x["category"])+"' data-status='"+x["status"]+"'><div class='thumb'>"+thumb+"</div><div class='body'><b>"+html.escape(x["file"])+"</b><div class='meta'>"+html.escape(LABELS.get(x["category"],x["category"]))+" · "+x["format"].upper()+" · "+str(round(x["bytes"]/1024))+" KB</div><div class='chips'>"+chips+"</div><div class='links'><a href='"+html.escape(src)+"' target='_blank'>打开原图</a> · "+links+"</div><details><summary>数据 / 文档引用</summary><p><code>"+detail+"</code></p><p>"+html.escape(" · ".join(x["used_by"]) if x["used_by"] else "未在本地文档引用")+"</p></details></div></article>")
    counts=payload["counts"]
    stat=[(len(figures),"可视化文件"),(len(scripts),"绘图脚本"),(counts.get("referenced",0),"referenced"),(counts.get("orphan-review",0),"待复核"),(f"{sum(x['bytes'] for x in figures)/1e6:.1f} MB","图文件体积")]
    opts="".join("<option value='"+k+"'>"+html.escape(LABELS[k])+"</option>" for k in LABELS if any(x["category"]==k for x in figures))
    css=".hidden{display:none!important}body{margin:0;background:radial-gradient(circle at 15% 0,#1c3445,#0b1118 42%);color:#eaf1f7;font:14px/1.55 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}main{max-width:1500px;margin:auto;padding:38px 28px 70px}h1{font-size:34px;margin:0}.sub,.meta{color:#91a4b5}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:22px 0}.stat,.toolbar,.card{background:linear-gradient(145deg,#152634,#101923);border:1px solid #2a4050;border-radius:14px}.stat{padding:13px}.stat b{display:block;font-size:25px;color:#6bd7e8}.stat span{color:#91a4b5;font-size:12px}.toolbar{padding:12px;display:flex;gap:9px;position:sticky;top:8px;z-index:3;backdrop-filter:blur(12px)}input,select,button{background:#0c151e;color:#eaf1f7;border:1px solid #2a4050;border-radius:8px;padding:9px 11px}input{flex:1}button{cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:14px}.card{overflow:hidden}.thumb{height:170px;background:#091017;display:flex;align-items:center;justify-content:center}.thumb img{width:100%;height:100%;object-fit:contain}.thumb iframe{width:100%;height:100%;border:0;background:#fff}.body{padding:13px}.body>b{word-break:break-word}.chips{display:flex;gap:5px;flex-wrap:wrap;margin:9px 0}.chip{font-size:11px;border-radius:99px;padding:2px 7px;background:#203545;color:#6bd7e8}.chip.current{color:#8ad6a1}.chip.outdated{color:#ef8c83}.chip.orphan-review{color:#e7bd74}a{color:#6bd7e8;text-decoration:none;font-size:12px}details{color:#91a4b5;font-size:12px}code{word-break:break-all}@media(max-width:700px){main{padding:24px 14px}.stats{grid-template-columns:repeat(2,1fr)}.toolbar{flex-wrap:wrap}input{flex-basis:100%}}"
    stat_html="".join("<div class='stat'><b>"+str(v)+"</b><span>"+html.escape(k)+"</span></div>" for v,k in stat)
    page="<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Figure Index · ngram-gap-lab</title><style>"+css+"</style></head><body><main><h1>Figure Index</h1><p class='sub'>把论文叙事、图片、绘图代码、数据产物和集群坐标放在同一条可审计路径上。索引由仓库内容生成；未声明集群的图会明确提示按 run 核对。</p><p class='meta'>commit "+payload["commit"]+" · "+payload["generated_at"]+" · 实测 "+str(len(figures))+" 个可视化文件</p><div class='stats'>"+stat_html+"</div><div class='toolbar'><input id='q' placeholder='搜索文件名、脚本、run、数据或集群…'><select id='cat'><option value=''>全部上下文</option>"+opts+"</select><select id='status'><option value=''>全部状态</option><option>referenced</option><option>outdated</option><option>orphan-review</option></select><button id='reset'>重置</button></div><h2>按论文上下文浏览</h2><section class='grid' id='grid'>"+"".join(cards)+"</section><p class='sub'>数据口径：训练产物优先使用 <code>data/runs_fixed/&lt;run_id&gt;_fixed/</code>；novel 没有 train loss，不在 gap 曲线中伪造。状态是整理建议，结论仍以 experiment-lines / experiment-log / claims-ledger 为准。</p></main><script>const q=document.querySelector('#q'),cat=document.querySelector('#cat'),st=document.querySelector('#status');function go(){const z=q.value.toLowerCase();document.querySelectorAll('.card').forEach(x=>x.classList.toggle('hidden',!!(z&&!x.dataset.search.toLowerCase().includes(z))||!!(cat.value&&x.dataset.cat!==cat.value)||!!(st.value&&x.dataset.status!==st.value)))}q.oninput=cat.onchange=st.onchange=go;document.querySelector('#reset').onclick=()=>{q.value=cat.value=st.value='';go()};</script></body></html>"
    (ROOT / "docs/figure-index.html").write_text(page)
    print("wrote docs/figure-index.html and docs/figure-index.json: "+str(len(figures))+" figures, "+str(len(scripts))+" scripts")
if __name__ == "__main__": main()
