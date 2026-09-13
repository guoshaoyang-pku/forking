#!/usr/bin/env python3
"""Emit a Feishu XML table covering every visual asset."""
import html, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REG=ROOT/'docs/figure-registry.json'
OUT=ROOT/'figure-assets-table.xml'
def main():
    data=json.loads(REG.read_text())
    scripts=[(p,p.read_text(errors='ignore')) for p in sorted((ROOT/'docs/plot_scripts').glob('*.py'))]
    rows=[]
    for x in data['assets']:
        name=Path(x['path']).name
        hits=[p.relative_to(ROOT).as_posix() for p,t in scripts if name in t or Path(name).stem in t]
        code='；'.join(hits[:4]) if hits else '待按输出文件名核实'
        rows.append('<tr><td>'+html.escape(x['number'])+'</td><td><code>'+html.escape(x['path'])+'</code></td><td>'+x['asset_type'].upper()+'</td><td><code>'+html.escape(code)+'</code></td><td>待按对应 run 核实</td><td>Git 资产</td></tr>')
    xml='<h2>八、全量图片资产表（稳定编号）</h2><p>本表覆盖仓库 docs/figs/ 下全部 '+str(len(rows))+' 个图片与交互图资产。正式图沿用“图章节-序号”；原文尚未编号的资产使用稳定资产号“图0-N”。编号记录在 docs/figure-registry.json，已有编号不会因重排改变。图片资产和绘图代码均保留 Git 路径，重画后可直接核查实物。</p><table><thead><tr><th>图号/资产号</th><th>图片资产（Git 路径）</th><th>格式</th><th>绘图代码位置</th><th>数据/集群</th><th>状态</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    OUT.write_text(xml)
    print(OUT, len(rows))
if __name__=='__main__': main()
