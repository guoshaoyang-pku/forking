# paper/ — 论文正文本仓库镜像

本目录把 **论文正文** 与 **实验证据** 放进同一个仓库：实验在 `code/` + `docs/`，论文在 `paper/`。
这样一条 claim 从正文到 run_id 不需要跨仓库跳转。

## 唯一权威源与镜像关系

| 角色 | 位置 | 权限 |
|---|---|---|
| 正文写作源（**source of truth**） | Overleaf 项目 `git.overleaf.com/6a9b071075d832e243e895a2`，本地 checkout `~/Desktop/workdir/overleaf-paper` | 协作者在 Overleaf 网页端编辑，只有这里能改正文 |
| 本仓库镜像（只读） | `paper/`（本目录） | 由脚本从 Overleaf 拉取，**不要在这里手改后当成正文** |
| 证据源 | `docs/figs/`（图）、`docs/plot_scripts/`（作图源码）、`docs/experiment-log.md`（run 登记） | 见 `agents.md` P7 |

`paper/` 是**镜像**，不是第二个写作入口。改正文请改 Overleaf，然后重跑同步脚本。

## 同步

```sh
# 把 Overleaf 当前工作树镜像进来
paper/scripts/sync_from_overleaf.sh

# 先 git fetch + ff-only 合并 Overleaf 的最新提交，再镜像
paper/scripts/sync_from_overleaf.sh --pull
```

脚本会把 `main.tex` / `paper.tex` / `sections/*.tex` / `references.bib` / 样式文件 / `figures/*`
拷进本目录，并写一份 `SOURCE_REVISION.txt` 记录这次镜像对应的 Overleaf commit。
非默认 checkout 位置用 `OVERLEAF_DIR=/path/to/overleaf-paper` 覆盖。

推送前按仓库规则先 `git fetch` 合并 Overleaf 的 `origin/main`（协作者会往 `main` 提交 `Update on Overleaf.`）。

## 编译

根文档是 **`main.tex`**（Overleaf 实际编译的文件，23 页）：

```sh
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper.tex` 是早期遗留的副本，**缺少 `wrapfig` / `subcaption` 和浮动体间距设置**，直接编译会报
`Environment wrapfigure undefined`。保留它只为忠实镜像 Overleaf 仓库，不要用它当根文档。

编译产物（`main.pdf`、`*.aux`、`*.log`、`*.bbl` 等）由 `paper/.gitignore` 排除，不进仓库。

## 目录

    paper/
    ├── main.tex                  根文档（权威编译入口）
    ├── paper.tex                 遗留副本，不要用作根文档
    ├── sections/                 正文与附录分章 tex
    ├── figures/                  论文用图的扁平构建镜像
    ├── references.bib            正文引用
    ├── iclr2027_conference.*     样式与 bst（vendored，保证可独立编译）
    ├── SOURCE_REVISION.txt       本次镜像对应的 Overleaf commit
    ├── FIGURE_PROVENANCE.md      每张图 → docs/figs 源 → plot script 的追溯表
    └── scripts/
        ├── sync_from_overleaf.sh     Overleaf → paper/ 镜像
        └── figure_provenance.py      刷新 FIGURE_PROVENANCE.md

## 图的追溯

论文需要 `figures/` 是一个扁平目录，而仓库的证据规范（`agents.md` P7）要求图进 `docs/figs/`、
作图源码进 `docs/plot_scripts/`。本目录因此是**构建镜像**，不是证据存放地。

`FIGURE_PROVENANCE.md` 给出每张论文图到仓库内同名源文件和作图脚本的映射：

```sh
python3 paper/scripts/figure_provenance.py --write
```

当前 47 张图中有 24 张能按同名回溯到 `docs/figs/` 或 `docs/appendices/`。
其余标 `**not found**` 的是更名过或早于当前证据布局的图；**引用前需要人工核对来源**，
不要在正文里凭文件名推断 run_id。

## 与实验侧的关系

- 正文里的数值仍然必须回到 `docs/experiment-log.md` 的 run_id / step / seed，以及
  `data/runs_fixed/<run_id>_fixed/`；`paper/` 不提供任何独立数值。
- 新图先在 `docs/figs/` 与 `docs/plot_scripts/` 落地并 commit，再上传到 Overleaf，
  最后用同步脚本回填到 `paper/figures/`。
- 论文语言与符号口径见 `docs/coordination/d1-paper-language-symbol-audit-20260913.md`。
