# paper/ — 论文正文本仓库镜像（lab 侧说明）

> 本文件是 lab 仓库对 `paper/` 的管理说明（本文件）。`paper/` 本身是 Overleaf 项目的
> git clone，Overleaf 侧有自己的 `README.md`（讲 manuscript 构建），两者并存。

本目录把 **论文正文** 与 **实验证据** 放进同一个仓库：实验在 `code/` + `docs/`，论文在 `paper/`。
这样一条 claim 从正文到 run_id 不需要跨仓库跳转。

## 唯一权威源与镜像关系

| 角色 | 位置 | 权限 |
|---|---|---|
| 正文写作源（**source of truth**） | Overleaf 项目 `git.overleaf.com/6a9b071075d832e243e895a2` | 协作者在 Overleaf 网页端编辑，只有这里能改正文 |
| 仓库内 clone（只读） | `paper/`（自带 `.git`） | 只通过 `git fetch` + ff-only merge 同步，**不要在这里手改后当成正文** |
| 证据源 | `docs/figs/`（图）、`docs/plot_scripts/`（作图源码）、`docs/experiment-log.md`（run 登记） | 见 `agents.md` P7 |

`paper/` 是 **Overleaf 项目的完整 git clone**（不再是外部 checkout 的拷贝镜像）。
版本历史直接看 `git -C paper log`；外部 checkout `~/Desktop/workdir/overleaf-paper`
已于 2026-09-24 废弃（备份在 `~/Desktop/workdir/overleaf-paper.deprecated-20260924`）。

> 注意：由于 `paper/` 含自己的 `.git`，lab 仓库**不能**跟踪其内部任何文件
> （git 只能把它当作 embedded repo）。lab 侧文件因此全部放在 `paper/` 之外：
> 同步/追溯脚本在 `scripts/`，本说明在 `docs/notes/method/paper-mirror.md`。
> Overleaf 侧文件整体由 `paper/` 的 git 历史管理（根 `.gitignore` 里 `paper/`）。

## 同步

```sh
# 在 paper/ 内 fetch + ff-only merge 到 Overleaf 最新提交
scripts/sync_from_overleaf.sh
```

Overleaf 的版本号即 `paper/` 的 HEAD commit（如 `a8dbc5c`），不再维护
`SOURCE_REVISION.txt`。协作者在网页端的编辑会以 `Update on Overleaf.` 提交出现在
`origin/main` 上，pull 即可拿到。

## 编译

根文档是 **`main.tex`**（Overleaf 实际编译的文件，23 页）：

```sh
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper.tex` 是早期遗留的副本，**缺少 `wrapfig` / `subcaption` 和浮动体间距设置**，直接编译会报
`Environment wrapfigure undefined`。保留它只为忠实镜像 Overleaf 仓库，不要用它当根文档。

编译产物（`main.pdf`、`*.aux`、`*.log`、`*.bbl` 等）由 Overleaf 侧 `.gitignore` +
`paper/.git/info/exclude` 排除，不进任何仓库。

## 目录

    paper/                        Overleaf 项目 git clone（lab 仓库不跟踪其内部文件）
    ├── main.tex                  根文档（权威编译入口，Overleaf 侧）
    ├── README.md                 Overleaf 侧 README（manuscript 构建）
    ├── paper.tex                 遗留副本，不要用作根文档
    ├── sections/                 正文与附录分章 tex（Overleaf 侧）
    ├── figures/                  论文用图（Overleaf 侧）
    ├── references.bib            正文引用（Overleaf 侧）
    └── iclr2027_conference.*     样式与 bst（Overleaf 侧）

    scripts/                      lab 侧（lab 仓库跟踪）
    ├── sync_from_overleaf.sh     paper/ 内 fetch + ff-only merge
    ├── figure_provenance.py      刷新 FIGURE_PROVENANCE.md
    └── FIGURE_PROVENANCE.md      每张图 → docs/figs 源 → plot script 的追溯表

    docs/notes/method/paper-mirror.md   本文件

## 图的追溯

论文需要 `figures/` 是一个扁平目录，而仓库的证据规范（`agents.md` P7）要求图进 `docs/figs/`、
作图源码进 `docs/plot_scripts/`。`paper/figures/` 因此是**构建镜像**，不是证据存放地。

`scripts/FIGURE_PROVENANCE.md` 给出每张论文图到仓库内同名源文件和作图脚本的映射：

```sh
python3 scripts/figure_provenance.py --write
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
