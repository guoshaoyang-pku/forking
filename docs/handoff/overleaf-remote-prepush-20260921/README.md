# Overleaf 远程 pre-push 存档（2026-09-21）

推送本地 §3–§4 压缩版之前，对 overleaf-paper `origin/main` 的状态做的完整存档。

- 远程 commit：`b08dd2b`（haoranzhao "Update on Overleaf."，appendix math notation）
- 存档时间：2026-09-21 17:27（本地编译）
- 内容：
  - `main-remote-b08dd2b.pdf`：远程 tex 在本地 `latexmk -pdf` 的编译产物（24 页口径，§3–§4 为压缩前布局）
  - `tex/`：远程全部 tex 源（`main.tex`、`paper.tex`、`sections/*.tex`、`references.bib`、样式与 bst）
- 恢复方式：Overleaf 仓库内亦可用 `git show b08dd2b:<path>` 取任意文件；本目录仅为离线快照。
- 对应本地推送：overleaf-paper `83ad3eb`（含 `daea3c0` 压缩、`db30243` 合并、`83ad3eb` 三图一排/Fig2 侧栏/§4.1 压缩）。
