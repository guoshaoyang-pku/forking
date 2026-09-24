#!/usr/bin/env bash
# Push local commits in paper/ (the Overleaf git clone) back to Overleaf,
# staying compatible with concurrent online edits.
#
#   1. refuse if paper/ has uncommitted tracked changes (commit first)
#   2. fetch Overleaf; merge origin/main into local main (never rebase, never force)
#   3. on merge conflict: abort the merge and stop (resolve by hand)
#   4. compile paper.tex with latexmk; stop if the build fails
#   5. plain `git push origin main` (non-fast-forward is rejected by the server,
#      so an online edit that lands mid-script can never be overwritten)
#
# Usage: scripts/push_to_overleaf.sh ["optional merge message"]

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../paper" && pwd)"
cd "$PAPER_DIR"
export GIT_TERMINAL_PROMPT=0

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "error: paper/ has uncommitted tracked changes; commit them first" >&2
  exit 1
fi
if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  echo "error: paper/ is not on main" >&2
  exit 1
fi

git fetch -q origin
if ! git merge-base --is-ancestor origin/main HEAD; then
  msg="${1:-Merge Overleaf online edits before push}"
  if ! git merge --no-edit -m "$msg" origin/main; then
    git merge --abort || true
    echo "error: conflict with online Overleaf edits; resolve manually" >&2
    exit 1
  fi
fi

if [ "$(git rev-list --count origin/main..HEAD)" = "0" ]; then
  echo "nothing to push; HEAD == origin/main ($(git log --oneline -1))"
  exit 0
fi

if ! latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/tmp/overleaf_prepush_build.log 2>&1; then
  echo "error: paper.tex does not compile; see /tmp/overleaf_prepush_build.log" >&2
  exit 1
fi
undef="$(grep -c 'undefined' paper.log || true)"
echo "build ok (undefined refs/cites: ${undef})"

git push origin main
echo "pushed: $(git log --oneline -1 origin/main)"
