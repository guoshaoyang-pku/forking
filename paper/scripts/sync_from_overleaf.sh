#!/usr/bin/env bash
# Mirror the live Overleaf manuscript into this repo's paper/ directory.
#
# The Overleaf project (git.overleaf.com/6a9b071075d832e243e895a2) is the
# source of truth for the manuscript text; paper/ is a read-only mirror kept
# in the experiment repo so that experiments and paper live in one place.
# Never edit paper/sections/*.tex here and push back to Overleaf by hand --
# edit on Overleaf, then re-run this script.
#
# Usage:
#   paper/scripts/sync_from_overleaf.sh            # copy current working tree
#   paper/scripts/sync_from_overleaf.sh --pull     # git fetch + ff-only merge first
#
# Override the source checkout with OVERLEAF_DIR=/path/to/overleaf-paper.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/paper"
SRC="${OVERLEAF_DIR:-/Users/guoshaoyang/Desktop/workdir/overleaf-paper}"

PULL=0
for arg in "$@"; do
  case "$arg" in
    --pull) PULL=1 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -d "$SRC/.git" ]; then
  echo "error: $SRC is not a git checkout (set OVERLEAF_DIR)" >&2
  exit 1
fi

if [ "$PULL" = "1" ]; then
  echo "== pulling Overleaf =="
  git -C "$SRC" fetch origin
  git -C "$SRC" merge --ff-only origin/main
fi

if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
  echo "warning: $SRC has uncommitted tracked changes; mirroring the working tree as-is" >&2
fi

echo "== mirroring $SRC -> $DEST =="
echo "   source HEAD: $(git -C "$SRC" log --oneline -1)"

# Manuscript sources. Styles and .bst are vendored so paper/ compiles standalone.
for f in main.tex paper.tex references.bib \
         iclr2027_conference.sty iclr2027_conference.bst \
         fancyhdr.sty natbib.sty; do
  if [ -f "$SRC/$f" ]; then
    cp -p "$SRC/$f" "$DEST/$f"
  else
    echo "warning: missing $f in $SRC" >&2
  fi
done

cp -p "$SRC"/sections/*.tex "$DEST/sections/"
cp -p "$SRC"/figures/* "$DEST/figures/"

# Record the mirrored revision so a reader can tell which Overleaf commit this is.
{
  echo "source_repo: $SRC"
  echo "source_remote: $(git -C "$SRC" remote get-url origin)"
  echo "source_commit: $(git -C "$SRC" rev-parse HEAD)"
  echo "source_commit_subject: $(git -C "$SRC" log --pretty=%s -1)"
  echo "source_commit_date: $(git -C "$SRC" log --pretty=%cI -1)"
  echo "mirrored_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$DEST/SOURCE_REVISION.txt"

echo "== done; review with: git -C $REPO_ROOT status --short paper =="
