#!/usr/bin/env bash
# Sync the in-repo Overleaf clone at paper/ to the latest online revision.
#
# paper/ IS the git clone of the Overleaf project
# (https://git.overleaf.com/6a9b071075d832e243e895a2) — there is no external
# checkout anymore. Syncing = fetch + ff-only merge inside paper/.
#
# The external checkout ~/Desktop/workdir/overleaf-paper is DEPRECATED as of
# 2026-09-24 (moved to ~/Desktop/workdir/overleaf-paper.deprecated-20260924).
#
# Never edit paper/sections/*.tex here and push back to Overleaf by hand --
# edit on Overleaf, then re-run this script.
#
# Usage:
#   scripts/sync_from_overleaf.sh

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../paper" && pwd)"

cd "$PAPER_DIR"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "warning: paper/ has uncommitted changes; resolving them first" >&2
  exit 1
fi

echo "== fetching Overleaf =="
git fetch origin

if git merge --ff-only origin/main; then
  echo "== synced =="
else
  echo "error: local main diverged from origin/main; resolve manually" >&2
  exit 1
fi

echo "   HEAD: $(git log --oneline -1)"
