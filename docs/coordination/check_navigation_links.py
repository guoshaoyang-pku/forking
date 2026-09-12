#!/usr/bin/env python3
"""Check relative Markdown links in the project's navigation documents."""
from pathlib import Path
import re

FILES = [
    Path('README.md'),
    Path('docs/resource-index.md'),
    Path('docs/coordination/README.md'),
    Path('docs/coordination/overnight-queue-20260913.md'),
    Path('docs/coordination/todo-completion-matrix-20260913.md'),
]
LINK = re.compile(r'\[[^\]]+\]\(([^)]+)\)')

checked = 0
missing = []
for source in FILES:
    for target in LINK.findall(source.read_text()):
        if target.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        checked += 1
        path = (source.parent / target.split('#', 1)[0]).resolve()
        if not path.exists():
            missing.append((str(source), target))
print(f'checked={checked} missing={len(missing)}')
for source, target in missing:
    print(f'{source}: {target}')
raise SystemExit(1 if missing else 0)
