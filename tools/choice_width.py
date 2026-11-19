# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""List likely choice menus (EN with one line per JP line, no quotes, no colour
header, no {sp} table) whose longest option is wider than LIMIT cells (default 18).
build_mbms.py wraps any line over 20 cells silently, which splits an
option over two lines and breaks the choice cursor.
Usage: choice_width.py [LIMIT]"""
import sys, os, re, glob
sys.path.insert(0, os.path.dirname(__file__))
import build_mbms as B
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 18
for f in sorted(glob.glob('tl/MG*.txt')):
    jp = None
    for l in open(f, encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('## '): sid = l[3:]
        elif l.startswith('JP: '): jp = l[4:]
        elif l.startswith('EN: ') and jp:
            en = l[4:]
            if '{n}' not in en or '"' in en or '{sp}' in en or '  ' in en:
                continue
            if en.startswith('{col:'): continue
            jpages = [p for p in re.split(r'\{p\}|\{end\}', jp) if p]
            for k, page in enumerate(p for p in re.split(r'\{p\}|\{end\}', en) if p):
                opts = page.split('{n}')
                if len(opts) < 2: continue
                # a choice keeps one EN line per JP line
                if k >= len(jpages) or jpages[k].count('{n}') != page.count('{n}'): continue
                if any(o.rstrip().endswith(('.', '!', '?', '...')) and len(o) > 30 for o in opts): continue
                w = max(B.cells(o) for o in opts)
                if w > limit:
                    print(f'{sid}  {w:2d}  {page}')
