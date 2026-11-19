# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Show contexts for glyph ordinals: ctx.py ORD [ORD...]"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
from dump import *
want = set(int(a) for a in sys.argv[1:])
db = {}
for l in open('data/kanji.tsv'):
    g, r, o = l.rstrip('\n').split('\t')
    db[g] = f'<<{r}#{o}>>' if int(o) in want else r
seen = {}
for fn in files():
    d = open(fn, 'rb').read(); m = charmap(d, db)
    for gi, si, s, e, ids in segments(d):
        t = decode(struct.unpack_from('<%dH' % ((e-s)//2), d, s), m)
        i = t.find('<<')
        while i >= 0:
            j = t.index('>>', i); key = t[i+2:j]
            c = t[max(0, i-15):j+15]
            seen.setdefault(key, set()).add(c)
            i = t.find('<<', j)
for k, v in seen.items():
    print('==', k, len(v))
    for c in list(v)[:5]: print('   ', c)
