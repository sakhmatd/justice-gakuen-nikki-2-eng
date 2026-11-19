# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Record manual glyph readings. Usage: kanji_add.py START 'chars...'
Each char maps to consecutive ordinals of data/ocr.tsv; [xyz] = one glyph reading 'xyz'.
Writes/updates data/kanji.tsv (gid<TAB>reading<TAB>ordinal)."""
import sys, re, os
rows = [l.rstrip('\n').split('\t') for l in open('data/ocr.tsv')]
start = int(sys.argv[1]); toks = re.findall(r'\[[^\]]*\]|.', sys.argv[2].replace(' ', ''))
toks = [t[1:-1] if t.startswith('[') else t for t in toks]
db = {}
if os.path.exists('data/kanji.tsv'):
    for l in open('data/kanji.tsv'):
        g, r, o = l.rstrip('\n').split('\t'); db[g] = (r, int(o))
for k, t in enumerate(toks):
    db[rows[start+k][0]] = (t, start+k)
with open('data/kanji.tsv', 'w') as f:
    for g, (r, o) in sorted(db.items(), key=lambda x: x[1][1]): f.write(f'{g}\t{r}\t{o}\n')
print('added', len(toks), 'total', len(db))
