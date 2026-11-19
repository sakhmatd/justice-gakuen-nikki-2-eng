# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
import sys, os, re
# runs of >=4 SJIS chars, mostly kana/kanji
pat = re.compile(rb'(?:[\x81-\x84\x88-\x9f\xe0-\xea][\x40-\x7e\x80-\xfc]){4,}')
def good(run):
    s = run.decode('cp932', errors='replace')
    kana = sum(1 for c in s if '぀' <= c <= 'ヿ')
    return kana >= max(2, len(s)//4), s
res = []
for root, _, files in os.walk(sys.argv[1]):
    for fn in files:
        p = os.path.join(root, fn); d = open(p,'rb').read()
        hits = [(m.start(), good(m.group())) for m in pat.finditer(d)]
        hits = [(o, s) for o,(g,s) in hits if g]
        if hits: res.append((sum(len(s) for _,s in hits), p, hits))
res.sort(reverse=True)
for tot, p, hits in res:
    print(f'{tot:7d} chars {len(hits):5d} runs {os.path.relpath(p, sys.argv[1])}  e.g. {hits[0][1][:30]!r}')
