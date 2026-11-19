# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Create tl/NAME.txt with every Japanese segment not yet listed in any
tl file (first occurrence wins). Usage: tl_init.py NAME [NAME...]"""
import sys, os, struct, glob
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from dump import kanji_db, charmap, decode
def jp_segments(name, db):
    d = open(os.path.join(UNPACKED, name + '.BIZ.lz'), 'rb').read(); m = charmap(d, db)
    for gi, to, ptrs in groups(d):
        starts = sorted(set(ptrs))
        for si, s in enumerate(starts):
            e = starts[si+1] if si + 1 < len(starts) else to
            jp = decode(struct.unpack_from('<%dH' % ((e - s)//2), d, s), m)
            while jp.endswith('{end}{end}'): jp = jp[:-5]
            yield gi, si, jp
if __name__ == '__main__':
    db = kanji_db(); os.makedirs('tl', exist_ok=True)
    known = set()
    for f in glob.glob('tl/*.txt'):
        known |= {l[4:].rstrip('\n') for l in open(f, encoding='utf-8') if l.startswith('JP: ')}
    for name in sys.argv[1:]:
        path = f'tl/{name}.txt'
        assert not os.path.exists(path), path + ' exists'
        n = 0
        with open(path, 'w', encoding='utf-8') as o:
            for gi, si, jp in jp_segments(name, db):
                if jp in known: continue
                known.add(jp); n += 1
                o.write(f'## {name} g{gi} s{si:04d}\nJP: {jp}\nEN: \n\n')
        print(name, n, 'new segments')
