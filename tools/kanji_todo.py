# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Unread kanji glyphs: list, sheet and record readings.

  kanji_todo.py list [START COUNT]    unread glyphs (ordinals of data/ocr.tsv)
                                      with OCR top-5 and up to 3 contexts
  kanji_todo.py sheet START COUNT OUT.png
                                      3x glyph images of the same selection
  kanji_todo.py set ORD=R [ORD=R...]  record readings in data/kanji.tsv
                                      (R may be several characters)

START/COUNT index the list of unread glyphs, not the ordinals. Contexts
come from private/script (run tools/dump.py after recording readings).
Write with the lock: flock private/kanji.lock python3 tools/kanji_todo.py set ...
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
from mbms import glyph, UNPACKED

def rows():
    return [l.rstrip('\n').split('\t') for l in open('data/ocr.tsv', encoding='utf-8')]

def known():
    db = {}
    if os.path.exists('data/kanji.tsv'):
        for l in open('data/kanji.tsv', encoding='utf-8'):
            g, r, o = l.rstrip('\n').split('\t'); db[g] = (r, int(o))
    return db

def todo(sel):
    rs = rows(); db = known()
    un = [(k, r) for k, r in enumerate(rs) if r[0] not in db]
    if sel: un = un[int(sel[0]):int(sel[0]) + int(sel[1])]
    return un, len(rs), len(db)

def contexts(fn, i, n=3):
    name = fn.split('.')[0]; tag = '{g:%d}' % int(i); out = []
    p = f'private/script/{name}.txt'
    if not os.path.exists(p): return out
    for l in open(p, encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('##') or tag not in l: continue
        j = l.index(tag)
        c = l[max(0, j - 18):j + len(tag) + 18].replace(tag, '[*]')
        c = re.sub(r'\{g:\d+\}', '?', c.replace('{n}', ' ').replace('{p}', ' '))
        if c not in out: out.append(c)
        if len(out) == n: break
    return out

def cmd_list(sel):
    un, total, nk = todo(sel)
    print(f'-- {nk} of {total} read; listing {len(un)}')
    for k, r in un:
        print(f'{k}\t{r[3]}')
        for c in contexts(r[4], r[5]): print('\t  ' + c)

def cmd_sheet(sel, out):
    from PIL import Image, ImageDraw, ImageFont
    un, _, _ = todo(sel)
    small = ImageFont.load_default(); cache = {}; cols = 10; S = 3
    cw, ch = 20*S + 8, 20*S + 14
    im = Image.new('L', (cols*cw, max(1, (len(un) + cols - 1)//cols)*ch), 50)
    dr = ImageDraw.Draw(im)
    for n, (k, r) in enumerate(un):
        fn, i = r[4], int(r[5])
        if fn not in cache: cache[fn] = open(os.path.join(UNPACKED, fn), 'rb').read()
        g = glyph(cache[fn], i)
        gi = Image.new('L', (20, 20)); gi.putdata([((g[q//2] >> (4*(q&1))) & 15)*17 for q in range(400)])
        x, y = (n % cols)*cw, (n // cols)*ch
        im.paste(gi.resize((20*S, 20*S), Image.NEAREST), (x+4, y+2))
        dr.text((x+4, y+20*S+2), str(k), font=small, fill=200)
    im.save(out); print('wrote', out, len(un), 'glyphs')

def cmd_set(args):
    rs = rows(); db = known()
    for a in args:
        o, r = a.split('=', 1); o = int(o)
        db[rs[o][0]] = (r, o)
    with open('data/kanji.tsv', 'w', encoding='utf-8') as f:
        for g, (r, o) in sorted(db.items(), key=lambda x: x[1][1]):
            f.write(f'{g}\t{r}\t{o}\n')
    print('recorded', len(args), '; read', len(db), 'of', len(rs))

if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'list': cmd_list(a[1:3])
    elif a[0] == 'sheet': cmd_sheet(a[1:3], a[3])
    elif a[0] == 'set': cmd_set(a[1:])
    else: sys.exit(__doc__)
