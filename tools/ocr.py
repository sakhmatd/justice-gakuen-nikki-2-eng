# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Match unique message-file kanji glyphs against system font renderings.

Writes data/ocr.tsv: gid, best char, score, top-5 alternatives, example (file, index).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import fontpath

def g2arr(g):
    a = np.zeros(400, np.float32)
    for k in range(400): a[k] = (g[k//2] >> (4*(k&1))) & 15
    return a.reshape(20, 20) / 15.0

def norm(a, size=24):
    """Crop to bbox, scale into size x size keeping aspect, blur, zero-mean unit-norm."""
    ys, xs = np.nonzero(a > 0.3)
    if len(xs) == 0: return None
    a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    im = Image.fromarray((a*255).astype(np.uint8))
    h, w = a.shape; s = size / max(h, w)
    im = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.BILINEAR)
    c = Image.new('L', (size, size)); c.paste(im, ((size-im.width)//2, (size-im.height)//2))
    c = c.filter(ImageFilter.GaussianBlur(1.2))
    v = np.asarray(c, np.float32).ravel(); v -= v.mean(); n = np.linalg.norm(v)
    return v / n if n else None

def candidates():
    chars = []
    for hi in range(0x88, 0xa0):
        for lo in range(0x40, 0xfd):
            try: c = bytes([hi, lo]).decode('cp932')
            except UnicodeDecodeError: continue
            if '一' <= c <= '鿿' or c == '々': chars.append(c)
    for hi in range(0xe0, 0xeb):
        for lo in range(0x40, 0xfd):
            try: c = bytes([hi, lo]).decode('cp932')
            except UnicodeDecodeError: continue
            if '一' <= c <= '鿿': chars.append(c)
    return sorted(set(chars))

def render_set(chars, fonts):
    vecs = []; labels = []
    for fp, idx, px in fonts:
        f = fontpath.truetype(fp, px, index=idx)
        for c in chars:
            im = Image.new('L', (40, 40)); ImageDraw.Draw(im).text((4, 4), c, font=f, fill=255)
            v = norm(np.asarray(im, np.float32) / 255)
            if v is not None: vecs.append(v); labels.append(c)
    return np.array(vecs), labels

if __name__ == '__main__':
    uniq = {}
    for fn in files():
        d = open(fn, 'rb').read()
        for i in range(FIRST, font_count(d)):
            g = glyph(d, i)
            if any(g): uniq.setdefault(gid(g), (g, os.path.basename(fn), i))
    print('unique kanji glyphs', len(uniq), file=sys.stderr)
    chars = candidates(); print('candidates', len(chars), file=sys.stderr)
    fonts = [('msgothic.ttc', 0, 20), ('YuGothB.ttc', 0, 20),
             ('YuGothM.ttc', 0, 20)]
    M, labels = render_set(chars, fonts)
    with open('data/ocr.tsv', 'w') as o:
        for k, (g, fn, i) in uniq.items():
            v = norm(g2arr(g))
            if v is None: continue
            sc = M @ v
            order = np.argsort(-sc)
            seen = []; alts = []
            for j in order:
                if labels[j] in seen: continue
                seen.append(labels[j]); alts.append(f'{labels[j]}{sc[j]:.2f}')
                if len(seen) == 5: break
            o.write(f'{k}\t{seen[0]}\t{sc[order[0]]:.3f}\t{" ".join(alts)}\t{fn}\t{i}\n')
