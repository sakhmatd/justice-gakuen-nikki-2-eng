# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Big glyph sheets for manual transcription. Usage: glyph_review.py start count out.png"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from PIL import Image, ImageDraw, ImageFont
rows = [l.rstrip('\n').split('\t') for l in open('data/ocr.tsv')]
start, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
small = ImageFont.load_default()
cache = {}; cols = 10; S = 3; cw, ch = 20*S+8, 20*S+14
sel = rows[start:start+n]
im = Image.new('L', (cols*cw, ((len(sel)+cols-1)//cols)*ch), 50)
dr = ImageDraw.Draw(im)
for k, r in enumerate(sel):
    fn, i = r[4], int(r[5])
    if fn not in cache: cache[fn] = open(os.path.join(UNPACKED, fn), 'rb').read()
    g = glyph(cache[fn], i)
    gi = Image.new('L', (20, 20)); gi.putdata([((g[q//2] >> (4*(q&1))) & 15)*17 for q in range(400)])
    x, y = (k % cols)*cw, (k // cols)*ch
    im.paste(gi.resize((20*S, 20*S), Image.NEAREST), (x+4, y+2))
    dr.text((x+4, y+20*S+2), str(start+k), font=small, fill=200)
im.save(out)
