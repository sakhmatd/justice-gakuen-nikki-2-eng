# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Review sheet: game glyph next to OCR guess. Usage: ocr_sheet.py start count out.png"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from PIL import Image, ImageDraw, ImageFont
import fontpath
rows = [l.rstrip('\n').split('\t') for l in open('data/ocr.tsv')]
start, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
font = fontpath.truetype('msgothic.ttc', 36)
small = ImageFont.load_default()
cache = {}
cols = 10; cw, ch = 96, 56
sel = rows[start:start+n]
im = Image.new('L', (cols*cw, ((len(sel)+cols-1)//cols)*ch), 40)
dr = ImageDraw.Draw(im)
for k, r in enumerate(sel):
    gidv, best, sc, alts, fn, i = r
    if fn not in cache: cache[fn] = open(os.path.join(UNPACKED, fn), 'rb').read()
    g = glyph(cache[fn], int(i))
    gi = Image.new('L', (20, 20)); gi.putdata([((g[q//2] >> (4*(q&1))) & 15)*17 for q in range(400)])
    x, y = (k % cols)*cw, (k // cols)*ch
    im.paste(gi.resize((40, 40), Image.NEAREST), (x+2, y+2))
    dr.text((x+46, y+2), best, font=font, fill=255)
    dr.text((x+2, y+44), f'{start+k}', font=small, fill=160)
im.save(out)
