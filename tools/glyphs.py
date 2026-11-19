# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Glyph access for MBMS* message fonts: 4bpp, 20x20 cells, from 0x180."""
from PIL import Image, ImageDraw
GW, GH, BASE = 20, 20, 0x180
GB = GW * GH // 2
def glyph(d, i):
    return d[BASE + i*GB: BASE + (i+1)*GB]
def count(d):
    return (len(d) - BASE) // GB
def to_img(g, scale=1):
    im = Image.new('L', (GW, GH))
    im.putdata([((g[k//2] >> (4*(k&1))) & 15) * 17 for k in range(GW*GH)])
    return im.resize((GW*scale, GH*scale), Image.NEAREST) if scale > 1 else im
def sheet(d, idxs, out, cols=20, scale=2, label=True):
    cw, ch = GW*scale + 4, GH*scale + (12 if label else 4)
    rows = (len(idxs) + cols - 1) // cols
    im = Image.new('L', (cols*cw, rows*ch), 60)
    dr = ImageDraw.Draw(im)
    for n, i in enumerate(idxs):
        x, y = (n % cols)*cw, (n // cols)*ch
        im.paste(to_img(glyph(d, i), scale), (x+2, y+2))
        if label: dr.text((x+2, y+GH*scale+1), str(i), fill=255)
    im.save(out)
