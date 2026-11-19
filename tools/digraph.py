# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Render two-letter 20x20 4bpp glyph tiles (10 px per letter)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
FONT = ('Iosevka-Bold.ttc', 2, 19, -3)  # path, face, size, y offset
_f = None
def font():
    global _f
    if _f is None: _f = fontpath.truetype(FONT[0], FONT[2], index=FONT[1])
    return _f
def levels(pair):
    """Return 20x20 array of palette indices 0..14."""
    im = Image.new('L', (20, 20))
    dr = ImageDraw.Draw(im)
    for k, c in enumerate(pair):
        if c != ' ': dr.text((k*10, FONT[3]), c, font=font(), fill=255)
    a = np.asarray(im, np.float32) / 255
    main = np.where(a > 0.12, np.rint(3 + 11*a), 0)
    sh = np.zeros_like(main); sh[1:, 1:] = (a[:-1, :-1] > 0.4)
    out = np.where(main > 0, main, np.where(sh > 0, 3, 0)).astype(np.uint8)
    return out
def pack(lv):
    """20x20 indices -> 200 bytes, low nibble first."""
    f = lv.ravel()
    return bytes((int(f[i]) | int(f[i+1]) << 4) for i in range(0, 400, 2))
def preview(text, out, scale=2):
    cells = [text[i:i+2].ljust(2) for i in range(0, len(text), 2)]
    pal = [0] + [int(x) for x in np.linspace(40, 255, 14)]
    im = Image.new('L', (20*len(cells), 20))
    for k, p in enumerate(cells):
        lv = levels(p)
        im.paste(Image.fromarray(np.vectorize(lambda v: pal[v] if v else 30)(lv).astype(np.uint8)), (20*k, 0))
    im.resize((im.width*scale, im.height*scale), Image.NEAREST).save(out)

_fs = None
def levels_small(text):
    """One 20x20 tile holding up to 3 small letters (for 1-cell labels)."""
    global _fs
    if _fs is None: _fs = fontpath.truetype(FONT[0], 13, index=FONT[1])
    im = Image.new('L', (20, 20))
    w = _fs.getlength(text)
    ImageDraw.Draw(im).text(((20 - w) / 2, 1), text, font=_fs, fill=255)
    a = np.asarray(im, np.float32) / 255
    main = np.where(a > 0.12, np.rint(3 + 11*a), 0)
    sh = np.zeros_like(main); sh[1:, 1:] = (a[:-1, :-1] > 0.4)
    return np.where(main > 0, main, np.where(sh > 0, 3, 0)).astype(np.uint8)

def letter_font():
    """96 letters (0x20-0x7f), each 20 rows x 5 bytes (10 px, 4bpp)."""
    out = bytearray()
    for c in range(0x20, 0x80):
        lv = levels(chr(c) + ' ' if c < 0x7f else '  ')[:, :10]
        f = lv.ravel()
        out += bytes((int(f[i]) | int(f[i+1]) << 4) for i in range(0, 200, 2))
    return bytes(out)
