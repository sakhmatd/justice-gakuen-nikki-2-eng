# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Extras viewer help panel (next page / prev page / back to Extras) in
English. Port of the first game's build_title.help_panel().

The panel is one sprite in the MOJI2 font TIM (4bpp, VRAM (768,0), 1024x256
px): TIM pixels (888,160) 128x72, CLUT 12. KOUBAI2 shows it in the
illustration and memories viewers. KOUBAI2 loads DAT2/MOJI2.TIZ itself
(0x800a71bc, 0x800b0ed8), so the disc MOJI2.TIZ is the copy in VRAM there.
MENU.BIZ TIM 0 is a byte copy of the same TIM (topic menu); build_title.py
calls patch_menu() so both copies match.

Text: help_next, help_prev, help_back in tl/EXTRASGFX.txt.
build_extras.py writes private/build/MOJI2.TIZ through build_moji2().
"""
import os, struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.join(ROOT, 'tl/EXTRASGFX.txt')
MOJI2_SRC = os.path.join(ROOT, 'private/unpacked/MOJI2.TIZ.lz')
MOJI2_SLOT = 42 * 2048                  # LBA 122142 -> next file at 122184
MENU_FONT_TIM = 0x0                     # MENU.BIZ TIM 0 = MOJI2 copy
HELP_RECT = (888, 160, 128, 72)         # panel in TIM pixels
HELP_FONT, HELP_SIZE = 'RobotoCondensed-Bold.ttf', 15
# (tl id, x0, x1, baseline, cleared rows, last cleared col) in panel
# pixels. The circle/cross icons start at x 96 and the START mark at x 92;
# the JP text reaches x 89. Same panel pixels as the first game.
HELP_LINES = [('help_next', 6, 93, 17, (4, 25), 95), ('help_prev', 6, 93, 38, (25, 45), 95),
              ('help_back', 6, 88, 61, (45, 66), 90)]

def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def unpack(d, off, w, h):
    raw = np.frombuffer(bytes(d[off:off + w * h // 2]), np.uint8).reshape(h, w // 2)
    px = np.empty((h, w), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def pack(d, off, px):
    d[off:off + px.size // 2] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def help_line(text, x0, x1, base, shape):
    """Alpha (shape) of one line, left at x0, squeezed to end by x1."""
    ss = 4
    f = fontpath.truetype(HELP_FONT, HELP_SIZE * ss)
    b = f.getbbox(text); asc, desc = f.getmetrics()
    im = Image.new('L', (b[2] - b[0] + 4 * ss, asc + desc + 2 * ss), 0)
    ImageDraw.Draw(im).text((2 * ss - b[0], ss), text, font=f, fill=255)
    sx = min(1.0, (x1 - x0) * ss / (b[2] - b[0]))
    sm = np.asarray(im.resize((max(1, round(im.width * sx / ss)), round(im.height / ss)),
                              Image.LANCZOS)) / 255.0
    a = np.zeros(shape)
    oy, ox = round(base - 1 - asc / ss), round(x0 - 2 * sx)
    h, w = min(sm.shape[0], shape[0] - oy), min(sm.shape[1], shape[1] - ox)
    a[oy:oy + h, ox:ox + w] = sm[:h, :w]
    return a, sx

def help_panel(d, tim, tl):
    """Redraw the panel text in the font TIM at file offset `tim` (in place).
    White fill, 1 px dark outline, quantised to the panel's grey ramp
    (CLUT 12, indices 1-8); cleared areas get the 9/0 checker back.
    Returns the horizontal squeeze per line."""
    pal = struct.unpack_from('<16H', d, tim + 20 + 12 * 32)
    greys = sorted(range(1, 9), key=lambda i: lum(pal[i]))
    lv = np.array([lum(pal[i]) * 255 / 186 for i in greys])
    x, y, w, h = HELP_RECT
    px = unpack(d, tim + 0x1c0, 1024, 256)
    pan = px[y:y + h, x:x + w]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where((yy + xx) % 2 == 1, 9, 0).astype(np.uint8)
    squeeze = []
    for rid, x0, x1, base, (r0, r1), xc in HELP_LINES:
        pan[r0:r1, 4:xc + 1] = checker[r0:r1, 4:xc + 1]
        a, sx = help_line(tl[rid], x0, x1, base, (h, w))
        squeeze.append(sx)
        o = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                o = np.maximum(o, np.roll(np.roll(a, dy, 0), dx, 1))
        v = 16 * o * (1 - a) + 248 * a
        q = np.array(greys, np.uint8)[np.abs(v[..., None] - lv).argmin(-1)]
        m = o > 0.3
        m[:, xc + 1:] = False
        m[:r0] = m[r1:] = False
        pan[m] = q[m]
    pack(d, tim + 0x1c0, px)
    return squeeze

def problems(squeeze):
    return [f'{rid}: squeezed to {sx:.2f}' for (rid, *_), sx in zip(HELP_LINES, squeeze) if sx < 0.85]

def build_moji2(tl=None):
    """-> (decompressed MOJI2.TIZ bytes, squeeze list)."""
    m2 = bytearray(open(MOJI2_SRC, 'rb').read())
    sq = help_panel(m2, 0, tl or load_tl())
    return m2, sq

def patch_menu(d, tl=None):
    """Paint the panel into MENU.BIZ (decompressed, bytearray) TIM 0."""
    src = open(MOJI2_SRC, 'rb').read()
    assert bytes(d[MENU_FONT_TIM:MENU_FONT_TIM + 0x1c0]) == src[:0x1c0], 'MENU.BIZ font TIM moved'
    return help_panel(d, MENU_FONT_TIM, tl or load_tl())

def panel_rgb(d, tim=0):
    """Panel as an RGB array (for previews)."""
    x, y, w, h = HELP_RECT
    pal = struct.unpack_from('<16H', d, tim + 20 + 12 * 32)
    rgb = np.array([[(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8] for c in pal], np.uint8)
    return rgb[unpack(d, tim + 0x1c0, 1024, 256)[y:y + h, x:x + w]]
