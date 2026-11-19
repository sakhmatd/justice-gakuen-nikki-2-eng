# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Render a 1024x512 VRAM dump (Redux getVRAM, u16 15-bit) to PNG.
Usage:
  vram.py DUMP OUT.png                          whole VRAM as 15-bit colour
  vram.py DUMP OUT.png X Y W H CX CY [SCALE]   4bpp region with CLUT at CX,CY
X is in VRAM halfwords, W and H in pixels. Index-0 colour 0 shows magenta.
Prefer the region form: a cropped image costs far fewer tokens to view.
Merged from private/re/*/vram.py, vram4.py, vram2png.py."""
import sys
import numpy as np
from PIL import Image


def load(path):
    return np.frombuffer(open(path, 'rb').read()[:1024 * 512 * 2], '<u2').reshape(512, 1024)


def rgb(v):
    return np.stack([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], -1).astype(np.uint8)


def region4(v, x, y, w, h, cx, cy):
    blk = v[y:y + h, x:x + (w + 3) // 4]
    px = np.stack([(blk >> s) & 15 for s in (0, 4, 8, 12)], -1).reshape(h, -1)[:, :w]
    clut = v[cy, cx:cx + 16]
    pal = rgb(clut); pal[clut == 0] = (255, 0, 255)
    return Image.fromarray(pal[px])


if __name__ == '__main__':
    a = sys.argv
    v = load(a[1])
    if len(a) == 3:
        Image.fromarray(rgb(v)).save(a[2])
    else:
        x, y, w, h, cx, cy = map(int, a[3:9])
        im = region4(v, x, y, w, h, cx, cy)
        sc = int(a[9]) if len(a) > 9 else 1
        if sc > 1: im = im.resize((w * sc, h * sc), Image.NEAREST)
        im.save(a[2])
