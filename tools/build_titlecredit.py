# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Paint the translation credit into the boot title screen (DAT2/TITLE.BIZ).

Writes private/build/TITLE.BIZ. No other script writes this file.
--dry writes nothing to private/build.
--png FILE writes a preview (JP | EN bottom-right area, 1x and 2x).
RE: private/re/title/findings.org, "Boot title screen: translation credit".

TITLE.BIZ decompressed: a VAB header (0x6000 bytes), then an 8bpp TIM,
CLUT (512,480) 256 entries at 0x6014, pixels 512x480 at VRAM (512,0) from
0x6220. The screen shows it 1:1 (pixel (x,y) = screen (x,y)). The logo and
the CAPCOM copyright are part of the bitmap and stay as they are. The
credit sits in the bottom-right corner, right edge x 496, bottom row 465,
in a pale blue with a dark-blue drop shadow (low contrast, variant B).
Every painted pixel is quantised to the existing CLUT; the CLUT is not
changed (all 256 entries are in use).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
import lz

CREDIT = 'v1.0 EN by sakhmatd - 2026'

SRC = 'private/disc/DAT2/TITLE.BIZ'
OUT = 'private/build/TITLE.BIZ'
SLOT = 88 * 2048                      # disc slot (LBA 158593 .. 158681)
CLUT_OFF, PIX_OFF, W, H = 0x6014, 0x6220, 512, 480
FONT = 'RobotoCondensed-Medium.ttf'
SIZE = 10
RIGHT, BOTTOM = 496, 465              # last ink column / row incl. shadow (title-safe)
FILL = (0xa8, 0xd8, 0xf8)
SHADOW = (0x00, 0x18, 0xb0)
SH_ALPHA = 0.85


def rgb15(v):
    v = np.asarray(v).astype(int)
    return np.stack([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], -1)


def text_alpha(text):
    f = fontpath.truetype(FONT, SIZE)
    im = Image.new('L', (int(f.getlength(text)) + 8, SIZE * 2))
    d = ImageDraw.Draw(im); x = 2
    for ch in text:
        d.text((x, 2), ch, font=f, fill=255); x += f.getlength(ch)
    a = np.array(im) / 255.0
    ys, xs = np.nonzero(a > 0.02)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.clip(a / a.max(), 0, 1) ** 0.8


def paint(idx, pal, text):
    """Return the new index image and the ink box (x0, y0, x1, y1)."""
    a = text_alpha(text)
    h, w = a.shape
    # The shadow adds 1 px right and down: ink + shadow end at RIGHT, BOTTOM.
    x0, y0 = RIGHT - w, BOTTOM - h
    pad = 2
    A = np.pad(a, pad); hh, ww = A.shape
    ry, rx = y0 - pad, x0 - pad
    reg = pal[idx[ry:ry + hh, rx:rx + ww]].astype(float)
    S = np.zeros_like(A); S[1:, 1:] = A[:-1, :-1]
    reg = reg * (1 - S[..., None] * SH_ALPHA) + np.array(SHADOW, float) * (S[..., None] * SH_ALPHA)
    reg = reg * (1 - A[..., None]) + np.array(FILL, float) * A[..., None]
    touched = (S * SH_ALPHA > 0.2) | (A > 0.2)
    q = ((np.clip(reg, 0, 255)[..., None, :] - pal[None, None]) ** 2).sum(-1).argmin(-1)
    out = idx.copy()
    sub = out[ry:ry + hh, rx:rx + ww]; sub[touched] = q[touched]
    ys, xs = np.nonzero(out != idx)
    return out, (xs.min(), ys.min(), xs.max(), ys.max())


def build(text=CREDIT):
    comp = open(SRC, 'rb').read()
    d = bytearray(lz.decompress(comp)[0])
    assert len(d) == 0x42800 and d[0x6000:0x6004] == b'\x10\0\0\0', 'unexpected TITLE.BIZ'
    pal = rgb15(np.frombuffer(bytes(d[CLUT_OFF:CLUT_OFF + 512]), '<u2'))
    idx = np.frombuffer(bytes(d[PIX_OFF:PIX_OFF + W * H]), np.uint8).reshape(H, W)
    new, box = paint(idx, pal, text)
    assert box[2] <= RIGHT and box[3] <= BOTTOM and box[1] >= 450, box  # clear of the copyright (rows 431-446)
    d[PIX_OFF:PIX_OFF + W * H] = new.tobytes()
    c = lz.compress(bytes(d), name='TITLE.BIZ')
    assert lz.decompress(c)[0] == bytes(d)
    assert len(c) <= SLOT, f'TITLE.BIZ {len(c)} > slot {SLOT}'
    return c, pal, idx, new, box


def preview(path, pal, idx, new):
    y0, x0 = 400, 256
    jp, en = pal[idx].astype(np.uint8), pal[new].astype(np.uint8)
    one = np.concatenate([jp[y0:, x0:], np.full((H - y0, 4, 3), 32, np.uint8), en[y0:, x0:]], 1)
    img = Image.fromarray(one)
    big = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    s = Image.new('RGB', (big.width, img.height + big.height + 4), (32, 32, 32))
    s.paste(img, (0, 0)); s.paste(big, (0, img.height + 4)); s.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--png')
    a = ap.parse_args()
    c, pal, idx, new, box = build()
    print(f'TITLE.BIZ: credit {CREDIT!r} ink x {box[0]}-{box[2]} rows {box[1]}-{box[3]}, '
          f'{int((new != idx).sum())} px, LZ {len(c)} B (slot {SLOT})')
    if a.png: preview(a.png, pal, idx, new)
    if not a.dry:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        tmp = OUT + '.tmp'; open(tmp, 'wb').write(c); os.replace(tmp, OUT)


if __name__ == '__main__':
    main()
