# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the move-name sheets of DAT2/EDITCE.BIZ (CPU Edit screen of EDIT
Register, drawn by the PASS overlay) in English. Topic editce.

Reads tl/EDITCM.txt (the EDITCM card names; same ids) and the disc copy of
EDITCE.BIZ. Writes private/build/EDITCE.BIZ only. --check builds but
writes nothing. --png DIR writes ceNN.png (JP | EN rows, 1x) to DIR.

EDITCE.BIZ is an archive of 32 LZ entries (entry k = EDIT style k + 1,
tl ids cm{k+1:02d}.s1.. then .b1..). Each entry is one 4bpp TIM without a
CLUT, 256x256 texels at VRAM (512,0). Row r of the list is the texel rect
(0, 20r, 200, 20); PASS draws it at screen x 40 with CLUT (128,481):
index 0 transparent, 2-15 a grey ramp from light (27/31) to dark (4/31).
The JP names are centred on x 99.5 and use the full row height. Entry 3
also holds an index-1 fill at y 192-255 that no row reaches; it is kept.
The archive cannot grow (EDITCM.BIZ follows), so each entry is packed into
its JP slot; every EN entry is smaller.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
import lz, arc
from build_editcm import load_tl, SS, FONT

ARC = 'private/disc/DAT2/EDITCE.BIZ'
OUT = 'private/build/EDITCE.BIZ'
N = 32
RW, RH, NROW = 200, 20, 8           # row rect and rows per sheet
CX = 100                            # centre of the row (JP measured 99.5)
MAXW = 198                          # ink width limit (1 px margin each side)
SIZES = [x / 4 for x in range(76, 55, -1)]  # 19 .. 14 px
SQUEEZE = (1.0, 0.96, 0.92, 0.88)  # more squeeze looked cramped (variants.png)
# CLUT (128,481) grey levels (5-bit) of indices 0-15; 0 is transparent.
GREY = [31, 0, 27, 25, 22, 20, 19, 17, 15, 14, 12, 11, 9, 7, 6, 4]
BG5 = 28                            # preview ground (screen is not seen yet)
INK = list(range(6, 16))            # coverage ramp indices, 15 = solid

def names(tl, n):
    """EN names of style n in sheet row order: specials, then BVAs."""
    out = []
    for kind in 'sb':
        i = 1
        while f'cm{n:02d}.{kind}{i}' in tl:
            out.append(tl[f'cm{n:02d}.{kind}{i}'].replace('|', ' ')); i += 1
    return out

def render_row(text, size, sx):
    """Coverage (0..1) of text on one 20-row strip, baseline placed so the
    cap-top .. descender span is centred. Returns (alpha, ink width)."""
    f = fontpath.truetype(FONT, int(round(size * SS)))
    cap = f.getbbox('H')                   # (x0, top, x1, baseline-ish)
    desc = f.getbbox('g')[3]
    top, bot = cap[1], desc                # in SS pixels from the draw origin
    oy = (RH * SS - (bot - top)) / 2 - top
    b = f.getbbox(text)
    im = Image.new('L', (b[2] - b[0] + 4 * SS, RH * SS), 0)
    ImageDraw.Draw(im).text((2 * SS - b[0], oy), text, font=f, fill=255)
    w = max(1, int(round(im.width * sx / SS)))
    a = np.asarray(im.resize((w, RH), Image.LANCZOS), float) / 255
    cols = np.where(a.max(0) > 0.05)[0]
    a = a[:, cols[0]:cols[-1] + 1]
    return a, a.shape[1]

def fit(text):
    """Largest size / least squeeze that fits MAXW, scored as in
    build_editcm.fit_lines (size - 10 * (1 - squeeze))."""
    best = None
    for s in SIZES:
        for sx in SQUEEZE:
            score = s - (1 - sx) * 10
            if best and score <= best[0]: continue
            a, w = render_row(text, s, sx)
            if w <= MAXW:
                best = (score, s, sx, a)
                break
    return best

def quant(a):
    """Coverage -> CLUT index: nearest grey of a blend of ink 15 over a
    ground at index 2, limited to 6..15; faint edges -> 0."""
    target = GREY[2] - (GREY[2] - GREY[15]) * a
    g = np.array([GREY[i] for i in INK], float)
    idx = np.array(INK)[np.abs(target[..., None] - g).argmin(-1)]
    return np.where(a < 0.17, 0, idx).astype(np.uint8)

def unpack(u):
    a = np.frombuffer(u[20:20 + 32768], np.uint8)
    px = np.zeros(65536, np.uint8); px[0::2] = a & 15; px[1::2] = a >> 4
    return px.reshape(256, 256)

def pack(u, px):
    f = px.reshape(-1)
    return bytes(u[:20]) + bytes((f[0::2] | f[1::2] << 4).astype(np.uint8)) + bytes(u[20 + 32768:])

def used_rows(px):
    # ce09 row 5 spills 2 texels into y 120; a real row has far more ink
    return [r for r in range(NROW) if np.count_nonzero(px[RH * r:RH * r + RH, :RW]) > 8]

def preview(px, rows):
    pal = np.array([[g * 255 // 31] * 3 for g in GREY], np.uint8)
    pal[0] = BG5 * 255 // 31
    return pal[px[:RH * max(rows, 1), :RW]]

def main(argv):
    check = '--check' in argv
    pdir = argv[argv.index('--png') + 1] if '--png' in argv else None
    if pdir: os.makedirs(pdir, exist_ok=True)
    tl = load_tl()
    src = open(ARC, 'rb').read()
    toc = arc.toc(src)
    assert len(toc) == N and toc[0][0] == 0x800, toc[:1]
    out = bytearray(src)
    problems = 0
    for k, (o, sz) in enumerate(toc):
        n = k + 1; err = []
        u = lz.decompress(src[o:o + sz])[0]
        assert struct.unpack_from('<2I', u, 0) == (0x10, 0) and len(u) == 32788
        jp = unpack(u); px = jp.copy()
        nm = names(tl, n)
        jr = used_rows(jp)
        if jr != list(range(len(nm))): err.append(f'JP rows {jr} vs {len(nm)} names')
        px[:RH * NROW, :] = 0
        info = []
        for r, t in enumerate(nm):
            if not t: err.append(f'row {r}: no EN'); continue
            f = fit(t)
            if f is None: err.append(f'row {r}: "{t}" does not fit'); continue
            _, s, sx, a = f
            x0 = int(round(CX - a.shape[1] / 2))
            q = quant(a)
            x0 = max(0, min(RW - a.shape[1], x0))
            blk = px[RH * r:RH * r + RH, x0:x0 + a.shape[1]]
            blk[:] = np.maximum(blk, q)
            if s < 19 or sx < 1: info.append(f'{t!r} {s:g}px x{sx:g}')
        d = pack(u, px)
        slot = (toc[k + 1][0] if k + 1 < N else len(src)) - o
        c = lz.compress(d, name='EDITCE.BIZ')
        if len(c) > slot: err.append(f'compressed {len(c)} > slot {slot}')
        if lz.decompress(c)[0][:len(d)] != d: err.append('LZ round trip failed')
        if not err:
            out[o:o + slot] = c + bytes(slot - len(c))
            struct.pack_into('<II', out, 8 * k, o, len(c))
        if pdir:
            rows = len(nm)
            pj, pe = preview(jp, rows), preview(px, rows)
            gap = np.full((pj.shape[0], 4, 3), 255, np.uint8)
            Image.fromarray(np.hstack([pj, gap, pe])).save(f'{pdir}/ce{k:02d}.png')
        print(f'ce{k:02d} (style {n}): {len(nm)} rows, {len(c)} B (slot {slot})'
              + (' fit: ' + '; '.join(info) if info else '')
              + ''.join(f'\n  {e}' for e in err))
        problems += len(err)
    assert len(out) == len(src) and out[:4] == src[:4]
    if not check and not problems:
        open(OUT, 'wb').write(out)
    print(f'EDITCE: problems {problems}' + ('' if check or problems else f', wrote {OUT}'))
    return 1 if problems else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
