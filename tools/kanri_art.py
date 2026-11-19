# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""EDIT Register art for tools/build_kanri.py (topic kanri, register_room).

EDITMOJI.TIZ (unpacked) holds two TIMs:
- TIM 0: 4bpp, VRAM (640,256), 1024x256 texels, CLUT block (0,481) 512x3.
  Pixels at file 0xc20, 512 B per row.
- TIM 1: 4bpp, VRAM (704,0), 256x256 texels, no CLUT. Pixels at file
  0x21014, 128 B per row. KANRI glyph code 0xffRC is the 20x20 cell
  (C*20, R*20), drawn with CLUT (a3*16, 481).

Pieces drawn here (all in place, same texels as the JP art):
- buttons: TIM 1 cells ff20-25, ff26-2b (row 2), ff30-35, ff36-3b (row 3),
  120x20 strips on the button CLUTs (a3 0xf cursor, 0x16 other: grey ramp,
  14 white). Style of the MOJI2 Latin letters: white face, grey edge, dark
  shadow right and below.
- header EDITキャラクター登録: TIM 0 (512,136) 334x32 (two sprites, 255 +
  79 px, KANRI records 0x210/0x226 and PASS 0x76c/0x782, 0x808/0x81e),
  CLUT (112,481): red letters on a gold lens.
- tab 登録キャラ一覧: TIM 0 (400,216) 111x32, CLUT (0,481), record 0x1a2.
- school labels: TIM 0 72x20 or 80x20 rects (KANRI 0xb5c.., PASS
  0x244..), CLUT (288,481): pink ramp, dark red shadow (N1 school_art).
"""
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

T0_PIX, T1_PIX = 0xc20, 0x21014
IMPACT = 'impact.ttf'
ARIALBD = 'arialbd.ttf'
BARLOW = 'BarlowCondensed-Bold.ttf'
BARLOW_SB = 'BarlowCondensed-SemiBold.ttf'

BTN_CELLS = {'a0': (2, 0), 'a3': (2, 6), 'a30': (3, 0), 'a33': (3, 6)}  # (row, first col)
HDR_RECT = (512, 136, 334, 32)
TAB_RECT = (400, 216, 111, 32)
SCH_RECTS = [(440, 96, 72), (440, 116, 72), (400, 196, 80), (400, 176, 72), (512, 212, 80)]
SCH_JP = ['太陽学園', '五輪高校', 'パシフィックHS', '外道高校', 'ジャスティス学園']


# ---------------------------------------------------------------- TIM access
def t0_get(d):
    raw = np.frombuffer(bytes(d[T0_PIX:T0_PIX + 512 * 256]), np.uint8).reshape(256, 512)
    px = np.zeros((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px


def t0_put(d, px):
    d[T0_PIX:T0_PIX + 512 * 256] = (px[:, 0::2] | px[:, 1::2] << 4).astype(np.uint8).tobytes()


def clut(d, cx, cy=481):
    v = struct.unpack_from('<16H', bytes(d), 0x14 + ((cy - 481) * 512 + cx) * 2)
    return np.array([[(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3, 255 if c else 0]
                     for c in v], np.uint8)


# ---------------------------------------------------------------- masks
def shift(m, dx, dy):
    o = np.zeros_like(m); H, W = m.shape
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r * 0.6


def text_cover(parts, H, base, ss=4, track=0.0, sx=1.0):
    """Coverage (0..1) of text parts [(text, font, cap px)] on one baseline,
    H rows high with the baseline at row base. sx squeezes horizontally."""
    W = 900
    im = Image.new('L', (W * ss, H * ss), 0); dr = ImageDraw.Draw(im)
    x = 8 * ss
    for text, font, cap in parts:
        f0 = fontpath.truetype(font, 200)
        b = f0.getbbox('H'); size = 200 * cap / (b[3] - b[1])
        f = fontpath.truetype(font, round(size * ss))
        for ch in text:
            dr.text((x, base * ss), ch, font=f, fill=255, anchor='ls')
            x += f.getlength(ch) + track * ss
    if sx != 1.0:
        im = im.resize((round(W * ss * sx), H * ss), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    w = a.shape[1] // ss * ss
    a = a[:, :w].reshape(H, ss, w // ss, ss).mean((1, 3))
    cols = np.where(a.max(0) > 0.02)[0]
    return a[:, max(cols[0] - 1, 0):cols[-1] + 2]


# ---------------------------------------------------------------- buttons
def button_art(text, font=BARLOW, cap=12, base=16, track=0.6, sx=1.0):
    """Index image 20 rows, white face on the button ramp: coverage ->
    3..14, shadow 3 one pixel right and below (as the MOJI2 letters)."""
    a = text_cover([(text, font, cap)], 20, base, track=track, sx=sx)
    idx = np.where(a < 0.12, 0, np.clip(np.rint(3 + 11 * a), 4, 14)).astype(np.uint8)
    face = a >= 0.5
    sh = (shift(face, 1, 1) | shift(face, 1, 0) | shift(face, 0, 1)) & (a < 0.3)
    idx[sh & (idx == 0)] = 3
    idx[sh & (idx > 0) & (idx < 6)] = 3
    idx = np.pad(idx, ((0, 0), (0, 1)))
    cols = np.where(idx.any(0))[0]
    return idx[:, cols[0]:cols[-1] + 1]


def put_button(t1, key, img):
    """Centre img in the 6-cell strip of key; returns problems list."""
    r, c0 = BTN_CELLS[key]
    y, x0 = 20 * r, 20 * c0
    t1[y:y + 20, x0:x0 + 120] = 0
    w = img.shape[1]
    if w > 116: return [f'{key}: {w} px > 116']
    x = x0 + (120 - w) // 2
    t1[y:y + 20, x:x + w] = img
    return []


# ---------------------------------------------------------------- header
def lens(W=334, H=32, cy=17.0, hc=12.0):
    """Gold lens (CLUT 112: 8 dark gold .. 14 yellow, 15 pale highlight,
    7 dark rim) as an index image."""
    idx = np.zeros((H, W), np.uint8)
    for x in range(W):
        t = (x + 0.5 - W / 2) / (W / 2)
        hh = hc * (1 - t * t) ** 0.55 + 1.5
        for y in range(H):
            s = (y + 0.5 - cy) / hh
            if abs(s) > 1: continue
            v = (1 - abs(t) ** 2.2) * (1 - 0.45 * s * s)
            if s < -0.15: v += 0.12 * (1 - abs(t))        # lit upper half
            i = 9 + int(round(np.clip(v, 0, 1) * 5.4))
            if abs(s) > 0.86 or abs(t) > 0.97: i = 8 if abs(t) < 0.9 else 7
            idx[y, x] = min(i, 14)
        # thin highlight along the upper edge in the middle third
    hl = np.zeros_like(idx, bool)
    for x in range(W):
        t = (x + 0.5 - W / 2) / (W / 2)
        if abs(t) < 0.45:
            hh = hc * (1 - t * t) ** 0.55 + 1.5
            y = int(round(cy - 0.62 * hh))
            hl[y, x] = True
    idx[hl] = 15
    return idx


def header_art(text='EDIT Character Register', W=334, H=32, caps=(21, 17), base=26,
               style='outline', track=0.5):
    """Red Impact letters (CLUT 112: 5 red, 4 and 3 darker, 2 dark red, 1
    black) over the gold lens."""
    if text.startswith('EDIT '):
        parts = [('EDIT', IMPACT, caps[0]), (' ' + text[5:], IMPACT, caps[1])]
    else:
        parts = [(text, IMPACT, caps[1])]
    a = text_cover(parts, H, base, track=track)
    face = a >= 0.45
    out = ndimage.binary_dilation(face, disk(1)) & ~face
    shadow = (shift(face | out, 2, 2) | shift(face | out, 1, 1)) & ~face & ~out
    idx = lens(W, H)
    w = a.shape[1]
    if w > W - 4: raise ValueError(f'header {w} px > {W - 4}')
    x0 = (W - w) // 2
    sl = (slice(0, H), slice(x0, x0 + w))
    rows = np.where(face.any(1))[0]; t0, t1 = rows[0], rows[-1]
    yy = ((np.arange(H) - t0) / max(1, t1 - t0))[:, None].repeat(w, 1)
    fidx = np.where(yy < 0.55, 5, np.where(yy < 0.85, 4, 3)).astype(np.uint8)
    sub = idx[sl]
    sub[shadow] = 1
    if style == 'outline':
        sub[out] = 2
    sub[face] = fidx[face]
    # anti-aliased face edge: partial coverage on the lens reads as dark red
    edge = (a >= 0.2) & ~face & ~out & ~shadow
    sub[edge] = 2
    idx[sl] = sub
    return idx


# ---------------------------------------------------------------- tab
def tab_art(text, W=111, H=32, cap=13, base=24, font=IMPACT, track=0.5, sx=1.0):
    """CLUT 0: face 1 yellow -> 6 orange -> 8 red top to bottom, outline 10
    dark red, shadow 15 near black (the JP 登録キャラ一覧)."""
    a = text_cover([(text, font, cap)], H, base, track=track, sx=sx)
    face = a >= 0.45
    out = ndimage.binary_dilation(face, disk(1)) & ~face
    shadow = shift(face | out, 1, 1) & ~face & ~out
    w = a.shape[1]
    if w > W: raise ValueError(f'tab {w} px > {W}')
    idx = np.zeros((H, W), np.uint8)
    x0 = 1
    rows = np.where(face.any(1))[0]; t0, t1 = rows[0], rows[-1]
    yy = (np.arange(H) - t0) / max(1, t1 - t0)
    ramp = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.uint8)
    g = ramp[np.clip((yy * 7.99).astype(int), 0, 7)][:, None].repeat(w, 1)
    sub = np.zeros((H, w), np.uint8)
    sub[shadow] = 15; sub[out] = 10; sub[face] = g[face]
    idx[:, x0:x0 + w] = sub[:, :W - x0]
    return idx


# ---------------------------------------------------------------- schools
def school_art(text, maxw, cap=10, font=ARIALBD, thr=0.42, track=0.0, sx=1.0):
    """N1 school_art: horizontal deep-pink to light-pink ramp (CLUT 288 idx
    11 9 10 12 13 14), dark red shadow 2, partial edge 7. 20 rows."""
    a = text_cover([(text, font, cap)], 20, 15, track=track, sx=sx)
    face = a >= thr
    edge = (a >= 0.2) & ~face
    shadow = (shift(face, 1, 1) | shift(face, 0, 1)) & ~face
    idx = np.zeros(a.shape, np.uint8)
    idx[shadow] = 2
    cols = np.where(face.any(0))[0]; c0, c1 = cols[0], cols[-1]
    ramp = [11, 9, 10, 12, 13, 14]
    xx = (np.arange(a.shape[1]) - c0) / max(1, c1 - c0)
    g = np.array([ramp[min(len(ramp) - 1, int(max(v, 0) * len(ramp)))] for v in xx], np.uint8)
    gi = np.repeat(g[None, :], 20, 0)
    idx[face] = gi[face]
    idx[edge & ~shadow & shift(face, -1, 0)] = 7
    cols = np.where(idx.any(0))[0]
    idx = idx[:, cols[0]:cols[-1] + 1]
    if idx.shape[1] > maxw: raise ValueError(f'{text!r} {idx.shape[1]} px > {maxw}')
    return idx


def put_rect(t0, x, y, w, h, img, align='left'):
    t0[y:y + h, x:x + w] = 0
    ih, iw = img.shape
    dx = 0 if align == 'left' else (w - iw) // 2
    dy = (h - ih) // 2
    t0[y + dy:y + dy + ih, x + dx:x + dx + iw] = img


# ---------------------------------------------------------------- build
BTN_STYLE = dict(font=BARLOW, cap=15, base=17, track=0.5)
HDR_STYLE = dict(caps=(24, 19), base=28)
TAB_STYLE = dict(cap=13, font='BarlowCondensed-ExtraBold.ttf',
                 track=0.6)
SCH_STYLE = dict(cap=12, font=BARLOW, track=0.2)
IDS = set(BTN_CELLS) | {'reghdr', 'tab'} | {f'sch{i}' for i in range(5)}


def build(d, t1, tl, problems):
    """Draw the register art into EDITMOJI d (TIM 0) and t1 (TIM 1 texel
    array). Returns {id: index image} for previews."""
    t0 = t0_get(d); out = {}
    for key in BTN_CELLS:
        en = tl.get(key, '')
        if en in ('', '='): continue
        img = button_art(en, **BTN_STYLE)
        problems += put_button(t1, key, img); out[key] = img
    en = tl.get('reghdr', '')
    if en not in ('', '='):
        try:
            img = header_art(en, **HDR_STYLE)
            # two sprites as the JP: 255 px from texel 512, the rest from
            # texel 768 (texel 767 is not drawn)
            x, y, w, h = HDR_RECT
            t0[y:y + h, 512:767] = img[:, :255]; t0[y:y + h, 768:768 + w - 255] = img[:, 255:]
            out['reghdr'] = img
        except ValueError as e: problems.append(f'reghdr: {e}')
    en = tl.get('tab', '')
    if en not in ('', '='):
        try:
            img = tab_art(en, **TAB_STYLE)
            x, y, w, h = TAB_RECT; t0[y:y + h, x:x + w] = img; out['tab'] = img
        except ValueError as e: problems.append(f'tab: {e}')
    for i, (x, y, w) in enumerate(SCH_RECTS):
        en = tl.get(f'sch{i}', '')
        if en in ('', '='): continue
        try:
            img = school_art(en, w, **SCH_STYLE)
            put_rect(t0, x, y, w, 20, img); out[f'sch{i}'] = img
        except ValueError as e: problems.append(f'sch{i}: {e}')
    t0_put(d, t0)
    return out
