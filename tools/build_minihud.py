# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Mini-game HUD graphics in English: CEFT5.BIZ (HUD labels and the
play-again box), SPRINT.BIZ and DANCE.BIZ (their own copies of the box,
the Dance intro lines), CEFT4.BIZ and CEFT6.BIZ (開始/終了 sprites and the
exam HUD label 科目).

Reads tl/MINIHUD.txt. RE: private/re/minihud/findings.org; the CEFT5 code
is the first game's build_festpics.py HUD part (CEFT5.BIZ is identical on
both discs); it differs from N1's CEFT5 only in 'Best', set flush right.

- CEFT5 TIM 0x10800 (4bpp 768x256): labels redrawn in place in their
  sprite rects (HUD); the box at (656,168) 112x72, CLUT 49, is erased to
  its stipple and redrawn (paint_again).
- SPRINT TIM 0 holds a pixel copy of the box at (0,56) with an identical
  CLUT; DANCE TIM 0 holds a copy at (256,70) whose right frame column
  differs. Both get the CEFT5 box change as a delta (only pixels that the
  CEFT5 paint changed).
- DANCE intro lines: rows of 24 px at x 512 (lines 0-3 JP, 4-7 already
  EN). The game draws them as scrolling FT4s of fixed widths (126, 118,
  254, 126 + 126 for line 3 split at u 128); lines 0-3 are redrawn.
- CEFT4/CEFT6 TIM 0x10800 (same pixels in the painted rects): 開始 and 終了 are
  80x96 sprites at (256,0) and (336,0), redrawn as two lines each (N1
  build_examgfx big_start). 科目 is two 24x16 cells (184,96) and (184,112)
  that SIKEN draws 20 px apart; the EN label is split across them.

The TIMs come from private/unpacked (originals), so reruns are clean.
- MGAME01/02/03/05/06/07.BIN: the start menu records of 戻る get a new X
  so that "Back" is centred on the menu like the other items.

Writes private/build/{CEFT4,CEFT5,CEFT6,DANCE,SPRINT}.BIZ and .BIZ.lz and
private/build/MGAME0{1,2,3,5,6,7}.BIN (from private/disc; no other tool
writes these overlays yet; a later tool must start from these copies).
Build order: before hook.py (any place; no other tool writes these files).
Options: --dry (no writes), --png (previews to private/re/minihud/en/).
"""
import sys, os, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz
import minittl_atlas

F_BODY = 'tahomabd.ttf'
F_HEAD = 'seguibli.ttf'
F_IMPACT = 'impact.ttf'
F_COND = 'DejaVuSansCondensed-Bold.ttf'
PREV = 'private/re/minihud/en'

def load_tl(path='tl/MINIHUD.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

_fcache = {}
def font(path, size):
    k = (path, size)
    if k not in _fcache: _fcache[k] = fontpath.truetype(path, size)
    return _fcache[k]

class Tim4:
    """4bpp TIM in a decompressed buffer: clut (u16 list), px (h, w)."""
    def __init__(self, buf, off):
        self.buf = buf
        bl = struct.unpack_from('<I', buf, off + 8)[0]
        cw, ch = struct.unpack_from('<HH', buf, off + 16)
        self.clut = list(struct.unpack_from(f'<{cw * ch}H', buf, off + 20))
        p = off + 8 + bl
        _, _, _, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.H, self.W = p + 12, h, w * 4
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        self.px = np.empty((h, w * 4), np.uint8)
        self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        self.orig = self.px.copy()

    def pal(self, c):
        return np.array([rgb15(v) for v in self.clut[c * 16:c * 16 + 16]])

    def store(self):
        raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8)
        self.buf[self.pix:self.pix + raw.size] = raw.tobytes()

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

# ---------------------------------------------------------------- CEFT5 HUD
# Port of N1 build_festpics.py (paint_hud, paint_again); keep in step.

CEFT_TIM, CEFT_PIX, CEFT_W, CEFT_H = 0x10800, 0x11020, 768, 256
# id: (x, y, w, h, clut, kind); kind 'plate' = white text on the orange bar
HUD = {
    'hud_konjo': (0, 56, 88, 16, 4, 'plate'),
    'hud_ato_kyu': (0, 72, 128, 16, 4, 'plate'),
    'hud_genzai': (720, 128, 48, 16, 2, 'small'),
    'hud_saiko': (200, 240, 56, 16, 2, 'small'),
    'hud_ten': (240, 0, 16, 16, 2, 'small'),
    'hud_geijutsu': (216, 56, 40, 16, 2, 'small'),
    'hud_frame': (136, 72, 40, 16, 2, 'small'),
}
# 'Best' is a caption left of right-aligned score digits: it is set flush
# right in its cell (2 px for the shadow) so it sits next to the score
# field. The other captions stay on their box's or column's left edge
# (Now: green box, Art/Frame: under the box's POINT caption).
HUD_RIGHT = {'hud_saiko'}
# old text columns on the plates (left part, right part)
HUD_TEXT = {'hud_konjo': [(0, 62)], 'hud_ato_kyu': [(0, 56), (100, 128)]}

def hud_quant(rgb, cov, pal, allowed):
    """Nearest palette index (from allowed) for covered pixels, else 0."""
    out = np.zeros(cov.shape, np.uint8)
    P = pal[allowed]
    d = ((rgb[:, :, None, :] - P[None, None]) ** 2).sum(-1)
    q = np.array(allowed)[d.argmin(-1)]
    out[cov] = q[cov]
    return out

def hud_text(text, size, shear, path=F_BODY, ss=4):
    f = font(path, int(size * ss))
    im = Image.new('L', (int(f.getlength(text)) + 12 * ss, int(size * 2 * ss)), 0)
    ImageDraw.Draw(im).text((4 * ss, 2 * ss), text, font=f, fill=255)
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * im.height * 0.7, 0, 1, 0),
                          resample=Image.BICUBIC)
    a = np.asarray(im.resize((im.width // ss, im.height // ss), Image.BOX), float) / 255
    rows_ = np.where(a.max(1) > 0.05)[0]; cols = np.where(a.max(0) > 0.05)[0]
    return a[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]

# end-of-round box: rect in the atlas, CLUT index, fill area, question and
# choice centres (the cursor frames sit at x 11 and 54, 44 wide).
AGAIN = dict(x=656, y=168, w=112, h=72, clut=49, fill=(2, 14, 110, 70),
             q=('hud_again', 56, 31, 100, 19), yes=('hud_yes', 33, 55, 38),
             no=('hud_no', 76, 55, 38))

def again_draw(b, pal, a, x0, y0, top, bot, ramp):
    """Text a at (x0, y0): vertical gradient top->bot, 1 px outline and a
    +1,+1 shadow in index 1, quantized to [1] + ramp."""
    h, w = b.shape
    f = np.zeros((h, w)); f[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    o = ndimage.grey_dilation(f, footprint=np.ones((3, 3)))
    s = np.zeros((h, w)); s[1:, 1:] = o[:-1, :-1]
    dark = np.maximum(s, o)
    t = np.clip((np.arange(h)[:, None] - y0) / max(1, a.shape[0] - 1), 0, 1)[..., None]
    rgb = pal[1] * (1 - f[..., None]) + (pal[top] * (1 - t) + pal[bot] * t) * f[..., None]
    cov = (f > 0.3) | (dark > 0.4)
    q = hud_quant(rgb, cov, pal, [1] + ramp)
    b[cov] = q[cov]

def paint_again(px, d, tl, log):
    """Erase the JP text of the box to the fill stipple and draw the English
    question (ramp 2-7) and choices (ramp 8-12)."""
    A = AGAIN
    if not all(A[k][0] in tl for k in ('q', 'yes', 'no')): return None
    clut = struct.unpack_from('<1024H', d, CEFT_TIM + 0x14)
    pal = np.array([rgb15(v) for v in clut[A['clut'] * 16:A['clut'] * 16 + 16]])
    b = px[A['y']:A['y'] + A['h'], A['x']:A['x'] + A['w']]
    x0, y0, x1, y1 = A['fill']
    yy, xx = np.mgrid[0:A['h'], 0:A['w']]
    m = np.zeros(b.shape, bool); m[y0:y1, x0:x1] = True
    m &= ~np.isin(b, (13, 14))                   # keep the frame
    b[m] = np.where((xx + yy) % 2 == 1, 15, 0)[m]
    def txt(s, size, maxw, maxh):
        while True:
            a = hud_text(s, size, 0.0, F_BODY)
            if (a.shape[1] + 3 <= maxw and a.shape[0] + 3 <= maxh) or size <= 9: break
            size -= 1
        return np.clip((a - 0.15) / 0.6, 0, 1), size
    key, cx, cy, mw, mh = A['q']
    a, sz = txt(tl[key], 15, mw, mh)
    x, y = cx - (a.shape[1] + 1) // 2, cy - (a.shape[0] + 1) // 2
    again_draw(b, pal, a, x, y, 7, 5, list(range(2, 8)))
    log.append(f'{key}: "{tl[key]}" {a.shape[1]}x{a.shape[0]} size {sz} at {x},{y}')
    for k in ('yes', 'no'):
        key, cx, cy, mw = A[k]
        a, sz = txt(tl[key], 11, mw, 12)
        x, y = cx - (a.shape[1] + 1) // 2, cy - (a.shape[0] + 1) // 2
        again_draw(b, pal, a, x, y, 12, 11, list(range(8, 13)))
        log.append(f'{key}: "{tl[key]}" {a.shape[1]}x{a.shape[0]} size {sz} at {x},{y}')
    return pal

def paint_hud(d, tl, log):
    """Returns (before, after) index arrays of the whole CEFT5 atlas."""
    clut = struct.unpack_from('<512H', d, CEFT_TIM + 0x14)
    raw = np.frombuffer(bytes(d[CEFT_PIX:CEFT_PIX + 384 * CEFT_H]), np.uint8).reshape(CEFT_H, 384)
    px = np.empty((CEFT_H, CEFT_W), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    before = px.copy()
    for rid, (x, y, w, h, c, kind) in HUD.items():
        if rid not in tl: continue
        pal = np.array([rgb15(v) for v in clut[c * 16:c * 16 + 16]])
        sub = px[y:y + h, x:x + w]
        parts = tl[rid].split('|')
        if kind == 'plate':
            # restore the bar under the old text from a clean column
            clean = sub[:, 80].copy()
            for (a0, a1) in HUD_TEXT[rid]:
                for r in range(h):
                    for xx in range(a0, a1):
                        left = sub[r, :].nonzero()[0]
                        lx = left.min() if len(left) else w
                        sub[r, xx] = clean[r] if (xx >= lx and r >= 8) else 0
            # text: white fill, 1 px dark outline, grey ramp antialias
            a = hud_text(parts[0], 14, 0.25, F_HEAD)
            if a.shape[0] > 13:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (max(1, int(a.shape[1] * 13 / a.shape[0])), 13), Image.LANCZOS), float) / 255
            ox, oy = 6, 0
            span = HUD_TEXT[rid][0][1] - ox
            if a.shape[1] + 2 > span: log.append(f'{rid}: "{parts[0]}" {a.shape[1]} px > {span}')
            f_ = np.zeros((h, w))
            hh, ww = min(a.shape[0], h - oy - 1), min(a.shape[1], w - ox - 1)
            f_[oy + 1:oy + 1 + hh, ox + 1:ox + 1 + ww] = a[:hh, :ww]
            o = np.clip(ndimage.grey_dilation(f_, footprint=np.ones((3, 3))) * 1.6, 0, 1)
            base = pal[sub].astype(float)
            cov = (sub > 0) | (o > 0.3)
            rgb = base * (1 - o[..., None]) + pal[1] * o[..., None]
            rgb = rgb * (1 - f_[..., None]) + pal[15] * f_[..., None]
            q = hud_quant(rgb, cov, pal, list(range(1, 16)))
            keep = o < 0.3                 # bar pixels the text does not touch
            q[keep] = sub[keep]
            sub[:] = q
            log.append(f'{rid}: "{parts[0]}" w {a.shape[1]}')
        else:
            sub[:] = 0
            a = hud_text(parts[0], 11, 0.0, F_BODY)
            if a.shape[0] > 10:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (max(1, int(a.shape[1] * 10 / a.shape[0])), 10), Image.LANCZOS), float) / 255
            if a.shape[1] + 1 > w:
                log.append(f'{rid}: "{parts[0]}" {a.shape[1] + 1} px > {w}; squeezed')
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (w - 1, a.shape[0]), Image.LANCZOS), float) / 255
            a = np.clip((a - 0.15) / 0.6, 0, 1)          # crisper strokes
            ax = w - 2 - a.shape[1] if rid in HUD_RIGHT else 0
            f_ = np.zeros((h, w)); f_[0:a.shape[0], ax:ax + a.shape[1]] = a
            s_ = np.zeros((h, w)); s_[1:, 1:] = f_[:-1, :-1]
            cov = (f_ > 0.15) | (s_ > 0.3)
            rgb = np.zeros((h, w, 3)) + pal[2] * (s_[..., None] > 0.3)
            rgb = rgb * (1 - f_[..., None]) + pal[14] * f_[..., None]
            sub[:] = hud_quant(rgb, cov, pal, list(range(2, 15)))
            log.append(f'{rid}: "{parts[0]}" ink {ax}-{ax + a.shape[1]} of {w}')
    paint_again(px, d, tl, log)
    d[CEFT_PIX:CEFT_PIX + 384 * CEFT_H] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    return before, px

# ------------------------------------------------------ box copies (delta)

BOX_COPIES = {           # file: (x, y) of the box picture's top-left (row 0)
    'SPRINT': (0, 56),
    'DANCE': (256, 70),
}

def copy_box(t, before, after, xy, name, log):
    """Apply the CEFT5 box change to the copy at xy (rows 12-71 hold ink)."""
    A = AGAIN
    sl = np.s_[A['y'] + 12:A['y'] + A['h'], A['x']:A['x'] + A['w']]
    b0, b1 = before[sl], after[sl]
    x, y = xy
    dst = t.px[y + 12:y + A['h'], x:x + A['w']]
    same = (dst == b0).mean()
    ch = b0 != b1
    assert same > 0.97, (name, same)
    assert not (ch & (dst != b0)).any(), f'{name}: box copy differs under the text'
    dst[ch] = b1[ch]
    log.append(f'{name}: box at {x},{y}, {same:.3f} equal to CEFT5, {ch.sum()} px changed')

# ------------------------------------------------------------ DANCE lines

DANCE_X = 512
# key: (row y, x0, max width); line 3 is drawn as two pieces at u 0 and 128
DANCE_LINES = {
    'dance_l0': (0, 0, 126), 'dance_l1': (24, 0, 118), 'dance_l2': (48, 0, 254),
    'dance_l3a': (72, 0, 120), 'dance_l3b': (72, 128, 126),
}
DANCE_CAP = 17           # cap height of the art's EN lines (rows 0-21 ink)

def dance_alpha(text, maxw, h=23, base=19, path=F_IMPACT, cap=DANCE_CAP, ss=4):
    """Alpha (h, w) with the baseline at row base, squeezed to maxw."""
    f0 = font(path, 100 * ss)
    hb = f0.getbbox('H', anchor='ls'); capr = (hb[3] - hb[1]) / (100 * ss)
    size = cap / capr
    f = font(path, int(round(size * ss)))
    adv = f.getlength(text) / ss
    W = int(np.ceil(adv)) + 4
    im = Image.new('L', (W * ss, h * ss), 0)
    ImageDraw.Draw(im).text((2 * ss, base * ss), text, font=f, fill=255, anchor='ls')
    sx = min(1.0, (maxw - 3) / adv)
    im = im.resize((max(1, int(round(W * sx))), h), Image.LANCZOS)
    a = np.clip(np.asarray(im, float) / 255, 0, 1)
    cols = np.where(a.max(0) > 0.05)[0]
    return a[:, cols[0]:cols[-1] + 1], sx

def dance_rowfill(px):
    """Face index per row (1 bright .. 7 deep) from the original JP lines."""
    rows = []
    for r in range(24):
        c = collections.Counter()
        for k in range(4):
            v = px[24 * k + r, DANCE_X:DANCE_X + 256]
            c.update(int(i) for i in v if 1 <= i <= 7)
        rows.append(c.most_common(1)[0][0] if c else None)
    known = [r for r in range(24) if rows[r]]
    return [rows[r] or rows[min(known, key=lambda q: abs(q - r))] for r in range(24)]

def dance_style(a, rowfill):
    """Face by row from rowfill, antialias 8/9, a 2 px extrusion to the
    lower right in 10-12 (as in the art's lettering)."""
    h, w = a.shape
    a = np.pad(a, ((0, 0), (0, 3)))
    m = a > 0.5
    e1 = (shift(m, 0, 1) | shift(m, 1, 1) | shift(m, 1, 0)) & ~m
    e2 = (shift(m, 1, 2) | shift(m, 2, 1) | shift(m, 2, 2) | shift(m, 0, 2)) & ~m & ~e1
    out = np.zeros(a.shape, np.uint8)
    out[e2] = 12
    out[e1] = 11
    edge = (a > 0.22) & ~m
    out[edge] = np.where(a[edge] > 0.36, 8, 9)
    for r in range(h):
        out[r, m[r]] = rowfill[r]
    return out

def paint_dance(t, tl, log):
    px = t.px
    rowfill = dance_rowfill(t.orig)
    px[0:96, DANCE_X:DANCE_X + 256] = 0
    for key, (y, x0, maxw) in DANCE_LINES.items():
        a, sx = dance_alpha(tl[key], maxw)
        out = dance_style(a, rowfill[:23] + [rowfill[22]])
        w = out.shape[1]
        assert w <= maxw, (key, w, maxw)
        sub = px[y:y + 23, DANCE_X + x0:DANCE_X + x0 + w]
        sub[out > 0] = out[out > 0]
        log.append(f'{key}: "{tl[key]}" w {w} of {maxw}, sx {sx:.2f}')

# ------------------------------------------------------------ CEFT4/CEFT6

def group_sx(texts, size, maxw, path):
    w = max(font(path, size * 4).getlength(s) / 4 for s in texts)
    return min(1.0, maxw / w)

def line_alpha(text, h, size, path, base, sx, ss=4):
    f = font(path, size * ss)
    adv = f.getlength(text) / ss
    W = int(np.ceil(adv)) + 1
    im = Image.new('L', (W * ss, h * ss), 0)
    ImageDraw.Draw(im).text((0, round(base * ss)), text, font=f, fill=255, anchor='ls')
    w = max(1, int(np.ceil(adv * sx)))
    im = im.resize((max(w, int(round(W * sx))), h), Image.BOX)
    return (np.asarray(im, np.float32) / 255)[:, :w]

def big_start(t, rect, lines):
    """開始/終了 style (N1 build_examgfx): red->yellow fill by row taken
    from the original, thin dark outline, cyan drop shadow lower right."""
    x0, y0, W, H = rect
    orig = t.orig[y0:y0 + H, x0:x0 + W]
    fills = (4, 6, 7, 8, 9, 10)
    rowfill = []
    for r in range(H):
        c = collections.Counter(int(v) for v in orig[r] if v in fills)
        rowfill.append(c.most_common(2) if c else None)
    known = [r for r in range(H) if rowfill[r]]
    a = np.zeros((H, W), np.float32)
    pitch = H // len(lines)
    sx = group_sx(lines, 40, W - 6, F_IMPACT)
    for i, s in enumerate(lines):
        la = line_alpha(s, pitch, 40, F_IMPACT, base=pitch - 5, sx=sx)
        lx = (W - la.shape[1]) // 2          # face centred; the shadow hangs right
        a[i * pitch:(i + 1) * pitch, lx:lx + la.shape[1]] = np.maximum(
            a[i * pitch:(i + 1) * pitch, lx:lx + la.shape[1]], la)
    m = a > 0.45
    o = ndimage.binary_dilation(m, np.ones((3, 3), bool))
    sh = shift(o, 2, 3) | shift(o, 1, 2) | shift(o, 2, 2) | shift(o, 1, 1)
    out = np.zeros((H, W), np.uint8)
    out[sh] = 15
    out[sh & ~shift(sh, 1, 1)] = 14
    for r in range(H):
        out[r, o[r]] = 1 if r < H // 2 else 3
    for r in range(H):
        k = rowfill[r] or rowfill[min(known, key=lambda q: abs(q - r))]
        a_, b_ = k[0][0], (k[1][0] if len(k) > 1 else k[0][0])
        pb = (k[1][1] / (k[0][1] + k[1][1])) if len(k) > 1 else 0
        for c in np.nonzero(m[r])[0]:
            out[r, c] = b_ if (pb > 0.3 and (r + c) % 2 == 0) else a_
    t.px[y0:y0 + H, x0:x0 + W] = out
    return sx

def st_hud(a):
    """CEFT4 pal 8 ramp from 1 (white) to 15 (dark blue): 1 px white
    outline, fill 15, antialias through the ramp (N1 build_examgfx)."""
    a = np.pad(a, 1)
    o = ndimage.grey_dilation(a, footprint=np.ones((3, 3))) > 0.3
    out = np.zeros(a.shape, np.uint8)
    out[o] = 1
    m = a > 0.12
    out[m] = np.clip(np.rint(1 + a[m] * 14), 2, 15)
    return out[1:-1]

def hud_alpha(text, size, base, h=16, shear=0.0, ss=4, path=F_COND):
    f = font(path, size * ss)
    W = (int(f.getlength(text)) // ss + 8 + int(shear * h)) * ss
    im = Image.new('L', (W, h * ss), 0)
    ImageDraw.Draw(im).text((4 * ss, base * ss), text, font=f, fill=255, anchor='ls')
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * base * ss, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(im.resize((W // ss, h), Image.BOX), np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    return a[:, cols.min():cols.max() + 1]

CEFT4_RECTS = ((256, 0, 80, 96), (336, 0, 80, 96), (184, 96, 24, 32))
SUBJ_CELLS = ((184, 96), (184, 112))   # 科, 目: 24x16 cells, drawn at x 30, 50
SUBJ_STEP, SUBJ_MAX = 20, 39            # screen step between cells; digit at x 70
SUBJ_END = 38                           # last ink column (screen x 30 + 38 = 68)

def paint_ceft4(t, tl, log):
    for key, rect in (('ceft_start', (256, 0, 80, 96)), ('ceft_end', (336, 0, 80, 96))):
        sx = big_start(t, rect, tl[key].split('|'))
        log.append(f'{key}: "{tl[key]}" sx {sx:.2f}')
    s = st_hud(hud_alpha(tl['ceft_subject'], 14, 11, shear=0.2))
    w = s.shape[1]
    assert w <= SUBJ_MAX and not s[15].any(), w
    s = np.pad(s, ((0, 0), (SUBJ_END + 1 - w, 0)))   # flush right before the digit
    for k, (cx, cy) in enumerate(SUBJ_CELLS):
        t.px[cy:cy + 16, cx:cx + 24] = 0
        part = s[:, k * SUBJ_STEP:(k + 1) * SUBJ_STEP]
        t.px[cy:cy + 16, cx:cx + part.shape[1]] = part
    log.append(f'ceft_subject: "{tl["ceft_subject"]}" w {w}, screen x {30 + SUBJ_END + 1 - w}-{30 + SUBJ_END}')

# ---------------------------------------------------------------- build

def slot(name):
    s = os.path.getsize(f'private/disc/DAT2/{name}')
    return s + (-s % 2048)

def save(name, d, dry, log):
    if dry: return 0
    c = lz.compress(bytes(d), name=name); s = slot(name)
    log.append(f'{name}: {len(c)} bytes, slot {s}')
    if len(c) > s:
        log.append(f'{name}: too large'); return 1
    open(f'private/build/{name}.lz', 'wb').write(d)
    open(f'private/build/{name}', 'wb').write(c)
    return 0

def rd(name):
    return bytearray(open(f'private/unpacked/{name}.lz', 'rb').read())

# ------------------------------------------------- start menu 戻る records
# Records s16 n0, cnt, X, Y, (u/16, row, w/8); screen x = X - 8*n0. The
# other items sit centred on the cursor bar; 戻る (n0 2, X 256, 40 px)
# sat on 260.
MENU_X = 270.5                  # centre of the cursor bar, x 184-357
BACK_RECS = {'MGAME01.BIN': (0xa0, 0x100), 'MGAME02.BIN': (0x30,), 'MGAME03.BIN': (0x30,),
             'MGAME05.BIN': (0x30,), 'MGAME06.BIN': (0x94, 0xf4), 'MGAME07.BIN': (0x174, 0x1d4)}

def patch_back(log):
    """Move the 戻る records so that the EN label's ink centre is on the bar."""
    d = bytearray(open('private/unpacked/MINITTL1.BIZ.lz', 'rb').read())
    prev = minittl_atlas.paint(d)
    a, b = minittl_atlas.ink_span('title_back', prev)
    left = int(np.floor(MENU_X - (a + b) / 2 + 0.5))   # screen x of the sprite
    out = {}
    for name, offs in BACK_RECS.items():
        o = bytearray(open(f'private/disc/DAT1/{name}', 'rb').read())
        for off in offs:
            n0, cnt, X, Y, u, row, w = struct.unpack_from('<7h', o, off)
            assert (n0, cnt, X, u, row, w) == (2, 1, 256, 9, 4, 5), (name, hex(off))
            struct.pack_into('<h', o, off + 4, left + 8 * n0)
        out[name] = o
    log.append(f'Back records: X {left + 16} (sprite x {left}, ink {left + a}-{left + b})')
    return out

def build(png=False, dry=False):
    tl = load_tl(); log = []; problems = 0
    out = {}
    d = rd('CEFT5.BIZ')
    before, after = paint_hud(d, tl, log)
    out['CEFT5.BIZ'] = d
    for name, xy in BOX_COPIES.items():
        dd = rd(f'{name}.BIZ'); t = Tim4(dd, 0)
        copy_box(t, before, after, xy, name, log)
        if name == 'DANCE': paint_dance(t, tl, log)
        t.store(); out[f'{name}.BIZ'] = (dd, t)
    for name in ('CEFT4.BIZ', 'CEFT6.BIZ'):
        # CEFT6 differs from CEFT4 in a few cells (e.g. the 1-3 回目 column)
        # but not in the painted ones
        dd = rd(name); t = Tim4(dd, 0x10800)
        if name == 'CEFT6.BIZ':
            t4 = out['CEFT4.BIZ'][1]
            for x, y, w, h in CEFT4_RECTS:
                assert (t.orig[y:y + h, x:x + w] == t4.orig[y:y + h, x:x + w]).all(), (name, x, y)
            assert t.clut == t4.clut
        paint_ceft4(t, tl, log if name == 'CEFT4.BIZ' else [])
        t.store(); out[name] = (dd, t)
    for name, v in out.items():
        problems += save(name, v[0] if isinstance(v, tuple) else v, dry, log)
    for name, o in patch_back(log).items():
        if not dry: open(f'private/build/{name}', 'wb').write(o)
    for line in log: print(line)
    if png: previews(before, after, d, out)
    print(f'minihud: problems {problems}')
    return problems

def previews(before, after, d5, out):
    """Raw JP/EN atlas crops (1x) for a quick look; screen mock-ups are
    made by private/re/minihud/mock.py."""
    os.makedirs(PREV, exist_ok=True)
    clut5 = struct.unpack_from('<1024H', d5, CEFT_TIM + 0x14)
    def rgb(px, clut, c, sl):
        pal = np.array([rgb15(v) for v in clut[c * 16:c * 16 + 16]]); pal[0] = (40, 40, 40)
        return pal[px[sl]]
    A = AGAIN; sl = np.s_[A['y']:A['y'] + A['h'], A['x']:A['x'] + A['w']]
    Image.fromarray(np.concatenate([rgb(before, clut5, 49, sl), rgb(after, clut5, 49, sl)], 1)
                    .astype(np.uint8)).save(f'{PREV}/again.png')
    dd, t = out['DANCE.BIZ']
    sl = np.s_[0:192, 512:768]
    Image.fromarray(np.concatenate([rgb(t.orig, t.clut, 36, sl), rgb(t.px, t.clut, 36, sl)], 1)
                    .astype(np.uint8)).save(f'{PREV}/dance_lines.png')
    c4, t4 = out['CEFT4.BIZ']
    sl = np.s_[0:96, 256:416]
    Image.fromarray(np.concatenate([rgb(t4.orig, t4.clut, 2, sl), rgb(t4.px, t4.clut, 2, sl)], 1)
                    .astype(np.uint8)).save(f'{PREV}/ceft4.png')

if __name__ == '__main__':
    sys.exit(1 if build('--png' in sys.argv, '--dry' in sys.argv) else 0)
