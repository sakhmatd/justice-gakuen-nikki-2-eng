# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Extras card viewer cards (DAT2/CARD.BIZ, entries 0-59) in English.

Every card is one 8bpp TIM (304x384, own 256-colour CLUT). The JP text is
painted into the picture; this script erases it, draws the EN from
tl/CARDBIZ.txt and requantises the changed pixels into the card's own
CLUT. Groups: footers (40 cards), letters (029-048, on a clean copy of
the shared ruled paper), illustrator names (009-015), anime credits
(049-057) and the design-note cards. Every repainted entry is
recompressed (lz.py, CARD.BIZ cap); the archive is relaid at 0x800
boundaries with a new TOC and must fit the JP length (0x525000).

Writes private/build/CARD.BIZ only. --png [N...] writes previews to
private/re/card/review/ (NNN.png, en|jp side by side) and writes nothing
else. Facts: private/re/card/findings.org.
Usage: python3 tools/build_card.py [--png [N...]]
"""
import os, sys, re, struct
import numpy as np
from scipy import ndimage
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFont
import fontpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lz, arc
import inpaint as I

NAME = 'CARD.BIZ'
SRC = 'private/disc/DAT2/' + NAME
OUT = 'private/build/' + NAME
TL = 'tl/CARDBIZ.txt'
REVIEW = 'private/re/card/review/'

# ---------------------------------------------------------------- TIM

class Card:
    """One card TIM: idx (384, 304) CLUT indices, pal (256, 3) RGB."""
    def __init__(self, n, data):
        self.n, self.data = n, bytearray(data)
        cl = struct.unpack_from('<I', data, 8)[0]
        cw, ch = struct.unpack_from('<2H', data, 16)
        self.clut = np.array(struct.unpack_from(f'<{cw * ch}H', data, 20))
        p = 8 + cl
        iw, ih = struct.unpack_from('<2H', data, p + 8)
        self.w, self.h, self.off = iw * 2, ih, p + 12
        self.idx = np.frombuffer(bytes(data[self.off:self.off + self.w * self.h]),
                                 np.uint8).reshape(self.h, self.w).copy()
        c = self.clut[:256]
        self.pal = np.stack([(c & 31), (c >> 5) & 31, (c >> 10) & 31], -1) * 255.0 / 31
        # 0x0000 is transparent on the GPU: never pick it for new pixels
        self.bad = [i for i in range(256) if c[i] == 0]
        self.orig = self.idx.copy()

    def rgb(self):
        return self.pal[self.idx]

    def put(self, rgb, where=None):
        """Quantise rgb into the CLUT where it changed (or inside where)."""
        ch = np.abs(rgb - self.pal[self.idx]).sum(-1) > 3
        if where is not None: ch &= where
        self.idx = quantise(rgb, self.pal, self.idx, ch, self.bad)

    def tim(self):
        out = bytearray(self.data)
        out[self.off:self.off + self.w * self.h] = self.idx.tobytes()
        return bytes(out)

    def image(self, idx=None):
        return Image.fromarray(self.pal[self.idx if idx is None else idx].astype(np.uint8))

def quantise(rgb, pal, idx, ch, bad=()):
    """Nearest CLUT colour (weighted RGB) for every pixel in ch."""
    out = idx.copy()
    wt = np.array([0.30, 0.59, 0.11]) ** 0.5
    pw = pal * wt
    sel = np.where(ch.ravel())[0]
    flat = rgb.reshape(-1, 3)[sel]
    for k in range(0, len(flat), 4096):
        q = flat[k:k + 4096] * wt
        dd = ((q[:, None, :] - pw[None]) ** 2).sum(-1)
        if len(bad): dd[:, list(bad)] = 1e12
        out.ravel()[sel[k:k + 4096]] = dd.argmin(1)
    return out

# ---------------------------------------------------------------- erase

def exemplar(idx, pal, mask, avoid=None, P=7, R=30):
    """Exemplar fill of mask (build_beta.py): copy CLUT indices from the
    best matching patch within R px that has no pixel in mask or avoid,
    most-known target patch first."""
    idx = idx.copy()
    if avoid is None: avoid = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if not len(ys): return idx
    hp = P // 2
    y0, y1 = max(0, ys.min() - R - P), min(idx.shape[0], ys.max() + R + P + 1)
    x0, x1 = max(0, xs.min() - R - P), min(idx.shape[1], xs.max() + R + P + 1)
    sub = idx[y0:y1, x0:x1]
    mk = mask[y0:y1, x0:x1].copy()
    orig_known = ~(mk | avoid[y0:y1, x0:x1])
    src_ok = ndimage.minimum_filter(orig_known.astype(np.uint8), P, mode='constant') > 0
    src_free = ndimage.minimum_filter((~mk).astype(np.uint8), P, mode='constant') > 0
    known = orig_known.copy()
    rgb = pal[sub]
    h, w = sub.shape
    win = lambda: sliding_window_view(np.pad(rgb, ((hp, hp), (hp, hp), (0, 0))), (P, P), (0, 1))
    win_rgb = win()
    while mk.any():
        cnt = ndimage.uniform_filter(known.astype(float), P, mode='constant')
        front = mk & ndimage.binary_dilation(known)
        if not front.any(): front = mk
        c = np.where(front, cnt, -1)
        ty, tx = np.unravel_index(np.argmax(c), c.shape)
        tgt = win_rgb[ty, tx]
        kn = np.pad(known, hp)[ty:ty + P, tx:tx + P]
        sy0, sy1 = max(hp, ty - R), min(h - hp, ty + R + 1)
        sx0, sx1 = max(hp, tx - R), min(w - hp, tx + R + 1)
        cand = win_rgb[sy0:sy1, sx0:sx1]
        ssd = (((cand - tgt) ** 2) * kn).sum((2, 3, 4))
        ok = src_ok[sy0:sy1, sx0:sx1]
        if not ok.any():
            ok = src_free[sy0:sy1, sx0:sx1]
        if not ok.any():
            raise SystemExit('exemplar: no source patch (widen R or shrink P)')
        ssd = np.where(ok, ssd, np.inf)
        gy, gx = np.mgrid[sy0:sy1, sx0:sx1]
        ssd = ssd * (1 + 0.002 * np.hypot(gy - ty, gx - tx))
        by, bx = np.unravel_index(np.argmin(ssd), ssd.shape)
        by += sy0; bx += sx0
        for dy in range(-hp, hp + 1):
            yy = ty + dy
            if not 0 <= yy < h: continue
            for dx in range(-hp, hp + 1):
                xx = tx + dx
                if 0 <= xx < w and mk[yy, xx]:
                    sub[yy, xx] = sub[by + dy, bx + dx]
                    rgb[yy, xx] = pal[sub[yy, xx]]
                    mk[yy, xx] = False; known[yy, xx] = True
        win_rgb = win()
    idx[y0:y1, x0:x1] = sub
    return idx

def smooth_fill(rgb, mk, iters=400):
    """Fill mk by repeated averaging of the 4 neighbours (flat panels and
    gradients)."""
    out = rgb.copy()
    if not mk.any(): return out
    ys, xs = np.nonzero(mk)
    y0, y1 = max(0, ys.min() - 2), min(mk.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(mk.shape[1], xs.max() + 3)
    sub = out[y0:y1, x0:x1]; m = mk[y0:y1, x0:x1]
    known = ~m
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    for c in range(3):
        ch = sub[..., c]
        ch[m] = ch[known].mean()
    # coarse start: rows and columns interpolated from the known edge
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            avg = ndimage.convolve(ch, k, mode='nearest') / 4
            ch[m] = avg[m]
    return out

def rect_mask(shape, r):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = r
    m[max(0, y0):y1, max(0, x0):x1] = True
    return m

def ink_mask(rgb, r, thr=60, size=7, grow=1, sel=None):
    """Thin text strokes in rect r: pixels far from their local median
    (size px), grown by grow px. sel(rgb) -> bool limits it further."""
    m = np.zeros(rgb.shape[:2], bool)
    x0, y0, x1, y1 = r
    a = rgb[y0:y1, x0:x1]
    med = np.stack([ndimage.median_filter(a[..., c], size) for c in range(3)], -1)
    d = np.abs(a - med).sum(-1) > thr
    if sel is not None: d &= sel(a)
    m[y0:y1, x0:x1] = d
    if grow: m = ndimage.binary_dilation(m, iterations=grow) & rect_mask(m.shape, r)
    return m

# ---------------------------------------------------------------- text

_fonts = {}
def font(path, size):
    k = (path, size)
    if k not in _fonts:
        _fonts[k] = fontpath.truetype(path, size)
    return _fonts[k]

def text_alpha(lines, ft, lead, w, h, align='left', spacing=0):
    """Alpha (h, w) float 0-1 with lines drawn from the top, line pitch
    lead. Returns alpha and the list of line widths."""
    im = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(im)
    widths = []
    for i, ln in enumerate(lines):
        tw = textlen(ln, ft, spacing)
        widths.append(tw)
        x = {'left': 0, 'center': (w - tw) / 2, 'right': w - tw}[align]
        draw_line(d, round(x), i * lead, ln, ft, spacing)
    return np.asarray(im, float) / 255, widths

def textlen(s, ft, spacing=0):
    s = s.replace('~~', '')
    return ft.getlength(s) + spacing * max(0, len(s) - 1)

def draw_line(d, x, y, s, ft, spacing=0, fill=255, strike=None, anchor='la'):
    """Draw s at (x, y); ~~word~~ is drawn struck out (strike() gets the
    span)."""
    parts = s.split('~~')
    for k, p in enumerate(parts):
        if not p: continue
        if spacing:
            for ch in p:
                d.text((x, y), ch, font=ft, fill=fill, anchor=anchor)
                x += ft.getlength(ch) + spacing
        else:
            d.text((x, y), p, font=ft, fill=fill, anchor=anchor)
            x0 = x; x += ft.getlength(p)
            if k % 2 and strike: strike(d, x0, x, y)

def wrap(text, ft, width, spacing=0):
    out = []
    for para in text.split(' / '):
        cur = ''
        for wd in para.split(' '):
            t = (cur + ' ' + wd).strip()
            if cur and textlen(t, ft, spacing) > width:
                out.append(cur); cur = wd
            else:
                cur = t
        out.append(cur)
    return out

def grow(a, r):
    """Max filter of an alpha map with a disc of radius r (outline)."""
    if r <= 0: return a
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (np.hypot(yy, xx) <= r + 0.01)
    return ndimage.maximum_filter(a, footprint=disc)

def shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

def comp(rgb, alpha, col, x, y):
    """Blend colour col into rgb with alpha placed at (x, y)."""
    h, w = alpha.shape
    H, W = rgb.shape[:2]
    ax0, ay0 = max(0, -x), max(0, -y)
    bx0, by0 = max(0, x), max(0, y)
    bx1, by1 = min(W, x + w), min(H, y + h)
    if bx1 <= bx0 or by1 <= by0: return
    a = alpha[ay0:ay0 + by1 - by0, ax0:ax0 + bx1 - bx0][..., None]
    reg = rgb[by0:by1, bx0:bx1]
    reg[:] = reg * (1 - a) + np.asarray(col, float) * a

# ---------------------------------------------------------------- tl

def load_tl():
    tl = {}
    for ln in open(TL, encoding='utf-8'):
        if ln.startswith('#') or ':' not in ln: continue
        k, v = ln.split(':', 1)
        tl[k.strip()] = v.strip()
    return tl

# ---------------------------------------------------------------- groups

# Footer "熱血カード NO.nn": top left of 熱 per card (found by matching
# the footer of 012). Right footers keep their right edge (x + 76), left
# ones their left edge.
FOOT = {0: (208, 371), 1: (209, 370), 2: (208, 371), 3: (208, 371), 4: (214, 371), 5: (10, 368), 6: (207, 371), 7: (208, 371), 8: (204, 371), 9: (219, 373), 10: (208, 366), 11: (212, 364), 12: (14, 364), 13: (204, 374), 14: (217, 369), 15: (217, 370), 16: (208, 372), 17: (14, 372), 18: (210, 371), 19: (220, 7), 20: (4, 5), 21: (217, 368), 22: (215, 369), 23: (219, 371), 24: (9, 366), 25: (219, 371), 26: (207, 370), 27: (218, 366), 28: (221, 369), 49: (221, 371), 50: (217, 369), 51: (217, 370), 52: (217, 368), 53: (225, 375), 54: (216, 367), 55: (215, 374), 56: (217, 369), 57: (217, 368), 58: (213, 375), 59: (9, 370)}
FOOT_W = 76
# Footers on a translucent or textured band: band rows (top, bottom). The
# background is sampled inside the band and the fill copies band pixels.
FOOT_BAND = {13: (372, 383), 27: (364, 376), 51: (369, 379), 53: (374, 383),
             54: (365, 376), 58: (374, 383), 59: (368, 380)}
# Footers where "far from the background" must not count as text (an art
# edge runs through the box).
FOOT_NEAR = {6}
FOOT_FONT = ('msgothic.ttc', 10)

# Core strokes of 熱血カード (footer of 012), 9 x 50 from (x, y).
FOOT_INK = np.array([[c == '#' for c in r] for r in """
..#...#......#....................................
.####.#......#.........#.................#..##....
########...######...######...............#..#.....
.#.#####..#.#.#.#...#######..............##.......
####.###..#.#.#.#.....#...#..########....#.##.....
..##.#.##.#.#.#.#.....#...#..............#..#.....
####.#.##.#.#.#.#....##..##..............#........
.#.#.#.#..#.#.#.#...##..##...............#........
##.#.#..##########..#...#................#........
""".split()])

def footer_rows(n, x, y):
    return FOOT_BAND.get(n, (max(0, y - 3), min(383, y + 10)))

def footer_bg(rgb, n, x, y):
    """Per-column background just above and below the footer, (W, 3)."""
    t, b = footer_rows(n, x, y)
    return (rgb[t, x - 2:x + FOOT_W + 3] + rgb[b, x - 2:x + FOOT_W + 3]) / 2

def footer_colour(rgb, n, x, y):
    """JP footer fill: median of the 熱血カード core strokes that differ
    most from the background."""
    core = rgb[y:y + 9, x:x + 50]
    bg = footer_bg(rgb, n, x, y)[2:52][None].repeat(9, 0)
    d = np.abs(core - bg).sum(-1)[FOOT_INK]
    px = core[FOOT_INK]
    return np.median(px[d >= np.median(d)], 0)

def footer_mask(rgb, n, x, y, fill):
    """JP footer pixels: closer to the fill than to the column background
    or far from that background, grown by 1 px."""
    x0, y0, x1, y1 = x - 2, y - 1, x + FOOT_W + 3, min(384, y + 10)
    if n in FOOT_BAND:
        y0, y1 = FOOT_BAND[n][0] + 1, FOOT_BAND[n][1] + 1
    a = rgb[y0:y1, x0:x1]
    bg = footer_bg(rgb, n, x, y)[None]
    dbg = np.abs(a - bg).sum(-1)
    dfi = np.abs(a - fill).sum(-1)
    t = dfi < dbg
    if n not in FOOT_NEAR: t |= dbg > 90
    m = np.zeros(rgb.shape[:2], bool)
    m[y0:y1, x0:x1] = t
    m = ndimage.binary_dilation(m, iterations=1)
    if n in FOOT_BAND:
        m &= rect_mask(m.shape, (0, y0, 304, y1))
    return m

def footer(card, tl, log, style=None):
    n = card.n
    x, y = FOOT[n]
    rgb = card.rgb()
    fill = footer_colour(rgb, n, x, y)
    m = footer_mask(rgb, n, x, y, fill)
    avoid = None
    if n in FOOT_BAND:
        t, b = FOOT_BAND[n]
        avoid = ~rect_mask(m.shape, (0, t, 304, b + 1))
    erase(card, None, mask=m, avoid=avoid, P=5, R=90 if avoid is not None else 30)
    rgb = card.rgb()
    st = dict(font=FOOT_FONT, shadow=None)
    if style: st.update(style)
    ft = font(*st['font'])
    t = tl['footer'].replace('{nn}', f'{n + 1:02d}')
    tw = int(np.ceil(textlen(t, ft)))
    a, _ = text_alpha([t], ft, 12, tw + 2, 13)
    tx = x + (FOOT_W - tw) // 2      # centred on the footer box
    ty = y - 1
    if st['shadow']:
        comp(rgb, shift(a, 1, 1) * st['shadow'][1], st['shadow'][0], tx, ty)
    comp(rgb, a, fill, tx, ty)
    card.put(rgb)
    log.append(f'{n:03d} footer {t!r} x {tx}-{tx + tw} fill {tuple(int(v) for v in fill)}')

# ---------------------------------------------------------------- letters

LETTERS = range(29, 49)
PAPER_Y = 228                          # cream ruled paper rows 228-383
RULES = [248, 270, 293, 316, 338, 362]  # ruled lines (text sits on them)
LETTER_X = (12, 294)
# Writer fonts (handwriting, one per writer) and the size range tried.
HAND = {
    29: 'GochiHand-Regular.ttf',            # Batsu
    30: 'FuzzyBubbles-Regular.ttf',      # Hinata
    31: 'Kalam-Light.ttf',                       # Kyosuke
    32: 'IndieFlower-Regular.ttf',        # Shoma
    33: 'PatrickHand-Regular.ttf',        # Natsu
    34: 'ArchitectsDaughter-Regular.ttf',  # Roberto
    35: 'KiteOne-Regular.ttf',                # Roy
    36: 'GloriaHallelujah.ttf',      # Tiffany
    37: 'KleeOne-Regular.ttf',                # Boman
    38: 'CaveatBrush-Regular.ttf',        # Edge
    39: 'Handlee-Regular.ttf',                 # Akira
    40: 'Sriracha-Regular.ttf',               # Gan
    41: 'Kalam-Regular.ttf',                     # Hideo
    42: 'Cedarville-Cursive.ttf',   # Kyoko
    43: 'ZenKurenaido-Regular.ttf',      # Hyo
    44: 'Mali-Regular.ttf',                       # Sakura
    45: 'Kalam-Bold.ttf',                        # Daigo
    46: 'Dekko-Regular.ttf',                     # Hayato
    47: 'Neucha.ttf',                           # Nagare
    48: 'Gaegu-Regular.ttf',                     # Ran
}
# Signature pixels that stay (Latin in the JP): 035 Roy Bromwell, 036 xxx.
SIGN_KEEP = {35: (167, 338, 292, 378), 36: (274, 338, 300, 368)}
_paper = None

def paper(cards_src):
    """Consensus paper (rows 228-383): per pixel the 10th of the 20 letter
    cards' colours in sorted order (a real card colour, not a mean)."""
    global _paper
    if _paper is None:
        A = np.stack([Card(n, d).rgb()[PAPER_Y:] for n, d in cards_src()])
        _paper = np.sort(A, 0)[9]
    return _paper

def xheight(ft):
    bb = ft.getbbox('x')
    return bb[3] - bb[1]

def letter_size(path, text, nlines, sign, sxh=(6, 9)):
    """Largest size with x-height in sxh whose wrap fits nlines lines and
    leaves room for the signature line."""
    best = None
    for sz in range(8, 40):
        ft = font(path, sz)
        xh = xheight(ft)
        if xh < sxh[0]: continue
        if xh > sxh[1]: break
        lines = wrap(text, ft, LETTER_X[1] - LETTER_X[0])
        if len(lines) <= nlines: best = (sz, ft, lines)
    return best

def strike(d, x0, x1, y):
    """Crossed-out word: two loose strokes through the x-height."""
    for k, dy in enumerate((-4, -2)):
        d.line([(x0 - 1, y + dy + k), (x1 + 1, y + dy - k)], fill=255, width=1)

def letter(card, tl, log):
    n = card.n
    rgb = card.rgb()
    P = paper(_letter_src)
    sub = rgb[PAPER_Y:]
    core = np.abs(sub - P).sum(-1) > 40
    ink = ndimage.binary_dilation(core, iterations=1)
    if n in SIGN_KEEP:
        x0, y0, x1, y1 = SIGN_KEEP[n]
        ink[y0 - PAPER_Y:y1 - PAPER_Y, x0:x1] = False
    lum = sub[core].mean(-1)
    inkc = np.median(sub[core][lum <= np.percentile(lum, 10)], 0)
    sub[ink] = P[ink]
    body, sign = tl[f'c{n:03d}'], tl.get(f'c{n:03d}_sign')
    path = HAND[n]
    got = letter_size(path, body, 5, sign)
    if got is None: raise SystemExit(f'{n:03d}: letter does not fit')
    sz, ft, lines = got
    im = Image.new('L', (card.w, card.h), 0)
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        draw_line(d, LETTER_X[0], RULES[i] - 3, ln, ft, strike=strike, anchor='ls')
    row = min(len(lines), 5)
    if sign:
        right = LETTER_X[1] - 4
        if n == 36: right = SIGN_KEEP[36][0] - 4
        ss = sz + 2
        while textlen(sign, font(path, ss)) > right - LETTER_X[0] - 30: ss -= 1
        sf = font(path, ss)
        sw = textlen(sign, sf)
        if n != 35:
            draw_line(d, round(right - sw), RULES[row] - 3, sign, sf, anchor='ls')
    a = np.asarray(im, float) / 255
    comp(rgb, a, inkc, 0, 0)
    card.put(rgb)
    log.append(f'{n:03d} letter {path.split("/")[0]} {sz}px {len(lines)} lines, ink {tuple(int(v) for v in inkc)}')

def _letter_src():
    orig = open(SRC, 'rb').read()
    toc = arc.toc(orig)
    for n in LETTERS:
        o, s = toc[n]
        yield n, lz.decompress(orig[o:o + s])[0]

# ---------------------------------------------------------------- labels

# Fill for every erase: 'best' = LaMa (private/models/big-lama.pt) if
# torch and the weights are there, else inpaint.py 'auto'. CARD_FILL=legacy
# gives the old exemplar / diffusion fills.
FILL = os.environ.get('CARD_FILL', 'best')
RING = 5            # rim around the script's mask (JP shadows and glows)
DITHER_MAX = 24     # max sum |RGB| error of a dithered fill pixel
STATS = {}          # card -> fill pixels that fell back to nearest colour

def grow_mask(rgb, mask, protect=None, ring=RING, thr=40, clip=None):
    """The script's mask plus a ring px rim of pixels that differ from the
    background diffused in around it (outline, shadow, glow, halo). clip:
    the step's rect; the rim stays inside it (plus the 1 px halo)."""
    if not mask.any() or not ring: return mask
    r = I._bbox(mask, ring + 2, rgb.shape)
    if clip is not None:
        r = (max(r[0], clip[0]), max(r[1], clip[1]), min(r[2], clip[2]), min(r[3], clip[3]))
    return I.text_mask(rgb, r, pred=lambda a: mask, ring=ring, thr=thr, halo=1,
                       min_px=0, close=0, protect=protect)

def erase(card, r, how='exemplar', thr=60, size=9, grow_px=1, sel=None,
          mask=None, avoid=None, P=7, R=30, ring=RING, fill=None):
    """Erase JP strokes in rect r. mask: the text pixels (default: stroke
    detection in r, or the whole rect if how is 'rect'). The mask gets a
    ring px rim (grow_mask), is filled with FILL and written back into the
    CLUT with dither inside the mask only. avoid: never used as a source
    and never added by the rim. fill: per-step method (only when FILL is
    'best'). how/P/R only matter for CARD_FILL=legacy
    ('smooth' = diffusion, else exemplar copy)."""
    rgb = card.rgb()
    if mask is None:
        if how == 'rect': mask = rect_mask(rgb.shape[:2], r)
        else: mask = ink_mask(rgb, r, thr, size, grow_px, sel)
    if FILL == 'legacy':
        if how == 'smooth':
            card.put(smooth_fill(rgb, mask), mask)
        else:
            card.idx = exemplar(card.idx, card.pal, mask, avoid, P, R)
        return mask
    drawn = getattr(card, 'drawn', None)     # EN drawn by earlier steps
    if drawn is not None and drawn.any():
        mask = mask & ~drawn
        avoid = drawn if avoid is None else avoid | drawn
    protect = None if avoid is None else avoid & ~mask
    m = grow_mask(rgb, mask, protect, ring, clip=r)
    meth = FILL
    if fill and FILL == 'best' and (not fill.startswith('lama') or I.lama_ok()): meth = fill
    out = I.fill(rgb, m, meth, avoid=avoid)
    # keep_stp off: the STP entry of most cards is 0x8000 (opaque black,
    # the card sprites are not semi-transparent); with it on, JP pixels in
    # 0x8000 could only become black again (as quantise() does, any
    # non-0x0000 entry may be chosen)
    fs = I.to_clut(out, card.idx, card.clut[:256], m, pal=card.pal, keep_stp=False)
    near = I.to_clut(out, card.idx, card.clut[:256], m, pal=card.pal, keep_stp=False,
                     dither='none')
    # as mgbg_cards.py: where the dithered pick is far from the fill (a
    # saturated art colour as a stray dot), take the nearest colour
    stray = np.abs(card.pal[fs] - out).sum(-1) > DITHER_MAX
    STATS[card.n] = STATS.get(card.n, 0) + int((stray & m).sum())
    card.idx = np.where(stray, near, fs)
    return m

def label(card, lines, ft, fill, x, y, align='left', lead=None, outline=None,
          shadow=None, underline=None, spacing=0, gradient=None):
    """Draw lines with the top-left (align left), centre or right end at x
    and the first baseline at y. outline (colour, radius), shadow (colour,
    dx, dy, alpha), underline (colour, dy, thickness). gradient (top,
    bottom) colours replace fill."""
    if lead is None: lead = round(ft.size * 1.2)
    asc, desc = ft.getmetrics()
    pad = 6
    widths = [textlen(l, ft, spacing) for l in lines]
    W = int(max(widths)) + 2 * pad + 2
    H = asc + desc + lead * (len(lines) - 1) + 2 * pad
    im = Image.new('L', (W, H), 0); ul = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im); du = ImageDraw.Draw(ul)
    for i, (ln, w) in enumerate(zip(lines, widths)):
        lx = pad + {'left': 0, 'center': (max(widths) - w) / 2, 'right': max(widths) - w}[align]
        by = pad + asc + i * lead
        draw_line(d, round(lx), by, ln, ft, spacing, anchor='ls')
        if underline:
            for t in range(underline[2]):
                du.line([(round(lx), by + underline[1] + t), (round(lx + w), by + underline[1] + t)], fill=255)
    a = np.asarray(im, float) / 255
    u = np.asarray(ul, float) / 255
    mw = max(widths)
    ox = x - pad - {'left': 0, 'center': mw / 2, 'right': mw}[align]
    oy = y - pad - asc
    ox, oy = int(round(ox)), int(round(oy))
    rgb = card.rgb()
    if shadow:
        c, dx, dy, al = shadow
        sa = np.maximum(grow(a, outline[1]) if outline else a, u)
        comp(rgb, shift(sa, dx, dy) * al, c, ox, oy)
    if outline:
        comp(rgb, grow(np.maximum(a, u), outline[1]), outline[0], ox, oy)
    if underline:
        comp(rgb, u, underline[0], ox, oy)
    if gradient:
        top, bot = np.asarray(gradient[0], float), np.asarray(gradient[1], float)
        ys = np.nonzero(a.max(1) > 0.1)[0]
        g0, g1 = (ys.min(), ys.max()) if len(ys) else (0, 1)
        t = np.clip((np.arange(a.shape[0]) - g0) / max(1, g1 - g0), 0, 1)[:, None, None]
        colimg = top * (1 - t) + bot * t
        h, w = a.shape
        reg = rgb[max(0, oy):oy + h, max(0, ox):ox + w]
        aa = a[max(0, -oy):max(0, -oy) + reg.shape[0], max(0, -ox):max(0, -ox) + reg.shape[1]][..., None]
        cc = np.broadcast_to(colimg, (h, w, 3))[max(0, -oy):max(0, -oy) + reg.shape[0], max(0, -ox):max(0, -ox) + reg.shape[1]]
        reg[:] = reg * (1 - aa) + cc * aa
    else:
        comp(rgb, a, fill, ox, oy)
    card.put(rgb)
    return ox, oy, a.shape

# ---------------------------------------------------------------- illustrators

TI, TB, TBI = 'timesi.ttf', 'timesbd.ttf', 'timesbi.ttf'
GOTH = 'msgothic.ttc'
# Per card: tl key, erase rect, erase mode, ink colour of the JP name (the
# erase mask is the pixels near it, grown by grow px; None = stroke
# detection), font, size, fill, x, baseline y, align, label options.
ILL = {
    9: dict(key='ill_shimamoto', r=(133, 3, 243, 27), how='exemplar', ink=None,
            font='times.ttf', size=18, fill=(255, 255, 255), x=139, y=22,
            align='left', opt=dict(shadow=((32, 0, 49), 1, 1, 0.8))),
    10: dict(key='ill_anzai', r=(11, 33, 100, 63), how='smooth', ink=(90, 82, 139),
             grow=2, font='ariali.ttf', size=16, fill=(90, 82, 139), x=65, y=55,
             align='center', opt=dict(outline=((246, 250, 255), 1.0),
                                      underline=((90, 82, 139), 3, 1))),
    11: dict(key='ill_anzai', r=(10, 40, 112, 65), how='smooth', ink=(230, 0, 24),
             grow=2, font=TBI, size=17, fill=(230, 0, 24), x=63, y=58,
             align='center', opt=dict(outline=((255, 236, 226), 1.0),
                                      underline=((230, 0, 24), 3, 1))),
    # 012: the glow of the name reaches x 190-282; the red "illustration"
    # underline above (y 25) is kept out of the fill context
    12: dict(key='ill_ohse', r=(188, 27, 286, 58), avoid=(150, 0, 304, 27),
             pred=lambda a: a[..., 1] < 225,         # ink and glow (on white)
             how='smooth', ink=(238, 24, 24),
             grow=2, font='arialbd.ttf', size=15, fill=(238, 24, 24), x=243,
             y=46, align='center', opt=dict(outline=((246, 164, 172), 1.0),
                                            underline=((238, 24, 24), 2, 1))),
    14: dict(key='ill_ohno', r=(86, 24, 170, 48), how='smooth', ink=None,
             font=TBI, size=19, fill=(82, 0, 90), x=165, y=43, align='right',
             opt=dict(underline=((82, 0, 90), 2, 1))),
    15: dict(key='ill_ohno', r=(102, 5, 174, 29), how='exemplar', ink=None,
             font='consolab.ttf', size=20, fill=(205, 213, 82), x=106, y=24,
             align='left', opt=dict(shadow=((40, 32, 0), 2, 2, 0.9))),
}

def colour_mask(rgb, r, col, dist=150, grow_px=2):
    """Pixels in r near colour col (or any of a list of colours), grown."""
    m = np.zeros(rgb.shape[:2], bool)
    x0, y0, x1, y1 = r
    cols = [col] if np.ndim(col) == 1 else col
    for c in cols:
        m[y0:y1, x0:x1] |= np.abs(rgb[y0:y1, x0:x1] - c).sum(-1) < dist
    if grow_px > 0: m = ndimage.binary_dilation(m, iterations=grow_px)
    return m & rect_mask(m.shape, r)

def illustrator(card, tl, log):
    n = card.n
    c = ILL[n]
    mask = colour_mask(card.rgb(), c['r'], c['ink'], grow_px=c['grow']) if c['ink'] else None
    if 'pred' in c:
        mask = c['pred'](card.rgb()) & rect_mask(card.idx.shape, c['r'])
    avoid = rect_mask(card.idx.shape, c['avoid']) if 'avoid' in c else None
    erase(card, c['r'], c['how'], thr=50, size=15, grow_px=1, mask=mask, avoid=avoid,
          ring=c.get('ring', RING), fill=c.get('method'))
    ft = font(c['font'], c['size'])
    t = tl[c['key']]
    ox, oy, (h, w) = label(card, [t], ft, c['fill'], c['x'], c['y'], c['align'], **c['opt'])
    log.append(f'{n:03d} illustrator {t!r} box x {ox}-{ox + w} y {oy}-{oy + h}')

def illustrator13(card, tl, log):
    """013: vertical name in the left border, read top to bottom like the
    Latin "illustration" on the right (column centre x 34, bottom y 370)."""
    r = (19, 287, 48, 373)
    m = colour_mask(card.rgb(), r, (180, 238, 238), 160, 2)
    erase(card, r, 'exemplar', mask=m, P=5, R=20)
    t = tl['ill_ohse']
    a, _ = text_alpha([t], font(GOTH, 17), 20, int(textlen(t, font(GOTH, 17))) + 4, 22)
    a = np.rot90(a, -1)                       # clockwise
    ys, xs = np.nonzero(a > 0.1)
    ox = 34 - (xs.min() + xs.max()) // 2
    oy = 370 - ys.max()
    rgb = card.rgb()
    comp(rgb, shift(a, 1, 1) * 0.9, (0, 24, 20), ox, oy)
    comp(rgb, a, (180, 238, 238), ox, oy)
    card.put(rgb)
    log.append(f'013 illustrator {t!r} vertical y {oy + ys.min()}-370')

# ---------------------------------------------------------------- anime credits

ROUND = 'MPLUSRounded1c-ExtraBold.ttf'
# Per card: erase rect, JP fill colour (mask = pixels near it, grown by
# grow px), EN fill, x (left edge, or centre if align is center), the
# baselines of the lines, max line width, style options for label().
ANIME = {
    # 049: lama paints light streaks into the flat blue (left edge, between
    # lines); 'auto' (gradient + texture) is clean there
    49: dict(r=(0, 288, 304, 364), ink=(240, 248, 0), grow=2, fill=(240, 248, 0), method='auto',
             x=6, base=[305, 322, 340, 357], width=294, align='left',
             opt=dict(outline=((16, 24, 64), 1.0))),
    50: dict(r=(0, 296, 304, 368), ink=(248, 200, 0), grow=3, fill=(248, 200, 0),
             x=8, base=[311, 328, 345, 362], width=290, align='left',
             opt=dict(shadow=((64, 16, 8), 2, 2, 1.0))),
    56: dict(r=(0, 184, 304, 255), ink=[(248, 248, 248), (40, 16, 32), (48, 32, 72)],
             dist=60, grow=2, fill=(248, 248, 248),
             x=11, base=[199, 215, 231, 248], width=284, align='left',
             opt=dict(outline=((40, 16, 40), 1.0), shadow=((48, 32, 72), 2, 2, 1.0))),
    51: dict(r=(4, 228, 300, 368), ink=(248, 248, 248), dist=100, grow=3, max=17,
             glow=[(248, 120, 232), (224, 32, 144), (216, 40, 128), (208, 48, 112),
                   (248, 216, 200), (240, 160, 220)], P=9, R=80,
             plate=((0, 230, 270, 368), (48, 0, 32), 0.6, 40), fill=(248, 248, 248), x=10,
             base=[248, 267, 286, 305, 324, 343, 362], width=280, align='left',
             opt=dict(outline=((216, 40, 128), 1.5))),
    # 053: ring 8 takes the faint glow outlines of the JP brackets; lama_x2
    # left the fewest dark smears on the face at the bottom
    53: dict(method='lama_x2', ring=8, r=(66, 118, 256, 270), ink=[(248, 248, 248), (216, 208, 255), (184, 176, 248),
                                         (160, 152, 240), (136, 120, 232)], dist=70, grow=3,
             glow=[(120, 24, 168), (104, 72, 192), (112, 56, 200), (120, 88, 168)], P=7, R=80, fill=(248, 248, 248), x=160,
             base=[133 + 15 * k for k in range(10)], width=180, align='center',
             opt=dict(outline=((120, 24, 168), 1.5))),
    54: dict(r=(106, 1, 304, 126), ink=(0, 0, 0), dist=150, grow=3, fill=(0, 0, 0),
             x=210, base=[15 + 15 * k for k in range(8)], width=188, align='center',
             opt=dict(outline=((255, 232, 176), 1.5))),
    55: dict(r=(12, 168, 200, 368), ink=(248, 248, 248), dist=100, grow=4, fill=(248, 248, 248), x=101,
             base=[181, 209, 237, 251, 265, 293, 321, 349, 363], width=180, align='center',
             opt=dict(shadow=((8, 32, 40), 2, 2, 1.0))),
    57: dict(r=(0, 306, 304, 378), ink=[(56, 0, 32), (208, 184, 200), (152, 128, 144), (112, 88, 104)], grow=2, dist=60, fill=(56, 0, 32),
             x=13, base=[319, 334, 349, 364], width=284, align='left',
             opt=dict(outline=((208, 184, 200), 1.4))),
}

def plate(card, r, col, alpha, fade=24):
    """Translucent panel over r that fades out over fade px at its right
    end and 3 px at the other edges (hides a busy fill behind text)."""
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    ax = np.clip(np.minimum(np.arange(w) + 1, (w - np.arange(w)) * 3 / fade * 3) / 3, 0, 1)
    ax = np.minimum(ax, np.clip((w - np.arange(w)) / fade, 0, 1))
    ay = np.clip(np.minimum(np.arange(h) + 1, h - np.arange(h)) / 3, 0, 1)
    a = np.outer(ay, ax) * alpha
    rgb = card.rgb()
    comp(rgb, a, col, x0, y0)
    card.put(rgb)

def anime_lines(n, tl):
    pair = tl[f'pair_{n:03d}']
    if f'anime_{n:03d}' in tl:
        lines = tl[f'anime_{n:03d}'].split(' / ')
    else:
        lines = [tl[f'anime_{k}'] for k in (1, 2, 3, 4)]
    return [l.replace('{pair}', pair) for l in lines]

def anime(card, tl, log):
    n = card.n
    c = ANIME[n]
    lines = anime_lines(n, tl)
    assert len(lines) == len(c['base']), (n, lines)
    m = colour_mask(card.rgb(), c['r'], c['ink'], c.get('dist', 140), c['grow'])
    avoid = None
    if 'glow' in c:        # glow-tinted pixels are not used as fill sources
        x0, y0, x1, y1 = c['r']
        avoid = colour_mask(card.rgb(), (x0 - 40, y0 - 40, x1 + 40, y1 + 40), c['glow'], 60, 2)
    erase(card, c['r'], mask=m, avoid=avoid, P=c.get('P', 7), R=c.get('R', 60),
          fill=c.get('method'), ring=c.get('ring', RING))
    if 'plate' in c: plate(card, *c['plate'])
    for sz in range(c.get('max', 15), 9, -1):
        ft = font(ROUND, sz)
        if max(textlen(l, ft) for l in lines) <= c['width']: break
    for l, b in zip(lines, c['base']):
        label(card, [l], ft, c['fill'], c['x'], b, c['align'], **c['opt'])
    log.append(f'{n:03d} anime credit {len(lines)} lines {sz}px')

# 052: one line per translucent bar (rows top-bottom), left / right
# aligned alternately as in the JP. Sources for the fill stay in the bar.
BARS52 = [(229, 248, 'left'), (259, 278, 'right'), (289, 308, 'left'), (319, 338, 'right')]

def anime52(card, tl, log):
    lines = anime_lines(52, tl)
    for k, (t, b, _) in enumerate(BARS52):
        r = (0, t, 304, b + 1)
        m = colour_mask(card.rgb(), r, (255, 255, 0), 160, 2)
        # bar 1: the art under its right end is smooth; a texture copy left
        # blocky traces there
        erase(card, r, 'smooth' if k == 0 else 'exemplar', mask=m,
              avoid=~rect_mask(m.shape, r), P=5, R=80)
    for sz in range(15, 9, -1):
        ft = font(ROUND, sz)
        if max(textlen(l, ft) for l in lines) <= 250: break
    for l, (t, b, al) in zip(lines, BARS52):
        label(card, [l], ft, (255, 255, 0), 5 if al == 'left' else 299, t + 15, al,
              shadow=((96, 64, 32), 1, 1, 0.8))
    log.append(f'052 anime credit 4 bars {sz}px')

# ---------------------------------------------------------------- notes

RB = 'MPLUSRounded1c-Bold.ttf'
RX = 'MPLUSRounded1c-ExtraBold.ttf'

def layout(parts, ft, mft, width, tl, mpad=3, hang=True):
    """Lines of (x, text) and mark positions for paragraphs (mark, key):
    the mark pill starts the paragraph and the text hangs after it."""
    out = []                      # (kind, x, line index, text)
    li = 0
    for mark, key in parts:
        text = tl[key] if key in tl else key
        ind = 0
        if mark:
            mw = mft.getlength(tl[mark]) + 2 * mpad
            out.append(('mark', 0, li, tl[mark], mw))
            ind = round(mw + 4)
        if hang or not ind:
            lines = wrap(text, ft, width - ind)
            xs = [ind] * len(lines)
        else:                     # only the first line follows the mark
            first = wrap(text, ft, width - ind)[0]
            rest = text[len(first):].strip()
            lines = [first] + (wrap(rest, ft, width) if rest else [])
            xs = [ind] + [0] * (len(lines) - 1)
        for x, ln in zip(xs, lines):
            out.append(('text', x, li, ln, 0)); li += 1
    return out, li

def block(card, tl, log, r, parts, font_path=RB, smax=12, smin=8, fill=(255, 255, 255),
          outline=None, shadow=None, mark=None, align='left', valign='top',
          lead=1.25, gradient=None, mark_font=None, tag='', plate_bg=None, hang=True):
    """Fit paragraphs into box r (largest size that fits) and draw them.
    mark = (pill colour, label colour[, pill outline colour])."""
    x0, y0, x1, y1 = r
    for sz in range(smax, smin - 1, -1):
        ft = font(font_path, sz)
        mft = font(mark_font or RX, max(7, sz - 2))
        lp = round(sz * lead)
        items, n = layout(parts, ft, mft, x1 - x0, tl, hang=hang)
        asc, desc = ft.getmetrics()
        h = lp * (n - 1) + asc + desc
        widest = max(it[1] + textlen(it[3], ft) for it in items if it[0] == 'text')
        if h <= y1 - y0 and widest <= x1 - x0: break
    else:
        raise SystemExit(f'{card.n:03d} {tag}: text does not fit {r}')
    top = y0 + {'top': 0, 'center': (y1 - y0 - h) // 2, 'bottom': y1 - y0 - h}[valign]
    bw = max(it[1] + (textlen(it[3], ft) if it[0] == 'text' else it[4]) for it in items)
    if plate_bg:                  # translucent panel hugging the text block
        col, al, mg = plate_bg
        bx = {'left': x0, 'center': x0 + (x1 - x0 - bw) / 2, 'right': x1 - bw}[align]
        plate(card, (int(bx) - mg, top - mg, int(np.ceil(bx + bw)) + mg + 1, top + h + mg), col, al, 6)
    # whole block aligned in the box, lines left-aligned inside it, except
    # for single-paragraph centred blocks where every line is centred
    if align == 'center' and all(not m for m, _ in parts):
        for it in items:
            if it[0] != 'text': continue
            label(card, [it[3]], ft, fill, (x0 + x1) / 2, top + asc + it[2] * lp, 'center',
                  outline=outline, shadow=shadow, gradient=gradient)
    else:
        bx = {'left': x0, 'center': x0 + (x1 - x0 - bw) / 2, 'right': x1 - bw}[align]
        for it in items:
            y = top + asc + it[2] * lp
            if it[0] == 'mark':
                pill(card, round(bx), y, it[3], mft, it[4], asc, mark, outline)
            else:
                label(card, [it[3]], ft, fill, round(bx + it[1]), y, 'left',
                      outline=outline, shadow=shadow, gradient=gradient)
    log.append(f'{card.n:03d} {tag} {n} lines {sz}px in {r}')

def pill(card, x, base, text, mft, w, asc, mark, outline):
    """Rounded tag with text (the circled デ / 企 marks of the JP)."""
    bg, fg = mark[0], mark[1]
    ring = mark[2] if len(mark) > 2 else (outline[0] if outline else None)
    h = asc + 1
    y = base - asc + 1
    W, H = int(w) + 4, h + 4
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).rounded_rectangle((2, 2, 2 + w - 1, 2 + h - 1), radius=h // 2, fill=255)
    a = np.asarray(im, float) / 255
    rgb = card.rgb()
    if ring is not None: comp(rgb, grow(a, 1.0), ring, x - 2, y - 2)
    comp(rgb, a, bg, x - 2, y - 2)
    card.put(rgb)
    ta = mft.getbbox(text)
    label(card, [text], mft, fg, x + w / 2, y + (h + (ta[3] - ta[1])) // 2 - 1 + 0, 'center')

def erase_step(card, r, ink=None, dist=80, grow=2, oink=None, odist=60, oring=3,
               keep=None, inside=None, how='exemplar', P=7, R=30, src_clean=True, pred=None,
               fill=None):
    """Erase text in r: pixels near the fill colours ink (grown by grow px)
    plus pixels near the outline colours oink within oring px of them.
    Pixels near ink/oink colours are not used as fill sources; inside
    limits the sources to a rect (a panel)."""
    rgb = card.rgb()
    if pred is not None:                 # colour predicate instead of a list
        core = np.zeros(rgb.shape[:2], bool)
        x0, y0, x1, y1 = r
        core[y0:y1, x0:x1] = pred(rgb[y0:y1, x0:x1])
    else:
        core = colour_mask(rgb, r, ink, dist, 0)
    m = (ndimage.binary_dilation(core, iterations=grow) if grow > 0 else core) & rect_mask(core.shape, r)
    if oink is not None:
        near = ndimage.binary_dilation(core, iterations=oring)
        m |= near & colour_mask(rgb, r, oink, odist, 0)
        m = ndimage.binary_dilation(m, iterations=1) & rect_mask(core.shape, r)
    if keep is not None: m &= ~rect_mask(m.shape, keep)
    avoid = np.zeros_like(m)
    if inside is not None: avoid |= ~rect_mask(m.shape, inside)
    if src_clean:
        x0, y0, x1, y1 = r
        big = (x0 - R - P, y0 - R - P, x1 + R + P, y1 + R + P)
        if pred is not None:
            pm = np.zeros_like(m); bx0, by0 = max(0, big[0]), max(0, big[1])
            pm[by0:big[3], bx0:big[2]] = pred(rgb[by0:big[3], bx0:big[2]])
            avoid |= ndimage.binary_dilation(pm)
        else:
            avoid |= colour_mask(rgb, big, ink, dist, 1)
        if oink is not None: avoid |= colour_mask(rgb, big, oink, odist, 0)
    erase(card, r, how, mask=m, avoid=avoid, P=P, R=R, fill=fill)

def run_steps(card, tl, log, steps):
    """Run the steps; pixels changed by a draw step (card.drawn) are kept
    out of later erase masks, rims and fill sources."""
    card.drawn = np.zeros(card.idx.shape, bool)
    for kind, kw in steps:
        before = card.idx.copy()
        if kind == 'erase':
            erase_step(card, **kw)
        elif kind == 'block':
            block(card, tl, log, **kw)
        elif kind == 'rot':
            rot_label(card, tl, log, **kw)
        elif kind == 'plate':
            plate(card, **kw)
        elif kind == 'fn':
            kw(card, tl, log)
        if kind != 'erase': card.drawn |= card.idx != before
    card.drawn = None

def rot_label(card, tl, log, r, key, font_path=RX, smax=40, fill=(255, 255, 255),
              outline=None, shadow=None, gradient=None, tag=''):
    """Text turned clockwise (reads top to bottom) centred in column r,
    the largest size whose length and height fit."""
    x0, y0, x1, y1 = r
    t = tl[key]
    for sz in range(smax, 6, -1):
        ft = font(font_path, sz)
        bb = ft.getbbox(t)
        if bb[2] - bb[0] <= y1 - y0 - 4 and bb[3] - bb[1] <= x1 - x0 - 4: break
    pad = 6
    W, H = bb[2] - bb[0] + 2 * pad, bb[3] + 2 * pad
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad), t, font=ft, fill=255)
    a = np.asarray(im, float) / 255
    ys = np.nonzero(a.max(1) > 0.1)[0]
    a = np.rot90(a, -1)              # clockwise: rows -> columns
    h, w = a.shape
    # ink extent across the column (from the unrotated rows)
    c0, c1 = H - 1 - ys.max(), H - 1 - ys.min()
    ox = round((x0 + x1) / 2 - (c0 + c1) / 2)
    oy = round((y0 + y1) / 2 - h / 2)
    rgb = card.rgb()
    if shadow:
        c, dx, dy, al = shadow
        sa = grow(a, outline[1]) if outline else a
        comp(rgb, shift(sa, dx, dy) * al, c, ox, oy)
    if outline: comp(rgb, grow(a, outline[1]), outline[0], ox, oy)
    if gradient:
        g0, g1 = np.asarray(gradient[0], float), np.asarray(gradient[1], float)
        t_ = np.clip((np.arange(w) - c0) / max(1, c1 - c0), 0, 1)[None, :, None]
        colimg = np.broadcast_to(g1 * (1 - t_) + g0 * t_, (h, w, 3))
        reg = rgb[oy:oy + h, ox:ox + w]
        aa = a[:reg.shape[0], :reg.shape[1]][..., None]
        reg[:] = reg * (1 - aa) + colimg[:reg.shape[0], :reg.shape[1]] * aa
    else:
        comp(rgb, a, fill, ox, oy)
    card.put(rgb)
    log.append(f'{card.n:03d} {tag} rotated {sz}px in {r}')

# Design-note cards: per card a list of steps (erase, plate, block, fn).
D, P = 'mark_d', 'mark_p'
DARK = (24, 24, 24)
BLUE = [(0, 88, 248), (0, 96, 248), (0, 112, 248), (0, 136, 248), (0, 120, 248),
        (16, 120, 248), (32, 128, 248), (48, 128, 248)]
CREAM = [(240, 240, 160), (232, 232, 152), (248, 248, 200), (224, 224, 144)]
NAVY = [(32, 32, 80), (40, 40, 96), (24, 24, 64), (48, 48, 104)]
OLIVE = [(104, 112, 16), (88, 96, 8), (96, 96, 16), (112, 120, 16), (112, 112, 16),
         (128, 136, 24), (144, 152, 32), (160, 168, 40), (176, 184, 48)]
def ink004(a):
    """004 text: red/orange ink, or yellow ink within 2 px of the dark
    text outline (the plate over the yellow art is yellow too)."""
    red = I.near(a, [(248, 96, 96), (248, 120, 128), (248, 128, 104), (248, 152, 64)], 90)
    yel = I.near(a, [(224, 216, 80), (208, 192, 80)], 90)
    ol = I.near(a, [(24, 24, 24), (24, 24, 16)], 12)
    return red | (yel & ndimage.binary_dilation(ol, iterations=2))

NOTES = {
    0: [('erase', dict(r=(212, 28, 290, 235), pred=lambda a: a[..., 2] - a[..., 1] > 25, grow=1,
                       oink=[(8, 8, 16), (16, 16, 24), (24, 24, 32), (32, 32, 40), (40, 40, 48)],
                       odist=50, oring=7, P=5, R=40)),
        ('rot', dict(tag='name', r=(222, 34, 278, 228), key='c000_name', font_path='georgia.ttf',
                     smax=44, fill=(184, 152, 232), shadow=((8, 8, 16), 2, 2, 0.9),
                     gradient=((224, 208, 248), (152, 112, 208)))),
        ('erase', dict(r=(18, 143, 94, 367), ink=[(192, 160, 96), (176, 144, 88), (200, 168, 104)],
                       dist=70, grow=1, inside=(18, 143, 94, 367), P=5, R=60)),
        ('block', dict(tag='note', r=(22, 150, 91, 362), parts=[(D, 'c000_d'), (P, 'c000_p')],
                       fill=(192, 160, 96), mark=((192, 160, 96), (40, 40, 96)), smax=10, smin=7)),
    ],
    # 001: plain lama left a light smudge in the pink diamond
    1: [('erase', dict(r=(175, 174, 284, 192), ink=[(0, 64, 208)], dist=90, grow=1, P=5, R=30,
                       fill='lama_ctx')),
        ('block', dict(tag='label', r=(172, 175, 288, 191), parts=[(None, 'shelved')], font_path=RX,
                       smax=12, fill=(0, 64, 208), align='center')),
        ('erase', dict(r=(172, 191, 288, 227), ink=[(8, 0, 240)], dist=110, grow=1, how='smooth',
                       fill='lama_ctx')),
        ('block', dict(tag='name', r=(178, 195, 282, 224), parts=[(None, 'c001_name')], font_path=RX,
                       smax=15, fill=(8, 0, 240), align='center', valign='center')),
        ('erase', dict(r=(174, 213, 300, 368), ink=[(104, 200, 248), (200, 72, 160), (208, 72, 168),
                                                    (248, 248, 248)],
                       dist=80, grow=2, oink=[(8, 16, 56), (8, 24, 64), (8, 48, 88), (8, 40, 72),
                                              (24, 40, 96), (40, 64, 120)],
                       odist=60, oring=6, P=5, R=90)),
        # the navy plate covers the whole JP panel (a lighter box was left
        # under the text when the plate hugged it)
        ('plate', dict(r=(176, 222, 296, 366), col=(8, 16, 56), alpha=0.75, fade=6)),
        ('block', dict(tag='note', r=(180, 226, 298, 362), parts=[(D, 'c001_d'), (P, 'c001_p')],
                       valign='center',
                       fill=(104, 200, 248), outline=((8, 16, 56), 1.5), smax=11,
                       mark=((32, 72, 200), (255, 255, 255), (255, 255, 255)))),
    ],
    2: [('erase', dict(r=(8, 16, 70, 360), ink=OLIVE, dist=40, grow=1, oink=[(48, 48, 0), (56, 64, 8)],
                       odist=30, how='smooth')),
        ('rot', dict(tag='name', r=(12, 20, 66, 356), key='c002_name', fill=(200, 208, 56),
                     outline=((48, 48, 0), 1.5), gradient=((232, 240, 96), (112, 120, 16)))),
        ('erase', dict(r=(236, 16, 294, 360), ink=OLIVE, dist=40, grow=1, oink=[(48, 48, 0), (56, 64, 8)],
                       odist=30, how='smooth')),
        ('rot', dict(tag='label', r=(240, 20, 292, 356), key='shelved', fill=(200, 208, 56),
                     outline=((48, 48, 0), 1.5), gradient=((232, 240, 96), (112, 120, 16)))),
        ('erase', dict(r=(56, 272, 256, 365), ink=[(224, 168, 168), (232, 184, 184), (216, 160, 160)],
                       dist=60, grow=1, oink=[(48, 16, 24), (64, 16, 24), (32, 8, 8), (80, 24, 32)],
                       odist=35, P=5, R=40)),
        ('block', dict(tag='note', r=(60, 276, 248, 363), parts=[(D, 'c002_d'), (P, 'c002_p')],
                       fill=(232, 184, 184), shadow=((48, 8, 16), 1, 1, 1.0),
                       mark=((232, 184, 184), (104, 32, 48)))),
    ],
    3: [('erase', dict(r=(240, 10, 296, 156), ink=[(232, 232, 232), (240, 240, 240), (72, 16, 24),
                                                   (24, 0, 0)],
                       dist=60, grow=1, oink=[(72, 16, 24), (48, 8, 8), (96, 32, 24)], odist=40,
                       how='smooth')),
        ('rot', dict(tag='label', r=(266, 10, 288, 126), key='shelved', font_path=RX,
                     smax=12, fill=(72, 16, 24), outline=((248, 200, 160), 1.0))),
        ('rot', dict(tag='name', r=(232, 14, 268, 182), key='c003_name', font_path='georgiab.ttf',
                     smax=24, fill=(248, 248, 248), outline=((72, 16, 24), 1.5),
                     shadow=((40, 8, 8), 2, 2, 0.8))),
        ('erase', dict(r=(196, 182, 298, 366), ink=[(24, 0, 0), (32, 8, 8)], dist=60, grow=1,
                       oink=[(136, 192, 120), (128, 192, 112), (152, 200, 136), (112, 176, 104), (96, 160, 88), (168, 208, 152)], odist=50, oring=4, how='smooth')),
        ('erase', dict(r=(38, 182, 82, 368), ink=[(24, 0, 0), (32, 8, 8)], dist=60, grow=1,
                       oink=[(136, 192, 120), (128, 192, 112), (152, 200, 136), (112, 176, 104), (96, 160, 88), (168, 208, 152)], odist=50, oring=4, how='smooth')),
        ('block', dict(tag='note1', r=(231, 188, 301, 364), parts=[(D, 'c003_d')], fill=(40, 8, 8), hang=False, smax=11,
                       outline=((176, 216, 160), 1.0), mark=((40, 8, 8), (248, 200, 140)))),
        ('block', dict(tag='note2', r=(8, 188, 86, 362), parts=[(P, 'c003_p')], fill=(40, 8, 8), hang=False, smax=11,
                       outline=((176, 216, 160), 1.0), mark=((40, 8, 8), (248, 200, 140)))),
    ],
    4: [('erase', dict(r=(5, 262, 198, 366), pred=ink004, grow=1, oink=[(24, 24, 24), (24, 24, 16), (40, 32, 32)], odist=30,
                       inside=(5, 262, 198, 366), P=5, R=60)),
        ('block', dict(tag='label', r=(10, 265, 192, 281), parts=[(None, 'shelved')], font_path=RX,
                       smax=11, fill=(248, 96, 96), outline=(DARK, 1.0), align='center')),
        ('block', dict(tag='name', r=(10, 281, 192, 305), parts=[(None, 'c004_name')], font_path=RX,
                       smax=19, fill=(248, 120, 128), outline=(DARK, 1.5), align='center',
                       gradient=((255, 200, 160), (240, 88, 80)))),
        ('block', dict(tag='note', r=(10, 308, 194, 365), parts=[(D, 'c004_d'), (P, 'c004_p')],
                       fill=(224, 216, 80), outline=(DARK, 1.0), mark=((248, 152, 64), DARK))),
    ],
    5: [('erase', dict(r=(112, 245, 302, 367), ink=[(224, 248, 192), (176, 184, 240), (0, 72, 40),
                                                    (56, 112, 80), (80, 128, 96), (24, 80, 56),
                                                    (88, 136, 96), (32, 96, 64), (72, 120, 88)],
                       dist=60, grow=2, oink=[(32, 0, 72), (32, 16, 72), (48, 32, 88)], odist=40,
                       oring=4, how='smooth')),
        ('block', dict(tag='label', r=(120, 248, 296, 264), parts=[(None, 'shelved')], font_path=RX,
                       smax=11, fill=(224, 248, 192), outline=(DARK, 1.0), align='center')),
        ('block', dict(tag='name', r=(120, 264, 296, 295), parts=[(None, 'c005_name')], font_path=RX,
                       smax=19, fill=(120, 200, 140), outline=((0, 56, 32), 1.5), align='center',
                       gradient=((200, 248, 200), (56, 160, 96)))),
        ('block', dict(tag='note', r=(118, 297, 300, 366), parts=[(D, 'c005_d'), (P, 'c005_p')],
                       fill=(176, 184, 240), outline=((32, 0, 72), 1.0),
                       mark=((176, 184, 240), (32, 0, 72)))),
    ],
    16: [('erase', dict(r=(28, 35, 135, 225), grow=2, P=5, R=40,
                        pred=lambda a: (a[..., 1] > 214) | (a[..., 0] < 150))),
         ('rot', dict(tag='name', r=(26, 34, 134, 226), key='c016_name', font_path='georgia.ttf',
                      smax=80, fill=(0, 176, 192), outline=((240, 232, 248), 1.5),
                      gradient=((112, 232, 232), (0, 152, 176)))),
         ('erase', dict(r=(36, 222, 154, 362), ink=[(112, 0, 80), (120, 16, 88), (128, 32, 96),
                                                    (136, 48, 104)], dist=40, grow=1, P=5, R=40)),
         ('block', dict(tag='note', r=(16, 228, 150, 352), parts=[(None, 'c016_note')],
                        fill=(112, 0, 80), smax=11)),
    ],
    17: [('erase', dict(r=(100, 262, 294, 365), grow=2, how='smooth',
                        pred=lambda a: (a.max(-1) < 70) | ((a[..., 0] > 150) & (a[..., 1] < 160)
                                                           & (a[..., 2] > 100)))),
         ('block', dict(tag='title', r=(104, 264, 292, 280), parts=[(None, 'c017_title')], font_path=RX,
                        smax=14, fill=(240, 88, 168), outline=((40, 0, 48), 1.5), align='center')),
         ('block', dict(tag='note', r=(106, 283, 292, 358), parts=[(None, 'c017_note')],
                        fill=(232, 128, 208), outline=((16, 0, 24), 1.0), smax=11)),
    ],
    18: [('erase', dict(r=(8, 262, 298, 288), ink=[(232, 112, 16), (224, 104, 16)], dist=70, grow=2,
                        oink=[(144, 64, 64), (120, 48, 80), (96, 40, 96), (176, 88, 56), (56, 16, 120),
                              (72, 48, 104), (64, 32, 104), (80, 24, 104), (48, 8, 88)], odist=30,
                        oring=5, P=5, R=60)),
         ('block', dict(tag='label', r=(15, 265, 292, 286), parts=[(None, 'c018_label')], font_path=RX,
                        smax=15, fill=(232, 112, 16), outline=((56, 16, 120), 2.0), align='center',
                        valign='center')),
         ('erase', dict(r=(20, 292, 286, 348), ink=[(8, 184, 0), (8, 176, 0), (24, 152, 40),
                                                    (32, 136, 56)],
                        dist=80, grow=2, oink=[(24, 24, 24)], odist=30, inside=(0, 290, 304, 350),
                        P=5, R=80)),
         ('plate', dict(r=(18, 292, 288, 349), col=(0, 24, 72), alpha=0.6, fade=10)),
         ('block', dict(tag='note', r=(24, 294, 284, 347), parts=[(D, 'c018_d'), (P, 'c018_p')],
                        fill=(8, 184, 0), outline=((0, 16, 40), 1.0), mark=((8, 184, 0), (0, 24, 72)))),
    ],
    23: [('erase', dict(r=(20, 40, 140, 275), ink=[(248, 248, 248), (216, 232, 248)], dist=60, grow=1,
                        oink=BLUE, odist=40, how='smooth')),
         ('rot', dict(tag='name', r=(112, 34, 138, 116), key='c023_name', font_path=RX,
                      smax=16, fill=(248, 248, 248), outline=((0, 88, 248), 1.5))),
         ('block', dict(tag='note', r=(14, 44, 110, 272), parts=[(D, 'c023_d'), (P, 'c023_p')],
                        fill=(248, 248, 248), smax=11, outline=((0, 112, 248), 1.5),
                        mark=((248, 248, 248), (0, 112, 248)))),
    ],
    26: [('erase', dict(r=(16, 236, 104, 258), ink=[(248, 248, 248)], dist=60, grow=1,
                        oink=[(0, 88, 32), (0, 80, 24)], odist=30, how='smooth')),
         ('block', dict(tag='name', r=(18, 238, 160, 256), parts=[(None, 'c026_name')], font_path=RX,
                        smax=12, fill=(248, 248, 248), outline=((0, 88, 32), 1.0))),
         ('erase', dict(r=(18, 262, 264, 368), grow=2, how='smooth', oink=[(24, 24, 40), (56, 56, 80)],
                        odist=30, oring=3,
                        pred=lambda a: (a.min(-1) > 200) | ((a[..., 1] > a[..., 0] + 30)
                                                            & (a[..., 1] > a[..., 2] + 10)
                                                            & (a[..., 1] < 150)))),
         ('block', dict(tag='note', r=(22, 265, 262, 356), parts=[(D, 'c026_d'), (P, 'c026_p')],
                        fill=(248, 248, 248), outline=((24, 24, 40), 1.0),
                        mark=((248, 248, 248), (24, 24, 40)))),
    ],
    28: [('erase', dict(r=(12, 312, 292, 348), ink=[(0, 0, 0), (40, 40, 40), (16, 16, 16), (160, 160, 160),
                                                    (144, 144, 144)], dist=60, grow=1, how='smooth')),
         ('block', dict(tag='title', r=(15, 315, 290, 346), parts=[(None, 'c028_title')], font_path=RX,
                        smax=24, fill=(0, 0, 0), align='center', valign='center')),
    ],
    58: [('erase', dict(r=(12, 248, 294, 368), ink=[(248, 248, 248)], dist=70, grow=1,
                        oink=[(16, 16, 16), (24, 24, 24), (32, 32, 40)], odist=35, how='smooth')),
         ('block', dict(tag='note', r=(18, 252, 290, 366), parts=[(None, 'c058_title'), (None, 'c058_note')],
                        fill=(248, 248, 248), outline=(DARK, 1.0), smax=13)),
    ],
    59: [('erase', dict(r=(126, 248, 303, 368), grow=1, oink=[(64, 0, 24), (80, 0, 32), (0, 72, 176),
                                                          (0, 112, 208), (0, 96, 200)], odist=30, oring=4,
                        P=5, R=40, pred=lambda a: (a.min(-1) > 150)
                                                  | ((a[..., 0] > 90) & (a[..., 2] > 180)))),
         ('block', dict(tag='note', r=(130, 252, 300, 366), parts=[(None, 'c059_title'), (None, 'c059_note')],
                        fill=(248, 248, 248), outline=((64, 0, 24), 1.0), smax=12)),
    ],
    6: [('erase', dict(r=(262, 12, 292, 148), ink=CREAM, dist=50, grow=1, oink=NAVY, odist=40,
                       oring=2, P=5, R=30)),
        ('rot', dict(tag='label', r=(263, 15, 291, 146), key='shelved', fill=(240, 240, 168),
                     outline=((56, 48, 96), 1.0), shadow=((32, 32, 80), 1, 1, 1.0))),
        ('erase', dict(r=(13, 177, 42, 288), ink=CREAM, dist=50, grow=1, oink=NAVY, odist=40,
                       oring=2, P=5, R=30)),
        ('rot', dict(tag='name', r=(13, 180, 41, 286), key='c006_name', fill=(240, 240, 168),
                     outline=((56, 48, 96), 1.0), shadow=((32, 32, 80), 1, 1, 1.0))),
        ('erase', dict(r=(20, 288, 210, 364), ink=[(128, 248, 128)], dist=100, grow=1,
                       oink=[(24, 24, 24), (32, 32, 32)], odist=40, P=5, R=40)),
        # the armband 実習 stays art (user); its translation goes in the note
        ('block', dict(tag='note', r=(24, 291, 202, 363), parts=[(D, 'c006_d'), (P, 'c006_p'),
                                                                 (None, 'c006_band')],
                       fill=(128, 248, 128), outline=(DARK, 1.0), mark=((128, 248, 128), DARK))),
    ],
    7: [('erase', dict(r=(105, 25, 292, 62), ink=[(208, 136, 56), (216, 152, 72), (192, 120, 48)],
                       dist=60, grow=1, oink=[(88, 8, 8), (64, 8, 8)], odist=40, oring=4, how='smooth')),
        ('block', dict(tag='title', r=(108, 28, 288, 58), parts=[(None, 'c007_title')],
                       font_path='georgiab.ttf', smax=20, fill=(216, 144, 64),
                       shadow=((88, 8, 8), 2, 2, 1.0), align='right', valign='center')),
        ('erase', dict(r=(10, 312, 296, 367), ink=[(32, 224, 56)], dist=100, grow=1,
                       oink=[(8, 16, 56), (8, 40, 56)], odist=50,
                       inside=(0, 310, 304, 368), P=5, R=80, src_clean=True)),
        ('block', dict(tag='note', r=(15, 315, 292, 366), parts=[(None, 'c007_note')],
                       fill=(32, 224, 56), outline=((8, 16, 56), 1.0), smax=13)),
    ],
    8: [('erase', dict(r=(105, 25, 292, 62), ink=[(80, 200, 144), (96, 216, 160), (64, 184, 128)],
                       dist=60, grow=1, oink=[(32, 48, 0), (40, 56, 0)], odist=40, oring=4, how='smooth')),
        ('block', dict(tag='title', r=(108, 28, 288, 58), parts=[(None, 'c008_title')],
                       font_path='georgiab.ttf', smax=20, fill=(80, 200, 144),
                       shadow=((32, 48, 0), 2, 2, 1.0), align='right', valign='center')),
        ('erase', dict(r=(10, 298, 296, 367), ink=[(128, 208, 216)], dist=100, grow=1,
                       oink=[(8, 16, 16), (16, 8, 8)], odist=40, P=5, R=60)),
        ('block', dict(tag='note', r=(15, 300, 292, 366), parts=[(None, 'c008_note')],
                       fill=(128, 208, 216), outline=((8, 16, 16), 1.0), smax=13)),
    ],
}

def steps(n):
    out = []
    if n in LETTERS: out.append(letter)
    if n in ILL: out.append(illustrator)
    if n == 13: out.append(illustrator13)
    if n in ANIME: out.append(anime)
    if n == 52: out.append(anime52)
    if n in NOTES: out.append(lambda c, t, l: run_steps(c, t, l, NOTES[c.n]))
    if n in FOOT: out.append(footer)        # last: later fills must not copy it
    return out

def paint(card, tl, log):
    for fn in steps(card.n):
        fn(card, tl, log)

# ---------------------------------------------------------------- archive

def build(only=None):
    orig = open(SRC, 'rb').read()
    toc = arc.toc(orig)
    tl = load_tl()
    blobs, cards, log = [], {}, []
    for i, (o, s) in enumerate(toc):
        blob = orig[o:o + s]
        if i < 60 and (only is None or i in only) and steps(i):
            data = lz.decompress(blob)[0]
            c = Card(i, data)
            paint(c, tl, log)
            cards[i] = c
            new = c.tim()
            assert len(new) == len(data)
            if new != data and only is None:
                blob = lz.compress(new, name=NAME)
                assert lz.decompress(blob)[0] == new
        blobs.append(blob)
    out = bytearray(0x800)
    for i, b in enumerate(blobs):
        o = len(out)
        struct.pack_into('<II', out, 8 * i, o, len(b))
        out += b + bytes(-len(b) % 0x800)
    if len(out) > len(orig):
        raise SystemExit(f'{NAME}: {len(out)} bytes > JP length {len(orig)}')
    out += bytes(len(orig) - len(out))
    return bytes(out), cards, log

def main():
    args = sys.argv[1:]
    if args and args[0] == '--png':
        only = {int(a) for a in args[1:]} or None
        _, cards, log = build(only if only else set(range(60)))
        os.makedirs(REVIEW, exist_ok=True)
        for n, c in cards.items():
            im = Image.new('RGB', (2 * c.w + 4, c.h))
            im.paste(c.image(), (0, 0)); im.paste(c.image(c.orig), (c.w + 4, 0))
            im.save(f'{REVIEW}{n:03d}.png')
        print('\n'.join(log)); print('previews', len(cards))
        return
    data, cards, log = build()
    print('\n'.join(log))
    os.makedirs('private/build', exist_ok=True)
    tmp = OUT + '.tmp'
    open(tmp, 'wb').write(data); os.replace(tmp, OUT)
    print(f'wrote {OUT} ({len(cards)} cards repainted)')

if __name__ == '__main__':
    main()
