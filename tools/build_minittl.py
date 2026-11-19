# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Repaint the mini-game title cards (DAT2/MINITTL1-9, A-C) in English.

Reads tl/MINITTL.txt. RE: private/re/minittl/findings.org (card layout),
private/re/practice/findings.org (which overlay loads which card).

Each MINITTLn.BIZ holds the title card as an 8bpp 512x480 TIM at 0x18800.
Normal cards (1-4, 9, A) carry the game title and a tagline on a slanted
plate. The practice cards (5-8, B, C) carry the title, a red 練習用 stamp
and the coach's line on a rhombus. The JP title, stamp and plate text are
painted out and redrawn in English; Doki-Waku-Dance (A, C) is already
Latin and stays. Text removal: tools/inpaint.py (text_mask rim around
the masks below, fill('best'), to_clut with the real CLUT).

The rules pages (TIMs 0x55000 and 0x91800) are painted by
tools/minittl_rules.py from tl/RULES.txt and data/minittl_rules.tsv.

The start menu label atlas (TIM 0 at offset 0, the same in every
MINITTL file) is painted by tools/minittl_atlas.py from tl/MINIHUD.txt
and tl/FEST.txt; this tool calls it for all 12 files and also writes
MINITTL0.BIZ (atlas plus the Festival menu background, whose 五輪祭 logo
tools/fest_menu.py redraws) and DAT1/MINISEL.BIN (Festival menu sprite
records for the moved or widened f_* labels, fest_menu.minisel()).

Build order: after build_beta.py, before hook.py. No other builder
writes these files. The TIMs come from private/unpacked (originals), so
reruns are clean.
Writes private/build/MINITTLn.BIZ (LZ) and .BIZ.lz (plain) for the 12
cards and MINITTL0, and private/build/MINISEL.BIN. Options: --dry (no writes), --png (JP/EN previews and a contact
sheet in private/re/minittl/en/), card ids (e.g. 1 5 B) to limit the run
(previews only; a real build paints all 12).
"""
import sys, os, struct, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz
import inpaint as IP
import minittl_atlas
import minittl_rules
import fest_menu

F_ROUND = 'MPLUSRounded1c-Black.ttf'
F_ROUNDX = 'MPLUSRounded1c-ExtraBold.ttf'
F_BLACKI = 'seguibli.ttf'
F_IMPACT = 'impact.ttf'
TITLE = 0x18800
PW, PH = 512, 480
PREV = 'private/re/minittl/en'
CARDS = '123456789ABC'
PRACTICE = '5678BC'
RNG = np.random.default_rng(1)
EXEMPLAR = os.environ.get('MINITTL_EXEMPLAR') == '1'   # old exemplar fill on 3, 7, C

# ---------------------------------------------------------------- data

def load_tl(path='tl/MINITTL.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

class Page:
    """One 8bpp 512x480 picture with its 256-colour CLUT."""
    def __init__(self, src, off):
        self.off = off
        self.clut = list(struct.unpack_from('<256H', src, off + 0x14))
        self.pal = np.array([rgb15(v) for v in self.clut])
        self.idx = np.frombuffer(bytes(src[off + 0x220:off + 0x220 + PW * PH]),
                                 np.uint8).reshape(PH, PW).copy()
        self.img = self.pal[self.idx]
        self.orig = self.img.copy()
        self.fillm = np.zeros((PH, PW), bool)   # pixels filled by IP.fill
        self.erased = None                      # img after the erase step

    def store(self, d):
        """Nearest colour for changed pixels; Floyd-Steinberg (IP.to_clut,
        real CLUT) for filled pixels no EN text was drawn over."""
        bad = [i for i, v in enumerate(self.clut) if v == 0]
        img = np.clip(self.img, 0, 255)
        new = quantize(img, self.pal, self.idx, bad)
        w = self.fillm.copy()
        if self.erased is not None:
            w &= np.abs(self.img - self.erased).sum(-1) < 1e-6
        if w.any():
            dith = IP.to_clut(img, self.idx, self.clut, w, pal=self.pal)
            new[w] = dith[w]
        d[self.off + 0x220:self.off + 0x220 + PW * PH] = new.tobytes()
        self.idx = new
        return self.pal[new]

def quantize(img, pal, idx, bad=()):
    """Nearest CLUT colour for every pixel that changed (weighted RGB).
    Entries in bad (value 0x0000, transparent on the GPU) are never chosen."""
    out = idx.copy()
    ch = np.abs(img - pal[idx]).sum(-1) > 3
    flat = img[ch]; wt = np.array([0.30, 0.59, 0.11]) ** 0.5
    pw = pal * wt; sel_all = np.where(ch.ravel())[0]
    for k in range(0, len(flat), 4096):
        q = flat[k:k + 4096] * wt
        dd = ((q[:, None, :] - pw[None]) ** 2).sum(-1)
        if len(bad): dd[:, list(bad)] = 1e12
        out.ravel()[sel_all[k:k + 4096]] = dd.argmin(1)
    return out

# ---------------------------------------------------------------- fill

def inpaint(img, mk, domain=None, iters=300):
    """Fill mk by repeated averaging of the 4 neighbours inside domain.
    Large holes start from a half-resolution solution."""
    out = img.copy()
    if not mk.any(): return out
    if domain is None: domain = np.ones(mk.shape, bool)
    ys, xs = np.where(mk)
    y0, y1 = max(0, ys.min() - 4), min(mk.shape[0], ys.max() + 5)
    x0, x1 = max(0, xs.min() - 4), min(mk.shape[1], xs.max() + 5)
    sub = out[y0:y1, x0:x1].copy(); m = mk[y0:y1, x0:x1]; dm = domain[y0:y1, x0:x1] | m
    known = dm & ~m
    if not known.any(): return out
    _solve(sub, m, dm, known, iters)
    out[y0:y1, x0:x1] = sub
    return out

def _solve(sub, m, dm, known, iters):
    h, w = m.shape
    if min(h, w) > 16 and ndimage.distance_transform_edt(m).max() > 6:
        hs, ws = (h + 1) // 2, (w + 1) // 2
        def down(a, f):
            p = np.zeros((hs * 2, ws * 2) + a.shape[2:], a.dtype); p[:h, :w] = a
            return f(p.reshape(hs, 2, ws, 2, *a.shape[2:]))
        kc = down(known.astype(float), lambda p: p.sum((1, 3)))
        sc = down(sub * known[..., None], lambda p: p.sum((1, 3)))
        dc = down(dm, lambda p: p.any((1, 3)))
        csub = sc / np.maximum(kc, 1)[..., None]
        cknown = kc > 0
        cm = dc & ~cknown
        if cm.any() and cknown.any():
            _solve(csub, cm, dc, cknown, iters)
        up = np.repeat(np.repeat(csub, 2, 0), 2, 1)[:h, :w]
        sub[m] = up[m]
    else:
        for c in range(3):
            sub[..., c][m] = sub[..., c][known].mean()
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    wgt = np.maximum(ndimage.convolve(dm.astype(float), k, mode='constant'), 1e-6)
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            avg = ndimage.convolve(ch * dm, k, mode='constant') / wgt
            ch[m] = avg[m]

def grain(img, mk, ref, amount=1.0, ignore=None):
    """Noise with the local standard deviation of the pixels around the hole."""
    hi = ref - ndimage.uniform_filter(ref, size=(3, 3, 1))
    k = np.ones((9, 9))
    bad = mk if ignore is None else mk | ignore
    known = (~ndimage.binary_dilation(bad, iterations=3)).astype(float)
    var = ndimage.convolve((hi ** 2).sum(-1) * known, k) / np.maximum(ndimage.convolve(known, k), 1)
    var = ndimage.grey_dilation(var, size=(5, 5))
    sd = np.sqrt(var / 3) * amount
    out = img.copy()
    n = RNG.standard_normal(img.shape[:2])
    out[mk] += (n[mk] * sd[mk])[:, None]
    return out

def exemplar(idx, pal, mask, avoid, P=9, R=36):
    """Exemplar fill of mask (from build_beta.py): copy CLUT indices from the
    best matching patch within R px that has no pixel in avoid, most-known
    target patch first."""
    from numpy.lib.stride_tricks import sliding_window_view
    idx = idx.copy()
    ys, xs = np.nonzero(mask)
    if not len(ys): return idx
    hp = P // 2
    y0, y1 = max(0, ys.min() - R - P), min(idx.shape[0], ys.max() + R + P + 1)
    x0, x1 = max(0, xs.min() - R - P), min(idx.shape[1], xs.max() + R + P + 1)
    sub = idx[y0:y1, x0:x1]
    mk = mask[y0:y1, x0:x1].copy()
    orig_known = ~(mk | avoid[y0:y1, x0:x1])
    src_ok = ndimage.minimum_filter(orig_known.astype(np.uint8), P, mode='constant') > 0
    known = orig_known.copy()
    rgb = pal[sub]
    h, w = sub.shape
    win_rgb = sliding_window_view(np.pad(rgb, ((hp, hp), (hp, hp), (0, 0))), (P, P), (0, 1))
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
        ssd = np.where(src_ok[sy0:sy1, sx0:sx1], ssd, np.inf)
        gy, gx = np.mgrid[sy0:sy1, sx0:sx1]
        ssd = ssd * (1 + 0.002 * np.hypot(gy - ty, gx - tx))
        if not np.isfinite(ssd).any():
            mk[ty, tx] = False; known[ty, tx] = True; continue
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
    idx[y0:y1, x0:x1] = sub
    return idx

RING = 5        # px of shadow/glow rim added around a JP mask (IP.text_mask)

def ring_mask(img, core, protect=None, ring=RING, thr=40, halo=1, shadow=None):
    """core plus the pixels within ring px of it that differ from the
    background (IP.text_mask with core as the ink)."""
    if not core.any(): return core
    ys, xs = np.where(core)
    rect = (xs.min() - ring - 2, ys.min() - ring - 2, xs.max() + ring + 3, ys.max() + ring + 3)
    m = IP.text_mask(img, rect, pred=lambda _: core, ring=ring, thr=thr, halo=halo,
                     shadow=shadow, min_px=1, protect=protect)
    # small holes (stray shadow pixels) would seed blotches in the fill
    hole = ndimage.binary_fill_holes(m) & ~m
    lab, n = ndimage.label(hole)
    if n:
        sz = ndimage.sum(hole, lab, np.arange(1, n + 1))
        m |= np.isin(lab, 1 + np.nonzero(sz <= 40)[0])
    if protect is not None: m &= ~protect
    return m

def fill(pg, img, mk, avoid=None):
    """IP.fill (LaMa, CPU fallback auto) of mk; marks the pixels for the
    dithered write-back."""
    if not mk.any(): return img
    pg.fillm |= mk
    return IP.fill(img, mk, method='best', avoid=avoid)

def rect_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool); m[max(0, y0):y1, max(0, x0):x1] = True
    return m

def poly_mask(shape, pts):
    im = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.asarray(im) > 127

def diff_mask(img, rect, thr=40, grow=2, unknown=None):
    """Pixels in rect far from a background diffused in from its border."""
    x0, y0, x1, y1 = rect
    r = rect_mask(img.shape[:2], x0, y0, x1, y1)
    bg = inpaint(img, r if unknown is None else r | unknown, iters=200)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    mk = ndimage.binary_closing(r & (dist > thr), iterations=1)
    return ndimage.binary_dilation(mk, iterations=grow) & ndimage.binary_dilation(r, iterations=grow)

def mode_colour(img, m):
    px = img[m].astype(int)
    key = px[:, 0] // 8 * 1024 + px[:, 1] // 8 * 32 + px[:, 2] // 8
    v, c = np.unique(key, return_counts=True)
    return np.median(px[key == v[c.argmax()]], 0).astype(float)

# ---------------------------------------------------------------- text

_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc: _fc[k] = fontpath.truetype(path, size)
    return _fc[k]

def render(text, path, size, shear=0.0, sx=1.0, ss=4, track=0.0):
    """Alpha (h, w) of one line, cropped to the ink. Returns (alpha, top)
    where top is the ink top relative to the line's cap top."""
    f = font(path, int(round(size * ss)))
    asc, desc = f.getmetrics()
    pad = int(4 * ss + abs(shear) * (asc + desc))
    w = int(f.getlength(text) + track * ss * len(text) + 2 * pad)
    im = Image.new('L', (w, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    x = pad
    if track:
        for ch in text:
            d.text((x, pad), ch, font=f, fill=255); x += f.getlength(ch) + track * ss
    else:
        d.text((pad, pad), text, font=f, fill=255)
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * (pad + asc), 0, 1, 0),
                          resample=Image.BICUBIC)
    nw = max(1, int(round(im.width * sx / ss))); nh = max(1, int(round(im.height / ss)))
    a = np.asarray(im.resize((nw, nh), Image.BOX), float) / 255.0
    cols = np.where(a.max(0) > 0.02)[0]; rows = np.where(a.max(1) > 0.02)[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

def text_width(text, path, size, sx=1.0):
    return font(path, int(round(size * 4))).getlength(text) * sx / 4

def over(img, x, y, a, col, alpha=1.0):
    """Composite colour (RGB or HxWx3 map) at (x, y) with alpha a."""
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return
    aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    c = np.asarray(col, float)
    c = c if c.ndim == 1 else c[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = img[y0:y1, x0:x1]
    reg[:] = reg * (1 - aa) + c * aa

def dilate(a, r):
    """Round dilation of an alpha map, padded by r on each side."""
    if r <= 0: return a
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (xx * xx + yy * yy) <= r * r + 0.5
    return ndimage.grey_dilation(np.pad(a, r), footprint=fp)


def grad(h, w, stops, axis=0):
    """Gradient with stops [(t, rgb)] along rows (axis 0) or columns (1)."""
    n = h if axis == 0 else w
    t = np.linspace(0, 1, n)
    ts = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops], float)
    col = np.stack([np.interp(t, ts, cs[:, c]) for c in range(3)], -1)
    if axis == 0: return np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    return np.broadcast_to(col[None, :, :], (h, w, 3)).copy()

# ---------------------------------------------------------------- layout

# Per card: game key for tl/MINITTL.txt; title erase rect (x0, y0, x1, y1)
# and diff threshold (None = title kept); band = rows of the PK banner that
# is rebuilt per column; plate = rough rect of the tagline plate, its text
# colour; stamp = outer corners TL TR BR BL of the 練習用 frame (measured
# from 3x crops, +-1 px). Practice cards share the game of their pair.
CFG = {
    '1': dict(game='hr', erase=(50, 14, 470, 78), thr=34, plate=(250, 115, 505, 215), ink=(232, 224, 88)),
    '2': dict(game='pk', oneline=True, band=(29, 83), erase=(40, 29, 470, 83), thr=40, plate=(5, 115, 265, 220), ink=(248, 32, 136)),
    '3': dict(game='ex', erase=(16, 14, 486, 76), thr=26, grow=3, fill='ex', avoid_brown=True, R=48, plate=(5, 90, 265, 210), ink=(224, 40, 192)),
    '4': dict(game='sv', erase=(44, 8, 458, 84), thr=34, grow=5, shadow=(-2, 4), plate=(235, 125, 510, 235), ink=(248, 248, 0)),
    '9': dict(game='sp', erase=(0, 16, 466, 82), thr=24, grow=3, plate=(245, 120, 505, 225), ink=(176, 80, 248)),
    'A': dict(game='dd', erase=None, plate=(220, 115, 510, 225), ink=(248, 240, 0)),
    '5': dict(game='hr', erase=(50, 20, 476, 86), thr=34, plate=(250, 140, 510, 300), ink=(248, 192, 232),
              stamp=[(375, 79.3), (465, 60.7), (470, 99), (381.7, 118.3)]),
    '6': dict(game='pk', oneline=True, band=(29, 83), erase=(40, 29, 470, 83), thr=40, plate=(260, 130, 510, 270),
              ink=(248, 224, 104), stamp=[(385.7, 80), (480.7, 55), (488.3, 92.7), (392.7, 116)]),
    '7': dict(game='ex', erase=(26, 18, 494, 80), thr=26, grow=3, fill='ex', diff_rect=(430, 10, 512, 125), plate=(10, 120, 270, 260), ink=(248, 216, 144),
              stamp=[(396, 84.3), (485, 61.7), (490, 100), (403.3, 117.3)]),
    '8': dict(game='sv', erase=(50, 6, 470, 80), thr=34, grow=5, shadow=(-2, 4), keep=[(180, 48, 345, 100)], plate=(260, 140, 510, 280), ink=(176, 248, 216),
              stamp=[(376, 77.7), (466.7, 55), (470, 91.7), (385, 115)]),
    'B': dict(game='sp', erase=(0, 16, 482, 84), thr=24, grow=3, plate=(270, 130, 510, 270), ink=(248, 96, 112),
              stamp=[(390.7, 76.7), (480, 55), (483.3, 90), (396.7, 111.7)]),
    'C': dict(game='dd', erase=None, fill='ex', plate=(210, 150, 510, 270), ink=(248, 232, 40),
              stamp=[(390.7, 81.7), (486.7, 56.7), (491.7, 95), (398.3, 116.7)]),
}

def rot_rect(c4):
    """Rotated rect (cx, cy, w, h, deg) from four measured corners."""
    p = np.array(c4, float)
    cx, cy = p.mean(0)
    top, bot = p[1] - p[0], p[2] - p[3]
    ang = np.degrees(np.arctan2(-(top[1] + bot[1]), top[0] + bot[0]))
    w = (np.hypot(*top) + np.hypot(*bot)) / 2
    h = (np.hypot(*(p[3] - p[0])) + np.hypot(*(p[2] - p[1]))) / 2
    return cx, cy, w, h, ang

def rect_corners(cx, cy, w, h, ang, grow=0.0):
    t = np.radians(ang); u = np.array([np.cos(t), -np.sin(t)]); v = np.array([np.sin(t), np.cos(t)])
    w2, h2 = w / 2 + grow, h / 2 + grow
    c = np.array([cx, cy])
    return [c - u * w2 - v * h2, c + u * w2 - v * h2, c + u * w2 + v * h2, c - u * w2 + v * h2]

# ---------------------------------------------------------------- erase

def plate_shape(img, cfg, practice):
    """Inside mask of the tagline plate and its colour."""
    shp = img.shape[:2]
    R = rect_mask(shp, *cfg['plate'])
    pcol = mode_colour(img, R)
    near = np.sqrt(((img - pcol) ** 2).sum(-1)) < 28
    lab, _ = ndimage.label(near & R)
    k = np.bincount(lab.ravel()); k[0] = 0
    comp = lab == k.argmax()
    if practice:
        ys, xs = np.where(comp)
        t = (xs[ys == ys.min()].mean(), ys.min()); b = (xs[ys == ys.max()].mean(), ys.max())
        l = (xs.min(), ys[xs == xs.min()].mean()); r = (xs.max(), ys[xs == xs.max()].mean())
        inside = poly_mask(shp, [t, r, b, l])
    else:
        ys = np.where(comp.any(1))[0]
        inside = np.zeros(shp, bool)
        y0, y1 = ys.min() + 1, ys.max() - 1
        # straight slanted sides from the row extents near top and bottom
        a0 = np.where(comp[y0 + 2])[0]; a1 = np.where(comp[y1 - 2])[0]
        inside = poly_mask(shp, [(a0.min(), y0 + 2), (a0.max(), y0 + 2), (a1.max(), y1 - 2),
                                 (a1.min(), y1 - 2)])
        inside = ndimage.binary_dilation(inside, iterations=2) & ndimage.binary_dilation(comp, iterations=3)
        inside |= comp
        inside = ndimage.binary_fill_holes(inside)
    return inside, pcol

def ink_style(img, inside, ink):
    """Text shadow colour and the share of shadow at +1+1 vs outline."""
    F = (np.sqrt(((img - ink) ** 2).sum(-1)) < 40) & inside
    S = np.roll(np.roll(F, 1, 0), 1, 1) & ~F & inside
    c = img[S]; c = c[c.sum(1) < np.median(c.sum(1)) + 1]
    return np.median(c, 0), F

def _fit_x(ys, xs, it=400, tol=1.0):
    """Robust line x = a*y + c (RANSAC, then least squares on inliers)."""
    rng = np.random.default_rng(0); best = None
    for _ in range(it):
        i, j = rng.choice(len(ys), 2, replace=False)
        if ys[i] == ys[j]: continue
        a = (xs[j] - xs[i]) / (ys[j] - ys[i]); c = xs[i] - a * ys[i]
        inl = np.abs(xs - (a * ys + c)) <= tol
        if best is None or inl.sum() > best.sum(): best = inl
    return np.polyfit(ys[best], xs[best], 1)

def rhombus(img, cfg, pcol):
    """The whole practice rhombus: its four edges fitted to the row extents
    of the plate-coloured area (glyphs that touch the edge are outliers).
    plate_shape's inside is up to ~10 px smaller on some edges."""
    R = rect_mask(img.shape[:2], *cfg['plate'])
    near = np.sqrt(((img - pcol) ** 2).sum(-1)) < 28
    lab, _ = ndimage.label(near & R)
    k = np.bincount(lab.ravel()); k[0] = 0
    comp = lab == k.argmax()
    ys, xs = np.where(comp)
    t, b = ys.min(), ys.max()
    ly = ys[xs == xs.min()].mean(); ry = ys[xs == xs.max()].mean()
    rows = np.arange(t, b + 1); has = comp[t:b + 1].any(1)
    L = np.array([np.where(r)[0].min() if r.any() else -1 for r in comp[t:b + 1]])
    Rr = np.array([np.where(r)[0].max() if r.any() else -1 for r in comp[t:b + 1]])
    ln = []
    for X, y0, y1 in ((L, t, ly), (L, ly, b), (Rr, t, ry), (Rr, ry, b)):
        k = (rows > y0 + 2) & (rows < y1 - 2) & has
        ln.append(_fit_x(rows[k].astype(float), X[k].astype(float)))
    def isect(p, q):
        y = (q[1] - p[1]) / (p[0] - q[0]); return (p[0] * y + p[1], y)
    tl, bl, tr, br = ln
    return poly_mask(img.shape[:2], [isect(tl, tr), isect(tr, br), isect(br, bl), isect(bl, tl)])

def erase_plate(pg, cfg, practice, log):
    img = pg.img; shp = img.shape[:2]
    inside, pcol = plate_shape(img, cfg, practice)
    pl_col = pcol
    shcol, F = ink_style(img, inside, np.array(cfg['ink'], float))
    # inside sets the EN layout; full is the whole plate for the erase
    full = rhombus(img, cfg, pcol) | inside if practice else inside
    inner = ndimage.binary_erosion(full, iterations=1)
    near = np.sqrt(((img - pcol) ** 2).sum(-1)) < 4      # the plates are flat
    img[inner & ~near] = pcol
    # glyph pixels on the 1 px edge of the plate (nearly all edge pixels
    # are in the plate colour)
    ed = full & ~inner
    edink = ed & ((np.sqrt(((img - cfg["ink"]) ** 2).sum(-1)) < 80) |
                  (np.sqrt(((img - shcol) ** 2).sum(-1)) < 40))
    img[edink] = pcol
    # text that runs off the plate: ink or shadow pixels outside it, inside
    # the bounding box of all JP ink in the plate rect
    R = rect_mask(shp, *cfg['plate'])
    Fa = (np.sqrt(((pg.orig - cfg['ink']) ** 2).sum(-1)) < 40) & R
    Fa = Fa & ndimage.binary_dilation(F, iterations=24)
    ys, xs = np.where(Fa)
    box = rect_mask(shp, xs.min() - 3, ys.min() - 3, xs.max() + 4, ys.max() + 5)
    o = pg.orig
    inkm = (np.sqrt(((o - cfg['ink']) ** 2).sum(-1)) < 60) | (np.sqrt(((o - shcol) ** 2).sum(-1)) < 40)
    if practice:
        # glyph pixels in the plate colour just outside the rhombus edge
        nearp = np.sqrt(((o - pl_col) ** 2).sum(-1)) < 28
        inkm |= nearp & ~ndimage.binary_dilation(full, iterations=1) & ndimage.binary_dilation(full, iterations=6)
    spill = ndimage.binary_dilation(inkm & box & ~full, iterations=2) & ~full & box
    if spill.any():
        spill = ring_mask(img, spill, protect=full)
        # plate-coloured edge pixels just outside the plate are no source
        img = fill(pg, img, spill, avoid=ndimage.binary_dilation(full, iterations=2))
        # no dither along the plate edge (it scatters plate-coloured dots)
        pg.fillm &= ~ndimage.binary_dilation(full, iterations=3)
    pg.img = img
    log.append(f'plate {tuple(int(v) for v in pcol)} shadow {tuple(int(v) for v in shcol)} spill {int(spill.sum())}')
    return dict(inside=inside, pcol=pcol, shcol=shcol)

def erase_title(pg, cfg, smk, log):
    img = pg.img; shp = img.shape[:2]
    mk = np.zeros(shp, bool)
    if cfg.get('erase'):
        mk = diff_mask(img, cfg['erase'], thr=cfg['thr'], grow=cfg.get('grow', 2), unknown=smk)
        for r in cfg.get('keep', []):
            # art under the title (hair): only bright title ink is removed
            km = rect_mask(shp, *r)
            bright = (img.sum(-1) > 330) | ((img.max(-1) > 200) & (img.max(-1) - img.min(-1) > 120))
            mk &= ~km | ndimage.binary_dilation(bright, iterations=1)
    prot = None
    for r in cfg.get('keep', []):
        km = rect_mask(shp, *r)
        prot = (km & ~mk) if prot is None else prot | (km & ~mk)
    if 'band' in cfg:
        # the PK title stays inside its banner; the dark strip below the
        # banner and the art around it are not touched
        band = rect_mask(shp, 0, cfg['band'][0], PW, cfg['band'][1])
        mk = ring_mask(img, mk & band, protect=~band) | ring_mask(img, smk)
    else:
        mk = ring_mask(img, mk | smk, protect=prot, thr=cfg.get('rthr', 40),
                       halo=cfg.get('halo', 1), shadow=cfg.get('shadow'))
    if 'band' in cfg:
        b0, b1 = cfg['band']
        band = rect_mask(shp, 0, b0, PW, b1)
        new = img.copy()
        # band colour per column: the commonest colour that is not title ink
        # (yellow face, orange/brown outline and shadow have blue < 16)
        bc = np.full((PW, 3), np.nan)
        for x in range(PW):
            col = img[b0:b1, x]
            ok = (col[:, 2] >= 16) & ~((col[:, 1] > 140) & (col[:, 2] < 110))
            if not ok.any(): continue
            c = col[ok].astype(int)
            key = c[:, 0] * 65536 + c[:, 1] * 256 + c[:, 2]
            v, cnt = np.unique(key, return_counts=True); k = v[cnt.argmax()]
            bc[x] = (k >> 16, k >> 8 & 255, k & 255)
        good = ~np.isnan(bc[:, 0])
        for c in range(3): bc[:, c] = np.interp(np.arange(PW), np.where(good)[0], bc[good, c])
        # columns whose commonest colour is still glyph rim: take the
        # median of the neighbouring columns
        med = ndimage.median_filter(bc, size=(9, 1), mode='nearest')
        odd = np.abs(bc - med).sum(-1) > 40
        bc[odd] = med[odd]
        for x in range(PW):
            col = img[b0:b1, x]
            ink = (col[:, 2] < 16) | ((col[:, 1] > 140) & (col[:, 2] < 110))
            far = np.abs(col - bc[x]).sum(-1) > 60
            sel = (mk[b0:b1, x] & far) | (ink & np.isin(np.arange(b0, b1), np.arange(b0, b1)) & (far | ink))
            if not cfg['erase'][0] <= x < cfg['erase'][2]:
                sel = mk[b0:b1, x] & far
            new[b0:b1, x][sel] = bc[x]
        img = new
        mk &= ~band
    if mk.any() and cfg.get('fill') == 'ex' and EXEMPLAR:
        bad = [i for i, v in enumerate(pg.clut) if v == 0]
        idx = quantize(np.clip(img, 0, 255), pg.pal, pg.idx, bad)
        av = mk.copy()
        if cfg.get('avoid_brown'):
            o = pg.orig
            brown = (o[..., 0] > o[..., 1]) & (o[..., 1] > o[..., 2]) & (o[..., 0] - o[..., 2] > 60) & (o[..., 2] < 100)
            av |= ndimage.binary_dilation(brown, iterations=2)
        dr = np.zeros(shp, bool)
        if 'diff_rect' in cfg: dr = rect_mask(shp, *cfg['diff_rect']) & mk
        idx = exemplar(idx, pg.pal, mk & ~dr, av, P=cfg.get('P', 9), R=cfg.get('R', 36))
        img = img.copy(); img[mk & ~dr] = pg.pal[idx][mk & ~dr]
        if dr.any():
            img = fill(pg, img, dr)
    elif mk.any():
        img = fill(pg, img, mk)
    pg.img = img
    log.append(f'title erase {int(mk.sum())} px')
    return mk


# ---------------------------------------------------------------- stamp

def stamp_style(img, c4):
    """Rect and colours of the 練習用 stamp: frame colour (3 px band inside
    the outer edge) and its dark outline."""
    shp = img.shape[:2]
    rect = rot_rect(c4)
    ring = poly_mask(shp, rect_corners(*rect)) & ~poly_mask(shp, rect_corners(*rect, grow=-6))
    px = img[ring].astype(int); key = px[:, 0] // 16 * 256 + px[:, 1] // 16 * 16 + px[:, 2] // 16
    v, c = np.unique(key, return_counts=True)
    cl = [np.median(px[key == v[i]], 0) for i in np.argsort(-c)[:5]]
    fc = next(x for x in cl if x.max() >= 200)            # commonest bright cluster
    dark = min(cl, key=lambda x: x.sum())                 # darkest of the top five
    return dict(rect=rect, frame=fc, outline=dark)

# Title boxes (x0, y0, x1, y1) the EN lockup is fitted into, and styles.
# A title "Name's|Game" is set as a small first line over a big second one.
TBOX = {'1': (54, 12, 462, 80), '5': (56, 16, 440, 84), '2': (50, 30, 462, 82),
        '6': (50, 30, 420, 82), '3': (22, 12, 482, 80), '7': (30, 12, 440, 82),
        '4': (48, 8, 456, 80), '8': (56, 6, 430, 78), '9': (36, 14, 462, 82),
        'B': (46, 14, 430, 84)}
TSTYLE = {'hr': 'shoma', 'pk': 'pk', 'ex': 'ex', 'sv': 'sv', 'sp': 'sp'}
STYLE = {
    # font, shear, small/big size ratio, max big size, pads (l, t, r, b)
    'shoma': dict(font=F_BLACKI, shear=0.0, k=0.55, big=60, pad=(1, 1, 6, 6)),
    'pk': dict(font=F_ROUND, shear=0.0, k=0.55, big=56, pad=(2, 2, 3, 3)),
    'ex': dict(font=F_ROUND, shear=0.25, k=0.55, big=60, pad=(4, 1, 4, 4)),
    'sv': dict(font=F_ROUND, shear=0.0, k=0.55, big=60, pad=(3, 3, 5, 5)),
    'sp': dict(font=F_ROUND, shear=0.22, k=0.55, big=60, pad=(18, 2, 3, 3)),
}

def lockup(text, st, w, h):
    """Alpha of the title lockup that fits w x h, and the big size used."""
    parts = text.split('|')
    for big in np.arange(st['big'], 16, -1.0):
        als = [render(t, st['font'], big * (st['k'] if i == 0 and len(parts) > 1 else 1), st['shear'])
               for i, t in enumerate(parts)]
        gap = -int(round(big * 0.02))
        W = max(a.shape[1] for a in als)
        if len(als) > 1:
            ind = int(round(big * 0.3))
            W = max(als[0].shape[1] + ind, als[1].shape[1])
        H = sum(a.shape[0] for a in als) + gap * (len(als) - 1)
        if W <= w and H <= h: break
    A = np.zeros((H, W)); y = 0
    for i, a in enumerate(als):
        x = (W - a.shape[1]) // 2 if len(als) == 1 else (0 if i else 0)
        if len(als) > 1 and i == 1: x = W - a.shape[1]
        A[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(A[y:y + a.shape[0], x:x + a.shape[1]], a)
        y += a.shape[0] + gap
    return A, big

def lerp(c0, c1, t): return np.array(c0, float) * (1 - t) + np.array(c1, float) * t

def draw_title(pg, n, text):
    img = pg.img
    sname = TSTYLE[CFG[n]['game']]; st = STYLE[sname]
    x0, y0, x1, y1 = TBOX[n]; pl, pt, pr, pb = st['pad']
    A, big = lockup(text, st, x1 - x0 - pl - pr, y1 - y0 - pt - pb)
    h, w = A.shape
    X = int(round(x0 + pl + (x1 - x0 - pl - pr - w) / 2)); Y = int(round(y0 + pt + (y1 - y0 - pt - pb - h) / 2))
    if sname == 'shoma':
        # pale face, blue-violet extrusion down-right, dark rim
        o = dilate(A, 1)
        for k in range(6, 0, -1):
            over(img, X - 1 + k, Y - 1 + k, o, lerp((40, 24, 120), (104, 88, 224), (6 - k) / 5))
        over(img, X - 1, Y - 1, o, (56, 40, 168))
        g = grad(h, w, [(0, (248, 248, 255)), (0.45, (216, 208, 248)), (0.8, (168, 160, 240)), (1, (200, 192, 248))])
        over(img, X, Y, A, g)
    elif sname == 'pk':
        o = dilate(A, 1)
        over(img, X + 1, Y + 2, o, (48, 0, 16), 0.9)
        over(img, X - 1, Y - 1, o, (168, 80, 0))
        g = grad(h, w, [(0, (248, 248, 160)), (0.35, (248, 240, 24)), (0.75, (248, 208, 0)), (1, (240, 160, 0))])
        over(img, X, Y, A, g)
    elif sname == 'ex':
        o = dilate(A, 1)
        over(img, X - 4, Y + 2, o, (216, 24, 168))
        over(img, X + 3, Y + 3, o, (32, 40, 200))
        over(img, X - 1, Y - 1, o, (96, 48, 0), 0.7)
        g = grad(h, w, [(0, (248, 248, 120)), (0.5, (248, 240, 24)), (1, (232, 216, 0))])
        over(img, X, Y, A, g)
    elif sname == 'sv':
        o2 = dilate(A, 2)
        sh = ndimage.gaussian_filter(dilate(A, 3), 1.2)
        over(img, X - 1, Y, sh, (16, 48, 32), 0.75)
        over(img, X - 2, Y - 2, o2, (240, 248, 232))
        g = grad(h, w, [(0, (40, 216, 64)), (0.45, (176, 232, 32)), (0.7, (248, 232, 24)), (1, (248, 48, 32))], axis=1)
        g = g * grad(h, w, [(0, (1.08, 1.08, 1.08)), (1, (0.88, 0.88, 0.88))])
        over(img, X, Y, A, np.clip(g, 0, 255))
    elif sname == 'sp':
        # motion trail to the left
        o = dilate(A, 1)
        for k in range(16, 0, -2):
            over(img, X - 1 - k, Y - 1, o, lerp((232, 72, 16), (248, 160, 32), k / 16), 0.55 * (1 - k / 18))
        over(img, X + 1, Y + 1, o, (96, 16, 48), 0.9)
        over(img, X - 1, Y - 1, o, (128, 40, 0))
        g = grad(h, w, [(0, (248, 240, 16)), (0.5, (200, 248, 24)), (1, (112, 232, 48))], axis=1)
        over(img, X, Y, A, g)
    return f'title {sname} big {big:.0f} {w}x{h} at {X},{Y}'

def draw_plate(pg, n, pl, text, smax=17.0):
    """Tagline: each line centred on the plate interior at its own row,
    ink colour with a dark drop shadow (+1+1, 1 px spread)."""
    img = pg.img; inside = pl['inside']
    inner = ndimage.binary_erosion(inside, iterations=1)
    lines = text.split('|'); k = len(lines)
    ys = np.where(inside.any(1))[0]
    # the visible face of the plate; the slanted plates have a dark 3D
    # edge below, which is not in inside
    cy = (ys.min() + ys.max()) / 2
    ink = np.array(CFG[n]['ink'], float); sh = pl['shcol']
    best = None
    cands = sorted(((size * sx, sx, size) for size in np.arange(smax, 10.9, -0.5)
                    for sx in (1.0, 0.94, 0.88)), key=lambda c: (-round(c[0] * 2), -c[1]))
    for _, sx, size in cands:
        pitch = round(size * 1.3)
        top = cy - pitch * k / 2
        ext = []
        for i in range(k):
            ly = int(round(top + i * pitch + pitch / 2))
            band = inner[ly - int(size * 0.45):ly + int(size * 0.45) + 1].all(0)
            xs = np.where(band)[0]
            if not len(xs): ext = None; break
            ext.append((xs.min(), xs.max()))
        if ext is None: continue
        if all(text_width(t, F_ROUNDX, size, sx) <= (b - a) - 10 for t, (a, b) in zip(lines, ext)):
            best = (size, sx, pitch, top, ext); break
    size, sx, pitch, top, ext = best
    for i, t in enumerate(lines):
        a = render(t, F_ROUNDX, size, sx=sx)
        a_, b_ = ext[i]
        # cap/x-height centre on the row centre
        ly = int(round(top + i * pitch + (pitch - a.shape[0]) / 2))
        lx = int(round(a_ + (b_ - a_ - a.shape[1]) / 2))
        over(img, lx, ly, dilate(a, 1), sh, 0.95)          # 1 px spread, offset +1+1
        over(img, lx, ly, a, ink)
    return f'tag size {size} sx {sx} lines {k}'

def draw_stamp(pg, st, text):
    """Rubber stamp: dark-rimmed frame and dark-rimmed letters in the frame
    colour, rotated like the original; the inside stays see-through."""
    cx, cy, w, h, ang = st['rect']
    ss = 4; W, H = int(round(w * ss)), int(round(h * ss))
    m = 6 * ss
    fr = Image.new('L', (W + 2 * m, H + 2 * m), 0); d = ImageDraw.Draw(fr)
    t_out, t_frame = 1.0 * ss, 3.0 * ss
    d.rectangle([m, m, m + W - 1, m + H - 1], fill=255)
    d.rectangle([m + t_frame + 2 * t_out, m + t_frame + 2 * t_out,
                 m + W - 1 - t_frame - 2 * t_out, m + H - 1 - t_frame - 2 * t_out], fill=0)
    face = Image.new('L', fr.size, 0); d2 = ImageDraw.Draw(face)
    d2.rectangle([m + t_out, m + t_out, m + W - 1 - t_out, m + H - 1 - t_out], fill=255)
    d2.rectangle([m + t_frame + t_out, m + t_frame + t_out,
                  m + W - 1 - t_frame - t_out, m + H - 1 - t_frame - t_out], fill=0)
    # letters: condensed caps filling the inner box
    iw = W - 2 * (t_frame + 2 * t_out) - 4 * ss; ih = H - 2 * (t_frame + 2 * t_out) - 5 * ss
    f = font(F_IMPACT, int(ih * 1.25))
    tim = Image.new('L', (int(f.getlength(text)) + 8 * ss, int(ih * 2)), 0)
    ImageDraw.Draw(tim).text((4 * ss, 0), text, font=f, fill=255)
    tim = tim.crop(tim.getbbox())
    iw, ih = int(iw), int(ih)
    tim = tim.resize((min(iw, int(tim.width * ih / tim.height)), ih), Image.LANCZOS)
    tx = m + (W - tim.width) // 2; ty = m + (H - tim.height) // 2
    face.paste(255, (tx, ty), tim)
    rim = fr.copy(); rim.paste(255, (tx, ty), tim)
    def rot(im):
        r = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        return np.asarray(r.resize((r.width // ss, r.height // ss), Image.BOX), float) / 255
    fa, ra = rot(face), rot(rim)
    ra = np.clip(np.maximum(ra, dilate(fa, 1)[1:-1, 1:-1]) * 1.15, 0, 1)
    fa = np.clip(fa * 1.15, 0, 1)
    X = int(round(cx - fa.shape[1] / 2)); Y = int(round(cy - fa.shape[0] / 2))
    over(pg.img, X + 2, Y + 3, ra, st['outline'], 0.75)
    over(pg.img, X, Y, ra, st['outline'])
    over(pg.img, X, Y, fa, st['frame'])
    return (f'stamp {w:.0f}x{h:.0f} {ang:.1f} deg frame {tuple(int(v) for v in st["frame"])} '
            f'outline {tuple(int(v) for v in st["outline"])}')

# ---------------------------------------------------------------- build

def paint_card(pg, n, tl, erase_only=False):
    cfg = CFG[n]; log = []; practice = n in PRACTICE
    shp = pg.img.shape[:2]
    smk = np.zeros(shp, bool)
    st = None
    if 'stamp' in cfg:
        st = stamp_style(pg.img, cfg['stamp'])
        smk = poly_mask(shp, rect_corners(*st['rect'], grow=3))
        smk |= np.roll(np.roll(smk, 3, 0), 2, 1)
    pl = erase_plate(pg, cfg, practice, log)
    erase_title(pg, cfg, smk, log)
    pg.erased = pg.img.copy()
    if erase_only: return log
    g = cfg['game']
    if cfg.get('erase'):
        t = tl[f'{g}_title']
        log.append(draw_title(pg, n, t.replace('|', ' ') if cfg.get('oneline') else t))
    log.append(draw_plate(pg, n, pl, tl[f'{g}_tag'] if not practice else tl['practice_tag']))
    if st:
        log.append(draw_stamp(pg, st, tl['practice_stamp']))
    return log

def slot(name):
    s = os.path.getsize(f'private/disc/DAT2/{name}')
    return s + (-s % 2048)

def sheet(pairs, path, scale=0.5):
    """JP | EN for each card, two cards per row."""
    w, h = int(PW * scale), int(PH * scale)
    rows = (len(pairs) + 1) // 2
    im = Image.new('RGB', (4 * w + 12, rows * (h + 4)), (0, 0, 0))
    for i, (a, b) in enumerate(pairs):
        x = (i % 2) * (2 * w + 12); y = (i // 2) * (h + 4)
        for k, p in enumerate((a, b)):
            im.paste(Image.fromarray(p.astype(np.uint8)).resize((w, h), Image.LANCZOS), (x + k * w, y))
    im.save(path)

def build(png=False, only=None, dry=False):
    tl = load_tl(); atl = minittl_atlas.load_tl()
    rtl = minittl_rules.load_tl(); rrows = minittl_rules.load_rows()
    problems = 0; pairs = {}
    if png: os.makedirs(PREV, exist_ok=True)
    for n in CARDS:
        if only and n not in only: continue
        name = f'MINITTL{n}.BIZ'
        d = bytearray(open(f'private/unpacked/{name}.lz', 'rb').read())
        pg = Page(d, TITLE)
        for line in paint_card(pg, n, tl, os.environ.get('MINITTL_ERASE_ONLY')):
            print(f'{n}: {line}')
        after = pg.store(d)
        alog = []
        minittl_atlas.paint(d, atl, alog)
        if n == '1':
            for line in alog: print(f'atlas: {line}')
        rlog = []
        rpages = minittl_rules.paint(d, n, rtl, rlog, rrows)
        for line in rlog: print(f'rules: {line}')
        if png:
            os.makedirs(minittl_rules.PREV, exist_ok=True)
            for pid, (a, b) in rpages.items():
                minittl_rules.preview([(a, b)], f'{minittl_rules.PREV}/{pid}.png')
        pairs[n] = (pg.orig, after)
        if png: Image.fromarray(after.astype(np.uint8)).save(f'{PREV}/T{n}.png')
        if dry or only: continue
        c = lz.compress(bytes(d), name=name); s = slot(name)
        print(f'{name}: {len(c)} bytes, slot {s}')
        if len(c) > s: print(f'{name}: too large'); problems += 1; continue
        open(f'private/build/{name}.lz', 'wb').write(d)
        open(f'private/build/{name}', 'wb').write(c)
    if not only:                  # MINITTL0: atlas and Festival menu header
        name = 'MINITTL0.BIZ'
        d = bytearray(open(f'private/unpacked/{name}.lz', 'rb').read())
        pg = Page(d, fest_menu.TIM); flog = []
        fest_menu.header(pg, atl['header'], log=flog)
        bg = pg.store(d)
        minittl_atlas.paint(d, atl)
        ms, mlog = fest_menu.minisel(open('private/disc/DAT1/MINISEL.BIN', 'rb').read(),
                                     minittl_atlas.ATLAS)
        for line in flog + mlog: print(f'fest: {line}')
        if png: Image.fromarray(bg.astype(np.uint8)).save(f'{PREV}/T0.png')
        if not dry: open('private/build/MINISEL.BIN', 'wb').write(ms)
    if not (dry or only):
        c = lz.compress(bytes(d), name=name); s = slot(name)
        print(f'{name}: {len(c)} bytes, slot {s}')
        if len(c) > s: print(f'{name}: too large'); problems += 1
        else:
            open(f'private/build/{name}.lz', 'wb').write(d)
            open(f'private/build/{name}', 'wb').write(c)
    if png:      # one game per row: normal card, then its practice card
        sheet([pairs[n] for n in '152637489BAC' if n in pairs], f'{PREV}/sheet.png')
    print(f'minittl: problems {problems}')
    return problems

if __name__ == '__main__':
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    sys.exit(1 if build('--png' in sys.argv, only or None, '--dry' in sys.argv) else 0)
