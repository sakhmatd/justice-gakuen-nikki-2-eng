# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Repaint the mini-game rules pages (DAT2/MINITTLn.BIZ) in English.

Called by tools/build_minittl.py for each MINITTL file; also runs alone
for previews (--png writes JP|EN pairs to private/re/minittl/rules/en/).
Wording: tl/RULES.txt. Rects and styles: data/minittl_rules.tsv. RE:
private/re/minittl/findings.org. Port of the rules-page part of the first
game's build_festpics.py (band header, section headings on a redrawn
ellipse, body text, grey label plates, the 戻る hint).

Text removal (erase_fill): each erase step's mask gets a 5 px rim of
pixels that differ from the local background (inpaint.text_mask) and is
filled with inpaint.fill('best'): LaMa if private/models/big-lama.pt and
torch are there, else the OpenCV 'auto' fill plus grain. Filled pixels are
written back with Floyd-Steinberg dither into the page's own CLUT; the EN
drawn afterwards keeps nearest-colour quantisation.

Pages (8bpp 512x480 TIMs): r1 MINITTL1 0x55000, r2a/r2b MINITTL2
0x55000/0x91800, r3 MINITTL3, r4 MINITTL4, r9 MINITTL9, rA MINITTLA. The
practice files 5-8, B, C hold byte-identical copies and get the same
result. Pages are painted from the original bytes in the file given, so
reruns are clean.
"""
import sys, os, struct, colorsys, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import inpaint as IP

F_BODY = 'tahomabd.ttf'
F_HEAD = 'seguibli.ttf'
PW, PH = 512, 480
PREV = 'private/re/minittl/rules/en'
RNG = np.random.default_rng(1)
# file id -> pages it holds (page id, TIM offset)
PAGES = {'1': [('r1', 0x55000)], '2': [('r2a', 0x55000), ('r2b', 0x91800)],
         '3': [('r3', 0x55000)], '4': [('r4', 0x55000)],
         '9': [('r9', 0x55000)], 'A': [('rA', 0x55000)]}
COPY = {'5': '1', '6': '2', '7': '3', '8': '4', 'B': '9', 'C': 'A'}

# ---------------------------------------------------------------- data

def load_tl(path='tl/RULES.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb_of(s):
    m = re.findall(r'(\d+),(\d+),(\d+)', s)
    return [np.array([float(a), float(b), float(c)]) for a, b, c in m]

def load_rows(path='data/minittl_rules.tsv'):
    rows = {}
    for line in open(path, encoding='utf-8'):
        if line.startswith(('#', 'id\t')) or not line.strip(): continue
        f = line.rstrip('\n').split('\t')
        r = dict(id=f[0], page=f[1], x=int(f[2]), y=int(f[3]), w=int(f[4]),
                 h=int(f[5]), kind=f[6], fill=f[7], shadow=f[8],
                 notes=f[9] if len(f) > 9 else '')
        for key in ('num', 'icon', 'botan', 'plate'):
            r[key] = [tuple(int(v) for v in m.split(','))
                      for m in re.findall(key + r'@(\d+,\d+,\d+,\d+)', r['notes'])]
        m = re.search(r'area@(\d+),(\d+),(\d+),(\d+)', r['notes'])
        r['area'] = tuple(int(v) for v in m.groups()) if m else \
            (r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h'])
        m = re.match(r'(\d+),(\d+),(\d+)/\+(\d+)\+(\d+)', r['shadow'])
        r['sh_col'] = np.array([float(m[1]), float(m[2]), float(m[3])]) if m else None
        r['sh_off'] = (int(m[4]), int(m[5])) if m else (1, 2)
        # an icon box ends where the caption under it starts
        for i, (x, y, w, h) in enumerate(r['icon']):
            for (bx, by, bw, bh) in r['botan']:
                if bx < x + w and x < bx + bw and y < by < y + h: h = by - y
            r['icon'][i] = (x, y, w, h)
        r['moved'] = []          # icons the EN text places itself ({iN})
        rows[r['id']] = r
    return rows

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

class Page:
    """One 8bpp 512x480 picture with its 256-colour CLUT."""
    def __init__(self, src, off):
        self.off = off
        assert struct.unpack_from('<II', src, off) == (0x10, 9), hex(off)
        self.clut = list(struct.unpack_from('<256H', src, off + 0x14))
        self.pal = np.array([rgb15(v) for v in self.clut])
        self.idx = np.frombuffer(bytes(src[off + 0x220:off + 0x220 + PW * PH]),
                                 np.uint8).reshape(PH, PW).copy()
        self.img = self.pal[self.idx]
        self.orig = self.img.copy()
        self.filled = np.zeros((PH, PW), bool)   # pixels set by erase_fill
        self.pre = None                          # the image after the last fill

    def store(self, d):
        img = np.clip(self.img, 0, 255)
        new = quantize(img, self.pal, self.idx,
                       [i for i, v in enumerate(self.clut) if v == 0])
        if self.filled.any():
            # fill pixels that no later drawing touched: dithered
            clut = np.array(self.clut)
            w = self.filled & (np.abs(self.img - self.pre).sum(-1) < 0.5)
            w &= np.abs(img - self.pal[self.idx]).sum(-1) > 3
            w &= clut[self.idx] != 0
            flat, main = flat_near(self.idx, self.filled)
            # flat ground: the ground colour where the fill is near it (LaMa
            # leaves a faint haze on flat panels), else nearest colour
            snap = w & flat & (np.abs(img - self.pal[main]).sum(-1) < SNAP)
            new[snap] = main[snap]
            w &= ~flat
            new = dither_local(img, new, self.idx, clut, self.pal, w, self.filled)
        d[self.off + 0x220:self.off + 0x220 + PW * PH] = new.tobytes()
        self.idx = new
        return self.pal[new]

SNAP = 36      # flat-ground snap distance (sum of |dRGB|, x << 3 scale)

def flat_near(idx, filled, size=9, share=0.85):
    """True on filled pixels whose nearest unfilled pixel sits on flat
    ground: one CLUT index holds >= share of the unfilled pixels in its
    size x size window (and at least a quarter of the window unfilled).
    Dithering there would only add stipple to a flat panel. Returns
    (mask, the window's main index at the nearest unfilled pixel)."""
    if filled.all(): return np.zeros(filled.shape, bool), idx
    known = ~filled
    nk = ndimage.uniform_filter(known.astype(float), size=size)
    best = np.zeros(idx.shape); main = idx.copy()
    near = ndimage.binary_dilation(filled, iterations=size)
    for k in np.unique(idx[known & near]):
        f = ndimage.uniform_filter(((idx == k) & known).astype(float), size=size)
        main[f > best] = k; best = np.maximum(best, f)
    flat = (best >= share * nk) & (nk >= 0.25)
    _, (iy, ix) = ndimage.distance_transform_edt(filled, return_indices=True)
    return flat[iy, ix] & filled, main[iy, ix]

def dither_local(img, new, orig, clut, pal, w, filled, tile=32, pad=8, floor=0.02):
    """Floyd-Steinberg (inpaint.to_clut) on w in tile x tile cells, each
    with only the CLUT entries that hold >= floor of the unfilled pixels
    within pad px of the cell, plus the nearest entries of fill colours
    more than SNAP from all of them. With the whole CLUT, or the colours of a
    distant part of the art, the error builds up into far colours (light
    dots on a flat-looking panel)."""
    H, W = w.shape
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            wk = np.zeros_like(w); wk[y0:y0 + tile, x0:x0 + tile] = w[y0:y0 + tile, x0:x0 + tile]
            if not wk.any(): continue
            ys = slice(max(0, y0 - pad), y0 + tile + pad)
            xs = slice(max(0, x0 - pad), x0 + tile + pad)
            k = orig[ys, xs][~filled[ys, xs]]
            k = k[clut[k] != 0]
            if not len(k): continue
            ids, cnt = np.unique(k, return_counts=True)
            ids = ids[cnt >= floor * len(k)]
            # fill colours none of these come near (art the fill continued
            # into the cell): add their nearest entries
            q = img[wk]
            far = np.abs(q[:, None, :] - pal[ids][None]).sum(-1).min(1) > SNAP
            ids = np.union1d(ids, new[wk][far])
            sub = np.zeros_like(clut); sub[ids] = clut[ids]
            base = new.copy(); base[wk] = ids[0]
            new = IP.to_clut(img, base, sub, wk, pal=pal, keep_stp=False)
    return new

def quantize(img, pal, idx, bad=()):
    """Nearest CLUT colour for every pixel that changed (weighted RGB).
    CLUT entries in bad (0x0000, transparent) are never chosen."""
    out = idx.copy()
    pal = pal.copy(); pal[list(bad)] = 1e6
    ch = np.abs(img - pal[idx]).sum(-1) > 3
    flat = img[ch]; wt = np.array([0.30, 0.59, 0.11]) ** 0.5
    pw = pal * wt; sel_all = np.where(ch.ravel())[0]
    for k in range(0, len(flat), 4096):
        q = flat[k:k + 4096] * wt
        dd = ((q[:, None, :] - pw[None]) ** 2).sum(-1)
        out.ravel()[sel_all[k:k + 4096]] = dd.argmin(1)
    return out

# ---------------------------------------------------------------- inpaint

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
        # coarse start
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
    wgt = ndimage.convolve(dm.astype(float), k, mode='constant')
    wgt = np.maximum(wgt, 1e-6)
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            avg = ndimage.convolve(ch * dm, k, mode='constant') / wgt
            ch[m] = avg[m]

def erase_fill(pg, mk, domain=None, amount=0.8, ignore=None, limit=None, ring=5,
               thr=40, method='best'):
    """Erase mk on pg in place: add a 5 px rim of pixels that differ from
    the background (shadows, glows), fill with inpaint.fill('best'), add
    grain only on the CPU fallback. Pixels outside domain are neither
    filled nor copied from; the rim stays inside limit if given (art next
    to the text). Returns the mask that was filled."""
    if not mk.any(): return mk
    avoid = None if domain is None else ~domain
    m2 = IP.text_mask(pg.img, IP._bbox(mk, 5, mk.shape), pred=lambda a: mk, ring=ring,
                      thr=thr, halo=1, min_px=0, close=0,
                      protect=None if avoid is None else avoid & ~mk)
    if avoid is not None: m2 &= ~avoid
    if limit is not None: m2 &= limit | mk
    # small unmasked islands inside the hole seed dark blobs in LaMa
    isl = ndimage.binary_fill_holes(m2) & ~m2
    if isl.any():
        lab, n = ndimage.label(isl)
        sz = ndimage.sum(isl, lab, np.arange(1, n + 1))
        isl = np.isin(lab, 1 + np.nonzero(sz < 40)[0])
        if avoid is not None: isl &= ~avoid
        m2 |= isl
    new = IP.fill(pg.img, m2, method, avoid=avoid)
    if amount and not (method == 'best' and IP.lama_ok() or method.startswith('lama')): new = grain(new, m2, pg.img, amount, ignore)
    pg.img = new
    pg.filled |= m2
    pg.pre = new.copy()
    return m2

def protect_mask_all(sh, rows, grow=0):
    p = np.zeros(sh, bool)
    for r in rows: p |= protect_mask(sh, r, grow)
    return p

def grain(img, mk, ref, amount=1.0, ignore=None):
    """Add noise with the local standard deviation of the unmasked pixels
    around the hole (the picture's quantisation noise)."""
    hi = ref - ndimage.uniform_filter(ref, size=(3, 3, 1))
    k = np.ones((9, 9))
    bad = mk if ignore is None else mk | ignore
    known = (~ndimage.binary_dilation(bad, iterations=3)).astype(float)
    var = ndimage.convolve((hi ** 2).sum(-1) * known, k) / np.maximum(ndimage.convolve(known, k), 1)
    var = ndimage.grey_dilation(var, size=(5, 5))
    sd = np.minimum(np.sqrt(var / 3), 6.0) * amount
    out = img.copy()
    n = RNG.standard_normal(img.shape[:2])
    out[mk] += (n[mk] * sd[mk])[:, None]
    return out

def rect_mask(shape, x, y, w, h, grow=0):
    m = np.zeros(shape, bool)
    m[max(0, y - grow):y + h + grow, max(0, x - grow):x + w + grow] = True
    return m

# ---------------------------------------------------------------- text

_fcache = {}
def font(path, size):
    k = (path, size)
    if k not in _fcache: _fcache[k] = fontpath.truetype(path, size)
    return _fcache[k]

def render(text, path, size, shear=0.0, sx=1.0, ss=4, track=0.0, per_letter=False):
    """Alpha (h, w) of one line of text. Returns (alpha, baseline_y, labels)
    where labels[y, x] is the index of the letter that owns the pixel (or -1)
    when per_letter is set. size is the font size in output pixels."""
    f = font(path, int(round(size * ss)))
    asc, desc = f.getmetrics()
    pad = int(2 * ss + abs(shear) * (asc + desc))
    xs = []; x = 0.0
    for i, ch in enumerate(text):
        xs.append(x)
        x += f.getlength(ch) + track * ss
    # kerning-aware total for plain text
    tw = int(max(x, f.getlength(text)) + 2 * pad)
    th = asc + desc + 2 * pad
    def draw(sub, at):
        im = Image.new('L', (tw, th), 0)
        ImageDraw.Draw(im).text((pad + at, pad), sub, font=f, fill=255)
        if shear:
            im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * (pad + asc), 0, 1, 0),
                              resample=Image.BICUBIC)
        nw = max(1, int(round(tw * sx / ss))); nh = max(1, int(round(th / ss)))
        return np.asarray(im.resize((nw, nh), Image.BOX), float) / 255.0
    if per_letter or track:
        layers = [draw(ch, xs[i]) for i, ch in enumerate(text)]
        a = np.max(layers, axis=0) if layers else draw('', 0)
        lab = np.argmax(layers, axis=0) if layers else np.zeros(a.shape, int)
        lab[a < 0.02] = -1
    else:
        a = draw(text, 0); lab = None
    # crop empty columns/rows but remember the baseline
    cols = np.where(a.max(0) > 0.01)[0]; rows_ = np.where(a.max(1) > 0.01)[0]
    base = (pad + asc) / ss
    if not len(cols): return np.zeros((1, 1)), 0, None
    a2 = a[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]
    if lab is not None: lab = lab[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]
    return a2, base - rows_[0], lab

def text_width(text, path, size, sx=1.0, track=0.0):
    f = font(path, max(1, int(round(size * 4))))
    return (f.getlength(text) + track * 4 * max(0, len(text) - 1)) * sx / 4

def wrap(text, path, size, width, first_indent=0, sx=1.0):
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            lim = width - (first_indent if not lines else 0)
            if not cur or text_width(t, path, size, sx) <= lim: cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

def fit_block(text, path, w, h, smax, smin, pitch_k=1.35, indent=0, sx_min=0.88):
    """Largest size (and horizontal squeeze) whose wrapped lines fit w x h."""
    for size in np.arange(smax, smin - 0.01, -0.5):
        for sx in (1.0, 0.94, sx_min):
            lines = wrap(text, path, size, w, indent, sx)
            pitch = max(size * pitch_k, size + 2)
            if all(text_width(l, path, size, sx) <= w - (indent if i == 0 else 0)
                   for i, l in enumerate(lines)) and \
               (len(lines) - 1) * pitch + size * 1.0 <= h:
                return size, sx, lines, pitch
    lines = wrap(text, path, smin, w, indent, sx_min)
    return smin, sx_min, lines, max(smin * pitch_k, smin + 2)

def over(img, x, y, a, col, alpha=1.0):
    """Composite colour (RGB or HxWx3 map) at (x, y) with alpha a."""
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return
    aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    c = col if np.ndim(col) == 1 else col[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = img[y0:y1, x0:x1]
    reg[:] = reg * (1 - aa) + c * aa

def dilate(a, r):
    if r <= 0: return a
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (xx * xx + yy * yy) <= r * r + 0.5
    p = np.pad(a, r)
    return ndimage.grey_dilation(p, footprint=fp)

# ---------------------------------------------------------------- masks

def ink_mask(img, r, fill=None, tol=90, grow=2, extra=2):
    """Text fill pixels close to the fill colour, plus their drop shadow,
    in the row's rect (grown by `extra`), minus protected icon rects."""
    sh = img.shape[:2]
    R = rect_mask(sh, r['x'], r['y'], r['w'], r['h'], extra)
    if fill is None: fill = rgb_of(r['fill'])[0]
    f = (np.sqrt(((img - fill) ** 2).sum(-1)) < tol) & R
    sdx, sdy = r['sh_off']
    m = f.copy()
    for dy in range(0, sdy + 2):
        for dx in range(0, sdx + 2):
            m |= np.roll(np.roll(f, dy, 0), dx, 1)
    m = ndimage.binary_dilation(m, iterations=grow) & rect_mask(sh, r['x'], r['y'], r['w'], r['h'], extra + 2)
    return m & ~protect_mask(sh, r)

def protect_mask(sh, r, grow=1):
    """Icons and numbered circles are round: protect the inscribed ellipse
    (so commas in the rect corners can be erased); plates are rects."""
    p = np.zeros(sh, bool)
    yy, xx = np.ogrid[0:sh[0], 0:sh[1]]
    for key in ('num', 'icon'):
        for i, (x, y, w, h) in enumerate(r[key]):
            if key == 'icon' and i in r.get('moved', ()): continue
            cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
            p |= ((xx - cx) / (w / 2 + grow + 0.5)) ** 2 + ((yy - cy) / (h / 2 + grow + 0.5)) ** 2 <= 1
    bot = min([b[1] for b in r.get('botan', [])], default=None)
    for (x, y, w, h) in r['plate']:
        if bot is not None and y < bot < y + h + grow:
            h = bot - y - grow       # the caption under the plate stays erasable
        p |= rect_mask(sh, x, y, w, h, grow)
    return p

def diff_mask(img, x0, y0, x1, y1, thr=40, grow=2, unknown=None):
    """Pixels in the rect far from a background diffused in from its border
    (pixels in `unknown` do not feed the background estimate)."""
    rect = np.zeros(img.shape[:2], bool); rect[y0:y1, x0:x1] = True
    bg = inpaint(img, rect if unknown is None else rect | unknown, iters=200)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    mk = rect & (dist > thr)
    mk = ndimage.binary_closing(mk, iterations=1)
    return ndimage.binary_dilation(mk, iterations=grow) & ndimage.binary_dilation(rect, iterations=grow)

def mode_colour(img, m):
    px = img[m].astype(int)
    if not len(px): return None
    key = px[:, 0] // 8 * 1024 + px[:, 1] // 8 * 32 + px[:, 2] // 8
    v, c = np.unique(key, return_counts=True)
    k = v[c.argmax()]
    return np.median(px[key == k], 0).astype(float)

# ---------------------------------------------------------------- body text

def draw_lines(img, lines, x, y, path, size, sx, pitch, fill, shadow, soff, indent=0,
               align='left', width=None, outline=None):
    """Draw lines with their tops at y + i * pitch. Returns the bounding box."""
    asc = font(path, int(round(size * 4))).getmetrics()[0] / 4
    box = [9999, 9999, 0, 0]
    for i, t in enumerate(lines):
        if not t: continue
        a, base, _ = render(t, path, size, sx=sx)
        ly = int(round(y + i * pitch + (pitch - size) / 2 + asc - base - size * 0.12))
        if align == 'centre':
            lx = int(round(x + (width - a.shape[1]) / 2))
        else:
            lx = x + (indent if i == 0 else 0)
        if shadow is not None:
            over(img, lx + soff[0], ly + soff[1], a, shadow, 0.95)
        if outline is not None:
            o = dilate(a, 1); over(img, lx - 1, ly - 1, o, outline)
        over(img, lx, ly, a, fill)
        box = [min(box[0], lx), min(box[1], ly), max(box[2], lx + a.shape[1]), max(box[3], ly + a.shape[0])]
    return box

# per-row layout overrides: area (x0, y0, x1, y1) the new text may use,
def erase_body(pg, r):
    """Ink near the fill colour plus its shadow (ink_mask), and anything in
    the box that differs clearly from the panel colour around it (thin
    strokes whose antialiasing is too far from the fill)."""
    mk = ink_mask(pg.img, r)
    sh = pg.img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    ring = rect_mask(sh, x, y, w, h, 5) & ~rect_mask(sh, x, y, w, h, 2)
    ring &= ~protect_mask(sh, r, 2)
    bg = np.median(pg.img[ring], 0)
    far = np.sqrt(((pg.img - bg) ** 2).sum(-1)) > 55
    far &= rect_mask(sh, x, y, w, h, 2)
    far = ndimage.binary_dilation(far, iterations=1) & rect_mask(sh, x, y, w, h, 3)
    mk |= far & ~protect_mask(sh, r)
    for (x, y, w, h) in r['botan']:
        mk |= botan_mask(pg.img, x, y, w, h, r)
    return mk

def botan_mask(img, x, y, w, h, r):
    sh = img.shape[:2]
    R = rect_mask(sh, x, y, w, h, 1)
    ring = rect_mask(sh, x, y, w, h, 4) & ~R
    bg = np.median(img[ring & ~protect_mask(sh, r, 0)], 0)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    m = R & (dist > 30)
    m = ndimage.binary_dilation(m, iterations=1) & rect_mask(sh, x, y, w, h, 2)
    # keep the icon itself (botan rect touches the icon bottom)
    for (ix, iy, iw, ih) in r['icon'] + r['plate']:
        m &= ~rect_mask(sh, ix, iy, iw, ih - 1, 0)
    return m

def header_band(pg, r, text, jobs):
    img = pg.img
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    ring = rect_mask(img.shape[:2], x, y, w, h) & ~rect_mask(img.shape[:2], x + 3, y + 3, w - 6, h - 6)
    band = mode_colour(img, ring)
    near = np.sqrt(((img - band) ** 2).sum(-1)) < 28
    # band pixels near the rect (the shadow can cut the band in pieces)
    comp = near & rect_mask(img.shape[:2], 0, y, PW, h, 8)
    comp = ndimage.binary_opening(comp, iterations=1)
    ys, xs = np.where(comp)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    dom = np.zeros(img.shape[:2], bool); dom[by0 + 1:by1 - 1, bx0 + 1:bx1 - 1] = True
    mk = dom & ~comp
    mk = ndimage.binary_dilation(mk, iterations=1) & dom
    erase_fill(pg, mk, dom, amount=0)
    # text
    sh_col = r['sh_col']; soff = r['sh_off']
    avail_w = (bx1 - bx0) - 24; avail_h = (by1 - by0) - 8
    size = 34
    while True:
        a, base, lab_ = render(text, F_HEAD, size, per_letter=True)
        sx = min(1.0, avail_w / (a.shape[1] + 2 + soff[0]))
        if (sx >= 0.86 and a.shape[0] + soff[1] + 2 <= avail_h) or size <= 16: break
        size -= 1
    if sx < 1.0:
        a, base, lab_ = render(text, F_HEAD, size, sx=sx, per_letter=True)
    letters = [i for i, c in enumerate(text) if c.strip()]
    hue = {}
    for j, i in enumerate(letters):
        t = j / max(1, len(letters) - 1)
        hue[i] = (1.0 - 0.97 * t) % 1.0
    colmap = np.zeros(a.shape + (3,))
    for i, hh in hue.items():
        rgbc = np.array(colorsys.hsv_to_rgb(hh, 0.95, 1.0)) * 255
        colmap[lab_ == i] = rgbc
    # light top, slightly darker bottom per glyph (as the original)
    shade = np.linspace(1.0, 0.82, a.shape[0])[:, None, None]
    colmap = colmap * shade
    o = dilate(a, 1)
    cx = (bx0 + bx1) / 2; cy = (by0 + by1) / 2
    X = int(round(cx - o.shape[1] / 2 - soff[0] / 2)); Y = int(round(cy - o.shape[0] / 2 - soff[1] / 2))
    def job():
        over(pg.img, X + soff[0], Y + soff[1], o, sh_col)
        over(pg.img, X, Y, o, np.array([248., 248, 248]))
        over(pg.img, X + 1, Y + 1, a, colmap)
    jobs.append(job)
    return f'{r["id"]}: size {size} sx {sx:.2f} band {bx0},{by0}-{bx1},{by1}'

def section_heading(pg, r, text, jobs):
    img = pg.img; shp = img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    # plate colour: the commonest colour in the rect that is not also in
    # strips above and below it (the background)
    strips = rect_mask(shp, x - 20, y - 10, w + 40, 6) | rect_mask(shp, x - 20, y + h + 4, w + 40, 6)
    key = lambda px: (px[..., 0].astype(int) // 8 * 1024 + px[..., 1].astype(int) // 8 * 32
                      + px[..., 2].astype(int) // 8)
    bgk = set(np.unique(key(img[strips])).tolist())
    rm = rect_mask(shp, x, y, w, h)
    cand = rm & ~np.isin(key(img), list(bgk))
    plate = mode_colour(img, cand)
    win = rect_mask(shp, x - 40, y - 6, w + 80, h + 12)
    near = (np.sqrt(((img - plate) ** 2).sum(-1)) < 40) & win
    near = ndimage.binary_closing(near, structure=np.ones((3, 9)), iterations=2)
    lab, _ = ndimage.label(near)
    k = np.bincount(lab[y:y + h, x:x + w].ravel()); k[0] = 0
    comp = lab == k.argmax()
    ys, xs = np.where(comp)
    ex0, ex1, ey0, ey1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cy = (ey0 + ey1) / 2; b = (ey1 - ey0) / 2
    # erase: ellipse + text
    yy, xx = np.mgrid[0:shp[0], 0:shp[1]]
    ell = ((xx - (ex0 + ex1) / 2) / ((ex1 - ex0) / 2 + 2)) ** 2 + ((yy - cy) / (b + 2)) ** 2 <= 1
    mk = ell | diff_mask(img, x - 2, y - 2, x + w + 4, y + h + 4, thr=45, grow=1)
    mk = ndimage.binary_dilation(mk, iterations=1)
    erase_fill(pg, mk, amount=0.8)
    # radial colour profile of the old plate (centre -> rim)
    rho = np.sqrt(((xx - (ex0 + ex1) / 2) / ((ex1 - ex0) / 2)) ** 2 + ((yy - cy) / b) ** 2)
    text_px = diff_mask(img, x, y, x + w, y + h, thr=45, grow=1)
    prof = []
    for k_ in range(12):
        sel = comp & ~text_px & (rho >= k_ / 12) & (rho < (k_ + 1) / 12)
        prof.append(np.median(img[sel], 0) if sel.sum() > 3 else None)
    last = plate
    for k_ in range(12):
        if prof[k_] is None: prof[k_] = last
        last = prof[k_]
    # text
    cols = rgb_of(r['fill'])
    sh_col = r['sh_col']; soff = r['sh_off']
    size = 24
    a, base, _ = render(text, F_HEAD, size)
    while a.shape[0] > h + 2 and size > 14:
        size -= 1; a, base, _ = render(text, F_HEAD, size)
    tw = a.shape[1]
    margin = max(8, ((ex1 - ex0) - w) / 2)
    nw = max(ex1 - ex0, tw + 2 * margin)
    nx0 = ex0; nx1 = ex0 + nw
    ncx = (nx0 + nx1) / 2
    gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    t = np.clip((gx / max(1, a.shape[1]) * 0.5 + gy / max(1, a.shape[0]) * 0.5), 0, 1)
    # alternate light/dark per letter like the original's 2-tone glyphs
    colmap = cols[0] * (1 - t[..., None]) + cols[1] * t[..., None]
    tx = int(round(ncx - tw / 2)); ty = int(round(y + (h - a.shape[0]) / 2))
    def job():
        im = pg.img
        rr = np.sqrt(((xx - ncx) / (nw / 2)) ** 2 + ((yy - cy) / b) ** 2)
        for k_ in range(12):
            sel = (rr >= k_ / 12) & (rr < (k_ + 1) / 12)
            im[sel] = prof[k_]
        edge = (rr >= 0.97) & (rr < 1.06)
        wgt = np.clip((1.06 - rr) / 0.09, 0, 1)
        im[edge] = (im[edge] * (1 - wgt[edge, None]) + prof[11] * wgt[edge, None])
        over(im, tx + soff[0], ty + soff[1], a, sh_col)
        over(im, tx, ty, a, colmap)
    jobs.append(job)
    return f'{r["id"]}: size {size} plate {ex1 - ex0}->{nw:.0f}'

def back_hint(pg, r, text, jobs):
    img = pg.img
    mk = ink_mask(img, r, tol=80, extra=1)
    # the rim must not reach the arrow right of the rect
    lim = rect_mask(img.shape[:2], r['x'] - 4, r['y'] - 4, r['w'] + 5, r['h'] + 8)
    erase_fill(pg, mk, amount=0, limit=lim)
    fill = rgb_of(r['fill'])[0]
    sh = r['sh_col'] if r['sh_col'] is not None else None
    size = 13.5
    a, base, _ = render(text, F_BODY, size)
    while a.shape[1] > r['w'] + 12 and size > 10:
        size -= 0.5; a, base, _ = render(text, F_BODY, size)
    tx = r['x'] + r['w'] - a.shape[1]; ty = int(round(r['y'] + (r['h'] - a.shape[0]) / 2))
    def job():
        if sh is not None: over(pg.img, tx + 1, ty + 1, a, sh)
        over(pg.img, tx, ty, a, fill)
    jobs.append(job)
    return f'{r["id"]}: size {size} w {a.shape[1]} (rect {r["w"]})'

# ---------------------------------------------------------------- body text
# EN text may name the row's icons as {i0}, {i1}... The icon is then cut
# from its old place and drawn inline, as a word of its own width.
# {icons} at the start keeps the icons in place and indents the first line.

TOK = re.compile(r'\{i(\d)\}')
ICON_GAP = 2

def icon_cut(pg, r, k):
    x, y, w, h = r['icon'][k]
    m = ellipse_mask(pg.img.shape[:2], x, y, w, h, 1)[y - 2:y + h + 2, x - 2:x + w + 2]
    return pg.orig[y - 2:y + h + 2, x - 2:x + w + 2].copy(), m

def seg_width(word, r, size, sx):
    """Pixel width of a word that may contain icon tokens."""
    w = 0.0
    for i, part in enumerate(TOK.split(word)):
        if i % 2: w += r['icon'][int(part)][2] + 2 * ICON_GAP
        elif part: w += text_width(part, F_BODY, size, sx)
    return w

def wrap_tok(text, r, size, width, indent, sx):
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            lim = width - (indent if not lines else 0)
            if not cur or seg_width(t, r, size, sx) <= lim: cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

def fit_tok(text, r, w, h, smax, smin, pitch_k, indent, sx_min=0.88, cap=99):
    """Largest size that fits; up to 2 px smaller is preferred to a
    horizontal squeeze."""
    near = [(z, 1.0) for z in np.arange(smax, max(smin, smax - 2) - 0.01, -0.5)]
    rest = [(z, sx) for z in np.arange(smax, smin - 0.01, -0.5) for sx in (1.0, 0.94, sx_min)]
    for size, sx in near + rest:
        if size <= cap + 0.01:
            lines = wrap_tok(text, r, size, w, indent, sx)
            pitch = max(size * pitch_k, size + 2)
            if all(seg_width(l, r, size, sx) <= w - (indent if i == 0 else 0)
                   for i, l in enumerate(lines)) and \
               (len(lines) - 1) * pitch + size * 1.0 <= h:
                return size, sx, lines, pitch, True
    lines = wrap_tok(text, r, smin, w, indent, sx_min)
    return smin, sx_min, lines, max(smin * pitch_k, smin + 2), False

def draw_tok(pg, r, lines, x, y, size, sx, pitch, fill, shadow, soff, indent):
    asc = font(F_BODY, int(round(size * 4))).getmetrics()[0] / 4
    box = [9999, 9999, 0, 0]
    for i, t in enumerate(lines):
        if not t: continue
        top = y + i * pitch + (pitch - size) / 2
        lx = x + (indent if i == 0 else 0)
        for j, part in enumerate(TOK.split(t)):
            if j % 2:
                k = int(part); px, m = icon_cut(pg, r, k)
                ih = r['icon'][k][3]
                ix = int(round(lx + ICON_GAP)); iy = int(round(top + size * 0.45 - ih / 2))
                reg = pg.img[iy - 2:iy - 2 + px.shape[0], ix - 2:ix - 2 + px.shape[1]]
                reg[m] = px[m]
                lx += r['icon'][k][2] + 2 * ICON_GAP
                continue
            if not part: continue
            # keep the spaces around icons
            core = part.strip(' ')
            if not core:
                lx += text_width(part, F_BODY, size, sx); continue
            lead = len(part) - len(part.lstrip(' '))
            if lead: lx += text_width(' ' * lead, F_BODY, size, sx)
            if core:
                a, base, _ = render(core, F_BODY, size, sx=sx)
                ly = int(round(top + asc - base - size * 0.12))
                ix = int(round(lx))
                if shadow is not None:
                    over(pg.img, ix + soff[0], ly + soff[1], a, shadow, 0.95)
                over(pg.img, ix, ly, a, fill)
                box = [min(box[0], ix), min(box[1], ly), max(box[2], ix + a.shape[1]),
                       max(box[3], ly + a.shape[0])]
                lx += text_width(core, F_BODY, size, sx)
            trail = len(part) - len(part.rstrip(' '))
            if trail: lx += text_width(' ' * trail, F_BODY, size, sx)
    return box

def body(pg, r, text, jobs, cap=None):
    """Lay out and queue one text block. cap limits the font size (one
    size for all numbered items, one for all labels of a page). Returns
    (log line, size)."""
    x0, y0, x1, y1 = r['area']
    if r['num']:                  # keep clear of the numbered circle
        x0 = max(x0, r['num'][0][0] + r['num'][0][2] + 4)
    small = r['kind'] == 'label'
    smax = 13.5 if small else 16.5
    smin = 10 if small else 11
    m = re.search(r'smax@([\d.]+)', r['notes'])
    if m: smax = float(m[1])
    indent = 0
    if text.startswith('{icons}'):
        text = text[len('{icons}'):].strip()
        indent = max(ix + iw for ix, iy, iw, ih in r['icon']) + 4 - x0
    pk = 1.3 if small else 1.45
    # vertical anchor: the first line sits on the numbered circle, or on
    # the icon / plate left of a label; other blocks are centred in their
    # area (the panel or panel row)
    anchor = r['num'][:1] or [b for b in r['icon'] + r['plate'] if b[0] < x0][:1]
    cy = None
    if anchor:
        ax, ay, aw, ah = anchor[0]
        cy = ay + (ah - (3 if r['num'] else 2)) / 2      # box includes the shadow
    top = smax
    while True:
        size, sx, lines, pitch, ok = fit_tok(text, r, x1 - x0, y1 - y0, top, smin, pk, indent,
                                             cap=cap or 99)
        if pitch * len(lines) > y1 - y0 + 4:
            pitch = max(size + 1, (y1 - y0) / len(lines))
        ty = cy - pitch / 2 if cy is not None else \
            y0 + max(0, (y1 - y0) - pitch * len(lines)) / 2
        if ty + pitch * len(lines) <= y1 + 3 or top <= smin: break
        top -= 0.5
    if ty + pitch * len(lines) > y1 + 3: ok = False
    y0 = ty
    if r['id'].endswith('_rule'):     # free paragraph: centred in its panel
        x0 = x0 + max(0, (x1 - x0) - max(seg_width(l, r, size, sx) for l in lines)) / 2
    fill = rgb_of(r['fill'])[0]
    sh = r['sh_col'] if r['sh_col'] is not None else np.array([16., 16, 16])
    res = {}
    jobs.append(lambda: res.setdefault('box', draw_tok(pg, r, lines, x0, y0, size, sx, pitch,
                                                       fill, sh, r['sh_off'], indent)))
    return f'{r["id"]}: size {size} sx {sx} lines {len(lines)}' + ('' if ok else ' OVERFLOW'), size

# ---------------------------------------------------------------- plates

def plate_text(text):
    size = 14
    a, base, _ = render(text, F_BODY, size)
    while a.shape[1] > 150 and size > 10:
        size -= 0.5; a, base, _ = render(text, F_BODY, size)
    return a, size

def grey_plate(pg, r, text, jobs, minw=0):
    """Grey label plate (flat fill, light/dark bevel at the bottom and
    right, 2 px drop shadow): erase it, draw it again at least as wide as
    the EN text needs, text centred."""
    img = pg.img; shp = img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    fill = rgb_of(r['fill'])[0]
    a, size = plate_text(text)
    pw_ = max(w, a.shape[1] + 14, minw)
    mk = rect_mask(shp, x - 1, y - 1, w + 4, h + 4)
    erase_fill(pg, mk, amount=0.8)
    tx = int(round(x + (pw_ - 2 - a.shape[1]) / 2)); ty = int(round(y + (h - 2 - a.shape[0]) / 2))
    def job():
        im = pg.img
        im[y + 2:y + h + 2, x + 2:x + pw_ + 2] = (40, 40, 40)
        im[y:y + h, x:x + pw_] = fill
        im[y + h - 2, x:x + pw_ - 1] = (184, 184, 184)
        im[y + h - 1, x:x + pw_] = (120, 120, 116)
        im[y:y + h - 1, x + pw_ - 2] = (176, 176, 176)
        im[y:y + h, x + pw_ - 1] = (136, 136, 136)
        over(im, tx + 1, ty + 1, a, np.array([40., 40, 40]))
        over(im, tx, ty, a, np.array([248., 248, 248]))
    jobs.append(job)
    return f'{r["id"]}: plate {w}->{pw_} size {size}'

# ---------------------------------------------------------------- page

def ellipse_mask(sh, x, y, w, h, grow=1):
    yy, xx = np.ogrid[0:sh[0], 0:sh[1]]
    cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
    return ((xx - cx) / (w / 2 + grow + 0.5)) ** 2 + ((yy - cy) / (h / 2 + grow + 0.5)) ** 2 <= 1

def paint_page(pg, rows, tl):
    """All rows of one rules page: erase first, then draw."""
    jobs = []; log = []
    shp = pg.img.shape[:2]
    for r in rows:
        t = tl.get(r['id'], '')
        r['moved'] = sorted({int(k) for k in TOK.findall(t)})
    is_text = lambda r: r['kind'] in ('body', 'label')
    # 1. body / label text, botan captions and moved icons: one mask
    mk = np.zeros(shp, bool); prot = np.zeros(shp, bool)
    for r in rows:
        if is_text(r):
            mk |= erase_body(pg, r)
            for k in r['moved']:
                mk |= ellipse_mask(shp, *r['icon'][k], 2)
            prot |= protect_mask(shp, r, 1)
    mk &= ~protect_mask_all(shp, rows, 0)
    erase_fill(pg, mk, ~prot, amount=0.9, ignore=prot)
    group = lambda r: ('c' if r['num'] else 'k' if r['kind'] == 'label' else
                       't' if r['id'].endswith('_t') else r['id'])
    cap = {}
    for r in rows:
        if is_text(r) and r['id'] in tl:
            g = group(r); z = body(pg, r, tl[r['id']], [])[1]
            cap[g] = min(cap.get(g, 99), z)
    for r in rows:
        t = tl.get(r['id'])
        if t is None:
            log.append(f'{r["id"]}: no EN'); continue
        if is_text(r): log.append(body(pg, r, t, jobs, cap[group(r)])[0])
        elif r['kind'] == 'plate':     # plates of a page share one width
            minw = max(q['w'] if q['id'] not in tl else max(q['w'], plate_text(tl[q['id']])[0].shape[1] + 14)
                       for q in rows if q['kind'] == 'plate')
            log.append(grey_plate(pg, r, t, jobs, minw))
        elif r['kind'] == 'header': log.append(header_band(pg, r, t, jobs))
        elif r['kind'] == 'heading': log.append(section_heading(pg, r, t, jobs))
        elif r['kind'] == 'back': log.append(back_hint(pg, r, t, jobs))
    if not os.environ.get("RULES_ERASE_ONLY"):
        for j in jobs: j()
    return log

_done = {}

def paint(d, n, tl=None, log=None, rows=None):
    """Paint the rules pages held by MINITTLn (bytearray d, plain). Returns
    {page id: (JP rgb, EN rgb)}."""
    tl = tl if tl is not None else load_tl()
    rows = rows if rows is not None else load_rows()
    src = COPY.get(n, n)
    out = {}
    for pid, off in PAGES.get(src, []):
        pg = Page(d, off)
        key = (pid, bytes(d[off:off + 0x220 + PW * PH]))
        if key in _done:          # practice copy: same bytes, same result
            d[off + 0x220:off + 0x220 + PW * PH] = _done[key][2]
            out[pid] = _done[key][:2]; continue
        prs = [r for r in rows.values() if r['page'] == pid]
        lines = paint_page(pg, prs, tl)
        if log is not None: log.extend(lines)
        after = pg.store(d)
        _done[key] = (pg.orig, after, bytes(d[off + 0x220:off + 0x220 + PW * PH]))
        out[pid] = (pg.orig, after)
    return out

def preview(pairs, path):
    """JP | EN at 1x, one page per row."""
    im = Image.new('RGB', (2 * PW + 8, len(pairs) * (PH + 4)), (0, 0, 0))
    for i, (a, b) in enumerate(pairs):
        im.paste(Image.fromarray(a.astype(np.uint8)), (0, i * (PH + 4)))
        im.paste(Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)), (PW + 8, i * (PH + 4)))
    im.save(path)

if __name__ == '__main__':
    only = [a for a in sys.argv[1:] if not a.startswith('--')] or list(PAGES)
    os.makedirs(PREV, exist_ok=True)
    tl = load_tl(); rows = load_rows()
    for n in only:
        d = bytearray(open(f'private/unpacked/MINITTL{n}.BIZ.lz', 'rb').read())
        log = []
        res = paint(d, n, tl, log, rows)
        for l in log: print(l)
        for pid, (a, b) in res.items():
            preview([(a, b)], f'{PREV}/{pid}.png')
