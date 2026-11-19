# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the month title cards (DAT2/MGBETA.TZZ) in English.

Reads tl/BETA.txt (JP:/EN: pairs). Writes private/build/MGBETA.TZZ.
--dry writes nothing to private/build.
--png DIR writes the JP and EN cards and a JP/EN contact sheet to DIR.
RE: private/re/beta/findings.org.

MGBETA.TZZ is an archive (TOC of (offset, size) at 0, entries 0x800
aligned) of 12 LZ entries, entry = month - 1. Each entry is an 8bpp TIM,
512x480 at VRAM (512,0), one 256-colour CLUT at (0,480). Entry 3 (April)
is a copy of entry 4 (May). The text is baked into the art: the build
removes it (tools/inpaint.py: the JP mask plus a 5 px rim, LaMa or the
OpenCV fallback, dithered back into the card's CLUT), draws the EN text
and quantises it to the card's CLUT (nearest colour).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import fontpath
import arc, lz, inpaint

SRC = 'private/disc/DAT2/MGBETA.TZZ'
OUT = 'private/build/MGBETA.TZZ'
TL = 'tl/BETA.txt'

# JP text rectangles (x, y, w, h) per entry, from private/re/beta/labels.tsv:
# the title first, then the event lines top to bottom. Each event is
# (month, day text, JP date key, JP event).
TITLE = {0: (43, 28, 180, 113), 1: (44, 34, 178, 113), 2: (44, 33, 182, 114),
         4: (44, 35, 179, 113), 5: (47, 34, 170, 112), 6: (53, 41, 169, 105),
         7: (44, 35, 192, 115), 8: (44, 34, 178, 112), 9: (33, 35, 274, 110),
         10: (33, 38, 260, 110), 11: (41, 42, 272, 106)}
EVENTS = {
    0: [((234, 367, 152, 39), '１日', '正月')],
    1: [((73, 368, 406, 38), '１４日', 'バレンタインデー')],
    2: [((146, 276, 308, 36), '６日', '学年末試験'),
        ((146, 316, 348, 38), '１４日', 'ホワイトデー'),
        ((146, 360, 278, 40), '２１日', '進級試験'),
        ((146, 403, 240, 41), '２４日', '終業式')],
    4: [((86, 365, 382, 42), '２４日', '１学期中間試験')],
    5: [((198, 372, 228, 42), '１２日', '太Ｐ戦')],
    6: [((91, 336, 376, 42), '１２日', '１学期期末試験'),
        ((89, 384, 234, 42), '３１日', 'プール')],
    7: [((86, 338, 384, 42), '１３～１４日', '臨海合宿'),
        ((196, 377, 234, 46), '２８日', '夏祭り')],
    8: [((203, 365, 232, 39), '２５日', '学園祭')],
    9: [((89, 365, 380, 42), '２５日', '２学期中間試験')],
    10: [((195, 364, 231, 42), '２７日', '五輪祭')],
    11: [((140, 338, 343, 42), '６日', '２学期期末試験'),
         ((107, 388, 301, 38), '２４日', 'クリスマス')],
}
FW = str.maketrans('０１２３４５６７８９', '0123456789')
def jp_month(e): return '１２３４５６７８９'[e] + '月' if e < 9 else '１' + '０１２'[e - 9] + '月'

# ---------------------------------------------------------------- TIM

def parse(d):
    """CLUT (256,) u16, pixel offset, pixel array (480, 512) u8."""
    assert struct.unpack_from('<II', d, 0) == (0x10, 9)
    cl = struct.unpack_from('<I', d, 8)[0]
    assert struct.unpack_from('<4H', d, 12) == (0, 480, 256, 1)
    clut = np.frombuffer(d[20:20 + 512], '<u2').copy()
    p = 8 + cl
    assert struct.unpack_from('<4H', d, p + 4) == (512, 0, 256, 480)
    px = np.frombuffer(d[p + 12:p + 12 + 512 * 480], np.uint8).reshape(480, 512).copy()
    return clut, p + 12, px

def clut_rgb(clut):
    c = clut.astype(int)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(float)

def hsv(rgb):
    r, g, b = (rgb[..., i] / 255.0 for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    dd = np.maximum(d, 1e-9)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6
    return np.where(d > 0, h, 0), s, mx

# ---------------------------------------------------------------- removal

def glyph_parts(rgb, rect, big):
    """Window, dark outline, glyph body (outline + fill) and removal mask of
    the JP text (fill, dark outline, colour shadow) in rect."""
    H, S, V = hsv(rgb)
    sel = S > 0.3
    hm = np.median(H[sel])
    dh = np.abs(H - hm); dh = np.minimum(dh, 1 - dh)
    x, y, w, h = rect
    m = 9 if big else 5
    win = (slice(max(0, y - m), y + h + m), slice(max(0, x - m), x + w + m))
    dark = (V < 0.36)[win]
    lab, n = ndimage.label(dark, np.ones((3, 3)))
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    dark = np.isin(lab, 1 + np.nonzero(sizes >= (40 if big else 12))[0])
    distinct = (((dh > 0.07) & (S > 0.3)) | ((S < 0.2) & (V > 0.86)))[win]
    # local contrast against a background estimate that ignores candidates
    a = rgb[win]
    keep = ~(dark | distinct)
    sg = 4
    num = ndimage.gaussian_filter(a * keep[..., None], (sg, sg, 0))
    den = ndimage.gaussian_filter(keep.astype(float), sg)[..., None]
    bg = num / np.maximum(den, 1e-3)
    diff = np.sqrt(((a - bg) ** 2).sum(-1)) > 60
    cand = dark | distinct | diff
    dist = ndimage.distance_transform_edt(~dark)
    mask = dark | (cand & (dist <= (13 if big else 6)))
    # glyph body: regions enclosed by the outline whose pixels are mostly
    # one flat colour (the fill); textured enclosed regions are counters
    idx = rgb[win] @ np.array([1, 256, 65536])
    ring = ndimage.binary_closing(dark, np.ones((3, 3)))
    inner = ndimage.binary_fill_holes(ring) & ~ring
    lab, n = ndimage.label(inner)
    body = ring.copy()
    for k in range(1, n + 1):
        c = lab == k
        v, cnt = np.unique(idx[c], return_counts=True)
        if cnt.max() >= 0.5 * c.sum(): body |= c
    # colour shadow: the body swept down-right by up to 7 (3) px
    sh = 7 if big else 3
    sweep = body.copy()
    for d in range(1, sh + 1):
        for dx, dy in ((d, d), (d, d - 1), (d - 1, d)):
            sweep[dy:, dx:] |= body[:body.shape[0] - dy, :body.shape[1] - dx]
    mask |= sweep
    mask = ndimage.binary_closing(mask, np.ones((3, 3)))
    mask = ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=2 if big else 1)
    return win, dark, body, mask

def text_mask(rgb, rect, big):
    win, _, _, mask = glyph_parts(rgb, rect, big)
    full = np.zeros(rgb.shape[:2], bool)
    full[win] = mask
    return full

def shadow_ramp(rgb, rect, big):
    """JP colour shadow sampled per column: (x0, x1, (n, 3) colours) over the
    shadow's x extent, smoothed, gaps interpolated."""
    win, dark, body, _ = glyph_parts(rgb, rect, big)
    dx, dy = SH_TITLE if big else SH_EVENT
    sb = np.zeros_like(body)
    sb[dy:, dx:] = body[:body.shape[0] - dy, :body.shape[1] - dx]
    a = rgb[win]
    ex = ndimage.binary_dilation(body, np.ones((3, 3)), iterations=8 if big else 4)
    num = ndimage.gaussian_filter(a * (~ex)[..., None], (8, 8, 0))
    den = ndimage.gaussian_filter((~ex).astype(float), 8)[..., None]
    bg = num / np.maximum(den, 1e-3)
    _, S, V = hsv(a)
    shp = sb & ~body & (np.sqrt(((a - bg) ** 2).sum(-1)) > 50) & (S > 0.4) & (V > 0.5)
    ys, xs = np.nonzero(shp)
    x0, x1 = xs.min(), xs.max()
    cols = np.full((x1 - x0 + 1, 3), np.nan)
    for x in range(x0, x1 + 1):
        c = a[ys[xs == x], x]
        if len(c) >= 2: cols[x - x0] = np.median(c, 0)
    ok = ~np.isnan(cols[:, 0])
    t = np.arange(len(cols))
    cols = np.stack([np.interp(t, t[ok], cols[ok, i]) for i in range(3)], -1)
    k = 15 if big else 9
    cols = ndimage.median_filter(cols, (k, 1), mode='nearest')
    cols = ndimage.uniform_filter1d(cols, k, 0, mode='nearest')
    off = win[1].start
    return x0 + off, x1 + off, cols

RING = 5        # text_mask rim (px): takes out the JP glyphs' soft shadow
ISLAND = 30     # unmasked islands under this size (px) are filled too

def removal_mask(rgb, e):
    """Fill mask of card e: the script's own JP mask per rect (outline,
    fill, colour shadow) plus a RING px rim of pixels that differ from the
    background diffused in around it."""
    rects = [(TITLE[e], True)] + [(r, False) for r, _, _ in EVENTS[e]]
    full = np.zeros(rgb.shape[:2], bool)
    for r, big in rects:
        m = text_mask(rgb, r, big)
        x0, y0, x1, y1 = inpaint._bbox(m, RING, rgb.shape)
        full |= inpaint.text_mask(rgb, (x0, y0, x1, y1), pred=lambda a, m=m: m,
                                  ring=RING, halo=1, min_px=0, close=0)
    # small unmasked islands inside the mask (dark texture specks in a
    # glyph counter) make LaMa grow a dark smudge from them (8月)
    holes = ndimage.binary_fill_holes(full) & ~full
    lab, n = ndimage.label(holes)
    if n:
        sz = ndimage.sum(holes, lab, np.arange(1, n + 1))
        full |= np.isin(lab, 1 + np.nonzero(sz < ISLAND)[0])
    return full

def plate(px, clut, pal, e, method='best'):
    """Card e with the JP text removed (CLUT indices) and the fill mask.
    fill(): LaMa if the model and torch are there, else the OpenCV 'auto'
    fill; written back into the card's CLUT, dithered inside the mask."""
    rgb = pal[px]
    mask = removal_mask(rgb, e)
    out = inpaint.fill(rgb, mask, method)
    return inpaint.to_clut(out, px, clut, mask, pal=pal), mask

# ---------------------------------------------------------------- EN text

FONT_TITLE = 'MPLUSRounded1c-ExtraBold.ttf'
FONT_EVENT = 'MPLUSRounded1c-Bold.ttf'
DARK = (48, 48, 48)                     # outline colour (CLUT index 0)
WHITE = (248, 248, 248)                 # event name fill
SH_TITLE, SH_EVENT = (4, 5), (2, 2)     # colour shadow offsets of the JP
OL_TITLE, OL_EVENT = 2.0, 1.6           # outline radius
# Fill colours of the JP title and date digits per entry.
TITLE_FILL = {0: (120, 248, 200), 1: (248, 160, 208), 2: (248, 248, 168),
              4: (0, 248, 248), 5: (96, 248, 168), 6: (248, 144, 0),
              7: (0, 232, 248), 8: (248, 0, 152), 9: (248, 232, 0),
              10: (248, 216, 128), 11: (248, 200, 152)}
DATE_FILL = {0: (248, 232, 0), 1: (224, 248, 0), 2: (104, 248, 152),
             4: (224, 248, 120), 5: (248, 240, 136), 6: (0, 248, 208),
             7: (200, 248, 32), 8: (40, 232, 248), 9: (80, 216, 248),
             10: (192, 112, 216), 11: (248, 208, 248)}
SAFE = (24, 488)                        # x range kept clear of overscan
TITLE_X, TITLE_MID, TITLE_MAXW, TITLE_CAP = 40, 86, 410, 64
# Event lines: one axis for every card, the centre of the long JP lines
# (the short JP lines sit further right, at x ~310).
EVENT_AXIS = 280
EVENT_CAP, EVENT_GAP = 23, 16
SS = 4

def cap_size(font, cap):
    """Font size (at SS) whose 'H' is cap px tall."""
    f = fontpath.truetype(font, 100 * SS)
    h = -f.getbbox('H', anchor='ls')[1]
    return max(1, int(round(100 * SS * cap * SS / h)))

def ink(text, font, cap, track=0.0, sx=1.0):
    """High-res coverage (SS x) of one line, baseline row (high-res) and
    the left offset; sx squeezes horizontally."""
    f = fontpath.truetype(font, cap_size(font, cap))
    l, t, r, b = f.getbbox(text, anchor='ls')
    pad = 12 * SS
    W = int(r - l + 2 * pad + track * SS * len(text))
    H = int(b - t + 2 * pad)
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    x = pad - l
    for ch in text:
        dr.text((x, pad - t), ch, font=f, fill=255, anchor='ls')
        x += f.getlength(ch) + track * SS
    a = np.asarray(im, float) / 255
    if sx != 1.0:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(1, int(W * sx)), H), Image.LANCZOS), float) / 255
    return a, pad - t

def layers(a, base, ol, sh):
    """Downsampled fill, silhouette (fill + outline) and shadow coverage of
    hi-res ink a; the result is cropped to its ink box. Returns the three
    arrays, the baseline row and the x of column 0 relative to the ink."""
    g = a > 0.5
    d = ndimage.distance_transform_edt(~g)
    sil = np.maximum(a, np.clip(ol * SS + 0.5 - d, 0, 1))
    dx, dy = sh[0] * SS, sh[1] * SS
    shd = np.zeros_like(sil)
    shd[dy:, dx:] = sil[:sil.shape[0] - dy, :sil.shape[1] - dx]
    ys, xs = np.nonzero(np.maximum(sil, shd) > 0.01)
    y0, x0 = ys.min() // SS * SS, xs.min() // SS * SS
    y1, x1 = -(-(ys.max() + 1) // SS) * SS, -(-(xs.max() + 1) // SS) * SS
    def down(z):
        z = z[y0:y1, x0:x1]
        return z.reshape(z.shape[0] // SS, SS, z.shape[1] // SS, SS).mean((1, 3))
    return down(a), down(sil), down(shd), (base - y0) // SS

def ramp_at(ramp, w):
    """Colours of a JP shadow ramp (x0, x1, cols) stretched over w columns."""
    jx0, jx1, cols = ramp
    t = np.linspace(0, len(cols) - 1, w)
    return np.stack([np.interp(t, np.arange(len(cols)), cols[:, i]) for i in range(3)], -1)

class Piece:
    """One drawn line: coverages, baseline, width; colours set later."""
    def __init__(self, text, font, cap, ol, sh, track=0.0, sx=1.0):
        a, base = ink(text, font, cap, track, sx)
        self.fill, self.sil, self.shd, self.base = layers(a, base, ol, sh)
        self.h, self.w = self.fill.shape
        g = np.nonzero(self.fill.max(0) > 0.05)[0]
        self.gx0, self.gx1 = g[0], g[-1] + 1     # glyph ink columns

def paint(rgb, p, x, y, fill, shcols, cover):
    """Composite piece p with ink column 0 at x and baseline at y."""
    y0 = y - p.base
    sl = (slice(y0, y0 + p.h), slice(x, x + p.w))
    assert y0 >= 0 and x >= 0 and y0 + p.h <= rgb.shape[0] and x + p.w <= rgb.shape[1], (x, y0)
    r = rgb[sl]
    sh = p.shd[..., None]; si = p.sil[..., None]; fi = p.fill[..., None]
    r[:] = r * (1 - sh) + shcols[None, :, :] * sh
    r[:] = r * (1 - si) + np.array(DARK, float) * si
    r[:] = r * (1 - fi) + np.array(fill, float) * fi
    cover[sl] |= np.maximum(p.sil, p.shd) > 0.02

def load_tl(path=TL):
    out, jp = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            out[jp] = line[4:].strip(); jp = None
    return out

def fit(text, font, cap, ol, sh, maxw, track, mincap):
    """Piece at most maxw wide: lower the cap height down to mincap, then
    squeeze horizontally."""
    while True:
        p = Piece(text, font, cap, ol, sh, track)
        if p.w <= maxw: return p, cap, 1.0
        if cap <= mincap: break
        cap -= 1
    sx = 1.0
    while p.w > maxw:
        sx -= 0.01
        p = Piece(text, font, cap, ol, sh, track, sx)
    return p, cap, sx

def quantise(rgb, pal, cover, out):
    """Nearest CLUT index for covered pixels (weighted RGB distance)."""
    ys, xs = np.nonzero(cover)
    c = rgb[ys, xs]
    d = (((c[:, None, :] - pal[None]) ** 2) * (3, 4, 2)).sum(-1)
    out[ys, xs] = d.argmin(1)
    return out

def draw_card(e, px, clut, pal, tl, log, method='best'):
    """EN pixel indices for card e."""
    jp = pal[px]
    base, _ = plate(px, clut, pal, e, method)
    rgb = pal[base].copy()
    cover = np.zeros(px.shape, bool)
    # title: left column of the JP titles, cap centred on the JP band
    name = tl[jp_month(e)]
    p, cap, sx = fit(name, FONT_TITLE, TITLE_CAP, OL_TITLE, SH_TITLE,
                     TITLE_MAXW, 2.0, 50)
    paint(rgb, p, TITLE_X, TITLE_MID + cap // 2, TITLE_FILL[e],
          ramp_at(shadow_ramp(jp, TITLE[e], True), p.w), cover)
    log.append(f'e{e} title {name!r} cap {cap} sx {sx:.2f} x {TITLE_X}-{TITLE_X + p.w}')
    title_y1 = TITLE_MID + cap // 2 - p.base + p.h
    assert TITLE_X + p.w <= SAFE[1], name
    # events: a date column and an event column, centred as one block on
    # EVENT_AXIS, rows at the JP rows
    ev = EVENTS[e]
    axis = EVENT_AXIS
    m = jp_month(e)
    texts = [(tl[m + day], tl[name_]) for _, day, name_ in ev]
    cap, sx = EVENT_CAP, 1.0
    while True:
        dps = [Piece(d, FONT_EVENT, cap, OL_EVENT, SH_EVENT, 1.0, sx) for d, _ in texts]
        eps = [Piece(n, FONT_EVENT, cap, OL_EVENT, SH_EVENT, 1.0, sx) for _, n in texts]
        dcol = max(q.w for q in dps)
        bw = dcol + EVENT_GAP + max(q.w for q in eps)
        if bw <= SAFE[1] - SAFE[0]: break
        if cap > 20: cap -= 1
        else: sx -= 0.02
    x0 = int(round(min(max(axis - bw / 2, SAFE[0]), SAFE[1] - bw)))
    prev = title_y1
    for (r, _, _), dp, ep, (dt, nt) in zip(ev, dps, eps, texts):
        y = r[1] + 19 + cap // 2
        ex = x0 + dcol + EVENT_GAP
        cols = ramp_at(shadow_ramp(jp, r, False), ex + ep.w - x0)
        paint(rgb, dp, x0, y, DATE_FILL[e], cols[:dp.w], cover)
        paint(rgb, ep, ex, y, WHITE, cols[ex - x0:ex - x0 + ep.w], cover)
        log.append(f'e{e} line {dt!r} {nt!r} cap {cap} sx {sx:.2f} x {x0}-{ex + ep.w} y {y - dp.base}-{y - dp.base + dp.h}')
        assert SAFE[0] <= x0 and ex + ep.w <= SAFE[1], (e, dt)
        top = y - max(dp.base, ep.base)
        assert top >= prev, (e, dt, top, prev)          # no overlap with the line above
        prev = y - max(dp.base, ep.base) + max(dp.h, ep.h)
        assert prev <= 470, (e, dt)
    return quantise(rgb, pal, cover, base.copy())

# ---------------------------------------------------------------- build

def build(pngdir=None, write=True):
    D = open(SRC, 'rb').read()
    toc = arc.toc(D)
    tl = load_tl()
    log, cards, jp_cards, data = [], {}, {}, {}
    for e in range(12):
        if e == 3: continue
        o, s = toc[e]
        d = bytearray(lz.decompress(D[o:o + s])[0])
        clut, off, px = parse(bytes(d))
        pal = clut_rgb(clut)
        new = draw_card(e, px, clut, pal, tl, log)
        d[off:off + new.size] = new.tobytes()
        data[e] = bytes(d)
        cards[e], jp_cards[e] = pal[new], pal[px]
    data[3] = data[4]
    # archive: keep the JP offsets when every entry fits its slot
    comp = {e: lz.compress(data[e], name='MGBETA.TZZ') for e in range(12) if e != 3}
    comp[3] = comp[4]
    for e in range(12):
        assert lz.decompress(comp[e])[0] == data[e], e
    ends = [toc[e + 1][0] for e in range(11)] + [len(D)]
    room = [ends[e] - toc[e][0] - len(comp[e]) for e in range(12)]
    out = bytearray(len(D))
    if min(room) >= 0:
        offs = [o for o, _ in toc]
    else:
        offs, pos = [], 0x800
        for e in range(12):
            offs.append(pos); pos += -(-len(comp[e]) // 0x800) * 0x800
        assert pos <= len(D), f'archive {pos:#x} > {len(D):#x}'
        log.append('relaid archive: ' + ' '.join(f'{o:#x}' for o in offs))
    for e in range(12):
        struct.pack_into('<II', out, 8 * e, offs[e], len(comp[e]))
        out[offs[e]:offs[e] + len(comp[e])] = comp[e]
    log.append('room ' + ' '.join(str(r) for r in room))
    print('\n'.join(log))
    if write:
        os.makedirs('private/build', exist_ok=True)
        open(OUT, 'wb').write(out)
        print('wrote', OUT, len(out))
    if pngdir: previews(jp_cards, cards, pngdir)

def previews(jp, en, out):
    os.makedirs(out, exist_ok=True)
    for e in en:
        Image.fromarray(en[e].astype(np.uint8)).save(f'{out}/en_{e:02d}.png')
    # contact sheet: JP | EN pairs, two pairs per row, half size
    ks = sorted(en)
    tw, th = 256, 240
    W = Image.new('RGB', (4 * tw + 12, -(-len(ks) // 2) * (th + 4)), (0, 0, 0))
    for i, e in enumerate(ks):
        for j, a in enumerate((jp[e], en[e])):
            im = Image.fromarray(a.astype(np.uint8)).resize((tw, th), Image.LANCZOS)
            W.paste(im, ((i % 2) * (2 * tw + 12) + j * tw, (i // 2) * (th + 4)))
    W.save(f'{out}/contact.png')
    # the same with layout guides on the EN halves: safe area (red) and the
    # event axis (white)
    dr = ImageDraw.Draw(W)
    for i, e in enumerate(ks):
        ox, oy = (i % 2) * (2 * tw + 12) + tw, (i // 2) * (th + 4)
        for x, c in ((SAFE[0], (255, 0, 0)), (SAFE[1], (255, 0, 0)), (EVENT_AXIS, (255, 255, 255))):
            for y in range(0, th, 4):
                dr.point((ox + x // 2, oy + y), fill=c)
    W.save(f'{out}/contact_guides.png')

if __name__ == '__main__':
    png = sys.argv[sys.argv.index('--png') + 1] if '--png' in sys.argv else None
    build(png, write='--dry' not in sys.argv)
