# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the EDIT command-list cards (DAT2/EDITCM.BIZ) in English.

Port of the first game's build_editcm.py (topic editchar). Reads
tl/EDITCM.txt (ID: EN). Writes private/build/EDITCM.BIZ and
private/build/KANRI.BIN (new Team-Up name records). --png writes previews
to private/re/editchar/en/. --check builds everything but writes nothing.

EDITCM.BIZ is an archive of 32 LZ entries, one per EDIT style (entry k =
style k + 1, tl id cm{k+1:02d}), 0x2b800 bytes unpacked each:
- TIM1 at 0x0: 4bpp 412x369, one 16-colour CLUT at VRAM (0,485), pixels at
  VRAM (512,0). Headings left, move names middle, commands right.
- TIM2 at 0x13000: 4bpp 512x256 at VRAM (640,256), CLUT (0,484). The 23
  Team-Up names; identical in all 32 entries.
Entries 0-23 are the first game's EDITCM01-24 cards; 11 are identical in
RGB, 13 have changed rows, 8 are new (private/re/editchar/rows.txt). A
card is redrawn only when tl/EDITCM.txt has every row of it; otherwise
only its Team-Up sheet changes and the card stays JP.
KANRI.BIN (load 0x800a2000) draws the card at screen y 64 with 4 sprites
(records at file 0xf78) and hides unlearned moves with rectangles whose
tops come from per-style tables at file 0x1264 (special moves) and 0x13e4
(Burning Vigor), 32 styles x two (x, y, h). Team-Up name: 23 records of
20 bytes at file 0x1038 (flag, tpage x, tpage y, clut y, u, v, w, h, x,
y), chosen through 24 pointers at 0x1204 by the Team-Up id.
"""
import sys, os, struct, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz, arc

ARC = 'private/disc/DAT2/EDITCM.BIZ'
KANRI = 'private/disc/DAT1/KANRI.BIN'
NCARD = 32
SP_TAB, BV_TAB = 0x1264, 0x13e4          # cover rect tables (KANRI file)
TU_REC, NTU = 0x1038, 23                 # Team-Up name records
FONT = 'DejaVuSansCondensed-Bold.ttf'
FONT_H = 'DejaVuSansCondensed-Bold.ttf'
W, H = 412, 369
TIM2 = 0x13000
SS = 4                                   # supersampling for text

def load_tl(path='tl/EDITCM.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb15(v): return [(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3]

# ---------------------------------------------------------------- text
def render(text, size, font=FONT, sx=1.0):
    """Alpha (float, 0..1) of one line, cropped to the ink, horizontally
    scaled by sx. Height is the font's full ascent+descent box."""
    f = fontpath.truetype(font, size * SS)
    asc, desc = f.getmetrics()
    b = f.getbbox(text)
    im = Image.new('L', (b[2] - b[0] + 4 * SS, asc + desc), 0)
    ImageDraw.Draw(im).text((2 * SS - b[0], 0), text, font=f, fill=255)
    w = max(1, int(round(im.width * sx / SS))); h = max(1, int(round(im.height / SS)))
    a = np.asarray(im.resize((w, h), Image.LANCZOS), float) / 255
    cols = np.where(a.max(0) > 0.05)[0]
    return a[:, cols[0]:cols[-1] + 1] if len(cols) else a

def ink_rows(a):
    r = np.where(a.max(1) > 0.05)[0]
    return a[r[0]:r[-1] + 1]

def fit_lines(text, maxw, maxh, size, minsize, font=FONT, minsx=0.78, lead=0):
    """Lines of alpha arrays that fit maxw x maxh: one line if possible
    (shrinking, then condensing), else two lines (forced with '|')."""
    def try_lines(lines):
        for s in range(size, minsize - 1, -1):
            for sx in (1.0, 0.94, 0.88, 0.83, minsx):
                al = [ink_rows(render(l, s, font, sx)) if l else None for l in lines]
                # line pitch from the font size, ink of the tallest line
                pitch = int(round(s * 0.82)) + lead
                tot = pitch * (len(lines) - 1) + max(a.shape[0] for a in al)
                if max(a.shape[1] for a in al) <= maxw and tot <= maxh:
                    return al, pitch, s - (1.0 - sx) * 10
        return None
    if '|' in text:
        return try_lines(text.split('|'))
    one = try_lines([text])
    words = text.split(' ')
    best = None
    for k in range(1, len(words)):
        l1, l2 = ' '.join(words[:k]), ' '.join(words[k:])
        r = try_lines([l1, l2])
        if r and (best is None or r[2] > best[2] or r[2] == best[2] and
                  max(a.shape[1] for a in r[0]) < max(a.shape[1] for a in best[0])):
            best = r
    if one and (best is None or one[2] >= min(14, best[2] - 1)):
        return one
    return best

def stamp(lines, pitch, cx, cy, top_ink=None):
    """Composite placement: returns (alpha canvas, x0, y0) centred on cx, cy.
    Each line is centred horizontally; baselines are pitch apart (aligned on
    the ink bottoms of letters without descenders is close enough)."""
    wmax = max(a.shape[1] for a in lines)
    hs = [a.shape[0] for a in lines]
    tot = pitch * (len(lines) - 1) + hs[-1]
    can = np.zeros((tot + 2, wmax + 2))
    for i, a in enumerate(lines):
        y = i * pitch; x = (wmax - a.shape[1]) // 2
        can[1 + y:1 + y + a.shape[0], 1 + x:1 + x + a.shape[1]] = np.maximum(
            can[1 + y:1 + y + a.shape[0], 1 + x:1 + x + a.shape[1]], a)
    rows = np.where(can.max(1) > 0.05)[0]
    can = can[rows[0]:rows[-1] + 1]
    x0 = int(round(cx - can.shape[1] / 2)); y0 = int(round(cy - can.shape[0] / 2))
    return can, x0, y0

def blend_quant(img_idx, pal, can, x0, y0, fg, cands=None, thr=0.04):
    """Draw alpha `can` in colour fg over the indexed image, choosing for
    each pixel the palette entry nearest to the blend. cands limits the
    entries considered (default: all)."""
    h, w = can.shape
    sub = img_idx[y0:y0 + h, x0:x0 + w]
    bg = pal[sub].astype(float)
    a = can[:sub.shape[0], :sub.shape[1], None]
    col = bg * (1 - a) + np.array(fg, float) * a
    cl = np.arange(len(pal)) if cands is None else np.array(cands)
    d = ((col[..., None, :] - pal[cl][None, None].astype(float)) ** 2).sum(-1)
    new = cl[d.argmin(-1)]
    m = a[..., 0] > thr
    sub[m] = new[m]

def segment_cands(pal, c1, c2, tol=48):
    """Palette entries near the colour segment c1-c2 (antialias tones)."""
    c1, c2 = np.array(c1, float), np.array(c2, float)
    out = []
    for i, p in enumerate(pal.astype(float)):
        t = np.clip(np.dot(p - c1, c2 - c1) / max(1, np.dot(c2 - c1, c2 - c1)), 0, 1)
        if np.linalg.norm(p - (c1 + t * (c2 - c1))) < tol: out.append(i)
    return out

def nearest(pal, c): return int(((pal.astype(int) - np.array(c)) ** 2).sum(1).argmin())

# ---------------------------------------------------------------- card
def load_card(d):
    clut = struct.unpack_from('<16H', d, 0x14)
    po = 8 + struct.unpack_from('<I', d, 8)[0]
    _, x, y, w, h = struct.unpack_from('<IHHHH', d, po)
    raw = np.frombuffer(bytes(d[po + 12:po + 12 + w * 2 * h]), np.uint8).reshape(h, w * 2)
    px = np.empty((h, w * 4), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    pal = np.array([rgb15(v) for v in clut], np.uint8)
    return px, pal, po + 12, w

def store_card(d, px, off, w):
    p = px.copy()
    if p.shape[1] % 2: p = np.pad(p, ((0, 0), (0, 1)))
    d[off:off + w * 2 * H] = (p[:, 0::2] | (p[:, 1::2] << 4)).astype(np.uint8).tobytes()

def runs(v):
    out = []; s = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]: out.append((s, i - 1, v[s])); s = i
    return out

def geometry(px, grey):
    """Cell edges from the grey separator lines (columns / rows that are
    grey over most of a section)."""
    g = {}
    def lines(cnt, thr):
        idx = [i for i in range(len(cnt)) if cnt[i] > thr]
        return [i for i in idx if i - 1 not in idx]
    ys = lines((px[:, 2:100] == grey).sum(1), 80)
    # ys: 0, 159, 272 (BVA heading bottom), 319, 367
    assert len(ys) == 5, ys
    g['y_sp'] = (2, ys[1])                      # special cells
    g['y_bv'] = (ys[1] + 2, ys[3])              # BVA row cells
    g['y_bvh'] = (ys[1] + 2, ys[2])             # BVA heading cell
    g['y_easy'] = (ys[2] + 2, ys[3])            # Easy Command (red)
    g['y_tu'] = (ys[3] + 2, ys[4])              # Team-Up row
    xs = lines((px[ys[1] + 2:ys[3], :] == grey).sum(0), 100)
    # xs: 0, 102, level column right edge, 282, 409
    assert len(xs) == 5, xs
    g['x_mid'] = (104, xs[3])
    g['x_bv'] = (xs[2] + 2, xs[3])
    g['x_lvl'] = (104, xs[2])
    g['x_cmd'] = (xs[3] + 2, xs[4])
    return g

def bands(mask, gap=3, minh=4):
    on = np.where(mask.any(1))[0]
    if not len(on): return []
    out = [[on[0], on[0]]]
    for k in on[1:]:
        if k - out[-1][1] <= gap: out[-1][1] = k
        else: out.append([k, k])
    return [(a, b + 1) for a, b in out if b + 1 - a >= minh]

def masks_for(style):
    d = open(KANRI, 'rb').read()
    sp = [struct.unpack_from('<3H', d, SP_TAB + ((style - 1) * 6 + k * 3) * 2) for k in (0, 1)]
    bv = [struct.unpack_from('<3H', d, BV_TAB + ((style - 1) * 6 + k * 3) * 2) for k in (0, 1)]
    # tops and exclusive bottoms of the cover rects in card rows (the card
    # sprite sits at screen y 64). KANRI draws them at y - 3 (special
    # moves, 0x8009fce0) and y - 2 (Burning Vigor, 0x8009ff84), h rows.
    return ((sorted({y - 3 - 64 for _, y, _ in sp}), sorted({y - 3 - 64 + h for _, y, h in sp})),
            (sorted({y - 2 - 64 for _, y, _ in bv}), sorted({y - 2 - 64 + h for _, y, h in bv})))

def clean_gradient(px, pal, x0, x1, y0, y1, tx0, tx1, ty0, ty1):
    """Paint the heading text out of a vertical-gradient cell. Each row is
    nearly uniform, so every pixel of the text box gets the pixel of the
    same row in a clean 8-pixel window (no dark text pixels nearby),
    repeated with period 8: two LZ words, which keeps the stream small.
    Rows without a clean window reuse the previous row's window."""
    lum = pal[px].astype(int).sum(-1)
    dirty = ndimage.binary_dilation(lum < 200, iterations=1)
    wins = list(range(x0 + (-x0) % 4, x1 - 7, 4))
    last = None
    for y in range(ty0, ty1):
        cand = [w for w in wins if not dirty[y, w:w + 8].any()]
        if cand:
            w = min(cand, key=lambda w: min(abs(w - tx0), abs(w + 8 - tx1)))
            pat = px[y, w:w + 8].copy(); off = w
        elif last is not None:
            pat, off = last
        else:
            continue
        last = (pat, off)
        for x in range(tx0, tx1):
            px[y, x] = pat[(x - off) % 8]

def heading(px, pal, cell, text, fg, size, minsize, shadow=None, find='dark', rng=None, flat=None):
    (x0, x1), (y0, y1) = cell
    sub = px[y0:y1, x0:x1].astype(int)
    lum = pal[sub].astype(int).sum(-1)
    if find == 'dark': tm = lum < 200
    else: tm = lum > 600
    ys, xs = np.where(tm)
    if len(ys):
        tx0, tx1 = x0 + max(0, xs.min() - 2), x0 + min(x1 - x0, xs.max() + 3)
        ty0, ty1 = y0 + max(0, ys.min() - 2), y0 + min(y1 - y0, ys.max() + 3)
        if flat is not None:
            px[ty0:ty1, tx0:tx1] = flat
        else:
            clean_gradient(px, pal, x0, x1, y0, y1, tx0, tx1, ty0, ty1)
    r = fit_lines(text, x1 - x0 - 6, y1 - y0 - 4, size, minsize, FONT_H, lead=2)
    if not r: return f'heading {text!r} does not fit'
    lines, pitch = r[0], r[1]
    can, cx0, cy0 = stamp(lines, pitch, (x0 + x1) / 2, (y0 + y1) / 2)
    if shadow is not None:
        sh = np.zeros_like(can); sh[1:, 1:] = can[:-1, :-1]
        sh = np.maximum(sh - can, 0)
        blend_quant(px, pal, sh, cx0, cy0, shadow, thr=0.3)
    blend_quant(px, pal, can, cx0, cy0, fg, thr=0.25)
    return None

def annotations(px, pal, x0, x1, y0, y1, red, white):
    """Red note clusters in a command row: red components that do not touch
    the icons (non-white, non-red pixels), grouped by gaps < 4 px."""
    sub = px[y0:y1, x0:x1]
    col = pal[sub].astype(int)
    redish = (col[..., 0] > 200) & (col[..., 1] < 200) & (col[..., 2] < 200)
    # icons (joystick dots, '+', fist/foot outlines) have near-black pixels;
    # the red arrows sit within an icon's columns, the notes outside them
    dark = col.sum(-1) < 150
    dc = np.where(dark.any(0))[0]
    iconcols = np.zeros(sub.shape[1], bool)
    if len(dc):
        s0 = p0 = dc[0]
        for c in list(dc[1:]) + [None]:
            if c is None or c - p0 > 3:
                iconcols[max(0, s0 - 4):p0 + 5] = True
                if c is not None: s0 = c
            if c is not None: p0 = c
    keep = redish & ~iconcols[None, :]
    # whole components that lie mostly outside the icon columns
    lab, n = ndimage.label(redish, structure=np.ones((3, 3)))
    for k in range(1, n + 1):
        m = lab == k
        if (m & keep).sum() * 2 >= m.sum(): keep |= m
    # an ellipse badge contains white text: fill holes so it is one blob
    cols = np.where(keep.any(0))[0]
    clusters = []
    if len(cols):
        s = cols[0]; p = cols[0]
        for c in list(cols[1:]) + [None]:
            if c is None or c - p > 4:
                m = keep[:, s:p + 1]
                ys = np.where(m.any(1))[0]
                box = (x0 + s, y0 + ys.min(), x0 + p + 1, y0 + ys.max() + 1)
                filled = ndimage.binary_fill_holes(m)
                badge = (p + 1 - s) >= 20 and m.sum() > 0.45 * filled.sum() and \
                    filled.sum() > 0.6 * (p + 1 - s) * (ys.max() + 1 - ys.min())
                clusters.append((box, badge, iconcols))
                if c is not None: s = c
            if c is not None: p = c
    return clusters

def clear_note(px, pal, box, white, icols, xoff):
    """White out a note: its box grown by 1 px, except icon columns (where
    only the red pixels go)."""
    bx0, by0, bx1, by1 = box
    for y in range(by0 - 1, by1 + 1):
        for x in range(bx0 - 1, bx1 + 1):
            c = pal[px[y, x]].astype(int)
            red = c[0] > 200 and c[1] < 200 and c[2] < 200
            if red or not icols[x - xoff]: px[y, x] = white

def draw_note(px, pal, box, text, red, white, badge, room=None, icols=None, xoff=0):
    bx0, by0, bx1, by1 = box
    clear_note(px, pal, box, white, icols, xoff)
    cy = (by0 + by1) / 2
    if badge:
        # redraw the ellipse and put white text in it
        w, h = bx1 - bx0, by1 - by0
        yy, xx = np.mgrid[0:h, 0:w]
        e = ((xx + 0.5 - w / 2) / (w / 2)) ** 2 + ((yy + 0.5 - h / 2) / (h / 2)) ** 2 <= 1.0
        px[by0:by1, bx0:bx1][e] = red
        r = fit_lines(text, w - 6, h - 2, 11, 7)
        if not r: return f'badge {text!r} does not fit'
        can, x, y = stamp(r[0], r[1], (bx0 + bx1) / 2, cy)
        blend_quant(px, pal, can, x, y, pal[white], cands=[red, white], thr=0.45)
        return None
    lo, hi = room if room else (bx0, bx1)
    r = fit_lines(text, hi - lo, max(10, by1 - by0 + 2), 11, 7, minsx=0.7)
    if not r: return f'note {text!r} does not fit {hi - lo}px'
    can, x, y = stamp(r[0], r[1], (bx0 + bx1) / 2, cy)
    x = min(max(x, lo), hi - can.shape[1])
    cands = segment_cands(pal, pal[white], pal[red])
    blend_quant(px, pal, can, x, y, pal[red], cands=cands, thr=0.2)
    return None

def pending(n, tl, u):
    """Rows of card n that have no EN in the tl (None if complete)."""
    px, pal, off, tw = load_card(u)
    px = px[:, :W]
    grey = nearest(pal, (112, 112, 112)); white = nearest(pal, (248, 248, 248))
    g = geometry(px, grey)
    miss = []
    for kind, (xa, xb), (ya, yb) in (('s', g['x_mid'], g['y_sp']), ('b', g['x_bv'], g['y_bv'])):
        for i in range(len(bands(px[ya:yb, xa:xb] != white))):
            key = f'cm{n:02d}.{kind}{i + 1}'
            if not tl.get(key): miss.append(key)
    return miss or None

def build_card(n, tl, rng, u):
    d = bytearray(u)
    px, pal, off, tw = load_card(d)
    px = px[:, :W].copy()
    problems = []
    grey = nearest(pal, (112, 112, 112)); white = nearest(pal, (248, 248, 248))
    black = nearest(pal, (16, 16, 16)); red = nearest(pal, (248, 0, 0))
    g = geometry(px, grey)
    neutral = [i for i, p in enumerate(pal.astype(int)) if p.max() - p.min() <= 24]
    sp_masks, bv_masks = masks_for(n)
    # ---- middle cells: rows, clear, draw
    for kind, (xa, xb), (ya, yb), mk in (('s', g['x_mid'], g['y_sp'], sp_masks),
                                         ('b', g['x_bv'], g['y_bv'], bv_masks)):
        cell = px[ya:yb, xa:xb]
        bd = bands(cell != white)
        keys = [k for k in tl if re.fullmatch(f'cm{n:02d}\\.{kind}\\d', k)]
        if len(bd) != len(keys):
            problems.append(f'{kind}: {len(bd)} rows on the card, {len(keys)} in the tl'); continue
        cmdrows = []
        for i, (a, b) in enumerate(bd):
            key = f'cm{n:02d}.{kind}{i + 1}'
            top = ya + (0 if i == 0 else (bd[i - 1][1] + a) // 2)
            bot = ya + (yb - ya if i == len(bd) - 1 else (b + bd[i + 1][0]) // 2)
            mtop, mbot = mk
            for m in mtop:
                if ya + a >= m > top: top = m
                if ya + b <= m < bot: bot = m
                if ya + a < m < ya + b: problems.append(f'{key}: JP crosses mask {m}')
            # a cover rect may end above the cell bottom: keep the
            # last name inside it (stray descenders showed below it)
            for m in mbot:
                if ya + b <= m < bot: bot = m
                # the JP name itself pokes out below the rect (cm06.b3 at
                # level 1, 2 px): keep the EN name above the rect end
                if ya + a < m < ya + b: bot = min(bot, m)
            cmdrows.append((top, bot, ya + a, ya + b, key))
        px[ya:yb, xa:xb] = white
        for top, bot, ja, jb, key in cmdrows:
            r = fit_lines(tl[key], xb - xa - 6, bot - top - 1, 17, 11, lead=1)
            if not r: problems.append(f'{key}: {tl[key]!r} does not fit'); continue
            can, x0, y0 = stamp(r[0], r[1], (xa + xb) / 2, (ja + jb) / 2)
            y0 = min(max(y0, top + 1), bot - can.shape[0])
            blend_quant(px, pal, can, x0, y0, pal[black], cands=neutral)
            if y0 < top or y0 + can.shape[0] > bot: problems.append(f'{key}: outside its band')
        # ---- command column notes for these rows
        xc0, xc1 = g['x_cmd']
        for top, bot, ja, jb, key in cmdrows:
            cl = annotations(px, pal, xc0, xc1 - 1, top, bot, red, white)
            notes = tl[key + '.ann'].split('|') if key + '.ann' in tl else []
            texts = [c for c in cl if not c[1]]
            if len(texts) < len(notes):
                problems.append(f'{key}: {len(texts)} notes found, {len(notes)} in the tl')
                continue
            # extra red clusters outside the icons are arrow heads: keep
            # the widest ones as notes, in left-to-right order
            wide = sorted(texts, key=lambda c: c[0][2] - c[0][0], reverse=True)[:len(notes)]
            cl = [c for c in cl if c[1] or c in wide]
            ti = 0
            for box, badge, icols in cl:
                if badge:
                    e = draw_note(px, pal, box, tl['ann.badge'], red, white, True, None, icols, xc0)
                else:
                    # free room around the note: up to the nearest non-white
                    # pixel left and right in the row (icons)
                    clear_note(px, pal, box, white, icols, xc0)
                    bx0, by0, bx1, by1 = box
                    row = (px[by0:by1, :] != white)
                    lo = bx0
                    while lo - 1 > xc0 and not row[:, lo - 1].any(): lo -= 1
                    hi = bx1
                    while hi < xc1 - 1 and not row[:, hi].any(): hi += 1
                    e = draw_note(px, pal, box, notes[ti], red, white, False, (lo + 1, hi - 1), icols, xc0)
                    ti += 1
                if e: problems.append(f'{key}: {e}')
    # ---- headings
    xh = (2, 102)
    e = heading(px, pal, (xh, (2, g['y_sp'][1])), tl['head.special'], pal[black], 17, 12, rng=rng)
    if e: problems.append(e)
    e = heading(px, pal, (xh, g['y_bvh']), tl['head.bva'], pal[black], 17, 12, rng=rng)
    if e: problems.append(e)
    e = heading(px, pal, (xh, g['y_easy']), tl['head.easy'], pal[white], 15, 10,
                shadow=pal[nearest(pal, (64, 56, 64))], find='light', flat=red)
    if e: problems.append(e)
    e = heading(px, pal, (xh, g['y_tu']), tl['head.teamup'], pal[black], 15, 10, rng=rng)
    if e: problems.append(e)
    # ---- team-up buttons 弱 / 強 -> L / H (red glyphs left of each icon)
    xc0, xc1 = g['x_cmd']; ty0, ty1 = g['y_tu']
    mid = (ty0 + ty1) // 2
    for (a, b), key in zip(((ty0, mid), (mid, ty1)), ('tu.light', 'tu.heavy')):
        cl = annotations(px, pal, xc0, xc1 - 1, a, b, red, white)
        for box, badge, icols in cl:
            bx0, by0, bx1, by1 = box
            clear_note(px, pal, box, white, icols, xc0)
            r = fit_lines(tl[key], bx1 - bx0 + 2, by1 - by0 + 2, 14, 9)
            can, x, y = stamp(r[0], r[1], (bx0 + bx1) / 2, (by0 + by1) / 2)
            blend_quant(px, pal, can, x, y, pal[red], cands=segment_cands(pal, pal[white], pal[red]), thr=0.2)
    full = np.zeros((H, tw * 4), np.uint8); full[:, :W] = px
    store_card(d, full, off, tw)
    return d, px, pal, problems

# ---------------------------------------------------------------- team-up
TU_CX, TU_Y, TU_H = 223, 397, 21
TU_PER = 256 // TU_H                     # names per tpage column (12)

def build_teamup(d, tl):
    """Redraw TIM2 in place (shared by all files). Returns the new records
    (u, v, w, h, tpage_x) per name and problems."""
    clut = struct.unpack_from('<16H', d, TIM2 + 0x14)
    pal = np.array([rgb15(v) for v in clut], np.uint8)
    po = TIM2 + 8 + struct.unpack_from('<I', d, TIM2 + 8)[0]
    _, vx, vy, w, h = struct.unpack_from('<IHHHH', d, po)
    off = po + 12
    white = nearest(pal, (248, 248, 248)); black = nearest(pal, (16, 16, 16))
    neutral = [i for i, p in enumerate(pal.astype(int)) if p.max() - p.min() <= 24]
    px = np.full((h, w * 4), white, np.uint8)
    recs, problems = [], []
    for k in range(NTU):
        text = tl[f'tu{k:02d}']
        r = fit_lines(text, 174, TU_H - 2, 17, 11)
        if not r: problems.append(f'tu{k:02d} {text!r} does not fit'); continue
        can, _, _ = stamp(r[0], r[1], 0, 0)
        cw = can.shape[1] + 2
        tp = 640 if k < TU_PER else 704
        u, v = 0, (k % TU_PER) * TU_H
        x0 = (tp - 640) * 4 + u + 1
        y0 = v + (TU_H - can.shape[0]) // 2
        blend_quant(px, pal, can, x0, y0, pal[black], cands=neutral)
        recs.append((tp, u, v, cw, TU_H))
    d[off:off + w * 2 * h] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    return px, pal, recs, problems

def patch_kanri(recs):
    k = bytearray(open(KANRI, 'rb').read())
    for i, (tp, u, v, w, h) in enumerate(recs):
        o = TU_REC + 20 * i
        old = struct.unpack_from('<10h', k, o)
        # flag, tpage x, tpage y, clut y (x 0), u, v, w, h, screen x, y
        assert old[1] in (640, 704) and old[2] == 256 and old[3] == 484, (i, old)
        struct.pack_into('<10h', k, o, old[0], tp, 256, old[3], u, v, w, h,
                         TU_CX - w // 2, TU_Y)
    return k

# ---------------------------------------------------------------- main
def main(argv):
    png = '--png' in argv; check = '--check' in argv
    only = [int(a) for a in argv if a.isdigit()] or range(1, NCARD + 1)
    tl = load_tl()
    rng = np.random.default_rng(1)
    total = 0
    src = open(ARC, 'rb').read()
    toc = arc.toc(src)
    assert len(toc) == NCARD, len(toc)
    out = bytearray(src)
    if png: os.makedirs('private/re/editchar/en', exist_ok=True)
    tu_px = tu_recs = None
    built, kept = [], []
    for k, (o, sz) in enumerate(toc):
        n = k + 1
        u = lz.decompress(src[o:o + sz])[0]
        problems = []
        miss = pending(n, tl, u) if n in only else ['(not selected)']
        if miss:
            d = bytearray(u); kept.append(n)
            if n in only and png:
                px, pal, _, _ = load_card(d)
                Image.fromarray(pal[px[:, :W]]).save(f'private/re/editchar/en/cm{n:02d}.png')
        else:
            d, px, pal, problems = build_card(n, tl, rng, u); built.append(n)
            if png:
                Image.fromarray(pal[px]).save(f'private/re/editchar/en/cm{n:02d}.png')
        tpx, tpal, recs, tprob = build_teamup(d, tl)
        if k == 0: problems += tprob
        tu_px, tu_pal, tu_recs = tpx, tpal, recs
        slot = (toc[k + 1][0] if k + 1 < len(toc) else len(src)) - o
        c = lz.compress(bytes(d), 256, name='EDITCM.BIZ')
        if len(c) > slot: problems.append(f'compressed {len(c)} > slot {slot}')
        if lz.decompress(c)[0][:len(d)] != bytes(d): problems.append('LZ round trip failed')
        if not problems:
            out[o:o + slot] = c + bytes(slot - len(c))
            struct.pack_into('<II', out, 8 * k, o, len(c))
        print(f'cm{n:02d}: {"EN" if not miss else "JP card, %d rows without EN" % len(miss)}, '
              f'{len(c)} bytes (slot {slot})' + ''.join(f'\n  {p}' for p in problems))
        total += len(problems)
    assert len(out) == len(src)
    print(f'EN cards: {" ".join(map(str, built))}')
    print(f'JP cards: {" ".join(map(str, kept))}')
    k = None
    if tu_recs and len(tu_recs) == NTU:
        k = patch_kanri(tu_recs)
        if png:
            Image.fromarray(tu_pal[tu_px]).save('private/re/editchar/en/teamup.png')
    else:
        total += 1
    if not check and not total:
        open('private/build/EDITCM.BIZ', 'wb').write(out)
        open('private/build/KANRI.BIN', 'wb').write(k)
    print(f'EDITCM: problems {total}')
    return 1 if total else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
