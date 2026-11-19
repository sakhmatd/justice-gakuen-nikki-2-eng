# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary exam sheets in English: the record sheet (DAT2/SIKEN10.BIZ, TIM
0x55000) and the two result sheets (DAT2/SIKEN20.BIZ entries 0 and 1, TIM
0x8800). The SIKEN2 overlay shows them (diary exams and the hub's Exam
Practice). Wording: tl/EXAMSHEET.txt. RE: private/re/examsheet/findings.org.

Each sheet is an 8bpp TIM, 512x480 at VRAM (512,0), one 256-colour CLUT at
(0,484). The labels are baked into the art. The build removes each JP
label (tools/inpaint.py: the label's ink plus a 5 px rim, LaMa fill,
write-back into the sheet's CLUT; the grid is never in the mask), draws the EN label laid out for its own width and quantises it to
the sheet's CLUT. Untouched: the grid, the digits and the record and
result rows the game draws on top, and the % sign.

Writes private/build/SIKEN10.BIZ and private/build/SIKEN20.BIZ (sheets,
result atlas pieces, stamps). Exam Practice (tools/examprac.py, tl/
EXAMGFX.txt): the menu atlas in SIKEN1/SIKEN10 TIM 1, the practice menu
pieces in SIKEN20 TIM 0, SIKEN1.BIZ (atlas, record sheet), SIKEN2.BIZ
(SIKEN20's TIM 0 and stamps, grade sheet) and the list tables and x
immediates in SIKEN.BIN and SIKEN2.BIN (read from private/build when
present; build_testmj.py runs after this and keeps them). --extra also
writes SIKENKK.BIZ (lesson names 0-29 and exam names 30-32 in the result
row) and SIKENG21/22.BIN (the unit record that points at the EN count
unit, from the disc copies, so build_testmj.py --terms must run after it);
without it the result row keeps the JP names and the JP 回. --dry writes nothing to private/build. --png DIR writes the JP and EN
sheets side by side to DIR.
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import fontpath
import arc, lz, streamsim
import examprac
import inpaint as IP
from mgbg_examhud import text_alpha


TL = 'tl/EXAMSHEET.txt'
SRC10, SRC20 = 'private/disc/DAT2/SIKEN10.BIZ', 'private/disc/DAT2/SIKEN20.BIZ'
OUT10, OUT20 = 'private/build/SIKEN10.BIZ', 'private/build/SIKEN20.BIZ'
OUTKK = 'private/build/SIKENKK.BIZ'
TIM10, TIM20 = 0x55000, 0x8800

F_TITLE = 'SourceSerif4-Bold.otf'          # OFL 1.1
F_LABEL = 'SourceSerif4Caption-Regular.otf' # OFL 1.1
F_SMALL = 'SourceSerif4Caption-Semibold.otf'
F_HEAD = 'MPLUS1p-Black.ttf'                        # OFL 1.1

# Ink colours of the current sheet, sampled from its JP title (sample_ink):
# fill (title and cells), the title's 1 px edge to the lower right and its
# shadow beyond the edge. SIKEN10 and SIKEN20/0: (88,40,248), (48,24,128),
# grey; SIKEN20/1: purple (184,64,248) with a dark edge.
STYLE = {}
GREEN = (0, 248, 112)          # 試験結果 fill
GREEN_SHADOW = (24, 24, 24)    # its drop shadow, 2 px down and right

# Layout boxes (x0, y0, x1, y1), inclusive pixels inside the grid lines.
# Lower table (SIKEN10 and SIKEN20 entry 0): grid rows 241-244, 275-276,
# 307-308, 339-340, 371-375; columns 73-75, 198-199, 383-384, 436-438.
LOW_COLS = [(76, 197), (200, 382), (385, 435)]
LOW_ROWS = [(245, 274), (277, 306), (309, 338), (341, 370)]
# Upper table (SIKEN20 entries 0, 1): rows 144-147, 178-179, 209-212;
# columns 72-75, 198-199, 376-377, 435-438.
UP_COLS = [(76, 197), (200, 375), (378, 434)]
UP_HEAD = (148, 177)

# JP label rects to clear (x0, y0, x1, y1), from the ink boxes.
LOW_JP = {'試験': (105, 247, 166, 275), '最高記録': (232, 247, 346, 275),
          '判定': (385, 247, 435, 275),
          '１学期期末試験': (76, 279, 197, 306), '２学期期末試験': (76, 311, 197, 338),
          '学年末試験': (76, 343, 197, 370),
          'のこりTIME': (200, 277, 244, 306), '回': (296, 312, 328, 338)}
UP_JP = {'試験': (80, 149, 142, 178), '記録': (256, 149, 323, 178),
         '評価': (378, 149, 434, 178)}
TITLE_JP = {'s10': (98, 186, 414, 224), 's20': (98, 109, 414, 143)}
BUTTON = {'s10': (340, 381, 435, 407), 's20b': (336, 408, 434, 434)}
ICON = {'s10': (345, 386, 14, 16), 's20b': (341, 413, 14, 16)}   # x, y, w, h
HEAD_JP = (176, 32, 310, 70)                       # 試験結果, entry 1 only

def load_tl(path=TL):
    out, jp = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            if line[4:].strip(): out[jp] = line[4:].strip()
            jp = None
    need = ["「'９９年度試験」成績表", '試験結果', '試験', '最高記録', '判定', '記録',
            '評価', '１学期期末試験', '２学期期末試験', '学年末試験', 'のこりTIME',
            '回', 'ボタン-戻る']
    miss = [k for k in need if k not in out]
    if miss: raise SystemExit(f'build_examgfx: no EN for {miss} in {path}')
    return out

# ---------------------------------------------------------------- TIM

def parse(d, o):
    """CLUT (256,) u16, offset of the pixels, pixels (480, 512) u8."""
    assert struct.unpack_from('<II', d, o) == (0x10, 9)
    cl = struct.unpack_from('<I', d, o + 8)[0]
    assert struct.unpack_from('<4H', d, o + 12) == (0, 484, 256, 1)
    clut = np.frombuffer(d[o + 20:o + 20 + 512], '<u2').copy()
    p = o + 8 + cl
    assert struct.unpack_from('<4H', d, p + 4) == (512, 0, 256, 480)
    px = np.frombuffer(d[p + 12:p + 12 + 512 * 480], np.uint8).reshape(480, 512).copy()
    return clut, p + 12, px

def clut_rgb(clut):
    c = clut.astype(int)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(float)

# ---------------------------------------------------------------- removal

def grid_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g > r + 20) & (g > b + 20) & (g < 200)

PAPER_REF = {'s10': (386, 310, 434, 337), 's20a': (202, 182, 374, 208),
             's20b': (202, 182, 374, 208)}      # empty cells

def paper_idx(px, ref):
    """CLUT indices that make up the paper: each covers >= 1% of an empty
    cell."""
    x0, y0, x1, y1 = ref
    v, n = np.unique(px[y0:y1 + 1, x0:x1 + 1], return_counts=True)
    return v[n >= 0.01 * n.sum()]

def mode_colour(rgb, m):
    v, n = np.unique(rgb[m].reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(c) for c in v[n.argmax()])

def sample_ink(px, rgb, rect, paper):
    """Fill, edge and shadow colours of the JP title in rect: the fill is
    the commonest strongly blue texel, the edge the commonest colour 1 px to
    its lower right, the shadow the commonest one 2 px down-right."""
    x0, y0, x1, y1 = rect
    sub, sp = rgb[y0:y1 + 1, x0:x1 + 1], px[y0:y1 + 1, x0:x1 + 1]
    pap = np.isin(sp, paper)
    blue = (sub[..., 2] > sub[..., 1] + 60) & ~pap
    fill = mode_colour(sub, blue)
    F = (sub == fill).all(-1)
    e = (shift(F, 1, 1) | shift(F, 0, 1)) & ~F & ~pap
    edge = mode_colour(sub, e)
    E = (sub == edge).all(-1)
    shm = shift(F, 2, 2) & ~F & ~E & ~pap
    return fill, edge, mode_colour(sub, shm)

RING = 5            # text_mask rim around the JP ink (shadows, soft edges)
ISLAND = 40         # unmasked islands inside a hole up to this size are filled too
DITHER_MAX = 24     # max sum |RGB| error of a dithered fill texel (else nearest)

def paper_core(px, rgb, rect, paper, grid=grid_mask):
    """JP ink in rect: every texel that is not paper and not grid (grid: a
    predicate on RGB, or None where the rect holds no grid line)."""
    x0, y0, x1, y1 = rect
    m = np.zeros(px.shape, bool)
    m[y0:y1 + 1, x0:x1 + 1] = ~np.isin(px[y0:y1 + 1, x0:x1 + 1], paper)
    if grid is not None: m[y0:y1 + 1, x0:x1 + 1] &= ~grid(rgb[y0:y1 + 1, x0:x1 + 1])
    return m

FILLS = {}          # sheet name -> RGB after the erase (review previews)
PAPER_MIN = 150     # paper texels have min(R, G, B) >= this (all five sheets)
TILE = 12           # paper grain is copied in tiles of this size

def paper_grain(rgb, out, m, struct):
    """Add the paper's fibre grain to the LaMa fill (which is smooth): the
    high-pass residual of clean paper (not in m, not struct, sigma 3
    normalised blur as the base), copied per TILE x TILE tile of m from
    the nearest offset whose whole tile lies on clean paper."""
    clean = ~ndimage.binary_dilation(m | struct, iterations=1)
    w = ndimage.gaussian_filter(clean.astype(float), 3)
    base = np.stack([ndimage.gaussian_filter(rgb[..., c] * clean, 3) for c in range(3)], -1)
    res = np.where(clean[..., None], rgb - base / np.maximum(w, 1e-6)[..., None], 0)
    H, W = m.shape
    offs = sorted(((dy, dx) for dy in range(-90, 91, 3) for dx in range(-90, 91, 3)
                   if abs(dy) >= TILE or abs(dx) >= TILE), key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    out = out.copy()
    miss = 0
    ys, xs = np.nonzero(m)
    for ty in range(ys.min() // TILE * TILE, ys.max() + 1, TILE):
        for tx in range(xs.min() // TILE * TILE, xs.max() + 1, TILE):
            ty_, tx_ = np.nonzero(m[ty:ty + TILE, tx:tx + TILE])
            if not len(ty_): continue
            ty_ = ty_ + ty; tx_ = tx_ + tx
            for dy, dx in offs:
                sy, sx = ty_ + dy, tx_ + dx
                if sy.min() < 0 or sx.min() < 0 or sy.max() >= H or sx.max() >= W: continue
                if clean[sy, sx].all():
                    out[ty_, tx_] += res[sy, sx]; break
            else: miss += len(ty_)
    return out, miss

def erase(px, pal, clut, core, limit, ring=RING, method='best', grid=True, paper=False,
          log=None):
    """Paint core out (tools/inpaint.py): core plus a ring px rim of texels
    that differ from the local background, kept inside limit and off the
    grid (if grid); small islands inside the hole are added. Fill: method
    ('best' = LaMa, CPU 'auto' without the model). On paper (paper=True)
    every darker texel (grid, art, other labels) is hidden from the model,
    which otherwise blends the grid colour into narrow cells, and the
    paper grain is added (paper_grain). Write-back into the sheet's CLUT
    with dither inside the mask; where the dithered pick is further than
    DITHER_MAX from the fill, the nearest colour. Returns (px, mask)."""
    if not core.any(): return px, core
    rgb = pal[px].astype(float)
    protect = ~limit | grid_mask(rgb) if grid else ~limit
    m = IP.text_mask(rgb, IP._bbox(core, ring + 1, px.shape), pred=lambda a: core & ~protect,
                     ring=ring, halo=1, min_px=0, close=0, protect=protect)
    isl = ndimage.binary_fill_holes(m) & ~m & ~protect
    if isl.any():
        lab, n = ndimage.label(isl)
        sz = ndimage.sum(isl, lab, np.arange(1, n + 1))
        m |= np.isin(lab, 1 + np.nonzero(sz <= ISLAND)[0])
    struct = None
    if paper:
        struct = rgb.min(-1) < PAPER_MIN
        if grid: struct |= grid_mask(rgb)
        struct = ndimage.binary_dilation(struct, iterations=1) & ~m
    out = IP.fill(rgb, m, method, avoid=struct)
    miss = 0
    if paper: out, miss = paper_grain(rgb, out, m, struct)
    # to_clut leaves texels of a 0x0000 (clear) entry alone; the p1 title
    # outline uses one, so those texels are first given an opaque entry
    src = px.copy()
    z = m & (clut[px] == 0)
    if z.any(): src[z] = np.nonzero(clut != 0)[0][0]
    fs = IP.to_clut(out, src, clut, m, pal=pal)
    near = IP.to_clut(out, src, clut, m, pal=pal, dither='none')
    stray = np.abs(pal[fs] - out).sum(-1) > DITHER_MAX
    if log is not None:
        log.append(f'erase {m.sum()} px ({core.sum()} ink), {stray.sum()} undithered'
                   + (f', {miss} px without grain' if miss else ''))
    return np.where(stray, near, fs), m

# ---------------------------------------------------------------- drawing

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

def fit(text, path, cap, maxw, minsx=0.8, track=0.0):
    """Alpha at cap; squeezed down to minsx, then a smaller cap, to fit
    maxw. -> (alpha, row of the cap top, cap)."""
    a, top = text_alpha(text, path, cap, track)
    if a.shape[1] <= maxw: return a, top, cap
    sx = maxw / a.shape[1]
    if sx >= minsx:
        a, top = text_alpha(text, path, cap, track, sx * 0.99)
        return a, top, cap
    while True:
        cap -= 0.5
        a, top = text_alpha(text, path, cap, track, minsx)
        if a.shape[1] <= maxw: return a, top, cap

def comp(canvas, layers, x, y):
    """Blend layers [(alpha, rgb)] (same shape) bottom-up onto canvas at
    x, y; returns the coverage mask (union of alphas)."""
    h, w = layers[0][0].shape
    cov = np.zeros(canvas.shape[:2], np.float32)
    for a, col in layers:
        reg = canvas[y:y + h, x:x + w]
        reg[:] = reg * (1 - a[..., None]) + np.array(col, float) * a[..., None]
        cov[y:y + h, x:x + w] = np.maximum(cov[y:y + h, x:x + w], a)
    return cov

def plain(a):
    return [(a, STYLE['fill'])]

def button_layers(a):
    """SIKEN20/1 button text on the art: fill with a dark shadow 1 px to
    the lower right, as in the JP."""
    a = np.pad(a, ((0, 2), (0, 2)))
    sh = np.maximum(shift(a, 1, 1), shift(a, 0, 1))
    return [(sh, STYLE['btn_shadow']), (a, STYLE['btn_fill'])]

def title_layers(a):
    """Fill with a 1 px dark-blue edge to the lower right and a grey shadow
    1-2 px further, as measured on the JP title."""
    a = np.pad(a, ((0, 3), (0, 3)))
    edge = np.maximum.reduce([shift(a, 0, 1), shift(a, 1, 0), shift(a, 1, 1)])
    sh = np.maximum.reduce([shift(a, 1, 2), shift(a, 2, 2), shift(a, 2, 1)])
    return [(sh * 0.85, STYLE['shadow']), (edge, STYLE['edge']), (a, STYLE['fill'])]

def head_layers(a):
    """試験結果 style: green fill and a dark drop shadow 2 px down-right."""
    a = np.pad(a, ((0, 3), (0, 3)))
    sh = np.maximum(shift(a, 2, 2), shift(a, 1, 1))
    return [(sh, GREEN_SHADOW), (a, GREEN)]

def place(canvas, a, top, cap, box, style, align='c', vbase=None):
    """Draw alpha a in box (x0, y0, x1, y1): centred (or left/right) in x,
    the cap height centred in y unless vbase gives the cap-top row."""
    x0, y0, x1, y1 = box
    w = a.shape[1]
    x = {'c': x0 + (x1 - x0 + 1 - w) // 2, 'l': x0, 'r': x1 + 1 - w}[align]
    capt = vbase if vbase is not None else y0 + int(round((y1 - y0 + 1 - cap) / 2))
    y = capt - top
    return comp(canvas, style(a), x, y), (x, y, w, a.shape[0])

def quantise(rgb, pal, cover, px):
    """Nearest CLUT index for covered texels (weighted RGB distance)."""
    m = cover > 0.03
    if not m.any(): return px
    P = pal[None, :, :]
    c = rgb[m][:, None, :]
    d = (((c - P) * np.array([0.30, 0.59, 0.11])) ** 2).sum(-1)
    out = px.copy(); out[m] = d.argmin(1).astype(np.uint8)
    return out

# ---------------------------------------------------------------- sheets

CAP_TITLE, CAP_HEAD, CAP_ROW, CAP_SMALL, CAP_GREEN = 17, 13, 13, 8, 20

def draw_title(canvas, tl, box, log):
    a, top, cap = fit(tl["「'９９年度試験」成績表"], F_TITLE, CAP_TITLE, box[2] - box[0] - 8, track=0.4)
    c, r = place(canvas, a, top, cap, (box[0], box[1], box[2], box[3] - 2), title_layers)
    log.append(f'title at {r}, cap {cap}')
    return c

def draw_cells(canvas, items, log, minsx=0.72):
    """items: (text, box, cap, align). The group shares one cap (the
    smallest any label needs); each label is squeezed to its own box."""
    cov = np.zeros(canvas.shape[:2], np.float32)
    maxw = lambda box: box[2] - box[0] + 1 - 6
    cap = min(fit(t, F_LABEL, c, maxw(b), minsx)[2] for t, b, c, _ in items)
    for text, box, _, align in items:
        a, top, c = fit(text, F_LABEL, cap, maxw(box), minsx)
        cv, r = place(canvas, a, top, c, box, plain, align)
        cov = np.maximum(cov, cv)
        log.append(f'{text!r} at {r} cap {c}')
    return cov

def low_table(canvas, tl, log):
    (h0, h1), r1, r2, r3 = LOW_ROWS
    C = LOW_COLS
    hdr = [(tl['試験'], (C[0][0], h0, C[0][1], h1), CAP_HEAD, 'c'),
           (tl['最高記録'], (C[1][0], h0, C[1][1], h1), CAP_HEAD, 'c'),
           (tl['判定'], (C[2][0], h0, C[2][1], h1), CAP_HEAD, 'c')]
    rows = [(tl[k], (C[0][0] + 2, y0, C[0][1], y1), CAP_ROW, 'c')
            for k, (y0, y1) in zip(('１学期期末試験', '２学期期末試験', '学年末試験'), (r1, r2, r3))]
    cov = np.maximum(draw_cells(canvas, hdr, log), draw_cells(canvas, rows, log))
    # remaining-time tag: two small lines at the left of the record cell
    lines = tl['のこりTIME'].split('|')
    y = r1[0] + 2
    for s in lines:
        a, top = text_alpha(s, F_SMALL, CAP_SMALL, 0.3)
        cv, r = place(canvas, a, top, CAP_SMALL, (C[1][0] + 2, y, C[1][1], y + CAP_SMALL), plain, 'l', vbase=y)
        cov = np.maximum(cov, cv); y += CAP_SMALL + 4
        log.append(f'{s!r} at {r}')
    # count unit: starts where 回 started, so the digits to its left keep
    # their place
    a, top = text_alpha(tl['回'], F_LABEL, CAP_ROW, 0.2)
    cv, r = place(canvas, a, top, CAP_ROW, (299, r2[0], C[1][1], r2[1]), plain, 'l')
    log.append(f'unit at {r}')
    return np.maximum(cov, cv)

def up_table(canvas, tl, log):
    h0, h1 = UP_HEAD
    C = UP_COLS
    return draw_cells(canvas, [(tl['試験'], (C[0][0], h0, C[0][1], h1), CAP_HEAD, 'c'),
                               (tl['記録'], (C[1][0], h0, C[1][1], h1), CAP_HEAD, 'c'),
                               (tl['評価'], (C[2][0], h0, C[2][1], h1), CAP_HEAD, 'c')], log)

def button(px, rgb, tl, name, log):
    """X icon + 'Back', right-aligned to where the JP label ended. Returns
    the text alpha, its cap-top row, text x, cap-top y, the icon (mask,
    indices) and the new icon x, y."""
    x0, y0, x1, y1 = BUTTON[name]
    ix0, iy0, iw, ih = ICON[name]
    sub = rgb[iy0:iy0 + ih, ix0:ix0 + iw]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    ink = ((np.abs(r - b) < 30) & (np.abs(r - g) < 30) & (r > 60) & (r < 200)) | (b > r + 20)
    yy, xx = np.mgrid[0:ih, 0:iw]
    disk = (yy - (ih - 1) / 2) ** 2 + (xx - (iw - 1) / 2) ** 2 <= (iw / 2 + 0.3) ** 2
    icon = (ndimage.binary_fill_holes(ink & disk) & disk, px[iy0:iy0 + ih, ix0:ix0 + iw].copy())
    a, top = text_alpha(tl['ボタン-戻る'], F_LABEL, CAP_ROW, 0.2)
    tx = x1 - 1 - a.shape[1]
    capt = iy0 + (ih - CAP_ROW) // 2
    log.append(f'button icon at ({tx - 5 - iw},{iy0}), text at x {tx}')
    return a, top, tx, capt, icon, tx - 5 - iw, iy0

def build_sheet(name, px, pal, tl, log, clut):
    """EN indices for sheet name ('s10', 's20a', 's20b'); clut: its CLUT."""
    rgb = pal[px]
    clear = []                                      # (rect) on paper
    if name in ('s10', 's20a'):
        clear += list(LOW_JP.values())
    if name in ('s20a', 's20b'):
        clear += list(UP_JP.values())
    clear.append(TITLE_JP['s20' if name != 's10' else 's10'])
    btn = None
    if name in BUTTON:
        a, top, tx, capt, icon, ix, iy = button(px, rgb, tl, name, log)
        btn = (a, top, tx, capt, icon, ix, iy)
    paper = paper_idx(px, PAPER_REF[name])
    tb = TITLE_JP['s10'] if name == 's10' else TITLE_JP['s20']
    ink = sample_ink(px, rgb, tb, paper)
    STYLE.update(fill=ink[0], edge=ink[1], shadow=ink[2])
    log.append(f'{name}: paper {len(paper)} indices, ink {ink}')
    if name == 's20b':
        x0, y0, x1, y1 = BUTTON[name]
        sub = rgb[y0:y1 + 1, ICON[name][0] + ICON[name][2] + 1:x1 + 1]
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        STYLE['btn_fill'] = mode_colour(sub, (b > r + 30) & (b > 150))
        STYLE['btn_shadow'] = mode_colour(sub, sub.max(-1) < 40)
        log.append(f"button ink {STYLE['btn_fill']} {STYLE['btn_shadow']}")
    # all paper labels in one fill, so no JP label is context for another
    core = np.zeros(px.shape, bool); lim = np.zeros(px.shape, bool)
    for x0, y0, x1, y1 in clear:
        core |= paper_core(px, rgb, (x0, y0, x1, y1), paper)
        lim[y0:y1 + 1, x0:x1 + 1] = True
    px, m = erase(px, pal, clut, core, lim, paper=True, log=log)
    bad = (m & (pal[px].min(-1) < 176)).sum()
    if bad: log.append(f'WARNING {name}: {bad} dark texels after the fill')
    if btn is not None:
        x0, y0, x1, y1 = BUTTON[name]
        rgb = pal[px]
        # the label and icon only: bluish or grey texels
        sub = rgb[y0:y1 + 1, x0:x1 + 1]
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        ink = b > r + 20
        if name == 's20b':                          # grey icon ring on the art
            ink |= (np.abs(r - b) < 30) & (np.abs(r - g) < 30) & (r > 60) & (r < 200)
        m = np.zeros(px.shape, bool); m[y0:y1 + 1, x0:x1 + 1] = ink
        ix0, iy0, iw, ih = ICON[name]
        m[iy0:iy0 + ih, ix0:ix0 + iw + 1] = True    # the whole old icon
        m = ndimage.binary_dilation(m, np.ones((3, 3), bool), iterations=2 if name == 's20b' else 1)
        m |= ndimage.binary_dilation(m, np.ones((3, 3), bool)) & (rgb.max(-1) < 48)   # dark shadow
        lim = np.zeros(px.shape, bool); lim[y0:y1 + 3, x0 - 40:x1 + 9] = True     # off the grid and edge lines
        px, _ = erase(px, pal, clut, m, lim, grid=False, paper=name == 's10', log=log)
    if name == 's20b':
        x0, y0, x1, y1 = HEAD_JP
        sub = rgb[y0:y1 + 1, x0:x1 + 1]
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        gr = (g > r + 40) & (g > b + 20)
        m = np.zeros(px.shape, bool); m[y0:y1 + 1, x0:x1 + 1] = gr
        m = ndimage.binary_dilation(m, iterations=1)
        m |= shift(m, 1, 1) | shift(m, 2, 2) | shift(m, 3, 3)
        lim = np.zeros(px.shape, bool); lim[y0 - 4:y1 + 5, x0 - 4:x1 + 5] = True
        px, _ = erase(px, pal, clut, m, lim, grid=False, log=log)
    FILLS[name] = pal[px]
    canvas = pal[px].astype(float)
    cov = np.zeros(px.shape, np.float32)
    tb = TITLE_JP['s10'] if name == 's10' else TITLE_JP['s20']
    cov = np.maximum(cov, draw_title(canvas, tl, tb, log))
    if name in ('s10', 's20a'): cov = np.maximum(cov, low_table(canvas, tl, log))
    if name in ('s20a', 's20b'): cov = np.maximum(cov, up_table(canvas, tl, log))
    if name == 's20b':
        a, top, cap = fit(tl['試験結果'], F_HEAD, CAP_GREEN, 300, track=0.6)
        cv, r = place(canvas, a, top, cap, (0, HEAD_JP[1], 511, HEAD_JP[1] + 30), head_layers)
        cov = np.maximum(cov, cv); log.append(f'head at {r}')
    if btn is not None:
        a, top, tx, capt, icon, ix, iy = btn
        cv, r = place(canvas, a, top, CAP_ROW, (tx, capt, tx + a.shape[1] - 1, capt + CAP_ROW),
                       button_layers if name == 's20b' else plain, 'l', vbase=capt)
        cov = np.maximum(cov, cv)
    out = quantise(canvas, pal, cov, px)
    if btn is not None:
        im, ic = icon
        reg = out[iy:iy + im.shape[0], ix:ix + im.shape[1]]
        reg[im] = ic[im]
    return out

# ---------------------------------------------------------------- pieces
# Result screen sprites (RE: private/re/examsheet/findings.org).
# - SIKEN20 TIM 0 (both entries, identical): 4bpp 256x256 atlas at VRAM
#   (896,256), 9 CLUTs at (0,483). The result row uses CLUT 7: index 0
#   transparent, 2 darkest ink, 3-8 lighter antialias.
# - SIKENG21/22 (SIKEN2's result sub-overlay) 0x80: unit pieces, 5 u16
#   (x, u, v, w, h) each, groups ended by a 0: time (' at 280, " at 325,
#   のこりTIME at 202), count (回 at 300), percent (% at 300). y is 184.
# - SIKEN20 TIMs 2 and 3: the big 合格 (red) / 不合格 (blue) stamps, 4bpp
#   328/344x480 at VRAM (768,0); drawn as FT4 quads at x 85, y 0.
# - SIKENKK entries 30-32: the exam name sprite of the result row (124x28
#   4bpp at VRAM (896,0), CLUT (0,486); drawn at 75,182).

F_PIECE = 'SourceSerif4Caption-Semibold.otf'
F_STAMP = 'ariblk.ttf'                  # as in N1 (PASSED/FAILED)
F_KK = 'DejaVuSansCondensed-Bold.ttf'   # as N1's red sheet strips

NOKORI = (144, 80, 40, 26)                # atlas rect of のこりTIME
KAI_NEW = (40, 112)                       # free atlas texels for the 回 unit
OVALS = {'合格（小）': (200, 38, 56, 33), '不合格（小）': (200, 70, 56, 34)}
G2_UNIT = 0x80                            # SIKENG21/22 unit table
G2_KAI = 0xa0                             # its 回 record (x, u, v, w, h)
G2_FILES = ('SIKENG21.BIN', 'SIKENG22.BIN')
KK_EXAMS = {30: '１学期期末試験', 31: '２学期期末試験', 32: '学年末試験'}

class Tim4:
    """4bpp TIM inside a buffer (one CLUT row may hold several CLUTs)."""
    def __init__(self, buf, off):
        self.buf, self.off = buf, off
        flag, bl = struct.unpack_from('<II', buf, off + 4)
        assert flag & 3 == 0
        cw, ch = struct.unpack_from('<2H', buf, off + 16)
        self.clut = np.frombuffer(bytes(buf[off + 20:off + 20 + 2 * cw * ch]), '<u2').copy()
        p = off + 8 + bl
        _, _, _, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.stride, self.H = p + 12, w * 2, h
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        self.px = np.empty((h, w * 4), np.uint8)
        self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        self.W = self.px.shape[1]
        self.orig = self.px.copy()

    def store(self):
        raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8)
        self.buf[self.pix:self.pix + self.stride * self.H] = raw.tobytes()

    def rgb(self, k):
        c = self.clut[16 * k:16 * k + 16].astype(int)
        return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.uint8)

def ink_idx(a, lo=2, hi=8, cut=0.12):
    """Alpha -> atlas CLUT 7 ramp: lo = full ink, hi = faintest; 0 clear."""
    out = np.clip(np.rint(lo + (1 - a) * (hi - lo)), lo, hi).astype(np.uint8)
    out[a < cut] = 0
    return out

def put_alpha(px, a, x, y, w, h, align='c', top=None, cap=None, fn=ink_idx):
    """Write fn(a) into px rect (x, y, w, h), cleared first is up to the
    caller. Centred in x; the cap centred in y unless top gives its row."""
    ah, aw = a.shape
    assert aw <= w, (aw, w)
    dx = {'c': (w - aw) // 2, 'l': 0}[align]
    dy = (h - ah) // 2 if top is None else top
    reg = px[y + dy:y + dy + ah, x + dx:x + dx + aw]
    v = fn(a)
    reg[v > 0] = v[v > 0]

def oval_words(px, ovals, tl, log):
    """Small pass/fail ovals in a 4bpp atlas: keep the rim, refill the body
    with its commonest index and write the word in the index farthest from
    it (the JP text index). ovals: {tl key: (x, y, w, h)}."""
    for key, (x, y, w, h) in ovals.items():
        sub = px[y:y + h, x:x + w]
        body = ndimage.binary_fill_holes(sub != 0)
        inner = ndimage.binary_erosion(body, iterations=3)
        v, n = np.unique(sub[inner], return_counts=True)
        fill = int(v[n.argmax()])
        far = int(v[np.argmax(np.abs(v.astype(int) - fill))])  # text index
        sub[inner] = fill
        ys, xs = np.nonzero(inner)
        iw = xs.max() - xs.min() + 1 - 6
        a, top, cap = fit(tl[key], F_STAMP, 9, iw, minsx=0.8, track=0.3)
        ramp = lambda al: np.where(al < 0.12, 0, np.rint(fill + (far - fill) * np.clip(al, 0, 1))).astype(np.uint8)
        ah, aw = a.shape
        cy = int(round(ys.mean())) - ah // 2 + 1
        cx = int(round(xs.mean())) - aw // 2
        reg = sub[cy:cy + ah, cx:cx + aw]
        val = ramp(a)
        m = (val > 0) & inner[cy:cy + ah, cx:cx + aw]
        reg[m] = val[m]
        log.append(f'oval {tl[key]!r} at {x},{y} fill {fill} text {far} cap {cap}')

def atlas_pieces(t, tl, log):
    """Redraw のこりTIME, add the 回 unit and the two small ovals."""
    px = t.px
    x, y, w, h = NOKORI
    px[y:y + h, x:x + w] = 0
    lines = tl['のこりTIME'].split('|')
    for k, s_ in enumerate(lines):
        a, top = text_alpha(s_, F_PIECE, 8, 0.3)
        put_alpha(px, a, x, y + 1 + 13 * k, w, 11, 'c', top=1 - top)
    # 回 -> unit word at free texels; the old piece stays unreferenced
    a, top = text_alpha(tl['回'], F_LABEL, 10, 0.2)
    ux, uy = KAI_NEW
    assert not px[uy:uy + 20, ux:ux + a.shape[1] + 2].any()
    put_alpha(px, a, ux, uy, a.shape[1] + 2, 20, 'l', top=5 - top)
    kai_w = a.shape[1] + 2
    oval_words(px, OVALS, tl, log)
    log.append(f'atlas: unit {tl["回"]!r} at u {ux} w {kai_w}')
    return kai_w

def stamp(t, text):
    """N1 build_examgfx.stamp: rounded double frame and the word along the
    long axis of the original's rotated rectangle. Ink 14, antialias 13..1
    (to white), 15 transparent. The last 4 columns are cleared."""
    W = t.W - 4
    ink = t.orig[:, :W] != 15
    ys, xs = np.nonzero(ink)
    c = np.array([xs.mean(), ys.mean()])
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xs - c[0], ys - c[1]])))
    ax = evec[:, 1]
    if ax[1] > 0: ax = -ax
    ang = np.degrees(np.arctan2(-ax[1], ax[0]))
    pr = np.stack([xs - c[0], ys - c[1]], 1)
    u = pr @ ax; v = pr @ np.array([-ax[1], ax[0]])
    L = np.percentile(u, 99.7) - np.percentile(u, 0.3)
    S = np.percentile(v, 99.7) - np.percentile(v, 0.3)
    ss = 3
    Lp, Sp = int(L * ss), int(S * ss)
    im = Image.new('L', (Lp, Sp), 0)
    dr = ImageDraw.Draw(im)
    th = int(S * 0.09 * ss); r = int(S * 0.16 * ss)
    dr.rounded_rectangle((th // 2, th // 2, Lp - th // 2, Sp - th // 2), radius=r, outline=255, width=th)
    g = int(S * 0.15 * ss); th2 = max(ss, int(th * 0.45))
    dr.rounded_rectangle((g, g, Lp - g, Sp - g), radius=int(r * 0.6), outline=255, width=th2)
    inner_w, inner_h = Lp - 2 * g - 6 * th2, Sp - 2 * g - 6 * th2
    size = inner_h
    while True:
        f = fontpath.truetype(F_STAMP, size)
        b = f.getbbox(text)
        if (b[2] - b[0] <= inner_w and b[3] - b[1] <= inner_h * 0.9) or size < 10: break
        size -= 2
    dr.text(((Lp - (b[2] - b[0])) // 2 - b[0], (Sp - (b[3] - b[1])) // 2 - b[1]), text, font=f, fill=255)
    rng = np.random.default_rng(7)
    a = np.asarray(im, np.float32) / 255
    noise = ndimage.gaussian_filter(rng.random(a.shape), 6 * ss / 3)
    noise = (noise - noise.mean()) / noise.std()
    a = a * np.clip(1.0 - np.maximum(0, noise - 2.2) * 3, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).rotate(ang, resample=Image.BICUBIC, expand=True)
    im = im.resize((im.width // ss, im.height // ss), Image.BOX)
    a = np.asarray(im, np.float32) / 255
    full = np.zeros((t.H, W), np.float32)
    h, w = a.shape
    x0 = int(round(c[0] - w / 2)); y0 = int(round(c[1] - h / 2))
    sx0, sy0, dx0, dy0 = max(0, -x0), max(0, -y0), max(0, x0), max(0, y0)
    hh = min(h - sy0, t.H - dy0); ww = min(w - sx0, W - dx0)
    full[dy0:dy0 + hh, dx0:dx0 + ww] = a[sy0:sy0 + hh, sx0:sx0 + ww]
    out = np.full((t.H, W), 15, np.uint8)
    lvl = np.clip(np.rint(full * 14), 0, 14).astype(np.uint8)
    out[full > 0.08] = np.maximum(lvl[full > 0.08], 1)
    t.px[:, :W] = out
    t.px[:, W:] = 15
    return ang

def kk_title(t, text):
    """SIKENKK exam name: orange-red index 1, antialias 2..15 to cream, 0
    clear; centred in the 124x28 image (the sprite x is fixed)."""
    t.px[:] = 0
    a, top, cap = fit(text, F_KK, 13, t.W - 4, minsx=0.8, track=0.2)
    fn = lambda al: np.where(al < 0.1, 0, np.clip(np.rint(1 + (1 - al) * 14), 1, 15)).astype(np.uint8)
    put_alpha(t.px, a, 0, 0, t.W, t.H, 'c', top=(t.H - int(round(cap))) // 2 - top, fn=fn)
    return cap

def patch_g2(kai_w, problems):
    """Point the 回 record of SIKENG21/22 at the new unit texels. No other
    builder writes these overlays, so the disc copies are the input."""
    out = {}
    for n in G2_FILES:
        d = bytearray(open(f'private/disc/DAT1/{n}', 'rb').read())
        rec = struct.unpack_from('<5H', d, G2_KAI)
        if rec != (300, 0, 112, 20, 20):
            problems.append(f'{n}: unexpected unit record {rec}'); continue
        struct.pack_into('<5H', d, G2_KAI, 300, KAI_NEW[0], KAI_NEW[1], kai_w, 20)
        out[n] = bytes(d)
    return out

def build_kk(texts, log, problems, prev):
    """texts: {entry: EN}. Entries 25-28 have another CLUT order; they are
    drawn for entry 30's CLUT and remapped (examprac.kk_remap)."""
    D = open('private/disc/DAT2/SIKENKK.BIZ', 'rb').read()
    toc = arc.toc(D)
    out = bytearray(D)
    o, s_ = toc[30]
    ref = Tim4(bytearray(lz.decompress(D[o:o + s_])[0]), 0).rgb(0)
    for e, text in sorted(texts.items()):
        o, s_ = toc[e]
        d = bytearray(lz.decompress(D[o:o + s_])[0])
        t = Tim4(d, 0)
        jp = t.rgb(0)[t.px]
        cap = kk_title(t, text) if isinstance(text, str) else examprac.kk_lesson(sys.modules[__name__], t, *text)
        if t.clut[0] != 0: examprac.kk_remap(t, ref)
        t.store()
        prev.append(('kk%d' % e, jp, t.rgb(0)[t.px]))
        end = toc[e + 1][0] if e + 1 < len(toc) else -(-len(D) // 2048) * 2048
        c = pack(f'SIKENKK/{e}', bytes(d), end - o, log, problems)
        out[o:end] = bytes(end - o)
        out[o:o + len(c)] = c
        struct.pack_into('<II', out, 8 * e, o, len(c))
        log.append(f'SIKENKK {e} {text!r}: {cap}')
    return bytes(out[:len(D)])

# ---------------------------------------------------------------- files

def pack(name, data, slot, log, problems):
    c = lz.compress(data, name=name.split('/')[0] + '.BIZ')
    assert lz.decompress(c)[0] == data
    o, done = streamsim.run(c)
    if not done or o[:len(data)] != data: problems.append(f'{name}: streamsim failed')
    log.append(f'{name}: {len(c)} bytes (room {slot - len(c)})')
    if len(c) > slot: problems.append(f'{name}: {len(c)} > {slot}')
    return c

def slot_of(path):
    return -(-os.path.getsize(path) // 2048) * 2048

def build(pngdir=None, write=True, extra=False):
    tl = load_tl(); log, problems = [], []
    prev, pieces = [], []
    ptl = examprac.load_tl(); titles = examprac.lesson_titles()
    # Exam Practice menu atlas (SIKEN1 TIM 1 = SIKEN10 TIM 1) and lists
    mt, d1, lists1, imm1, lists2, imm2 = examprac.build_menu(ptl, titles, problems, log)
    if lists1 is None:
        print('\n'.join(log + problems)); print('build_examgfx: nothing written'); return 1
    oval_words(mt.px, examprac.MENU_OVALS, tl, log)    # record sheet ovals (SIKEN, SIKEN2)
    mt.store()
    mo, mn = examprac.MENU_TIM, examprac.tim_size(d1, examprac.MENU_TIM)
    # SIKEN10: one LZ stream
    D = open(SRC10, 'rb').read()
    d = bytearray(lz.decompress(D)[0])
    clut, off, px = parse(bytes(d), TIM10)
    pal = clut_rgb(clut)
    new = build_sheet('s10', px, pal, tl, log, clut)
    d[off:off + new.size] = new.tobytes()
    jp1 = lz.decompress(open(examprac_src(1), 'rb').read())[0]
    assert d[mo:mo + mn] == jp1[mo:mo + mn], 'SIKEN10 TIM 1 differs from SIKEN1 TIM 1'
    d[mo:mo + mn] = d1[mo:mo + mn]
    out10 = pack('SIKEN10', bytes(d), slot_of(SRC10), log, problems)
    prev.append((pal[px], pal[new]))
    # SIKEN20: archive of two LZ entries
    D = open(SRC20, 'rb').read()
    toc = arc.toc(D)
    assert len(toc) == 2, toc
    comp_e = []
    for e, (o, s) in enumerate(toc):
        d = bytearray(lz.decompress(D[o:o + s])[0])
        clut, off, px = parse(bytes(d), TIM20)
        pal = clut_rgb(clut)
        new = build_sheet('s20a' if e == 0 else 's20b', px, pal, tl, log, clut)
        d[off:off + new.size] = new.tobytes()
        at = Tim4(d, 0)
        kai_w = atlas_pieces(at, tl, log if e == 0 else [])
        at.store()
        pt = examprac.Tim(d, 0)                     # practice menu pieces (rows 0-39)
        pm, pmw = examprac.practice(pt, ptl, problems)
        pt.store()
        for so, key in ((0x45000, '合格'), (0x58800, '不合格')):
            st = Tim4(d, so)
            ang = stamp(st, tl[key]); st.store()
            if e == 0:
                log.append(f'stamp {tl[key]!r} angle {ang:.1f}')
                pieces.append((key, st.rgb(0)[st.orig], st.rgb(0)[st.px]))
        if e == 0:
            pieces.append(('atlas', np.concatenate([at.rgb(7)[at.orig], at.rgb(6)[at.orig]], 0),
                           np.concatenate([at.rgb(7)[at.px], at.rgb(6)[at.px]], 0)))
            pieces.append(('practice', pt.rgba(1)[pt.orig[:40]][..., :3], pt.rgba(1)[pt.px[:40]][..., :3]))
            d20 = bytes(d)
            log.append(f'practice menu lists {pm}')
        ends = toc[e + 1][0] if e + 1 < len(toc) else -(-len(D) // 2048) * 2048
        comp_e.append(pack(f'SIKEN20/{e}', bytes(d), ends - o, log, problems))
        prev.append((pal[px], pal[new]))
    out20 = bytearray(D)
    for e, (o, s) in enumerate(toc):
        end = toc[e + 1][0] if e + 1 < len(toc) else len(D)
        out20[o:end] = bytes(end - o)
        out20[o:o + len(comp_e[e])] = comp_e[e]
        struct.pack_into('<II', out20, 8 * e, o, len(comp_e[e]))
    out20 = bytes(out20)
    # SIKEN1.BIZ: menu atlas + record sheet; SIKEN2.BIZ: SIKEN20's atlas
    # and stamps + grade sheet
    clut, off, px = parse(bytes(d1), 0x55000)
    pal = clut_rgb(clut)
    new = examprac.build_sheet(sys.modules[__name__], 'p1', px, pal, ptl | tl, log, clut)
    d1[off:off + new.size] = new.tobytes()
    prev.append((pal[px], pal[new]))
    out1 = pack('SIKEN1', bytes(d1), slot_of(examprac_src(1)), log, problems)
    d2 = bytearray(lz.decompress(open(examprac_src(2), 'rb').read())[0])
    for o in (0, 0x45000, 0x58800):
        n = examprac.tim_size(d2, o)
        assert d2[o:o + n] == lz.decompress(D[toc[0][0]:toc[0][0] + toc[0][1]])[0][o:o + n], hex(o)
        d2[o:o + n] = d20[o:o + n]
    clut, off, px = parse(bytes(d2), 0x8800)
    pal = clut_rgb(clut)
    new = examprac.build_sheet(sys.modules[__name__], 'p2', px, pal, ptl | tl, log, clut)
    d2[off:off + new.size] = new.tobytes()
    prev.append((pal[px], pal[new]))
    out2 = pack('SIKEN2', bytes(d2), slot_of(examprac_src(2)), log, problems)
    ovl = {n: examprac.patch_overlay(n, l, i, pm, pmw, problems, log)
           for n, l, i in (('SIKEN', lists1, imm1), ('SIKEN2', lists2, imm2))}
    kk = {e: tl[k] for e, k in KK_EXAMS.items()}
    kk.update(examprac.kk_lessons(sys.modules[__name__], titles, log))
    outkk = build_kk(kk, log, problems, pieces)
    outg2 = patch_g2(kai_w, problems)
    print('\n'.join(log))
    if problems:
        print('\n'.join(problems)); print('build_examgfx: nothing written'); return 1
    if write:
        os.makedirs('private/build', exist_ok=True)
        files = {OUT10: out10, OUT20: out20, 'private/build/SIKEN1.BIZ': out1,
                 'private/build/SIKEN2.BIZ': out2,
                 'private/build/SIKEN.BIN': ovl['SIKEN'], 'private/build/SIKEN2.BIN': ovl['SIKEN2']}
        for p, data in files.items(): open(p, 'wb').write(data)
        print('wrote', ' '.join(files))
        if extra:          # SIKENKK exam and lesson names, the SIKENG21/22 unit record
            open(OUTKK, 'wb').write(outkk)
            for n, d in outg2.items(): open(f'private/build/{n}', 'wb').write(d)
            print('wrote', OUTKK, ' '.join(outg2))
    if pngdir:
        os.makedirs(pngdir, exist_ok=True)
        for k, (jp, en) in zip(('siken10', 'siken20_0', 'siken20_1', 'siken1_record', 'siken2_grade'), prev):
            im = np.concatenate([jp, en], 1).astype(np.uint8)
            Image.fromarray(im).save(os.path.join(pngdir, k + '.png'))
        for k, jp, en in pieces:
            im = np.concatenate([jp, np.zeros((jp.shape[0], 4, 3), np.uint8), en], 1)
            Image.fromarray(im.astype(np.uint8)).save(os.path.join(pngdir, f'piece_{k}.png'))
        kks = [en for k, jp, en in pieces if k.startswith('kk')]
        jps = [jp for k, jp, en in pieces if k.startswith('kk')]
        grid = np.concatenate([np.concatenate([np.pad(j, ((1, 1), (1, 1), (0, 0))), np.pad(e, ((1, 1), (1, 1), (0, 0)))], 1)
                               for j, e in zip(jps, kks)], 0)
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(pngdir, 'kk_all.png'))
        sc = examprac.screens(mt, lists1, imm1, lists2, imm2)
        crop = [s[30:440, 80:432] for _, s in sc]
        g = np.concatenate([np.concatenate(crop[0:3], 1), np.concatenate(crop[3:6], 1)], 0)
        Image.fromarray(g).save(os.path.join(pngdir, 'menus.png'))
        Image.fromarray(examprac.lesson_sheet(mt, lists1, imm1)).save(os.path.join(pngdir, 'lessons.png'))
        print('previews in', pngdir)
    return 0

def examprac_src(n):
    return f'private/disc/DAT2/SIKEN{n}.BIZ'

if __name__ == '__main__':
    a = sys.argv[1:]
    png = a[a.index('--png') + 1] if '--png' in a else None
    sys.exit(build(png, write='--dry' not in a, extra='--extra' in a))
