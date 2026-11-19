# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Repaint the PocketStation pages (DAT2/PDATIM.BIZ) in English.

PDATIM.BIZ is an archive of five LZ entries, each one 8bpp 512x480 TIM
with its own 256-colour CLUT: 0 J-BANK page, 1-4 the four help pages
(Part 1 Poke-J Mail, 2 Shoma, 3 Ran, 4 Nagare). All text is painted into
the pictures. Wording: tl/PDATIM.txt. RE: private/re/pocket/findings.org
(sections "PDATIM.BIZ", "J-BANK page", "Disc room").

Each page is erased by masks (pixels that differ from a background
diffused in from the box border, minus kept callout lines), filled once
with minittl_rules.erase_fill (5 px rim, LaMa or the OpenCV fallback),
then the EN is drawn in the original's colours: red/green header with a
black outline and drop shadow, blue italic title on a redrawn banner band,
white or yellow body text with a black outline, button labels and grey
action labels on the callouts. The PocketStation LCD shots, callout lines,
the "Part N" tags and the footer icons stay as they are.

The archive is relaid (entries at 0x800 boundaries, TOC rewritten) and
padded to the JP size; the loader reads only the on-disc TOC.
Usage: build_pdatim.py [--png] [--check]
  --png    write JP|EN previews to private/re/pocket/pdatim/en/
  --check  build everything, write nothing to private/build
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from scipy import ndimage
import arc, lz
import minittl_rules as M
from minittl_rules import render, over, dilate, inpaint, grain, Page, font, erase_fill

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARC = 'private/disc/DAT2/PDATIM.BIZ'
OUT = 'private/build/PDATIM.BIZ'
PREV = 'private/re/pocket/pdatim/en'
NAME = 'PDATIM.BIZ'
F_HEAD = 'seguibl.ttf'        # Segoe UI Black (upright, as the JP header)
F_BAN = 'seguibli.ttf'        # Segoe UI Black Italic (as the rules pages)
F_BODY = 'tahomabd.ttf'       # as the mini-game rules pages
F_FOOT = 'impact.ttf'         # condensed, as the NEXT/BACK tags
F_JB = 'bahnschrift.ttf'      # J-BANK labels (DIN-like, as the J-BANK text)
PW, PH = 512, 480

WHITE = (248, 248, 248); BLACK = (8, 8, 8); RED = (248, 0, 0)
YELLOW = (248, 216, 0); BLUE = (0, 0, 248); GREEN_T = (24, 160, 0)
PURPLE = (88, 0, 160); GREY = (200, 200, 200); NAVY = (32, 56, 80)

def C(c): return np.array(c, float)

# ---------------------------------------------------------------- data

def load_tl(path='tl/PDATIM.txt'):
    out = {}
    for line in open(os.path.join(ROOT, path), encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

# Per page: erase boxes (x0, y0, x1, y1, thr, keep) and the drawing jobs.
# keep: colour class whose pixels stay (callout lines): 'red', 'green'.
# Boxes avoid the LCD shots, the Part N tags and the footer icons.
HDR = (52, 245, 33, 70)               # x0, centre x, y0, y1 of the JP header
ERASE = {
    0: [(222, 112, 292, 144, 28, None), (222, 193, 292, 224, 28, None)],
    1: [(48, 26, 442, 76, 60, None),               # header
        (28, 80, 296, 123, 45, None),              # banner + band
        (22, 112, 470, 232, 60, None),             # body
        (22, 238, 420, 298, 60, None),             # note
        (200, 312, 268, 403, 50, 'red'),           # button labels
        (268, 312, 396, 356, 50, 'red'),           # actions (rows 1-2)
        (280, 356, 396, 403, 50, 'red'),           # actions (rows 3-4)
        (292, 422, 327, 448, 40, None)],           # 戻る
    2: [(48, 26, 442, 76, 60, None),
        (28, 80, 296, 123, 45, None),
        (34, 112, 492, 198, 60, None),
        (34, 199, 492, 284, 60, None),
        (354, 298, 440, 334, 50, 'red'),
        (364, 318, 426, 354, 50, 'red'),
        (292, 422, 327, 448, 40, None)],
    3: [(48, 22, 444, 70, 60, None),
        (26, 64, 340, 113, 45, 'green'),
        (30, 118, 218, 300, 60, 'green'),
        (30, 300, 320, 370, 60, 'green'),
        (30, 382, 322, 412, 60, 'green'),
        (392, 160, 510, 240, 50, 'green'),
        (322, 218, 470, 270, 50, 'green'),
        (322, 282, 502, 333, 50, 'green'),
        (322, 346, 470, 396, 50, 'green'),
        (292, 422, 327, 448, 40, None)],
    4: [(48, 22, 442, 72, 60, None),
        (30, 82, 230, 124, 45, None),
        (38, 121, 168, 176, 50, 'red'),
        (118, 158, 233, 214, 50, 'red'),
        (114, 246, 266, 298, 50, 'red'),
        (238, 90, 495, 170, 'wb', None),
        (238, 175, 495, 294, 'wb', None),
        (10, 305, 316, 420, 'yb', None),
        (290, 393, 356, 420, 'yb', None),
        (348, 422, 383, 447, 40, None)],
}
# The header on each page: fill and shadow factor (shadow = bg * k).
HEAD = {1: (RED, 0.79), 2: (RED, 0.42), 3: ((0, 248, 184), 0.45), 4: (RED, 0.5)}
# Banner band: JP band rows (top, bottom), left x, band colours (cap, mid)
BAND = {1: (93, 109, 40), 2: (93, 109, 41), 3: (83, 99, 36), 4: (93, 109, 41)}
BAN_MAXW = {1: 330, 2: 340, 3: 296, 4: 350}

# ---------------------------------------------------------------- masks

def keep_mask(img, kind):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    if kind == 'red':
        return (r > 150) & (g < 70) & (b < 70)
    if kind == 'green':
        return (g > 130) & (r < 90) & (b < 120) & (g > r + 60)
    return np.zeros(img.shape[:2], bool)

def text_mask(img, box):
    """(erase mask, kept line pixels) for one box."""
    x0, y0, x1, y1, thr, keep = box
    rect = np.zeros(img.shape[:2], bool); rect[y0:y1, x0:x1] = True
    K = keep_mask_core(img, keep) & ndimage.binary_dilation(rect, iterations=3) if keep \
        else np.zeros(rect.shape, bool)
    if isinstance(thr, str):
        # text colours: w white, y yellow, b black outline (+1 px rim)
        cols = {'w': WHITE, 'y': (248, 220, 0), 'b': BLACK}
        hit = np.zeros(rect.shape, bool)
        for ch in thr:
            hit |= np.sqrt(((img - C(cols[ch])) ** 2).sum(-1)) < (90 if ch == 'b' else 60)
        mk = rect & hit
    else:
        bg = inpaint(img, rect | K, iters=200)
        d = np.sqrt(((img - bg) ** 2).sum(-1))
        mk = rect & (d > thr)
    mk = ndimage.binary_closing(mk, iterations=1) & rect
    mk = ndimage.binary_dilation(mk, iterations=2) & ndimage.binary_dilation(rect, iterations=1)
    return mk & ~K, K

def keep_mask_core(img, kind):
    """Line pixels plus their 1 px antialias rim, only where the line is
    at least 2 px long (drops stray text pixels of the same hue)."""
    k = keep_mask(img, kind)
    lab, n = ndimage.label(k, np.ones((3, 3)))
    if n:
        sz = ndimage.sum(k, lab, range(1, n + 1))
        k = np.isin(lab, 1 + np.where(sz >= 6)[0])
    return ndimage.binary_dilation(k, iterations=1)

FOOT_FILL = 'telea'     # footer fill: flat ground, LaMa invents strokes there

def erase(pg, boxes):
    mk = np.zeros(pg.img.shape[:2], bool); K = mk.copy()
    if pg is not None and boxes is not ERASE[0]:
        K[73:100, 396:464] = True            # the Part N tag (ink from y 75)
    for b in boxes:
        m, k = text_mask(pg.img, b); mk |= m; K |= k
    mk &= ~K
    # the rim stays inside the boxes: the footer
    # arrows sit right next to them. Ring 4: ring 5 squares off the
    # PocketStation outline under the page 4 text.
    # The footer box (y >= 420) ends on the arrow's light left edge: the arrow is
    # kept and not copied from, or the fill paints its white into the gap.
    lim = np.zeros(mk.shape, bool)
    for x0, y0, x1, y1, *_ in boxes:
        lim[y0:y1, x0:x1] = True
        if y0 >= 420: K[max(0, y0 - 1):y1 + 1, x1 - 1:x1 + 12] = True
    mk &= ~K
    # footer (y >= 420): its own fill, see FOOT_FILL
    foot = np.zeros(mk.shape, bool); foot[420:] = True
    m1 = erase_fill(pg, mk & ~foot, ~K, amount=0.6, ignore=K, limit=lim & ~foot, ring=4)
    m2 = erase_fill(pg, mk & foot, ~K, amount=0.6, ignore=K, limit=lim & foot, ring=4,
                    method=FOOT_FILL)
    return m1 | m2

# ---------------------------------------------------------------- text

def tw(text, path, size, sx=1.0):
    return font(path, max(1, int(round(size * 4)))).getlength(text) * sx / 4

def metrics(path, size):
    f = font(path, int(round(size * 4)))
    return f.getmetrics()[0] / 4

def draw(img, text, path, size, x, base_y, fill, outline=None, ow=1, shadow=None,
         soff=(3, 3), sx=1.0, shear=0.0, align='left', grad=None):
    """Draw one line with its baseline at base_y. x is the left edge, the
    centre (align='centre') or the right edge. shadow: RGB or ('mul', k)
    (bg darkened by k) under the outline shape. grad: (top, bottom) fill
    colours. Returns the ink box."""
    if shear:
        a, base, _ = render(text, path, size, shear=shear, sx=sx)
    else:
        a, base, _ = render(text, path, size, sx=sx)
    w = a.shape[1]
    lx = {'left': x, 'centre': x - w / 2, 'right': x - w}[align]
    lx = int(round(lx)); ly = int(round(base_y - base))
    shape = dilate(a, ow) if outline is not None else a
    o = ow if outline is not None else 0
    if shadow is not None:
        sxo, syo = lx - o + soff[0], ly - o + soff[1]
        if isinstance(shadow, tuple) and shadow[0] == 'mul':
            h_, w_ = shape.shape
            y0, x0 = max(0, syo), max(0, sxo)
            reg = img[y0:syo + h_, x0:sxo + w_]
            col = np.zeros(shape.shape + (3,))
            col[y0 - syo:y0 - syo + reg.shape[0], x0 - sxo:x0 - sxo + reg.shape[1]] = reg * shadow[1]
            over(img, sxo, syo, shape, col)
        else:
            over(img, sxo, syo, shape, C(shadow))
    if outline is not None:
        over(img, lx - o, ly - o, shape, C(outline))
    if grad is not None:
        t = np.linspace(0, 1, a.shape[0])[:, None, None]
        col = C(grad[0]) * (1 - t) + C(grad[1]) * t
        col = np.broadcast_to(col, a.shape + (3,))
        over(img, lx, ly, a, col)
    else:
        over(img, lx, ly, a, C(fill))
    return (lx - o, ly - o, lx + w + o, ly + a.shape[0] + o)

def wrap(text, path, size, wfn, sx=1.0):
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            if not cur or tw(t, path, size, sx) <= wfn(len(lines)): cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

def fit(text, path, h, wfn_of, smax, smin, pk=1.3):
    """Largest size whose wrapped lines fit. wfn_of(size, pitch) gives the
    width function for line i."""
    for size in np.arange(smax, smin - 0.01, -0.5):
        for sx in (1.0, 0.95, 0.9):
            pitch = max(round(size * pk), size + 2)
            wfn = wfn_of(size, pitch)
            lines = wrap(text, path, size, wfn, sx)
            if all(tw(l, path, size, sx) <= wfn(i) for i, l in enumerate(lines)) and \
               (len(lines) - 1) * pitch + size * 0.95 <= h:
                return size, sx, pitch, lines
    raise ValueError('does not fit: %r' % text[:40])

def block(img, text, path, box, smax, smin, fill, outline=BLACK, pk=1.3, wfn_of=None,
          align='left', valign='top', log=None, key=''):
    x0, y0, x1, y1 = box
    if wfn_of is None: wfn_of = lambda s, p: (lambda i: x1 - x0)
    size, sx, pitch, lines = fit(text, path, y1 - y0, wfn_of, smax, smin, pk)
    asc = size * 0.76            # cap height of Tahoma bold ~0.73 em
    used = (len(lines) - 1) * pitch + asc
    top = y0 + (y1 - y0 - used) / 2 if valign == 'middle' else y0
    for i, l in enumerate(lines):
        bl = top + asc + i * pitch
        if align == 'centre':
            draw(img, l, path, size, (x0 + x1) / 2, bl, fill, outline, sx=sx, align='centre')
        else:
            draw(img, l, path, size, x0, bl, fill, outline, sx=sx)
    if log is not None:
        log.append(f'{key}: size {size} sx {sx} pitch {pitch} lines {len(lines)} '
                   f'bottom {top + used + size * 0.25:.0f} (box {y1})')
    return lines

# ---------------------------------------------------------------- pieces

def header(pg, n, text, log):
    fill, k = HEAD[n]
    x0, cx, y0, y1 = HDR
    size = 33
    while tw(text, F_HEAD, size) > 376 and size > 20: size -= 0.5
    # JP glyphs fill y 36..66; EN caps sit on the same baseline
    draw(pg.img, text, F_HEAD, size, cx, 63, fill, BLACK, ow=2, shadow=('mul', k),
         soff=(3, 3), align='centre')
    log.append(f'hdr: size {size} w {tw(text, F_HEAD, size):.0f}')

# Band colours: cap colour at both ends, blending to the middle colour
# (JP: red caps with an orange-yellow middle; Part 3 teal with cyan).
BAND_COL = {1: ((192, 0, 0), (232, 144, 0)), 2: ((192, 0, 0), (232, 144, 0)),
            3: ((0, 192, 136), (0, 176, 224)), 4: ((192, 0, 0), (232, 144, 0))}

def band_cols(n, L, cap=14):
    c0, c1 = C(BAND_COL[n][0]), C(BAND_COL[n][1])
    out = []
    for i in range(L):
        d = min(i, L - 1 - i)
        t = np.clip((d - cap) / max(1, L / 2 - cap - 10), 0, 1)
        t = t * t * (3 - 2 * t)
        out.append(c0 * (1 - t) + c1 * t)
    return np.array(out)

def banner(pg, n, text, log):
    yt, yb, xl = BAND[n]
    size = 29.0; sx = 1.0
    # band = 14 px lead, the text's ink (italic overhang and 2 px rim
    # included), 22 px tail (6 px of full band, then the 16 px cap)
    def ink(size, sx):
        return render(text, F_BAN, size, sx=sx)[0].shape[1] + 4
    while 14 + ink(size, sx) + 22 > BAN_MAXW[n]:
        if sx > 0.86: sx -= 0.02
        else: size -= 0.5
    w = ink(size, sx)
    L = 14 + w + 22
    np_ = band_cols(n, L)
    x0 = xl
    hh = (yb - yt) / 2; cy = (yt + yb) / 2
    img = pg.img
    for i, col in enumerate(np_):
        x = x0 + i
        # pointed caps: 0.5 px per row
        d = min(i, len(np_) - 1 - i)
        half = min(hh + 0.5, d * 2.0 + 0.5)
        for y in range(yt, yb + 1):
            a = np.clip(half - abs(y - cy) + 0.5, 0, 1)
            if a > 0: img[y, x] = img[y, x] * (1 - a) + col * a
    base = yb + 3
    draw(img, text, F_BAN, size, x0 + 16, base, BLUE, WHITE, ow=2, sx=sx)
    log.append(f'banner: size {size} sx {sx:.2f} w {w:.0f} band {x0}..{x0 + len(np_)}')

def label(img, text, size, x, base, fill, outline, align='left', path=None, ow=1, grad=None,
          maxw=999):
    path = path or F_BODY
    sx = min(1.0, maxw / tw(text, path, size))
    return draw(img, text, path, size, x, base, fill, outline, ow=ow, align=align, grad=grad, sx=sx)

def back(pg, n, text, log):
    cx = 365.5 if n == 4 else 309.5
    k = HEAD[n][1]
    size = 21.0
    a, _, _ = render(text, F_FOOT, size)
    while a.shape[0] > 17: size -= 0.5; a, _, _ = render(text, F_FOOT, size)
    sx = min(1.0, 27 / a.shape[1])
    draw(pg.img, text, F_FOOT, size, cx, 442.5, RED, None, shadow=('mul', min(0.8, k + 0.15)),
         soff=(2, 2), sx=sx, align='centre')
    log.append(f'back: size {size} sx {sx:.2f}')

# grey actions: light top, darker bottom, 2 px black rim (JP: heavy dark rim)
ACT = dict(grad=((236, 236, 236), (196, 196, 196)), ow=2)

# ---------------------------------------------------------------- pages

def page0(pg, tl, log):
    # labels centred on x 256 in the panel; JP glyphs y 118..140 / 198..220
    for key, base in (('jb_wallet', 139), ('jb_deposit', 219)):
        size = 28                  # caps ~20 px, as J-BANK (y 318..340)
        draw(pg.img, tl[key], F_JB, size, 256.5, base, NAVY, None, align='centre')
        log.append(f'{key}: w {tw(tl[key], F_JB, size):.0f}')

def page1(pg, tl, log):
    im = pg.img
    header(pg, 1, tl['hdr'], log)
    banner(pg, 1, tl['p1_title'], log)
    block(im, tl['p1_body'], F_BODY, (40, 121, 474, 236), 16, 12, WHITE, pk=1.25, log=log, key='body')
    block(im, tl['p1_note'], F_BODY, (40, 246, 420, 294), 16, 12, YELLOW, log=log, key='note')
    rows = (331, 351, 371, 391)
    for i, base in enumerate(rows):
        label(im, tl[f'p1_b{i + 1}'], 15, 206, base, GREEN_T, WHITE, ow=2)
        label(im, tl[f'p1_a{i + 1}'], 15, 284, base, GREY, BLACK, **ACT)
    back(pg, 1, tl['back'], log)

def page2(pg, tl, log):
    im = pg.img
    header(pg, 2, tl['hdr'], log)
    banner(pg, 2, tl['p2_title'], log)
    block(im, tl['p2_body'], F_BODY, (40, 120, 488, 196), 16, 12, WHITE, log=log, key='body')
    block(im, tl['p2_note'], F_BODY, (40, 206, 488, 282), 16, 12, YELLOW, log=log, key='note')
    # the red line starts at x 424, y 326-327: the button label ends just
    # left of it with its caps centred on the line, the action under it
    a = label(im, tl['p2_b1'], 15, 420, 332, GREEN_T, WHITE, align='right', ow=2)
    label(im, tl['p2_a1'], 16, (a[0] + a[2]) / 2, 353, GREY, BLACK, align='centre', **ACT)
    back(pg, 2, tl['back'], log)

def page3(pg, tl, log):
    im = pg.img
    header(pg, 3, tl['hdr'], log)
    banner(pg, 3, tl['p3_title'], log)
    # lines beside the LCD shots (x 220..296, y < 302) stop at x 214
    def wfn_of(size, pitch):
        return lambda i: (214 - 36) if 124 + i * pitch + size < 304 else (300 - 36)
    block(im, tl['p3_body'], F_BODY, (36, 124, 318, 372), 16, 12, WHITE, wfn_of=wfn_of,
          pk=1.32, log=log, key='body')
    block(im, tl['p3_note'], F_BODY, (36, 374, 322, 416), 15, 11, YELLOW, valign='middle',
          log=log, key='note')
    # the Down line ends at (443, 169): one label line right under it
    label(im, tl['p3_b1'].replace('|', ' / '), 15, 443, 184, PURPLE, WHITE, align='centre', ow=2)
    label(im, tl['p3_a1'], 17, 443, 205, GREY, BLACK, align='centre', **ACT)
    # Left: line end y ~220; Up: hook y 292; Right: hook y 380 (at the action)
    for k, (yb, ya) in zip((2, 3, 4), ((236, 257), (298, 319), (366, 387))):
        label(im, tl[f'p3_b{k}'], 15, 330, yb, PURPLE, None)
        label(im, tl[f'p3_a{k}'], 17, 330, ya, GREY, BLACK, **ACT)
    back(pg, 3, tl['back'], log)

def page4(pg, tl, log):
    im = pg.img
    header(pg, 4, tl['hdr'], log)
    banner(pg, 4, tl['p4_title'], log)
    label(im, tl['p4_b1'], 15, 45, 140, PURPLE, None)
    label(im, tl['p4_a1'], 16, 45, 162, GREY, BLACK, **ACT)
    label(im, tl['p4_b2'], 15, 184, 178, PURPLE, None, align='centre')
    label(im, tl['p4_a2'], 16, 180, 200, GREY, BLACK, align='centre', maxw=112, **ACT)
    a = label(im, tl['p4_b3'], 15, 128, 266, PURPLE, None)
    label(im, tl['p4_a3'], 16, (a[0] + a[2]) / 2, 288, GREY, BLACK, align='centre', **ACT)
    block(im, tl['p4_body1'], F_BODY, (240, 126, 492, 190), 16, 12, WHITE, log=log, key='body1')
    block(im, tl['p4_body2'], F_BODY, (240, 196, 492, 299), 16, 12, WHITE, log=log, key='body2')
    block(im, tl['p4_note'], F_BODY, (18, 308, 314, 420), 15, 11, YELLOW, log=log, key='note')
    back(pg, 4, tl['back'], log)

PAGES = {0: page0, 1: page1, 2: page2, 3: page3, 4: page4}

def paint(d, n, tl, log):
    pg = Page(d, 0)
    erase(pg, ERASE[n])
    if n: PAGES[n](pg, tl, log)
    else: PAGES[0](pg, tl, log)
    return pg.store(d)

# ---------------------------------------------------------------- main

def main(argv):
    png = '--png' in argv; check = '--check' in argv
    os.chdir(ROOT)
    tl = load_tl()
    src = open(ARC, 'rb').read()
    toc = arc.toc(src)
    assert len(toc) == 5, toc
    out = bytearray(0x800); pos = 0x800; problems = []
    if png: os.makedirs(PREV, exist_ok=True)
    for n, (o, sz) in enumerate(toc):
        u = lz.decompress(src[o:o + sz])[0]
        d = bytearray(u)
        jp = Page(bytes(u), 0).img
        log = []
        en = paint(d, n, tl, log)
        assert len(d) == len(u)
        c = lz.compress(bytes(d), name=NAME)
        if lz.decompress(c)[0][:len(d)] != bytes(d): problems.append(f'{n}: LZ round trip failed')
        struct.pack_into('<II', out, 8 * n, pos, len(c))
        out += c + bytes(-len(c) % 0x800); pos = len(out)
        print(f'page {n}: {len(c)} B (JP {sz})' + ''.join('\n  ' + l for l in log))
        if png:
            s = np.zeros((PH, PW * 2, 3), np.uint8)
            s[:, :PW] = np.clip(jp, 0, 255); s[:, PW:] = np.clip(en, 0, 255)
            Image.fromarray(s).save(f'{PREV}/p{n}.png')
            Image.fromarray(np.clip(en, 0, 255).astype(np.uint8)).save(f'{PREV}/en{n}.png')
    if len(out) > len(src): problems.append(f'archive {len(out)} > JP {len(src)}')
    out += bytes(len(src) - len(out))
    print(f'PDATIM: {pos} B used of {len(src)} ({(len(src) - pos) // 0x800} sectors free); '
          f'problems {len(problems)}' + ''.join('\n  ' + p for p in problems))
    if not check and not problems:
        open(OUT, 'wb').write(out)
        print('wrote', OUT)
    return 1 if problems else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
