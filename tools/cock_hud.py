# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary hub header and status page labels in DAT2/MGCOCK.TIZ.

Called by tools/build_cock.py as apply(tim, mgame): tim.px is the 256x1024
index sheet (VRAM 512,256, tpages 512/576/640/704), tim.clut the 96x16
palettes, mgame a bytearray of MGAME.BIN (file offset = RAM - 0x800a2000).
Wording: tl/COCK.txt, section "hud labels" (keys ROLE:JP). RE facts:
private/re/cock/findings.org.

- Date header "4/16 Fri": the 9 date sprites (16x24 each, own DR_TPAGE)
  are set up by MGAME 0x800b4c70. Its digit block (0x800b504c-0x800b5177)
  is rewritten: digits as before, 月 sprite = "/", 日 ( ) = blank, the
  weekday sprite switches to tpage 576 (strip v216-239) with its own
  width, and the x table 0x800a3bf4 is recomputed per date so the line is
  centred on the panel and has no gap for a missing tens digit. Weekday
  table 0x800a3be4 holds (u, width) per day.
- Stat names (48x16, two uv tables), club names (64x20), school names,
  character names (60x20): redrawn in their cells, centred (stats left
  aligned, ending 3 px or more before the gauge at cell x 48; clubs
  centred on the slot between the badge and the frame).
- Status page tabs and birthday line (records 0x800a4f98, xy 0x800a4fb0):
  "Status", "Items", "Friendship"; "Male"/"Female" (female offset is the
  immediate at 0x800c64ac); "Born M/D": 日生まれ = "Born" left of the
  digits, 月 = "/". The digit block of 0x800c5cec (0x800c65fc-0x800c66ff)
  is rewritten: a day below 10 is drawn in the tens slot, a month below
  10 moves "Born" right by one digit pitch.
Fonts (all OFL 1.1, hinted 1x): Antonio (stats wght 600, clubs and save
list school names 700; per-glyph layout, no anti-aliasing), Barlow Condensed
Bold (status page), Roboto Condensed (date).
"""
import os, re, struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook import Asm, mul_const

TL = 'tl/COCK.txt'
FONT = 'RobotoCondensed-Bold.ttf'
BARLOW = 'BarlowCondensed-Bold.ttf'
MEDIUM = 'RobotoCondensed-Medium.ttf'
# group: (font, size, baseline row in the cell). Labels shown together
# (names, tabs) share one size. Stat and school names are Antonio
# (STAT_ANT, SCHOOL_ANT).
G = {'wd': (MEDIUM, 22, 21), 'slash': (FONT, 30, 21),
     'tab': (BARLOW, 23, 18), 'tab2': (BARLOW, 19, 15), 'sex': (BARLOW, 20, 16), 'born': (BARLOW, 20, 16),
     'bslash': (BARLOW, 22, 17), 'name': (BARLOW, 20, 15)}
B = 0x800a2000                       # MGAME load address

# Sheet rectangles this module may write (sheet x, y, w, h). Sheet x =
# (tpage - 512) * 4 + u.
SPACE = [(160, 0, 96, 24), (0, 24, 48, 24),            # date marks, weekdays
         (0, 48, 192, 16), (80, 32, 96, 16),             # stat names
         (192, 48, 64, 80), (128, 64, 64, 60), (64, 64, 64, 40),   # clubs
         (512, 0, 224, 40),                              # school names
         (512, 40, 248, 20),                             # tabs
         (512, 60, 240, 100),                            # character names
         (512, 160, 140, 20),                            # sex, birthday words
         (256, 216, 256, 24), (400, 200, 112, 16), (416, 240, 96, 16)]


def col(tp, u): return (tp - 512) * 4 + u


def load_tl(path=TL):
    out, k = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): k = line[4:]
        elif line.startswith('EN: ') and k is not None:
            if ':' in k and k not in out: out[k] = line[4:]
            k = None
    return out


# ------------------------------------------------------------- rendering
def cover(text, size, base, h, track=0.0, font=FONT):
    """Hinted 1x coverage (h, w) in 0..1, baseline at row base, cropped to
    the ink columns. track adds px per letter (rounded per glyph)."""
    f = fontpath.truetype(font, size)
    W = int(f.getlength(text) + abs(track) * len(text)) + 12
    im = Image.new('L', (W, h), 0)
    if not track:
        ImageDraw.Draw(im).text((4, base), text, font=f, fill=255, anchor='ls')
    else:
        a = np.zeros((h, W))
        for i, ch in enumerate(text):
            if ch == ' ': continue
            x = f.getlength(text[:i + 1]) - f.getlength(ch) + i * track
            g = Image.new('L', (W, h), 0)
            ImageDraw.Draw(g).text((4 + round(x), base), ch, font=f, fill=255, anchor='ls')
            a = np.maximum(a, np.asarray(g))
        im = Image.fromarray(a.astype(np.uint8))
    a = np.asarray(im) / 255.0
    cols = np.where((a > 0.15).any(0))[0]
    return a[:, cols[0]:cols[-1] + 1]


def shift(m, dx, dy):
    o = np.zeros_like(m)
    h, w = m.shape
    o[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return o


def style(a, core, mid, outline=None, shadow=None, conn=8, sh=1, lpad=None):
    """Palette indices for coverage a: core (index, or a function of the
    column -> index), mid for anti-aliased pixels, a 1 px outline, and a
    drop shadow sh px to the lower right. Columns are padded for both."""
    lp = (1 if outline is not None else 0) if lpad is None else lpad
    rp = (1 if outline is not None else 0) + (sh if shadow is not None else 0)
    A = np.pad(a, ((0, 0), (lp, rp)))
    ink = A > 0.5
    body = A > 0.28
    out = np.zeros(A.shape, np.uint8)
    if shadow is not None:
        s = body.copy()
        if outline is not None:
            s = ndimage.binary_dilation(body, np.ones((3, 3), bool) if conn == 8 else None)
        out[shift(s, sh, sh)] = shadow
    if outline is not None:
        o = ndimage.binary_dilation(body, np.ones((3, 3), bool) if conn == 8 else None)
        out[o] = outline
    out[body & ~ink] = mid
    if callable(core):
        for x in range(A.shape[1]):
            c = ink[:, x]
            out[c, x] = core(x / max(1, A.shape[1] - 1), np.where(c)[0])
    else:
        out[ink] = core
    return out


def engraved(a):
    """Date style (palette 0): text pressed into the panel. Core 7, dark
    left edge 1, lighter right edge 8, light rim e/f below and right."""
    A = np.pad(a, ((0, 0), (1, 2)))
    core = A > 0.45
    part = (A > 0.18) & ~core
    out = np.zeros(A.shape, np.uint8)
    rim_r = shift(core, 1, 0) & ~core
    rim_b = shift(core, 0, 1) & ~core
    out[rim_b] = 14
    out[rim_r] = 15
    out[part] = 4
    out[core] = 7
    out[core & ~shift(core, 1, 0)] = 1          # left edge
    out[core & ~shift(core, -1, 0)] = 8         # right edge
    return out


def put(px, x, y, img, w=None, h=None, align='centre'):
    """Clear the cell (w x h) and place img in it (centred or left)."""
    w = w or img.shape[1]; h = h or img.shape[0]
    if img.shape[1] > w or img.shape[0] > h:
        raise SystemExit(f'cock_hud: piece {img.shape[1]}x{img.shape[0]} > cell {w}x{h} at {x},{y}')
    px[y:y + h, x:x + w] = 0
    dx = (w - img.shape[1]) // 2 if align == 'centre' else 0
    px[y:y + img.shape[0], x + dx:x + dx + img.shape[1]] = img
    return dx


def gcover(text, g, h, track=0.0):
    font, size, base = G[g]
    return cover(text, size, base, h, track, font)


def fit_group(texts, g, h, maxw, style_fn, track_min=-0.5, log=False):
    """One size for a group: the largest size up to G[g] at which every
    text fits maxw (int or list) with tracking down to track_min."""
    font, size, base = G[g]
    ws = maxw if isinstance(maxw, list) else [maxw] * len(texts)
    while True:
        out = []
        for t, w in zip(texts, ws):
            tr = 0.0
            while True:
                img = style_fn(cover(t, size, base, h, tr, font))
                if img.shape[1] <= w or tr <= track_min + 1e-6: break
                tr = round(tr - 0.1, 1)
            out.append((img, tr))
        if all(i.shape[1] <= w for (i, _), w in zip(out, ws)) or size <= 9:
            break
        size -= 1
    if log and (size != G[g][1] or any(tr for _, tr in out)):
        print(f'cock_hud: {g} {texts[0] if len(texts) == 1 else ""} size {size}, tracking',
              {t: tr for t, (_, tr) in zip(texts, out) if tr})
    return [i for i, _ in out]


def fit(text, g, h, maxw, style_fn, track_min=-0.8):
    """Render, tracking tighter in -0.1 steps until the styled piece fits."""
    tr = 0.0
    while True:
        img = style_fn(gcover(text, g, h, tr))
        if img.shape[1] <= maxw or tr <= track_min + 1e-6: return img, tr
        tr = round(tr - 0.1, 1)


# ------------------------------------------------------------- date
DATE_CODE, DATE_END = 0x800b504c, 0x800b5178
DATE_HEAD = [0x3c02801f, 0x90421855, 0x3c05cccc, 0x34a5cccd]
WD_TAB, MARK_TAB, X_TAB = 0x800a3be4, 0x800a3bd0, 0x800a3bf4
WD_V = 216                            # tpage 576 strip v216-239
WD_ORDER = '日月火水木金土'             # Sun..Sat (0x801f1857)
SLASH_UV, BLANK_UV = (160, 0), (176, 0)
# x layout (screen x - 256): digit pitch, slash advance, gap before the
# weekday, centre (screen 99). The panel face is x 13-191; the dark corner
# triangle (frame, x 177-184 over the text rows) narrows it to 13-182
# there. 99 puts the ink centre at 99-100.5 for all dates, between the
# two (topic hubhdr).
D_P, D_S, D_G, D_C = 15, 13, 6, 99 - 256


def date_code(K):
    a = Asm(DATE_CODE)
    a.lui('t0', 0x801f)
    a.lbu('a0', 0x1855, 't0')          # month
    a.lbu('a1', 0x1856, 't0')          # day
    a.lbu('t2', 0x1857, 't0')          # weekday
    a.addiu('t1', 'zero', 10)
    a.divu('a0', 't1')
    a.mflo('t3')                       # month tens
    a.r(0, 'zero', 'zero', 't5', 0, 0x10)   # mfhi t5: month ones
    a.sll('v0', 't3', 4)
    a.sh('v0', 0x424, 's2')            # u = 16 * digit, v = 0
    a.sll('v0', 't5', 4)
    a.divu('a1', 't1')
    a.sh('v0', 0x45c, 's2')
    a.mflo('t4')                       # day tens
    a.r(0, 'zero', 'zero', 't6', 0, 0x10)   # mfhi t6: day ones
    a.sll('v0', 't4', 4)
    a.sh('v0', 0x494, 's2')
    a.sll('v0', 't6', 4)
    a.sh('v0', 0x4cc, 's2')
    a.lui('t7', 0x800a)                # weekday: (u, w) table, v fixed
    a.sll('v0', 't2', 1)
    a.addu('v0', 't7', 'v0')
    a.lbu('v1', WD_TAB & 0xffff, 'v0')
    a.lbu('t8', (WD_TAB & 0xffff) + 1, 'v0')
    a.addiu('v0', 'zero', WD_V)
    a.sb('v1', 0x504, 's2')
    a.sb('v0', 0x505, 's2')
    a.sh('t8', 0x508, 's2')            # sprite w
    a.lw('v0', 0x4f4, 's2')            # DR_TPAGE word
    a.nop()
    a.addiu('v0', 'v0', 1)             # tpage 512 -> 576
    a.sw('v0', 0x4f4, 's2')
    a.sltu('t3', 'zero', 't3')         # has month tens
    a.sltu('t4', 'zero', 't4')         # has day tens
    a.addu('v0', 't3', 't4')
    mul_const(a, 'v1', 'v0', D_P, 't9')
    a.addu('v0', 'v1', 't8')
    a.addiu('v0', 'v0', K)
    a.sra('v0', 'v0', 1)
    a.addiu('a3', 'zero', D_C)
    a.subu('a3', 'a3', 'v0')           # x of the first piece
    x = X_TAB & 0xffff
    a.sh('a3', x + 8, 't7')            # month tens
    mul_const(a, 'v1', 't3', D_P, 't9')
    a.addu('a3', 'a3', 'v1')
    a.sh('a3', x + 10, 't7')           # month ones
    a.addiu('a3', 'a3', D_P)
    a.sh('a3', x + 0, 't7')            # "/"
    a.addiu('a3', 'a3', D_S)
    a.sh('a3', x + 12, 't7')           # day tens
    mul_const(a, 'v1', 't4', D_P, 't9')
    a.addu('a3', 'a3', 'v1')
    a.sh('a3', x + 14, 't7')           # day ones
    a.addiu('a3', 'a3', D_P + D_G)
    a.sh('a3', x + 16, 't7')           # weekday
    a.r(0, 'zero', 'zero', 's3', 0, 0x21)   # move s3, zero
    a.r(0, 's2', 'zero', 'a2', 0, 0x21)     # move a2, s2
    code = a.finish()
    n = (DATE_END - DATE_CODE)
    assert len(code) <= n, (len(code), n)
    return code + bytes(n - len(code))


def build_date(tl, px, mg, log):
    o = DATE_CODE - B
    if list(struct.unpack_from('<4I', mg, o)) != DATE_HEAD:
        raise SystemExit('cock_hud: date code is not the original (run hook.py first)')
    for x, y, w, h in ((160, 0, 96, 24), (0, 24, 48, 24)):
        px[y:y + h, x:x + w] = 0
    # "/" in the old 月 cell
    sl = engraved(gcover(tl['日付:月'], 'slash', 24))
    put(px, SLASH_UV[0], SLASH_UV[1], sl, 16, 24)
    # weekdays packed into tpage 576 v216-239
    u = 0
    tab = []
    pieces = []
    for k, jp in enumerate(WD_ORDER):
        img = engraved(gcover(tl['曜日:' + jp], 'wd', 24))
        w = img.shape[1]
        if u + w > 256: raise SystemExit('cock_hud: weekdays do not fit v216-239')
        px[WD_V:WD_V + 24, col(576, u):col(576, u) + w] = img
        tab += [u, w]; pieces.append(img); u += w
    struct.pack_into('14B', mg, WD_TAB - B, *tab)
    struct.pack_into('8B', mg, MARK_TAB - B, *SLASH_UV, *BLANK_UV, *BLANK_UV, *BLANK_UV)
    K = 2 * D_P + D_S + D_G
    mg[o:o + DATE_END - DATE_CODE] = date_code(K)
    if log: print(f'cock_hud: weekdays {tab[1::2]} px, strip {u}/256')
    return pieces, sl


def date_layout(m, d, wdw):
    """Screen x of (month tens, month ones, slash, day tens, day ones,
    weekday) as the patched code computes them (None = not drawn)."""
    mt, dt = m // 10, d // 10
    tot = D_P * ((mt > 0) + (dt > 0)) + wdw + 2 * D_P + D_S + D_G
    x = D_C - (tot >> 1)
    r = [x if mt else None]
    x += D_P * (mt > 0); r.append(x); x += D_P; r.append(x); x += D_S
    r.append(x if dt else None); x += D_P * (dt > 0); r.append(x); x += D_P + D_G
    r.append(x)
    return [None if v is None else v + 256 for v in r]


# ------------------------------------------------------------- stats
STAT_TABS = (0x800a4254, 0x800a516c)
STAT_MAXW = 45                        # gauge at cell x 48, 3 px gap
STAT_JP = ['攻撃力', '防御力', '体力', '根性', '潜在能力', '知力']


def stat_style(a):
    """As the JP labels (palette 2): white face e, a 1 px magenta (6) edge
    right of and below each stroke, dark purple shadow 3 one further down
    and right. No anti-aliasing: grey-level pixels on both sides of the
    strokes read as blur. The edge is left out where it would close a
    1 px gap to the next stroke (the "tt" crossbars), so the shadow
    separates them. One blank column on the left."""
    core = np.pad(a, ((0, 0), (0, 2))) > 0.45
    ext = ((shift(core, 1, 0) & ~shift(core, -1, 0)) |
           (shift(core, 0, 1) & ~shift(core, 0, -1))) & ~core
    out = np.zeros(core.shape, np.uint8)
    out[shift(core | ext, 1, 1) & ~core & ~ext] = 3
    out[ext] = 6
    out[core] = 14
    cols = np.where(out.any(0))[0]
    return np.pad(out[:, cols[0]:cols[-1] + 1], ((0, 0), (1, 0)))


# Stat and school names in Antonio, set as the club names (per-glyph 1x
# render, binarised, set by ink; topic hubhdr). cap = rows of "H" (11
# renders as 12), base = baseline row in the cell, kern see ant_mask. A
# group that does not fit drops weight in steps of 20. Stats: 600 (1 px
# stems at this size; 700 has 2 px stems and with the edge read heavy).
STAT_ANT = dict(wght=600, cap=12, gap=2, space=2, base=13, kern=1)
SCHOOL_ANT = dict(wght=700, cap=14, gap=2, space=3, base=16)


def ant_cover(text, p, base, h, wght):
    """Antonio mask for a cell h rows high with the baseline at row base,
    as 0/1 coverage for the style functions."""
    m = ant_mask(text, wght, p['cap'], p['gap'], p['space'], p.get('kern'))
    assert not m[:20 - base].any(), (text, 'ink above the cell')
    assert not m[20 - base + h:].any(), (text, 'ink below the cell')
    return m[20 - base:20 - base + h].astype(float)


def ant_fit(texts, p, base, h, maxw, style_fn, log=False):
    """One weight for the group: the heaviest, in steps of 20 from
    p['wght'] down to 600, at which every styled text fits maxw."""
    ws = maxw if isinstance(maxw, list) else [maxw] * len(texts)
    for w in range(p['wght'], 590, -20):
        imgs = [style_fn(ant_cover(t, p, base, h, w)) for t in texts]
        if all(i.shape[1] <= m for i, m in zip(imgs, ws)): break
    bad = [t for t, i, m in zip(texts, imgs, ws) if i.shape[1] > m]
    if bad: raise SystemExit(f'cock_hud: {bad} too wide in Antonio')
    if log and w != p['wght']: print(f'cock_hud: {texts} Antonio weight {w}')
    return imgs


def stat_pieces(tl, log=False):
    texts = [tl['能力:' + jp] for jp in STAT_JP]
    return ant_fit(texts, STAT_ANT, STAT_ANT['base'], 16, STAT_MAXW, stat_style, log)


def build_stats(tl, px, mg, log):
    uv = [tuple(mg[STAT_TABS[0] - B + 2 * k:STAT_TABS[0] - B + 2 * k + 2]) for k in range(6)]
    assert uv == [tuple(mg[STAT_TABS[1] - B + 2 * k:STAT_TABS[1] - B + 2 * k + 2]) for k in range(6)]
    out = stat_pieces(tl, log)
    for k, img in enumerate(out):
        u, v = uv[k]
        put(px, col(512, u), v, img, 48, 16, align='left')
    return out


# ------------------------------------------------------------- clubs
CLUB_TAB = 0x800a3c74
CLUB_JP = ['野球部', 'バレー部', 'サッカー部', '相撲部', '美術部', '吹奏楽部',
           '演劇部', '化学部', '帰宅部']


# The cell is drawn at screen x 124 (64x20). The slot it sits in runs from
# the school badge (ends x 120) to the panel frame (starts x 188): x
# 121-187, centre 154. JP ink filled 124-184 (61 px). EN ink (incl.
# shadow) is at most 62 px; only Brass Band needs that much.
CLUB_X, CLUB_SLOT_C, CLUB_MAXW = 124, 154, 62
# Antonio (OFL 1.1) at cap height 14, baseline row 15. Each glyph is
# rendered hinted at 1x and binarised on its own, so a letter has the
# same pixels wherever it appears. Glyphs are then set by their ink:
# the closest approach between neighbours (rows within +-1) is CLUB_GAP
# empty columns, which the left letter's shadow fills. A space adds
# CLUB_SPACE columns. A name wider than CLUB_MAXW drops weight in steps.
CLUB_FONT = 'Antonio[wght].ttf'
CLUB_WGHT, CLUB_CAP, CLUB_BASE, CLUB_GAP, CLUB_SPACE = 700, 14, 15, 1, 2


_GLYPHS = {}


def ant_glyph(ch, wght, cap=None):
    """Antonio glyph at weight wght, sized so "H" is cap rows (default
    CLUB_CAP) -> binary mask (28 rows, baseline row 20) cropped to its ink."""
    cap = cap or CLUB_CAP
    key = (ch, wght, cap)
    if key in _GLYPHS: return _GLYPHS[key]
    for s10 in range(80, 300, 5):
        f = fontpath.truetype(CLUB_FONT, s10 / 10)
        f.set_variation_by_axes([wght])
        im = Image.new('L', (60, 28), 0)
        ImageDraw.Draw(im).text((10, 20), 'H', font=f, fill=255, anchor='ls')
        if (np.asarray(im) >= 128).any(1).sum() >= cap: break
    im = Image.new('L', (60, 28), 0)
    ImageDraw.Draw(im).text((10, 20), ch, font=f, fill=255, anchor='ls')
    m = np.asarray(im) >= 128
    xs = np.where(m.any(0))[0]
    _GLYPHS[key] = m[:, xs[0]:xs[-1] + 1]
    return _GLYPHS[key]


def ant_mask(text, wght, cap=None, gap=None, space=None, kern=None):
    """Set Antonio glyphs by ink: the closest approach between neighbours
    (rows within +-1) is gap empty columns; a space adds space columns.
    kern: a pair whose ink comes closer than gap columns on at most kern
    rows of the same row at gap - 1 (a crossbar or a diagonal only, as
    "tt", "ti", "ot") is set one column tighter. -> mask (28 rows,
    baseline 20)."""
    gap = CLUB_GAP if gap is None else gap
    space = CLUB_SPACE if space is None else space
    out = np.zeros((28, 10 * len(text) + 8), bool)
    prev, pend = None, 0
    for ch in text:
        if ch == ' ':
            pend += space
            continue
        m = ant_glyph(ch, wght, cap)
        x = 0
        if prev is not None:
            pm, px = prev
            for a in range(28):
                ra = np.where(pm[a])[0]
                if not len(ra): continue
                for b in range(max(a - 1, 0), min(a + 2, 28)):
                    lb = np.where(m[b])[0]
                    if len(lb): x = max(x, px + ra[-1] + gap + 1 - lb[0])
            x = max(x, px + pm.shape[1] + gap - 1)
            if kern is not None and not pend and gap > 1:
                t = x - 1                     # one column tighter
                close = 0
                for a in range(28):
                    ra, lb = np.where(pm[a])[0], np.where(m[a])[0]
                    if len(ra) and len(lb) and t + lb[0] - (px + ra[-1]) - 1 < gap:
                        close += 1
                ok = all(t + lb[0] - (px + ra[-1]) - 1 >= gap - 1
                         for a in range(28) for b in range(max(a - 1, 0), min(a + 2, 28))
                         for ra, lb in [(np.where(pm[a])[0], np.where(m[b])[0])]
                         if len(ra) and len(lb))
                if ok and close <= kern: x = t
            x += pend
        out[:, x:x + m.shape[1]] |= m
        prev, pend = (m, x), 0
    xs = np.where(out.any(0))[0]
    return out[:, :xs[-1] + 1]


def club_glyph(ch, wght): return ant_glyph(ch, wght, CLUB_CAP)


def club_mask(text, wght): return ant_mask(text, wght, CLUB_CAP, CLUB_GAP, CLUB_SPACE)


def club_style(m):
    """As the JP labels (palette 50): red-orange face f with e on the right
    column of each stroke, near-black drop shadow 2 (1-2 px down and
    right), no outline and no anti-aliasing. m: club_mask output."""
    core = np.pad(m[20 - CLUB_BASE:40 - CLUB_BASE], ((0, 0), (0, 2)))
    assert not m[40 - CLUB_BASE - 2:].any(), 'club name: descender + shadow below the cell'
    assert not m[:20 - CLUB_BASE].any(), 'club name: ink above the cell'
    sh = shift(core, 1, 1) | shift(core, 2, 2) | shift(core, 2, 1) | shift(core, 1, 2)
    out = np.zeros(core.shape, np.uint8)
    out[sh & ~core] = 2
    out[core] = 15
    out[core & ~shift(core, -1, 0)] = 14
    return out


def club_piece(text, log=False):
    for w in range(CLUB_WGHT, 590, -20):
        img = club_style(club_mask(text, w))
        if img.shape[1] <= CLUB_MAXW: break
    assert img.shape[1] <= CLUB_MAXW, (text, img.shape[1])
    if log and w != CLUB_WGHT: print(f'cock_hud: club {text} weight {w}')
    return img


def build_clubs(tl, px, mg, log):
    out = [club_piece(tl['部:' + jp], log) for jp in CLUB_JP]
    for k, img in enumerate(out):
        u, v = mg[CLUB_TAB - B + 2 * k], mg[CLUB_TAB - B + 2 * k + 1]
        x = col(512, u)
        dx = round(CLUB_SLOT_C - (img.shape[1] - 1) / 2) - CLUB_X
        assert 0 <= dx and dx + img.shape[1] <= 64
        px[v:v + 20, x:x + 64] = 0
        px[v:v + 20, x + dx:x + dx + img.shape[1]] = img
    return out


# ------------------------------------------------------------- schools
SCHOOL_REC = 0x800a4600
SCHOOL_JP = ['太陽学園', '五輪高校', 'パシフィックHS', '外道高校', 'ジャスティス学園']


def school_style(a):
    # Save list rows (blue cursor bar, dark green rows). Palette 51: 4 pale
    # (192,224,232), 14 (0,224,168), 15 (0,216,96) as a top-down fill, 11
    # for edge pixels, 5 (0,72,56) outline and shadow. Indices 2, 3, 8, 9
    # have the STP bit and are avoided. The old light shadow (1) and pale
    # edge pixels (4) read as a blur.
    def fill(xf, ys):
        return np.where(ys < 8, 4, np.where(ys < 12, 14, 15))
    return style(a, fill, 11, outline=5, shadow=5)


def school_piece(text, maxw, log=False):
    """One school label, styled, at most maxw wide (also used by
    build_mgtit for the title Continue list)."""
    return ant_fit([text], SCHOOL_ANT, SCHOOL_ANT['base'], 20, maxw, school_style, log)[0]


def build_schools(tl, px, mg, log):
    recs = [mg[SCHOOL_REC - B + 4 * k:SCHOOL_REC - B + 4 * k + 4] for k in range(5)]
    imgs = [school_piece(tl['学校:' + jp], r[2], log) for jp, r in zip(SCHOOL_JP, recs)]
    for (u, v, w, _), img in zip(recs, imgs):
        put(px, col(640, u), v, img, w, 20)


# ------------------------------------------------------------- status page
REC, REC_XY = 0x800a4f98, 0x800a4fb0          # (u, v, w, pal) x 6; (x, y) x 6
DIG_XY = 0x800a4fdc                           # birthday digits (x, y) x 4
FEMALE_IMM = 0x800c64ac                       # addiu v0,v0,0x14
BDAY_CODE, BDAY_END = 0x800c65fc, 0x800c6700
BDAY_HEAD = [0x24020001, 0x12420012, 0x2a420002]
TAB_JP = ['ステータス', 'アイテム', '友好度']
# Status and Items are the page tabs (no descenders, larger); Friendship
# heads the lower panel and must end before the first name column (x 96).
# Placement per label: ('c', centre x) or ('l', left x), top y.
# Status: centred on its bar (screen x 0-127). Items: left edge on the item
# list column of the magenta panel (x 316-511, icons from x 320).
# Friendship: centred in the lower panel's legend column, from the panel
# edge (x 16) to the first name cell (x 96); 2 px margin each side.
TAB_POS = [('c', 64, 40), ('l', 320, 40), ('c', 56, 186)]
TAB2_MAXW = 76
# Birthday line: centre of "Born M/DD" (1-digit month), digit pitch, gap.
BD_C, BD_Y, BD_P, BD_GAP, BD_SL = 200, 96, 16, 5, 12
# Sex label centred under the family name field (default Seigi, x 25-71,
# centre 48), as the birthday sits under the given name.
SEX_C, SEX_Y = 49, 96
NAME_TAB = 0x800a4f20
NAME_JP = ['バツ', 'ひなた', '恭介', 'ラン', '将馬', '夏', 'ロベルト', '流', 'ロイ',
           'ティファニー', 'ボーマン', '隼人', 'エッジ', 'アキラ', '岩', '醍醐', '英雄',
           '響子', '雹']


def tab_status_style(a, px):
    """Palette 24: the core runs through the original's colours left to
    right (red, yellow, cyan, blue, pink), dark outline 1."""
    src = px[40:60, 512:512 + 104]
    seq = []
    for x in range(src.shape[1]):
        c = [i for i in src[4:16, x] if i not in (0, 1, 3, 11)]
        if c: seq.append(max(set(c), key=c.count))
    def core(t, rows):
        return seq[min(len(seq) - 1, int(t * len(seq)))]
    return style(a, core, 1, outline=1, shadow=1, conn=8)


def tab_item_style(a):
    # palette 5: 1 yellow, 5 darker yellow, e near black
    return style(a, 1, 5, outline=14, shadow=14, conn=8)


def bday_style(a):
    # palette 25: e yellow, c darker yellow, 4 dark
    return style(a, 14, 12, outline=4, shadow=4, conn=8)


def name_style(a):
    # palette 31: e white, c light grey, 5 dark grey
    return style(a, 14, 12, outline=None, shadow=5, sh=1, lpad=1)


def bday_code(mp, dp):
    a = Asm(BDAY_CODE)
    a.lui('t0', 0x801f)
    a.srl('v0', 's2', 1)
    a.addu('t0', 't0', 'v0')
    a.lbu('a0', 0x1858, 't0')          # month (s2 0, 1) or day (2, 3)
    a.addiu('t1', 'zero', 10)
    a.divu('a0', 't1')
    a.lhu('a2', 0, 's5')               # x
    a.lhu('a3', 0, 's7')               # y
    a.andi('t2', 's2', 1)
    a.mflo('v1')                       # tens
    a.r(0, 'zero', 'zero', 'a0', 0, 0x10)   # mfhi a0: ones
    a.branch('beq', 't2', 'zero', 'GOT')
    a.r(0, 'v1', 'zero', 'a1', 0, 0x21)     # move a1, v1 (tens)
    a.r(0, 'a0', 'zero', 'a1', 0, 0x21)     # move a1, a0 (ones)
    a.label('GOT')
    a.branch('bne', 'v1', 'zero', 'XY')
    a.addiu('t3', 'zero', 3)
    a.branch('bne', 's2', 't3', 'M1')
    a.addiu('t3', 'zero', 1)
    a.branch('beq', 'zero', 'zero', 'XY')
    a.addiu('a2', 'a2', -dp)           # day < 10: ones in the tens slot
    a.label('M1')
    a.branch('bne', 's2', 't3', 'XY')
    a.nop()
    a.lhu('t4', 0x7e8, 's6')           # month < 10: "Born" moves right
    a.nop()
    a.addiu('t4', 't4', mp)
    a.sh('t4', 0x7e8, 's6')
    a.sh('t4', 0x804, 's6')
    a.label('XY')
    a.sll('a1', 'a1', 1)
    a.lui('t1', 0x800a)
    a.addu('t1', 't1', 'a1')
    a.lbu('v0', 0x4fc8, 't1')
    a.lbu('v1', 0x4fc9, 't1')
    a.sb('v0', 0xc, 's1')
    a.sb('v1', 0xd, 's1')
    a.sh('a2', 8, 's1')
    a.sh('a3', 0xa, 's1')
    a.addiu('s4', 's4', 0x38)
    a.addiu('s2', 's2', 1)
    a.addiu('s5', 's5', 4)
    a.addiu('s7', 's7', 4)
    code = a.finish()
    n = BDAY_END - BDAY_CODE
    assert len(code) <= n, (len(code), n)
    return code + bytes(n - len(code))


def build_status(tl, px, mg, log):
    o = BDAY_CODE - B
    if list(struct.unpack_from('<3I', mg, o)) != BDAY_HEAD:
        raise SystemExit('cock_hud: birthday code is not the original (run hook.py first)')
    assert struct.unpack_from('<I', mg, FEMALE_IMM - B)[0] == 0x24420014
    recs = [list(mg[REC - B + 4 * k:REC - B + 4 * k + 4]) for k in range(6)]
    pieces = {}
    # Pieces go first-fit into the cleared label rows of tpage 640: the
    # tab strip v40-59 (u 0-247) and the birthday row v160-179 (u 0-139).
    src_status = px.copy()
    rows = [[40, 0, 248], [160, 0, 140]]          # v, next free u, end u
    for v, u0, u1 in rows: px[v:v + 20, col(640, u0):col(640, u1)] = 0

    def alloc(img, w=None):
        w = w or img.shape[1]
        for r in rows:
            if r[1] + w <= r[2]:
                u = r[1]; r[1] += w
                put(px, col(640, u), r[0], img, w, 20)
                return u, r[0]
        raise SystemExit('cock_hud: status page labels do not fit the tab and birthday rows')
    tabs = [tab_status_style(gcover(tl['タブ:' + TAB_JP[0]], 'tab', 20), src_status),
            tab_item_style(gcover(tl['タブ:' + TAB_JP[1]], 'tab', 20)),
            tab_item_style(gcover(tl['タブ:' + TAB_JP[2]], 'tab2', 20))]
    if max(t.shape[1] for t in tabs[:2]) > 104 or tabs[2].shape[1] > TAB2_MAXW:
        raise SystemExit('cock_hud: tab labels too wide')
    for k, img in enumerate(tabs):
        w = img.shape[1]
        u, v = alloc(img)
        recs[k][0:3] = [u, v, w]
        mode, x, y = TAB_POS[k]
        struct.pack_into('<2h', mg, REC_XY - B + 4 * k, x - w // 2 if mode == 'c' else x, y)
        pieces[f'tab{k}'] = img
    male = bday_style(gcover(tl['性別:男'], 'sex', 20))
    female = bday_style(gcover(tl['性別:女'], 'sex', 20))
    sl = bday_style(gcover(tl['誕生日:月'], 'bslash', 20))
    born = bday_style(gcover(tl['誕生日:日生まれ'], 'born', 20))
    sw = max(male.shape[1], female.shape[1])
    pair = np.zeros((20, 2 * sw), np.uint8)       # female = male u + sw
    put(pair, 0, 0, male, sw, 20); put(pair, sw, 0, female, sw, 20)
    u, v = alloc(pair)
    recs[5][0:3] = [u, v, sw]
    u, v = alloc(sl, BD_SL)
    recs[3][0:3] = [u, v, BD_SL]
    u, v = alloc(born)
    recs[4][0:3] = [u, v, born.shape[1]]
    struct.pack_into('<I', mg, FEMALE_IMM - B, 0x24420000 | sw)
    # birthday layout for a 2-digit month; the code moves "Born" right by
    # one pitch for months below 10, so centre that common case.
    bw = born.shape[1]
    tot = bw + BD_GAP + BD_P + BD_SL + 2 * BD_P   # Born, M, /, DD (ink approx.)
    xb = BD_C - tot // 2 - BD_P
    xm10 = xb + bw + BD_GAP - 2           # digit cells have 2-4 blank columns
    xm1 = xm10 + BD_P
    xs = xm1 + BD_P + 1
    xd10 = xs + BD_SL - 2
    xd1 = xd10 + BD_P
    struct.pack_into('<8h', mg, DIG_XY - B, xm10, BD_Y, xm1, BD_Y, xd10, BD_Y, xd1, BD_Y)
    struct.pack_into('<2h', mg, REC_XY - B + 12, xs, BD_Y)
    struct.pack_into('<2h', mg, REC_XY - B + 16, xb, BD_Y)
    struct.pack_into('<2h', mg, REC_XY - B + 20, SEX_C - sw // 2, BD_Y)
    for k in range(6): mg[REC - B + 4 * k:REC - B + 4 * k + 4] = bytes(recs[k])
    mg[o:o + BDAY_END - BDAY_CODE] = bday_code(BD_P, BD_P)
    pieces.update(male=male, female=female, slash=sl, born=born)
    # character names, 60x20 cells in place
    imgs = fit_group([tl['名前:' + jp] for jp in NAME_JP], 'name', 20, 60, name_style, log=log)
    for k, img in enumerate(imgs):
        u, v = mg[NAME_TAB - B + 2 * k], mg[NAME_TAB - B + 2 * k + 1]
        put(px, col(640, u), v, img, 60, 20)
    if log:
        print(f'cock_hud: tabs {[i.shape[1] for i in tabs]}, sex cell {sw}, born x {xb}, '
              f'digits {xm10} {xm1} / {xs} {xd10} {xd1}')
    return pieces


# ------------------------------------------------------------- entry
def apply(tim, mgame, tl=None, log=True):
    px = tim.px
    assert px.shape == (256, 1024)
    tl = tl or load_tl()
    build_date(tl, px, mgame, log)
    build_stats(tl, px, mgame, log)
    build_clubs(tl, px, mgame, log)
    build_schools(tl, px, mgame, log)
    build_status(tl, px, mgame, log)
