# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Extras collection (KOUBAI2 overlay) in English: list strings,
PocketStation and memory card dialogs, the card grid, the list screen
labels and the viewer help panel.

Reads tl/EXTRAS.txt (font strings, ## sNNN = table index, hN = hard-coded
lists) and tl/EXTRASGFX.txt (TIM 0 pieces, help panel). Writes
private/build/KOUBAI2.BIN, private/build/KOUBAI1.TIZ (+ .lz) and
private/build/MOJI2.TIZ (+ .lz). Does not touch KOUBAI.BIN/KOUBAI.TIZ/
KOUBAI0.TIZ (build_koubai.py). MENU.BIZ gets the same help panel through
build_title.py (extras_help.patch_menu).
--check reports problems and writes nothing. --png writes previews to
private/re/extras/build/.
RE: private/re/extras/findings.org, private/re/pocket/findings.org.

Font strings (KOUBAI1.TIZ TIM 1, 4 blocks of 12x12 cells of 20x20 px):
- EN is drawn two letters per cell (tools/digraph.py letters, 10 px each).
  Every distinct cell bitmap gets one free font cell. Free cells: block 3's
  empty cells, then every kana/kanji/Latin cell no remaining string uses.
  Kept: block 0 row 0 (digits, drawn by code), '/' (page counter), the
  button icons and ▼ (0x800b0718).
- String table 0x800a31e0, 142 pointers; the pool 0x800a2010-0x800a31e0
  is rewritten. 3 bytes per glyph (block, col, row), 63 63 63 = 20 px
  space (no sprite), 0x80 new line, 0x81 end.
- Each draw call has a block of 13 sprites (a2); a longer string runs on
  into a2+1. A string may use as many blocks as its JP did.
- Centred strings (0x800ab5ac, s089-s141 but s133-s135): each line is
  centred on its cell count (x = 256 - 10 * triples). The builder may
  shift a line by one letter to share more cells. A line whose letters
  are not centred that way starts with an x-adjust triple 62 dx 00
  (signed dx): it counts as a cell in the centring and moves x by dx
  with no sprite. 0x800ab6a0 branches to a 28-byte cave at CAVE (end of
  the string pool) for it. Odd lines get dx 15 (or 5 after a leading
  half space), so every line is centred to the pixel.
- Hard-coded lists: h0 (所持, two block 2 cells set by immediates),
  h1 (n/4ページ: 4 records + page digit x), h2/h3 (count + triples,
  centred on the count; odd EN is drawn 5 px into its cells), h4 (Yes/No, 5 glyphs at x 186/206 and 266/286/306).
  s094 (pocket Yes/No) is laid out on the highlight boxes (centres 196,
  306) from its EN, split at the run of spaces.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
import lz, digraph, extras_help
from build_title import line_alpha, ramp, compose, quantise, cap_size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(p): return os.path.join(ROOT, p)
K2_DISC = P('private/disc/DAT1/KOUBAI2.BIN')
K1_SRC = P('private/unpacked/KOUBAI1.TIZ.lz')
OUT = P('private/build')
PNG_DIR = P('private/re/extras/build')
K2_SLOT = (359 - 312) * 2048            # LBA 312 -> MGAME.BIN at 359
K1_SLOT = (27672 - 27602) * 2048        # LBA 27602 -> KOUBAI0... at 27672
BASE = 0x800a2000
TAB, NSTR = 0x800a31e0, 142
POOL0, POOL1 = 0x800a2010, 0x800a31e0
CAVE = POOL1 - 32                       # x-adjust code for 0x800ab5ac
XADJ = 0x62                             # x-adjust triple: 62 dx 00
SLOT_CELLS = 13
T1 = 0x21000                            # KOUBAI1 TIM 1 (font)
T1PIX = T1 + 8 + 428 + 12               # 1024x256 4bpp, 512 bytes per row
SPACE = b'\x63\x63\x63'
def F(a): return a - BASE

# ---------------------------------------------------------------- text

def load_extras(path=P('tl/EXTRAS.txt')):
    """-> {id: (jp, en)}; one entry may carry several ids."""
    out, ids, jp = {}, [], None
    for l in open(path, encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('## '): ids = l[3:].split()
        elif l.startswith('JP: '): jp = l[4:]
        elif l.startswith('EN:'):
            for i in ids: out[i] = (jp, l[4:].strip())
    return out

def load_gfx(path=P('tl/EXTRASGFX.txt')):
    return extras_help.load_tl(path)

def wrap(text, L, balance=False):
    """Greedy wrap at spaces to L letters ({n} forces a break). balance:
    same number of lines, shortest longest line."""
    out = []
    for part in text.split('{n}'):
        words, lines, line = part.split(' '), [], ''
        for w in words:
            t = (line + ' ' + w) if line else w
            if len(t) <= L or not line: line = t
            else: lines.append(line); line = w
        lines.append(line)
        if balance and len(lines) > 1:
            best = None
            n = len(lines)
            def rec(i, k, acc):
                nonlocal best
                if k == 1:
                    cand = acc + [' '.join(words[i:])]
                    m = max(len(c) for c in cand)
                    if m <= L and (best is None or m < best[0]): best = (m, cand)
                    return
                for j in range(i + 1, len(words) - k + 2):
                    rec(j, k - 1, acc + [' '.join(words[i:j])])
            rec(0, n, [])
            if best: lines = best[1]
        out += lines
    return out

# ---------------------------------------------------------------- cells

def strip(items, W):
    """digraph letters at pixel x (items: (x, ch)) on a W x 20 strip ->
    palette indices (same look as digraph.levels)."""
    im = Image.new('L', (W, 20))
    dr = ImageDraw.Draw(im)
    for x, c in items:
        if c != ' ': dr.text((x, digraph.FONT[3]), c, font=digraph.font(), fill=255)
    a = np.asarray(im, np.float32) / 255
    main = np.where(a > 0.12, np.rint(3 + 11 * a), 0)
    sh = np.zeros_like(main); sh[1:, 1:] = (a[:-1, :-1] > 0.4)
    return np.where(main > 0, main, np.where(sh > 0, 3, 0)).astype(np.uint8)

_pair = {}
def pair_cell(p):
    if p not in _pair: _pair[p] = strip([(0, p[0]), (10, p[1])], 20)
    return _pair[p]

def free_cells(W, text_items, ncell):
    """Word(s) at free pixel positions -> ncell cell bitmaps (None = blank)."""
    a = strip(text_items, W)
    return [a[:, 20 * k:20 * k + 20] if a[:, 20 * k:20 * k + 20].any() else None
            for k in range(ncell)]

def line_pairs(s, parity):
    t = ' ' * parity + s
    if len(t) % 2: t += ' '
    return [t[i:i + 2] for i in range(0, len(t), 2)]

# ---------------------------------------------------------------- font

def font_px(k1):
    raw = np.frombuffer(bytes(k1[T1PIX:T1PIX + 512 * 256]), np.uint8).reshape(256, 512)
    px = np.empty((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def put_font(k1, px):
    k1[T1PIX:T1PIX + 512 * 256] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def cell_view(px, b, c, r):
    return px[r * 20:r * 20 + 20, b * 256 + c * 20:b * 256 + c * 20 + 20]

KEEP = {(0, c, 0) for c in range(12)} | {(0, 9, 5)} | {(1, c, 8) for c in (9, 10, 11)} \
    | {(1, c, 9) for c in range(12)}     # digits, '/', icons, ▼ row

def decode_cells(d, p):
    """JP string at file offset p -> (list of lines of triples, end)."""
    lines, cur = [], []
    while True:
        b = d[p]
        if b == 0x81: lines.append(cur); return lines, p + 1
        if b == 0x80: lines.append(cur); cur = []; p += 1; continue
        cur.append(tuple(d[p:p + 3])); p += 3


# ---------------------------------------------------------------- TIM 0

T0PIX, T0W = 0x200, 1024                # KOUBAI1 TIM 0: 4bpp 1024x256 at 0x200
MR = 'MPLUSRounded1c-%s.ttf'
HELP_FONT = 'RobotoCondensed-Bold.ttf'
CATS = ['cat_card', 'cat_poster', 'cat_cd', 'cat_video', 'cat_illust', 'cat_pocket', 'back']
REC_LAB = 0x800a3528                    # 21 x (y, u, v, w, h, clut)
REC_HELP = 0x800a34e0                   # 5 x (clut, x, y, u, v, w, h)
REC_GRID = 0x800a3624                   # 5 x (clut, x, y, u, v, w, h)
SMALL_JP = [(0, 48, 101), (0, 72, 120), (0, 96, 80), (0, 120, 101), (0, 144, 141), (0, 168, 200),
            (142, 96, 48)]              # TIM x, y, w (h 24, CLUT 6)
LARGE_JP = [(256, 0, 120), (256, 32, 128), (256, 64, 99), (256, 96, 120), (256, 128, 138),
            (256, 160, 248), (456, 0, 56)]   # h 32, CLUT 0
ICON_W = 15                             # circle / cross icon columns in the help pieces

def disc_grow(a, r):
    o = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + 0.5: continue
            s = np.zeros_like(a)
            ys, ye = max(0, dy), a.shape[0] + min(0, dy)
            xs, xe = max(0, dx), a.shape[1] + min(0, dx)
            s[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
            o = np.maximum(o, s)
    return o

def shifted(a, dx, dy):
    s = np.zeros_like(a)
    s[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return s

def text_cov(text, font, cap, H, maxw, pad, base=None):
    """Coverage (H, w) of one line: pad px each side, baseline so the cap
    height is centred (or at row `base`). Squeezed to maxw if needed."""
    size = cap_size(font, cap)
    a, B = line_alpha(text, font, size)
    sx = 1.0
    if a.shape[1] > maxw - 2 * pad:
        sx = (maxw - 2 * pad) / a.shape[1]
        a, B = line_alpha(text, font, size, sx)
    w = a.shape[1] + 2 * pad
    cov = np.zeros((H, w))
    y0 = (base if base is not None else (H - cap) // 2 + cap) - B
    h = min(a.shape[0], H - y0)
    cov[y0:y0 + h, pad:pad + a.shape[1]] = a[:h]
    return cov, sx

class Tim0:
    def __init__(self, k1, k2, gfx, problems):
        self.k1, self.k2, self.gfx, self.problems = k1, k2, gfx, problems
        raw = np.frombuffer(bytes(k1[T0PIX:T0PIX + 512 * 256]), np.uint8).reshape(256, 512)
        self.px = np.empty((256, 1024), np.uint8)
        self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        self.jp = self.px.copy()
        clut = struct.unpack_from('<240H', k1, 20)
        self.pal = [np.array([[(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8]
                              for v in clut[r * 16:r * 16 + 16]], float) for r in range(15)]
        self.squeeze = {}

    def save(self):
        self.k1[T0PIX:T0PIX + 512 * 256] = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8).tobytes()

    def put(self, x, y, idx):
        h, w = idx.shape
        self.px[y:y + h, x:x + w] = idx

    def clear(self, x, y, w, h):
        self.px[y:y + h, x:x + w] = 0

    # styles ---------------------------------------------------------------
    def title(self, text):
        """Red-orange fill, thick yellow outline, dark rim (CLUT 1)."""
        W, H = 248, 48
        cov, sx = text_cov(text, MR % 'Black', 30, H, W, 4)
        full = np.zeros((H, W)); x0 = (W - cov.shape[1]) // 2
        full[:, x0:x0 + cov.shape[1]] = cov
        ys = np.nonzero(full.max(1) > 0.05)[0]
        P = self.pal[1]
        o1, o2 = disc_grow(full, 3), disc_grow(full, 4)
        fill = ramp([P[11], P[13], P[15], P[15]], H, ys[0], ys[-1] + 1)
        rgb = compose([(o2, P[1]), (o1, P[8]), (full, fill)], P[1], (H, W))
        self.squeeze['title'] = sx
        return quantise(rgb, P, list(range(1, 16)), o2 > 0.35)

    def large(self, text, maxw):
        """Blue fill with a light top, white outline, black rim (CLUT 0)."""
        H = 32
        cov, sx = text_cov(text, MR % 'Black', 20, H, maxw, 3)
        ys = np.nonzero(cov.max(1) > 0.05)[0]
        P = self.pal[0]
        o1, o2 = disc_grow(cov, 2), disc_grow(cov, 3)
        fill = ramp([P[10], P[12], P[15], P[15]], H, ys[0], ys[-1] + 1)
        rgb = compose([(o2, P[1]), (o1, P[6]), (cov, fill)], P[1], cov.shape)
        return quantise(rgb, P, list(range(1, 16)), o2 > 0.35), sx

    def small(self, text, maxw):
        """Dim flat grey (CLUT 6, indices 5-10)."""
        H = 24
        cov, sx = text_cov(text, MR % 'Bold', 14, H, maxw, 0)
        P = self.pal[6]
        rgb = compose([(cov, P[10])], P[4], cov.shape)
        return quantise(rgb, P, list(range(5, 11)), cov > 0.2), sx

    def helptext(self, text, maxw):
        """White text, dark shadow down-right (CLUT 3), 16 rows."""
        H = 16
        cov, sx = text_cov(text, HELP_FONT, 10, H, maxw, 1, base=13)
        P = self.pal[3]
        sh = shifted(disc_grow(cov, 1), 1, 1)
        rgb = compose([(sh, P[4]), (cov, P[15])], P[4], cov.shape)
        return quantise(rgb, P, list(range(4, 16)), np.maximum(sh, cov) > 0.3), sx

    # pieces ---------------------------------------------------------------
    def labels(self):
        d, g = self.k2, self.gfx
        for x, y, w in SMALL_JP: self.clear(x, y, w, 24)
        for x, y, w in LARGE_JP: self.clear(x, y, w, 32)
        # small: rows 0-5 at TIM x 0 (up to the icons at x 200); Back at x 142
        # in the CD row. Large: rows at TIM x 256; Back at the end of row 0.
        bw = None
        for i, key in enumerate(CATS):
            text = g[key]
            sx0, sy0, _ = SMALL_JP[i]
            maxw = 58 if i == 6 else 142 if i == 2 else 200
            img, sq = self.small(text, maxw)
            self.put(sx0, sy0, img)
            o = F(REC_LAB) + 12 * i
            struct.pack_into('<6h', d, o, *struct.unpack_from('<h', d, o), sx0, sy0, img.shape[1], 24, 6)
            if i == 6:
                img, sq2 = self.large(text, 120)
                bw = img.shape[1]; u = 256 - bw
                self.put(256 + u, 0, img)
            else:
                maxw = 248
                img, sq2 = self.large(text, maxw)
                if i == 0: self.row0 = img.shape[1]
                self.put(256, 32 * i, img); u = 0
            o = F(REC_LAB) + 12 * (7 + i)
            struct.pack_into('<6h', d, o, *struct.unpack_from('<h', d, o), u, 32 * i if i < 6 else 0,
                             img.shape[1], 32, 0)
            self.squeeze[key] = min(sq, sq2)
        if self.row0 + bw > 256: self.problems.append('large row 0: %s + Back too wide' % g['cat_card'])

    def help_line(self):
        d, g, px = self.k2, self.gfx, self.px
        # (key, TIM x, y, JP w, icon kept)
        pieces = [('line_select', 32, 208, 112, False), ('line_ok', 144, 208, 104, True),
                  ('line_back', 0, 224, 104, True), ('line_stop', 0, 240, 176, True)]
        widths = {}
        for key, x, y, w, icon in pieces:
            x0 = x + (ICON_W + 3 if icon else 0)
            self.clear(x + (ICON_W if icon else 0), y, w - (ICON_W if icon else 0), 16)
            img, sq = self.helptext(g[key], w - (x0 - x))
            self.put(x0, y, img)
            widths[key] = x0 - x + img.shape[1]
            self.squeeze[key] = sq
        # records 1-3; arrows (0, 4) stay at x 52 and 444
        ws = [widths['line_select'], widths['line_ok'], widths['line_back']]
        gap = (444 - 68 - sum(ws)) / 4
        x = 68 + gap
        for k, (key, u, v) in enumerate([('line_select', 32, 208), ('line_ok', 144, 208), ('line_back', 0, 224)]):
            o = F(REC_HELP) + 14 * (k + 1)
            struct.pack_into('<7h', d, o, 3, round(x), 431, u, v, widths[key], 16)
            x += widths[key] + gap
        self.help_x = [52] + [struct.unpack_from('<h', d, F(REC_HELP) + 14 * k + 2)[0] for k in (1, 2, 3)] + [444]

    def grid(self):
        d, g, px = self.k2, self.gfx, self.px
        text = g['grid_view']
        icon = text.endswith(' O')
        if icon: text = text[:-2]
        ic = self.jp[216:232, 602:617].copy()
        self.clear(512, 216, 104, 16)
        img, sq = self.helptext(text, 104 - (ICON_W + 3 if icon else 0))
        self.put(512, 216, img)
        w = img.shape[1]
        if icon:
            px[216:232, 512 + w + 2:512 + w + 2 + 15] = ic
            w += 2 + 15
        self.squeeze['grid_view'] = sq
        o = F(REC_GRID) + 14 * 2
        struct.pack_into('<7h', d, o, 3, 314 - w, 432, 0, 216, w, 16)
        # EXIT：×・START: the middle dot (TIM x 697-703) becomes a slash
        if g['grid_exit'] != 'EXIT: X/START':
            self.problems.append('grid_exit: only "EXIT: X/START" is supported')
        self.clear(696, 216, 9, 16)
        sl, _ = self.helptext('/', 12)
        sl = sl[:, :9] if sl.shape[1] > 9 else sl
        px[216:232, 696:696 + sl.shape[1]] |= sl

    def run(self):
        self.put(0, 0, self.title(self.gfx['title']))
        self.labels()
        self.help_line()
        self.grid()
        for k, v in self.squeeze.items():
            if v < 0.8: self.problems.append('%s: squeezed to %.2f' % (k, v))
        self.save()

# ---------------------------------------------------------------- build

# Strings drawn while PDATIM (J-BANK page, 8bpp 512x480 at VRAM 704,0) is
# in VRAM. It covers font block 3 (VRAM x 704-767), so their cells must be
# in blocks 0-2. Found by scanning the string-table loads in the code
# ranges below (pdatim_strings); the J-BANK exit and the help pages reload
# KOUBAI1 before they draw the menu again. RE: private/re/pocket/findings.org.
PDATIM_CODE = [(0x800acb14, 0x800ad53c),    # J-BANK 0x800acb14 (all states)
               (0x800b0878, 0x800b0c1c)]    # error dialog 0x800b0878
PDATIM_EXTRA = {119, 129}                   # J-BANK save texts, no code ref
LOW_BLOCKS = 3                              # blocks 0-2 stay in VRAM

def pdatim_strings(k2):
    """String ids loaded (lw/addiu with a table offset) in PDATIM_CODE."""
    ids = set(PDATIM_EXTRA)
    for lo, hi in PDATIM_CODE:
        for a in range(lo, hi, 4):
            w = struct.unpack_from('<I', k2, F(a))[0]
            imm = w & 0xffff
            if w >> 26 in (0x23, 0x09) and TAB & 0xffff <= imm < (TAB & 0xffff) + 4 * NSTR \
                    and imm % 4 == 0:
                ids.add((imm - (TAB & 0xffff)) // 4)
    return ids

CENTRED = set(range(89, 142)) - {133, 134, 135}
# centred-range strings no code references (unused): no parity shift and
# no x-adjust triple. The others all go to 0x800ab5ac.
UNSURE = {96, 102, 106, 110, 115, 119, 125, 129, 138}

class Build:
    def __init__(self):
        self.k2 = bytearray(open(K2_DISC, 'rb').read())
        self.k1 = bytearray(open(K1_SRC, 'rb').read())
        self.tl = load_extras()
        self.gfx = load_gfx()
        self.problems = []
        self.notes = []

    def jp_strings(self):
        d = self.k2
        ptr = struct.unpack_from('<142I', d, F(TAB))
        return [decode_cells(d, F(p))[0] for p in ptr]

    def plan(self):
        """Decide every string's cells. Cell = pair string (standard),
        ('bm', bytes) (free bitmap), None (space), or ('jp', triple)."""
        jp = self.jp_strings()
        self.jp = jp
        tl = self.tl
        strings = {}                     # i -> list of lines of cells
        flex = []                        # (i, line no, text) centred lines
        for i in range(NSTR):
            jpt, en = tl.get('s%03d' % i, ('', ''))
            jcells = [[('jp', t) if t[0] != 0x63 else None for t in ln] for ln in jp[i]]
            if not en or self.same(jpt, en):
                strings[i] = jcells
                continue
            if i == 94:
                strings[i] = [self.yesno_pocket(en)]
                continue
            L = 20 if i <= 88 else 24 if i in (89, 90, 114, 116, 117) else 12 if i == 115 \
                else {133: 4, 134: 16, 135: 16}.get(i, 40)
            lines = wrap(en, L, balance=i in CENTRED)
            strings[i] = [None] * len(lines)
            for k, s in enumerate(lines):
                if i in CENTRED and i not in UNSURE: flex.append((i, k, s))
                else: strings[i][k] = line_pairs(s, 0)
        # parity choice for centred lines: share as many cells as possible
        fixed = {}
        def add(cs, n=1):
            for c in cs:
                if isinstance(c, str) and c != '  ': fixed[c] = fixed.get(c, 0) + n
        for i, ls in strings.items():
            for ln in ls:
                if ln is not None: add(ln)
        choice = {}
        for it in range(3):
            for (i, k, s) in flex:
                if (i, k) in choice: add(choice[(i, k)], -1)
                cap = self.capacity(i)
                best = None
                for par in (0, 1):
                    cs = line_pairs(s, par)
                    new = len({c for c in cs if c != '  ' and not fixed.get(c)})
                    exact = (len(s) + par) % 2 == 0 if len(s) % 2 == 0 else False
                    key = (new, len(cs), not exact, par)
                    if best is None or key < best[0]: best = (key, cs)
                choice[(i, k)] = best[1]; add(best[1])
        for (i, k, s) in flex:
            strings[i][k] = choice[(i, k)]
        # slot capacity: shrink parity where a leading space overflows
        for i, ls in strings.items():
            if self.sprites(ls) > self.capacity(i):
                for k, ln in enumerate(ls):
                    if ln and ln[0][0] == ' ' and ln[0] != '  ' and (i, k) in choice:
                        txt = ''.join(ln).strip()
                        ls[k] = line_pairs(txt, 0)
            if self.sprites(ls) > self.capacity(i):
                self.problems.append('s%03d: %d sprites > %d' % (i, self.sprites(ls), self.capacity(i)))
        # x-adjust triple where the letters are not centred on the cells
        for (i, k, s) in flex:
            ln = strings[i][k]
            t = ''.join(c for c in ln)
            lead = len(t) - len(t.lstrip(' '))
            n = len(t.strip(' '))
            dx = 10 * len(ln) - 10 * lead - 5 * n
            if dx: strings[i][k] = [('dx', dx + 10)] + ln
        self.strings = strings
        # hard-coded lists
        self.hard = {}
        self.plan_hard()

    def same(self, jp, en):
        """EN that equals the JP text (Latin full-width) keeps the JP bytes."""
        z = jp.replace('ー', '-').replace('　', ' ')
        z = ''.join(chr(ord(c) - 0xfee0) if 0xff01 <= ord(c) <= 0xff5e else c for c in z)
        return z == en

    def capacity(self, i):
        n = sum(1 for ln in self.jp[i] for t in ln if t[0] != 0x63)
        return SLOT_CELLS * max(1, -(-n // SLOT_CELLS))

    @staticmethod
    def sprites(ls):
        return sum(1 for ln in ls if ln for c in ln
                   if c is not None and c != '  ' and not (isinstance(c, tuple) and c[0] == 'dx'))

    def yesno_pocket(self, en):
        """s094: Yes centred on x 196, No on x 306; the line is 8 cells
        centred on 256 (x 176-336), as the JP はい　　　いいえ."""
        parts = [p for p in re.split(r'\s{2,}', en.strip()) if p]
        if len(parts) != 2:
            self.problems.append('s094: EN must be "Yes<spaces>No": %r' % en); parts = ['Yes', 'No']
        items = []
        for word, cx in zip(parts, (196 - 176, 306 - 176)):
            x0 = cx - 5 * len(word)
            items += [(x0 + 10 * k, ch) for k, ch in enumerate(word)]
        return [('bm', c.tobytes()) if c is not None else None for c in free_cells(160, items, 8)]

    def plan_hard(self):
        tl = self.tl
        h = self.hard
        # h0 所持: two cells at x 282, 302 (block 2 only)
        en = tl['h0'][1]
        if len(en) > 4: self.problems.append('h0: more than 4 letters')
        x0 = 20 - 5 * len(en)
        h['h0'] = [('bm', c.tobytes()) if c is not None else ('blank',)
                   for c in free_cells(40, [(x0 + 10 * k, ch) for k, ch in enumerate(en)], 2)]
        # h1 page counter: label, 10 px gap, d/d
        en = tl['h1'][1]
        t = en if len(en) % 2 else ' ' + en
        lab = line_pairs(t + ' ', 0) if t else []
        lab = [c for c in lab]
        if len(lab) > 3: self.problems.append('h1: label too long (%d cells)' % len(lab)); lab = lab[:3]
        h['h1'] = lab
        # h2/h3: counted, centred on the count. Odd EN: letters 5 px into
        # the cells (own bitmaps), so the line is centred to the pixel.
        for k, cap in (('h2', 7), ('h3', 14)):
            s = tl[k][1]
            if len(s) % 2:
                n = (len(s) + 1) // 2
                cs = [('bm', c.tobytes()) if c is not None else '  ' for c in
                      free_cells(20 * n, [(5 + 10 * j, ch) for j, ch in enumerate(s)], n)]
            else:
                cs = line_pairs(s, 0)
            if len(cs) > cap: self.problems.append('%s: %d cells > %d' % (k, len(cs), cap))
            h[k] = cs
        # h4 Yes/No: boxes 184-228 (centre 206) and 264-328 (centre 296)
        y, n = tl['h4a'][1], tl['h4b'][1]
        ys = free_cells(40, [(20 - 5 * len(y) + 10 * k, c) for k, c in enumerate(y)], 2)
        ns = free_cells(60, [(30 - 5 * len(n) + 10 * k, c) for k, c in enumerate(n)], 3)
        h['h4'] = [('bm', c.tobytes()) if c is not None else ('blank',) for c in ys + ns]

    # ------------------------------------------------------------ font cells

    def allocate(self):
        """Give every needed cell bitmap a font cell."""
        px = font_px(self.k1)
        self.px = px
        used_jp = set(KEEP)
        for ls in self.strings.values():
            for ln in ls:
                for c in ln:
                    if isinstance(c, tuple) and c[0] == 'jp': used_jp.add(c[1])
        need = {}                        # key -> bitmap
        def want(c):
            if c is None or c == '  ' or c[0] == 'dx': return
            if isinstance(c, str): need.setdefault(c, pair_cell(c))
            elif c[0] == 'bm': need.setdefault(c, np.frombuffer(c[1], np.uint8).reshape(20, 20))
        for ls in self.strings.values():
            for ln in ls:
                for c in ln: want(c)
        for k in ('h1', 'h2', 'h3', 'h4'):
            for c in self.hard[k]:
                if c != ('blank',): want(c)
        h0 = [c for c in self.hard['h0'] if c != ('blank',)]
        # free cells: block 3 first, then blocks 2, 1, 0 (all-blank cells
        # and cells no kept string uses)
        free = []
        for b in (3, 2, 1, 0):
            for r in range(12):
                for c in range(12):
                    if (b, c, r) not in used_jp: free.append((b, c, r))
        blank = next(f for f in free if f[0] == 3 and not cell_view(px, *f).any())
        free.remove(blank)
        self.blank = blank
        where = {}
        # h0 needs block 2
        # (0x800a86d8 'addiu s1, zero, 1' is the first cell's column and
        # also the loop's compare constant, so that cell must be in column 1)
        for k, c in enumerate(h0):
            if c in where: continue
            f2 = [f for f in free if f[0] == 2 and (k or f[1] == 1)]
            if not f2: self.problems.append('h0: no free block 2 cell'); break
            f = f2[0]; free.remove(f); where[c] = f
        keys = [k for k in need if k not in where]
        self.cells_needed = len(keys) + len(where)
        self.cells_free = len(free)
        if len(keys) > len(free):
            self.problems.append('font: %d cells needed, %d free' % (len(keys), len(free)))
            keys = keys[:len(free)]
        # PDATIM strings first, into blocks 0-2; the rest fill block 3 first
        self.pdatim = pdatim_strings(self.k2)
        pri = []
        for i in sorted(self.pdatim):
            for ln in self.strings[i]:
                for c in ln:
                    if c in need and c not in where and c not in pri: pri.append(c)
        low = [f for f in free if f[0] < LOW_BLOCKS]
        if len(pri) > len(low):
            self.problems.append('font: %d PDATIM cells, %d free in blocks 0-2' % (len(pri), len(low)))
        for k, f in zip(pri, low): where[k] = f
        rest = [f for f in free if f not in set(where.values())]
        keys = [k for k in keys if k not in where]
        for k, f in zip(keys, rest): where[k] = f
        self.pri_cells = len(pri)
        self.free_left = len(rest) - len(keys)
        # clear freed JP cells, then paint
        for f in free: cell_view(px, *f)[:] = 0
        for k, f in where.items():
            cell_view(px, *f)[:] = need[k] if k in need else np.frombuffer(k[1], np.uint8).reshape(20, 20)
        self.where = where

    def code(self, c):
        if c is None or c == '  ': return SPACE
        if c == ('blank',): return bytes(self.blank)
        if isinstance(c, tuple) and c[0] == 'dx': return bytes([XADJ, c[1] & 0xff, 0])
        if isinstance(c, tuple) and c[0] == 'jp': return bytes(c[1])
        return bytes(self.where[c])

    # ------------------------------------------------------------ KOUBAI2

    def write_pool(self):
        d = self.k2
        pool = bytearray(); at = {}
        ptrs = []
        for i in range(NSTR):
            s = b'\x80'.join(b''.join(self.code(c) for c in ln) for ln in self.strings[i]) + b'\x81'
            if s not in at:
                at[s] = POOL0 + len(pool); pool += s
            ptrs.append(at[s])
        self.pool_bytes = len(pool)
        if POOL0 + len(pool) > CAVE:
            self.problems.append('pool: %d bytes > %d' % (len(pool), CAVE - POOL0))
            return
        d[F(POOL0):F(CAVE)] = bytes(pool) + bytes(CAVE - POOL0 - len(pool))
        struct.pack_into('<142I', d, F(TAB), *ptrs)
        self.write_cave()
        self.check_pdatim()

    def check_pdatim(self):
        """Every glyph of a PDATIM string must be in blocks 0-2."""
        d = self.k2
        for i in sorted(self.pdatim):
            p = F(struct.unpack_from('<I', d, F(TAB) + 4 * i)[0])
            bad = []
            while d[p] != 0x81:
                if d[p] == 0x80: p += 1; continue
                t = tuple(d[p:p + 3]); p += 3
                if t[0] in (0x63, XADJ): continue
                if t[0] >= LOW_BLOCKS: bad.append(t)
            if bad: self.problems.append('s%03d: drawn over PDATIM but uses block 3 %r' % (i, bad))

    def write_cave(self):
        """0x800ab6a0 'bne v1, v0(0x63), draw' -> 'bne v1, v0, CAVE' (delay
        slot 'move a0, zero' kept). The cave sends any block byte but
        XADJ on to the draw path; XADJ adds the signed second byte to x
        (s3) and skips the triple."""
        d = self.k2
        def j(a): return 0x08000000 | ((a >> 2) & 0x3ffffff)
        def br(op, rs, rt, at, to): return op << 26 | rs << 21 | rt << 16 | (((to - at - 4) >> 2) & 0xffff)
        V0, V1, S2, S3 = 2, 3, 18, 19
        DRAW, NEXT = 0x800ab6b4, 0x800ab824
        code = [0x24000000 | V0 << 16 | XADJ,          # addiu v0, zero, XADJ
                br(5, V1, V0, CAVE + 4, DRAW),         # bne v1, v0, DRAW
                0,                                     # nop
                0x80000000 | S2 << 21 | V0 << 16 | 1,  # lb v0, 1(s2)
                0x24000000 | S2 << 21 | S2 << 16 | 3,  # addiu s2, s2, 3
                j(NEXT),                               # j NEXT
                S3 << 21 | V0 << 16 | S3 << 11 | 0x21]  # addu s3, s3, v0
        o = F(0x800ab6a0)
        assert struct.unpack_from('<I', d, o)[0] == br(5, V1, V0, 0x800ab6a0, DRAW), 'ab6a0'
        assert struct.unpack_from('<I', d, F(DRAW))[0] == 0x26520001    # addiu s2, s2, 1
        struct.pack_into('<I', d, o, br(5, V1, V0, 0x800ab6a0, CAVE))
        struct.pack_into('<7I', d, F(CAVE), *code)

    def patch_imm(self, addr, old, new):
        """addiu rt, zero/rs, imm: replace the low half-word."""
        o = F(addr)
        w = struct.unpack_from('<I', self.k2, o)[0]
        assert w & 0xffff == old, ('imm', hex(addr), hex(w))
        struct.pack_into('<I', self.k2, o, (w & 0xffff0000) | (new & 0xffff))

    def write_hard(self):
        d = self.k2
        # h0: 0x800a86d8 col0, 0x800a86e0 row0, 0x800a86e4 col1, 0x800a86e8 row1
        c0, c1 = (self.where.get(c, self.blank) if c != ('blank',) else self.blank
                  for c in self.hard['h0'])
        if c0[0] != 2 or c1[0] != 2 or c0[1] != 1: self.problems.append('h0: cells not in block 2 / column 1')
        self.patch_imm(0x800a86d8, 1, c0[1]); self.patch_imm(0x800a86e0, 7, c0[2])
        self.patch_imm(0x800a86e4, 10, c1[1]); self.patch_imm(0x800a86e8, 6, c1[2])
        # h1: records (xcell, tpage/64, col, row); x = R + 20 xcell
        lab = self.hard['h1']
        n = len(lab)
        # label cells, 10 px gap built into the last cell, digit, '/', digit
        # (a leading half space, for even labels, is left out of the width)
        lead = 10 if lab and lab[0][0] == ' ' else 0
        width = 20 * n + 60 - lead
        R = 256 - width // 2 - lead
        recs = []
        for k, c in enumerate(lab):
            b, cc, r = self.code(c) if c != '  ' else bytes(self.blank)
            recs.append((k, b + 8, cc, r))
        recs.append((n + 1, 8, 9, 5))    # JP '/'
        while len(recs) < 4:
            b, cc, r = self.blank; recs.append((0, b + 8, cc, r))
        for k, r in enumerate(recs):
            d[F(0x800a366c) + 4 * k:F(0x800a366c) + 4 * k + 4] = bytes(r)
        self.patch_imm(0x800aa5fc, 0xc4, R)
        self.patch_imm(0x800aa72c, 0xc4, R + 20 * n)
        self.patch_imm(0x800aa858, 0xc4, R + 20 * n)   # second digit = +0x28
        self.page_x = (R, R + 20 * n)
        # h2/h3 in place (0x800a367c 24 bytes, 0x800a3694 44 bytes)
        for k, a, size in (('h2', 0x800a367c, 24), ('h3', 0x800a3694, 44)):
            cs = self.hard[k]
            b = bytes([len(cs)]) + b''.join(self.code(c) if c != '  ' else bytes(self.blank) for c in cs)
            if len(b) > size: self.problems.append('%s: %d bytes > %d' % (k, len(b), size)); continue
            d[F(a):F(a) + size] = b + bytes(size - len(b))
        # card received line: [s133 Rare at x 48] s134 at x 88 (48 if not
        # rare), card numbers at s134 x + 0xa0. EN: s134 a little right of
        # Rare (word gap) and the numbers right after its last cell.
        n134 = len(self.strings[134][0])
        X134 = 0x5e
        self.patch_imm(0x800af330, 0x58, X134); self.patch_imm(0x800af428, 0x58, X134)
        self.patch_imm(0x800af358, 0xa0, 20 * n134)
        self.patch_imm(0x800af3f8, 0x10c, X134 + 20 * n134 + 20)
        # h4: 5 triples
        b = b''.join(self.code(c) for c in self.hard['h4'])
        d[F(0x800a36c0):F(0x800a36c0) + 15] = b

    # ------------------------------------------------------------ run

    def text(self):
        self.plan()
        self.allocate()
        self.write_pool()
        self.write_hard()
        put_font(self.k1, self.px)

    def run(self, write=True):
        self.text()
        self.t0 = Tim0(self.k1, self.k2, self.gfx, self.problems)
        self.t0.run()
        self.moji2, sq = extras_help.build_moji2(self.gfx)
        self.problems += extras_help.problems(sq)
        assert len(self.k2) == 94704
        self.k1c = lz.compress(bytes(self.k1), name='KOUBAI1.TIZ')
        assert lz.decompress(self.k1c)[0] == bytes(self.k1)
        self.m2c = lz.compress(bytes(self.moji2), name='MOJI2.TIZ')
        if len(self.k1c) > K1_SLOT: self.problems.append('KOUBAI1.TIZ: %d > slot %d' % (len(self.k1c), K1_SLOT))
        if len(self.m2c) > extras_help.MOJI2_SLOT:
            self.problems.append('MOJI2.TIZ: %d > slot %d' % (len(self.m2c), extras_help.MOJI2_SLOT))
        if write and not self.problems:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, 'KOUBAI2.BIN'), 'wb').write(self.k2)
            open(os.path.join(OUT, 'KOUBAI1.TIZ.lz'), 'wb').write(self.k1)
            open(os.path.join(OUT, 'KOUBAI1.TIZ'), 'wb').write(self.k1c)
            open(os.path.join(OUT, 'MOJI2.TIZ.lz'), 'wb').write(self.moji2)
            open(os.path.join(OUT, 'MOJI2.TIZ'), 'wb').write(self.m2c)

if __name__ == '__main__':
    check = '--check' in sys.argv
    bd = Build()
    bd.run(write=not check)
    print('font cells: %d used by EN, %d free before, %d left; pool %d of %d bytes' %
          (bd.cells_needed, bd.cells_free, bd.free_left, bd.pool_bytes, CAVE - POOL0))
    print('PDATIM strings (blocks 0-2 only, %d cells): %s' %
          (bd.pri_cells, ' '.join('s%03d' % i for i in sorted(bd.pdatim))))
    print('KOUBAI1.TIZ %d (slot %d), MOJI2.TIZ %d (slot %d), KOUBAI2.BIN %d (slot %d)' %
          (len(bd.k1c), K1_SLOT, len(bd.m2c), extras_help.MOJI2_SLOT, len(bd.k2), K2_SLOT))
    for p in bd.problems: print('PROBLEM', p)
    if '--png' in sys.argv:
        sys.path.insert(0, PNG_DIR)
        import preview                   # private/re/extras/build/preview.py
        preview.write(bd, PNG_DIR)
    if bd.problems: sys.exit(1)
    print('check OK' if check else 'written')
