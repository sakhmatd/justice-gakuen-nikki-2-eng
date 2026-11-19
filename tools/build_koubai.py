# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Extras shop (購買部) and Extras top menu in English.

Reads tl/KOUBAI.txt (shop strings), tl/KOUBAIGFX.txt (graphic labels),
the disc DAT1/KOUBAI.BIN and private/unpacked/KOUBAI.TIZ.lz,
KOUBAI0.TIZ.lz. Writes private/build/KOUBAI.BIN, KOUBAI.TIZ and
KOUBAI0.TIZ (nothing else). --check writes nothing and reports problems;
--png writes previews to private/re/shop/en/.
Facts: private/re/shop/findings.org, private/re/extras/findings.org
(section KOUBAI0.TIZ).

KOUBAI.BIN (load 0x800a2000, file = RAM - base):
- Strings are 3-byte cells (T, col, row) into the KOUBAI.TIZ font TIM
  (tpage x 576 + 64*T, 12x12 cells of 20x20 px), 0x80 new line, 0x63 xx xx
  blank cell (no primitive), 0x81 end.
- Three pointer tables (desc 0x2458 x83, names 0x2cbc x83, messages
  0x371c x20) point into three pools; the pools are filled as one pool
  with identical strings shared. Card pieces 0x3bac / 0x3bbc are count
  byte + cells, patched in place.
- EN letters are drawn two per cell (letter pairs, tools/digraph.py, the
  diary look) into the font cells; the digits, the slash and the 「」
  cells used by code stay.
- Top menu labels: (w, x, y) u16 table at 0x3a20.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lz, digraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0x800a2000
BIN_DISC = 'private/disc/DAT1/KOUBAI.BIN'
TIZ_SRC = 'private/unpacked/KOUBAI.TIZ.lz'
TIZ0_SRC = 'private/unpacked/KOUBAI0.TIZ.lz'
SLOT_BIN, SLOT_TIZ, SLOT_TIZ0 = 19 * 2048, 98 * 2048, 100 * 2048   # disc sectors (lba.txt)

TABLES = {'d': (0x2458, 83), 'n': (0x2cbc, 83), 'm': (0x371c, 20)}
POOLS = [(0x0010, 0x2458), (0x25a4, 0x2cbc), (0x2e08, 0x371c)]
CARD = {'c00': (0x3bac, 5), 'c01': (0x3bbc, 6)}
MENU_TAB = 0x3a20                       # 3 x (w, x, y) u16
CONFIRM = ('m01', 'm04', 'm11', 'm14')  # Yes/No on line 3
YES_X, NO_X, LINE_X = 180, 340, 80
MSG_CELLS = 70                          # glyph buffer of the text box
FONT_OFF = 0x2e800                      # TIM 4: font, 4 tpages of 12x12 cells
# Font cells other code draws: digits 0-9 (tp0 row 0), slash (tp0 9,5),
# 「 」 of the description header (tp1 7,9 and 8,9).
KEEP = {(0, c, 0) for c in range(10)} | {(0, 9, 5), (1, 7, 9), (1, 8, 9)}
SPACE = b'\x63\x63\x63'
# Description header 「name」 (loop 0x800a9bf0-0x800a9d70, its own code):
# 「 x80, name from x100, the code's 」 at the x after the last cell. The
# loop is patched so the header reads past the list terminator: 0x81 =
# skip 3 bytes and x -= 10, 0x80 = x += 10 (names have no line breaks),
# 0x82 = end, 0x63 is no longer special there. Each name is stored as
# list bytes .. 0x81, 2 pad bytes, [0x80 if even], 0x82, so odd names
# pull 」 back over their blank half cell. The list drawer stops at 0x81.
HDR_LOOP = [(0x7bd8, 0x24020081, 0x24020082),   # first end check
            (0x7d68, 0x24020081, 0x24020082),   # loop end check
            (0x7bf8, 0x24020063, 0x24020081),   # skip code 0x63 -> 0x81
            (0x7c14, 0x26730014, 0x2673fff6),   # skip: x += 20 -> x -= 10
            (0x7bfc, 0x26d6001a, 0x2673000a),   # 0x80: y += 26 -> x += 10
            (0x7c00, 0x24130040, 0x00000000)]   # 0x80: x = 64 -> nop


# ---------------------------------------------------------------- tl files
def load_tl(path='tl/KOUBAI.txt'):
    """-> {id: (JP, EN, limit)} with limit (letters/line, lines, total)."""
    out = {}; ids = []; lim = None; jp = None
    for l in open(os.path.join(ROOT, path), encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('## '): ids = l[3:].split(); lim = None
        m = re.match(r'# limit: (\d+) letters/line, (\d+) lines?(?:, (\d+) letters total)?', l)
        if m: lim = tuple(int(x) if x else None for x in m.groups())
        if l.startswith('JP: '): jp = l[4:]
        if l.startswith('EN:'):
            for i in ids: out[i] = (jp, l[4:].strip(), lim)
    return out


def load_gfx(path='tl/KOUBAIGFX.txt'):
    out = {}
    for l in open(os.path.join(ROOT, path), encoding='utf-8'):
        if l.startswith('#') or ':' not in l: continue
        k, v = l.split(':', 1); out[k.strip()] = v.strip()
    return out


def wrap(text, L):
    """Greedy wrap at spaces (same rule as private/re/extras/tlcheck.py)."""
    out = []
    for part in text.split('{n}'):
        line = ''
        for w in part.split(' '):
            t = (line + ' ' + w) if line else w
            if len(t) <= L or not line: line = t
            else: out.append(line); line = w
        out.append(line)
    return out


def text_lines(key, en, lim, tl):
    """EN of one entry -> list of letter lines as drawn."""
    L, N, T = lim
    lines = wrap(en, L)
    if key in CONFIRM:
        if len(lines) == 1: lines.append('')
        assert len(lines) == 2, (key, lines)
        yes, no = tl['yes'][1], tl['no'][1]
        a, b = (YES_X - LINE_X) // 10, (NO_X - LINE_X) // 10
        lines.append(' ' * a + yes.ljust(b - a) + no)
    return lines


def pairs(line):
    """Letter line -> list of 2-letter cells, trailing blanks removed."""
    if len(line) % 2: line += ' '
    ps = [line[i:i + 2] for i in range(0, len(line), 2)]
    while ps and ps[-1] == '  ': ps.pop()
    return ps


# ---------------------------------------------------------------- font
def font_cells():
    """All 576 cells (T, col, row) in fill order."""
    return [(t, c, r) for t in range(4) for r in range(12) for c in range(12)]


class Font:
    """Allocates font cells for letter pairs."""
    def __init__(self, reserved):
        self.free = [c for c in font_cells() if c not in reserved]
        self.map = {}
    def cell(self, p):
        if p not in self.map:
            if not self.free: raise OverflowError('font full')
            self.map[p] = self.free.pop(0)
        return self.map[p]


def encode(lines, font):
    out = bytearray(); cells = 0
    for i, line in enumerate(lines):
        if i: out.append(0x80)
        for p in pairs(line):
            if p == '  ': out += SPACE
            else: out += bytes(font.cell(p)); cells += 1
    out.append(0x81)
    return bytes(out), cells


def jp_string(disc, kind, i):
    """Raw JP bytes of table entry i (through 0x81)."""
    o = struct.unpack_from('<I', disc, TABLES[kind][0] + 4 * i)[0] - BASE
    e = o
    while disc[e] != 0x81: e += 1 if disc[e] in (0x80,) else 3
    return disc[o:e + 1]


def jp_cells(raw):
    out = set(); o = 0
    while raw[o] != 0x81:
        if raw[o] == 0x80: o += 1; continue
        if raw[o] != 0x63: out.add(tuple(raw[o:o + 3]))
        o += 3
    return out


def build_text(disc, tl, problems):
    """-> (patched KOUBAI.BIN bytearray, Font, stats)."""
    d = bytearray(disc)
    strings = {}                         # (kind, i) -> bytes
    reserved = set(KEEP)
    todo = []
    for kind, (tab, n) in TABLES.items():
        for i in range(n):
            key = '%s%02d' % (kind, i)
            jp, en, lim = tl.get(key, (None, '', None))
            if not en:
                raw = jp_string(disc, kind, i)
                problems.append('%s: no EN, JP kept' % key)
                strings[(kind, i)] = raw; reserved |= jp_cells(raw)
            else: todo.append((kind, i, key, en, lim))
    font = Font(reserved)
    cells_of = {}
    for kind, i, key, en, lim in todo:
        lines = text_lines(key, en, lim, tl)
        L, N, T = lim
        q = [x for x in lines if x is not None]
        if key not in CONFIRM and len(lines) > N:
            problems.append('%s: %d lines > %d' % (key, len(lines), N))
        if any(len(x) > 40 for x in lines):
            problems.append('%s: line over 20 cells' % key)
        raw, nc = encode(lines, font)
        if kind in 'dm' and nc > MSG_CELLS:
            problems.append('%s: %d cells > %d' % (key, nc, MSG_CELLS))
        strings[(kind, i)] = raw; cells_of[key] = lines
    # card pieces: count byte + cells, in place, no blank cells
    for key, (off, mx) in CARD.items():
        jp, en, lim = tl.get(key, (None, '', None))
        if not en: problems.append('%s: no EN, JP kept' % key); continue
        # c00 right-aligned with one blank letter before the number (x180);
        # c01 starts with one blank letter after it (x220)
        if key == 'c00': en = (en.strip() + ' ').rjust(2 * mx)
        if key == 'c01': en = ' ' + en.strip()
        ps = pairs(en)
        if len(ps) > mx or '  ' in ps:
            problems.append('%s: %d cells > %d or blank cell' % (key, len(ps), mx)); ps = ps[:mx]
        b = bytes([len(ps)]) + b''.join(bytes(font.cell(p)) for p in ps)
        d[off:off + 1 + 3 * mx] = b.ljust(1 + 3 * mx, b'\0')
    # header tails of the names (see HDR_LOOP)
    for i in range(TABLES['n'][1]):
        raw = strings[('n', i)]
        if b'\x80' in raw or any(raw[o] == 0x63 for o in range(0, len(raw) - 1, 3)):
            problems.append('n%02d: blank cell or line break in a name' % i)
        jp, en, lim = tl.get('n%02d' % i, (None, '', None))
        odd = bool(en) and len(en.rstrip()) % 2 == 1
        strings[('n', i)] = raw + b'\0\0' + (b'' if odd else b'\x80') + b'\x82'
    # one pool over three regions, identical strings shared, largest first;
    # the header table (4-aligned) opens the first region
    for a, b in POOLS: d[a:b] = bytes(b - a)
    uniq = sorted(set(strings.values()), key=lambda s: (-len(s), s))  # no hash order
    free = [list(r) for r in POOLS]; where = {}
    for s in uniq:
        for r in free:
            if r[1] - r[0] >= len(s):
                where[s] = r[0]; d[r[0]:r[0] + len(s)] = s; r[0] += len(s); break
        else: raise OverflowError('string pool full (%d unique strings)' % len(uniq))
    for (kind, i), s in strings.items():
        struct.pack_into('<I', d, TABLES[kind][0] + 4 * i, BASE + where[s])
    for off, old, new in HDR_LOOP:
        w = struct.unpack_from('<I', d, off)[0]
        assert w == old, (hex(off), hex(w))
        struct.pack_into('<I', d, off, new)
    used = sum(map(len, uniq)); total = sum(b - a for a, b in POOLS)
    return d, font, dict(pool_used=used, pool_total=total, pairs=len(font.map),
                         pairs_free=len(font.free), lines=cells_of)


def font_tim(tiz, font):
    """Clear every non-kept font cell and draw the pairs. JP ink indices:
    the letters use digraph levels (3 shadow, 4-14 face) as the message
    font does."""
    q = FONT_OFF + 8 + struct.unpack_from('<I', tiz, FONT_OFF + 8)[0]
    w, h = struct.unpack_from('<HH', tiz, q + 8)
    po = q + 12; stride = w * 2
    a = np.frombuffer(bytes(tiz[po:po + stride * h]), np.uint8).reshape(h, stride)
    px = np.zeros((h, stride * 2), np.uint8); px[:, 0::2] = a & 15; px[:, 1::2] = a >> 4
    new = np.zeros_like(px)
    for t, c, r in KEEP:
        y, x = r * 20, t * 256 + c * 20
        new[y:y + 20, x:x + 20] = px[y:y + 20, x:x + 20]
    for p, (t, c, r) in font.map.items():
        y, x = r * 20, t * 256 + c * 20
        new[y:y + 20, x:x + 20] = digraph.levels(p)
    b = (new[:, 0::2] | (new[:, 1::2] << 4)).astype(np.uint8)
    tiz[po:po + stride * h] = b.tobytes()
    return new


# ---------------------------------------------------------------- graphics
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

F_HDR = ('Antonio[wght].ttf', (700,))   # as the hub club names
FREDOKA = 'Fredoka[wdth,wght].ttf'   # axes (weight, width)
F_LBL = (FREDOKA, (600, 80))
F_LOGO = (FREDOKA, (700, 100))


def tim_at(d, off):
    """-> (pixel offset, w_texels, h, bpp, clut array)."""
    flag = struct.unpack_from('<I', d, off + 4)[0]; bpp = 8 if flag & 1 else 4
    q = off + 8; cl = None
    if flag & 8:
        bl, cx, cy, cw, ch = struct.unpack_from('<IHHHH', d, q)
        cl = np.frombuffer(bytes(d[q + 12:q + 12 + 2 * cw * ch]), np.uint16).copy(); q += bl
    bl, x, y, w, h = struct.unpack_from('<IHHHH', d, q)
    return q + 12, w * (4 if bpp == 4 else 2), h, bpp, cl


def tim_get(d, off):
    po, w, h, bpp, cl = tim_at(d, off)
    a = np.frombuffer(bytes(d[po:po + w * h * bpp // 8]), np.uint8)
    if bpp == 8: return a.reshape(h, w).copy()
    a = a.reshape(h, w // 2); p = np.zeros((h, w), np.uint8)
    p[:, 0::2] = a & 15; p[:, 1::2] = a >> 4
    return p


def tim_put(d, off, px):
    po, w, h, bpp, cl = tim_at(d, off)
    b = px.astype(np.uint8) if bpp == 8 else (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8)
    d[po:po + b.size] = b.tobytes()


def rgb15(c):
    c = np.asarray(c, np.int32)
    return np.stack([(c & 31) << 3, ((c >> 5) & 31) << 3, ((c >> 10) & 31) << 3], -1)


def text_alpha(text, font, cap, sx=1.0, track=0.0, ss=4):
    """Coverage (h, w) float of one line; cap = capital height in px.
    Returns (alpha, baseline row)."""
    path, axes = font if isinstance(font, tuple) else (font, None)
    def load(sz):
        f = fontpath.truetype(path, sz)
        if axes: f.set_variation_by_axes(list(axes))
        return f
    size = cap * 2
    f = load(size * ss)
    hb = f.getbbox('H', anchor='ls'); size = size * cap / ((hb[3] - hb[1]) / ss)
    f = load(max(1, int(round(size * ss))))
    asc, desc = f.getmetrics()
    xs = [0.0]
    for i, ch in enumerate(text):
        xs.append(xs[-1] + f.getlength(ch) + track * ss)
    W = int(xs[-1] + 4 * ss); H = asc + desc + 4 * ss
    im = Image.new('L', (W, H)); dr = ImageDraw.Draw(im)
    for i, ch in enumerate(text): dr.text((xs[i] + 2 * ss, asc + 2 * ss), ch, font=f, fill=255, anchor='ls')
    if sx != 1.0: im = im.resize((int(W * sx), H), Image.LANCZOS); W = im.width
    W2, H2 = -(-W // ss) * ss, -(-H // ss) * ss
    pad = Image.new('L', (W2, H2)); pad.paste(im, (0, 0))
    a = np.asarray(pad, np.float32).reshape(H2 // ss, ss, W2 // ss, ss).mean((1, 3)) / 255
    ys, xs_ = np.where(a > 0.02)
    a = a[ys.min():ys.max() + 1, xs_.min():xs_.max() + 1]
    return a, (asc + 2 * ss) // ss - ys.min()


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r * 0.6


def grow(a, r):
    return ndimage.grey_dilation(a, footprint=disk(r)) if r > 0 else a


def shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def pad(a, m):
    return np.pad(a, m)


def compose(a, layers, face):
    """Index image from coverage a (already padded). layers: back to front
    list of (r, dx, dy, index); face(a) -> index array for a > 0.5 pixels
    (an AA ramp toward the ring index is up to the caller)."""
    out = np.zeros(a.shape, np.uint8)
    for r, dx, dy, idx in layers:
        m = shift(grow(a, r), dx, dy) > 0.45
        out[m] = idx
    m = a > 0.12
    out[m] = face(a)[m]
    return out


def fit(text, font, cap, maxw, minsx=0.75, track=0.0, margin=0):
    """Coverage of text at cap height, squeezed (down to minsx) or shrunk
    to fit maxw - 2*margin."""
    while True:
        a, base = text_alpha(text, font, cap, 1.0, track)
        sx = min(1.0, (maxw - 2 * margin) / a.shape[1])
        if sx >= minsx: break
        cap -= 1
    if sx < 1.0: a, base = text_alpha(text, font, cap, sx, track)
    return a, base, cap


def warm_label(text, w, h, cap, font=F_HDR, track=0.5, ring=2):
    """Yellow face, red ring (shop header CLUT 2, points CLUT 1: index 5 =
    red, 15 = yellow, 6-14 the ramp between). -> (index image, a)."""
    a, base, cap = fit(text, font, cap, w, track=track, margin=ring)
    a = pad(a, ring)
    img = compose(a, [(ring, 0, 0, 5)],
                  lambda a: np.clip(np.rint(5 + 10 * (a - 0.12) / 0.6), 6, 15).astype(np.uint8))
    return img, base + ring


def menu_label(text, cap=19, maxw=254, font=F_LBL):
    """Top menu label (KOUBAI0 TIM 1 CLUT 0): violet-blue face (4) with an
    AA ramp 5-11, cyan ring (12), teal rim (13), dark shadow (2) one px
    down-right, as the JP pieces."""
    a, base, cap = fit(text, font, cap, maxw - 4, minsx=0.8)
    a = pad(a, 3)
    img = compose(a, [(2, 1, 1, 2), (2, 0, 0, 13), (1, 0, 0, 12)],
                  lambda a: np.clip(np.rint(12 - 8 * (a - 0.15) / 0.6), 4, 11).astype(np.uint8))
    ys, xs = np.where(img > 0)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], base + 3 - ys.min()


LOGO_BOX = (170, 44, 346, 140)          # JP おまけ logo area in TIM 0 (x0, y0, x1, y1)
LOGO_CX, LOGO_CY = 256, 92              # centre of the new logo
SHADOW_RGB = (0x28, 0x28, 0x30)


def hsv_rgb(h, s=1.0, v=0.97):
    """h in degrees (array) -> (..., 3) floats 0-255."""
    h = (np.asarray(h, np.float32) % 360) / 60
    c = v * s; x = c * (1 - np.abs(h % 2 - 1)); m = v - c
    z = np.zeros_like(h); i = h.astype(int) % 6
    r = np.choose(i, [c + z, x, z, z, x, c + z]); g = np.choose(i, [x, c + z, c + z, x, z, z])
    b = np.choose(i, [z, z, x, c + z, c + z, x])
    return np.stack([r + m, g + m, b + m], -1) * 255


def logo_rgba(text, cap=40, maxw=236, font=F_LOGO, ring=5, sh=(3, 3), track=1.0):
    """White letters, rainbow ring (violet at the left to red at the right),
    dark drop shadow, as the JP おまけ logo. -> (rgb float, alpha 0..1)."""
    a, base, cap = fit(text, font, cap, maxw - 2 * ring - sh[0], minsx=0.85, track=track)
    m = ring + max(sh) + 1
    a = pad(a, m)
    ringc = grow(a, ring)
    shad = shift(ringc, *sh)
    h, w = a.shape
    xs = np.where(ringc.max(0) > 0.5)[0]
    t = np.clip((np.arange(w) - xs[0]) / max(1, xs[-1] - xs[0]), 0, 1)
    hue = 285 * (1 - t) - 5
    rainbow = np.broadcast_to(hsv_rgb(hue)[None], (h, w, 3))
    rgb = np.zeros((h, w, 3), np.float32); al = np.zeros((h, w), np.float32)
    def over(c, k):
        nonlocal rgb, al
        k = np.clip(k, 0, 1)[..., None]
        rgb = rgb * (1 - k) + np.asarray(c, np.float32) * k; al = np.maximum(al, k[..., 0])
    over(SHADOW_RGB, np.clip((shad - 0.3) * 3, 0, 1))
    over(rainbow, np.clip((ringc - 0.3) * 3, 0, 1))
    over((248, 248, 248), np.clip((a - 0.25) * 2.5, 0, 1))
    return rgb, al


def logo_mask(bg_rgb):
    """JP logo pixels inside LOGO_BOX: saturated, white or very dark, closed
    and grown by 2 px."""
    x0, y0, x1, y1 = LOGO_BOX
    c = bg_rgb[y0:y1, x0:x1].astype(np.float32)
    mx, mn = c.max(-1), c.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    m = (sat > 0.55) | (mn > 238) | (mx < 120)
    m = ndimage.binary_closing(m, disk(2))
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, disk(1))
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.where(sizes > 40)[0])
    m = ndimage.binary_dilation(m, disk(2))
    full = np.zeros(bg_rgb.shape[:2], bool); full[y0:y1, x0:x1] = m
    return full


def inpaint(rgb, mask, iters=600):
    """Fill mask by repeated neighbour averaging (the background there is a
    soft blur)."""
    out = rgb.astype(np.float32).copy()
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    sub = out[y0:y1, x0:x1]; m = mask[y0:y1, x0:x1]
    # start from the row/column mean of the known pixels
    known = ~m
    for ch in range(3):
        s = sub[..., ch]
        fill = np.where(known, s, np.nan)
        rowm = np.nanmean(fill, 1, keepdims=True); colm = np.nanmean(fill, 0, keepdims=True)
        s[m] = ((np.nan_to_num(rowm) + np.nan_to_num(colm)) / 2 * np.ones_like(s))[m]
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4
    for _ in range(iters):
        for ch in range(3):
            s = sub[..., ch]
            s[m] = ndimage.convolve(s, k, mode='nearest')[m]
    out[y0:y1, x0:x1] = sub
    return out


def quantise(rgb, pal):
    """RGB float (h, w, 3) -> nearest palette index (pal (n, 3))."""
    p = pal.astype(np.float32)
    d = ((rgb[..., None, :] - p[None, None]) ** 2 * np.array([0.3, 0.59, 0.11])).sum(-1)
    return d.argmin(-1).astype(np.uint8)


# TIM 3 header piece. JP u104 v72 122x42 at (195,40); the free texels
# around it allow u101-230, v70-117 (icons from u232, the bar above v69).
HDR_JP = dict(u=104, v=72, w=122, h=42, x=195, y=40)
HDR = dict(u=101, v=70, w=130, h=48, x=256 - 65, y=61 - 24)
PTS = dict(u=0, v=40, w=256, h=32, bar0=20)            # points label piece
HDR_CODE = {'u': 0x4bc4, 'v': 0x4bcc, 'w': 0x4bd4, 'h': 0x4bdc, 'x': 0x4be4, 'y': 0x4bec}


def header_piece(text, font=F_HDR, cap=34, lines=1, track=0.8):
    """Index image (HDR h x w) for the shop header, text centred."""
    W, H = HDR['w'], HDR['h']
    out = np.zeros((H, W), np.uint8)
    parts = [text] if lines == 1 else text.split(' ', 1)
    imgs = [warm_label(p, W, H, cap, font, track)[0] for p in parts]
    tot = sum(i.shape[0] for i in imgs) - 2 * (len(imgs) - 1)
    y = (H - tot) // 2
    for img in imgs:
        ys, xs = np.where(img > 0); img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape; x = (W - w) // 2
        if h > H or w > W: raise ValueError('header %r %dx%d > %dx%d' % (text, w, h, W, H))
        sub = out[y:y + h, x:x + w]; sub[img > 0] = img[img > 0]
        y += h - 2
    return out


def points_piece(jp, text, font=F_HDR, cap=17, x0=12, track=0.8):
    """Rebuild the points label piece: bar rows from the JP piece with the
    text removed, then the EN text (same ink rules) at x0, on the JP
    baseline."""
    W, H, B0 = PTS['w'], PTS['h'], PTS['bar0']
    out = np.zeros((H, W), np.uint8)
    for r in range(B0, H):
        row = jp[r]
        if not row.any(): continue
        nz = np.where(row > 0)[0]
        fill = row[W - 4]
        out[r, nz[0]:] = fill
        out[r, nz[0]:x0 - 1] = row[nz[0]:x0 - 1]
    img, base = warm_label(text, W - x0, H, cap, font, track)
    ys, xs = np.where(img > 0); img = img[ys.min():, xs.min():xs.max() + 1]; base -= ys.min()
    y = max(0, 19 - base)                 # JP baseline: row 19 of the piece
    h, w = img.shape
    sub = out[y:y + h, x0:x0 + w]; m = img > 0; sub[m] = img[m]
    return out


# ---------------------------------------------------------------- patches
TIM3, TIM0, TIM1 = 0x1e000, 0x0, 0x3c800
MENU_KEYS = ('menu_store', 'menu_list', 'menu_back')
MENU_CAP, MENU_FONT = 20, (FREDOKA, (500, 75))
LOGO_CAP, LOGO_FONT = 44, (FREDOKA, (650, 90))
BAR_MID = 16                          # cursor bar centre row in a 40-row label slot


def patch_tim3(tiz, gfx, code):
    """Header and points label in KOUBAI.TIZ TIM 3; header rect in code."""
    px = tim_get(tiz, TIM3)
    j = HDR_JP; px[j['v']:j['v'] + j['h'], j['u']:j['u'] + j['w']] = 0
    h = HDR; px[h['v']:h['v'] + h['h'], h['u']:h['u'] + h['w']] = header_piece(gfx['store_hdr'])
    p = PTS
    jp = px[p['v']:p['v'] + p['h'], p['u']:p['u'] + p['w']].copy()
    px[p['v']:p['v'] + p['h'], p['u']:p['u'] + p['w']] = points_piece(jp, gfx['points_lbl'])
    tim_put(tiz, TIM3, px)
    for k, off in HDR_CODE.items():
        w = struct.unpack_from('<I', code, off)[0]
        assert w >> 16 == 0x2402 and (w & 0xffff) == HDR_JP[k], (k, hex(w))
        struct.pack_into('<I', code, off, 0x24020000 | HDR[k])


def menu_piece(text):
    """40-row slot image (cropped to ink width) for one top menu label."""
    img, base = menu_label(text, cap=MENU_CAP, font=MENU_FONT)
    out = np.zeros((40, img.shape[1]), np.uint8)
    y = BAR_MID + MENU_CAP // 2 - base
    h = img.shape[0]
    if y < 0 or y + h > 40: raise ValueError('menu label %r rows %d-%d' % (text, y, y + h))
    out[y:y + h] = img
    return out


def patch_menu(tiz0, gfx, code, problems):
    """KOUBAI0 TIM 1 labels (rows 0-120 and 140-180) and the (w, x, y)
    table; the labels are centred on the cursor bar (x 128-383)."""
    px = tim_get(tiz0, TIM1)
    for k, key in enumerate(MENU_KEYS + ('menu_store2',)):
        v = 40 * k if k < 3 else 140
        img = menu_piece(gfx[key]); w = img.shape[1]
        if w > 256: problems.append('%s: %d px > 256' % (key, w))
        px[v:v + 40, :] = 0; px[v:v + 40, :w] = img
        if k < 3:
            o = MENU_TAB + 6 * k
            jw, jx, jy = struct.unpack_from('<3H', code, o)
            struct.pack_into('<3H', code, o, w, 256 - w // 2, jy)
    tim_put(tiz0, TIM1, px)


def logo_layer(text):
    return logo_rgba(text, cap=LOGO_CAP, font=LOGO_FONT)


def patch_logo(tiz0, gfx):
    """Repaint the background behind おまけ and draw the EN logo in
    KOUBAI0 TIM 0 (8bpp, palette kept)."""
    px = tim_get(tiz0, TIM0); cl = tim_at(tiz0, TIM0)[4]
    pal = rgb15(cl); rgb = pal[px].astype(np.float32)
    mask = logo_mask(rgb.astype(np.uint8))
    rgb = inpaint(rgb, mask)
    lr, la = logo_layer(gfx['logo'])
    h, w = la.shape; x = LOGO_CX - w // 2; y = LOGO_CY - h // 2
    sub = rgb[y:y + h, x:x + w]
    rgb[y:y + h, x:x + w] = sub * (1 - la[..., None]) + lr * la[..., None]
    ys, xs = np.where(mask | np.pad(la > 0.02, ((y, px.shape[0] - y - h), (x, px.shape[1] - x - w))))
    ok = np.where(cl != 0)[0]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    q = ok[quantise(rgb[y0:y1, x0:x1], pal[ok])]
    keep = px[y0:y1, x0:x1]
    chg = (mask | np.pad(la > 0.02, ((y, px.shape[0] - y - h), (x, px.shape[1] - x - w))))[y0:y1, x0:x1]
    px[y0:y1, x0:x1] = np.where(chg, q, keep)
    tim_put(tiz0, TIM0, px)
    return dict(logo=(x, y, w, h), repaint=int(mask.sum()))


# ---------------------------------------------------------------- main
def build(problems):
    disc = open(os.path.join(ROOT, BIN_DISC), 'rb').read()
    tl = load_tl(); gfx = load_gfx()
    code, font, st = build_text(disc, tl, problems)
    tiz = bytearray(open(os.path.join(ROOT, TIZ_SRC), 'rb').read())
    tiz0 = bytearray(open(os.path.join(ROOT, TIZ0_SRC), 'rb').read())
    n0, n00 = len(tiz), len(tiz0)
    font_tim(tiz, font)
    patch_tim3(tiz, gfx, code)
    patch_menu(tiz0, gfx, code, problems)
    st.update(patch_logo(tiz0, gfx))
    assert len(code) == len(disc) and len(tiz) == n0 and len(tiz0) == n00
    return code, tiz, tiz0, font, st


def main():
    os.chdir(ROOT)
    check = '--check' in sys.argv
    problems = []
    code, tiz, tiz0, font, st = build(problems)
    ctiz = lz.compress(bytes(tiz), name='KOUBAI.TIZ')
    ctiz0 = lz.compress(bytes(tiz0), name='KOUBAI0.TIZ')
    for name, data, slot in (('KOUBAI.BIN', code, SLOT_BIN), ('KOUBAI.TIZ', ctiz, SLOT_TIZ),
                             ('KOUBAI0.TIZ', ctiz0, SLOT_TIZ0)):
        if len(data) > slot: problems.append('%s: %d bytes > slot %d' % (name, len(data), slot))
    for p in problems: print('PROBLEM', p)
    print('koubai: pairs %d used, %d cells free; pool %d of %d bytes; KOUBAI.TIZ %d of %d, '
          'KOUBAI0.TIZ %d of %d bytes%s' % (st['pairs'], st['pairs_free'], st['pool_used'],
          st['pool_total'], len(ctiz), SLOT_TIZ, len(ctiz0), SLOT_TIZ0,
          ' (check only)' if check else ''))
    if check:
        sys.exit(1 if any('bytes > slot' in p or 'cells >' in p for p in problems) else 0)
    if any('bytes > slot' in p for p in problems): sys.exit('not written')
    open('private/build/KOUBAI.BIN', 'wb').write(code)
    open('private/build/KOUBAI.TIZ', 'wb').write(ctiz)
    open('private/build/KOUBAI0.TIZ', 'wb').write(ctiz0)


if __name__ == '__main__':
    main()
