# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""PASS overlay in English: the EDIT Register password input and display
screens and the CPU Edit (熱血戦術指導) mode. Topic: private/re/pass.

Reads tl/PASS.txt (ID: EN), data/latin.tsv (MOJI2 Latin codes),
private/unpacked/MOJI2.TIZ.lz (glyph shapes, for ink positions), the disc
PASS.BIN and private/build/EDITMOJI.TIZ (written by build_kanri.py, so run
this after it). Writes private/build/PASS.BIN and private/build/EDITMOJI.TIZ.
Idempotent on its own output. --check only reports problems.

PASS.BIN (loads at 0x8017a000, slot 0x8000 B):
- Drawer 0x8017af88 (records: s16 x, s16 y, u32 ptr, s16 tighten, u16 0):
  blank cells (0xfff) no longer take a 0x38-byte primitive slot, so the
  buffer limits count letters only; the centre flag now gives
  x = (rec.x + 512 - 14 n) / 2, so centred strings use tighten 6 and
  rec.x = 2 * start - 512 + 14 n.
- Strings are rebuilt into the JP pools and the file tail.
- Status panel: right column at +224 px (JP +200), values through a stub
  that sets tighten 7, Vigor "N本" -> N.
- Header loop: 4 sprites of 40x40 from TIM1 (0,0)-(159,39) at x 176.
- Grid 決定 -> OK (MOJI2 Latin codes).
EDITMOJI.TIZ: header art, 熱血戦術指導 label, usage words (TIM1, table
0x69c), range tab text (TIM0, table 0x664).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import fontpath
import lz
from build_sysmsg import charmap

PASS_DISC = 'private/disc/DAT1/PASS.BIN'
PASS_OUT = 'private/build/PASS.BIN'
EM_OUT = 'private/build/EDITMOJI.TIZ'
EM_SLOT = 49 * 2048
MOJI = 'private/unpacked/MOJI2.TIZ.lz'
TL = 'tl/PASS.txt'
BASE = 0x8017a000
ORIG_LEN = 0x7fb4
MAX_LEN = 0x8000
# JP string pools, the dead routine 0x8017b3ec-0x8017b61f (no jal, branch or
# pointer to it in PASS, KANRI, the exe or any DAT1 overlay) and the file
# tail (the last 12 B hold the value stub)
FREE = [(0x00c, 0x124), (0x3bc, 0x420), (0x4a4, 0x524), (0x56c, 0x59c), (0x13ec, 0x1620),
        (ORIG_LEN, MAX_LEN)]
DRAWER = 0x8017af88


def F(a): return a - BASE


def jal(t): return 0x0c000000 | (t >> 2 & 0x3ffffff)


def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.startswith('#') or ':' not in line: continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------- glyphs
def moji_px():
    d = open(MOJI, 'rb').read()
    p = 8 + struct.unpack_from('<I', d, 8)[0] + 12
    raw = np.frombuffer(d[p:p + 512 * 256], np.uint8).reshape(256, 512)
    px = np.zeros((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px


class Glyphs:
    def __init__(self, tim1):
        self.m = moji_px(); self.t1 = tim1; self.cm = charmap()

    def cell(self, c):
        c &= 0xfff
        r, col = c >> 4 & 15, c & 15
        if c >> 8 == 0xf:
            return self.t1[20 * r:20 * r + 20, (20 * col) % 256:(20 * col) % 256 + 20]
        t = c >> 8
        return self.m[20 * r:20 * r + 20, 256 * t + 20 * col:256 * t + 20 * col + 20]

    def encode(self, s):
        out = []
        for ch in s:
            if ch not in self.cm: raise ValueError(f'no glyph for {ch!r} in {s!r}')
            out.append(self.cm[ch])
        return out

    def ink(self, codes, t, start=0):
        """(left, right) ink columns relative to the string start."""
        lo = hi = None
        for i, c in enumerate(codes):
            if c & 0xfff == 0xfff: continue
            cols = np.where(self.cell(c).any(0))[0]
            if not len(cols): continue
            a = start + i * (20 - t)
            lo = a + cols[0] if lo is None else min(lo, a + cols[0])
            hi = a + cols[-1] if hi is None else max(hi, a + cols[-1])
        return lo, hi


# ---------------------------------------------------------------- layout
# Screen facts (findings.org): status panel x 17-491, help panel x 24-362,
# CPU Edit buttons x 372-490 (centre 431), choice cursor boxes centred at
# 280 + 60k + 21 (0x8017d7a4), registered Yes/No boxes centred 202 / 312.
LAB_L, VAL_L, LAB_R, COL_D = 30, 144, 269, 224   # ink left x; value col offset
PANEL_T = 7                                       # status panel tighten
BTN_C, FOOT_C = 431, 193
CHOICE_C = (301, 361, 421)
YESNO_C = (202, 312)
NICK_X = 140                                      # nickname cell x (0x8017dd14)

# id -> (records, slots, centred flag, mode, arg, tighten, clut a3)
# mode: 'left' arg = ink left x ('jp' = JP ink left); 'centre' arg = centre.
LAYOUT = {
    'wrong':  ([0x148], 16, 1, 'centre', 256, 6, 0xa),
    'done1':  ([0x184], 13, 1, 'centre', 256, 6, 0xa),
    'done2':  ([0x190], 19, 1, 'centre', 256, 6, 0xa),
    'tconf':  ([0x1a8], 7, 0, 'left', 'jp', 6, 0xf),
    'btn0':   ([0x420], 6, 0, 'centre', BTN_C, 6, 0xf),
    'btn1':   ([0x42c], 6, 0, 'centre', BTN_C, 6, 0xf),
    'btn2':   ([0x438], 6, 0, 'centre', BTN_C, 6, 0xf),
    'btn3':   ([0x444], 6, 0, 'centre', BTN_C, 6, 0xf),
    'lab0':   ([0x450], 8, 0, 'left', LAB_L, PANEL_T, 9),
    'lab1':   ([0x45c], 8, 0, 'left', LAB_L, PANEL_T, 9),
    'lab2':   ([0x468], 8, 0, 'left', LAB_L, PANEL_T, 9),
    'lab3':   ([0x474], 8, 0, 'left', LAB_R, PANEL_T, 9),
    'lab4':   ([0x480], 8, 0, 'left', LAB_R, PANEL_T, 9),
    'male':   ([0x48c], 6, 0, 'left', LAB_R, PANEL_T, 0xf),
    'female': ([0x498], 6, 0, 'left', LAB_R, PANEL_T, 0xf),
    'help0.1': ([0x524], 16, 0, 'left', 'jp', 6, 0xf),
    'help0.2': ([0x530], 16, 0, 'left', 'jp', 6, 0xf),
    'help1.1': ([0x53c], 16, 0, 'left', 'jp', 6, 0xf),
    'help1.2': ([0x548], 16, 0, 'left', 'jp', 6, 0xf),
    'help2.1': ([0x554], 16, 0, 'left', 'jp', 6, 0xf),
    'help2.2': ([0x560], 16, 0, 'left', 'jp', 6, 0xf),
    'val0':   ([0x59c], 16, 0, 'left', VAL_L, PANEL_T, 0xf),
    'val1':   ([0x5a8], 16, 0, 'left', VAL_L, PANEL_T, 0xf),
    'val2':   ([0x5b4], 16, 0, 'left', VAL_L, PANEL_T, 0xf),
    'val3':   ([0x5c0], 16, 0, 'left', VAL_L, PANEL_T, 0xf),
    'val4':   ([0x5cc, 0x5d8], 16, 0, 'left', VAL_L, PANEL_T, 0xf),
}
# anykey: one string, two records (0x178 centred under the wrong password
# line, 0x16c footer under a shown password, flag 0)
ANYKEY = ([0x178], [0x16c], 16)
# choice strings: id prefix, record, slots, centred flag, words (suffix,
# target centre), allowed tighten values, clut
CHOICES = {
    'conf':  (0x13c, 32, 0, [('name', None), ('add', CHOICE_C[0]), ('quit', CHOICE_C[1]),
                             ('stats', CHOICE_C[2])], (6, 7), 0xf),
    'tconf': (0x124, 9, 0, [('yes', CHOICE_C[0]), ('no', CHOICE_C[1])], (6,), 0xf),
    'done3': (0x19c, 8, 1, [('yes', YESNO_C[0]), ('no', YESNO_C[1])], (6,), 0xf),
}
HINTS = ([0x154, 0x160], 10)                      # two lines, columns aligned


# ---------------------------------------------------------------- PASS code
def patch_word(p, a, old, new, problems):
    cur = struct.unpack_from('<I', p, F(a))[0]
    if cur not in (old, new):
        problems.append(f'{a:#x}: {cur:#010x}, expected {old:#010x}'); return
    struct.pack_into('<I', p, F(a), new)


def patch_code(p, disc, problems):
    # drawer: blanks take no slot
    patch_word(p, 0x8017afd0, 0x0080b821, 0x2497ffc8, problems)   # addiu s7, a0, -0x38
    patch_word(p, 0x8017b078, 0x00000000, 0x26f70038, problems)   # addiu s7, s7, 0x38
    patch_word(p, 0x8017b17c, 0x26f70038, 0x00000000, problems)   # (loop delay) nop
    # drawer: centred x = (rec.x + 512 - 14 n) >> 1
    for a, o, n in zip(range(0x8017afe8, 0x8017b000, 4),
                       (0x001e1880, 0x007e1821, 0x00031880, 0x24020200, 0x00431023, 0x0002a043),
                       (0x001e18c0, 0x007e1823, 0x00031840, 0x26820200, 0x00431023, 0x0002a043)):
        patch_word(p, a, o, n, problems)
    # header: 4 sprites from x 176, table -> TIM1 cells (0..3, 0)
    patch_word(p, 0x8017bf3c, 0x24160074, 0x241600b0, problems)
    patch_word(p, 0x8017c0b4, 0x2ae20007, 0x2ae20004, problems)
    p[0x64c:0x658] = bytes([0x0b, 0, 0, 0x0b, 1, 0, 0x0b, 2, 0, 0x0b, 3, 0])
    # status values: right column +224, tighten stub, Vigor N本 -> N
    patch_word(p, 0x801801d4, 0x000710c0, 0x00071040, problems)   # sll v0, a3, 1
    patch_word(p, 0x801801dc, 0x000210c0, 0x00021140, problems)   # sll v0, v0, 5
    patch_word(p, 0x80180278, 0x240200a9, 0x24023001, problems)   # 本 -> end
    patch_word(p, 0x80180264, 0x240700ae, 0x24070000 | (VAL_L - 2), problems)


def value_stub(p, problems):
    """jal at 0x80180218 -> stub (tighten 7 at sp+0x20, then the drawer)."""
    org = BASE + len(p)
    p += struct.pack('<3I', 0x24080000 | PANEL_T, 0x08000000 | (DRAWER >> 2 & 0x3ffffff),
                     0xa7a80020)
    cur = struct.unpack_from('<I', p, F(0x80180218))[0]
    if cur != jal(DRAWER): problems.append(f'value jal: {cur:#x}')
    struct.pack_into('<I', p, F(0x80180218), jal(org))


# ---------------------------------------------------------------- strings
def rec_get(p, r): return struct.unpack_from('<hhIhH', p, r)


def jp_ink_left(disc, g, r):
    x, y, ptr, t, _ = rec_get(disc, r)
    o = F(ptr); n = struct.unpack_from('<H', disc, o)[0]
    codes = struct.unpack_from('<%dH' % n, disc, o + 2)
    return x + g.ink(codes, t)[0]


def place_words(g, words, t, x0=None):
    """words: [(codes, target centre or None)]; the first word starts the
    string (at x0 if given, else placed with the others). Each later word
    goes on the cell grid with at least one blank before it; the cell
    indices (and a free x0) minimise the largest centre error.
    Returns (codes, start x, max error)."""
    import itertools
    pitch = 20 - t
    inkc = [(lambda lo, hi: (lo + hi + 1) / 2)(*g.ink(w, t)) for w, _ in words]
    base = [0]
    for i in range(1, len(words)):
        base.append(base[-1] + len(words[i - 1][0]) + 1)
    best = None
    for extra in itertools.product(range(0, 22), repeat=len(words) - 1):
        ks = [0]
        for i, e in enumerate(extra):
            ks.append(ks[-1] + len(words[i][0]) + 1 + e)
        r = [tc - (k * pitch + c) for (w, tc), k, c in zip(words, ks, inkc) if tc is not None]
        x = x0 if x0 is not None else round((min(r) + max(r)) / 2)
        err = max(abs(v - x) for v in r)
        if best is None or err < best[2] - 1e-9: best = (ks, x, err)
    ks, x, err = best
    codes = []
    for (w, _), k in zip(words, ks):
        codes += [0xfff] * (k - len(codes)) + list(w)
    return codes, x, err


def strings(tl, g, disc, problems):
    """-> list of dicts: recs [(rec, x)], codes, t, clut, flag, id."""
    out = []

    def enc(key):
        en = tl.get(key, '')
        if en in ('', '='):
            problems.append(f'{key}: no EN'); return None
        try: return g.encode(en)
        except ValueError as e: problems.append(f'{key}: {e}'); return None

    def nslots(codes): return sum(1 for c in codes if c & 0xfff != 0xfff)

    def rx_for(flag, start, n):
        return 2 * start - 512 + 14 * n if flag else start

    for key, (recs, lim, flag, mode, arg, t, clut) in LAYOUT.items():
        codes = enc(key)
        if codes is None: continue
        if nslots(codes) > lim: problems.append(f'{key}: {nslots(codes)} letters > {lim}')
        lo, hi = g.ink(codes, t)
        if mode == 'left':
            left = jp_ink_left(disc, g, recs[0]) if arg == 'jp' else arg
            start = left - lo
        else:
            start = round(arg - (lo + hi + 1) / 2)
        out.append(dict(id=key, recs=[(r, rx_for(flag, start, len(codes))) for r in recs],
                        start=start, codes=codes, t=t, clut=clut, flag=flag))
    # anykey
    codes = enc('anykey')
    if codes:
        if nslots(codes) > ANYKEY[2]: problems.append(f'anykey: too long')
        lo, hi = g.ink(codes, 6)
        s1 = round(256 - (lo + hi + 1) / 2); s0 = round(FOOT_C - (lo + hi + 1) / 2)
        out.append(dict(id='anykey', recs=[(ANYKEY[0][0], rx_for(1, s1, len(codes)))],
                        start=s1, codes=codes, t=6, clut=0xa, flag=1))
        out.append(dict(id='anykey', recs=[(ANYKEY[1][0], s0)], start=s0, codes=codes, t=6,
                        clut=0xa, flag=0, share=True))
    # choice strings
    for key, (rec, lim, flag, words, ts, clut) in CHOICES.items():
        ws = [(enc(f'{key}.{w}'), c) for w, c in words]
        if any(w is None for w, _ in ws): continue
        best = None
        for t in ts:
            x0s = [None]
            if words[0][1] is None:                 # left word near the JP ink left
                j = jp_ink_left(disc, g, rec) - g.ink(ws[0][0], t)[0]
                x0s = range(j - 6, j + 3)
            for x0 in x0s:
                r = place_words(g, ws, t, x0)
                if best is None or r[2] < best[0][2]: best = (r, t)
        (codes, x0, err), t = best
        if err > 5.5: problems.append(f'{key}: word centre off by {err:.1f} px')
        if nslots(codes) > lim: problems.append(f'{key}: {nslots(codes)} letters > {lim}')
        if key == 'conf':
            hi = g.ink(ws[0][0], t)[1] + x0
            if hi >= NICK_X: problems.append(f'conf.name ends at {hi} >= {NICK_X}')
        out.append(dict(id=key, recs=[(rec, rx_for(flag, x0, len(codes)))], start=x0,
                        codes=codes, t=t, clut=clut, flag=flag, err=err))
    # hints: "WORD  word" with the second words in one column, block centred
    w = [[enc(f'hint{i}.0'), enc(f'hint{i}.1')] for i in (1, 2)]
    if all(x is not None for l in w for x in l):
        col = max(len(l[0]) for l in w) + 1
        lines = [l[0] + [0xfff] * (col - len(l[0])) + l[1] for l in w]
        ink = [g.ink(c, 6) for c in lines]
        lo = min(i[0] for i in ink); hi = max(i[1] for i in ink)
        s = round(256 - (lo + hi + 1) / 2)
        for i, (r, c) in enumerate(zip(HINTS[0], lines)):
            if nslots(c) > HINTS[1]: problems.append(f'hint{i + 1}: {nslots(c)} letters > {HINTS[1]}')
            out.append(dict(id=f'hint{i + 1}', recs=[(r, rx_for(1, s, len(c)))], start=s, codes=c,
                            t=6, clut=0xa, flag=1))
    return out


def write_strings(p, items, problems):
    free = [list(f) for f in FREE]
    for a, b in free: p[a:b] = b'\0' * (b - a)
    blobs = {}
    for it in items: blobs.setdefault(tuple(it['codes']), []).append(it)
    for codes in sorted(blobs, key=len, reverse=True):
        blob = struct.pack('<H%dH' % len(codes), len(codes), *codes)
        for f in free:
            if f[1] - f[0] >= len(blob):
                at = f[0]; p[at:at + len(blob)] = blob; f[0] += len(blob); break
        else:
            problems.append(f'no room for {blobs[codes][0]["id"]} ({len(blob)} B)'); continue
        for it in blobs[codes]:
            for r, x in it['recs']:
                y = struct.unpack_from('<h', p, r + 2)[0]
                struct.pack_into('<hhIhH', p, r, x, y, BASE + at, it['t'], 0)
    return sum(b - a for a, b in free)


# ---------------------------------------------------------------- EDITMOJI
TIM0 = None                     # set by em_load: pixel offset of TIM0
TIM1_OFF = 0x21000 + 0x14


def em_load(d):
    o = 8 + struct.unpack_from('<I', d, 8)[0] + 12
    raw = np.frombuffer(bytes(d), np.uint8, 512 * 256, o).reshape(256, 512)
    t0 = np.zeros((256, 1024), np.uint8); t0[:, 0::2] = raw & 15; t0[:, 1::2] = raw >> 4
    raw = np.frombuffer(bytes(d), np.uint8, 128 * 256, TIM1_OFF).reshape(256, 128)
    t1 = np.zeros((256, 256), np.uint8); t1[:, 0::2] = raw & 15; t1[:, 1::2] = raw >> 4
    return t0, o, t1


def em_store(d, t0, o0, t1):
    d[o0:o0 + 512 * 256] = (t0[:, 0::2] | t0[:, 1::2] << 4).astype(np.uint8).tobytes()
    d[TIM1_OFF:TIM1_OFF + 128 * 256] = (t1[:, 0::2] | t1[:, 1::2] << 4).astype(np.uint8).tobytes()


def em_clut(d, x, y):
    a = np.frombuffer(bytes(d), '<u2', 512 * 3, 20).reshape(3, 512)
    return a[y - 481, x:x + 16]


def rgb15(c):
    c = np.asarray(c, np.int32)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.float32)


def font(name, size):
    return fontpath.truetype(name, size)


def coverage(text, fnt, ss=4, track=0.0, sx=1.0):
    """Anti-aliased coverage (float 0..1) of one line, tight box, plus the
    baseline row. sx squeezes horizontally."""
    f = fontpath.truetype(fnt.path, round(fnt.size * ss))
    w = int(f.getlength(text) + track * ss * len(text)) + 8 * ss
    asc, desc = f.getmetrics()
    im = Image.new('L', (w, asc + desc + 4 * ss), 0); dr = ImageDraw.Draw(im)
    x = 2 * ss
    for ch in text:
        dr.text((x, 2 * ss), ch, font=f, fill=255)
        x += f.getlength(ch) + track * ss
    if sx != 1.0: im = im.resize((round(im.width * sx), im.height), Image.LANCZOS)
    W = -(-im.width // ss) * ss; H = -(-im.height // ss) * ss
    big = Image.new('L', (W, H), 0); big.paste(im, (0, 0))
    a = np.asarray(big, np.float32).reshape(H // ss, ss, W // ss, ss).mean((1, 3)) / 255
    base = (2 * ss + asc) / ss
    rows = np.where(a.max(1) > 0.02)[0]; cols = np.where(a.max(0) > 0.02)[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], base - rows[0]


def cap_px(fnt):
    a, _ = coverage('H', fnt); return a.shape[0]


def fit_font(name, cap, text=None, maxw=None, squeeze=0.80):
    """Font at cap height 'cap' px; squeeze if wider than maxw."""
    s = cap * 1.4
    for _ in range(6):
        f = font(name, s); c = cap_px(f); s *= cap / c
    f = font(name, s)
    sx = 1.0
    if text and maxw:
        w = coverage(text, f)[0].shape[1]
        if w > maxw: sx = max(squeeze, maxw / w)
    return f, sx


# header: MOJI2 CLUT (352,480): 1 black, 3-10 green dark->bright, 12/13/15
# blue-purple glow, 14 bright blue
HDR = dict(font='RobotoCondensed-Bold.ttf', cap=28, box=(160, 40))


def header_art(text, font_name=None, cap=None, two_lines=False):
    W, H = HDR['box']
    fn = font_name or HDR['font']; cap = cap or HDR['cap']
    if two_lines:
        a1, _ = coverage(text.split(' ', 1)[0], fit_font(fn, cap)[0])
        a2, _ = coverage(text.split(' ', 1)[1], fit_font(fn, cap)[0])
        lines = [a1, a2]
    else:
        f, sx = fit_font(fn, cap, text, W - 8)
        lines = [coverage(text, f, sx=sx)[0]]
    cov = np.zeros((H, W), np.float32)
    th = sum(l.shape[0] for l in lines) + 2 * (len(lines) - 1)
    y = (H - 3 - th) // 2
    for l in lines:
        h, w = l.shape
        if w > W - 6: l = np.asarray(Image.fromarray((l * 255).astype(np.uint8)).resize((W - 6, h), Image.LANCZOS), np.float32) / 255; w = W - 6
        x = (W - 3 - w) // 2 + 1
        cov[y:y + h, x:x + w] = np.maximum(cov[y:y + h, x:x + w], l); y += h + 2
    return style_header(cov)


def style_header(cov):
    """Green gradient fill, 1-2 px black outline, blue drop glow."""
    H, W = cov.shape
    fill = cov > 0.45
    im = Image.fromarray((fill * 255).astype(np.uint8))
    ol = np.asarray(im.filter(ImageFilter.MaxFilter(3)), bool) & ~fill
    ol2 = np.asarray(Image.fromarray(((fill | ol) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)), bool)
    shadow = np.zeros_like(fill); shadow[2:, 2:] = ol2[:-2, :-2]
    out = np.zeros((H, W), np.uint8)
    out[shadow & ~(fill | ol)] = 13
    out[ol2 & ~(fill | ol)] = 15
    out[ol] = 1
    rows = np.where(fill.any(1))[0]; top, bot = rows[0], rows[-1]
    for y in range(top, bot + 1):
        f = (y - top) / max(1, bot - top)
        idx = int(round(6 + 4 * f))              # 6 (mid) -> 10 (bright)
        out[y][fill[y]] = idx
    # top-left bevel: fill pixels with outline above or left -> darker
    up = np.zeros_like(fill); up[1:] = ol[:-1]
    lf = np.zeros_like(fill); lf[:, 1:] = ol[:, :-1]
    bev = fill & (up | lf)
    out[bev] = np.maximum(4, out[bev].astype(int) - 2).astype(np.uint8)
    return out


# 熱血戦術指導 label: TIM1 u0 v140 104x20, CLUT (240,481): 14 white, 4 dark
# grey outline, 5-13 greys between
TAC = dict(rect=(0, 140, 104, 20), font='RobotoCondensed-Bold.ttf', cap=12)


def ramp_index(a, lo_idx, hi_idx):
    """Coverage 0..1 -> index ramp lo..hi (inclusive)."""
    return np.rint(lo_idx + (hi_idx - lo_idx) * a).astype(np.uint8)


def tac_art(text):
    u, v, w, h = TAC['rect']
    f, sx = fit_font(TAC['font'], TAC['cap'], text, w - 6)
    a, base = coverage(text, f, sx=sx)
    ah, aw = a.shape
    y = max(2, min(int(round(15 - base)), h - ah - 1))     # baseline row 15
    x = (w - aw) // 2
    cov = np.zeros((h, w), np.float32); cov[y:y + ah, x:x + aw] = a
    ink = cov > 0.12
    body = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)), bool)
    out = np.zeros((h, w), np.uint8)
    out[body] = 4                                  # dark grey outline
    out[ink] = np.rint(6 + 8 * np.clip((cov[ink] - 0.12) / 0.5, 0, 1)).astype(np.uint8)
    return out


# usage words: TIM1 CLUT (240,481): 2 black, 3-10 greys; table 0x69c
# (u, v, w, x) per word; JP order 0 あまり使わない, 1 普通, 2 よく使う,
# 3 とてもよく使う. Free texels: row 7 u104-255, row 8 u0-179 (kanrinick
# uses row 8 from u180 and row 9 from u100).
USE = dict(font='DejaVuSansCondensed.ttf', cap=11, slots=[(140, 104, 256), (160, 0, 180)],
           table=0x69c, x=252)


def use_art(text):
    f, _ = fit_font(USE['font'], USE['cap'])
    a, base = coverage(text, f)
    out = np.zeros((20, a.shape[1] + 2), np.uint8)
    y = int(round(15 - base))                    # baseline at row 15
    h = a.shape[0]
    y = max(0, min(y, 20 - h))
    cov = np.zeros((20, a.shape[1] + 2), np.float32); cov[y:y + h, 1:1 + a.shape[1]] = a
    m = cov > 0.10
    out[m] = (10 - np.rint(8 * cov[m])).clip(2, 10).astype(np.uint8)
    return out


# range tabs: TIM0 96x20 tabs; table 0x664 (tpage x, clut x, u, v, screen
# x); texel x = (tpage x - 640) * 4 + u. 0 遠 (x 275), 1 中 (216),
# 2 近 (147), 3 超近 (81). Text centre (tab-local) where the overlapping
# neighbours leave it visible.
RNG = dict(font='RobotoCondensed-Bold.ttf', cap=11, centre=(50, 37, 43, 34), baseline=17)


def tab_ramp(clut, fill_idx):
    """Indices from grey (index 1) to the fill colour, by blend fraction."""
    g = rgb15(clut[1]); f = rgb15(clut[fill_idx]); c = rgb15(clut)
    d = f - g; n = (d * d).sum()
    out = []
    for i in range(1, 16):
        al = float(((c[i] - g) * d).sum() / n)
        res = float(np.abs(g + al * d - c[i]).max())
        if -0.05 <= al <= 1.05 and res < 20: out.append((al, i))
    return sorted(out)


def draw_tab(t0, d, ent, text, centre):
    tpx, cx, u, v, sx = ent
    x0 = (tpx - 640) * 4 + u
    tab = t0[v:v + 20, x0:x0 + 96]
    clut = em_clut(d, cx, 482)
    fill = int(np.bincount(tab[6:18, 2:40].ravel(), minlength=16)[1:].argmax() + 1)
    edge = []
    for r in range(5, 20):
        nz = np.where(tab[r] != 0)[0]
        if len(nz) < 3: edge.append(95); continue
        edge.append(nz[-1]); tab[r, 1:nz[-1]] = fill
    ramp = tab_ramp(clut, fill)
    f, _ = fit_font(RNG['font'], RNG['cap'])
    a, base = coverage(text, f)
    h, w = a.shape
    y = int(round(RNG['baseline'] - base)); x = int(round(centre - w / 2))
    x = max(2, x)
    als = np.array([al for al, _ in ramp]); ids = [i for _, i in ramp]
    for yy in range(h):
        for xx in range(w):
            c = a[yy, xx]
            if c < 0.08: continue
            X, Y = x + xx, y + yy
            if Y < 5 or Y >= 20 or X >= edge[Y - 5] - 1: continue
            al = 1 - c                              # 0 = grey text, 1 = fill
            tab[Y, X] = ids[int(np.abs(als - al).argmin())]
    return x, x + w - 1, fill


def patch_em(d, tl, p, problems):
    t0, o0, t1 = em_load(d)
    # header
    en = tl.get('hdr', '')
    if en: t1[0:40, 0:160] = header_art(en)
    # 熱血戦術指導
    en = tl.get('tac', '')
    if en:
        u, v, w, h = TAC['rect']; t1[v:v + h, u:u + w] = tac_art(en)
    # usage words
    slots = [list(s) for s in USE['slots']]
    for v, a, b in USE['slots']: t1[v:v + 20, a:b] = 0
    for k in range(4):
        en = tl.get(f'use{k}', '')
        if not en: continue
        art = use_art(en); w = art.shape[1]
        for s in slots:
            if s[2] - s[1] >= w:
                t1[s[0]:s[0] + 20, s[1]:s[1] + w] = art
                struct.pack_into('<4H', p, USE['table'] + 8 * k, s[1], s[0], w, USE['x'])
                s[1] += w; break
        else: problems.append(f'use{k}: no room ({w} px)')
    # range tabs
    for k in range(4):
        en = tl.get(f'rng{k}', '')
        if not en: continue
        ent = struct.unpack_from('<5H', p, 0x664 + 10 * k)
        lo, hi, _ = draw_tab(t0, d, ent, en, RNG['centre'][k])
        if hi - lo > 64: problems.append(f'rng{k}: {hi - lo + 1} px wide')
    em_store(d, t0, o0, t1)


# ---------------------------------------------------------------- main
def build(tl, problems):
    disc = open(PASS_DISC, 'rb').read()
    p = bytearray(disc)
    if not os.path.exists(EM_OUT):
        problems.append(f'{EM_OUT} missing: run build_kanri.py first'); return None, None, None
    d = bytearray(lz.decompress(open(EM_OUT, 'rb').read())[0])
    g = Glyphs(em_load(d)[2])
    patch_code(p, disc, problems)
    # 決定 -> OK
    ok = g.encode(tl.get('ok', 'OK'))
    if len(ok) != 2: problems.append('ok: must be 2 cells')
    else: struct.pack_into('<2H', p, 0x368, *ok)
    items = strings(tl, g, disc, problems)
    # strings fill the JP pools and the file tail minus 12 B for the stub
    p = p[:ORIG_LEN] + b'\0' * (MAX_LEN - 12 - ORIG_LEN)
    FREE[-1] = (ORIG_LEN, MAX_LEN - 12)
    left = write_strings(p, items, problems)
    value_stub(p, problems)
    patch_em(d, tl, p, problems)
    if len(p) > MAX_LEN: problems.append(f'PASS {len(p):#x} > {MAX_LEN:#x}')
    return p, d, dict(items=items, free=left, g=g)


def main(argv):
    check = '--check' in argv
    tl = load_tl(); problems = []
    p, d, info = build(tl, problems)
    if p is None:
        for x in problems: print('  ' + x)
        return 1
    c = lz.compress(bytes(d), name='EDITMOJI.TIZ')
    if len(c) > EM_SLOT: problems.append(f'EDITMOJI compressed {len(c)} > slot {EM_SLOT}')
    if lz.decompress(c)[0][:len(d)] != bytes(d): problems.append('EDITMOJI LZ round trip failed')
    print(f'PASS: {len(p):#x} bytes, {info["free"]} B string space left; EDITMOJI: {len(c)} bytes (slot {EM_SLOT})')
    for it in info['items']:
        if 'err' in it: print(f'  {it["id"]}: t {it["t"]}, word centres within {it["err"]:.1f} px')
    for x in problems: print('  ' + x)
    if not check and not problems:
        open(PASS_OUT, 'wb').write(p)
        open(EM_OUT, 'wb').write(c)
    print(f'PASS: problems {len(problems)}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
