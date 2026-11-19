# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Exam Practice mode in English (SIKEN and SIKEN2 overlays). Called by
build_examgfx.py, which owns every file written here. Wording:
tl/EXAMGFX.txt, tl/EXAMSHEET.txt and the lesson titles in tl/EXAM.txt.
RE: private/re/exprac/findings-gfx.org and findings-text.org.

- Menu atlas: SIKEN1.BIZ TIM 1 (0x3c800, 4bpp 768x256 at VRAM (768,0)) and
  the identical SIKEN10.BIZ TIM 1. Tpages 12 and 13 (atlas x 0-511) are
  cleared except the digits, the < > brackets, the cursor mark and four
  unlisted pieces; every string is drawn as one strip (N1 build_siken1
  style: DejaVu Sans Condensed Bold, white with a soft shadow, CLUT 1).
  Tpage 14 keeps its art; its empty rows 128-255 take the overflow. The
  title is redrawn in the JP's green CLUT 2 look. The 決定/戻る box at
  (512,64) gets OK/Back.
- Menu lists: SIKEN file 0x420 (49) and SIKEN2 file 0x210 (8) are
  repacked inside their byte spans (only the pointer tables refer to
  them) and every item is centred on x 256 through the addiu immediates.
  The lesson line "Subj <name>" keeps Subj in place; a blank spacer piece
  centres each <name> in its column.
- Practice menu: rows 0-39 of SIKEN20 TIM 0 (= SIKEN2.BIZ TIM 0) get the
  first game's three pieces (same code, CLUT 1 identical); results lists
  46, 47, 55 of SIKEN 0x7ec and SIKEN2 0x538 point at them and the item
  x at file 0x28/0x2c/0x34 centres them.
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

BASE = 0x800a2000                 # SIKEN, SIKEN2 load address
DAT1, BUILD = 'private/disc/DAT1', 'private/build'
MENU_TIM = 0x3c800                # SIKEN1 / SIKEN10 TIM 1
FONT = 'DejaVuSansCondensed-Bold.ttf'          # as N1's menus
F_TITLE = 'MPLUSRounded1c-Black.ttf'  # OFL 1.1
TLS = ('tl/EXAMGFX.txt', 'tl/EXAMSHEET.txt')

# SIKEN menu lists: file 0x420, u8 n + n x (tp/64, u, v, w, h); tp 12-14.
S_PTAB, S_N, S_SPAN = 0x420, 49, (0xa4, 0x420)
S2_PTAB, S2_N, S2_SPAN = 0x210, 8, (0x158, 0x210)
# Max pieces per list: sprite slots up to the next draw call's first slot.
S_SLOTS = {0: 1, 1: 5, 8: 4, 9: 1, 46: 8, 47: 11}
S_SLOTS.update({2 + k: 3 for k in range(6)})
S_SLOTS.update({16 + L: 12 for L in range(30)})
S2_SLOTS = {0: 2, 1: 8, 2: 11, 4: 10, 5: 10, 6: 6, 7: 4}
# x immediates (addiu a2) of each list's draw calls.
S_IMM = {0: (0x800a45dc, 0x800a46f0, 0x800a954c), 1: (0x800a4600,), 8: (0x800a4720,),
         9: (0x800a47cc,), 46: (0x800a958c,), 47: (0x800a9570,),
         'exams': (0x800a461c, 0x800a4634, 0x800a464c, 0x800a4664, 0x800a467c, 0x800a4694,
                   0x800a4738),     # the last one: chosen exam on the subject screen
         'lesson': (0x800a47ec,)}
S2_IMM = {0: (0x800a3d5c, 0x800a9618), 1: (0x800a963c,), 2: (0x800a9658,), 4: (0x800a3d80,),
          5: (0x800a3d9c,), 6: (0x800a3db4,), 7: (0x800a3dcc,)}
# Results lists (60, u8 n + n x (tpx/64, tpy/256, clx/16, cly-480, u, v, w,
# h)) and the practice menu item positions (u16 x, y).
RES_PTAB = {'SIKEN': 0x7ec, 'SIKEN2': 0x538}
PM_POS = {46: 0x28, 47: 0x2c, 55: 0x34}
PM_KEYS = {46: 'もう一度復習する', 47: '他の課題を練習する', 55: '戻る'}
CX = 256                          # menu centre (panel x 88-424, JP items)
MH = 20                           # menu strip height (JP pieces)
LESSON_GAP = 8                    # Subj .. <
BOX = (512, 64, 104, 64)          # 決定/戻る box in the atlas, CLUT 5
# Small 合格/不合格 ovals the record sheets draw from the atlas (SIKEN file
# 0x84 / SIKEN2 0x138: u 104 or 160, v 64, 56x32, tpage (896,0), CLUT 6);
# redrawn by build_examgfx.oval_words like the SIKEN20 ones.
MENU_OVALS = {'合格（小）': (616, 64, 56, 29), '不合格（小）': (672, 64, 56, 29)}

# ---------------------------------------------------------------- TIM

class Tim:
    """4bpp or 8bpp TIM inside a buffer (N1 build_examgfx.Tim)."""
    def __init__(self, buf, off):
        self.buf, self.off = buf, off
        flag, bl = struct.unpack_from('<II', buf, off + 4)
        self.bpp = 4 if flag & 3 == 0 else 8
        cw, ch = struct.unpack_from('<2H', buf, off + 16)
        self.clut = np.frombuffer(bytes(buf[off + 20:off + 20 + 2 * cw * ch]), '<u2').copy()
        p = off + 8 + bl
        _, self.vx, self.vy, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.stride, self.H = p + 12, w * 2, h
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        if self.bpp == 4:
            self.px = np.empty((h, w * 4), np.uint8)
            self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        else:
            self.px = raw.copy()
        self.W = self.px.shape[1]
        self.orig = self.px.copy()

    def store(self):
        raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8) if self.bpp == 4 else self.px
        self.buf[self.pix:self.pix + self.stride * self.H] = raw.tobytes()

    def rgba(self, k):
        n = 16 if self.bpp == 4 else 256
        c = self.clut[k * n:(k + 1) * n].astype(int)
        return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3,
                         np.where(c == 0, 0, 255)], -1).astype(np.uint8)

def tim_size(d, o):
    flag = struct.unpack_from('<I', d, o + 4)[0]
    p = o + 8
    if flag & 8: p += struct.unpack_from('<I', d, p)[0]
    return p + struct.unpack_from('<I', d, p)[0] - o

# ---------------------------------------------------------------- text

def load_tl():
    out = {}
    for path in TLS:
        jp = None
        for line in open(path, encoding='utf-8'):
            line = line.rstrip('\n')
            if line.startswith('JP: '): jp = line[4:]
            elif line.startswith('EN: ') and jp is not None:
                if line[4:].strip(): out.setdefault(jp, line[4:].strip())
                jp = None
    need = ['試験に出る実技', '試験に出る実技’９９', 'どの試験に向けて勉強しますか？',
            'どの科目を重点的に行いますか？', '試験（１～６）', '科目', '基礎のおさらい',
            '成績表を見る', '試験を行う', '○－決定', '×－戻る', 'もう一度復習する',
            '他の課題を練習する', '戻る', '１学期期末試験', '２学期期末試験', '学年末試験']
    miss = [k for k in need if k not in out]
    if miss: raise SystemExit(f'examprac: no EN for {miss} in {TLS}')
    return out

def lesson_titles():
    from build_testmj import parse_tl
    lessons, _ = parse_tl()
    return [lessons[L]['title'] for L in range(30)]

# ---------------------------------------------------------------- render
# line_alpha, blend_idx, st_menu and Atlas are N1 build_examgfx's.

_fonts = {}
def font(path, size):
    if (path, size) not in _fonts: _fonts[path, size] = fontpath.truetype(path, size)
    return _fonts[path, size]

def line_alpha(text, h, size, path=FONT, maxw=None, minsx=0.72, ss=4, base=None):
    """Alpha (h, w) of one line, capitals centred in h (or baseline at row
    base). Squeezed down to minsx, then smaller, to fit maxw."""
    while True:
        f = font(path, size * ss)
        adv = f.getlength(text) / ss
        sx = 1.0 if not maxw or adv <= maxw else maxw / adv
        if sx >= minsx or size <= 8: break
        size -= 1
    hb = f.getbbox('H', anchor='ls')
    cap = (hb[3] - hb[1]) / ss
    by = base if base is not None else (h + cap) / 2
    W = int(np.ceil(adv)) + 1
    im = Image.new('L', (W * ss, h * ss), 0)
    ImageDraw.Draw(im).text((0, round(by * ss)), text, font=f, fill=255, anchor='ls')
    w = max(1, int(np.ceil(adv * sx)))
    im = im.resize((max(w, int(round(W * sx))), h), Image.BOX)
    return (np.asarray(im, np.float32) / 255)[:, :w]

def nearest(rgb, pal, cand):
    cand = list(cand)
    P = np.array([pal[i][:3] for i in cand], float)
    d = ((rgb[..., None, :] - P) ** 2).sum(-1)
    return np.array(cand, np.uint8)[d.argmin(-1)]

def blend_idx(a, fg, bg, pal, cand, cut=0.1):
    fg, bg = np.array(fg, float), np.array(bg, float)
    out = nearest(bg + (fg - bg) * a[..., None], pal, cand)
    out[a < cut] = 0
    return out

def pad(a, l=1, r=1, t=0, b=0):
    return np.pad(a, ((t, b), (l, r)))

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

def st_menu(pal):
    """White text with a soft dark shadow to the lower right (CLUT 1)."""
    greys = list(range(2, 15))
    def f(a):
        a = pad(a, 0, 1, 0, 0)
        sh = np.maximum(shift(a, 1, 1), shift(a, 0, 1) * 0.6)
        out = np.zeros(a.shape, np.uint8)
        out[sh > 0.3] = 4
        t = blend_idx(a, (248, 248, 248), (40, 40, 40), pal, greys, cut=0.25)
        return np.where(t > 0, t, out)
    return f

# Fill index by relative row in the glyphs, as measured on the JP title:
# a thin dark top, mid green (6) above the middle, bright (9) below.
TITLE_RAMP = ((0.08, 3), (0.15, 4), (0.24, 5), (0.45, 6), (0.55, 7), (0.66, 8), (9, 9))

def st_title(a, h=32):
    """JP title look in CLUT 2: green fill getting brighter downwards
    (TITLE_RAMP, 3 dark .. 9 bright), 14 where the glyph edge meets the
    2 px black (13) outline, and a violet drop shadow 1-3 px down-right
    (12, 11, 1). a: alpha of height h with 3 px free on every side."""
    o = ndimage.grey_dilation(a, footprint=ndimage.generate_binary_structure(2, 1)) > 0.25
    o = ndimage.binary_dilation(o, ndimage.generate_binary_structure(2, 2))
    out = np.zeros(a.shape, np.uint8)
    for k, idx in ((3, 1), (2, 11), (1, 12)):
        out[shift(o, k, k) & ~o] = idx
    out[o] = 13
    ys = np.nonzero((a > 0.5).any(1))[0]
    y0, y1 = ys.min(), ys.max()
    tt = (np.arange(a.shape[0]) - y0) / max(1, y1 - y0)
    g = np.array([next(v for lim, v in TITLE_RAMP if x <= lim) for x in tt], np.uint8)
    fill = np.broadcast_to(g[:, None], a.shape)
    out[a > 0.35] = 14
    out[a > 0.6] = fill[a > 0.6]
    return out

def title_alpha(text, h=32, cap=21, maxw=250, track=0.5):
    """Title alpha with 3 px margins: Rounded M+ Black at cap px, tracked,
    squeezed to maxw."""
    ss = 4
    probe = font(F_TITLE, 100)
    b = probe.getbbox('H', anchor='ls')
    size = cap * 100 / (b[3] - b[1])
    f = font(F_TITLE, int(round(size * ss)))
    adv = sum(f.getlength(c) for c in text) / ss + track * (len(text) - 1)
    W = int(np.ceil(adv)) + 2
    im = Image.new('L', (W * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    by = (h + cap) / 2 - 1
    x = 0.0
    for c in text:
        dr.text((x * ss, by * ss), c, font=f, fill=255, anchor='ls')
        x += f.getlength(c) / ss + track
    sx = min(1.0, (maxw - 6) / W)
    im = im.resize((max(1, int(round(W * sx))), h), Image.BOX)
    a = np.asarray(im, np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    a = a[:, cols.min():cols.max() + 1]
    return pad(a, 3, 4)

class Atlas:
    """Free-texel bookkeeping (N1). Strips never cross a 256-texel tpage."""
    def __init__(self, px, free, tpw=256):
        self.px, self.free, self.tpw = px, free.copy(), tpw

    def _fits(self, w, h):
        F = self.free.astype(np.int32)
        S = np.zeros((F.shape[0] + 1, F.shape[1] + 1), np.int32)
        S[1:, 1:] = F.cumsum(0).cumsum(1)
        ok = (S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]) == w * h
        xs = np.arange(ok.shape[1])
        ok &= ((xs % self.tpw) + w <= self.tpw)[None, :]
        return ok

    def alloc(self, w, h):
        ys, xs = np.nonzero(self._fits(w, h))
        if not len(ys): return None
        y, x = int(ys[0]), int(xs[0])
        self.free[y:y + h, x:x + w] = False
        return x, y

    def put(self, strip, maxpieces=1):
        h, W = strip.shape
        if W <= self.tpw:
            p = self.alloc(W, h)
            if p:
                self.px[p[1]:p[1] + h, p[0]:p[0] + W] = strip
                return [(p[0], p[1], W)]
        if maxpieces <= 1: return None
        for w in range(min(W, self.tpw) - 1, 7, -1):
            ys, xs = np.nonzero(self._fits(w, h))
            if not len(ys): continue
            y, x = int(ys[0]), int(xs[0])
            self.free[y:y + h, x:x + w] = False
            rest = self.put(strip[:, w:], maxpieces - 1)
            if rest is None:
                self.free[y:y + h, x:x + w] = True; return None
            self.px[y:y + h, x:x + w] = strip[:, :w]
            return [(x, y, w)] + rest
        return None

    def pack(self, items, problems, what):
        """items: (key, strip, max pieces); taller and wider first."""
        got = {}
        for key, strip, budget in sorted(items, key=lambda i: (-i[1].shape[0], -i[1].shape[1])):
            r = self.put(strip, budget)
            if r is None:
                problems.append(f'{what}: no room for {key} ({strip.shape[1]}x{strip.shape[0]})'); r = []
            got[key] = r
        return got

# ---------------------------------------------------------------- lists

def read_lists(d, ptab, n, size):
    out = []
    for i in range(n):
        p = struct.unpack_from('<I', d, ptab + 4 * i)[0] - BASE
        out.append([tuple(d[p + 1 + size * j:p + 1 + size * (j + 1)]) for j in range(d[p])])
    return out

def repack(d, ptab, span, lists, size=5):
    """Write all lists back to back from span[0] and point the table at
    them; the rest of the span is zeroed."""
    pos = span[0]
    for i, pieces in enumerate(lists):
        d[pos] = len(pieces)
        for j, e in enumerate(pieces):
            d[pos + 1 + size * j:pos + 1 + size * (j + 1)] = bytes(e)
        struct.pack_into('<I', d, ptab + 4 * i, BASE + pos)
        pos += 1 + size * len(pieces)
    assert pos <= span[1], (hex(pos), hex(span[1]))
    d[pos:span[1]] = bytes(span[1] - pos)
    return span[1] - pos

def set_imm(d, ram, val):
    o = ram - BASE
    w = struct.unpack_from('<I', d, o)[0]
    assert w >> 16 == 0x2406, (hex(ram), hex(w))       # addiu a2, zero, imm
    struct.pack_into('<I', d, o, (w & 0xffff0000) | (val & 0xffff))

def piece(x, y, w, h):
    return (12 + x // 256, x % 256, y, w, h)

def pw(pieces):
    return sum(e[3] for e in pieces)

# ---------------------------------------------------------------- menu atlas

def box_words(px, tl, log):
    """決定/戻る box: clear the dashes and words (the 7/0 checker of the
    box fill comes back) and write OK / Back after the button icons in the
    JP glyph colours (CLUT 5: 1 white, 2-5 greys, 6 dark edge)."""
    x0, y0, w, h = BOX
    sub = px[y0:y0 + h, x0:x0 + w]
    yy, xx = np.mgrid[0:h, 0:w]
    clr = (xx >= 31) & (xx <= 100) & (yy >= 6) & (yy <= 56)
    sub[clr] = np.where((xx + yy)[clr] % 2 == 0, 7, 0)
    grey = {1: 248, 2: 200, 3: 160, 4: 112, 5: 72, 6: 24}
    for key, icon_c in (('○－決定', 17.5), ('×－戻る', 41.5)):
        a = line_alpha(tl[key], 24, 17, FONT, maxw=60)
        ys = np.nonzero((a > 0.3).any(1))[0]
        a = pad(a, 0, 2, 0, 2)
        sh = np.maximum(shift(a, 1, 1), shift(a, 0, 1))
        out = np.zeros(a.shape, np.uint8)
        out[sh > 0.3] = 6
        v = 248 * a + 40 * (1 - a)
        idx = np.array([min(grey, key=lambda k: abs(grey[k] - c)) for c in v.ravel()],
                       np.uint8).reshape(a.shape)
        out[a > 0.2] = idx[a > 0.2]
        top = int(round(icon_c - (ys.min() + ys.max()) / 2))
        ty, tx = top, 35
        reg = sub[ty:ty + out.shape[0], tx:tx + out.shape[1]]
        m = out > 0
        reg[m] = out[m]
        log.append(f'box {tl[key]!r} at x {x0 + tx}, rows {ty}..{ty + out.shape[0]}')

def build_menu(tl, titles, problems, log):
    """Repaint the menu atlas pixels of TIM t (SIKEN1 TIM 1) and return the
    new SIKEN and SIKEN2 lists and x immediates: (lists1, imm1, lists2,
    imm2)."""
    from lz import decompress
    d1 = bytearray(decompress(open('private/disc/DAT2/SIKEN1.BIZ', 'rb').read())[0])
    t = Tim(d1, MENU_TIM)
    J1 = read_lists(open(f'{DAT1}/SIKEN.BIN', 'rb').read(), S_PTAB, S_N, 5)
    J2 = read_lists(open(f'{DAT1}/SIKEN2.BIN', 'rb').read(), S2_PTAB, S2_N, 5)
    px = t.px
    free = np.zeros(px.shape, bool)
    free[:, :512] = True
    for x, y, w, h in ((0, 0, 240, 20),        # digits, > <
                       (224, 20, 20, 20),      # cursor mark (list 15)
                       (120, 212, 120, 44)):   # 度 ' " のこりTIME: no list, kept
        free[y:y + h, x:x + w] = False
    px[free] = 0
    free[128:, 512:] = px[128:, 512:] == 0
    A = Atlas(px, free)
    white = st_menu(t.rgba(1))
    def strip(text, maxw=330):
        return white(line_alpha(text, MH, 15, FONT, maxw=maxw, minsx=0.6))
    exam = tl['試験（１～６）']
    items = [('title', st_title(title_alpha(tl['試験に出る実技'])), 1),
             ('title99', st_title(title_alpha(tl['試験に出る実技’９９'], maxw=280)), 2),
             ('q_exam', strip(tl['どの試験に向けて勉強しますか？']), 4),
             ('q_subj', strip(tl['どの科目を重点的に行いますか？']), 4),
             ('exam', strip(exam + ' '), 1),
             ('kamoku', strip(tl['科目'], 60), 1),
             ('records', strip(tl['成績表を見る'], 250), 1),
             ('basics', strip(tl['基礎のおさらい'], 250), 1),
             ('take', strip(tl['試験を行う'], 250), 1),
             ('term1', strip(tl['１学期期末試験'], 250), 1),
             ('term2', strip(tl['２学期期末試験'], 250), 1),
             ('term3', strip(tl['学年末試験'], 250), 1)]
    for L in range(30):
        items.append((f'L{L}', strip(titles[L], 200), 2))
    got = A.pack(items, problems, 'menu atlas')
    if problems: return t, d1, None, None, None, None
    W = {k: sum(w for _, _, w in r) for k, r in got.items()}
    P = {k: [piece(x, y, w, 32 if k.startswith('title') else MH) for x, y, w in r]
         for k, r in got.items()}
    # lesson line: Subj, then <name> centred in a column as wide as the
    # longest <name>; the spacer piece reads blank texels
    lt, gt = J1[16][0], J1[16][-1]                 # < and > (20 px each)
    inner = max(W[f'L{L}'] for L in range(30))
    pads = [(inner - W[f'L{L}']) // 2 for L in range(30)]
    sp = A.alloc(max(pads), MH) if max(pads) else None
    if max(pads) and sp is None: problems.append('menu atlas: no room for the spacer')
    total = W['kamoku'] + LESSON_GAP + 20 + inner + 20
    x0 = CX - total // 2
    lists1 = [list(l) for l in J1]
    lists2 = [list(l) for l in J2]
    imm1, imm2 = {}, {}
    lists1[0] = P['title']; imm1[0] = CX - W['title'] // 2
    lists1[1] = P['q_exam']; imm1[1] = CX - W['q_exam'] // 2
    lists1[8] = P['q_subj']; imm1[8] = CX - W['q_subj'] // 2
    for k in range(6):
        lists1[2 + k] = P['exam'] + [J1[2 + k][-1]]          # + the JP digit 1-6
    imm1['exams'] = CX - (W['exam'] + 20) // 2
    lists1[9] = P['kamoku']; imm1[9] = x0
    for L in range(30):
        sp_p = [piece(sp[0], sp[1], pads[L], MH)] if pads[L] else []
        lists1[16 + L] = sp_p + [lt] + P[f'L{L}'] + [gt]
    imm1['lesson'] = x0 + W['kamoku'] + LESSON_GAP
    lists1[46] = P['records']; imm1[46] = CX - W['records'] // 2
    lists1[47] = P['basics']; imm1[47] = CX - W['basics'] // 2
    lists2[0] = P['title99']; imm2[0] = CX - W['title99'] // 2
    lists2[1] = P['take']; imm2[1] = CX - W['take'] // 2
    lists2[2] = P['records']; imm2[2] = imm1[46]
    lists2[4] = P['q_exam']; imm2[4] = imm1[1]
    for i, k in ((5, 'term1'), (6, 'term2'), (7, 'term3')):
        lists2[i] = P[k]; imm2[i] = CX - W[k] // 2
    for lists, slots, name in ((lists1, S_SLOTS, 'SIKEN'), (lists2, S2_SLOTS, 'SIKEN2')):
        for i, n in slots.items():
            if len(lists[i]) > n: problems.append(f'{name} list {i}: {len(lists[i])} pieces > {n} slots')
    box_words(px, tl, log)
    end = x0 + total
    log.append(f'menu: title w {W["title"]}, lesson line x {x0}..{x0 + W["kamoku"] + LESSON_GAP + 20 + inner + 20}'
               f' (longest name {inner}), exam item w {W["exam"] + 20}')
    if x0 < 92 or end > 420: problems.append(f'menu: lesson line {x0}..{end} leaves the panel')
    for k in ('q_exam', 'q_subj'):
        if W[k] > 330: problems.append(f'menu: {k} {W[k]} px')
    log.append(f'menu atlas: tpage 14 rows 128+ used: {int((~A.free[128:, 512:] & free[128:, 512:]).sum())} texels')
    t.store()
    return t, d1, lists1, imm1, lists2, imm2

# ---------------------------------------------------------------- practice menu

def practice(t0, tl, problems):
    """N1 practice pieces into rows 0-39 of the SIKEN20 / SIKEN2.BIZ TIM 0
    (4bpp 256x256 at VRAM (896,256)). Returns {list: pieces}, {list: w}."""
    free = np.zeros(t0.px.shape, bool)
    for x, y, w, h in ((0, 0, 256, 20), (0, 20, 160, 20)):
        t0.px[y:y + h, x:x + w] = 0; free[y:y + h, x:x + w] = True
    PA = Atlas(t0.px, free)
    st = st_menu(t0.rgba(1))
    lists, width = {}, {}
    for li in (46, 47, 55):
        s = st(line_alpha(tl[PM_KEYS[li]], 20, 15, FONT))
        r = PA.put(s, 1)
        if r is None: problems.append(f'practice menu: no room for {li}'); continue
        (x, y, w), = r
        lists[li] = [(14, 1, 1, 3, x, y, w, 20)]
        width[li] = w
    return lists, width

# ---------------------------------------------------------------- overlays

def patch_overlay(name, lists, imm, pm, pmw, problems, log):
    """SIKEN or SIKEN2.BIN from private/build (when another tool wrote it)
    or the disc; every byte written here is absolute, so reruns are
    clean."""
    p = f'{BUILD}/{name}.BIN'
    d = bytearray(open(p if os.path.exists(p) else f'{DAT1}/{name}.BIN', 'rb').read())
    ptab, span = (S_PTAB, S_SPAN) if name == 'SIKEN' else (S2_PTAB, S2_SPAN)
    room = repack(d, ptab, span, lists)
    IMM = S_IMM if name == 'SIKEN' else S2_IMM
    for k, rams in IMM.items():
        for ram in rams: set_imm(d, ram, imm[k])
    disc = open(f'{DAT1}/{name}.BIN', 'rb').read()
    R = RES_PTAB[name]
    for li, pieces in pm.items():
        q = struct.unpack_from('<I', disc, R + 4 * li)[0] - BASE
        assert len(pieces) <= disc[q], (name, li)
        d[q] = len(pieces)
        d[q + 1:q + 1 + 8 * disc[q]] = bytes(8 * disc[q])
        for j, e in enumerate(pieces): d[q + 1 + 8 * j:q + 9 + 8 * j] = bytes(e)
        oy = struct.unpack_from('<2H', disc, PM_POS[li])[1]
        struct.pack_into('<2H', d, PM_POS[li], CX - pmw[li] // 2, oy)
    log.append(f'{name}: lists repacked, {room} bytes spare in the span')
    return bytes(d)

# ---------------------------------------------------------------- previews

def draw_list(canvas, t, pieces, x, y, clut):
    for tp, u, v, w, h in pieces:
        ax = (tp - 12) * 256 + u
        src = t.rgba(clut)[t.px[v:v + h, ax:ax + w]]
        sub = canvas[y:y + h, x:x + w]
        s = src[:sub.shape[0], :sub.shape[1]]
        m = s[..., 3] > 0
        sub[m] = s[..., :3][m]
        x += w

def backdrop():
    """SIKEN1 TIM 0 (the menu backdrop) with the half-dark panel."""
    from lz import decompress
    d = decompress(open('private/disc/DAT2/SIKEN1.BIZ', 'rb').read())[0]
    t = Tim(bytearray(d), 0)
    bg = t.rgba(0)[t.px][..., :3].copy()
    bg[120:376, 88:425] //= 2
    return bg

def screens(t, lists1, imm1, lists2, imm2, jp=False):
    """Mock-ups of the five menu screens: [(name, rgb)] with the lists at
    their call-site x, y (SIKEN and SIKEN2 draw calls)."""
    bg = backdrop()
    out = []
    def scr(calls):
        c = bg.copy()
        for lst, x, y, cl in calls: draw_list(c, t, lst, x, y, cl)
        return c
    L1, L2 = lists1, lists2
    box = (L1[48], 352, 388, 5)
    out.append(('top', scr([(L1[0], imm1[0], 48, 2), (L1[47], imm1[47], 206, 1),
                            (L1[46], imm1[46], 254, 1), box])))
    out.append(('exams', scr([(L1[0], imm1[0], 48, 2), (L1[1], imm1[1], 136, 1)] +
                             [(L1[2 + k], imm1['exams'], 176 + 32 * k, 1) for k in range(6)] + [box])))
    for L in (0, 11):
        e = L // 5
        out.append((f'subj{L + 1}', scr([(L1[0], imm1[0], 48, 2), (L1[8], imm1[8], 136, 1),
                                         (L1[2 + e], imm1['exams'], 176, 1)] +
                                        [(L1[10 + i], 142 + 52 * i, 272, 1) for i in range(5)] +
                                        [(L1[15], 142 + 52 * (L % 5), 240, 1), (L1[9], imm1[9], 336, 1),
                                         (L1[16 + L], imm1['lesson'], 336, 1), box])))
    out.append(('top99', scr([(L2[0], imm2[0], 48, 2), (L2[1], imm2[1], 206, 1),
                              (L2[2], imm2[2], 254, 1), (L2[3], 352, 388, 5)])))
    out.append(('exams99', scr([(L2[0], imm2[0], 48, 2), (L2[4], imm2[4], 168, 1),
                                (L2[5], imm2[5], 208, 1), (L2[6], imm2[6], 240, 1),
                                (L2[7], imm2[7], 272, 1), (L2[3], 352, 388, 5)])))
    return out

def lesson_sheet(t, lists1, imm1):
    """All 30 lesson lines (Subj <name>) on the panel colour, 20 px apart."""
    rows = []
    for L in range(30):
        c = np.zeros((MH + 2, 512, 3), np.uint8); c[:] = (24, 40, 80)
        draw_list(c, t, lists1[9], imm1[9], 0, 1)
        draw_list(c, t, lists1[16 + L], imm1['lesson'], 0, 1)
        rows.append(c[:, 80:432])
    return np.concatenate(rows, 0)

def jp_state():
    from lz import decompress
    d1 = bytearray(decompress(open('private/disc/DAT2/SIKEN1.BIZ', 'rb').read())[0])
    t = Tim(d1, MENU_TIM)
    J1 = read_lists(open(f'{DAT1}/SIKEN.BIN', 'rb').read(), S_PTAB, S_N, 5)
    J2 = read_lists(open(f'{DAT1}/SIKEN2.BIN', 'rb').read(), S2_PTAB, S2_N, 5)
    i1 = {0: 144, 1: 106, 8: 106, 9: 114, 46: 196, 47: 186, 'exams': 216, 'lesson': 164}
    i2 = {0: 116, 1: 206, 2: 196, 4: 106, 5: 186, 6: 186, 7: 206}
    return t, J1, i1, J2, i2

# ---------------------------------------------------------------- sheets
# 8bpp 512x480 sheets at VRAM (512,0), CLUT (0,484). Inner cell boxes
# (x0, y0, x1, y1) between the grid lines, measured (sheet_geom.py).
# p1 = SIKEN1 0x55000, the record sheet (成績表を見る): orange grid, rows
#      177-180, 211-212 ..., columns 73-75, 171-172, 224-225 ...
# p2 = SIKEN2.BIZ 0x8800, the grade sheet after a lesson: upper table rows
#      80-83, 114-115, 145-148, columns 72-75, 198-199, 257-258, 317,
#      376-377, 435-438; lower table as p1 (teal grid).
P1_TITLE = (98, 102, 414, 139)        # 「基礎のおさらい」成績表, ink 114-389 x 107-137
P1_TASSEI = (200, 140, 300, 175)      # 達成率, ink 208-284; % at 329 is kept
P1_HDR = (76, 181, 170, 210)          # 試験/科目
P1_BUTTON, P1_ICON = (330, 408, 426, 431), (337, 412, 14, 16)
P1_INK = (40, 8, 168)                 # label blue
P2_HEAD = (150, 46, 360, 79)          # 成績発表, ink 163-342 x 46-79
P2_COLS = [(76, 197), (200, 256), (259, 316), (318, 375), (378, 434)]
P2_ROW = (84, 113)
P2_TASSEI = (190, 149, 300, 177)      # ink 195-287
P2_HDR = (76, 181, 170, 210)
P2_INK = (128, 24, 24)                # label dark red
P2_FILL, P2_HI, P2_SHADOW = (224, 56, 0), (224, 152, 96), (8, 8, 8)
PAPER = (176, 215, 221, 240)          # an empty lower-table cell (both)
F_BLACK = 'SourceSerif4-Black.otf'

def red_head(a):
    """成績発表 look: red-orange, lighter towards the top, dark drop shadow
    1-3 px down-right."""
    a = np.pad(a, ((0, 4), (0, 4)))
    sh = np.maximum.reduce([shift(a, 1, 1), shift(a, 2, 2), shift(a, 3, 3)])
    ys = np.nonzero((a > 0.5).any(1))[0]
    y0, y1 = ys.min(), ys.max()
    t = np.clip((np.arange(a.shape[0]) - y0) / max(1, (y1 - y0) * 0.45), 0, 1)
    hi = (1 - t)[:, None] * a
    return [(sh, P2_SHADOW), (a, P2_FILL), (hi * 0.8, P2_HI)]

def build_sheet(G, name, px, pal, tl, log, clut):
    """EN indices of sheet name ('p1', 'p2'), with build_examgfx's removal
    (G.erase: inpaint.py LaMa fill, real CLUT) and drawing (G = that
    module). clut: the sheet's CLUT."""
    rgb = pal[px]
    paper = G.paper_idx(px, PAPER)
    if name == 'p1':
        clear = [P1_TITLE, P1_TASSEI, P1_HDR]
        ink = G.sample_ink(px, rgb, P1_TITLE, paper)
        G.STYLE.update(fill=ink[0], edge=ink[1], shadow=ink[2])
        G.BUTTON['p1'], G.ICON['p1'] = P1_BUTTON, P1_ICON
        btn = G.button(px, rgb, tl, 'p1', log)
    else:
        clear = [P2_HEAD, P2_TASSEI, P2_HDR] + [(x0, P2_ROW[0], x1, P2_ROW[1]) for x0, x1 in P2_COLS]
        btn = None
    log.append(f'{name}: paper {len(paper)} indices')
    core = np.zeros(px.shape, bool); lim = np.zeros(px.shape, bool)
    for x0, y0, x1, y1 in clear:
        core |= G.paper_core(px, rgb, (x0, y0, x1, y1), paper, grid=None)   # rects inside the grid
        lim[y0:y1 + 1, x0:x1 + 1] = True
    px, m = G.erase(px, pal, clut, core, lim, grid=False, paper=True, log=log)
    bad = (m & (pal[px].min(-1) < 150)).sum()
    if bad: log.append(f'WARNING {name}: {bad} dark texels after the fill')
    if btn is not None:                  # as build_examgfx s10: label and icon on paper
        x0, y0, x1, y1 = P1_BUTTON
        rgb = pal[px]
        sub = rgb[y0:y1 + 1, x0:x1 + 1]
        m = np.zeros(px.shape, bool); m[y0:y1 + 1, x0:x1 + 1] = sub[..., 2] > sub[..., 0] + 20
        ix0, iy0, iw, ih = P1_ICON
        m[iy0:iy0 + ih, ix0:ix0 + iw + 1] = True
        m = ndimage.binary_dilation(m, np.ones((3, 3), bool))
        lim = np.zeros(px.shape, bool); lim[y0:y1 + 3, x0 - 40:x1 + 9] = True     # off the grid and edge lines
        px, _ = G.erase(px, pal, clut, m, lim, grid=False, paper=True, log=log)
    G.FILLS[name] = pal[px]
    canvas = pal[px].astype(float)
    cov = np.zeros(px.shape, np.float32)
    if name == 'p1':
        x0, y0, x1, y1 = P1_TITLE
        a, top, cap = G.fit(tl['「基礎のおさらい」成績表'], G.F_TITLE, G.CAP_TITLE, x1 - x0 - 8, track=0.4)
        cv, r = G.place(canvas, a, top, cap, (x0, y0, x1, y1 - 2), G.title_layers)
        cov = np.maximum(cov, cv); log.append(f'p1 title at {r} cap {cap}')
        G.STYLE['fill'] = P1_INK
        tas, hdr = P1_TASSEI, P1_HDR
        right = 284
    else:
        x0, y0, x1, y1 = P2_HEAD
        a, top, cap = G.fit(tl['成績発表'], F_BLACK, 22, x1 - x0, track=1.2)
        cv, r = G.place(canvas, a, top, cap, (x0, y0, x1, y1 - 3), red_head)
        cov = np.maximum(cov, cv); log.append(f'p2 head at {r} cap {cap}')
        G.STYLE['fill'] = P2_INK
        # Exam alone; the four narrow stat cells share one cap
        cells = [(tl[k], (c0, P2_ROW[0], c1, P2_ROW[1]), G.CAP_HEAD, 'c')
                 for k, (c0, c1) in zip(('試験', '行動力', '正確さ', '素早さ', '評価'), P2_COLS)]
        cov = np.maximum(cov, G.draw_cells(canvas, cells[:1], log))
        cov = np.maximum(cov, G.draw_cells(canvas, cells[1:], log, minsx=0.64))
        tas, hdr = P2_TASSEI, P2_HDR
        right = 287
    # 達成率: right-aligned where the JP ended; the % digits follow it
    a, top, cap = G.fit(tl['達成率'], G.F_LABEL, 16, right - 150)
    cv, r = G.place(canvas, a, top, cap, (tas[0], tas[1], right, tas[3]), G.plain, 'r')
    cov = np.maximum(cov, cv); log.append(f'{name} completion at {r} cap {cap}')
    cov = np.maximum(cov, G.draw_cells(canvas, [(tl['試験／科目'], hdr, G.CAP_HEAD, 'c')], log))
    if btn is not None:
        a, top, tx, capt, icon, ix, iy = btn
        cv, r = G.place(canvas, a, top, G.CAP_ROW, (tx, capt, tx + a.shape[1] - 1, capt + G.CAP_ROW),
                        G.plain, 'l', vbase=capt)
        cov = np.maximum(cov, cv)
    out = G.quantise(canvas, pal, cov, px)
    if btn is not None:
        im, ic = icon
        reg = out[iy:iy + im.shape[0], ix:ix + im.shape[1]]
        reg[im] = ic[im]
    return out

# ---------------------------------------------------------------- SIKENKK

def kk_lessons(G, titles, log):
    """SIKENKK entries 0-29: {entry: (subject number, title)}."""
    return {L: (str(L % 5 + 1), titles[L]) for L in range(30)}

KK_CAP = 11

def kk_lesson(G, t, num, name):
    """One SIKENKK lesson sprite (124x28, drawn for entry 30's CLUT: 0
    clear, 1 ink .. 15 cream, as G.kk_title). As in the JP: a large
    subject number, then the title at cap KK_CAP, squeezed to 0.72; titles
    that would need less go on two lines at cap 10. Returns a log text."""
    H, W = t.H, t.W
    al = np.zeros((H, W), np.float32)
    def put(a, x, y):
        h, w = a.shape
        reg = al[y:y + h, x:x + w]
        np.maximum(reg, a[:reg.shape[0], :reg.shape[1]], out=reg)
    a, top = G.text_alpha(num, G.F_KK, 17, 0.0)
    nx = 3
    put(a, nx, (H - 17) // 2 - top)
    x0 = nx + a.shape[1] + 4
    room = W - 2 - x0
    a, top, cap = G.fit(name, G.F_KK, KK_CAP, room, minsx=0.72, track=0.2)
    if cap >= KK_CAP:
        put(a, x0 + (room - a.shape[1]) // 2, (H - int(round(cap))) // 2 - top)
        how = f'1 line cap {cap}'
    else:
        words = name.split()
        ok = [k for k in range(1, len(words)) if words[k - 1] != '&' and words[k] != '&']
        best = min(ok, key=lambda k: abs(len(' '.join(words[:k])) - len(' '.join(words[k:]))))
        lines = [' '.join(words[:best]), ' '.join(words[best:])]
        cap = 10
        fits = [G.fit(s_, G.F_KK, cap, room, minsx=0.75, track=0.2) for s_ in lines]
        cap = min(f[2] for f in fits)
        fits = [G.fit(s_, G.F_KK, cap, room, minsx=0.75, track=0.2) for s_ in lines]
        lh = int(round(cap)) + 4
        y = (H - (2 * int(round(cap)) + 4)) // 2
        for (a, top, c) in fits:
            put(a, x0 + (room - a.shape[1]) // 2, y - top); y += lh
        how = f'2 lines {lines} cap {cap}'
    t.px[:] = np.where(al < 0.1, 0, np.clip(np.rint(1 + (1 - al) * 14), 1, 15)).astype(np.uint8)
    return how

def kk_remap(t, ref_rgb):
    """Map indices drawn for the reference CLUT onto t's own CLUT."""
    words = t.clut[:16]
    clear = [i for i in range(16) if words[i] == 0]
    own = t.rgb(0).astype(int)
    cand = [i for i in range(16) if i not in clear]
    lut = np.zeros(16, np.uint8)
    lut[0] = clear[0]
    for k in range(1, 16):
        d = ((own[cand] - ref_rgb[k].astype(int)) ** 2).sum(-1)
        lut[k] = cand[int(d.argmin())]
    t.px[:] = lut[t.px]
