# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English entry form (DAT2/MA_ENT.BIZ, MGENT.BIN records), from tl/ENTRY.txt.

Redraws, in the unpacked MA_ENT data (0x33800 B, length kept):
- TIM 0 (the 入学願書 form, 4bpp 448x464 at VRAM (512,0)): the printed
  labels, in a serif that matches the printed Mincho. Rows >= 457 overlap
  TIM 1's header and are never written.
- TIM 1 (glyph sheet, 4bpp 768x256 at VRAM (768,0)): the handwritten
  popup items (schools, sex, subjects, favourite words, face editor parts)
  in the hand of the original alnum keys, and the printed pop-style items
  (word prompt, confirm prompt and buttons, keyboard side buttons, face
  type label) in a rounded bold sans with the original outline, shadow
  and per-row fill colours.
The keyboard keys (alnum, kana) are left as they are.

Handwritten text uses the original key glyphs (tpage 14 rows 0-5) at their
natural size where the rect allows, else the squeezed field letters
(variant e of private/re/nameentry/fonts/orig/: proportional widths and a
stroke threshold). field_font() is the 10x20 field letter font that
tools/mgentfix.py letter_font() appends to MA_ENT.

Called from tools/mgentfix.py: redraw(data) inside patch_maent(), and
records() inside patch() for the MGENT record tables (new u/v/w). hook.py
writes both files. RE: private/re/entry/findings.org,
private/re/face/findings.org, private/re/nameentry/findings.org.

Usage: python3 tools/build_entry.py [--png]   (report; --png previews to
private/re/entry/preview/). Writes nothing to private/build.
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAENT = os.path.join(ROOT, 'private/disc/DAT2/MA_ENT.BIZ')
TL = os.path.join(ROOT, 'tl/ENTRY.txt')
MAENT_SIZE = 0x33800
T0_PIX, T0_STRIDE, T0_W, T0_ROWS = 0x40, 224, 448, 457   # rows < 457 only
T0_CLUT = 0x14
T1, T1_CLUT, T1_PIX, T1_STRIDE = 0x19000, 0x19014, 0x192e0, 384
F_SERIF = 'BIZUDMincho-Regular.ttf'
F_POP = 'MPLUSRounded1c-ExtraBold.ttf'


def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out


def orig_maent():
    import lz
    return lz.decompress(open(MAENT, 'rb').read())[0]


# ---------------------------------------------------------------- data
class Maent:
    """Unpacked MA_ENT: t0 (457x448 indices), sheet (256x768 indices),
    pal0 (16 RGB), pal1 (22 x 16 RGB)."""
    def __init__(self, data):
        assert len(data) >= MAENT_SIZE
        self.data = bytearray(data[:MAENT_SIZE])
        self.t0 = self._unpack(T0_PIX, T0_STRIDE, T0_ROWS)
        self.sheet = self._unpack(T1_PIX, T1_STRIDE, 256)
        c0 = np.array(struct.unpack_from('<16H', self.data, T0_CLUT))
        c1 = np.array(struct.unpack_from('<352H', self.data, T1_CLUT)).reshape(22, 16)
        self.raw1 = c1
        self.pal0, self.pal1 = rgb15(c0), rgb15(c1)

    def _unpack(self, off, stride, rows):
        raw = np.frombuffer(bytes(self.data[off:off + stride * rows]), np.uint8).reshape(rows, stride)
        p = np.zeros((rows, stride * 2), np.uint8)
        p[:, 0::2] = raw & 15; p[:, 1::2] = raw >> 4
        return p

    def _pack(self, p, off, stride):
        b = (p[:, 0::2] | p[:, 1::2] << 4).astype(np.uint8).tobytes()
        self.data[off:off + len(b)] = b

    def bytes(self):
        assert self.t0.shape == (T0_ROWS, T0_W) and self.sheet.shape == (256, 768)
        self._pack(self.t0, T0_PIX, T0_STRIDE)
        self._pack(self.sheet, T1_PIX, T1_STRIDE)
        return bytes(self.data)

    def cell(self, tp, u, v, w=20, h=20):
        x = (tp - 12) * 256 + u
        return self.sheet[v:v + h, x:x + w]


def rgb15(v):
    v = np.asarray(v)
    return np.stack([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], -1).astype(np.int32)


# ---------------------------------------------------------- hand letters
ALNUM = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
HAND = ALNUM + "-!?.'"


def cov(lv):
    """sheet level -> ink coverage: 2 = 1.0 .. 13 = 1/12, 0 = 0."""
    return np.where(lv > 0, (14 - lv.astype(np.float32)) / 12, 0)


def quant(a, thr=0.06):
    return np.where(a > thr, np.clip(np.rint(14 - 12 * a), 2, 13), 0).astype(np.uint8)


def area(n_in, n_out):
    """(n_out, n_in) area-resampling matrix."""
    m = np.zeros((n_out, n_in), np.float32); s = n_in / n_out
    for o in range(n_out):
        a, b = o * s, (o + 1) * s
        for i in range(int(a), min(n_in, int(np.ceil(b)))):
            m[o, i] = max(0, min(b, i + 1) - max(a, i))
    return m / s


def cols(g):
    xs = np.nonzero(g.any(0))[0]
    return g[:, xs.min():xs.max() + 1]


class Hand:
    """The original hand: key glyphs (natural, 20 rows, cropped to ink) and
    the squeezed variant-e letters, from the unmodified sheet."""
    NARROW = set("il1jtfr!.'")

    def __init__(self, m):
        src = {}
        for i, ch in enumerate(ALNUM):
            src[ch] = m.cell(14, 20 * (i % 12), 20 * (i // 12)).copy()
        src['!'] = m.cell(12, 160, 120).copy()
        bar = cols(m.cell(12, 180, 120))
        src['-'] = np.zeros((20, 20), np.uint8)
        src['-'][:, 2:10] = quant(cov(bar) @ area(bar.shape[1], 8).T)   # 8 px hyphen
        src['?'] = m.cell(12, 200, 120).copy()
        ex = src['!']
        dot = np.zeros((20, 20), np.uint8); dot[15:19, 8:12] = ex[15:19, 8:12]
        src['.'] = dot
        apo = np.zeros((20, 20), np.uint8); apo[0:6] = ex[0:6]           # top of the ! stroke
        src["'"] = apo
        self.src = src
        self.nat = {c: cols(g) for c, g in src.items()}
        self.sq = {c: cols(self._var_e(c)) for c in src}

    def _tw(self, ch):
        w = self.nat[ch].shape[1]
        if ch in self.NARROW: return min(w, 9)
        if ch == '-': return 6
        return min(w, int(round(9 - (19 - w) * 2 / 9)))

    def _var_e(self, ch):
        g = self.nat[ch]; w = g.shape[1]; tw = self._tw(ch)
        a = cov(g) @ area(w, tw).T
        c = np.zeros((20, 10), np.float32)
        x0 = (10 - tw) // 2
        c[:, x0:x0 + tw] = a
        return quant(np.clip((c - 0.22) / 0.55, 0, 1), 0.08)

    def field_font(self):
        """96 x 100 B: ASCII 0x20-0x7f, 10x20, 4bpp low nibble first."""
        out = bytearray()
        for c in range(0x20, 0x80):
            ch = chr(c)
            lv = self._var_e(ch) if ch in self.src else np.zeros((20, 10), np.uint8)
            f = lv.ravel()
            out += bytes(int(f[i]) | int(f[i + 1]) << 4 for i in range(0, 200, 2))
        assert len(out) == 9600
        return bytes(out)

    def text(self, s, mode='nat', gap=2, space=None, maxov=3):
        """Level array (20 x w) of s, kerned by row profiles: at least gap
        clear columns between ink in the same or a neighbouring row."""
        gl = self.nat if mode == 'nat' else self.sq
        space = space if space is not None else (6 if mode == 'nat' else 4)
        canvas = np.zeros((20, 400), np.uint8)
        right = np.full(20, -99)            # rightmost ink column per row
        x = 0; pend = 0; end = 0
        for ch in s:
            if ch == ' ':
                pend += space; continue
            g = gl[ch]; w = g.shape[1]
            ink = g > 0
            left = np.where(ink.any(1), ink.argmax(1), 99)
            # dilate the previous profile by one row
            rp = np.maximum(right, np.maximum(np.r_[right[1:], -99], np.r_[-99, right[:-1]]))
            need = max([rp[r] + gap + 1 - left[r] for r in range(20) if left[r] < 99] + [-99])
            x = max(need + pend, end + pend - maxov) if end else 0
            if pend and end: x = max(x, end + pend)
            pend = 0
            sub = canvas[:, x:x + w]
            canvas[:, x:x + w] = np.where(g > 0, np.where(sub > 0, np.minimum(sub, g), g), sub)
            rr = np.where(ink.any(1), w - 1 - ink[:, ::-1].argmax(1) + x, -99)
            right = np.maximum(right, rr)
            end = max(end, x + w)
        return canvas[:, :end]


# ------------------------------------------------------- printed text
def lum(c): return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]


def alpha(text, font, size, ss=4, track=0, w=None, h=None, dy=0):
    """Coverage (h x w) of text, centred; (a, ink width). track in px."""
    f = fontpath.truetype(font, size * ss)
    xs, x = [], 0
    for ch in text:
        xs.append(x)
        x += f.getlength(ch) + track * ss
    tw = x - track * ss
    asc, desc = f.getmetrics()
    W = w * ss if w else int(tw) + 8 * ss
    H = h * ss if h else (asc + desc)
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    x0 = (W - tw) / 2
    # vertical: centre the cap height
    cap = f.getbbox('H')
    y0 = (H - (cap[3] - cap[1])) / 2 - cap[1] + dy * ss
    for ch, x in zip(text, xs):
        d.text((x0 + x, y0), ch, font=f, fill=255)
    im = im.resize((W // ss, H // ss), Image.BOX)
    return np.asarray(im, np.float32) / 255, tw / ss


def ink_width(a, thr=0.18):
    c = np.nonzero((a > thr).any(0))[0]
    return int(c.max() - c.min() + 1) if len(c) else 0


def nearest(pal, idx, c):
    """palette index among idx closest to RGB c."""
    best = min(idx, key=lambda i: sum((int(pal[i][k]) - c[k]) ** 2 for k in range(3)))
    return best


class Pop:
    """Pop-style printed text in the palette roles of an original item:
    outline/shadow in its darkest colour, fill with its per-row colours."""
    def __init__(self, orig, pal, kind='shadow'):
        self.pal = pal
        used = sorted(set(orig[orig > 0].ravel().tolist()))
        self.idx = used
        L = {i: lum(pal[i]) for i in used}
        self.dark = min(used, key=L.get)
        rows = []
        for r in range(orig.shape[0]):
            v = orig[r][orig[r] > 0]
            v = [i for i in v if L[i] >= np.percentile([L[j] for j in orig[orig > 0]], 60)]
            rows.append(np.mean([pal[i] for i in v], 0) if v else None)
        ok = [r for r in range(len(rows)) if rows[r] is not None]
        self.rows = [rows[r] if rows[r] is not None else rows[min(ok, key=lambda k: abs(k - r))]
                     for r in range(len(rows))]
        self.kind = kind

    def render(self, a, fill_thr=0.5):
        h, w = a.shape
        core = a > 0.18
        if self.kind == 'full':
            out_m = ndimage.binary_dilation(core, np.ones((3, 3), bool))
        else:  # drop shadow right/below plus a thin left/top rim
            sh = np.zeros_like(core); sh[1:, 1:] = core[:-1, :-1]
            out_m = core | sh
        res = np.zeros((h, w), np.uint8)
        dk = np.array(self.pal[self.dark], np.float32)
        for y in range(h):
            fc = np.array(self.rows[min(y, len(self.rows) - 1)], np.float32)
            for x in range(w):
                if not out_m[y, x]: continue
                t = min(1.0, a[y, x] / fill_thr)
                res[y, x] = nearest(self.pal, self.idx, dk + (fc - dk) * t)
        return res


# ------------------------------------------------------------ layout
F_FORM = 'SourceSerif4-Regular.otf'
F_FORM_SMALL = 'SourceSerif4Caption-Regular.otf'   # <= 12 px
F_FACE = 'MPLUS1p-Medium.ttf'
POP_CLUT = 12                     # popup items (white on blue); field CLUT 1

# Schools: (tl id, record file offset of the name rect, v, clear u end).
SCHOOLS = [('sch_taiyo', 0x15fe, 160, 180), ('sch_gorin', 0x160a, 180, 180),
           ('sch_pacific', 0x1616, 200, 256), ('sch_gedo', 0x1622, 140, 180),
           ('sch_justice', 0x162e, 220, 200)]
SCH_U = 32
SEX = [('sex_m', 180), ('sex_f', 200)]          # tp12 v140, 20x20, fixed by code
SUBJ_REC, WORD_REC = 0x1674, 0x1724
GRID_W, GRID_MAX = 64, 66         # popup cursor brackets span x-4 .. x+68
SUBJ_ROWS = [(13, 140), (13, 160)]
WORD_ROWS = [(13, 180), (13, 200), (13, 220), (14, 120)]
FACE_REC = 0x118                  # 5 x 7 s16: x, y, tp, u, v, w, clut/16
FACE = [('face_eyes', 0), ('face_nose', 1), ('face_mouth', 2)]
# The JP labels (one 20 px glyph at x 74) are centred at x ~84. The EN words
# are centred on one column at FACE_CX and must end by FACE_R (4 px before
# the arrow tip), so they use the squeezed hand (variant e).
FACE_CX, FACE_L, FACE_R = 82, 58, 100
FACE_JP_X = 74                    # JP record x of every face label
BRACKET = (72, 128)               # 決定-row cursor: 24x26 SPRTs at x 72 and 104 (code)
# Free 20-row runs for them: after the Pacific name, the old 鼻口 cells
# (tp12 u180-219 v180, face only) plus the empty run to u255, tp13 u200-255 (u180-199 is a katakana cell).
FACE_FREE = [(12, 200), (12, 180), (13, 120)]
# Printed pop items: id -> (tp, u, v, w, h, clut, kind, size, pieces)
POP = {
    'words_prompt': ((14, 0, 180, 240, 20), 8, 'shadow', 19, [(14, 0, 200, 40, 20)]),
    'confirm_q': ((14, 0, 140, 240, 20), 9, 'full', 19, [(14, 0, 160, 60, 20)]),
    'confirm_ok': ((14, 60, 160, 80, 20), 10, 'full', 18, []),
    'confirm_redo': ((14, 140, 160, 80, 20), 10, 'full', 18, []),
    'btn_0': ((14, 48, 200, 48, 24), 18, 'shadow', 17, []),
    'btn_1': ((14, 96, 200, 48, 24), 18, 'shadow', 17, []),
    'btn_2': ((14, 144, 200, 48, 24), 18, 'shadow', 17, []),
    'btn_3': ((14, 0, 224, 48, 24), 19, 'shadow', 17, []),
    'btn_4': ((14, 48, 224, 48, 24), 19, 'shadow', 17, []),
    'btn_5': ((14, 96, 224, 48, 24), 19, 'shadow', 17, []),
    'btn_6': ((14, 144, 224, 48, 24), 17, 'shadow', 17, []),
}
FACETYPE = (14, 40, 100, 160, 20)  # face_type, CLUT 8; arrows from u 200

# TIM 0 labels: id -> (erase rects, free rect x0, y0, x1, y1, align, size,
# font, baselines). Ink must stay inside the free rect (the paper between
# the printed rules). Text is rendered hinted at 1x on an integer baseline;
# baselines None centres cap top .. baseline in the free rows.
F_FORM_SB = 'SourceSerif4Caption-Semibold.otf'   # 11-13 px labels
LBL = 16
FORM = {
    'form_shimei': ([(19, 71, 117, 111)], (19, 69, 117, 113), 'c', LBL, F_FORM, None),
    'form_yobina': ([(19, 115, 117, 150)], (19, 115, 117, 152), 'c', LBL, F_FORM, None),
    'form_shibou': ([(19, 157, 117, 191)], (19, 155, 117, 192), 'c', LBL, F_FORM, None),
    'form_seibetsu': ([(19, 195, 117, 231)], (19, 194, 117, 233), 'c', LBL, F_FORM, None),
    'form_seigappi': ([(19, 235, 117, 270)], (19, 234, 117, 271), 'c', LBL, F_FORM, None),
    'form_tokui': ([(19, 277, 117, 311)], (19, 276, 117, 311), 'c', LBL, F_FORM, None),
    'form_suki': ([(19, 315, 117, 351)], (21, 313, 117, 352), 'c', LBL, F_FORM, (330, 346)),
    'form_sei': ([(124, 69, 142, 79)], (125, 69, 280, 79), 'l', 11, F_FORM_SB, None),
    'form_mei': ([(285, 69, 302, 79)], (286, 69, 430, 79), 'l', 11, F_FORM_SB, None),
    'form_examdate': ([(316, 50, 420, 66)], (300, 50, 417, 66), 'r', 12, F_FORM_SMALL, None),
    'form_tsuki': ([(178, 243, 195, 263)], (178, 244, 195, 262), 'c', 16, F_FORM, None),
    'form_hi': ([(237, 244, 253, 263)], None, 'c', 16, F_FORM, None),
    'form_kaijou': ([(329, 234, 387, 248)], (285, 234, 428, 249), 'c', 13, F_FORM_SB, None),
    'form_hongaku': ([(287, 254, 428, 268)], (285, 251, 428, 271), 'c', 13, F_FORM_SB, None),
    'form_note': ([(30, 354, 184, 370)], (32, 354, 300, 370), 'l', 12, F_FORM_SMALL, None),
}
TITLE = (156, 22, 296, 48)        # 入学願書 + its underline (rows 46-47)
TITLE_CX, TITLE_UL = 225, 46


def fit_hand(h, s, limit, gaps=(2, 1)):
    """-> (levels, mode): natural key glyphs if they fit, else squeezed."""
    for g in gaps:
        a = h.text(s, 'nat', g)
        if a.shape[1] <= limit: return a, f'natural gap {g}'
    a = h.text(s, 'sq', 1)
    return a, 'squeezed' if a.shape[1] <= limit else f'OVERFLOW {a.shape[1]} > {limit}'


def blit(m, tp, u, v, a):
    h, w = a.shape
    assert u + w <= 256 and v + h <= 256, (tp, u, v, w)
    m.cell(tp, u, v, w, h)[:] = a


def pack(items, rows):
    """items [(key, w)] -> {key: (tp, u, v)}: first fit, in order."""
    used = {r: 0 for r in rows}; out = {}
    for k, w in items:
        for r in rows:
            if used[r] + w <= 256:
                out[k] = (r[0], used[r], r[1]); used[r] += w; break
        else:
            raise ValueError(f'no sheet room for {k} ({w} px)')
    return out


class Layout:
    """Everything build_entry changes, computed from the original MA_ENT
    and tl/ENTRY.txt: new sheet/TIM 0 pixels, MGENT records, a report."""
    def __init__(self, data=None, tl=None):
        self.m = Maent(data if data is not None else orig_maent())
        self.orig = Maent(self.m.bytes())
        self.tl = tl or load_tl()
        self.hand = Hand(self.orig)
        self.rec = []            # (file offset, old bytes, new bytes)
        self.report = []         # (id, EN, how, width, limit)
        self.items = {}          # id -> (tp, u, v, w, h, clut) for previews
        self.sheet_items(); self.pop_items(); self.face_type(); self.form()

    def note(self, k, how, w, lim):
        self.report.append((k, self.tl.get(k, ''), how, w, lim))

    def rec_bytes(self, off, old, new, fmt='6B'):
        self.rec.append((off, struct.pack('<' + fmt, *old), struct.pack('<' + fmt, *new)))

    # -------- handwritten items
    def sheet_items(self):
        m, h, tl = self.m, self.hand, self.tl
        mg = mgent_orig()
        ends = {}
        for k, off, v, uend in SCHOOLS:
            m.cell(12, SCH_U, v, uend - SCH_U)[:] = 0
            lim = (uend if k != 'sch_pacific' else 248) - SCH_U
            a, how = fit_hand(h, tl[k], lim)
            blit(m, 12, SCH_U, v, a)
            old = struct.unpack_from('<6B', mg, off)
            assert old[2:4] == (SCH_U, v) and old[0] == 12, (k, old)
            self.rec_bytes(off, old, old[:4] + (a.shape[1],) + old[5:])
            self.note(k, how, a.shape[1], lim)
            self.items[k] = (12, SCH_U, v, a.shape[1], 20, POP_CLUT)
            ends[v] = (SCH_U + a.shape[1] + 4, uend if uend < 256 else 256)
        for k, u in SEX:
            a, how = fit_hand(h, tl[k], 20)
            c = m.cell(12, u, 140); c[:] = 0
            x = (20 - a.shape[1]) // 2; c[:, x:x + a.shape[1]] = a
            self.note(k, how, a.shape[1], 20)
            self.items[k] = (12, u, 140, 20, 20, POP_CLUT)
        for tp, v in SUBJ_ROWS + WORD_ROWS:
            m.cell(tp, 0, v, 256)[:] = 0
        for keys, rows, rec in (([f'subj_{i}' for i in range(8)], SUBJ_ROWS, SUBJ_REC),
                                ([f'word_{i:02d}' for i in range(16)], WORD_ROWS, WORD_REC)):
            arts = {}
            for k in keys:
                a, how = fit_hand(h, tl[k], GRID_MAX)
                pad = max(0, (GRID_W - a.shape[1]) // 2)
                arts[k] = np.hstack([np.zeros((20, pad), np.uint8), a])
                self.note(k, how, a.shape[1], GRID_MAX)
            pos = pack([(k, arts[k].shape[1]) for k in keys], rows)
            for i, k in enumerate(keys):
                tp, u, v = pos[k]; a = arts[k]
                blit(m, tp, u, v, a)
                off = rec + 6 * i
                old = struct.unpack_from('<6B', mg, off)
                assert old[5] == 20 and old[4] == 60, (k, old)
                self.rec_bytes(off, old, (tp, old[1], u, v, a.shape[1], 20))
                self.items[k] = (tp, u, v, a.shape[1], 20, POP_CLUT)
        # face editor part labels, in the rows freed after the school names
        m.cell(12, 180, 180, 40)[:] = 0              # old 鼻口
        free = {(12, 200): list(ends[200]), (12, 180): [180, 256], (13, 120): [200, 256]}
        assert not m.cell(12, 220, 180, 36).any() and not m.cell(13, 200, 120, 56).any()
        for k, i in FACE:
            for g in (1, 0):
                a = h.text(tl[k], 'sq', g)
                fx = FACE_CX - a.shape[1] // 2
                if fx >= FACE_L and fx + a.shape[1] <= FACE_R: break
            how = f'squeezed gap {g}' if fx + a.shape[1] <= FACE_R else 'OVERFLOW'
            for slot in FACE_FREE:
                lo, hi = free[slot]
                if hi - lo >= a.shape[1]:
                    tp, v = slot; u = lo; free[slot][0] = lo + a.shape[1] + 4; break
            else:
                raise ValueError(k)
            blit(m, tp, u, v, a)
            off = FACE_REC + 14 * i
            old = struct.unpack_from('<7h', mg, off)
            assert old[0] == FACE_JP_X, (k, old)
            self.rec_bytes(off, old, (fx, old[1], tp, u, v, a.shape[1], old[6]), '7h')
            self.note(k, f'{how} x {fx}-{fx + a.shape[1]}', a.shape[1], FACE_R - FACE_L)
            self.items[k] = (tp, u, v, a.shape[1], 20, old[6])

    # -------- printed pop items
    def pop_items(self):
        for k, ((tp, u, v, w, hh), clut, kind, size, extra) in POP.items():
            orig = self.orig.cell(tp, u, v, w, hh)
            if extra:
                o2 = [self.orig.cell(*e) for e in extra]
                orig = np.hstack([orig] + o2)
            st = Pop(orig, self.m.pal1[clut], kind)
            W = orig.shape[1]
            H = 20
            s = self.tl[k]
            sz, track = size, 0
            pad = 2 if kind == 'full' else 1
            while True:
                iw = ink_width(alpha(s, F_POP, sz, w=W + 40, h=H, track=track)[0]) + pad
                if iw <= W - 2 or sz <= 9: break
                if track > -1: track -= 0.5
                else: sz -= 1
            a, tw = alpha(s, F_POP, sz, w=W, h=H, track=track)
            art = np.zeros((orig.shape[0], W), np.uint8)
            art[:H] = st.render(a)
            x = 0
            for (tp2, u2, v2, w2, h2) in [(tp, u, v, w, hh)] + extra:
                self.m.cell(tp2, u2, v2, w2, h2)[:] = art[:h2, x:x + w2]; x += w2
            how = f'printed {os.path.basename(F_POP)[:-4]} {sz}px track {track}'
            self.note(k, how if iw <= W - 2 else 'OVERFLOW ' + how, iw, W - 2)
            self.items[k] = (tp, u, v, w, hh, clut)

    def face_type(self):
        tp, u, v, w, hh = FACETYPE
        orig = self.orig.cell(tp, u, v, w, hh)
        st = Pop(orig, self.m.pal1[8], 'shadow')
        s = self.tl['face_type']
        sz = 16
        while True:
            a, tw = alpha(s, F_FACE, sz, w=w, h=hh, track=0.5)
            if tw <= w - 2 or sz <= 9: break
            sz -= 1
        # left-align: shift the centred art to column 1
        cols_ = np.nonzero((a > 0.18).any(0))[0]
        a = np.roll(a, 1 - cols_.min(), axis=1)
        self.m.cell(tp, u, v, w, hh)[:] = st.render(a)
        self.note('face_type', f'printed {os.path.basename(F_FACE)[:-4]} {sz}px', round(tw), w - 2)
        mg = mgent_orig()
        off = FACE_REC + 14 * 4
        old = struct.unpack_from('<7h', mg, off)
        # left edge of the ink where the JP ink began (JP record x 74)
        jx = int(np.nonzero(orig.any(0))[0].min())
        assert old[0] == FACE_JP_X, old
        self.rec_bytes(off, old, (FACE_JP_X + jx - 1,) + old[1:], '7h')
        off = FACE_REC + 14 * 3            # 決定 = keyboard OK (btn_6)
        old = struct.unpack_from('<7h', mg, off)
        assert old[0] == FACE_JP_X, old
        ok = self.m.cell(14, 144, 224, 48, 20)
        xs = np.nonzero(ok.any(0))[0]
        # centre the OK ink in the cursor brackets (決定 filled them)
        x = int(round((BRACKET[0] + BRACKET[1] - 1) / 2 - (xs.min() + xs.max()) / 2))
        self.rec_bytes(off, old, (x,) + old[1:], '7h')
        self.note('face_ok', f'x {x}, ink {x + xs.min()}-{x + xs.max()}', 0, 0)
        self.items['face_type'] = (tp, u, v, w, hh, 8)

    # -------- TIM 0 form
    def ink(self, a):
        pal = self.m.pal0
        paper, ink = np.array(pal[3], float), np.array(pal[13], float)
        out = np.full(a.shape, 3, np.uint8)
        for (y, x), val in np.ndenumerate(a):
            if val > 0.04:
                out[y, x] = nearest(pal, range(15), paper + (ink - paper) * min(1.0, val * 1.15))
        return out

    def text0(self, s, free, align, size, font, base=None, track=0):
        """Draw s on TIM 0 inside free (x0, y0, x1, y1): hinted 1x
        rendering, baseline on an integer row. -> ink (x0, y0, x1, y1)."""
        f = fontpath.truetype(font, size)
        W = int(sum(f.getlength(c) + track for c in s)) + 16
        asc, desc = f.getmetrics()
        H = asc + desc + 8
        im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
        x, b = 4, asc + 4
        for c in s:
            d.text((round(x), b), c, font=f, fill=255, anchor='ls'); x += f.getlength(c) + track
        a = np.asarray(im, np.float32) / 255
        ink = a > 0.04
        ys, xs = np.nonzero(ink.any(1))[0], np.nonzero(ink.any(0))[0]
        a = a[:, xs.min():xs.max() + 1]; w = a.shape[1]
        x0, y0, x1, y1 = free
        if base is None:
            cap = -f.getbbox('H', anchor='ls')[1]
            base = int(round(y0 + (y1 - y0 - cap) / 2 + cap))
        x = {'c': x0 + (x1 - x0 - w) // 2, 'l': x0, 'r': x1 - w}[align]
        top = base - b
        reg = self.m.t0
        sub = self.ink(a)
        for (yy, xx), v in np.ndenumerate(a):
            if v > 0.04:
                Y, X = top + yy, x + xx
                assert x0 <= X < x1 and y0 <= Y < y1, (s, X, Y, free)
                reg[Y, X] = sub[yy, xx]
        return x, top + ys.min(), x + w, top + ys.max() + 1

    def form(self):
        t, tl = self.m.t0, self.tl
        for k, (erase, free, align, size, font, bases) in FORM.items():
            for x0, y0, x1, y1 in erase:
                t[y0:y1, x0:x1] = 3
            if free is None or not tl.get(k):
                self.note(k, 'erased', 0, 0); continue
            s = tl[k]
            lim = free[2] - free[0] - 4
            if bases:
                l1, l2 = s.split(' ', 1)
                b1 = self.text0(l1, free, align, size, font, bases[0])
                b2 = self.text0(l2, free, align, size, font, bases[1])
                w = max(b1[2] - b1[0], b2[2] - b2[0]); how = ' 2 lines'
            else:
                bb = self.text0(s, free, align, size, font)
                w = bb[2] - bb[0]; how = ''
            self.note(k, f'printed {os.path.basename(font)[:-4]} {size}px' + how, w, lim)
        # title: letter-spaced caps, underline re-centred to the text
        x0, y0, x1, y1 = TITLE
        ul = t[TITLE_UL:TITLE_UL + 2, 200].copy()
        t[y0:y1, x0:x1] = 3
        s = tl['form_title']
        bb = self.text0(s, (TITLE_CX - 110, 23, TITLE_CX + 110, 46), 'c', 20, F_FORM, track=2.5)
        half = (bb[2] - bb[0]) // 2 + 8
        t[TITLE_UL:TITLE_UL + 2, TITLE_CX - half:TITLE_CX + half] = ul[:, None]
        self.note('form_title', 'printed serif 20px, tracked', bb[2] - bb[0], 220)


_MG = None
def mgent_orig():
    global _MG
    if _MG is None:
        _MG = open(os.path.join(ROOT, 'private/disc/DAT1/MGENT.BIN'), 'rb').read()
    return _MG


# ------------------------------------------------------------ API
def redraw(data):
    """Unpacked MA_ENT (0x33800 B) -> the same length with the EN form."""
    assert len(data) == MAENT_SIZE, hex(len(data))
    out = Layout(data).m.bytes()
    assert len(out) == MAENT_SIZE
    return out


def records():
    """[(MGENT file offset, old bytes, new bytes)] for the EN layout."""
    return Layout().rec


def field_font(data=None):
    return Hand(Maent(data if data is not None else orig_maent())).field_font()


def main():
    L = Layout()
    bad = 0
    for k, en, how, w, lim in L.report:
        flag = '!!' if 'OVERFLOW' in how else '  '
        bad += flag == '!!'
        print(f'{flag} {k:14s} {en[:24]:24s} {w:4d}/{lim:<4d} {how}')
    print(f'{len(L.rec)} MGENT record edits; {bad} overflow(s)')
    if '--png' in sys.argv:
        import importlib.util
        p = os.path.join(ROOT, 'private/re/entry/preview/previews.py')
        spec = importlib.util.spec_from_file_location('previews', p)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.run(L)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
