# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English diary title menu, prompts, memory card messages and Continue
list (MGTIT overlay, MA_TIT glyph sheet).

Reads tl/MGTIT.txt (ID: EN). Writes private/build/MGTIT.BIZ (DAT1, LZ;
the loose DAT1/MGTIT.BIN is unused) and private/build/MA_TIT.BIZ (DAT2,
LZ). --png writes previews to private/re/title/preview/.
RE: private/re/title/findings.org.

MA_TIT TIM0 is a 4bpp 768x256 sheet at VRAM (768,0): three tpages of
12x12 cells of 20x20. MGTIT strings are u16 count + u16 cell codes (bits
0-3 col, 4-7 row, 8-11 tpage-12; 0xfff blank) and are drawn by
0x800a50b4 from an 18-entry table at file 0x2a0 (x, y, ptr, adv, pad).
- EN letters are drawn two per cell (10x20 each) into cells that only
  these strings use (FREE). Name cells (katakana, ー) and tpage 960 are
  not touched; the logo rectangles in tpage 896 are kept clear.
- The string area 0x10-0x29f (incl. 8 unused strings) is repacked and the
  18 pointers rewritten. Fixed-x strings are re-centred on the JP centre;
  Yes/No are centred in their cursor bars.
- Continue list rows (findings: Continue list). School names are the
  Backup list's labels (tl/COCK.txt, cock_hud.school_piece), redrawn as
  five 80x20 cells (records 0x660 get new u, v, w) with CLUT 7 replaced
  by the Backup palette colours, drawn at x 232 (12 px after the name
  column 120-220; names end at x 200 at most). Dates use the letter-font digits at a 10 px pitch
  as the Backup list: the date routine 0x800a5e0c gets its own uv table
  (narrow digits in t768 row 11 cols 7-11) and step 10; the 月 cell holds
  "/" in its left 10 px, the 日 cell is blank. Month right-aligned, "/"
  fixed, day left-aligned (day call 0x800a6540 goes through a stub after
  the strings that moves x left by 10 when the day is below 10), the 50
  px box centred on the JP date column (centre x 368): "4/8", "12/24".
- Full nickname (topic titlenick): the row shows the save's +0x96 pair
  cells as up to 8 narrow letters (squeezed sheet letters, 10 px pitch) via
  NAME_INIT/NAME_DSTUB; other saves keep the 4 wide +0x86 glyphs.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz, digraph, build_logo

TL = 'tl/MGTIT.txt'
SRC_OVL = 'private/disc/DAT1/MGTIT.BIZ'
SRC_TIT = 'private/disc/DAT2/MA_TIT.BIZ'
OUT = 'private/build'
PREV = 'private/re/title/preview'
SLOT_OVL = 14 * 2048                  # disc allocation (26890 B)
SLOT_TIT = 56 * 2048                  # disc allocation (113696 B)

BASE = 0x800a2000
TAB, NENT = 0x2a0, 18                 # entries: x, y, ptr, adv, pad
AREA = (0x10, 0x2a0)                  # string area to repack
TIM_PIX = 0x174 + 12                  # TIM0 pixel data, 384 B per row
TIM_CLUT = 20                         # 11 CLUTs of 16 colours
MAX_CELLS = {4: 18}                   # entry 4: packets 0-17, entry 5 at 18
RIGHT_EDGE = {2: 512, 0: 408, 1: 408, 17: 408}   # small logo at x 410, y 30-94
FIXED_X = (0, 1, 17)                  # left-aligned like the JP
CENTRED = range(6, 17)                # drawn with the centre flag
BARS = ((178, 64), (258, 64))         # Yes/No cursor bars (x, w)

# Cells only these strings use. t832 row 9 cols 2/3 (月日) are the list's
# date markers; t896 row 6 and row 7 col 4 lie inside the logo rectangles.
# t768 row 11 cols 7-11 hold the narrow date digits (NDIG).
NDIG = [(0, 11, c) for c in range(7, 12)]   # digits 2k, 2k+1 in cell k
# t768 rows 8-11 cols 0-7 hold the narrow name letters (namefull): letter k
# (0-9, A-Z, a-z) in cell m = k >> 1 (row 8 + m >> 3, col m & 7), half k & 1.
NAME_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
NNAME = [(0, 8 + (m >> 3), m & 7) for m in range(31)]
FREE = ([(1, r, c) for r in (0, 1) for c in range(12)] +
        [(1, 9, c) for c in range(12) if c not in (2, 3)] +
        [(1, 10, c) for c in range(3)] +
        [(0, r, c) for r in range(7, 12) for c in range(12)
         if (r, c) != (7, 1) and (0, r, c) not in NDIG + NNAME] +
        [(2, 7, c) for c in range(5, 12)] +
        [(2, r, c) for r in (8, 9) for c in range(5, 12)] +
        [(2, r, c) for r in (10, 11) for c in range(12)])
MONTH, DAY = (1, 9, 2), (1, 9, 3)     # 月, 日 (u40/u60 v180 t832)
SCHOOL_REC = 0x660                    # 5 x 16 B: ?, tpage, ?, u, v, w, h, ?
SCHOOLS = ('taiyo', 'gorin', 'pacific', 'gedo', 'justice')   # record order
SCHOOL_JP = ('太陽学園', '五輪高校', 'パシフィックHS', '外道高校',
             'ジャスティス学園')         # tl/COCK.txt keys (Backup list)
# School cells, all 80x20 in t832 rows 10-11 (row 10 u0-59 are FREE).
SCHOOL_UV = ((64, 200), (144, 200), (0, 220), (80, 220), (160, 220))
SCHOOL_W = 80
CLUT_SCHOOL = 7                       # (112,481), used only by the schools
CLUT_DATE = 0                         # date digits and "/" (yellow ramp)

# Continue list row (screen x). Name: 5 cells from x 120 (ends <= 220).
# School cell after a 12 px gap. Date: month right-aligned, "/" fixed,
# day left-aligned (as the Backup list), the 50 px box centred on the JP
# date column (308-428, centre 368).
NAME_X = 120
SCHOOL_X = 232
DATE_X = 368 - 25
PITCH = 10
DAY_CALL = 0x800a6540                 # jal 0x800a5e0c (day digits)
DIGITS = 0x800a5e0c
# (addr, old imm, new imm) of addiu immediates in the list code
IMMS = [(0x800a6004, 0x14, PITCH),                  # digit step in 0x800a5e0c
        (0x800a6264, 0x64, SCHOOL_X - NAME_X),      # school x = name x + ...
        (0x800a6390, 0x58, DATE_X - SCHOOL_X),      # month x
        (0x800a6494, 0x28, 2 * PITCH),              # "/" (月 sprite) x
        (0x800a64f0, 0x14, PITCH)]                  # day x = "/" x + 10
DTAB_REFS = ((0x800a5ea8, 0x2638, 0), (0x800a5f94, 0x263b, 3))  # digit uv table

# ------------------------------------------------------------ letters
def letters(text, w=20, dx=0):
    """The single letter source: text in 10 px columns from x = dx on a
    20-row canvas w px wide, as CLUT 6 indices (0 clear, 3 shadow, 4-14
    ramp to white). Pair cells are letters(pair). Uses the message font
    (tools/digraph.py) for now; a different title font goes here."""
    fpath, face, size, yoff = digraph.FONT
    im = Image.new('L', (w, 20))
    dr = ImageDraw.Draw(im)
    for k, c in enumerate(text):
        if c != ' ': dr.text((dx + k * 10, yoff), c, font=digraph.font(), fill=255)
    a = np.asarray(im, np.float32) / 255
    main = np.where(a > 0.12, np.rint(3 + 11 * a), 0)
    sh = np.zeros_like(main); sh[1:, 1:] = a[:-1, :-1] > 0.4
    return np.where(main > 0, main, np.where(sh > 0, 3, 0)).astype(np.uint8)

# ------------------------------------------------------------ helpers
def load_tl(path=TL):
    tl = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.startswith('#'): continue
        k, _, v = line.partition(':')
        tl[k.strip()] = v[1:] if v.startswith(' ') else v
    return tl

def sheet_of(tim):
    px = np.frombuffer(tim, np.uint8, 384 * 256, TIM_PIX).reshape(256, 384)
    a = np.zeros((256, 768), np.uint8); a[:, 0::2] = px & 15; a[:, 1::2] = px >> 4
    return a

def put_sheet(tim, a):
    tim[TIM_PIX:TIM_PIX + 384 * 256] = (a[:, 0::2] | a[:, 1::2] << 4).astype(np.uint8).tobytes()

def cell_xy(t, r, c): return t * 256 + c * 20, r * 20
def code_of(t, r, c): return t << 8 | r << 4 | c
def cell_of(code): return code >> 8, code >> 4 & 15, code & 15

def strings(ovl):
    out = []
    for i in range(NENT):
        x, y, p, adv = struct.unpack_from('<HHIH', ovl, TAB + 12 * i)
        n = struct.unpack_from('<H', ovl, p - BASE)[0]
        out.append((x, y, adv, list(struct.unpack_from('<%dH' % n, ovl, p - BASE + 2))))
    return out

def clut_rgb(tim, k):
    v = np.frombuffer(tim, np.uint16, 16, TIM_CLUT + 32 * k).astype(np.int32)
    return np.stack([(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8], 1).astype(np.uint8)

# ------------------------------------------------------------ layout
def layout(tl, jp):
    """Per entry: (x, cells); a cell is a 20x20 index array or None."""
    out = {}
    for i in range(NENT):
        text = tl['s%d' % i]
        x, y, adv, codes = jp[i]
        if i == 3:
            words = text.split()
            assert len(words) == 2, 's3 needs two words (Yes, No)'
            xs = [bx + (bw - 10 * len(wd)) // 2 for (bx, bw), wd in zip(BARS, words)]
            x0 = xs[0]
            w = (xs[1] - x0 + 10 * len(words[1]) + 19) // 20 * 20
            strip = np.maximum(letters(words[0], w, 0), letters(words[1], w, xs[1] - x0))
            cells = [strip[:, k:k + 20] for k in range(0, w, 20)]
            out[i] = (x0, [c if c.any() else None for c in cells])
            continue
        if len(text) % 2: text += ' '
        cells = [letters(text[k:k + 2]) if text[k:k + 2].strip() else None
                 for k in range(0, len(text), 2)]
        if i in FIXED_X or i in CENTRED: nx = x
        else:
            centre = x + len(codes) * 10
            nx = int(round(centre - len(tl['s%d' % i]) * 10 / 2))
        out[i] = (nx, cells)
    return out

def check(tl, lay):
    """Hard problems (the build stops) and fit warnings (text too long for
    its place; the build goes on, the tl text needs shortening)."""
    probs, warn = [], []
    for i, (x, cells) in lay.items():
        n = len(cells)
        if i in MAX_CELLS and n > MAX_CELLS[i]:
            probs.append(f's{i}: {n} cells > {MAX_CELLS[i]}')
        end = x + 10 * len(tl['s%d' % i].rstrip())
        if i in RIGHT_EDGE and end > RIGHT_EDGE[i]:
            warn.append(f's{i}: ends at x {end} > {RIGHT_EDGE[i]}')
        if i in CENTRED and n * 20 > 480:
            warn.append(f's{i}: {n * 20} px wide, over the screen')
    return probs, warn

# ------------------------------------------------------------ schools
def school_cells():
    """The Backup list's school labels (tools/cock_hud.py: same text from
    tl/COCK.txt, font, size and style), centred in 80x20 cells, as CLUT 7
    indices. Returns ({name: img}, CLUT 7 as 16 u16)."""
    import cock_hud
    tl = cock_hud.load_tl()
    # cock_hud palette 51 index -> CLUT 7 index, colour (r, g, b)
    pmap = {5: (1, (0, 72, 56)), 11: (2, (0, 184, 112)), 4: (3, (192, 224, 232)),
            14: (4, (0, 224, 168)), 15: (5, (0, 216, 96))}
    lut = np.zeros(16, np.uint8)
    for k, (i, _) in pmap.items(): lut[k] = i
    out = {}
    for name, jp in zip(SCHOOLS, SCHOOL_JP):
        img = cock_hud.school_piece(tl['学校:' + jp], SCHOOL_W)
        bad = set(np.unique(img)) - set(pmap) - {0}
        assert not bad, (name, bad)
        cell = np.zeros((20, SCHOOL_W), np.uint8)
        dx = (SCHOOL_W - img.shape[1]) // 2
        cell[:img.shape[0], dx:dx + img.shape[1]] = lut[img]
        out[name] = cell
    clut = [0] * 16
    for i, (r, g, b) in pmap.values():
        clut[i] = 0x8000 | (b >> 3) << 10 | (g >> 3) << 5 | r >> 3
    return out, clut

def school_recs(ovl):
    out = []
    for k in range(5):
        r = struct.unpack_from('<8H', ovl, SCHOOL_REC + 16 * k)
        assert r[1] == 0x340, r
        out.append((r[3], r[4], r[5], r[6]))           # u, v, w, h in t832
    return out

def narrow_date_cells(sheet):
    """Digits 0-9 (letter font, 10 px, CLUT 0 ramp as letters()) into the
    NDIG cells and "/" into the 月 cell (left 10 px); 日 is cleared.
    Returns the digit uv table (u, v, w, h bytes x 10)."""
    tab = b''
    for k, (t, r, c) in enumerate(NDIG):
        x, y = cell_xy(t, r, c)
        sheet[y:y + 20, x:x + 20] = 0
        for j in (0, 1):
            d = 2 * k + j
            sheet[y:y + 20, x + 10 * j:x + 10 * j + 10] = letters(str(d))[:, :10]
            tab += bytes((c * 20 + 10 * j, r * 20, PITCH, 20))
    mx, my = cell_xy(*MONTH); dx, dy = cell_xy(*DAY)
    sheet[my:my + 20, mx:mx + 20] = 0
    sheet[my:my + 20, mx:mx + 10] = letters('/')[:, :10]
    sheet[dy:dy + 20, dx:dx + 20] = 0
    return tab

# ------------------------------------------------------------ full nickname
# Continue list name (findings: private/re/titlenick). The drawer 0x800a6054
# reads 4 glyph codes at record +0x86. NAME_INIT (jal at 0x800a60ac) reads the
# 4 pair cells at record +0x96 (0x8000 | c1 << 7 | c2; ' ' halves skipped,
# 0x061e or 0x3001 ends). If every letter is 0-9, A-Z or a-z it writes up to
# 8 narrow codes 0x8000 | half << 12 | cell + 0x3001 to NAME_BUF, points s2
# there, sets s7 = -4 (9 loop passes) and pitch 10 in the frame slot 0x24.
# Otherwise (JP kana, empty, other codes) nothing changes: +0x86, pitch 20.
# NAME_DSTUB replaces the table lookup: negative codes give t0 = code and
# at = half * 10, others the table entry and at = 0; at is added to u.
TABFILL = (0x4cc, 0x578)              # name table 0x378 entries 170-255 (all "*")
NAME_PATCH = [   # addr, JP word, new word (None: filled in build)
    (0x800a60ac, 0x03c3f021, None),               # addu fp,fp,v1 -> jal init
    (0x800a60d8, 0x3c01800a, None),               # lui at -> jal dstub
    (0x800a60dc, 0x00220821, 0),                  # addu at,at,v0 -> nop
    (0x800a60e0, 0x84282378, 0),                  # lh t0,0x2378(at) -> nop
    (0x800a6154, 0x00000000, 0x02c1b021),         # nop -> addu s6,s6,at
    (0x800a61c4, 0x00000000, 0x24030014),         # nop -> addiu v1,zero,20
    (0x800a61e4, 0x24020014, 0x87a20024),         # -> lh v0,0x24(sp) (w)
    (0x800a61f4, 0xa6020012, 0xa6030012),         # sh v0 -> sh v1,0x12(s0) (h)
    (0x800a6210, 0x00000000, 0x87a30024),         # nop -> lh v1,0x24(sp)
    (0x800a6218, 0x26730014, 0x02639821)]         # step -> addu s3,s3,v1

def name_glyph(ch):
    """10x20 CLUT 0 indices: the 20 px sheet letter (t768 rows 0-5) squeezed
    horizontally to 0.6 of its ink width (at most 9 px), area-averaged,
    centred in 10 px. Edge columns under half coverage are darkened."""
    i = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'.index(ch)
    r, c = divmod(i, 12)
    g = _SHEET0[r * 20:r * 20 + 20, c * 20:c * 20 + 20]
    cols = np.where(g.any(0))[0]
    g = g[:, cols[0]:cols[-1] + 1]; W = g.shape[1]
    wt = max(1, min(9, int(round(W * 0.6)))); s = W / wt
    lum = _LUM[g] * (g > 0); op = (g > 0).astype(np.float32)
    ramp = np.arange(3, 15)
    out = np.zeros((20, 10), np.uint8); x0 = (10 - wt) // 2
    for j in range(wt):
        w = np.array([max(0, min((j + 1) * s, k + 1) - max(j * s, k)) for k in range(W)], np.float32)
        al = (op * w).sum(1) / w.sum()
        lu = (lum * w).sum(1) / np.maximum((op * w).sum(1), 1e-6)
        for y in range(20):
            if al[y] < 0.3: continue
            v = lu[y] * min(1, al[y] / 0.5)
            out[y, x0 + j] = ramp[np.abs(_LUM[ramp] - v).argmin()]
    return out

def narrow_name_cells(sheet0, tim0, sheet):
    global _SHEET0, _LUM
    _SHEET0, _LUM = sheet0, clut_rgb(tim0, 0)[:, 0].astype(np.float32)
    for t, r, c in NNAME:
        x, y = cell_xy(t, r, c); sheet[y:y + 20, x:x + 20] = 0
    for k, ch in enumerate(NAME_CHARS):
        t, r, c = NNAME[k >> 1]
        x, y = cell_xy(t, r, c)
        sheet[y:y + 20, x + 10 * (k & 1):x + 10 * (k & 1) + 10] = name_glyph(ch)

def name_code(ch):
    """The narrow code NAME_INIT writes for ch (offline mirror)."""
    k = NAME_CHARS.index(ch); m = k >> 1
    return 0x8080 | (k & 1) << 12 | (m + (m & 0x18))

def name_asm(area_org):
    """Code for the two free ranges. Returns {org: bytes}."""
    from hook import Asm, R
    class A(Asm):
        def lh(self, rt, off, rs): self.i(0x21, rs, rt, off)
        def bltz(self, rs, lab): self.fix.append((len(self.code), lab)); self.i(1, rs, 'zero', 0)
        def bnez(self, rs, lab): self.branch('bne', rs, 'zero', lab)
        def beqz(self, rs, lab): self.branch('beq', rs, 'zero', lab)
        def b(self, lab): self.branch('beq', 'zero', 'zero', lab)
        def finish(self):
            for idx, lab in self.fix:
                off = (self.labels[lab] - (self.org + 4 * idx + 4)) >> 2
                assert -0x8000 <= off < 0x8000
                self.code[idx] = (self.code[idx] & 0xffff0000) | (off & 0xffff)
            return b''.join(struct.pack('<I', w) for w in self.code)
    fill = BASE + TABFILL[0]
    labels = {}
    def two(pass_labels):
        f = A(fill); f.labels.update(pass_labels)
        g = A(area_org); g.labels = f.labels
        # ---- table filler: BUF, dstub, init head, done, fail
        f.label('buf')
        for _ in range(5): f.emit(0)
        f.label('dstub')                  # v1 code, v0 = code * 2
        f.bltz('v1', 'narrow')
        f.lui('at', 0x800a)
        f.addu('at', 'at', 'v0')
        f.lh('t0', 0x2378, 'at')
        f.jr('ra')
        f.addu('at', 'zero', 'zero')
        f.label('narrow')
        f.andi('at', 'v1', 0x1000)
        f.srl('at', 'at', 9)               # 8 * half
        f.srl('t0', 'at', 2)               # 2 * half
        f.addu('at', 'at', 't0')
        f.jr('ra')
        f.addu('t0', 'v1', 'zero')
        f.label('init')                   # a1 record, delay slot sw s3
        f.addu('fp', 'fp', 'v1')
        f.addiu('t0', 'zero', 20)
        f.sh('t0', 0x24, 'sp')
        f.addiu('t1', 'a1', 0x96)
        b = f.labels.get('buf', fill)
        f.lui('t3', (b + 0x8000) >> 16)
        f.addiu('t3', 't3', b & 0xffff)
        f.addiu('t4', 't1', 8)
        f.b('cell')
        f.addu('t2', 't3', 'zero')
        f.label('chn')                    # next letter, next cell
        f.bnez('a3', 'ch')
        f.nop()
        f.branch('bne', 't1', 't4', 'cell')
        f.nop()
        f.label('done')
        f.branch('beq', 't2', 't3', 'fail')
        f.addiu('v1', 'zero', 0x3001)
        f.sh('v1', 0, 't2')
        f.addu('s2', 't3', 'zero')
        f.addiu('s7', 'zero', -4)
        f.addiu('t0', 'zero', 10)
        f.jr('ra')
        f.sh('t0', 0x24, 'sp')
        f.label('fail')
        f.jr('ra')
        f.nop()
        # ---- string area: cell loop, letters
        g.label('cell')
        g.lh('a2', 0, 't1')
        g.addiu('t1', 't1', 2)
        g.bltz('a2', 'pair')
        g.addiu('a3', 'zero', 2)
        g.addiu('v1', 'a2', -0x3001)
        g.beqz('v1', 'done')
        g.addiu('v1', 'a2', -0x061e)
        g.beqz('v1', 'done')
        g.nop()
        g.b('fail')
        g.nop()
        g.label('pair')
        g.sll('a2', 'a2', 18)             # c1 in bits 25-31
        g.label('ch')
        g.srl('a0', 'a2', 25)
        g.sll('a2', 'a2', 7)
        g.addiu('a3', 'a3', -1)
        g.addiu('v1', 'a0', -0x20)
        g.beqz('v1', 'chn')               # ' ' half: skip
        g.addiu('v1', 'a0', -0x30)
        g.sltiu('v0', 'v1', 10)
        g.bnez('v0', 'put')               # 0-9: k 0-9
        g.addiu('t5', 'a0', -0x41)
        g.sltiu('v0', 't5', 26)
        g.bnez('v0', 'put')               # A-Z: k 10-35
        g.addiu('v1', 't5', 10)
        g.addiu('t5', 'a0', -0x61)
        g.sltiu('v0', 't5', 26)
        g.beqz('v0', 'fail')              # a-z: k 36-61, else fail
        g.addiu('v1', 't5', 36)
        g.label('put')
        g.srl('v0', 'v1', 1)              # m
        g.andi('a0', 'v0', 0x18)
        g.addu('v0', 'v0', 'a0')          # (m >> 3) << 4 | m & 7
        g.andi('a0', 'v1', 1)
        g.sll('a0', 'a0', 12)
        g.or_('v0', 'v0', 'a0')
        g.ori('v0', 'v0', 0x8080)         # row 8 + ..., narrow flag
        g.sh('v0', 0, 't2')
        g.b('chn')
        g.addiu('t2', 't2', 2)
        return f, g
    f, g = two({})                     # pass 1: labels
    f, g = two(dict(f.labels))         # pass 2: forward refs resolved
    fb, gb = f.finish(), g.finish()
    assert TABFILL[0] + len(fb) <= TABFILL[1], len(fb)
    return {fill: fb, area_org: gb}, f.labels

# ------------------------------------------------------------ code patch
def j_op(op, target): return op << 26 | (target >> 2) & 0x3ffffff

def set_imm(ovl, addr, old, new):
    o = addr - BASE
    w = struct.unpack_from('<I', ovl, o)[0]
    assert w >> 26 == 9 and w & 0xffff == old, (hex(addr), hex(w))
    struct.pack_into('<I', ovl, o, w & 0xffff0000 | new & 0xffff)

def day_stub(addr):
    """jal'd in place of jal DIGITS: a1 day, a3 x."""
    return [
        0x2ca2000a,                  # sltiu v0, a1, 10
        0x10400002,                  # beqz v0, +2
        0x00000000,                  # nop
        0x24e70000 | (-PITCH & 0xffff),   # addiu a3, a3, -10
        j_op(2, DIGITS),             # j DIGITS
        0x00000000]

# ------------------------------------------------------------ build
def build(png=False):
    tl = load_tl()
    ovl = bytearray(lz.decompress(open(SRC_OVL, 'rb').read())[0])
    tim = bytearray(lz.decompress(open(SRC_TIT, 'rb').read())[0])
    jp = strings(ovl)
    sheet0 = sheet_of(tim)
    sheet = sheet0.copy()
    lay = layout(tl, jp)
    probs, warn = check(tl, lay)
    for p in warn: print('FIT', p)
    # unique cells -> free cells
    uniq = {}
    for i in range(NENT):
        for c in lay[i][1]:
            if c is not None: uniq.setdefault(c.tobytes(), c)
    if len(uniq) > len(FREE):
        probs.append(f'{len(uniq)} distinct cells > {len(FREE)} free')
    for p in probs: print('PROBLEM', p)
    if probs: sys.exit(1)
    codes = {}
    for (key, c), (t, r, col) in zip(uniq.items(), FREE):
        x, y = cell_xy(t, r, col)
        sheet[y:y + 20, x:x + 20] = c
        codes[key] = code_of(t, r, col)
    for t, r, col in FREE[len(uniq):]:
        x, y = cell_xy(t, r, col); sheet[y:y + 20, x:x + 20] = 0
    # date: narrow digits, "/" (月), blank 日
    dtab = narrow_date_cells(sheet)
    narrow_name_cells(sheet0, tim, sheet)          # full nickname letters
    # schools: new 80x20 cells, Backup list style, own CLUT 7
    sch, clut7 = school_cells()
    sheet[200:240, 256 + 60:512] = 0                 # old school cells
    for name, (u, v) in zip(SCHOOLS, SCHOOL_UV):
        sheet[v:v + 20, 256 + u:256 + u + SCHOOL_W] = sch[name]
    o = TIM_CLUT + 32 * CLUT_SCHOOL
    struct.pack_into('<16H', tim, o, *clut7)
    put_sheet(tim, sheet)
    build_logo.patch('tit', tim)         # EN diary logos (big, small)
    # repack strings
    area = bytearray()
    ptrs = []
    for i in range(NENT):
        while len(area) % 4: area += b'\0\0'
        ptrs.append(AREA[0] + len(area))
        cs = [0xfff if c is None else codes[c.tobytes()] for c in lay[i][1]]
        area += struct.pack('<%dH' % (len(cs) + 1), len(cs), *cs)
    while len(area) % 4: area += b'\0\0'
    stub_off = AREA[0] + len(area)
    stub = day_stub(BASE + stub_off)
    area += struct.pack('<%dI' % len(stub), *stub)
    dtab_off = AREA[0] + len(area)
    area += dtab
    name_org = BASE + AREA[0] + len(area)
    blobs, nlab = name_asm(name_org)
    area += blobs[name_org]
    room = AREA[1] - AREA[0]
    print(f'strings+stub {len(area)} B of {room}; {len(uniq)} cells of {len(FREE)} free')
    if len(area) > room: sys.exit(f'PROBLEM string area {len(area)} > {room}')
    ovl[AREA[0]:AREA[1]] = bytes(area).ljust(room, b'\0')
    for i in range(NENT):
        struct.pack_into('<HHI', ovl, TAB + 12 * i, lay[i][0], jp[i][1], BASE + ptrs[i])
    o = DAY_CALL - BASE
    assert struct.unpack_from('<I', ovl, o)[0] == j_op(3, DIGITS)
    struct.pack_into('<I', ovl, o, j_op(3, BASE + stub_off))
    for addr, old, k in DTAB_REFS:
        set_imm(ovl, addr, old, dtab_off + 0x2000 + k)
    for addr, old, new in IMMS:
        set_imm(ovl, addr, old, new)
    fill = bytes(ovl[TABFILL[0]:TABFILL[1]])
    assert fill == b'\x5b\0' * ((TABFILL[1] - TABFILL[0]) // 2), 'name table filler changed'
    fb = blobs[BASE + TABFILL[0]]
    ovl[TABFILL[0]:TABFILL[0] + len(fb)] = fb
    for addr, old, new in NAME_PATCH:
        if new is None: new = j_op(3, nlab['init' if addr == 0x800a60ac else 'dstub'])
        assert struct.unpack_from('<I', ovl, addr - BASE)[0] == old, hex(addr)
        struct.pack_into('<I', ovl, addr - BASE, new)
    print(f'name: filler {len(fb)} B of {TABFILL[1] - TABFILL[0]}, area code {len(blobs[name_org])} B')
    for k, (u, v) in enumerate(SCHOOL_UV):
        struct.pack_into('<3H', ovl, SCHOOL_REC + 16 * k + 6, u, v, SCHOOL_W)
    # compress, check, write
    zo, zt = lz.compress(bytes(ovl), name='MGTIT.BIZ'), lz.compress(bytes(tim), name='MA_TIT.BIZ')
    assert lz.decompress(zo)[0] == bytes(ovl) and lz.decompress(zt)[0] == bytes(tim)
    print(f'MGTIT.BIZ {len(zo)} B (slot {SLOT_OVL}), MA_TIT.BIZ {len(zt)} B (slot {SLOT_TIT})')
    if len(zo) > SLOT_OVL or len(zt) > SLOT_TIT: sys.exit('PROBLEM: does not fit its slot')
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'MGTIT.BIZ'), 'wb').write(zo)
    open(os.path.join(OUT, 'MA_TIT.BIZ'), 'wb').write(zt)
    if png: previews(tim, sheet0, sheet, jp, lay, codes, sch)

# ------------------------------------------------------------ previews
def previews(tim, sheet0, sheet, jp, lay, codes, sch):
    os.makedirs(PREV, exist_ok=True)
    c6, c7, c0 = clut_rgb(tim, 6), clut_rgb(tim, 7), clut_rgb(tim, 0)
    bg = np.array([24, 16, 184], np.uint8)   # list row blue
    def rgb(idx, pal):
        o = pal[idx]; o[idx == 0] = bg; return o
    def draw(codes_, sh):
        out = np.zeros((20, 20 * max(len(codes_), 1)), np.uint8)
        for k, cd in enumerate(codes_):
            if cd == 0xfff: continue
            x, y = cell_xy(*cell_of(cd)); out[:, 20 * k:20 * k + 20] = sh[y:y + 20, x:x + 20]
        return out
    rows = []
    for i in range(NENT):
        x, y, adv, jc = jp[i]
        en = [0xfff if c is None else codes[c.tobytes()] for c in lay[i][1]]
        a = draw(jc, sheet0); b = draw(en, sheet)
        # place both on a 512 px line at their screen x (centred ones at 256)
        line = np.zeros((44, 520), np.uint8)
        for k, (xx, img) in enumerate(((x, a), (lay[i][0], b))):
            if i in CENTRED: xx = (512 - img.shape[1]) // 2
            line[2 + 22 * k:22 + 22 * k, xx:xx + img.shape[1]] = img[:, :520 - xx]
        rows.append(rgb(line, c6))
        rows.append(np.full((2, 520, 3), 255, np.uint8))
    im = np.vstack(rows)
    Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST).save(f'{PREV}/strings.png')
    # schools and date marker cells
    parts = []
    tim0 = lz.decompress(open(SRC_TIT, 'rb').read())[0]
    c7j = clut_rgb(tim0, 7)
    ovl0 = lz.decompress(open(SRC_OVL, 'rb').read())[0]
    for name, (u, v, w, h), (nu, nv) in zip(SCHOOLS, school_recs(ovl0), SCHOOL_UV):
        j = np.zeros((20, SCHOOL_W, 3), np.uint8); j[:] = bg
        j[:, :w] = rgb(sheet0[v:v + h, 256 + u:256 + u + w], c7j)
        e = rgb(sheet[nv:nv + 20, 256 + nu:256 + nu + SCHOOL_W], c7)
        parts.append(np.vstack([j, np.zeros((2, SCHOOL_W, 3), np.uint8), e]))
    for t, r, c in (MONTH, DAY) + tuple(NDIG):
        x, y = cell_xy(t, r, c)
        parts.append(np.vstack([rgb(sheet0[y:y + 20, x:x + 20], c0), np.zeros((2, 20, 3), np.uint8),
                                rgb(sheet[y:y + 20, x:x + 20], c0)]))
    W = sum(p.shape[1] + 6 for p in parts)
    im = np.zeros((42, W, 3), np.uint8); x = 0
    for p in parts: im[:, x:x + p.shape[1]] = p; x += p.shape[1] + 6
    Image.fromarray(im).resize((W * 2, 84), Image.NEAREST).save(f'{PREV}/cells.png')
    print('previews in', PREV)

if __name__ == '__main__':
    build(png='--png' in sys.argv)
