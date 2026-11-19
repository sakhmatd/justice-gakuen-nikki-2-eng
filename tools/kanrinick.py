# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""KANRI nickname tag with up to 8 letters (called by build_kanri.py;
notes: private/re/kanrinick/findings.org, private/re/nickreaders/).

The diary data page (call 0x800a77dc) and the card screen (0x800a6c18)
draw the fighter record's 4 nickname codes with the name drawer
0x800a3f74 (20 px MOJI2 cells, pitch 20, x 44, y 40, CLUT 0xc). This
module sends those two calls to NDRAW at the KANRI file end:
- if the record's 4 codes equal the diary nickname's first 4 letters
  (0x801f1886), it rebuilds the name from the 4 letter-pair cells at
  0x801f1896 (0x8000 | c1 << 7 | c2; ' ' and 0x061e cells dropped) into
  BUF; any other case (kana, unknown codes) goes to the JP drawer;
- it draws one SPRT per letter as the JP drawer does (same frame,
  prim layout 0x38 per letter, same library calls), but from narrow
  proportional glyphs in free texels of the EDITMOJI TIM at VRAM (704,0),
  centred on the tag box (x 42-125), advance = width + GAP, 1 px tighter
  when the name is wider than BOXW.
The glyphs are the screen's own MOJI2 letters (KANRI code table 0xd74),
squeezed horizontally to SQUEEZE with a rebuilt 1 px shadow (index 3).
The guts gauge and the register screens keep the JP drawer.
"""
import struct
import numpy as np
from hook import Asm, JALR_V0

BASE = 0x800a2000
DRAWER = 0x800a3f74
SITES = (0x800a77dc, 0x800a6c18)          # data page, card screen
NICK, PAIRS = 0x801f1886, 0x801f1896
CODE_TAB = 0xd74                          # ids 1-62 -> MOJI2 cell (file offset)
IDS = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
TPAGE_X = 704                             # EDITMOJI TIM 1
# free 20-row bands of the TIM (row, first free u); rows 10/11 start after
# the header art cells (build_kanri passes their codes)
BANDS = ((8, 180), (9, 100), (10, 0), (11, 0))
SQUEEZE, BOOST, GAP = 0.55, 1.2, 1
BOXW = 80                                 # x 44..124 inside the tag box 42-125
NTAB = 0x7b - 0x30                        # TAB index = ASCII - 0x30


class A(Asm):
    def srlv(self, rd, rt, rs): self.r(0, rs, rt, rd, 0, 0x06)
    def lh(self, rt, off, rs): self.i(0x21, rs, rt, off)
    def b(self, lab): self.branch('beq', 'zero', 'zero', lab)
    def j(self, t): self.emit(0x08000000 | (t >> 2 & 0x3ffffff))
    def jalr(self, rs): self.emit(R_JALR[rs])
    def la(self, rt, v):
        self.lui(rt, (v + 0x8000) >> 16 & 0xffff); self.addiu(rt, rt, v & 0xffff)


R_JALR = {'v0': JALR_V0, 'v1': 0x0060f809}


def jal(t): return 0x0c000000 | (t >> 2 & 0x3ffffff)


# ---------------------------------------------------------------- glyphs
def squeeze(g, s=SQUEEZE, boost=BOOST):
    """20x20 MOJI2 index glyph -> 20 x w index glyph: fill (idx 6-14 as
    coverage) area-resampled to s of its width, shadow rebuilt (idx 3)
    right, below and diagonally."""
    g = g.astype(int)
    F = np.clip((g - 5) / 9.0, 0, 1) * (g >= 6)
    cols = np.where(F.any(0))[0]
    F = F[:, cols[0]:cols[-1] + 1]
    w0 = F.shape[1]; w1 = max(2, int(round(w0 * s)))
    out = np.zeros((20, w1))
    for j in range(w1):
        a, b = j * w0 / w1, (j + 1) * w0 / w1
        for i in range(int(a), int(np.ceil(b))):
            ov = min(b, i + 1) - max(a, i)
            if ov > 0: out[:, j] += F[:, i] * ov
    out = np.clip(out / (w0 / w1) * boost, 0, 1)
    idx = np.where(out < 0.15, 0, np.clip(np.rint(5 + 9 * out), 6, 14)).astype(np.uint8)
    full = np.zeros((20, w1 + 1), np.uint8); full[:, :w1] = idx
    m = full > 0
    sh = np.zeros_like(m)
    sh[:, 1:] |= m[:, :-1]; sh[1:, :] |= m[:-1, :]; sh[1:, 1:] |= m[:-1, :-1]
    full[~m & sh] = 3
    return full


def letters(kanri, moji_px, glyph):
    """{char: 20 x w glyph} from the KANRI code table and MOJI2 pixels."""
    tab = struct.unpack_from('<63h', kanri, CODE_TAB)
    return {ch: squeeze(glyph(moji_px, tab[i] & 0xfff)) for i, ch in enumerate(IDS, 1)}


def pack(tim, gl, problems, hdr_codes=()):
    """Place the glyphs in the free bands of tim (256x256 indices, edited
    in place), after the header cells 0xffRC in hdr_codes.
    Returns {char: (u, v, w)}."""
    free = []
    for r, u0 in BANDS:
        band = tim[20 * r:20 * r + 20]
        cells = [c & 15 for c in hdr_codes if c >> 4 & 15 == r]
        if cells: u0 = max(u0, 20 * (max(cells) + 1))
        if band[:, u0:].any():
            problems.append(f'kanrinick: TIM row {r} not free from u {u0}'); continue
        free.append([u0, 20 * r])
    out = {}
    for ch in sorted(gl, key=lambda c: -gl[c].shape[1]):
        w = gl[ch].shape[1]
        for f in free:
            if f[0] + w <= 256:
                u, v = f
                tim[v:v + 20, u:u + w] = gl[ch]; out[ch] = (u, v, w); f[0] += w
                break
        else:
            problems.append(f'kanrinick: no texels left for {ch!r}')
    return out


def table(place):
    t = bytearray(3 * NTAB + (-3 * NTAB) % 4)
    for ch, (u, v, w) in place.items():
        o = 3 * (ord(ch) - 0x30)
        t[o:o + 3] = bytes((u, v, w))
    return t


# ---------------------------------------------------------------- code
def build_code(org, tab_at, buf_at):
    """NDRAW at org. TAB at tab_at, BUF at buf_at
    (8 codes + 0x3001, then the s16 gap at +18). TAB: NTAB x
    (u, v, w); the advance is w + GAP."""
    a = A(org)
    var = buf_at + 18
    # ---- expand: compare the record codes with the diary nickname
    a.li32('t1', NICK)
    a.addu('t0', 'a1', 'zero')
    a.addiu('t2', 'zero', 4)
    a.addu('t9', 'zero', 'zero')
    a.label('cmp')
    a.lhu('t3', 0, 't0'); a.lhu('t5', 0, 't1')
    a.addiu('t0', 't0', 2); a.addiu('t1', 't1', 2)
    a.xor('t5', 't5', 't3'); a.or_('t9', 't9', 't5')
    a.addiu('t2', 't2', -1)
    a.branch('bne', 't2', 'zero', 'cmp'); a.nop()
    a.branch('bne', 't9', 'zero', 'orig')
    a.addiu('t1', 't1', PAIRS - NICK - 8)
    a.la('t4', buf_at)
    a.la('t8', tab_at)
    a.addiu('t2', 'zero', 4)              # t9 = 0 here: sum of advances
    a.label('cell')
    a.lhu('t3', 0, 't1'); a.addiu('t1', 't1', 2)
    a.addiu('t5', 'zero', 0x3001)
    a.branch('beq', 't3', 't5', 'done')
    a.andi('t6', 't3', 0x8000)
    a.branch('bne', 't6', 'zero', 'pair')
    a.addiu('t5', 'zero', 0x61e)
    a.branch('beq', 't3', 't5', 'next'); a.nop()
    a.b('orig'); a.nop()
    a.label('pair')
    a.addiu('t7', 'zero', 7)
    a.label('half')
    a.srlv('t6', 't3', 't7'); a.andi('t6', 't6', 0x7f)
    a.addiu('t6', 't6', -0x20)
    a.branch('beq', 't6', 'zero', 'skip')
    a.addiu('t6', 't6', -0x10)
    a.sltiu('t5', 't6', NTAB)
    a.branch('beq', 't5', 'zero', 'orig')
    a.sll('t5', 't6', 1)
    a.addu('t5', 't5', 't6'); a.addu('t5', 't5', 't8'); a.lbu('t0', 2, 't5')
    a.sh('t6', 0, 't4')
    a.branch('beq', 't0', 'zero', 'orig')
    a.addu('t9', 't9', 't0')
    a.addiu('t9', 't9', GAP)
    a.addiu('t4', 't4', 2)
    a.label('skip')
    a.branch('bne', 't7', 'zero', 'half')
    a.addu('t7', 'zero', 'zero')
    a.label('next')
    a.addiu('t2', 't2', -1)
    a.branch('bne', 't2', 'zero', 'cell'); a.nop()
    a.label('done')
    a.la('t5', buf_at)
    a.branch('beq', 't4', 't5', 'orig')
    a.addiu('t0', 'zero', 0x3001)
    a.sh('t0', 0, 't4')
    # ---- width, tighten, centre
    a.addiu('t9', 't9', -GAP)
    a.slti('t0', 't9', BOXW + 1)
    a.branch('bne', 't0', 'zero', 'fit')
    a.addiu('t1', 'zero', GAP)
    a.subu('t0', 't4', 't5'); a.srl('t0', 't0', 1); a.addiu('t0', 't0', -1)
    a.subu('t9', 't9', 't0')
    a.addiu('t1', 'zero', GAP - 1)
    a.label('fit')
    a.lui('t0', (var + 0x8000) >> 16 & 0xffff); a.sh('t1', var & 0xffff, 't0')
    a.addiu('t0', 'zero', BOXW); a.subu('t0', 't0', 't9'); a.sra('t0', 't0', 1)
    a.addu('a3', 'a3', 't0')
    a.addu('a1', 't5', 'zero')
    # ---- draw (frame and calls as the JP drawer 0x800a3f74)
    a.addiu('sp', 'sp', -0x38)
    a.sw('s5', 0x24, 'sp'); a.addu('s5', 'a0', 'zero')
    a.sw('s6', 0x28, 'sp'); a.addu('s6', 'a1', 'zero')
    a.sw('fp', 0x30, 'sp'); a.addu('fp', 'a2', 'zero')
    a.sw('s4', 0x20, 'sp'); a.addu('s4', 'a3', 'zero')
    a.sw('s7', 0x2c, 'sp'); a.sw('ra', 0x34, 'sp')
    a.sw('s3', 0x1c, 'sp'); a.sw('s2', 0x18, 'sp')
    a.sw('s1', 0x14, 'sp'); a.sw('s0', 0x10, 'sp')
    a.label('loop')
    a.lhu('v1', 0, 's6'); a.addiu('v0', 'zero', 0x3001)
    a.branch('beq', 'v1', 'v0', 'ret')
    a.addiu('s6', 's6', 2)
    a.sll('s7', 'v1', 1); a.addu('s7', 's7', 'v1')
    a.lui('at', (tab_at + 0x8000) >> 16 & 0xffff); a.addu('s7', 's7', 'at')
    a.addiu('s7', 's7', tab_at & 0xffff)
    a.lbu('s2', 0, 's7'); a.lbu('s1', 1, 's7')
    a.addu('a0', 'zero', 'zero'); a.addu('a1', 'zero', 'zero')
    a.addiu('a2', 'zero', TPAGE_X); a.addu('a3', 'zero', 'zero')
    a.lui('v0', 0x801f); a.lw('v0', 0x608c, 'v0'); a.nop()
    a.sll('s0', 'v0', 3); a.subu('s0', 's0', 'v0')
    a.lui('v0', 0x801f); a.lw('v0', 0x6370, 'v0')
    a.sll('s0', 's0', 2); a.lw('v0', 0x18, 'v0'); a.nop()
    a.jalr('v0'); a.addu('s0', 's5', 's0')                 # GetTPage
    a.addu('a0', 's0', 'zero'); a.addu('a1', 'zero', 'zero')
    a.lui('v1', 0x801f); a.lw('v1', 0x6360, 'v1')
    a.addiu('a2', 'zero', 1); a.lw('v1', 0x34, 'v1'); a.nop()
    a.jalr('v1'); a.andi('a3', 'v0', 0xffff)               # SetDrawTPage
    a.lui('v0', 0x801f); a.lw('v0', 0x6360, 'v0')
    a.addiu('s3', 's0', 8); a.lw('v0', 0x68, 'v0'); a.nop()
    a.jalr('v0'); a.addu('a0', 's3', 'zero')               # SetSprt
    a.addiu('a1', 'zero', 0x1e1)
    a.lui('v0', 0x801f); a.lw('v0', 0x6370, 'v0')
    a.lw('t0', 0x4c, 'sp'); a.lw('v0', 0x14, 'v0'); a.nop()
    a.jalr('v0'); a.sll('a0', 't0', 4)                     # GetClut
    a.sh('v0', 0x16, 's0')
    a.addiu('v0', 'zero', 0x80)
    a.sb('v0', 0xc, 's0'); a.sb('v0', 0xd, 's0'); a.sb('v0', 0xe, 's0')
    a.lbu('v0', 2, 's7')
    a.sb('s2', 0x14, 's0'); a.sb('s1', 0x15, 's0')
    a.sh('v0', 0x18, 's0')
    a.addiu('v0', 'zero', 20); a.sh('v0', 0x1a, 's0')
    a.sh('s4', 0x10, 's0')
    a.lhu('t0', 0x48, 'sp'); a.nop(); a.sh('t0', 0x12, 's0')
    a.lui('v0', 0x801f); a.lw('v0', 0x6360, 'v0')
    a.addu('a1', 's3', 'zero'); a.lw('v0', 0x20, 'v0'); a.nop()
    a.jalr('v0'); a.addu('a0', 's0', 'zero')
    a.addu('a1', 's0', 'zero'); a.sll('v1', 'fp', 2)
    a.lui('v0', 0x801f); a.lw('v0', 0x6360, 'v0')
    a.lui('a0', 0x801f); a.lw('a0', 0x6090, 'a0')
    a.lw('v0', 0, 'v0'); a.nop()
    a.jalr('v0'); a.addu('a0', 'a0', 'v1')                 # AddPrim
    a.lbu('t0', 2, 's7')
    a.lui('t1', (var + 0x8000) >> 16 & 0xffff); a.lh('t1', var & 0xffff, 't1')
    a.addu('s4', 's4', 't0')
    a.addu('s4', 's4', 't1')
    a.b('loop')
    a.addiu('s5', 's5', 0x38)
    a.label('ret')
    a.lw('ra', 0x34, 'sp'); a.lw('fp', 0x30, 'sp'); a.lw('s7', 0x2c, 'sp')
    a.lw('s6', 0x28, 'sp'); a.lw('s5', 0x24, 'sp'); a.lw('s4', 0x20, 'sp')
    a.lw('s3', 0x1c, 'sp'); a.lw('s2', 0x18, 'sp'); a.lw('s1', 0x14, 'sp')
    a.lw('s0', 0x10, 'sp')
    a.jr('ra'); a.addiu('sp', 'sp', 0x38)
    a.label('orig')
    a.j(DRAWER); a.nop()
    return a.finish()


def restore(k, disc):
    """Put the JP call words back (build_kanri truncates and rebuilds)."""
    for s in SITES:
        o = s - BASE; k[o:o + 4] = disc[o:o + 4]


def patch(k, tim, kanri_disc, moji_px, glyph, problems, hdr_codes=()):
    """Append NDRAW, TAB and BUF to k (bytearray, word aligned), redirect
    the two diary calls, write the glyphs into tim (256x256 indices).
    Returns {char: (u, v, w)}."""
    for s in SITES:
        if struct.unpack_from('<I', k, s - BASE)[0] != jal(DRAWER):
            problems.append(f'kanrinick: call at {s:#x} is not jal {DRAWER:#x}'); return {}
    place = pack(tim, letters(kanri_disc, moji_px, glyph), problems, hdr_codes)
    org = BASE + len(k)
    size = len(build_code(org, org, org))
    tab_at = org + size
    buf_at = tab_at + len(table({}))
    code = build_code(org, tab_at, buf_at)
    assert len(code) == size
    k += code + table(place) + bytes(20)
    for s in SITES:
        struct.pack_into('<I', k, s - BASE, jal(org))
    print(f'kanrinick: NDRAW {org:#x} ({size} B), TAB {tab_at:#x}, BUF {buf_at:#x}, '
          f'{len(place)} glyphs')
    return place


def layout(text, place, x=44):
    """[(char, x)] as NDRAW draws text (the visible letters)."""
    adv = [place[c][2] + GAP for c in text if c != ' ']
    tot = sum(adv) - GAP; d = 0
    if tot > BOXW: tot -= len(adv) - 1; d = -1
    x += (BOXW - tot) // 2
    out = []
    for c in (c for c in text if c != ' '):
        out.append((c, x)); x += place[c][2] + GAP + d
    return out
