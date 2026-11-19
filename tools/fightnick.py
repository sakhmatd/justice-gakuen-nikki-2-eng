# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Fight HUD player nickname with up to 8 letters (called by
build_fight.py; notes: private/re/fightnames/findings.org).

The renderer 0x800290cc draws the player's name into the id 27-30 cells
of CEFT1 TIM1 from COCKEDIT.TIZ glyphs (24 x 8 cells, codes = shared
glyph ids). The JP code takes the first 4 codes of the fighter record.
This module
- moves the player cells to TIM (0..95, V..V+H) (renderer ClearImage and
  slot y, name table ids 27-30), with a glyph pitch of 7 rows;
- adds EXPAND at 0x80010bf0 (cave between the hook and build_sysmsg's
  stub at 0x80010d0c): when the record's 4 codes are the diary
  nickname's first 4 letters (0x801f1886), it rebuilds the glyph list
  from the full nickname's letter pairs at 0x801f1896 (up to 8 letters);
- redraws COCKEDIT glyphs 1-62 (digits, A-Z, a-z) in the style of the EN
  fight names (build_fight.name_style): rotated clockwise, palette 48,
  one rim per glyph, ink in rows 0..6. a-z are drawn as capitals, like
  the other EN names. M is hand-drawn (DRAWN). Kana glyphs (63+) are
  kept.
"""
import os, struct
import numpy as np
from PIL import Image
import lz, streamsim
from hook import Asm

EB = 0x8000f800                     # RAM - exe file offset
EXPAND = 0x80010bf0
BUF = 0x80010cf8                    # 20 bytes, ends 0x80010d0c (sysmsg EXE_STUB)
V, H, PITCH = 200, 56, 7            # player cells: TIM (0..95, 200..255)
CELLS = (0, V, 96, H)               # x, y, w, h (runtime-written texels)
OLD_CELLS = (0, 64, 96, 32)         # JP player cells, free for names now
FONT_SRC = 'private/unpacked/COCKEDIT.TIZ.lz'
FONT_DISC = 'private/disc/DAT2/COCKEDIT.TIZ'
FONT_OUT = 'private/build/COCKEDIT.TIZ'
GLYPHS = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # codes 1-62

class A(Asm):
    def srlv(self, rd, rt, rs): self.r(0, rs, rt, rd, 0, 0x06)
    def b(self, lab): self.branch('beq', 'zero', 'zero', lab)
    def j(self, t): self.emit(0x08000000 | (t >> 2 & 0x3ffffff))

def jal(t): return 0x0c000000 | (t >> 2 & 0x3ffffff)

def build_code():
    """EXPAND. Entry from 0x800291b4 (delay slot: t0 = v1 << 2; a0 = record
    base, v1 = index). Copies the 4 record codes (record + 0x10) to BUF;
    if they equal 0x801f1886..+7, rebuilds BUF from the 4 pair cells at
    0x801f1896 (0x8000 | c1 << 7 | c2; 0x061e pads and ' ' dropped,
    '1'-'9' -> 1-9, '0' -> 10, A-Z -> 11-36, a-z -> 37-62). Any other cell
    (kana) restarts with the record copy only. Appends 0x3001, returns
    s0 = BUF - 0x10. Uses t0-t9."""
    a = A(EXPAND)
    # Only records 16+ hold the diary player (EDIT/password slots are
    # 0-15): t9 = 1 there forces the plain record copy.
    a.sltiu('t9', 'v1', 0x10)
    a.label('restart')
    a.addu('t0', 't0', 'v1'); a.sll('t0', 't0', 3); a.addu('t0', 't0', 'a0')
    a.lui('t1', 0x801f); a.ori('t1', 't1', 0x1886)
    a.lui('t4', BUF >> 16); a.addiu('t4', 't4', BUF & 0xffff)
    a.addiu('t2', 'zero', 4)
    a.label('cmp')
    a.lhu('t3', 0x10, 't0'); a.lhu('t5', 0, 't1')
    a.addiu('t0', 't0', 2); a.addiu('t1', 't1', 2)
    a.sh('t3', 0, 't4'); a.xor('t5', 't5', 't3'); a.or_('t9', 't9', 't5')
    a.addiu('t2', 't2', -1)
    a.branch('bne', 't2', 'zero', 'cmp')
    a.addiu('t4', 't4', 2)
    a.branch('bne', 't9', 'zero', 'done')
    a.addiu('t1', 't1', 8)                  # 0x801f1896
    a.addiu('t4', 't4', -8)
    a.addiu('t2', 'zero', 4)
    a.label('cell')
    a.lhu('t3', 0, 't1'); a.addiu('t1', 't1', 2)
    a.addiu('t5', 'zero', 0x3001)
    a.branch('beq', 't3', 't5', 'done')
    a.andi('t6', 't3', 0x8000)
    a.branch('bne', 't6', 'zero', 'pair')
    a.addiu('t5', 'zero', 0x61e)
    a.branch('beq', 't3', 't5', 'next')
    a.nop()
    a.addiu('t9', 'zero', 1)                # kana etc.: record copy only
    a.j(EXPAND + 4)
    a.sll('t0', 'v1', 2)
    a.label('pair')
    a.addiu('t7', 'zero', 7)
    a.label('half')
    a.srlv('t6', 't3', 't7'); a.andi('t6', 't6', 0x7f)
    a.addiu('t8', 't6', -0x30); a.sltiu('t5', 't8', 10)
    a.branch('beq', 't5', 'zero', 'alpha')
    a.nop()
    a.branch('bne', 't8', 'zero', 'store')
    a.nop()
    a.b('store')
    a.addiu('t8', 'zero', 10)
    a.label('alpha')
    a.andi('t8', 't6', 0xdf); a.addiu('t8', 't8', -0x41); a.sltiu('t5', 't8', 26)
    a.branch('beq', 't5', 'zero', 'skip')
    a.andi('t5', 't6', 0x20)
    a.branch('beq', 't5', 'zero', 'store')
    a.addiu('t8', 't8', 11)
    a.addiu('t8', 't8', 26)
    a.label('store')
    a.sh('t8', 0, 't4'); a.addiu('t4', 't4', 2)
    a.label('skip')
    a.branch('bne', 't7', 'zero', 'half')
    a.addu('t7', 'zero', 'zero')
    a.label('next')
    a.addiu('t2', 't2', -1)
    a.branch('bne', 't2', 'zero', 'cell')
    a.nop()
    a.label('done')
    a.addiu('t5', 'zero', 0x3001); a.sh('t5', 0, 't4')
    a.lui('s0', ((BUF - 0x10) + 0x8000) >> 16)
    a.jr('ra')
    a.addiu('s0', 's0', (BUF - 0x10) & 0xffff)
    code = a.finish()
    assert EXPAND + len(code) <= BUF, hex(EXPAND + len(code))
    return code

# Renderer edits (RAM, JP word, new word).
EDITS = [
    (0x80029128, 0x24020140, 0x24020100 + V),        # ClearImage y 320 -> 256 + V
    (0x80029138, 0x24020020, 0x24020000 + H),        # ClearImage h 32 -> H
    (0x80029164, 0x24020008, 0x24020000 + PITCH),    # glyph rect h 8 -> PITCH
    (0x800291ac, 0x24020140, 0x24020100 + V),        # slot y start
    (0x800291b4, 0x00031080, jal(EXPAND)),           # sll v0,v1,2 -> jal EXPAND
    (0x800291b8, 0x00431021, 0x00034080),            # -> sll t0,v1,2 (delay)
    (0x800291bc, 0x000210c0, 0),
    (0x800291c0, 0x00448021, 0),
    (0x8002920c, 0x24420008, 0x24420000 + PITCH),    # y step 8 -> PITCH
    (0x80029214, 0x2a220004, 0x2a220008),            # 4 -> 8 letters
]
TABLE = 0x8008ede8                  # fighter-name table (u, v, h) x 31
PLAYER_IDS = (27, 28, 29, 30)
RANGES = ([(r - EB, r - EB + 4) for r, _, _ in EDITS] + [(EXPAND - EB, BUF - EB)]
          + [(TABLE - EB + 3 * i, TABLE - EB + 3 * i + 3) for i in PLAYER_IDS])

def patch_exe(exe, disc, problems):
    """Writes EDITS, EXPAND and table ids 27-30 into exe (bytearray).
    Every value is checked against the disc exe; a rerun on its own
    output is allowed."""
    for r, old, new in EDITS:
        o = r - EB
        if struct.unpack_from('<I', disc, o)[0] != old:
            problems.append(f'nick: disc word at {r:#x} is not {old:#010x}'); continue
        struct.pack_into('<I', exe, o, new)
    for i in PLAYER_IDS:
        o = TABLE - EB + 3 * i
        if disc[o:o + 3] != bytes((0, 64, 32)):
            problems.append(f'nick: name table id {i} is {tuple(disc[o:o + 3])}'); continue
        exe[o:o + 3] = bytes((0, V, H))
    code = build_code()
    o0, o1 = EXPAND - EB, BUF + 20 - EB
    if any(disc[o0:o1]): problems.append('nick: cave not empty on the disc exe')
    cur = bytes(exe[o0:o1])
    # A rerun finds an older EXPAND (same length, buffer still zero).
    if any(cur[len(code):]):
        problems.append(f'nick: cave {EXPAND:#x}..{BUF + 20:#x} is used by another tool')
    else:
        exe[o0:o0 + len(code)] = code
    return len(code)

# ---------------------------------------------------------------- font ---

# Hand-drawn letters (upright, 5 wide x 12 tall fill, X = fill), used
# instead of the squeezed font. The font M at 5 px read as N at 1x: its
# middle notch was 3 rows deep and the right counter was lost. This M has
# a 7-row middle notch and two counters below the vertex.
DRAWN = {
    'M': ['XX.XX'] * 7 + ['X.X.X'] * 5,
}

def drawn(rows):
    """Upright 12 x 5 bitmap -> 7 x 24 cell, placed like the font glyphs
    (fill rows 7..18 of the upright 24-row cell, columns 1..5)."""
    import build_fight as bf
    up = np.zeros((24, 7), np.float32)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c == 'X': up[7 + y, 1 + x] = 1
    return bf.name_style(np.rot90(up, -1).copy())

def glyph(ch):
    """24 x 7 glyph: the letter as in build_fight.name_strip (cap 12,
    Inconsolata Extra Condensed Black, rotated clockwise), squeezed to at
    most 5 rows of fill and centred in rows 1..5, with its own rim.
    Letters in DRAWN use their bitmap."""
    import build_fight as bf
    if ch in DRAWN: return drawn(DRAWN[ch])
    def mw(a):
        c = np.nonzero((a > 0.45).any(0))[0]
        return c[-1] - c[0] + 1 if len(c) else 0
    s = 1.0
    while True:
        a = bf.text_alpha(ch, bf.NAME_CAP, bf.F_NAME, sx=s, minsx=0.05)
        if mw(a) <= 5: break
        s -= 0.02
    c = np.nonzero((a > 0.45).any(0))[0]
    a = a[:, c[0]:c[-1] + 1]
    h, w = a.shape
    full = np.zeros((22, 7), np.float32)
    y, x = (22 - h) // 2, 1 + (5 - w) // 2
    full[y:y + h, x:x + w] = a
    s2 = np.zeros((7, 24), np.float32)
    s2[:, 0:22] = np.rot90(full, -1)
    return bf.name_style(s2)

def font_px(d):
    raw = np.frombuffer(bytes(d[0x14:0x14 + 12 * 1408]), np.uint8).reshape(1408, 12)
    px = np.empty((1408, 24), np.uint8)
    px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def build_font(problems):
    """Returns (decompressed, compressed) COCKEDIT.TIZ."""
    d = bytearray(open(FONT_SRC, 'rb').read())
    if struct.unpack_from('<I4H', d, 8) != (16908, 512, 0, 6, 1408):
        problems.append('COCKEDIT: unexpected TIM header'); return d, b''
    px = font_px(d)
    for code, ch in enumerate(GLYPHS, 1):
        px[8 * code:8 * code + 8] = 0
        px[8 * code:8 * code + 7] = glyph(ch)
    raw = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8)
    d[0x14:0x14 + raw.size] = raw.tobytes()
    slot = os.path.getsize(FONT_DISC); slot += -slot % 2048
    c = lz.compress(bytes(d), name='COCKEDIT.TIZ')
    out, done = streamsim.run(c)
    cb, cap = lz.cbmax(c), lz.jp_cap('COCKEDIT.TIZ')
    ok = done and out[:len(d)] == bytes(d)
    print(f'COCKEDIT.TIZ: {len(c)} bytes, slot {slot} ({slot - len(c):+d}), per-callback max '
          f'{cb:#x} (JP cap {cap:#x}), streamsim {"ok" if ok else "FAILED"}')
    if cb > cap: problems.append(f'COCKEDIT.TIZ: callback output {cb:#x} > {cap:#x}')
    if len(c) > slot: problems.append(f'COCKEDIT.TIZ: {len(c)} > slot {slot}')
    if not ok: problems.append('COCKEDIT.TIZ: streamsim failed')
    return d, c

def cell(px, codes):
    """Preview: the 24 x H player cell the renderer builds from codes."""
    out = np.zeros((H, 24), np.uint8)
    for k, g in enumerate(codes[:8]):
        out[PITCH * k:PITCH * k + PITCH] = px[8 * g:8 * g + PITCH]
    return out

def codes(text):
    return [int(ch) or 10 if ch.isdigit() else
            ord(ch) - 0x41 + 11 if ch.isupper() else ord(ch) - 0x61 + 37 for ch in text if ch != ' ']
