# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English name entry (DAT1/MGENT.BIN, load 0x8017a000, uncompressed).

Changes (RE: private/re/nameentry/findings.org):
- Romaji start: surname, given name and nickname open on the alnum page
  (page 2) with the keyboard cursor on "A" (key 10).
- Defaults "Seigi" / "Makoto" / "Makoto" as letter pairs.
- Keys (0x801810f0-0x801812a4, old delete + input): one body with two
  entries, delete at 0x801810f0 (old entry), input at 0x801810f8 (jal
  0x8017be18 retargeted). Latin keys pack two letters per cell, 0x8000 |
  c1<<7 | c2. The field cursor stays on a cell (c, ' ') until its second
  letter is typed, so the surname and given name hold 12 letters, the
  nickname 8. Delete turns (c1, c2) into (c1, ' '); on a single letter or a
  kana it empties the cell; on an empty cell it first steps back one cell.
- Pad 0x8018086c (OK on a field) drops empty cells (0xfff, 0x061e) inside
  the name, so 0x061e is trailing padding only (topic namefix).
- Name draw 0x80181454: a pair code gets a dynamic cell. The two letters
  are composed into a per-slot buffer, uploaded with LoadImage to VRAM
  (768 + 5*i, 416 + 20*field) and drawn from tpage (768,256). The code
  sits in the dead keyboard-draw copy 0x801816f4.
- Fighting-mode nickname (0x801f1886): jal 0x8017c000 calls a new routine
  after the pair code, which unpacks pairs into single shared glyphs and
  keeps 4 + 0x3001 (N1 NICK). The old 0x80180a10 is left unused.

- Entry form records (subjects 0x1674, words 0x1724, school name widths,
  face editor 0x118) for the EN sheet layout of tools/build_entry.py, whose
  redraw of MA_ENT TIM 0/1 patch_maent() applies.

The letter font (letter_font(), 96 x 100 B, CLUT (16,483) indices) and the
slot data live after the MA_ENT.BIZ data: MA_ENT unpacks to 0x800d0000,
0x33800 B, so the font lands at 0x80103800 (stale VAB staging data during
name entry). MGENT.BIN cannot grow: its 18 sectors leave 68 B of slack and
MGFIN.BIN follows on the disc. hook.py writes both files.

Usage: python3 tools/mgentfix.py   -> prints the patched code (writes nothing)
"""
import os, sys, struct, zlib
sys.path.insert(0, os.path.dirname(__file__))
import hook

BASE = 0x8017a000
MAENT_SIZE = 0x33800                 # unpacked MA_ENT.BIZ (load 0x800d0000)
MAENT_ALLOC = 54 * 2048              # MA_ENT.BIZ sectors on the disc (LBA 28248)
LF = 0x800d0000 + MAENT_SIZE         # letter font, 96 x 100 B
SLOTS = LF + 96 * 100                # 24 slots x 224 B (slot_data())
MAENT_END = SLOTS + 24 * 224         # 0x80107280
VRAM_X, VRAM_Y = 768, 416            # free area 64 hw x 64 rows
SOUND = 0x80182c08                   # play sound a0
GFX, GFX2 = 0x801f6360, 0x801f6370   # libgpu function tables
PAGES = 0x8017ad08                   # u32[3] keyboard page code tables
CODES = 0x8017ad14                   # u16 code list, 0xffff end
CONV = 0x8017b198                    # u16 fighting-mode glyph per list index
NICKBUF = 0x801f1886
PRIM0 = 0x801c8118                   # name prim slots, 0x38 B per cell
DRAW_BACK = 0x801815bc               # SetDrawTPage(s2, 0, 1, v0) onwards
DEFAULT = 0x80180a98                 # copy default name, a0 = field

# Ranges rewritten whole: (start, end, CRC32 of the original bytes).
KEYS = (0x801810f0, 0x801812a4, 0xab302e95)  # delete + input 0x80181194
PAD = (0x8018086c, 0x80180988, 0x4f5986a9)
DEAD = (0x801816f4, 0x80181940, 0xbdd26eff)  # dead keyboard-draw copy
STUB = 0x80181524                    # in the draw loop: lui a1 / addiu a1
# jal sites retargeted: (RAM, old word, entry name in layout())
CALLS = [(0x8017be18, 0x0c060465, 'input'),  # was jal 0x80181194
         (0x8017c000, 0x0c060284, 'nick')]   # was jal 0x80180a10 (left dead)

def pair(a, b=' '): return 0x8000 | ord(a) << 7 | ord(b)

# (file offset, old u16 codes, new u16 codes); 0x3001 follows each.
NAMES = [
    (0x10, (0xde, 0xd2, 0x100), (pair('S', 'e'), pair('i', 'g'), pair('i'))),
    (0x18, (0xef, 0xda, 0xe4), (pair('M', 'a'), pair('k', 'o'), pair('t', 'o'))),
    (0x20, (0x75, 0x60, 0x6a), (pair('M', 'a'), pair('k', 'o'), pair('t', 'o'))),
]

# Init: +8 = 0 and +9 = 2 by one sh, keyboard cursor +c/+d = 10 ("A").
ROMAJI = [
    (0x8017ba5c, 0xa2000009, 0xa6020008),   # sh v0,8(s0)  (v0 = 0x200)
    (0x8017ba64, 0xa200000d, 0x24040a0a),   # addiu a0,zero,0x0a0a
    (0x8017ba68, 0xa200000c, 0xa604000c),   # sh a0,0xc(s0)
    (0x8017bb18, 0xa200000d, 0x24020a0a),   # addiu v0,zero,0x0a0a
    (0x8017bb1c, 0xa200000c, 0xa602000c),   # sh v0,0xc(s0)
    (0x8017bfa4, 0x24020001, 0x24020002),   # addiu v0,zero,2 (nickname page)
    (0x8017bfb8, 0xa200000d, 0x24030a0a),   # addiu v1,zero,0x0a0a
    (0x8017bfbc, 0xa200000c, 0xa603000c),   # sh v1,0xc(s0)
]

class Asm(hook.Asm):
    def lb(self, rt, off, rs): self.i(0x20, rs, rt, off)
    def multu(self, rs, rt): self.r(0, rs, rt, 'zero', 0, 0x19)
    def slt(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x2a)
    def j(self, t): self.emit(0x08000000 | (t >> 2 & 0x3ffffff))
    def jal(self, t): self.emit(0x0c000000 | (t >> 2 & 0x3ffffff))
    def jalr(self, rs): self.r(0, rs, 'zero', 'ra', 0, 0x09)
    def move(self, rd, rs): self.addu(rd, rs, 'zero')
    def b(self, lab): self.branch('beq', 'zero', 'zero', lab)
    def beqz(self, rs, lab): self.branch('beq', rs, 'zero', lab)
    def bnez(self, rs, lab): self.branch('bne', rs, 'zero', lab)

def fit(a, rng):
    code = a.finish()
    assert a.org + len(code) <= rng[1], (hex(a.org), hex(a.org + len(code)))
    return code + bytes(rng[1] - a.org - len(code))

def build_keys():
    """Delete (entry DELETE[0]) and input (entry KEYS_INPUT) share one body.
    a0 = struct. s1 = sound id, then the last cell index."""
    a = Asm(KEYS[0])
    a.b('common')
    a.addiu('a1', 'zero', 0x92)             # delay: delete
    a.addiu('a1', 'zero', 0x91)             # input entry
    a.label('common')
    a.addiu('sp', 'sp', -0x20)
    a.sw('ra', 0x18, 'sp'); a.sw('s1', 0x14, 'sp'); a.sw('s0', 0x10, 'sp')
    a.move('s0', 'a0')
    a.move('s1', 'a1')
    a.jal(SOUND)
    a.move('a0', 'a1')                      # delay
    a.lbu('t0', 0xf, 's0')                  # field
    a.lbu('t1', 0xa, 's0')                  # cell cursor p
    a.sltiu('t2', 't0', 3)
    a.beqz('t2', 'ret')
    a.sll('t3', 't0', 4)                    # delay
    a.addu('t3', 't3', 's0')
    a.sll('t4', 't1', 1)
    a.addu('t4', 't4', 't3')                # cell = 0x12(t4)
    a.lhu('t5', 0x12, 't4')
    a.addiu('t2', 'zero', 0x92)
    a.branch('beq', 's1', 't2', 'delete')
    a.addiu('s1', 'zero', 5)                # delay: last cell
    a.addiu('t2', 'zero', 2)
    a.branch('bne', 't0', 't2', 'f6')
    a.nop()
    a.addiu('s1', 'zero', 3)
    a.label('f6')
    # glyph = *(*(PAGES + 4*page) + 2*key)
    a.lbu('v0', 9, 's0')
    a.lb('v1', 0xc, 's0')
    a.sll('v0', 'v0', 2)
    a.lui('at', (PAGES + 0x8000) >> 16)
    a.addu('at', 'at', 'v0')
    a.lw('a0', PAGES & 0xffff, 'at')
    a.sll('v1', 'v1', 1)
    a.addu('a0', 'a0', 'v1')
    a.lhu('v0', 0, 'a0')                    # v0 = glyph
    a.nop()
    # Latin glyph 1-62 -> ASCII t6
    a.addiu('t6', 'v0', -1)
    a.sltiu('t7', 't6', 62)
    a.beqz('t7', 'plain')
    a.sltiu('t7', 'v0', 37)                 # delay
    a.beqz('t7', 'have')
    a.addiu('t6', 'v0', 60)                 # delay: 37-62 -> 'a'-'z'
    a.sltiu('t7', 'v0', 11)
    a.beqz('t7', 'have')
    a.addiu('t6', 'v0', 54)                 # delay: 11-36 -> 'A'-'Z'
    a.sltiu('t7', 'v0', 10)
    a.bnez('t7', 'have')
    a.addiu('t6', 'v0', 48)                 # delay: 1-9 -> '1'-'9'
    a.addiu('t6', 'zero', 48)               # 10 -> '0'
    a.label('have')
    a.andi('t7', 't5', 0x807f)
    a.xori('t7', 't7', 0x8020)              # 0 if cell = (c1, ' ')
    a.bnez('t7', 'single')
    a.sll('t8', 't6', 7)                    # delay
    a.addu('t5', 't5', 't6')
    a.addiu('t5', 't5', -0x20)              # (c1, c), advance
    a.b('adv')
    a.sh('t5', 0x12, 't4')                  # delay
    a.label('single')
    a.ori('t8', 't8', 0x8020)               # (c, ' '), cursor stays
    a.b('ret')
    a.sh('t8', 0x12, 't4')                  # delay
    a.label('plain')                        # other glyph (kana)
    a.andi('t7', 't5', 0x807f)
    a.xori('t7', 't7', 0x8020)
    a.bnez('t7', 'store')
    a.slt('t8', 't1', 's1')                 # delay
    a.beqz('t8', 'store')                   # single letter here: use the
    a.nop()                                 # next cell, if there is one
    a.addiu('t1', 't1', 1)
    a.addiu('t4', 't4', 2)
    a.label('store')
    a.sh('v0', 0x12, 't4')
    a.label('adv')                          # original advance / field full
    a.slt('t8', 't1', 's1')
    a.beqz('t8', 'full')
    a.addiu('t1', 't1', 1)                  # delay
    a.b('ret')
    a.sb('t1', 0xa, 's0')                   # delay
    a.label('full')
    a.addiu('t8', 'zero', 0x60)             # keyboard cursor -> OK
    a.b('ret')
    a.sb('t8', 0xc, 's0')                   # delay
    a.label('delete')
    a.addiu('t6', 'zero', 0xfff)
    a.branch('beq', 't5', 't6', 'empty')
    a.addiu('t6', 'zero', 0x61e)            # delay
    a.branch('bne', 't5', 't6', 'dhave')
    a.nop()
    a.label('empty')                        # empty cell: step back first
    a.beqz('t1', 'ret')
    a.addiu('t1', 't1', -1)                 # delay
    a.sb('t1', 0xa, 's0')
    a.addiu('t4', 't4', -2)
    a.lhu('t5', 0x12, 't4')
    a.nop()
    a.label('dhave')
    a.andi('t6', 't5', 0x8000)
    a.beqz('t6', 'erase')
    a.andi('t7', 't5', 0x7f)                # delay
    a.addiu('t8', 'zero', 0x20)
    a.branch('beq', 't7', 't8', 'erase')
    a.andi('t5', 't5', 0xff80)              # delay
    a.ori('t5', 't5', 0x20)                 # (c1, c2) -> (c1, ' ')
    a.b('ret')
    a.sh('t5', 0x12, 't4')                  # delay
    a.label('erase')                        # single letter or glyph
    a.addiu('t5', 'zero', 0xfff)
    a.sh('t5', 0x12, 't4')
    a.label('ret')
    a.lw('ra', 0x18, 'sp'); a.lw('s1', 0x14, 'sp'); a.lw('s0', 0x10, 'sp')
    a.jr('ra')
    a.addiu('sp', 'sp', 0x20)
    return fit(a, KEYS)

def build_pad():
    """a0 = struct, a1 = field. Compact, pad with 0x061e, default if empty."""
    a = Asm(PAD[0])
    a.sltiu('t0', 'a1', 3)
    a.beqz('t0', 'ret')
    a.sll('t0', 'a1', 4)                    # delay
    a.addu('a2', 'a0', 't0')
    a.addiu('a2', 'a2', 0x12)               # buffer
    a.move('t1', 'a2')                      # read
    a.move('t2', 'a2')                      # write
    a.addiu('t3', 'zero', 0x3001)
    a.addiu('t4', 'zero', 0xfff)
    a.addiu('t5', 'zero', 0x61e)
    a.label('rd')
    a.lhu('v0', 0, 't1')
    a.addiu('t1', 't1', 2)
    a.branch('beq', 'v0', 't3', 'fill')
    a.nop()
    a.branch('beq', 'v0', 't4', 'rd')
    a.nop()
    a.branch('beq', 'v0', 't5', 'rd')
    a.nop()
    a.sh('v0', 0, 't2')
    a.b('rd')
    a.addiu('t2', 't2', 2)                  # delay
    a.label('fill')
    a.addiu('t1', 't1', -2)                 # at the 0x3001
    a.subu('t6', 't2', 'a2')                # bytes kept
    a.label('fl')
    a.branch('beq', 't2', 't1', 'done')
    a.nop()
    a.sh('t5', 0, 't2')
    a.b('fl')
    a.addiu('t2', 't2', 2)                  # delay
    a.label('done')
    a.bnez('t6', 'ret')
    a.nop()
    a.j(DEFAULT)                            # all blank: default name
    a.move('a0', 'a1')                      # delay
    a.label('ret')
    a.jr('ra')
    a.nop()
    return a

def build_nick(org):
    """a0 = struct. Nickname (+0x32) -> NICKBUF: 4 glyphs + 0x3001."""
    a = Asm(org)
    a.addiu('a0', 'a0', 0x32)
    a.lui('a3', NICKBUF >> 16)
    a.ori('a3', 'a3', NICKBUF & 0xffff)
    a.addiu('t0', 'zero', 4)                # glyphs left
    a.addiu('t9', 'zero', 0x3001)
    a.label('cell')
    a.lhu('t1', 0, 'a0')
    a.addiu('a0', 'a0', 2)
    a.branch('beq', 't1', 't9', 'pad')
    a.andi('t2', 't1', 0x8000)              # delay
    a.bnez('t2', 'pair')
    a.srl('t3', 't1', 7)                    # delay
    a.lui('a1', (CODES + 0x8000) >> 16)     # plain: list index -> CONV
    a.addiu('a1', 'a1', CODES & 0xffff)
    a.move('v0', 'zero')
    a.ori('t3', 'zero', 0xffff)
    a.label('lk')
    a.lhu('v1', 0, 'a1')
    a.addiu('a1', 'a1', 2)
    a.branch('beq', 'v1', 't3', 'got')
    a.nop()
    a.branch('bne', 'v1', 't1', 'lk')
    a.addiu('v0', 'v0', 1)                  # delay
    a.addiu('v0', 'v0', -1)
    a.label('got')
    a.sll('v0', 'v0', 1)
    a.lui('at', (CONV + 0x8000) >> 16)
    a.addu('at', 'at', 'v0')
    a.lhu('v0', CONV & 0xffff, 'at')
    a.addiu('t4', 'zero', 0x20)             # no second letter
    a.label('put')
    a.sh('v0', 0, 'a3')
    a.addiu('t0', 't0', -1)
    a.beqz('t0', 'end')
    a.addiu('a3', 'a3', 2)                  # delay
    a.addiu('t5', 'zero', 0x20)
    a.branch('beq', 't4', 't5', 'cell')
    a.move('t3', 't4')                      # delay: second letter next
    a.b('let')
    a.move('t4', 't5')                      # delay
    a.label('pair')
    a.andi('t3', 't3', 0x7f)                # c1
    a.andi('t4', 't1', 0x7f)                # c2
    a.label('let')                          # ASCII t3 -> shared glyph v0
    a.addiu('v0', 't3', -0x61)
    a.sltiu('at', 'v0', 26)
    a.bnez('at', 'put')
    a.addiu('v0', 'v0', 37)                 # delay: 'a'-'z' 37-62
    a.addiu('v0', 't3', -0x41)
    a.sltiu('at', 'v0', 26)
    a.bnez('at', 'put')
    a.addiu('v0', 'v0', 11)                 # delay: 'A'-'Z' 11-36
    a.addiu('v0', 't3', -0x31)
    a.sltiu('at', 'v0', 9)
    a.bnez('at', 'put')
    a.addiu('v0', 'v0', 1)                  # delay: '1'-'9'
    a.addiu('v0', 't3', -0x30)
    a.sltiu('v0', 'v0', 1)                  # '0' -> 1, other -> 0
    a.sll('at', 'v0', 3)
    a.sll('v0', 'v0', 1)
    a.b('put')
    a.addu('v0', 'v0', 'at')                # delay: '0' -> 10
    a.label('pad')                          # 0 (blank) up to 4 glyphs
    a.sh('zero', 0, 'a3')
    a.addiu('t0', 't0', -1)
    a.bnez('t0', 'pad')
    a.addiu('a3', 'a3', 2)                  # delay
    a.label('end')
    a.jr('ra')
    a.sh('t9', 0, 'a3')                     # delay
    return a

def build_pair(org):
    """Draw-loop pair path. Entry from STUB with v0 = code & 0x8000,
    a3 = code. Sets s2/s1 (prims), composes and uploads the cell, calls
    GetTPage, then s0 = the slot's cell entry and rejoins at DRAW_BACK.
    Slot = SLOTS + 4 * (s5 - PRIM0), 224 B per prim slot (slot_data())."""
    a = Asm(org)
    a.bnez('v0', 'pair')
    a.lui('v1', 0x801f)                     # delay (original insn)
    a.lui('a1', (CODES + 0x8000) >> 16)     # not a pair: original loop
    a.j(STUB + 8)
    a.addiu('a1', 'a1', CODES & 0xffff)     # delay
    a.label('pair')
    a.lw('v1', 0x608c, 'v1')                # double-buffer index
    a.lui('t0', PRIM0 >> 16)
    a.ori('t0', 't0', PRIM0 & 0xffff)
    a.sll('v0', 'v1', 3)
    a.subu('v0', 'v0', 'v1')
    a.sll('v0', 'v0', 2)
    a.addu('s2', 's5', 'v0')                # DR_TPAGE
    a.addiu('s1', 's2', 8)                  # SPRT
    a.subu('t0', 's5', 't0')
    a.sll('t0', 't0', 2)
    a.lui('t4', (SLOTS + 0x8000) >> 16)
    a.addiu('t4', 't4', SLOTS & 0xffff)
    a.addu('s0', 't0', 't4')                # slot: RECT, entry, pixels
    a.srl('t2', 'a3', 7)
    a.andi('t2', 't2', 0x7f)                # c1
    a.andi('t3', 'a3', 0x7f)                # c2
    for r in ('t2', 't3'):                  # *100 = 64 + 32 + 4
        a.sll('t4', r, 6); a.sll('t5', r, 5); a.addu('t4', 't4', 't5')
        a.sll('t5', r, 2); a.addu(r, 't4', 't5')
    lf0 = LF - 0x20 * 100                   # letter c at lf0 + 100c
    a.lui('t4', (lf0 + 0x8000) >> 16)
    a.addiu('t4', 't4', lf0 & 0xffff)
    a.addu('t2', 't2', 't4')                # left letter
    a.addu('t3', 't3', 't4')                # right letter
    a.addiu('t4', 's0', 12)                 # pixels
    a.addiu('t5', 'zero', 20)
    a.label('row')
    for k in range(5):
        a.lbu('t6', k, 't2'); a.lbu('t7', k, 't3')
        a.sb('t6', k, 't4'); a.sb('t7', 5 + k, 't4')
    a.addiu('t2', 't2', 5)
    a.addiu('t3', 't3', 5)
    a.addiu('t5', 't5', -1)
    a.bnez('t5', 'row')
    a.addiu('t4', 't4', 10)                 # delay
    a.lui('v0', GFX >> 16)                  # LoadImage(slot, slot + 12)
    a.lw('v0', GFX & 0xffff, 'v0')
    a.move('a0', 's0')
    a.lw('v0', 0x1c, 'v0')
    a.addiu('a1', 's0', 12)
    a.jalr('v0')
    a.nop()
    a.lui('v0', GFX2 >> 16)                 # GetTPage(0, 0, 768, 256)
    a.lw('v0', GFX2 & 0xffff, 'v0')
    a.move('a0', 'zero')
    a.lw('v0', 0x18, 'v0')
    a.move('a1', 'zero')
    a.addiu('a2', 'zero', VRAM_X)
    a.jalr('v0')
    a.addiu('a3', 'zero', 256)              # delay
    a.j(DRAW_BACK)
    a.addiu('s0', 's0', 8)                  # delay: cell entry
    return a

def slot_data():
    """24 slots of 224 B (prim slot k = 8*field + i): RECT (768 + 5i,
    416 + 20*field, 5, 20), cell entry (tpage, col i, row 8 + field), then
    200 B of pixels written at run time."""
    out = bytearray()
    for k in range(24):
        f, i = divmod(k, 8)
        s = struct.pack('<4H4B', VRAM_X + 5 * i, VRAM_Y + 20 * f, 5, 20,
                        VRAM_X // 64, i, (VRAM_Y - 256) // 20 + f, 0)
        out += s.ljust(224, b'\0')
    return bytes(out)

def stub_words(pair_org):
    # 0x80181524 lui a1,0x8018 / 0x80181528 addiu a1,a1,-0x52ec
    a = Asm(STUB)
    a.j(pair_org)
    a.andi('v0', 'a3', 0x8000)              # delay
    return [0x3c058018, 0x24a5ad14], list(struct.unpack('<2I', a.finish()))

def layout():
    """[(start, end, crc, code)] of the rewritten ranges, and the entries."""
    keys = build_keys()
    padc = build_pad().finish()
    assert PAD[0] + len(padc) <= PAD[1], 'pad too big'
    pairc = build_pair(DEAD[0]).finish()
    nick_org = DEAD[0] + len(pairc)
    nickc = build_nick(nick_org).finish()
    assert nick_org + len(nickc) <= DEAD[1], hex(nick_org + len(nickc))
    def pad_to(c, r): return c + bytes(r[1] - r[0] - len(c))
    rngs = [(*KEYS, keys), (*PAD, pad_to(padc, PAD)),
            (*DEAD, pad_to(pairc + nickc, DEAD))]
    return rngs, {'input': KEYS[0] + 8, 'nick': nick_org}

def patch(data):
    d = bytearray(data)
    assert len(d) == 36796, len(d)
    rngs, entry = layout()
    for lo, hi, crc, code in rngs:
        o = lo - BASE
        cur = zlib.crc32(bytes(d[o:hi - BASE]))
        assert cur == crc, (hex(lo), hex(cur))
        assert len(code) == hi - lo
        d[o:hi - BASE] = code
    old, new = stub_words(DEAD[0])
    o = STUB - BASE
    assert list(struct.unpack_from('<2I', d, o)) == old, hex(STUB)
    struct.pack_into('<2I', d, o, *new)
    for ram, old, name in CALLS:
        o = ram - BASE
        assert struct.unpack_from('<I', d, o)[0] == old, hex(ram)
        struct.pack_into('<I', d, o, 0x0c000000 | entry[name] >> 2 & 0x3ffffff)
    for ram, old, new in ROMAJI:
        o = ram - BASE
        assert struct.unpack_from('<I', d, o)[0] == old, hex(ram)
        struct.pack_into('<I', d, o, new)
    for o, old, new in NAMES:
        assert struct.unpack_from('<4H', d, o) == (*old, 0x3001), hex(o)
        struct.pack_into('<3H', d, o, *new)
    # EN entry form: popup, field and face editor rects (tools/build_entry.py)
    import build_entry
    for o, old, new in build_entry.records():
        assert bytes(d[o:o + len(old)]) == old, hex(o)
        d[o:o + len(new)] = new
    return bytes(d)

def letter_font(maent=None):
    """96 letters (ASCII 0x20-0x7f), 20 rows x 5 B (10 px, 4bpp, low nibble
    first), in CLUT (16,483) indices: 0 clear, 2 darkest ink .. 13 lightest.
    The letters are squeezed from the original hand-drawn alnum keys of the
    MA_ENT sheet (variant e, tools/build_entry.py Hand), plus - ! ? . ' in
    the same hand. maent: unpacked MA_ENT (default: the disc file)."""
    import build_entry
    return build_entry.field_font(maent)

def patch_maent(data, font=None):
    """Unpacked MA_ENT.BIZ -> the EN form and sheet (tools/build_entry.py,
    same length) + letter font + slot data."""
    import build_entry
    assert len(data) == MAENT_SIZE, hex(len(data))
    font = font or letter_font(data)
    assert len(font) == 9600
    out = build_entry.redraw(data) + font + slot_data()
    assert 0x800d0000 + len(out) == MAENT_END
    return out

def main():
    import mips
    orig = open('private/disc/DAT1/MGENT.BIN', 'rb').read()
    new = patch(orig)
    rngs, entry = layout()
    print('; entries', {k: hex(v) for k, v in entry.items()})
    for lo, hi, _, code in rngs:
        n = len(code.rstrip(b'\0')) // 4 + 1
        print(f'; {lo:08x}-{hi:08x}: {n} of {(hi - lo) // 4} words')
        for x, w, t in mips.dis(new, BASE, lo, n): print(f'{x:08x}: {w:08x}  {t}')
    print(f'; stub {STUB:08x}')
    for x, w, t in mips.dis(new, BASE, STUB, 2): print(f'{x:08x}: {w:08x}  {t}')

if __name__ == '__main__':
    main()
