# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Runtime letter-pair glyphs (port of the first game's hook, core only).

Text codes 0x8000 | c1<<7 | c2 (c1, c2 ASCII 0x20-0x7f) are drawn as one
20x20 cell with two 10x20 letters. MGAME uploads each message-file glyph
with LoadImage through v0 (jalr v0) after src = 0x800d0180 + code*200, at
four sites (private/re/hook/findings.org, section a). Each jalr becomes
jal HOOK; the delay slots stay. The hook passes normal codes through. For
pair codes it copies the two letters from the letter font (in the message
file) into a ring buffer and passes that instead.

HOOK lives in unused padding of the exe (0x80010ab0..0x80010d28). Letter
font, ring index and ring live in the free font area of each rebuilt
message file (LF_OFF etc. in game.py; build_mbms.py writes the letters).
MGAME is loaded from DAT1/MGAME.BIN (file idx 0x2bd, uncompressed). No
other overlay uploads message-file glyphs (findings section c). The CLUT
upload at 0x800af5e0 is not hooked.

MGAME half cells: tools/namefix.py. Dates: tools/datefix.py. Sprite
immediates: every tools/mgbg_*.py with MGAME_PATCH(bytes) -> bytes (e.g.
mgbg_examhud.py), applied after namefix and datefix. Name entry: tools/mgentfix.py.
Not ported yet: the first game's up_tables.

Usage: hook.py   -> writes patched SLPS_021.20, MGAME.BIN, MGENT.BIN and
                    MA_ENT.BIZ to private/build/
"""
import struct, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from game import (EXE_NAME, HOOK, HOOK_END, MSG_LOAD, FONTPIX, LF_OFF,
                  STATE_OFF, RING_OFF, RING_N, OVL_BASE)

LF, STATE, RING = MSG_LOAD + LF_OFF, MSG_LOAD + STATE_OFF, MSG_LOAD + RING_OFF

SITES = {
    # main upload (texture pass 0x800af578), name copy, digit helper,
    # single-glyph routine 0x800af4b0 (no callers found)
    'MGAME.BIN': (OVL_BASE, [0x800af924, 0x800afa2c, 0x800afb1c, 0x800af50c]),
}
JALR_V0 = 0x0040f809

R = {n: i for i, n in enumerate('zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 '
                                's0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra'.split())}

class Asm:
    def __init__(self, org):
        self.org = org; self.code = []; self.labels = {}; self.fix = []
    def pc(self): return self.org + 4 * len(self.code)
    def label(self, n): self.labels[n] = self.pc()
    def emit(self, w): self.code.append(w & 0xffffffff)
    def r(self, op, rs, rt, rd, sh, fn):
        self.emit(op << 26 | R[rs] << 21 | R[rt] << 16 | R[rd] << 11 | sh << 6 | fn)
    def i(self, op, rs, rt, imm): self.emit(op << 26 | R[rs] << 21 | R[rt] << 16 | (imm & 0xffff))
    def nop(self): self.emit(0)
    def lui(self, rt, imm): self.i(0x0f, 'zero', rt, imm)
    def ori(self, rt, rs, imm): self.i(0x0d, rs, rt, imm)
    def addiu(self, rt, rs, imm): self.i(0x09, rs, rt, imm)
    def andi(self, rt, rs, imm): self.i(0x0c, rs, rt, imm)
    def lbu(self, rt, off, rs): self.i(0x24, rs, rt, off)
    def lhu(self, rt, off, rs): self.i(0x25, rs, rt, off)
    def sh(self, rt, off, rs): self.i(0x29, rs, rt, off)
    def xori(self, rt, rs, imm): self.i(0x0e, rs, rt, imm)
    def slti(self, rt, rs, imm): self.i(0x0a, rs, rt, imm)
    def sltiu(self, rt, rs, imm): self.i(0x0b, rs, rt, imm)
    def sra(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x03)
    def lw(self, rt, off, rs): self.i(0x23, rs, rt, off)
    def sb(self, rt, off, rs): self.i(0x28, rs, rt, off)
    def sw(self, rt, off, rs): self.i(0x2b, rs, rt, off)
    def addu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x21)
    def subu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x23)
    def and_(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x24)
    def or_(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x25)
    def xor(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x26)
    def sltu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x2b)
    def sll(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x00)
    def srl(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x02)
    def jr(self, rs): self.r(0, rs, 'zero', 'zero', 0, 0x08)
    def divu(self, rs, rt): self.r(0, rs, rt, 'zero', 0, 0x1b)
    def mflo(self, rd): self.r(0, 'zero', 'zero', rd, 0, 0x12)
    def li32(self, rt, v):
        self.lui(rt, v >> 16); self.ori(rt, rt, v & 0xffff)
    def branch(self, kind, rs, rt, lab):
        op, rtn = {'beq': (4, rt), 'bne': (5, rt), 'bgez': (1, 'at')}[kind]
        if kind == 'bgez': rtn = 'at'                         # REGIMM rt=1 -> bgez
        self.fix.append((len(self.code), lab)); self.i(op, rs, rtn, 0)
    def finish(self):
        for idx, lab in self.fix:
            off = (self.labels[lab] - (self.org + 4*idx + 4)) >> 2
            self.code[idx] = (self.code[idx] & 0xffff0000) | (off & 0xffff)
        return b''.join(struct.pack('<I', w) for w in self.code)

def mul_const(a, dst, src, k, tmp):
    """dst = src * k using shifts (k = 100 or 200). dst != src."""
    bits = [b for b in range(16) if k >> b & 1]
    a.sll(dst, src, bits[0])
    for b in bits[1:]:
        a.sll(tmp, src, b); a.addu(dst, dst, tmp)

def build_hook():
    a = Asm(HOOK)
    a.li32('t0', FONTPIX)
    a.subu('t1', 'a1', 't0')            # t1 = code*200 for MBMS font sources
    a.lui('t2', 0x0064)                 # 0x8000*200
    a.subu('t3', 't1', 't2')            # pair codes: 0 <= t3 < 0x8000*200 (unsigned)
    a.sltu('t3', 't3', 't2')
    a.branch('beq', 't3', 'zero', 'out')
    a.addiu('t3', 'zero', 200)          # delay
    a.divu('t1', 't3')
    a.mflo('t1')                        # code
    a.andi('t2', 't1', 0x7f)            # c2
    a.srl('t1', 't1', 7)
    a.andi('t1', 't1', 0x7f)            # c1
    a.addiu('t1', 't1', -0x20)
    a.addiu('t2', 't2', -0x20)
    mul_const(a, 't4', 't1', 100, 't5'); mul_const(a, 't6', 't2', 100, 't5')
    a.li32('t5', LF)
    a.addu('t1', 't4', 't5')            # left letter
    a.addu('t2', 't6', 't5')            # right letter
    a.li32('t6', STATE)
    a.lw('t7', 0, 't6')
    a.nop()
    a.addiu('t8', 't7', 1)
    a.andi('t8', 't8', RING_N - 1)
    a.sw('t8', 0, 't6')
    a.andi('t7', 't7', RING_N - 1)
    mul_const(a, 't9', 't7', 200, 't8')
    a.li32('t8', RING)
    a.addu('a1', 't8', 't9')            # new source
    a.addu('t9', 'a1', 'zero')          # dst cursor
    a.addiu('t3', 'zero', 20)           # rows
    a.label('row')
    for k in range(5):
        a.lbu('t4', k, 't1'); a.lbu('t5', k, 't2'); a.nop()
        a.sb('t4', k, 't9'); a.sb('t5', 5 + k, 't9')
    a.addiu('t1', 't1', 5)
    a.addiu('t2', 't2', 5)
    a.addiu('t3', 't3', -1)
    a.branch('bne', 't3', 'zero', 'row')
    a.addiu('t9', 't9', 10)             # delay
    a.label('out')
    a.jr('v0')
    a.nop()
    code = a.finish()
    assert HOOK + len(code) <= HOOK_END, hex(HOOK + len(code))
    return code

def patch_all(disc='private/disc', out='private/build'):
    os.makedirs(out, exist_ok=True)
    code = build_hook()
    exe = bytearray(open(os.path.join(disc, EXE_NAME), 'rb').read())
    off = 0x800 + HOOK - 0x80010000
    assert not any(exe[off - 12:HOOK_END - 0x80010000 + 0x800]), 'hook area not empty'
    exe[off:off + len(code)] = code
    open(os.path.join(out, EXE_NAME), 'wb').write(exe)
    jal = 0x0c000000 | (HOOK >> 2 & 0x3ffffff)
    for name, (base, sites) in SITES.items():
        d = bytearray(open(os.path.join(disc, 'DAT1', name), 'rb').read())
        for s in sites:
            o = s - base
            assert struct.unpack_from('<I', d, o)[0] == JALR_V0, (name, hex(s))
            struct.pack_into('<I', d, o, jal)
        if name == 'MGAME.BIN':                     # half-cell name inserts
            import namefix
            d = bytearray(namefix.patch(bytes(d)))
            import datefix                          # 10 px date digits, save list
            d = bytearray(datefix.patch_mgame(bytes(d)))
            # sprite immediates of graphics modules (tools/mgbg_*.py)
            import build_mgbg
            for m in build_mgbg.modules():
                if hasattr(m, 'MGAME_PATCH'):
                    d = bytearray(m.MGAME_PATCH(bytes(d)))
            import mgbg_map                         # location map label records
            d = bytearray(mgbg_map.patch_mgame(bytes(d)))
        open(os.path.join(out, name), 'wb').write(d)
    # Name entry: romaji start, pair packing and drawing (tools/mgentfix.py).
    # MA_ENT.BIZ carries the letter font and slot data for it.
    import mgentfix, lz
    d = open(os.path.join(disc, 'DAT1', 'MGENT.BIN'), 'rb').read()
    open(os.path.join(out, 'MGENT.BIN'), 'wb').write(mgentfix.patch(d))
    d = lz.decompress(open(os.path.join(disc, 'DAT2', 'MA_ENT.BIZ'), 'rb').read())[0]
    biz = lz.compress(mgentfix.patch_maent(d), name='MA_ENT.BIZ')
    assert len(biz) <= mgentfix.MAENT_ALLOC, len(biz)
    open(os.path.join(out, 'MA_ENT.BIZ'), 'wb').write(biz)
    print(f'hook {len(code)} bytes at {HOOK:#x}..{HOOK + len(code):#x}; '
          f'patched', ', '.join(SITES), '+ MGENT.BIN, MA_ENT.BIZ')

if __name__ == '__main__':
    patch_all()
