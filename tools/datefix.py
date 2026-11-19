# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Compact EN dates in MGAME (port of the first game's textfix DAYFIX idea).

Text box ({mm}/{dd}, page renderer 0x800b7be4): the digit table
0x800a2dd0 (glyph per digit, read by the texture pass digit helper, a
hooked upload) holds half pairs (digit, 0x7f) from the letter font, and
the three digit column steps advance 1 (10 px) instead of 2. The "/"
between {mm} and {dd} is a half pair too (build_mbms.py pads a letter
before {mm}/{dd} with 0x7f). Result "4/8", "10/25" in 10 px letters.

Backup save list (row 0x800c3624): the six date slots are drawn from the
pre-rendered g5 s2 cells (12 slots: 0-9, "/", blank) at x table
0x800a462c. The table becomes a 10 px pitch and the slot code (39 words
at 0x800c4028, was a jump table) picks the glyph by position:
p0 month tens (skipped if 0), p1 month units, p2 "/", p3 day tens or the
units of a 1-digit day, p4 day units after a tens digit, p5 nothing. So
the day is left aligned, the month right aligned. g5 s2 holds (c, 0x7f)
half pairs (tl), g2 s20 CLEAR! too, since it shares the x table.

patch_mgame(data) runs after namefix.patch() and asserts every old word.
RE: private/re/datefix/findings.org.
"""
import os, struct, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import OVL_BASE
from namefix import asm

HALF = 0x807f
DIGITS = 0x800a2dd0                  # u16 x 10: glyph of 0..9 (0 -> 0x0a)
DATE_X = 0x800a462c                  # s16 x 6: slot x offsets (also CLEAR!)
DATE_X_OLD = (180, 200, 220, 240, 260, 280)
DATE_X_NEW = (180, 190, 200, 210, 220, 230)
# digit column steps of the page renderer, as namefix leaves them
COLS = [0x800b7f68, 0x800b8268, 0x800b8378]
COL_OLD, COL_NEW = 0x24420002, 0x24420001       # addiu v0,v0,2 -> 1
SLOT, SLOT_END = 0x800c4028, 0x800c40c4          # date slot code
SLOT_CRC = 0x03cde592                            # disc words (jump table cases)
OUT, SKIP = 0x800c40c4, 0x800c414c               # draw slot s4 / next slot

# s0 = position 0..5, s6 = save record (+0x55 month, +0x56 day),
# s7 = 0xcccccccd. Sets s4 = glyph slot and goes to OUT, or to SKIP.
# t0, t1 are unused in this routine. a1 must survive.
SLOT_SRC = '''
    sltiu v0, s0, 5
    beq v0, zero, SKIP          # p5: nothing
    sltiu v0, s0, 2
    bne v0, zero, div           # p0, p1: month
    lbu a2, 0x55(s6)
    addiu v1, zero, 2
    beq s0, v1, OUT             # p2: "/"
    addiu s4, zero, 10
    lbu a2, 0x56(s6)            # p3, p4: day
div:
    nop
    multu a2, s7
    mfhi t0
    srl t0, t0, 3               # tens
    sll v0, t0, 2
    addu v0, v0, t0
    sll v0, v0, 1
    subu t1, a2, v0             # units
    addiu v1, zero, 1
    beq s0, v1, OUT             # p1: month units
    move s4, t1
    addiu v1, zero, 4
    bne s0, v1, p03
    nop
    bne t0, zero, OUT           # p4: day units after a tens digit
    nop
    b SKIP
    nop
p03:
    bne t0, zero, OUT           # p0, p3: tens digit if there is one
    move s4, t0
    bne s0, zero, OUT           # p3: else the units of a 1-digit day
    move s4, t1
    b SKIP                      # p0: 1-digit month, nothing
    nop
'''


def slot_code():
    code = asm(SLOT_SRC, SLOT, {'OUT': OUT, 'SKIP': SKIP})
    n = (SLOT_END - SLOT) // 4
    assert len(code) <= n, (len(code), n)
    return code + [0] * (n - len(code))


def half(c): return HALF | ord(c) << 7


def patch_mgame(data):
    d = bytearray(data)
    o = DIGITS - OVL_BASE
    assert struct.unpack_from('<10H', d, o) == (10, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    struct.pack_into('<10H', d, o, *[half(str(k)) for k in range(10)])
    for a in COLS:
        assert struct.unpack_from('<I', d, a - OVL_BASE)[0] == COL_OLD, hex(a)
        struct.pack_into('<I', d, a - OVL_BASE, COL_NEW)
    o = DATE_X - OVL_BASE
    assert struct.unpack_from('<6h', d, o) == DATE_X_OLD
    struct.pack_into('<6h', d, o, *DATE_X_NEW)
    o, n = SLOT - OVL_BASE, (SLOT_END - SLOT) // 4
    crc = zlib.crc32(bytes(d[o:o + 4 * n]))
    assert crc == SLOT_CRC, hex(crc)
    struct.pack_into('<%dI' % n, d, o, *slot_code())
    return bytes(d)


if __name__ == '__main__':
    import mips
    src = sys.argv[1] if len(sys.argv) > 1 else 'private/disc/DAT1/MGAME.BIN'
    code = asm(SLOT_SRC, SLOT, {'OUT': OUT, 'SKIP': SKIP})
    blob = struct.pack('<%dI' % len(code), *code)
    for a, w, t in mips.dis(blob, SLOT, SLOT, len(code)):
        print(f'{a:08x}: {w:08x}  {t}')
    print(len(code), 'of', (SLOT_END - SLOT) // 4, 'words')
