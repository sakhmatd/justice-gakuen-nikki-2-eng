# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""MGAME text layout in 10 px columns (half cells after names and half pairs).

Letter pairs (0x8000 | c1<<7 | c2) put two 10 px letters in one 20 px cell.
An odd-length text part ends in a half cell. MGAME lays out each text line
with a column counter on the stack and x = tab[box type].x + col*20. This
patch counts col in 10 px units instead: every col*20 becomes col*10 and
every col += 1 becomes col += 2. A half cell adds 1:
- a plain glyph (c, 0x7f) in the page renderer and the choice line
  (build_mbms.py pads the last pair before an insert or tag with 0x7f);
- in the page name helper, a cell (c, 0x7f), or a cell (c, ' ') that is
  followed by a non-pair (0x3001 end, 0x061e padding) or by a pair that
  starts with '-' (a -kun/-san/-chan suffix in the built buffer).
Full 20 px cells (all JP glyphs, digits, 0x061e) still advance 20 px, so JP
text lays out exactly as before. The texture pass 0x800af578 is unchanged:
it allocates tiles by sprite index.

Covered (RE: private/re/hook/findings.org section b, private/re/namefix/):
- page renderer 0x800b7be4 (col at 0x18(sp)): loop x, digit x and col
  sites, 0x061e, plain glyph path (half test in five load-delay nops);
- page name helper 0x800b7a54 (all inserts 0x2000-0x2009): rewritten;
- choice line 0x800bc8f8 (col at 0x10(sp)): loop x, digit sites, 0x061e,
  the shared col tail and the plain glyph path (half test in nops and in
  two redundant andi, t1/t2 are free there);
- choice name helper 0x800bc788: col units only (choice strings hold no
  name inserts).
Not touched: name object 0x800b5d78, save row 0x800c44ac, debug viewer
0x800b3224, the digit table 0x800a2dd0 (digits stay full cells).

patch(data) takes the original MGAME.BIN image and returns it patched. It
asserts every old word (the helper range by CRC32) before writing.
Usage: namefix.py [MGAME.BIN] -> prints old/new disassembly, writes nothing.
"""
import os, struct, sys, zlib
sys.path.insert(0, os.path.dirname(__file__))
from game import OVL_BASE                     # MGAME base 0x800a2000

R = {n: i for i, n in enumerate(
    'zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 '
    's0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra'.split())}
RTYPE = {'addu': 0x21, 'subu': 0x23, 'and': 0x24, 'or': 0x25, 'xor': 0x26,
         'sltu': 0x2b, 'slt': 0x2a}
SHIFT = {'sll': 0x00, 'srl': 0x02, 'sra': 0x03}
ITYPE = {'addiu': 0x09, 'slti': 0x0a, 'sltiu': 0x0b, 'andi': 0x0c,
         'ori': 0x0d, 'xori': 0x0e}
MEM = {'lb': 0x20, 'lh': 0x21, 'lw': 0x23, 'lbu': 0x24, 'lhu': 0x25,
       'sb': 0x28, 'sh': 0x29, 'sw': 0x2b}

def asm(src, org, ext=None):
    """Assemble MIPS I text (one insn or 'label:' per line, '#' comments).
    ext: extra labels (name -> address) for branches out of the block."""
    lines = []
    labels = dict(ext or {})
    for ln in src.splitlines():
        ln = ln.split('#')[0].strip()
        if not ln: continue
        if ln.endswith(':'):
            labels[ln[:-1]] = org + 4 * len(lines); continue
        op, _, rest = ln.partition(' ')
        lines.append((op, [x.strip() for x in rest.split(',') if x.strip()]))
    out = []
    for k, (op, a) in enumerate(lines):
        pc = org + 4 * k
        num = lambda s: int(s, 0)
        if op == 'nop': w = 0
        elif op == 'move': w = R[a[1]] << 21 | R[a[0]] << 11 | 0x21
        elif op in RTYPE: w = R[a[1]] << 21 | R[a[2]] << 16 | R[a[0]] << 11 | RTYPE[op]
        elif op in SHIFT: w = R[a[1]] << 16 | R[a[0]] << 11 | num(a[2]) << 6 | SHIFT[op]
        elif op in ITYPE: w = ITYPE[op] << 26 | R[a[1]] << 21 | R[a[0]] << 16 | num(a[2]) & 0xffff
        elif op == 'lui': w = 0x0f << 26 | R[a[0]] << 16 | num(a[1]) & 0xffff
        elif op in MEM:
            off, base = a[1].rstrip(')').split('(')
            w = MEM[op] << 26 | R[base] << 21 | R[a[0]] << 16 | num(off) & 0xffff
        elif op in ('beq', 'bne', 'b'):
            if op == 'b': op, a = 'beq', ['zero', 'zero'] + a
            d = (labels[a[2]] - pc - 4) >> 2
            assert -0x8000 <= d < 0x8000
            w = (4 if op == 'beq' else 5) << 26 | R[a[0]] << 21 | R[a[1]] << 16 | d & 0xffff
        elif op == 'j':
            t = labels[a[0]] if a[0] in labels else num(a[0])
            assert t >> 28 == pc >> 28
            w = 2 << 26 | t >> 2 & 0x3ffffff
        elif op == 'jr': w = R[a[0]] << 21 | 0x08
        elif op == 'multu': w = R[a[0]] << 21 | R[a[1]] << 16 | 0x19
        elif op == 'mfhi': w = R[a[0]] << 11 | 0x10
        else: raise ValueError(ln)
        out.append(w)
    return out

# Single words: (RAM, old word, new insn, note).
WORDS = [
    # page renderer 0x800b7be4
    (0x800b7c74, 0x00023880, 'sll a3, v0, 1', 'page: x = tab.x + col*10'),
    (0x800b7f90, 0x00031880, 'sll v1, v1, 1', 'page: units digit x (200a/b/c)'),
    (0x800b8290, 0x00031880, 'sll v1, v1, 1', 'page: units digit x (200d)'),
    (0x800b7d6c, 0x24820001, 'addiu v0, a0, 2', 'page: 0x061e col += 2'),
    (0x800b7f68, 0x24420001, 'addiu v0, v0, 2', 'page: tens digit col += 2'),
    (0x800b8268, 0x24420001, 'addiu v0, v0, 2', 'page: tens digit col += 2 (200d)'),
    (0x800b8378, 0x24420001, 'addiu v0, v0, 2', 'page: units digit col += 2'),
    (0x800b83c4, 0x00000000, 'lhu t1, 0(s1)', 'page plain: t1 = code'),
    (0x800b83fc, 0x00000000, 'andi t1, t1, 0x807f', 'page plain:'),
    (0x800b8424, 0x00000000, 'xori t1, t1, 0x807f', 'page plain: 0 = half pair'),
    (0x800b8430, 0x00000000, 'sltu t1, zero, t1', 'page plain:'),
    (0x800b8438, 0x00000000, 'addiu t1, t1, 1', 'page plain: t1 = 1 half, 2 full'),
    (0x800b8488, 0x24420001, 'addu v0, v0, t1', 'page plain: col += t1'),
    # choice line 0x800bc8f8
    (0x800bc980, 0x00023880, 'sll a3, v0, 1', 'choice: x = tab.x + col*10'),
    (0x800bcc14, 0x00031880, 'sll v1, v1, 1', 'choice: units digit x'),
    (0x800bcdfc, 0x00031880, 'sll v1, v1, 1', 'choice: units digit x'),
    (0x800bcfe4, 0x00031880, 'sll v1, v1, 1', 'choice: units digit x'),
    (0x800bd1cc, 0x00031880, 'sll v1, v1, 1', 'choice: units digit x'),
    (0x800bca70, 0x24820001, 'addiu v0, a0, 2', 'choice: 0x061e col += 2'),
    (0x800bcbec, 0x24420001, 'addiu v0, v0, 2', 'choice: tens digit col += 2'),
    (0x800bcdd4, 0x24420001, 'addiu v0, v0, 2', 'choice: tens digit col += 2'),
    (0x800bcfbc, 0x24420001, 'addiu v0, v0, 2', 'choice: tens digit col += 2'),
    (0x800bd1a4, 0x24420001, 'addiu v0, v0, 2', 'choice: tens digit col += 2'),
    (0x800bd360, 0x24420001, 'addiu v0, v0, 2', 'choice: tail col += 2 (units, plain)'),
    # choice plain glyph 0x800bd2c4: v1 keeps the sprite index, the u
    # remainder moves to t2, the freed words test for a half pair and
    # store col - 1 before the shared tail adds 2.
    (0x800bd2c8, 0x00000000, 'lhu t1, 0(s1)', 'choice plain: t1 = code'),
    (0x800bd2e4, 0x00621823, 'subu t2, v1, v0', 'choice plain: t2 = idx % 12'),
    (0x800bd2e8, 0x306300ff, 'andi t1, t1, 0x807f', 'choice plain: (was andi v1, 0xff)'),
    (0x800bd2ec, 0x00031080, 'sll v0, t2, 2', 'choice plain: u = t2*20'),
    (0x800bd2f0, 0x00431021, 'addu v0, v0, t2', 'choice plain:'),
    (0x800bd2fc, 0x92620558, 'xori t1, t1, 0x807f', 'choice plain: (was lbu v0, idx)'),
    (0x800bd300, 0x00000000, 'sltiu t1, t1, 1', 'choice plain: t1 = 1 half pair'),
    (0x800bd304, 0x00520019, 'multu v1, s2', 'choice plain: idx / 12 (v1 = idx)'),
    (0x800bd30c, 0x00000000, 'lw t2, 0x10(sp)', 'choice plain: col'),
    (0x800bd324, 0x308400ff, 'subu t2, t2, t1', 'choice plain: (was andi a0, 0xff)'),
    (0x800bd340, 0x00000000, 'sw t2, 0x10(sp)', 'choice plain: col -= half'),
    # choice name helper 0x800bc788
    (0x800bc7e4, 0x00021080, 'sll v0, v0, 1', 'choice helper: x = tab.x + (col+t0)*10'),
    (0x800bc820, 0x25080001, 'addiu t0, t0, 2', 'choice helper: blank t0 += 2'),
    (0x800bc8d4, 0x24420001, 'addiu v0, v0, 2', 'choice helper: cell col += 2'),
]

# Page name helper 0x800b7a54-0x800b7be4 (a0 window, a1 sprite, a2 code
# buffer, a3 line, 0x10(sp) = &col; returns the next sprite). Same output
# as the original for full cells; the loop invariants are hoisted.
HELPER = 0x800b7a54
HELPER_END = 0x800b7be4
HELPER_SRC = '''
    move  t2, a0
    move  t4, a1
    lw    t3, 0x10(sp)          # &col
    lhu   v0, 0(a2)
    lw    t1, 0x2c(t2)          # text object
    addiu v1, zero, 0x3001
    beq   v0, v1, ret
    move  t0, zero              # pending blank columns
    lbu   a0, 3(t2)             # box type
    lui   at, 0x800a
    sll   v0, a0, 2
    addu  at, at, v0
    lhu   t5, 0x3ccc(at)        # tab.x
    lhu   t7, 0x3cce(at)        # tab.y
    sll   v0, a3, 3
    subu  v0, v0, a3
    sll   v0, v0, 2
    addu  t7, t7, v0            # y = tab.y + line*28
    addiu t6, zero, 4           # CLUT row of box type 5
    addiu v0, zero, 5
    beq   a0, v0, row
    lui   v0, 0x801f
    lbu   v0, 0x18f7(v0)
    lui   at, 0x800a
    addu  at, at, v0
    lbu   t6, 0x3d00(at)        # CLUT row of the other types
row:
    lui   a3, 0xaaaa
    ori   a3, a3, 0xaaab        # /12
loop:
    lw    a0, 0(t3)             # col
    lhu   v0, 0(a2)             # code
    addu  v1, a0, t0
    sll   at, v1, 2
    addu  v1, at, v1
    sll   v1, v1, 1             # (col + pending) * 10
    addu  v1, v1, t5
    sh    v1, 0x38(t4)          # x
    addiu at, zero, 0x61e
    bne   v0, at, cell
    sh    t7, 0x3a(t4)          # y
    b     next
    addiu t0, t0, 2             # blank: 20 px, dropped at the end
cell:
    addu  a0, a0, t0
    move  t0, zero
    lhu   t9, 2(a2)             # next code
    andi  t8, v0, 0x807f
    xori  at, t8, 0x807f
    sltiu at, at, 1             # at = (c, 0x7f)
    xori  t8, t8, 0x8020
    sltiu t8, t8, 1             # t8 = (c, ' ')
    andi  v1, t9, 0xff80
    xori  v1, v1, 0x9680
    sltiu v1, v1, 1             # v1 = next is ('-', c)
    srl   t9, t9, 15
    xori  t9, t9, 1             # t9 = next is not a pair
    or    t9, t9, v1
    and   t9, t9, t8
    or    t9, t9, at            # t9 = 1: half cell
    addiu a0, a0, 2
    subu  a0, a0, t9
    sw    a0, 0(t3)             # col += 2 or 1
    lbu   v1, 0x1548(t1)        # sprite index
    nop
    multu v1, a3
    mfhi  t8
    srl   t8, t8, 3             # idx / 12
    sll   v0, t8, 1
    addu  v0, v0, t8
    sll   v0, v0, 2
    subu  v0, v1, v0            # idx % 12
    sll   at, v0, 2
    addu  v0, at, v0
    sll   v0, v0, 2
    sh    v0, 0x3c(t4)          # u = (idx % 12) * 20
    sll   at, t8, 2
    addu  at, at, t8
    sll   at, at, 2
    sh    at, 0x3e(t4)          # v = (idx / 12) * 20
    sb    zero, 0x40(t4)
    sb    t6, 0x41(t4)
    addiu v1, v1, 1
    sb    v1, 0x1548(t1)
    addiu t4, t4, 0x44
next:
    addiu a2, a2, 2
    lhu   v0, 0(a2)
    addiu v1, zero, 0x3001
    bne   v0, v1, loop
    nop
ret:
    jr    ra
    move  v0, t4
'''
HELPER_CRC = 0x348c292a          # zlib.crc32 of the original 100 words

def helper_code():
    code = asm(HELPER_SRC, HELPER)
    n = (HELPER_END - HELPER) // 4
    assert len(code) <= n, (len(code), n)
    return code + [0] * (n - len(code))

def changes():
    """[(ram, [old words or None], [new words], note)]"""
    out = [(ram, [old], asm(src, ram), note) for ram, old, src, note in WORDS]
    out.append((HELPER, None, helper_code(), 'page name helper (rewritten)'))
    return sorted(out)

def patch(data):
    d = bytearray(data)
    for ram, old, new, note in changes():
        o = ram - OVL_BASE
        n = len(new)
        cur = list(struct.unpack_from('<%dI' % n, d, o))
        if old is None:
            crc = zlib.crc32(bytes(d[o:o + 4 * n]))
            assert crc == HELPER_CRC, (hex(ram), hex(crc))
        else:
            assert cur == old, (hex(ram), [hex(x) for x in cur], note)
        struct.pack_into('<%dI' % n, d, o, *new)
    return bytes(d)

def main():
    import mips
    path = sys.argv[1] if len(sys.argv) > 1 else 'private/disc/DAT1/MGAME.BIN'
    orig = open(path, 'rb').read()
    new = patch(orig)
    used = len(asm(HELPER_SRC, HELPER))
    for ram, old, words, note in changes():
        n = len(words)
        a = list(mips.dis(orig, OVL_BASE, ram, n))
        b = list(mips.dis(new, OVL_BASE, ram, n))
        if n > 1:
            print(f'; {ram:08x}-{ram + 4*n:08x} {note}: {used} of {n} words')
        for (pa, wo, to), (_, wn, tn) in zip(a, b):
            if n > 1: print(f'{pa:08x} {wo:08x} {to:28s} | {wn:08x} {tn}')
            else: print(f'{pa:08x} {wo:08x} {to:24s} -> {wn:08x} {tn:24s} {note}')
    nw = sum(1 for i in range(0, len(orig), 4) if orig[i:i+4] != new[i:i+4])
    print(f'; {nw} words changed')

if __name__ == '__main__':
    main()
