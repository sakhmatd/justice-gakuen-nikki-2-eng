# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Capstone helpers for the PSX exe and DAT1 overlays."""
import capstone, struct, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from game import EXE, ovl_base
md = capstone.Cs(capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS32 | capstone.CS_MODE_LITTLE_ENDIAN)
def load_exe(path=EXE):
    d = open(path, 'rb').read()
    pc0, gp, taddr, tsize = struct.unpack_from('<IIII', d, 0x10)
    return taddr, d[0x800:0x800 + tsize]
def dis(code, base, start, count=None, end=None):
    off = start - base
    blob = code[off:(end - base) if end else off + 4 * (count or 64)]
    for i in range(0, len(blob) - 3, 4):
        insn = next(md.disasm(blob[i:i+4], start + i), None)
        w = struct.unpack_from('<I', blob, i)[0]
        yield start + i, w, (f'{insn.mnemonic} {insn.op_str}' if insn else f'.word {w:#010x}')
def show(code, base, start, count=64):
    for a, w, t in dis(code, base, start, count): print(f'{a:08x}: {w:08x}  {t}')
def refs(code, base, target):
    """Find lui/addiu or lui/ori/lw pairs that build `target` (simple, same-register, within 8 insns)."""
    hi = (target + 0x8000) >> 16 if True else 0
    lo = target & 0xffff
    out = []
    ws = struct.unpack('<%dI' % (len(code)//4), code[:len(code)//4*4])
    for i, w in enumerate(ws):
        if w >> 26 == 0x0f and (w & 0xffff) in ((target >> 16), hi):
            rt = (w >> 16) & 31
            for j in range(i+1, min(i+12, len(ws))):
                v = ws[j]; op = v >> 26; rs = (v >> 21) & 31
                if rs == rt and op in (0x09, 0x0d, 0x23, 0x21, 0x25, 0x20, 0x24, 0x2b, 0x29, 0x28):
                    imm = v & 0xffff
                    full = ((w & 0xffff) << 16) + (imm - 0x10000 if imm & 0x8000 and op != 0x0d else imm)
                    if full & 0xffffffff == target: out.append(base + 4*i); break
    return out
def calls(code, base, target):
    jal = 0x0c000000 | ((target >> 2) & 0x3ffffff)
    ws = struct.unpack('<%dI' % (len(code)//4), code[:len(code)//4*4])
    return [base + 4*i for i, w in enumerate(ws) if w == jal]
def funcs(code, base):
    """Function starts: addiu $sp,$sp,-N."""
    ws = struct.unpack('<%dI' % (len(code)//4), code[:len(code)//4*4])
    return [base + 4*i for i, w in enumerate(ws) if (w & 0xffff8000) == 0x27bd8000]
def func_of(fs, addr):
    import bisect
    i = bisect.bisect_right(fs, addr) - 1
    return fs[i] if i >= 0 else None
def guess_base(code, lo=0x80080000, hi=0x80200000, step=0x10):
    """Pick load base maximizing internal jal targets that hit function prologues."""
    ws = struct.unpack('<%dI' % (len(code)//4), code[:len(code)//4*4])
    starts = {4*i for i, w in enumerate(ws) if (w & 0xffff8000) == 0x27bd8000}
    tg = [((w & 0x3ffffff) << 2) | 0x80000000 for w in ws if w >> 26 == 3]
    import collections
    c = collections.Counter()
    for t in tg:
        for s in starts:
            b = t - s
            if lo <= b < hi and b % step == 0: c[b] += 1
    return c.most_common(3)
