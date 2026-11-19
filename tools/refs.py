# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Find loads/stores of given absolute addresses (lui + offset) in the exe and all DAT1 overlays.
Usage: refs.py ADDR... (hex)
Copied from private/re/fight/refs.py."""
import sys, struct, glob, os
sys.path.insert(0, 'tools')
from mips import load_exe, md
from game import ovl_base
tg = [int(a, 16) for a in sys.argv[1:]]
bins = [('EXE', *load_exe())]
for p in sorted(glob.glob('private/disc/DAT1/*.BIN')):
    d = open(p, 'rb').read()
    base = ovl_base(p)
    bins.append((os.path.basename(p), base, d))
for name, base, code in bins:
    ws = struct.unpack('<%dI' % (len(code)//4), code[:len(code)//4*4])
    for i, w in enumerate(ws):
        op = w >> 26
        if op not in (0x20, 0x24, 0x28, 0x21, 0x25, 0x29, 0x23, 0x2b): continue
        rs = (w >> 21) & 31; imm = w & 0xffff; simm = imm - 0x10000 if imm & 0x8000 else imm
        for j in range(i - 1, max(i - 10, -1), -1):
            v = ws[j]
            if v >> 26 == 0x0f and (v >> 16) & 31 == rs:
                full = ((v & 0xffff) << 16) + simm & 0xffffffff
                if full in tg:
                    a = base + 4*i
                    insn = next(md.disasm(code[4*i:4*i+4], a))
                    print(f'{name:12s} {a:08x} file {4*i:#07x} {insn.mnemonic} {insn.op_str}  [{full:08x}]')
                break
