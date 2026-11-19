# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Parse the exe's CD file table: 12-byte entries (BCD MSF u32, size u32, param u32)."""
import struct, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from game import EXE, TAB, NTAB
def bcd(b): return (b >> 4) * 10 + (b & 15)
def load_lba(path='private/lba.txt'):
    m = {}
    for l in open(path):
        if l[0] != ' ': continue
        lba, sz, name = l.split()[:3]
        m[int(lba)] = (int(sz), name.split(';')[0])
    return m
def entries(exe, lbamap):
    """Scan for runs of entries that resolve to real files."""
    out = []
    for off in range(0x800, len(exe) - 12, 4):
        m0, s, f, fr = exe[off:off+4]
        if s > 0x99 or fr > 0x99 or f != 0 and False: pass
        try:
            lba = (bcd(m0) * 60 + bcd(s)) * 75 + bcd(f) - 150
        except Exception: continue
        size, param = struct.unpack_from('<II', exe, off + 4)
        if lba in lbamap and lbamap[lba][0] == size:
            out.append((off, lba, size, param, lbamap[lba][1]))
    return out
if __name__ == '__main__':
    exe = open(EXE, 'rb').read()
    for off, lba, size, param, name in entries(exe, load_lba()):
        print(f'{off:#07x} idx? lba={lba:6d} size={size:9d} param={param:#010x} {name}')
