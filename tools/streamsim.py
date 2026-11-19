# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Simulate the game's streaming LZ load (exe 0x80015c0c / 0x80015dbc).
Usage: streamsim.py FILE.BIZ EXPECTED.lz"""
import sys, struct
RING, RING_END = 0x801f9000, 0x801fd000
def run(comp):
    mem = bytearray(0x10000)                     # covers 0x801f9000..0x801fffff
    M = lambda a: a - RING
    out = bytearray()
    st = {'e0': RING, 'e4': RING, 'ec': len(comp), 'f0': 0, 'done': False}
    def rd16(a): return mem[M(a)] | mem[M(a)+1] << 8
    def dec(a0, a1):
        if ((a1 + 0x7fe07000) & 0x7800) == 0x4000:
            mem[M(RING_END):M(RING_END)+0x40] = mem[0:0x40]
        while True:
            flags = rd16(a0); a0 += 2
            for _ in range(16):
                if not flags & 0x8000:
                    out.extend(mem[M(a0):M(a0)+2]); a0 += 2
                else:
                    v = rd16(a0); a0 += 2
                    ln, d = v >> 11, v & 0x7ff
                    if ln == 0:
                        ln = rd16(a0); a0 += 2
                    if d == 0 and ln == 0:
                        st['done'] = True; return
                    p = len(out) - d*2
                    for _ in range(ln): out.extend(out[p:p+2]); p += 2
                flags = (flags << 1) & 0xffff
            if not a0 < a1: break
        st['e4'] = a0 - 0x4000 if ((a0 + 0x7fe07000) & 0x7800) == 0x4000 else a0
    pos = 0
    while st['ec'] > 0:
        # a sector arrives at e0
        sec = comp[pos:pos+0x800].ljust(0x800, b'\0'); pos += 0x800
        mem[M(st['e0']):M(st['e0'])+0x800] = sec
        if st['ec'] <= 0x800:
            st['f0'] += st['ec']
            if st['f0'] > 0x800:
                dec(st['e4'], (st['e4'] & ~0x7ff) + 0x800); st['f0'] -= 0x800
            st['e0'] = RING if ((st['e0'] + 0x7fe07000) & 0x7800) == 0x3800 else st['e0'] + 0x800
            if not st['done']: dec(st['e4'], (st['e4'] & ~0x7ff) + st['f0'])
            st['ec'] = 0
        else:
            st['f0'] += 0x800
            if st['f0'] >= 0x1000:
                dec(st['e4'], (st['e4'] & ~0x7ff) + 0x800); st['f0'] -= 0x800
            st['e0'] = RING if ((st['e0'] + 0x7fe07000) & 0x7800) == 0x3800 else st['e0'] + 0x800
            st['ec'] -= 0x800
    return bytes(out), st['done']
if __name__ == '__main__':
    comp = open(sys.argv[1], 'rb').read(); exp = open(sys.argv[2], 'rb').read()
    out, done = run(comp)
    n = next((i for i in range(min(len(out), len(exp))) if out[i] != exp[i]), None)
    print(f'{sys.argv[1]}: end marker {"seen" if done else "NOT seen"}, out {len(out)} / {len(exp)}, first diff {n}')
