# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Scan a RAM dump for GPU packets (OT entries): SPRT, SPRT_8/16 and
POLY_FT4 with their tpage.
Usage: prims.py RAM.bin [FILTER]
Prints addr, kind, screen rect, tpage, uv, wh, clut. FILTER is a Python
expression over a k x y w h u v clut tp tpx, e.g. "clut>>6 >= 500".
Copied from private/re/exam/prims.py."""
import sys, struct
def scan(r):
    out = []; n = len(r) // 4
    W = struct.unpack('<%dI' % n, r[:n*4])
    for i in range(n - 2):
        tag = W[i]; ln = tag >> 24; nxt = tag & 0xffffff
        if ln == 0 or ln > 16 or not (nxt == 0xffffff or (nxt < 0x200000 and nxt % 4 == 0)): continue
        j = i + 1; end = i + 1 + ln; tp = None; ok = False; items = []
        while j < end and j < n:
            w = W[j]; c = w >> 24
            if c == 0xe1: tp = w & 0x1ff; j += 1; continue
            if c in (0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0): j += 1; continue
            if 0x64 <= c <= 0x67 and j + 3 < end + 1:
                x, y = struct.unpack('<hh', struct.pack('<I', W[j+1]))
                u, v, clut = W[j+2] & 0xff, W[j+2] >> 8 & 0xff, W[j+2] >> 16
                w_, h_ = struct.unpack('<hh', struct.pack('<I', W[j+3]))
                items.append(('SPRT', x, y, w_, h_, u, v, clut, tp, w & 0xffffff)); j += 4; continue
            if c in (0x74, 0x75, 0x76, 0x77, 0x7c, 0x7d, 0x7e, 0x7f):
                x, y = struct.unpack('<hh', struct.pack('<I', W[j+1]))
                u, v, clut = W[j+2] & 0xff, W[j+2] >> 8 & 0xff, W[j+2] >> 16
                s = 8 if c < 0x7c else 16
                items.append(('SPRT%d' % s, x, y, s, s, u, v, clut, tp, w & 0xffffff)); j += 3; continue
            if 0x2c <= c <= 0x2f and j + 8 < end + 1:
                xs = []; us = []
                for k in range(4):
                    x, y = struct.unpack('<hh', struct.pack('<I', W[j+1+2*k])); xs.append((x, y))
                    us.append((W[j+2+2*k] & 0xff, W[j+2+2*k] >> 8 & 0xff))
                clut = W[j+2] >> 16; tpg = W[j+4] >> 16 & 0x1ff
                x0 = min(p[0] for p in xs); y0 = min(p[1] for p in xs)
                x1 = max(p[0] for p in xs); y1 = max(p[1] for p in xs)
                u0 = min(p[0] for p in us); v0 = min(p[1] for p in us)
                u1 = max(p[0] for p in us); v1 = max(p[1] for p in us)
                items.append(('FT4', x0, y0, x1 - x0, y1 - y0, u0, v0, clut, tpg, (u1 - u0, v1 - v0))); j += 9; continue
            break
        else:
            ok = True
        if ok and items:
            for it in items: out.append((0x80000000 + 4*i,) + it)
    return out
def fmt(e):
    a, k, x, y, w, h, u, v, clut, tp, extra = e
    tpx = (tp & 15) * 64 if tp is not None else -1; tpy = (tp >> 4 & 1) * 256 if tp is not None else -1
    bpp = (tp >> 7 & 3) if tp is not None else -1
    return f'{a:08x} {k:6s} xy {x:4d},{y:4d} wh {w:3d}x{h:<3d} tp {tpx:4d},{tpy:3d} b{bpp} uv {u:3d},{v:3d} clut {(clut & 63) * 16:4d},{clut >> 6:3d} {extra if k == "FT4" else ""}'
if __name__ == '__main__':
    r = open(sys.argv[1], 'rb').read()
    flt = sys.argv[2] if len(sys.argv) > 2 else None
    seen = set()
    for e in scan(r):
        key = e[1:]
        if key in seen: continue
        seen.add(key)
        a, k, x, y, w, h, u, v, clut, tp, extra = e
        tpx = (tp & 15) * 64 if tp is not None else -1
        if flt and not eval(flt): continue
        print(fmt(e))
