# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""List TIMs at any 4-byte offset in a file, with CLUT and image rects.
Usage: timscan.py [-o PREFIX] [-c CLUTIDX] FILE...
-o renders each TIM to PREFIX<file>_<n>.png with tim2png (CLUT row CLUTIDX).
Merged from private/re/{boot,fest,menus}/timscan.py."""
import sys, struct, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def scan(d):
    """-> [(offset, bpp code, clut rect or None, image rect, end offset)]"""
    out = []
    for p in range(0, len(d) - 20, 4):
        m, fl = struct.unpack_from('<II', d, p)
        if m != 0x10 or fl & ~0xb or (fl & 3) == 3: continue
        q = p + 8; clut = None
        try:
            if fl & 8:
                bl, x, y, w, h = struct.unpack_from('<IHHHH', d, q)
                if bl != 12 + w*h*2 or w == 0 or h == 0: continue
                clut = (x, y, w, h); q += bl
            bl, x, y, w, h = struct.unpack_from('<IHHHH', d, q)
            if bl != 12 + w*h*2 or w == 0 or h == 0 or x >= 1024 or y >= 512: continue
        except struct.error: continue
        out.append((p, fl & 3, clut, (x, y, w, h), q + bl))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', dest='prefix')
    ap.add_argument('-c', dest='clut', type=int, default=0)
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    for f in a.files:
        d = open(f, 'rb').read()
        for n, (p, bpp, c, r, e) in enumerate(scan(d)):
            print(f, n, hex(p), 'bpp', [4, 8, 16, 24][bpp], 'clut', c, 'img', r, 'end', hex(e))
            if a.prefix:
                from tim2png import tim
                out = f'{a.prefix}{os.path.basename(f).split(".")[0]}_{n:02d}.png'
                try: tim(d[p:e], out, 1, a.clut)
                except Exception as ex: print('  render error', ex)
