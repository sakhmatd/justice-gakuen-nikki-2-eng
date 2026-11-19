# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Replace the Japanese "ただいまロード中..." strip on the picture loading screens.

DAT2/ALOAD32.BIZ is an archive of 50 LZ entries, one loading picture each
(exe 0x8001495c picks one at random, file index 0x3b). Every entry holds a
4bpp strip TIM at 0x3c800: 6 CLUTs at VRAM (512,481), pixels at (768,0),
header 256x256 but the entry ends at 0x3e000 (46 rows; the lettering uses
rows 0-31). Header and CLUTs are the same in all 50 entries; six strip
designs exist. Design c26041 is Japanese (entries in JP). The builder copies
the Latin strip of another entry over it (0x3c800 to 0x3e000). SRC holds
the two italic "NOW LOADING..." designs (closest to the italic JP strip),
used in turn. The script design (entry 0) is hard to read on busy art, and
the entry 7 design has descenders below row 45, past the end of a
0x3e000-byte entry.

Writes private/build/ALOAD32.BIZ only. Run inside
private/build/.rebuild.lock, before hook.py.
"""
import sys, os, struct, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import lz, arc

NAME = 'ALOAD32.BIZ'
SRCF = 'private/disc/DAT2/' + NAME
OUT = 'private/build/' + NAME
JP = [2, 6, 12, 16, 25, 35, 42, 45]
JP_HASH = 'c26041'                 # md5 of the JP strip pixels, first 6 digits
SRC = [1, 3]                       # italic Latin strips (ink in rows 0-31)
A, B = 0x3c800, 0x3e000            # TIM header + CLUTs (0xe0 B), then pixels

def pixhash(d):
    return hashlib.md5(d[A + 0xe0:B]).hexdigest()[:6]

def build():
    orig = open(SRCF, 'rb').read()
    toc = arc.toc(orig)
    ent = lambda i: lz.decompress(orig[toc[i][0]:toc[i][0] + toc[i][1]])[0]
    # Every entry with the JP strip must be in JP, and no other.
    found = [i for i in range(len(toc)) if pixhash(ent(i)) == JP_HASH]
    assert found == JP, f'JP strip in {found}, expected {JP}'
    srcs = {i: ent(i) for i in SRC}
    blobs = [orig[o:o + s] for o, s in toc]
    for k, i in enumerate(JP):
        d = bytearray(ent(i)); en = srcs[SRC[k % len(SRC)]]
        assert len(d) == len(en) == B, (i, len(d))
        assert d[A:A + 0xe0] == en[A:A + 0xe0], f'entry {i}: TIM header/CLUT differ'
        d[A:B] = en[A:B]
        c = lz.compress(bytes(d), name='ALOAD32.BIZ')
        assert lz.decompress(c)[0] == bytes(d)
        blobs[i] = c
    # Keep every entry at its offset if it fits its slot, else move the rest.
    slots = [(toc[i + 1][0] if i + 1 < len(toc) else len(orig)) - o
             for i, (o, s) in enumerate(toc)]
    over = [i for i in JP if len(blobs[i]) > slots[i]]
    if not over:
        out = bytearray(orig)
        for i in JP:
            o = toc[i][0]
            out[o:o + slots[i]] = blobs[i] + bytes(slots[i] - len(blobs[i]))
            struct.pack_into('<II', out, 8 * i, o, len(blobs[i]))
    else:
        alloc = (len(orig) + 0x7ff) // 0x800 * 0x800
        out = bytearray(orig[:0x800])
        for i, b in enumerate(blobs):
            struct.pack_into('<II', out, 8 * i, len(out), len(b))
            out += b + bytes(-len(b) % 0x800)
        if len(out) > alloc:
            raise SystemExit(f'{NAME}: {len(out)} > disc allocation {alloc}')
        out += bytes(max(0, len(orig) - len(out)))
    open(OUT, 'wb').write(out)
    room = min(slots[i] - len(blobs[i]) for i in JP)
    print(f'ALOAD32: {len(JP)} strips replaced, entries moved: {over or "none"}, '
          f'least slot room {room} B')

if __name__ == '__main__':
    build()
