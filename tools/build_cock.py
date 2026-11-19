# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Build DAT2/MGCOCK.TIZ (diary hub sheet) and patch its tables in MGAME.

Run after tools/hook.py: hook.py writes private/build/MGAME.BIN fresh from
the disc and this script edits that file in place. It refuses to run twice
on the same MGAME.BIN (marker private/build/.cock_mgame holds the hash of
the file it wrote); rerun hook.py first.

The original sheet comes from private/unpacked/MGCOCK.TIZ.lz (one 4bpp TIM,
1024x256 texels at VRAM 512,256; topic private/re/cock/). Every module
tools/cock_*.py is imported in name order and its apply(tim, mgame) called:
tim.px is a writable (256, 1024) uint8 index array (sheet x = (tpage -
512) * 4 + u), tim.clut a (96, 16) uint16 array of the palettes, mgame a
bytearray of MGAME.BIN (file offset = RAM - 0x800a2000). A module that
defines SPACE (list of sheet rectangles x, y, w, h) is checked: it must not
change texels outside them.

Writes private/build/MGCOCK.TIZ (compressed, must fit the disc slot),
MGCOCK.TIZ.lz (decompressed) and MGAME.BIN. --dry writes nothing.
Module names as arguments (e.g. cock_hud) run only those (for tests).
"""
import os, sys, glob, struct, hashlib, importlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lz

SRC = 'private/unpacked/MGCOCK.TIZ.lz'
DISC = 'private/disc/DAT2/MGCOCK.TIZ'
OUT = 'private/build/MGCOCK.TIZ'
MGAME = 'private/build/MGAME.BIN'
MARK = 'private/build/.cock_mgame'
PIX, CLUT = 0xc20, 20                 # pixel data, CLUT data (512x3 u16)


class Tim:
    def __init__(self, data):
        self.data = bytearray(data)
        assert struct.unpack_from('<2I', self.data, 0) == (0x10, 8), 'not a 4bpp TIM'
        assert struct.unpack_from('<4H', self.data, PIX - 8) == (512, 256, 256, 256)
        raw = np.frombuffer(bytes(self.data[PIX:PIX + 512 * 256]), np.uint8).reshape(256, 512)
        self.px = np.empty((256, 1024), np.uint8)
        self.px[:, 0::2] = raw & 15
        self.px[:, 1::2] = raw >> 4
        self.clut = np.array(struct.unpack_from('<1536H', self.data, CLUT), np.uint16).reshape(96, 16)

    def pack(self):
        p = self.px.astype(np.uint8)
        assert p.max() < 16
        self.data[PIX:PIX + 512 * 256] = (p[:, 0::2] | (p[:, 1::2] << 4)).tobytes()
        self.data[CLUT:CLUT + 3072] = self.clut.astype('<u2').tobytes()
        return bytes(self.data)


def modules(only=None):
    here = os.path.dirname(os.path.abspath(__file__))
    names = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(here, 'cock_*.py')))
    if only: names = [n for n in names if n in only]
    return [importlib.import_module(n) for n in names]


def check_space(name, before, after, space):
    ok = np.zeros(before.shape, bool)
    for x, y, w, h in space: ok[y:y + h, x:x + w] = True
    bad = (before != after) & ~ok
    if bad.any():
        ys, xs = np.where(bad)
        sys.exit(f'{name}: {bad.sum()} texels changed outside its SPACE, '
                 f'first at sheet ({xs[0]}, {ys[0]})')


def build(dry=False, only=None):
    mg = bytearray(open(MGAME, 'rb').read())
    h = hashlib.sha256(mg).hexdigest()
    if os.path.exists(MARK) and open(MARK).read().strip() == h:
        if not dry: sys.exit('build_cock: MGAME.BIN was already patched; run hook.py first')
        print('build_cock: MGAME.BIN already patched; dry run on the disc MGAME.BIN')
        mg = bytearray(open('private/disc/DAT1/MGAME.BIN', 'rb').read())
    tim = Tim(open(SRC, 'rb').read())
    for m in modules(only):
        before = tim.px.copy()
        m.apply(tim, mg)
        if hasattr(m, 'SPACE'): check_space(m.__name__, before, tim.px, m.SPACE)
    data = tim.pack()
    c = lz.compress(data, name='MGCOCK.TIZ')
    assert lz.decompress(c)[0] == data
    slot = os.path.getsize(DISC); slot += -slot % 2048
    print(f'MGCOCK.TIZ: {len(c)} bytes (slot {slot})')
    if len(c) > slot: sys.exit('build_cock: MGCOCK.TIZ does not fit its slot')
    if dry: return
    open(OUT + '.lz', 'wb').write(data)
    open(OUT, 'wb').write(c)
    open(MGAME, 'wb').write(mg)
    open(MARK, 'w').write(hashlib.sha256(mg).hexdigest() + '\n')


if __name__ == '__main__':
    only = [a for a in sys.argv[1:] if not a.startswith('-')]
    build('--dry' in sys.argv, only)
