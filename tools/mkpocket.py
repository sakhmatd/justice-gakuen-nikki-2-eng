# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Write a memory card image that holds the PocketStation app (Poke J Mail).

Usage: python3 tools/mkpocket.py [--image BIN | --app FILE] [--unlock] OUT

The app is DAT2/EMTBL.BIZ on the disc, an LZ file that decodes to a PS1
card save holding the PocketStation program. In the game, Extras >
PocketStation > Download writes it to the card the same way.

  --image BIN  take the app from a disc image (MODE2/2352 .bin), e.g. the
               patched "Nekketsu Seishun Nikki 2 (English).bin"
  --app FILE   take a decoded app file (starts with "SC"); default
               private/build/EMTBL.BIZ.lz from the last build
  --unlock     mark all 20 characters as unlocked; the game sets one
               character for each diary ending seen
OUT ending in .gme gets a DexDrive header (MAME pockstat, DexDrive tools).
Any other name (.mcd, .mcr) gets a raw 128 KB card image for emulators and
card tools. The save is named BISLPSP021201, as the game writes it.
"""
import argparse, os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from lz import decompress

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILT = os.path.join(ROOT, 'private/build/EMTBL.BIZ.lz')   # decoded
NAME = b'BISLPSP021201'      # KOUBAI2 0x800b71f0; 'P' marks a PocketStation file
FR, BLK = 0x80, 0x2000


def iso_file(path, want):
    """Bytes of the file at ISO path want ('DAT2/EMTBL.BIZ') in a 2352 image."""
    f = open(path, 'rb')

    def sector(lba):
        f.seek(lba * 2352 + 24)
        return f.read(2048)

    def records(lba, size):
        data = b''.join(sector(lba + i) for i in range((size + 2047) // 2048))
        o = 0
        while o < len(data):
            n = data[o]
            if n == 0:
                o = (o // 2048 + 1) * 2048
                continue
            ln = data[o + 32]
            name = data[o + 33:o + 33 + ln].decode('ascii', 'replace').split(';')[0]
            yield name, struct.unpack_from('<I', data, o + 2)[0], struct.unpack_from('<I', data, o + 10)[0]
            o += n

    pvd = sector(16)
    if pvd[1:6] != b'CD001':
        sys.exit(f'{path}: no ISO 9660 volume (expected a raw MODE2/2352 .bin)')
    lba, size = struct.unpack_from('<I', pvd, 156 + 2)[0], struct.unpack_from('<I', pvd, 156 + 10)[0]
    for part in want.split('/'):
        for name, l, s in records(lba, size):
            if name.upper() == part.upper():
                lba, size = l, s
                break
        else:
            sys.exit(f'{path}: {want} not found')
    return b''.join(sector(lba + i) for i in range((size + 2047) // 2048))[:size]


def frame(b):
    b = bytearray(b.ljust(FR, b'\0'))
    c = 0
    for x in b[:0x7f]:
        c ^= x
    b[0x7f] = c
    return bytes(b)


def card(app, name):
    """128 KB card: header frame, 15 directory frames, no broken sectors,
    the app as one save of app[3] blocks from block 1."""
    nb = app[3]
    assert app[:2] == b'SC' and 1 <= nb <= 15 and len(app) <= nb * BLK, (app[:4], len(app))
    app = app.ljust(nb * BLK, b'\0')
    c = bytearray(0x20000)
    c[0:FR] = frame(b'MC')
    for i in range(15):
        if i < nb:
            st = 0x51 if i == 0 else (0x53 if i == nb - 1 else 0x52)
            size = nb * BLK if i == 0 else 0
            nxt = i + 1 if i < nb - 1 else 0xffff
            e = struct.pack('<IIH', st, size, nxt) + (name if i == 0 else b'').ljust(21, b'\0')
        else:
            e = struct.pack('<IIH', 0xa0, 0, 0xffff)
        c[(1 + i) * FR:(2 + i) * FR] = frame(e)
    for i in range(20):
        c[(16 + i) * FR:(17 + i) * FR] = frame(struct.pack('<I', 0xffffffff) + b'\0' * 4 + b'\xff\xff')
    for i in range(36, 63):
        c[i * FR:(i + 1) * FR] = b'\xff' * FR
    c[63 * FR:64 * FR] = frame(b'MC')
    c[BLK:BLK + nb * BLK] = app
    return bytes(c)


def gme(c):
    h = bytearray(0xf40)
    h[0:11] = b'123-456-STD'
    h[0x12] = 1; h[0x14] = 1; h[0x15] = ord('M')
    for i in range(15):
        h[0x16 + i] = c[(1 + i) * FR]
        h[0x26 + i] = c[(1 + i) * FR + 8]
    return bytes(h) + c


def main():
    ap = argparse.ArgumentParser(description='Write a PocketStation app card image.')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--image', help='disc image (MODE2/2352 .bin)')
    src.add_argument('--app', default=BUILT, help='decoded app file')
    ap.add_argument('--unlock', action='store_true', help='unlock all 20 characters')
    ap.add_argument('out')
    a = ap.parse_args()
    if a.image:
        app, _ = decompress(iso_file(a.image, 'DAT2/EMTBL.BIZ'))
        what = a.image
    else:
        app, what = open(a.app, 'rb').read(), a.app
    app = bytearray(app)
    if a.unlock:
        for i in range(20):                  # character records at 0x380 + 16 i
            app[0x380 + 16 * i] = 1
        app[0x4d0] = 0                       # selected character
    c = card(bytes(app), NAME)
    out = gme(c) if a.out.lower().endswith('.gme') else c
    open(a.out, 'wb').write(out)
    print(f'{a.out}: app from {what}, {app[3]} blocks, {len(out)} bytes')


if __name__ == '__main__':
    main()
