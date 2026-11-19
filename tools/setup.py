# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Extract the game data for building.

Usage: python3 tools/setup.py JP.bin USA.bin

JP.bin   Shiritsu Justice Gakuen - Nekketsu Seishun Nikki 2 (Japan),
         SLPS-02120, one MODE2/2352 track (raw .bin; an .ecm image must be
         decoded first, e.g. with ecm2bin).
USA.bin  Rival Schools - United by Fate (USA) (Disc 1) (Arcade Disc),
         SLUS-00681. build_fight.py and build_selp.py take English art
         from it.

Writes (all gitignored, skipped when present):
  private/disc/      the ISO file system
  private/lba.txt    directory listing with LBAs
  private/unpacked/  DAT2 files after decompression
  private/script/    the dumped message text
  private/usa/disc/, private/usa/unpacked/  the same for the USA disc
"""
import os, sys, subprocess, zlib
sys.path.insert(0, os.path.dirname(__file__))
from lz import decompress
from arc import toc
from game import ORIG, DISC_SIZE, DISC_CRC, EXE_NAME, USA_SIZE, USA_CRC, USA_DIR
PY = [sys.executable, '-I']


def check(path, size=DISC_SIZE, crc=DISC_CRC):
    if os.path.getsize(path) != size:
        sys.exit(f'{path}: size {os.path.getsize(path)}, expected {size}')
    c = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            c = zlib.crc32(b, c)
    if c & 0xffffffff != crc:
        sys.exit(f'{path}: CRC32 {c & 0xffffffff:08x}, expected {crc:08x}')


def try_lz(d):
    try:
        out, end = decompress(d)
    except Exception:
        return None
    return out if len(d) - 0x800 <= end <= len(d) and len(out) >= len(d) else None


def unpack(src, dst):
    """Same layout as tools/unpack_all.py."""
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        d = open(os.path.join(src, fn), 'rb').read()
        t = toc(d)
        if t:
            os.makedirs(os.path.join(dst, fn), exist_ok=True)
            for i, (o, s) in enumerate(t):
                e = d[o:o + s]
                out = try_lz(e)
                tag = 'lz' if out is not None else 'raw'
                open(os.path.join(dst, fn, f'{i:03d}.{tag}'), 'wb').write(
                    out if out is not None else e)
        else:
            out = try_lz(d)
            tag = 'lz' if out is not None else 'raw'
            open(os.path.join(dst, f'{fn}.{tag}'), 'wb').write(out if out is not None else d)


def step(done, msg, fn):
    if os.path.exists(done):
        print(f'{msg}: {done} exists, skipped')
        return
    print(msg + '...')
    fn()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    disc, usa = (os.path.abspath(p) for p in sys.argv[1:])
    check(disc)
    check(usa, USA_SIZE, USA_CRC)
    if not os.path.exists(ORIG):
        os.makedirs(os.path.dirname(ORIG), exist_ok=True)
        os.symlink(disc, ORIG)
        print('linked', ORIG)
    tools = os.path.dirname(os.path.abspath(__file__))
    run = lambda *a, **kw: subprocess.run(PY + [os.path.join(tools, a[0])] + list(a[1:]),
                                          check=True, **kw)
    step('private/disc/' + EXE_NAME, 'Extracting the disc',
         lambda: run('extract.py', disc, 'private/disc'))

    def lba():
        with open('private/lba.txt', 'w') as o:
            run('isols.py', disc, stdout=o)
    step('private/lba.txt', 'Listing the disc', lba)
    step('private/unpacked', 'Unpacking DAT2', lambda: unpack('private/disc/DAT2', 'private/unpacked'))
    step('private/script', 'Dumping the message text', lambda: run('dump.py'))
    step(USA_DIR + '/disc/SLUS_006.81', 'Extracting the USA disc',
         lambda: run('extract.py', usa, USA_DIR + '/disc'))
    step(USA_DIR + '/unpacked', 'Unpacking USA DAT2',
         lambda: unpack(USA_DIR + '/disc/DAT2', USA_DIR + '/unpacked'))
    print('Done. Build with: sh tools/build_all.sh')


if __name__ == '__main__':
    main()
