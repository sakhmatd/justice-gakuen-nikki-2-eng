# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Make the release patches from the original dump and the built image.

Usage: python3 tools/make_release.py VERSION [PATCHED.bin]

PATCHED.bin defaults to private/out/nikki2-en.bin. The original is the
Redump dump at game.ORIG (linked by setup.py). Writes a .bps and a .ppf
patch and checks both against PATCHED.bin on a temporary copy. Each patch
then goes into its own directory private/dist/nikki2-en-VERSION-FMT/ with
apply_patch.py, README.txt, LICENSE and LICENSE-TEXT, zipped to
private/dist/nikki2-en-VERSION-FMT.zip (FMT bps or ppf). The patches are
large (~190 MB), so users download only the format they need.
Run from the project root. Needs numpy.
"""
import hashlib, os, shutil, struct, sys, zipfile, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from game import ORIG, OUT, DISC_SIZE, DISC_CRC
import apply_patch

CHUNK = 1 << 26


def diff_runs(a_path, b_path, gap):
    """(start, end) byte ranges where the files differ; gaps < gap merge."""
    idx = []
    with open(a_path, 'rb') as a, open(b_path, 'rb') as b:
        base = 0
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if not x:
                break
            d = np.nonzero(np.frombuffer(x, np.uint8) != np.frombuffer(y, np.uint8))[0]
            idx.append(d + base)
            base += len(x)
    idx = np.concatenate(idx)
    if not len(idx):
        return []
    br = np.nonzero(np.diff(idx) > gap)[0]
    starts = idx[np.r_[0, br + 1]]
    ends = idx[np.r_[br, len(idx) - 1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def varint(n):
    out = bytearray()
    while True:
        x = n & 0x7f
        n >>= 7
        if n == 0:
            out.append(0x80 | x)
            return bytes(out)
        out.append(x)
        n -= 1


def write_bps(src, dst, path):
    size = os.path.getsize(src)
    assert os.path.getsize(dst) == size, 'the patcher keeps the image size'
    p = bytearray(b'BPS1' + varint(size) + varint(size) + varint(0))
    pos = 0
    with open(dst, 'rb') as t:
        for s, e in diff_runs(src, dst, 8):
            if s > pos:
                p += varint((s - pos - 1) << 2 | 0)          # SourceRead
            t.seek(s)
            p += varint((e - s - 1) << 2 | 1) + t.read(e - s)  # TargetRead
            pos = e
    if pos < size:
        p += varint((size - pos - 1) << 2 | 0)
    p += struct.pack('<II', apply_patch.crc32_file(src), apply_patch.crc32_file(dst))
    p += struct.pack('<I', zlib.crc32(p) & 0xffffffff)
    open(path, 'wb').write(p)


def write_ppf(src, dst, path, desc):
    p = bytearray(b'PPF30\x02' + desc.encode('ascii')[:50].ljust(50, b' '))
    p += bytes([0, 1, 0, 0])                 # BIN image, block check, no undo
    with open(src, 'rb') as f:
        f.seek(0x9320)
        p += f.read(1024)
    with open(dst, 'rb') as t:
        for s, e in diff_runs(src, dst, 9):
            t.seek(s)
            data = t.read(e - s)
            for k in range(0, len(data), 255):
                c = data[k:k + 255]
                p += struct.pack('<QB', s + k, len(c)) + c
    open(path, 'wb').write(p)


README = """\
Shiritsu Justice Gakuen: Nekketsu Seishun Nikki 2 - English patch {ver}
========================================================================

This patch translates the Japanese "Shiritsu Justice Gakuen: Nekketsu
Seishun Nikki 2" (SLPS-02120) into English: the diary mode and the other
modes that had no Western release. You need your own dump of that disc.
The patch does not contain the game.

By Sergei Akhmatdinov (sakhmatd). Copyright 2026.
apply_patch.py: Apache License 2.0 (LICENSE).
Translation text: Creative Commons BY-NC-SA 4.0 (LICENSE-TEXT).
The game and its Japanese text belong to Capcom. This is an unofficial
fan project, not affiliated with Capcom.

The patch expects the Redump dump (one MODE2/2352 track),
"Shiritsu Justice Gakuen - Nekketsu Seishun Nikki 2 (Japan).bin":
  file size {size} bytes
  CRC32     {crc:08x}
  MD5       {md5}
  SHA-1     {sha}

Patched image:
  CRC32     {pcrc:08x}
  SHA-1     {psha}

Applying the patch
------------------
Patch file: {patch}
  for {tools}.
The same patch is also released as a .{other} file
  for {othertools}.

With Python 3 (any system):
  python3 apply_patch.py {patch} "ORIGINAL.bin"

This writes "Nekketsu Seishun Nikki 2 (English).bin" and a .cue next to
the original and leaves the original unchanged. Open the .cue in your
emulator or burn it.

If the check fails, the dump differs from the Redump one (another dump,
or a file converted to .ecm, .chd or .iso). Make a raw 2352-byte
dump, or convert yours back to .bin/.cue, and try again. An older dump
of this disc with the same size (CRC32 865b1432) is damaged and does not
work.

PocketStation
-------------
The PocketStation game Pocket Justice is translated as well. In the
patched game, Extras > PocketStation > Download writes it to a memory
card (11 free blocks). tools/mkpocket.py in the source repository writes
a memory card image (.gme or .mcd) with it from the patched .bin.
MAME's pockstat driver has a bug that loses most button presses in this
game (also in the Japanese original). tools/mame_pockstat.lua in the
source repository works around it; see README.md there.
{changes}"""

TOOLS = {'bps': 'Flips, Rom Patcher JS (in a browser) or apply_patch.py',
         'ppf': 'PPF-O-Matic, apply_patch.py, or emulators that load PPF'}

# Release notes per version, appended to README.txt.
CHANGES = {}


def digest(path, algo='sha1'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip())
        return 2
    ver = sys.argv[1]
    patched = sys.argv[2] if len(sys.argv) == 3 else OUT + '.bin'
    name = f'nikki2-en-{ver}'
    assert os.path.getsize(ORIG) == DISC_SIZE and apply_patch.crc32_file(ORIG) == DISC_CRC, \
        f'{ORIG} is not the Redump dump (run tools/setup.py)'
    dist = os.path.join('private', 'dist')
    os.makedirs(dist, exist_ok=True)
    paths = {'bps': os.path.join(dist, f'{name}.bps'), 'ppf': os.path.join(dist, f'{name}.ppf')}
    print('BPS...')
    write_bps(ORIG, patched, paths['bps'])
    print('PPF...')
    write_ppf(ORIG, patched, paths['ppf'], f'Nekketsu Seishun Nikki 2 English {ver}')
    want = apply_patch.crc32_file(patched)
    tmp = os.path.join(dist, '.verify.bin')
    try:
        for fmt, fn in (('bps', apply_patch.apply_bps), ('ppf', apply_patch.apply_ppf)):
            fn(open(paths[fmt], 'rb').read(), ORIG, tmp)
            got = apply_patch.crc32_file(tmp)
            assert got == want, f'{fmt}: CRC32 {got:08x}, expected {want:08x}'
            print(fmt, 'verified')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    info = dict(ver=ver, size=os.path.getsize(ORIG), crc=apply_patch.crc32_file(ORIG),
                sha=digest(ORIG), md5=digest(ORIG, 'md5'), pcrc=want, psha=digest(patched),
                changes=f'\nChanges in {ver}\n----------------\n' + CHANGES[ver] if ver in CHANGES else '')
    for fmt, other in (('bps', 'ppf'), ('ppf', 'bps')):
        sub_name = f'{name}-{fmt}'
        out = os.path.join(dist, sub_name)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        shutil.move(paths[fmt], os.path.join(out, f'{name}.{fmt}'))
        shutil.copy(os.path.join(os.path.dirname(__file__), 'apply_patch.py'), out)
        for lic in ('LICENSE', 'LICENSE-TEXT'):
            shutil.copy(lic, out)
        open(os.path.join(out, 'README.txt'), 'w', newline='\r\n').write(README.format(
            patch=f'{name}.{fmt}', tools=TOOLS[fmt], other=other, othertools=TOOLS[other], **info))
        zpath = out + '.zip'
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for fn in sorted(os.listdir(out)):
                z.write(os.path.join(out, fn), f'{sub_name}/{fn}')
        for fn in sorted(os.listdir(out)):
            print(f'{os.path.getsize(os.path.join(out, fn)):>10}  {sub_name}/{fn}')
        print(f'{os.path.getsize(zpath):>10}  {zpath}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
