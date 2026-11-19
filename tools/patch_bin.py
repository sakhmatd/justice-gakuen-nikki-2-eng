# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Patch replacement files into a copy of the original image in place.

Each file in private/build/ (not *.lz) overwrites the sectors of the disc file
with the same name; it must fit in the original allocation. Files in
game.RELOCATE move instead to sectors taken from the start of
game.RELOC_DONOR (zero padding); the exe file table MSF and the ISO record
LBA then change as well, and the donor's record starts after them.
The ISO directory record and the exe file table size and check word (param =
the file's first u32) are updated, and EDC/ECC are regenerated for every
rewritten sector.
Output: private/out/nikki2-en.bin/.cue. Run from the project root.
"""
import os, sys, struct, shutil
sys.path.insert(0, os.path.dirname(__file__))
from filetab import load_lba, bcd

from game import ORIG, OUT, TAB, NTAB, EXE_NAME, RELOCATE, RELOC_DONOR
SS = 2352

# --- EDC/ECC (Mode 2 Form 1) ---
EDC = []
for i in range(256):
    e = i
    for _ in range(8): e = (e >> 1) ^ (0xD8018001 if e & 1 else 0)
    EDC.append(e)
FL = [0] * 256; BL = [0] * 256
for i in range(256):
    j = ((i << 1) ^ (0x11D if i & 0x80 else 0)) & 0xff
    FL[i] = j; BL[i ^ j] = i

def edc(b):
    e = 0
    for x in b: e = (e >> 8) ^ EDC[(e ^ x) & 0xff]
    return e

def ecc_block(s, major_count, minor_count, major_mult, minor_inc):
    size = major_count * minor_count
    out = bytearray(major_count * 2)
    for major in range(major_count):
        idx = (major >> 1) * major_mult + (major & 1)
        a = b = 0
        for _ in range(minor_count):
            t = s[idx]; idx += minor_inc
            if idx >= size: idx -= size
            a ^= t; b ^= t; a = FL[a]
        a = BL[FL[a] ^ b]
        out[major] = a; out[major + major_count] = a ^ b
    return out

def fix_sector(sec):
    sec[0x818:0x81c] = struct.pack('<I', edc(sec[0x10:0x818]))
    s = bytearray(sec[0xc:])
    s[0:4] = b'\0\0\0\0'
    p = ecc_block(s, 86, 24, 2, 86)
    s[0x810:0x810 + 172] = p
    q = ecc_block(s, 52, 43, 86, 88)
    sec[0x81c:0x81c + 172] = p
    sec[0x8c8:0x8c8 + 104] = q

def selftest(f):
    for lba in (16, 22765, 22766):
        f.seek(lba * SS); sec = bytearray(f.read(SS)); ref = bytes(sec)
        fix_sector(sec)
        assert bytes(sec) == ref, f'EDC/ECC self-test failed at {lba}'

def find_dirrec(f, lba):
    """Locate the directory record pointing at lba; return absolute byte offset."""
    def rd(l):
        f.seek(l * SS + 24); return f.read(2048)
    pvd = rd(16)
    root = struct.unpack_from('<I', pvd, 156 + 2)[0], struct.unpack_from('<I', pvd, 156 + 10)[0]
    stack = [root]
    while stack:
        dl, dsz = stack.pop()
        for k in range((dsz + 2047) // 2048):
            blk = rd(dl + k); i = 0
            while i < 2048 and blk[i]:
                l = blk[i]; el = struct.unpack_from('<I', blk, i + 2)[0]
                flags = blk[i + 25]; nl = blk[i + 32]
                if blk[i + 33:i + 33 + nl] not in (b'\0', b'\1'):
                    if flags & 2: stack.append((el, struct.unpack_from('<I', blk, i + 10)[0]))
                    elif el == lba: return (dl + k) * SS + 24 + i, dl + k
                i += l
    raise KeyError(lba)

def msf(lba):
    """BCD MSF bytes (minute, second, frame) of an LBA."""
    lba += 150
    b = lambda v: (v // 10) << 4 | v % 10
    return bytes([b(lba // 4500), b(lba // 75 % 60), b(lba % 75)])

def write_file(f, lba, alloc, data, subs=None):
    n = (len(data) + 2047) // 2048
    assert n <= alloc, f'{n} sectors > {alloc}'
    f.seek(lba * SS); first_sub = f.read(SS)[16:24]
    f.seek((lba + alloc - 1) * SS); last_sub = f.read(SS)[16:24]
    if subs: first_sub, last_sub = subs
    for k in range(alloc):
        f.seek((lba + k) * SS); sec = bytearray(f.read(SS))
        chunk = data[k*2048:(k+1)*2048].ljust(2048, b'\0')
        sec[24:24 + 2048] = chunk
        if k == n - 1: sec[16:24] = last_sub        # EOF/EOR flags move to the new end
        elif k >= n or k == alloc - 1: sec[16:24] = first_sub
        fix_sector(sec)
        f.seek((lba + k) * SS); f.write(sec)

def patch_sector_bytes(f, lba, off, data):
    f.seek(lba * SS); sec = bytearray(f.read(SS))
    sec[24 + off:24 + off + len(data)] = data
    fix_sector(sec); f.seek(lba * SS); f.write(sec)

def main(out=None, only=None):
    global OUT
    if out: OUT = out
    old = load_lba('private/lba.txt')                       # lba -> (size, path)
    byname = {p.split('/')[-1]: (l, s) for l, (s, p) in old.items()}
    # Build under a temporary name and rename it into place at the end, so
    # emulators running on the old image keep reading their open file.
    tmp = OUT + '.bin.tmp'
    shutil.copyfile(ORIG, tmp)
    open(OUT + '.cue', 'w').write(f'FILE "{os.path.basename(OUT)}.bin" BINARY\n'
                                  '  TRACK 01 MODE2/2352\n    INDEX 01 00:00:00\n')
    exe = bytearray(open('private/disc/' + EXE_NAME, 'rb').read())
    repl = {f: os.path.join('private/build', f) for f in sorted(os.listdir('private/build'))
            if not f.endswith('.lz') and not f.startswith('.') and (only is None or f in only)}
    if EXE_NAME in repl: exe = bytearray(open(repl.pop(EXE_NAME), 'rb').read())
    # relocations: name -> (new LBA, sectors); the donor shrinks from the front
    reloc, at = {}, None
    if any(name in repl for name, _ in RELOCATE):
        dl, dsize = byname[RELOC_DONOR]
        at = dl
        for name, n in RELOCATE:
            if name in repl: reloc[name] = (at, n); at += n
        assert (at - dl) * 2048 < dsize
    with open(tmp, 'r+b') as f:
        selftest(f)
        for name, path in repl.items():
            lba, size = byname[name]
            alloc = (size + 2047) // 2048
            data = open(path, 'rb').read()
            f.seek(lba * SS + 24); orig4 = f.read(4)
            new = lba
            if name in reloc:
                new, alloc = reloc[name]
                f.seek(lba * SS); s0 = f.read(SS)[16:24]
                f.seek((lba + (size + 2047) // 2048 - 1) * SS); s1 = f.read(SS)[16:24]
                write_file(f, new, alloc, data, (s0, s1))
            else:
                write_file(f, lba, alloc, data)
            off, dlba = find_dirrec(f, lba)
            rel = off - dlba * SS - 24
            if new != lba:
                patch_sector_bytes(f, dlba, rel + 2, struct.pack('<I', new) + struct.pack('>I', new))
            patch_sector_bytes(f, dlba, rel + 10, struct.pack('<I', len(data)) + struct.pack('>I', len(data)))
            for k in range(NTAB):
                e = exe[TAB + 12*k: TAB + 12*k + 12]
                if (bcd(e[0]) * 60 + bcd(e[1])) * 75 + bcd(e[2]) - 150 == lba:
                    if new != lba: exe[TAB + 12*k: TAB + 12*k + 3] = msf(new)
                    exe[TAB + 12*k + 4: TAB + 12*k + 8] = struct.pack('<I', len(data))
                    # The param is a check word: the loader (0x80015b08,
                    # 0x80015c74) compares it with the file's first u32
                    # and retries forever on a mismatch.
                    if exe[TAB + 12*k + 8: TAB + 12*k + 12] == orig4:
                        exe[TAB + 12*k + 8: TAB + 12*k + 12] = data[:4]
            print(f'{name}: {len(data)} bytes at LBA {new} (was {size} at {lba}, {alloc} sectors)')
        if reloc:
            dl, dsize = byname[RELOC_DONOR]
            off, dlba = find_dirrec(f, dl)
            rel = off - dlba * SS - 24
            nsize = dsize - (at - dl) * 2048
            patch_sector_bytes(f, dlba, rel + 2, struct.pack('<I', at) + struct.pack('>I', at)
                               + struct.pack('<I', nsize) + struct.pack('>I', nsize))
            print(f'{RELOC_DONOR}: now at LBA {at}, {nsize} bytes')
        elba, esize = byname[EXE_NAME]
        assert len(exe) == esize
        write_file(f, elba, (esize + 2047) // 2048, bytes(exe))
    os.replace(tmp, OUT + '.bin')
    print('wrote', OUT + '.bin')

if __name__ == '__main__':
    # patch_bin.py [OUTNAME [FILE...]]  (default: all of private/build)
    a = sys.argv[1:]
    if a and a[0].startswith('-'):
        sys.exit(__doc__.strip())
    main('private/out/' + a[0] if a else None, set(a[1:]) or None)
