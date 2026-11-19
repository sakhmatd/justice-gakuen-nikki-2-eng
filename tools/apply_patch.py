#!/usr/bin/env python3
# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Apply the English patch (.bps or .ppf) to a Nekketsu Seishun Nikki 2 dump.

Usage: python3 apply_patch.py PATCH ORIGINAL.bin [OUTPUT.bin]

ORIGINAL.bin is the Redump dump of Shiritsu Justice Gakuen: Nekketsu
Seishun Nikki 2 (Japan, SLPS-02120), one MODE2/2352 track. The original is
not modified. The output defaults to "Nekketsu Seishun Nikki 2
(English).bin" next to the original, and a matching .cue is written beside
it.
Needs only Python 3.6 or newer (no extra packages).
"""
import os, struct, sys, zlib

CHUNK = 1 << 24
# The Redump dump. PPF has no checksum of its own, so it is checked here.
ORIG_SIZE, ORIG_CRC = 663216960, 0xd97d0aaa


def crc32_file(path):
    c = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return c & 0xffffffff
            c = zlib.crc32(b, c)


def copy_range(src, dst, n):
    while n:
        b = src.read(min(n, CHUNK))
        if not b:
            raise ValueError('source file ends early')
        dst.write(b)
        n -= len(b)


class Reader:
    def __init__(self, data):
        self.d, self.i = data, 0

    def take(self, n):
        b = self.d[self.i:self.i + n]
        if len(b) != n:
            raise ValueError('patch file is truncated')
        self.i += n
        return b

    def varint(self):
        v, shift = 0, 1
        while True:
            x = self.take(1)[0]
            v += (x & 0x7f) * shift
            if x & 0x80:
                return v
            shift <<= 7
            v += shift


def apply_bps(patch, src_path, out_path):
    if zlib.crc32(patch[:-4]) & 0xffffffff != struct.unpack('<I', patch[-4:])[0]:
        raise ValueError('the patch file is damaged (CRC mismatch)')
    r = Reader(patch)
    if r.take(4) != b'BPS1':
        raise ValueError('not a BPS patch')
    src_size, dst_size, meta = r.varint(), r.varint(), r.varint()
    r.take(meta)
    src_crc, dst_crc = struct.unpack('<II', patch[-12:-4])
    if os.path.getsize(src_path) != src_size:
        raise ValueError(f'original has {os.path.getsize(src_path)} bytes, '
                         f'expected {src_size}')
    print('Checking the original...')
    if crc32_file(src_path) != src_crc:
        raise ValueError('the original does not match (CRC32 differs). '
                         'Use an unmodified Redump dump.')
    end = len(patch) - 12
    out_pos = src_rel = dst_rel = 0
    print('Patching...')
    with open(src_path, 'rb') as src, open(out_path, 'w+b') as out:
        while r.i < end:
            v = r.varint()
            cmd, n = v & 3, (v >> 2) + 1
            if cmd == 0:                                    # SourceRead
                src.seek(out_pos)
                copy_range(src, out, n)
            elif cmd == 1:                                  # TargetRead
                out.write(r.take(n))
            else:
                o = r.varint()
                o = -(o >> 1) if o & 1 else o >> 1
                if cmd == 2:                                # SourceCopy
                    src_rel += o
                    src.seek(src_rel)
                    copy_range(src, out, n)
                    src_rel += n
                else:                                       # TargetCopy
                    dst_rel += o
                    for _ in range(n):
                        out.seek(dst_rel)
                        b = out.read(1)
                        out.seek(out_pos)
                        out.write(b)
                        out_pos += 1
                        dst_rel += 1
                    continue
            out_pos += n
        if out_pos != dst_size:
            raise ValueError('patch produced the wrong output size')
    if crc32_file(out_path) != dst_crc:
        raise ValueError('patched image has the wrong CRC32')


def apply_ppf(patch, src_path, out_path):
    if patch[:5] != b'PPF30' or patch[5] != 2:
        raise ValueError('only PPF 3.0 patches are supported')
    imagetype, blockcheck, undo = patch[0x38], patch[0x39], patch[0x3a]
    i = 0x3c
    with open(src_path, 'rb') as src:
        if blockcheck:
            src.seek(0x9320 if imagetype == 0 else 0x80a0)
            if src.read(1024) != patch[i:i + 1024]:
                raise ValueError('the original does not match the patch '
                                 '(PPF block check failed)')
            i += 1024
    if b'@BEGIN_FILE_ID.DIZ' in patch[-4096:]:
        raise ValueError('PPF patches with a file_id.diz block are not supported')
    if os.path.getsize(src_path) != ORIG_SIZE:
        raise ValueError(f'original has {os.path.getsize(src_path)} bytes, '
                         f'expected {ORIG_SIZE} (the Redump dump)')
    print('Checking the original...')
    if crc32_file(src_path) != ORIG_CRC:
        raise ValueError(f'the original does not match (CRC32 differs, '
                         f'expected {ORIG_CRC:08x}). Use the Redump dump.')
    end = len(patch)
    print('Patching...')
    with open(src_path, 'rb') as src, open(out_path, 'wb') as out:
        copy_range(src, out, os.path.getsize(src_path))
    with open(out_path, 'r+b') as out:
        while i < end:
            off, n = struct.unpack_from('<QB', patch, i)
            i += 9
            out.seek(off)
            out.write(patch[i:i + n])
            i += n + (n if undo else 0)


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__.strip())
        return 2
    patch_path, src = argv[1], argv[2]
    out = argv[3] if len(argv) == 4 else os.path.join(
        os.path.dirname(os.path.abspath(src)), 'Nekketsu Seishun Nikki 2 (English).bin')
    if os.path.abspath(out) == os.path.abspath(src):
        print('error: the output must be a different file than the original')
        return 1
    patch = open(patch_path, 'rb').read()
    try:
        if patch[:4] == b'BPS1':
            apply_bps(patch, src, out)
        elif patch[:3] == b'PPF':
            apply_ppf(patch, src, out)
        else:
            raise ValueError('unknown patch format (expected .bps or .ppf)')
    except (ValueError, OSError) as e:
        print('error:', e)
        if os.path.exists(out) and os.path.abspath(out) != os.path.abspath(src):
            os.remove(out)
        return 1
    cue = os.path.splitext(out)[0] + '.cue'
    with open(cue, 'w', newline='\r\n') as f:
        f.write(f'FILE "{os.path.basename(out)}" BINARY\n'
                '  TRACK 01 MODE2/2352\n    INDEX 01 00:00:00\n')
    print('Wrote', out)
    print('Wrote', cue)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
