# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English memory card save titles (shown only in the BIOS card manager).

The PSX card header title is up to 64 bytes of Shift-JIS. The EN text is
written as full-width ASCII (SJIS 0x8140 space, 0x824f digits, 0x8260 A-Z,
0x8281 a-z) within the JP byte length and NUL-padded.

- Exe file offset 0x74c50 (RAM 0x80084450, pointer at exe 0x7a5a8):
  card title of every save, 52 bytes + NUL.
- MGAME file offset 0x28af4 (RAM 0x800caaf4, pointer at MGAME 0x2358):
  the same string for diary saves.
- EMTBL.BIZ (DAT2, LZ): a complete card file (SC header, 10 blocks) for the
  PocketStation app; title at +4, 64 bytes padded with full-width spaces.

Run after tools/hook.py and tools/build_sysmsg.py (it edits the exe and
MGAME.BIN they wrote in private/build/ in place) and before build_cock.py.
EMTBL.BIZ is written fresh from private/unpacked/.
--check prints the encoded titles and writes nothing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import lz
from game import EXE_NAME

TITLE = 'Rival Schools Diary 2 Save'        # 26 full-width chars = 52 bytes
PS_TITLE = 'Rival Schools: Pocket Justice'  # 29 chars, padded to 32

JP = '私立ジャスティス学園熱血青春日記２のセーブデータです'.encode('shift_jis')
PS_JP = '私立ジャスティス学園熱血青春日記２　ポケジャス'.encode('shift_jis')
SITES = [('private/build/' + EXE_NAME, 0x74c50),
         ('private/build/MGAME.BIN', 0x28af4)]
EMTBL_SRC = 'private/unpacked/EMTBL.BIZ.lz'   # decompressed
EMTBL_ORIG = 'private/disc/DAT2/EMTBL.BIZ'    # for the allocation check

def fullwidth(s):
    """ASCII -> full-width SJIS bytes (2 per char)."""
    out = bytearray()
    for c in s:
        if c == ' ': u = '　'
        elif '!' <= c <= '~': u = chr(ord(c) + 0xfee0)
        else: raise ValueError(f'not ASCII: {c!r}')
        b = u.encode('shift_jis')
        assert len(b) == 2, c
        out += b
    return bytes(out)

def patch_slot(buf, off, jp, en, what):
    """Write en over jp at off; accept a buffer that already has en."""
    n = len(jp)
    cur = bytes(buf[off:off + n])
    if cur != jp and cur != en.ljust(n, b'\0'):
        raise SystemExit(f'{what} 0x{off:x}: unexpected bytes {cur[:16].hex()}')
    if buf[off + n] != 0:
        raise SystemExit(f'{what} 0x{off:x}: no NUL after the JP string')
    if len(en) > n:
        raise SystemExit(f'{what}: EN {len(en)} bytes > JP {n}')
    buf[off:off + n] = en.ljust(n, b'\0')

def main(check=False):
    en = fullwidth(TITLE)
    ps = fullwidth(PS_TITLE.ljust(32))
    print(f'card title {len(en)}/{len(JP)} B: {en.decode("shift_jis")}')
    print(f'PocketStation title {len(ps)}/64 B: {ps.decode("shift_jis")}')
    # EMTBL: title field at +4..+0x44, then zeros.
    d = bytearray(open(EMTBL_SRC, 'rb').read())
    assert d[:2] == b'SC', 'EMTBL is not a card file'
    field = bytes(d[4:0x44])
    assert field[:len(PS_JP)] == PS_JP, 'EMTBL title differs from the JP'
    d[4:0x44] = ps
    c = lz.compress(bytes(d), name='EMTBL.BIZ')
    assert lz.decompress(c)[0] == bytes(d)
    alloc = -(-os.path.getsize(EMTBL_ORIG) // 0x800) * 0x800
    assert len(c) <= alloc, f'EMTBL {len(c)} B > allocation {alloc}'
    print(f'EMTBL.BIZ {len(c)} B (JP {os.path.getsize(EMTBL_ORIG)}, '
          f'alloc {alloc})')
    if check: return
    for path, off in SITES:
        buf = bytearray(open(path, 'rb').read())
        patch_slot(buf, off, JP, en, os.path.basename(path))
        open(path, 'wb').write(buf)
        print(f'{path}: 0x{off:x}')
    open('private/build/EMTBL.BIZ.lz', 'wb').write(d)
    open('private/build/EMTBL.BIZ', 'wb').write(c)
    print('private/build/EMTBL.BIZ')

if __name__ == '__main__':
    main(check='--check' in sys.argv)
