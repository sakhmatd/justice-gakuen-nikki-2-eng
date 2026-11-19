# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the diary background archives (DAT2/MGBG.TZZ, MGBG?nn.BZZ) from
the patch modules tools/mgbg_*.py.

Each module exposes PATCHES = [(archive, entry, fn)], fn(bytes) -> bytes
on the decompressed entry. Patches for one entry run in module-name order.
Every patched entry is recompressed with lz.py. An entry that still fits
its slot (to the next 0x800-aligned entry offset) stays in place; if one
grows past its slot, the entries after it move down (0x800 aligned) and the
TOC offsets follow. The game's archive loader reads the TOC from disc
(exe slot 0x15c, private/re/beta/findings.org), so entries may move. The
archive must still fit the file's disc allocation (patch_bin.py does not
move files); the length of the original file is kept.

Writes private/build/NAME for each patched archive. Run inside
private/build/.rebuild.lock, after build_mbms.py ALL.
Usage: python3 tools/build_mgbg.py [--dry]
"""
import os, sys, glob, importlib, struct
sys.path.insert(0, os.path.dirname(__file__))
import lz, arc

SRC = 'private/disc/DAT2/'
OUT = 'private/build/'

def modules():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sorted(glob.glob(os.path.join(here, 'mgbg_*.py'))):
        yield importlib.import_module(os.path.basename(p)[:-3])

def rebuild(name, fns):
    """fns: entry -> [fn]. Returns (archive bytes, report lines)."""
    orig = open(SRC + name, 'rb').read()
    toc = arc.toc(orig)
    alloc = (len(orig) + 0x7ff) // 0x800 * 0x800
    blobs = []; moved = False; notes = []
    for i, (o, s) in enumerate(toc):
        blob = orig[o:o + s]
        if i in fns:
            data = lz.decompress(blob)[0]
            new = data
            for fn in fns[i]: new = fn(new)
            if new != data:
                blob = lz.compress(new, name=name)
                assert lz.decompress(blob)[0] == new
        slot = (toc[i + 1][0] if i + 1 < len(toc) else len(orig)) - o
        if len(blob) > slot:
            moved = True; notes.append(f'entry {i}: {len(blob)} > slot {slot}, later entries move')
        blobs.append(blob)
    out = bytearray(orig[:0x800])
    if not moved:
        out = bytearray(orig)
        for i, ((o, s), b) in enumerate(zip(toc, blobs)):
            slot = (toc[i + 1][0] if i + 1 < len(toc) else len(orig)) - o
            out[o:o + slot] = b + bytes(slot - len(b))
            struct.pack_into('<II', out, 8 * i, o, len(b))
    else:
        for i, b in enumerate(blobs):
            o = len(out)
            struct.pack_into('<II', out, 8 * i, o, len(b))
            out += b + bytes(-len(b) % 0x800)
        if len(out) > alloc:
            raise SystemExit(f'{name}: {len(out)} bytes > disc allocation {alloc}')
        out += bytes(len(orig) - len(out)) if len(out) < len(orig) else b''
    return bytes(out), notes

def main(dry=False):
    reg = {}
    for m in modules():
        for a, i, fn in m.PATCHES: reg.setdefault(a, {}).setdefault(i, []).append(fn)
    for name in sorted(reg):
        data, notes = rebuild(name, reg[name])
        print(f'{name}: {len(reg[name])} entries patched, {len(data)} bytes'
              + ''.join(f'\n  {n}' for n in notes))
        if not dry: open(OUT + name, 'wb').write(data)

if __name__ == '__main__':
    main('--dry' in sys.argv)
