# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Message files (MGC*, MGK*, MGM*, MGT*, MGE03, MGF02, MGGnn[GJOPT]):
4bpp TIM font (20x20 glyphs) + text at MSG_TEXT. Same format as the
first game's MBMS files, with another load address and text offset.

Loaded at MSG_LOAD. Text area: 8 u32 pointers to groups at MSG_TEXT (null
for an empty slot), each group = u16 strings followed by a u32 pointer
table to them.
"""
import struct, glob, os, hashlib
from game import MSG_LOAD as LOAD, MSG_TEXT as TEXT, MSG_GLOB, MSG_SIZE, FIRST
GW = GH = 20; GB = 200; FBASE = 0x180
UNPACKED = os.path.join(os.path.dirname(__file__), '..', 'private', 'unpacked')

def font_count(d):
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, 0x174)
    return h // GH
def glyph(d, i):
    return d[FBASE + i*GB: FBASE + (i+1)*GB]
def gid(g):
    return hashlib.sha1(g).hexdigest()[:12]
def slots(d):
    """The 8 group table pointers at TEXT (0 = empty slot)."""
    return list(struct.unpack_from('<8I', d, TEXT))
def groups(d):
    """Return [(slot, table_offset, [string offsets])] for each non-null
    group. slot is the index in the 8-pointer table, so numbering stays
    the same in files that leave a slot empty."""
    tops = [(k, t - LOAD) for k, t in enumerate(slots(d)) if t]
    out = []
    for n, (slot, to) in enumerate(tops):
        nxt = next_block(d, tops[n+1][1]) if n + 1 < len(tops) else None
        ptrs = []
        o = to
        while o + 4 <= len(d):
            p = struct.unpack_from('<I', d, o)[0]
            if not (LOAD + TEXT < p < LOAD + to): break
            ptrs.append(p - LOAD); o += 4
            if nxt is not None and o >= nxt: break
        out.append((slot, to, ptrs))
    return out
def next_block(d, to):
    """Offset of the first string of the group whose table is at `to`."""
    return struct.unpack_from('<I', d, to)[0] - LOAD
def files():
    out = []
    for g in MSG_GLOB: out += glob.glob(os.path.join(UNPACKED, g))
    return sorted(out)
def names():
    return [os.path.basename(f).split('.')[0] for f in files()]
