# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
import struct
def toc(d):
    """Archive: table of (offset,size) u32 pairs at 0, first offset = 0x800. Returns list or None."""
    if len(d) < 16: return None
    o0 = struct.unpack_from('<I', d, 0)[0]
    if o0 != 0x800 or len(d) < 0x800: return None
    ents=[]
    for i in range(0, 0x800, 8):
        o, s = struct.unpack_from('<II', d, i)
        if o == 0 and s == 0: break
        if o + s > len(d) or o % 0x800: return None
        ents.append((o, s))
    return ents
