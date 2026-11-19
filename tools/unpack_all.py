# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Decompress DAT2 files (and archive entries) into private/unpacked/."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from lz import decompress
from arc import toc
SRC, DST = 'private/disc/DAT2', 'private/unpacked'
def try_lz(d):
    try:
        out, end = decompress(d)
    except Exception:
        return None
    # accept if stream ended within the last 2 KB padding
    return out if len(d) - 0x800 <= end <= len(d) and len(out) >= len(d) else None
os.makedirs(DST, exist_ok=True)
for fn in sorted(os.listdir(SRC)):
    d = open(os.path.join(SRC, fn), 'rb').read()
    t = toc(d)
    if t:
        os.makedirs(os.path.join(DST, fn), exist_ok=True)
        for i, (o, s) in enumerate(t):
            e = d[o:o+s]; out = try_lz(e)
            tag = 'lz' if out is not None else 'raw'
            open(os.path.join(DST, fn, f'{i:03d}.{tag}'), 'wb').write(out if out is not None else e)
    else:
        out = try_lz(d)
        tag = 'lz' if out is not None else 'raw'
        open(os.path.join(DST, f'{fn}.{tag}'), 'wb').write(out if out is not None else d)
