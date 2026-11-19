# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Max decompressed bytes produced per 0x800 bytes of compressed input."""
import sys
def bursts(src):
    out = 0; pos = 0; per = {}
    while pos + 1 < len(src):
        flags = src[pos] | src[pos+1] << 8; pos += 2
        for _ in range(16):
            s = pos // 0x800
            if not flags & 0x8000: pos += 2; n = 2
            else:
                v = src[pos] | src[pos+1] << 8; pos += 2
                ln, d = v >> 11, v & 0x7ff
                if ln == 0: ln = src[pos] | src[pos+1] << 8; pos += 2
                if ln == 0 and d == 0: return per
                n = ln * 2
            per[s] = per.get(s, 0) + n
            flags = (flags << 1) & 0xffff
    return per
if __name__ == '__main__':
    for f in sys.argv[1:]:
        p = bursts(open(f, 'rb').read()); last = max(p)
        mid = {k: v for k, v in p.items() if k < last - 1}
        k = max(mid, key=mid.get) if mid else None
        print(f'{f}: max burst (excluding last 2 sectors) {mid.get(k, 0):#x} at sector {k}/{last}')
