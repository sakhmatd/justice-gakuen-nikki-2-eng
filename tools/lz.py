# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Word-based LZ used by the game (exe 0x80015dbc).

Stream of u16 LE. Flag word, 16 items MSB-first. Bit 0: copy one literal
word. Bit 1: ref word v; len = v >> 11, dist = v & 0x7ff (halfwords); if
len == 0 the next word is len. dist == 0 and len == 0 ends the stream.
"""
import struct, sys

def decompress(src, pos=0):
    out = bytearray()
    n = len(src)
    while pos + 1 < n:
        flags = src[pos] | src[pos+1] << 8; pos += 2
        for _ in range(16):
            if not flags & 0x8000:
                out += src[pos:pos+2]; pos += 2
            else:
                v = src[pos] | src[pos+1] << 8; pos += 2
                ln, dist = v >> 11, v & 0x7ff
                if ln == 0:
                    ln = src[pos] | src[pos+1] << 8; pos += 2
                if dist == 0 and ln == 0:
                    return bytes(out), pos
                p = len(out) - dist * 2
                for _ in range(ln):
                    out += out[p:p+2]; p += 2
            flags = (flags << 1) & 0xffff
    return bytes(out), pos

if __name__ == '__main__':
    d = open(sys.argv[1], 'rb').read()
    out, end = decompress(d)
    print(f'in {len(d)} consumed {end} out {len(out)}', file=sys.stderr)
    open(sys.argv[2], 'wb').write(out)

CAP = 0x1800        # upper bound of one regular callback's output
CAP_UNNAMED = 0x1400   # when the caller gives no file name
_jp = {}

def jp_cap(name):
    """Callback cap for a rebuilt file: the JP file's largest regular
    callback (over all archive entries), at most CAP."""
    import os, arc
    if name in _jp: return _jp[name]
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(root, 'private', 'disc')):
        root = os.path.dirname(root)
        if root == '/': raise FileNotFoundError('lz.jp_cap: no private/disc')
    root = os.path.join(root, 'private', 'disc')
    for sub in ('DAT2', 'DAT1', 'DAT3', 'DAT4'):
        p = os.path.join(root, sub, name)
        if os.path.exists(p): break
    else:
        raise FileNotFoundError(f'lz.jp_cap: no JP file {name}')
    d = open(p, 'rb').read()
    toc = arc.toc(d) or [(0, len(d))]
    m = max(cbmax(d[o:o + s]) for o, s in toc)
    _jp[name] = min(CAP, m) if m >= 0x880 else CAP
    return _jp[name]

def callbacks(src):
    """Output bytes per regular streaming callback, and of the final one.

    The loader (exe 0x80015cfc) decodes with 0x80015eac one sector behind
    the CD. A regular callback decodes every flag group whose flag word
    lies in one input sector; the group that straddles the sector end is
    decoded whole. The callback for the last sector decodes the last two
    sectors. Returns (list for sectors 0..n-3, final).
    """
    n = len(src); pos = 0; per = {}; end = False
    while pos + 1 < n and not end:
        s = pos // 0x800; o = 0
        flags = src[pos] | src[pos+1] << 8; pos += 2
        for _ in range(16):
            if not flags & 0x8000:
                pos += 2; o += 2
            else:
                v = src[pos] | src[pos+1] << 8; pos += 2
                ln, d = v >> 11, v & 0x7ff
                if ln == 0:
                    ln = src[pos] | src[pos+1] << 8; pos += 2
                if ln == 0 and d == 0:
                    end = True; break
                o += ln * 2
            flags = (flags << 1) & 0xffff
        per[s] = per.get(s, 0) + o
    k = (n + 0x7ff) // 0x800 - 2
    return ([per.get(i, 0) for i in range(max(0, k))],
            sum(v for i, v in per.items() if i >= k))

def cbmax(src):
    """Largest regular-callback output of a compressed stream."""
    return max(callbacks(src)[0] or [0])

def compress(data, chain=64, burst=None, name=None):
    """Greedy LZ matching the game's format. data length must be even.

    The game decodes while the CD streams into an 8-sector ring, inside the
    read callback (see callbacks()). A callback that runs too long lets the
    ring overrun. So the output of all flag groups whose flag word lies in
    one input sector is held to at most `burst` bytes (JP files reach
    ~0x18ce). With `name` the cap is jp_cap(name); `burst` overrides it.
    The cap is hard: when a ref would leave too little room to
    reach the sector end with literals, a shorter ref or a literal is
    emitted. Only the trailing all-zero tail is exempt; its items sit in
    the last two sectors, which decode in the final callback after the
    last read.
    """
    import array
    if burst is None: burst = jp_cap(name) if name else CAP_UNNAMED
    assert len(data) % 2 == 0
    assert burst >= 0x880, 'cap below one sector of literals'
    w = array.array('H', data)
    n = len(w)
    tail = n
    while tail > 0 and w[tail-1] == 0: tail -= 1
    head = {}; prev = [-1] * n
    def insert(i):
        if i + 1 < n:
            k = w[i] | w[i+1] << 16
            prev[i] = head.get(k, -1); head[k] = i
    items = []                       # (is_ref, words)
    pos = 0                          # compressed bytes so far
    gsec = -1                        # input sector of the open group's flag
    gout = 0                         # output of groups flagged in gsec
    def fill(p, m, B):
        # literal output needed after an item that ends at p as item m of
        # its group, until a group flag word lands at or past B
        out = 2 * (16 - m); p += 2 * (16 - m)
        if p < B: out += 32 * ((B - p + 33) // 34)
        return out
    i = 0
    while i < n:
        m = len(items) % 16
        p = pos + (2 if m == 0 else 0)          # where this item starts
        if m == 0:
            s = pos // 0x800
            if s != gsec: gsec, gout = s, 0
        B = (gsec + 1) * 0x800
        best_len, best_d = 0, 0
        if i >= tail:
            cap = 0xffff
        else:
            r2 = (burst - gout - fill(p + 2, m + 1, B)) // 2
            r4 = (burst - gout - fill(p + 4, m + 1, B)) // 2
            cap = max(min(31, r2), r4 if r4 > 31 else 0)
            cap = min(0xffff, cap, max(2, tail - i))
        if i + 1 < n and cap >= 2:
            j = head.get(w[i] | w[i+1] << 16, -1); c = 0
            while j >= 0 and i - j <= 0x7ff and c < chain:
                L = 0
                while i + L < n and w[j+L] == w[i+L] and L < cap: L += 1
                if L > best_len: best_len, best_d = L, i - j
                if best_len >= cap: break
                j = prev[j]; c += 1
        if i < tail and best_len <= 31: best_len = min(best_len, r2)
        if best_len >= 2:
            ws = [(best_len << 11) | best_d] if best_len <= 31 else [best_d, best_len]
            items.append((1, ws)); gout += best_len * 2
            for k in range(best_len): insert(i + k)
            i += best_len
        else:
            ws = [w[i]]; items.append((0, ws)); gout += 2
            insert(i); i += 1
        pos = p + 2 * len(ws)
    items.append((1, [0, 0]))
    out = array.array('H')
    for s0 in range(0, len(items), 16):
        grp = items[s0:s0+16]; f = 0
        for k, (r, _) in enumerate(grp):
            if r: f |= 0x8000 >> k
        out.append(f)
        for _, ws in grp: out.extend(ws)
    out = out.tobytes()
    assert cbmax(out) <= burst
    return out
