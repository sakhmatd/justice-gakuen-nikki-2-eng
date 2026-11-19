# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Write the glyph-coded system messages (memory card, records, options,
game system, key config) in English, into both copies of the pool.

Run after tools/hook.py: hook.py writes private/build/SLPS_021.20 fresh
from the disc, and this script edits that file in place. NEWOPT.BIN is read
from the disc and written to private/build/NEWOPT.BIN. Rerunning hook.py
drops the exe part of this build, so always run both, in this order.
Inputs: tl/SYSMSG.txt, column limits in data/sysmsg_use.tsv, font codes in
data/latin.tsv. --check only reports problems. --tl PATH reads another
translation file (for previews; the normal build uses tl/SYSMSG.txt).

Format: u16 len + len u16 codes, code 0xTRC = DAT2/MOJI2.TIZ tpage T,
row R, col C; 0xfff is a blank cell. Each string is one row of a 24 x 21
grid (no bounds check), so a string must fit the columns its screen gives
it. The exe table (148 u32 at 0x80090580) serves records, the card dialogs
and the boot auto-load message. NEWOPT.BIN has a byte-identical copy of
pool and table (table at file 0xb90) for the options screens. Both tables
get the same exe addresses; the exe pool is always resident. Strings go to
the old exe pool: first those the exe draws, then the unused ones, then the
NEWOPT-only ones (data/sysmsg_use.tsv column 2). What does not fit goes to
NEWOPT's own pool copy, which is resident only while NEWOPT is loaded, so
only NEWOPT-only and unused strings may go there. Both tables point at it
for those ids. The image tail after the BSS clear (0x800a0b68-0x800a1000)
is not free: crt0 calls InitHeap(0x800a0b6c, ...), and the heap header
there overwrote the 2nd and 3rd cells of the string placed at 0x800a0b68
("Reset" drawn as "R11et").
Row-1 titles are drawn with 40x40 glyphs; TITLE_SMALL switches those calls
to 20 px glyphs. The titles are then centred on the 24-column grid, drawn
in colour 2 (green ramp; colour 6 is the outlined 40 px palette and comes
out dark green) and moved 10 px down (and 10 px right for odd lengths) so
they sit on the blue title bar. The move is a 7-word stub on the sprite
position call of each grid renderer: exe 0x80020f40 -> EXE_STUB (end of
the hook spare), NEWOPT 0x800a5358 -> NEWOPT_STUB (start of its unused
pool copy). It adds the offset for grid row 1 only.

Options main (NEWOPT): s014 is both the label prefix of rows r7/r8 (s005
and s006 follow at c5) and the unit after the match count. If the EN of
s005 and s006 are whole labels (not starting with a lowercase letter), the
prefix entries draw an empty string and s005/s006 move to c2. The cursor
highlight lengths and bar widths of the records screen and the three
NEWOPT screens are set from the EN lengths.
EN '=' or empty keeps the original string.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
from game import EXE_NAME

DISC = 'private/disc'
EXE = 'private/build/' + EXE_NAME
NEWOPT_IN, NEWOPT_OUT = DISC + '/DAT1/NEWOPT.BIN', 'private/build/NEWOPT.BIN'
NBASE = 0x800a2000                       # NEWOPT load address
PTRS, NPTR = 0x80090580, 148
NPTRS = 0x800a2b90                       # NEWOPT copy of the table
EXE_POOL = (0x8008fd5c, 0x80090580)
NOP_POOL = (0x800a2388, 0x800a2b90)  # NEWOPT pool copy after NEWOPT_STUB
NEWOPT_ONLY = ('options main', 'game system', 'key config', 'c2 if s014', 'unit')
TITLE_SMALL = [0xd8f0]                   # exe file offsets, records title
TITLE_SMALL_N = [0x19d4, 0x1e70, 0x26ec]  # NEWOPT: options, game sys, keys
# layout quads (col, row, colour, id)
REC_LAYOUT = 0x80089f4c                  # exe, records (6)
REC_BAR = 0x80089f64                     # 4 widths: load, save, auto save, exit
REC_HI = 0x8001d128                      # addiu a2,zero,6: label highlight
MAIN, SYS, KEYS = 0x800a2090, 0x800a2100, 0x800a2300   # NEWOPT (16/12/7)
MAIN_BAR, MAIN_ROW, MAIN_HI = 0x800a20d0, 0x800a20e0, 0x800a20f0
KEY_BAR = 0x800a2338                     # 10 widths: 8 actions, default, exit
KEY_HI = 0x800a47bc                      # addiu a2,zero,9: value highlight
SYS_HI = 0x800a3ee8                      # addiu a2,zero,6: value highlight
SYS_BAR = 0x800a3f18                     # addiu a3,zero,5: value bar width
SYS_COL = 0x800a3e5c                     # addiu s4,zero,10: P1 value column (P2 = +7)
EXE_STUB, EXE_CALL, EXE_POS = 0x80010d0c, 0x80020f40, 0x80013b40
NEWOPT_STUB, NEWOPT_CALL = 0x800a236c, 0x800a5358
TITLE_COLOUR = 2
BLANK = 87                               # unused id, emptied for the s014 split
SPECIAL = {' ': 0xfff, '.': 0x2b6, ',': 0x070, '-': 0x071, "'": 0x066}
ICONS = {'{CIRCLE}': 0x2a0, '{SQUARE}': 0x2a1, '{CROSS}': 0x2a2, '{TRIANGLE}': 0x2a3,
         '{L1}': 0x2a4, '{L2}': 0x2a5, '{R1}': 0x2a6, '{R2}': 0x2a7}
def F(a): return a - 0x80010000 + 0x800
def N(a): return a - NBASE

def charmap():
    m = {}
    for line in open('data/latin.tsv', encoding='utf-8').read().split('\n')[1:]:
        if '\t' not in line: continue
        name, code = line.split('\t')
        if len(name) == 1: m.setdefault(name, int(code, 16))
    m.update(SPECIAL)
    return m

def load_tl(path='tl/SYSMSG.txt'):
    out = {}; sid = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('## SYSMSG s'): sid = int(line[11:])
        elif line.startswith('EN: ') and sid is not None: out[sid] = line[4:].strip()
    return out

def screens():
    """'exe' if the exe may draw the id, 'nop' if only NEWOPT does, 'none'."""
    cls = {}
    for line in open('data/sysmsg_use.tsv', encoding='utf-8').read().split('\n')[1:]:
        f = line.split('\t')
        if len(f) < 2 or not f[0].startswith('s'): continue
        parts = [p.strip() for p in f[1].split(';')]
        cls[int(f[0][1:])] = ('none' if f[1].startswith('unused') else
                              'nop' if all(p.startswith(NEWOPT_ONLY) for p in parts) else 'exe')
    return cls

def limits():
    lim = {}
    for line in open('data/sysmsg_use.tsv', encoding='utf-8').read().split('\n')[1:]:
        f = line.split('\t')
        if len(f) >= 3 and f[2].strip().isdigit(): lim[int(f[0][1:])] = int(f[2])
    return lim

def encode(text, cm):
    codes = []
    for tok in re.split(r'(\{[A-Z0-9]+\}|\{sp\})', text):
        if not tok: continue
        if tok == '{sp}': codes.append(0xfff)
        elif tok in ICONS: codes.append(ICONS[tok])
        else:
            for ch in tok:
                if ch not in cm: raise ValueError(f'no glyph for {ch!r}')
                codes.append(cm[ch])
    return codes

def stub(buf, off, addr, xs, jump):
    """if row (fp) == 1: y (a2) += 10, x (a1) += xs; then jump on."""
    words = [0x0e << 26 | 30 << 21 | 25 << 16 | 1,      # xori t9, fp, 1
             0x05 << 26 | 25 << 21 | 3,                 # bnez t9, +3
             0,
             0x09 << 26 | 6 << 21 | 6 << 16 | 10,       # addiu a2, a2, 10
             0x09 << 26 | 5 << 21 | 5 << 16 | xs,       # addiu a1, a1, xs
             jump, 0]
    struct.pack_into('<7I', buf, off, *words)

def split_match(tl):
    """True if s005/s006 are whole labels, so s014 is only the unit."""
    return all(tl.get(i, '') not in ('', '=') and not tl[i][0].islower() for i in (5, 6))

def sys_layout(lab, val):
    """Game system columns (label, P1 value) for the longest label and value.
    Disc: labels c2, values c10/c17. Longer labels move P1 right, as far as
    P2 (P1 + 7) still ends by c23; then the labels move left (not below c1)."""
    if lab <= 7: return 2, 10
    v1 = min(24 - 7 - val, 2 + lab + 1)
    return v1 - lab - 1, v1

def build(check=False, tlpath='tl/SYSMSG.txt'):
    cm = charmap(); tl = load_tl(tlpath); lim = limits()
    split = split_match(tl)
    if split:
        lim[5] = lim.get(5, 12) + 3; lim[6] = lim.get(6, 12) + 3; lim[14] = 5
    disc = open(os.path.join(DISC, EXE_NAME), 'rb').read()
    ptr = [struct.unpack_from('<I', disc, F(PTRS) + 4 * i)[0] for i in range(NPTR)]
    orig = {}                            # original strings by pointer (ids share some)
    for p in set(ptr):
        n = struct.unpack_from('<H', disc, F(p))[0]
        orig[p] = list(struct.unpack_from('<%dH' % n, disc, F(p) + 2))
    problems = 0; new = {}
    for i in range(NPTR):
        en = tl.get(i, '')
        if en in ('', '='): continue
        try: codes = encode(en, cm)
        except ValueError as e: print(f's{i:03d}: {e}'); problems += 1; continue
        if i in lim and len(codes) > lim[i]:
            print(f's{i:03d}: {len(codes)} cells > {lim[i]}: {en}'); problems += 1
        new[i] = codes
    if split: new[BLANK] = []
    Lc = lambda i: len(new[i]) if i in new else len(orig[ptr[i]])
    glab = max(Lc(i) for i in range(91, 96))
    gval = max(Lc(i) for i in list(range(96, 107)) + [19, 20, 61, 112])
    gcol = sys_layout(glab, gval)
    if gcol[0] < 1 or gval > 6:
        print(f'game system: labels {glab} + values {gval} cells do not fit'); problems += 1
    if check:
        print(f'SYSMSG: {len(new)} translated, s014 split {split}, '
              f'game system cols {gcol[0]}/{gcol[1]}/{gcol[1] + 7}, problems {problems}')
        return problems
    exe = bytearray(open(EXE, 'rb').read())
    assert exe[F(PTRS):F(PTRS) + 4 * NPTR] == disc[F(PTRS):F(PTRS) + 4 * NPTR], \
        'exe table already patched: run tools/hook.py first'
    nop = bytearray(open(NEWOPT_IN, 'rb').read())
    # lay out: every distinct pointer gets one string (EN of its first id)
    blobs = {}; rank = {}; cls = screens(); order = {'exe': 0, 'none': 1, 'nop': 2}
    for i in range(NPTR):
        if ptr[i] not in blobs: blobs[ptr[i]] = new.get(i, orig[ptr[i]])
        rank[ptr[i]] = min(rank.get(ptr[i], 3), order[cls.get(i, 'exe')])
    exe[F(EXE_POOL[0]):F(EXE_POOL[1])] = bytes(EXE_POOL[1] - EXE_POOL[0])
    nop[N(NOP_POOL[0]):N(NOP_POOL[1])] = bytes(NOP_POOL[1] - NOP_POOL[0])
    cur = [list(EXE_POOL), list(NOP_POOL)]; where = {}
    for p, codes in sorted(blobs.items(), key=lambda kv: (rank[kv[0]], -len(kv[1]))):
        size = 2 + 2 * len(codes); size += -size % 4
        for k, c in enumerate(cur[:1] if rank[p] == 0 else cur):
            if c[0] + size <= c[1]:
                where[p] = c[0]; c[0] += size; break
        else:
            print('out of pool space'); return problems + 1
        buf, off = (exe, F(where[p])) if k == 0 else (nop, N(where[p]))
        struct.pack_into('<H%dH' % len(codes), buf, off, len(codes), *codes)
    for i in range(NPTR):
        struct.pack_into('<I', exe, F(PTRS) + 4 * i, where[ptr[i]])
        struct.pack_into('<I', nop, N(NPTRS) + 4 * i, where[ptr[i]])
    L = Lc

    def setw(buf, off, old, val):        # patch an addiu immediate or a byte
        if old > 0xff:
            w = struct.unpack_from('<I', buf, off)[0]
            assert w == old, (hex(off), hex(w))
            struct.pack_into('<I', buf, off, w & 0xffff0000 | val)
        else:
            assert buf[off] == old, (hex(off), buf[off], old); buf[off] = val

    def putw(buf, off, old, new):        # replace a whole word
        assert struct.unpack_from('<I', buf, off)[0] == old, hex(off)
        struct.pack_into('<I', buf, off, new)

    titles = (42, 62, 63, 90)
    if all(t in new for t in titles):
        for o in TITLE_SMALL: setw(exe, o, 0x24070001, 0)
        for o in TITLE_SMALL_N: setw(nop, o, 0x24070001, 0)
        setw(exe, F(REC_LAYOUT), 10, (24 - L(42)) // 2)
        setw(nop, N(MAIN), 7, (24 - L(62)) // 2)
        setw(nop, N(SYS), 3, (24 - L(90)) // 2)
        setw(nop, N(KEYS), 6, (24 - L(63)) // 2)
        for buf, off in ((exe, F(REC_LAYOUT)), (nop, N(MAIN)), (nop, N(SYS)), (nop, N(KEYS))):
            setw(buf, off + 2, 6, TITLE_COLOUR)
        odd = [L(i) % 2 for i in (62, 63, 90)]
        if len(set(odd)) > 1: print('NEWOPT titles: mixed odd/even lengths, x not centred')
        assert not any(exe[F(EXE_STUB):F(EXE_STUB) + 28]), 'exe stub area in use'
        stub(exe, F(EXE_STUB), EXE_STUB, 10 * (L(42) % 2), 0x08000000 | EXE_POS >> 2 & 0x3ffffff)
        putw(exe, F(EXE_CALL), 0x0c004ed0, 0x0c000000 | EXE_STUB >> 2 & 0x3ffffff)
        stub(nop, N(NEWOPT_STUB), NEWOPT_STUB, 10 * min(odd), 0x00400008)   # jr v0
        putw(nop, N(NEWOPT_CALL), 0x0040f809, 0x0c000000 | NEWOPT_STUB >> 2 & 0x3ffffff)
    # records: s043 r4, s044 r6, s045 + s044 at c5 r8 (s017 c9 / s018 c10), s112 r10
    setw(exe, F(REC_HI), 0x24060006, max(6, L(43), L(44), 3 + L(44), L(112)))
    for k, w in enumerate((L(43), L(44), max(7 + L(17), 8 + L(18)), L(112))):
        exe[F(REC_BAR) + k] = w
    # options main: s014 split, label highlight lengths, bar widths
    if split:
        for q in (0x800a20a0, 0x800a20a8): setw(nop, N(q) + 3, 14, BLANK)
        for q in (0x800a20a4, 0x800a20ac): setw(nop, N(q), 5, 2)
        r78 = [L(5), L(6)]
    else:
        r78 = [3 + L(5), 3 + L(6)]
    # per cursor: label length, last column of the value (None: item row)
    lab = [L(2), L(3), L(4)] + r78 + [L(89), L(10), 0, L(11), L(7), L(12), L(13), L(61), L(112)]
    v56 = max(18 + L(17), 19 + L(18))   # s017 at c19, s018 at c20
    vend = [21, 21, 17 + max(L(21), 4), 18 + L(14), 18 + L(14), v56, v56, None,
            19 + max(L(19), L(20)), 17 + max(L(15), L(16)), None, None, None, None]
    for k in range(14):
        if k == 7: continue
        nop[N(MAIN_HI) + k] = lab[k]
        nop[N(MAIN_BAR) + k] = lab[k] if vend[k] is None else max(20, vend[k] - 1)
    # key config: actions s024-s034 at c3 after the icon at c2
    kv = max(L(i) for i in range(24, 35))
    setw(nop, N(KEY_HI), 0x24060009, max(9, kv))
    for k in range(8): nop[N(KEY_BAR) + k] = max(10, kv + 1)
    nop[N(KEY_BAR) + 8] = L(61); nop[N(KEY_BAR) + 9] = L(112)
    # game system: values at c10/c17 (cursor rows 6-10, default r12, exit r14)
    gv = gval
    lc, v1 = gcol
    # quads 1-11 (P1/P2 headers, five labels, reset, exit): disc columns
    for k, c in enumerate((10, 17, 2, 2, 2, 2, 2, 10, 17, 10, 17), 1):
        setw(nop, N(SYS) + 4 * k, c, lc if c == 2 else v1 + c - 10)
    setw(nop, SYS_COL - NBASE, 0x2414000a, v1)
    setw(nop, SYS_HI - NBASE, 0x24060006, max(6, gv))
    setw(nop, SYS_BAR - NBASE, 0x24070005, max(5, gv))
    open(EXE, 'wb').write(exe)
    open(NEWOPT_OUT, 'wb').write(nop)
    used = [c[0] - q[0] for c, q in zip(cur, (EXE_POOL, NOP_POOL))]
    print(f'SYSMSG: {len(new)} translated, exe pool {used[0]} of '
          f'{EXE_POOL[1] - EXE_POOL[0]} B, NEWOPT pool {used[1]} of '
          f'{NOP_POOL[1] - NOP_POOL[0]} B, s014 split {split}, '
          f'problems {problems}')
    return problems

if __name__ == '__main__':
    a = sys.argv[1:]
    tlp = a[a.index('--tl') + 1] if '--tl' in a else 'tl/SYSMSG.txt'
    build('--check' in a, tlp)
