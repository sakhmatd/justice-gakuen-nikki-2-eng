# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""KANRI pre-fight EDIT character page in English: KANRI.BIN glyph strings
(stat labels and values) and the command-list header art in EDITMOJI.TIZ.
Port of the first game's tools/build_kanri.py (topic register there).

Also the 8-letter nickname tag on the data page and card screen
(tools/kanrinick.py: NDRAW code at the file end, narrow letters in free
EDITMOJI TIM rows 8-11), and the EDIT Register screens: every register
string (tools/kanri_reg.py) and the register art in EDITMOJI
(tools/kanri_art.py: four art buttons in TIM 1, header, list tab and school
labels in TIM 0). KANRI is relocated by patch_bin.py (game.RELOCATE), so
it may grow to that many sectors.

Reads tl/KANRI.txt (## KANRI id / JP / EN), data/latin.tsv (MOJI2 Latin
codes), private/unpacked/MOJI2.TIZ.lz (glyph shapes, for centring) and
private/unpacked/EDITMOJI.TIZ.lz. Edits private/build/KANRI.BIN (written by
build_editcm.py; the disc copy if it is missing) and writes
private/build/EDITMOJI.TIZ. Run it after build_editcm.py, which rewrites
KANRI.BIN from the disc copy. It is idempotent on its own output.
--check only reports problems; --png writes the header art and a sheet of
the strings to private/re/editchar/kanri/.

KANRI facts (topic editchar, section KANRI strings): string records are
12 bytes (s16 x, s16 y, u32 RAM pointer, s16 tighten, u16 0); strings are
u16 count + u16 codes. Drawer 0x800a3b3c: 20x20 cells, pitch 20 - tighten,
0x38 bytes of primitive buffer per cell. Status labels: records 0x5b0 +
12k (8 cells each). Values: records 0x9d4 + 12k (16 cells; the caller
copies x and the pointer to the stack and zeroes the tighten, so a stub
sets it). Header: record 0x52c (card screen), codes 0xfRC are 20x20 cells
of the EDITMOJI TIM at VRAM (704,0) (CLUT 160,481, an orange ramp).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
import lz, kanrinick, kanri_reg, kanri_art
from game import RELOCATE
from build_sysmsg import charmap

KANRI_DISC = 'private/disc/DAT1/KANRI.BIN'
KANRI_OUT = 'private/build/KANRI.BIN'
EM_SRC = 'private/unpacked/EDITMOJI.TIZ.lz'
EM_DISC = 'private/disc/DAT2/EDITMOJI.TIZ'
EM_OUT = 'private/build/EDITMOJI.TIZ'
EM_SLOT = 49 * 2048                 # LBA 22934 -> next file at 22983
MOJI = 'private/unpacked/MOJI2.TIZ.lz'
TL = 'tl/KANRI.txt'
PNG_DIR = 'private/re/editchar/kanri'
BASE = 0x800a2000
ORIG_LEN = 0x83a8
MAX_LEN = dict(RELOCATE)['KANRI.BIN'] * 2048   # relocated (patch_bin.py)

# id: (records, cell limit, centre x, tighten)
LEFT_C = 94                         # JP label/value column centre
RIGHT_C = 415                       # the portraits under s24/s25 (x 401-429)
LAYOUT = {}
for i, k in enumerate(range(19, 24)):
    LAYOUT[f's{k}'] = ([0x5b0 + 12 * i], 8, LEFT_C, 6)
LAYOUT['s24'] = ([0x5ec], 8, RIGHT_C, 6)
LAYOUT['s25'] = ([0x5f8], 8, RIGHT_C, 6)
for k in range(5):
    LAYOUT[f'v{k}'] = ([0x9d4 + 12 * k] + ([0xa10] if k == 4 else []), 16, LEFT_C, 6)
# JP strings that only these records use (pool space to reuse); the
# register strings free the rest of the pools (kanri_reg.free_list)
FREE = [(0x404, 0x450), (0x9a4, 0x9d4)]

# value call (0x800a7a28 jal 0x800a3b3c): send it through a stub that
# stores tighten 6 at sp+0x20 (the caller wrote 0 there)
VAL_JAL, DRAWER, VAL_T = 0x5a28, 0x800a3b3c, 6
# Guts gauge "N本": stack word 0xa9 (本) -> 0x3001 (end), digit cell x
NHON_GLYPH, NHON_X = 0x5a74, 0x5a60
NHON_CX = LEFT_C

# header: record 0x52c, art in free rows of the EDITMOJI TIM at (704,0)
HDR_REC = 0x52c
HDR_TIM = 0x21000                   # TIM header; pixels at +0x14, 128 B/row
HDR_ROWS = (10, 11)                 # glyph rows (y 200-239), 12 cells each
CARD_C = 237                        # card x 34-439 on the command-list screen
HDR_FONT = 'DejaVuSansCondensed-Bold.ttf'  # card text font
HDR_CAP = 13
HDR_TOP = 3                         # cap top row inside the 20 px cell
HDR_MAXW = 2 * (CARD_C - 135)       # clear of the nickname tag (x 42-125)


def load_tl(path=TL):
    out = {}; key = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('## KANRI '): key = line[9:].strip()
        elif line.startswith('EN: ') and key: out[key] = line[4:].strip(); key = None
    return out


# ---------------------------------------------------------------- glyphs
def moji_glyphs():
    d = open(MOJI, 'rb').read()
    p = 8 + struct.unpack_from('<I', d, 8)[0] + 12
    raw = np.frombuffer(d[p:p + 512 * 256], np.uint8).reshape(256, 512)
    px = np.zeros((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px


def glyph(px, c):
    t, r, col = c >> 8, c >> 4 & 15, c & 15
    return px[20 * r:20 * r + 20, 256 * t + 20 * col:256 * t + 20 * col + 20]


def ink(px, codes, tighten):
    lo = hi = None
    for i, c in enumerate(codes):
        if c == 0xfff: continue
        cols = np.where(glyph(px, c).any(0))[0]
        if not len(cols): continue
        a = i * (20 - tighten)
        lo = a + cols[0] if lo is None else min(lo, a + cols[0])
        hi = a + cols[-1] if hi is None else max(hi, a + cols[-1])
    return lo, hi


def encode(text, cm):
    out = []
    for ch in text:
        if ch not in cm: raise ValueError(f'no glyph for {ch!r} in {text!r}')
        out.append(cm[ch])
    return out


# ---------------------------------------------------------------- header art
def header_art(text, font=HDR_FONT, cap=HDR_CAP, top=HDR_TOP, ss=4, track=0.0):
    """20-row index image: coverage -> CLUT idx 3..14 (the JP glyphs are
    anti-aliased orange on index 0, no outline)."""
    f0 = fontpath.truetype(font, 100)
    b = f0.getbbox('H'); size = 100 * cap / (b[3] - b[1])
    f = fontpath.truetype(font, round(size * ss))
    W = int(f.getlength(text) + len(text) * track * ss) + 8 * ss
    W += -W % ss
    im = Image.new('L', (W, 20 * ss), 0); dr = ImageDraw.Draw(im)
    htop = f.getbbox('H')[1]
    x = 2 * ss
    for ch in text:
        dr.text((x, top * ss - htop), ch, font=f, fill=255)
        x += f.getlength(ch) + track * ss
    a = np.asarray(im, np.float32).reshape(20, ss, W // ss, ss).mean((1, 3)) / 255
    idx = np.where(a < 0.08, 0, np.clip(np.rint(2 + 12 * a), 3, 14)).astype(np.uint8)
    cols = np.where(idx.any(0))[0]
    return idx[:, cols[0]:cols[-1] + 1]


def header_cells(art):
    """Split into 20 px cells; pad left so the ink centre stays known.
    Returns (cells image 20 x 20n, left pad)."""
    w = art.shape[1]; n = -(-w // 20)
    pad = (20 * n - w) // 2
    out = np.zeros((20, 20 * n), np.uint8); out[:, pad:pad + w] = art
    return out, pad


def em_patch(d, cells):
    """Write header cells into the TIM at HDR_TIM (glyph rows HDR_ROWS).
    Returns the glyph codes."""
    p = HDR_TIM + 0x14
    raw = np.frombuffer(bytes(d[p:p + 128 * 256]), np.uint8).reshape(256, 128)
    px = np.zeros((256, 256), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    for r in HDR_ROWS: px[20 * r:20 * r + 20] = 0
    codes = []
    for i in range(cells.shape[1] // 20):
        r, c = HDR_ROWS[i // 12], i % 12
        px[20 * r:20 * r + 20, 20 * c:20 * c + 20] = cells[:, 20 * i:20 * i + 20]
        codes.append(0xff00 | r << 4 | c)
    d[p:p + 128 * 256] = (px[:, 0::2] | px[:, 1::2] << 4).astype(np.uint8).tobytes()
    return codes


# ---------------------------------------------------------------- KANRI
def patch_kanri(k, tl, problems, hdr_codes=None, hdr_x=None):
    disc = open(KANRI_DISC, 'rb').read()
    k = bytearray(k[:ORIG_LEN])
    for a, b in FREE + kanri_reg.POOLS + kanri_reg.RESTORE: k[a:b] = disc[a:b]
    for recs, *_ in LAYOUT.values():
        for r in recs: k[r:r + 12] = disc[r:r + 12]
    k[HDR_REC:HDR_REC + 12] = disc[HDR_REC:HDR_REC + 12]
    for o in (VAL_JAL, NHON_GLYPH, NHON_X): k[o:o + 4] = disc[o:o + 4]
    kanrinick.restore(k, disc)
    cm = charmap(); gpx = moji_glyphs()
    strings = []                                  # (records, codes, x, tighten)
    for key, (recs, lim, cx, t) in LAYOUT.items():
        en = tl.get(key, '')
        if en in ('', '='): continue
        try: codes = encode(en, cm)
        except ValueError as e: problems.append(f'{key}: {e}'); continue
        if len(codes) > lim: problems.append(f'{key}: {en!r} has {len(codes)} cells > {lim}'); continue
        lo, hi = ink(gpx, codes, t)
        strings.append((recs, codes, round(cx - (lo + hi + 1) / 2), t))
    if hdr_codes:
        strings.append(([HDR_REC], hdr_codes, hdr_x, 0))
    known = set(LAYOUT) | {'hdr'} | kanri_reg.ids() | kanri_art.IDS
    for key in tl:
        if key not in known: problems.append(f'{key}: unknown id')
    reg, words = kanri_reg.strings(tl, disc, cm, gpx, ink, encode, problems)
    for recs, codes, t in reg:
        strings.append((recs, codes, None, t))
    for o, b in words: k[o:o + len(b)] = b
    free = kanri_reg.free_list(disc)
    for f in free: k[f[0]:f[1]] = b'\0' * (f[1] - f[0])
    for recs, codes, x, t in strings:
        blob = struct.pack('<H%dH' % len(codes), len(codes), *codes)
        blob += b'\0' * (-len(blob) % 4)
        for f in free:
            if f[1] - f[0] >= len(blob):
                at = f[0]; k[at:at + len(blob)] = blob; f[0] += len(blob); break
        else:
            at = len(k); k += blob
        for r in recs:
            r, x, y = r if isinstance(r, tuple) else (r, x, None)
            if y is None: y = struct.unpack_from('<h', k, r + 2)[0]
            struct.pack_into('<hhIhH', k, r, x, y, BASE + at, t, 0)
    # value tighten stub
    assert struct.unpack_from('<I', k, VAL_JAL)[0] == 0x0c000000 | (DRAWER >> 2 & 0x3ffffff)
    org = BASE + len(k)
    k += struct.pack('<3I', 0x24080000 | VAL_T,                # addiu t0, zero, 6
                     0x08000000 | (DRAWER >> 2 & 0x3ffffff),   # j drawer
                     0xa7a80020)                               # sh t0, 0x20(sp)
    struct.pack_into('<I', k, VAL_JAL, 0x0c000000 | (org >> 2 & 0x3ffffff))
    # N本 -> N, digit cell centred on the column
    assert struct.unpack_from('<I', k, NHON_GLYPH)[0] == 0x240200a9
    assert struct.unpack_from('<I', k, NHON_X)[0] == 0x2407004a
    struct.pack_into('<I', k, NHON_GLYPH, 0x24020000 | 0x3001)
    struct.pack_into('<I', k, NHON_X, 0x24070000 | (NHON_CX - 10))
    if len(k) > MAX_LEN: problems.append(f'KANRI {len(k):#x} > {MAX_LEN:#x}')
    return k


def build(tl, problems):
    d = bytearray(open(EM_SRC, 'rb').read())
    hdr_codes = hdr_x = art = None
    en = tl.get('hdr', '')
    if en not in ('', '='):
        art = header_art(en)
        if art.shape[1] > HDR_MAXW:
            problems.append(f'hdr: {art.shape[1]} px > {HDR_MAXW}')
        cells, pad = header_cells(art)
        if cells.shape[1] > 20 * 12 * len(HDR_ROWS):
            problems.append('hdr: too many cells')
        else:
            hdr_codes = em_patch(d, cells)
            hdr_x = round(CARD_C - pad - art.shape[1] / 2)
    src = KANRI_OUT if os.path.exists(KANRI_OUT) else KANRI_DISC
    k = patch_kanri(bytearray(open(src, 'rb').read()), tl, problems, hdr_codes, hdr_x)
    # 8-letter nickname tag: NDRAW after the strings, glyphs in free TIM rows
    tim = tim_px(d)
    kanrinick.patch(k, tim, open(KANRI_DISC, 'rb').read(), moji_glyphs(), glyph, problems,
                    hdr_codes or ())
    reg_art = kanri_art.build(d, tim, tl, problems)
    tim_put(d, tim)
    if len(k) > MAX_LEN: problems.append(f'KANRI {len(k):#x} > {MAX_LEN:#x}')
    return k, d, art, src, reg_art


def tim_px(d):
    p = HDR_TIM + 0x14
    raw = np.frombuffer(bytes(d[p:p + 128 * 256]), np.uint8).reshape(256, 128)
    px = np.zeros((256, 256), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px


def tim_put(d, px):
    p = HDR_TIM + 0x14
    d[p:p + 128 * 256] = (px[:, 0::2] | px[:, 1::2] << 4).astype(np.uint8).tobytes()


def main(argv):
    check = '--check' in argv; png = '--png' in argv
    tl = load_tl(); problems = []
    k, d, art, src, reg_art = build(tl, problems)
    c = lz.compress(bytes(d), name='EDITMOJI.TIZ')
    if len(c) > EM_SLOT: problems.append(f'EDITMOJI compressed {len(c)} > slot {EM_SLOT}')
    if lz.decompress(c)[0][:len(d)] != bytes(d): problems.append('EDITMOJI LZ round trip failed')
    print(f'KANRI: {len(k):#x} bytes (from {src}); EDITMOJI: {len(c)} bytes (slot {EM_SLOT})')
    if png and art is not None:
        os.makedirs(PNG_DIR, exist_ok=True)
        clut = [0x0000, 0x84a0, 0x8422, 0x8464, 0x8487, 0x84c9, 0x84ec, 0x8551,
                0x8593, 0x85b5, 0x85f8, 0x861a, 0x865d, 0x86bf, 0x86ff, 0xd022]
        pal = np.array([[(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3, 255 if i else 0]
                        for i, c in enumerate(clut)], np.uint8)      # VRAM (160,481)
        Image.fromarray(pal[art]).save(f'{PNG_DIR}/hdr.png')
    for p in problems: print('  ' + p)
    if not check and not problems:
        open(KANRI_OUT, 'wb').write(k)
        open(EM_OUT, 'wb').write(c)
    print(f'KANRI: problems {len(problems)}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
