# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary location captions (MGBG.TZZ 000-092) in English.
Patch module for tools/build_mgbg.py (PATCHES) and hook.py (MGAME_PATCH);
wording in tl/CAPTIONS.txt. RE: private/re/captions/findings.org.

Each location background is an 8bpp TIM 256x244 with one 256-colour CLUT.
Rows 0-219 hold the 512x110 picture, rows 220-243 the caption strip. The
event script op 03 (MGAME 0x800ac4b0) starts a task (0x800b4018) that
draws the strip as one semi-transparent (additive) SPRT u 0, v 220, w 96,
h 24 at y 112, sliding in from x 640 to x 384, over an opaque G4 bar
(x-16 .. x+112, y 124-135, green (128,240,64) to blue (64,96,240)). The
strip's black is CLUT 0x8000, so it adds nothing; the white text and its
dark grey shadow are opaque (STP 0).

The EN strip is 128 wide: the sprite becomes 128 wide and stops at x 352,
the bar grows to x-16 .. x+144, so it still ends at x 496 and keeps its
16 px margins (MGAME_PATCH). The strip texels u 96-127 are free (the
picture ends at row 219). The EN line is drawn fresh, centred on the strip
(screen x 416), white italic with a dark grey extrusion down-right, and
mapped to the neutral greys of each picture's own CLUT. CLUT and header
stay unchanged.

81 of the 84 location TIMs have 1-65 byte-identical copies (969 in all)
in the MGBG?nn archives, at several offsets in bundle entries (COPIES);
000, 039 and 064 are only in MGBG.TZZ. The patcher scans
an entry for TIM headers and replaces every TIM whose bytes equal an
original location TIM.

Usage: python3 tools/mgbg_captions.py [--png DIR]  previews only.
"""
import os, sys, struct, hashlib, functools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath

OVL = 0x800a2000
TIM_LEN = 8 + 12 + 512 + 12 + 256 * 244
PIX = 8 + 12 + 512 + 12
HDR = bytes.fromhex('1000000009000000')       # TIM, 8bpp with CLUT
ROW0, H = 220, 24                              # strip rows in the TIM
W = 128                                        # EN strip / sprite width
X_HOLD = 352                                   # sprite x when held (JP 384)
BAR_L, BAR_R = 16, W + 16                      # bar x-16 .. x+144 (JP +112)
MAXW = 124                                     # EN ink width limit (px)
FONT = 'BarlowCondensed-SemiBoldItalic.ttf'  # OFL
CAP = 15                                       # cap height (px)
BASE = 17                                      # baseline row in the strip
FACE, SHADE = 248.0, 52.0                      # grey levels
EXTRUDE = ((1, 1), (2, 2))                     # shadow offsets (dx, dy)
KEY = 14                                       # darker than this = transparent
SHEAR = 0.15                                   # extra slant (x per y)
TRACK = 0.0                                    # extra letter spacing (px)
SS = 4

# JP caption -> MGBG.TZZ entries (distinct TIMs; 016 = 015, 017 = 003,
# 023 = 022, 051 = 050, 073 = 072, 083 = 082 are patched by content).
ENTRIES = {
    'ヨットハーバー': (0,), '教室': (1, 13, 14), '体育館': (2, 15),
    'グラウンド': (3, 11, 12, 85), '保健室': (4, 18), '屋上': (5, 19),
    '校門前': (6,), '2F廊下': (7, 22), '1F廊下': (8, 24, 25),
    '渡り廊下': (9, 26), '廊下': (10,), '正門前': (20, 57),
    '部室棟': (21, 27), 'プール': (28,), 'プールサイド': (29, 89, 90, 91),
    '相撲部部室': (30,), '下駄箱前': (31,), '階段': (33,), 'ステージ': (34,),
    '中庭': (35,), '喫茶店': (36, 69, 78), '商店街': (39, 54),
    '自宅': (40, 43, 56), '住宅街': (41, 42, 62, 63), '公園': (44,),
    'ホテル前': (45, 46, 64), 'ビーチ': (47, 48), '海の家': (49,),
    'パラソルビーチ': (50,), '部屋': (52, 53), '繁華街': (55,), '広場': (58,),
    '乗り場前': (59,), '観覧車': (60,), '池': (61,), 'パーティ会場': (65,),
    '根性神社': (66,), '恋愛神社': (67, 72), '河原': (68, 74, 75), '街': (70,),
    '縁日': (71, 81), 'パーティ会場中庭': (76,), 'ゲームセンター': (77, 82),
    'デパート': (79, 80), 'ロビー': (84,), '遊戯場': (86,), '入場口': (87,),
    '中央広場': (88,), '高台': (92,),
}

# Archive entries holding location TIMs: 'nn' -> MGBGnn.BZZ entries, 'ixk'
# when entry i holds k location TIMs (private/re/captions/copies.py).
COPIES = {
    'C00': '0 1 2 3 4 5 6 7 8',
    'E03': '0x2 1 2x2 3x2 4x2 5 6x2 7x2 8 9x2 10 11 12x2 13 14 15 16 17 '
           '18 19 20 21x2 22',
    'F02': '0 1x4 2 3 4 5 6 7x2 8',
    'G04': '0 1 2 3x2 4 5 6 7 8 9 10 11 12 13 14x2 15 16x3 17 18 19 20 '
           '21x3 22 23 24 25x2 26 27 28 29x2 30x2 31x2 32 33x2 34 35 36 '
           '37x2',
    'G06': '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 '
           '23x2 24x2 25x2 26x2',
    'G08': '0 1x2 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20x2',
    'G09': '0 1 2 3 4 5 6 7 8 9 10x2 11x2 12x2 13x2 14x2 15x2 16x2 17x2 '
           '18x2 19x2 20x2 21x2 22x2 23x2 24x2 25x2 26x2 27x2 28x2 29 30 '
           '31 32',
    'G11': '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 '
           '23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39',
    'K01': '0x2 1 2x2 3x2 4x2 5x2 6 7x2 8x2 9x2 10 11 12x3 13x2 14x2 '
           '15x2 16 17x2 18x2 19 20 21 22 23 24 25 26 27 28 29 30x2',
    'K02': '0 1 2 3 4x2 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20x2 21 '
           '22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 '
           '42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 '
           '62 63 64 65',
    'K03': '0 1 2x2 3x2 4 5x3 6x3 7x3 8x3 9x3 10x3 11x3 12x3 13x3 14 '
           '15x2 16x3 17x3 18x3 19 20x2 21x3 22 23x2 24 25x2 26 27 28 29 '
           '30 31 32x5 33x2 34x2 35x2 36x2 37x2 38x2 39x2 40x2 41x2 42x2 '
           '43x2 44x2 45x2 46x2 47x2 48x2 49x2 50x2 51x2',
    'K04': '0 1',
    'K05': '0 2',
    'K07': '0x2 1x2 2x2 3x2 4x2 5x2 6 7 8x3 9x3 10x3 11x3 12x3 13x3 14x2 '
           '15x3 16x3 17x2 18x3 19x2 20x3 21x3 22x3 23x3 24x3',
    'K08': '0 1 2x2 3 4x3 5x3 6x2 7x3 8x2 9x2 10x2 11x2 12x2 13 14x3 '
           '15x2 16x3 17 18x2 19x2 20x3 21x2',
    'K09': '0 1 2 3 4 5 6 7 8 9',
    'K10': '0 2',
    'K12': '0 1 2 3 4 5 6 7 8x2 9x2 10x2 11x2 12x2 13x2 14x2 15x2 16x2 '
           '17x2 18x2 19x2 20x2 21x2 22x2 23x2 24x2 25x2 26x2',
    'M01': '0 1x2 2 3 4 5 6 7x2 8 9x2 10 11x2 12 13 14x2 15 16 17 18',
    'M02': '0x2 1x2 2x2 3 4 5 6 7x2 8 9 10 11 12 13x2 14x2 15 16x2 17 18',
    'M04': '0 1 2 3 4 5x2 6',
    'M05': '0x2 1 2 3 4 5',
    'M06': '0x3 1 2 3 4 5',
    'M07': '0x2 1 2 3x2 4x2 5',
    'M08': '0 1 2 3 4x2 5',
    'M09': '0 1x4 2 3 4 5 6x2 7x2 8x3 9x2 10x3 11',
    'M10': '0x2 1x2 2 3 4 5 6x2 7x2 8x2 9 10',
    'M11': '0x2 1 2x2 3 4 5 6 7 8 9 10',
    'M12': '0x2 1 2 3 4 5 6 7 8 9x3 10 11 12x3 13 14x2 15',
    'T00': '0 1 2x2 3x2 4x2',
    'T01': '0 1 2x2 3x2 4x2',
    'T02': '0 1x2 2 3x2 4x2 5x2',
    'T03': '0 1x2 2x2 3x2 4x2',
    'T04': '0x2 1x3 2x2 3x2 4x2 5x2',
    'T05': '0 1x3 2 3x2 4x2',
    'T06': '0 1 2 3x2 4x2',
    'T07': '0x2 1 2x2 3x2 4x2',
    'T08': '0 1 2 3x2 4x2',
    'T09': '0x2 1 2x2 3x2 4x2',
    'T0A': '0 1 2x3 3x3 4x2 5x2',
    'T0B': '0 1x2 2x2 3x2 4x2',
    'T0C': '0 1x2 2 3x2 4x2',
    'T0D': '0x2 1x2 2x2 3x2 4x2',
    'T0F': '0 1 2 3x2 4x2',
    'T10': '0x2 1x2 2x2 3x2 4x2',
    'T11': '0 1 2 3x2',
    'T12': '1x3 2x2 3x2 4x2',
    'T13': '0 1 2 3x2 4x2',
}

def load_tl(path='tl/CAPTIONS.txt'):
    out = {}; jp = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            out[jp] = line[4:].strip(); jp = None
    return out

@functools.lru_cache(None)
def _archive(name):
    import arc
    d = open('private/disc/DAT2/' + name, 'rb').read()
    return d, arc.toc(d)

@functools.lru_cache(None)
def orig_tim(e):
    import lz
    d, toc = _archive('MGBG.TZZ')
    o, s = toc[e]
    return lz.decompress(d[o:o + s])[0][:TIM_LEN]

LOC = [e for es in ENTRIES.values() for e in es]

@functools.lru_cache(None)
def by_hash():
    """md5 of an original location TIM -> its MGBG.TZZ entry."""
    return {hashlib.md5(orig_tim(e)).digest(): e for e in LOC}

def unpack(tim):
    clut = np.frombuffer(tim, '<u2', 256, 20).astype(int)
    pal = np.stack([(clut & 31) << 3, (clut >> 5 & 31) << 3, (clut >> 10 & 31) << 3], -1)
    idx = np.frombuffer(tim, np.uint8, 256 * 244, PIX).reshape(244, 256)
    return clut, pal, idx

# --- lettering ----------------------------------------------------------

@functools.lru_cache(None)
def font_ss():
    """Font at SS x scale with cap height CAP * SS."""
    for sz in range(CAP * SS, CAP * SS * 3):
        f = fontpath.truetype(FONT, sz)
        b = f.getbbox('H', anchor='ls')
        if -b[1] >= CAP * SS: return f
    raise SystemExit('mgbg_captions: no font size for the cap height')

def down(a):
    h, w = a.shape
    return a.reshape(h // SS, SS, w // SS, SS).mean((1, 3))

def shift(a, dx, dy):
    out = np.zeros_like(a)
    out[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return out

@functools.lru_cache(None)
def line_alpha(text):
    """-> (face alpha H x W, ink width px, x squeeze factor)."""
    f = font_ss()
    im = Image.new('L', (W * SS * 3, H * SS), 0)
    dr = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        x = W * SS // 2 + f.getlength(text[:i]) + i * TRACK * SS
        dr.text((x, BASE * SS), ch, font=f, fill=255, anchor='ls')
    if SHEAR:   # slant about the baseline
        im = im.transform(im.size, Image.AFFINE, (1, SHEAR, -SHEAR * BASE * SS, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(im, float) / 255
    cols = np.nonzero(a.max(0) > 0.02)[0]
    a = a[:, cols[0]:cols[-1] + 1]
    ink = a.shape[1] / SS
    sq = 1.0
    if ink > MAXW:
        sq = MAXW / ink
        assert sq >= 0.9, f'mgbg_captions: "{text}" is {ink:.0f} px, more than 10% over {MAXW}'
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (int(round(a.shape[1] * sq)), a.shape[0]), Image.LANCZOS), float) / 255
    # Centre the ink on the strip, then shift right by half the extrusion.
    x0 = int(round((W * SS - a.shape[1]) / 2 - EXTRUDE[-1][0] * SS / 2))
    x0 = max(0, min(x0, W * SS - a.shape[1]))
    canvas = np.zeros((H * SS, W * SS))
    canvas[:, x0:x0 + a.shape[1]] = a
    return down(canvas), a.shape[1] / SS, sq

def strip_grey(text):
    """-> grey value H x W (0 = nothing) for one EN caption."""
    A, _, _ = line_alpha(text)
    S = np.zeros_like(A)
    for dx, dy in EXTRUDE: S = np.maximum(S, shift(A, dx, dy))
    return A * FACE + (1 - A) * S * SHADE

def quantise(grey, clut, pal, bg):
    neut = [i for i in range(256) if clut[i] and not clut[i] >> 15
            and pal[i].max() - pal[i].min() <= 16]
    lv = pal[neut].mean(1)
    k = np.abs(grey[..., None] - lv[None, None, :]).argmin(-1)
    out = np.array(neut, np.uint8)[k]
    out[grey < KEY] = bg
    return out

@functools.lru_cache(None)
def new_tim(e):
    tl = load_tl()
    jp = next(j for j, es in ENTRIES.items() if e in es)
    en = tl.get(jp)
    tim = orig_tim(e)
    if not en: return tim
    clut, pal, idx = unpack(tim)
    s = idx[ROW0:ROW0 + H, :96]
    bg = int(np.bincount(s.ravel()).argmax())
    assert clut[bg] == 0x8000, (e, hex(clut[bg]))
    new = idx.copy()
    new[ROW0:ROW0 + H, :W] = quantise(strip_grey(en), clut, pal, bg)
    out = bytearray(tim)
    out[PIX:PIX + 256 * 244] = new.tobytes()
    return bytes(out)

# --- patching -----------------------------------------------------------

def patch_entry(data, expect):
    """Replace every original location TIM in a bundle entry."""
    d = bytearray(data); n = 0; o = d.find(HDR)
    while o >= 0:
        e = by_hash().get(hashlib.md5(d[o:o + TIM_LEN]).digest())
        if e is not None:
            d[o:o + TIM_LEN] = new_tim(e); n += 1
            o = d.find(HDR, o + TIM_LEN)
        else:
            o = d.find(HDR, o + 4)
    assert n == expect, f'mgbg_captions: {n} location TIMs, expected {expect}'
    return bytes(d)

def patcher(expect):
    return lambda data: patch_entry(data, expect)

def copies():
    for k, v in COPIES.items():
        for t in v.split():
            i, _, c = t.partition('x')
            yield f'MGBG{k}.BZZ', int(i), int(c or 1)

PATCHES = [('MGBG.TZZ', e, patcher(1)) for e in range(93) if e not in (32, 37, 38)]
PATCHES += [(a, i, patcher(c)) for a, i, c in copies()]

def set_imm(d, ram, old, new):
    o = ram - OVL
    w = struct.unpack_from('<I', d, o)[0]
    assert w & 0xffff == old & 0xffff, (hex(ram), hex(w), old)
    struct.pack_into('<I', d, o, (w & 0xffff0000) | (new & 0xffff))

def MGAME_PATCH(data):
    """Caption sprite width, hold x and bar width (task 0x800b4018)."""
    d = bytearray(data)
    set_imm(d, 0x800b4070, 0x180, X_HOLD)      # start x without slide-in
    set_imm(d, 0x800b4388, 0x181, X_HOLD + 1)  # slide-in stops (slti)
    set_imm(d, 0x800b4398, 0x180, X_HOLD)      # hold x
    set_imm(d, 0x800b4258, 0x60, W)            # SPRT w
    set_imm(d, 0x800b4474, -0x10, -BAR_L)      # bar left  = x - 16
    set_imm(d, 0x800b4478, 0x70, BAR_R)        # bar right = x + 144
    return bytes(d)

# --- previews -----------------------------------------------------------

BAR0, BAR1 = np.array([128., 240., 64.]), np.array([64., 96., 240.])

def scene(tim, x, w, crop=(304, 100, 496, 152)):
    """Frame crop (x0, y0, x1, y1) of the 512x480 screen: picture at y 100
    (rows doubled), bar, then the caption sprite at (x, 112), u 0..w-1."""
    clut, pal, idx = unpack(tim)
    pic = np.concatenate([idx[:110], idx[110:220]], 1)
    fr = np.zeros((480, 512, 3)); fr[100:320] = np.repeat(pal[pic], 2, 0)
    t = np.linspace(0, 1, w + 32)[:, None]
    bx = np.arange(x - 16, x + w + 16)
    keep = (bx >= 0) & (bx < 496)
    fr[124:136, bx[keep]] = (BAR0 * (1 - t) + BAR1 * t)[keep][None]
    s = idx[ROW0:ROW0 + H, :w]; c = clut[s]
    op = (c != 0) & (c >> 15 == 0)
    add = (c >> 15 == 1)
    reg = fr[112:136, x:x + w]
    reg[op] = pal[s][op]
    reg[add] = np.minimum(reg[add] + pal[s][add], 248)
    fr[:, 496:] = (0, 32, 128)
    x0, y0, x1, y1 = crop
    return fr[y0:y1, x0:x1]

def sheet(path, cols=3):
    """JP and EN caption scenes for every distinct location, one cell each."""
    cells = []
    for jp, es in ENTRIES.items():
        for e in es:
            a = scene(orig_tim(e), 384, 96); b = scene(new_tim(e), X_HOLD, W)
            cells.append(np.concatenate([a, np.zeros((a.shape[0], 4, 3)), b], 1))
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    out = np.full((rows * (h + 3), cols * (w + 6), 3), 40.)
    for k, c in enumerate(cells):
        r, q = divmod(k, cols)
        out[r * (h + 3):r * (h + 3) + h, q * (w + 6):q * (w + 6) + w] = c
    Image.fromarray(out.astype(np.uint8)).save(path)

if __name__ == '__main__':
    if '--png' in sys.argv:
        d = sys.argv[sys.argv.index('--png') + 1]; os.makedirs(d, exist_ok=True)
        sheet(os.path.join(d, 'captions_sheet.png'))
        for jp, en in load_tl().items():
            _, w, sq = line_alpha(en)
            print(f'{en}: {w:.0f} px' + (f', squeezed {sq:.3f}' if sq < 1 else ''))
