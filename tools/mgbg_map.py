# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Mid-month location map labels in English (after-class "Move" map).
Patch module for tools/build_mgbg.py; reads tl/MAP.txt (locations 0-5).

Pixels: a 4bpp 512x256 TIM at offset 0x2b800 of one entry in each of the
11 MGBGM archives (ENTRIES, byte-identical copies). The game uploads it to
VRAM (512,0), CLUTs at (0,484). The labels use CLUT 6 (a grey ramp) in
tpage (512,0): JP 教室 グラウンド 屋上 / 体育館 保健室 部室棟 at u 0-167,
v 168-199, two rows of 16 px. The red dot at u 168-183, v 168-183 stays.
u 184-255 of v 168-199 and all of v 200-255 are free (index 0).

Records: MGAME (base 0x800a2000), one 16-byte entry per location at
0x800a406c: bubble x,y; frame dx,dy; label dx,dy; dot x,y (halfwords).
Label width bytes at 0x800a40cc + i, label u,v bytes at 0x800a40d6 + 4i.
Label height 16 is fixed in code (0x800bba20). patch_mgame() rewrites
widths, uv and the label offset; hook.py calls it on MGAME.BIN.

Layout: each EN label gets a new 16 px tall cell packed into the free
texels. It is centred on its bubble: the ink centre sits at the ellipse
centre, bubble x + ECX[i], at the JP label row (bubble y + 40), on the bubble's bottom
rim. The table is per location (index = location id), so the six places
are the same on every day; a day with fewer places just skips some.
The JP labels hung off the lower-right rim instead. Style (STYLE, as the JP): dark
italic capitals (Barlow Semi Condensed ExtraBold Italic, OFL, cap 11 px)
with a 1 px light edge and a dark drop shadow down-right, quantised to
CLUT 6 (index 5 has the STP bit and is not used). The other STYLES are the
variants that were compared (private/re/map/findings.org).

Usage: python3 tools/mgbg_map.py --png DIR [--styles A,B]   previews only
(needs the screenshots in private/re/map/emu/b and emu/c).
"""
import os, sys, struct, functools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath

ENTRIES = [('MGBGM01.BZZ', 18), ('MGBGM02.BZZ', 18), ('MGBGM04.BZZ', 6),
           ('MGBGM05.BZZ', 5), ('MGBGM06.BZZ', 5), ('MGBGM07.BZZ', 5),
           ('MGBGM08.BZZ', 5), ('MGBGM09.BZZ', 11), ('MGBGM10.BZZ', 10),
           ('MGBGM11.BZZ', 10), ('MGBGM12.BZZ', 15)]
TIM_OFF = 0x2b800
CLUT = 6
OVL = 0x800a2000
LOC = 0x800a406c          # 6 x 16 bytes
WID = 0x800a40cc          # 6 width bytes
UV = 0x800a40d6           # u,v bytes, stride 4
JP_UV = [(0, 168, 40), (40, 168, 88), (128, 168, 40), (0, 184, 56), (56, 184, 56), (112, 184, 56)]
H = 16
DOT = (168, 168, 16, 16)  # u, v, w, h kept
FREE_V = (168, 184, 200, 216, 232)
# Ellipse centre - bubble x, measured on screenshots. The ellipse spans
# bubble x + 2..110 where the frame dx is negative (tail to the left) and
# bubble x + 0..108 where it is 0 (Gym, Clubhouse: tail below).
ECX = [56.5, 56.5, 56.5, 54.5, 56.5, 54.5]
XMIN, XMAX = 16, 496      # inside the blue panel frame

BARLOW = 'BarlowSemiCondensed-ExtraBoldItalic.ttf'
MPLUS = 'MPLUS1p-ExtraBold.ttf'
STYLES = {
    # font, cap px, shear (for upright fonts), tracking px, all caps, edge mode
    'barlow': dict(font=BARLOW, cap=10, shear=0.0, track=0.3, caps=False, mode='ol'),
    'barlow-ex': dict(font=BARLOW, cap=10, shear=0.0, track=0.3, caps=False, mode='ex'),
    'barlow-caps': dict(font=BARLOW, cap=11, shear=0.0, track=0.6, caps=True, mode='ol'),
    'barlow-caps-ex': dict(font=BARLOW, cap=11, shear=0.0, track=0.6, caps=True, mode='ex'),
    'mplus-caps': dict(font=MPLUS, cap=11, shear=0.22, track=0.3, caps=True, mode='ol'),
    'mplus-caps-ex': dict(font=MPLUS, cap=11, shear=0.22, track=0.3, caps=True, mode='ex'),
    'barlow-caps-emb': dict(font=BARLOW, cap=11, shear=0.0, track=0.8, caps=True, mode='emb'),
    'barlow-caps-inv': dict(font=BARLOW, cap=11, shear=0.0, track=0.8, caps=True, mode='inv'),
    'mplus-caps-emb': dict(font=MPLUS, cap=11, shear=0.22, track=0.5, caps=True, mode='emb'),
    'mplus-caps-inv': dict(font=MPLUS, cap=11, shear=0.22, track=0.5, caps=True, mode='inv'),
}
STYLE = 'barlow-caps-inv'
SS = 4

def load_tl(path='tl/MAP.txt'):
    out = []; jp = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            out.append((jp, line[4:].strip())); jp = None
    assert len(out) == 6, len(out)
    return out

# ---------------------------------------------------------------- TIM
@functools.lru_cache(None)
def orig_entry():
    import lz, arc
    name, i = ENTRIES[2]
    d = open('private/disc/DAT2/' + name, 'rb').read()
    o, s = arc.toc(d)[i]
    return lz.decompress(d[o:o + s])[0]

def tim_layout(d):
    """-> (clut row 6 as 16 u16, pixel offset, row bytes)"""
    assert struct.unpack_from('<II', d, TIM_OFF) == (0x10, 0x8)
    cl = struct.unpack_from('<I', d, TIM_OFF + 8)[0]
    cw = struct.unpack_from('<H', d, TIM_OFF + 16)[0]
    clut = np.frombuffer(d, '<u2', 16, TIM_OFF + 20 + 2 * 16 * CLUT)
    q = TIM_OFF + 8 + cl
    w, h = struct.unpack_from('<HH', d, q + 8)
    assert (w, h) == (128, 256)
    return clut.copy(), q + 12, w * 2

def unpack(d):
    _, p, rb = tim_layout(d)
    b = np.frombuffer(d, np.uint8, rb * 256, p).reshape(256, rb)
    return np.stack([b & 15, b >> 4], -1).reshape(256, rb * 2)

def pack(d, idx):
    _, p, rb = tim_layout(d)
    b = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    out = bytearray(d); out[p:p + rb * 256] = b.tobytes()
    return bytes(out)

def greys(clut):
    return np.array([(int(c) & 31) << 3 for c in clut], float)

# ---------------------------------------------------------------- drawing
def font(path, size): return fontpath.truetype(path, size)

@functools.lru_cache(None)
def size_for_cap(path, cap):
    s = cap * SS
    for _ in range(4):
        b = font(path, s).getbbox('H')
        s = int(round(s * cap * SS / (b[3] - b[1])))
    return s

def alpha(text, style):
    """Letter coverage at 1x, H rows; cap top at row 1, baseline row 1+cap."""
    st = STYLES[style]
    path, cap, shear, track = st['font'], st['cap'], st['shear'], st['track']
    if st['caps']: text = text.upper()
    f = font(path, size_for_cap(path, cap))
    asc = f.getbbox('H')[1]
    W = (int(f.getlength(text) + track * SS * len(text)) // SS + 16) * SS
    im = Image.new('L', (W, H * SS), 0); dr = ImageDraw.Draw(im)
    x = 4 * SS
    for ch in text:
        dr.text((x, 1 * SS - asc), ch, font=f, fill=255)
        x += f.getlength(ch) + track * SS
    if shear:
        base = (1 + cap) * SS
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * base, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(im, float) / 255
    a = a.reshape(H, SS, -1, SS).mean((1, 3))
    cols = np.nonzero(a.max(0) > 0.06)[0]
    return a[:, cols[0]:cols[-1] + 1]

def shift(a, dx, dy):
    o = np.zeros_like(a)
    h, w = a.shape
    o[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return o

def label(text, style, gr):
    """-> CLUT index array (H x w) for one label."""
    a = alpha(text, style)
    a = np.pad(a, ((0, 0), (1, 3)))
    a[a < 0.12] = 0
    hard = np.clip((a - 0.15) / 0.5, 0, 1)
    mode = STYLES[style]['mode']
    rows = np.arange(H)[:, None]
    if mode in ('emb', 'inv'):              # dark letters, light edge
        hl = np.zeros_like(a)
        ring = ((-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1))
        for dx, dy in (ring if mode == 'inv' else ring[:3] + ring[6:]):
            hl = np.maximum(hl, shift(hard, dx, dy))
        sh = np.zeros_like(a)
        for dx, dy in ((1, 1), (2, 1), (2, 2)):
            sh = np.maximum(sh, shift(np.maximum(hard, hl), dx, dy))
        light = np.interp(rows, [0, 6, 15], [248, 232, 192]) + 0 * a
        dark = np.interp(rows, [1, 14], [40, 8]) + 0 * a
        col = np.where(sh > 0, 8.0, 0.0)
        col = np.where(hl > 0.3, light, col)
        col = np.where(a > 0, dark * a + col * (1 - a) + (a == 0) * 0, col)
        col = np.where((a > 0) & (hl <= 0.3) & (sh == 0), dark, col)
        tot = np.maximum.reduce([a * 1.5, hl, sh * 0.9])
    else:
        ol = np.zeros_like(a)
        if mode == 'ol':                    # 1 px outline, shadow 1-2 px
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ol = np.maximum(ol, shift(hard, dx, dy) * (0.85 if dx and dy else 1))
            sh = np.maximum(shift(ol, 1, 1), shift(ol, 2, 1))
        else:                               # extrusion down-right
            for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
                ol = np.maximum(ol, shift(hard, dx, dy) * (0.6 if dx < 0 or dy < 0 else 1))
            sh = np.zeros_like(a)
            for dx, dy in ((2, 1), (2, 2), (3, 2), (1, 2)):
                sh = np.maximum(sh, shift(hard, dx, dy))
        fill = np.interp(rows, [1, 4, 11, 14], [248, 240, 192, 176]) + 0 * a
        dark_ol, dark_sh = 16.0, 8.0
        col = fill * a + dark_ol * (1 - a)
        vis = np.maximum(a, ol)
        col = np.where(vis > 0.35, col, dark_sh)
        tot = np.maximum(vis, sh * 0.9)
    idx = np.zeros(a.shape, int)
    cand = [i for i in range(1, 16) if i != 5]
    g = gr[cand]
    on = tot > 0.4
    idx[on] = np.array(cand)[np.abs(col[on][:, None] - g[None]).argmin(1)]
    cols = np.nonzero(idx.any(0))[0]
    return idx[:, :cols[-1] + 1]

# ---------------------------------------------------------------- layout
def pack_cells(widths):
    """Greedy: place each width in the free texel rows. -> [(u, v)]"""
    spans = []
    for v in FREE_V:
        if v == DOT[1]: spans += [[v, 0, DOT[0]], [v, DOT[0] + DOT[2], 256]]
        else: spans.append([v, 0, 256])
    out = []
    for w in widths:
        for s in spans:
            if s[2] - s[1] >= w:
                out.append((s[1], s[0])); s[1] += w + 1; break
        else: raise SystemExit(f'mgbg_map: no free texels for a {w} px label')
    return out

def mgame_locs(d):
    o = LOC - OVL
    return [struct.unpack_from('<8h', d, o + 16 * i) for i in range(6)]

@functools.lru_cache(None)
def build(style=STYLE):
    """-> (new TIM index array, [(u, v, w, dx, dy)] per location)"""
    d = orig_entry()
    clut, _, _ = tim_layout(d)
    gr = greys(clut)
    idx = unpack(d).copy()
    # JP labels sit in rows 168-199, x 0-167 of the TIM's left half.
    assert not idx[168:256, 184:256].any() and not idx[200:256, :256].any()
    idx[168:200, 0:168] = 0
    labs = [label(en, style, gr) for jp, en in load_tl()]
    cells = pack_cells([l.shape[1] for l in labs])
    mg = open('private/disc/DAT1/MGAME.BIN', 'rb').read()
    recs = []
    for i, (l, (u, v)) in enumerate(zip(labs, cells)):
        w = l.shape[1]
        idx[v:v + H, u:u + w] = l
        bx, by, fx, fy, lx, ly, _, _ = mgame_locs(mg)[i]
        cols = np.nonzero(l.any(0))[0]
        x = bx + int(np.floor(ECX[i] - (cols[0] + cols[-1]) / 2 + 0.5))
        assert XMIN <= x and x + w <= XMAX, (i, x, w)
        recs.append((u, v, w, x - bx, ly))
    assert (idx[DOT[1]:DOT[1] + 16, DOT[0]:DOT[0] + 16] == unpack(d)[DOT[1]:DOT[1] + 16, DOT[0]:DOT[0] + 16]).all()
    return idx, recs

def patch_mgame(d):
    """MGAME.BIN bytes -> bytes with the EN label records."""
    d = bytearray(d)
    for i, (u, v, w), in enumerate(JP_UV):
        assert d[WID - OVL + i] == w and tuple(d[UV - OVL + 4 * i:UV - OVL + 4 * i + 2]) == (u, v), i
    for i, (u, v, w, dx, dy) in enumerate(build()[1]):
        d[WID - OVL + i] = w
        d[UV - OVL + 4 * i] = u; d[UV - OVL + 4 * i + 1] = v
        struct.pack_into('<hh', d, LOC - OVL + 16 * i + 8, dx, dy)
    return bytes(d)

def patcher(data):
    # Bytes before TIM_OFF may differ: mgbg_captions.py patches the location
    # TIM at 0xe000 first. Check the map TIM header and CLUTs instead.
    assert data[TIM_OFF:TIM_OFF + 0x400] == orig_entry()[TIM_OFF:TIM_OFF + 0x400], 'mgbg_map: unexpected entry'
    assert unpack(data).tobytes() == unpack(orig_entry()).tobytes(), 'mgbg_map: map TIM differs from MGBGM04/006'
    return pack(data, build()[0])

PATCHES = [(a, i, patcher) for a, i in ENTRIES]

# ---------------------------------------------------------------- preview
def composite(bg, idx, recs, gr, mg, sel=0):
    """Draw labels on a screenshot (screen coords = sprite coords)."""
    out = bg.copy()
    for i, (u, v, w, dx, dy) in enumerate(recs):
        bx, by = mgame_locs(mg)[i][:2]
        c = idx[v:v + H, u:u + w]
        g = gr[c] * (1.0 if i == sel else 0.5)
        m = c > 0
        x, y = bx + dx, by + dy
        reg = out[y:y + H, x:x + w]
        reg[m] = g[m][:, None]
    return out

def preview(dirn, styles=None):
    mg = open('private/disc/DAT1/MGAME.BIN', 'rb').read()
    d = orig_entry(); gr = greys(tim_layout(d)[0])
    clean = np.asarray(Image.open('private/re/map/emu/c/clean.png').convert('RGB')).astype(float)
    jp = np.asarray(Image.open('private/re/map/emu/b/s160.png').convert('RGB')).astype(float)
    crop = lambda a: a[110:320, 0:512]
    rows = [crop(jp)]
    for st in styles or [STYLE]:
        build.cache_clear()
        idx, recs = build(st)
        rows.append(crop(composite(clean, idx, recs, gr, mg)))
    sep = np.full((3, 512, 3), 255.)
    img = np.concatenate([x for r in rows for x in (r, sep)][:-1], 0)
    Image.fromarray(img.astype(np.uint8)).save(os.path.join(dirn, 'map_preview.png'))
    build.cache_clear()
    idx, recs = build()
    tex = Image.fromarray(gr[idx[160:256, 0:256]].astype(np.uint8))
    tex.resize((512, 192), Image.NEAREST).save(os.path.join(dirn, 'map_texels.png'))
    return recs

if __name__ == '__main__':
    if '--png' in sys.argv:
        dn = sys.argv[sys.argv.index('--png') + 1]; os.makedirs(dn, exist_ok=True)
        st = sys.argv[sys.argv.index('--styles') + 1].split(',') if '--styles' in sys.argv else None
        print(preview(dn, st))
