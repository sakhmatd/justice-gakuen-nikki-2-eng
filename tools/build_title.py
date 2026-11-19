# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the mode select screen graphics (DAT2/MENU.BIZ) in English.

Reads tl/TITLE.txt (ID: EN). Writes private/build/MENU.BIZ (LZ, for
patch_bin) and private/build/MENU.BIZ.lz (decompressed). MODESEL is not
changed: every EN piece fits the JP rectangle.
--dry writes nothing to private/build.
--png DIR writes the EN pieces (2x, with the JP piece on the left) to DIR.
RE: private/re/menu/findings.org.

MENU.BIZ (decompressed 0x9f800 bytes):
- TIM2 at 0x22800: CLUTs at VRAM (0,482) 512x2, 4bpp 1280x256 pixels at
  0x23020 (640 bytes per row). Column labels, selected labels, the points
  label and the logo live here.
- TIM 3+m at 0x4b800 + m*0x2800: description of mode m, 4bpp 176x112, no
  CLUT (drawn with CLUT (128,482), 166 px wide).
- TIM 0 is a copy of the MOJI2 font TIM; extras_help.patch_menu() paints
  the EN Extras help panel into it (same paint as build_extras.py's
  MOJI2.TIZ).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
import lz, build_logo, extras_help

SRC = 'private/unpacked/MENU.BIZ.lz'
MODESEL = 'private/disc/DAT1/MODESEL.BIN'
SLOT = 188 * 2048                       # disc allocation of MENU.BIZ
T2 = 0x22800
T2CLUT = T2 + 20                        # 512x2 entries, VRAM (0,482)
T2PIX, T2W, T2H = T2CLUT + 512 * 2 * 2 + 12, 1280, 256
DESC = 0x4b800                          # TIM 3: + m * 0x2800, pixels at +0x14
DESC_W, DESC_H, DESC_VIS = 176, 112, 166

FONT = 'DejaVuSansCondensed-Bold.ttf'   # as in N1
DFONT = 'RobotoCondensed-Medium.ttf'

# screen row of each mode in the label column; rows 0-7 are 104x24 at TIM2
# (888,64+24r), rows 8-13 are 96x24 at (1024,24(r-8)). Only the first 96x23
# is shown on the panel tab, so EN text stays inside it.
ROW = {6: 0, 0: 1, 1: 2, 10: 3, 8: 4, 9: 5, 7: 6, 13: 7, 2: 8, 11: 9, 12: 10,
       5: 11, 4: 12, 3: 13}
COL_W, COL_H = 96, 23
POINTS = (536, 224, 96, 20)

def col_rect(m):
    r = ROW[m]
    return (888, 64 + 24 * r, 104, 24) if r < 8 else (1024, 24 * (r - 8), 96, 24)

def sel_records(path=MODESEL):
    """m -> (x, y, w, h) in TIM2 pixels, screen (x, y), from MODESEL 0x1ac."""
    M = open(path, 'rb').read()
    out = {}
    for m in range(14):
        f, z, tx, ty, cx, cy, u, v, w, h, sx, sy = struct.unpack_from('<12h', M, 0x1ac + 24 * m)
        out[m] = ((tx - 512) * 4 + u, v, w, h), (sx, sy)
    return out

def load_tl(path='tl/TITLE.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def unpack(d, off, w, h):
    raw = np.frombuffer(bytes(d[off:off + w * h // 2]), np.uint8).reshape(h, w // 2)
    px = np.empty((h, w), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def pack(d, off, px):
    d[off:off + px.size // 2] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def palette(d, cx, cy):
    """16 RGB rows (int) of the TIM2 CLUT at VRAM (cx, cy)."""
    v = np.array(struct.unpack_from('<16H', d, T2CLUT + 2 * ((cy - 482) * 512 + cx)))
    return np.stack([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], -1).astype(float)

# ---------------------------------------------------------------- text

def cap_size(font, cap):
    """Smallest font size whose 'H' is at least `cap` px tall."""
    for s in range(6, 60):
        b = fontpath.truetype(font, s).getbbox('H', anchor='ls')
        if -b[1] >= cap: return s
    return 60

def line_alpha(text, font, size, sx=1.0, ss=4, track=0.0):
    """Coverage (h, w) of one line and its baseline row. The baseline falls
    on a pixel boundary; sx squeezes horizontally; track adds px per gap."""
    f = fontpath.truetype(font, size * ss)
    l, t, r, b = f.getbbox(text, anchor='ls')
    B = -(t // ss) + 1                  # rows above the baseline
    H = B + -(-b // ss) + 1
    extra = int(round(track * ss * max(0, len(text) - 1)))
    W = r - l + 2 * ss + extra
    im = Image.new('L', (W, H * ss), 0)
    dr = ImageDraw.Draw(im)
    if track:
        x = ss - l
        for ch in text:
            dr.text((x, B * ss), ch, font=f, fill=255, anchor='ls')
            x += f.getlength(ch) + track * ss
    else:
        dr.text((ss - l, B * ss), text, font=f, fill=255, anchor='ls')
    w = max(1, int(round(W * sx / ss)))
    a = np.asarray(im.resize((w, H), Image.BOX), float) / 255.0
    xs = np.nonzero(a.max(0) > 0.05)[0]
    a = a[:, xs[0]:xs[-1] + 1] if len(xs) else a
    return a, B

def place(dst, a, x0, y0):
    """Max-blend a into dst at (x0, y0); True if ink falls outside."""
    h, w = a.shape
    ys, xs = max(0, -y0), max(0, -x0)
    ye, xe = min(h, dst.shape[0] - y0), min(w, dst.shape[1] - x0)
    if ye > ys and xe > xs:
        sub = dst[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
        np.maximum(sub, a[ys:ye, xs:xe], out=sub)
    inside = np.zeros_like(a, bool); inside[ys:ye, xs:xe] = True
    return bool((a[~inside] > 0.05).any())

def fit_line(text, font, cap, maxw, minsx=0.82):
    """Size and squeeze so `text` is at most maxw px wide; shrink the cap
    height only when a squeeze below minsx would be needed."""
    while True:
        size = cap_size(font, cap)
        a, _ = line_alpha(text, font, size)
        sx = min(1.0, maxw / a.shape[1])
        if sx >= minsx or cap <= 7: return size, sx, cap
        cap -= 1

def label_alpha(lines, w, h, font, cap, pad=1, minsx=0.82, base=None, pitch=None):
    """Centred label coverage (h, w), plus the squeeze and cap used. pad
    keeps room for the outline. base: baseline row of the first line."""
    size, sx, cap = min((fit_line(t, font, cap, w - 2 * pad, minsx) for t in lines),
                        key=lambda r: (r[2], r[1]))
    out = np.zeros((h, w))
    pitch = pitch or cap + 4
    if base is None: base = (h - cap - (len(lines) - 1) * pitch) // 2 + cap
    clip = False
    for i, t in enumerate(lines):
        a, B = line_alpha(t, font, size, sx)
        clip |= place(out, a, (w - a.shape[1]) // 2, base + i * pitch - B)
    return out, sx, cap, clip

# ---------------------------------------------------------------- colour

def grow(a, r=1, diag=True):
    o = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not diag and dx and dy: continue
            s = np.zeros_like(a)
            ys, ye = max(0, dy), a.shape[0] + min(0, dy)
            xs, xe = max(0, dx), a.shape[1] + min(0, dx)
            s[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
            o = np.maximum(o, s)
    return o

def shift(a, dx, dy):
    s = np.zeros_like(a)
    s[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return s

def ramp(cols, h, y0=0, y1=None):
    """(h, 3) colours, top->bottom through `cols` between rows y0 and y1."""
    y1 = h if y1 is None else y1
    t = np.clip((np.arange(h) - y0) / max(1, y1 - y0 - 1), 0, 1) * (len(cols) - 1)
    i = np.minimum(t.astype(int), len(cols) - 2) if len(cols) > 1 else np.zeros(h, int)
    f = (t - i)[:, None]
    c = np.array(cols, float)
    return c[i] * (1 - f) + c[np.minimum(i + 1, len(cols) - 1)] * f if len(cols) > 1 else np.repeat(c, h, 0)

def compose(layers, bg, shape):
    """layers: (coverage, colour) bottom->top; colour (3,) or (h, 3)."""
    h, w = shape
    rgb = np.broadcast_to(np.array(bg, float), (h, w, 3)).copy()
    for cov, col in layers:
        col = np.array(col, float)
        col = col[:, None, :] if col.ndim == 2 else col[None, None, :]
        rgb = rgb * (1 - cov[..., None]) + col * cov[..., None]
    return rgb

def quantise(rgb, pal, allowed, solid):
    """Nearest allowed index per pixel; 0 where solid is False."""
    idx = np.array(allowed)
    dist = ((rgb[..., None, :] - pal[idx][None, None]) ** 2 * (2, 3, 1)).sum(-1)
    out = idx[dist.argmin(-1)].astype(np.uint8)
    out[~solid] = 0
    return out

# ---------------------------------------------------------------- pieces

COL_BG = (80, 224, 80)                  # column panel green
SEL_BG = (248, 200, 112)                # highlight bar under the selected label
PTS_BG = (248, 232, 88)                 # notebook yellow
DESC_BG = (248, 240, 104)

# Short forms for labels too long for their rectangle; a tl key labelN_short
# overrides them. "Hot-Blooded Youth Diary 2" fits 96x24 only in two lines of
# 7 px capitals, which are hard to read.
SHORT = {6: 'Youth Diary 2'}

def label_text(tl, m):
    return tl.get(f'label{m}_short', SHORT.get(m, tl[f'label{m}'])).split('|')

def col_label(lines, d, cap=12):
    """Blue text (12), light yellow-green outline (7..4), green outer edge
    (3, 2); 8-11 and 13-15 are the blends, as in the JP labels."""
    pal = palette(d, 16, 482)
    h, w = 24, 104
    if len(lines) == 1:
        a, sx, cap, clip = label_alpha(lines, COL_W, COL_H, FONT, cap, pad=2, minsx=0.78)
    else:
        a, sx, cap, clip = label_alpha(lines, COL_W, COL_H, FONT, 7, pad=2, base=9, pitch=11)
    a = np.pad(a, ((0, h - COL_H), (0, w - COL_W)))
    o = grow(a)
    o2 = grow(o, 1, diag=False) * 0.5
    rgb = compose([(o2 * (1 - o), pal[3]), (o, pal[7]), (a, pal[12])], COL_BG, a.shape)
    img = quantise(rgb, pal, range(2, 16), o > 0.3)
    return img, sx, cap, clip

def sel_label(lines, w, h, d, cap=15):
    """Red fill (1) with AA 2/10/11, dark drop shadow (3, blends 7-9, 13),
    orange-to-yellow outline (5, 4, 12 top; 15, 14 bottom)."""
    pal = palette(d, 112, 482)
    if len(lines) == 1:
        a, sx, cap, clip = label_alpha(lines, w, h - 2, FONT, cap, pad=2)
    else:
        a, sx, cap, clip = label_alpha(lines, w, h - 2, FONT, 9, pad=2, base=12, pitch=13)
    a = np.pad(a, ((0, 2), (0, 0)))
    sh = np.maximum(shift(a, 1, 1), shift(a, 2, 2))
    o = grow(np.maximum(a, sh))
    ys = np.nonzero(o.max(1) > 0.3)[0]
    ycol = ramp([pal[5], pal[15], pal[14]], h, ys[0], ys[-1] + 1)
    rgb = compose([(o, ycol), (sh, pal[3]), (a, pal[1])], SEL_BG, a.shape)
    img = quantise(rgb, pal, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], o > 0.3)
    return img, sx, cap, clip

def pts_label(text, d, cap=12):
    """Red text (1, 2), yellow outline (15), blends 8-14."""
    x, y, w, h = POINTS
    pal = palette(d, 272, 483)
    a, sx, cap, clip = label_alpha([text], w, h, FONT, cap, pad=2)
    o = grow(a)
    rgb = compose([(o, pal[15]), (a, pal[1])], PTS_BG, a.shape)
    img = quantise(rgb, pal, [1, 2] + list(range(8, 16)), o > 0.3)
    return img, sx, cap, clip

DESC_GREYS = [1, 13, 14, 15, 9, 3, 4, 10, 11, 5, 12, 6]   # dark -> light

def wrap(text, font, size, width):
    f = fontpath.truetype(font, size)
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            if f.getlength(t) <= width or not cur: cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

DESC_CAP, DESC_PITCH = 10, (20, 15)       # capital height, max/min pitch
DESC_TEXT_W = 162

def desc_img(text, d, cap=DESC_CAP, font=None, pitches=DESC_PITCH):
    """Dark grey text (1) with grey antialias, left aligned, block centred
    vertically. One capital height for all modes; the line pitch shrinks
    from pitches[0] to fit 112 rows, as the JP pitch does (25 to 19)."""
    font = font or DFONT
    pal = palette(d, 128, 482)
    size = cap_size(font, cap)
    lines = wrap(text, font, size, DESC_TEXT_W)
    f = fontpath.truetype(font, size)
    desc = -(-f.getbbox('gjpqy', anchor='ls')[3])
    n = len(lines)
    pitch = pitches[0] if n < 2 else min(pitches[0], (DESC_H - cap - desc) // (n - 1))
    need = (n - 1) * pitch + cap + desc
    a = np.zeros((DESC_H, DESC_W))
    base = (DESC_H - need) // 2 + cap
    clip = need > DESC_H or pitch < pitches[1]
    for i, t in enumerate(lines):
        al, B = line_alpha(t, font, size)
        clip |= place(a[:, :DESC_VIS], al, 1, base + i * pitch - B)
    lv = pal[DESC_GREYS][:, 0]
    g = 0x28 * a + 0xd8 * (1 - a)
    img = np.array(DESC_GREYS, np.uint8)[np.abs(g[..., None] - lv).argmin(-1)]
    img[a < 0.22] = 0
    return img, (cap, pitch, n, need, clip, lines)

# ---------------------------------------------------------------- build

def build(pngdir=None, write=True):
    tl = load_tl()
    d = bytearray(open(SRC, 'rb').read())
    assert len(d) == 0x9f800
    jp = bytes(d)
    t2 = unpack(d, T2PIX, T2W, T2H)
    t2jp = t2.copy()
    sels = sel_records()
    problems = []
    prev = []                            # (name, jp idx, en idx, (cx, cy))

    for m in range(14):
        lines = label_text(tl, m)
        x, y, w, h = col_rect(m)
        img, sx, cap, clip = col_label(lines, d)
        t2[y:y + h, x:x + w] = img[:h, :w]
        prev.append((f'label{m:02d}', t2jp[y:y + h, x:x + w], img[:h, :w], (16, 482)))
        if clip or sx < 0.78: problems.append(f'label{m}: clip {clip} squeeze {sx:.2f}')
        (x, y, w, h), _ = sels[m]
        img, sx, cap, clip = sel_label(lines, w, h, d)
        t2[y:y + h, x:x + w] = img
        prev.append((f'sel{m:02d}', t2jp[y:y + h, x:x + w], img, (112, 482)))
        if clip or sx < 0.82: problems.append(f'sel{m}: clip {clip} squeeze {sx:.2f}')

    x, y, w, h = POINTS
    img, sx, cap, clip = pts_label(tl['points'], d)
    t2[y:y + h, x:x + w] = img
    prev.append(('points', t2jp[y:y + h, x:x + w], img, (272, 483)))
    pack(d, T2PIX, t2)

    for m in range(14):
        off = DESC + m * 0x2800 + 0x14
        old = unpack(d, off, DESC_W, DESC_H)
        img, info = desc_img(tl[f'desc{m}'], d)
        pack(d, off, img)
        prev.append((f'desc{m:02d}', old, img, (128, 482)))
        cap, pitch, n, need, clip, _ = info
        print(f'desc{m}: {n} lines, cap {cap}, pitch {pitch}, {need} rows')
        if clip: problems.append(f'desc{m}: {need} rows > {DESC_H}')

    build_logo.patch('menu', d)          # EN diary logo (120x72)
    for p in extras_help.problems(extras_help.patch_menu(d)):   # Extras help panel copy
        problems.append(p)
    assert len(d) == len(jp)
    c = lz.compress(bytes(d), name='MENU.BIZ')
    if len(c) > SLOT: problems.append(f'MENU.BIZ: {len(c)} bytes > slot {SLOT}')
    assert lz.decompress(c)[0] == bytes(d)
    if write:
        os.makedirs('private/build', exist_ok=True)
        open('private/build/MENU.BIZ.lz', 'wb').write(d)
        open('private/build/MENU.BIZ', 'wb').write(c)
    if pngdir: write_previews(d, prev, pngdir)
    for p in problems: print('PROBLEM', p)
    print(f'MENU.BIZ: {len(c)} bytes (slot {SLOT}, JP 383296), problems {len(problems)}')
    return d, prev, problems

def to_rgb(idx, pal, bg=(255, 0, 255)):
    rgb = pal[idx].astype(np.uint8)
    rgb[idx == 0] = bg
    return rgb

def write_previews(d, prev, out):
    """One 2x PNG per piece (JP left, EN right, on the piece's background
    colour), plus labels.png and desc.png sheets of all pieces."""
    os.makedirs(out, exist_ok=True)
    bgs = {'label': COL_BG, 'sel': SEL_BG, 'points': PTS_BG, 'desc': DESC_BG}
    pairs = {}
    for name, jp, en, (cx, cy) in prev:
        pal = palette(d, cx, cy)
        bg = bgs[name.rstrip('0123456789')]
        h = max(jp.shape[0], en.shape[0])
        a = np.full((h, jp.shape[1] + en.shape[1] + 4, 3), 64, np.uint8)
        a[:jp.shape[0], :jp.shape[1]] = to_rgb(jp, pal, bg)
        a[:en.shape[0], jp.shape[1] + 4:] = to_rgb(en, pal, bg)
        a = np.repeat(np.repeat(a, 2, 0), 2, 1)
        Image.fromarray(a).save(f'{out}/{name}.png')
        pairs[name] = a
    def sheet(names, ncol):
        ims = [pairs[n] for n in names]
        cw = max(i.shape[1] for i in ims) + 8; rh = max(i.shape[0] for i in ims) + 8
        nrow = -(-len(ims) // ncol)
        sh = np.full((nrow * rh, ncol * cw, 3), 32, np.uint8)
        for k, i in enumerate(ims):
            y, x = (k % nrow) * rh, (k // nrow) * cw
            sh[y:y + i.shape[0], x:x + i.shape[1]] = i
        return sh
    order = sorted(range(14), key=lambda m: ROW[m])
    Image.fromarray(sheet([f'label{m:02d}' for m in order] + [f'sel{m:02d}' for m in order]
                          + ['points'], 2)).save(f'{out}/labels.png')
    Image.fromarray(sheet([f'desc{m:02d}' for m in range(14)], 2)).save(f'{out}/desc.png')

if __name__ == '__main__':
    out = sys.argv[sys.argv.index('--png') + 1] if '--png' in sys.argv else None
    build(out, write='--dry' not in sys.argv)
