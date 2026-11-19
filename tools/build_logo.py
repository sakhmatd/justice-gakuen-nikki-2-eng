# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English diary-mode logo (熱血青春日記2 -> "HOT-BLOODED YOUTH DIARY 2").

Three copies of the logo are redrawn (private/re/logo/findings.org):
- Big: MA_TIT.BIZ TIM0 tpage 896 u0 v0 256x136, CLUT 9, drawn 1:1 at
  (240,300) on the diary title and Continue screens.
- Small: same TIM, u0 v136 96x64, CLUT 8, at (410,30). The game title
  (私立ジャスティス学園, rows 0-24) is kept; the diary part below is new.
- Mode select: MENU.BIZ TIM2 (1120,0) 120x72, CLUT (160,482), at
  (368,16). The game title (rows 0-27) is kept.

Design (same family as the JP logo): tall condensed lettering (Anton)
fanned out and arched, a white -> yellow -> orange -> red fill from top to
bottom, 1 px light bevel on top edges and a magenta/brown inner edge at
the bottom, a black outline with a drop shadow and a thin dark rim, and a
big red "2" (Russo One) under the arch. "HOT-BLOODED" is a green tracked
line on top, in the place of the JP romaji. No CLUT is changed: each
piece is drawn with indices that already hold the right colour.

build_mgtit.py and build_title.py call patch() before they compress.
A standalone run patches private/build/MA_TIT.BIZ and MENU.BIZ(.lz) in
place (the same result; it can run any number of times); --png writes JP
over EN previews to private/re/logo/en/. Fonts: Anton and Russo One (OFL).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz

ANTON = 'Anton-Regular.ttf'          # OFL
RUSSO = 'RussoOne-Regular.ttf'   # OFL
SS = 8                                   # supersampling of text masks
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9],
                  [15, 7, 13, 5]]) / 16 - 0.5

# ------------------------------------------------------------ text masks

def text_ss(text, font, cap, track=0.0, ss=SS):
    """Supersampled coverage of one line, cropped to the cap box (rows from
    cap top to baseline) and the ink columns. cap in 1x px, track in 1x px
    per gap. Descenders and overshoot are kept in a margin below/above."""
    size = 4
    while True:
        f = fontpath.truetype(font, size)
        b = f.getbbox('H', anchor='ls')
        if -b[1] >= cap * ss: break
        size += 1
    capt = -f.getbbox('H', anchor='ls')[1]
    adv = [f.getlength(c) for c in text]
    W = int(sum(adv) + track * ss * (len(text) - 1) + 4 * capt)
    pad = capt // 4
    im = Image.new('L', (W, capt + 2 * pad), 0)
    dr = ImageDraw.Draw(im)
    x = capt
    for c, a in zip(text, adv):
        dr.text((x, pad + capt), c, font=f, fill=255, anchor='ls')
        x += a + track * ss
    a = np.asarray(im, float) / 255
    cols = np.where(a.max(0) > 0)[0]
    a = a[:, cols[0]:cols[-1] + 1]
    rows = np.where(a.max(1) > 0)[0]
    # crop to cap box, but keep overshoot rows (round letters)
    top, bot = min(pad, rows[0]), max(pad + capt, rows[-1] + 1)
    return a[top:bot]

def resize_ss(a, w, h):
    """Resize a supersampled mask to w x h (1x px) at SS resolution."""
    im = Image.fromarray((a * 255).astype(np.uint8))
    return np.asarray(im.resize((max(1, int(w * SS)), max(1, int(h * SS))),
                                Image.BICUBIC), float) / 255

def shear_ss(a, k):
    """Italic: shift row y right by k * (h - y)."""
    h, w = a.shape
    out = np.zeros((h, w + int(abs(k) * h) + 2))
    for y in range(h):
        s = k * (h - 1 - y)
        i = int(np.floor(s)); f = s - i
        out[y, i:i + w] += a[y] * (1 - f)
        out[y, i + 1:i + 1 + w] += a[y] * f
    return np.clip(out, 0, 1)

def warp(a, H, W, top, bot, xl, xr, ox=0, oy=0):
    """Map the source mask a onto an H x W (1x px) canvas at SS: source
    u (0..1 across) and v (0..1 down) go to x between xl(v) and xr(v) and
    y between top(u) and bot(u). Returns SS coverage of the canvas."""
    hs, ws = a.shape
    Y, X = np.mgrid[0:H * SS, 0:W * SS].astype(float)
    X = (X + 0.5) / SS - ox; Y = (Y + 0.5) / SS - oy
    v = np.full_like(X, 0.5)
    for _ in range(6):
        l, r = xl(v), xr(v)
        u = (X - l) / (r - l)
        t, b = top(u), bot(u)
        v = (Y - t) / (b - t)
    ok = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    out = ndimage.map_coordinates(a, [np.clip(v, 0, 1) * (hs - 1),
                                      np.clip(u, 0, 1) * (ws - 1)], order=1)
    return out * ok

def down(a):
    """SS coverage -> 1x coverage (box mean)."""
    h, w = a.shape[0] // SS, a.shape[1] // SS
    return a[:h * SS, :w * SS].reshape(h, SS, w, SS).mean((1, 3))

def paste_ss(canvas, a, x, y):
    """Add mask a (SS) into canvas (SS) at 1x position (x, y), clipped."""
    X, Y = int(round(x * SS)), int(round(y * SS))
    h, w = a.shape
    H, W = canvas.shape
    x0, y0 = max(0, X), max(0, Y)
    x1, y1 = min(W, X + w), min(H, Y + h)
    if x1 <= x0 or y1 <= y0: return canvas
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1],
                                      a[y0 - Y:y1 - Y, x0 - X:x1 - X])
    return canvas

# ------------------------------------------------------------ painting

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r * 0.6

def shift(m, dx, dy):
    o = np.zeros_like(m)
    H, W = m.shape
    o[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
        m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return o

def ramp_idx(shape, y0, y1, stops, dith=1.0):
    """Index map: rows y0..y1 run through the index list stops (top to
    bottom) with ordered dithering between neighbours."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.clip((yy - y0) / max(1, y1 - y0), 0, 1) * (len(stops) - 1)
    k = np.clip(np.rint(t + BAYER[yy % 4, xx % 4] * dith), 0, len(stops) - 1)
    return np.asarray(stops)[k.astype(int)]

def paint(out, cov, st, thr=0.5):
    """Paint one piece (1x coverage) onto the index image out with style
    st (dict). Layers: rim, shadow, outline, fill ramp, bevels."""
    body = cov >= thr
    if not body.any(): return out
    ol = ndimage.binary_dilation(body, disk(st['ol'])) & ~body if st['ol'] else body & False
    shape = body | ol
    sx, sy = st.get('sh', (0, 0))
    sh = np.zeros_like(shape)
    for i in range(1, max(abs(sx), abs(sy)) + 1):
        sh |= shift(shape, int(round(sx * i / max(abs(sx), abs(sy)))),
                    int(round(sy * i / max(abs(sx), abs(sy)))))
    sh &= ~shape
    if st.get('rim'):
        rim = ndimage.binary_dilation(shape | sh, disk(1)) & ~(shape | sh)
        out[rim & (out == 0)] = st['rim']
    out[sh] = st['shc']
    out[ol] = st['olc']
    if st.get('ol2') is not None:          # inner dark edge in the outline
        ring = ndimage.binary_dilation(body, disk(1)) & ~body & ol
        out[ring] = st['ol2']
    rows = np.where(body.any(1))[0]
    y0, y1 = st.get('span', (rows[0], rows[-1]))
    g = ramp_idx(body.shape, y0, y1, st['stops'], st.get('dith', 1.0))
    out[body] = g[body]
    if st.get('lo') is not None:           # bottom/right inner edge
        lo = body & (~shift(body, 0, -1) | ~shift(body, -1, 0))
        lo &= ~(~shift(body, 0, 1))        # not on a top edge
        out[lo] = st['lo'](g)[lo] if callable(st['lo']) else st['lo']
    if st.get('hi') is not None:           # top/left inner edge
        hi = body & (~shift(body, 0, 1) | ~shift(body, 1, 0) & st.get('hileft', True))
        out[hi] = st['hi'](g)[hi] if callable(st['hi']) else st['hi']
    return out

# ------------------------------------------------------------ design

def arch(c, a):
    return lambda u: c + a * (1 - (2 * u - 1) ** 2)

def lin(a, b):
    return lambda v: a + (b - a) * v

def lut(d):
    k = np.zeros(16, np.uint8)
    for i in range(16): k[i] = d.get(i, i)
    return lambda g: k[g]

# Index roles per CLUT. stops: fill ramp top -> bottom; lo/hi: inner edge
# colour for each fill index; olc outline, shc shadow, rim outer rim.
ROLES = {
    # MA_TIT CLUT 9 (144,481): 1 black, 2/3 reds, 4 white, 5/6 dark reds,
    # 7-11 brown -> yellow, 12 dark green, 13/14 greens, 15 magenta.
    'big': dict(stops=[4, 4, 11, 10, 9, 8, 3, 2, 2], two=[8, 3, 2, 2, 2],
                lo={4: 7, 11: 7, 10: 7, 9: 6, 8: 6, 3: 15, 2: 15},
                hi={9: 11, 8: 11, 3: 9, 2: 9}, olc=1, shc=1, rim=12,
                green=dict(stops=[14, 14, 13], olc=12, shc=1)),
    # MA_TIT CLUT 8 (128,481): 1/2 reds, 3/4 oranges, 5 white, 6 black,
    # 7/8 dark reds, 9 brown, 10 magenta red, 11 yellow, 12 dark grey,
    # 13-15 greys/cream. No green: the top line is yellow.
    'small': dict(stops=[5, 15, 11, 4, 3, 2, 1, 1], two=[3, 2, 1, 1],
                  lo={5: 9, 15: 9, 11: 9, 4: 8, 3: 8, 2: 10, 1: 10},
                  hi={4: 11, 3: 11, 2: 4, 1: 4}, olc=6, shc=6, rim=12,
                  green=dict(stops=[5, 11, 11], olc=6, shc=6)),
    # MENU.BIZ CLUT (160,482): 1 green, 2 dark brown, 3 brown-grey, 4 white,
    # 5 red, 6-10 salmon -> orange, 11/12 yellows, 13 cream, 14/15 browns.
    'menu': dict(stops=[4, 4, 13, 12, 11, 10, 9, 8, 7, 7, 5], two=[9, 8, 7, 7, 5],
                 lo={4: 15, 13: 15, 12: 15, 11: 14, 10: 14, 9: 14, 8: 6, 7: 6, 5: 6},
                 hi={8: 11, 7: 11, 5: 10}, olc=3, shc=14, rim=None,
                 green=dict(stops=[1, 1], olc=3, shc=14)),
}

# Pixel letters for the small copies' top line (7 rows, 2 px stems, round
# O). Russo One at cap 6 gave boxy O's that read as "BL[]]DED" at 1x.
PIX = {
    'H': ['##..##', '##..##', '##..##', '######', '##..##', '##..##', '##..##'],
    'O': ['.####.', '##..##', '##..##', '##..##', '##..##', '##..##', '.####.'],
    'T': ['######', '######', '..##..', '..##..', '..##..', '..##..', '..##..'],
    '-': ['....', '....', '....', '####', '....', '....', '....'],
    'B': ['#####.', '##..##', '##..##', '#####.', '##..##', '##..##', '#####.'],
    'L': ['##....', '##....', '##....', '##....', '##....', '##....', '######'],
    'D': ['####..', '##.##.', '##..##', '##..##', '##..##', '##.##.', '####..'],
    'E': ['######', '##....', '##....', '#####.', '##....', '##....', '######'],
}

def pix_line(text, gap=1):
    cols = []
    for i, c in enumerate(text):
        g = np.array([[ch == '#' for ch in row] for row in PIX[c]], float)
        if i: cols.append(np.zeros((7, gap)))
        cols.append(g)
    return np.concatenate(cols, 1)

# Layout per copy (1x px in the copy's own rectangle). keep: rows of the
# JP art (game title) kept on top. top: (text, cap, track, y, outline).
# main: arch top (centre y, rise), bottom (centre y, rise), x left and
# right at top/bottom. two: (w, h, x, y). ol/sh: outline radius, shadow.
SPECS = {
    'big': dict(W=256, H=136, keep=0, top=('HOT-BLOODED', 12, 2.5, 3, 2),
                main=((21, 4), (108, -23), (14, 5), (242, 251)),
                two=(60, 40, 98, 88), ol=3, sh=(3, 3)),
    'small': dict(W=96, H=64, keep=25, top=('HOT-BLOODED', 7, 0, 26, 1),
                  main=((36, 1), (53, -6), (9, 3), (87, 93)),
                  two=(22, 12, 37, 49), ol=1, sh=(1, 1), shear=0, thr=0.4, bevel=False, dith=0.6, pix=True),
    'menu': dict(W=120, H=72, keep=28, top=('HOT-BLOODED', 7, 0, 29, 1),
                 main=((39, 1), (60, -7), (10, 3), (110, 117)),
                 two=(26, 13, 47, 56), ol=1, sh=(1, 1), shear=0, thr=0.4, bevel=False, dith=0.6, pix=True),
}

def design(name, spec=None):
    sp = spec or SPECS[name]
    R = ROLES[name]
    H, W = sp['H'], sp['W']
    fire = dict(ol=sp['ol'], sh=sp['sh'], olc=R['olc'], shc=R['shc'],
                rim=R['rim'], stops=R['stops'], dith=sp.get('dith', 1.0))
    if sp.get('bevel', True): fire.update(lo=lut(R['lo']), hi=lut(R['hi']))
    out = np.zeros((H, W), np.uint8)
    (tc, tr), (bc, br), (l0, l1), (r0, r1) = sp['main']
    # the 2
    w, h, x, y = sp['two']
    a = resize_ss(text_ss('2', RUSSO, 40), w, h)
    cov = down(paste_ss(np.zeros((H * SS, W * SS)), a, x, y))
    paint(out, cov, dict(fire, stops=R['two'], span=(y, y + h)), sp.get('thr', 0.5))
    # main line, fanned and arched
    m = text_ss('YOUTH DIARY', ANTON, 40, 3)
    cov = down(warp(m, H, W, arch(tc, tr), arch(bc, br), lin(l0, l1), lin(r0, r1)))
    paint(out, cov, dict(fire, span=(tc, bc - 8)), sp.get('mthr', 0.5))
    # top line
    text, cap, track, y, ol = sp['top']
    cx = (l0 + r0) / 2
    if sp.get('pix'):                    # hand-drawn 7 px letters
        a = pix_line(text)
        cov = np.zeros((H, W))
        x = int(round(cx - a.shape[1] / 2))
        cov[y:y + a.shape[0], x:x + a.shape[1]] = a
    else:
        a = shear_ss(text_ss(text, RUSSO, cap, track), sp.get('shear', 0.2))
        cov = down(paste_ss(np.zeros((H * SS, W * SS)), a, cx - a.shape[1] / SS / 2, y))
    g = R['green']
    paint(out, cov, dict(ol=ol, sh=(1, 1), olc=g['olc'], shc=g['shc'],
                         stops=g['stops']), sp.get('thr', 0.5))
    return out

# ------------------------------------------------------------ files

JP_TIT = 'private/disc/DAT2/MA_TIT.BIZ'
JP_MENU = 'private/disc/DAT2/MENU.BIZ'
TIT_PIX, TIT_STRIDE = 0x180, 384          # MA_TIT TIM0 4bpp 768x256
TIT_CLUT = 0x14                           # 11 CLUTs of 16 colours
MENU_CLUT = 0x22800 + 20                  # TIM2 CLUT block, VRAM (0,482)
MENU_PIX, MENU_STRIDE = MENU_CLUT + 2048 + 12, 640   # TIM2 4bpp 1280x256
# copy: (file, pixel offset, stride, x, y in the TIM, w, h)
RECTS = {'big': ('tit', 512, 0), 'small': ('tit', 512, 136),
         'menu': ('menu', 1120, 0)}

def get4(d, off, stride, x, y, w, h):
    px = np.frombuffer(bytes(d), np.uint8, stride * (y + h), off).reshape(-1, stride)[y:]
    a = np.zeros((h, stride * 2), np.uint8); a[:, 0::2] = px & 15; a[:, 1::2] = px >> 4
    return a[:, x:x + w]

def put4(d, off, stride, x, y, img):
    h, w = img.shape
    assert x % 2 == 0 and w % 2 == 0
    for r in range(h):
        o = off + (y + r) * stride + x // 2
        d[o:o + w // 2] = (img[r, 0::2] | img[r, 1::2] << 4).astype(np.uint8).tobytes()

def logo(name, jp):
    """EN index image for one copy; jp is the JP index image (kept rows)."""
    img = design(name)
    k = SPECS[name]['keep']
    top = jp[:k]
    img[:k][top > 0] = top[top > 0]
    return img

_jp = {}
def jp_files():
    if not _jp:
        _jp['tit'] = lz.decompress(open(JP_TIT, 'rb').read())[0]
        _jp['menu'] = lz.decompress(open(JP_MENU, 'rb').read())[0]
    return _jp

def patch(kind, d):
    """Write the EN logos of file kind ('tit' MA_TIT, 'menu' MENU.BIZ) into
    the decompressed bytes d. Returns {name: (jp, en)}."""
    off, stride = (TIT_PIX, TIT_STRIDE) if kind == 'tit' else (MENU_PIX, MENU_STRIDE)
    src = jp_files()[kind]
    done = {}
    for name, (k, x, y) in RECTS.items():
        if k != kind: continue
        sp = SPECS[name]
        jp = get4(src, off, stride, x, y, sp['W'], sp['H'])
        en = logo(name, jp)
        put4(d, off, stride, x, y, en)
        done[name] = (jp, en)
    return done

def palette(d, kind, name):
    if kind == 'tit':
        o = TIT_CLUT + 32 * (9 if name == 'big' else 8)
    else:
        o = MENU_CLUT + 2 * 160
    v = np.frombuffer(bytes(d), np.uint16, 16, o).astype(np.int32)
    return np.stack([(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8], 1).astype(np.uint8)

def previews(done, outdir='private/re/logo/en'):
    """JP over EN for each copy, 3x, on grey."""
    os.makedirs(outdir, exist_ok=True)
    src = jp_files()
    for name, (jp, en) in done.items():
        kind = RECTS[name][0]
        pal = palette(src[kind], kind, name)
        rows = []
        for img in (jp, en):
            rgb = np.full(img.shape + (3,), 150, np.uint8)
            rgb[img > 0] = pal[img][img > 0]
            rows.append(np.pad(rgb, ((2, 2), (2, 2), (0, 0)), constant_values=150))
        a = np.concatenate(rows, 1)
        Image.fromarray(a).resize((a.shape[1] * 3, a.shape[0] * 3), Image.NEAREST).save(
            f'{outdir}/{name}.png')

def main():
    """Standalone run: patch private/build/MA_TIT.BIZ and MENU.BIZ in place
    (both must exist, from build_mgtit.py and build_title.py)."""
    pt, pm = 'private/build/MA_TIT.BIZ', 'private/build/MENU.BIZ.lz'
    for p in (pt, pm):
        if not os.path.exists(p): sys.exit(f'PROBLEM: {p} missing; run build_mgtit.py/build_title.py')
    tit = bytearray(lz.decompress(open(pt, 'rb').read())[0])
    menu = bytearray(open(pm, 'rb').read())
    done = patch('tit', tit)
    done.update(patch('menu', menu))
    zt, zm = lz.compress(bytes(tit), name='MA_TIT.BIZ'), lz.compress(bytes(menu), name='MENU.BIZ')
    assert lz.decompress(zt)[0] == bytes(tit) and lz.decompress(zm)[0] == bytes(menu)
    st, sm = 56 * 2048, 188 * 2048
    print(f'logo: MA_TIT.BIZ {len(zt)} B (slot {st}), MENU.BIZ {len(zm)} B (slot {sm})')
    if len(zt) > st or len(zm) > sm: sys.exit('PROBLEM: does not fit its slot')
    open(pt, 'wb').write(zt)
    open(pm, 'wb').write(menu)
    open('private/build/MENU.BIZ', 'wb').write(zm)
    if '--png' in sys.argv: previews(done)

if __name__ == '__main__':
    main()
