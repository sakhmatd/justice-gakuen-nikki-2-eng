# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary event title cards (MGBG.TZZ entries 093-111) in English.
Patch module for tools/build_mgbg.py; reads tl/CARDS.txt.

Each card is an 8bpp TIM at offset 0 of its entry: 256x220 at VRAM
(812,256), one 256-colour CLUT at (0,482). The picture is 512x110, stored
as two 110-row halves (rows 0-109 = x 0-255, rows 110-219 = x 256-511).
Every card has byte-identical copies at offset 0 of bundle entries in the
MGBGG/K/E archives (COPIES); all of them get the same new pixels.

JP lettering: a date line (rows ~20-39) above the event name (rows
~56-89), or one exam name (rows ~39-68). Each line has a rainbow fill
(hue falls from magenta-red at the left end of the line to red at the
right end), a 1 px white outline and a dark drop shadow ~2 px down-right.
The builder paints the JP out (tools/inpaint.py: the JP mask plus a 5 px
rim, LaMa fill, dithered write-back into the CLUT), lays each EN line out
fresh on the card's axis, draws it in the same style and maps the EN
pixels to the nearest CLUT colour. CLUT and header stay unchanged.

Usage: python3 tools/mgbg_cards.py [--png DIR]  previews only (JP/EN sheet).
"""
import os, sys, struct, functools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import inpaint
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

FIRST, COUNT = 93, 19
TIM_LEN = 8 + 12 + 512 + 12 + 256 * 220         # header, CLUT block, image block
PIX = 8 + 12 + 512 + 12
FONT = 'MPLUSRounded1c-Medium.ttf'      # OFL
FONT_DATE = 'MPLUSRounded1c-Bold.ttf'  # same stroke at the smaller size
CX = 254.5                  # design axis: mean centre of the JP lines
X0, X1 = 30, 480            # EN ink (shadow included) stays inside
SS = 4                      # supersampling
RING = 5                    # text_mask rim around the JP mask (px)
DITHER_MAX = 24             # max sum |RGB| error of a dithered fill pixel
# Line metrics in picture rows: cap height and the y of the cap centre.
DATE = dict(cap=14, cy=29.0, maxw=X1 - X0, track=3.0, font=FONT_DATE)
TITLE = dict(cap=22, cy=72.0, maxw=X1 - X0, track=2.0)
SINGLE = dict(cap=23, cy=53.5, maxw=X1 - X0, track=2.0)
SHADOW = (2.0, 1.5)         # shadow offset (px)
WHITE = np.array([240., 240., 240.])
HUE0, HUE1 = 345.0, 5.0     # rainbow hue at the left and right end of a line
SAT, VAL = 0.95, 0.95

# Copies of each card (archive, entry), from private/re/cards/copies.py.
COPIES = {
    93: [('MGBGG04.BZZ', n) for n in (0, 22, 26, 30, 34)],
    94: [('MGBGK05.BZZ', 0)],
    95: [('MGBGG06.BZZ', n) for n in (0, 4, 9, 13, 20)],
    96: [('MGBGK07.BZZ', 0)],
    97: [('MGBGG08.BZZ', n) for n in (0, 16, 17, 18, 19)],
    98: [('MGBGG09.BZZ', n) for n in (0, 29, 30, 31, 32)],
    99: [('MGBGK10.BZZ', 0)],
    100: [('MGBGG11.BZZ', n) for n in (0, 8, 16, 24, 32)],
    101: [('MGBGK12.BZZ', 0)],
    102: [('MGBGK12.BZZ', 6)],
    103: [('MGBGK01.BZZ', 0), ('MGBGK01.BZZ', 30)],
    104: [('MGBGK02.BZZ', 0), ('MGBGK02.BZZ', 27)],
    105: [('MGBGK03.BZZ', 29)],
    106: [('MGBGK03.BZZ', 26)],
    107: [('MGBGE03.BZZ', 0)],
    108: [('MGBGG04.BZZ', n) for n in (14, 25, 29, 33, 37)],
    109: [('MGBGK07.BZZ', 6), ('MGBGK07.BZZ', 7)],
    110: [('MGBGK03.BZZ', 0), ('MGBGK03.BZZ', 32)],
    111: [('MGBGK08.BZZ', 0), ('MGBGK08.BZZ', 1)],
}

def load_tl(path='tl/CARDS.txt'):
    """Card entry -> (JP lines, EN lines), in file order."""
    out = {}; jp = None; k = FIRST
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            en = line[4:].strip()
            out[k] = (jp.split('|'), en.split('|') if en else None); k += 1; jp = None
    return out

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

def unpack(tim):
    clut = struct.unpack_from('<256H', tim, 20)
    pal = np.array([rgb15(v) for v in clut])
    a = np.frombuffer(bytes(tim[PIX:PIX + 256 * 220]), np.uint8).reshape(220, 256)
    return pal, np.concatenate([a[:110], a[110:]], 1)

def pack(tim, idx):
    out = bytearray(tim)
    out[PIX:PIX + 256 * 220] = np.concatenate([idx[:, :256], idx[:, 256:]], 0).astype(np.uint8).tobytes()
    return bytes(out)

# --- JP removal ---------------------------------------------------------

def jp_lines(img):
    """Row bands of the JP lettering (y0, y1, x0, x1), from the white
    outline pixels."""
    white = img.min(-1) > 200
    rows = np.where(white.any(1))[0]
    out = []; s = p = rows[0]
    for r in list(rows[1:]) + [10 ** 6]:
        if r > p + 2:
            cols = np.where(white[s:p + 1].any(0))[0]
            out.append((s, p, cols[0], cols[-1])); s = r
        p = r
    return white, out

def jp_mask(img):
    """Lettering, outline and shadow: the white outline grown by 3 px plus
    the same shape moved 2 px down-right (shadow), plus pixels near it that
    differ from the background, inside the line bands."""
    white, lines = jp_lines(img)
    band = np.zeros_like(white)
    for y0, y1, x0, x1 in lines: band[max(0, y0 - 4):y1 + 6, max(0, x0 - 4):x1 + 7] = True
    # outline pixels can be tinted (e.g. 192,192,216 on 110): looser test
    white = band & (img.min(-1) > 180) & (img.max(-1) - img.min(-1) < 56)
    m = ndimage.binary_dilation(white, iterations=3)
    m |= np.roll(np.roll(m, 2, 0), 2, 1)
    # plus anything far from a background diffused in from the band edges
    bg = diffuse(img, band, iters=300)
    far = band & (np.sqrt(((img - bg) ** 2).sum(-1)) > 36)
    far = ndimage.binary_opening(far) | (far & m)
    m = ndimage.binary_dilation(m | (far & ndimage.binary_dilation(white, iterations=8)), iterations=1)
    return m & band, lines

def diffuse(img, mk, iters=500):
    out = img.copy()
    ys, xs = np.where(mk)
    if not len(ys): return out
    y0, y1 = max(0, ys.min() - 3), ys.max() + 4
    x0, x1 = max(0, xs.min() - 3), xs.max() + 4
    sub = out[y0:y1, x0:x1]; m = mk[y0:y1, x0:x1]
    # start from a row-wise fill so diffusion converges quickly
    for c in range(3):
        ch = sub[..., c]
        for r in range(ch.shape[0]):
            good = ~m[r]
            if good.any() and m[r].any():
                ch[r, m[r]] = np.interp(np.where(m[r])[0], np.where(good)[0], ch[r, good])
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            ch[m] = ndimage.convolve(ch, k, mode='nearest')[m]
    out[y0:y1, x0:x1] = sub
    return out

def erase(img, mk, clut, pal, idx):
    """Paint the JP out: the script's mask (jp_mask) plus a 5 px rim of
    pixels that differ from the local background (soft shadow and glow of
    the JP glyphs). Fill: LaMa on a 128 px context crop (lama_ctx; cleaner
    than 'lama' at the 101 eraser, equal elsewhere), or inpaint's 'best'
    ('auto') without the model. Write-back into the card's CLUT with
    dither inside the mask, nearest colour where the dithered pick is
    further than DITHER_MAX from the fill. mk: jp_mask(img).
    Returns (fill mask, new indices)."""
    x0, y0, x1, y1 = inpaint._bbox(mk, 5, img.shape)
    m = inpaint.text_mask(img, (x0, y0, x1, y1), pred=lambda a: mk, ring=RING,
                          halo=1, min_px=0, close=0)
    out = inpaint.fill(img, m, 'lama_ctx' if inpaint.lama_ok() else 'best')
    fs = inpaint.to_clut(out, idx, clut, m, pal=pal)
    near = inpaint.to_clut(out, idx, clut, m, pal=pal, dither='none')
    # the CLUTs hold the canvas tones and the saturated art colours with
    # few steps between; dither may pick an art colour as a stray dot
    stray = np.abs(pal[fs] - out).sum(-1) > DITHER_MAX
    return m, np.where(stray, near, fs)

# --- EN lettering -------------------------------------------------------

@functools.lru_cache(None)
def font(size, path=None):
    return fontpath.truetype(path or FONT, size)

def cap_px(size, path=None):
    b = font(size * SS, path).getbbox('H')
    return (b[3] - b[1]) / SS

def line_alpha(text, cap, maxw, track=0.0, path=None):
    """Glyph coverage at SS x for a cap height in px, condensed down to
    0.85 and then shrunk until the outlined ink fits maxw. track is extra
    letter spacing in px at a 20 px font size. Returns (alpha at SS scale,
    baseline row, cap top row, cap px)."""
    size = 8
    while cap_px(size + 1, path) <= cap: size += 1
    while True:
        f = font(size * SS, path)
        tr = track * size / 20 * SS
        b = list(f.getbbox(text)); b[2] += tr * (len(text) - 1)
        w = (b[2] - b[0]) / SS + 2 + SHADOW[0]
        sx = min(1.0, maxw / w)
        if sx >= 0.85 or size <= 8: break
        size -= 1
    pad = 4 * SS
    im = Image.new('L', (int(b[2] - b[0]) + 2 * pad, b[3] - b[1] + 2 * pad), 0)
    dr = ImageDraw.Draw(im)
    for i, ch in enumerate(text):        # advance with kerning, plus tracking
        dr.text((pad - b[0] + f.getlength(text[:i]) + tr * i, pad - b[1]), ch, font=f, fill=255)
    if sx < 1.0: im = im.resize((int(im.width * sx), im.height), Image.LANCZOS)
    hb = f.getbbox('H')
    return np.asarray(im, float) / 255, pad - b[1] + hb[3], pad - b[1] + hb[1], cap_px(size, path)

def down(a):
    """SS-scale alpha to px (box filter); the array is padded to SS."""
    h, w = a.shape
    a = np.pad(a, ((0, -h % SS), (0, -w % SS)))
    return a.reshape(a.shape[0] // SS, SS, a.shape[1] // SS, SS).mean((1, 3))

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r

def hsv2rgb(h, s, v):
    h = (h % 360) / 60.0; i = np.floor(h).astype(int); f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    rgb = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate(sel):
        m = (i % 6) == k
        for c, val in enumerate((r, g, b)):
            rgb[..., c][m] = (val[m] if np.ndim(val) else val)
    return rgb * 255

def draw_line(img, text, spec, shadow):
    """Draw one line centred on CX at spec's cap centre. Returns the ink
    box (x0, y0, x1, y1) including outline and shadow."""
    a, base, top, cap = line_alpha(text, spec['cap'], spec['maxw'], spec['track'], spec.get('font'))
    o = ndimage.grey_dilation(a, footprint=disk(SS))              # 1 px outline
    sdx, sdy = int(round(SHADOW[0] * SS)), int(round(SHADOW[1] * SS))
    s = np.zeros_like(o); s[sdy:, sdx:] = o[:o.shape[0] - sdy, :o.shape[1] - sdx]
    cols = np.where(np.maximum(o, s).max(0) > 0.02)[0]
    ink_l, ink_r = cols[0], cols[-1] + 1
    inkw = (ink_r - ink_l - sdx) / SS                            # centre without the shadow
    # place: SS-array column ink_l at x = CX - inkw/2, cap centre at cy
    x = CX - inkw / 2 - ink_l / SS
    y = spec['cy'] - (top + base) / 2 / SS
    xi, yi = int(np.floor(x)), int(np.floor(y))
    fx, fy = int(round((x - xi) * SS)), int(round((y - yi) * SS))
    def place(arr):
        arr = np.pad(arr, ((fy, 0), (fx, 0)))
        return down(arr)
    A, O, S = place(a), place(o), place(s)
    h, w = A.shape
    H, W = img.shape[:2]
    reg = img[yi:yi + h, xi:xi + w]
    assert reg.shape[:2] == (h, w) and xi >= 0 and yi >= 0, (text, xi, yi, w, h)
    # rainbow across the line's glyph ink
    gc = np.where(A.max(0) > 0.05)[0]
    t = np.clip((np.arange(w) - gc[0]) / max(1, gc[-1] - gc[0]), 0, 1)
    hue = HUE0 + (HUE1 - HUE0) * t
    fill = np.broadcast_to(hsv2rgb(hue, SAT, VAL)[None], (h, w, 3))
    for alpha, col in ((S, shadow), (O, WHITE), (A, fill)):
        al = alpha[..., None]
        reg[:] = reg * (1 - al) + col * al
    ys, xs = np.where(np.maximum(O, S) > 0.1)
    return xi + xs.min(), yi + ys.min(), xi + xs.max() + 1, yi + ys.max() + 1

def shadow_colour(img, white, mk):
    near = ndimage.binary_dilation(white, iterations=4) & mk & ~ndimage.binary_dilation(white, iterations=1)
    lum = img.mean(-1)
    cand = img[near & (lum < np.percentile(lum[near], 25))]
    return np.median(cand, 0)

def quantise(img, pal, idx):
    out = idx.copy()
    ch = np.abs(img - pal[idx]).sum(-1) > 0.5
    flat = img[ch]; sel = np.where(ch.ravel())[0]
    for k in range(0, len(flat), 4096):
        d = ((flat[k:k + 4096, None, :] - pal[None]) ** 2).sum(-1)
        out.ravel()[sel[k:k + 4096]] = d.argmin(1)
    return out

INK = {}                    # entry -> list of ink boxes (for checks)

def render(e, tim, en):
    pal, idx = unpack(tim)
    clut = struct.unpack_from('<256H', tim, 20)
    img = pal[idx]
    white, _ = jp_lines(img)
    mk, _ = jp_mask(img)
    shadow = shadow_colour(img, white, mk)
    _, fidx = erase(img, mk, clut, pal, idx)
    img = pal[fidx]
    specs = [SINGLE] if len(en) == 1 else [DATE, TITLE]
    INK[e] = [draw_line(img, t, s, shadow) for t, s in zip(en, specs)]
    return pal, idx, quantise(np.clip(img, 0, 255), pal, fidx)

@functools.lru_cache(None)
def card_tim(e):
    """New TIM bytes for card e, or None if it has no EN."""
    tl = load_tl()
    tim = orig_tim(e)
    if e not in tl or not tl[e][1]: return None
    return pack(tim, render(e, tim, tl[e][1])[2])

@functools.lru_cache(None)
def orig_tim(e):
    import lz
    d = open('private/disc/DAT2/MGBG.TZZ', 'rb').read()
    o, s = struct.unpack_from('<II', d, 8 * e)
    return lz.decompress(d[o:o + s])[0][:TIM_LEN]

def patcher(e):
    def fn(data):
        assert data[:TIM_LEN] == orig_tim(e), f'card {e}: entry does not start with the JP card TIM'
        new = card_tim(e)
        return data if new is None else new + data[TIM_LEN:]
    return fn

PATCHES = []
for _e in range(FIRST, FIRST + COUNT):
    PATCHES.append(('MGBG.TZZ', _e, patcher(_e)))
    for _a, _n in COPIES[_e]: PATCHES.append((_a, _n, patcher(_e)))

def preview(path):
    """JP/EN pairs of all cards, one row each, at 1x."""
    tl = load_tl()
    rows = []
    for e in range(FIRST, FIRST + COUNT):
        tim = orig_tim(e)
        pal, idx, new = render(e, tim, tl[e][1])
        rows.append(np.concatenate([pal[idx], np.full((110, 4, 3), 0.), pal[new]], 1))
        rows.append(np.zeros((3, rows[-1].shape[1], 3)))
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)

if __name__ == '__main__':
    if '--png' in sys.argv:
        d = sys.argv[sys.argv.index('--png') + 1]; os.makedirs(d, exist_ok=True)
        preview(os.path.join(d, 'cards_sheet.png'))
        for e, boxes in INK.items(): print(e, boxes)
