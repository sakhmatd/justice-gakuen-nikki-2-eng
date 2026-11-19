# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Written-exam HUD graphics in English. Patch module for build_mgbg.py
(PATCHES) and hook.py (MGAME_PATCH); wording in tl/EXAMHUD.txt.

The HUD is a 4bpp TIM at offset 0xe000 of MGBGK04 entry 2 and MGBGK05/10
entry 1 (same bytes): 768x256 at VRAM (768,0), 32 CLUTs at (0,489), so
TIM x 0-255, 256-511 and 512-767 are tpages 12, 13 and 14. MGAME
(base 0x800a2000) sets every sprite with immediates (RE: topic examhud).

| label    | JP rect (tp,u,v,w,h) | CLUT    | EN rect            | screen  |
|----------+----------------------+---------+--------------------+---------|
| 正解     | 12,0,0,176,224       | 1 idx 1 | 12,0,0,256,168     | stamp   |
| 不正解   | 13,0,0,176,224       | 1 idx 2 | 13,0,0,256,168     | stamp   |
| 問題     | 12,184,176,40,20     | 20      | 12,0,176,w,20      | 84,108  |
| 試験終了 | 14,0,144,200,48      | 1       | 14,0,144,256,48    | slides  |
| 評価UP   | 14,152,80,104,64     | 3-14    | 14,0,192,w,h       | centred |

The stamps share one rect: both are redrawn landscape (W x H) and the
sprite is centred on the quiz panel (x 50-462, y 102-265). Rows 168+ of
tp12 hold the star, the marks and the score bar, so H stays below 168.
"Question" is left-aligned with the question text (x 84). 試験終了
slides in from x 512 and stops centred on x 256 (y 160-207); the EN rect
is 256 wide. 評価UP cycles CLUTs 3-14 (rotations of one 12-colour
rainbow); the fill index runs 12 -> 1 from left to right as in the JP.

Usage: python3 tools/mgbg_examhud.py --png DIR   previews only.
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

TIM_OFF = 0xe000
ENTRIES = [('MGBGK04.BZZ', 2), ('MGBGK05.BZZ', 1), ('MGBGK10.BZZ', 1)]
TL = 'tl/EXAMHUD.txt'
OVL = 0x800a2000
FONT_STAMP = 'PottaOne-Regular.ttf'           # OFL 1.1
FONT_Q = 'RobotoCondensed-Bold.ttf'                  # OFL 1.1
FONT_END = 'MPLUS1p-Black.ttf'                # OFL 1.1
FONT_UP = 'MPLUSRounded1c-Black.ttf'    # OFL 1.1
SS = 4

# Stamp sprite (both stamps), screen centre.
ST_W, ST_H = 256, 168
ST_CX, ST_CY = 256, 184
ST_ANGLE = {'正解': 10.0, '不正解': -12.0}   # degrees, + = ccw (baseline rises)
ST_INK = {'正解': 1, '不正解': 2}
ST_TP = {'正解': 0, '不正解': 256}          # TIM x of the tpage
# "Question" strip in tp12.
Q_U, Q_V, Q_H, Q_X, Q_Y = 0, 176, 20, 84, 108
# 試験終了 rect (tp14): JP 200 wide, EN 256 (x 712-767 were free).
END_X, END_Y, END_W, END_H = 512, 144, 256, 48
# 評価UP: new strip in tp14, centred on x 256, ink centre y 384 (JP).
UP_U, UP_V, UP_CY = 0, 192, 384

# --------------------------------------------------------------- text ---

def font(path, size):
    return fontpath.truetype(path, size)

def text_alpha(text, path, cap, track=0.0, sx=1.0, ss=SS):
    """Float alpha of one line at 1x, cropped to the ink, and the row of the
    cap top in it. cap: cap height in px. track: extra px per letter."""
    probe = font(path, 100)
    b = probe.getbbox('H')
    size = int(round(100 * cap * ss / (b[3] - b[1])))
    f = font(path, size)
    asc = f.getbbox('H')[1]
    w = int(sum(f.getlength(c) for c in text) + track * ss * len(text) + size)
    h = int(size * 1.6)
    im = Image.new('L', (w, h), 0); dr = ImageDraw.Draw(im)
    x = size // 4
    for c in text:
        dr.text((x, size // 4), c, font=f, fill=255)
        x += f.getlength(c) + track * ss
    a = np.asarray(im, np.float32) / 255
    if sx != 1.0:
        im2 = Image.fromarray((a * 255).astype(np.uint8))
        im2 = im2.resize((max(1, int(round(im2.width * sx))), im2.height), Image.LANCZOS)
        a = np.asarray(im2, np.float32) / 255
    # pad to multiples of ss with the cap top on a pixel boundary
    top = size // 4 + asc
    pad_t = (-top) % ss
    a = np.pad(a, ((pad_t, 0), (0, 0)))
    top += pad_t
    H = -(-a.shape[0] // ss) * ss; W = -(-a.shape[1] // ss) * ss
    a = np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])))
    a = a.reshape(H // ss, ss, W // ss, ss).mean((1, 3))
    ys, xs = np.nonzero(a > 0.02)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return a, top // ss - ys.min()

def fit_line(text, path, cap, track, maxw, minsx=0.8):
    """text_alpha at cap with a horizontal squeeze down to minsx, then a
    smaller cap, until the ink is at most maxw wide. -> (alpha, top, cap)."""
    a, top = text_alpha(text, path, cap, track)
    if a.shape[1] <= maxw: return a, top, cap
    sx = maxw / a.shape[1]
    if sx >= minsx:
        a, top = text_alpha(text, path, cap, track, sx * 0.995)
        return a, top, cap
    while True:
        cap -= 0.5
        a, top = text_alpha(text, path, cap, track, minsx)
        if a.shape[1] <= maxw: return a, top, cap

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

# ------------------------------------------------------------- stamps ---

def stamp_alpha(text, cap, angle, path, ss=SS, padx=0.32, pady=0.30, thick=0.16):
    """Float alpha of the tilted stamp at 1x: a frame sized from the word
    (cap height cap px) with brush breaks, the word centred in it."""
    f = font(path, 400); b = f.getbbox('H')
    size = int(round(400 * cap * ss / (b[3] - b[1])))
    f = font(path, size)
    tb = f.getbbox(text); hb = f.getbbox('H')
    tw = tb[2] - tb[0]; C = cap * ss
    t = max(5 * ss, int(thick * C))
    bw = tw + 2 * (padx * C + t); bh = C + 2 * (pady * C + t)
    m = 2 * ss
    Wp, Hp = int(bw) + 2 * m, int(bh) + 2 * m
    im = Image.new('L', (Wp, Hp), 0); dr = ImageDraw.Draw(im)
    x0, y0, x1, y1 = m, m, Wp - m, Hp - m
    dr.rounded_rectangle((x0, y0, x1, y1), radius=int(0.6 * t), outline=255, width=t)
    for edge, f0, f1 in (('top', 0.70, 0.75), ('bot', 0.18, 0.22), ('right', 0.28, 0.42),
                         ('left', 0.60, 0.72)):
        if edge == 'top': dr.rectangle((x0 + f0 * (x1 - x0), y0 - 1, x0 + f1 * (x1 - x0), y0 + t + 1), fill=0)
        if edge == 'bot': dr.rectangle((x0 + f0 * (x1 - x0), y1 - t - 1, x0 + f1 * (x1 - x0), y1 + 1), fill=0)
        if edge == 'right': dr.rectangle((x1 - t - 1, y0 + f0 * (y1 - y0), x1 + 1, y0 + f1 * (y1 - y0)), fill=0)
        if edge == 'left': dr.rectangle((x0 - 1, y0 + f0 * (y1 - y0), x0 + t + 1, y0 + f1 * (y1 - y0)), fill=0)
    dr.text(((Wp - tw) / 2 - tb[0], Hp / 2 - C / 2 - hb[1]), text, font=f, fill=255)
    a = np.asarray(im.rotate(angle, resample=Image.BICUBIC, expand=True), np.float32) / 255
    h, w = a.shape
    h2, w2 = -(-h // ss) * ss, -(-w // ss) * ss
    a = np.pad(a, ((0, h2 - h), (0, w2 - w))).reshape(h2 // ss, ss, w2 // ss, ss).mean((1, 3))
    ys, xs = np.nonzero(a > 0.3)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def stamp(text, ink, angle, W=ST_W, H=ST_H, path=FONT_STAMP, seed=3, **kw):
    """Rubber stamp like the JP: one thick frame with brush breaks, the word
    inside, tilted, single ink index (CLUT 1 index 1 red, 2 blue). The cap
    height is the largest that fits W-2 x H-2."""
    lo, hi = 10.0, 80.0
    while hi - lo > 0.25:
        c = (lo + hi) / 2
        h, w = stamp_alpha(text, c, angle, path, **kw).shape
        if w <= W - 2 and h <= H - 2: lo = c
        else: hi = c
    a = stamp_alpha(text, lo, angle, path, **kw)
    # rough edges and wear (deterministic)
    rng = np.random.default_rng(seed)
    n1 = ndimage.gaussian_filter(rng.standard_normal(a.shape), 1.2)
    n1 /= n1.std()
    n2 = ndimage.gaussian_filter(rng.standard_normal(a.shape), 3.0)
    n2 /= n2.std()
    on = (a > 0.5 + 0.12 * n1) & ~(n2 > 2.1)
    ys, xs = np.nonzero(on)
    on = on[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out = np.zeros((H, W), np.uint8)
    h, w = on.shape
    oy, ox = (H - h) // 2, (W - w) // 2
    assert oy >= 0 and ox >= 0, (text, on.shape)
    out[oy:oy + h, ox:ox + w][on] = ink
    return out

# ----------------------------------------------------------- labels ---

def question(text, cap=13, path=FONT_Q, track=0.3):
    """Yellow header (CLUT 20): fill 14 (yellow), AA down the 3..14 ramp,
    1 px dark outline (3) and a dark shadow (4) down-right."""
    a, top = text_alpha(text, path, cap, track)
    h, w = a.shape
    can = np.zeros((Q_H, w + 3), np.float32)
    y = int(round((Q_H - 1) / 2 - cap / 2)) - top + 0      # cap centred
    y = max(0, min(y, Q_H - h - 1)) if h < Q_H else 0
    can[y:y + h, 1:1 + w] = a[:Q_H - y]
    fill = can
    body = fill > 0.35
    sh = shift(body, 1, 1) & ~body
    ol = ndimage.binary_dilation(body, np.ones((3, 3), bool)) & ~body & ~sh
    out = np.zeros(can.shape, np.uint8)
    out[ol] = 3
    out[sh] = 4
    lv = fill > 0.12
    out[lv] = np.clip(np.rint(3 + fill[lv] * 11), 3, 14)
    return out

def exam_over(text, cap=36, path=FONT_END, track=1.0, sx=0.8, W=None):
    """Lime fill (CLUT 1 index 15), AA down the 3..15 ramp, red shadow
    (index 1) 2 px down-right; centred in the W x 48 rect."""
    W, H = W or END_W, END_H
    a, top, cap = fit_line(text, path, cap, track, W - 6, sx)
    h, w = a.shape
    can = np.zeros((H, W), np.float32)
    x = (W - (w + 2)) // 2
    y = (H - (h + 2)) // 2                     # caps only: centre the ink
    assert 0 <= y and y + h + 2 <= H, (text, h)
    can[y:y + h, x:x + w] = a
    body = can > 0.4
    sh = shift(body, 2, 2) & ~body
    out = np.zeros((H, W), np.uint8)
    out[sh] = 1
    lv = can > 0.15
    out[lv] = np.clip(np.rint(3 + can[lv] * 12), 3, 15)
    return out

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32

def rating_up(text, cap=24, path=FONT_UP, track=1.0, ol=2):
    """Rainbow label (CLUTs 3-14): fill index 12 -> 1 across the width with
    ordered dither, AA ring 14, black outline 13 (ol px) plus 1 px more
    down-right as a shadow."""
    a, top = text_alpha(text, path, cap, track)
    h, w = a.shape
    P = ol + 2
    can = np.zeros((h + 2 * P, w + 2 * P), np.float32)
    can[P:P + h, P:P + w] = a
    body = can > 0.5
    ring = can > 0.12
    st = ndimage.generate_binary_structure(2, 1)
    o = ndimage.binary_dilation(ring, st, iterations=ol)
    o = o | shift(o, 1, 1)
    out = np.zeros(can.shape, np.uint8)
    out[o] = 13
    out[ring & ~body] = 14
    xs = np.arange(can.shape[1])[None, :].repeat(can.shape[0], 0)
    ys = np.arange(can.shape[0])[:, None].repeat(can.shape[1], 1)
    t = (xs - P) / max(1, w - 1) * 11.0                    # 0 .. 11
    lvl = 12 - np.floor(t + BAYER[ys % 4, xs % 4] - 0.5)
    lvl = np.clip(lvl, 1, 12).astype(np.uint8)
    out[body] = lvl[body]
    ys_, xs_ = np.nonzero(out)
    return out[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1], P - ys_.min()

# --------------------------------------------------------------- TIM ---

def load_tl(path=TL):
    out = {}; jp = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            if line[4:].strip(): out[jp] = line[4:].strip()
            jp = None
    for k in ('正解', '不正解', '問題', '試験終了', '評価UP'):
        if k not in out: raise SystemExit(f'mgbg_examhud: no EN for {k} in {path}')
    return out

def tim_idx(e):
    t = e[TIM_OFF:]
    assert struct.unpack_from('<II', t, 0) == (0x10, 8)
    clen = struct.unpack_from('<I', t, 8)[0]
    p = 8 + clen
    blen, x, y, w, h = struct.unpack_from('<IHHHH', t, p)
    assert (x, y, w, h) == (768, 0, 192, 256)
    px = np.frombuffer(t[p + 12:p + blen], np.uint8).reshape(h, w * 2)
    idx = np.zeros((h, w * 4), np.uint8)
    idx[:, 0::2] = px & 15; idx[:, 1::2] = px >> 4
    return idx, TIM_OFF + p + 12

def clut(e):
    return np.frombuffer(e[TIM_OFF + 20:TIM_OFF + 20 + 1024], '<u2').reshape(32, 16)

def layout(tl):
    """-> dict of label -> (index image, TIM x, TIM y) and sprite facts."""
    L = {}
    for jp in ('正解', '不正解'):
        L[jp] = (stamp(tl[jp], ST_INK[jp], ST_ANGLE[jp]), ST_TP[jp], 0)
    q = question(tl['問題'])
    assert q.shape[1] <= 176, q.shape
    L['問題'] = (q, Q_U, Q_V)
    L['試験終了'] = (exam_over(tl['試験終了']), END_X, END_Y)
    up, _ = rating_up(tl['評価UP'])
    assert up.shape[1] <= 256 and up.shape[0] <= 64, up.shape
    L['評価UP'] = (up, 512 + UP_U, UP_V)
    return L

def patch_tim(e, tl=None):
    tl = tl or load_tl()
    idx, pix = tim_idx(e)
    idx = idx.copy()
    idx[0:224, 0:176] = 0                     # old 正解 (tp12)
    idx[0:224, 256:432] = 0                   # old 不正解 (tp13)
    idx[END_Y:END_Y + END_H, END_X:END_X + END_W] = 0
    for jp, (im, x, y) in layout(tl).items():
        h, w = im.shape
        if jp in ('正解', '不正解'):
            idx[y:y + h, x:x + w] = im
        else:
            assert not idx[y:y + h, x:x + w].any() or jp == '試験終了', jp
            idx[y:y + h, x:x + w] = im
    px = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8).tobytes()
    e = bytearray(e)
    e[pix:pix + len(px)] = px
    return bytes(e)

PATCHES = [(name, k, patch_tim) for name, k in ENTRIES]

# ------------------------------------------------------------- MGAME ---

def set_imm(d, ram, old, new):
    o = ram - OVL
    w = struct.unpack_from('<I', d, o)[0]
    assert w & 0xffff == old & 0xffff, (hex(ram), hex(w), old)
    struct.pack_into('<I', d, o, (w & 0xffff0000) | (new & 0xffff))

def MGAME_PATCH(data, tl=None):
    """Sprite rects and positions (MGAME immediates, RE: topic examhud)."""
    tl = tl or load_tl()
    L = layout(tl)
    d = bytearray(data)
    # stamps (both): x, y, w, h; u = v = 0, tpage 0x300 / 0x340
    set_imm(d, 0x800bece8, 0xa0, ST_CX - ST_W // 2)
    set_imm(d, 0x800becf0, 0x60, ST_CY - ST_H // 2)
    set_imm(d, 0x800beda4, 0xb0, ST_W)
    set_imm(d, 0x800bedac, 0xe0, ST_H)
    # 問題: x 84, y 108 stay; u, v, w, h
    q = L['問題'][0]
    set_imm(d, 0x800be94c, 0xb8, Q_U)
    set_imm(d, 0x800be954, 0xb0, Q_V)
    set_imm(d, 0x800be95c, 0x28, q.shape[1])
    set_imm(d, 0x800be964, 0x14, Q_H)
    # 試験終了: width and slide. It enters from x 512 in steps of 28, stops
    # at x 256 - w / 2 and leaves when x < 1 - w.
    set_imm(d, 0x800bffe8, 0xc8, END_W)
    set_imm(d, 0x800c00b4, 0x9d, 256 - END_W // 2 + 1)
    set_imm(d, 0x800c00d0, 0x9c, 256 - END_W // 2)
    set_imm(d, 0x800c0128, -0xc7, 1 - END_W)
    # 評価UP: x, y, u, v, w, h
    up = L['評価UP'][0]
    h, w = up.shape
    set_imm(d, 0x800bf69c, 0xcc, 256 - w // 2)
    set_imm(d, 0x800bf6a4, 0x16c, UP_CY - h // 2)
    set_imm(d, 0x800bf748, 0x98, UP_U)
    set_imm(d, 0x800bf750, 0x50, UP_V)
    set_imm(d, 0x800bf758, 0x68, w)
    set_imm(d, 0x800bf760, 0x40, h)
    return bytes(d)

# ----------------------------------------------------------- previews ---

def rgb(cl, im, bg=None):
    c = cl[im].astype(int)
    out = np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.uint8)
    if bg is not None:
        out = np.where((im > 0)[..., None], out, bg)
    return out

def main(outdir):
    sys.path.insert(0, os.path.dirname(__file__))
    import lz, arc
    os.makedirs(outdir, exist_ok=True)
    d = open('private/disc/DAT2/MGBGK04.BZZ', 'rb').read()
    o, s = arc.toc(d)[2]
    e = lz.decompress(d[o:o + s])[0]
    new = patch_tim(e)
    cl = clut(e)
    for tag, ee in (('jp', e), ('en', new)):
        idx, _ = tim_idx(ee)
        Image.fromarray(rgb(cl, idx, np.array([40, 40, 90], np.uint8))).save(f'{outdir}/tim_{tag}.png')
    print('wrote', outdir)

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--png':
        main(sys.argv[2])
    else:
        print(__doc__)
