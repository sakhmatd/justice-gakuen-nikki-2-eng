# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Festival mode menu (mode 11): the 五輪祭 header logo and the MINISEL
sprite records. Called by build_minittl.py; the menu labels themselves
are painted by minittl_atlas.py (f_* pieces).

- Header: painted into the 8bpp 512x480 background TIM of MINITTL0.BIZ
  (0x18800, drawn 1:1). The JP logo (brush kanji, rainbow fill running
  left to right, thin white outline, dark green drop shadow) is cut out by
  its white outline, filled from the surrounding crowd picture and
  redrawn as two centred lines in the same style. N1 model:
  justice-mtl tools/build_fest.py header() (the first game's patch).
- MINISEL.BIN (DAT1, base 0x800a2000): 14 sprite lists at file 0x10;
  each list is s16 count, then count x 11 s16 (mode, tpx, tpy, clx, cly,
  u, v, w, h, x, y). Lists 0-7 are the menu items. minisel() writes tpx,
  u, v and w of the items whose EN sprite moved or grew (minittl_atlas
  ATLAS f_*). RE: private/re/fest/findings.org.
Wording: tl/FEST.txt (header; '|' breaks the line).
"""
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

TIM = 0x18800
LOGO_RECT = (130, 10, 380, 112)       # search rect of the JP logo (x0, y0, x1, y1)
CX = 256                              # header centre: the screen centre
BOX = (14, 104)                       # header text block, screen y top/bottom
FONT = 'seguibli.ttf'   # as the title cards (build_minittl F_BLACKI)
OUTLINE = 2.0                         # white outline width, px
SHADOW = (3, 3)                       # drop shadow offset (dx, dy)

# MINISEL menu lists: id -> file offset of the list (count at +0)
LISTS = {'f_hr': 0x10, 'f_pk': 0x28, 'f_shoot': 0x40, 'f_ace': 0x58,
         'f_sprint': 0x70, 'f_dance': 0x88, 'f_watch': 0xa0, 'f_back': 0xb8}


def minisel(src, atlas):
    """Patched copy of MINISEL.BIN for the f_* rects in atlas (minittl_atlas
    ATLAS). Returns (bytes, log lines)."""
    d = bytearray(src); log = []
    for rid, off in LISTS.items():
        x, y, w, h = atlas[rid][:4]
        cnt, mode, tpx, tpy, clx, cly, u, v, ow, oh, sx, sy = struct.unpack_from('<12h', d, off)
        assert cnt == 1 and oh == h == 24 and 768 <= tpx <= 896, (rid, hex(off))
        page, u_new = divmod(x, 256)
        assert u_new + w <= 256, rid
        new = (768 + 64 * page, u_new, y, w)
        if new != (tpx, u, v, ow):
            struct.pack_into('<h', d, off + 4, new[0])
            struct.pack_into('<3h', d, off + 12, *new[1:])
            log.append(f'MINISEL {rid}: tpage {tpx}->{new[0]} uv {u},{v}->{new[1]},{new[2]} '
                       f'w {ow}->{w}')
    return bytes(d), log


# ---------------------------------------------------------------- header

def logo_mask(img):
    """Logo (fill + white outline) and its drop shadow, as bool masks."""
    x0, y0, x1, y1 = LOGO_RECT
    sub = img[y0:y1, x0:x1]
    mn, mx = sub.min(-1), sub.max(-1); sat = mx - mn
    white = (mn >= 200) & (sat < 70)
    g = sub[..., 1]
    nongreen = (sat > 60) & ~((g >= sub[..., 0]) & (g >= sub[..., 2]) & (sat < 140))
    body = ndimage.binary_fill_holes(ndimage.binary_closing(white | nongreen, iterations=2))
    lab, n = ndimage.label(body)
    sizes = ndimage.sum(body, lab, range(1, n + 1))
    body = np.isin(lab, 1 + np.where(sizes > 150)[0])
    sh = np.zeros_like(body)
    for k in range(1, 7): sh |= np.roll(np.roll(body, k, 0), k, 1)
    shadow = sh & ~body & (sub.sum(-1) < 260)
    full = np.zeros(img.shape[:2], bool); fs = full.copy()
    full[y0:y1, x0:x1] = body; fs[y0:y1, x0:x1] = shadow
    return full, fs


def logo_style(img, body, shadow):
    """Colour ramp of the fill (left to right), outline and shadow colour."""
    core = ndimage.binary_erosion(body, iterations=3)
    sat = img.max(-1) - img.min(-1)
    core &= sat > 120
    xs = np.where(core.any(0))[0]
    ramp = []
    for x in range(xs[0], xs[-1] + 1):
        sel = core[:, x]
        if sel.sum() < 3: ramp.append(None); continue
        ramp.append(np.median(img[sel, x], 0))
    good = [i for i, c in enumerate(ramp) if c is not None]
    arr = np.array([ramp[i] for i in good])
    full = np.stack([np.interp(np.arange(len(ramp)), good, arr[:, c]) for c in range(3)], -1)
    full = ndimage.uniform_filter1d(full, 9, axis=0)
    ring = body & ~ndimage.binary_erosion(body, iterations=1)
    line = np.median(img[ring & (img.min(-1) >= 200)], 0)
    shade = np.median(img[shadow], 0)
    return full, line, shade


def text_alpha(lines, w, h, font, gap=0.06, ss=4):
    """Alpha (h, w) of centred lines, as large as fits in w x h."""
    best = None
    for size in range(80, 10, -1):
        f = fontpath.truetype(font, size * ss)
        bbs = [f.getbbox(t) for t in lines]
        lw = [b[2] - b[0] for b in bbs]
        cap = max(b[3] - b[1] for b in bbs)
        tot = cap * len(lines) + int(gap * cap * ss / ss) * (len(lines) - 1)
        if max(lw) <= (w - 8) * ss and tot <= (h - 8) * ss:
            best = (f, bbs, lw, cap); break
    f, bbs, lw, cap = best
    pitch = cap + int(gap * cap)
    tot = cap * len(lines) + int(gap * cap) * (len(lines) - 1)
    top = (h * ss - tot) // 2
    out = []
    for i, (t, b, tw) in enumerate(zip(lines, bbs, lw)):
        # each line's own ink box is centred; caps share a baseline pitch
        im = Image.new('L', (w * ss, h * ss), 0)
        ImageDraw.Draw(im).text(((w * ss - tw) // 2 - b[0], top + i * pitch - b[1] + (cap - (b[3] - b[1]))),
                                t, font=f, fill=255)
        out.append(np.asarray(im.resize((w, h), Image.LANCZOS)) / 255.0)
    return np.clip(sum(out), 0, 1), out


def soft_dilate(a, r):
    """Round dilation of an alpha by r px (distance based, antialiased)."""
    dist = ndimage.distance_transform_edt(a < 0.5)
    return np.clip(r + 0.5 - dist, 0, 1) if r > 0 else a


def header(pg, text, fill_mode='inpaint', font=FONT, tilt=0.15, per_line=True, log=None):
    """Erase the JP logo of Page pg (build_minittl.Page) and draw text."""
    import build_minittl as BM
    img = pg.img
    body, shadow = logo_mask(img)
    ramp, line, shade = logo_style(img, body, shadow)
    mk = ndimage.binary_dilation(body | shadow, iterations=1)
    if fill_mode == 'ex':
        bad = [i for i, v in enumerate(pg.clut) if v == 0]
        idx = BM.exemplar(pg.idx, pg.pal, mk, mk, P=9, R=40)
        img = img.copy(); img[mk] = pg.pal[idx][mk]
    else:
        img = BM.grain(BM.inpaint(img, mk, None, iters=300), mk, pg.img, 1.0)
    lines = text.split('|')
    w = 300; h = BOX[1] - BOX[0]
    x0 = CX - w // 2; y0 = BOX[0]
    a, per = text_alpha(lines, w, h, font)
    o = soft_dilate(a, OUTLINE)
    sh = ndimage.shift(o, (SHADOW[1], SHADOW[0]), order=0)
    reg = img[y0:y0 + h, x0:x0 + w]
    reg[:] = reg * (1 - sh[..., None]) + shade * sh[..., None]
    reg[:] = reg * (1 - o[..., None]) + line * o[..., None]
    # fill: the original's ramp along a slightly tilted axis (bottom-left
    # starts the ramp, as in the JP logo). per_line: every line runs
    # through the whole ramp, as each JP line does.
    yy, xx = np.mgrid[0:h, 0:w]
    groups = per if per_line else [a]
    owner = np.argmax(np.stack(groups), 0)
    t = np.zeros((h, w))
    for k, g in enumerate(groups):
        m = g > 0.3
        ys, xs = np.nonzero(m)
        tb = (xx - xs.min()) / max(1, xs.max() - xs.min()) + tilt * (ys.max() - yy) / max(1, ys.max() - ys.min())
        tb = (tb - tb[m].min()) / max(1e-6, np.ptp(tb[m]))
        t[owner == k] = tb[owner == k]
    col = ramp[np.clip((t * (len(ramp) - 1)).astype(int), 0, len(ramp) - 1)]
    reg[:] = reg * (1 - a[..., None]) + col * a[..., None]
    pg.img = img
    if log is not None:
        ink = np.nonzero(o > 0.3)
        log.append(f'header: "{text}" ink x {ink[1].min() + x0}-{ink[1].max() + x0} '
                   f'y {ink[0].min() + y0}-{ink[0].max() + y0}, JP logo erased {int(mk.sum())} px')
    return img
