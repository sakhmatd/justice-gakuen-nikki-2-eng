# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Mini-game start menu labels in English: the label atlas (TIM 0 at file
offset 0 of every MINITTL0-C.BIZ; identical copies), called by
build_minittl.py for each file it writes.

The atlas is 4bpp 768x256 at VRAM (768,0), 12 CLUTs at (0,481), pixels at
0x1a0 (384 bytes per row). The start menu of every mini-game overlay
(MGAME01-07) draws its items from it through records of s16 n0, cnt, X, Y
and cnt x (u/16, row, w/8): u = 16*a, v = 24*row, h = 24, screen
x = X - 8*n0 (tables e.g. MGAME02 file 0x10-0x50, MGAME06 0x74-0xf4,
MGAME07 0x154-0x1d4). The menu items are centred on the cursor bar
behind the selected item (screen x 184-357); build_minihud.py moves the
戻る records (JP sprite centre 260) so that "Back" is centred there too.

Wording: tl/MINIHUD.txt (title_*, unit_pt). Style: the first game's
build_fest.py atlas() (shadow/outline from the darkest used index, the
most used index as fill, an in-between index for the antialias edge).
The Festival menu (DAT1/MINISEL, mode 11) draws its items and the record
label from the same atlas (f_* below, wording tl/FEST.txt). Some EN items
are wider than the JP sprites or moved to free atlas space; fest_menu.py
writes the matching w/u/v into MINISEL.BIN. RE: private/re/fest/
findings.org. Not painted: the player-select popup names and the 遅/速
icons.
"""
import struct, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

ATLAS_END = 0x181a0        # end of TIM 0; the rest of the file is not touched
PIX, STRIDE, W, H = 0x1a0, 384, 768, 256
FONT = 'DejaVuSansCondensed-Bold.ttf'
# id: (x, y, w, h, clut, align); a number = ink centre in the sprite.
# The record label is centred on the score field below it (digits and
# unit, x 361-452 for 99.99s; the sprite starts at screen x 322). The
# menu items are centred on the cursor bar (screen x 184-357, centre
# 270.5); their sprites start at x 192 (Game Start 200).
ATLAS = {
    'title_start': (0, 96, 144, 24, 1, 70.5),
    'title_back': (144, 96, 40, 24, 1, 'centre'),
    'title_1p': (0, 120, 160, 24, 1, 78.5),
    'title_vs': (0, 144, 160, 24, 1, 78.5),
    'title_rules': (0, 168, 160, 24, 1, 78.5),
    'title_record': (0, 216, 160, 24, 2, 84),
    'unit_pt': (160, 216, 16, 24, 2, 'left'),
    # Festival menu (MINISEL lists 0-7): left-aligned in the menu column
    # (screen x 38, as the JP items). Atlas page 2 rows 192-255 (x 512-767)
    # are empty in the TIM and in VRAM at run time; PK moved there because
    # 1P/VS glyphs follow the JP piece at x 152.
    'f_hr': (0, 0, 208, 24, 0, 'left'),
    'f_pk': (512, 192, 192, 24, 0, 'left'),
    'f_shoot': (0, 48, 216, 24, 0, 'left'),
    'f_ace': (0, 72, 184, 24, 0, 'left'),
    'f_sprint': (512, 48, 224, 24, 0, 'left'),
    'f_dance': (512, 24, 224, 24, 0, 'left'),
    'f_watch': (512, 168, 112, 24, 0, 'left'),
    'f_back': (512, 0, 208, 24, 0, 'left'),
    # record label (MINISEL lists 8-13, screen x 323): centred on the
    # preview picture frame above it (screen x 330-441).
    'f_record': (0, 192, 120, 24, 2, 62.5),
}
# Explicit CLUT indices for labels whose JP art is not the simple
# shadow/edge/fill scheme. Festival items (CLUT 0, painter item_jp): the
# JP look of the items, Roboto Condensed Medium Italic 18 with 1 px
# tracking, purple row gradient 5/6/7 down the cap height and one light
# 13 row above the baseline, AA edge 4, navy 2 outline on the top/left
# and a maroon 12 drop shadow down-right (mockup B, private/re/fest/mock).
# Record label (CLUT 2): red fill in a light outline.
FONT_RCMI = 'RobotoCondensed-MediumItalic.ttf'
_ITEM = dict(painter='item_jp', font=FONT_RCMI, size=18, track=1, base=18,
             cap=0.711, outline=2, shadow=12, edge=4, grad=(5, 6, 7), light=13)
STYLE = {k: _ITEM for k in ATLAS if k.startswith('f_') and k != 'f_record'}
STYLE['f_record'] = dict(font=FONT, size=16, shadow=None, outline=14, edge=8, fill=7, light=None, ring=2)

def _cov(text, font, size, track):
    """1x hinted coverage (32 rows, baseline row 22) cropped to the ink
    columns; track = extra px after each letter but spaces (kerning kept)."""
    f = fontpath.truetype(font, size)
    im = Image.new('L', (420, 32), 0)
    dr = ImageDraw.Draw(im)
    x = 4.0
    for i, ch in enumerate(text):
        dr.text((x, 22), ch, font=f, fill=255, anchor='ls')
        nxt = text[i:i + 2]
        x += (f.getlength(nxt) - f.getlength(text[i + 1:i + 2])) if len(nxt) == 2 else 0
        x += track if ch != ' ' else 0
    a = np.asarray(im) / 255.0
    xs = np.where((a > 0.15).any(0))[0]
    return a[:, xs[0]:xs[-1] + 1]

def item_jp(text, w, h, st):
    """Festival item in the JP look (STYLE painter 'item_jp'): index image
    (h, w), ink from column 1, baseline at row st['base']. Shrinks the
    size until the ink plus outline fits. Returns (out, size)."""
    size, base = st['size'], st['base']
    while True:
        c = _cov(text, st['font'], size, st['track'])
        if c.shape[1] <= w - 3 or size <= 10: break
        size -= 1
    a = np.zeros((h, w))
    src = c[22 - base:22 - base + h]
    a[:src.shape[0], 1:1 + src.shape[1]] = src[:, :w - 1]
    def sh(m):
        o = np.zeros_like(m); o[1:, 1:] = m[:-1, :-1]; return o
    edge, core = a > 0.3, a > 0.5
    o = ndimage.binary_dilation(edge, structure=np.ones((3, 3), bool))
    out = np.zeros((h, w), np.uint8)
    out[sh(o)] = st['shadow']
    out[o] = st['outline']
    out[sh(core) & ~core & ~edge] = st['shadow']
    out[edge & ~core] = st['edge']
    cap = base - round(size * st['cap'])
    g = np.zeros((h, w), np.uint8)
    for r in range(h):
        t = (r - cap) / (base - cap)
        g[r] = (st['grad'][0] if t < 0.2 else st['grad'][1] if t < 0.6 else
                st['grad'][2] if t < 0.84 or r >= base else st['light'])
    out[core] = g[core]
    return out, size

def styled(a, st):
    """Index image of alpha a in style st (STYLE entry)."""
    fp = np.ones((3, 3)) if st.get('ring', 1) == 1 else np.array(
        [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    o = ndimage.grey_dilation(a, footprint=fp)
    out = np.zeros(a.shape, np.uint8)
    if st['shadow'] is not None:
        sh = np.zeros_like(a); sh[1:, 1:] = o[:-1, :-1]
        out[sh > 0.4] = st['shadow']
    out[o > 0.4] = st['outline']
    out[a > 0.3] = st['edge']
    core = a > 0.55
    out[core] = st['fill']
    if st['light'] is not None:          # top facing edge of the fill
        up = np.zeros_like(core); up[1:] = core[:-1]
        out[core & ~up] = st['light']
    return out

def load_tl(paths=('tl/MINIHUD.txt', 'tl/FEST.txt')):
    out = {}
    if isinstance(paths, str): paths = (paths,)
    for line in (l for p in paths for l in open(p, encoding='utf-8')):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def mask(text, w, h, size, align, ss=4, font=FONT):
    """Alpha (h, w) of one line; squeezes to 0.7, then shrinks, to fit
    (N1 build_title.mask)."""
    while True:
        f = fontpath.truetype(font, int(size * ss))
        b = f.getbbox(text)
        tw = b[2] - b[0]
        sx = min(1.0, (w - 4) * ss / max(1, tw))
        if sx >= 0.7 or size < 8: break
        size -= 1
    asc, desc = b[1], b[3]
    im = Image.new('L', (tw + 2 * ss, desc - asc + 2 * ss), 0)
    ImageDraw.Draw(im).text((ss - b[0], ss - asc), text, font=f, fill=255)
    nw = max(1, int(im.width * sx / ss)); nh = max(1, int(im.height / ss))
    sm = np.asarray(im.resize((nw, nh), Image.LANCZOS)) / 255.0
    a = np.zeros((h, w))
    y0 = max(0, (h - nh) // 2)
    if align == 'left': x0 = 0
    elif align == 'centre': x0 = max(0, (w - 1 - nw) // 2)
    else: x0 = max(0, min(w - 2 - nw, int(np.floor(align - (nw - 1) / 2 + 0.5))))
    hh, ww = min(nh, h - y0), min(nw, w - x0)
    a[y0:y0 + hh, x0:x0 + ww] = sm[:hh, :ww]
    return a, size, sx, nw, x0

def paint(d, tl=None, log=None):
    """Repaint the labels in buffer d (a decompressed MINITTLn.BIZ) in
    place. Returns {id: (before, after, 16-colour CLUT)}."""
    tl = tl or load_tl()
    nclut = struct.unpack_from('<HH', d, 0x10)
    clut = struct.unpack_from(f'<{nclut[0] * nclut[1]}H', d, 0x14)
    raw = np.frombuffer(bytes(d[PIX:PIX + STRIDE * H]), np.uint8).reshape(H, STRIDE)
    px = np.empty((H, W), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    prev = {}
    for rid, (x, y, w, h, c, align) in ATLAS.items():
        if rid not in tl: continue
        pc = clut[c * 16:c * 16 + 16]
        sub = px[y:y + h, x:x + w]
        if rid in STYLE:
            st = STYLE[rid]
            if st.get('painter') == 'item_jp':
                assert align == 'left', rid
                out, size = item_jp(tl[rid], w, h, st); sx = 1.0
            else:
                a, size, sx, tw, tx = mask(tl[rid], w, h, st['size'], align, font=st['font'])
                out = styled(a, st)
        else:
            cnt = collections.Counter(sub.ravel().tolist())
            used = [i for i, n in cnt.items() if i and n > 2]
            ramp = sorted(used, key=lambda i: lum(pc[i]))
            main = max((i for i in used if i != ramp[0]), key=lambda i: cnt[i])
            a, size, sx, tw, tx = mask(tl[rid], w, h, 16, align)
            o = ndimage.grey_dilation(a, footprint=np.ones((3, 3)))
            sh = np.zeros_like(a); sh[1:, 1:] = o[:-1, :-1]
            out = np.zeros((h, w), np.uint8)
            out[np.maximum(o, sh) > 0.4] = ramp[0]
            edge = sorted(used, key=lambda i: abs(lum(pc[i]) - (lum(pc[main]) + lum(pc[ramp[0]])) / 2))[0]
            out[a > 0.3] = edge
            out[a > 0.55] = main
        if not isinstance(align, str):          # move the ink centre to align
            ic = np.where((out > 0).any(0))[0]
            dx = int(np.floor(align - (ic[0] + ic[-1]) / 2 + 0.5))
            dx = max(-ic[0], min(w - 1 - ic[-1], dx))
            out = np.roll(out, dx, 1)
        prev[rid] = (sub.copy(), out, pc)
        px[y:y + h, x:x + w] = out
        if log is not None:
            ic = np.where((out > 0).any(0))[0]
            log.append(f'{rid}: "{tl[rid]}" ink {ic[0]}-{ic[-1]} of {w} (centre '
                       f'{(ic[0] + ic[-1]) / 2:.1f}), size {size}, sx {sx:.2f}')
    d[PIX:PIX + STRIDE * H] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    return prev

def ink_span(rid, prev):
    """(first, last) ink column of a painted label inside its sprite."""
    ic = np.where((prev[rid][1] > 0).any(0))[0]
    return int(ic[0]), int(ic[-1])
