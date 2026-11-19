# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Year-end Team Up Technique select (3/24) in English.
Patch module for tools/build_mgbg.py; reads tl/TEAMUP.txt (entry 0 the
prompt, entries 1-22 the move names in record order).

Pixels: a 4bpp 1024x256 TIM at offset 0xf800 of MGBGE03.BZZ entry 22
(entry 22 also holds background 020 at offset 0, patched by
mgbg_captions.py). The game uploads it to VRAM x 640 (the header says
512), CLUTs at (0,485). The prompt and the names sit in three 4bpp
tpages, 256 texels wide each: tpage 704 (TIM x 256-511), 768 (512-767)
and 832 (768-1023). The LEGION OF HEROES logos (TIM x 0-255 and tpage 704
v 0-99) are English and stay.
- Prompt: two sprites, 20 px tall, at tpage 704 u 0 v 100 (240 wide) and
  v 120 (200 wide), drawn side by side at screen (16,40) and (256,40),
  CLUT 4 (red-orange ramp). Code: MGAME 0x800c8764 (x 0x10 at 0x800c877c,
  v 0x64 at 0x800c8784, widths 0xf0/0xc8 at 0x800c8878/0x800c8874).
- Names: 22 records at MGAME 0x800a523c, {u8 tpage x / 64, u8 v, u16 w}
  (w is read with lbu, so at most 255), u 0, height 20, CLUT 6 (grey
  ramp). Id k is row k // 2, column k % 2, drawn at x 56 + 180 * col, y 82
  + 32 * row (0x800c8de0). The chosen name is the same sprite at (96,222)
  on a box from x 60 to w + 102 (0x800c90d8), so the box follows w.
- Cursor bar: TILE at x 54 + 180 * col, width 148 (left) or 244 (right),
  immediates at 0x800b8ee0/0x800b8ed8. Unchanged. Left names must end
  before the right column's face icons (x ~204): w <= LEFT_MAX.

Layout: every name keeps its JP cell (tpage, v) and is redrawn from u 0
in its own width; MGAME_PATCH writes the new widths. The prompt is one
line drawn into a 440 px strip (the two sprites side by side), centred on
the screen centre (x 256).
Style (STYLE): white Roboto Condensed Bold (OFL), cap 13 px, with a 1 px
dark outline and a dark 1 px drop shadow down-right, as the JP names; the
prompt in the red-orange ramp of CLUT 4. The other STYLES are the variants
that were compared (private/re/teamup/findings.org).

Usage: python3 tools/mgbg_teamup.py --png OUT.png [--styles A,B]  preview
(needs the clean background shots in private/re/teamup/bg/).
"""
import os, sys, struct, functools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath

ARCH, ENTRY, TIM_OFF = 'MGBGE03.BZZ', 22, 0xf800
OVL = 0x800a2000
REC = 0x800a523c                       # 22 x {u8 tp, u8 v, u16 w}
JP_REC = [(11, 140, 80), (11, 160, 100), (11, 180, 124), (11, 200, 180),
          (11, 220, 120), (12, 0, 220), (12, 20, 60), (12, 40, 160),
          (12, 60, 80), (12, 80, 180), (12, 100, 60), (12, 120, 140),
          (12, 140, 120), (12, 160, 60), (12, 180, 80), (12, 200, 100),
          (12, 220, 140), (13, 0, 220), (13, 20, 140), (13, 40, 160),
          (13, 60, 100), (13, 80, 240)]
VRAM_X = 640                           # runtime VRAM x of the TIM
PROMPT = ((100, 240), (120, 200))      # tpage 704 rows: v, sprite width
PROMPT_X = 16                          # screen x of the first prompt sprite
H = 20
LEFT_MAX, RIGHT_MAX = 144, 240         # name ink limits (px)
CLUT_NAME, CLUT_PROMPT = 6, 4
SS = 4

ROBOTO = 'RobotoCondensed-Bold.ttf'                       # OFL 1.1
BARLOW = 'BarlowCondensed-Bold.ttf'          # OFL 1.1
BARLOW_XB = 'BarlowCondensed-ExtraBold.ttf'     # OFL 1.1
STYLES = {
    # names: font, cap px, baseline row, fill top/bottom grey;
    # prompt: pfont, pcap, pbase, pcaps (all capitals)
    'roboto': dict(font=ROBOTO, cap=13, base=15, top=248, bot=248,
                   pfont=ROBOTO, pcap=13, pbase=15, pcaps=False),
    'roboto-grad': dict(font=ROBOTO, cap=13, base=15, top=248, bot=200,
                        pfont=ROBOTO, pcap=13, pbase=15, pcaps=False),
    'barlow': dict(font=BARLOW, cap=14, base=16, top=248, bot=248,
                   pfont=BARLOW, pcap=14, pbase=16, pcaps=False),
    'roboto-grad-caps': dict(font=ROBOTO, cap=13, base=15, top=248, bot=200,
                             pfont=ROBOTO, pcap=15, pbase=17, pcaps=True),
    'roboto-grad-bxb': dict(font=ROBOTO, cap=13, base=15, top=248, bot=200,
                            pfont=BARLOW_XB, pcap=15, pbase=17, pcaps=True),
}
STYLE = 'roboto-grad-caps'
EDGE, SHADOW = 24, 8                   # outline and drop shadow grey
P_TOP, P_BOT = (248, 104, 40), (216, 88, 40)   # prompt fill (CLUT 4 ramp)
P_EDGE, P_SHADOW = (40, 16, 40), (40, 16, 40)
KEY = 0.35                             # coverage below this = transparent

def load_tl(path='tl/TEAMUP.txt'):
    out = []; jp = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            out.append((jp, line[4:].strip())); jp = None
    assert len(out) == 23 and all(e for _, e in out), 'tl/TEAMUP.txt: 23 filled entries'
    return out

# ---------------------------------------------------------------- TIM
@functools.lru_cache(None)
def orig_entry():
    import lz, arc
    d = open('private/disc/DAT2/' + ARCH, 'rb').read()
    o, s = arc.toc(d)[ENTRY]
    return lz.decompress(d[o:o + s])[0]

def layout(d):
    """-> (CLUTs 32 x 16 u16, pixel offset); asserts the 1024x256 4bpp TIM."""
    assert struct.unpack_from('<II', d, TIM_OFF) == (0x10, 0x8)
    cl = struct.unpack_from('<I', d, TIM_OFF + 8)[0]
    assert struct.unpack_from('<4H', d, TIM_OFF + 12) == (0, 490, 512, 1)
    q = TIM_OFF + 8 + cl
    assert struct.unpack_from('<4H', d, q + 4) == (512, 0, 256, 256)
    clut = np.frombuffer(d, '<u2', 512, TIM_OFF + 20).reshape(32, 16)
    return clut.copy(), q + 12

def unpack(d):
    _, p = layout(d)
    b = np.frombuffer(d, np.uint8, 512 * 256, p).reshape(256, 512)
    return np.stack([b & 15, b >> 4], -1).reshape(256, 1024)

def pack(d, idx):
    _, p = layout(d)
    b = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    out = bytearray(d); out[p:p + 512 * 256] = b.tobytes()
    return bytes(out)

def rgb(clut):
    c = clut.astype(int)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(float)

def cell_x(tp):
    """TIM pixel x of u 0 in tpage tp (VRAM x tp * 64)."""
    return (tp * 64 - VRAM_X) * 4

# ---------------------------------------------------------------- render
@functools.lru_cache(None)
def font(path, cap):
    for sz in range(cap * SS, cap * SS * 3):
        f = fontpath.truetype(path, sz)
        if -f.getbbox('H', anchor='ls')[1] >= cap * SS: return f
    raise SystemExit('mgbg_teamup: no font size for the cap height')

def down(a):
    h, w = a.shape
    return a.reshape(h // SS, SS, w // SS, SS).mean((1, 3))

def shift(a, dx, dy):
    out = np.zeros_like(a)
    out[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return out

def dilate(a, r):
    """Disc max filter of radius r (no wrap-around)."""
    h, w = a.shape
    p = np.pad(a, r)
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + r: continue
            out = np.maximum(out, p[r + dy:r + dy + h, r + dx:r + dx + w])
    return out

def layers(text, path, cap, base, width):
    """-> fill, outline, shadow coverage (H x width) with the ink from x 1."""
    f = font(path, cap)
    im = Image.new('L', (width * SS, H * SS), 0)
    ImageDraw.Draw(im).text((SS, base * SS), text, font=f, fill=255, anchor='ls')
    F = np.asarray(im, float) / 255
    O = dilate(F, SS)
    S = shift(O, SS, SS)
    return down(F), down(O), down(S)

def paint(F, O, S, top, bot, edge, shadow, pal, allowed):
    """Quantise to the palette indices in `allowed`; -> (idx, ink width)."""
    top, bot, edge, shadow = (np.array(c, float).reshape(-1)[:3] if np.ndim(c) else np.array([c] * 3, float)
                              for c in (top, bot, edge, shadow))
    t = np.linspace(0, 1, H)[:, None, None]
    fill = top + (bot - top) * t
    back = np.where((O >= S)[..., None], edge, shadow)
    col = F[..., None] * fill + (1 - F[..., None]) * back
    a = np.maximum(np.maximum(F, O), S)
    P = pal[allowed]
    k = ((col[:, :, None, :] - P[None, None]) ** 2).sum(-1).argmin(-1)
    idx = np.array(allowed, np.uint8)[k]
    idx[a < KEY] = 0
    cols = np.nonzero(idx.any(0))[0]
    return idx, (int(cols[-1]) + 1 if len(cols) else 0)

def grey_allowed(pal):
    return [i for i in range(1, 16) if np.ptp(pal[i]) == 0]

@functools.lru_cache(None)
def build(style=STYLE):
    """-> (new TIM index array, widths[22], prompt strip idx)."""
    st = STYLES[style]
    tl = load_tl()
    clut, _ = layout(orig_entry())
    idx = unpack(orig_entry()).copy()
    pn, pp = rgb(clut[CLUT_NAME]), rgb(clut[CLUT_PROMPT])
    widths = []
    for k, ((tp, v, _), (_, en)) in enumerate(zip(JP_REC, tl[1:])):
        cell, w = paint(*layers(en, st['font'], st['cap'], st['base'], 256), st['top'], st['bot'], EDGE, SHADOW,
                        pn, grey_allowed(pn))
        lim = LEFT_MAX if k % 2 == 0 else RIGHT_MAX
        if w > lim: raise SystemExit(f'mgbg_teamup: "{en}" is {w} px, over {lim}')
        x = cell_x(tp)
        idx[v:v + H, x:x + 256] = cell
        widths.append(w)
    # prompt: one strip over both sprites, centred on screen x 256
    en = tl[0][1].upper() if st['pcaps'] else tl[0][1]
    cell, w = paint(*layers(en, st['pfont'], st['pcap'], st['pbase'], 512), P_TOP, P_BOT, P_EDGE, P_SHADOW,
                    pp, list(range(2, 15)))
    total = sum(pw for _, pw in PROMPT)
    x0 = 256 - PROMPT_X - w // 2
    assert 0 <= x0 and x0 + w <= total, ('prompt too wide', w)
    strip = np.zeros((H, total), np.uint8)
    strip[:, x0:x0 + w] = cell[:, :w]
    u = 0
    for v, pw in PROMPT:
        x = cell_x(11)
        idx[v:v + H, x:x + 256] = 0
        idx[v:v + H, x:x + pw] = strip[:, u:u + pw]
        u += pw
    return idx, widths, strip

def patch_entry(data, style=STYLE):
    """Rewrite the TIM at TIM_OFF only (the background at 0 is another
    module's)."""
    o = orig_entry()
    _, p = layout(data)
    assert data[TIM_OFF:p + 512 * 256] == o[TIM_OFF:p + 512 * 256], 'TIM at 0xf800 already changed'
    idx, _, _ = build(style)
    return pack(data, idx)

PATCHES = [(ARCH, ENTRY, patch_entry)]

def MGAME_PATCH(d):
    """Name widths in the record table (applied by hook.py)."""
    _, widths, _ = build()
    out = bytearray(d)
    for k, ((tp, v, w0), w) in enumerate(zip(JP_REC, widths)):
        o = REC - OVL + 4 * k
        t, vv, ww = struct.unpack_from('<BBH', out, o)
        assert (t, vv) == (tp, v) and ww in (w0, w), ('MGAME team-up record', k, t, vv, ww)
        struct.pack_into('<H', out, o + 2, w)
    return bytes(out)

# ---------------------------------------------------------------- preview
BG = 'private/re/teamup/bg/'

def sprite(idx, clut, k, x, y, w, h, img, u=0, v=0, tp=11):
    pal = rgb(clut[k]).astype(np.uint8)
    cell = idx[v:v + h, cell_x(tp) + u:cell_x(tp) + u + w]
    a = np.asarray(img).copy()
    reg = a[y:y + h, x:x + w]
    m = cell > 0
    reg[m] = pal[cell[m]]
    return Image.fromarray(a)

def scene(style, badge_id=15):
    idx, widths, _ = build(style)
    clut, _ = layout(orig_entry())
    sel = Image.open(BG + 'sel15.png').convert('RGB')
    x = PROMPT_X
    for v, pw in PROMPT:
        sel = sprite(idx, clut, CLUT_PROMPT, x, 40, pw, H, sel, v=v); x += pw
    for k, (tp, v, _) in enumerate(JP_REC):
        sel = sprite(idx, clut, CLUT_NAME, 56 + 180 * (k % 2), 82 + 32 * (k // 2),
                     widths[k], H, sel, v=v, tp=tp)
    bad = Image.open(BG + 'badge2.png').convert('RGB')
    tp, v, _ = JP_REC[badge_id]
    bad = sprite(idx, clut, CLUT_NAME, 96, 222, widths[badge_id], H, bad, v=v, tp=tp)
    return sel, bad

def preview(out, styles=None):
    styles = styles or [STYLE]
    ims = []
    jp_sel = Image.open('private/re/final7/t7/d2480.png').convert('RGB')
    jp_bad = Image.open('private/re/final7/t7/c2.png').convert('RGB')
    rows = [('JP', jp_sel, jp_bad)] + [(s,) + scene(s) for s in styles]
    W, Hh = 512 + 352, 480
    sheet = Image.new('RGB', (W * min(2, len(rows)), (Hh + 14) * ((len(rows) + 1) // 2)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (name, s, b) in enumerate(rows):
        x0, y0 = (i % 2) * W, (i // 2) * (Hh + 14)
        dr.text((x0 + 4, y0), name, fill=(255, 255, 0))
        sheet.paste(s, (x0, y0 + 14))
        sheet.paste(b.crop((40, 180, 380, 270)), (x0 + 512 + 4, y0 + 14 + 10))
    sheet.save(out)

if __name__ == '__main__':
    if '--png' in sys.argv:
        st = None
        if '--styles' in sys.argv: st = sys.argv[sys.argv.index('--styles') + 1].split(',')
        preview(sys.argv[sys.argv.index('--png') + 1], st)
    else:
        _, widths, _ = build()
        for (jp, en), w in zip(load_tl()[1:], widths): print(f'{w:4d} {en}')
