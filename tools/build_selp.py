# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Character select of PLSEL1 in English: heading, Lead/Partner labels, the
name labels and the Exam Prep '99 popup. PLSEL1 serves Solo, Versus, Exam
Prep '99 and Training, so all four show the EN pieces. Wording: tl/SELP.txt.
RE: private/re/selp/findings.org.

Art: SELP1.BIZ TIM 2 (0x17800, 4bpp 1024x256 at VRAM (768,0), CLUT row 482).
- Heading: 32x32 cells of the grid at TIM x 256-511 y 0-127, CLUT 30. The
  EN line is drawn as one strip in the JP heading look (Rounded M+ Black,
  orange-to-yellow fill, black outline, red shadow) and cut into cells.
  Strings: PLSEL1 0x56c (count, then (col,row) per cell, max 11) and 0x584
  (Versus, max 3 cells). x = (512 - 32 n) / 2.
- Lead/Partner: 56x16 at TIM (0,240) and (56,240), CLUT 0 (yellow). The
  draw code 0x800a44b8 is patched from 40x20 at u 40*n, v 220, x+5 to
  56x16 at u 56*n, v 240, x-4 (centred on the 48 px portrait), y+2.
- Names: 48x16 cells, CLUT 1, table PLSEL1 0x2a0 (unchanged). Ids 0-23 at
  TIM (512+u, v); ids 24 (Ran) and 25 (Nagare) at TIM (768+u, 96) (tpage
  960, code 0x800a46e0). Names that the first game's USA label art has
  (private/usa/unpacked/SELP3.BIZ.lz TIM 2, same layout and CLUT)
  are copied from it; others are put together from its letters.
- Popup ('99 only): two rows of 20 px at TIM (876,112) and (876,132),
  CLUT 11 (grey ramp), records PLSEL1 0x954 {u,v,w,h} (tpage 960), x =
  (512 - w) / 2 in a 180 px box.

Writes private/build/SELP1.BIZ and private/build/PLSEL1.BIN (from the disc
copy; no other build script writes PLSEL1). Idempotent.
Options: --check (no writes), --png (previews in private/re/selp/prev/).
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import lz
from examprac import font, line_alpha, blend_idx, pad, shift, F_TITLE, FONT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SELP_SRC = 'private/unpacked/SELP1.BIZ.lz'
SELP_OUT = 'private/build/SELP1.BIZ'
SELP_SLOT = 109 * 2048               # LBA 157025, JP 221538 B
PLSEL_DISC = 'private/disc/DAT1/PLSEL1.BIN'
PLSEL_OUT = 'private/build/PLSEL1.BIN'
PLSEL_MAX = 12 * 2048
USA = 'private/usa/unpacked/SELP3.BIZ.lz'
USA_TIM = 0x15800
TIM2 = 0x17800
TL = 'tl/SELP.txt'
PREV = 'private/re/selp/prev'

# ---------------------------------------------------------------- data
HEAD_TBL, HEAD_MAX = 0x56c, 11       # 24 bytes up to 0x584
VS_TBL, VS_MAX = 0x584, 3            # 8 bytes up to 0x58c
# Cells (col, row) of the grid. The JP heading's 9 cells plus free ones;
# (0,0) is the blank, (4,0) (5,0) are JP 対戦 (left alone).
HEAD_CELLS = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (2, 1),
              (0, 3), (0, 2), (6, 0)]
VS_CELLS = [(1, 0), (2, 0), (3, 0)]
POP_REC = 0x954
POP_U, POP_V = 108, (112, 132)       # TIM (876, 112) and (876, 132)
POP_W = 148                          # u + w <= 256; the box is 180 wide
LAB_U, LAB_V, LAB_W, LAB_H = 0, 240, 56, 16
NAME_TBL = 0x2a0
# id -> tl key; ids 24-28 are placeholders except 24/25 (tpage 960)
IDS = ['kyosuke', 'batsu', 'hinata', 'shoma', 'natsu', 'roberto', 'roy',
       'tiffany', 'boman', 'edge', 'akira', 'gan', 'hideo', 'kyoko', 'raizo',
       'hyo', 'akira_b', 'sakura', 'daigo', 'hayato', 'hinata2', 'natsu2',
       'tiffany2', 'kyoko2', 'ran', 'nagare']
# USA SELP3 TIM 2 name cells, 48x16 at (512 + 48 (i % 5), 16 (i // 5))
USA_NAMES = ['BATSU', 'HINATA', 'KYOSUKE', 'SHOMA', 'NATSU', 'ROBERTO', 'ROY',
             'TIFFANY', 'BOMAN', 'EDGE', 'AKIRA', 'GAN', 'HIDEO', 'KYOKO',
             'RAIZO', 'HYO', 'SAKURA', 'DAIGO', 'HAYATO', 'akira', 'HINATA2',
             'NATSU2', 'AKIRA2', 'TIFFANY2', 'KYOKO2']
# Letters cut from those cells: (cell, first col, last col)
USA_LETTERS = {'N': (20, 13, 18), 'A': (20, 19, 25), 'G': (9, 24, 32),
               'R': (5, 0, 6), 'E': (5, 22, 27)}
# Label draw code 0x800a44b8.. (file offset, JP word, EN word)
LABEL_CODE = [
    (0x24b8, 0x00171080, 0x001710c0),   # sll v0,s7,2 -> sll v0,s7,3
    (0x24bc, 0x00571021, 0x00571023),   # addu v0,v0,s7 -> subu: u = 56 n
    (0x24c8, 0x240200dc, 0x240200f0),   # v 220 -> 240
    (0x24d0, 0x24020028, 0x24020038),   # w 40 -> 56
    (0x24d8, 0x24020014, 0x24020010),   # h 20 -> 16
    (0x24e8, 0x25220005, 0x2522fffc),   # x = portrait x + 5 -> - 4
    (0x24f0, 0x27c2ffe8, 0x27c2ffea),   # y = fp - 24 -> fp - 22
]


def load_tl(path=TL):
    tl = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        tl[k.strip()] = v.strip()
    return tl


def tim4(d, off):
    """(pixel offset, CLUTs (n,16) u16, index array h x w) of a 4bpp TIM."""
    bl, cx, cy, cw, ch = struct.unpack_from('<IHHHH', d, off + 8)
    cl = np.array(struct.unpack_from('<%dH' % (cw * ch), d, off + 20)).reshape(-1, 16)
    p = off + 8 + bl
    bl2, x, y, w, h = struct.unpack_from('<IHHHH', d, p)
    px = np.frombuffer(d[p + 12:p + 12 + w * 2 * h], np.uint8).reshape(h, w * 2)
    return p + 12, cl, np.stack([px & 15, px >> 4], -1).reshape(h, w * 4).copy()


def pal_rgb(clut):
    c = np.array(clut, int)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1)


# ---------------------------------------------------------------- styles
# JP heading (CLUT 30): fill 6 (orange) at the top to 9 (yellow) at the
# bottom, 1 near-black outline, red shadow 12/13 down-right.
HEAD_RAMP = ((0.30, 6), (0.50, 7), (0.68, 8), (9, 9))
HEAD_STYLE = {'heading': dict(cap=23, track=0.3), 'versus': dict(cap=18, track=0)}


def head_alpha(text, w, cap=21, track=0.6, path=F_TITLE, h=32):
    """Alpha (h, w): text at cap px, tracked, squeezed to fit w - 8 and
    centred in w."""
    ss = 4
    probe = font(path, 100)
    b = probe.getbbox('H', anchor='ls')
    f = font(path, int(round(cap * 100 / (b[3] - b[1]) * ss)))
    adv = sum(f.getlength(c) for c in text) / ss + track * (len(text) - 1)
    W = int(np.ceil(adv)) + 2
    im = Image.new('L', (W * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    by = (h + cap) / 2 - 1.5
    x = 0.0
    for c in text:
        dr.text((x * ss, by * ss), c, font=f, fill=255, anchor='ls')
        x += f.getlength(c) / ss + track
    sx = min(1.0, (w - 8) / W)
    im = im.resize((max(1, int(round(W * sx))), h), Image.BOX)
    a = np.asarray(im, np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    a = a[:, cols.min():cols.max() + 1]
    out = np.zeros((h, w), np.float32)
    x0 = (w - a.shape[1] - 2) // 2          # the shadow adds 2 px right
    out[:, x0:x0 + a.shape[1]] = a
    return out, sx


def st_head(a, bevel=True):
    core = a > 0.3
    o = ndimage.binary_dilation(core, ndimage.generate_binary_structure(2, 2))
    out = np.zeros(a.shape, np.uint8)
    out[shift(o, 2, 2) & ~o] = 13
    out[shift(o, 1, 1) & ~o] = 12
    out[o] = 1
    fill = a > 0.55
    ys = np.nonzero(fill.any(1))[0]
    y0, y1 = ys.min(), ys.max()
    tt = (np.arange(a.shape[0]) - y0) / max(1, y1 - y0)
    g = np.array([next(v for lim, v in HEAD_RAMP if t <= lim) for t in tt], np.uint8)
    out[core] = 5
    out[fill] = np.broadcast_to(g[:, None], a.shape)[fill]
    if bevel:      # darker top edge of each stroke, as the JP glyphs
        top = fill & ~shift(fill, 1, 0)
        out[top] = 5
    return out


def st_flat(a, face, pal, sh_idx=2):
    """Face colour with a dark AA ramp (indices 2-14 of a dark-to-face CLUT)
    and a 1 px dark shadow down-right: the JP popup and label look."""
    ramp = list(range(2, 15))
    t = blend_idx(a, face, (8, 8, 8), pal, ramp, cut=0.2)
    sh = shift(a, 1, 1) > 0.3
    out = np.zeros(a.shape, np.uint8)
    out[sh] = sh_idx
    return np.where(t > 0, t, out)


def flat_line(text, h, size, maxw, path=FONT, base=None):
    a = line_alpha(text, h, size, path, maxw=maxw - 1, base=base)
    return pad(a, 0, 1)


# ---------------------------------------------------------------- names
def usa_cells():
    d = open(os.path.join(ROOT, USA), 'rb').read()
    _, cl, idx = tim4(d, USA_TIM)
    cells = {}
    for i, n in enumerate(USA_NAMES):
        x, y = 512 + 48 * (i % 5), 16 * (i // 5)
        cells[n] = idx[y:y + 16, x:x + 48]
    return cells, cl


def name_art(text, cells, problems):
    """16 x w index art of a name in the USA label font."""
    if text in cells:
        c = cells[text]
        cols = np.nonzero(c.any(0))[0]
        return c[:, cols.min():cols.max() + 1]
    parts = []
    for ch in text:
        if ch not in USA_LETTERS:
            problems.append(f'name {text!r}: no USA letter {ch!r}'); return None
        i, c0, c1 = USA_LETTERS[ch]
        parts.append(cells[USA_NAMES[i]][:, c0:c1 + 1])
    return np.hstack(parts)          # USA letters touch (shadow column)


def put(idx, x, y, art, w, h, problems, what):
    """Clear the w x h cell at (x, y) and centre art in it."""
    if art.shape[1] > w or art.shape[0] > h:
        problems.append(f'{what}: {art.shape[1]}x{art.shape[0]} > {w}x{h}'); return
    idx[y:y + h, x:x + w] = 0
    x0 = x + (w - art.shape[1]) // 2
    idx[y:y + art.shape[0], x0:x0 + art.shape[1]] = art


# ---------------------------------------------------------------- build
def build(tl, problems):
    sd = bytearray(open(os.path.join(ROOT, SELP_SRC), 'rb').read())
    pix, cl, idx = tim4(sd, TIM2)
    pal = [pal_rgb(c) for c in cl]
    pl = bytearray(open(os.path.join(ROOT, PLSEL_DISC), 'rb').read())
    res = {}

    # heading and Versus strips
    for key, cells, tbl, mx in (('heading', HEAD_CELLS, HEAD_TBL, HEAD_MAX),
                                ('versus', VS_CELLS, VS_TBL, VS_MAX)):
        text = tl[key]
        n = min(len(cells), mx, 10)
        a, sx = head_alpha(text, 32 * n, **HEAD_STYLE[key])
        if sx < 0.8: problems.append(f'{key}: squeezed to {sx:.2f}')
        art = st_head(a)
        used = [k for k in range(n) if art[:, 32 * k:32 * k + 32].any()]
        lo, hi = used[0], used[-1] + 1     # trim empty end cells
        art = art[:, 32 * lo:32 * hi]
        n = hi - lo
        res[key] = art
        tb = bytes([n]) + b''.join(bytes(c) for c in cells[:n])
        room = 1 + 2 * mx
        pl[tbl:tbl + room] = tb + bytes(room - len(tb))
        for k in range(n):
            c, r = cells[k]
            idx[32 * r:32 * r + 32, 256 + 32 * c:288 + 32 * c] = art[:, 32 * k:32 * k + 32]

    # Lead / Partner
    for k, key in enumerate(('lead', 'partner')):
        a = flat_line(tl[key], LAB_H - 1, 15, LAB_W - 2)
        art = st_flat(pad(a, 0, 0, 0, 1), (248, 248, 0), pal[0])
        res[key] = art
        put(idx, LAB_U + LAB_W * k, LAB_V, art, LAB_W, LAB_H, problems, key)
    for off, jp, en in LABEL_CODE:
        cur = struct.unpack_from('<I', pl, off)[0]
        if cur != jp: problems.append(f'PLSEL1 {off:#x}: {cur:#010x}, expected {jp:#010x}')
        struct.pack_into('<I', pl, off, en)

    # names
    cells, ucl = usa_cells()
    if not np.array_equal(ucl[1], cl[1]): problems.append('USA CLUT 1 differs from JP')
    tbl = struct.unpack_from('<60H', pl, NAME_TBL)
    for i, key in enumerate(IDS):
        text = tl['name_' + key]
        art = name_art(text, cells, problems)
        if art is None: continue
        u, v = tbl[2 * i], tbl[2 * i + 1]
        x = (768 if i in (24, 25) else 512) + u
        res['name_' + key] = art
        put(idx, x, v, art, 48, 16, problems, 'name_' + key)

    # popup
    idx[112:152, 768 + POP_U:1024] = 0
    recs = []
    for k, key in enumerate(('exam99', 'basics')):
        a = flat_line(tl[key], 19, 18, POP_W - 1)
        art = st_flat(pad(a, 0, 0, 0, 1), (216, 216, 216), pal[11])
        res[key] = art
        w = art.shape[1]
        if w > POP_W: problems.append(f'{key}: {w} px > {POP_W}')
        put(idx, 768 + POP_U, POP_V[k], art, w, 20, problems, key)
        recs += [POP_U, POP_V[k], w, 20]
    pl[POP_REC:POP_REC + 16] = struct.pack('<8H', *recs)

    sd[pix:pix + idx.size // 2] = (idx[:, 0::2] | idx[:, 1::2] << 4).astype(np.uint8).tobytes()
    return sd, pl, idx, pal, res


def preview(idx, pal, res, pl):
    """EN pieces drawn on the JP-free captures (emu/r5) as the game places
    them, next to the JP screen, at 1x."""
    os.makedirs(os.path.join(ROOT, PREV), exist_ok=True)
    def blit(img, art, p, x, y):
        rgb = p[art]
        m = art > 0
        sub = img[y:y + art.shape[0], x:x + art.shape[1]]
        sub[m] = rgb[m]
    shots = {}
    for name in ('cs', 'pop'):
        f = os.path.join(ROOT, f'private/re/selp/emu/r5/{name}.png')
        if os.path.exists(f): shots[name] = np.asarray(Image.open(f).convert('RGB')).copy()
    if not shots: return
    n = pl[HEAD_TBL]
    for name, img in shots.items():
        blit(img, res['heading'], pal[30], (512 - 32 * n) // 2, 24)
        for k, key in enumerate(('lead', 'partner')):
            px = (80, 160)[k]
            a = res[key]
            blit(img, a, pal[0], px - 4 + (LAB_W - a.shape[1]) // 2, 74)
        nm = res['name_batsu']
        for px in (80, 160):
            if name == 'cs' and px == 160: continue
            blit(img, nm, pal[1], px + (48 - nm.shape[1]) // 2, 168)
        if name == 'pop':
            for k, key in enumerate(('exam99', 'basics')):
                a = res[key]
                blit(img, a, pal[11], (512 - a.shape[1]) // 2, 320 + 40 * k)
        Image.fromarray(img).save(os.path.join(ROOT, PREV, f'{name}_en.png'))


def main(argv):
    check = '--check' in argv
    tl = load_tl(os.path.join(ROOT, TL)); problems = []
    sd, pl, idx, pal, res = build(tl, problems)
    c = lz.compress(bytes(sd), name='SELP1.BIZ')
    if len(c) > SELP_SLOT: problems.append(f'SELP1 compressed {len(c)} > slot {SELP_SLOT}')
    if lz.decompress(c)[0][:len(sd)] != bytes(sd): problems.append('SELP1 LZ round trip failed')
    if len(pl) > PLSEL_MAX: problems.append(f'PLSEL1 {len(pl)} > {PLSEL_MAX}')
    print(f'SELP1: {len(c)} bytes (slot {SELP_SLOT}); PLSEL1: {len(pl)} bytes; '
          f'heading {pl[HEAD_TBL]} cells, versus {pl[VS_TBL]} cells')
    if '--png' in argv:
        np.save(os.path.join(ROOT, 'private/re/selp/en_idx.npy'), idx)
        preview(idx, pal, res, pl)
    for p in problems: print('  ' + p)
    if not check and not problems:
        open(os.path.join(ROOT, SELP_OUT), 'wb').write(c)
        open(os.path.join(ROOT, PLSEL_OUT), 'wb').write(pl)
    print(f'SELP: problems {len(problems)}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
