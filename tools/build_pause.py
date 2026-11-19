# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""In-round pause menu (1P/2P PAUSE, Continue, Back) in English.

Port of the first game's tool. The same four-string block is in the
mini-games MGAME01/02/03/05/06, the Exam Practice fights SIKENGO1/2, the
diary term-exam fights SIKENG21/22, the Hayato fight SKHAY and the exe
(fight modes, item 2 = Back to Mode Menu). MGAME061, MGAME07, SIKEN
and SIKEN2 hold no block.
- Glyph strip: DAT2/OVER.TIZ, one 4bpp TIM 256x240 (pixels at 0x3c0, 128
  bytes per row). Cells are 24x12 at u 192-239, v 0-203 (17 rows x 2).
  The strip and the CLUTs are byte-identical to the first game's.
- String record: u8 width_cells, u8 count, count x (a, b, c), padded to 4
  bytes, with u = (24a - 64) & 0xff, v = 12b, w = 24c.
  Screen x0 = (512 - 24*width)/2. Each file holds the records in one block
  followed by a 4-entry pointer table (1P title, 2P title, item 1, item 2).
- Title CLUT 28 (cyan), items CLUT 6 (yellow). Index 1 is the dark
  outline, 5-15 the face ramp (15 brightest).
Reads tl/PAUSE.txt and private/unpacked/OVER.TIZ.lz. Patches the
private/build copies written by build_testmj.py (SIKENGO1/2, SIKENG21/22),
build_examgfx.py --extra (SIKENG21/22), build_minihud.py (MGAME01-06) and
hook.py/build_sysmsg.py/build_cardtitle.py/build_cock.py (exe), so it runs
after build_cock.py. Files without a build copy come from the disc (SKHAY;
the exe must exist). The old block bytes are asserted. Writes
private/build/OVER.TIZ (+ .lz) and the patched files. --check writes
nothing; --png writes the preview. Facts: private/re/pause/findings.org.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz
from game import EXE_NAME

PIX, STRIDE = 0x3c0, 128
STRIP_U, STRIP_ROWS = 192, 17            # u 192-239, v 0-203
CW, CH = 24, 12
CLUT_TITLE, CLUT_ITEM = 28, 6
SECTORS = 9

# name: (base, block offset, block size, ptr table offset, file header)
FILES = {
    'MGAME01.BIN': (0x800a2000, 0x51c, 52, 0x550, 0),
    'MGAME02.BIN': (0x800a2000, 0xc8, 52, 0xfc, 0),
    'MGAME03.BIN': (0x800a2000, 0xc8, 52, 0xfc, 0),
    'MGAME05.BIN': (0x800a2000, 0x550, 52, 0x584, 0),
    'MGAME06.BIN': (0x800a2000, 0x5a4, 52, 0x5d8, 0),
    'SKHAY.BIN': (0x800a2000, 0x34, 52, 0x68, 0),
    'SIKENGO1.BIN': (0x80152000, 0x2c4, 52, 0x2f8, 0),
    'SIKENGO2.BIN': (0x8017a000, 0x2c4, 52, 0x2f8, 0),
    'SIKENG21.BIN': (0x80152000, 0x304, 52, 0x338, 0),
    'SIKENG22.BIN': (0x8017a000, 0x304, 52, 0x338, 0),
    EXE_NAME: (0x80010000, 0x7a870, 72, 0x7a8b8, 0x800),
}
# Box quad after each pointer table (x0, y0, x1, y1 ...): overlays 148-364,
# exe 112-400, y 90-150.
# JP strings in the order of the pointer table; the exe's item 2 differs.
JP_OV = ['1Pポーズ', '2Pポーズ', 'つづける', '戻る']
JP_EXE = ['1Pポーズ', '2Pポーズ', 'つづける', 'モードメニューに戻る']

# Glyph style. cap: capital height in px, top: row of the cap top,
# xs: horizontal stretch (the menu runs in 512-wide mode), track: extra
# px between letters, grad: face index per cell row 0-11.
STYLE = dict(font='ariblk.ttf', cap=11, top=0, xs=1.5, track=1.0,
             shear=0.12, outline=8, shadow=(1, 1),
             grad=[15, 15, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13])


def load_tl(path='tl/PAUSE.txt'):
    out, jp = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None: out[jp] = line[4:]; jp = None
    return out


def coverage(text, st, ss=8, trim=True):
    """Float coverage, 12 rows high, width trimmed to the ink if trim."""
    f0 = fontpath.truetype(st['font'], 100)
    capb = f0.getbbox('H')
    size = st['cap'] * ss * 100 / (capb[3] - capb[1])
    f = fontpath.truetype(st['font'], round(size))
    cb = f.getbbox('H')
    H = CH * ss
    W = int(sum(f.getlength(ch) for ch in text) + len(text) * st['track'] * ss / st['xs'] + 4 * ss)
    im = Image.new('L', (W, H))
    dr = ImageDraw.Draw(im)
    x = 2 * ss
    y = st['top'] * ss - cb[1]
    for ch in text:
        dr.text((x, y), ch, font=f, fill=255)
        x += f.getlength(ch) + st['track'] * ss / st['xs']
    im = im.resize((round(W * st['xs']), H), Image.LANCZOS)
    if st['shear']:
        k = st['shear']; w = im.width + int(k * H) + 1
        im = im.transform((w, H), Image.AFFINE, (1, k, -k * H, 0, 1, 0), Image.BICUBIC)
    w1 = (im.width + ss - 1) // ss
    big = Image.new('L', (w1 * ss, H)); big.paste(im, (0, 0))
    a = np.asarray(big.resize((w1, CH), Image.BOX), np.float32) / 255
    if not trim: return a
    cols = np.where(a.max(0) > 0.06)[0]
    return a[:, cols[0]:cols[-1] + 1] if len(cols) else a[:, :0]


def to_index(a, st, pad=3):
    """Coverage -> 4bpp indices with outline and shadow. Pads 3 px each side."""
    h, w = a.shape
    a = np.pad(a, ((0, 0), (pad, pad)))
    face = a >= 0.5
    ink = a >= 0.12
    out = np.zeros(a.shape, np.uint8)
    fp = np.ones((3, 3), bool) if st['outline'] == 8 else ndimage.generate_binary_structure(2, 1)
    ol = ndimage.binary_dilation(face, fp) if st['outline'] else np.zeros_like(face)
    dx, dy = st['shadow']
    sh = np.zeros_like(face)
    if dx or dy:
        src = ndimage.binary_dilation(face, fp) if st['outline'] else face
        sh[dy:, dx:] = src[:h - dy if dy else h, :a.shape[1] - dx]
    out[ol | sh] = 1
    g = np.array(st['grad'], np.float32)[:, None]
    # face: gradient index; edge pixels fall toward the dark end of the ramp
    idx = np.where(a >= 0.85, g, 5 + (g - 5) * np.clip((a - 0.12) / 0.73, 0, 1))
    out[ink] = np.clip(np.rint(idx[ink]), 5, 15).astype(np.uint8)
    return out


def place(g, cells=None, dx=0, text=''):
    """Centre index image g (cropped to its ink) in 24-px cells, moved by dx."""
    n = cells or -(-(g.shape[1] + 4) // CW)
    x = (n * CW - g.shape[1]) // 2 + dx
    if x < 0 or x + g.shape[1] > n * CW: raise ValueError(f'{text!r} does not fit {n} cells')
    out = np.zeros((CH, n * CW), np.uint8)
    out[:, x:x + g.shape[1]] = g
    return out


def crop(*gs):
    """Crop images of equal width to the union of their ink columns."""
    c = np.where(np.any([g.any(0) for g in gs], 0))[0]
    return [g[:, c[0]:c[-1] + 1] for g in gs]


def line_image(text, st, cells=None, dx=0):
    return place(crop(to_index(coverage(text, st), st))[0], cells, dx, text)


def title_images(tl, st):
    """'1P PAUSE' and '2P PAUSE' share every cell but the first, which must
    hold all of the digit. Returns (line 1P, line 2P, width in cells)."""
    t1, t2 = tl['1Pポーズ'], tl['2Pポーズ']
    assert t1[1:] == t2[1:], 'titles must differ only in the digit'
    g1, g2 = crop(*(to_index(coverage(t, st, trim=False), st) for t in (t1, t2)))
    n = -(-(g1.shape[1] + 4) // CW)
    for dx in sorted(range(-8, 9), key=abs):
        try: a, b = place(g1, n, dx, t1), place(g2, n, dx, t2)
        except ValueError: continue
        if (a[:, CW:] == b[:, CW:]).all(): return a, b, n
    raise ValueError('title digit does not fit in the first cell')


class Strip:
    """Packs pieces of 1 or 2 cells into the 17x2 cell strip."""
    def __init__(self):
        self.px = np.zeros((STRIP_ROWS * CH, 2 * CW), np.uint8)
        self.free = [[True, True] for _ in range(STRIP_ROWS)]
        self.cache = {}

    def put(self, img):
        c = img.shape[1] // CW
        key = img.tobytes()
        if key in self.cache: return self.cache[key]
        for b in range(STRIP_ROWS):
            for a in range(3 - c):
                if all(self.free[b][a:a + c]):
                    self.px[b * CH:(b + 1) * CH, a * CW:(a + c) * CW] = img
                    for k in range(a, a + c): self.free[b][k] = False
                    self.cache[key] = (a, b, c); return a, b, c
        raise ValueError('pause strip full')

    def pieces(self, img):
        """Split a line image into 2-cell pieces (+ one 1-cell piece)."""
        n = img.shape[1] // CW; out = []; x = 0
        while x < n:
            c = 2 if n - x >= 2 else 1
            sub = img[:, x * CW:(x + c) * CW]
            if c == 2 and not sub.any(): c = 1; sub = img[:, x * CW:(x + 1) * CW]
            out.append(self.put(sub)); x += c
        return out


def record(width, pieces):
    b = bytes([width, len(pieces)])
    for a, bb, c in pieces: b += bytes([a, bb, c])
    return b


def build_all(tl, st=STYLE):
    """Returns the strip pixels, the line images and the two record sets."""
    s = Strip()
    l1, l2, wt = title_images(tl, st)
    t1 = [s.put(l1[:, :CW])] + s.pieces(l1[:, CW:])
    t2 = [s.put(l2[:, :CW])] + s.pieces(l2[:, CW:])
    lines = {'title1': l1, 'title2': l2}
    recs = {}
    for k in ('つづける', '戻る', 'モードメニューに戻る'):
        img = line_image(tl[k], st)
        lines[k] = img
        recs[k] = record(img.shape[1] // CW, s.pieces(img))
    ov = [record(wt, t1), record(wt, t2), recs['つづける'], recs['戻る']]
    exe = [record(wt, t1), record(wt, t2), recs['つづける'], recs['モードメニューに戻る']]
    return s, lines, ov, exe


def tim_load():
    return bytearray(open('private/unpacked/OVER.TIZ.lz', 'rb').read())


def tim_px(d):
    raw = np.frombuffer(bytes(d[PIX:PIX + STRIDE * 240]), np.uint8).reshape(240, STRIDE)
    px = np.empty((240, 256), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px


def tim_store(d, px):
    d[PIX:PIX + STRIDE * 240] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()


def patch_block(d, name, recs):
    base, off, size, ptab, hdr = FILES[name]
    blob = b''; ptrs = []
    for r in recs:
        ptrs.append(base + off - hdr + len(blob)); blob += r
    if len(blob) > size: raise ValueError(f'{name}: records {len(blob)} > {size} bytes')
    d[off:off + size] = blob + bytes(size - len(blob))
    for i, p in enumerate(ptrs): struct.pack_into('<I', d, ptab + 4 * i, p)
    return len(blob)


def check_block(d, disc, name):
    """The block and pointer table must hold the disc bytes or our output."""
    base, off, size, ptab, hdr = FILES[name]
    a, b = d[off:ptab + 16], disc[off:ptab + 16]
    if a == b: return 'disc'
    p = struct.unpack_from('<I', d, ptab)[0]
    if p == base + off - hdr: return 'ours'
    raise ValueError(f'{name}: pause block changed by another tool')


def main():
    png = '--png' in sys.argv
    check = '--check' in sys.argv
    tl = load_tl()
    s, lines, ov, exe = build_all(tl)
    used = sum(not f for row in s.free for f in row)
    d = tim_load(); px = tim_px(d)
    px[0:STRIP_ROWS * CH, STRIP_U:STRIP_U + 2 * CW] = s.px
    tim_store(d, px)
    comp = lz.compress(bytes(d), name='OVER.TIZ')
    if len(comp) > SECTORS * 2048: raise ValueError(f'OVER.TIZ {len(comp)} bytes > {SECTORS} sectors')
    for k, img in lines.items():
        w = img.shape[1]; x0 = (512 - w) // 2
        box = (112, 400) if k == 'モードメニューに戻る' else (148, 364)
        ink = np.where(img.any(0))[0]
        if x0 + ink[0] < box[0] + 4 or x0 + ink[-1] > box[1] - 4:
            raise ValueError(f'{k}: ink x {x0 + ink[0]}-{x0 + ink[-1]} outside box {box}')
    out = {}
    for name in FILES:
        exe_file = name == EXE_NAME
        src = 'private/disc/' + ('' if exe_file else 'DAT1/') + name
        disc = open(src, 'rb').read()
        bp = 'private/build/' + name
        if exe_file and not os.path.exists(bp):
            raise SystemExit(f'{bp} missing: run hook.py ... build_cock.py first')
        d2 = bytearray(open(bp if os.path.exists(bp) else src, 'rb').read())
        if len(d2) != len(disc): raise ValueError(f'{name}: size {len(d2)} != disc {len(disc)}')
        check_block(d2, disc, name)
        patch_block(d2, name, exe if exe_file else ov)
        out[name] = d2
    if not check:
        open('private/build/OVER.TIZ.lz', 'wb').write(d)
        open('private/build/OVER.TIZ', 'wb').write(comp)
        for name, d2 in out.items(): open('private/build/' + name, 'wb').write(d2)
    if png: previews(s, lines)
    print(f'pause: {used} of {2 * STRIP_ROWS} cells, OVER.TIZ {len(comp)} bytes '
          f'({-(-len(comp) // 2048)} sectors), records ov {sum(map(len, ov))} '
          f'exe {sum(map(len, exe))} bytes' + (' (check only)' if check else ''))


def clut(d, c):
    v = np.array(struct.unpack_from('<16H', d, 0x14 + 32 * c))
    rgb = np.stack([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], -1)
    return rgb.astype(np.float32), v


def draw(bg, img, x, y, d, c, bright=0x80):
    """Draw an index image like the GPU: index 0 / colour 0 transparent."""
    rgb, v = clut(d, c)
    sub = bg[y:y + img.shape[0], x:x + img.shape[1]]
    m = (img > 0) & (v[img] != 0)
    sub[m] = np.minimum(rgb[img][m] * bright / 0x80, 255)


BG = 'private/re/testmj/emu/b/s005.png'   # JP pause in an Exam Practice fight


def screen(lines, d, exe=False, sel=1, player=1, bg=BG):
    """Pause screen mock-up on the real exam-fight frame. The box is the
    measured one: border 240 grey at x 148-364, y 90-150, fill (0,0,64), bar
    (0,0,216) x 160-351, 15 rows from y 112 + 16*(sel-1)."""
    bg = np.asarray(Image.open(bg).convert('RGB'), np.float32).copy()
    x0, x1 = (112, 400) if exe else (148, 364)
    bg[90:151, x0:x1 + 1] = (240, 240, 240)
    bg[91:150, x0 + 1:x1] = (0, 0, 64)
    bg[112 + 16 * (sel - 1):127 + 16 * (sel - 1), x0 + 12:x1 - 12] = (0, 0, 216)
    t = lines['title1' if player == 1 else 'title2']
    draw(bg, t, (512 - t.shape[1]) // 2, 98, d, CLUT_TITLE)
    for i, k in enumerate(('つづける', 'モードメニューに戻る' if exe else '戻る')):
        im = lines[k]
        draw(bg, im, (512 - im.shape[1]) // 2, 114 + 16 * i, d, CLUT_ITEM, 0x80 if sel == i + 1 else 0x40)
    return bg


def previews(s, lines, out='private/re/pause'):
    """Real JP frame on top, then the overlay box (1P, item 1) and the exe
    box (2P, item 2)."""
    os.makedirs(out, exist_ok=True)
    d = tim_load()
    jp = np.asarray(Image.open(BG).convert('RGB'), np.float32)[84:156, 100:412]
    a = screen(lines, d)[84:156, 100:412]
    b = screen(lines, d, exe=True, sel=2, player=2)[84:156, 100:412]
    Image.fromarray(np.vstack([jp, a, b]).astype(np.uint8)).save(out + '/preview.png')


if __name__ == '__main__':
    main()
