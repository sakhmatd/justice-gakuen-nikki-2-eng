# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English labels in the PocketStation app's 1bpp LCD frames (EMTBL.BIZ).

A MODULES entry of build_emtbl.py: patch(d) redraws the frames in the
decoded file, png(outdir) writes the JP | EN review sheet.

Frames are sprite-set entries (pointer table 0x68d8; entry = u16 offset
from the set start, u8 bytes per row - 1, u8 rows), rows LSB-left, set bit
= ink. The kernel icon is a plain 32x32 frame at 0x280. RE: private/re/
pocket/findings.org, "LCD frames with Japanese" and "EMTBL frames".

Each frame gets a list of ops on a 0/1 bitmap: the JP text area is cleared
(art outside it is kept), then EN text is drawn in one of the styles the JP
frames use:
  ink      ink letters
  paper    paper letters (inside an ink band)
  outline  paper letters with a 1 px ink outline (the hollow JP style)
  bold     ink letters doubled 1 px to the right
"""
import struct, os

# ------------------------------------------------------------------ fonts
# Proportional; '#' = ink, rows separated by spaces.
F5 = {  # 5 rows, mostly 3 wide
 'A': '.#. #.# ### #.# #.#', 'B': '##. #.# ##. #.# ##.',
 'C': '.## #.. #.. #.. .##', 'D': '##. #.# #.# #.# ##.',
 'E': '### #.. ##. #.. ###', 'F': '### #.. ##. #.. #..',
 'G': '.## #.. #.# #.# .##', 'H': '#.# #.# ### #.# #.#',
 'I': '### .#. .#. .#. ###', 'J': '..# ..# ..# #.# .#.',
 'K': '#.# #.# ##. #.# #.#', 'L': '#.. #.. #.. #.. ###',
 'M': '#...# ##.## #.#.# #...# #...#', 'N': '#..# ##.# #.## #..# #..#',
 'O': '.#. #.# #.# #.# .#.', 'P': '##. #.# ##. #.. #..',
 'R': '##. #.# ##. #.# #.#', 'S': '.## #.. .#. ..# ##.',
 'T': '### .#. .#. .#. .#.', 'U': '#.# #.# #.# #.# ###',
 'V': '#.# #.# #.# #.# .#.', 'W': '#...# #...# #.#.# ##.## #...#',
 'Y': '#.# #.# .#. .#. .#.', '!': '# # # . #', '?': '##. ..# .#. ... .#.',
 '.': '. . . . #', '-': '... ... ### ... ...', ' ': '. . . . .',
 '#': '.#.#. ##### .#.#. ##### .#.#.', '1': '.#. ##. .#. .#. ###',
}
F7 = {  # 7 rows, mostly 4 wide
 'A': '.##. #..# #..# #### #..# #..# #..#',
 'B': '###. #..# #..# ###. #..# #..# ###.',
 'C': '.##. #..# #... #... #... #..# .##.',
 'D': '###. #..# #..# #..# #..# #..# ###.',
 'E': '### #.. #.. ### #.. #.. ###',
 'F': '### #.. #.. ### #.. #.. #..',
 'G': '.##. #..# #... #.## #..# #..# .###',
 'H': '#..# #..# #..# #### #..# #..# #..#',
 'I': '### .#. .#. .#. .#. .#. ###',
 'J': '...# ...# ...# ...# ...# #..# .##.',
 'K': '#..# #..# #.#. ##.. #.#. #..# #..#',
 'L': '#.. #.. #.. #.. #.. #.. ###',
 'M': '#...# ##.## #.#.# #.#.# #...# #...# #...#',
 'N': '#..# #..# ##.# #.## #..# #..# #..#',
 'O': '.##. #..# #..# #..# #..# #..# .##.',
 'P': '###. #..# #..# ###. #... #... #...',
 'R': '###. #..# #..# ###. #.#. #..# #..#',
 'S': '.### #... #... .##. ...# ...# ###.',
 'T': '### .#. .#. .#. .#. .#. .#.',
 'U': '#..# #..# #..# #..# #..# #..# .##.',
 'V': '#...# #...# #...# .#.#. .#.#. ..#.. ..#..',
 'W': '#...# #...# #...# #.#.# #.#.# ##.## #...#',
 'Y': '#.# #.# #.# .#. .#. .#. .#.',
 '!': '# # # # # . #', '?': '.##. #..# ...# ..#. .#.. .... .#..',
 '.': '. . . . . . #', ' ': '. . . . . . .',
}
F6 = {  # 6 rows, mostly 4 wide (names, AGAIN?)
 'A': '.##. #..# #..# #### #..# #..#', 'B': '###. #..# ###. #..# #..# ###.',
 'C': '.##. #..# #... #... #..# .##.', 'D': '###. #..# #..# #..# #..# ###.',
 'E': '### #.. ### #.. #.. ###', 'F': '### #.. ### #.. #.. #..',
 'G': '.### #... #.## #..# #..# .###', 'H': '#..# #..# #### #..# #..# #..#',
 'I': '### .#. .#. .#. .#. ###', 'K': '#..# #.#. ##.. #.#. #..# #..#',
 'M': '#...# ##.## #.#.# #...# #...# #...#',
 'N': '#..# ##.# #.## #..# #..# #..#', 'O': '.##. #..# #..# #..# #..# .##.',
 'R': '###. #..# #..# ###. #.#. #..#', 'S': '.### #... .##. ...# ...# ###.',
 'T': '### .#. .#. .#. .#. .#.', 'U': '#..# #..# #..# #..# #..# .##.',
 'Y': '#.# #.# #.# .#. .#. .#.', '?': '.##. #..# ..#. .#.. .... .#..',
 '!': '# # # # . #', ' ': '. . . . . .',
}
FB = {  # 10 rows, 2 px strokes
 'G': '.#####. ####### ##...## ##..... ##..... ##.#### ##...## ##...## '
      '####### .#####.',
 'E': '###### ###### ##.... ##.... #####. #####. ##.... ##.... ###### ######',
 'T': '###### ###### ..##.. ..##.. ..##.. ..##.. ..##.. ..##.. ..##.. ..##..',
 '!': '## ## ## ## ## ## .. .. ## ##',
}


def glyph(font, ch):
    rows = font[ch].split()
    return [[1 if c == '#' else 0 for c in r] for r in rows]


def text_mask(font, s, gap=1, space=2):
    """Bitmap (rows of 0/1) of s; letters gap px apart, ' ' = space px."""
    h = len(next(iter(font.values())).split())
    out = [[] for _ in range(h)]
    for i, ch in enumerate(s):
        if ch == ' ':
            for r in out: r.extend([0] * space)
            continue
        g = glyph(font, ch)
        if i and s[i - 1] != ' ':
            for r in out: r.extend([0] * gap)
        for r in range(h): out[r].extend(g[r])
    return out


# ------------------------------------------------------------------ bitmap
def read_frame(d, off, w, h):
    return [[(d[off + r * w + x // 8] >> (x % 8)) & 1 for x in range(8 * w)]
            for r in range(h)]


def write_frame(d, off, w, h, bm):
    for r in range(h):
        for b in range(w):
            v = 0
            for k in range(8):
                if bm[r][8 * b + k]: v |= 1 << k
            d[off + r * w + b] = v


def sprite(d, si, i):
    b = struct.unpack_from('<I', d, 0x68d8 + 4 * si)[0] - 0x2000000
    o, w, h = struct.unpack_from('<HBB', d, b + 4 * i)
    return b + o, w + 1, h


def put(bm, m, x0, y0, style):
    """Draw mask m with its top-left at (x0, y0) in the given style."""
    H, W = len(bm), len(bm[0])
    h, w = len(m), len(m[0])
    if style == 'bold':
        m = [[r[x] or (x and r[x - 1]) for x in range(w)] + [r[-1]] for r in m]
        w += 1
    if style == 'outline':
        for y in range(-1, h + 1):
            for x in range(-1, w + 1):
                near = any(0 <= y + dy < h and 0 <= x + dx < w and m[y + dy][x + dx]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                if near and 0 <= y0 + y < H and 0 <= x0 + x < W:
                    bm[y0 + y][x0 + x] = 1
    for y in range(h):
        for x in range(w):
            if m[y][x] and 0 <= y0 + y < H and 0 <= x0 + x < W:
                bm[y0 + y][x0 + x] = 0 if style in ('paper', 'outline') else 1


def run(bm, ops):
    for op in ops:
        k = op[0]
        if k == 'fill':                       # fill x0 y0 x1 y1 (incl.) v
            _, x0, y0, x1, y1, v = op
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1): bm[y][x] = v
        elif k == 'text':                     # text font s cx y style [gap]
            _, font, s, cx, y, style = op[:6]
            m = text_mask(font, s, *op[6:7])
            w = len(m[0])
            # cx: centre (x + 0.5 = between columns x and x + 1), ('l', x)
            # left edge or ('r', x) right edge
            if isinstance(cx, tuple):
                x0 = cx[1] if cx[0] == 'l' else cx[1] - w + 1
            else:
                x0 = int(cx + 0.5 - w / 2 + 0.25)
            put(bm, m, x0, y, style)
        elif k == 'vtext':                    # vtext font s cx y step style
            _, font, s, cx, y, step, style = op
            for i, ch in enumerate(s):
                m = text_mask(font, ch)
                put(bm, m, int(round(cx - len(m[0]) / 2)), y + i * step, style)
        elif k == 'thin':                     # thin y0 y1 x0 x1: 2:15 blink
            _, y0, y1, x0, x1 = op
            for y in range(y0, y1 + 1):
                if y % 2 == 0:
                    for x in range(x0, x1 + 1):
                        if x % 2 == 0: bm[y][x] = 0
        else:
            raise ValueError(op)


# ------------------------------------------------------------------ frames
NAMES = ['BATSU', 'HINATA', 'KYOSUKE', 'SHOMA', 'NATSU', 'ROBERTO', 'ROY',
         'TIFFANY', 'BOMAN', 'EDGE', 'AKIRA', 'GAN', 'HIDEO', 'KYOKO', 'HYO',
         'SAKURA', 'DAIGO', 'HAYATO', 'NAGARE', 'RAN']
# Feet that reach below row 23 (kept): name index -> (x0, x1, last row)
FEET = {8: (10, 25, 25), 9: (10, 24, 25), 11: (5, 26, 25), 14: (21, 24, 25),
        16: (8, 23, 24)}

FRAMES = {}
# names: row 23 is always art; the JP name fills rows 24..31
for i, n in enumerate(NAMES):
    ops = [('fill', 0, 24, 31, 31, 0)]
    FRAMES[(1, i)] = ops + [('text', F6, n, 15.5, 26, 'ink')]

OUT5 = 'outline'
FRAMES.update({
 # title logo: hollow (2:0) and solid (2:12) frames of one animation; the
 # arrow right of POCKET is kept
 (2, 0): [('fill', 0, 0, 23, 6, 0), ('fill', 0, 25, 31, 31, 0),
          ('text', F5, 'POCKET', 12.5, 1, OUT5),
          ('text', F5, 'JUSTICE', 15.5, 26, OUT5)],
 (2, 12): [('fill', 0, 0, 23, 6, 0), ('fill', 0, 25, 31, 31, 0),
           ('text', F5, 'POCKET', 12.5, 1, 'ink'),
           ('text', F5, 'JUSTICE', 15.5, 26, 'ink')],
 # mode screens; Dr.K box and art kept
 (2, 5): [('fill', 0, 0, 16, 6, 0), ('text', F5, 'BE A', 8, 1, 'ink')],
 (2, 6): [('fill', 1, 1, 6, 30, 0), ('vtext', F5, 'SNAP!', 3.5, 2, 6, 'ink'),
          ('fill', 8, 0, 31, 12, 0), ('text', F5, 'PHOTO', 20, 1, 'ink'),
          ('text', F5, 'OP!', 20, 7, 'ink')],
 (2, 7): [('fill', 0, 0, 18, 6, 0), ('text', F5, 'DIVE', 8.5, 1, 'ink'),
          ('fill', 0, 8, 12, 14, 0), ('text', F5, 'ACE', 6, 9, 'ink')],
 (2, 8): [('fill', 4, 0, 27, 6, 0), ('text', F5, 'WALLET', 15.5, 1, OUT5)],
 (2, 9): [('fill', 4, 0, 27, 6, 0), ('text', F5, 'PHONE', 15.5, 1, OUT5)],
 (2, 10): [('fill', 4, 0, 28, 6, 0), ('text', F5, 'WALK', 15.5, 1, OUT5)],
 (2, 13): [('fill', 4, 0, 28, 6, 0), ('text', F5, 'WALK', 15.5, 1, OUT5)],
 (2, 11): [('fill', 0, 0, 27, 18, 0), ('text', F5, 'BATTERY', 15.5, 2, OUT5),
           ('text', F7, 'LOW!', 15.5, 10, OUT5)],
 (2, 14): [('fill', 0, 0, 31, 11, 0), ('text', FB, 'GET!', 15.5, 1, 'ink', 2)],
 (2, 15): [('fill', 0, 0, 31, 11, 0), ('text', FB, 'GET!', 15.5, 1, 'ink', 2),
           ('thin', 1, 11, 0, 31)],
 # dialogs (YES/NO is a separate sprite below)
 (3, 5): [('fill', 0, 0, 31, 7, 1), ('text', F6, 'AGAIN?', 15.5, 1, 'paper')],
 (3, 6): [('fill', 0, 0, 31, 22, 0), ('text', F7, 'WRITE', 15.5, 2, OUT5),
          ('text', F7, 'BACK?', 15.5, 13, OUT5)],
 (3, 7): [('fill', 0, 0, 31, 22, 0), ('text', F7, 'REPLY', 15.5, 2, OUT5),
          ('text', F7, 'WITH?', 15.5, 13, OUT5)],
 # all clear: hollow (boxed) and solid frames; hearts kept
 (4, 241): [('fill', 6, 6, 24, 13, 0), ('text', F5, 'CLEAR!', 15.5, 8, OUT5)],
 (4, 242): [('fill', 6, 7, 24, 12, 0), ('text', F5, 'CLEAR!', 15.5, 8, 'ink')],
 # minigame counters: the code draws the number in the middle
 (7, 1): [('fill', 0, 0, 13, 9, 0), ('fill', 18, 19, 31, 31, 0),
          ('text', F5, 'LEFT', ('l', 3), 3, OUT5),
          ('text', F5, 'BALLS', ('r', 29), 25, OUT5)],
 (9, 0): [('fill', 0, 0, 13, 9, 0), ('fill', 18, 19, 31, 31, 0),
          ('text', F5, 'LEFT', ('l', 3), 3, OUT5),
          ('text', F5, 'DIVES', ('r', 29), 25, OUT5)],
})
# minigame results (Shoma 7, Ran 8, Nagare 9): label in rows 26..31
for si, a, b in ((7, 29, 30), (8, 3, 4), (9, 22, 23)):
    for i, t in ((a, 'GREAT!'), (b, 'TOO BAD!')):
        FRAMES[(si, i)] = [('fill', 0, 26, 31, 31, 0),
                           ('text', F5, t, 15.5, 26, OUT5)]
FRAMES[('icon', 0)] = [('fill', 0, 0, 31, 7, 0),
                       ('text', F5, 'POKEJUS', 15.5, 1, OUT5)]


def frames_list(d):
    out = []
    for (si, i), ops in FRAMES.items():
        if si == 'icon': off, w, h = 0x280, 4, 32
        else: off, w, h = sprite(d, si, i)
        out.append((f'{si}:{i}', off, w, h, ops))
    return out


def patch(d):
    for lab, off, w, h, ops in frames_list(d):
        bm = read_frame(d, off, w, h)
        run(bm, ops)
        write_frame(d, off, w, h, bm)


# ------------------------------------------------------------------ preview
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JP = os.path.join(ROOT, 'private/unpacked/EMTBL.BIZ.lz')
REVIEW = os.path.join(ROOT, 'private/re/pocket/emtbl_frames_review.png')


def png(outdir, path=REVIEW, Z=4, cols=4, only=None):
    from PIL import Image, ImageDraw
    jp = open(JP, 'rb').read()
    en = bytearray(jp)
    patch(en)
    INK, PAPER, BG = (40, 48, 40), (176, 190, 160), (255, 255, 255)
    fl = [f for f in frames_list(jp) if only is None or f[0] in only]
    cw, ch = 2 * (32 * Z) + 3 * Z + 12, 32 * Z + 16
    im = Image.new('RGB', (cols * cw + 4, -(-len(fl) // cols) * ch + 4), BG)
    dr = ImageDraw.Draw(im)
    for n, (lab, off, w, h, ops) in enumerate(fl):
        x0, y0 = 4 + (n % cols) * cw, 4 + (n // cols) * ch
        dr.text((x0, y0), lab, fill=(200, 0, 0))
        for k, src in enumerate((jp, en)):
            bm = read_frame(src, off, w, h)
            xo = x0 + k * (32 * Z + 3 * Z)
            for r in range(h):
                for x in range(8 * w):
                    dr.rectangle([xo + x * Z, y0 + 12 + r * Z, xo + x * Z + Z - 1,
                                  y0 + 12 + r * Z + Z - 1],
                                 fill=INK if bm[r][x] else PAPER)
    im.save(path)
    return path


if __name__ == '__main__':
    import sys
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    print(png(None, *(sys.argv[1:2] or []), only=only))
