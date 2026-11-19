# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English mail text for the PocketStation app (DAT2/EMTBL.BIZ).

EMTBL.BIZ is an LZ file that decodes to a PSX card save file holding the
PocketStation program; KOUBAI2 (Extras > PocketStation > Download) loads
it whole to 0x800d4000 and writes it to the card. The app maps at
0x02000000 (file offset = address - 0x2000000). RE: private/re/pocket/
findings.org, sections "EMTBL app text" and "EMTBL build".

Text: tl/EMTBL.txt (## EMTBL sNNNN, JP:, EN:), 405 mail lines, index =
character * 20 + mail, 400-404 special lines. The app shows each line as a
one-line marquee under the sender's portrait (routine 0x2002454).

What this script changes:
- Font: 8x7 cells at 0xf139 + 7 * code (one byte per row, bit 0 = left,
  clear bit = ink). Cells 0x20..0x7e are redrawn as ASCII (5x7 glyphs, a-z
  drawn as A-Z), so the strings are plain ASCII. {heart} is byte 0x2b, the
  JP heart cell, redrawn 5 px wide. Cells 0x01..0x0a (digits) and 0xd0,
  0xd1 (+, -) stay: the affection counter draws them through sprite set 10
  (BCD digits only). The other old cells are blanked.
- Advance 8 -> 6 px: 0x24ae movs r0,#8 -> #6 (advance), 0x248c movs r1,#8
  -> #6 (scroll wrap), 0x2504 movs r5,#5 -> #6 (window of 7 bytes; the
  extra byte is RAM 0x2c7 = [r7+0x6f], otherwise unused).
- Strings: new u16 offset list at 0xf700 (relative to 0xf700), strings
  packed into 0xfa2c..0x11eab (the JP string area) and then the file tail
  from 0x1344c.
- Size: the decoded file grows to 11 blocks (0x16000, zero padded). Header
  byte 3 = 11. KOUBAI2.BIN in private/build gets the two download sizes
  0x14000 -> 0x16000 (ori at file 0x9fdc and 0xa5e8). The "10+ free
  blocks" note (EXTRAS s091) is in tl/EXTRAS.txt and says 11.
- Frames: the LCD sprite frames with Japanese (title, mode screens,
  dialogs, names, minigame labels, kernel icon) are redrawn by
  tools/emtbl_frames.py, listed in MODULES; it gets patch(d) on the
  decoded file before compression and png(outdir) for previews.

Input: private/build/EMTBL.BIZ.lz (decoded, written by build_cardtitle.py
with the EN card title); everything outside the title field is taken from
the JP file private/unpacked/EMTBL.BIZ.lz, so a rerun gives the same
output. Run after build_cardtitle.py and build_extras.py.
Writes private/build/EMTBL.BIZ (+ .lz) and patches private/build/KOUBAI2.BIN.
Usage: build_emtbl.py [--check] [--png]
  --check  build and report, write nothing to private/build
  --png    previews in private/re/pocket/emtbl/ (font sheet, marquees)
"""
import sys, os, re, struct, importlib
sys.path.insert(0, os.path.dirname(__file__))
import lz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JP_SRC = os.path.join(ROOT, 'private/unpacked/EMTBL.BIZ.lz')
CT_SRC = os.path.join(ROOT, 'private/build/EMTBL.BIZ.lz')   # build_cardtitle
DISC = os.path.join(ROOT, 'private/disc/DAT2/EMTBL.BIZ')
OUT = os.path.join(ROOT, 'private/build/EMTBL.BIZ')
K2 = os.path.join(ROOT, 'private/build/KOUBAI2.BIN')
TL = os.path.join(ROOT, 'tl/EMTBL.txt')
TL_EXTRAS = os.path.join(ROOT, 'tl/EXTRAS.txt')
PNG_DIR = os.path.join(ROOT, 'private/re/pocket/emtbl')

MODULES = ['emtbl_frames']   # LCD frames with Japanese (tools/emtbl_frames.py)

JP_SIZE = 0x1344c
BLOCKS = 11
SIZE = BLOCKS * 0x2000
FONT = 0xf139           # glyph = FONT + 7 * code
TAB = 0xf700            # u16 offsets relative to TAB
N = 405
AREAS = [(0xfa2c, 0x11eac), (JP_SIZE, SIZE)]   # string space, [lo, hi)
HEART = 0x2b
# Cells no string uses now: the old A-U (0x0b..0x1f), the kana
# (0x7f..0xcf) and 0xd2, blanked
# (0xff) so the file compresses better. Debug-only 8x8/4x4 ASCII font,
# read only by two uncalled hex printers (0x200069e, 0x200072c): zeroed.
BLANK = list(range(0x0b, 0x20)) + list(range(0x7f, 0xd0)) + [0xd2]
DEBUG_FONT = (0x13228, 0x13426)
ADV = 6

# Thumb patches: offset -> (JP halfword, EN halfword)
CODE = {0x24ae: (0x2008, 0x2000 | ADV),       # movs r0,#8: advance
        0x248c: (0x2108, 0x2100 | ADV),       # movs r1,#8: wrap at -8
        0x2504: (0x2505, 0x2506)}             # movs r5,#5: 6 -> 7 bytes
# KOUBAI2 download sizes: ori a3,a3,0x4000 after lui a3,1
K2_SITES = [0x9fdc, 0xa5e8]
K2_JP, K2_EN = 0x34e74000, 0x34e70000 | (SIZE - 0x10000)

# 5x7 glyphs, '#' = ink, columns 0-4 (left to right), rows 0-6.
G = {
 ' ': '..... ..... ..... ..... ..... ..... .....',
 '!': '..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..',
 '"': '.#.#. .#.#. ..... ..... ..... ..... .....',
 '%': '##... ##..# ...#. ..#.. .#... #..## ...##',
 '&': '.##.. #..#. #.#.. .#... #.#.# #..#. .##.#',
 "'": '..#.. ..#.. .#... ..... ..... ..... .....',
 '(': '...#. ..#.. .#... .#... .#... ..#.. ...#.',
 ')': '.#... ..#.. ...#. ...#. ...#. ..#.. .#...',
 '*': '..... ..#.. #.#.# .###. #.#.# ..#.. .....',
 ',': '..... ..... ..... ..... .##.. ..#.. .#...',
 '-': '..... ..... ..... .###. ..... ..... .....',
 '.': '..... ..... ..... ..... ..... .##.. .##..',
 '/': '..... ....# ...#. ..#.. .#... #.... .....',
 '0': '.###. #...# #..## #.#.# ##..# #...# .###.',
 '1': '..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.',
 '2': '.###. #...# ....# ...#. ..#.. .#... #####',
 '3': '##### ...#. ..#.. ...#. ....# #...# .###.',
 '4': '...#. ..##. .#.#. #..#. ##### ...#. ...#.',
 '5': '##### #.... ####. ....# ....# #...# .###.',
 '6': '..##. .#... #.... ####. #...# #...# .###.',
 '7': '##### ....# ...#. ..#.. .#... .#... .#...',
 '8': '.###. #...# #...# .###. #...# #...# .###.',
 '9': '.###. #...# #...# .#### ....# ...#. .##..',
 ':': '..... .##.. .##.. ..... .##.. .##.. .....',
 '=': '..... ..... ##### ..... ##### ..... .....',
 '?': '.###. #...# ....# ...#. ..#.. ..... ..#..',
 'A': '.###. #...# #...# ##### #...# #...# #...#',
 'B': '####. #...# #...# ####. #...# #...# ####.',
 'C': '.###. #...# #.... #.... #.... #...# .###.',
 'D': '###.. #..#. #...# #...# #...# #..#. ###..',
 'E': '##### #.... #.... ####. #.... #.... #####',
 'F': '##### #.... #.... ####. #.... #.... #....',
 'G': '.###. #...# #.... #.### #...# #...# .####',
 'H': '#...# #...# #...# ##### #...# #...# #...#',
 'I': '.###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.',
 'J': '..### ...#. ...#. ...#. ...#. #..#. .##..',
 'K': '#...# #..#. #.#.. ##... #.#.. #..#. #...#',
 'L': '#.... #.... #.... #.... #.... #.... #####',
 'M': '#...# ##.## #.#.# #.#.# #...# #...# #...#',
 'N': '#...# #...# ##..# #.#.# #..## #...# #...#',
 'O': '.###. #...# #...# #...# #...# #...# .###.',
 'P': '####. #...# #...# ####. #.... #.... #....',
 'Q': '.###. #...# #...# #...# #.#.# #..#. .##.#',
 'R': '####. #...# #...# ####. #.#.. #..#. #...#',
 'S': '.#### #.... #.... .###. ....# ....# ####.',
 'T': '##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..',
 'U': '#...# #...# #...# #...# #...# #...# .###.',
 'V': '#...# #...# #...# #...# #...# .#.#. ..#..',
 'W': '#...# #...# #...# #.#.# #.#.# #.#.# .#.#.',
 'X': '#...# #...# .#.#. ..#.. .#.#. #...# #...#',
 'Y': '#...# #...# #...# .#.#. ..#.. ..#.. ..#..',
 'Z': '##### ....# ...#. ..#.. .#... #.... #####',
 '+': '..... .#.#. ##### ##### .###. ..#.. .....',   # cell 0x2b = heart
}


def glyph_bytes(rows):
    """'#'/'.' rows -> 7 cell bytes (bit 0 = left, clear bit = ink)."""
    out = []
    for r in rows.split():
        b = 0xff
        for x, ch in enumerate(r):
            if ch == '#': b &= ~(1 << x)
        out.append(b)
    assert len(out) == 7
    return bytes(out)


def font_cells():
    """code -> 7 bytes for 0x20..0x7e (undefined ASCII cells blank)."""
    cells = {}
    for c in range(0x20, 0x7f):
        ch = chr(c).upper()
        cells[c] = glyph_bytes(G.get(ch, G[' ']))
    return cells


def read_tl():
    t = open(TL, encoding='utf-8').read()
    ents = re.findall(r'^## EMTBL s(\d{4})[^\n]*\nJP: (.*)\nEN: (.*)$', t, re.M)
    if len(ents) != N or [int(k) for k, _, _ in ents] != list(range(N)):
        raise SystemExit(f'tl/EMTBL.txt: expected entries s0000..s{N - 1:04d}')
    return [(j, e) for _, j, e in ents]


OK = set(" .,'!?%-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def encode(en, k, problems):
    s = en.replace('{heart}', '\x00')
    bad = sorted(set(c for c in s if c != '\x00' and c not in OK))
    if bad: problems.append(f's{k:04d}: characters not allowed {bad}')
    if not s.strip(): problems.append(f's{k:04d}: empty')
    b = bytes(HEART if c == '\x00' else ord(c) for c in s)
    if len(b) > 240: problems.append(f's{k:04d}: {len(b)} B > 240 (pos is a byte)')
    return b


def _cost(buf):
    t = bytes(buf[-6000:])
    if len(t) % 2: t += b'\0'
    return len(lz.compress(t, burst=0x1800))


def pack(strings, problems):
    """Place NUL-ended strings into AREAS in order; returns {k: offset}.
    The LZ matches whole halfwords, so a repeated word only matches when
    both copies have the same byte parity. Each string starts at the
    parity (an extra NUL before it if needed) that compresses better with
    the last 6 KB before it."""
    offs, ai, buf = {}, 0, bytearray()
    for k, b in enumerate(strings):
        best = None
        for pad in (0, 1):
            t = buf + bytes(pad) + b + b'\0'
            if AREAS[ai][0] + len(t) > AREAS[ai][1]: continue
            c = _cost(t)
            if best is None or c < best[0]: best = (c, pad)
        if best is None:
            ai += 1
            if ai == len(AREAS):
                problems.append(f'strings do not fit (s{k:04d} onward)')
                return offs, None
            buf = bytearray()
            best = (0, 0)
        buf += bytes(best[1])
        offs[k] = AREAS[ai][0] + len(buf)
        buf += b + b'\0'
    used = sum(len(b) + 1 for b in strings)
    room = sum(h - l for l, h in AREAS)
    return offs, (used, room, ai, AREAS[ai][0] + len(buf))


def build():
    problems = []
    jp = open(JP_SRC, 'rb').read()
    assert len(jp) == JP_SIZE and jp[:2] == b'SC' and jp[3] == 10, 'JP EMTBL'
    ct = open(CT_SRC, 'rb').read() if os.path.exists(CT_SRC) else None
    if ct is None or ct[4:0x44] == jp[4:0x44]:
        raise SystemExit('run build_cardtitle.py first (no EN title in '
                         'private/build/EMTBL.BIZ.lz)')
    d = bytearray(jp) + bytes(SIZE - JP_SIZE)
    d[4:0x44] = ct[4:0x44]
    d[3] = BLOCKS
    for off, (old, new) in CODE.items():
        assert struct.unpack_from('<H', d, off)[0] == old, hex(off)
        struct.pack_into('<H', d, off, new)
    for c, cell in font_cells().items():
        d[FONT + 7 * c:FONT + 7 * c + 7] = cell
    for c in BLANK:
        d[FONT + 7 * c:FONT + 7 * c + 7] = b'\xff' * 7
    d[DEBUG_FONT[0]:DEBUG_FONT[1]] = bytes(DEBUG_FONT[1] - DEBUG_FONT[0])
    tl = read_tl()
    strings = [encode(e, k, problems) for k, (j, e) in enumerate(tl)]
    for lo, hi in AREAS:
        d[lo:hi] = bytes(hi - lo)
    offs, stat = pack(strings, problems)
    if stat:
        for k, b in enumerate(strings):
            o = offs[k]
            d[o:o + len(b)] = b
            struct.pack_into('<H', d, TAB + 2 * k, o - TAB)
        struct.pack_into('<H', d, TAB + 2 * N, 0)
    for name in MODULES:
        importlib.import_module(name).patch(d)
    assert len(d) == SIZE
    c = lz.compress(bytes(d), name='EMTBL.BIZ')
    assert lz.decompress(c)[0] == bytes(d)
    alloc = -(-os.path.getsize(DISC) // 0x800) * 0x800
    if len(c) > alloc: problems.append(f'LZ {len(c)} B > disc slot {alloc}')
    return d, c, strings, stat, alloc, problems


def patch_k2(write):
    k = bytearray(open(K2, 'rb').read())
    for off in K2_SITES:
        w = struct.unpack_from('<I', k, off)[0]
        hi = struct.unpack_from('<I', k, off - 0x2c if off == 0x9fdc else off - 0x24)[0]
        if w not in (K2_JP, K2_EN) or hi != 0x3c070001:
            raise SystemExit(f'KOUBAI2 0x{off:x}: unexpected {w:08x} / {hi:08x}')
        struct.pack_into('<I', k, off, K2_EN)
    if write: open(K2, 'wb').write(k)
    return [hex(o) for o in K2_SITES]


def check_extras():
    t = open(TL_EXTRAS, encoding='utf-8').read()
    m = re.search(r'^JP: ダウンロードには10ブロック.*\nEN: (.*)$', t, re.M)
    if not m or f'{BLOCKS}+' not in m.group(1):
        return [f'tl/EXTRAS.txt s091 does not say {BLOCKS}+ blocks']
    return []


# ---------------------------------------------------------------- previews

def lcd_window(d, s, adv=ADV, width=32):
    """Marquee rows (7 x width) after s pixels of scroll, as 0/1 ink."""
    # simulate 0x2002454: x starts at 32; x -= 1; x <= -adv -> x = 0, pos++
    x, pos = 32, 0
    for _ in range(s):
        x -= 1
        if x <= -adv: x, pos = 0, pos + 1
    img = [[0] * width for _ in range(7)]
    for i in range(adv + 1 if adv < 8 else 6):
        c = d[pos + i] if pos + i < len(d) else 0
        if c == 0: break
        for r in range(7):
            b = FONT_D[FONT + 7 * c + r]
            for bx in range(8):
                px = x + adv * i + bx
                if 0 <= px < width:
                    img[r][px] = 0 if (b >> bx) & 1 else 1
    return img, pos


FONT_D = None


def png(d, strings):
    global FONT_D
    from PIL import Image, ImageDraw
    FONT_D = d
    os.makedirs(PNG_DIR, exist_ok=True)
    INK, PAPER, BG = (40, 48, 40), (176, 190, 160), (255, 255, 255)
    # font sheet: cells 0x20..0x5f at 6x, plus the kept cells
    S = 6
    codes = list(range(0x20, 0x60)) + list(range(1, 11)) + [0xd0, 0xd1]
    cols = 16
    sh = Image.new('RGB', (cols * (8 * S + 4) + 4, ((len(codes) + cols - 1) // cols) * (7 * S + 16) + 4), BG)
    dr = ImageDraw.Draw(sh)
    for n, c in enumerate(codes):
        x0, y0 = 4 + (n % cols) * (8 * S + 4), 4 + (n // cols) * (7 * S + 16)
        dr.text((x0, y0), f'{c:02x}', fill=(200, 0, 0))
        for r in range(7):
            b = d[FONT + 7 * c + r]
            for bx in range(8):
                col = PAPER if bx < ADV else (220, 220, 220)
                if not (b >> bx) & 1: col = INK
                dr.rectangle([x0 + bx * S, y0 + 12 + r * S, x0 + bx * S + S - 1, y0 + 12 + r * S + S - 1], fill=col)
    sh.save(os.path.join(PNG_DIR, 'font.png'))
    # marquee strips: portrait (set 4) + scrolling line, frames every 10 px
    pick = [0, 41, 85, 133, 207, 260, 399, 403, 404]
    port = sprite_set(d, 4)
    Z, step, nf = 3, 12, 10
    fw = 32 * Z + 4
    im = Image.new('RGB', (nf * fw + 4, len(pick) * (32 * Z + 18) + 4), BG)
    dr = ImageDraw.Draw(im)
    for row, k in enumerate(pick):
        y0 = 4 + row * (32 * Z + 18)
        txt = strings[k].replace(bytes([HEART]), b'<3').decode()
        dr.text((4, y0), f's{k:04d} {txt}'[:150], fill=(0, 0, 0))
        pw = port[(k // 20 * 12) % len(port)] if k < 400 else port[241]
        for f in range(nf):
            win, _ = lcd_window(strings[k] + b'\0' * 8, f * step)
            x0 = 4 + f * fw
            for yy in range(32):
                for xx in range(32):
                    if yy < 22: ink = pw[yy][xx] if yy < len(pw) and xx < len(pw[yy]) else 0
                    elif 24 <= yy < 31: ink = win[yy - 24][xx]
                    else: ink = 0
                    dr.rectangle([x0 + xx * Z, y0 + 14 + yy * Z, x0 + xx * Z + Z - 1, y0 + 14 + yy * Z + Z - 1],
                                 fill=INK if ink else PAPER)
    im.save(os.path.join(PNG_DIR, 'marquee.png'))
    # whole lines at 2x as one band each (every mail of the first two
    # characters), for a quick read of the spacing
    Z = 2
    lines = list(range(0, 40))
    wmax = max(len(strings[k]) for k in lines) * ADV + 8
    im = Image.new('RGB', (wmax * Z + 8, len(lines) * (9 * Z) + 8), PAPER)
    dr = ImageDraw.Draw(im)
    for n, k in enumerate(lines):
        for i, c in enumerate(strings[k]):
            for r in range(7):
                b = d[FONT + 7 * c + r]
                for bx in range(ADV):
                    if not (b >> bx) & 1:
                        x, y = 4 + (i * ADV + bx) * Z, 4 + (n * 9 + r) * Z
                        dr.rectangle([x, y, x + Z - 1, y + Z - 1], fill=INK)
    im.save(os.path.join(PNG_DIR, 'lines.png'))
    for name in MODULES:
        m = importlib.import_module(name)
        if hasattr(m, 'png'): m.png(PNG_DIR)
    print('previews in', PNG_DIR)


def sprite_set(d, si):
    b = struct.unpack_from('<I', d, 0x68d8 + 4 * si)[0] - 0x2000000
    first = struct.unpack_from('<H', d, b)[0]
    out = []
    for a in range(b, b + first, 4):
        o, w, h = struct.unpack_from('<HBB', d, a)
        w += 1
        out.append([[(d[b + o + r * w + x // 8] >> (x % 8)) & 1 for x in range(8 * w)] for r in range(h)])
    return out


def main():
    check = '--check' in sys.argv
    d, c, strings, stat, alloc, problems = build()
    problems += check_extras()
    if stat:
        used, room, ai, end = stat
        print(f'strings {used} B of {room} (ends at 0x{end:x} in area {ai})')
    print(f'EMTBL.BIZ: decoded 0x{len(d):x} ({BLOCKS} blocks), LZ {len(c)} B '
          f'(JP {os.path.getsize(DISC)}, slot {alloc})')
    if '--png' in sys.argv: png(d, strings)
    for p in problems: print('PROBLEM', p)
    if problems: sys.exit(1)
    sites = patch_k2(not check)
    if check:
        print('check only, nothing written'); return
    open(OUT, 'wb').write(c)
    open(OUT + '.lz', 'wb').write(d)
    print(f'{OUT} (+ .lz); KOUBAI2.BIN download size 0x{SIZE:x} at {sites}')


if __name__ == '__main__':
    main()
