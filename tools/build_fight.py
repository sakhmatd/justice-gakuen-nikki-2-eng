# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Arcade fight HUD in EN: DAT2/CEFT1.BIZ (HUD TIM1 at 0x10800, 4bpp
768x256 at VRAM 768,256, CLUT 512x4 at 0,500) and the exe records that cut
it. The diary sparring fights (mode 0) and the arcade modes load it.

Port of the first game's tool (justice-mtl tools/build_fight.py). The text
pieces come from the first game's USA disc (CEFT1 TIM1 of SLUS-00681), as
in the first game's patch; the rest of the TIM stays N2 art. The tool
- clears the N2 JP callouts, banners, labels and fighter names;
- copies the USA pieces in: the tpage 832 block at its USA place (ROUND,
  FINAL, FIGHT, digits 1 2, K.O., FINISH, HIT), the rest
  (TIME OVER, DRAW, banners, FIRST ATTACK, swap prompts, READY!!) packed
  into the freed N2 texels;
- draws the four result grades (lists 14-17, tl/FIGHT.txt win_*) in the
  style of the USA VICTORY piece, and copies the USA round marks BV and 2
  over the JP marks 完全 and 2ﾂｰﾌﾟﾗ;
- draws ROUND digits 3-6 in the USA style (as the first game) if there is
  room, and our fighter names (tl/FIGHT.txt name_*, rotated, Inconsolata
  Extra Condensed Black) in the N2 name palette;
- puts the USA palettes into the N2 palette slots that only the replaced
  JP callouts used (60-64, 67, 68); FIGHT, TIME OVER and DRAW share slot
  64 (colours merged).
Exe changes (file offsets, file = RAM - 0x8000f800):
- 0x7d860-0x7dd13: callout lists 0-17, pointer table 0x7dd14. Lists 7, 8
  keep the disc bytes (the exams draw them from CEFT4).
- 0x7f5e8: fighter-name table, ids 1-26 (id 0 kept; ids 27-30 by
  fightnick.py).
- fightnick.py: 8-letter player nickname (renderer 0x800290cc edits,
  EXPAND at 0x80010bf0, player cells moved to TIM 0..95,200..255) and
  DAT2/COCKEDIT.TIZ glyphs 1-62. The names are packed exactly into the
  JP name cells and the old player cells (0..95, 64..95).
- moved records (MOVED): first attack, swap prompts, banners, FINISH,
  READY!! P1/P2.
- swap prompt uv code 0x4aa6c-0x4aa9c (v, w, h, u stride).
- hit counter label 0x4ee48, 0x4f0b0-0x4f104 (HIT; digits kept).
- win callout x 0x49208 (side 1 +120 -> +184).
--png writes previews to private/re/fighthud/; --check builds and checks
but writes nothing. Notes: private/re/fighthud/findings.org."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz, streamsim, fightnick

EXE = 'private/build/SLPS_021.20'
DISC_EXE = 'private/disc/SLPS_021.20'
USA_EXE = 'private/usa/disc/SLUS_006.81'
SRC = 'private/unpacked/CEFT1.BIZ.lz'
SRC_USA = 'private/usa/unpacked/CEFT1.BIZ.lz'
PNG = 'private/re/fighthud'
BG = 'private/re/examfight/r2/s08920.png'       # sparring fight frame (512x240)
TL = 'tl/FIGHT.txt'
F_DIGIT = 'verdanab.ttf'
F_NAME = 'Inconsolata-ExtraCondensedBlack.ttf'
EB = 0x8000f800                     # RAM - exe file offset
TIM1 = 0x10800
PTAB = 0x8008d514 - EB              # 18 list pointers
LISTS = 0x8008d060 - EB             # list block, ends at PTAB
USA_PTAB = 0x71aa4
NAMES = 0x7f5e8                     # 31 x (u, v, h)
KEEP_LISTS = (7, 8)

# ----------------------------------------------------------- USA pieces
# name: (USA TIM x, y, w, h, USA palette). The rect is the USA record's
# texture rect. 'own' (x, y, w, h) limits the colour remap where a USA
# rect overlaps a neighbour's texels.
PIECES = {
    'round':    (256, 0, 256, 49, 60), 'final': (256, 48, 193, 49, 60),
    'd1':       (256, 144, 49, 49, 60), 'd2': (304, 144, 49, 49, 60),
    'fight':    (256, 96, 193, 49, 64), 'ko': (256, 192, 129, 49, 62),
    'finish':   (384, 192, 128, 13, 66),
    'hit':      (384, 208, 64, 12, 66),
    'timeover': (512, 32, 193, 33, 64), 'draw': (640, 64, 128, 33, 61),
    'first':    (512, 64, 129, 13, 68), 'tardy': (512, 80, 128, 13, 68),
    'burn':     (512, 96, 128, 25, 67), 'team': (640, 96, 128, 25, 67),
    'swap0':    (512, 128, 89, 12, 47), 'swap1': (512, 144, 89, 12, 47),
    'ready':    (176, 184, 81, 12, 2)}
# FIGHT's col 448 is the USA WIN's (not copied).
OWN = {'fight': (256, 96, 192, 49), 'timeover': (512, 32, 192, 33),   # col 704: USA stray line
       'swap0': (512, 128, 88, 12), 'swap1': (512, 144, 88, 12)}   # col 600+: PUSH BUTTON, box
FIXED = ('round', 'final', 'd1', 'd2', 'fight', 'ko', 'finish', 'hit')                      # kept at the USA place
# Palette slot (N2 CLUT index, 16 colours) -> pieces. Slots 60-64, 67, 68
# were used only by the N2 JP callouts and banners (exe records; no other
# GetClut on them). 63 holds the USA VICTORY palette (USA 64) for the
# result grades drawn here (GRADES). 65 (hit digits), 66 (OVER.TIZ pause strip), 47 and 2
# stay N2. USA 47 equals N2 49. READY uses N2 slot 2 (MAX! shares it).
SLOTS = {60: ['round', 'final', 'd1', 'd2'], 61: ['finish', 'hit'], 62: ['ko'],
         64: ['fight', 'timeover', 'draw'],
         67: ['burn', 'team'], 68: ['first', 'tardy'], 49: ['swap0', 'swap1'],
         2: ['ready']}
FREE_SLOTS = (60, 61, 62, 63, 64, 67, 68)

# N2 texels freed (x, y, w, h): JP callouts, banners, labels. tpage 832:
# 第一二三四五, 最終戦, 開始!, 決着 (三四五 are referenced by no list).
# tpage 896: win grades + shadows, 時間切れ, 引き分け, 勝, 愛と友情の..,
# 辛, フィニッシュ, 先制, 発, 完全燃焼/根性 banners, 主役/相棒交代?.
# tpage 768: 準備OK!!. Runtime-written texels (player name cells
# 0-95,64-95; faces 512-607,208-247 and 640-767,208-239) are not free.
FREE = [(256, 0, 256, 232), (256, 232, 160, 16),
        (512, 32, 192, 80), (512, 112, 64, 48), (576, 112, 128, 32), (704, 96, 64, 48),
        (576, 144, 137, 17), (712, 144, 24, 16), (512, 160, 256, 32), (512, 192, 160, 12),
        (160, 96, 81, 12)]

# ------------------------------------------------------------- records
# (N2 offset, USA offset, halfwords) of the records the USA moved.
MOVED = {'first': (0x7d3aa, 0x7142e, 12), 'burn': (0x7e36a, 0x720de, 12),
         'tardy': (0x7e386, 0x720fa, 12), 'team': (0x7e3a2, 0x72116, 12),
         'finish': (0x7e3be, 0x72132, 12)}
READY = (0x7f920, 0x7f936)          # 11 halfwords: tpx tpy clx cly u v w h x y f
SWAP = (0x7e292, 0x7e2f6)           # 12 halfwords; uv set by code
# The JP prompt (80 wide at x 56 / 376) is centred on its swap block
# (faces + icon, x 48-146 and 366-464): centres 96.5 and 416.5.
SWAP_CX = (96.5, 416.5)
SWAP_Y = 170
# Swap prompt code (RAM, disc word): u = stride * byte 0x8009d2ab.
SW_V, SW_W, SW_H = 0x8005a26c, 0x8005a270, 0x8005a280
SW_SLL1, SW_ADD, SW_SLL2 = 0x8005a290, 0x8005a294, 0x8005a29c
# u strides the code can make: (sll1, add/sub, sll2) -> (2^a +- 1) << b.
STRIDES = {96: (1, 'addu', 5), 112: (3, 'subu', 4), 144: (3, 'addu', 4), 160: (2, 'addu', 5)}
# Hit counter (N2 0x8005e648-0x8005e950): (RAM, disc imm); new values are
# set in patch_counter. Digits stay N2.
CTR = {'p2x': (0x8005e648, 0x178), 'tpx': (0x8005e8b0, 0x380), 'clx': (0x8005e8c8, 0x10),
       'cly': (0x8005e8cc, 0x1f6), 'y': (0x8005e8e0, 0x3e), 'w': (0x8005e8e4, 0x18),
       'h': (0x8005e8e8, 0x10), 'u': (0x8005e8fc, 0xc8), 'v': (0x8005e900, 0x90),
       'uvw': (0x8005e904, 0x18)}
# Win callout x: piece x += -184 (side 0) or +120 (side 1); y += -20.
# The USA pieces are single vertical strips centred on x, placed at 256 +-
# 184 as on the USA disc.
WIN_X1 = (0x80058a08, 0x24630078, 0x246300b8)

RANGES = ([(LISTS, PTAB + 72), (NAMES, NAMES + 93)]
          + [(o, o + 24) for o, _, _ in MOVED.values()]
          + [(o, o + 22) for o in READY] + [(o, o + 24) for o in SWAP]
          + [(r - EB, r - EB + 4) for r in (SW_V, SW_W, SW_H, SW_SLL1, SW_ADD, SW_SLL2)]
          + [(r - EB, r - EB + 4) for r, _ in CTR.values()] + [(WIN_X1[0] - EB, WIN_X1[0] - EB + 4)]
          + fightnick.RANGES)

# ---------------------------------------------------------------- TIM ---

class Tim:
    """4bpp TIM inside a decompressed buffer."""
    def __init__(self, buf, off):
        self.buf, self.off = buf, off
        bl = struct.unpack_from('<I', buf, off + 8)[0]
        cw, ch = struct.unpack_from('<2H', buf, off + 16)
        self.clut = list(struct.unpack_from(f'<{cw * ch}H', buf, off + 20))
        p = off + 8 + bl
        _, vx, vy, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.stride, self.H = p + 12, w * 2, h
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        self.px = np.empty((h, w * 4), np.uint8)
        self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        self.W = self.px.shape[1]

    def store(self):
        raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8)
        self.buf[self.pix:self.pix + self.stride * self.H] = raw.tobytes()
        struct.pack_into(f'<{len(self.clut)}H', self.buf, self.off + 20, *self.clut)

    def pal(self, k):
        return self.clut[16 * k:16 * k + 16]

    def rgb(self, k):
        """(16, 4) RGBA; CLUT value 0 is transparent."""
        return np.array([((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3, 255 if v else 0)
                         for v in self.pal(k)], np.uint8)

def clut_xy(k):
    return (k % 32) * 16, 500 + k // 32

def tp_uv(x, y):
    """TIM x, y -> (tpage x, u, v)."""
    return 768 + 64 * (x // 256), x % 256, y

# ---------------------------------------------------------- rendering ---

_fonts = {}
def font(path, size):
    if (path, size) not in _fonts: _fonts[path, size] = fontpath.truetype(path, size)
    return _fonts[path, size]

def text_alpha(text, cap, path, maxw=None, minsx=0.6, ss=4, sx=1.0, track=0):
    """Alpha of one line with capitals `cap` px tall, rows = cap + 2.
    Squeezed horizontally to maxw (not below minsx, then the size drops)."""
    while True:
        size = 8
        while True:
            f = font(path, size * ss)
            b = f.getbbox('H', anchor='ls')
            if (b[3] - b[1]) / ss >= cap - 0.25: break
            size += 1
        adv = f.getlength(text) / ss + track * (len(text) - 1)
        s = sx
        if maxw and adv * s > maxw: s = maxw / adv
        if s >= minsx or cap <= 8: break
        cap -= 1
    h = cap + 2
    W = int(np.ceil(adv)) + 4
    im = Image.new('L', (W * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        x = 2 * ss + f.getlength(text[:i]) + track * ss * i
        dr.text((x, (1 + cap) * ss), ch, font=f, fill=255, anchor='ls')
    w = max(1, int(round(W * s)))
    a = np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255
    cols = np.nonzero(a.max(0) > 0.05)[0]
    return a[:, cols[0]:cols[-1] + 1] if len(cols) else a

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

DISK2 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], bool)

# ------------------------------------------------------ ROUND digits ---
# As in the first game: Verdana Bold, widened so the '2' matches the USA
# '2' (fill 42x35), USA layers (fill 9-13 sampled from the ROUND fill, 2 px
# outline 1, 2 on the top edge, shadow 8). Palette slot 60.
DIGIT_H, DIGIT_TOP, DIGIT_W2 = 35, 7, 42

def digit_mask(d):
    f = font(F_DIGIT, 160)
    im = Image.new('L', (200, 220), 0)
    ImageDraw.Draw(im).text((20, 10), str(d), font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.5)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def big_digit(d, tex, texm):
    m2 = digit_mask(2)
    k = DIGIT_W2 / (m2.shape[1] / m2.shape[0] * DIGIT_H)
    m = digit_mask(d)
    h, w = m.shape
    W = min(DIGIT_W2, int(round(w / h * DIGIT_H * k)))
    s = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, DIGIT_H), Image.BOX),
                   np.float32) / 255 > 0.5
    F = np.zeros((48, 48), bool)
    x0 = 23 - W // 2
    F[DIGIT_TOP:DIGIT_TOP + DIGIT_H, x0:x0 + W] = s
    S = ndimage.binary_dilation(F, DISK2)
    out = np.zeros((48, 48), np.uint8)
    out[(shift(S, 1, 1) | shift(S, 2, 2) | shift(S, 1, 2) | shift(S, 2, 1)) & ~S] = 8
    out[S] = 1
    out[S & ~shift(S, 1, 0)] = 2
    for r in range(48):
        cols = np.nonzero(F[r])[0]
        if not len(cols): continue
        seq = tex[r][texm[r]]
        out[r, cols] = seq[(40 * d + cols - cols[0]) % len(seq)]
    return out

# -------------------------------------------------------------- names ---
# N2 name cell (u, v) -> tl key. Ids 21-24 and 17 reuse these cells.
NAME_KEYS = {(144, 160): 'kyosuke', (48, 152): 'batsu', (0, 208): 'hinata', (24, 152): 'shoma',
             (0, 184): 'natsu', (144, 200): 'roberto', (96, 160): 'roy', (96, 192): 'tiffany',
             (48, 208): 'boman', (168, 160): 'edge', (192, 160): 'akira', (72, 192): 'gan',
             (216, 160): 'hideo', (72, 152): 'kyoko', (72, 216): 'raizo', (24, 184): 'hyo',
             (24, 208): 'sakura', (0, 152): 'daigo', (120, 192): 'hayato', (120, 160): 'ran',
             (48, 184): 'nagare'}
NAME_IDS = range(1, 27)             # 0 is unused, 27-30 are the runtime player cells
NAME_CAP = 12
NAME_ADV = 5.8
NAME_PAL = 48                       # N2 template record CLUT (256,501)
# N2 palette 48: 3 red ... 15 pale yellow. JP names: bright core, red rim.
N_FILL, N_EDGE, N_RIM = 15, 12, 3

def name_style(a):
    """Vertical name as the N2 JP names: pale yellow fill, orange on the
    right edge of the strokes, a red ring, no shadow (variants compared in
    private/re/fighthud/names_var.png)."""
    m = a > 0.45
    k = ndimage.binary_dilation(m, np.ones((3, 3), bool))
    out = np.zeros(a.shape, np.uint8)
    out[k] = N_RIM
    out[m] = N_FILL
    out[m & ~shift(m, 0, -1)] = N_EDGE
    return out

def name_strip(text):
    """24 x len strip: the name rotated to read top to bottom."""
    a = text_alpha(text, NAME_CAP, F_NAME, maxw=NAME_ADV * len(text), minsx=0.3)
    h, w = a.shape
    full = np.zeros((22, w + 3), np.float32)
    y = (22 - h) // 2
    full[y:y + h, 1:1 + w] = a
    r = np.rot90(full, -1)                    # clockwise: top of letters to the right
    s = np.zeros((r.shape[0], 24), np.float32)
    s[:, 0:22] = r
    return name_style(s)

def name_room(disc):
    """Free name columns [u, top, bottom]: the JP name cells (ids 1-26),
    minus the runtime player cells (fightnick.CELLS), plus the JP player
    cells (fightnick.OLD_CELLS), 24 px wide each."""
    cells = {tuple(disc[NAMES + 3 * i:NAMES + 3 * i + 3]) for i in NAME_IDS}
    cols = {}
    for u, v, h in sorted(cells):
        cols.setdefault(u, []).append([v, v + h])
    room = []
    for u, rs in cols.items():
        cur = list(rs[0])
        for a, b in rs[1:]:
            if a <= cur[1]: cur[1] = max(cur[1], b)
            else: room.append([u, *cur]); cur = [a, b]
        room.append([u, *cur])
    cx, cy, cw, ch = fightnick.CELLS
    out = []
    for u, a, b in room:
        if cx <= u < cx + cw:
            out += [r for r in ([u, a, min(b, cy)], [u, max(a, cy + ch), b]) if r[2] > r[1]]
        else:
            out.append([u, a, b])
    ox, oy, ow, oh = fightnick.OLD_CELLS
    out += [[u, oy, oy + oh] for u in range(ox, ox + ow, 24)]
    return cells, out

def pack_exact(items, room):
    """items: [(key, h)]; room: [u, top, bottom]. Exact search (largest
    first, best-fit order, equal gaps tried once). -> {key: (u, v)}."""
    items = sorted(items, key=lambda x: (-x[1], x[0]))
    left = [b - a for _, a, b in room]
    asg = [None] * len(items)
    def rec(i):
        if i == len(items): return True
        h, seen = items[i][1], set()
        for j in sorted(range(len(room)), key=lambda j: (left[j], room[j][0], room[j][1])):
            if left[j] < h or left[j] in seen: continue
            seen.add(left[j]); left[j] -= h; asg[i] = j
            if rec(i + 1): return True
            left[j] += h
        return False
    if not rec(0): return None
    top = {j: room[j][1] for j in range(len(room))}
    pos = {}
    for (k, h), j in zip(items, asg):
        pos[k] = (room[j][0], top[j]); top[j] += h
    return pos

def build_names(tl, t, exe, disc, problems):
    cells, room = name_room(disc)
    for u, v, h in cells:
        t.px[v:v + h, u:u + 24] = 0
    for x, y, w, h in (fightnick.CELLS, fightnick.OLD_CELLS):
        t.px[y:y + h, x:x + w] = 0
    strips = {k: name_strip(tl['name_' + k]) for k in sorted(set(NAME_KEYS.values()))}
    pos = pack_exact([(k, s.shape[0]) for k, s in strips.items()], room)
    if pos is None:
        problems.append(f'names: {sum(s.shape[0] for s in strips.values())} rows do not fit the '
                        f'name room ({sum(b - a for _, a, b in room)} rows)'); return {}
    slot = {}
    for k, (u, v) in pos.items():
        h = strips[k].shape[0]
        t.px[v:v + h, u:u + 24] = strips[k]
        slot[k] = (u, v, h)
    for i in NAME_IDS:
        o = NAMES + 3 * i
        k = NAME_KEYS.get(tuple(disc[o:o + 2]))
        if k is None or k not in slot:
            problems.append(f'name id {i}: cell {tuple(disc[o:o + 3])} has no key'); continue
        exe[o:o + 3] = bytes(slot[k])
    return slot

# ------------------------------------------------------- round marks ---
# Round-result marks under the health bars: table 0x8008eff0 (9 x u8
# CLUT x/16 at row 500, u, v, w), 24x16 cells in tpage 768. V, P, T, D
# are Latin. Two hold JP text; they take the USA marks as on the USA disc:
# 完全 (Burning Vigor finish, 160,128, palette 14) <- USA BV (184,144);
# 2ﾂｰﾌﾟﾗ (Team Up finish, 184,144, palette 13) <- USA 2 (160,128). Palettes
# 12-15 are the same on both discs except the order of 13.
MARKS = [((184, 144), (160, 128), None),
         ((160, 128), (184, 144), {5: 11, 6: 10, 7: 9, 8: 5, 9: 6, 10: 7, 11: 8})]

def copy_marks(t, us, problems):
    for (sx, sy), (dx, dy), remap in MARKS:
        a = us.px[sy:sy + 16, sx:sx + 24].copy()
        if remap:
            m = np.arange(16, dtype=np.uint8)
            for k, v in remap.items(): m[k] = v
            a = m[a]
        t.px[dy:dy + 16, dx:dx + 24] = a
    if t.pal(13)[5:12] != [us.pal(13)[i] for i in (8, 9, 10, 11, 7, 6, 5)]:
        problems.append('mark palette 13: order not as expected')

# ------------------------------------------------------------- packing ---

class Packer:
    def __init__(self):
        self.free = np.zeros((256, 768), bool)
        for x, y, w, h in FREE: self.free[y:y + h, x:x + w] = True

    def ok(self, x, y, w, h):
        if x // 256 != (x + w - 1) // 256 or y + h > 256: return False
        return self.free[y:y + h, x:x + w].all()

    def take(self, x, y, w, h):
        assert self.ok(x, y, w, h), (x, y, w, h)
        self.free[y:y + h, x:x + w] = False

    def column(self, w, need, xs=range(256, 768), minrun=2):
        """Free column w wide: (x, y, run), the shortest run >= need, else
        the longest run (tpages 832 and 896)."""
        best = None
        for x in xs:
            if x + w > 768: continue
            if x // 256 != (x + w - 1) // 256: continue
            col = np.concatenate([[False], self.free[:, x:x + w].all(1), [False]])
            d = np.diff(col.astype(np.int8))
            for y0, y1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                run = y1 - y0
                if run < min(need, minrun): continue
                key = (0, run) if run >= need else (1, -run)
                if best is None or key < best[0]: best = (key, (x, y0, run))
        return best[1] if best else None

    def find(self, w, h, xs=range(256, 768)):
        for y in range(0, 257 - h):
            row = self.free[y:y + h].all(0)
            for x in xs:
                if row[x:x + w].all() and x // 256 == (x + w - 1) // 256: return x, y
        return None

def place_pieces(problems, tl):
    """-> {piece: (TIM x, y)} and the swap stride."""
    pk = Packer(); pos = {}
    for k in FIXED:                           # USA rects may share empty edges
        x, y, w, h, _ = PIECES[k]
        if not pk.ok(x, y, w, h): problems.append(f'{k}: USA rect not free')
        pos[k] = (x, y)
    for k in FIXED:
        x, y, w, h, _ = PIECES[k]
        pk.free[y:y + h, x:x + w] = False
    grades = pack_grades(pk, tl, problems)       # before the packed pieces
    # swap prompts: u = stride * index at the same v, u of the first = 0
    stride = None
    w, h = PIECES['swap0'][2:4]
    for y in range(0, 257 - h):
        for base in (512, 256):
            for s in sorted(STRIDES):
                if s + w <= 256 and pk.ok(base, y, w, h) and pk.ok(base + s, y, w, h):
                    stride = s; break
            if stride: break
        if stride: break
    if not stride:
        problems.append('swap prompts: no row with room'); return pos, None, pk, grades
    pk.take(base, y, w, h); pk.take(base + stride, y, w, h)
    pos['swap0'], pos['swap1'] = (base, y), (base + stride, y)
    for k in ('timeover', 'draw', 'burn', 'team', 'first', 'tardy', 'ready'):
        w, h = PIECES[k][2:4]
        p = pk.find(w, h, range(0, 768))
        if p is None: problems.append(f'{k}: no room ({w}x{h})'); continue
        pk.take(*p, w, h); pos[k] = p
    return pos, stride, pk, grades

def build_palettes(t, us, pos, problems):
    """Fill the palette slots and remap merged pieces. -> {piece: slot}"""
    of = {}
    for slot, ks in SLOTS.items():
        srcs = {PIECES[k][4] for k in ks}
        for k in ks: of[k] = slot
        if slot not in FREE_SLOTS:                     # existing N2 slot
            if len(srcs) != 1: problems.append(f'slot {slot}: mixed sources'); continue
            s = srcs.pop(); a, b = t.pal(slot), us.pal(s)
            used = set()
            for k in ks:
                x, y, w, h = PIECES[k][:4]
                used |= set(np.unique(us.px[y:y + h, x:x + w]).tolist()) - {0}
            diff = [i for i in used if a[i] != b[i]]
            if diff and slot != 2:
                problems.append(f'slot {slot}: N2 colours differ from USA {s} at {diff}')
            continue
        if len(srcs) == 1:
            t.clut[16 * slot:16 * slot + 16] = us.pal(srcs.pop()); continue
        cols = [0]; maps = {}
        for k in ks:
            x, y, w, h, s = PIECES[k]
            ox, oy, ow, oh = OWN.get(k, (x, y, w, h))
            used = sorted(set(np.unique(us.px[oy:oy + oh, ox:ox + ow]).tolist()) - {0})
            m = np.arange(16, dtype=np.uint8)
            for i in used:
                c = us.pal(s)[i]
                if c not in cols: cols.append(c)
                m[i] = cols.index(c)
            maps[k] = m
        if len(cols) > 16:
            problems.append(f'slot {slot}: {len(cols) - 1} colours'); continue
        t.clut[16 * slot:16 * slot + 16] = cols + [0] * (16 - len(cols))
        for k in ks:
            x, y, w, h, _ = PIECES[k]
            ox, oy, ow, oh = OWN.get(k, (x, y, w, h))
            px, py = pos[k][0] + ox - x, pos[k][1] + oy - y
            a = t.px[py:py + oh, px:px + ow]
            a[:] = maps[k][a]
    return of

def build_art(t, us, tl, problems):
    pos, stride, pk, grades = place_pieces(problems, tl)
    for x, y, w, h in FREE: t.px[y:y + h, x:x + w] = 0
    for k, (px, py) in pos.items():
        x, y, w, h, _ = PIECES[k]
        ox, oy, ow, oh = OWN.get(k, (x, y, w, h))
        src = us.px[oy:oy + oh, ox:ox + ow]          # only the piece's own texels
        dst = t.px[py + oy - y:py + oy - y + oh, px + ox - x:px + ox - x + ow]
        dst[src != 0] = src[src != 0]
    of = build_palettes(t, us, pos, problems)
    copy_marks(t, us, problems)
    t.clut[16 * GRADE_SLOT:16 * GRADE_SLOT + 16] = us.pal(GRADE_SRC)
    for x, y, a in grades[2]:
        t.px[y:y + a.shape[0], x:x + a.shape[1]] = a
    # ROUND digits 3-6, each cut to its ink box (palette slot 60)
    tex = us.px[0:48, 256:512]; texm = tex >= 9
    digits = {}
    for d in (3, 4, 5, 6):
        c_d = big_digit(d, tex, texm)
        ys, xs = np.nonzero(c_d)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bw, bh = x1 - x0, y1 - y0
        bh += bh % 2; parts = []; r0 = 0
        cell = np.pad(c_d[y0:y1, x0:x1], ((0, bh - (y1 - y0)), (0, 0)))
        while r0 < bh:                       # row chunks if no cell fits
            c = pk.column(bw, bh - r0)
            if c is None or c[2] < 2: break
            h = min(c[2], bh - r0); h -= h % 2
            pk.take(c[0], c[1], bw, h)
            t.px[c[1]:c[1] + h, c[0]:c[0] + bw] = cell[r0:r0 + h]
            parts.append((c[0], c[1], r0, h)); r0 += h
        if r0 < bh: problems.append(f'ROUND digit {d}: no room for {bw}x{bh}'); continue
        digits[d] = (parts, (x0, y0, bw, bh))
    return pos, stride, of, digits, grades

# ------------------------------------------------------ result grades ---
# Lists 14-17 (code 0x80051170, winner side in object +0x34): 17 制圧 when
# the CPU beat the human player. Otherwise (human winner, or any winner
# in 2P versus): 15 勝利 after a time over (flag 0x801f64ac), else 14 圧勝
# at full HP, 16 辛勝 with HP < 40, 15 勝利 in between. Drawn here in
# the style of the USA VICTORY piece: capitals stacked top to bottom in a
# 30-texel strip drawn 2x wide, yellow fill 8, orange ring 5, 3 px red
# outline 1, lower fill edge 7, USA palette 64 in slot 63. Strips that do
# not fit in one free column are cut into row chunks, stacked by the
# record y offset (like N1's ROUND 6).
GRADE_SLOT, GRADE_SRC = 63, 64
F_GRADE = 'ariblk.ttf'
GRADE_CAP, GRADE_PITCH, GRADE_SPACE = 10, 14, 6
GRADE_W, GRADE_MAXW, GRADE_K = 30, 24, 2.2    # strip width; fill width limit; widening
GRADE_KEYS = {14: 'win_perfect', 15: 'win', 16: 'win_close', 17: 'win_other'}
# Screen centre of a grade is y ~100 (callout centre 120, win code -20).
# Tall strips move down so that they start below the health bars (y 46).
GRADE_CY, GRADE_TOP = 100, 46
DISK3 = np.hypot(*np.mgrid[-3:4, -3:4]) <= 3.2

def grade_letter(ch):
    im = Image.new('L', (400, 400), 0)
    ImageDraw.Draw(im).text((100, 50), ch, font=font(F_GRADE, 200), fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.5)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    W = max(3, min(GRADE_MAXW, int(round(w / h * GRADE_CAP * GRADE_K))))
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, GRADE_CAP), Image.BOX),
                      np.float32) / 255 > 0.5

def grade_strip(text):
    F = np.zeros((GRADE_PITCH * len(text) + 16, GRADE_W), bool)
    y = 4
    for ch in text:
        if ch == ' ': y += GRADE_SPACE; continue
        g = grade_letter(ch); h, w = g.shape
        x0 = GRADE_W // 2 - (w + 1) // 2
        F[y:y + h, x0:x0 + w] |= g
        y += GRADE_PITCH
    H = y - GRADE_PITCH + GRADE_CAP + 4
    F = F[:H + H % 2]
    d1 = ndimage.binary_dilation(F, np.ones((3, 3), bool)) & ~F
    d2 = ndimage.binary_dilation(F, DISK3) & ~F & ~d1
    out = np.zeros(F.shape, np.uint8)
    out[d2] = 1; out[d1] = 5; out[F] = 8
    out[F & ~shift(F, -1, 0)] = 7
    return out

def pack_grades(pk, tl, problems):
    """Reserve texels for the grade strips. -> ({list: records},
    {list: strip}, [(x, y, strip rows)])"""
    clx, cly = clut_xy(GRADE_SLOT)
    chunks = []
    strips = {i: grade_strip(tl[k].upper()) for i, k in GRADE_KEYS.items()}
    recs = {}
    for i in sorted(strips, key=lambda i: -strips[i].shape[0]):
        s = strips[i]; H = s.shape[0]; y0 = 0; recs[i] = []
        dy = max(0, GRADE_TOP - (GRADE_CY - H // 2))
        while y0 < H:
            c = (pk.column(GRADE_W, H - y0, range(256, 512), 16)
                 or pk.column(GRADE_W, H - y0, minrun=16))
            if c is None or c[2] < 2:
                problems.append(f'grade {i}: no room for {H - y0} rows'); break
            x, y, run = c
            h = min(run, H - y0); h -= h % 2
            if 0 < H - y0 - h < 16 and H - y0 - 16 >= 8: h = H - y0 - 16   # no tiny last chunk
            pk.take(x, y, GRADE_W, h)
            chunks.append((x, y, s[y0:y0 + h]))
            tpx, u, v = tp_uv(x, y)
            recs[i].append((0, y0 + h // 2 - H // 2 + dy, 0, tpx, 256, clx, cly, u, v, GRADE_W - 1,
                            h - 1, 0, 0, 2 * GRADE_W - 1, h - 1))
            y0 += h
    return recs, strips, chunks

# ------------------------------------------------------------- lists ---

def read_lists(exe, ptab, eb):
    L = {}
    for i in range(18):
        p = struct.unpack_from('<I', exe, ptab + 4 * i)[0] - eb
        n = struct.unpack_from('<h', exe, p)[0]
        L[i] = [struct.unpack_from('<15h', exe, p + 2 + 30 * k) for k in range(n)]
    return L

def usa_piece(r):
    """USA list record -> piece name (by its texture rect)."""
    x = (r[3] - 768) * 4 + r[7]
    for k, (px, py, w, h, _) in PIECES.items():
        if (px, py, w, h) == (x, r[8], r[9] + 1, r[10] + 1): return k
    raise KeyError(f'USA record {r}')

def n2_rec(r, k, pos, of, win=False):
    """USA record (N1 builder: y = a + b) -> N2 record (y = a, x = b).
    The win code adds the y -20 itself, as the USA b did."""
    tpx, u, v = tp_uv(*pos[k])
    clx, cly = clut_xy(of[k])
    a = r[1] if win else r[1] + r[2]
    return (r[0], a, 0, tpx, 256, clx, cly, u, v) + tuple(r[9:15])

def build_lists(usa, pos, of, digits, grades):
    U = read_lists(usa, USA_PTAB, 0x80010000 - 0x800)
    L = {}
    for i in (0, 1, 6, 10, 11, 12, 13):
        L[i] = [n2_rec(r, usa_piece(r), pos, of) for r in U[i]]
    L.update(grades)
    word, dig = L[0]
    for n in (3, 4, 5, 6):
        if n not in digits: L[n - 1] = L[0]; continue
        parts, (bx, by, bw, bh) = digits[n]
        recs = [word]
        for x, y, r0, h in parts:           # USA digit: 49x49 cell drawn 1:1
            tpx, u, v = tp_uv(x, y)
            r = list(dig)
            r[2] = int(round(bx + bw / 2 - 24.5))
            r[1] = dig[1] + int(round(by + r0 + h / 2 - 24.5))
            r[3], r[7], r[8] = tpx, u, v
            r[9], r[10], r[13], r[14] = bw - 1, h - 1, bw - 1, h - 1
            recs.append(tuple(r))
        L[n - 1] = recs
    L[9] = L[0]
    return L

def pack_lists(exe, disc, L):
    ptr = lambda i: struct.unpack_from('<I', disc, PTAB + 4 * i)[0] - EB
    blob = bytearray(); where = {}
    for i in range(18):
        if i == 9: continue
        if i in KEEP_LISTS:
            p = ptr(i); n = struct.unpack_from('<h', disc, p)[0]
            body = disc[p:p + 2 + 30 * n]
        else:
            body = struct.pack('<h', len(L[i])) + b''.join(struct.pack('<15h', *r) for r in L[i])
        where[i] = LISTS + len(blob); blob += body
    where[9] = where[0]
    if LISTS + len(blob) > PTAB:
        return f'lists need {len(blob)} bytes, block has {PTAB - LISTS}'
    blob += bytes(PTAB - LISTS - len(blob))
    exe[LISTS:PTAB] = blob
    for i in range(18):
        struct.pack_into('<I', exe, PTAB + 4 * i, where[i] + EB)
    return None

# ----------------------------------------------------------- records ---

def imm(exe, disc, ram, old, new, problems, what):
    o = ram - EB
    w = struct.unpack_from('<I', disc, o)[0]
    if w & 0xffff != old & 0xffff:
        problems.append(f'{what}: unexpected code at {ram:#x} ({w:#010x})'); return
    struct.pack_into('<I', exe, o, (w & 0xffff0000) | (new & 0xffff))

def patch_records(exe, disc, usa, pos, stride, of, problems):
    for k, (o, uo, n) in MOVED.items():
        a = list(struct.unpack_from(f'<{n}h', disc, o)); b = list(struct.unpack_from(f'<{n}h', usa, uo))
        if a[1] != 256 or b[1] != 256 or a[0] not in (832, 896):
            problems.append(f'record {o:#x}: N2/USA pairing looks wrong'); continue
        tpx, u, v = tp_uv(*pos[k])
        clx, cly = clut_xy(of[k])
        struct.pack_into(f'<{n}h', exe, o, tpx, 256, clx, cly, u, v, *b[6:])
    # READY!!: same 81x12 rect as the JP piece, at the JP x, y (same width).
    tpx, u, v = tp_uv(*pos['ready'])
    clx, cly = clut_xy(of['ready'])
    for o in READY:
        a = list(struct.unpack_from('<11h', disc, o))
        if a[:4] != [768, 256, 32, 500] or a[6:8] != [80, 11]:
            problems.append(f'READY record {o:#x} unexpected'); continue
        struct.pack_into('<11h', exe, o, tpx, 256, clx, cly, u, v, 80, 11, *a[8:])
    # swap prompts
    if stride:
        w = PIECES['swap0'][2]
        tpx, u, v = tp_uv(*pos['swap0'])
        clx, cly = clut_xy(of['swap0'])
        for o, cx in zip(SWAP, SWAP_CX):
            a = list(struct.unpack_from('<12h', disc, o))
            if a[:4] != [896, 256, 240, 501] or a[6:8] != [79, 11]:
                problems.append(f'swap record {o:#x} unexpected'); continue
            x = int(round(cx - w / 2))
            struct.pack_into('<12h', exe, o, tpx, 256, clx, cly, u, v, w - 1, 11, x, SWAP_Y, w - 1, 11)
        imm(exe, disc, SW_V, 0xc0, v, problems, 'swap v')
        imm(exe, disc, SW_W, 0x4f, w - 1, problems, 'swap uv w')
        imm(exe, disc, SW_H, 0xb, 11, problems, 'swap uv h')
        s1, op, s2 = STRIDES[stride]
        sll = lambda rd, rt, sa: (rt << 16) | (rd << 11) | (sa << 6)
        words = {SW_SLL1: (0x00032880, sll(5, 3, s1)),                       # sll a1,v1,s1
                 SW_ADD: (0x00a32821, 0x00a32821 if op == 'addu' else 0x00a32823),
                 SW_SLL2: (0x00052900, sll(5, 5, s2))}                         # sll a1,a1,s2
        for ram, (old, new) in words.items():
            o = ram - EB
            if struct.unpack_from('<I', disc, o)[0] != old:
                problems.append(f'swap stride: unexpected code at {ram:#x}'); continue
            struct.pack_into('<I', exe, o, new)
    # win x, side 1
    o = WIN_X1[0] - EB
    if struct.unpack_from('<I', disc, o)[0] != WIN_X1[1]: problems.append('win x: unexpected code')
    else: struct.pack_into('<I', exe, o, WIN_X1[2])

def patch_counter(exe, disc, pos, of, problems):
    """Label 発 (24x16 at y 62) -> HIT (64x12, bottom kept at 78). The P2
    x moves left by the extra width so the P2 right edge stays at 440."""
    x, y, w, h, _ = PIECES['hit']
    tpx, u, v = tp_uv(*pos['hit'])
    clx, cly = clut_xy(of['hit'])
    new = {'p2x': 0x178 - (w - 24), 'tpx': tpx, 'clx': clx, 'cly': cly, 'y': 0x3e + 16 - h,
           'w': w, 'h': h, 'u': u, 'v': v, 'uvw': w}
    for k, (ram, old) in CTR.items():
        imm(exe, disc, ram, old, new[k], problems, f'counter {k}')

# ------------------------------------------------------------- output ---

def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def compress_check(data, problems):
    slot = os.path.getsize('private/disc/DAT2/CEFT1.BIZ'); slot += -slot % 2048
    c = lz.compress(bytes(data), name='CEFT1.BIZ')
    out, done = streamsim.run(c)
    cb, cap = lz.cbmax(c), lz.jp_cap('CEFT1.BIZ')
    ok = done and out[:len(data)] == bytes(data)
    print(f'CEFT1.BIZ: {len(c)} bytes, slot {slot} ({slot - len(c):+d}), per-callback max {cb:#x} '
          f'(JP cap {cap:#x}), streamsim {"ok" if ok else "FAILED"}')
    if cb > cap: problems.append(f'CEFT1.BIZ: callback output {cb:#x} > {cap:#x}')
    if len(c) > slot: problems.append(f'CEFT1.BIZ: {len(c)} > slot {slot}')
    if not ok: problems.append('CEFT1.BIZ: streamsim failed')
    return c

def build(png=False, check=False):
    tl = load_tl(); problems = []
    if not os.path.exists(EXE):
        print(f'{EXE} missing: run tools/hook.py first'); return 1
    exe = bytearray(open(EXE, 'rb').read())
    disc = open(DISC_EXE, 'rb').read()
    usa = open(USA_EXE, 'rb').read()
    if len(exe) != len(disc): problems.append('exe size differs from disc')
    d = bytearray(open(SRC, 'rb').read())
    t = Tim(d, TIM1)
    us = Tim(bytearray(open(SRC_USA, 'rb').read()), TIM1)
    pos, stride, of, digits, (grades, _, _) = build_art(t, us, tl, problems)
    L = build_lists(usa, pos, of, digits, grades)
    err = pack_lists(exe, disc, L)
    if err: problems.append(err)
    patch_records(exe, disc, usa, pos, stride, of, problems)
    patch_counter(exe, disc, pos, of, problems)
    slots = build_names(tl, t, exe, disc, problems)
    fightnick.patch_exe(exe, disc, problems)
    font, fc = fightnick.build_font(problems)
    t.store()
    c = compress_check(d, problems)
    print('layout: ' + ', '.join(f'{k} {x},{y}' for k, (x, y) in pos.items() if k not in FIXED)
          + '; grades ' + ' '.join(f'{i}:' + '+'.join(f'{(r[3] - 768) * 4 + r[7]},{r[8]}x{r[10] + 1}' for r in L[i]) for i in (14, 15, 16, 17))
          + f'; swap stride {stride}; digits ' + ', '.join(f'{n} ' + '+'.join(f'{x},{y}x{h}' for x, y, _, h in p) for n, (p, b) in digits.items()))
    if png: previews(t, exe, slots, font)
    if problems:
        for p in problems: print(p)
        print(f'fight: {len(problems)} problems, nothing written'); return 1
    if check:
        print('fight: --check, problems 0, nothing written'); return 0
    open('private/build/CEFT1.BIZ.lz', 'wb').write(d)
    open('private/build/CEFT1.BIZ', 'wb').write(c)
    open(fightnick.FONT_OUT + '.lz', 'wb').write(font)
    open(fightnick.FONT_OUT, 'wb').write(fc)
    open(EXE, 'wb').write(exe)
    print('fight: wrote CEFT1.BIZ, COCKEDIT.TIZ, SLPS_021.20 (lists, records, names, counter, '
          'swap, nickname), problems 0')
    return 0

# ----------------------------------------------------------- previews ---

CX, CY = 256, 120                   # callout centre on screen (JP frames)

def piece(t, tpx, clx, cly, u, v, w, h):
    x = (tpx - 768) * 4 + u
    return t.rgb((cly - 500) * 32 + clx // 16)[t.px[v:v + h, x:x + w]]

def blit(canvas, src, x, y):
    h, w = src.shape[:2]; H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[..., 3] > 0
    canvas[y0:y1, x0:x1][m] = s[..., :3][m]

def scaled(img, w, h):
    return np.asarray(Image.fromarray(img).resize((max(1, w), max(1, h)), Image.NEAREST))

def list_screen(t, recs, bg, dx=0, dy=0):
    """Final frame of one list (N2 builder: quad sw+1 x sh+1 centred on
    CX + b, CY + a)."""
    scr = bg.copy()
    for r in recs:
        tpx, tpy, clx, cly, u, v, tw, th = r[3:11]
        img = piece(t, tpx, clx, cly, u, v, tw + 1, th + 1)
        hw, hh = (r[13] + 1) // 2, (r[14] + 1) // 2
        img = scaled(img, 2 * hw, 2 * hh)
        blit(scr, img, CX + r[2] + dx - hw, CY + r[1] + dy - hh)
    return scr

def previews(t, exe, slots, font=None):
    os.makedirs(PNG, exist_ok=True)
    bg = np.asarray(Image.open(BG).convert('RGB')).copy()
    L = read_lists(exe, PTAB, EB)
    names = {0: 'ROUND 1', 1: 'ROUND 2', 2: 'ROUND 3', 3: 'ROUND 4', 4: 'ROUND 5', 5: 'ROUND 6',
             6: 'FINAL ROUND', 10: 'FIGHT', 11: 'K.O.', 12: 'DRAW', 13: 'TIME OVER',
             14: 'P1 won, full HP', 15: 'P2 won', 16: 'P1 won, HP < 40', 17: 'CPU (P2) won'}
    tiles = []
    for i, lab in names.items():
        dx = (-184 if i in (14, 16) else 184) if i >= 14 else 0
        scr = list_screen(t, L[i], bg, dx, -20 if i >= 14 else 0)
        im = Image.fromarray(scr); ImageDraw.Draw(im).text((4, 226), f'list {i}: {lab}', fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
    Image.fromarray(np.vstack(rows)).save(f'{PNG}/prev_lists.png')
    # HUD: banner + first attack + counter on the P2 side, swap prompts,
    # READY, names. Positions from the JP frame (P2 banner at x 340).
    scr = bg.copy()
    rec = lambda o, n: struct.unpack_from(f'<{n}h', exe, o)
    bx = 340
    for k, y in (('first', 62), ('burn', 78), ('finish', 104)):
        r = rec(MOVED[k][0], 12)
        img = scaled(piece(t, r[0], r[2], r[3], r[4], r[5], r[6] + 1, r[7] + 1), r[10] + 1, r[11] + 1)
        blit(scr, img, bx + r[8], r[9] + (30 if k == 'first' else 0))
    for k, x0 in (('tardy', 40), ('team', 40)):
        r = rec(MOVED[k][0], 12)
        img = scaled(piece(t, r[0], r[2], r[3], r[4], r[5], r[6] + 1, r[7] + 1), r[10] + 1, r[11] + 1)
        blit(scr, img, x0 + r[8], r[9] + (0 if k == 'tardy' else 28))
    w = lambda ram: struct.unpack_from('<I', exe, ram - EB)[0] & 0xffff
    lx = w(CTR['p2x'][0]) + 40                  # two digits on P2
    lab = piece(t, w(CTR['tpx'][0]), w(CTR['clx'][0]), w(CTR['cly'][0]), w(CTR['u'][0]), w(CTR['v'][0]),
                w(CTR['uvw'][0]), w(CTR['h'][0]))
    for j in range(2):                          # stand-in digits: N2 '1' cells
        blit(scr, scaled(piece(t, 896, 16, 502, 192 + 20 * j, 80, 20, 15), 20, 23), lx - 40 + 20 * j, 0x36)
    blit(scr, lab, lx, w(CTR['y'][0]))
    for o in SWAP:
        r = rec(o, 12)
        for j in range(2):
            img = piece(t, r[0], r[2], r[3], r[4] + j * (_stride(exe)), r[5], r[6] + 1, r[7] + 1)
            blit(scr, img, r[8], r[9] + 14 * j)
    for o in READY[:1]:
        r = rec(o, 11)
        blit(scr, piece(t, r[0], r[2], r[3], r[4], r[5], r[6] + 1, r[7] + 1), r[8], r[9])
    for k, xx in (('hayato', 8), ('sakura', 478)):
        u, v, h = slots[k]
        blit(scr, t.rgb(NAME_PAL)[t.px[v:v + h, u:u + 24]], xx, 48)
    for j, ((_, (x, y), _), p) in enumerate(zip(MARKS, (14, 13))):   # round marks
        blit(scr, t.rgb(p)[t.px[y:y + 16, x:x + 24]], 300 + 30 * j, 47)
    Image.fromarray(scr).save(f'{PNG}/prev_hud.png')
    # all names side by side (1x) on a sky-coloured strip
    ks = sorted(slots)
    scr = np.zeros((max(h for _, _, h in slots.values()) + 8, 26 * len(ks) + 4, 3), np.uint8)
    scr[:] = (180, 120, 90)
    for j, k in enumerate(ks):
        u, v, h = slots[k]
        blit(scr, t.rgb(NAME_PAL)[t.px[v:v + h, u:u + 24]], 4 + 26 * j, 4)
    Image.fromarray(scr).save(f'{PNG}/prev_names.png')
    if font is not None:
        nick_preview(t, slots, fightnick.font_px(font))

def nick_preview(t, slots, fpx):
    """private/re/fightnames/prev_nick.png: top, the HUD corners of the
    sparring frame with the player cell (Makoto, P1) and SAKURA (P2); below,
    1x on sky: player cells Makoto, Seigiman, Mako, R2d2Xy09, then the 21
    names."""
    pal = t.rgb(NAME_PAL)
    bg = np.asarray(Image.open(BG).convert('RGB')).copy()
    blit(bg, pal[fightnick.cell(fpx, fightnick.codes('Makoto'))], 8, 48)
    u, v, h = slots['sakura']
    blit(bg, pal[t.px[v:v + h, u:u + 24]], 478, 48)
    top = np.hstack([bg[36:112, 0:120], np.zeros((76, 4, 3), np.uint8), bg[36:112, 392:512]])
    cells = [fightnick.cell(fpx, fightnick.codes(n)) for n in ('Makoto', 'Seigiman', 'Mako', 'R2d2Xy09')]
    cells += [t.px[v:v + h, u:u + 24] for u, v, h in (slots[k] for k in sorted(slots))]
    low = np.zeros((64, max(top.shape[1], 26 * len(cells) + 4), 3), np.uint8)
    low[:] = (180, 120, 90)
    for j, c in enumerate(cells):
        blit(low, pal[c], 4 + 26 * j + (6 if j >= 4 else 0), 4)
    W = max(top.shape[1], low.shape[1])
    pad = lambda a: np.hstack([a, np.zeros((a.shape[0], W - a.shape[1], 3), np.uint8)])
    Image.fromarray(np.vstack([pad(top), pad(low)])).save('private/re/fightnames/prev_nick.png')

def _stride(exe):
    s1 = (struct.unpack_from('<I', exe, SW_SLL1 - EB)[0] >> 6) & 31
    sub = struct.unpack_from('<I', exe, SW_ADD - EB)[0] & 63 == 0x23
    s2 = (struct.unpack_from('<I', exe, SW_SLL2 - EB)[0] >> 6) & 31
    return ((1 << s1) + (-1 if sub else 1)) << s2

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv, '--check' in sys.argv))
