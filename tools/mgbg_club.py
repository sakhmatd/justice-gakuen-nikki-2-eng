# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Club special practice menus and their stat radar in English.
Patch module for tools/build_mgbg.py; reads tl/CLUBPRAC.txt (the 51
distinct practice names, then the six radar axes Attack, Defense, Stamina,
Guts, Potential, Smarts).

Pixels: one 4bpp 512x256 TIM per club at offset 0x1d800 of MGBGC00.BZZ
entries 0-8 (club byte 0x801f185b order: Baseball, Volleyball, Soccer,
Sumo, Art, Brass Band, Drama, Chemistry, Go-Home). No copies elsewhere on
the disc. The game uploads the pixels to VRAM x 896 (the header says
512); CLUTs at (0,490).
- Labels: six SPRT 160x24 at screen (36, 134 + 24 * row), tpage 896,
  uv (0, 24 * row), CLUT 0 (grey ramp, 1 white .. 15 black). The green
  cursor bar covers the same 160 x 24 (x 36-195). No x table: the whole
  layout is in the texels.
- Radar: SPRT 136x152 at (199,132), tpage 960 uv (0,0) = TIM x 256-391,
  CLUT 2 (navy 1 .. white 15). The dark hexagon is a second sprite (TIM
  x 392-503) under it. The six axis letters are 12x12 kanji at the axis
  tips; the axes span sprite x 13-124, y 14-136. Same texels in all nine
  TIMs.

Layout: every label is drawn fresh and centred in its 160 px cell (the
cursor bar). A name wider than LAB_MAX is squeezed horizontally to fit.
Radar names sit outside the axis tips: Attack centred above the top tip,
Guts below the bottom tip, the four side names above (upper tips) or
below (lower tips) their tip, centred on the tip x and kept inside the
sprite.
Style (STYLE): labels in the JP look: white face, 1 px dark outline (dark
grey above/left, black below/right) and a black drop shadow one further
down-right; no anti-aliasing. Font Antonio wght 700 (as the hub-header
club names, cock_hud.py): each glyph is rendered hinted at 1x and
binarised on its own, so a letter has the same pixels everywhere, and
glyphs are set by their ink with a fixed face gap (ant_mask). Radar names:
the same Antonio glyphs, binary white with a navy shadow down-right. A
radar name that would touch an axis moves away from the hexagon.
STYLES/RSTYLES hold the variants that were compared
(private/re/clubprac/findings.org).

Usage: python3 tools/mgbg_club.py                    widths
       python3 tools/mgbg_club.py --png OUT [--styles A,B] [--sheet]
(needs the clean background private/re/clubprac/bg.png).
"""
import os, sys, struct, functools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath

ARCH, TIM_OFF, NCLUB = 'MGBGC00.BZZ', 0x1d800, 9
CLUBS = ['Baseball', 'Volleyball', 'Soccer', 'Sumo', 'Art', 'Brass Band', 'Drama', 'Chemistry', 'Go-Home']
# JP label per club and row (rows 0-5, v = 24 * row)
JP = [l.split() for l in """打撃練習 投球練習 守備練習 フォーメーション練習 ダッシュ ランニング
スパイク練習 レシーブ練習 サーブ練習 トス練習 ブロック練習 ランニング
シュート練習 ドリブル練習 パス練習 セービング練習 ダッシュ ランニング
股割り すり足 四股踏み 鉄砲 ぶつかり稽古 チャンコ
水彩画 油彩画 彫刻 デッサン 写生 美術鑑賞
腹筋 管楽器 打楽器 マーチング リズムトレーニング 音楽鑑賞
発声練習 ダンス練習 演技練習 衣装づくり ストレッチ 脚本読み
薬品調合 合成実験 観察・分析 人体実験 動物実験 レポート
カラオケ アルバイト ウィンドウショッピング 買い食い ゲームセンター 昼寝""".splitlines()]
AXES = ['攻撃力', '防御力', '体力', '根性', '潜在能力', '知力']
LAB_W, LAB_H, NROW = 160, 24, 6
LAB_MAX = 156                         # ink incl. outline and shadow
LAB_SCREEN = (36, 134)                # screen x, y of row 0 (preview)
RAD_X, RAD_W, RAD_H = 256, 136, 152   # radar sprite in the TIM
RAD_SCREEN = (199, 132)
# JP axis letters (sprite x0, x1, y0, y1), clockwise from the top
JP_AXIS = [(62, 73, 0, 11), (124, 135, 34, 45), (123, 135, 105, 117),
           (62, 73, 138, 150), (1, 12, 105, 116), (1, 12, 35, 46)]
# EN anchors: (tip x, text top row) per axis, clockwise from the top
AXIS_ANCHOR = [(68, 1), (121, 31), (121, 113), (68, 140), (14, 113), (14, 31)]
CLUT_LAB, CLUT_RAD = 0, 2
SS = 4

SAIRA_C = 'SairaCondensed-Bold.ttf'                # OFL 1.1
SAIRA_SC = 'SairaSemiCondensed-Bold.ttf'       # OFL 1.1
BARLOW_C = 'BarlowCondensed-Bold.ttf'             # OFL 1.1
ROBOTO_C = 'RobotoCondensed-Bold.ttf'                          # OFL 1.1
ANTONIO = 'Antonio[wght].ttf'                              # OFL 1.1
# labels: font, cap px, baseline row, face index top/bottom (CLUT 0)
STYLES = {
    'saira': dict(font=SAIRA_C, cap=15, base=18, top=1, bot=1),
    'saira-grad': dict(font=SAIRA_C, cap=15, base=18, top=1, bot=4),
    'barlow': dict(font=BARLOW_C, cap=15, base=18, top=1, bot=1),
    'roboto': dict(font=ROBOTO_C, cap=14, base=18, top=1, bot=1),
    # Antonio per-glyph: weight, face gap (2 = the outlines of neighbours
    # touch but do not merge), extra columns for a space
    'antonio': dict(font=ANTONIO, cap=15, base=18, top=1, bot=1, wght=700, gap=2, space=3),
}
STYLE = 'antonio'
# radar: font, cap px, anti-aliased (CLUT 2 ramp) or binary
RSTYLES = {
    'r-aa8': dict(font=SAIRA_C, cap=8, aa=True),
    'r-bin8': dict(font=SAIRA_C, cap=8, aa=False),
    'r-aa9': dict(font=SAIRA_C, cap=9, aa=True),
    'r-bin9r': dict(font=ROBOTO_C, cap=8, aa=False),
    'r-aa9b': dict(font=BARLOW_C, cap=9, aa=True),
    'r-ant10': dict(font=ANTONIO, cap=10, aa=False, wght=700, gap=1, space=2),
}
RSTYLE = 'r-ant10'
KEY = 0.5                              # coverage threshold for binary ink
EDGE_TL, EDGE_BR, SHADOW = 12, 15, 14  # CLUT 0: grey 56, black 8, black 24


def load_tl(path='tl/CLUBPRAC.txt'):
    d = {}; jp = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('JP: '): jp = line[4:]
        elif line.startswith('EN: ') and jp is not None:
            d[jp] = line[4:].strip(); jp = None
    for row in JP:
        for j in row: assert d.get(j), f'tl/CLUBPRAC.txt: no EN for {j}'
    for a in AXES: assert d.get('レーダー:' + a), f'tl/CLUBPRAC.txt: no EN for axis {a}'
    return d

# ---------------------------------------------------------------- TIM
@functools.lru_cache(None)
def orig_entry(i):
    import lz, arc
    d = open('private/disc/DAT2/' + ARCH, 'rb').read()
    o, s = arc.toc(d)[i]
    return lz.decompress(d[o:o + s])[0]

def layout(d):
    """-> (CLUTs 32 x 16 u16, pixel offset); asserts the 512x256 4bpp TIM."""
    assert struct.unpack_from('<II', d, TIM_OFF) == (0x10, 0x8)
    cl = struct.unpack_from('<I', d, TIM_OFF + 8)[0]
    assert struct.unpack_from('<4H', d, TIM_OFF + 12) == (0, 490, 512, 1)
    q = TIM_OFF + 8 + cl
    assert struct.unpack_from('<4H', d, q + 4) == (512, 0, 128, 256)
    clut = np.frombuffer(d, '<u2', 512, TIM_OFF + 20).reshape(32, 16)
    return clut.copy(), q + 12

def unpack(d):
    _, p = layout(d)
    b = np.frombuffer(d, np.uint8, 256 * 256, p).reshape(256, 256)
    return np.stack([b & 15, b >> 4], -1).reshape(256, 512)

def pack(d, idx):
    _, p = layout(d)
    b = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    out = bytearray(d); out[p:p + 256 * 256] = b.tobytes()
    return bytes(out)

def rgb(clut):
    c = clut.astype(int)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.uint8)

# ---------------------------------------------------------------- render
@functools.lru_cache(None)
def font(path, cap):
    for sz in range(cap * SS, cap * SS * 3):
        f = fontpath.truetype(path, sz)
        if -f.getbbox('H', anchor='ls')[1] >= cap * SS: return f
    raise SystemExit('mgbg_club: no font size for the cap height')

def down(a):
    h, w = a.shape
    return a.reshape(h // SS, SS, w // SS, SS).mean((1, 3))

def shift(a, dx, dy):
    out = np.zeros_like(a)
    out[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return out

def coverage(text, path, cap, base, h, maxw):
    """-> (coverage h x w at 1x with the ink from x 0, squeeze factor). The
    text is squeezed horizontally when its ink is wider than maxw."""
    f = font(path, cap)
    W = (len(text) * cap * 2 + 8) * SS
    im = Image.new('L', (W, h * SS), 0)
    ImageDraw.Draw(im).text((2 * SS, base * SS), text, font=f, fill=255, anchor='ls')
    a = np.asarray(im)
    cols = np.nonzero(a.any(0))[0]
    a = a[:, cols[0]:cols[-1] + 1]
    w = a.shape[1]
    sq = 1.0
    if w > maxw * SS:
        sq = maxw * SS / w
        a = np.asarray(Image.fromarray(a).resize((maxw * SS, h * SS), Image.LANCZOS))
    w = -(-a.shape[1] // SS) * SS
    a = np.pad(a, ((0, 0), (0, w - a.shape[1])))
    return down(a.astype(float) / 255), sq

@functools.lru_cache(None)
def ant_font(wght, cap):
    """Antonio at the size where 'H' is cap rows when rendered hinted at 1x."""
    for s10 in range(80, 400, 5):
        f = fontpath.truetype(ANTONIO, s10 / 10)
        f.set_variation_by_axes([wght])
        im = Image.new('L', (60, 40), 0)
        ImageDraw.Draw(im).text((10, 28), 'H', font=f, fill=255, anchor='ls')
        if (np.asarray(im) >= 128).any(1).sum() >= cap: return f
    raise SystemExit('mgbg_club: no Antonio size for the cap height')

@functools.lru_cache(None)
def ant_glyph(ch, wght, cap, base):
    """-> binary mask (24 rows, baseline row base) cropped to its ink."""
    im = Image.new('L', (80, 40), 0)
    ImageDraw.Draw(im).text((10, 28), ch, font=ant_font(wght, cap), fill=255, anchor='ls')
    m = np.asarray(im) >= 128
    assert not m[:28 - base].any() and not m[28 - base + LAB_H:].any(), ('glyph outside the cell', ch)
    xs = np.where(m.any(0))[0]
    return m[28 - base:28 - base + LAB_H, xs[0]:xs[-1] + 1]

def ant_mask(text, wght, cap, base, gap, space):
    """Set the glyphs by ink: the closest approach between neighbours
    (rows within +-1) is gap empty columns; a space adds space columns."""
    out = np.zeros((LAB_H, 12 * len(text) + 12), bool)
    prev, pend = None, 0
    for ch in text:
        if ch == ' ':
            pend += space
            continue
        m = ant_glyph(ch, wght, cap, base)
        x = 0
        if prev is not None:
            pm, px = prev
            for a in range(LAB_H):
                ra = np.where(pm[a])[0]
                if not len(ra): continue
                for b in range(max(a - 1, 0), min(a + 2, LAB_H)):
                    lb = np.where(m[b])[0]
                    if len(lb): x = max(x, px + ra[-1] + gap + 1 - lb[0])
            x = max(x, px + pm.shape[1] + gap - 1) + pend
        out[:, x:x + m.shape[1]] |= m
        prev, pend = (m, x), 0
    xs = np.where(out.any(0))[0]
    return out[:, :xs[-1] + 1]

def label_face(text, st):
    """-> (binary face 24 rows, squeeze or Antonio weight)."""
    if 'wght' not in st:
        cov, sq = coverage(text, st['font'], st['cap'], st['base'], LAB_H, LAB_MAX - 3)
        return cov >= KEY, sq
    for w in range(st['wght'], 590, -20):       # a too-wide name drops weight
        face = ant_mask(text, w, st['cap'], st['base'], st['gap'], st['space'])
        if face.shape[1] <= LAB_MAX - 3: break
    return face, w

def label_cell(text, st):
    """-> (24 x 160 index cell, ink width, squeeze or weight)."""
    face, sq = label_face(text, st)
    h, w = face.shape
    face = np.pad(face, ((0, 0), (1, 2)))               # room for edge and shadow
    ring = np.zeros_like(face)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ring |= np.roll(np.roll(face, dy, 0), dx, 1)
    ring &= ~face
    br = (shift(face, 1, 0) | shift(face, 0, 1) | shift(face, 1, 1)) & ring
    shadow = shift(face | ring, 1, 1) & ~face & ~ring
    t = np.linspace(0, 1, LAB_H)[:, None]
    fill = np.rint(st['top'] + (st['bot'] - st['top']) * t).astype(np.uint8) * np.ones_like(face, np.uint8)
    out = np.zeros(face.shape, np.uint8)
    out[shadow] = SHADOW
    out[ring] = EDGE_TL
    out[br] = EDGE_BR
    out[face] = fill[face]
    ink = out.shape[1]
    assert ink <= LAB_MAX, (text, ink)
    assert not out[0].any(), (text, 'ink touches row 0')
    cell = np.zeros((LAB_H, LAB_W), np.uint8)
    # centre the face + outline (the shadow hangs right)
    x0 = (LAB_W - (ink - 1)) // 2
    cell[:, x0:x0 + ink] = out
    return cell, ink, sq

def axis_cell(text, rs):
    """-> index block (h x w) for one radar name, CLUT 2."""
    cap = rs['cap']; h = cap + 6
    if 'wght' in rs:
        m = ant_mask(text, rs['wght'], cap, LAB_H - 6, rs['gap'], rs['space'])
        cov = np.pad(m, ((0, 1), (0, 1))).astype(float)
    else:
        cov, _ = coverage(text, rs['font'], cap, cap + 1, h, 60)
        cov = np.pad(cov, ((0, 1), (0, 1)))
    if rs['aa']:
        face = np.where(cov >= 0.2, np.rint(1 + 14 * np.clip(cov, 0, 1)), 0).astype(np.uint8)
    else:
        face = np.where(cov >= KEY, 15, 0).astype(np.uint8)
    sh = shift(face > 0, 1, 1) & (face == 0)
    out = face.copy(); out[sh] = 1
    rows = np.nonzero(out.any(1))[0]
    return out[rows[0]:rows[-1] + 1]

@functools.lru_cache(None)
def build_radar(rstyle=RSTYLE):
    """-> radar index block (152 x 136) for every TIM."""
    tl = load_tl(); rs = RSTYLES[rstyle]
    blk = unpack(orig_entry(0))[:RAD_H, RAD_X:RAD_X + RAD_W].copy()
    for x0, x1, y0, y1 in JP_AXIS: blk[y0:y1 + 1, x0:x1 + 1] = 0
    axes = blk.copy()
    for k, (a, (tx, top)) in enumerate(zip(AXES, AXIS_ANCHOR)):
        c = axis_cell(tl['レーダー:' + a], rs)
        h, w = c.shape
        x = min(max(tx - (w - 1) // 2, 0), RAD_W - w)
        if k == 3: top = RAD_H - h                   # bottom name flush with the sprite edge
        step = -1 if k in (0, 1, 5) else 1           # away from the hexagon
        while 0 <= top <= RAD_H - h and (axes[top:top + h, x:x + w][c > 0]).any(): top += step
        reg = blk[top:top + h, x:x + w]
        assert top >= 0 and reg.shape == c.shape, ('radar name outside the sprite', a)
        assert not (axes[top:top + h, x:x + w][c > 0]).any(), ('radar name over an axis', a)
        reg[c > 0] = c[c > 0]
    return blk

@functools.lru_cache(None)
def build(i, style=STYLE, rstyle=RSTYLE):
    """-> (new TIM index array for club i, [(en, ink, squeeze)])."""
    tl = load_tl(); st = STYLES[style]
    idx = unpack(orig_entry(i)).copy()
    info = []
    for r, jp in enumerate(JP[i]):
        cell, ink, sq = label_cell(tl[jp], st)
        idx[LAB_H * r:LAB_H * (r + 1), 0:LAB_W] = cell
        info.append((tl[jp], ink, sq))
    idx[:RAD_H, RAD_X:RAD_X + RAD_W] = build_radar(rstyle)
    return idx, info

def make_patch(i):
    def patch(data):
        o = orig_entry(i)
        _, p = layout(data)
        assert data[TIM_OFF:p + 256 * 256] == o[TIM_OFF:p + 256 * 256], f'{ARCH} {i}: TIM already changed'
        return pack(data, build(i)[0])
    return patch

PATCHES = [(ARCH, i, make_patch(i)) for i in range(NCLUB)]

# ---------------------------------------------------------------- preview
BG = 'private/re/clubprac/bg.png'
CROP = (28, 126, 342, 290)

def scene(i, style=STYLE, rstyle=RSTYLE, jp=False):
    clut, _ = layout(orig_entry(i))
    idx = unpack(orig_entry(i)) if jp else build(i, style, rstyle)[0]
    a = np.asarray(Image.open(BG).convert('RGB')).copy()
    x, y = LAB_SCREEN
    pal = rgb(clut[CLUT_LAB]); cell = idx[:LAB_H * NROW, :LAB_W]
    reg = a[y:y + LAB_H * NROW, x:x + LAB_W]; m = cell > 0; reg[m] = pal[cell[m]]
    x, y = RAD_SCREEN
    pal = rgb(clut[CLUT_RAD]); cell = idx[:RAD_H, RAD_X:RAD_X + RAD_W]
    reg = a[y:y + RAD_H, x:x + RAD_W]
    # only the letters: the axes are already in the background shot
    base = unpack(orig_entry(i))[:RAD_H, RAD_X:RAD_X + RAD_W].copy()
    for x0, x1, y0, y1 in JP_AXIS: base[y0:y1 + 1, x0:x1 + 1] = 0
    m = (cell > 0) & (base == 0); reg[m] = pal[cell[m]]
    return Image.fromarray(a).crop(CROP)

def preview(out, styles=None, clubs=(0, 7, 8)):
    """Variants: one row per style (label style:radar style), one column per club."""
    styles = styles or [f'{STYLE}:{RSTYLE}']
    W, H = CROP[2] - CROP[0], CROP[3] - CROP[1]
    rows = [('JP', None)] + [(s, s) for s in styles]
    sheet = Image.new('RGB', (W * len(clubs), (H + 12) * len(rows)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for r, (name, s) in enumerate(rows):
        y0 = r * (H + 12)
        dr.text((2, y0), name, fill=(255, 255, 0))
        for c, i in enumerate(clubs):
            if s is None: im = scene(i, jp=True)
            else:
                st, rs = (s.split(':') + [RSTYLE])[:2]
                im = scene(i, st, rs)
            sheet.paste(im, (c * W, y0 + 12))
    sheet.save(out)

def contact(out):
    """All nine menus, JP and EN side by side, two clubs per row."""
    W, H = CROP[2] - CROP[0], CROP[3] - CROP[1]
    n = (NCLUB + 1) // 2
    sheet = Image.new('RGB', (W * 4, (H + 12) * n), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i in range(NCLUB):
        x0, y0 = (i % 2) * 2 * W, (i // 2) * (H + 12)
        dr.text((x0 + 2, y0), f'{i} {CLUBS[i]} JP', fill=(255, 255, 0))
        dr.text((x0 + W + 2, y0), f'{i} {CLUBS[i]} EN', fill=(255, 255, 0))
        sheet.paste(scene(i, jp=True), (x0, y0 + 12))
        sheet.paste(scene(i), (x0 + W, y0 + 12))
    sheet.save(out)

if __name__ == '__main__':
    if '--png' in sys.argv:
        out = sys.argv[sys.argv.index('--png') + 1]
        if '--sheet' in sys.argv: contact(out)
        else:
            st = sys.argv[sys.argv.index('--styles') + 1].split(',') if '--styles' in sys.argv else None
            preview(out, st)
    else:
        for i in range(NCLUB):
            for en, ink, sq in build(i)[1]:
                print(f'{i} {ink:4d} {sq:g} {en}')
