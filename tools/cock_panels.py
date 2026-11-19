# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary menu panel labels in DAT2/MGCOCK.TIZ (part of build_cock.py).

apply(tim, mgame) draws the 19 panel labels (hub, diary system menu,
message speed, minigame practice) in English and rewrites their MGAME
records. Wording: tl/COCK.txt, section "Menu panels" (keys パネル:JP). RE: topic cock
(private/re/cock/findings.org, section Menu panel labels).

Records (MGAME file 0x1f84 + 8 * id): n, tpage x / 64, tpage y / 64, u, v,
half width, half height, 0. The game draws n sprites of 24x24 from
u + 24 i, spread over the panel width (pitch 2 * halfw / n); n = 255
draws one 144x24 sprite at x = cx - 72 (MGAME 0x800ba868, 0x800bb3a8; any
panel). A label whose ink fits 144 px in a wider panel uses n = 255 and a
144-texel strip. Otherwise n is 2 * halfw / 24: the pitch is 24 and the
cells join into one strip of the panel width. Either strip is centred on
the panel. The half widths (panel frames) are not changed. Each label is
laid out fresh and centred in its strip.

Texels: the original JP label rectangles (cleared first) and the free rows
listed in AREAS. tp640 u0-111 v128-255 is overwritten at runtime (portrait
in the message box), so only u112-255 is used there. A strip's transparent
margins may overlap a neighbour's margins, never its ink.

Style as the JP labels: index 1 fill (white, pal 4; yellow when selected,
pal 6), a grey outline (indices 10-12) and a dark drop shadow (15) down
and right. The 15 palette entries form a grey ramp, so the levels are
quantised by index.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage

REC, CTR = 0x1f84, 0x1f38          # MGAME file offsets (RAM - 0x800a2000)
TL = 'tl/COCK.txt'
KEY_PREFIX = 'パネル:'
FONT = 'RobotoCondensed-Bold.ttf'    # OFL 1.1
# per group: font size, letter spacing (px, may be fractional), baseline
# row. The lowercase x-height is centred on the 24-row cell; descenders,
# outline and shadow still fit. The minigame list is smaller because of
# the longest name (Showdown! Blazing Sprint, 168 px panel).
STYLE = {'hub': (22, 0.0, 18), 'sys': (22, 0.0, 18), 'speed': (22, 0.0, 18),
         'mini': (16, 0.0, 17)}
# per-label letter spacing where the group style does not fit the panel
TRACK = {13: -0.4}
# (id, JP key in tl/COCK.txt, group, original label rect: tpage x, u, v, w)
LABELS = [
    (0, '進める', 'hub', 576, 168, 72, 72),
    (1, '移動', 'hub', 576, 144, 96, 48),
    (2, 'ステータス', 'hub', 576, 0, 144, 120),
    (3, '日記', 'hub', 576, 120, 144, 48),
    (4, 'バックアップ', 'sys', 576, 0, 96, 144),
    (5, 'メッセージスピード', 'sys', 576, 0, 0, 216),
    (6, 'ミニゲームの練習', 'sys', 576, 0, 120, 192),
    (7, '試験の練習', 'sys', 512, 0, 216, 120),
    (8, '終了', 'sys', 576, 144, 24, 48),
    (9, '遅い', 'speed', 512, 56, 176, 48),
    (10, '普通', 'speed', 576, 192, 96, 48),
    (11, '速い', 'speed', 512, 56, 152, 48),
    (12, '一括', 'speed', 576, 192, 24, 48),
    (13, '決戦! 炎のスプリント', 'mini', 576, 0, 168, 144),
    (14, 'Doki―Waku―Dance', 'mini', 576, 0, 192, 144),
    (15, 'PK戦', 'mini', 576, 168, 48, 72),
    (16, '爆裂シュート', 'mini', 576, 0, 24, 144),
    (17, 'サービスエース', 'mini', 576, 0, 48, 168),
    (18, 'ホームラン競争', 'mini', 576, 0, 72, 168),
]
# Extra free rows (tpage x, v, u0, u1), 24 rows each, in order of use.
AREAS = [(640, 208, 112, 256), (640, 232, 112, 256), (640, 184, 200, 256),
         (704, 192, 0, 256), (704, 216, 0, 256), (704, 240, 0, 256),
         (704, 168, 120, 232)]
# grey level (0-31) of palette indices 1-15 (pal 4 at VRAM 64,480)
RAMP = np.array([31, 29, 27, 25, 22, 20, 18, 16, 14, 12, 10, 7, 5, 3, 1])
H = 24
BASE = 18          # default baseline row


def load_tl(path=TL):
    """JP -> EN pairs of the section "# --- Menu panels (tools/cock_panels.py)".
    Keys there are KEY_PREFIX + JP."""
    out, jp, on = {}, None, False
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('# --- '):
            on = 'cock_panels' in line
        elif not on:
            continue
        elif line.startswith('JP: '):
            jp = line[4:].removeprefix(KEY_PREFIX)
        elif line.startswith('EN: ') and jp is not None:
            out[jp] = line[4:].strip(); jp = None
    return out


def coverage(text, size, track=0.0, font=FONT, base=BASE):
    """Hinted 1x coverage (H, w) of text, cropped to its ink columns. The
    ink plus outline and shadow must stay inside the H rows."""
    f = fontpath.truetype(font, size)
    w = int(f.getlength(text) + abs(track) * len(text)) + 8
    im = np.zeros((H + 8, w))
    for i, ch in enumerate(text):
        if ch == ' ':
            continue
        x = 4 + f.getlength(text[:i]) + i * track
        g = Image.new('L', (w, H + 8), 0)
        ImageDraw.Draw(g).text((round(x), base + 4), ch, font=f, fill=255,
                               anchor='ls')
        im = np.maximum(im, np.asarray(g) / 255.0)
    rows = np.where((im > 0.1).any(1))[0]
    if rows[0] < 5 or rows[-1] > H + 1:
        raise SystemExit(f'cock_panels: {text!r} does not fit {H} rows')
    cols = np.where((im > 0.1).any(0))[0]
    return im[4:H + 4, cols[0]:cols[-1] + 1]


def style(a):
    """Palette indices (H, w + 3) of a coverage piece: 1 px outline on all
    sides, drop shadow 1 px down-right of the outline."""
    a = np.pad(a, ((0, 0), (1, 2)))
    k = np.array([[.5, 1, .5], [1, 1, 1], [.5, 1, .5]])
    ol = np.clip(ndimage.convolve(a, k, mode='constant'), 0, 1)
    sh = np.zeros_like(ol); sh[1:, 1:] = ol[:-1, :-1]
    # composite back to front: shadow (level 1), outline (11), fill (31)
    pm = np.zeros_like(a); al = np.zeros_like(a)
    for layer, g in ((sh, 1.0), (ol, 11.0), (a, 31.0)):
        pm = pm * (1 - layer) + g * layer
        al = al * (1 - layer) + layer
    lev = pm / np.maximum(al, 1e-6)
    out = np.zeros(a.shape, np.uint8)
    on = al > 0.35
    idx = np.abs(lev[..., None] - RAMP[None, None, :]).argmin(-1) + 1
    out[on] = idx[on]
    return out


def render(text, size, track=0.0, font=FONT, base=BASE):
    return style(coverage(text, size, track, font, base))


def strips(tl, styles=STYLE, font=FONT):
    """id -> (piece, strip length, group) for every label."""
    out = {}
    for i, jp, grp, *_ in LABELS:
        if jp not in tl or not tl[jp]:
            raise SystemExit(f'cock_panels: no EN for {jp} in {TL}')
        size, track, base = styles[grp]
        track = TRACK.get(i, track)
        out[i] = (render(tl[jp], size, track, font, base), grp)
    return out


def rows():
    """Usable rows: the JP rectangles (merged where adjacent) + AREAS."""
    rr = {}
    for _, _, _, tp, u, v, w in LABELS:
        rr.setdefault((tp, v), []).append((u, u + w))
    out = []
    for (tp, v), segs in rr.items():
        segs.sort(); cur = list(segs[0])
        for a, b in segs[1:]:
            if a <= cur[1]: cur[1] = max(cur[1], b)
            else: out.append((tp, v, *cur)); cur = [a, b]
        out.append((tp, v, *cur))
    return sorted(out, key=lambda r: (-(r[3] - r[2]), r)) + AREAS


def pack(pieces, lengths):
    """Place each strip (length L, ink centred) in a row: longest first,
    each at the leftmost free spot of the row it fills best (least room
    left to the right). Returns id -> (tp, v, u, ink_x0). Strips may
    overlap only where both are blank."""
    rr = rows(); placed = [[] for _ in rr]; res = {}
    order = sorted(pieces, key=lambda i: (-lengths[i], -pieces[i].shape[1]))
    for i in order:
        L, w = lengths[i], pieces[i].shape[1]; m = (L - w) // 2
        best = None
        for ri, (tp, v, u0, u1) in enumerate(rr):
            for u in range(u0, u1 - L + 1):
                if all((u + L <= o[2] or u >= o[3]) and
                       (u + m + w <= o[0] or u + m >= o[1]) for o in placed[ri]):
                    end = max([u + L] + [o[1] for o in placed[ri]])
                    if best is None or u1 - end < best[0]:
                        best = (u1 - end, ri, u)
                    break
        if best is None:
            raise SystemExit(f'cock_panels: no room for label {i}')
        _, ri, u = best
        placed[ri].append((u, u + L, u + m, u + m + w))
        res[i] = (rr[ri][0], rr[ri][1], u, u + m)
    return res


def col(tp, u):
    """Sheet column of tpage tp (VRAM x) texel u."""
    return (tp - 512) * 4 + u


# sheet rectangles (x, y, w, h) this module may write (build_cock check)
SPACE = [(col(tp, u), v, w, H) for _, _, _, tp, u, v, w in LABELS] + \
        [(col(tp, u0), v, u1 - u0, H) for tp, v, u0, u1 in AREAS]


def _pix(tim):
    for k in ('px', 'pix', 'pixels', 'idx'):
        if hasattr(tim, k): return getattr(tim, k)
        if isinstance(tim, dict) and k in tim: return tim[k]
    if isinstance(tim, (tuple, list)): return tim[0]
    return tim


def apply(tim, mgame, tl=None, styles=STYLE, font=FONT, log=True):
    px = _pix(tim)
    assert px.shape == (256, 1024), px.shape
    tl = tl or load_tl()
    pieces = {i: p for i, (p, g) in strips(tl, styles, font).items()}
    halfw = {i: mgame[REC + 8 * i + 5] for i, *_ in LABELS}
    lengths = {i: 144 if 2 * halfw[i] > 144 and pieces[i].shape[1] <= 142
               else 2 * halfw[i] for i in pieces}
    for i in pieces:
        if pieces[i].shape[1] > lengths[i]:
            raise SystemExit(f'cock_panels: label {i} is {pieces[i].shape[1]}'
                             f' px, panel {lengths[i]}')
        assert lengths[i] % 24 == 0
    # clear the JP rectangles and the extra rows
    for _, _, _, tp, u, v, w in LABELS:
        px[v:v + H, col(tp, u):col(tp, u + w)] = 0
    for tp, v, u0, u1 in AREAS:
        px[v:v + H, col(tp, u0):col(tp, u1)] = 0
    pos = pack(pieces, lengths)
    for i, (tp, v, u, k0) in pos.items():
        p = pieces[i]; x = col(tp, k0)
        px[v:v + H, x:x + p.shape[1]] = p
        r = REC + 8 * i
        n = 255 if lengths[i] != 2 * halfw[i] else lengths[i] // 24
        assert u + lengths[i] <= 256
        mgame[r:r + 5] = bytes((n, tp // 64, 256 // 64, u, v))
        if log:
            print(f'  panel {i:2d}: n {n:3d} tp{tp} u{u:3d} v{v:3d} '
                  f'ink {p.shape[1]}/{lengths[i]}')
    return pos
