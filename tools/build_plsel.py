# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Festival Sprint/Dance popups in English: the PLAYER SELECT popup
(MGAME06 Sprint, MGAME07 Dance) and the Sprint opponent list with its
fast/slow icons (MGAME06). Wording: tl/PLSEL.txt. RE:
private/re/fest/findings.org (sections PLAYER SELECT popup, Sprint
opponent list, Player select build).

PLAYER SELECT: TIM 3 at 0x91800 of MINITTL9/A/B/C.BIZ (identical copies),
4bpp 256x256 at VRAM (960,0), CLUTs 0-2 at (0,483), cells of 20x20, 12 per
row. MGAME06 0x800a63dc draws a line from an entry (count, then count x
(cell, ncells)): each run is one SPRT of 20*n x 20 at u = 20*(cell%12),
v = 20*(cell//12); the runs are laid side by side and the line is centred
on x 256 (arrows at 128 and 366). This tool redraws cell rows 0-7 and 11
with EN runs and rewrites the entries in MGAME06 (names 0x2b0, school+sex
0x268) and MGAME07 (0x280, 0x238). Player/Partner/OK are redrawn in their
JP rects; the fixed records (clut, u, v, w, x, y) at MGAME06 0x384 and
MGAME07 0x354 keep their rects, except that the VS set re-centres
"Player 1P"/"Player 2P".

Opponent list: label atlas TIM 0 of every MINITTL0-C.BIZ (the atlas
build_minittl.py paints; this tool only touches the JP opponent-name
cells x 160-255 y 168-239 minus the f_unit cells, and the empty strip
y 240-255). Records of 6 s16 (x, y, u, v, w, h) at MGAME06 0x1a8 (idx 0-4
names, 5 arrow, 6 slow, 7 fast; tpage (768,0) so u < 256).

Build order: after build_minittl.py and build_minihud.py (it patches their
private/build output in place; idempotent), before build_pause.py.
Options: --check (no writes), --png (previews in private/re/fest/plsel/).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import lz

FONT = 'DejaVuSansCondensed-Bold.ttf'
SIZE = 18                       # px, both screens (JP glyphs are ~18 px)
PREV = 'private/re/fest/plsel'
BUILD = 'private/build'

# ---------------------------------------------------------------- data

CHARS = ['kyosuke', 'batsu', 'hinata', 'shoma', 'natsu', 'roberto', 'roy',
         'tiffany', 'boman', 'edge', 'akira_k', 'gan', 'hideo', 'kyoko',
         'raizo', 'hyo', 'akira_h', 'sakura', 'daigo', 'hayato', 'ran',
         'nagare']                     # entry order (MGAME06 0x2b0 + 4*i)
LINES = [('taiyo', 'm'), ('taiyo', 'f'), ('gorin', 'm'), ('gorin', 'f'),
         ('pacific', 'm'), ('pacific', 'f'), ('gedo', 'm'), ('justice', 'm'),
         ('justice', 'f')]             # MGAME06 0x268 + 8*j
TABLES = {'MGAME06.BIN': (0x2b0, 0x268), 'MGAME07.BIN': (0x280, 0x238)}
# fixed pieces: rect in TIM 3 (x, y, w), CLUT
# and baseline row. OK moves from the JP rect (0,200), whose glyph bottom
# would reach TIM row SAFE_V, to the free cells 10-11 of row 8; its
# MGAME07 record gets the new u, v.
FIXED = {'player': (0, 160, 100, 2, 15), 'partner': (100, 160, 100, 2, 15),
         'ok': (200, 160, 40, 0, 15)}
OK_REC = {'MGAME07.BIN': 0x378}
PS_SX = 0.92                     # horizontal squeeze of the popup text
VS_REC = {'MGAME06.BIN': [(0x3a8, 0x3b4), (0x3c0, 0x3cc)]}  # (Player, 1P/2P)
VS_GAP = 8                       # px between "Player" and 1P/2P ink
# free cell ranges (row, first, end): rows 0-7 (JP names and schools) are
# redrawn whole; row 8 has 2 empty cells after Player/Partner. In Dance
# the TIM rows from 215 are zero in VRAM at run time (emulator VRAM dump,
# 2026-11-18, private/re/fest/v1): rows 10 and 11 cannot hold names, and
# the JP row 10 pieces (OK, arrows, 1P/2P) end at row 214.
ROWS = [(r, 0, 12) for r in range(8)]
CLEAR = ROWS + [(8, 10, 12), (10, 0, 2), (10, 8, 12), (11, 0, 12)]   # JP OK, reruns
SAFE_V = 215                     # no ink at or below this TIM row
SCHOOLS = ['taiyo', 'gorin', 'pacific', 'gedo', 'justice']
GAP = 7                          # ink gap between school and sex word, px
PS_OFF, PS_PIX = 0x91800, 0x91880
LINE_CX, LINE_HALF = 256, 106    # popup line centre, max ink half-width

# opponent list (label atlas)
ATL_PIX, ATL_STRIDE = 0x1a0, 384
OPP = ['hyo', 'shoma', 'batsu', 'boman', 'gan']   # record idx 0-4
OPP_REC, OPP_X = 0x1a8, 200
OPP_Y = [200, 230, 260, 290, 320]                 # JP record y (20-row sprites)
PILL_Y = {'fast': 200, 'slow': 320}               # JP icon y
JP_OPP = [(192, 216, 20, 20), (200, 192, 40, 20), (160, 192, 40, 20),
          (160, 168, 80, 20), (216, 216, 20, 20)]
# areas this tool owns in the label atlas (x, y, w, h)
OWN = [(160, 168, 96, 48), (192, 216, 64, 24), (240, 120, 16, 48),
       (0, 240, 256, 16)]
SLOTS = [(160, 168, 96, 20), (160, 190, 96, 20), (192, 216, 64, 20),
         (0, 240, 256, 16)]
PILL_REC = {'slow': 0x1f0, 'fast': 0x1fc}
PILL_CX = 170                    # pill centre x on screen (JP icon 160-179)

def load_tl(path='tl/PLSEL.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb15(v): return ((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3)

# ---------------------------------------------------------------- render

def text_alpha(text, size=SIZE, ss=4, sx=1.0):
    """Alpha of one line, squeezed to sx: (array, baseline row, ink x0,
    ink x1). The box starts at the ascent line of the font."""
    f = fontpath.truetype(FONT, size * ss)
    asc, desc = f.getmetrics()
    b = f.getbbox(text)
    w = b[2] + 2 * ss
    im = Image.new('L', (w, asc + desc + ss), 0)
    ImageDraw.Draw(im).text((ss, 0), text, font=f, fill=255)
    if sx != 1.0: im = im.resize((max(1, int(round(im.width * sx))), im.height), Image.LANCZOS)
    nw, nh = -(-im.width // ss), -(-im.height // ss)
    big = Image.new('L', (nw * ss, nh * ss), 0); big.paste(im, (0, 0))
    a = np.asarray(big, float).reshape(nh, ss, nw, ss).mean((1, 3)) / 255
    cols = np.where(a.max(0) > 0.08)[0]
    return a, asc / ss, int(cols[0]), int(cols[-1])

def place(text, w, h, align, base, pad=1, sx=1.0):
    """Alpha (h, w) with the ink placed by align ('left', 'right',
    'centre') and the baseline at row base. Raises if ink (or its 1 px
    shadow) would fall outside the h rows."""
    a, asc, x0, x1 = text_alpha(text, sx=sx)
    ink = x1 - x0 + 1
    if ink > w - (2 * pad if align == 'centre' else pad): raise ValueError(f'"{text}": ink {ink} > {w - 2 * pad}')
    if align == 'left': dx = pad - x0
    elif align == 'right': dx = w - pad - 1 - x1
    else: dx = (w - ink) // 2 - x0
    dy = int(round(base - asc))
    rows = np.where(a.max(1) > 0.08)[0] + dy
    if rows[0] < 0 or rows[-1] > h - 2:
        raise ValueError(f'"{text}": ink rows {rows[0]}-{rows[-1]} (+1 shadow) outside 0-{h - 1}')
    out = np.zeros((h, w))
    for y in range(a.shape[0]):
        ty = y + dy
        if 0 <= ty < h:
            xs = slice(max(0, dx), min(w, a.shape[1] + dx))
            out[ty, xs] = np.maximum(out[ty, xs], a[y, xs.start - dx:xs.stop - dx])
    return out

def to_index(alpha, lums, fill, shade):
    """Composite white-ish text (value fill) over a 1 px down-right
    shadow (value shade) on transparent; nearest index by luminance among
    the allowed indices lums {index: value}. 0 = transparent."""
    sh = np.zeros_like(alpha)
    sh[1:, 1:] = alpha[:-1, :-1]
    sh = np.maximum(sh, 0.5 * np.pad(alpha, ((2, 0), (2, 0)))[:alpha.shape[0], :alpha.shape[1]])
    sh = np.clip(sh, 0, 1)
    v = alpha * fill + (1 - alpha) * sh * shade
    keys = np.array(list(lums)); vals = np.array([lums[k] for k in keys], float)
    idx = keys[np.abs(v[..., None] - vals).argmin(-1)]
    idx[(alpha < 0.12) & (sh < 0.3)] = 0
    return idx.astype(np.uint8)

def lum(v): r, g, b = rgb15(v); return max(r, g, b)

# ---------------------------------------------------------------- 4bpp

def get4(d, off, stride, w, h):
    raw = np.frombuffer(bytes(d[off:off + stride * h]), np.uint8).reshape(h, stride)
    px = np.empty((h, stride * 2), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px[:, :w].copy()

def put4(d, off, px):
    d[off:off + px.size // 2] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

# ---------------------------------------------------------------- player select

def ink_w(t):
    _, _, x0, x1 = text_alpha(t, sx=PS_SX)
    return x1 - x0 + 1

def plsel_layout(tl):
    """Runs to paint: list of (key, text, ncells, align, pad). Names are
    centred in their run. A school + sex line is two runs side by side:
    the school right-aligned (pad G - L), the sex word left-aligned (pad
    L), so the ink gap is always G. L and one optional extra cell per
    school run are chosen to centre the lines best (the run widths are
    multiples of 20 px)."""
    runs, seen = [], set()
    for c in CHARS:
        t = tl['name_' + c]
        if t in seen: continue
        seen.add(t)
        runs.append(('name_' + c, t, -(-(ink_w(t) + 2) // 20), 'centre', 1))
    sch = {s: ink_w(tl['school_' + s]) for s in SCHOOLS}
    sx = {x: ink_w(tl['sex_' + x]) for x in 'mf'}
    best = None
    for L in range(0, GAP + 1):
        nx = {x: -(-(sx[x] + L) // 20) for x in 'mf'}
        X = {x: 20 * nx[x] - sx[x] - L for x in 'mf'}          # right slack
        worst, ns = 0, {}
        for s in SCHOOLS:
            opts = []
            for add in (0, 1):
                n = -(-(sch[s] + GAP - L) // 20) + add
                Ls = 20 * n - sch[s] - (GAP - L)                  # left slack
                opts.append((max(abs(Ls - X[x]) / 2 for x in 'mf' if (s, x) in LINES), add, n))
            o = min(opts); ns[s] = o[2]; worst = max(worst, o[0])
        cost = (worst, sum(ns.values()) + sum(nx.values()))
        if best is None or cost < best[0]: best = (cost, L, nx, ns)
    _, L, nx, ns = best
    for s in SCHOOLS:
        runs.append(('school_' + s, tl['school_' + s], ns[s], 'right', GAP - L))
    for x in 'mf':
        runs.append(('sex_' + x, tl['sex_' + x], nx[x], 'left', L))
    return runs

def pack(runs):
    """First-fit decreasing into the free cell ranges ROWS. Returns
    ({key: cell}, {row: cells used})."""
    free = {r: a for r, a, b in ROWS}
    cell = {}
    for key, t, n, al, pad in sorted(runs, key=lambda r: (-r[2], r[0])):
        for r, a, b in ROWS:
            if free[r] + n <= b:
                cell[key] = 12 * r + free[r]; free[r] += n; break
        else: raise ValueError(f'no room for {key} ({n} cells)')
    return cell, {r: free[r] - a for r, a, b in ROWS}

def paint_plsel(d, tl, log):
    """Repaint TIM 3 in buffer d. Returns (cells {key: (cell, n)}, ink {fixed key: (x0, x1)})."""
    hdr = struct.unpack_from('<IIIHHHH', d, PS_OFF)
    assert hdr == (16, 8, 108, 0, 483, 48, 1), hdr
    clut = struct.unpack_from('<48H', d, PS_OFF + 0x14)
    px = get4(d, PS_PIX, 128, 256, 256)
    runs = plsel_layout(tl)
    cell, free = pack(runs)
    for r, a, b in CLEAR: px[20 * r:20 * r + 20, 20 * a:20 * b] = 0
    out = {}
    lums = {i: lum(clut[i]) for i in range(2, 16)}
    for key, t, n, al, pad in runs:
        k = cell[key]; u, v = 20 * (k % 12), 20 * (k // 12)
        a = place(t, 20 * n, 20, al, 15, pad, PS_SX)
        px[v:v + 20, u:u + 20 * n] = to_index(a, lums, 248, 56)
        out[key] = (k, n)
    ink = {}
    for key, (x, y, w, c, base) in FIXED.items():
        cl = {i: lum(clut[16 * c + i]) for i in range(2, 16)}
        top = max(cl.values())
        a = place(tl[key], w, 20, 'centre', base, 1, PS_SX)
        sub = to_index(a, cl, top, 56 * top / 248)
        px[y:y + 20, x:x + w] = sub
        cols = np.where(sub.any(0))[0]; ink[key] = (int(cols[0]), int(cols[-1]))
    for key in ('1p', '2p'):            # Latin JP pieces, kept
        x = 80 if key == '1p' else 120
        cols = np.where(px[200:220, x:x + 40].any(0))[0]; ink[key] = (int(cols[0]), int(cols[-1]))
    for (x, y, w, c, base), key in ((v, k) for k, v in FIXED.items()):
        rows = np.where(px[y:y + 20, x:x + w].any(1))[0] + y
        if rows[-1] >= SAFE_V: log.append(f'PROBLEM {key}: ink to TIM row {rows[-1]} (>= {SAFE_V})')
    for key, (k, n) in out.items():
        u, v = 20 * (k % 12), 20 * (k // 12)
        rows = np.where(px[v:v + 20, u:u + 20 * n].any(1))[0] + v
        if rows[-1] >= SAFE_V: log.append(f'PROBLEM {key}: ink to TIM row {rows[-1]} (>= {SAFE_V})')
    put4(d, PS_PIX, px)
    log.append('plsel cells used per row: ' + ' '.join(f'{r}:{free[r]}' for r, a, b in ROWS))
    return out, ink, px, clut

def line_ink(px, entry):
    """Screen ink span (x0, x1) of one popup line drawn from entry."""
    runs = entry
    tot = 20 * sum(n for _, n in runs)
    x = (512 - tot) // 2; xs = []
    for k, n in runs:
        u, v = 20 * (k % 12), 20 * (k // 12)
        cols = np.where(px[v:v + 20, u:u + 20 * n].any(0))[0]
        if len(cols): xs += [x + cols[0], x + cols[-1]]
        x += 20 * n
    return min(xs), max(xs)

def patch_tables(name, d, cells, tl, ink, log):
    noff, soff = TABLES[name]
    for i, c in enumerate(CHARS):
        t = tl['name_' + c]
        key = next(k for k in cells if k.startswith('name_') and tl[k] == t)
        k, n = cells[key]
        d[noff + 4 * i:noff + 4 * i + 4] = bytes((1, k, n, 0))
    for j, (s, x) in enumerate(LINES):
        ks, ns = cells['school_' + s]; kx, nx = cells['sex_' + x]
        d[soff + 8 * j:soff + 8 * j + 8] = bytes((2, ks, ns, kx, nx, 0, 0, 0))
    inv = {v: k for k, v in cells.items()}                # self-check
    for i, c in enumerate(CHARS):
        cnt, k, n, _ = d[noff + 4 * i:noff + 4 * i + 4]
        if cnt != 1 or tl.get(inv.get((k, n)), None) != tl['name_' + c]:
            log.append(f'PROBLEM {name} entry {i} ({c}): cells {k}+{n} -> {inv.get((k, n))}')
    for j, (sc, x) in enumerate(LINES):
        e = tuple(d[soff + 8 * j:soff + 8 * j + 5])
        if e != (2, *cells['school_' + sc], *cells['sex_' + x]):
            log.append(f'PROBLEM {name} line {j} ({sc}/{x}): {e}')
    if name in OK_REC:
        off = OK_REC[name]; x, y, w = FIXED['ok'][:3]
        c, u, v, ow, sx_, sy = struct.unpack_from('<6h', d, off)
        assert ow == w == 40 and (u, v) in ((0, 200), (x, y)), (name, hex(off))
        struct.pack_into('<2h', d, off + 2, x, y)
    for prec, nrec in VS_REC.get(name, []):
        c, u, v, w, x, y = struct.unpack_from('<6h', d, prec)
        c2, u2, v2, w2, x2, y2 = struct.unpack_from('<6h', d, nrec)
        assert (u, v, w) == (0, 160, 100) and v2 == 200 and w2 == 40, (name, hex(prec))
        p0, p1 = ink['player']; q0, q1 = ink['1p' if u2 == 80 else '2p']
        tot = (p1 - p0 + 1) + VS_GAP + (q1 - q0 + 1)
        left = LINE_CX - tot // 2
        struct.pack_into('<h', d, prec + 8, left - p0)
        struct.pack_into('<h', d, nrec + 8, left + (p1 - p0 + 1) + VS_GAP - q0)
        log.append(f'{name} VS: Player x {x}->{left - p0}, {"1P" if u2 == 80 else "2P"} '
                   f'x {x2}->{left + (p1 - p0 + 1) + VS_GAP - q0}')

# ---------------------------------------------------------------- opponent list

PILL_FONT = {   # 7-row bitmaps
    'F': ['####', '#...', '#...', '###.', '#...', '#...', '#...'],
    'A': ['.##.', '#..#', '#..#', '####', '#..#', '#..#', '#..#'],
    'S': ['.###', '#...', '#...', '.##.', '...#', '...#', '###.'],
    'T': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
    'L': ['#...', '#...', '#...', '#...', '#...', '#...', '####'],
    'O': ['.##.', '#..#', '#..#', '#..#', '#..#', '#..#', '.##.'],
    'W': ['#...#', '#...#', '#...#', '#.#.#', '#.#.#', '##.##', '#...#'],
}

def pill(text):
    """Pill icon in CLUT 9 indices: red ring (f), dark inside (1), white
    letters (8) with a dark grey shadow (2), dark rim at the right and bottom
    (c). Returns (h, w) indices."""
    glyphs = [PILL_FONT[ch] for ch in text]
    tw = sum(len(g[0]) for g in glyphs) + len(glyphs) - 1
    w, h = tw + 6, 13
    m = np.zeros((h, w), bool)                       # rounded rect
    m[:, 2:w - 2] = True; m[1:h - 1, 1:w - 1] = True; m[2:h - 2, :] = True
    inner = np.zeros_like(m); inner[2:h - 2, 3:w - 3] = True; inner[3:h - 3, 2:w - 2] = True
    out = np.zeros((h, w), np.uint8)
    out[m] = 0xf; out[inner] = 1
    edge = m & ~np.pad(m, ((0, 1), (0, 1)))[1:, 1:]  # right/bottom outer edge
    out[edge] = 0xc
    x = 3
    for g in glyphs:
        for r, row in enumerate(g):
            for c, ch in enumerate(row):
                if ch == '#':
                    if out[3 + r + 1, x + c + 1] == 1: out[3 + r + 1, x + c + 1] = 2
        x += len(g[0]) + 1
    x = 3
    for g in glyphs:
        for r, row in enumerate(g):
            for c, ch in enumerate(row):
                if ch == '#': out[3 + r, x + c] = 8
        x += len(g[0]) + 1
    return out

def opp_sprites(tl, clut):
    """{key: (indices, top)}: names rendered 20 rows tall, then cropped to
    their ink rows (top = first row kept; the record y moves by top), and
    the two pills."""
    lums = {i: lum(clut[16 + i]) for i in range(2, 16)}
    out = {}
    for k in OPP:
        t = tl['opp_' + k]
        _, _, x0, x1 = text_alpha(t)
        a = place(t, x1 - x0 + 4, 20, 'left', 15, 1)
        sub = to_index(a, lums, 232, 64)
        rows = np.where(sub.any(1))[0]
        out['opp_' + k] = (sub[rows[0]:rows[-1] + 1], int(rows[0]))
    for k in ('fast', 'slow'): out[k] = (pill(tl[k]), 0)
    return out

def opp_layout(spr):
    """Places the sprites in SLOTS (tallest first). Returns {key: (u, v, w, h)}."""
    fill = [0] * len(SLOTS); pos = {}
    items = [(k, a.shape[1], a.shape[0]) for k, (a, _) in spr.items()]
    for key, w, h in sorted(items, key=lambda t: (-t[2], -t[1], t[0])):
        for s, (sx, sy, sw, shh) in enumerate(SLOTS):
            if h <= shh and fill[s] + w <= sw:
                pos[key] = (sx + fill[s], sy, w, h); fill[s] += w; break
        else: raise ValueError(f'no atlas room for {key} ({w}x{h})')
    return pos

def paint_opp(d, spr, pos, jp):
    """Paint the opponent names and pills into the label atlas of buffer d.
    jp = the JP atlas indices (for the guard). Returns the atlas indices."""
    clut = struct.unpack_from('<192H', d, 0x14)
    px = get4(d, ATL_PIX, ATL_STRIDE, 768, 256)
    new = px.copy()
    for x, y, w, h in OWN: new[y:y + h, x:x + w] = 0
    new[216:240, 160:192] = px[216:240, 160:192]          # f_unit, not ours
    for key, (u, v, w, h) in pos.items():
        new[v:v + h, u:u + w] = spr[key][0]
    for x, y, w, h in OWN:                                  # guard
        for yy in range(y, y + h):
            cols = [c for c in range(x, x + w) if not (216 <= yy < 240 and 160 <= c < 192)]
            a, b, c = px[yy, cols], jp[yy, cols], new[yy, cols]
            if not ((a == b).all() or (a == c).all()):
                raise SystemExit(f'label atlas row {yy} x {x}-{x + w - 1}: painted by another '
                                 'tool; check the OWN rects against minittl_atlas.ATLAS')
    put4(d, ATL_PIX, new)
    return new, clut

def patch_opp(d, spr, pos, log):
    for i, k in enumerate(OPP):
        off = OPP_REC + 12 * i
        nu, nv, nw, nh = pos['opp_' + k]
        struct.pack_into('<6h', d, off, OPP_X, OPP_Y[i] + spr['opp_' + k][1], nu, nv, nw, nh)
    for k, off in PILL_REC.items():
        cy = PILL_Y[k] + 10                          # JP icon 20x20
        nu, nv, nw, nh = pos[k]
        struct.pack_into('<6h', d, off, PILL_CX - nw // 2, cy - nh // 2, nu, nv, nw, nh)
        log.append(f'{k}: screen {PILL_CX - nw // 2},{cy - nh // 2} {nw}x{nh} atlas {nu},{nv}')

# ---------------------------------------------------------------- previews

def preview(ps_px, ps_clut, cells, tl, atl, atl_clut, pos, mg06, opsp):
    os.makedirs(PREV, exist_ok=True)
    pal = lambda c: np.array([rgb15(v) for v in ps_clut[16 * c:16 * c + 16]], float)
    bg = np.asarray(Image.open('private/re/fest/r3/c00.png').convert('RGB'), float)
    def spr(img, u, v, w, h, x, y, P, px, bright=1.0):
        sub = px[v:v + h, u:u + w]
        m = sub > 0
        img[y:y + h, x:x + w][m] = P[sub][m] * bright
    tiles = []
    entries = [[cells['name_' + next(k for k in CHARS if tl['name_' + k] == tl['name_' + c])]]
               for c in CHARS]
    entries += [[cells['school_' + s], cells['sex_' + x]] for s, x in LINES]
    for e in entries:
        img = bg.copy(); img[250:316, 140:372] = (16, 16, 16)
        spr(img, 0, 160, 100, 20, 206, 260, pal(2), ps_px)
        tot = 20 * sum(n for _, n in e); x = (512 - tot) // 2
        for k, n in e:
            spr(img, 20 * (k % 12), 20 * (k // 12), 20 * n, 20, x, 290, pal(0), ps_px); x += 20 * n
        spr(img, 40, 200, 16, 16, 128, 290, pal(0), ps_px)
        spr(img, 60, 200, 16, 16, 366, 290, pal(0), ps_px)
        tiles.append(img[252:316, 112:400])
    cols = 4; th, tw = tiles[0].shape[:2]
    sheet = np.zeros((((len(tiles) + cols - 1) // cols) * (th + 2), cols * (tw + 2), 3))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); sheet[r * (th + 2):r * (th + 2) + th, c * (tw + 2):c * (tw + 2) + tw] = t
    Image.fromarray(sheet.astype(np.uint8)).save(f'{PREV}/plsel_lines.png')
    # opponent list, cursor on row 0 and on row 3, JP | EN
    bg = np.asarray(Image.open('private/re/fest/r3/c01.png').convert('RGB'), float)
    pal9 = np.array([rgb15(v) for v in atl_clut[144:160]], float)
    pal1 = np.array([rgb15(v) for v in atl_clut[16:32]], float)
    bar = bg[205, 330].copy()
    out = []
    for sel in (0, 3):
        img = bg.copy(); img[196:340, 156:350] = (16, 16, 16)
        ys = [200, 230, 260, 290, 320]
        img[ys[sel] - 4:ys[sel] + 22, 190:342] = bar
        spr(img, 240, 24, 15, 64, 160, 230, pal9, atl)
        for i, k in enumerate(OPP):
            u, v, w, h = pos['opp_' + k]
            spr(img, u, v, w, h, OPP_X, ys[i] + opsp['opp_' + k][1], pal1, atl, 1.0 if i == sel else 0.5)
        for k, y in (('fast', 200), ('slow', 320)):
            u, v, w, h = pos[k]
            spr(img, u, v, w, h, PILL_CX - w // 2, y + 10 - h // 2, pal9, atl)
        out.append(img[180:360, 140:400])
    jp = bg[180:360, 140:400]
    im = np.concatenate([jp, np.zeros((180, 4, 3)), out[0], np.zeros((180, 4, 3)), out[1]], 1)
    Image.fromarray(im.astype(np.uint8)).save(f'{PREV}/opp.png')
    # Dance popup (Player, Partner, OK) on the popup background
    img = np.asarray(Image.open('private/re/fest/r3/c00.png').convert('RGB'), float).copy()
    img[250:420, 140:372] = (16, 16, 16)
    spr(img, 0, 160, 100, 20, 206, 260, pal(2), ps_px)
    spr(img, 100, 160, 100, 20, 206, 320, pal(2), ps_px)
    spr(img, 200, 160, 40, 20, 236, 390, pal(0), ps_px)
    for k, y in ((entries[0][0], 290), (entries[1][0], 350)):
        spr(img, 20 * (k[0] % 12), 20 * (k[0] // 12), 20 * k[1], 20, 256 - 10 * k[1], y, pal(0), ps_px)
    # VS lines from the patched MGAME06 records
    img2 = np.asarray(Image.open('private/re/fest/r3/c00.png').convert('RGB'), float).copy()
    img2[250:420, 140:372] = (16, 16, 16)
    for off in (0x3a8, 0x3b4, 0x3c0, 0x3cc):
        c, u, v, w, x, y = struct.unpack_from('<6h', mg06, off)
        spr(img2, u, v, w, 20, x, y, pal(c), ps_px)
    for k, y in ((entries[0][0], 290), (entries[7][0], 350)):
        spr(img2, 20 * (k[0] % 12), 20 * (k[0] // 12), 20 * k[1], 20, 256 - 10 * k[1], y, pal(0), ps_px)
    im = np.concatenate([img[240:420, 120:392], np.zeros((180, 4, 3)), img2[240:420, 120:392]], 1)
    Image.fromarray(im.astype(np.uint8)).save(f'{PREV}/popup.png')
    Image.fromarray((pal(0)[ps_px]).astype(np.uint8)).save(f'{PREV}/plsel_atlas.png')

# ---------------------------------------------------------------- main

def slot(name):
    s = os.path.getsize(f'private/disc/DAT2/{name}')
    return s + (-s % 2048)

def build(check=False, png=False):
    tl = load_tl(); log = []; problems = 0
    jp_atl = get4(open('private/unpacked/MINITTL0.BIZ.lz', 'rb').read(), ATL_PIX, ATL_STRIDE, 768, 256)
    jclut = struct.unpack_from('<192H', open('private/unpacked/MINITTL0.BIZ.lz', 'rb').read(), 0x14)
    spr = opp_sprites(tl, jclut)
    pos = opp_layout(spr)
    files = {}
    cells = ink = None
    for n in '0123456789ABC':
        name = f'MINITTL{n}.BIZ'
        p = f'{BUILD}/{name}.lz'
        if not os.path.exists(p): raise SystemExit(f'{p} missing: run build_minittl.py first')
        d = bytearray(open(p, 'rb').read())
        atl, aclut = paint_opp(d, spr, pos, jp_atl)
        if n in '9ABC':
            c2, k2, ps_px, ps_clut = paint_plsel(d, tl, log if n == '9' else [])
            cells, ink = c2, k2
        files[name] = d
    for key, (k, n) in sorted(cells.items(), key=lambda t: t[1]):
        log.append(f'  cell {k:3d} n {n} {key} "{tl[key]}"')
        if k % 12 + n > 12 or 20 * (k // 12) + 20 > SAFE_V:
            log.append(f'PROBLEM {key}: cells {k}+{n} cross a row or end below TIM row {SAFE_V}')
    mg = {}
    for name in TABLES:
        p = f'{BUILD}/{name}'
        d = bytearray(open(p if os.path.exists(p) else f'private/disc/DAT1/{name}', 'rb').read())
        patch_tables(name, d, cells, tl, ink, log)
        if name == 'MGAME06.BIN': patch_opp(d, spr, pos, log)
        mg[name] = d
    # line widths (the popup arrows sit at 128-143 and 366-381)
    for i, c in enumerate(CHARS + [f'{s}/{x}' for s, x in LINES]):
        e = ([cells[next(k for k in cells if k.startswith('name_') and tl[k] == tl['name_' + c])]]
             if i < len(CHARS) else [cells['school_' + LINES[i - 22][0]], cells['sex_' + LINES[i - 22][1]]])
        x0, x1 = line_ink(ps_px, e)
        off = (x0 + x1) / 2 - LINE_CX
        bad = x0 < LINE_CX - LINE_HALF or x1 > LINE_CX + LINE_HALF or abs(off) > 6
        if bad or i >= 22: log.append(f'line {c}: ink {x0}-{x1}, off-centre {off:+.1f}' + (' PROBLEM' if bad else ''))
        problems += bad
    for k, (u, v, w, h) in pos.items(): log.append(f'atlas {k}: {u},{v} {w}x{h}')
    problems += sum(line.startswith('PROBLEM') for line in log)
    for line in log: print(line)
    if png: preview(ps_px, ps_clut, cells, tl, atl, aclut, pos, mg['MGAME06.BIN'], spr)
    if check or problems:
        print(f'plsel: problems {problems}' + (' (nothing written)' if not check else ''))
        return problems
    for name, d in files.items():
        c = lz.compress(bytes(d), name=name)
        if len(c) > slot(name): print(f'{name}: {len(c)} > slot {slot(name)}'); return 1
        open(f'{BUILD}/{name}.lz', 'wb').write(d)
        open(f'{BUILD}/{name}', 'wb').write(c)
    for name, d in mg.items(): open(f'{BUILD}/{name}', 'wb').write(d)
    print('plsel: problems 0, wrote MINITTL0-C, MGAME06, MGAME07')
    return 0

if __name__ == '__main__':
    sys.exit(1 if build('--check' in sys.argv, '--png' in sys.argv) else 0)
